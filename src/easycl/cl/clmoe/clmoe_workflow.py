import os
import shutil
import torch
import numpy as np
from typing import TYPE_CHECKING, List, Optional
import re

from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.logging import get_logger
from llamafactory.extras.misc import calculate_tps, get_logits_processor
from llamafactory.extras.ploting import plot_loss
from llamafactory.model import load_tokenizer
from llamafactory.train.trainer_utils import create_modelcard_and_push
from llamafactory.train.sft.metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor

# Import CL-MoE specific components
from .clmoe_loader import load_clmoe_model # Changed from llamafactory.cl.clmoe...
from .clmoe_trainer import CLMoETrainer # Changed from llamafactory.cl.clmoe...
from .peft.tuners.clitmoelora import CLMoEMOELoraLinear # Changed from llamafactory.cl.clmoe...

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from llamafactory.hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments # Keep llamafactory
    from easycl.hparams import CLFinetuningArguments # Change to EasyCL


logger = get_logger(__name__)


def run_statistics_calculation(output_dir: str, task_id: str, top_k_experts: int):
    logger.info_rank0(f"Running statistics calculation for task: {task_id}")
    value_counts_file = os.path.join(output_dir, f"value_counts_{task_id}.txt")
    index_output_file = os.path.join(output_dir, f"index_{task_id}.txt")

    if not os.path.exists(value_counts_file):
        # Raise specific error if input file not found
        raise FileNotFoundError(f"Statistics input file not found: {value_counts_file}. Cannot proceed.")

    try:
        # Initialize a dictionary to store the sum for each index
        index_sum_dict = {}
        with open(value_counts_file, "r", encoding="utf-8") as txt_file:
            for line in txt_file:
                try:
                    index, number = line.strip().split(":")
                    index = int(index)
                    number = int(number)
                    index_sum_dict[index] = index_sum_dict.get(index, 0) + number
                except ValueError:
                    logger.warning_rank0(f"Skipping malformed line in {value_counts_file}: {line.strip()}")
                    continue

        # Sort the dictionary by values in descending order
        sorted_result = dict(sorted(index_sum_dict.items(), key=lambda item: item[1], reverse=True))

        # Collect the top k indices, excluding -1
        top_keys = []
        for key in sorted_result:
            if key != -1:
                top_keys.append(key)
            if len(top_keys) == top_k_experts:
                break

        # Write the top indices to the output file
        with open(index_output_file, "w", encoding="utf-8") as output_file:
            for key in top_keys:
                output_file.write(f"{key}\n")

        logger.info_rank0(f"Successfully calculated statistics and saved top {top_k_experts} indices to {index_output_file}")
        logger.info_rank0(f"Statistics result: {sorted_result}")
        return True
    except Exception as e:
        # logger.error_rank0(f"Error during statistics calculation for {task_id}: {e}")
        # Raise runtime error on calculation failure
        # raise RuntimeError(f"Error during statistics calculation for {task_id}: {e}")
        raise ValueError(f"Error during statistics calculation for {task_id}: {e}")


def run_parameter_alignment(output_dir: str, current_task_id: str, prev_task_id: str, prev_task_model_path: str):
    logger.info_rank0(f"Running parameter alignment: merging {prev_task_id} into {current_task_id}")
    not_aligned_dir = os.path.join(output_dir, "not_aligned")
    adapter_bin_name = "adapter_model.bin" # Assume standard name

    finetuned_model1_path = os.path.join(prev_task_model_path, adapter_bin_name)
    finetuned_model2_path = os.path.join(not_aligned_dir, adapter_bin_name)
    index1_path = os.path.join(prev_task_model_path, f"index_{prev_task_id}.txt")
    index2_path = os.path.join(output_dir, f"index_{current_task_id}.txt")

    # Check if all required files exist
    required_files = [finetuned_model1_path, finetuned_model2_path, index1_path, index2_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        # Raise specific error if required files are missing
        raise FileNotFoundError(f"Missing required files for parameter alignment: {', '.join(missing_files)}. Cannot proceed.")

    try:
        # Load finetuned weights
        finetuned_state_dict1 = torch.load(finetuned_model1_path, map_location="cpu")
        finetuned_state_dict2 = torch.load(finetuned_model2_path, map_location="cpu")

        # Define weights (alpha seems hardcoded in params.py, keeping it here)
        alpha = 0.7

        # Read index files
        def read_indices(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return [int(line.strip()) for line in f]

        index1 = read_indices(index1_path)
        index2 = read_indices(index2_path)

        # Determine indices that are only in task1 (previous task)
        exclusive_indices = [i for i in index1 if i not in index2]

        # Build the prefix list for targeted LoRA modules (Assuming 8 experts as in params.py)
        # TODO: Potentially make expert_num dynamic if needed
        total_experts = 8 # Hardcoded based on params.py, adjust if necessary
        exclusive_prefixes = [f"base_model.model.model.layers.{layer_idx}.mlp.experts.{i}." for layer_idx in range(32) for i in exclusive_indices] \
                         + [f"base_model.model.model.layers.{layer_idx}.mlp.lora_A.loraA.{i}." for layer_idx in range(32) for i in exclusive_indices] \
                         + [f"base_model.model.model.layers.{layer_idx}.mlp.lora_B.loraB.{i}." for layer_idx in range(32) for i in exclusive_indices]
                         # Add other LoRA layers (e.g., self_attn) if they also use experts similarly
        
        # Rework prefix generation based on typical PEFT LoRA naming conventions. 
        # The original params.py prefixes `loraA.{i}` are too generic. We need the full module path.
        # Finding exact paths dynamically is complex without loading the model structure.
        # Using a placeholder structure based on common LLaMA LoRA fine-tuning.
        # This part might need significant adjustment based on the *actual* keys in your state dicts.
        
        # Let's try a more pattern-based approach assuming keys look like:
        # 'base_model.model.model....lora_A.default.loraA.0.mlp.weight'
        # 'base_model.model.model....lora_B.default.loraB.0.mlp.weight'
        # 'base_model.model.model....lora_router.default.weight' (Router weights might not be indexed by expert)
        
        logger.info_rank0(f"Exclusive expert indices (in {prev_task_id} only): {exclusive_indices}")

        # Initialize combined model state_dict
        combined_state_dict = finetuned_state_dict2.copy()

        # Merge parameters
        merged_count = 0
        skipped_count = 0
        problematic_keys = []

        expert_pattern = re.compile(r"lora_[AB]\.([a-zA-Z0-9_]+)\.lora[AB]\.(\d+)\.")

        for name, param2 in finetuned_state_dict2.items():
            if name not in finetuned_state_dict1:
                # Keep param from model 2 if it doesn't exist in model 1
                skipped_count += 1
                continue

            param1 = finetuned_state_dict1[name]
            match = expert_pattern.search(name)
            is_expert_param = False
            expert_idx = -1

            if match:
                expert_idx = int(match.group(2))
                is_expert_param = True

            # Apply merging logic based on expert index
            if is_expert_param and expert_idx in exclusive_indices:
                # Exclusive to task 1: Weighted average (Model1 more influence)
                # Original params.py logic was: combined = (1 - alpha) * param1 + alpha * param2
                # Let's stick to that logic interpretation
                combined_state_dict[name] = (1 - alpha) * param1 + alpha * param2
                merged_count += 1
            elif is_expert_param: # Expert param but not exclusive (shared or exclusive to task 2)
                # Default/Shared: Weighted average (Model2 more influence)
                # Original params.py logic was: combined = alpha * param1 + (1 - alpha) * param2
                combined_state_dict[name] = alpha * param1 + (1 - alpha) * param2
                merged_count += 1
            else: # Not an expert-specific LoRA weight (e.g., router, embeddings, base weights if saved)
                # Keep the weights from the current task's model (model 2)
                # combined_state_dict[name] = param2 # Already copied by default
                skipped_count += 1

        logger.info_rank0(f"Parameter alignment summary: Merged {merged_count} tensors, Skipped/Kept {skipped_count} tensors.")
        if problematic_keys:
             logger.warning_rank0(f"Problematic keys during merge (check patterns): {problematic_keys[:10]}...")

        # --- Save merged model --- 
        # 1. Remove existing final output dir content carefully (except the not_aligned subdir)
        # The entire cleanup block is removed as requested.

        # 2. Copy necessary non-weight files (like config.json, tokenizer files) from not_aligned dir
        logger.info_rank0(f"Copying config/tokenizer files from {not_aligned_dir} to {output_dir}")
        shutil.copytree(not_aligned_dir, output_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns(adapter_bin_name))

        # 3. Save the combined state dict
        final_adapter_path = os.path.join(output_dir, adapter_bin_name)
        torch.save(combined_state_dict, final_adapter_path)
        logger.info_rank0(f"Saved aligned adapter weights to {final_adapter_path}")
        # --- End Save --- 

        logger.info_rank0("Parameter alignment completed successfully.")
        return True

    except Exception as e:
        # logger.error_rank0(f"Error during parameter alignment: {e}")
        # Raise runtime error on alignment failure
        # raise RuntimeError(f"Error during parameter alignment: {e}")
        raise ValueError(f"Error during parameter alignment: {e}")


def run_sft_clmoe( # Renamed from run_sft_moelora
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    cl_finetuning_args: "CLFinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    """
    Run supervised fine-tuning with CL-MoE adapters.
    
    This workflow is similar to the standard SFT workflow, but uses the 
    CL-MoE specific loader and trainer.
    """
    # --- Moved: Write task.txt at the beginning if training --- 
    if training_args.do_train:
        if hasattr(cl_finetuning_args, "current_task_id") and cl_finetuning_args.current_task_id:
            # --- Calculate the new path for task.txt ---
            original_output_dir = training_args.output_dir
            parent_dir = os.path.dirname(original_output_dir)
            base_name = os.path.basename(original_output_dir)

            # Handle edge case where output_dir might be in the root or relative to CWD
            if not parent_dir:
                parent_dir = "." # Use current directory as parent
                taskid_save_base = "taskid_save"
            else:
                taskid_save_base = os.path.join(parent_dir, "taskid_save")

            new_task_dir = os.path.join(taskid_save_base, base_name)
            task_file_path = os.path.join(new_task_dir, "task.txt")
            # --- End new path calculation ---
            try:
                # Ensure the new directory exists
                os.makedirs(new_task_dir, exist_ok=True) # Create the new directory structure
                with open(task_file_path, "w", encoding="utf-8") as t_file:
                    t_file.write(cl_finetuning_args.current_task_id)
                logger.info_rank0(f"Current task '{cl_finetuning_args.current_task_id}' written to {task_file_path}")
                # Add confirmation log
                logger.info_rank0(f"Successfully wrote task.txt to the new location at the start of training.")
            except Exception as e:
                # logger.error_rank0(f"Failed to write task.txt to {task_file_path}: {e}")
                # Raise error if writing fails
                # raise RuntimeError(f"Failed to write task.txt to {task_file_path} before training: {e}")
                raise ValueError(f"Failed to write task.txt to {task_file_path} before training: {e}")
        else:
            # Raise an error if current_task_id is crucial and missing during training
            raise ValueError("cl_finetuning_args.current_task_id is required for cl-MoE training but is not defined or empty.")
    # --- End Moved block --- 
    
    # Validate cl-MoE configuration
    if cl_finetuning_args.use_cl_moe:
        if cl_finetuning_args.expert_num is None or cl_finetuning_args.expert_num <= 1:
             # This should be caught by CLFinetuningArguments.__post_init__, but double-check
            raise ValueError("expert_num must be greater than 1 for cl-MoE.")
        logger.info_rank0("Running SFT with cl-MoE enabled.")
    elif cl_finetuning_args.use_moe:
         # Log if standard MoE-LoRA is used instead
         logger.info_rank0("Running SFT with standard MoE-LoRA enabled.")
    # No specific warning if neither is enabled and finetuning_type is lora, 
    # as it defaults to standard LoRA.
    
    # Load tokenizer and template
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    # training_args.save_safetensors = False # Keep this behavior from original moelora_workflow? Or allow True?
    # Consider if cl-MoE adapters are compatible with safetensors
    # Let's keep it False for now for consistency with the source
    training_args.save_safetensors = False 
    
    # Get dataset
    dataset_module = get_dataset(
        template=template,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        stage="sft",
        **tokenizer_module
    )
    
    # Load model using CL-MoE adapter loader
    model = load_clmoe_model( # Changed function call
        tokenizer=tokenizer,
        model_args=model_args,
        finetuning_args=finetuning_args,
        cl_finetuning_args=cl_finetuning_args,
        is_trainable=training_args.do_train
    )
    
    # For better serialization of quantized models during inference
    if getattr(model, "is_quantized", False) and not training_args.do_train:
        # Check if this is needed/compatible with CL-MoE adapters
        setattr(model, "_hf_peft_config_loaded", True) # Keep for now

    # Inject output_dir into CLMoEMOELoraLinear modules if training
    if training_args.do_train:
        logger.info_rank0("Injecting output_dir into CLMoEMOELoraLinear modules...")
        injected_count = 0
        for module in model.modules():
            if isinstance(module, CLMoEMOELoraLinear):
                module.output_dir = training_args.output_dir
                injected_count += 1
        if injected_count > 0:
            logger.info_rank0(f"Successfully injected output_dir into {injected_count} CLMoEMOELoraLinear module(s).")
        else:
            logger.warning_rank0("No CLMoEMOELoraLinear modules found to inject output_dir into.")

    # Prepare data collator
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )
    
    # Override the decoding parameters
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False  # Important for multimodal dataset
    
    # Set up metrics
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor
    
    # Prepare generation arguments
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()
    
    # Initialize the cl-MoE trainer
    trainer = CLMoETrainer( # Changed class
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        cl_finetuning_args=cl_finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs=gen_kwargs,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )
    
    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        
        # --- Conditional Model Saving, Statistics, and Alignment --- 
        logger.info_rank0("Training finished. Proceeding to saving and potential alignment.")
        
        # Check prerequisites for alignment
        can_align = ( cl_finetuning_args.prev_task_id and
                      cl_finetuning_args.current_task_id and
                      cl_finetuning_args.previous_task_model and
                      os.path.exists(cl_finetuning_args.previous_task_model) )
                      
        if not can_align:
             logger.info_rank0("First task or missing parameters/paths for alignment (prev_task_id, current_task_id, previous_task_model). Saving model directly.")
             # Save directly to output_dir for the first task or if alignment params missing
             try:
                 model.save_pretrained(training_args.output_dir)
                 logger.info_rank0(f"Model saved directly to {training_args.output_dir}")
             except Exception as e:
                 # logger.error_rank0(f"Failed to save model directly: {e}")
                 # raise RuntimeError(f"Failed to save model directly to {training_args.output_dir}: {e}")
                 raise ValueError(f"Failed to save model directly to {training_args.output_dir}: {e}")
             
             # --- Added: Run statistics calculation also for the first task --- 
             logger.info_rank0("Running statistics calculation for the first task...")
             try:
                 run_statistics_calculation(training_args.output_dir, cl_finetuning_args.current_task_id, cl_finetuning_args.top_k_experts)
                 logger.info_rank0("Statistics calculation for the first task completed successfully.")
             except (FileNotFoundError, RuntimeError) as stats_err:
                 # logger.error_rank0(f"Statistics calculation failed for the first task: {stats_err}")
                 # Raise error if statistics fail even for the first task
                 # raise RuntimeError(f"Statistics calculation failed for the first task: {stats_err}")
                 raise ValueError(f"Statistics calculation failed for the first task: {stats_err}")
             # --- End Added block ---
             
        else:
            logger.info_rank0("Running subsequent task. Preparing for alignment.")
            not_aligned_dir = os.path.join(training_args.output_dir, "not_aligned")
            
            # 1. Save current (unaligned) model to not_aligned subdir
            try:
                logger.info_rank0(f"Saving unaligned model to {not_aligned_dir}")
                model.save_pretrained(not_aligned_dir)
                logger.info_rank0("Unaligned model saved.")
            except Exception as e:
                # logger.error_rank0(f"Failed to save unaligned model to {not_aligned_dir}: {e}. Skipping alignment.")
                # Exit this block if saving fails, keep output_dir as is (likely empty or from previous run)
                # raise RuntimeError(f"Failed to save unaligned model to {not_aligned_dir}: {e}")
                raise ValueError(f"Failed to save unaligned model to {not_aligned_dir}: {e}")

            # 2. Run statistics calculation for the current task
            run_statistics_calculation(training_args.output_dir, cl_finetuning_args.current_task_id, cl_finetuning_args.top_k_experts)

            # 3. Run parameter alignment (which handles final saving)
            run_parameter_alignment(
                output_dir=training_args.output_dir,
                current_task_id=cl_finetuning_args.current_task_id,
                prev_task_id=cl_finetuning_args.prev_task_id,
                prev_task_model_path=cl_finetuning_args.previous_task_model
            )
            logger.info_rank0("Statistics and Alignment steps completed successfully.")

        # --- End Conditional Saving --- 
        
        # The original save_pretrained call is now handled within the conditional logic above.
        # model.save_pretrained(training_args.output_dir) # Removed/handled above
        
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )
        
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(
                training_args.output_dir, 
                keys=["loss", "eval_loss", "eval_accuracy"] # Check if other metrics are relevant for cl-MoE
            )
    
    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # Use left-padding in generation
    
    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        # TODO: Add cl-MoE specific evaluation metrics if needed
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    # Prediction
    if training_args.do_predict:
        logger.warning_rank0(
            "Batch generation can be very slow. "
            "Consider using `scripts/vllm_infer.py` instead."
        )
        predict_results = trainer.predict(
            dataset_module["eval_dataset"], 
            metric_key_prefix="predict", 
            **gen_kwargs
        )
        # TODO: Add cl-MoE specific prediction handling/metrics if needed
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(
            dataset_module["eval_dataset"], 
            predict_results, 
            generating_args.skip_special_tokens
        )
    
    # Create model card
    create_modelcard_and_push(
        trainer, model_args, data_args, training_args, finetuning_args
    ) 