# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import copy
from typing import TYPE_CHECKING, List, Optional
import traceback
from llamafactory.data import (
    SFTDataCollatorWith4DAttentionMask, 
    get_dataset, 
    get_template_and_fix_tokenizer,
)
from llamafactory.data.data_utils import merge_dataset
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.logging import get_logger
from llamafactory.extras.misc import calculate_tps, get_logits_processor
from llamafactory.extras.ploting import plot_loss
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.trainer_utils import create_modelcard_and_push
from llamafactory.train.sft.metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .lamol_trainer import LAMOLTrainer
from .lamol import LAMOLGenerator # Import LAMOLGenerator
from llamafactory.hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments
from easycl.hparams import CLFinetuningArguments


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback


logger = get_logger(__name__)


def run_sft_lamol(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    cl_finetuning_args: "CLFinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    """Run SFT training using LAMOL (pseudo-replay style)."""
    # --- 1. Load Tokenizer and Template ---    
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # --- Determine Task IDs --- 
    # Try to extract current_task_id from dataset name if not provided
    if not cl_finetuning_args.current_task_id and data_args.dataset:
        if isinstance(data_args.dataset, list):
            dataset_name = os.path.splitext(os.path.basename(data_args.dataset[-1]))[0]
        else:
            dataset_name = os.path.splitext(os.path.basename(data_args.dataset))[0]
        cl_finetuning_args.current_task_id = dataset_name.upper()
        logger.info_rank0(f"Auto-extracted current_task_id: {cl_finetuning_args.current_task_id}")
        # Assume prev_task_id is the one before the current if multiple datasets provided
        if isinstance(data_args.dataset, list) and len(data_args.dataset) > 1:
            prev_dataset_name = os.path.splitext(os.path.basename(data_args.dataset[-2]))[0]
            cl_finetuning_args.prev_task_id = prev_dataset_name.upper()
            logger.info_rank0(f"Auto-extracted prev_task_id: {cl_finetuning_args.prev_task_id}")
    
    is_first_task = not cl_finetuning_args.prev_task_id

    # --- 2. Load Original Dataset --- 
    logger.info_rank0("Loading original dataset for the current task...")
    orig_dataset_module = get_dataset(
        template=template, 
        model_args=model_args, 
        data_args=data_args, 
        training_args=training_args, 
        stage="sft", 
        **tokenizer_module
    )
    if "train_dataset" not in orig_dataset_module or len(orig_dataset_module["train_dataset"]) == 0:
         raise ValueError("Failed to load a valid training dataset for the current task.")
    logger.info_rank0(f"Loaded original dataset with {len(orig_dataset_module['train_dataset'])} samples.")

    # --- 3. Handle LAMOL Pseudo-Sample Generation and Merging (if not the first task) --- 
    merged_dataset_module = orig_dataset_module # Start with the original dataset

    if not is_first_task:
        logger.info_rank0(f"Current task ({cl_finetuning_args.current_task_id}) is not the first. Proceeding with LAMOL pseudo-sample generation.")
        
        # --- 3a. Initialize LAMOL Generator --- 
        lamol_generator = LAMOLGenerator(
            model_args=model_args,
            data_args=data_args,
            finetuning_args=finetuning_args,
            cl_finetuning_args=cl_finetuning_args,
        )

        # --- 3b. Generate Pseudo Samples --- 
        logger.info_rank0("Generating LAMOL pseudo samples...")
        pseudo_samples = lamol_generator.generate_pseudo_samples()

        if not pseudo_samples:
            logger.warning_rank0("No pseudo samples were generated. Training will continue with only the original dataset.")
        else:
            # --- 3c. Save Pseudo Samples --- 
            logger.info_rank0("Saving generated LAMOL pseudo samples...")
            pseudo_dir = lamol_generator.save_pseudo_samples(pseudo_samples)

            if pseudo_dir:
                # --- 3d. Load Pseudo Samples Dataset --- 
                data_args_pseudo = copy.deepcopy(data_args)
                data_args_pseudo.dataset_dir = pseudo_dir # Use the directory where samples were saved
                data_args_pseudo.dataset = [f"lamol_pseudo_{cl_finetuning_args.current_task_id}"] # Use the dataset name from dataset_info.json
                
                logger.info_rank0(f"Loading LAMOL pseudo samples dataset from: {pseudo_dir}")
                try:
                    dataset_module_pseudo = get_dataset(
                        template=template,
                        model_args=model_args,
                        data_args=data_args_pseudo,
                        training_args=training_args,
                        stage="sft",
                        **tokenizer_module
                    )
                    logger.info_rank0(f"Loaded {len(dataset_module_pseudo.get('train_dataset', []))} pseudo samples.")

                    # --- 3e. Merge Datasets --- 
                    if "train_dataset" in dataset_module_pseudo and len(dataset_module_pseudo["train_dataset"]) > 0:
                        merged_data_args = copy.deepcopy(data_args)
                        merged_data_args.mix_strategy = "concat" # Simple concatenation
                        
                        train_datasets = [
                            orig_dataset_module["train_dataset"],
                            dataset_module_pseudo["train_dataset"]
                        ]
                        merged_train_dataset = merge_dataset(
                            train_datasets,
                            merged_data_args,
                            seed=training_args.seed
                        )
                        merged_dataset_module = {
                            "train_dataset": merged_train_dataset,
                            # Keep original eval dataset unless pseudo eval exists
                            "eval_dataset": orig_dataset_module.get("eval_dataset") 
                        }
                        logger.info_rank0(
                            f"Merged original ({len(orig_dataset_module['train_dataset'])}) and "
                            f"LAMOL pseudo ({len(dataset_module_pseudo['train_dataset'])}) datasets. "
                            f"Total training samples: {len(merged_train_dataset)}."
                        )
                    else:
                        logger.warning_rank0("Failed to load pseudo samples dataset, using only original data.")

                except Exception as e:
                    logger.warning_rank0(f"Error loading or merging pseudo dataset: {e}. Training with original data only.")
            else:
                 logger.warning_rank0("Failed to save pseudo samples, using only original data.")
    else:
        logger.info_rank0("This is the first task, skipping LAMOL pseudo-sample generation.")

    # --- 4. Load Model --- 
    # Load the model AFTER potential pseudo-sample generation (if base model path was not set)
    # The generator might have used the current model state.
    logger.info_rank0("Loading model for training...")
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True) # Compatibility hack

    # --- 5. Initialize Data Collator --- 
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

    # --- 6. Configure Training Arguments --- 
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False # Important for multimodal

    # --- 7. Configure Metrics --- 
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # --- 8. Configure Generation Arguments --- 
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # --- 9. Initialize Trainer --- 
    # Use the potentially merged dataset module
    trainer = LAMOLTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        cl_finetuning_args=cl_finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **merged_dataset_module, # Use the final dataset
        **tokenizer_module,
        **metric_module,
        # gen_kwargs=gen_kwargs, # Pass gen_kwargs if needed by trainer
    )

    # --- 10. Training --- 
    if training_args.do_train:
        # No LAMOL-specific initialization needed in the trainer itself anymore
        logger.info_rank0("Starting LAMOL SFT training...")
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            # Use the merged dataset size for TPS calculation if applicable
            train_dataset_for_tps = merged_dataset_module.get("train_dataset", None)
            if train_dataset_for_tps:
                train_result.metrics["effective_tokens_per_second"] = calculate_tps(
                    train_dataset_for_tps, train_result.metrics, stage="sft"
                )
            else:
                logger.warning_rank0("Could not calculate effective TPS, training dataset not found.")

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"]) # Adjust keys if needed

    # --- 11. Evaluation --- 
    if training_args.predict_with_generate:
        tokenizer.padding_side = "left" # Use left-padding in generation

    if training_args.do_eval:
        eval_dataset = merged_dataset_module.get("eval_dataset", None)
        if eval_dataset:
            metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="eval", **gen_kwargs) # Pass gen_kwargs
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
        else:
             logger.warning_rank0("Evaluation dataset not found, skipping evaluation.")

    # --- 12. Prediction --- 
    if training_args.do_predict:
        logger.warning_rank0_once("Batch generation can be slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_dataset = merged_dataset_module.get("eval_dataset", None) # Usually predict on eval set
        if predict_dataset:
            predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict", **gen_kwargs) # Pass gen_kwargs
            trainer.log_metrics("predict", predict_results.metrics)
            trainer.save_metrics("predict", predict_results.metrics)
            trainer.save_predictions(predict_dataset, predict_results, generating_args.skip_special_tokens)
        else:
             logger.warning_rank0("Prediction dataset not found, skipping prediction.")

    # --- 13. Create Model Card --- 
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
