import os
import copy
import torch
import gc
import random
import math
from typing import TYPE_CHECKING, Optional

from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from llamafactory.data.data_utils import merge_dataset
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.logging import get_logger
from llamafactory.extras.misc import calculate_tps, get_logits_processor
from llamafactory.extras.ploting import plot_loss
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.trainer_utils import create_modelcard_and_push
from llamafactory.train.sft.metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from .dynamic_conpet_trainer import DynamicConPetTrainer
from .dynamic_conpet import DatasetClassifier, save_classifier, load_classifier

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from llamafactory.hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments
    from easycl.hparams import CLFinetuningArguments


logger = get_logger(__name__)


def run_sft_dynamic_conpet(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    cl_finetuning_args: "CLFinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    """
    Run sequence-to-sequence fine-tuning with Dynamic ConPet method
    This method will train two adapters: a shared adapter and a task-specific adapter
    """
    # Load tokenizer and dataset (only once)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # ============= Prepare current and historical datasets (1:1 ratio) =============
    if training_args.do_train:
        logger.info("\n" + "*" * 80)
        logger.info("*" + " " * 78 + "*")
        logger.info("*" + " " * 20 + "LOADING CURRENT AND HISTORICAL DATASETS (1:1 RATIO)" + " " * 20 + "*")
        logger.info("*" + " " * 78 + "*")
        logger.info("*" * 80 + "\n")

        merged_datasets = []
        datasets_info = []

        # Load current task dataset
        current_dataset_module = get_dataset(
            template=template,
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            stage="sft",
            **tokenizer_module
        )
        
        current_dataset = None
        current_dataset_size = 0
        if "train_dataset" in current_dataset_module:
            current_dataset = current_dataset_module["train_dataset"]
            current_dataset_size = len(current_dataset)
            logger.info(f"Loaded current dataset with {current_dataset_size} samples")
            
            # Add to merged datasets
            merged_datasets.append(current_dataset)
            datasets_info.append({
                "name": "current_task",
                "dataset": data_args.dataset,
                "size": current_dataset_size
            })

        # Load historical task datasets
        historical_datasets = []
        if cl_finetuning_args.replay_task_list and current_dataset_size > 0:
            original_dataset_dir = copy.deepcopy(data_args.dataset_dir)
            original_dataset = copy.deepcopy(data_args.dataset)
            replay_task_list = [task.strip() for task in cl_finetuning_args.replay_task_list.split(',')]
            
            # Calculate number of samples to load for each historical task
            samples_per_task = math.ceil(current_dataset_size / len(replay_task_list))
            
            maxsamples_list = None
            if cl_finetuning_args.maxsamples_list:
                maxsamples_list = [int(x.strip()) for x in cl_finetuning_args.maxsamples_list.split(',')]
                
            if cl_finetuning_args.previous_task_dataset:
                data_args.dataset_dir = cl_finetuning_args.previous_task_dataset

            for task_idx, task_name in enumerate(replay_task_list):
                data_args.dataset = [task_name]
                try:
                    replay_dataset_module = get_dataset(
                        template=template,
                        model_args=model_args,
                        data_args=data_args,
                        training_args=training_args,
                        stage="sft",
                        **tokenizer_module
                    )
                    if "train_dataset" in replay_dataset_module:
                        total_samples = len(replay_dataset_module["train_dataset"])
                        
                        # Determine number of samples to load
                        max_samples = (
                            min(maxsamples_list[task_idx], total_samples)
                            if maxsamples_list and task_idx < len(maxsamples_list)
                            else min(samples_per_task, total_samples)
                        )
                        
                        if max_samples < total_samples:
                            indices = random.sample(range(total_samples), max_samples)
                            replay_dataset = replay_dataset_module["train_dataset"].select(indices)
                            logger.info(f"Selected {max_samples}/{total_samples} samples from historical task {task_name}")
                        else:
                            replay_dataset = replay_dataset_module["train_dataset"]
                            logger.info(f"Using all {total_samples} samples from historical task {task_name}")
                        
                        historical_datasets.append(replay_dataset)
                        datasets_info.append({
                            "name": task_name,
                            "size": len(replay_dataset)
                        })
                except Exception as e:
                    logger.error(f"Failed to load historical task {task_name}: {str(e)}")
                    continue

            data_args.dataset_dir = original_dataset_dir
            data_args.dataset = original_dataset

            # Merge historical datasets
            if historical_datasets:
                # Calculate total samples in historical datasets
                total_historical_samples = sum(len(ds) for ds in historical_datasets)
                logger.info(f"Total historical samples: {total_historical_samples}")
                
                # Add historical datasets to merge list
                merged_datasets.extend(historical_datasets)

        # Merge all datasets
        if len(merged_datasets) > 0:
            dataset_module = {
                "train_dataset": merge_dataset(
                    merged_datasets,
                    data_args,
                    seed=training_args.seed
                )
            }
            logger.info(f"Final merged dataset size: {len(dataset_module['train_dataset'])} samples")
            
            # Add evaluation dataset
            if "eval_dataset" in current_dataset_module:
                dataset_module["eval_dataset"] = current_dataset_module["eval_dataset"]
        else:
            dataset_module = current_dataset_module
            logger.info("Using only current dataset")
    else:
        dataset_module = get_dataset(
            template=template,
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            stage="sft",
            **tokenizer_module
        )
    
    # Save original output directory
    original_output_dir = training_args.output_dir
    
    # Ensure adapters_save_path exists
    adapters_save_path = cl_finetuning_args.adapters_save_path
    if not adapters_save_path:
        # If not specified, use output_dir as default path
        adapters_save_path = os.path.join(original_output_dir, "adapters")
    
    os.makedirs(adapters_save_path, exist_ok=True)
    logger.info_rank0(f"Adapters will be saved to: {adapters_save_path}")

    # =========================== Train shared adapter ===========================
    adapter_name = "shared_adapter"
    adapter_output_dir = os.path.join(adapters_save_path, adapter_name)
    
    # Set shared adapter output directory
    training_args = copy.deepcopy(training_args)  # Create copy to avoid modifying original object
    training_args.output_dir = adapter_output_dir
    training_args.overwrite_output_dir = True  # Overwrite existing adapter
    
    os.makedirs(training_args.output_dir, exist_ok=True)
    logger.info_rank0(f"Training shared adapter: {adapter_name}, output directory: {training_args.output_dir}")
    
    # Create model args copy for shared adapter
    model_args_copy = copy.deepcopy(model_args)
    finetuning_args_copy = copy.deepcopy(finetuning_args)
    cl_finetuning_args_copy = copy.deepcopy(cl_finetuning_args)
    
    # Check if pretrained shared adapter exists, check by adapter_config.json file
    if os.path.exists(os.path.join(adapter_output_dir, "adapter_config.json")):
        # If pretrained shared adapter exists, load it
        model_args_copy.adapter_name_or_path = [adapter_output_dir]
        logger.info_rank0(f"Loading pretrained shared adapter from: {adapter_output_dir}")
    else:
        # Otherwise initialize new shared adapter
        model_args_copy.adapter_name_or_path = None
        logger.info_rank0(f"No pretrained shared adapter found, initializing a new one")
    
    # Load model (with shared adapter if exists)
    model = load_model(tokenizer, model_args_copy, finetuning_args_copy, training_args.do_train)
    
    # Create dataset classifier
    dataset_classifier = None
    dataset_names = []
    dataset_indices_map = {}
    
    if training_args.do_train and len(merged_datasets) > 0:
        # Set dataset information
        num_datasets = len(datasets_info)
        dataset_names = [info["name"] for info in datasets_info]
        
        # Get model hidden size
        hidden_size = model.config.hidden_size
        
        # Check if pretrained classifier exists
        classifier_path = os.path.join(adapters_save_path, "shared_adapter", "dataset_classifier")
        if os.path.exists(os.path.join(classifier_path, "classifier_config.json")):
            # Load and expand classifier
            logger.info_rank0(f"Loading and expanding dataset classifier from: {classifier_path}")
            dataset_classifier, old_dataset_names = load_classifier(
                classifier_path, 
                hidden_size, 
                new_num_datasets=num_datasets
            )
            # Record historical dataset names
            logger.info_rank0(f"Loaded classifier for datasets: {old_dataset_names}")
            logger.info_rank0(f"Expanded for current datasets: {dataset_names}")
        else:
            # Create new dataset classifier
            dataset_classifier = DatasetClassifier(hidden_size, num_datasets)
            logger.info_rank0(f"Created new dataset classifier for {num_datasets} datasets")
        
        # Build dataset index mapping for identifying which dataset a sample belongs to during training
        start_idx = 0
        for idx, ds_info in enumerate(datasets_info):
            end_idx = start_idx + ds_info["size"]
            dataset_indices_map[(start_idx, end_idx)] = idx
            start_idx = end_idx
        
        logger.info_rank0(f"Dataset classifier set up with {num_datasets} classes")
        for idx, name in enumerate(dataset_names):
            logger.info_rank0(f"  Dataset {idx}: {name}")

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args_copy.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args_copy.compute_dtype,
        **tokenizer_module,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False  # important for multimodal dataset

    # Metric utils
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args_copy.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Create callbacks copy for shared adapter
    current_callbacks = copy.deepcopy(callbacks) if callbacks is not None else []

    # Initialize Trainer
    if dataset_classifier is not None:
        # Use DynamicConPetTrainer
        trainer = DynamicConPetTrainer(
            model=model,
            args=training_args,
            finetuning_args=finetuning_args_copy,
            cl_finetuning_args=cl_finetuning_args_copy,
            data_collator=data_collator,
            callbacks=current_callbacks,
            dataset_classifier=dataset_classifier,
            dataset_names=dataset_names,
            dataset_indices_map=dataset_indices_map,
            gen_kwargs=gen_kwargs,
            **dataset_module,
            **tokenizer_module,
            **metric_module,
        )
        logger.info_rank0("Using DynamicConPetTrainer with dataset classification")
    else:
        # Use standard Trainer
        trainer = CustomSeq2SeqTrainer(
            model=model,
            args=training_args,
            finetuning_args=finetuning_args_copy,
            cl_finetuning_args=cl_finetuning_args_copy,
            data_collator=data_collator,
            callbacks=current_callbacks,
            gen_kwargs=gen_kwargs,
            **dataset_module,
            **tokenizer_module,
            **metric_module,
        )
        logger.info_rank0("Using standard CustomSeq2SeqTrainer")

    # Train shared adapter
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()  # Save to adapter_output_dir
        
        if finetuning_args_copy.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        if datasets_info:
            for idx, ds_info in enumerate(datasets_info):
                prefix = f"shared_adapter_dataset_{idx}"
                for key, value in ds_info.items():
                    # Ensure all values are formattable basic types (string, number, etc.)
                    if isinstance(value, (str, int, float, bool)):
                        train_result.metrics[f"{prefix}_{key}"] = value
                    elif isinstance(value, list):
                        # Convert list to string
                        train_result.metrics[f"{prefix}_{key}"] = str(value)
                    else:
                        train_result.metrics[f"{prefix}_{key}"] = str(value)
            
            if cl_finetuning_args.replay_task_list:
                # Ensure it's a string, not a list
                if isinstance(replay_task_list, list):
                    task_list_str = ','.join(replay_task_list)
                else:
                    task_list_str = str(replay_task_list)
                train_result.metrics["shared_adapter_replay_task_list"] = task_list_str

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args_copy.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy", "classification_loss", "total_loss"])

    # Clean up memory for task-specific adapter training
    del model
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================== Train task-specific adapter ===========================
    if not cl_finetuning_args.current_task_id:
        logger.warning_rank0("No current_task_id specified, using 'task' as default task ID")
        task_id = "task"
    else:
        task_id = cl_finetuning_args.current_task_id

    # Use same dataset as shared adapter, no need to reload
    logger.info("\n" + "*" * 80)
    logger.info("*" + " " * 78 + "*")
    logger.info("*" + " " * 15 + "USING SAME DATASET FOR TASK ADAPTER AS FOR SHARED ADAPTER" + " " * 15 + "*")
    logger.info("*" + " " * 78 + "*")
    logger.info("*" * 80 + "\n")

    adapter_name = task_id
    adapter_output_dir = os.path.join(adapters_save_path, adapter_name)
    
    # Set task-specific adapter output directory
    training_args = copy.deepcopy(training_args)  # Create new copy
    training_args.output_dir = adapter_output_dir
    
    os.makedirs(training_args.output_dir, exist_ok=True)
    logger.info_rank0(f"Training task-specific adapter: {adapter_name}, output directory: {training_args.output_dir}")
    
    # Create model args copy for task-specific adapter
    model_args_copy = copy.deepcopy(model_args)
    finetuning_args_copy = copy.deepcopy(finetuning_args)
    cl_finetuning_args_copy = copy.deepcopy(cl_finetuning_args)
    
    # Ensure shared adapter path exists
    shared_adapter_dir = os.path.join(adapters_save_path, "shared_adapter")
    if not os.path.exists(shared_adapter_dir) or not os.path.exists(os.path.join(shared_adapter_dir, "adapter_config.json")):
        logger.warning_rank0(f"Warning: Shared adapter not found: {shared_adapter_dir}, will continue but may not be able to compute orthogonal loss")
    
    # Set adapters_save_path in cl_finetuning_args_copy
    cl_finetuning_args_copy.adapters_save_path = adapters_save_path  
    
    # Only load base model, don't preload any adapters
    model_args_copy.adapter_name_or_path = None
    
    logger.info_rank0(f"Loading base model and initializing new task-specific adapter: {task_id}")
    
    # Load model (randomly initialize task-specific adapter)
    model = load_model(tokenizer, model_args_copy, finetuning_args_copy, training_args.do_train)
    
    # Check and log adapter configurations after model creation
    if hasattr(model, "peft_config"):
        for adapter_name, config in model.peft_config.items():
            logger.info_rank0(f"Adapter '{adapter_name}' configuration: {config}")
            if hasattr(config, "target_modules"):
                logger.info_rank0(f"Adapter '{adapter_name}' target modules: {config.target_modules}")

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args_copy.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args_copy.compute_dtype,
        **tokenizer_module,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False  # important for multimodal dataset

    # Metric utils
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args_copy.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Create callbacks copy for task-specific adapter
    current_callbacks = copy.deepcopy(callbacks) if callbacks is not None else []

    # Initialize Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args_copy,
        cl_finetuning_args=cl_finetuning_args_copy,
        data_collator=data_collator,
        callbacks=current_callbacks,
        gen_kwargs=gen_kwargs,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    # Train task-specific adapter
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()  # Save to adapter_output_dir
        
        if finetuning_args_copy.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )
            
        # Record used adapter information
        train_result.metrics["shared_adapter"] = os.path.join(adapters_save_path, "shared_adapter")
        train_result.metrics["task_adapter"] = os.path.join(adapters_save_path, adapter_name)
        
        # Add classifier related information
        classifier_path = os.path.join(adapters_save_path, "shared_adapter", "dataset_classifier")
        if os.path.exists(os.path.join(classifier_path, "classifier_config.json")):
            train_result.metrics["dataset_classifier"] = classifier_path
        
        # Log to logger
        logger.info(f"Dynamic ConPet Shared Adapter: {os.path.join(adapters_save_path, 'shared_adapter')}")
        logger.info(f"Dynamic ConPet Task Adapter: {os.path.join(adapters_save_path, adapter_name)}")

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args_copy.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy", "dynamic_conpet_loss", "shared_l2_loss"])

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluate task-specific adapter
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Generate predictions with task-specific adapter
    if training_args.do_predict:
        logger.warning_rank0("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    # Create model card
    create_modelcard_and_push(trainer, model_args_copy, data_args, training_args, finetuning_args_copy)
    
    # Clean up memory
    del model
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Restore original output directory
    logger.info_rank0(f"Dynamic ConPet training completed. Adapters saved to {adapters_save_path}")
    logger.info_rank0(f"  - Shared adapter: {os.path.join(adapters_save_path, 'shared_adapter')}")
    logger.info_rank0(f"  - Task-specific adapter: {os.path.join(adapters_save_path, task_id)}")
    
    return