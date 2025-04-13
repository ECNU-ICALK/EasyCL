# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
# Modifications copyright 2024 Your Name/Org (if applicable)
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
from typing import TYPE_CHECKING, Optional, List, Dict, Any
import random

from llamafactory.data.data_utils import merge_dataset
from llamafactory.data import (
    SFTDataCollatorWith4DAttentionMask, 
    get_dataset, 
    get_template_and_fix_tokenizer
)
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.logging import get_logger
from llamafactory.extras.misc import calculate_tps, get_logits_processor
from llamafactory.extras.ploting import plot_loss
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.trainer_utils import create_modelcard_and_push
from llamafactory.train.sft.metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .gem_trainer import GEMSeq2SeqTrainer

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from llamafactory.hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments
    from easycl.hparams.cl_finetuning_args import CLFinetuningArguments

logger = get_logger(__name__)


def run_sft_gem(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    cl_finetuning_args: "CLFinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    # Load tokenizer and model
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # Log replay settings
    if cl_finetuning_args.use_gem:
        logger.info("\n" + "*" * 80)
        logger.info("*" + " " * 78 + "*")
        logger.info("*" + " " * 28 + "GEM MODE ENABLED" + " " * 28 + "*")
        logger.info("*" + " " * 78 + "*")
        logger.info("*" + f" GEM Memory Strength: {cl_finetuning_args.gem_memory_strength}" + " " * (77 - len(f" GEM Memory Strength: {cl_finetuning_args.gem_memory_strength}")) + "*")
        logger.info("*" + f" Memory Ratio/Max Samples: {cl_finetuning_args.replay_ratio if cl_finetuning_args.maxsamples_list is None else 'List Specified'}" + " " * (77 - len(f" Memory Ratio/Max Samples: {cl_finetuning_args.replay_ratio if cl_finetuning_args.maxsamples_list is None else 'List Specified'}")) + "*")
        logger.info("*" + " " * 78 + "*")
        logger.info("*" * 80 + "\n")

        merged_datasets = []
        gem_datasets_info = []
        
        # 1. First load current task dataset
        current_dataset_module = get_dataset(
            template=template, 
            model_args=model_args, 
            data_args=data_args, 
            training_args=training_args, 
            stage="sft", 
            **tokenizer_module
        )
        
        if "train_dataset" in current_dataset_module:
            current_dataset = current_dataset_module["train_dataset"].map(lambda example: {"is_memory": False})
            merged_datasets.append(current_dataset)
            logger.info(f"Loaded current dataset with {len(current_dataset)} samples (marked as is_memory=False)")
            
            # Record current dataset info
            current_dataset_info = {
                "name": "current_task",
                "dataset": data_args.dataset,
                "size": len(current_dataset),
                "is_memory": False
            }
            gem_datasets_info.append(current_dataset_info)
        
        # 2. If replay_task_list is specified, load memory task data
        if cl_finetuning_args.replay_task_list and training_args.do_train:
            # Save current data directory and dataset configuration
            original_dataset_dir = copy.deepcopy(data_args.dataset_dir)
            original_dataset = copy.deepcopy(data_args.dataset)
            
            # Parse task list for replay (as memory)
            memory_task_list = [task.strip() for task in cl_finetuning_args.replay_task_list.split(',')]
            
            # Parse max samples list (if provided)
            maxsamples_list = None
            if cl_finetuning_args.maxsamples_list:
                try:
                    maxsamples_list = [int(x.strip()) for x in cl_finetuning_args.maxsamples_list.split(',')]
                    if len(maxsamples_list) != len(memory_task_list):
                        logger.warning(f"Length of maxsamples_list ({len(maxsamples_list)}) doesn't match memory_task_list ({len(memory_task_list)}). Will use replay_ratio instead.")
                        maxsamples_list = None
                except ValueError:
                    logger.warning(f"Invalid format in maxsamples_list: {cl_finetuning_args.maxsamples_list}. Will use replay_ratio instead.")
                    maxsamples_list = None
            
            # If previous_task_dataset is provided, use it as data directory
            if cl_finetuning_args.previous_task_dataset:
                data_args.dataset_dir = cl_finetuning_args.previous_task_dataset
                logger.info(f"Using custom dataset directory for memory: {data_args.dataset_dir}")
            
            logger.info(f"Memory task list: {memory_task_list}")
            if maxsamples_list:
                logger.info(f"Max samples per memory task: {maxsamples_list}")
            else:
                logger.info(f"Using memory ratio: {cl_finetuning_args.replay_ratio}")
            
            for task_idx, task_name in enumerate(memory_task_list):
                # Set current memory task
                data_args.dataset = [task_name]
                
                logger.info(f"Loading memory task {task_idx+1}/{len(memory_task_list)}: {task_name}")
                
                try:
                    # Load memory task dataset
                    memory_dataset_module = get_dataset(
                        template=template, 
                        model_args=model_args, 
                        data_args=data_args, 
                        training_args=training_args, 
                        stage="sft", 
                        **tokenizer_module
                    )
                    
                    if "train_dataset" in memory_dataset_module:
                        # Determine sample count
                        total_samples = len(memory_dataset_module["train_dataset"])
                        
                        # Determine max samples based on maxsamples_list or replay_ratio
                        if maxsamples_list and task_idx < len(maxsamples_list):
                            max_samples = min(maxsamples_list[task_idx], total_samples)
                            logger.info(f"Using max samples from list: {max_samples}")
                        else:
                            max_samples = int(total_samples * cl_finetuning_args.replay_ratio)
                            logger.info(f"Using ratio-based max samples: {max_samples}")
                        
                        if max_samples < total_samples:
                            # Randomly select specified number of samples
                            indices = random.sample(range(total_samples), max_samples)
                            memory_dataset_raw = memory_dataset_module["train_dataset"].select(indices)
                            logger.info(f"Selected {max_samples}/{total_samples} samples from task {task_name}")
                        else:
                            memory_dataset_raw = memory_dataset_module["train_dataset"]
                            logger.info(f"Using all {total_samples} samples from task {task_name}")
                        
                        # Add 'is_memory': True flag to memory data
                        memory_dataset = memory_dataset_raw.map(lambda example: {"is_memory": True})
                        merged_datasets.append(memory_dataset)
                        logger.info(f"Marked {len(memory_dataset)} samples from {task_name} as is_memory=True")
                        
                        # Record memory dataset info
                        memory_dataset_info = {
                            "name": task_name,
                            "size_original": total_samples,
                            "size_selected": len(memory_dataset),
                            "is_memory": True
                        }
                        gem_datasets_info.append(memory_dataset_info)
                    else:
                        logger.warning(f"No training data found for memory task: {task_name}")
                        
                except Exception as e:
                    logger.error(f"Failed to load memory task {task_name}: {str(e)}")
                    continue
                    
            # Restore original dataset configuration
            data_args.dataset_dir = original_dataset_dir
            data_args.dataset = original_dataset
        
        # 3. Merge all datasets (Current + Memory)
        if len(merged_datasets) > 0: # Should always be at least 1 if current task data exists
            logger.info(f"Merging {len(merged_datasets)} datasets (current + memory) with strategy: concat")
            merged_data_args = copy.deepcopy(data_args)
            merged_data_args.mix_strategy = "concat"
            
            dataset_module = {}
            dataset_module["train_dataset"] = merge_dataset(
                merged_datasets,
                merged_data_args,
                seed=training_args.seed
            )
            
            # Summarize merged dataset information
            total_samples = len(dataset_module["train_dataset"])
            logger.info("\n" + "#" * 80)
            logger.info("#" + " " * 78 + "#")
            logger.info("#" + " " * 22 + "GEM DATASET MERGE SUMMARY" + " " * 22 + "#")
            logger.info("#" + " " * 78 + "#")
            logger.info("#" + f" Total merged samples: {total_samples}" + " " * (77 - len(f" Total merged samples: {total_samples}")) + "#")
            
            for ds_info in gem_datasets_info:
                type_label = "Memory" if ds_info["is_memory"] else "Current"
                if ds_info["is_memory"]:
                    ds_status = f" {type_label} ({ds_info['name']}): {ds_info['size_selected']}/{ds_info['size_original']} samples"
                else:
                    ds_status = f" {type_label} ({ds_info['name']}): {ds_info['size']} samples"
                logger.info("#" + ds_status + " " * (77 - len(ds_status)) + "#")
            
            logger.info("#" + " " * 78 + "#")
            logger.info("#" * 80 + "\n")
            
            # Keep eval_dataset from current task
            if "eval_dataset" in current_dataset_module:
                dataset_module["eval_dataset"] = current_dataset_module["eval_dataset"]
        else:
            # Handle case where only current data exists (no memory tasks specified)
            logger.info("No memory tasks specified or loaded. Using only current task data.")
            dataset_module = current_dataset_module
            # Ensure 'is_memory' column exists even if only current data
            if "train_dataset" in dataset_module:
                 dataset_module["train_dataset"] = dataset_module["train_dataset"].map(lambda example: {"is_memory": False})

    else:
        # Load regular dataset when GEM is not enabled
        dataset_module = get_dataset(
            template=template, 
            model_args=model_args, 
            data_args=data_args, 
            training_args=training_args, 
            stage="sft", 
            **tokenizer_module
        )
        # Ensure 'is_memory' column exists and is False for standard SFT
        if "train_dataset" in dataset_module and training_args.do_train:
             logger.info("GEM not enabled, marking all training data as is_memory=False for compatibility.")
             # Check if column already exists to avoid errors
             if "is_memory" not in dataset_module["train_dataset"].column_names:
                 dataset_module["train_dataset"] = dataset_module["train_dataset"].map(lambda example: {"is_memory": False})
             else: # If it somehow exists, ensure it's boolean False
                 logger.warning("'is_memory' column already exists in non-GEM mode. Ensuring values are False.")
                 dataset_module["train_dataset"] = dataset_module["train_dataset"].map(lambda example: {"is_memory": False})

    
    # Now load model (after dataset preparation since we might need to adjust it)
    model = load_model(
        tokenizer=tokenizer,
        model_args=model_args,
        finetuning_args=finetuning_args,
        is_trainable=training_args.do_train
    )

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    # Initialize data collator
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        tokenizer=tokenizer,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )

    # Set up training arguments
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams

    # Set up metrics
    metric_kwargs = {}
    if training_args.predict_with_generate:
        metric_kwargs["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_kwargs["compute_metrics"] = ComputeAccuracy()
        metric_kwargs["preprocess_logits_for_metrics"] = eval_logit_processor

    # Set up generation arguments if needed
    if training_args.predict_with_generate or training_args.do_eval:
        gen_kwargs = generating_args.to_dict()
        gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
        gen_kwargs["logits_processor"] = get_logits_processor()
    else:
        gen_kwargs = None

    # Initialize the GEM trainer
    trainer = GEMSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        cl_finetuning_args=cl_finetuning_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **metric_kwargs
    )

    # Training
    if training_args.do_train:
        logger.info("*** Training ***")
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        
        # Calculate effective tokens per second if requested
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_second"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

    # Evaluation
    if training_args.do_eval and "eval_dataset" in dataset_module:
        if training_args.predict_with_generate:
            # Set left padding for generation
            tokenizer.padding_side = "left"
            
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            eval_dataset=dataset_module["eval_dataset"],
            metric_key_prefix="eval",
            **gen_kwargs
        )
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Model card and push to hub if requested
    if training_args.push_to_hub:
        create_modelcard_and_push(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            finetuning_args=finetuning_args,
            model_name=model_args.model_name_or_path.split("/")[-1],
        )

    # Plot training loss if enabled
    if trainer.is_world_process_zero() and finetuning_args.plot_loss:
        logger.info("Plotting training loss...")
        plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])
        
    return trainer, tokenizer
