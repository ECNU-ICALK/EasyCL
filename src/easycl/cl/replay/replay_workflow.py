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
from .replay_trainer import ReplaySeq2SeqTrainer

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from llamafactory.hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments
    from easycl.hparams.cl_finetuning_args import CLFinetuningArguments

logger = get_logger(__name__)


def run_sft_replay(
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
    
    # Record replay settings
    if cl_finetuning_args.use_replay:
        logger.info("\n" + "*" * 80)
        logger.info("*" + " " * 78 + "*")
        logger.info("*" + " " * 30 + "REPLAY MODE ENABLED" + " " * 30 + "*")
        logger.info("*" + " " * 78 + "*")
        logger.info("*" + f" Replay Ratio: {cl_finetuning_args.replay_ratio}" + " " * (77 - len(f" Replay Ratio: {cl_finetuning_args.replay_ratio}")) + "*")
        logger.info("*" + " " * 78 + "*")
        logger.info("*" * 80 + "\n")

        merged_datasets = []
        replay_datasets_info = []
        
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
            merged_datasets.append(current_dataset_module["train_dataset"])
            logger.info(f"Loaded current dataset with {len(current_dataset_module['train_dataset'])} samples")
            
            # Record current dataset info
            current_dataset_info = {
                "name": "current_task",
                "dataset": data_args.dataset,
                "size": len(current_dataset_module["train_dataset"])
            }
            replay_datasets_info.append(current_dataset_info)
        
        # 2. If replay_task_list is specified, load replay task data
        if cl_finetuning_args.replay_task_list and training_args.do_train:
            # Save current data directory and dataset configuration
            original_dataset_dir = copy.deepcopy(data_args.dataset_dir)
            original_dataset = copy.deepcopy(data_args.dataset)
            
            # Parse replay task list and maximum samples per task
            replay_task_list = [task.strip() for task in cl_finetuning_args.replay_task_list.split(',')]
            
            # Parse maximum samples list (if provided)
            maxsamples_list = None
            if cl_finetuning_args.maxsamples_list:
                try:
                    maxsamples_list = [int(x.strip()) for x in cl_finetuning_args.maxsamples_list.split(',')]
                    # Ensure maxsamples_list and replay_task_list have same length
                    if len(maxsamples_list) != len(replay_task_list):
                        logger.warning(f"Length of maxsamples_list ({len(maxsamples_list)}) doesn't match replay_task_list ({len(replay_task_list)}). Will use replay_ratio instead.")
                        maxsamples_list = None
                except ValueError:
                    logger.warning(f"Invalid format in maxsamples_list: {cl_finetuning_args.maxsamples_list}. Will use replay_ratio instead.")
                    maxsamples_list = None
            
            # If previous_task_dataset is provided, use it as data directory
            if cl_finetuning_args.previous_task_dataset:
                data_args.dataset_dir = cl_finetuning_args.previous_task_dataset
                logger.info(f"Using custom dataset directory: {data_args.dataset_dir}")
            
            logger.info(f"Replay task list: {replay_task_list}")
            if maxsamples_list:
                logger.info(f"Max samples per task: {maxsamples_list}")
            else:
                logger.info(f"Using replay ratio: {cl_finetuning_args.replay_ratio}")
            
            for task_idx, task_name in enumerate(replay_task_list):
                # Set current replay task
                data_args.dataset = [task_name]
                
                logger.info(f"Loading replay task {task_idx+1}/{len(replay_task_list)}: {task_name}")
                
                try:
                    # Load replay task dataset
                    replay_dataset_module = get_dataset(
                        template=template, 
                        model_args=model_args, 
                        data_args=data_args, 
                        training_args=training_args, 
                        stage="sft", 
                        **tokenizer_module
                    )
                    
                    if "train_dataset" in replay_dataset_module:
                        # Determine sample count
                        total_samples = len(replay_dataset_module["train_dataset"])
                        
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
                            replay_dataset = replay_dataset_module["train_dataset"].select(indices)
                            logger.info(f"Selected {max_samples}/{total_samples} samples from task {task_name}")
                        else:
                            replay_dataset = replay_dataset_module["train_dataset"]
                            logger.info(f"Using all {total_samples} samples from task {task_name}")
                        
                        merged_datasets.append(replay_dataset)
                        
                        # Record replay dataset info
                        replay_dataset_info = {
                            "name": task_name,
                            "size_original": total_samples,
                            "size_selected": len(replay_dataset)
                        }
                        replay_datasets_info.append(replay_dataset_info)
                    else:
                        logger.warning(f"No training data found for replay task: {task_name}")
                        
                except Exception as e:
                    logger.error(f"Failed to load replay task {task_name}: {str(e)}")
                    continue
                    
            # Restore original dataset configuration
            data_args.dataset_dir = original_dataset_dir
            data_args.dataset = original_dataset
        
        # 3. Merge all datasets
        if len(merged_datasets) > 1:
            logger.info(f"Merging {len(merged_datasets)} datasets with strategy: concat")
            # Use concat strategy to merge datasets
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
            logger.info("#" + " " * 25 + "DATASET MERGE SUMMARY" + " " * 25 + "#")
            logger.info("#" + " " * 78 + "#")
            logger.info("#" + f" Total merged samples: {total_samples}" + " " * (77 - len(f" Total merged samples: {total_samples}")) + "#")
            
            for ds_info in replay_datasets_info:
                if "size_selected" in ds_info:
                    # Replay dataset
                    ds_status = f" {ds_info['name']}: {ds_info['size_selected']}/{ds_info['size_original']} samples"
                else:
                    # Current dataset
                    ds_status = f" {ds_info['name']}: {ds_info['size']} samples"
                logger.info("#" + ds_status + " " * (77 - len(ds_status)) + "#")
            
            logger.info("#" + " " * 78 + "#")
            logger.info("#" * 80 + "\n")
            
            # Keep evaluation dataset
            if "eval_dataset" in current_dataset_module:
                dataset_module["eval_dataset"] = current_dataset_module["eval_dataset"]
        else:
            # Only current dataset, use it directly
            dataset_module = current_dataset_module
    else:
        # Replay not enabled, use regular dataset loading
        dataset_module = get_dataset(
            template=template, 
            model_args=model_args, 
            data_args=data_args, 
            training_args=training_args, 
            stage="sft", 
            **tokenizer_module
        )
    
    # Load model
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    # Configure data collator
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
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
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Initialize our Trainer
    trainer = ReplaySeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        cl_finetuning_args=cl_finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_second"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )
        
        # If replay info exists, add to metrics
        if cl_finetuning_args.use_replay and replay_datasets_info:
            for idx, ds_info in enumerate(replay_datasets_info):
                prefix = f"dataset_{idx}"
                for key, value in ds_info.items():
                    # Ensure all values are formattable (strings, numbers, etc.)
                    if isinstance(value, (str, int, float, bool)):
                        train_result.metrics[f"{prefix}_{key}"] = value
                    else:
                        train_result.metrics[f"{prefix}_{key}"] = str(value)
            
            # Add replay task list to metrics
            if cl_finetuning_args.replay_task_list:
                # Use string format instead of list
                task_list_str = ','.join(
                    [task.strip() for task in cl_finetuning_args.replay_task_list.split(',')]
                )
                train_result.metrics["replay_task_list"] = task_list_str
        
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])

    # Use left padding in generation
    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.warning("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
