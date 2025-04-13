import os
import copy
from typing import Optional, List, TYPE_CHECKING

import torch
from transformers import Seq2SeqTrainingArguments

from llamafactory.data import (
    get_dataset, 
    get_template_and_fix_tokenizer
)
from llamafactory.data.data_utils import merge_dataset
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.logging import get_logger
from llamafactory.extras.misc import calculate_tps, get_logits_processor
from llamafactory.extras.ploting import plot_loss
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.trainer_utils import create_modelcard_and_push
from llamafactory.train.sft.metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from llamafactory.data import SFTDataCollatorWith4DAttentionMask

from .pseudo_replay import PseudoReplay
from .pseudo_replay_trainer import PseudoReplayTrainer

if TYPE_CHECKING:
    from llamafactory.hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments
    from easycl.hparams import CLFinetuningArguments
    from transformers import TrainerCallback

logger = get_logger(__name__)


def run_sft_pseudo_replay(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    cl_finetuning_args: "CLFinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None
):
    """
    Run Pseudo Replay method for continual learning
    """
    # Check if this is the first task
    is_first_task = not cl_finetuning_args.prev_task_id
    
    # Load tokenizer
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # For the first task, skip pseudo sample generation and train normally
    if is_first_task:
        logger.info_rank0("This is the first task, skipping pseudo sample generation and proceeding with normal training")
        
        # Load current task dataset
        dataset_module = get_dataset(
            template=template,
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            stage="sft",
            **tokenizer_module
        )
        
        # Load model
        model = load_model(
            tokenizer=tokenizer,
            model_args=model_args,
            finetuning_args=finetuning_args,
            is_trainable=training_args.do_train,
        )
        
        # Regular training workflow
        run_regular_training(
            model=model,
            tokenizer_module=tokenizer_module,
            template=template,
            dataset_module=dataset_module,
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            finetuning_args=finetuning_args,
            cl_finetuning_args=cl_finetuning_args,
            generating_args=generating_args,
            callbacks=callbacks
        )
        
        return
    
    # For subsequent tasks, need to generate and process pseudo samples
    logger.info_rank0(f"This is a subsequent task ({cl_finetuning_args.current_task_id}), need to generate pseudo samples for continual learning")
    
    # Initialize PseudoReplay method
    pseudo_replay = PseudoReplay(model_args, data_args, finetuning_args, cl_finetuning_args)
    
    # Load original dataset
    logger.info_rank0("Loading original dataset for training and pseudo sample generation")
    orig_dataset_module = get_dataset(
        template=template,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        stage="sft",
        **tokenizer_module
    )
    
    # Check if dataset loaded successfully
    if "train_dataset" not in orig_dataset_module or len(orig_dataset_module["train_dataset"]) == 0:
        raise ValueError(f"Failed to load valid training dataset, please check dataset configuration")
    
    # Generate pseudo samples using PseudoReplay method
    logger.info_rank0("Starting pseudo sample generation...")
    pseudo_samples = pseudo_replay.generate_pseudo_samples(orig_dataset_module["train_dataset"])
    
    # Save pseudo samples
    logger.info_rank0("Saving pseudo samples...")
    pseudo_dir = pseudo_replay.save_pseudo_samples(
        pseudo_samples, 
        cl_finetuning_args.current_task_id,
        cl_finetuning_args.prev_task_id
    )
    
    # Prepare two sets of data arguments
    data_args_orig = copy.deepcopy(data_args)  # Original data arguments
    data_args_pseudo = copy.deepcopy(data_args)  # Pseudo sample data arguments
    
    # Fix: Use complete directory path for pseudo samples
    pseudo_samples_dir = cl_finetuning_args.pseudo_samples_dir
    if not os.path.isabs(pseudo_samples_dir):
        pseudo_samples_dir = os.path.join(os.getcwd(), pseudo_samples_dir)
    
    # Use pseudo sample path for current task
    current_pseudo_dir = os.path.join(pseudo_samples_dir, cl_finetuning_args.current_task_id)
    
    # Set different data directory and dataset name
    data_args_pseudo.dataset_dir = current_pseudo_dir
    data_args_pseudo.dataset = [f"pseudo_{cl_finetuning_args.current_task_id}"]
    
    logger.info_rank0(f"Loading pseudo sample dataset from: {current_pseudo_dir}")
    
    # Load pseudo sample dataset
    logger.info_rank0("Loading pseudo sample dataset...")
    try:
        dataset_module_pseudo = get_dataset(
            template=template,
            model_args=model_args,
            data_args=data_args_pseudo,
            training_args=training_args,
            stage="sft",
            **tokenizer_module
        )
    except Exception as e:
        logger.warning_rank0(f"Failed to load pseudo sample dataset: {e}")
        logger.warning_rank0("Will only use original dataset for training")
        dataset_module_pseudo = {}
    
    # Merge training sets
    merged_module = {}
    if "train_dataset" in orig_dataset_module and "train_dataset" in dataset_module_pseudo:
        # Set merge strategy
        merged_data_args = copy.deepcopy(data_args)
        merged_data_args.mix_strategy = "concat"  # Simple concatenation strategy
        
        # Merge training sets
        train_datasets = [
            orig_dataset_module["train_dataset"],
            dataset_module_pseudo["train_dataset"]
        ]
        merged_module["train_dataset"] = merge_dataset(
            train_datasets,
            merged_data_args,
            seed=training_args.seed
        )
        
        logger.info_rank0(f"Successfully merged original dataset ({len(orig_dataset_module['train_dataset'])}) and pseudo sample dataset ({len(dataset_module_pseudo['train_dataset'])}), total samples: {len(merged_module['train_dataset'])}")
    elif "train_dataset" in orig_dataset_module:
        merged_module["train_dataset"] = orig_dataset_module["train_dataset"]
        logger.warning_rank0(f"Only using original dataset ({len(orig_dataset_module['train_dataset'])}) for training")
    elif "train_dataset" in dataset_module_pseudo:
        merged_module["train_dataset"] = dataset_module_pseudo["train_dataset"]
        logger.warning_rank0(f"Only using pseudo sample dataset ({len(dataset_module_pseudo['train_dataset'])}) for training")
    else:
        raise ValueError("Failed to load valid training dataset, please check dataset configuration")
        
    # Merge validation sets (if exist)
    eval_dataset = {}
    if "eval_dataset" in orig_dataset_module:
        if isinstance(orig_dataset_module["eval_dataset"], dict):
            eval_dataset.update(orig_dataset_module["eval_dataset"])
        else:
            eval_dataset["orig"] = orig_dataset_module["eval_dataset"]
            
    if "eval_dataset" in dataset_module_pseudo:
        if isinstance(dataset_module_pseudo["eval_dataset"], dict):
            eval_dataset.update(dataset_module_pseudo["eval_dataset"])
        else:
            eval_dataset["pseudo"] = dataset_module_pseudo["eval_dataset"]
            
    if eval_dataset:
        merged_module["eval_dataset"] = eval_dataset
    
    # Load model with merged dataset
    model = load_model(
        tokenizer=tokenizer,
        model_args=model_args,
        finetuning_args=finetuning_args,
        is_trainable=training_args.do_train,
    )
    
    # Regular training workflow
    run_regular_training(
        model=model,
        tokenizer_module=tokenizer_module,
        template=template,
        dataset_module=merged_module,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        cl_finetuning_args=cl_finetuning_args,
        generating_args=generating_args,
        callbacks=callbacks
    )


def run_regular_training(
    model,
    tokenizer_module,
    template,
    dataset_module,
    model_args,
    data_args,
    training_args,
    finetuning_args,
    cl_finetuning_args,
    generating_args,
    callbacks=None
):
    """
    Execute regular training workflow
    """
    tokenizer = tokenizer_module["tokenizer"]
    
    # Data collator
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
    
    # Training arguments configuration
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False
    
    # Metric computation configuration
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor
    
    # Generation arguments configuration
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()
    
    # Initialize trainer
    trainer = PseudoReplayTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs=gen_kwargs,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )
    
    # Start training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )
            
        # Log pseudo-replay parameters
        train_result.metrics["num_pseudo_samples"] = num_pseudo_samples
        train_result.metrics["pseudo_lambda"] = cl_finetuning_args.pseudo_lambda
        train_result.metrics["pseudo_loss_type"] = cl_finetuning_args.pseudo_loss_type

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])
    
    # Prediction configuration
    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"
        
    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
    # Prediction
    if training_args.do_predict:
        logger.warning_rank0("Batch generation may be slow. Consider using `scripts/vllm_infer.py` instead.")
        dataset_to_predict = dataset_module.get("eval_dataset", {})
        predict_results = trainer.predict(dataset_to_predict, metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_to_predict, predict_results, generating_args.skip_special_tokens)
        
    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
