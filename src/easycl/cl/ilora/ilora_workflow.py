# Copyright 2025 the LlamaFactory team.
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

import os  # 添加此行导入os模块
from typing import TYPE_CHECKING, Optional

from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.logging import get_logger
from llamafactory.extras.misc import calculate_tps, get_logits_processor
from llamafactory.extras.ploting import plot_loss
from llamafactory.model import load_tokenizer
from llamafactory.train.sft.metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from llamafactory.train.trainer_utils import create_modelcard_and_push
from easycl.cl.ilora.ilora_loader import load_ilora_model
from easycl.cl.ilora.ilora_trainer import ILORATrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from llamafactory.hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments
    from easycl.hparams.cl_finetuning_args import CLFinetuningArguments


logger = get_logger(__name__)


def run_sft_ilora(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    cl_finetuning_args: "CLFinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    """
    Runs the supervised fine-tuning process with I-LORA.
    
    This function is based on run_sft but uses I-LORA specific components.
    """
    logger.info_rank0("Running SFT with I-LORA...")
    
    # Determine current task type (first task or subsequent task)
    is_first_task = cl_finetuning_args.prev_task_id is None
    if is_first_task:
        logger.info_rank0("Detected first task")
    else:
        logger.info_rank0(f"Detected subsequent task, previous task ID: {cl_finetuning_args.prev_task_id}")
        
        # If previous_task_model is specified, print path for confirmation
        if cl_finetuning_args.previous_task_model:
            logger.info_rank0(f"Using specified previous task model path: {cl_finetuning_args.previous_task_model}")
            
            # Check for EMA adapter existence
            ema_path = os.path.join(cl_finetuning_args.previous_task_model, "ema")
            ema_adapter_path = os.path.join(cl_finetuning_args.previous_task_model, "ema_adapter")
            
            if os.path.exists(ema_path):
                logger.info_rank0(f"Found previous task's EMA adapter: {ema_path}")
            elif os.path.exists(ema_adapter_path):
                logger.info_rank0(f"Found previous task's EMA adapter: {ema_adapter_path}")
            else:
                logger.warning_rank0(f"No EMA adapter found in {cl_finetuning_args.previous_task_model}")
    
    # Load tokenizer and get template
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # Get dataset
    dataset_module = get_dataset(
        template=template, 
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        stage="sft",
        **tokenizer_module
    )
    
    # Infer previous task path before loading model (if needed)
    if not is_first_task and not cl_finetuning_args.previous_task_model:
        # Try to automatically infer previous task path
        possible_prev_paths = []
        
        # Infer based on current output path
        current_dir_parts = training_args.output_dir.split('/')
        if len(current_dir_parts) >= 2:
            # Assume path format is base_dir/task_X_name
            base_dir = '/'.join(current_dir_parts[:-1])
            prev_task_path = os.path.join(base_dir, f"task_0_{cl_finetuning_args.prev_task_id}")
            possible_prev_paths.append(prev_task_path)
        
        # Look in saves directory
        saves_base = os.path.join("saves", "Llama-2-7B-Chat", "lora")
        prev_task_saves = os.path.join(saves_base, f"task_0_{cl_finetuning_args.prev_task_id}")
        possible_prev_paths.append(prev_task_saves)
        
        # Check which path exists
        for path in possible_prev_paths:
            if os.path.exists(path):
                logger.info_rank0(f"Automatically discovered previous task path: {path}")
                cl_finetuning_args.previous_task_model = path
                break
                
        if not cl_finetuning_args.previous_task_model:
            logger.warning_rank0(f"Could not automatically find previous task path, please specify --previous_task_model")
    
    # Load model with I-LORA adapter
    model = load_ilora_model(
        tokenizer=tokenizer,
        model_args=model_args,
        finetuning_args=finetuning_args,
        is_trainable=training_args.do_train
    )

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    # Set up data collator
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

    # Set up metrics
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Set up generation kwargs
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Initialize I-LORA trainer
    trainer = ILORATrainer(
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
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        # Print I-LORA specific training statistics
        if hasattr(model, "ilora"):
            logger.info_rank0("=" * 50)
            logger.info_rank0("I-LORA Training Complete, Statistics:")
            logger.info_rank0(f"Average Consistency Loss: {train_result.metrics.get('avg_consistency_loss', 'N/A')}")
            logger.info_rank0(f"Average Total Loss: {train_result.metrics.get('avg_ilora_total_loss', 'N/A')}")
            logger.info_rank0(f"EMA Smoothing Coefficient: {train_result.metrics.get('ilora_ema_alpha', 'N/A')}")
            logger.info_rank0(f"Consistency Loss Weight: {train_result.metrics.get('ilora_consistency_weight', 'N/A')}")
            logger.info_rank0(f"Buffer Size: {train_result.metrics.get('ilora_buffer_size', 'N/A')}")
            
            # Add adapter save path information
            if finetuning_args.save_ema_adapter:
                ema_adapter_path = finetuning_args.ema_adapter_path or "ema_adapter"
                if not os.path.isabs(ema_adapter_path):
                    ema_adapter_path = os.path.join(training_args.output_dir, ema_adapter_path)
                logger.info_rank0(f"EMA Adapter saved to: {ema_adapter_path}")
                
                # Add EMA adapter path to training metrics
                train_result.metrics["ema_adapter_path"] = ema_adapter_path
            
            # Add previous task path information to training metrics
            if cl_finetuning_args.previous_task_model is not None:
                train_result.metrics["previous_task_model"] = cl_finetuning_args.previous_task_model
                
                # Add previous task's EMA adapter load path
                loaded_ema_path = "none"
                for possible_path in ["ema", "ema_adapter"]:
                    full_path = os.path.join(cl_finetuning_args.previous_task_model, possible_path)
                    if os.path.exists(full_path):
                        loaded_ema_path = full_path
                        break
                train_result.metrics["loaded_ema_adapter_path"] = loaded_ema_path
                
            if cl_finetuning_args.current_task_id is not None:
                logger.info_rank0(f"Current Task ID: {cl_finetuning_args.current_task_id}")
                train_result.metrics["current_task_id"] = cl_finetuning_args.current_task_id
                
            if cl_finetuning_args.prev_task_id is not None:
                train_result.metrics["prev_task_id"] = cl_finetuning_args.prev_task_id
                
            logger.info_rank0("=" * 50)

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            # Add more loss keys for plotting
            plot_loss(
                training_args.output_dir, 
                keys=["loss", "eval_loss", "task_loss", "consistency_loss", 
                     "ilora_total_loss", "eval_accuracy"]
            )

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.warning_rank0("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)