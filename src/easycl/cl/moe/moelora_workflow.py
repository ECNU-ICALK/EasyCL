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

from typing import TYPE_CHECKING, List, Optional

from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.logging import get_logger
from llamafactory.extras.misc import calculate_tps, get_logits_processor
from llamafactory.extras.ploting import plot_loss
from llamafactory.model import load_tokenizer
from llamafactory.train.trainer_utils import create_modelcard_and_push
from llamafactory.train.sft.metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor

# Import MoE-LoRA specific components
# from llamafactory.cl.moe.moelora_loader import load_moelora_model
# from llamafactory.cl.moe.moelora_trainer import MoELoRATrainer
from .moelora_loader import load_moelora_model
from .moelora_trainer import MoELoRATrainer

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    # from llamafactory.hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments, CLFinetuningArguments
    from llamafactory.hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments
    from easycl.hparams import CLFinetuningArguments


logger = get_logger(__name__)


def run_sft_moelora(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    cl_finetuning_args: "CLFinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    """
    Run supervised fine-tuning with MoE-LoRA adapters.
    
    This workflow is similar to the standard SFT workflow, but uses the 
    MoE-LoRA specific loader and trainer.
    """
    # Validate MoE-LoRA configuration
    if getattr(cl_finetuning_args, "expert_num", None) is None or cl_finetuning_args.expert_num <= 1:
        logger.warning_rank0(
            "expert_num is not set or less than 2. "
            "This will fall back to standard LoRA training. "
            "Set expert_num > 1 to enable MoE-LoRA."
        )
    
    # Load tokenizer and template
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
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
    
    # Load model using MoE-LoRA adapter
    model = load_moelora_model(
        tokenizer=tokenizer,
        model_args=model_args,
        finetuning_args=finetuning_args,
        cl_finetuning_args=cl_finetuning_args,
        is_trainable=training_args.do_train
    )
    
    # For better serialization of quantized models during inference
    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)
    
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
    
    # Initialize the MoE-LoRA trainer
    trainer = MoELoRATrainer(
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
    
    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        
        #trainer.save_model()
        model.save_pretrained(training_args.output_dir)
        
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
                keys=["loss", "eval_loss", "eval_accuracy"]
            )
    
    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # Use left-padding in generation
    
    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
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
