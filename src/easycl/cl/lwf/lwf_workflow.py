# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
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
from typing import TYPE_CHECKING, Optional
import traceback
def debugprint(*args, **kwargs):
    pass
from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.logging import get_logger
from llamafactory.extras.misc import calculate_tps, get_logits_processor
from llamafactory.extras.ploting import plot_loss
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.trainer_utils import create_modelcard_and_push
from llamafactory.train.sft.metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .lwf_trainer import LWFTrainer
from llamafactory.hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments
from easycl.hparams import CLFinetuningArguments


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback


logger = get_logger(__name__)


def run_sft_lwf(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    cl_finetuning_args: "CLFinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    debugprint(f"run_sft_lwf 已调用.")
    debugprint(f"收到的 cl_finetuning_args: {cl_finetuning_args}")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

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
    debugprint(f"正在初始化 LWFTrainer.")
    debugprint(f"传递给 LWFTrainer 的参数 - use_lwf: {cl_finetuning_args.use_lwf}")
    debugprint(f"传递给 LWFTrainer 的参数 - lwf_temperature: {cl_finetuning_args.lwf_temperature}")
    debugprint(f"传递给 LWFTrainer 的参数 - lwf_alpha: {cl_finetuning_args.lwf_alpha}")
    debugprint(f"传递给 LWFTrainer 的参数 - previous_task_model 路径: {cl_finetuning_args.previous_task_model}")
    trainer = LWFTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        cl_finetuning_args=cl_finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        use_lwf=cl_finetuning_args.use_lwf,
        lwf_temperature=cl_finetuning_args.lwf_temperature,
        lwf_alpha=cl_finetuning_args.lwf_alpha,
        previous_task_model=cl_finetuning_args.previous_task_model,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )


    # Training
    if training_args.do_train:
        # Modify LWF logic
        if cl_finetuning_args.use_lwf:
            debugprint(f"LWF 已启用 (cl_finetuning_args.use_lwf = {cl_finetuning_args.use_lwf})。继续进行 LWF 设置。")
            if cl_finetuning_args.previous_task_model:
                logger.info("Loading previous task model for LWF...")
                debugprint(f"尝试从路径加载先前任务模型: {cl_finetuning_args.previous_task_model}")
                try:
                    # Save current adapter_name_or_path and model_name_or_path
                    current_adapter = copy.deepcopy(model_args.adapter_name_or_path)  # Deep copy to avoid reference issues
                    current_model = model_args.model_name_or_path
                    
                    # Normalize paths and perform detailed checks
                    previous_task_path = os.path.abspath(cl_finetuning_args.previous_task_model)
                    adapter_config_path = os.path.join(previous_task_path, "adapter_config.json")
                    adapter_model_path = os.path.join(previous_task_path, "adapter_model.safetensors")
                    
                    logger.info(f"Checking adapter files in: {previous_task_path}")
                    logger.info(f"Adapter config path: {adapter_config_path}")
                    logger.info(f"Adapter model path: {adapter_model_path}")
                    
                    if not os.path.exists(previous_task_path):
                        raise ValueError(f"Previous task model path does not exist: {previous_task_path}")
                    if not os.path.exists(adapter_config_path):
                        raise ValueError(f"Cannot find adapter_config.json in {previous_task_path}")
                    if not os.path.exists(adapter_model_path):
                        raise ValueError(f"Cannot find adapter_model.safetensors in {previous_task_path}")
                    
                    # Create new ModelArguments instance
                    prev_model_args = copy.deepcopy(model_args)
                    prev_model_args.model_name_or_path = current_model  # Use the same base model
                    prev_model_args.adapter_name_or_path = [previous_task_path]  # Use list format
                    
                    # Ensure finetuning_args copy uses correct configuration
                    prev_finetuning_args = copy.deepcopy(finetuning_args)
                    prev_cl_finetuning_args = copy.deepcopy(cl_finetuning_args)
                    prev_cl_finetuning_args.use_lwf = False  # Avoid recursive loading
                    prev_finetuning_args.create_new_adapter = False  # Ensure direct adapter loading
                    
                    # Load previous task model
                    logger.info(f"Loading model with adapter from: {prev_model_args.adapter_name_or_path}")
                    previous_task_model = load_model(
                        tokenizer,
                        prev_model_args,
                        prev_finetuning_args,
                        is_trainable=False  # Set to False as we don't need to train this model
                    )
                    
                    # Restore original adapter_name_or_path
                    model_args.adapter_name_or_path = current_adapter
                    
                    # Set to eval mode and move to correct device
                    previous_task_model.eval()
                    if hasattr(model, "device"):
                        previous_task_model.to(model.device)
                    
                    # Set previous task model
                    trainer.lwf.previous_task_model = previous_task_model
                    logger.info(f"Previous task model loaded successfully from: {previous_task_path}")
                    
                    # Record historical model path in trainer for use in metrics
                    trainer.previous_task_model_path = previous_task_path
                    
                except Exception as e:
                    logger.error(f"Failed to load previous task model: {str(e)}")
                    logger.error("Stack trace:")
                    logger.error(traceback.format_exc())
                    logger.error("LWF requires a valid previous task model. Training will be terminated.")
                    raise RuntimeError("LWF initialization failed: unable to load previous task model")
            else:
                logger.error("LWF enabled but no previous task model provided. Training will be terminated.")
                raise ValueError("LWF requires a previous task model path")

        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_second"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        # Add LWF-related metrics
        if cl_finetuning_args.use_lwf:
            debugprint(f"LWF 已使用 (cl_finetuning_args.use_lwf = {cl_finetuning_args.use_lwf})，正在添加 LWF 指标。")
            # Calculate LWF loss
            lwf_loss = trainer.lwf.lwf_loss(trainer.model(**next(iter(trainer.get_train_dataloader()))).logits, next(iter(trainer.get_train_dataloader()))).item()
            train_result.metrics["lwf_loss"] = lwf_loss
            # Record historical model path
            train_result.metrics["previous_task_model_path"] = trainer.previous_task_model_path
            logger.info(f"LWF Loss: {lwf_loss}")
            logger.info(f"Previous Task Model Path: {trainer.previous_task_model_path}")

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
