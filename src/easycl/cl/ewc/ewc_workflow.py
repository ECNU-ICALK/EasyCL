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

from typing import TYPE_CHECKING, Optional
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
from typing import List
from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.logging import get_logger
from llamafactory.extras.misc import calculate_tps, get_logits_processor
from llamafactory.extras.ploting import plot_loss
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.trainer_utils import create_modelcard_and_push
from llamafactory.train.sft.metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .ewc_trainer import EWCSeq2SeqTrainer
from llamafactory.hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments
from easycl.hparams import CLFinetuningArguments
from ..distributed_utils import (
    is_distributed, get_rank, get_world_size, is_main_process,
    get_deepspeed_zero_stage, gather_parameters, all_reduce_tensor
)
import traceback
def debugprint(*args, **kwargs):
    pass

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback


logger = get_logger(__name__)


def run_sft_ewc(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    cl_finetuning_args: "CLFinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    # --- Debug Print Start ---
    debugprint(f"CL Finetuning Args in run_sft_ewc (start): {cl_finetuning_args}")
    # --- Debug Print End ---

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

    # --- Debug Print Start ---
    debugprint(f"EWC Args for Trainer: use_ewc={cl_finetuning_args.use_ewc}, ewc_lambda={cl_finetuning_args.ewc_lambda}")
    # --- Debug Print End ---

    # Initialize our Trainer
    trainer = EWCSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        cl_finetuning_args=cl_finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        use_ewc=cl_finetuning_args.use_ewc,
        ewc_lambda=cl_finetuning_args.ewc_lambda,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )



    # Training
    if training_args.do_train:
        # If current_task_name is not specified, extract from dataset path

        # Modify EWC logic
        if cl_finetuning_args.use_ewc:
            if cl_finetuning_args.previous_task_data:
                logger.info("Loading previous task data for EWC...")
                # --- Debug Print Start ---
                debugprint(f"EWC Fisher Params: previous_task_data={cl_finetuning_args.previous_task_data}, ewc_num_samples={cl_finetuning_args.ewc_num_samples}")
                # --- Debug Print End ---
                try:
                    # Save current dataset configuration
                    current_dataset = data_args.dataset
                    data_args.dataset = [cl_finetuning_args.previous_task_data]

                    prev_dataset_module = get_dataset(
                        template,
                        model_args,
                        data_args,
                        training_args,
                        stage="sft",
                        **tokenizer_module
                    )

                    if "train_dataset" in prev_dataset_module:
                        logger.info("Computing Fisher information for EWC using previous task data...")
                        try:
                            # Create dataloader with distributed sampler if needed
                            prev_train_dataset = prev_dataset_module["train_dataset"]

                            if is_distributed():
                                # Use DistributedSampler for the previous task dataset
                                debugprint(f"[Rank {get_rank()}] Creating DistributedSampler for Fisher computation")
                                prev_sampler = DistributedSampler(
                                    prev_train_dataset,
                                    num_replicas=get_world_size(),
                                    rank=get_rank(),
                                    shuffle=True  # Shuffle samples for Fisher computation
                                )
                            else:
                                prev_sampler = None  # No sampler needed for single process

                            prev_dataloader = DataLoader(
                                prev_train_dataset,
                                batch_size=training_args.per_device_train_batch_size,
                                sampler=prev_sampler,  # Use the distributed sampler if applicable
                                collate_fn=data_collator,
                                shuffle=(prev_sampler is None)  # Shuffle only if not using DistributedSampler
                            )
                            success = trainer.prepare_for_new_task(prev_dataloader, cl_finetuning_args.ewc_num_samples)
                            if not success:
                                logger.warning("Failed to compute Fisher information. EWC will be disabled.")
                                logger.warning("Training will continue without EWC.")
                                trainer.use_ewc = False
                        except Exception as e:
                            logger.warning(f"Error during Fisher computation: {str(e)}")
                            logger.warning("Stack trace:")
                            logger.warning(traceback.format_exc())
                            trainer.use_ewc = False
                    else:
                        logger.warning("No training data found in previous task dataset. EWC will be disabled.")
                        trainer.use_ewc = False
                except Exception as e:
                    logger.warning(f"Failed to load previous task data: {str(e)}")
                    logger.warning("Stack trace:")
                    logger.warning(traceback.format_exc())
                    trainer.use_ewc = False
                finally:
                    # Restore current dataset configuration
                    data_args.dataset = current_dataset
            else:
                logger.warning("EWC enabled but no previous task data provided. EWC will be disabled.")
                trainer.use_ewc = False


        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_second"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        # Add EWC loss and previous task information printing
        if cl_finetuning_args.use_ewc:
            # Calculate EWC loss on all ranks to avoid deadlock in gather_parameters
            ewc_loss_val = trainer.ewc.ewc_loss().item()

            # Only log and add to metrics on the main process
            if is_main_process():
                train_result.metrics["ewc_loss"] = ewc_loss_val
                logger.info(f"EWC Loss (logged post-train): {ewc_loss_val}")

                # Add previous task information to metrics
                if cl_finetuning_args.previous_task_data:
                    train_result.metrics["previous_task_data"] = cl_finetuning_args.previous_task_data
                    train_result.metrics["ewc_num_samples"] = cl_finetuning_args.ewc_num_samples
                    logger.info(f"Previous Task Data: {cl_finetuning_args.previous_task_data}")
                    logger.info(f"EWC Samples Used: {cl_finetuning_args.ewc_num_samples}")
                    logger.info(f"Distributed Training: {is_distributed()}, World Size: {get_world_size()}")


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