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
import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer
from typing_extensions import override
from datasets import Dataset, concatenate_datasets
from llamafactory.extras import logging
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.packages import is_transformers_version_greater_than
from llamafactory.train.callbacks import SaveProcessorCallback
from llamafactory.train.trainer_utils import create_custom_optimizer, create_custom_scheduler
from llamafactory.hparams import FinetuningArguments
from easycl.hparams.cl_finetuning_args import CLFinetuningArguments

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput


logger = logging.get_logger(__name__)


class LAMOLTrainer(Seq2SeqTrainer):
    """
    Trainer for LAMOL (pseudo-replay style).
    Inherits from Seq2SeqTrainer and keeps standard training behavior.
    Data merging and pseudo-sample generation are handled in the workflow.
    """

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        cl_finetuning_args: "CLFinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # Standard Seq2SeqTrainer initialization
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: "PreTrainedTokenizer" = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.cl_finetuning_args = cl_finetuning_args
        if gen_kwargs is not None:
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)


    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler()

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Removes the prompt part in the generated tokens. (Standard override)
        """
        if hasattr(self, "_gen_kwargs") and not gen_kwargs:
            gen_kwargs = self._gen_kwargs
        
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            tokenizer = getattr(self, "processing_class", getattr(self, "tokenizer", None))
            if tokenizer:
                if "input_ids" in inputs:
                    input_length = inputs["input_ids"].size(-1)
                    generated_tokens[:, :input_length] = tokenizer.pad_token_id
                generated_tokens = generated_tokens.contiguous()
            else:
                 logger.warning("Tokenizer/Processing class not found in prediction_step, cannot adjust generated tokens.")


        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""
        Saves model predictions to `output_dir`. (Custom utility)
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")
        
        tokenizer = getattr(self, "processing_class", getattr(self, "tokenizer", None))
        if not tokenizer:
            logger.error("Tokenizer/Processing class not found, cannot save predictions.")
            return

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            tokenizer.pad_token_id,
        )

        for i in range(len(preds)):
            pad_indices = np.where(preds[i] == tokenizer.pad_token_id)[0]
            if len(pad_indices) > 0:
                first_pad_idx = pad_indices[0]
                non_pad_tokens = preds[i][:first_pad_idx]
                processed_pred = np.full_like(preds[i], tokenizer.pad_token_id)
                processed_pred[:len(non_pad_tokens)] = non_pad_tokens
                preds[i] = processed_pred


        decoded_inputs = ["Dataset missing 'input_ids'"] * len(preds)
        if "input_ids" in dataset.column_names:
             decoded_inputs = tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        else:
             logger.warning("Dataset is missing 'input_ids' column, unable to decode inputs for predictions file.")

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")