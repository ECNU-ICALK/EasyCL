import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from llamafactory.extras import logging
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.packages import is_transformers_version_greater_than
from llamafactory.train.callbacks import SaveProcessorCallback
from llamafactory.train.trainer_utils import create_custom_optimizer, create_custom_scheduler
from .olora import OLoRA
from llamafactory.hparams import FinetuningArguments
from easycl.hparams import CLFinetuningArguments
def debugprint(*args, **kwargs):
    pass

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput


logger = logging.get_logger(__name__)


class OLoRATrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    Adds O-LoRA specific functionality.
    """

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        cl_finetuning_args: "CLFinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: "PreTrainedTokenizer" = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.cl_finetuning_args = cl_finetuning_args
        if gen_kwargs is not None:
            self._gen_kwargs = gen_kwargs
        
        debugprint(f"OLoRATrainer __init__: 传入的 cl_finetuning_args: {self.cl_finetuning_args}")

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        # Initialize O-LoRA
        self.use_olora = cl_finetuning_args.use_olora
        if self.use_olora:
            debugprint(f"OLoRATrainer __init__: O-LoRA 已启用, use_olora={self.use_olora}")
            self.olora = OLoRA(
                model=self.model,
                orthogonal_lambda=cl_finetuning_args.orthogonal_lambda,
                l2_lambda=cl_finetuning_args.l2_lambda,
                olora_history_path=cl_finetuning_args.olora_history_path,
                model_output_dir=self.args.output_dir,
                device=self.args.device,
                prev_task_id=cl_finetuning_args.prev_task_id
            )
            # Only try to load when prev_task_id is provided
            if cl_finetuning_args.prev_task_id:
                debugprint(f"OLoRATrainer __init__: 尝试加载 prev_task_id: {cl_finetuning_args.prev_task_id}")
                load_success = self.olora.load_prev_adapter(cl_finetuning_args.prev_task_id)
                debugprint(f"OLoRATrainer __init__: 加载 prev_task_id={cl_finetuning_args.prev_task_id} 的结果: {load_success}")
                if not load_success:
                    logger.warning("Failed to load previous task adapter.")
                    logger.warning("Training will continue but without orthogonal constraints.")
                    self.use_olora = False
                    debugprint(f"OLoRATrainer __init__: 加载失败，禁用 O-LoRA, use_olora={self.use_olora}")
            else:
                logger.info("No previous task ID provided. This seems to be the first task.")
                debugprint("OLoRATrainer __init__: 没有提供 prev_task_id，视为第一个任务。")
        else:
            debugprint(f"OLoRATrainer __init__: O-LoRA 未启用, use_olora={self.use_olora}")

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
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    @override
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss

        # Add O-LoRA losses if enabled
        if self.use_olora and hasattr(self, 'olora') and self.olora is not None:
            orthogonal_loss = torch.tensor(0.0, device=self.args.device) # Initialize to zero
            l2_loss = torch.tensor(0.0, device=self.args.device)       # Initialize to zero

            # Only compute orthogonal loss if historical weights are loaded
            if self.olora.merged_historical_weights is not None:
                orthogonal_loss = self.olora.compute_orthogonal_loss()
                debugprint(f"OLoRATrainer compute_loss: 计算出的 orthogonal_loss: {orthogonal_loss.item():.4f}")
            else:
                 debugprint(f"OLoRATrainer compute_loss: 未加载历史权重，跳过 orthogonal_loss 计算。")

            # Compute L2 loss for the current adapter parameters
            l2_loss = self.olora.compute_l2_loss()
            debugprint(f"OLoRATrainer compute_loss: 计算出的 l2_loss: {l2_loss.item():.4f}")

            # Add computed losses to the main task loss
            loss = loss + orthogonal_loss + l2_loss
            debugprint(f"OLoRATrainer compute_loss: 添加 O-LoRA 损失后的总 loss: {loss.item():.4f}")
            
            if return_outputs:
                outputs.metrics = outputs.get("metrics", {})
                outputs.metrics.update({
                    "orthogonal_loss": orthogonal_loss.item(),
                    "l2_loss": l2_loss.item()
                })
        else:
            debugprint(f"OLoRATrainer compute_loss: O-LoRA 未启用或未初始化，跳过 O-LoRA 损失计算。")

        return (loss, outputs) if return_outputs else loss

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n") 