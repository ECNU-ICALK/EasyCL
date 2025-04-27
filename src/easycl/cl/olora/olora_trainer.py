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
from easycl.cl.distributed_utils import is_main_process, broadcast_object
import torch.distributed as dist

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
                    logger.warning_rank0("Failed to load previous task adapter.")
                    logger.warning_rank0("Training will continue but without orthogonal constraints.")
                    self.use_olora = False
                    debugprint(f"OLoRATrainer __init__: 加载失败，禁用 O-LoRA, use_olora={self.use_olora}")
            else:
                logger.info_rank0("No previous task ID provided. This seems to be the first task.")
                debugprint("OLoRATrainer __init__: 没有提供 prev_task_id，视为第一个任务。")
        else:
            debugprint(f"OLoRATrainer __init__: O-LoRA 未启用, use_olora={self.use_olora}")

        # Ensure O-LoRA state is consistent across processes
        if torch.distributed.is_initialized():
            self.use_olora = broadcast_object(self.use_olora)
            torch.distributed.barrier()

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
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        from easycl.cl.distributed_utils import get_deepspeed_zero_stage, is_deepspeed_zero3_enabled

        # Debug: Print the active adapter at the start of compute_loss
        active_adapter = "N/A"
        if hasattr(model, 'active_adapter'):
            active_adapter = model.active_adapter
        elif hasattr(model, 'active_adapters'): # Some PEFT versions might use plural
            active_adapter = model.active_adapters
        debugprint(f"OLoRATrainer compute_loss: Entering compute_loss. Active adapter(s): {active_adapter}")
        # 注意：O-LoRA 实现中只使用 'default' adapter 进行损失计算，不再使用 'current' adapter

        # 检测是否是Zero-3或zero-2环境
        is_zero3 = is_deepspeed_zero3_enabled() or get_deepspeed_zero_stage(model) == 3
        is_zero2 = get_deepspeed_zero_stage(model) == 2
        debugprint(f"OLoRATrainer compute_loss: 检测到 Zero-3 环境: {is_zero3}")
        debugprint(f"OLoRATrainer compute_loss: 检测到 Zero-2 环境: {is_zero2}")

        # 初始化辅助损失变量
        orthogonal_loss = torch.tensor(0.0, device=self.args.device)
        l2_loss = torch.tensor(0.0, device=self.args.device)

        # 创建模块到规范化路径的映射
        module_to_path_map = {}
        for name, mod in model.named_modules():
            # 移除可能的'module.'前缀
            clean_name = name.replace('module.', '')
            module_to_path_map[mod] = clean_name

        # 在Zero-3或zero-2环境下使用Forward Hooks计算辅助损失
        if self.use_olora and hasattr(self, 'olora') and self.olora is not None:
            if is_zero3 or is_zero2:
                debugprint(f"OLoRATrainer compute_loss: 在Zero-3或Zero-2环境下使用Forward Hooks计算辅助损失")

                # 创建钩子列表
                hooks = []
                matched_weights_count = 0  # 添加计数器，记录成功匹配的历史权重数量

                # 只有在历史权重加载时才计算正交损失
                if self.olora.merged_historical_weights is not None:
                    # 注册正交损失的Forward Hook
                    def orthogonal_hook_fn(module, inputs, outputs):
                        nonlocal orthogonal_loss, matched_weights_count
                        if hasattr(module, "lora_A") and hasattr(module.lora_A, "keys"):
                            #debugprint(f"orthogonal_hook_fn: 开始处理模块，可用的 lora_A keys: {list(module.lora_A.keys())}")

                            # 检查是否存在default adapter
                            if 'default' in module.lora_A:
                                new_weight = module.lora_A['default'].weight
                                #debugprint(f"orthogonal_hook_fn: 获取到default adapter的权重，shape: {new_weight.shape}")

                                # 使用预先构建的映射获取规范化的模块路径
                                module_path = module_to_path_map.get(module)
                                #debugprint(f"orthogonal_hook_fn: 原始模块在映射中的路径: {module_path}")

                                if module_path:
                                    merged_a_key = f"{module_path}.merged_A"
                                    #debugprint(f"orthogonal_hook_fn: 构建历史权重键: {merged_a_key}")
                                    #debugprint(f"orthogonal_hook_fn: 历史权重键列表前5个: {list(self.olora.merged_historical_weights.keys())[:5]}")

                                    if merged_a_key in self.olora.merged_historical_weights:
                                        old_weight = self.olora.merged_historical_weights[merged_a_key].to(new_weight.device)
                                        #debugprint(f"orthogonal_hook_fn: 获取到历史权重，shape: {old_weight.shape}")

                                        if new_weight.shape[1] == old_weight.shape[1]:
                                            dot_product = torch.mm(new_weight, old_weight.T)
                                            curr_loss = torch.abs(dot_product).sum()
                                            orthogonal_loss += curr_loss
                                            matched_weights_count += 1  # 增加匹配计数
                                            #debugprint(f"orthogonal_hook_fn: 模块 {module_path} 计算正交损失成功: {curr_loss.item():.4f}")
                                        else:
                                            debugprint(f"orthogonal_hook_fn: 模块 {module_path} 权重维度不匹配 - 新权重: {new_weight.shape}, 历史权重: {old_weight.shape}")
                                    else:
                                        debugprint(f"orthogonal_hook_fn: 未找到模块 {module_path} 的历史权重")
                                else:
                                    debugprint(f"orthogonal_hook_fn: 模块未在映射中找到对应路径")
                            else:
                                debugprint(f"orthogonal_hook_fn: 模块中未找到'default' adapter")

                    # 注册L2损失的Forward Hook
                    def l2_hook_fn(module, inputs, outputs):
                        nonlocal l2_loss
                        # 在这里计算L2损失，因为此时参数已经被DeepSpeed收集为完整参数
                        if hasattr(module, "lora_A") and hasattr(module.lora_A, "keys"):
                            # 检查是否存在default adapter
                            if 'default' in module.lora_A:
                                # 计算A矩阵的L2损失
                                a_weight = module.lora_A['default'].weight
                                if a_weight.numel() > 0:
                                    a_norm_sq = torch.sum(a_weight ** 2)
                                    l2_loss += a_norm_sq

                                # 计算B矩阵的L2损失
                                if hasattr(module, "lora_B") and 'default' in module.lora_B:
                                    b_weight = module.lora_B['default'].weight
                                    if b_weight.numel() > 0:
                                        b_norm_sq = torch.sum(b_weight ** 2)
                                        l2_loss += b_norm_sq

                    # 注册钩子到所有LoRA模块
                    for name, module in model.named_modules():
                        if hasattr(module, "lora_A") and hasattr(module.lora_A, "keys"):
                            # 注册正交损失钩子（如果有历史权重）
                            if self.olora.merged_historical_weights is not None:
                                h1 = module.register_forward_hook(orthogonal_hook_fn)
                                hooks.append(h1)

                            # 注册L2损失钩子
                            h2 = module.register_forward_hook(l2_hook_fn)
                            hooks.append(h2)

                # 执行前向传播，触发钩子
                outputs = model(**inputs)
                loss = outputs.loss

                # 应用L2损失的lambda系数
                l2_loss = l2_loss * self.olora.l2_lambda
                orthogonal_loss = orthogonal_loss * self.olora.orthogonal_lambda

                # 移除所有钩子
                for h in hooks:
                    h.remove()

                debugprint(f"OLoRATrainer compute_loss: Zero-3环境下计算的 orthogonal_loss: {orthogonal_loss.item():.4f}, 成功匹配历史权重数量: {matched_weights_count}")
                debugprint(f"OLoRATrainer compute_loss: Zero-3环境下计算的 l2_loss: {l2_loss.item():.4f}")
            else:
                # 非Zero-3环境，使用原来的方式计算损失
                outputs = model(**inputs)
                loss = outputs.loss

                # 只有在历史权重加载时才计算正交损失
                if self.olora.merged_historical_weights is not None:
                    orthogonal_loss = self.olora.compute_orthogonal_loss()
                    debugprint(f"OLoRATrainer compute_loss: 计算出的 orthogonal_loss: {orthogonal_loss.item():.4f}")
                else:
                    debugprint(f"OLoRATrainer compute_loss: 未加载历史权重，跳过 orthogonal_loss 计算。")

                # 计算当前adapter参数的L2损失
                l2_loss = self.olora.compute_l2_loss()
                debugprint(f"OLoRATrainer compute_loss: 计算出的 l2_loss: {l2_loss.item():.4f}")
        else:
            # O-LoRA未启用，直接计算主任务损失
            outputs = model(**inputs)
            loss = outputs.loss
            debugprint(f"OLoRATrainer compute_loss: O-LoRA 未启用或未初始化，跳过 O-LoRA 损失计算。")

        # 将辅助损失添加到主任务损失
        if self.use_olora and hasattr(self, 'olora') and self.olora is not None:
            loss = loss + orthogonal_loss + l2_loss
            debugprint(f"OLoRATrainer compute_loss: 添加 O-LoRA 损失后的总 loss: {loss.item():.4f}")

            if return_outputs:
                outputs.metrics = outputs.get("metrics", {})
                outputs.metrics.update({
                    "orthogonal_loss": orthogonal_loss.item(),
                    "l2_loss": l2_loss.item()
                })

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

        os.makedirs(os.path.dirname(output_prediction_file), exist_ok=True)
        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

        # Ensure all processes wait for predictions to be saved
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
