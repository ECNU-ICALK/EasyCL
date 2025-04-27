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

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import TrainOutput
from typing_extensions import override
from contextlib import nullcontext

from llamafactory.extras import logging
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from easycl.cl.distributed_utils import (
    is_distributed, get_rank, is_main_process, get_world_size,
    get_deepspeed_zero_stage, gather_parameters, all_reduce_tensor
)

def debugprint(*args, **kwargs):
    pass
if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from transformers.trainer import PredictionOutput

    from llamafactory.hparams import FinetuningArguments
    from easycl.hparams.cl_finetuning_args import CLFinetuningArguments


logger = logging.get_logger(__name__)


class ILORACallback(TrainerCallback):
    """
    Callback for updating EMA weights in I-LORA.
    """

    def __init__(self, trainer: "ILORATrainer"):
        self.trainer = trainer
        debugprint("ILORACallback 初始化")

    def on_step_end(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs
    ):
        """Update EMA weights after each step."""
        debugprint(f"ILORACallback.on_step_end - global_step: {state.global_step}")
        if hasattr(self.trainer.model, "ilora") and self.trainer.model.ilora is not None:
            debugprint("  检测到 ilora 实例，调用 update_ema_weights")
            self.trainer.model.ilora.update_ema_weights()
        else:
            debugprint("  未检测到 ilora 实例，跳过 EMA 更新")


class ILORATrainer(CustomSeq2SeqTrainer):
    r"""
    Trainer for I-LORA fine-tuning.
    Adds consistency loss computation and buffer management.
    """

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        cl_finetuning_args: "CLFinetuningArguments",
        **kwargs,
    ):
        debugprint("进入 ILORATrainer.__init__")
        debugprint(f"  finetuning_args: {finetuning_args}")
        debugprint(f"  cl_finetuning_args: {cl_finetuning_args}")
        super().__init__(finetuning_args=finetuning_args, **kwargs)

        # Add I-LORA callback for EMA weight updates
        debugprint("  添加 ILORACallback")
        self.add_callback(ILORACallback(self))

        # Store cl_finetuning_args for later use
        self.cl_finetuning_args = cl_finetuning_args
        debugprint("  已存储 cl_finetuning_args")

        # Check if model has I-LORA instance
        if hasattr(self.model, "ilora") and self.model.ilora is not None:
            logger.info_rank0("I-LORA instance detected. Consistency loss will be computed.")
            debugprint("  检测到 I-LORA 实例，将计算一致性损失")
            self.ilora = self.model.ilora
        else:
            logger.warning_rank0("No I-LORA instance found in model. Running without consistency loss.")
            debugprint("  未找到 I-LORA 实例，将不计算一致性损失")
            self.ilora = None
        debugprint("退出 ILORATrainer.__init__")

    @override
    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute training loss with additional consistency loss.
        Handles different DeepSpeed ZeRO stages appropriately.

        Args:
            model: The model to train
            inputs: The inputs and targets of the model
            return_outputs: If True, outputs will be returned along with the loss

        Returns:
            Loss or (loss, outputs) if return_outputs is True
        """
        debugprint("进入 ILORATrainer.compute_loss")

        # 检查 DeepSpeed ZeRO 阶段
        zero_stage = get_deepspeed_zero_stage(model)
        debugprint(f"  检测到 DeepSpeed ZeRO Stage: {zero_stage}")

        # 在 ZeRO-3 下，需要使用 gather_parameters 上下文管理器
        # 对于 ZeRO-1/2 或非 DeepSpeed，gather_parameters 返回 nullcontext
        with gather_parameters(model) if zero_stage == 3  else nullcontext():
            # Get standard task loss
            debugprint("  计算标准任务损失")
            outputs = model(**inputs)
            task_loss = outputs.loss
            debugprint(f"  标准任务损失 (task_loss): {task_loss.item() if task_loss is not None else 'None'}")

            # If I-LORA is enabled, compute and add consistency loss
            consistency_loss = torch.tensor(0.0, device=task_loss.device if task_loss is not None else inputs["input_ids"].device)

            if self.ilora is not None:
                debugprint("  I-LORA 启用，开始处理一致性损失")
                debugprint(f"  当前 cl_finetuning_args: {self.cl_finetuning_args}")

                # Compute consistency loss with current batch samples
                # Note: consistency_loss 方法内部已经处理了 ZeRO-3 和分布式同步
                debugprint("  使用当前批次样本计算一致性损失")
                # Add check for ema adapter before computing consistency loss
                if hasattr(self.model, 'peft_config') and 'ema' in self.model.peft_config:
                    debugprint("    检查通过: 'ema' 适配器存在于 model.peft_config 中。")
                    consistency_loss = self.ilora.compute_consistency_loss(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask", None),
                        labels=inputs.get("labels", None)
                    )
                else:
                    logger.warning_rank0("'ema' adapter not found in model.peft_config before computing consistency loss. Skipping loss computation.")
                    debugprint("    检查失败: 'ema' 适配器未找到。跳过一致性损失计算。")
                    if hasattr(self.model, 'peft_config'):
                        debugprint(f"    当前可用适配器: {list(self.model.peft_config.keys())}")
                    else:
                        debugprint("    模型没有 peft_config 属性。")
                    consistency_loss = torch.tensor(0.0, device=task_loss.device if task_loss is not None else inputs["input_ids"].device)

                debugprint(f"  计算得到的一致性损失 (consistency_loss): {consistency_loss.item()}")

                # Combine losses
                loss = task_loss + self.cl_finetuning_args.consistency_weight * consistency_loss
                debugprint(f"  总损失 (loss = task_loss + consistency_weight * consistency_loss): {loss.item()}")
                debugprint(f"    (consistency_weight: {self.cl_finetuning_args.consistency_weight})")

                # 始终记录一致性损失，无论其值，也无论是否return_outputs
                if not hasattr(self, "_consistency_loss_start_step"):
                    self._consistency_loss_start_step = self.state.global_step
                    self._cumulative_consistency_loss = 0.0
                    debugprint(f"  初始化一致性损失累加器 (start_step={self._consistency_loss_start_step})")

                cumulative_update = self.cl_finetuning_args.consistency_weight * consistency_loss.item()
                self._cumulative_consistency_loss += cumulative_update
                debugprint(f"  累加一致性损失: {cumulative_update}, 当前总累加值: {self._cumulative_consistency_loss}")

                # 将统计信息添加到输出，不再检查是否为零
                if return_outputs:
                    debugprint("  将一致性损失和总损失添加到输出 metrics")
                    if not hasattr(outputs, "metrics"):
                        outputs.metrics = {}
                    outputs.metrics.update({
                        "consistency_loss": consistency_loss.item(),
                        # 也添加总损失信息
                        "ilora_total_loss": loss.item()
                    })
            else:
                debugprint("  I-LORA 未启用，总损失等于任务损失")
                loss = task_loss

        # 在分布式环境中同步损失值（如果 compute_consistency_loss 内部没有同步）
        if is_distributed() and zero_stage != 3 :  # ZeRO-3 在 compute_consistency_loss 中已同步
            loss = all_reduce_tensor(loss)
            debugprint(f"  在分布式环境中同步总损失值，同步后: {loss.item()}")

        debugprint(f"退出 ILORATrainer.compute_loss, 返回损失: {loss.item() if loss is not None else 'None'}, return_outputs: {return_outputs}")
        return (loss, outputs) if return_outputs else loss

    @override
    def train(
        self,
        resume_from_checkpoint: Optional[str] = None,
        trial: Union["torch.optim.ParamGroupsHandler", Dict[str, Any], None] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ) -> TrainOutput:
        """
        Main training entry point.
        Overrides the standard training loop to handle I-LORA specific logic.
        """
        debugprint("进入 ILORATrainer.train")
        debugprint(f"  resume_from_checkpoint: {resume_from_checkpoint}")

        # Get distributed rank at the beginning of the method
        is_dist = is_distributed()
        rank = get_rank() if is_dist else 0
        debugprint(f"[rank={rank}] Entering ILORATrainer.train") # Add rank to initial debug print

        # At training start, record EMA adapter path information
        if hasattr(self.model, "ilora") and self.model.ilora is not None:
            debugprint("  检测到 ilora 实例，记录 I-LORA 相关信息")
            ilora_info = {
                "previous_task_model": self.model.ilora.previous_task_model,
                "current_task_id": self.model.ilora.current_task_id,
            }

            # Print EMA adapter information
            ema_adapter_path = self.cl_finetuning_args.ema_adapter_path or "ema_adapter"
            if not os.path.isabs(ema_adapter_path):
                output_dir = self.args.output_dir
                ema_adapter_path = os.path.join(output_dir, ema_adapter_path)

            logger.info_rank0(f"I-LORA Info - Previous Task Model: {self.model.ilora.previous_task_model}")
            logger.info_rank0(f"I-LORA Info - Current Task ID: {self.model.ilora.current_task_id}")
            logger.info_rank0(f"I-LORA Info - EMA Adapter will be saved to: {ema_adapter_path}")

            debugprint(f"  I-LORA 信息: previous_task_model={ilora_info['previous_task_model']}, current_task_id={ilora_info['current_task_id']}, ema_save_path={ema_adapter_path}")
        else:
            debugprint("  未检测到 ilora 实例")

        # Save current active_adapter before training
        original_adapter = None
        if hasattr(self.model, "active_adapter"):
            original_adapter = self.model.active_adapter
            debugprint(f"  训练开始前，保存当前活动适配器: {original_adapter}")
        else:
            debugprint("  训练开始前，模型没有 active_adapter 属性")

        # Standard train method
        debugprint("  调用父类 train 方法")
        result = super().train(
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
            **kwargs
        )
        debugprint("  父类 train 方法调用完成")
        debugprint(f"  训练结果 (TrainOutput): {result}")

        # Add I-LORA statistics to train_result.metrics
        if hasattr(self, "ilora") and self.ilora is not None:
            debugprint("  训练结束，添加 I-LORA 统计信息到 metrics")
            # Find last log entry containing consistency_loss
            avg_consistency_loss = 0.0
            log_found = False
            for log_entry in reversed(self.state.log_history):
                if "train_consistency_loss" in log_entry:
                    avg_consistency_loss = log_entry["train_consistency_loss"]
                    debugprint(f"  从日志历史记录中找到 train_consistency_loss: {avg_consistency_loss}")
                    log_found = True
                    break

            # If not found, check if we have cumulative consistency loss
            if not log_found and hasattr(self, "_cumulative_consistency_loss"):
                steps = self.state.global_step - getattr(self, "_consistency_loss_start_step", 0)
                if steps > 0:
                    avg_consistency_loss = self._cumulative_consistency_loss / steps
                    debugprint(f"  根据累积值计算平均一致性损失: {self._cumulative_consistency_loss} / {steps} = {avg_consistency_loss}")
                else:
                    debugprint("  有累积损失但训练步数为 0，无法计算平均值")
            elif not log_found:
                 debugprint("  未在日志历史记录中找到 train_consistency_loss 且没有累积损失记录")

            metrics_update = {
                "avg_consistency_loss": avg_consistency_loss,
                # "ilora_buffer_size": self.ilora.buffer_size, # Removed buffer size metric
                "ilora_ema_alpha": self.ilora.ema_alpha,
                "ilora_consistency_weight": self.ilora.reg_weight,
            }
            debugprint(f"  更新 metrics: {metrics_update}")
            result.metrics.update(metrics_update)

        # Special handling for I-LORA adapter saving
        if hasattr(self.model, "ilora") and self.args.should_save:
            # --- Start Refinement ---
            is_dist = is_distributed()
            rank = get_rank() if is_dist else 0
            debugprint(f"[rank={rank}] 进入 EMA 适配器保存逻辑 (移除 gather_parameters 和 barrier)") # Updated debug print
            debugprint(f"[rank={rank}]   should_save: {self.args.should_save}")
            debugprint(f"[rank={rank}]   save_ema_adapter: {self.cl_finetuning_args.save_ema_adapter}")
            debugprint(f"[rank={rank}]   ema_adapter_path: {self.cl_finetuning_args.ema_adapter_path}")
            debugprint(f"[rank={rank}]   is_world_process_zero: {self.is_world_process_zero()}")
            debugprint(f"[rank={rank}]   is_distributed: {is_dist}")

            if self.cl_finetuning_args.save_ema_adapter and self.cl_finetuning_args.ema_adapter_path:
                 if hasattr(self.model, "active_adapter") and hasattr(self.model, "peft_config") and "ema" in self.model.peft_config:
                     # 检查 DeepSpeed ZeRO 阶段 (所有进程都需要知道) - 但不再需要 gather_parameters
                     zero_stage = get_deepspeed_zero_stage(self.model)
                     debugprint(f"[rank={rank}]   检测到 DeepSpeed ZeRO Stage: {zero_stage} (不再使用 gather_parameters 包裹保存)")

                     # 实际保存操作仅在主进程执行
                     if self.is_world_process_zero():
                         debugprint(f"[rank={rank}] 条件满足且为主进程，开始保存 EMA 适配器 (无 gather_parameters)")
                         current_adapter = None
                         try:
                             current_adapter = self.model.active_adapter
                             debugprint(f"[rank={rank}]   当前适配器: {current_adapter}，准备切换到 ema")

                             # 确定 EMA 保存目录
                             ema_dir = self.cl_finetuning_args.ema_adapter_path
                             if not os.path.isabs(ema_dir):
                                 # If relative path, make it relative to output_dir for consistency
                                 output_dir = self.args.output_dir
                                 ema_dir = os.path.join(output_dir, ema_dir)
                                 logger.info_rank0(f"[rank={rank}] EMA adapter path '{self.cl_finetuning_args.ema_adapter_path}' is relative. Saving to: {ema_dir}")
                             else:
                                 # Use absolute path as is
                                 pass
                             debugprint(f"[rank={rank}]   确定 EMA 保存目录: {ema_dir}")
                             os.makedirs(ema_dir, exist_ok=True) # Ensure directory exists

                             try:
                                 debugprint(f"[rank={rank}]   尝试切换到 'ema' 适配器")
                                 self.model.set_adapter("ema")
                                 debugprint(f"[rank={rank}]   已切换到 'ema' 适配器")

                                 logger.info_rank0(f"[rank={rank}] Saving EMA adapter to custom path: {ema_dir}")
                                 debugprint(f"[rank={rank}]   调用 model.save_pretrained 保存 EMA 到: {ema_dir} (无 gather_parameters)")
                                 # 保存操作直接调用，不使用 gather_parameters
                                 self.model.save_pretrained(ema_dir)
                                 logger.info_rank0(f"[rank={rank}] EMA adapter saved successfully to custom path")
                                 debugprint(f"[rank={rank}]   EMA 适配器保存成功")

                             except Exception as e:
                                 logger.error(f"[rank={rank}] Failed to save EMA adapter: {e}", exc_info=True)
                                 debugprint(f"[rank={rank}]   在保存 EMA 适配器时发生异常: {e}")
                             finally:
                                 # 确保在 rank 0 恢复适配器
                                 if current_adapter is not None:
                                     debugprint(f"[rank={rank}]   (finally block in rank 0) 尝试恢复适配器为: {current_adapter}")
                                     self.model.set_adapter(current_adapter)
                                     debugprint(f"[rank={rank}]   (finally block in rank 0) 已恢复适配器")
                                 else:
                                     debugprint(f"[rank={rank}]   (finally block in rank 0) 无需恢复适配器 (current_adapter is None)")

                         except Exception as outer_e:
                             logger.error(f"[rank={rank}] Error during EMA saving process (rank 0): {outer_e}", exc_info=True)
                             debugprint(f"[rank={rank}]  在 rank 0 的 EMA 保存过程中出错: {outer_e}")
                             # Ensure adapter is restored even if error happens before try block (only rank 0)
                             if current_adapter is not None and hasattr(self.model, "active_adapter") and self.model.active_adapter != current_adapter:
                                 debugprint(f"[rank={rank}]   (outer exception handler in rank 0) 尝试恢复适配器为: {current_adapter}")
                                 try:
                                     self.model.set_adapter(current_adapter)
                                     debugprint(f"[rank={rank}]   (outer exception handler in rank 0) 已恢复适配器")
                                 except Exception as restore_e:
                                     logger.error(f"[rank={rank}] Failed to restore adapter in outer exception handler (rank 0): {restore_e}", exc_info=True)
                                     debugprint(f"[rank={rank}]   (outer exception handler in rank 0) 恢复适配器失败: {restore_e}")
                     # else: # 非主进程无需操作
                     #     debugprint(f"[rank={rank}] 非主进程，跳过 EMA 保存操作")

                 else:
                     # Log warning only on main process
                     if self.is_world_process_zero():
                          logger.warning_rank0("[rank=0] Cannot save EMA adapter: model does not have an 'ema' adapter or active_adapter")
                     debugprint(f"[rank={rank}] 跳过 EMA 保存：模型缺少 'ema' 适配器或 active_adapter 属性")
            else:
                 # Log info only on main process
                 if self.is_world_process_zero():
                     logger.info_rank0("[rank=0] Skipping EMA adapter saving: save_ema_adapter not enabled or ema_adapter_path not specified.")
                 debugprint(f"[rank={rank}] 跳过 EMA 保存：未启用 save_ema_adapter 或未指定 ema_adapter_path")

            # 移除显式的 barrier 同步
            # if is_dist:
            #     debugprint(f"[rank={rank}] 到达 barrier 前 (已移除)")
            #     # torch.distributed.barrier() # Removed explicit barrier
            #     debugprint(f"[rank={rank}] 已通过 barrier (已移除)")
            # --- End Refinement ---

        # Restore original adapter just in case (this was already here)
        # 注意：对于 ZeRO-3，此处的恢复逻辑可能需要重新审视，因为 adapter 的状态可能仅在 rank 0 被修改和恢复。
        # 但 PeftModel 的 set_adapter 应该能在内部处理好分布式状态，暂时保持不变。
        if original_adapter is not None and hasattr(self.model, "set_adapter"):
            # Check if the current adapter is already the original one (check on all ranks for safety?)
            needs_restore = True
            if hasattr(self.model, "active_adapter") and self.model.active_adapter == original_adapter:
                 needs_restore = False
                 debugprint(f"[rank={rank}]   训练结束，当前适配器已经是原始适配器 ({original_adapter})，无需恢复") # Log on all ranks

            if needs_restore:
                 # Use the rank defined at the method start
                 debugprint(f"[rank={rank}]   训练结束，尝试恢复原始适配器: {original_adapter}") # Log on all ranks
                 try:
                      # Assume set_adapter handles distributed state correctly
                      self.model.set_adapter(original_adapter)
                      # Use the rank defined at the method start
                      debugprint(f"[rank={rank}]   训练结束，已恢复原始适配器: {original_adapter}") # Log on all ranks
                 except Exception as e:
                      # Log error on rank 0, debug on all ranks
                      if self.is_world_process_zero():
                           logger.warning_rank0(f"Failed to restore original adapter '{original_adapter}': {e}", exc_info=True)
                      # Use the rank defined at the method start
                      debugprint(f"[rank={rank}]   训练结束，恢复原始适配器 '{original_adapter}' 失败: {e}") # Log on all ranks
        else:
            # Use the rank defined at the method start
            debugprint(f"[rank={rank}]   训练结束，无需恢复适配器 (无原始适配器或模型无 set_adapter 方法)") # Log on all ranks

        # Use the rank defined at the method start
        debugprint(f"[rank={rank}] 退出 ILORATrainer.train") # Log on all ranks
        return result