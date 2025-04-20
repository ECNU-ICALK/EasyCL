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

from llamafactory.extras import logging
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

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
        
        Args:
            model: The model to train
            inputs: The inputs and targets of the model
            return_outputs: If True, outputs will be returned along with the loss
            
        Returns:
            Loss or (loss, outputs) if return_outputs is True
        """
        debugprint("进入 ILORATrainer.compute_loss")
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
            # Store current samples in buffer for future consistency computation
            # debugprint("  将当前样本存入 Buffer") # Removed buffer storing
            # self.ilora.store_sample(
            #     input_ids=inputs["input_ids"].detach(),
            #     attention_mask=inputs.get("attention_mask", None).detach() if inputs.get("attention_mask", None) is not None else None,
            #     labels=inputs.get("labels", None).detach() if inputs.get("labels", None) is not None else None
            # ) # Removed buffer storing

            # Compute consistency loss with current batch samples
            debugprint("  使用当前批次样本计算一致性损失") # Updated log message
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
            debugprint("  检查是否需要保存 EMA 适配器")
            debugprint(f"    should_save: {self.args.should_save}")
            debugprint(f"    save_ema_adapter: {self.cl_finetuning_args.save_ema_adapter}")
            debugprint(f"    ema_adapter_path: {self.cl_finetuning_args.ema_adapter_path}")
            debugprint(f"    is_world_process_zero: {self.is_world_process_zero()}")
            if self.cl_finetuning_args.save_ema_adapter and self.cl_finetuning_args.ema_adapter_path and self.is_world_process_zero():
                if hasattr(self.model, "active_adapter") and hasattr(self.model, "peft_config") and "ema" in self.model.peft_config:
                    debugprint("  条件满足，开始保存 EMA 适配器")
                    current_adapter = self.model.active_adapter
                    debugprint(f"    当前适配器: {current_adapter}，切换到 ema")
                    self.model.set_adapter("ema")
                    # Save EMA adapter to user-specified separate directory
                    ema_dir = self.cl_finetuning_args.ema_adapter_path
                    if not os.path.isabs(ema_dir):
                        # If relative path, make it relative to output_dir
                        ema_dir = os.path.join(self.args.output_dir, ema_dir)
                    debugprint(f"    确定 EMA 保存目录: {ema_dir}")
                    
                    os.makedirs(ema_dir, exist_ok=True)
                    logger.info_rank0(f"Saving EMA adapter to custom path: {ema_dir}")
                    debugprint(f"    调用 model.save_pretrained 保存 EMA 到: {ema_dir}")
                    self.model.save_pretrained(ema_dir)
                    logger.info_rank0("EMA adapter saved successfully to custom path")
                    debugprint(f"    恢复适配器为: {current_adapter}")
                    self.model.set_adapter(current_adapter)
                else:
                    logger.warning_rank0("Cannot save EMA adapter: model does not have an 'ema' adapter or active_adapter")
                    debugprint("  无法保存 EMA 适配器：模型缺少 'ema' 适配器或 active_adapter 属性")
            elif not (self.cl_finetuning_args.save_ema_adapter and self.cl_finetuning_args.ema_adapter_path):
                 debugprint("  跳过 EMA 保存：未启用 save_ema_adapter 或未指定 ema_adapter_path")
            elif not self.is_world_process_zero():
                 debugprint("  跳过 EMA 保存：非主进程")

        # Restore original adapter just in case
        if original_adapter is not None and hasattr(self.model, "set_adapter"):
            debugprint(f"  训练结束，恢复原始适配器: {original_adapter}")
            self.model.set_adapter(original_adapter)
        else:
            debugprint("  训练结束，无需恢复适配器 (无原始适配器或模型无 set_adapter 方法)")
            
        debugprint("退出 ILORATrainer.train")
        return result