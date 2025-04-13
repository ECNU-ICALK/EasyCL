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
        
    def on_step_end(
        self, 
        args: "TrainingArguments", 
        state: "TrainerState", 
        control: "TrainerControl", 
        **kwargs
    ):
        """Update EMA weights after each step."""
        if hasattr(self.trainer.model, "ilora"):
            self.trainer.model.ilora.update_ema_weights()


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
        super().__init__(finetuning_args=finetuning_args, **kwargs)
        
        # Add I-LORA callback for EMA weight updates
        self.add_callback(ILORACallback(self))
        
        # Store cl_finetuning_args for later use
        self.cl_finetuning_args = cl_finetuning_args
        
        # Check if model has I-LORA instance
        if hasattr(self.model, "ilora"):
            logger.info_rank0("I-LORA instance detected. Consistency loss will be computed.")
            self.ilora = self.model.ilora
        else:
            logger.warning_rank0("No I-LORA instance found in model. Running without consistency loss.")
            self.ilora = None

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
        # Get standard task loss
        outputs = model(**inputs)
        task_loss = outputs.loss
        
        # If I-LORA is enabled, compute and add consistency loss
        consistency_loss = torch.tensor(0.0, device=task_loss.device)
        
        if self.ilora is not None:
            # Store current samples in buffer for future consistency computation
            self.ilora.store_sample(
                input_ids=inputs["input_ids"].detach(),
                attention_mask=inputs.get("attention_mask", None),
                labels=inputs.get("labels", None)
            )
            
            # Compute consistency loss with buffer samples
            consistency_loss = self.ilora.compute_consistency_loss(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                labels=inputs.get("labels", None)
            )
            
            # Combine losses
            loss = task_loss + self.cl_finetuning_args.consistency_weight * consistency_loss
            
            # 始终记录一致性损失，无论其值，也无论是否return_outputs
            if not hasattr(self, "_consistency_loss_start_step"):
                self._consistency_loss_start_step = self.state.global_step
                self._cumulative_consistency_loss = 0.0
                
            self._cumulative_consistency_loss += self.cl_finetuning_args.consistency_weight * consistency_loss.item()
            
            # 将统计信息添加到输出，不再检查是否为零
            if return_outputs:
                if not hasattr(outputs, "metrics"):
                    outputs.metrics = {}
                outputs.metrics.update({
                    "consistency_loss": consistency_loss.item(),
                    # 也添加总损失信息
                    "ilora_total_loss": loss.item()
                })
        else:
            loss = task_loss
            
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
        # At training start, record EMA adapter path information
        if hasattr(self.model, "ilora") and self.model.ilora is not None:
            ilora_info = {
                "previous_task_model": self.model.ilora.previous_task_model,
                "current_task_id": self.model.ilora.current_task_id,
            }
            
            # Print EMA adapter information
            ema_adapter_path = self.finetuning_args.ema_adapter_path or "ema_adapter"
            if not os.path.isabs(ema_adapter_path):
                output_dir = self.args.output_dir
                ema_adapter_path = os.path.join(output_dir, ema_adapter_path)
                
            logger.info_rank0(f"I-LORA Info - Previous Task Model: {self.model.ilora.previous_task_model}")
            logger.info_rank0(f"I-LORA Info - Current Task ID: {self.model.ilora.current_task_id}")
            logger.info_rank0(f"I-LORA Info - EMA Adapter will be saved to: {ema_adapter_path}")
            
            # Record buffer size
            logger.info_rank0(f"I-LORA Info - Buffer Size: {self.model.ilora.buffer_size}")

        # Save current active_adapter before training
        original_adapter = None
        if hasattr(self.model, "active_adapter"):
            original_adapter = self.model.active_adapter
        
        # Standard train method
        result = super().train(
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
            **kwargs
        )
        
        # Add I-LORA statistics to train_result.metrics
        if hasattr(self, "ilora") and self.ilora is not None:
            # Find last log entry containing consistency_loss
            avg_consistency_loss = 0.0
            for log_entry in reversed(self.state.log_history):
                if "train_consistency_loss" in log_entry:
                    avg_consistency_loss = log_entry["train_consistency_loss"]
                    break
                    
            # If not found, check if we have cumulative consistency loss
            if avg_consistency_loss == 0.0 and hasattr(self, "_cumulative_consistency_loss"):
                steps = self.state.global_step - getattr(self, "_consistency_loss_start_step", 0)
                if steps > 0:
                    avg_consistency_loss = self._cumulative_consistency_loss / steps
            
            result.metrics.update({
                "avg_consistency_loss": avg_consistency_loss,
                "ilora_buffer_size": self.ilora.buffer_size,
                "ilora_ema_alpha": self.ilora.ema_alpha,
                "ilora_consistency_weight": self.ilora.reg_weight,
            })
        
        # Special handling for I-LORA adapter saving
        if hasattr(self.model, "ilora") and self.args.should_save:
            if self.finetuning_args.save_ema_adapter and self.finetuning_args.ema_adapter_path and self.is_world_process_zero():
                if hasattr(self.model, "active_adapter") and "ema" in self.model.peft_config:
                    current_adapter = self.model.active_adapter
                    self.model.set_adapter("ema")
                    # Save EMA adapter to user-specified separate directory
                    ema_dir = self.finetuning_args.ema_adapter_path
                    if not os.path.isabs(ema_dir):
                        # If relative path, make it relative to output_dir
                        ema_dir = os.path.join(self.args.output_dir, ema_dir)
                    
                    os.makedirs(ema_dir, exist_ok=True)
                    logger.info_rank0(f"Saving EMA adapter to custom path: {ema_dir}")
                    self.model.save_pretrained(ema_dir)
                    logger.info_rank0("EMA adapter saved successfully to custom path")
                    self.model.set_adapter(current_adapter)
                else:
                    logger.warning_rank0("Cannot save EMA adapter: model does not have an 'ema' adapter")

        # Restore original adapter just in case
        if original_adapter is not None and hasattr(self.model, "set_adapter"):
            self.model.set_adapter(original_adapter)
            
        return result