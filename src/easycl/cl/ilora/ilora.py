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
import random
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from llamafactory.extras import logging


if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from peft import PeftModel

    from llamafactory.hparams import FinetuningArguments
    from easycl.hparams.cl_finetuning_args import CLFinetuningArguments


logger = logging.get_logger(__name__)


class Buffer:
    """
    Buffer for storing samples of previous tasks.
    Uses reservoir sampling for maintaining a fixed size buffer.
    """
    
    def __init__(self, buffer_size: int = 500):
        """
        Initialize buffer with a fixed size.
        
        Args:
            buffer_size: Maximum number of samples to store in the buffer.
        """
        self.buffer_size = buffer_size
        self.buffer = []
        self.sample_count = 0
        
    def add_data(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Add new samples to the buffer using reservoir sampling.
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            labels: Target labels
        """
        batch_size = input_ids.size(0)
        
        for i in range(batch_size):
            # Get single sample
            single_input = input_ids[i].detach().clone()
            single_mask = attention_mask[i].detach().clone() if attention_mask is not None else None
            single_label = labels[i].detach().clone() if labels is not None else None
            
            if len(self.buffer) < self.buffer_size:
                # Buffer not full, add directly
                self.buffer.append((single_input, single_mask, single_label))
            else:
                # Buffer full, use reservoir sampling
                j = random.randint(0, self.sample_count)
                if j < self.buffer_size:
                    self.buffer[j] = (single_input, single_mask, single_label)
            
            self.sample_count += 1
    
    def get_data(self, batch_size: int = 16) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get a random batch of samples from the buffer.
        
        Args:
            batch_size: Number of samples to return.
            
        Returns:
            Tuple of input_ids, attention_mask, and labels.
        """
        if not self.buffer:
            return None, None, None
            
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
            
        indices = random.sample(range(len(self.buffer)), batch_size)
        samples = [self.buffer[i] for i in indices]
        
        # Pad inputs to the same length
        input_ids = [sample[0] for sample in samples]
        attention_masks = [sample[1] for sample in samples]
        labels = [sample[2] for sample in samples]
        
        max_length = max(input_id.size(0) for input_id in input_ids)
        
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []
        
        for input_id, mask, label in zip(input_ids, attention_masks, labels):
            # Pad inputs
            padding_length = max_length - input_id.size(0)
            padded_input = F.pad(input_id, (0, padding_length), value=0)
            padded_input_ids.append(padded_input)
            
            # Pad attention masks
            if mask is not None:
                padded_mask = F.pad(mask, (0, padding_length), value=0)
                padded_attention_masks.append(padded_mask)
            
            # Pad labels
            if label is not None:
                padded_label = F.pad(label, (0, padding_length), value=-100)  # Use -100 to ignore padding in loss
                padded_labels.append(padded_label)
        
        # Stack tensors
        input_tensor = torch.stack(padded_input_ids, dim=0)
        attention_mask_tensor = torch.stack(padded_attention_masks, dim=0) if padded_attention_masks and padded_attention_masks[0] is not None else None
        label_tensor = torch.stack(padded_labels, dim=0) if padded_labels and padded_labels[0] is not None else None
        
        return input_tensor, attention_mask_tensor, label_tensor
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.buffer) == 0


class ILORA:
    """
    Implementation of I-LORA (Improved LoRA for Continual Learning).
    Uses EMA of model weights and consistency loss on hidden states to prevent catastrophic forgetting.
    """
    
    def __init__(
        self,
        model: "PeftModel",
        cl_finetuning_args: "CLFinetuningArguments",
        previous_task_model: Optional[str] = None,
        current_task_id: Optional[str] = None,
    ):
        """
        Initialize ILORA.
        
        Args:
            model: The PeftModel with LoRA adapters.
            cl_finetuning_args: Continual Learning fine-tuning arguments.
            previous_task_model: Path to previous task model.
            current_task_id: ID of the current task.
        """
        self.model = model
        self.ema_alpha = cl_finetuning_args.ema_alpha
        self.reg_weight = cl_finetuning_args.consistency_weight
        self.buffer_size = cl_finetuning_args.ilora_buffer_size
        self.selective_update = cl_finetuning_args.selective_update
        self.min_update_threshold = cl_finetuning_args.min_update_threshold
        self.hidden_state_layers = cl_finetuning_args.hidden_state_layers
        
        # Initialize buffer
        self.buffer = Buffer(buffer_size=self.buffer_size)
        
        # Initialize MSE loss
        self.consistency_loss = nn.MSELoss(reduction='none')
        
        # Set defaults for paths
        self.previous_task_model = previous_task_model
        self.current_task_id = current_task_id
        
        logger.info_rank0(f"Initialized ILORA with buffer size {self.buffer_size}, EMA alpha {self.ema_alpha}, and regularization weight {self.reg_weight}.")
        
    def update_ema_weights(self) -> None:
        """
        Update EMA adapter weights from the current adapter (default).
        """
        if not hasattr(self.model, "peft_config"):
            logger.warning_rank0("Model is not a PEFT model, cannot update EMA weights.")
            return
            
        # 检查模型是否同时具有default和ema适配器
        has_default = hasattr(self.model, "peft_config") and "default" in self.model.peft_config
        has_ema = hasattr(self.model, "peft_config") and "ema" in self.model.peft_config
        
        if not has_default or not has_ema:
            logger.warning_rank0("Model does not have both 'default' and 'ema' adapters, skipping EMA update.")
            return

        # 获取当前模型参数
        try:
            current_active = self.model.active_adapter
            self.model.set_adapter("default")
            default_state = {}
            
            # 单独收集default适配器的参数
            for name, param in self.model.named_parameters():
                if "lora" in name and param.requires_grad:
                    default_state[name] = param.data.clone()
            
            # 如果没有参数，可能是配置错误
            if not default_state:
                logger.warning_rank0("Default adapter appears to have no parameters, skipping EMA update.")
                return
                
            # 现在获取EMA适配器参数
            self.model.set_adapter("ema")
            ema_state = {}
            
            # 收集EMA适配器参数
            for name, param in self.model.named_parameters():
                if "lora" in name and "ema" in name and not param.requires_grad:
                    ema_state[name] = param.data.clone()
            
            # 如果没有EMA参数，尝试从default复制
            if not ema_state:
                logger.warning_rank0("EMA adapter has no parameters, initializing from default.")
                # 这里我们会在下面的更新循环中处理
        
            # 更新EMA权重
            for default_name, default_param in default_state.items():
                # 将default参数名称转换为对应的ema参数名称
                if "lora" in default_name and "default" not in default_name:
                    ema_name = default_name.replace("base_model.model.", "base_model.model.ema.")
                    
                    if ema_name in ema_state:
                        # EMA更新
                        ema_state[ema_name] = self.ema_alpha * default_param + (1 - self.ema_alpha) * ema_state[ema_name]
                    else:
                        # 如果EMA中没有对应参数，直接复制
                        ema_state[ema_name] = default_param.clone()
                
            # 将更新后的权重应用到EMA适配器
            self.model.set_adapter("ema")
            for name, param in self.model.named_parameters():
                if name in ema_state:
                    param.data = ema_state[name].to(param.device)
            
            # 恢复原始适配器
            self.model.set_adapter(current_active)
            
        except Exception as e:
            logger.warning_rank0(f"Updating EMA weights failed: {e}")
            self.model.set_adapter(current_active)  # 确保恢复adapter状态
        
    def compute_consistency_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute consistency loss between current model (default adapter) and EMA model (ema adapter).
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            labels: Target labels
            
        Returns:
            Consistency loss tensor
        """
        # If no buffer samples yet, return zero loss
        buffer_inputs, buffer_masks, buffer_labels = self.buffer.get_data(min(16, len(self.buffer)))
        
        if buffer_inputs is None:
            # Add a log to record that consistency loss is zero due to empty buffer
            return torch.tensor(0.0, device=input_ids.device)
        
        # Record buffer sample count for debugging
        
        buffer_inputs = buffer_inputs.to(input_ids.device)
        if buffer_masks is not None:
            buffer_masks = buffer_masks.to(input_ids.device)
        if buffer_labels is not None:
            buffer_labels = buffer_labels.to(input_ids.device)
            
        # Forward pass with the current model (plastic) using default adapter
        self.model.set_adapter("default")
        plastic_outputs = self.model(
            input_ids=buffer_inputs,
            attention_mask=buffer_masks,
            labels=buffer_labels,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
        
        plastic_hidden = plastic_outputs.hidden_states
        
        # Forward pass with the EMA model (stable) using ema adapter
        self.model.set_adapter("ema")
        with torch.no_grad():
            stable_outputs = self.model(
                input_ids=buffer_inputs,
                attention_mask=buffer_masks,
                labels=buffer_labels,
                output_hidden_states=True,
                return_dict=True,
                **kwargs
            )
            stable_hidden = stable_outputs.hidden_states
        
        # Select which hidden states to compute consistency on
        selected_plastic = []
        selected_stable = []
        
        for layer_idx in self.hidden_state_layers:
            if layer_idx < 0:
                layer_idx = len(plastic_hidden) + layer_idx  # Convert negative index to positive
            
            if 0 <= layer_idx < len(plastic_hidden):
                selected_plastic.append(plastic_hidden[layer_idx])
                selected_stable.append(stable_hidden[layer_idx])
        
        # Compute consistency loss with selective update
        layer_losses = []
        
        for plastic, stable in zip(selected_plastic, selected_stable):
            if self.selective_update:
                # Create mask for selective update - only update where plastic model is better
                plastic_loss = (plastic - buffer_labels.unsqueeze(-1)).abs()
                stable_loss = (stable - buffer_labels.unsqueeze(-1)).abs()
                update_mask = (plastic_loss < stable_loss).float() * (plastic_loss > self.min_update_threshold).float()
                
                # Compute MSE loss with mask
                layer_loss = self.consistency_loss(plastic, stable)
                masked_loss = layer_loss * update_mask
                layer_losses.append(masked_loss.mean())
            else:
                # Regular MSE loss
                layer_loss = self.consistency_loss(plastic, stable).mean()
                layer_losses.append(layer_loss)
        
        # Average losses across all selected layers
        if layer_losses:
            l_cons = torch.stack(layer_losses).mean()
        else:
            l_cons = torch.tensor(0.0, device=input_ids.device)
            
        # Reset adapter to default for further training
        self.model.set_adapter("default")
        
        return l_cons
    
    def store_sample(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> None:
        """
        Store a batch of samples in the buffer.
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            labels: Target labels
        """
        self.buffer.add_data(input_ids, attention_mask, labels)