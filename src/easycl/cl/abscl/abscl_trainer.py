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
import torch
import json
from typing import TYPE_CHECKING, Any, Dict, Optional
import safetensors.torch
from safetensors.torch import load_file as safe_load_file
import re

from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from llamafactory.extras import logging
from easycl.cl.olora.olora import OLoRA
from peft import PeftModel
from easycl.hparams import CLFinetuningArguments

if TYPE_CHECKING:
    from transformers import ProcessorMixin

logger = logging.get_logger(__name__)

class ABSCLTrainer(CustomSeq2SeqTrainer):
    """
    ABSCL Trainer - Uses O-LoRA method to compute orthogonal constraints and L2 regularization
    """
    
    def __init__(
        self,
        finetuning_args,
        cl_finetuning_args,
        processor: Optional["ProcessorMixin"] = None,
        gen_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            finetuning_args=finetuning_args,
            processor=processor,
            gen_kwargs=gen_kwargs,
            **kwargs
        )
        # Store cl_finetuning_args for later use
        self.cl_finetuning_args = cl_finetuning_args
        
        # Set loss weights
        self.orthogonal_lambda = cl_finetuning_args.abscl_orthogonal_lambda
        self.l2_lambda = cl_finetuning_args.abscl_shared_l2_lambda
        
        # Set task ID and paths
        self.task_id = cl_finetuning_args.current_task_id or "task"
        adapters_path = cl_finetuning_args.adapters_save_path or os.path.join(
            os.path.dirname(self.args.output_dir)
        )
        self.shared_adapter_path = os.path.join(adapters_path, "shared_adapter")
        
        # Use device string instead of device object
        device = self.args.device.type if hasattr(self.args.device, "type") else self.args.device
        
        # Create OLoRA instance for loss computation
        self.olora = OLoRA(
            model=self.model,
            orthogonal_lambda=self.orthogonal_lambda,
            l2_lambda=self.l2_lambda,
            olora_history_path=adapters_path,  # Use adapters_path as history path
            model_output_dir=self.args.output_dir,
            device=device,
            prev_task_id="shared_adapter"  # Set prev_task_id to shared_adapter
        )
        
        # Load shared adapter
        self._load_shared_adapter()
        
        logger.info_rank0(f"Configured ABSCL trainer with O-LoRA method")
        logger.info_rank0(f"- Shared adapter path: {self.shared_adapter_path}")
        logger.info_rank0(f"- Orthogonal constraint weight: {self.orthogonal_lambda}")
        logger.info_rank0(f"- L2 regularization weight: {self.l2_lambda}")

    def _load_shared_adapter(self):
        """Load shared adapter as reference adapter for orthogonal constraint"""
        try:
            # Check if shared adapter exists
            if not os.path.exists(self.shared_adapter_path):
                logger.warning_rank0(f"Shared adapter path does not exist: {self.shared_adapter_path}")
                return False
                
            # Verify shared adapter configuration files
            config_path = os.path.join(self.shared_adapter_path, "adapter_config.json")
            model_path = os.path.join(self.shared_adapter_path, "adapter_model.safetensors")
            
            if not os.path.exists(config_path):
                logger.warning_rank0(f"Shared adapter config file does not exist: {config_path}")
                return False
                
            if not os.path.exists(model_path):
                logger.warning_rank0(f"Shared adapter weights file does not exist: {model_path}")
                return False
            
            # Set up adapters using same method as O-LoRA
            result = self.olora.setup_adapters(self.task_id)
            if result:
                logger.info_rank0(f"Successfully set up O-LoRA adapters using shared adapter: shared_adapter and current adapter: {self.task_id}")
            else:
                logger.warning_rank0(f"Failed to set up adapters")
                
            # Load shared adapter as orthogonal reference
            result = self.olora.load_prev_adapter("shared_adapter")
            if result:
                logger.info_rank0(f"Successfully loaded shared adapter as orthogonal reference")
            else:
                logger.warning_rank0(f"Failed to load shared adapter")
                
            return result
        except Exception as e:
            logger.error_rank0(f"Error loading shared adapter: {str(e)}")
            return False

    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        """Compute loss using O-LoRA style loss computation"""
        # Calculate original loss
        outputs = model(**inputs)
        base_loss = outputs.loss
        
        # Calculate orthogonal loss and L2 loss using O-LoRA method
        orthogonal_loss = self.olora.compute_orthogonal_loss()
        l2_loss = self.olora.compute_l2_loss()
        
        # Combine losses
        total_loss = base_loss + orthogonal_loss + l2_loss
        
        # Save current loss values for logging
        self._current_orthogonal_loss = orthogonal_loss.item()
        self._current_l2_loss = l2_loss.item()
        
        # Log loss values every 100 steps
        if self.state.global_step % 100 == 0:
            logger.info_rank0(f"Step {self.state.global_step} losses - Base: {base_loss.item():.4f}, "
                  f"Orthogonal: {orthogonal_loss.item():.4f}, L2: {l2_loss.item():.4f}")
        
        # Update output metrics
        if return_outputs:
            outputs.metrics = outputs.get("metrics", {})
            outputs.metrics.update({
                "orthogonal_loss": orthogonal_loss.item(),
                "l2_loss": l2_loss.item(),
            })
        
        return (total_loss, outputs) if return_outputs else total_loss

    def get_extra_losses(self, model=None):
        """Get extra loss values for logging"""
        if hasattr(self, '_current_orthogonal_loss') and hasattr(self, '_current_l2_loss'):
            # Use cached loss values
            return {
                "orthogonal_loss": self._current_orthogonal_loss,
                "shared_l2_loss": self._current_l2_loss,  # For compatibility with original code
            }
        elif model is not None:
            # If no cache but model provided, recompute
            orthogonal_loss = self.olora.compute_orthogonal_loss()
            l2_loss = self.olora.compute_l2_loss()
            return {
                "orthogonal_loss": orthogonal_loss.item(),
                "shared_l2_loss": l2_loss.item(),  # For compatibility with original code
            }
        else:
            # No cache and no model, return zero values
            return {
                "orthogonal_loss": 0.0,
                "shared_l2_loss": 0.0,
            }
