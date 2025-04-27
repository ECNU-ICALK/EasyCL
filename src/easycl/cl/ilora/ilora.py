
import os
import random
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from contextlib import nullcontext
from llamafactory.extras import logging
from easycl.cl.distributed_utils import (
    is_distributed, get_rank, is_main_process, get_world_size,
    get_deepspeed_zero_stage, gather_parameters, all_reduce_tensor
)

def debugprint(*args, **kwargs):
    pass

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from peft import PeftModel

    from llamafactory.hparams import FinetuningArguments
    from easycl.hparams.cl_finetuning_args import CLFinetuningArguments


logger = logging.get_logger(__name__)


class ILORA:
    """
    Implementation of I-LORA (Improved LoRA for Continual Learning).
    Uses EMA of model weights and consistency loss on hidden states to prevent catastrophic forgetting.
    """

    def __init__(
        self,
        model: "PeftModel",
        finetuning_args: "FinetuningArguments",
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
        debugprint("进入 ILORA.__init__")
        self.model = model
        self.ema_alpha = cl_finetuning_args.ema_alpha
        self.reg_weight = cl_finetuning_args.consistency_weight
        self.selective_update = cl_finetuning_args.selective_update
        self.min_update_threshold = cl_finetuning_args.min_update_threshold
        self.hidden_state_layers = cl_finetuning_args.hidden_state_layers
        debugprint(f"  从 cl_finetuning_args 获取参数:")
        debugprint(f"    ema_alpha: {self.ema_alpha}")
        debugprint(f"    consistency_weight (reg_weight): {self.reg_weight}")
        debugprint(f"    selective_update: {self.selective_update}")
        debugprint(f"    min_update_threshold: {self.min_update_threshold}")
        debugprint(f"    hidden_state_layers: {self.hidden_state_layers}")

        # Initialize MSE loss
        self.consistency_loss = nn.MSELoss(reduction='mean')
        debugprint(f"  初始化一致性损失函数: {self.consistency_loss}")

        # Set defaults for paths
        self.previous_task_model = previous_task_model
        self.current_task_id = current_task_id
        debugprint(f"  previous_task_model: {self.previous_task_model}")
        debugprint(f"  current_task_id: {self.current_task_id}")

        logger.info_rank0(f"Initialized ILORA without buffer, EMA alpha {self.ema_alpha}, and regularization weight {self.reg_weight}.")
        debugprint("退出 ILORA.__init__")

    def update_ema_weights(self) -> None:
        """
        Update EMA adapter weights from the current adapter (default).
        Handles different DeepSpeed ZeRO stages appropriately.
        """
        debugprint("进入 ILORA.update_ema_weights")
        if not hasattr(self.model, "peft_config"):
            logger.warning_rank0("Model is not a PEFT model, cannot update EMA weights.")
            debugprint("  模型不是 PEFT 模型，跳过 EMA 更新")
            return

        # 检查模型是否同时具有 default 和 ema 适配器
        has_default = "default" in self.model.peft_config
        has_ema = "ema" in self.model.peft_config
        debugprint(f"  检查适配器存在性: default={has_default}, ema={has_ema}")

        if not has_default or not has_ema:
            logger.warning_rank0("Model does not have both 'default' and 'ema' adapters, skipping EMA update.")
            debugprint("  缺少 default 或 ema 适配器，跳过 EMA 更新")
            return

        current_active = None
        if hasattr(self.model, "active_adapter"):
             current_active = self.model.active_adapter
             debugprint(f"  当前活动适配器: {current_active}")
        else:
             debugprint("  警告: 模型没有 active_adapter 属性")
             return # 如果没有活动适配器，无法安全地切换和恢复

        # 检查 DeepSpeed ZeRO 阶段
        zero_stage = get_deepspeed_zero_stage(self.model)
        debugprint(f"  检测到 DeepSpeed ZeRO Stage: {zero_stage}")

        # 获取当前模型参数
        try:
            # 在 ZeRO-3 下，需要使用 gather_parameters 上下文管理器
            # 对于 ZeRO-1/2 或非 DeepSpeed，gather_parameters 返回 nullcontext
            with gather_parameters(self.model) if zero_stage == 3 else nullcontext():
                debugprint("  切换到 default 适配器获取参数")
                self.model.set_adapter("default")
                default_state = {}
                # 单独收集 default 适配器的可训练参数 (LoRA 参数)
                for name, param in self.model.named_parameters():
                    # 检查是否为 LoRA 参数且需要梯度
                    if "lora" in name and param.requires_grad:
                        # 提取相对于 base_model.model 的名称，用于匹配 EMA 参数
                        # 例如: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
                        rel_name = name.split("base_model.model.")[-1]
                        default_state[rel_name] = param.data.clone()
                        # debugprint(f"    获取 default 参数: {rel_name} (来自 {name})")

                # 如果没有参数，可能是配置错误
                if not default_state:
                    logger.warning_rank0("Default adapter appears to have no trainable LoRA parameters, skipping EMA update.")
                    debugprint("  Default 适配器没有可训练的 LoRA 参数，跳过 EMA 更新")
                    self.model.set_adapter(current_active)
                    return
                debugprint(f"  获取了 {len(default_state)} 个 default LoRA 参数")

                # 现在切换到 EMA 适配器并更新其参数
                debugprint("  切换到 ema 适配器进行更新")
                self.model.set_adapter("ema")
                update_count = 0
                # 遍历 EMA 适配器的参数 (这些参数应该是 frozen 的)
                for name, param in self.model.named_parameters():
                    # 检查是否为 EMA 的 LoRA 参数且不需要梯度
                    if "lora" in name and "ema." in name and not param.requires_grad:
                         # 提取相对于 base_model.model 的名称，并移除 'ema.'
                         # 例如: base_model.model.ema.model.layers.0.self_attn.q_proj.lora_A.weight
                         # -> model.layers.0.self_attn.q_proj.lora_A.weight
                         rel_name_ema = name.split("base_model.model.ema.")[-1]
                         # debugprint(f"    检查 EMA 参数: {rel_name_ema} (来自 {name})")

                         if rel_name_ema in default_state:
                             default_param = default_state[rel_name_ema]
                             # EMA 更新公式
                             new_ema_param = self.ema_alpha * default_param.to(param.device) + (1 - self.ema_alpha) * param.data
                             param.data = new_ema_param
                             update_count += 1
                             # debugprint(f"      更新 EMA 参数 {rel_name_ema}")
                         # else:
                             # debugprint(f"      警告: EMA 参数 {rel_name_ema} 在 default_state 中未找到对应项")

                debugprint(f"  更新了 {update_count} 个 EMA LoRA 参数")

                # 恢复原始适配器
                debugprint(f"  恢复活动适配器为: {current_active}")
                self.model.set_adapter(current_active)

        except Exception as e:
            logger.warning_rank0(f"Updating EMA weights failed: {e}", exc_info=True)
            debugprint(f"  更新 EMA 权重时发生异常: {e}")
            if current_active is not None: # 确保在异常时恢复 adapter 状态
                debugprint(f"  (异常处理) 恢复活动适配器为: {current_active}")
                self.model.set_adapter(current_active)
        finally:
             # 确保无论如何都尝试恢复
             if current_active is not None and hasattr(self.model, 'active_adapter') and self.model.active_adapter != current_active:
                  debugprint(f"  (finally 块) 恢复活动适配器为: {current_active}")
                  self.model.set_adapter(current_active)

        debugprint("退出 ILORA.update_ema_weights")

    def compute_consistency_loss(
        self,
        input_ids: torch.Tensor, # 当前 batch 的 input_ids
        attention_mask: Optional[torch.Tensor] = None, # 当前 batch
        labels: Optional[torch.Tensor] = None, # 当前 batch (未使用)
        **kwargs # 传递给 model forward 的其他参数
    ) -> torch.Tensor:
        """
        Compute consistency loss between current model (default adapter) and EMA model (ema adapter).
        Handles different DeepSpeed ZeRO stages appropriately.

        Args:
            input_ids: Input token ids from the current batch (used for device placement).
            attention_mask: Attention mask from the current batch (unused).
            labels: Target labels from the current batch (unused).

        Returns:
            Consistency loss tensor
        """
        debugprint("进入 ILORA.compute_consistency_loss")

        # --- Pre-check for EMA adapter ---
        debugprint("  前置检查: 'ema' 适配器是否存在?")
        if not hasattr(self.model, 'peft_config') or 'ema' not in self.model.peft_config:
            logger.warning_rank0("'ema' adapter not found in model.peft_config at the start of compute_consistency_loss. Returning zero loss.")
            debugprint("  前置检查失败: 'ema' 适配器未找到。直接返回零损失。")
            if hasattr(self.model, 'peft_config'):
                debugprint(f"    当前可用适配器: {list(self.model.peft_config.keys())}")
            else:
                debugprint("    模型没有 peft_config 属性。")
            return torch.tensor(0.0, device=input_ids.device)
        debugprint("  前置检查通过: 'ema' 适配器存在。")
        # --- End Pre-check ---

        target_device = input_ids.device
        # 使用当前批次数据
        debugprint(f"  使用当前批次数据 (device: {target_device}, shape: {input_ids.shape}) 计算一致性损失")

        # 获取当前活动适配器以备恢复
        current_active = None
        if hasattr(self.model, "active_adapter"):
            current_active = self.model.active_adapter
            debugprint(f"  当前活动适配器: {current_active}")
        else:
            debugprint("  警告: 模型没有 active_adapter 属性，无法安全计算一致性损失")
            return torch.tensor(0.0, device=input_ids.device)

        # 检查 DeepSpeed ZeRO 阶段
        zero_stage = get_deepspeed_zero_stage(self.model)
        debugprint(f"  检测到 DeepSpeed ZeRO Stage: {zero_stage}")

        l_cons = torch.tensor(0.0, device=target_device)
        try:
            # 在 ZeRO-3 下，需要使用 gather_parameters 上下文管理器
            # 对于 ZeRO-1/2 或非 DeepSpeed，gather_parameters 返回 nullcontext
            with gather_parameters(self.model) if zero_stage == 3 else nullcontext():
                # 使用 default 适配器进行前向传播
                self.model.set_adapter("default")
                plastic_outputs = self.model(
                    input_ids=input_ids,         # 使用当前批次的 input_ids
                    attention_mask=attention_mask, # 使用当前批次的 attention_mask
                    output_hidden_states=True,
                    return_dict=True,
                    **kwargs
                )
                plastic_hidden = plastic_outputs.hidden_states

                # 使用 ema 适配器进行前向传播 (no_grad)
                with torch.no_grad():
                    self.model.set_adapter("ema")
                    stable_outputs = self.model(
                        input_ids=input_ids,         # 使用当前批次的 input_ids
                        attention_mask=attention_mask, # 使用当前批次的 attention_mask
                        output_hidden_states=True,
                        return_dict=True,
                        **kwargs
                    )
                    stable_hidden = stable_outputs.hidden_states

                if not plastic_hidden or not stable_hidden or len(plastic_hidden) != len(stable_hidden):
                     debugprint("  获取的隐藏状态为空或数量不匹配，无法计算一致性损失")
                     self.model.set_adapter(current_active) # 恢复适配器
                     return l_cons

                # Select which hidden states to compute consistency on
                selected_plastic = []
                selected_stable = []
                debugprint(f"  根据 hidden_state_layers ({self.hidden_state_layers}) 选择隐藏层")
                num_layers = len(plastic_hidden)
                for layer_idx in self.hidden_state_layers:
                    actual_idx = layer_idx if layer_idx >= 0 else num_layers + layer_idx
                    if 0 <= actual_idx < num_layers:
                        selected_plastic.append(plastic_hidden[actual_idx])
                        selected_stable.append(stable_hidden[actual_idx])
                        debugprint(f"    选择层索引: {actual_idx} (原始: {layer_idx})")
                    else:
                        debugprint(f"    警告: 层索引 {actual_idx} (原始: {layer_idx}) 超出范围 [0, {num_layers-1}]，已跳过")

                if not selected_plastic:
                     debugprint("  没有成功选择任何隐藏层，一致性损失为 0")
                     self.model.set_adapter(current_active) # 恢复适配器
                     return l_cons

                debugprint(f"  共选择了 {len(selected_plastic)} 层用于计算一致性损失")
                # Compute consistency loss
                layer_losses = []
                debugprint(f"  计算所选层之间的 MSE 损失 (selective_update: {self.selective_update})")
                for i, (plastic, stable) in enumerate(zip(selected_plastic, selected_stable)):
                    debugprint(f"    计算层 {i+1}/{len(selected_plastic)} 的损失 (shape: {plastic.shape})")
                    # 直接计算 MSE 损失，不使用 selective update （保持与原始论文一致）
                    layer_loss = self.consistency_loss(plastic, stable) # 使用 mean reduction
                    debugprint(f"      层 {i} 损失: {layer_loss.item()}")
                    layer_losses.append(layer_loss)

                # Average losses across all selected layers
                if layer_losses:
                    l_cons = torch.stack(layer_losses).mean()
                    debugprint(f"  最终平均一致性损失 (l_cons): {l_cons.item()}")
                else:
                    debugprint("  layer_losses 列表为空，一致性损失为 0")
                    l_cons = torch.tensor(0.0, device=target_device)

                # 恢复原始适配器
                self.model.set_adapter(current_active)

        except Exception as e:
            logger.warning_rank0(f"Computing consistency loss failed: {e}", exc_info=True)
            debugprint(f"  计算一致性损失时发生异常: {e}")
            l_cons = torch.tensor(0.0, device=target_device) # 异常时返回 0

        finally:
            # Reset adapter to default (or original active) for further training
            if current_active is not None and hasattr(self.model, 'active_adapter') and self.model.active_adapter != current_active:
                debugprint(f"  (finally 块) 恢复活动适配器为: {current_active}")
                self.model.set_adapter(current_active)

        # 在分布式环境中同步损失值
        if is_distributed():
            l_cons = all_reduce_tensor(l_cons)
            debugprint(f"  在分布式环境中同步损失值，同步后: {l_cons.item()}")

        debugprint(f"退出 ILORA.compute_consistency_loss (使用当前批次), 返回损失: {l_cons.item()}")
        return l_cons