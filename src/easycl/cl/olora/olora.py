import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Dict, Optional, Union, List
import logging
from safetensors.torch import load_file as safe_load_file
from pathlib import Path
import re
from easycl.hparams import CLFinetuningArguments
from contextlib import nullcontext
import torch.distributed as dist
from easycl.cl.distributed_utils import (
    get_deepspeed_zero_stage,
    gather_parameters,
    is_main_process,
    broadcast_object,
    is_distributed,
    get_rank,
    get_world_size
)
def debugprint(*args, **kwargs):
    pass

logger = logging.getLogger(__name__)

@dataclass
class AdapterInfo:
    """Store adapter-related information"""
    task_id: str
    path: str
    config: Dict
    is_first_task: bool = False

class OLoRA:
    """O-LoRA implementation for orthogonal constraint and parameter management."""

    def __init__(
        self,
        model: nn.Module,
        orthogonal_lambda: float = 0.1,
        l2_lambda: float = 0.01,
        olora_history_path: str = "olora_history",
        model_output_dir: str = "model_output",
        device: str = "cuda",
        prev_task_id: str = None
    ):
        self.model = model
        self.orthogonal_lambda = orthogonal_lambda
        self.l2_lambda = l2_lambda
        self.olora_history_path = os.path.abspath(olora_history_path)
        self.model_output_dir = os.path.abspath(model_output_dir)
        self.device = device
        self.adapter_history: List[AdapterInfo] = []
        self.prev_task_id = prev_task_id
        self.merged_historical_weights: Optional[Dict[str, torch.Tensor]] = None

        debugprint(f"OLoRA __init__: 初始化参数:")
        debugprint(f"  - orthogonal_lambda: {self.orthogonal_lambda}")
        debugprint(f"  - l2_lambda: {self.l2_lambda}")
        debugprint(f"  - oloara_history_path: {self.olora_history_path}")
        debugprint(f"  - model_output_dir: {self.model_output_dir}")
        debugprint(f"  - device: {self.device}")
        debugprint(f"  - prev_task_id: {self.prev_task_id}")

        # Ensure history directory exists and load adapter history only on main process
        if is_main_process():
            os.makedirs(self.olora_history_path, exist_ok=True)
            # Load existing adapter history
            self._load_adapter_history()

    def _load_adapter_history(self):
        """Load existing adapter history information"""
        if not os.path.exists(self.olora_history_path):
            return

        prev_adapters = sorted([
            f for f in os.listdir(self.olora_history_path)
            if os.path.isdir(os.path.join(self.olora_history_path, f))
        ])

        for adapter_id in prev_adapters:
            adapter_path = os.path.join(self.olora_history_path, adapter_id)
            config_path = os.path.join(adapter_path, "adapter_config.json")

            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)

                adapter_info = AdapterInfo(
                    task_id=adapter_id,
                    path=adapter_path,
                    config=config
                )
                self.adapter_history.append(adapter_info)
                debugprint(f"_load_adapter_history: 加载到历史 adapter: {adapter_info}")

    def _validate_adapter_path(self, adapter_path: str) -> str:
        """Validate and normalize adapter path"""
        debugprint(f"_validate_adapter_path: 正在验证路径: {adapter_path}")
        adapter_path = os.path.abspath(adapter_path)

        if not os.path.exists(adapter_path):
            debugprint(f"_validate_adapter_path: 验证失败 - 路径不存在: {adapter_path}")
            raise ValueError(f"Adapter path does not exist: {adapter_path}")

        config_path = os.path.join(adapter_path, "adapter_config.json")
        model_path = os.path.join(adapter_path, "adapter_model.safetensors")

        if not os.path.exists(config_path):
            debugprint(f"_validate_adapter_path: 验证失败 - adapter_config.json 不存在于 {adapter_path}")
            raise ValueError(f"adapter_config.json not found in {adapter_path}")

        if not os.path.exists(model_path):
            debugprint(f"_validate_adapter_path: 验证失败 - adapter_model.safetensors 不存在于 {adapter_path}")
            raise ValueError(f"adapter_model.safetensors not found in {adapter_path}")

        debugprint(f"_validate_adapter_path: 验证成功: {adapter_path}")
        return adapter_path

    def load_adapter_weights(self, task_id: str, base_path: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Load LoRA weights from adapter file.
        If base_path is provided, it uses that path directly. Otherwise, defaults to model_output_dir.
        """
        # 初始化空字典，非主进程将直接返回这个空字典
        lora_weights = {}

        # 只在主进程上执行文件读取操作
        if is_main_process():
            if base_path:
                adapter_path = os.path.abspath(base_path)
                debugprint(f"load_adapter_weights: 尝试加载 task_id='{task_id}' 的权重，使用提供的 base_path: {adapter_path}")
            else:
                adapter_path = os.path.join(self.model_output_dir)
                debugprint(f"load_adapter_weights: 尝试加载 task_id='{task_id}' 的权重，使用默认路径 (model_output_dir): {adapter_path}")

            try:
                # Note: _validate_adapter_path now receives the determined path
                validated_path = self._validate_adapter_path(adapter_path)
            except ValueError as e:
                logger.error(f"Error validating adapter path: {str(e)}")
                return {}

            try:
                # Read adapter configuration
                config_path = os.path.join(validated_path, "adapter_config.json")
                with open(config_path, 'r') as f:
                    config = json.load(f)

                # Read adapter weights
                model_path = os.path.join(validated_path, "adapter_model.safetensors")
                state_dict = safe_load_file(model_path, device="cpu")

                # Extract LoRA weights
                # Record weight names for debugging
                weight_keys = list(state_dict.keys())
                logger.debug(f"Adapter weights in {task_id}: {weight_keys[:5]}...")
                logger.debug(f"Total weights: {len(weight_keys)}")
                debugprint(f"load_adapter_weights: 从 {validated_path} 加载了 {len(weight_keys)} 个权重张量。示例 keys: {weight_keys[:3]}...")

                for key, value in state_dict.items():
                    # In PEFT format, weights are typically named as:
                    # base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight
                    if re.search(r'\.lora_[AB]\.', key):
                        # Extract module path and weight type from original key
                        module_path = key.split('.lora_')[0]
                        weight_type = 'merged_A' if '.lora_A.' in key else 'merged_B'

                        # Create standardized key format for merging
                        new_key = f"{module_path}.{weight_type}"
                        lora_weights[new_key] = value

                if not lora_weights:
                    logger.warning(f"No LoRA weights found in adapter: {task_id}")
                    debugprint(f"load_adapter_weights: 在 adapter '{task_id}' 中未找到 LoRA 权重。")
                else:
                    logger.info(f"Loaded {len(lora_weights)} LoRA weights from adapter: {task_id}")
                    debugprint(f"load_adapter_weights: 成功加载 {len(lora_weights)} 个 LoRA 权重 (A/B matrices) from adapter: {task_id}")

            except Exception as e:
                logger.error(f"Error loading adapter weights for task {task_id}: {str(e)}")
                logger.error(f"Stack trace:", exc_info=True)
                return {}

        # 如果是分布式环境，将权重广播到所有进程
        # REMOVED: Broadcasting logic is moved to the caller (load_prev_adapter) to avoid nested broadcasts.
        # if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        #     lora_weights = broadcast_object(lora_weights, src=0)

        return lora_weights

    def save_merged_adapter(self, adapter_name: str):
        """Save adapter weights to the history path.
        For the first task: directly save current weights.
        For subsequent tasks: concatenate current weights with previous weights.
        Always uses 'default' adapter weights.
        """
        success = False

        # 只在主进程上执行文件保存操作
        if is_main_process():
            # Build current task's adapter directory path
            adapter_dir = os.path.join(self.olora_history_path, adapter_name)
            current_adapter_dir = os.path.join(self.model_output_dir)
            merged_dir = os.path.join(adapter_dir, "merged")
            os.makedirs(merged_dir, exist_ok=True)

            debugprint(f"save_merged_adapter: 准备保存 adapter '{adapter_name}'")
            debugprint(f"  - 当前 adapter 目录 (源): {current_adapter_dir}")
            debugprint(f"  - 目标历史目录: {adapter_dir}")
            debugprint(f"  - 目标 merged 目录: {merged_dir}")

            try:
                # Validate current adapter path
                adapter_path = self._validate_adapter_path(current_adapter_dir)

                # Read current adapter weights (always using 'default' adapter)
                current_weights = self.load_adapter_weights(adapter_name)

                if not current_weights:
                    logger.error(f"No LoRA weights found in current adapter: {adapter_name}")
                    success = False
                else:
                    # Check if this is the first task
                    is_first_task = self.prev_task_id is None

                    if is_first_task:
                        # First task: directly save current weights
                        debugprint("save_merged_adapter: 检测到是第一个任务。直接保存当前权重。")
                        save_path = os.path.join(merged_dir, "merged_adapter.pt")
                        torch.save(current_weights, save_path)

                        # Update adapter history
                        with open(os.path.join(current_adapter_dir, "adapter_config.json"), 'r') as f:
                            config = json.load(f)

                        self.adapter_history.append(AdapterInfo(
                            task_id=adapter_name,
                            path=current_adapter_dir,
                            config=config,
                            is_first_task=True
                        ))

                        logger.info(f"Saved first task adapter weights to {save_path}")
                        debugprint(f"save_merged_adapter: 第一个任务的 adapter 权重已保存至 {save_path}")
                        success = True

                    else:
                        # Subsequent tasks: load historical weights and concatenate
                        # Add check to prevent IndexError if history is empty for a subsequent task
                        debugprint(f"save_merged_adapter: 检测到是后续任务。前一个任务 ID: {self.prev_task_id}")

                        prev_merged_dir = os.path.join(self.olora_history_path, self.prev_task_id, "merged")
                        prev_path = os.path.join(prev_merged_dir, "merged_adapter.pt")
                        debugprint(f"save_merged_adapter: 尝试加载前一个任务的合并后权重: {prev_path}")

                        if not os.path.exists(prev_path):
                            debugprint(f"save_merged_adapter: 错误 - 未找到前一个任务的合并权重文件: {prev_path}")
                            raise ValueError(f"Previous merged adapter weights not found: {prev_path}")

                        prev_weights = torch.load(prev_path, map_location="cpu")
                        debugprint(f"save_merged_adapter: 成功加载前一个任务的合并权重，包含 {len(prev_weights)} 个张量。")

                        # Merge weights (using concatenation)
                        merged_weights = {}
                        for key in current_weights.keys():
                            if "merged_A" in key or "merged_B" in key:
                                if key in prev_weights:
                                    prev_weight = prev_weights[key]
                                    curr_weight = current_weights[key]

                                    if "merged_A" in key:
                                        # For A matrix, concatenate along output dimension
                                        merged_weights[key] = torch.cat([prev_weight, curr_weight], dim=0)
                                    elif "merged_B" in key:
                                        # For B matrix, concatenate along input dimension
                                        merged_weights[key] = torch.cat([prev_weight, curr_weight], dim=1)
                                else:
                                    merged_weights[key] = current_weights[key]
                            else:
                                # Non-LoRA weights are directly copied
                                merged_weights[key] = current_weights[key]

                        # Save merged weights to current task's merged subdirectory
                        save_path = os.path.join(merged_dir, "merged_adapter.pt")
                        debugprint(f"save_merged_adapter: 准备保存合并后的权重到: {save_path}")
                        torch.save(merged_weights, save_path)

                        # Update adapter history
                        with open(os.path.join(current_adapter_dir, "adapter_config.json"), 'r') as f:
                            config = json.load(f)

                        self.adapter_history.append(AdapterInfo(
                            task_id=adapter_name,
                            path=adapter_dir,
                            config=config
                        ))

                        logger.info(f"Saved concatenated adapter weights for task {adapter_name}")
                        debugprint(f"save_merged_adapter: 任务 '{adapter_name}' 的合并权重已保存。")

                        # Record dimension information for debugging
                        for key, value in merged_weights.items():
                            if "merged_A" in key or "merged_B" in key:
                                logger.debug(f"Merged weight {key} shape: {value.shape}")
                                debugprint(f"save_merged_adapter: 合并权重 {key} 形状: {value.shape}")

                        success = True

            except Exception as e:
                logger.error(f"Error saving adapter weights for task {adapter_name}: {str(e)}")
                logger.error(f"Stack trace:", exc_info=True)
                success = False

        # 添加同步屏障，确保所有进程等待主进程完成保存
        if dist.is_available() and dist.is_initialized():
            # 广播成功状态到所有进程
            success = broadcast_object(success, src=0)
            # 添加同步屏障
            dist.barrier()

        return success

    def get_adapter_info(self, task_id: str) -> Optional[AdapterInfo]:
        """Get adapter information for the specified task"""
        for adapter in self.adapter_history:
            if adapter.task_id == task_id:
                return adapter
        return None

    def compute_orthogonal_loss(self) -> torch.Tensor:
        """Calculate orthogonal loss between old and new LoRA A matrices"""
        # Check if merged historical weights are loaded
        if not self.merged_historical_weights:
            logger.info("No merged historical weights loaded, skipping orthogonal loss calculation")
            debugprint("compute_orthogonal_loss: 未加载合并的历史权重，跳过正交损失计算。")
            return torch.tensor(0.0, device=self.device)

        debugprint(f"compute_orthogonal_loss: 开始计算正交损失 (使用内存中的合并权重)")
        orth_loss = torch.tensor(0.0, device=self.device)
        num_matrices = 0
        matched_weights_count = 0  # 添加计数器，记录成功匹配的历史权重数量

        # 获取DeepSpeed ZeRO阶段
        stage = get_deepspeed_zero_stage(self.model)
        debugprint(f"compute_orthogonal_loss: 检测到 DeepSpeed ZeRO Stage: {stage}")

        # 定义上下文管理器，在ZeRO-3下收集完整参数
        ctx = gather_parameters(self.model) if stage == 3 else nullcontext()

        # Find adapter names list for LoRA modules
        adapter_names = set()

        # 在gather_parameters上下文中查找adapter名称
        with ctx:
            for name, module in self.model.named_modules():
                if hasattr(module, "lora_A") and hasattr(module.lora_A, "keys"):
                    adapter_keys = module.lora_A.keys()
                    adapter_names.update(adapter_keys)

        debugprint(f"compute_orthogonal_loss: 在模型中找到的 adapter names: {adapter_names}")
        if 'default' not in adapter_names:
             logger.warning(f"'default' adapter not found. Found adapters: {adapter_names}. Skipping orthogonal loss.")
             debugprint("compute_orthogonal_loss: 未找到 'default' adapter，跳过正交损失计算。")
             return torch.tensor(0.0, device=self.device)

        debugprint(f"compute_orthogonal_loss: 使用 'default' adapter 和内存中的合并历史权重进行计算。")

        # 在gather_parameters上下文中计算损失
        with ctx:
            # Iterate through modules again to calculate loss
            for name, module in self.model.named_modules():
                # Ensure the module has the default adapter and the necessary structure
                if hasattr(module, "lora_A") and 'default' in module.lora_A:
                    # 在上下文中访问权重
                    new_weight = module.lora_A['default'].weight

                    # Construct key for historical merged weights
                    merged_a_key = f"{name}.merged_A"

                    if merged_a_key in self.merged_historical_weights:
                        # 确保历史权重在正确的设备上
                        old_weight = self.merged_historical_weights[merged_a_key].to(new_weight.device)

                        # Calculate orthogonal loss: |A_new · A_historical^T|
                        try:
                            # Ensure dimensions match for matrix multiplication
                            # new_weight shape: (r_new, K)
                            # old_weight shape: (R_historical, K) -> transpose to (K, R_historical)
                            if new_weight.shape[1] == old_weight.shape[1]:
                                dot_product = torch.mm(new_weight, old_weight.T)
                                curr_loss = torch.abs(dot_product).sum()
                                orth_loss = orth_loss + curr_loss
                                num_matrices += 1
                                matched_weights_count += 1  # 增加匹配计数
                                logger.debug(f"Module {name} orthogonal loss contribution: {curr_loss.item():.4f}, New shape: {new_weight.shape}, Hist shape: {old_weight.shape}")
                            else:
                                 logger.warning(f"Dimension mismatch for orthogonal loss calculation in module {name}: "
                                               f"New A shape {new_weight.shape}, Historical A shape {old_weight.shape}. Skipping.")
                                 debugprint(f"compute_orthogonal_loss: 模块 {name} 维度不匹配，跳过。New: {new_weight.shape}, Hist: {old_weight.shape}")

                        except Exception as e:
                            logger.warning(f"Error calculating orthogonal loss for {name}: {str(e)}")
                            logger.warning(f"Shapes: new={new_weight.shape}, old={old_weight.shape}")
                            debugprint(f"compute_orthogonal_loss: 计算模块 {name} 正交损失时出错: {e}")
                    else:
                        # Log if a corresponding historical weight is missing
                        logger.debug(f"Historical weight key '{merged_a_key}' not found in merged_historical_weights for module {name}.")
                        # debugprint(f"compute_orthogonal_loss: 模块 {name} 的历史权重 key '{merged_a_key}' 未在 merged_historical_weights 中找到。")

        if num_matrices == 0:
            logger.warning(f"No valid matrix pairs found for orthogonal loss calculation using 'default' adapter and historical weights.")
            debugprint(f"compute_orthogonal_loss: 未找到有效的矩阵对进行正交损失计算。")

        final_loss = self.orthogonal_lambda * orth_loss
        debugprint(f"compute_orthogonal_loss: 计算完成。参与计算的矩阵对数量: {num_matrices}, 成功匹配历史权重数量: {matched_weights_count}, 原始 orth_loss: {orth_loss.item():.4f}, 带 lambda ({self.orthogonal_lambda}) 的最终损失: {final_loss.item():.4f}")
        return final_loss

    def compute_l2_loss(self) -> torch.Tensor:
        """Calculate L2 regularization loss for new LoRA parameters"""
        debugprint("compute_l2_loss: 开始计算 L2 正则化损失")
        l2_loss = torch.tensor(0.0, device=self.device)
        num_params = 0

        # 获取DeepSpeed ZeRO阶段
        stage = get_deepspeed_zero_stage(self.model)
        debugprint(f"compute_l2_loss: 检测到 DeepSpeed ZeRO Stage: {stage}")

        # 定义上下文管理器，在ZeRO-3下收集完整参数
        ctx = gather_parameters(self.model) if stage == 3 else nullcontext()

        # 在gather_parameters上下文中查找adapter名称
        adapter_names = set()
        with ctx:
            # Find all adapter names
            for name, module in self.model.named_modules():
                if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                    # Check which adapters this module has
                    if hasattr(module.lora_A, "keys"):
                        adapter_names.update(module.lora_A.keys())

        # If no adapters found, cannot calculate L2 loss
        if not adapter_names:
            logger.warning("No adapters found for L2 loss calculation")
            debugprint("compute_l2_loss: 未找到 adapters，无法计算 L2 损失。")
            return l2_loss

        # 检查是否存在default adapter
        if 'default' not in adapter_names:
            logger.warning("'default' adapter not found for L2 loss calculation")
            logger.warning(f"Available adapters: {adapter_names}")
            debugprint(f"compute_l2_loss: 未找到 'default' adapter。可用 adapters: {adapter_names}")
            return l2_loss

        debugprint(f"compute_l2_loss: 使用 'default' adapter 计算 L2 损失。")

        # 在gather_parameters上下文中计算L2损失
        with ctx:
            # Calculate L2 regularization loss for default adapter parameters
            for name, module in self.model.named_modules():
                if hasattr(module, "lora_A") and 'default' in module.lora_A:
                    # 在上下文中访问权重
                    a_weight = module.lora_A['default'].weight

                    # 确保权重不是空的（在ZeRO-3中可能发生）
                    if a_weight.numel() > 0:
                        # L2 norm for A matrix
                        a_norm_sq = torch.sum(a_weight ** 2)
                        l2_loss = l2_loss + a_norm_sq
                        num_params += 1

                        # L2 norm for B matrix
                        # Ensure lora_B exists and has the default adapter key
                        if hasattr(module, "lora_B") and 'default' in module.lora_B:
                            b_weight = module.lora_B['default'].weight
                            if b_weight.numel() > 0:
                                b_norm_sq = torch.sum(b_weight ** 2)
                                l2_loss = l2_loss + b_norm_sq
                                num_params += 1
                                logger.debug(f"Module {name} L2 squared norms: A={a_norm_sq.item():.4f}, B={b_norm_sq.item():.4f}")
                                # debugprint(f"compute_l2_loss: 模块 {name} - A 平方和: {a_norm_sq.item():.4f}, B 平方和: {b_norm_sq.item():.4f}")
                        else:
                            logger.debug(f"Module {name} L2 squared norm: A={a_norm_sq.item():.4f} (B not found or missing adapter)")
                            # debugprint(f"compute_l2_loss: 模块 {name} - A 平方和: {a_norm_sq.item():.4f} (B 未找到或缺失)")

        if num_params > 0:
            # The loss is already the sum of squares, which is the standard L2 penalty term (before lambda)
            pass
        else:
            logger.warning(f"No parameters found for L2 loss with 'default' adapter")
            debugprint(f"compute_l2_loss: 未找到用于计算 L2 损失的参数 (adapter: 'default')")

        final_loss = self.l2_lambda * l2_loss
        debugprint(f"compute_l2_loss: 计算完成。参与计算的参数数量: {num_params}, 原始 l2_loss: {l2_loss.item():.4f}, 带 lambda ({self.l2_lambda}) 的最终损失: {final_loss.item():.4f}")
        return final_loss

    def load_prev_adapter(self, prev_task_id: str) -> bool:
        """Load previous task's merged adapter parameters into memory.
        Prioritizes loading merged/merged_adapter.pt.
        If merged weights are not found, tries loading adapter_model.safetensors for the same task ID.
        Does NOT apply weights to the model's 'default' adapter anymore.
        Stores the loaded weights (either merged or from safetensors) in self.merged_historical_weights.
        Returns True if merged weights were successfully loaded into memory, False otherwise.
        """
        # 初始化变量
        self.merged_historical_weights = None
        load_success = False
        self.prev_task_id = prev_task_id  # Store for reference if needed elsewhere

        # If no prev_task_id provided, this is the first task
        if prev_task_id is None:
            logger.info(f"No previous task ID provided. This seems to be the first task.")
            debugprint(f"load_prev_adapter: prev_task_id 为 None，视为第一个任务。返回 False。")
            self.prev_task_id = None
            return False  # Cannot load previous if none exists

        # 只在主进程上执行文件读取操作
        if is_main_process():
            # Try loading merged weights from the specified previous task's history
            merged_dir = os.path.join(self.olora_history_path, prev_task_id, "merged")
            merged_load_path = os.path.join(merged_dir, "merged_adapter.pt")

            debugprint(f"load_prev_adapter: 尝试加载 prev_task_id='{prev_task_id}' 的合并权重。")
            debugprint(f"  - 尝试路径: {merged_load_path}")

            temp_weights = None  # 临时存储加载的权重

            if os.path.exists(merged_load_path):
                try:
                    # Load weights to CPU first to avoid potential GPU memory issues with large merged files
                    temp_weights = torch.load(merged_load_path, map_location="cpu")
                    logger.info(f"Successfully loaded {len(temp_weights)} merged historical weights from {merged_load_path} into memory.")
                    debugprint(f"load_prev_adapter: 成功从 {merged_load_path} 加载 {len(temp_weights)} 个合并权重到内存。")
                    # Log shapes of a few weights for debugging
                    keys_to_log = list(temp_weights.keys())[:3]
                    for key in keys_to_log:
                        if isinstance(temp_weights[key], torch.Tensor):
                            logger.debug(f"  - Loaded weight '{key}' shape: {temp_weights[key].shape}")
                            debugprint(f"  - 加载的权重 '{key}' 形状: {temp_weights[key].shape}")
                    load_success = True
                except Exception as e:
                    logger.error(f"Error loading merged historical weights from {merged_load_path}: {str(e)}")
                    logger.error(f"Stack trace:", exc_info=True)
                    debugprint(f"load_prev_adapter: 从 {merged_load_path} 加载合并权重失败: {str(e)}。将尝试加载 safetensors。")

                    # --- Fallback logic starts here ---
                    logger.warning(f"Could not load merged weights from {merged_load_path}. Trying adapter_model.safetensors for task '{prev_task_id}'.")
                    debugprint(f"load_prev_adapter: 尝试加载 task '{prev_task_id}' 的 adapter_model.safetensors。")
                    prev_task_history_path = os.path.join(self.olora_history_path, prev_task_id)  # Define history path
                    try:
                        # Use load_adapter_weights which handles loading safetensors and converting keys
                        # Pass the correct base_path for history
                        safetensors_weights = self.load_adapter_weights(prev_task_id, base_path=prev_task_history_path)
                        if safetensors_weights:
                            temp_weights = safetensors_weights
                            logger.info(f"Successfully loaded {len(safetensors_weights)} weights from adapter_model.safetensors in {prev_task_history_path} for task '{prev_task_id}' into memory.")
                            debugprint(f"load_prev_adapter: 成功从 {prev_task_history_path} 的 adapter_model.safetensors 加载 {len(safetensors_weights)} 个权重到内存。")
                            # Log shapes of a few weights for debugging
                            keys_to_log = list(temp_weights.keys())[:3]
                            for key in keys_to_log:
                                if isinstance(temp_weights[key], torch.Tensor):
                                    logger.debug(f"  - Loaded weight '{key}' shape: {temp_weights[key].shape}")
                                    debugprint(f"  - 加载的权重 '{key}' 形状: {temp_weights[key].shape}")
                            load_success = True
                    except Exception as safetensors_e:
                        logger.error(f"Error loading adapter_model.safetensors for task '{prev_task_id}': {str(safetensors_e)}")
                        debugprint(f"load_prev_adapter: 加载 adapter_model.safetensors 时出错: {safetensors_e}")
                    # --- Fallback logic ends here ---
            else:
                logger.warning(f"Merged historical adapter weights not found at {merged_load_path} for task '{prev_task_id}'. Trying adapter_model.safetensors.")
                debugprint(f"load_prev_adapter: 未找到任务 '{prev_task_id}' 的合并权重文件: {merged_load_path}。将尝试加载 safetensors。")
                # --- Fallback logic starts here ---
                prev_task_history_path = os.path.join(self.olora_history_path, prev_task_id)  # Define history path
                try:
                    # Use load_adapter_weights which handles loading safetensors and converting keys
                    # Pass the correct base_path for history
                    safetensors_weights = self.load_adapter_weights(prev_task_id, base_path=prev_task_history_path)
                    if safetensors_weights:
                        temp_weights = safetensors_weights
                        logger.info(f"Successfully loaded {len(safetensors_weights)} weights from adapter_model.safetensors in {prev_task_history_path} for task '{prev_task_id}' into memory.")
                        debugprint(f"load_prev_adapter: 成功从 {prev_task_history_path} 的 adapter_model.safetensors 加载 {len(safetensors_weights)} 个权重到内存。")
                        # Log shapes of a few weights for debugging
                        keys_to_log = list(temp_weights.keys())[:3]
                        for key in keys_to_log:
                            if isinstance(temp_weights[key], torch.Tensor):
                                logger.debug(f"  - Loaded weight '{key}' shape: {temp_weights[key].shape}")
                                debugprint(f"  - 加载的权重 '{key}' 形状: {temp_weights[key].shape}")
                        load_success = True
                except Exception as safetensors_e:
                    logger.error(f"Error loading adapter_model.safetensors for task '{prev_task_id}': {str(safetensors_e)}")
                    debugprint(f"load_prev_adapter: 加载 adapter_model.safetensors 时出错: {safetensors_e}")
                # --- Fallback logic ends here ---

        # 广播加载状态和权重到所有进程
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        debugprint(f"load_prev_adapter: [rank {rank}/{world_size-1}] 准备广播 load_success ({load_success}) 和权重。")
        debugprint(f"[DEBUG] load_prev_adapter: [rank {rank}/{world_size-1}] 准备广播 load_success ({load_success}) 和权重。")

        if dist.is_available() and dist.is_initialized():
            try:
                # 首先同步所有进程，确保主进程已完成文件读取
                debugprint(f"load_prev_adapter: [rank {rank}/{world_size-1}] 即将进入第一个 barrier (确保主进程文件读取完成)")
                debugprint(f"[DEBUG] load_prev_adapter: [rank {rank}/{world_size-1}] 即将进入第一个 barrier (确保主进程文件读取完成)")

                # 添加每个进程的状态打印
                for i in range(world_size):
                    if rank == i:
                        debugprint(f"[DEBUG] load_prev_adapter: [rank {rank}/{world_size-1}] 正在等待进入第一个barrier")

                dist.barrier()

                debugprint(f"[DEBUG] load_prev_adapter: [rank {rank}/{world_size-1}] 已通过第一个 barrier")
                debugprint(f"load_prev_adapter: [rank {rank}/{world_size-1}] 已通过第一个 barrier")

                # 广播加载状态
                debugprint(f"[DEBUG] load_prev_adapter: [rank {rank}/{world_size-1}] 即将广播 load_success...")
                debugprint(f"load_prev_adapter: [rank {rank}/{world_size-1}] 即将广播 load_success...")

                # 添加每个进程的状态打印
                for i in range(world_size):
                    if rank == i:
                        debugprint(f"[DEBUG] load_prev_adapter: [rank {rank}/{world_size-1}] 准备广播/接收 load_success_tensor")

                load_success_tensor = torch.tensor(int(load_success), dtype=torch.int, device=self.device) # Use tensor for broadcast
                dist.broadcast(load_success_tensor, src=0)
                load_success = bool(load_success_tensor.item()) # Convert back to bool

                debugprint(f"[DEBUG] load_prev_adapter: [rank {rank}/{world_size-1}] 已完成广播/接收 load_success: {load_success}")
                debugprint(f"load_prev_adapter: [rank {rank}/{world_size-1}] 已完成广播/接收 load_success: {load_success}")


                # 如果主进程成功加载了权重，则广播权重到所有进程
                if load_success:
                    debugprint(f"[DEBUG] load_prev_adapter: [rank {rank}/{world_size-1}] load_success 为 True，准备广播/接收权重...")
                    debugprint(f"load_prev_adapter: [rank {rank}/{world_size-1}] load_success 为 True，准备广播/接收权重...")

                    # 添加每个进程的状态打印
                    for i in range(world_size):
                        if rank == i:
                            debugprint(f"[DEBUG] load_prev_adapter: [rank {rank}/{world_size-1}] 准备广播/接收权重对象")

                    # Use torch.distributed primitives for broadcasting large objects if needed
                    # For simplicity, sticking with broadcast_object for now, but adding more logs
                    broadcast_list = [None] # Placeholder for broadcast_object
                    if is_main_process():
                        self.merged_historical_weights = temp_weights
                        broadcast_list[0] = self.merged_historical_weights

                        # 打印权重对象的大小信息
                        weights_size = sum(param.numel() * param.element_size() for param in self.merged_historical_weights.values() if isinstance(param, torch.Tensor)) / (1024 * 1024)
                        debugprint(f"[DEBUG] load_prev_adapter: [rank {rank}/{world_size-1}] (主进程) 准备广播权重对象，大小约 {weights_size:.2f} MB...")
                        debugprint(f"load_prev_adapter: [rank {rank}/{world_size-1}] (主进程) 准备广播权重对象，大小约 {weights_size:.2f} MB...")

                        debugprint(f"[DEBUG] load_prev_adapter: [rank {rank}/{world_size-1}] (主进程) 开始广播权重对象...")
                        dist.broadcast_object_list(broadcast_list, src=0)
                        debugprint(f"[DEBUG] load_prev_adapter: [rank {rank}/{world_size-1}] (主进程) 已发送权重对象。")
                        debugprint(f"load_prev_adapter: [rank {rank}/{world_size-1}] (主进程) 已发送权重对象。")
                    else:
                        debugprint(f"[DEBUG] load_prev_adapter: [rank {rank}/{world_size-1}] (非主进程) 准备接收权重对象...")
                        debugprint(f"load_prev_adapter: [rank {rank}/{world_size-1}] (非主进程) 准备接收权重对象...")

                        debugprint(f"[DEBUG] load_prev_adapter: [rank {rank}/{world_size-1}] (非主进程) 开始接收权重对象...")
                        dist.broadcast_object_list(broadcast_list, src=0)
                        self.merged_historical_weights = broadcast_list[0]
                        weight_count = len(self.merged_historical_weights) if self.merged_historical_weights is not None else 0

                        debugprint(f"[DEBUG] load_prev_adapter: [rank {rank}/{world_size-1}] (非主进程) 已接收权重对象，数量: {weight_count}。")
                        debugprint(f"load_prev_adapter: [rank {rank}/{world_size-1}] (非主进程) 已接收权重对象，数量: {weight_count}。")

                # 最后再次同步所有进程，确保所有进程都完成了权重加载/跳过
                debugprint(f"[DEBUG] load_prev_adapter: [rank {rank}/{world_size-1}] 即将进入最终 barrier (确保所有进程完成权重处理)")
                debugprint(f"load_prev_adapter: [rank {rank}/{world_size-1}] 即将进入最终 barrier (确保所有进程完成权重处理)")

                # 添加每个进程的状态打印
                for i in range(world_size):
                    if rank == i:
                        debugprint(f"[DEBUG] load_prev_adapter: [rank {rank}/{world_size-1}] 正在等待进入最终barrier")

                dist.barrier()

                debugprint(f"[DEBUG] load_prev_adapter: [rank {rank}/{world_size-1}] 已通过最终 barrier")
                debugprint(f"load_prev_adapter: [rank {rank}/{world_size-1}] 已通过最终 barrier")

            except Exception as e:
                # 捕获任何异常，防止某个进程卡住
                logger.error(f"[rank {rank}] Error in distributed communication during load_prev_adapter: {str(e)}", exc_info=True)
                debugprint(f"load_prev_adapter: [rank {rank}] 分布式通信出错: {str(e)}")
                # 尝试执行一次barrier，帮助其他进程解除阻塞，但要小心这本身也可能导致问题
                try:
                    debugprint(f"load_prev_adapter: [rank {rank}] 错误处理中，尝试进入错误处理 barrier")
                    dist.barrier()
                    debugprint(f"load_prev_adapter: [rank {rank}] 错误处理中，已通过错误处理 barrier")
                except Exception as barrier_e:
                    logger.error(f"[rank {rank}] Error during error handling barrier in load_prev_adapter: {str(barrier_e)}", exc_info=True)
                    debugprint(f"load_prev_adapter: [rank {rank}] 错误处理 barrier 出错: {str(barrier_e)}")
                # 设置加载失败
                load_success = False
                self.merged_historical_weights = None

        else:
            # 非分布式环境，直接设置权重
             debugprint(f"load_prev_adapter: [rank {rank}] 非分布式环境。")
             if is_main_process() and load_success:
                 self.merged_historical_weights = temp_weights

        debugprint(f"load_prev_adapter: [rank {rank}] 函数即将返回 load_success: {load_success}")
        return load_success

    def init_new_adapter(self, adapter_name: str):
        """Initialize new adapter parameters - This is usually handled by the PEFT library itself, this method may not be needed"""
        logger.info(f"Adapter initialization is handled by PEFT library. This method may not be needed.")
        pass

    def setup_adapters(self, current_task_id: str = "current") -> bool:
        """
        此函数已被弃用，因为 O-LoRA 实现中不再需要创建 "current" adapter。
        所有的正交损失和 L2 损失计算都只使用 'default' adapter。

        保留此函数仅为了向后兼容，始终返回 True。
        """
        debugprint(f"setup_adapters: 此函数已被弃用，不再创建 'current' adapter。所有计算只使用 'default' adapter。")
        logger.info_rank0("setup_adapters function is deprecated. All calculations now use only the 'default' adapter.")

        # 添加同步屏障确保所有进程同步
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        return True
