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

        # Ensure history directory exists
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
            lora_weights = {}
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
                
            return lora_weights
            
        except Exception as e:
            logger.error(f"Error loading adapter weights for task {task_id}: {str(e)}")
            logger.error(f"Stack trace:", exc_info=True)
            return {}

    def save_merged_adapter(self, adapter_name: str):
        """Save adapter weights to the history path.
        For the first task: directly save current weights.
        For subsequent tasks: concatenate current weights with previous weights.
        """
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
            
            # Read current adapter weights
            current_weights = self.load_adapter_weights(adapter_name)
            
            if not current_weights:
                logger.error(f"No LoRA weights found in current adapter: {adapter_name}")
                return False
                
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
                
            return True
            
        except Exception as e:
            logger.error(f"Error saving adapter weights for task {adapter_name}: {str(e)}")
            logger.error(f"Stack trace:", exc_info=True)
            return False

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

        # Find adapter names list for LoRA modules
        adapter_names = set()
        current_task_adapter_name = None # Will be determined below

        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module.lora_A, "keys"):
                adapter_keys = module.lora_A.keys()
                adapter_names.update(adapter_keys)
                # Identify the current task adapter (assuming it's not 'default')
                for key in adapter_keys:
                    if key != 'default':
                        current_task_adapter_name = key
                        break # Assume only one non-default active adapter for the current task

        debugprint(f"compute_orthogonal_loss: 在模型中找到的 adapter names: {adapter_names}")
        if current_task_adapter_name is None:
             logger.warning(f"Could not identify the current task adapter (non-default). Found adapters: {adapter_names}. Skipping orthogonal loss.")
             debugprint("compute_orthogonal_loss: 未能识别当前任务 adapter，跳过正交损失计算。")
             return torch.tensor(0.0, device=self.device)

        debugprint(f"compute_orthogonal_loss: 使用 current_task_adapter='{current_task_adapter_name}' 和内存中的合并历史权重进行计算。")

        # Iterate through modules again to calculate loss
        for name, module in self.model.named_modules():
            # Ensure the module has the current adapter and the necessary structure
            if hasattr(module, "lora_A") and current_task_adapter_name in module.lora_A:
                new_weight = module.lora_A[current_task_adapter_name].weight

                # Construct key for historical merged weights
                merged_a_key = f"{name}.merged_A"

                if merged_a_key in self.merged_historical_weights:
                    old_weight = self.merged_historical_weights[merged_a_key].to(self.device) # Ensure device consistency

                    # Calculate orthogonal loss: |A_new · A_historical^T|
                    try:
                        # Ensure dimensions match for matrix multiplication
                        # new_weight shape: (r_new, K)
                        # old_weight shape: (R_historical, K) -> transpose to (K, R_historical)
                        if new_weight.shape[1] == old_weight.shape[1]:
                            dot_product = torch.mm(new_weight, old_weight.T)
                            curr_loss = torch.abs(dot_product).sum()
                            orth_loss += curr_loss
                            num_matrices += 1
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
            logger.warning(f"No valid matrix pairs found for orthogonal loss calculation using current adapter '{current_task_adapter_name}' and historical weights.")
            debugprint(f"compute_orthogonal_loss: 未找到有效的矩阵对进行正交损失计算。")

        final_loss = self.orthogonal_lambda * orth_loss
        debugprint(f"compute_orthogonal_loss: 计算完成。参与计算的矩阵对数量: {num_matrices}, 原始 orth_loss: {orth_loss.item():.4f}, 带 lambda ({self.orthogonal_lambda}) 的最终损失: {final_loss.item():.4f}")
        return final_loss

    def compute_l2_loss(self) -> torch.Tensor:
        """Calculate L2 regularization loss for new LoRA parameters"""
        debugprint("compute_l2_loss: 开始计算 L2 正则化损失")
        l2_loss = torch.tensor(0.0, device=self.device)
        num_params = 0
        
        # Find all adapter names
        adapter_names = set()
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
            
        # Determine current adapter (non-default adapter)
        current_adapter = None
        if 'default' in adapter_names and len(adapter_names) > 1:
            # Find first non-default adapter
            for name in adapter_names:
                if name != 'default':
                    current_adapter = name
                    break
        # If only one adapter, use it regardless of name
        elif len(adapter_names) == 1:
            current_adapter = list(adapter_names)[0]
        
        if current_adapter is None:
            logger.warning("Could not determine current adapter for L2 loss")
            logger.warning(f"Available adapters: {adapter_names}")
            debugprint(f"compute_l2_loss: 无法确定当前 adapter。可用 adapters: {adapter_names}")
            return l2_loss
            
        debugprint(f"compute_l2_loss: 使用 current_adapter='{current_adapter}' 计算 L2 损失。")
        # Calculate L2 regularization loss for specified adapter parameters
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and current_adapter in module.lora_A:
                # L2 norm for A matrix
                a_norm = torch.norm(module.lora_A[current_adapter].weight)
                l2_loss += a_norm
                num_params += 1
                
                # L2 norm for B matrix
                # Ensure lora_B exists and has the current_adapter key
                if hasattr(module, "lora_B") and current_adapter in module.lora_B:
                    b_norm = torch.norm(module.lora_B[current_adapter].weight)
                    l2_loss += b_norm
                    num_params += 1
                    logger.debug(f"Module {name} L2 norms: A={a_norm.item():.4f}, B={b_norm.item():.4f}")
                    # debugprint(f"compute_l2_loss: 模块 {name} - A 范数: {a_norm.item():.4f}, B 范数: {b_norm.item():.4f}")
                else:
                    logger.debug(f"Module {name} L2 norm: A={a_norm.item():.4f} (B not found or missing adapter)")
                    # debugprint(f"compute_l2_loss: 模块 {name} - A 范数: {a_norm.item():.4f} (B 未找到或缺失)")

        if num_params > 0:
            l2_loss = l2_loss
        else:
            logger.warning(f"No parameters found for L2 loss with adapter '{current_adapter}'")
            debugprint(f"compute_l2_loss: 未找到用于计算 L2 损失的参数 (adapter: {current_adapter})")
            
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
        # Reset historical weights at the beginning
        self.merged_historical_weights = None

        # If no prev_task_id provided, this is the first task
        if prev_task_id is None:
            logger.info(f"No previous task ID provided. This seems to be the first task.")
            debugprint(f"load_prev_adapter: prev_task_id 为 None，视为第一个任务。返回 False。")
            self.prev_task_id = None
            return False # Cannot load previous if none exists

        self.prev_task_id = prev_task_id # Store for reference if needed elsewhere

        # Try loading merged weights from the specified previous task's history
        merged_dir = os.path.join(self.olora_history_path, prev_task_id, "merged")
        merged_load_path = os.path.join(merged_dir, "merged_adapter.pt")

        debugprint(f"load_prev_adapter: 尝试加载 prev_task_id='{prev_task_id}' 的合并权重。")
        debugprint(f"  - 尝试路径: {merged_load_path}")

        if os.path.exists(merged_load_path):
            try:
                # Load weights to CPU first to avoid potential GPU memory issues with large merged files
                loaded_state_dict = torch.load(merged_load_path, map_location="cpu")
                self.merged_historical_weights = loaded_state_dict # Store in memory
                logger.info(f"Successfully loaded {len(loaded_state_dict)} merged historical weights from {merged_load_path} into memory.")
                debugprint(f"load_prev_adapter: 成功从 {merged_load_path} 加载 {len(loaded_state_dict)} 个合并权重到内存。")
                # Log shapes of a few weights for debugging
                keys_to_log = list(self.merged_historical_weights.keys())[:3]
                for key in keys_to_log:
                     if isinstance(self.merged_historical_weights[key], torch.Tensor):
                         logger.debug(f"  - Loaded weight '{key}' shape: {self.merged_historical_weights[key].shape}")
                         debugprint(f"  - 加载的权重 '{key}' 形状: {self.merged_historical_weights[key].shape}")

                return True # Indicate success in loading historical weights
            except Exception as e:
                logger.error(f"Error loading merged historical weights from {merged_load_path}: {str(e)}")
                logger.error(f"Stack trace:", exc_info=True)
                debugprint(f"load_prev_adapter: 从 {merged_load_path} 加载合并权重失败: {str(e)}。将尝试加载 safetensors。")
                self.merged_historical_weights = None # Ensure it's None if loading failed
                # --- Fallback logic starts here ---
                logger.warning(f"Could not load merged weights from {merged_load_path}. Trying adapter_model.safetensors for task '{prev_task_id}'.")
                debugprint(f"load_prev_adapter: 尝试加载 task '{prev_task_id}' 的 adapter_model.safetensors。")
                prev_task_history_path = os.path.join(self.olora_history_path, prev_task_id) # Define history path
                try:
                    # Use load_adapter_weights which handles loading safetensors and converting keys
                    # Pass the correct base_path for history
                    safetensors_weights = self.load_adapter_weights(prev_task_id, base_path=prev_task_history_path)
                    if safetensors_weights:
                        self.merged_historical_weights = safetensors_weights
                        logger.info(f"Successfully loaded {len(safetensors_weights)} weights from adapter_model.safetensors in {prev_task_history_path} for task '{prev_task_id}' into memory.")
                        debugprint(f"load_prev_adapter: 成功从 {prev_task_history_path} 的 adapter_model.safetensors 加载 {len(safetensors_weights)} 个权重到内存。")
                        # Log shapes of a few weights for debugging
                        keys_to_log = list(self.merged_historical_weights.keys())[:3]
                        for key in keys_to_log:
                            if isinstance(self.merged_historical_weights[key], torch.Tensor):
                                logger.debug(f"  - Loaded weight '{key}' shape: {self.merged_historical_weights[key].shape}")
                                debugprint(f"  - 加载的权重 '{key}' 形状: {self.merged_historical_weights[key].shape}")
                        return True # Indicate success from fallback
                    else:
                        logger.warning(f"Failed to load weights from adapter_model.safetensors for task '{prev_task_id}'.")
                        debugprint(f"load_prev_adapter: 从 adapter_model.safetensors 加载权重失败。")
                        return False # Both methods failed
                except Exception as safetensors_e:
                    logger.error(f"Error loading adapter_model.safetensors for task '{prev_task_id}': {str(safetensors_e)}")
                    debugprint(f"load_prev_adapter: 加载 adapter_model.safetensors 时出错: {safetensors_e}")
                    return False # Both methods failed
                # --- Fallback logic ends here ---
        else:
            logger.warning(f"Merged historical adapter weights not found at {merged_load_path} for task '{prev_task_id}'. Trying adapter_model.safetensors.")
            debugprint(f"load_prev_adapter: 未找到任务 '{prev_task_id}' 的合并权重文件: {merged_load_path}。将尝试加载 safetensors。")
            # --- Fallback logic starts here ---
            prev_task_history_path = os.path.join(self.olora_history_path, prev_task_id) # Define history path
            try:
                # Use load_adapter_weights which handles loading safetensors and converting keys
                # Pass the correct base_path for history
                safetensors_weights = self.load_adapter_weights(prev_task_id, base_path=prev_task_history_path)
                if safetensors_weights:
                    self.merged_historical_weights = safetensors_weights
                    logger.info(f"Successfully loaded {len(safetensors_weights)} weights from adapter_model.safetensors in {prev_task_history_path} for task '{prev_task_id}' into memory.")
                    debugprint(f"load_prev_adapter: 成功从 {prev_task_history_path} 的 adapter_model.safetensors 加载 {len(safetensors_weights)} 个权重到内存。")
                    # Log shapes of a few weights for debugging
                    keys_to_log = list(self.merged_historical_weights.keys())[:3]
                    for key in keys_to_log:
                        if isinstance(self.merged_historical_weights[key], torch.Tensor):
                            logger.debug(f"  - Loaded weight '{key}' shape: {self.merged_historical_weights[key].shape}")
                            debugprint(f"  - 加载的权重 '{key}' 形状: {self.merged_historical_weights[key].shape}")
                    return True # Indicate success from fallback
                else:
                    logger.warning(f"Failed to load weights from adapter_model.safetensors for task '{prev_task_id}'.")
                    debugprint(f"load_prev_adapter: 从 adapter_model.safetensors 加载权重失败。")
                    return False # Both methods failed
            except Exception as safetensors_e:
                logger.error(f"Error loading adapter_model.safetensors for task '{prev_task_id}': {str(safetensors_e)}")
                debugprint(f"load_prev_adapter: 加载 adapter_model.safetensors 时出错: {safetensors_e}")
                return False # Both methods failed
            # --- Fallback logic ends here ---

    def init_new_adapter(self, adapter_name: str):
        """Initialize new adapter parameters - This is usually handled by the PEFT library itself, this method may not be needed"""
        logger.info(f"Adapter initialization is handled by PEFT library. This method may not be needed.")
        pass

    def setup_adapters(self, current_task_id: str = "current") -> bool:
        """
        Set up second adapter needed for O-LoRA training for orthogonal constraint.
        Use 'default' as previous adapter name and 'current' as current adapter name by default.
        """
        try:
            debugprint(f"setup_adapters: 开始设置 adapter，目标 current_task_id='{current_task_id}'")
            # Check if model has LoRA attributes
            has_lora = False
            for name, module in self.model.named_modules():
                if hasattr(module, "lora_A"):
                    has_lora = True
                    break
                    
            if not has_lora:
                logger.warning("Model does not have LoRA modules, cannot set up adapters")
                debugprint("setup_adapters: 模型没有 LoRA 模块，无法设置 adapters。返回 False。")
                return False
                
            # Get existing adapter names
            existing_adapters = set()
            for name, module in self.model.named_modules():
                if hasattr(module, "lora_A") and hasattr(module.lora_A, "keys"):
                    existing_adapters.update(module.lora_A.keys())
                    
            logger.info(f"Existing adapters before setup: {existing_adapters}")
            debugprint(f"setup_adapters: 设置前的 existing_adapters: {existing_adapters}")
            
            # If already have two adapters, check if they include default and current
            if len(existing_adapters) >= 2 and 'default' in existing_adapters and current_task_id in existing_adapters:
                logger.info(f"Both 'default' and '{current_task_id}' adapters already exist")
                debugprint(f"setup_adapters: 'default' 和 '{current_task_id}' adapters 已存在。返回 True。")
                return True
                
            # If only have default adapter, create second adapter
            if 'default' in existing_adapters and current_task_id not in existing_adapters:
                debugprint(f"setup_adapters: 只有 'default' adapter，开始创建 '{current_task_id}' adapter。")
                # Use PEFT library interface to add new adapter
                from peft.tuners.lora import LoraLayer
                
                # Iterate through all LoRA modules
                for name, module in self.model.named_modules():
                    if isinstance(module, LoraLayer):
                        # Copy default adapter configuration
                        if not hasattr(module, "lora_A") or 'default' not in module.lora_A:
                            continue
                            
                        # Get adapter configuration
                        r = module.r['default']  # LoRA rank
                        lora_alpha = module.lora_alpha['default']
                        lora_dropout = module.lora_dropout['default'] if hasattr(module, "lora_dropout") else 0.0
                        
                        # Get weight shapes
                        debugprint(f"setup_adapters: 正在为模块 {name} 创建 '{current_task_id}' adapter (基于 'default')。r={r}, alpha={lora_alpha}")
                        if hasattr(module, "lora_A") and hasattr(module.lora_A['default'], "weight"):
                            weight_shape = module.lora_A['default'].weight.shape
                            
                            # Create new A and B matrices
                            module.lora_A[current_task_id] = type(module.lora_A['default'])(
                                weight_shape[1], r, bias=False
                            ).to(self.device)
                            
                            B_shape = module.lora_B['default'].weight.shape
                            module.lora_B[current_task_id] = type(module.lora_B['default'])(
                                B_shape[0], r, bias=False
                            ).to(self.device)
                            
                            # Initialize weights - use same initialization as normal LoRA
                            if hasattr(module.lora_A[current_task_id], "weight"):
                                # Use kaiming initialization by default
                                nn.init.kaiming_uniform_(module.lora_A[current_task_id].weight, a=math.sqrt(5))
                                # Set B matrix to zero
                                nn.init.zeros_(module.lora_B[current_task_id].weight)
                                
                            # Ensure other configurations match default adapter
                            if hasattr(module, "r"):
                                module.r[current_task_id] = r
                            if hasattr(module, "lora_alpha"):
                                module.lora_alpha[current_task_id] = lora_alpha
                                
            # Update adapter names for validation
            new_adapters = set()
            for name, module in self.model.named_modules():
                if hasattr(module, "lora_A") and hasattr(module.lora_A, "keys"):
                    new_adapters.update(module.lora_A.keys())
                    
            logger.info(f"Adapters after setup: {new_adapters}")
            debugprint(f"setup_adapters: 设置后的 adapters: {new_adapters}")
            
            final_check = current_task_id in new_adapters
            debugprint(f"setup_adapters: 最终检查 '{current_task_id}' 是否在 adapters 中: {final_check}。返回 {final_check}。")
            return current_task_id in new_adapters
            
        except Exception as e:
            logger.error(f"Error setting up adapters: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            debugprint(f"setup_adapters: 设置 adapters 时发生错误: {str(e)}。返回 False。")
            return False
