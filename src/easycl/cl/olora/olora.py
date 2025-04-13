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
                    
                self.adapter_history.append(AdapterInfo(
                    task_id=adapter_id,
                    path=adapter_path,
                    config=config
                ))
                
    def _validate_adapter_path(self, adapter_path: str) -> str:
        """Validate and normalize adapter path"""
        adapter_path = os.path.abspath(adapter_path)
        
        if not os.path.exists(adapter_path):
            raise ValueError(f"Adapter path does not exist: {adapter_path}")
            
        config_path = os.path.join(adapter_path, "adapter_config.json")
        model_path = os.path.join(adapter_path, "adapter_model.safetensors")
        
        if not os.path.exists(config_path):
            raise ValueError(f"adapter_config.json not found in {adapter_path}")
            
        if not os.path.exists(model_path):
            raise ValueError(f"adapter_model.safetensors not found in {adapter_path}")
            
        return adapter_path

    def load_adapter_weights(self, task_id: str) -> Dict[str, torch.Tensor]:
        """Load LoRA weights from adapter file"""
        adapter_path = os.path.join(self.model_output_dir)
        
        try:
            adapter_path = self._validate_adapter_path(adapter_path)
        except ValueError as e:
            logger.error(f"Error validating adapter path: {str(e)}")
            return {}
            
        try:
            # Read adapter configuration
            config_path = os.path.join(adapter_path, "adapter_config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Read adapter weights
            model_path = os.path.join(adapter_path, "adapter_model.safetensors")
            state_dict = safe_load_file(model_path, device="cpu")
            
            # Extract LoRA weights
            lora_weights = {}
            # Record weight names for debugging
            weight_keys = list(state_dict.keys())
            logger.debug(f"Adapter weights in {task_id}: {weight_keys[:5]}...")
            logger.debug(f"Total weights: {len(weight_keys)}")
            
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
            else:
                logger.info(f"Loaded {len(lora_weights)} LoRA weights from adapter: {task_id}")
                
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
                
            else:
                # Subsequent tasks: load historical weights and concatenate
                # Add check to prevent IndexError if history is empty for a subsequent task
                
                prev_merged_dir = os.path.join(self.olora_history_path, self.prev_task_id, "merged")
                prev_path = os.path.join(prev_merged_dir, "merged_adapter.pt")
                
                if not os.path.exists(prev_path):
                    raise ValueError(f"Previous merged adapter weights not found: {prev_path}")
                    
                prev_weights = torch.load(prev_path, map_location="cpu")
                
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
                
                # Record dimension information for debugging
                for key, value in merged_weights.items():
                    if "merged_A" in key or "merged_B" in key:
                        logger.debug(f"Merged weight {key} shape: {value.shape}")
                
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
        # Check if prev_task_id exists, if not or None, return zero loss
        if not hasattr(self, "prev_task_id") or self.prev_task_id is None:
            #logger.info("No previous task ID found, skipping orthogonal loss calculation")
            return torch.tensor(0.0, device=self.device)
            
        orth_loss = torch.tensor(0.0, device=self.device)
        num_matrices = 0
        
        # In PEFT library, LoRA parameters are typically named as:
        # base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight
        
        # Collect all parameters in the model
        all_params = dict(self.model.named_parameters())
        
        # Find adapter names list for LoRA modules
        adapter_names = set()
        
        # Debug: Print all adapter names in the model
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                # Check which adapters this module has
                if hasattr(module.lora_A, "keys"):
                    adapter_names.update(module.lora_A.keys())
        
        # If less than two adapters found, cannot compute orthogonal loss
        if len(adapter_names) < 2:
            logger.warning(f"Need at least 2 adapters to calculate orthogonal loss, found {len(adapter_names)}")
            return orth_loss
            
        # Determine which is the previous adapter and which is the current adapter
        # Typically, adapter names might be 'default' and task names (like dbpedia etc.)
        # Or they might be 'default' and '0' etc.
        primary_adapter = None
        secondary_adapter = None
        
        # Assume 'default' is the previous adapter by priority
        if 'default' in adapter_names:
            primary_adapter = 'default'
            # Find other adapters that are not 'default'
            secondary_candidates = [name for name in adapter_names if name != 'default']
            if secondary_candidates:
                secondary_adapter = secondary_candidates[0]
        # If adapter_names has only one element but it's not 'default'
        elif len(adapter_names) == 1:
            logger.warning(f"Only one adapter found: {list(adapter_names)[0]}, cannot compute orthogonal loss")
            return orth_loss
        
        if primary_adapter is None or secondary_adapter is None:
            logger.warning("Could not determine primary and secondary adapters")
            logger.warning(f"Available adapters: {adapter_names}")
            return orth_loss
        
        # Now we have two adapter names, can calculate orthogonal loss
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and primary_adapter in module.lora_A and secondary_adapter in module.lora_A:
                old_weight = module.lora_A[primary_adapter].weight
                new_weight = module.lora_A[secondary_adapter].weight
                
                # Calculate orthogonal loss: |A·B^T|
                try:
                    dot_product = torch.mm(new_weight, old_weight.T)
                    curr_loss = torch.abs(dot_product).sum()
                    orth_loss += curr_loss
                    num_matrices += 1
                    
                    logger.debug(f"Module {name} orthogonal loss: {curr_loss.item():.4f}")
                except Exception as e:
                    logger.warning(f"Error calculating orthogonal loss for {name}: {str(e)}")
                    logger.warning(f"Shapes: new={new_weight.shape}, old={old_weight.shape}")
        
        if num_matrices > 0:
            orth_loss = orth_loss
        else:
            logger.warning(f"No matrix pairs found for orthogonal loss between {primary_adapter} and {secondary_adapter}")
            
        return self.orthogonal_lambda * orth_loss

    def compute_l2_loss(self) -> torch.Tensor:
        """Calculate L2 regularization loss for new LoRA parameters"""
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
            return l2_loss
            
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
                else:
                    logger.debug(f"Module {name} L2 norm: A={a_norm.item():.4f} (B not found or missing adapter)")

        if num_params > 0:
            l2_loss = l2_loss
        else:
            logger.warning(f"No parameters found for L2 loss with adapter '{current_adapter}'")
            
        return self.l2_lambda * l2_loss

    def load_prev_adapter(self, prev_task_id: str) -> bool:
        """Load previous task's adapter parameters.
        Prioritize loading merged/merged_adapter.pt, if not exists try loading adapter_model.safetensors.
        Return False if this is the first task.
        """
        # If no prev_task_id provided, this is the first task
        if prev_task_id is None:
            logger.info(f"No previous task ID provided. This seems to be the first task.")
            self.prev_task_id = None
            return False

        # Set prev_task_id attribute
        self.prev_task_id = prev_task_id

        # Try loading merged weights
        merged_dir = os.path.join(self.olora_history_path, prev_task_id, "merged")
        merged_load_path = os.path.join(merged_dir, "merged_adapter.pt")

        state_dict = None
        loaded_from = None

        if os.path.exists(merged_load_path):
            try:
                state_dict = torch.load(merged_load_path, map_location=self.device)
                loaded_from = "merged_adapter.pt"
                logger.info(f"Found previous adapter weights at {merged_load_path}")
            except Exception as e:
                logger.warning(f"Error loading {merged_load_path}: {str(e)}. Trying safetensors.")
                state_dict = None # Reset state_dict if loading failed
        else:
            logger.info(f"Previous merged adapter not found at {merged_load_path}. "
                          f"Attempting to load from adapter_model.safetensors.")
            # Try loading safetensors file directly
            try:
                # Use existing load_adapter_weights to load and convert format
                # Note: load_adapter_weights returns keys already in 'module.path.merged_A/B' format
                state_dict = self.load_adapter_weights(prev_task_id)
                if state_dict:
                    loaded_from = "adapter_model.safetensors"
                    logger.info(f"Successfully loaded weights from adapter_model.safetensors for task {prev_task_id}")
                else:
                    logger.warning(f"Failed to load weights from adapter_model.safetensors for task {prev_task_id}.")
                    state_dict = None # Ensure state_dict is None if loading failed
            except Exception as e:
                logger.error(f"Error loading adapter_model.safetensors for task {prev_task_id}: {str(e)}")
                logger.error(f"Stack trace:", exc_info=True)
                state_dict = None # Ensure state_dict is None on error

        # If both methods failed
        if state_dict is None:
            logger.warning(f"Could not load previous adapter weights for task {prev_task_id} from either .pt or .safetensors.")
            return False

        try:
            # Record which weights were loaded
            logger.info(f"Loaded {len(state_dict)} weights from {loaded_from} for task {prev_task_id}")
            logger.debug(f"Weight keys pattern: {list(state_dict.keys())[:3]}")

            # Check each module with lora_A
            loaded_count = 0
            missing_keys_count = 0
            for name, module in self.model.named_modules():
                # Check for lora_A attribute, which typically indicates a PEFT LoRA module
                if hasattr(module, "lora_A"):
                    # Construct expected merged weight key names (load_adapter_weights already handles this format)
                    merged_a_key = f"{name}.merged_A"
                    merged_b_key = f"{name}.merged_B"

                    if merged_a_key in state_dict and merged_b_key in state_dict:
                        # Load merged weights into default adapter
                        if "default" in module.lora_A:
                            try:
                                # Check for size mismatch before copying
                                expected_a_shape = module.lora_A["default"].weight.shape
                                got_a_shape = state_dict[merged_a_key].shape
                                expected_b_shape = module.lora_B["default"].weight.shape
                                got_b_shape = state_dict[merged_b_key].shape

                                if expected_a_shape == got_a_shape and expected_b_shape == got_b_shape:
                                    module.lora_A["default"].weight.data.copy_(state_dict[merged_a_key])
                                    module.lora_B["default"].weight.data.copy_(state_dict[merged_b_key])
                                    loaded_count += 1
                                else:
                                    logger.warning(f"Size mismatch loading {name} for 'default' adapter. Skipping.")
                                    logger.warning(f"  Expected A: {expected_a_shape}, Got A: {got_a_shape}")
                                    logger.warning(f"  Expected B: {expected_b_shape}, Got B: {got_b_shape}")

                            except RuntimeError as e:
                                 logger.warning(f"Runtime error loading {name} for 'default' adapter: {e}. Skipping.")
                                 logger.warning(f"  Expected A: {module.lora_A['default'].weight.shape}, Got A: {state_dict.get(merged_a_key, 'N/A')}")
                                 logger.warning(f"  Expected B: {module.lora_B['default'].weight.shape}, Got B: {state_dict.get(merged_b_key, 'N/A')}")

                        else:
                            logger.debug(f"Module {name} does not have a 'default' adapter to load into.")

                    else:
                        # Record missing keys, only log in detail in debug mode
                        if missing_keys_count < 5: # Limit logging spam
                            logger.debug(f"Keys '{merged_a_key}' or '{merged_b_key}' not found in loaded state_dict for module {name}.")
                        missing_keys_count += 1

            if missing_keys_count > 0:
                 logger.warning(f"Found {missing_keys_count} modules where expected merged keys were not in the loaded state_dict.")

            if loaded_count > 0:
                logger.info(f"Successfully applied weights to {loaded_count} LoRA modules in the 'default' adapter.")
                return True
            else:
                logger.warning(f"Loaded weights from {loaded_from}, but did not apply them to any LoRA modules. Check model structure and weight keys.")
                return False

        except Exception as e:
            logger.error(f"Error applying previous adapter weights: {str(e)}")
            logger.error(f"Stack trace:", exc_info=True)
            return False

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
            # Check if model has LoRA attributes
            has_lora = False
            for name, module in self.model.named_modules():
                if hasattr(module, "lora_A"):
                    has_lora = True
                    break
                    
            if not has_lora:
                logger.warning("Model does not have LoRA modules, cannot set up adapters")
                return False
                
            # Get existing adapter names
            existing_adapters = set()
            for name, module in self.model.named_modules():
                if hasattr(module, "lora_A") and hasattr(module.lora_A, "keys"):
                    existing_adapters.update(module.lora_A.keys())
                    
            logger.info(f"Existing adapters before setup: {existing_adapters}")
            
            # If already have two adapters, check if they include default and current
            if len(existing_adapters) >= 2 and 'default' in existing_adapters and current_task_id in existing_adapters:
                logger.info(f"Both 'default' and '{current_task_id}' adapters already exist")
                return True
                
            # If only have default adapter, create second adapter
            if 'default' in existing_adapters and current_task_id not in existing_adapters:
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
            
            return current_task_id in new_adapters
            
        except Exception as e:
            logger.error(f"Error setting up adapters: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            return False
