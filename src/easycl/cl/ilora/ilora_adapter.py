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
from typing import TYPE_CHECKING, Optional

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled

from llamafactory.extras import logging
from llamafactory.model.adapter import init_adapter
from llamafactory.model.model_utils.misc import find_all_linear_modules, find_expanded_modules
from llamafactory.model.model_utils.quantization import QuantizationMethod
from llamafactory.model.model_utils.unsloth import get_unsloth_peft_model, load_unsloth_peft_model
from llamafactory.model.model_utils.visual import patch_target_modules
from easycl.cl.ilora.ilora import ILORA


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from llamafactory.hparams import FinetuningArguments, ModelArguments
    from easycl.hparams.cl_finetuning_args import CLFinetuningArguments


logger = logging.get_logger(__name__)


def _setup_ilora_tuning(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    cl_finetuning_args: "CLFinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
    ilora_instance: Optional[ILORA] = None,
) -> "PeftModel":
    """
    Setup I-LORA tuning by initializing both default and EMA adapters.

    This function preserves all the functionality of the original _setup_lora_tuning function
    while adding support for the EMA adapter needed by I-LORA.
    """
    if is_trainable:
        logger.info_rank0("Fine-tuning method: I-LORA")

    adapter_to_resume = None
    adapter_to_merge = []
    
    # 处理前一个任务的EMA适配器加载
    is_first_task = finetuning_args.prev_task_id is None
    if not is_first_task and cl_finetuning_args.use_ilora and adapter_to_resume is None:
        # 尝试从前一个任务中加载EMA适配器
        try:
            # 构建前一个任务的EMA适配器路径模式
            prev_ema_paths = []
            
            # 按照要求修改路径加载顺序：
            # 1. 直接尝试ema_adapter_path
            if cl_finetuning_args.ema_adapter_path:
                prev_ema_paths.append(cl_finetuning_args.ema_adapter_path)
                
                # 2. 尝试从ema_adapter_path/ema和ema_adapter_path/ema_adapter读取
                prev_ema_paths.append(os.path.join(cl_finetuning_args.ema_adapter_path, "ema"))
                prev_ema_paths.append(os.path.join(cl_finetuning_args.ema_adapter_path, "ema_adapter"))
            
            # 3. 尝试从previous_task_model/ema或previous_task_model/ema_adapter读取
            if cl_finetuning_args.previous_task_model:
                prev_ema_paths.append(os.path.join(cl_finetuning_args.previous_task_model, "ema"))
                prev_ema_paths.append(os.path.join(cl_finetuning_args.previous_task_model, "ema_adapter"))
            
            # 尝试所有可能的路径
            loaded = False
            for ema_path in prev_ema_paths:
                if os.path.exists(ema_path):
                    logger.info_rank0(f"Attempting to load EMA adapter from {ema_path}...")
                    try:
                        # 可能的adapter子文件夹
                        adapter_dirs = ["", "adapter", "ema"]
                        
                        for adapter_dir in adapter_dirs:
                            try_path = os.path.join(ema_path, adapter_dir) if adapter_dir else ema_path
                            adapter_file = os.path.join(try_path, "adapter_model.safetensors")
                            adapter_file_bin = os.path.join(try_path, "adapter_model.bin")
                            
                            if os.path.exists(adapter_file) or os.path.exists(adapter_file_bin):
                                logger.info_rank0(f"Found adapter file: {adapter_file if os.path.exists(adapter_file) else adapter_file_bin}")
                                
                                init_kwargs = {
                                    "subfolder": "",
                                    "offload_folder": model_args.offload_folder,
                                    "cache_dir": model_args.cache_dir,
                                    "revision": model_args.model_revision,
                                    "token": model_args.hf_hub_token,
                                }
                                
                                try:
                                    model = PeftModel.from_pretrained(model, try_path, is_trainable=is_trainable, **init_kwargs)
                                    logger.info_rank0(f"Successfully loaded EMA adapter from {try_path}")
                                    loaded = True
                                    break
                                except Exception as e:
                                    logger.warning_rank0(f"Failed to load adapter from {try_path}: {e}")
                                    continue
                        
                        if loaded:
                            break
                    except Exception as load_err:
                        logger.warning_rank0(f"Failed to load from {ema_path}: {load_err}")
            
            if not loaded:
                error_msg = f"Failed to load EMA adapter. Tried paths: {prev_ema_paths}"
                if not is_first_task:
                    logger.warning_rank0(error_msg)  # 修改为warning_rank0，因为没有error_rank0方法
                    raise ValueError(error_msg)
                else:
                    logger.warning_rank0(f"{error_msg}. Continuing as first task.")
                
        except Exception as e:
            if not is_first_task:
                logger.warning_rank0(f"Failed to load EMA adapter: {e}")  # 修改为warning_rank0
                raise RuntimeError(f"Non-first task (prev_task_id = {cl_finetuning_args.prev_task_id}) failed to load EMA adapter. Ensure correct path specified or use --prev_task_id=None for first task.")
            else:
                logger.warning_rank0(f"Failed to load EMA adapter: {e}")
                logger.warning_rank0(f"Continuing as first task...")
    
    # Handle adapter loading similar to original _setup_lora_tuning
    if model_args.adapter_name_or_path is not None:
        is_mergeable = True
        if getattr(model, "quantization_method", None):  # Merge in quantized model is unstable
            assert len(model_args.adapter_name_or_path) == 1, "Quantized model only accepts a single adapter."
            is_mergeable = False

        if is_deepspeed_zero3_enabled():
            assert len(model_args.adapter_name_or_path) == 1, "Cannot use multiple adapters in DeepSpeed ZeRO-3."
            is_mergeable = False

        if model_args.use_unsloth:
            assert len(model_args.adapter_name_or_path) == 1, "Unsloth model only accepts a single adapter."
            is_mergeable = False

        if (is_trainable and not cl_finetuning_args.create_new_adapter) or (not is_mergeable):
            adapter_to_merge = model_args.adapter_name_or_path[:-1]
            adapter_to_resume = model_args.adapter_name_or_path[-1]
        else:
            adapter_to_merge = model_args.adapter_name_or_path

        init_kwargs = {
            "subfolder": model_args.adapter_folder,
            "offload_folder": model_args.offload_folder,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": model_args.hf_hub_token,
        }

        # Merge adapters as in original implementation
        for adapter in adapter_to_merge:
            model = PeftModel.from_pretrained(model, adapter, **init_kwargs)
            model = model.merge_and_unload()

        if len(adapter_to_merge) > 0:
            logger.info_rank0(f"Merged {len(adapter_to_merge)} adapter(s).")

        # For ILORA: Store adapter history information if needed
        if ilora_instance is not None:
            ilora_instance.merged_adapters = adapter_to_merge

        # Resume adapter training
        if adapter_to_resume is not None:
            if model_args.use_unsloth:
                model = load_unsloth_peft_model(config, model_args, is_trainable=is_trainable)
            else:
                model = PeftModel.from_pretrained(model, adapter_to_resume, is_trainable=is_trainable, **init_kwargs)

        logger.info_rank0("Loaded adapter(s): {}".format(",".join(model_args.adapter_name_or_path)))

    # Create new LoRA weights while training
    if is_trainable and adapter_to_resume is None:
        # Determine target modules as in original implementation
        if len(cl_finetuning_args.lora_target) == 1 and cl_finetuning_args.lora_target[0] == "all":
            target_modules = find_all_linear_modules(model, cl_finetuning_args.freeze_vision_tower)
        else:
            target_modules = cl_finetuning_args.lora_target

        if cl_finetuning_args.use_llama_pro:
            target_modules = find_expanded_modules(model, target_modules, cl_finetuning_args.freeze_trainable_layers)

        target_modules = patch_target_modules(model, cl_finetuning_args, target_modules)

        if (
            cl_finetuning_args.use_dora
            and getattr(model, "quantization_method", None) is not None
            and getattr(model, "quantization_method", None) != QuantizationMethod.BITS_AND_BYTES
        ):
            raise ValueError("DoRA is not compatible with PTQ-quantized models.")

        # Handle vocabulary resizing as in original implementation
        if model_args.resize_vocab and cl_finetuning_args.additional_target is None:
            input_embeddings = model.get_input_embeddings()
            output_embeddings = model.get_output_embeddings()
            module_names = set()
            for name, module in model.named_modules():
                if module in [input_embeddings, output_embeddings]:
                    module_names.add(name.split(".")[-1])

            cl_finetuning_args.additional_target = module_names
            logger.warning_rank0("Vocab has been resized, add {} to trainable params.".format(",".join(module_names)))

        # Set up PEFT kwargs as in original implementation
        peft_kwargs = {
            "r": cl_finetuning_args.lora_rank,
            "target_modules": target_modules,
            "lora_alpha": cl_finetuning_args.lora_alpha,
            "lora_dropout": cl_finetuning_args.lora_dropout,
            "use_rslora": cl_finetuning_args.use_rslora,
            "use_dora": cl_finetuning_args.use_dora,
            "modules_to_save": cl_finetuning_args.additional_target,
        }

        # Modify peft_kwargs for I-LORA if needed
        if ilora_instance is not None:
            # No specific modifications needed for I-LORA peft_kwargs
            pass

        if model_args.use_unsloth:
            model = get_unsloth_peft_model(model, model_args, peft_kwargs)
        else:
            # Handle PiSSA initialization as in original implementation
            if cl_finetuning_args.pissa_init:
                if cl_finetuning_args.pissa_iter == -1:
                    logger.info_rank0("Using PiSSA initialization.")
                    peft_kwargs["init_lora_weights"] = "pissa"
                else:
                    logger.info_rank0(f"Using PiSSA initialization with FSVD steps {cl_finetuning_args.pissa_iter}.")
                    peft_kwargs["init_lora_weights"] = f"pissa_niter_{cl_finetuning_args.pissa_iter}"

            # Create default adapter
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                **peft_kwargs,
            )
            model = get_peft_model(model, lora_config)
            
            # I-LORA: Add EMA adapter (copy of default adapter with inference mode=True)
            if is_trainable:
                ema_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=True,  # EMA adapter is never directly trained
                    **peft_kwargs,
                )
                
                logger.info_rank0("Adding EMA adapter for I-LORA.")
                model.add_adapter("ema", ema_config)
                
                # Check if default adapter exists
                has_default_adapter = hasattr(model, "peft_config") and "default" in model.peft_config
                
                if has_default_adapter:
                    logger.info_rank0("Default adapter exists, attempting to copy weights to EMA adapter.")
                    try:
                        # Get and copy default adapter weights to ema adapter
                        default_state = model.get_adapter_state_dict(adapter_name="default")
                        current_adapter = model.active_adapter
                        model.set_adapter("ema")
                        
                        for name, param in model.named_parameters():
                            if "lora" in name and "ema" in name and not param.requires_grad:
                                adapter_param_name = name.split(".")[-1]
                                module_name = ".".join(name.split(".")[:-1])
                                try:
                                    layer_name = module_name.split("ema.")[1]
                                    full_param_name = f"base_model.model.{layer_name}.{adapter_param_name}"
                                    if full_param_name in default_state:
                                        param.data = default_state[full_param_name].to(param.device)
                                except (IndexError, KeyError) as e:
                                    logger.warning_rank0(f"Failed to copy parameter {name}: {e}")
                        
                        # Restore original adapter
                        model.set_adapter(current_adapter)
                        logger.info_rank0("Successfully copied default adapter weights to EMA adapter.")
                        
                    except Exception as e:
                        logger.warning_rank0(f"Error copying weights to EMA adapter: {e}. Will use default initialization.")
                else:
                    # This is a new task start, or adapter loading failed
                    if is_first_task:
                        logger.info_rank0("First task, EMA adapter will use random initialization.")
                    else:
                        logger.warning_rank0(f"Although this is a subsequent task (prev_task_id={cl_finetuning_args.prev_task_id}), no valid default adapter found. Attempting to continue...")

    # Cast trainable parameters to fp32 if needed (same as original implementation)
    if is_trainable and cast_trainable_params_to_fp32:
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)

    return model


def init_ilora_adapter(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    cl_finetuning_args: "CLFinetuningArguments",
    is_trainable: bool,
) -> "PreTrainedModel":
    """
    Initialize I-LORA adapter for the model.
    
    Args:
        config: Model configuration.
        model: The model to add adapter to.
        model_args: Model arguments.
        finetuning_args: Fine-tuning arguments.
        cl_finetuning_args: Continual Learning fine-tuning arguments.
        is_trainable: Whether the model is being trained.
        
    Returns:
        Model with I-LORA adapter(s).
    """
    # Cast model weights to fp32 if needed
    cast_trainable_params_to_fp32 = (
        is_trainable and getattr(model, "quantization_method", None) and model_args.upcast_layernorm
    )
    
    # Only create I-LORA instance if we're trainable and using I-LORA
    if is_trainable and cl_finetuning_args.use_ilora:
        ilora_instance = ILORA(
            model=model,  # This will be replaced after PEFT model creation
            cl_finetuning_args=cl_finetuning_args,
            previous_task_model=cl_finetuning_args.previous_task_model,
            current_task_id=cl_finetuning_args.current_task_id,
        )
    
    # Use our custom setup function for I-LORA
    model = _setup_ilora_tuning(
        config=config,
        model=model,
        model_args=model_args,
        finetuning_args=finetuning_args,
        cl_finetuning_args=cl_finetuning_args,
        is_trainable=is_trainable,
        cast_trainable_params_to_fp32=cast_trainable_params_to_fp32,
        ilora_instance=ilora_instance,
    )
    
    # Attach I-LORA instance to the model for later use in the trainer
    if is_trainable and cl_finetuning_args.use_ilora:
        ilora_instance.model = model  # Update with the PEFT model
        model.ilora = ilora_instance
    
    return model