import os
from typing import TYPE_CHECKING, Optional
from contextlib import nullcontext

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from peft.utils import PeftType # Import PeftType for checking if model is PeftModel
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled

from llamafactory.extras import logging
from llamafactory.model.adapter import init_adapter
from llamafactory.model.model_utils.misc import find_all_linear_modules, find_expanded_modules
from llamafactory.model.model_utils.quantization import QuantizationMethod
from llamafactory.model.model_utils.unsloth import get_unsloth_peft_model, load_unsloth_peft_model
from llamafactory.model.model_utils.visual import patch_target_modules
from easycl.cl.ilora.ilora import ILORA
from easycl.cl.distributed_utils import (
    is_distributed, get_rank, is_main_process, get_world_size,
    get_deepspeed_zero_stage, gather_parameters, all_reduce_tensor, broadcast_object
)

def debugprint(*args, **kwargs):
    pass

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from llamafactory.hparams import FinetuningArguments, ModelArguments
    from easycl.hparams.finetuning_args import CLFinetuningArguments


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
    Refactored to load main adapter first, then EMA adapter.
    """
    debugprint("进入 _setup_ilora_tuning 函数 (修改后)")
    debugprint(f"  is_trainable: {is_trainable}")
    debugprint(f"  ilora_instance provided: {ilora_instance is not None}")
    debugprint(f"  finetuning_args: {finetuning_args}")
    debugprint(f"  cl_finetuning_args: {cl_finetuning_args}")
    debugprint(f"  model_args.adapter_name_or_path: {model_args.adapter_name_or_path}")

    if is_trainable:
        logger.info_rank0("Fine-tuning method: I-LORA")

    adapter_to_resume = None
    adapter_to_merge = []
    successfully_resumed_adapter = False # Flag to track if main adapter was resumed
    is_peft_model_at_start = isinstance(model, PeftModel)
    debugprint(f"  模型在开始时是否为 PeftModel: {is_peft_model_at_start}")


    # --- 1. 首先处理 model_args.adapter_name_or_path (加载/合并/恢复主适配器) ---
    if model_args.adapter_name_or_path is not None:
        debugprint(f"  处理 model_args.adapter_name_or_path: {model_args.adapter_name_or_path}")
        is_mergeable = True
        if getattr(model, "quantization_method", None):  # Merge in quantized model is unstable
            assert len(model_args.adapter_name_or_path) == 1, "Quantized model only accepts a single adapter."
            is_mergeable = False

        if is_deepspeed_zero3_enabled():
            assert len(model_args.adapter_name_or_path) == 1, "Cannot use multiple adapters in DeepSpeed ZeRO-3."
            is_mergeable = False

        if model_args.use_unsloth:
            assert len(model_args.adapter_name_or_path) == 1, "Unsloth model only accepts a single adapter."
            # For unsloth, merging might be implicitly handled or not supported, assume not mergeable here?
            # Let's keep is_mergeable=False for unsloth to be safe, aligns with resume logic below.
            is_mergeable = False

        debugprint(f"  is_mergeable: {is_mergeable}")

        # Determine which adapters to merge and which one to resume
        if (is_trainable and not finetuning_args.create_new_adapter) or (not is_mergeable):
            adapter_to_merge = model_args.adapter_name_or_path[:-1]
            adapter_to_resume = model_args.adapter_name_or_path[-1]
            debugprint(f"  将要合并的适配器: {adapter_to_merge}, 将要恢复训练的主适配器: {adapter_to_resume}")
        else:
            adapter_to_merge = model_args.adapter_name_or_path
            debugprint(f"  将要合并的适配器: {adapter_to_merge}, 无需恢复训练的适配器")

        # Define common kwargs for from_pretrained
        init_kwargs = {
            "subfolder": model_args.adapter_folder,
            "offload_folder": model_args.offload_folder,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": model_args.hf_hub_token,
        }

        # Merge adapters if any
        for adapter in adapter_to_merge:
            debugprint(f"  合并适配器: {adapter}")
            # Need to load the model as PeftModel first if it's not already
            if not isinstance(model, PeftModel):
                 logger.info_rank0(f"Model is not a PeftModel yet. Loading {adapter} to convert.")
                 model = PeftModel.from_pretrained(model, adapter, **init_kwargs)
            else:
                 # If already PeftModel, load and merge
                 model.load_adapter(adapter, adapter_name=f"merge_{adapter.split('/')[-1]}", **init_kwargs) # Give temp name
            model = model.merge_and_unload(adapter_names=[f"merge_{adapter.split('/')[-1]}"]) # Merge specific adapter
            debugprint(f"  合并完成: {adapter}")


        if len(adapter_to_merge) > 0:
            logger.info_rank0(f"Merged {len(adapter_to_merge)} adapter(s).")
            if ilora_instance is not None:
                 ilora_instance.merged_adapters = adapter_to_merge # Store merge history if needed
                 debugprint(f"  存储合并的适配器历史到 ilora_instance: {adapter_to_merge}")


        # Resume adapter training (this loads the main adapter, often named 'default')
        if adapter_to_resume is not None:
            debugprint(f"  恢复主适配器训练: {adapter_to_resume}")
            try:
                if model_args.use_unsloth:
                    # Unsloth handles PeftModel creation internally when loading adapters
                    model = load_unsloth_peft_model(config, model_args, is_trainable=is_trainable, adapter_name_or_path=adapter_to_resume)
                    debugprint("    使用 Unsloth 恢复适配器")
                else:
                    # If the base model was already loaded and potentially merged,
                    # we need to ensure it's a PeftModel before loading/resuming the target adapter.
                    if not isinstance(model, PeftModel):
                         debugprint(f"    模型在恢复前不是 PeftModel，使用 PeftModel.from_pretrained 恢复: {adapter_to_resume}, is_trainable={is_trainable}")
                         model = PeftModel.from_pretrained(model, adapter_to_resume, is_trainable=is_trainable, **init_kwargs)
                    else:
                         # If it's already a PeftModel (e.g., after merging), load the adapter to resume
                         debugprint(f"    模型已经是 PeftModel，使用 model.load_adapter 恢复: {adapter_to_resume}, is_trainable={is_trainable}")
                         # Ensure the adapter name is 'default' if it's the main one being resumed for training
                         # However, PeftModel.from_pretrained implicitly loads as 'default' if no name is specified.
                         # Let's load it with a temporary name and then set 'default' active if needed?
                         # Or rely on the fact that load_adapter might overwrite 'default' if it exists?
                         # Let's assume from_pretrained set the adapter name correctly, or load_adapter is used carefully.
                         # For simplicity, if resuming, we assume this becomes the 'default' active adapter.
                         # We might need to explicitly rename or set active later.
                         # Let's try loading directly. If `adapter_name` is specified in `adapter_config.json`, it will use that.
                         # If not, it defaults to 'default'.
                         model.load_adapter(adapter_to_resume, adapter_name="default", is_trainable=is_trainable, **init_kwargs)
                         debugprint(f"    尝试加载适配器 {adapter_to_resume} 为 'default'")

                successfully_resumed_adapter = True
                logger.info_rank0(f"Successfully resumed main adapter from {adapter_to_resume}")
                debugprint(f"  成功恢复主适配器: {adapter_to_resume}")
                # Check adapters after resuming
                if hasattr(model, 'peft_config'):
                     debugprint(f"    恢复后可用适配器: {list(model.peft_config.keys())}")
                # Ensure the resumed adapter is set as active if it wasn't automatically
                if hasattr(model, 'active_adapter') and model.active_adapter != "default":
                     if "default" in model.peft_config:
                          debugprint("    将恢复的适配器设置为活动适配器 ('default')")
                          model.set_adapter("default")

            except Exception as e:
                 logger.warning_rank0(f"Failed to resume main adapter from {adapter_to_resume}: {e}", exc_info=True)
                 debugprint(f"  恢复主适配器失败: {adapter_to_resume}, 错误: {e}")
                 # Depending on policy, we might want to raise here or continue cautiously
                 # raise e # Or just log and potentially create new adapters later

        if adapter_to_resume or adapter_to_merge:
             logger.info_rank0("Loaded/merged adapter(s): {}".format(",".join(model_args.adapter_name_or_path)))
    elif model_args.use_unsloth and not is_peft_model_at_start:
         # If using unsloth and no adapter path provided, the model might already be loaded by load_ilora_model
         # Check if it's already a PeftModel (Unsloth might have done this)
         if isinstance(model, PeftModel):
              debugprint("  Unsloth 模型已是 PeftModel (可能由 load_ilora_model 处理)，无需额外操作")
              is_peft_model_at_start = True # Update flag
         else:
              debugprint("  Unsloth 模型但非 PeftModel 且无 adapter_path，稍后可能创建新适配器")


    # --- 2. 如果是非首次任务且使用 I-LORA，加载 EMA 适配器 ---
    is_first_task = cl_finetuning_args.prev_task_id is None
    ema_loaded = False
    if not is_first_task and cl_finetuning_args.use_ilora:
        debugprint("非首任务且使用 I-LORA，尝试加载前一个任务的 EMA 适配器")

        # Ensure the model is a PeftModel before trying to load the EMA adapter
        if not isinstance(model, PeftModel):
             # This might happen if only merging occurred and the base model wasn't converted.
             # Or if no adapter_name_or_path was provided.
             # We need a base PeftModel structure to load into. Let's create a dummy one?
             # Or should we raise an error? Let's raise for now, as loading EMA requires a PeftModel.
             logger.error("Cannot load EMA adapter because the model is not a PeftModel instance at this point.")
             raise TypeError("Model must be a PeftModel before loading the EMA adapter for subsequent tasks.")

        # Build EMA path list
        prev_ema_paths = []
        debugprint(f"  检查 cl_finetuning_args.ema_adapter_path: {cl_finetuning_args.ema_adapter_path}")
        if cl_finetuning_args.ema_adapter_path:
            prev_ema_paths.append(cl_finetuning_args.ema_adapter_path)
            # Also check common subfolders within the specified path
            # prev_ema_paths.append(os.path.join(cl_finetuning_args.ema_adapter_path, "ema")) # Let's rely on probing below
            # prev_ema_paths.append(os.path.join(cl_finetuning_args.ema_adapter_path, "ema_adapter"))

        if cl_finetuning_args.previous_task_model:
            debugprint(f"  检查 cl_finetuning_args.previous_task_model: {cl_finetuning_args.previous_task_model}")
            # Check for ema/ema_adapter subfolders within the previous task model directory
            prev_ema_paths.append(os.path.join(cl_finetuning_args.previous_task_model, "ema"))
            prev_ema_paths.append(os.path.join(cl_finetuning_args.previous_task_model, "ema_adapter"))
            prev_ema_paths.append(cl_finetuning_args.previous_task_model) # Also check the root dir

        # Remove duplicates and non-existent paths before trying
        valid_paths = []
        for p in prev_ema_paths:
             if p and p not in valid_paths and os.path.exists(p):
                  valid_paths.append(p)
             elif p and p not in valid_paths:
                  debugprint(f"  路径不存在，跳过: {p}")

        debugprint(f"  尝试加载 EMA 适配器的有效存在路径: {valid_paths}")

        for ema_base_path in valid_paths:
            debugprint(f"  尝试基础路径: {ema_base_path}")
            try:
                # Probe for adapter files within the base path and common subdirs
                adapter_dirs_to_probe = ["", "adapter", "ema", "ema_adapter"]
                found_adapter_file = False
                try_path_for_load = None

                for subdir in adapter_dirs_to_probe:
                     current_try_path = os.path.join(ema_base_path, subdir) if subdir else ema_base_path
                     # Check existence of path and necessary files
                     if not os.path.isdir(current_try_path):
                          # debugprint(f"    跳过非目录: {current_try_path}")
                          continue

                     adapter_config_file = os.path.join(current_try_path, "adapter_config.json")
                     has_safetensors = os.path.exists(os.path.join(current_try_path, "adapter_model.safetensors"))
                     has_bin = os.path.exists(os.path.join(current_try_path, "adapter_model.bin"))

                     debugprint(f"    检查子路径: {current_try_path} (config: {os.path.exists(adapter_config_file)}, safetensors: {has_safetensors}, bin: {has_bin})")

                     if os.path.exists(adapter_config_file) and (has_safetensors or has_bin):
                          logger.info_rank0(f"Found potential EMA adapter files in: {current_try_path}")
                          debugprint(f"    找到潜在 EMA 适配器文件于: {current_try_path}")
                          try_path_for_load = current_try_path
                          found_adapter_file = True
                          break # Found in this base path

                if found_adapter_file and try_path_for_load:
                    # Use common kwargs, but ensure is_trainable=False for EMA
                    init_kwargs_ema = {
                        "subfolder": "", # try_path_for_load is the full path
                        "offload_folder": model_args.offload_folder,
                        "cache_dir": model_args.cache_dir,
                        "revision": model_args.model_revision,
                        "token": model_args.hf_hub_token,
                    }
                    debugprint(f"    尝试使用 model.load_adapter 加载 EMA: {try_path_for_load}, adapter_name='ema', is_trainable=False")
                    # Load the found adapter specifically as 'ema' and non-trainable
                    model.load_adapter(try_path_for_load, adapter_name="ema", is_trainable=False, **init_kwargs_ema)

                    logger.info_rank0(f"Successfully loaded EMA adapter as 'ema' from {try_path_for_load}")
                    debugprint(f"    成功加载 EMA 适配器并命名为 'ema': {try_path_for_load}")
                    ema_loaded = True
                    # Check adapters after loading EMA
                    if hasattr(model, 'peft_config'):
                         loaded_adapters = list(model.peft_config.keys())
                         debugprint(f"    当前可用适配器: {loaded_adapters}")
                         logger.info_rank0(f"Adapters after loading EMA: {loaded_adapters}")
                    break # EMA loaded successfully, exit the loop over base paths

                elif not found_adapter_file:
                     debugprint(f"    在 {ema_base_path} 及其子目录中未找到有效的 adapter 文件")

            except Exception as load_err:
                logger.warning_rank0(f"Failed attempt to load EMA adapter from base path {ema_base_path}: {load_err}", exc_info=True)
                debugprint(f"  尝试从 {ema_base_path} 加载 EMA 时出错: {load_err}")
                # Continue to the next path in valid_paths

        if not ema_loaded:
            # If EMA is required for a subsequent task and couldn't be loaded, it's an error
            error_msg = (f"Failed to load EMA adapter for subsequent task (prev_task_id={cl_finetuning_args.prev_task_id}). "
                         f"Please ensure the EMA adapter was saved correctly in the previous task and the path is accessible. "
                         f"Checked paths derived from ema_adapter_path='{cl_finetuning_args.ema_adapter_path}' and "
                         f"previous_task_model='{cl_finetuning_args.previous_task_model}'. "
                         f"Attempted valid existing base paths: {valid_paths}")
            debugprint(f"  最终未能加载 EMA 适配器。")
            logger.error(error_msg)
            raise ValueError(error_msg)

    # --- 3. 如果需要创建新适配器 (Check conditions carefully) ---
    # Conditions for creating NEW adapters:
    # A) It's trainable AND...
    # B) EITHER:
    #    1) No main adapter was successfully resumed/loaded via adapter_name_or_path AND (it's the first task OR I-LORA is not used OR EMA wasn't loaded)
    #    2) OR finetuning_args.create_new_adapter is explicitly True
    needs_new_default_adapter = False
    if is_trainable:
         condition_B1 = (
             not successfully_resumed_adapter and
             (is_first_task or not cl_finetuning_args.use_ilora or not ema_loaded)
         )
         condition_B2 = finetuning_args.create_new_adapter

         if condition_B1 or condition_B2:
              needs_new_default_adapter = True
              logger.info_rank0("Conditions met to create a new default LoRA adapter.")
              debugprint("  需要创建新的 default LoRA 适配器")
         else:
              debugprint("  不需要创建新的 default LoRA 适配器 (已恢复或非训练模式或非必要)")


    if needs_new_default_adapter:
        # --- Logic for creating new 'default' adapter ---
        debugprint("  执行创建新 default 适配器的逻辑")
        # Determine target modules
        if len(finetuning_args.lora_target) == 1 and finetuning_args.lora_target[0] == "all":
            target_modules = find_all_linear_modules(model, finetuning_args.freeze_vision_tower)
        else:
            target_modules = finetuning_args.lora_target
        debugprint(f"  目标模块 (原始): {target_modules}")

        if finetuning_args.use_llama_pro:
            target_modules = find_expanded_modules(model, target_modules, finetuning_args.freeze_trainable_layers)
            debugprint(f"  目标模块 (llama_pro 扩展后): {target_modules}")

        target_modules = patch_target_modules(model, finetuning_args, target_modules)
        debugprint(f"  目标模块 (patch 后): {target_modules}")

        if (
            finetuning_args.use_dora
            and getattr(model, "quantization_method", None) is not None
            and getattr(model, "quantization_method", None) != QuantizationMethod.BITS_AND_BYTES
        ):
            raise ValueError("DoRA is not compatible with PTQ-quantized models.")

        # Handle vocabulary resizing
        if model_args.resize_vocab and finetuning_args.additional_target is None:
            input_embeddings = model.get_input_embeddings()
            output_embeddings = model.get_output_embeddings()
            module_names = set()
            for name, module in model.named_modules():
                if module in [input_embeddings, output_embeddings]:
                    module_names.add(name.split(".")[-1])
            if module_names:
                 finetuning_args.additional_target = list(module_names) # Ensure it's a list
                 logger.warning_rank0("Vocab has been resized, adding {} to trainable params.".format(",".join(module_names)))
                 debugprint(f"  词汇表已调整，添加 {module_names} 到可训练参数")

        # Set up PEFT kwargs
        peft_kwargs = {
            "r": finetuning_args.lora_rank,
            "target_modules": target_modules,
            "lora_alpha": finetuning_args.lora_alpha,
            "lora_dropout": finetuning_args.lora_dropout,
            "use_rslora": finetuning_args.use_rslora,
            "use_dora": finetuning_args.use_dora,
            "modules_to_save": finetuning_args.additional_target,
        }
        debugprint(f"  PEFT kwargs for new default: {peft_kwargs}")

        if model_args.use_unsloth:
            # Unsloth might handle this differently, potentially within get_unsloth_peft_model
            # Assume get_unsloth_peft_model adds the 'default' adapter
            model = get_unsloth_peft_model(model, model_args, peft_kwargs)
            debugprint("  使用 Unsloth 创建/获取 PEFT 模型 (预期包含 default 适配器)")
        else:
            # Handle PiSSA initialization
            if finetuning_args.pissa_init:
                if finetuning_args.pissa_iter == -1:
                    logger.info_rank0("Using PiSSA initialization.")
                    peft_kwargs["init_lora_weights"] = "pissa"
                else:
                    logger.info_rank0(f"Using PiSSA initialization with FSVD steps {finetuning_args.pissa_iter}.")
                    peft_kwargs["init_lora_weights"] = f"pissa_niter_{finetuning_args.pissa_iter}"
                debugprint(f"  使用 PiSSA 初始化: {peft_kwargs['init_lora_weights']}")

            # Create default adapter config
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                **peft_kwargs,
            )
            debugprint(f"  创建新的 default LoRA 配置: {lora_config}")

            # Add the 'default' adapter
            if not isinstance(model, PeftModel):
                 # If the model isn't PeftModel yet, create it with the default adapter
                 model = get_peft_model(model, lora_config, adapter_name="default")
                 debugprint("  模型非 PeftModel，使用 get_peft_model 创建并添加 default 适配器")
            else:
                 # If it's already PeftModel, add the new default adapter
                 # Check if 'default' already exists (e.g., from merging/resuming failure?)
                 if "default" in model.peft_config:
                     logger.warning_rank0("'default' adapter already exists but creating a new one. Overwriting.")
                     # PEFT might handle overwriting or require deletion first. Let's assume add_adapter overwrites.
                 model.add_adapter("default", lora_config)
                 debugprint("  模型已是 PeftModel，使用 add_adapter 添加/覆盖 default 适配器")
                 # Ensure the new adapter is active
                 model.set_adapter("default")
                 debugprint("  已将新创建的 default 适配器设为活动状态")

        # Mark that a new default adapter was created (for EMA logic below)
        # successfully_resumed_adapter remains False, needs_new_default_adapter is True

    # --- 4. If using I-LORA, ensure 'ema' adapter exists and is initialized ---
    if is_trainable and cl_finetuning_args.use_ilora:
         # Check if 'ema' adapter exists (it should have been loaded if not is_first_task, or needs creation now)
         ema_exists = hasattr(model, 'peft_config') and 'ema' in model.peft_config
         debugprint(f"  I-LORA 启用，检查 EMA 适配器存在性: {ema_exists}")

         if not ema_exists:
              # EMA needs to be created (only if default was also newly created)
              if needs_new_default_adapter:
                   logger.info_rank0("Creating new EMA adapter as it doesn't exist and default was new.")
                   debugprint("  需要创建新的 EMA 适配器 (default 也是新的)")

                   # Use the same peft_kwargs as default, but set inference_mode=True
                   peft_kwargs_ema = peft_kwargs.copy() # Reuse kwargs from default creation above
                   if "init_lora_weights" in peft_kwargs_ema: # EMA shouldn't use PiSSA init directly
                       del peft_kwargs_ema["init_lora_weights"]

                   ema_config = LoraConfig(
                       task_type=TaskType.CAUSAL_LM,
                       inference_mode=True,  # EMA adapter is never directly trained
                       **peft_kwargs_ema,
                   )
                   debugprint(f"  为 I-LORA 添加新的 EMA 适配器，配置: {ema_config}")
                   model.add_adapter("ema", ema_config)
                   debugprint("  已添加 EMA 适配器")

                   # Initialize EMA weights from the newly created default adapter
                   debugprint("  从新创建的 default 适配器初始化新 EMA 适配器的权重 (使用参数迭代)")
                   try:
                        # Store original active adapter
                        original_active = None
                        if hasattr(model, "active_adapter"):
                            original_active = model.active_adapter
                            debugprint(f"    保存原始活动适配器: {original_active}")

                        # 检查 DeepSpeed ZeRO 阶段
                        zero_stage = get_deepspeed_zero_stage(model)
                        debugprint(f"    检测到 DeepSpeed ZeRO Stage: {zero_stage}")

                        # 在 ZeRO-3 下，需要使用 gather_parameters 上下文管理器
                        with gather_parameters(model) if zero_stage == 3 else nullcontext():
                            with torch.no_grad():
                                copied_count = 0
                                # Iterate through all parameters to find default and ema LoRA weights
                                default_params = {}
                                ema_params_refs = {}

                                # First pass: collect default LoRA params and references to EMA params
                                for name, param in model.named_parameters():
                                    if "lora" in name:
                                        # Check if it's a default parameter (assuming structure like '...layers.X.[...]lora_[A/B].weight')
                                        # Exclude parameters belonging to the 'ema' adapter by checking name prefix
                                        # Default adapter params might not have a specific 'default.' prefix if it's the base PeftModel adapter
                                        is_default_param = "base_model.model." in name and ".ema." not in name and param.requires_grad # Default should be trainable
                                        is_ema_param = "base_model.model.ema." in name and not param.requires_grad # EMA should be frozen

                                        if is_default_param:
                                            # Extract relative name for matching
                                            rel_name = name.split("base_model.model.")[-1]
                                            default_params[rel_name] = param.data.clone()
                                            # debugprint(f"      Found default param: {rel_name} (from {name})")
                                        elif is_ema_param:
                                            # Extract relative name for matching (removing 'ema.')
                                            rel_name = name.split("base_model.model.ema.")[-1]
                                            ema_params_refs[rel_name] = param # Store reference to EMA param
                                            # debugprint(f"      Found EMA param ref: {rel_name} (from {name})")

                                debugprint(f"    收集到 {len(default_params)} 个 default LoRA 参数和 {len(ema_params_refs)} 个 EMA LoRA 参数引用")

                                # Second pass: copy from default to EMA using collected references
                                for rel_name, default_param_data in default_params.items():
                                    if rel_name in ema_params_refs:
                                        ema_param = ema_params_refs[rel_name]
                                        if default_param_data.shape == ema_param.shape:
                                            ema_param.data.copy_(default_param_data.to(ema_param.device))
                                            copied_count += 1
                                            # debugprint(f"      Copied {rel_name}")
                                        else:
                                            debugprint(f"      形状不匹配，跳过复制: default {rel_name} ({default_param_data.shape}) vs ema {rel_name} ({ema_param.shape})")
                                    # else:
                                    #      debugprint(f"      警告: Default 参数 {rel_name} 在 EMA 引用中未找到")

                                if copied_count > 0:
                                    debugprint(f"  已将 {copied_count} 个权重从新 default 复制到新 EMA")
                                    logger.info_rank0("Initialized new EMA adapter weights from new default adapter.")
                                else:
                                    logger.warning_rank0("Could not copy any weights from new default to new EMA adapter.")
                                    debugprint("  未能从新 default 复制任何权重到新 EMA")

                        # 在分布式环境中同步，确保所有进程完成权重复制
                        if is_distributed():
                            torch.distributed.barrier()
                            debugprint(f"  进程 rank={get_rank()} 在 EMA 权重初始化后同步")

                        # Restore active adapter if it was changed
                        if original_active is not None and hasattr(model, "active_adapter"):
                            if model.active_adapter != original_active:
                                debugprint(f"    恢复活动适配器为: {original_active}")
                                model.set_adapter(original_active)
                            else:
                                # If default was the original active, ensure it's still active
                                if original_active == "default" and model.active_adapter != "default":
                                     debugprint("    恢复活动适配器为 'default'")
                                     model.set_adapter("default")

                   except Exception as e:
                        logger.warning_rank0(f"Error copying weights to new EMA adapter: {e}", exc_info=True)
                        debugprint(f"  复制权重到新 EMA 适配器时出错: {e}")
              else:
                   # This case should ideally not happen if logic is correct
                   # (EMA needed, but default wasn't new, and EMA wasn't loaded earlier)
                   logger.error("I-LORA EMA adapter ('ema') is missing, but conditions for creating a new one were not met. Configuration error.")
                   raise RuntimeError("I-LORA EMA adapter ('ema') missing unexpectedly.")
         else:
              debugprint("  EMA 适配器已存在 (先前加载或已创建)")


    # --- 5. Cast trainable parameters to fp32 if needed ---
    if is_trainable and cast_trainable_params_to_fp32:
        debugprint("  将可训练参数转换为 fp32")
        cast_count = 0
        # Ensure the model is in training mode for parameters to have requires_grad=True
        model.train()
        # Ensure the correct adapter (default) is active for training
        if "default" in model.peft_config:
            model.set_adapter("default")

        for name, param in model.named_parameters():
            if param.requires_grad:
                # debugprint(f"    Casting {name} to fp32")
                param.data = param.data.to(torch.float32)
                cast_count += 1
        debugprint(f"    转换了 {cast_count} 个可训练参数为 fp32")


    # --- Final Checks ---
    if isinstance(model, PeftModel) and hasattr(model, 'peft_config'):
         final_adapters = list(model.peft_config.keys())
         debugprint(f"  最终可用适配器: {final_adapters}")
         if 'default' not in final_adapters:
              logger.warning_rank0("Final check: 'default' adapter not found in peft_config.")
         if cl_finetuning_args.use_ilora and 'ema' not in final_adapters:
              # This should now be an error based on earlier checks
              logger.error("Final check: 'ema' adapter not found in peft_config, but I-LORA is enabled. This indicates a loading or creation failure.")
              # raise RuntimeError("I-LORA EMA adapter missing at the end of setup.") # Optionally raise
         if hasattr(model, 'active_adapter'):
             debugprint(f"  最终活动适配器: {model.active_adapter}")
             # Ensure default is active if trainable
             if is_trainable and "default" in final_adapters and model.active_adapter != "default":
                  debugprint("  将活动适配器设置为 'default' 以进行训练")
                  model.set_adapter("default")

    else:
         debugprint("  最终模型不是 PeftModel 或没有 peft_config 属性")


    debugprint("退出 _setup_ilora_tuning 函数 (修改后)")
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
    Handles different DeepSpeed ZeRO stages appropriately.

    Args:
        config: Model configuration.
        model: The model to add adapter to.
        model_args: Model arguments.
        finetuning_args: Fine-tuning arguments.
        finetuning_args: Continual Learning fine-tuning arguments.
        is_trainable: Whether the model is being trained.

    Returns:
        Model with I-LORA adapter(s).
    """
    debugprint("进入 init_ilora_adapter 函数")
    debugprint(f"  is_trainable: {is_trainable}")
    debugprint(f"  finetuning_args: {finetuning_args}")

    # 检查分布式环境
    is_dist = is_distributed()
    rank = get_rank() if is_dist else 0
    debugprint(f"  分布式环境: {is_dist}, rank: {rank}")

    # 检查 DeepSpeed ZeRO 阶段
    zero_stage = get_deepspeed_zero_stage(model)
    debugprint(f"  检测到 DeepSpeed ZeRO Stage: {zero_stage}")

    # Cast model weights to fp32 if needed
    cast_trainable_params_to_fp32 = (
        is_trainable and getattr(model, "quantization_method", None) and model_args.upcast_layernorm
    )
    debugprint(f"  是否需要转换可训练参数为 fp32: {cast_trainable_params_to_fp32}")

    ilora_instance = None
    # Only create I-LORA instance if we're trainable and using I-LORA
    if is_trainable and cl_finetuning_args.use_ilora:
        debugprint("  条件满足 (is_trainable=True, use_ilora=True)，创建 ILORA 实例")
        ilora_instance = ILORA(
            model=model,  # This will be replaced after PEFT model creation
            finetuning_args=finetuning_args,
            cl_finetuning_args=cl_finetuning_args,
            previous_task_model=cl_finetuning_args.previous_task_model,
            current_task_id=cl_finetuning_args.current_task_id,
        )
        debugprint("  ILORA 实例已创建 (model 稍后更新)")
    else:
        debugprint("  条件不满足，不创建 ILORA 实例")

    # Use our custom setup function for I-LORA
    debugprint("  调用 _setup_ilora_tuning")
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
    debugprint("  _setup_ilora_tuning 调用完成")

    # 在分布式环境中同步，确保所有进程完成适配器设置
    if is_dist:
        torch.distributed.barrier()
        debugprint(f"  进程 rank={rank} 在适配器设置后同步")

    # Add checks after _setup_ilora_tuning returns
    debugprint("  检查 _setup_ilora_tuning 返回后的适配器状态:")
    if hasattr(model, 'peft_config'):
        debugprint(f"    可用适配器: {list(model.peft_config.keys())}")
        debugprint(f"    'ema' in model.peft_config = {'ema' in model.peft_config}")
    else:
        debugprint("    模型没有 peft_config 属性")

    # Attach I-LORA instance to the model for later use in the trainer
    if is_trainable and cl_finetuning_args.use_ilora and ilora_instance is not None:
        debugprint("  条件满足 (is_trainable=True, use_ilora=True)，将 ILORA 实例附加到模型")
        # 在 ZeRO-3 下，需要使用 gather_parameters 上下文管理器
        with gather_parameters(model) if zero_stage == 3 else nullcontext():
            ilora_instance.model = model  # Update with the PEFT model
            model.ilora = ilora_instance
            debugprint("  ILORA 实例已更新并附加到模型")
    else:
        debugprint("  条件不满足或实例未创建，不附加 ILORA 实例")

    # 再次同步，确保所有进程完成 ILORA 实例附加
    if is_dist:
        torch.distributed.barrier()
        debugprint(f"  进程 rank={rank} 在 ILORA 实例附加后同步")

    debugprint("退出 init_ilora_adapter 函数")
    return model