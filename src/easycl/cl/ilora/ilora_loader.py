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
from typing import TYPE_CHECKING, Any, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForVision2Seq
from trl import AutoModelForCausalLMWithValueHead

from llamafactory.extras import logging
from llamafactory.extras.misc import count_parameters
from llamafactory.model.loader import _get_init_kwargs, load_config, load_tokenizer
from llamafactory.model.model_utils.liger_kernel import apply_liger_kernel
from llamafactory.model.model_utils.misc import register_autoclass
from llamafactory.model.model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model
from llamafactory.model.model_utils.unsloth import load_unsloth_pretrained_model
from llamafactory.model.model_utils.valuehead import load_valuehead_params
from llamafactory.model.patcher import patch_config, patch_model, patch_processor, patch_tokenizer, patch_valuehead_model
from easycl.cl.ilora.ilora_adapter import init_ilora_adapter

def debugprint(*args, **kwargs):
    pass

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from llamafactory.hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


def load_ilora_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    cl_finetuning_args: "CLFinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    """
    Loads pretrained model with I-LORA adapter.
    
    This function is based on the original load_model function but uses init_ilora_adapter
    instead of init_adapter for adapter initialization.
    """
    debugprint("进入 load_ilora_model 函数")
    debugprint(f"  model_args: {model_args}")
    debugprint(f"  finetuning_args: {finetuning_args}")
    debugprint(f"  is_trainable: {is_trainable}")
    debugprint(f"  add_valuehead: {add_valuehead}")

    init_kwargs = _get_init_kwargs(model_args)
    debugprint(f"  初始化 kwargs: {init_kwargs}")
    config = load_config(model_args)
    debugprint(f"  加载的 config 类型: {type(config)}")
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
    debugprint("  已 patch config")
    apply_liger_kernel(config, model_args, is_trainable, require_logits=(finetuning_args.stage not in ["pt", "sft"]))
    debugprint("  已应用 Liger kernel (如果需要)")

    model = None
    lazy_load = False
    if model_args.use_unsloth:
        debugprint("  检测到使用 Unsloth")
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
            debugprint(f"    存在 adapter_name_or_path ({model_args.adapter_name_or_path})，将进行惰性加载")
        elif is_trainable:
            debugprint("    可训练模式，尝试加载 Unsloth 预训练模型")
            model = load_unsloth_pretrained_model(config, model_args)
            debugprint(f"    Unsloth 模型加载完成 (类型: {type(model)})")
        else:
            debugprint("    非训练模式且无 adapter，Unsloth 不加载模型")
    else:
        debugprint("  不使用 Unsloth")

    if model is None and not lazy_load:
        debugprint("  模型为空且非惰性加载，开始标准模型加载流程")
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path

        if model_args.mixture_of_depths == "load":
            debugprint(f"  加载 MoD 模型: {init_kwargs['pretrained_model_name_or_path']}")
            model = load_mod_pretrained_model(**init_kwargs)
        else:
            if type(config) in AutoModelForVision2Seq._model_mapping.keys():  # assume built-in models
                load_class = AutoModelForVision2Seq
                debugprint("  使用 AutoModelForVision2Seq 加载")
            elif type(config) in AutoModelForSeq2SeqLM._model_mapping.keys():
                load_class = AutoModelForSeq2SeqLM
                debugprint("  使用 AutoModelForSeq2SeqLM 加载")
            else:
                load_class = AutoModelForCausalLM
                debugprint("  使用 AutoModelForCausalLM 加载")

            if model_args.train_from_scratch:
                debugprint(f"  从头开始训练，使用 {load_class.__name__}.from_config")
                model = load_class.from_config(config, trust_remote_code=model_args.trust_remote_code)
            else:
                debugprint(f"  加载预训练模型，使用 {load_class.__name__}.from_pretrained: {init_kwargs['pretrained_model_name_or_path']}")
                model = load_class.from_pretrained(**init_kwargs)

        debugprint(f"  标准模型加载完成 (类型: {type(model)})")
        if model_args.mixture_of_depths == "convert":
            debugprint("  转换预训练模型为 MoD 模型")
            model = convert_pretrained_model_to_mod(model, config, model_args)
            debugprint(f"  MoD 转换完成 (类型: {type(model)})")

    if not lazy_load:
        debugprint("  非惰性加载，进行模型 patching 和注册")
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)
        debugprint("  模型 patching 和注册完成")
    else:
        debugprint("  惰性加载模式，跳过模型 patching 和注册")
    
    # Key change: Use init_ilora_adapter instead of init_adapter
    debugprint("  调用 init_ilora_adapter 初始化 I-LORA 适配器")
    model = init_ilora_adapter(config, model, model_args, finetuning_args, cl_finetuning_args, is_trainable)
    debugprint(f"  init_ilora_adapter 调用完成 (模型类型: {type(model)})")

    if add_valuehead:
        debugprint("  添加 Value Head")
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)
        debugprint("  Value Head 添加并 patch 完成")

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
            debugprint(f"  从最后一个适配器路径加载 Value Head 参数: {vhead_path}")
        else:
            vhead_path = model_args.model_name_or_path
            debugprint(f"  从模型路径加载 Value Head 参数: {vhead_path}")

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            debugprint(f"  成功加载 Value Head 参数，应用到模型")
            model.load_state_dict(vhead_params, strict=False)
            logger.info_rank0(f"Loaded valuehead from checkpoint: {vhead_path}")
        else:
            debugprint("  未找到 Value Head 参数")

    if not is_trainable:
        debugprint("  非训练模式，设置 requires_grad=False 并转换 dtype")
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()
        debugprint("  模型已设置为评估模式")
    else:
        debugprint("  训练模式，设置模型为训练状态")
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = "trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    else:
        param_stats = f"all params: {all_param:,}"

    logger.info_rank0(param_stats)
    debugprint(f"  参数统计: {param_stats}")

    if model_args.print_param_status and int(os.getenv("LOCAL_RANK", "0")) == 0:
        debugprint("  打印模型参数状态 (名称, dtype, device, trainable)")
        for name, param in model.named_parameters():
            print(f"name: {name}, dtype: {param.dtype}, device: {param.device}, trainable: {param.requires_grad}")

    debugprint("退出 load_ilora_model 函数")
    return model