import os
import sys
import json
import torch
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import asdict
from transformers import HfArgumentParser
from transformers.utils import logging
from pathlib import Path
import yaml
import transformers
from transformers.trainer_utils import IntervalStrategy, SchedulerType
from transformers.training_args import OptimizerNames, ParallelMode

from llamafactory.hparams.model_args import ModelArguments
from llamafactory.hparams.data_args import DataArguments
from llamafactory.hparams.evaluation_args import EvaluationArguments
from llamafactory.hparams.finetuning_args import FinetuningArguments
from llamafactory.hparams.generating_args import GeneratingArguments
from llamafactory.hparams.training_args import RayArguments, TrainingArguments
from .cl_finetuning_args import CLFinetuningArguments
from .cl_evaluation_args import CLEvaluationArguments
from llamafactory.extras.misc import is_env_enabled, check_dependencies, check_version, get_current_device
from llamafactory.extras.constants import CHECKPOINT_NAMES

logger = logging.get_logger(__name__)

_MODEL_CLS = ModelArguments
_DATA_CLS = DataArguments
_EVAL_CLS = EvaluationArguments
_FINETUNING_CLS = FinetuningArguments
_CL_FINETUNING_CLS = CLFinetuningArguments

check_dependencies()


_TRAIN_ARGS = [ModelArguments, DataArguments, TrainingArguments, FinetuningArguments, CLFinetuningArguments, GeneratingArguments]
_TRAIN_CLS = Tuple[ModelArguments, DataArguments, TrainingArguments, FinetuningArguments, CLFinetuningArguments, GeneratingArguments]
_INFER_ARGS = [ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments]
_INFER_CLS = Tuple[ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments]
_EVAL_ARGS = [ModelArguments, DataArguments, EvaluationArguments, FinetuningArguments]
_EVAL_CLS = Tuple[ModelArguments, DataArguments, EvaluationArguments, FinetuningArguments]
_CL_EVAL_ARGS = [ModelArguments, DataArguments, CLEvaluationArguments, FinetuningArguments]
_CL_EVAL_CLS = Tuple[ModelArguments, DataArguments, CLEvaluationArguments, FinetuningArguments]


def read_args(args: Optional[Union[Dict[str, Any], List[str]]] = None) -> Union[Dict[str, Any], List[str]]:
    r"""
    Gets arguments from the command line or a config file.
    """
    if args is not None:
        return args

    if len(sys.argv) == 2 and (sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml")):
        return yaml.safe_load(Path(sys.argv[1]).absolute().read_text())
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return json.loads(Path(sys.argv[1]).absolute().read_text())
    else:
        return sys.argv[1:]


def _parse_args(
    parser: "HfArgumentParser", args: Optional[Union[Dict[str, Any], List[str]]] = None, allow_extra_keys: bool = False
) -> Tuple[Any]:
    args = read_args(args)
    if isinstance(args, dict):
        return parser.parse_dict(args, allow_extra_keys=allow_extra_keys)

    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(args=args, return_remaining_strings=True)

    if unknown_args and not allow_extra_keys:
        print(parser.format_help())
        print(f"Got unknown args, potentially deprecated arguments: {unknown_args}")
        raise ValueError(f"Some specified arguments are not used by the HfArgumentParser: {unknown_args}")

    return tuple(parsed_args)


def _set_transformers_logging() -> None:
    if os.getenv("LLAMAFACTORY_VERBOSITY", "INFO") in ["DEBUG", "INFO"]:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()


def _verify_model_args(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    finetuning_args: "FinetuningArguments",
) -> None:
    if model_args.adapter_name_or_path is not None and finetuning_args.finetuning_type != "lora":
        raise ValueError("Adapter is only valid for the LoRA method.")

    if model_args.quantization_bit is not None:
        if finetuning_args.finetuning_type != "lora":
            raise ValueError("Quantization is only compatible with the LoRA method.")

        if finetuning_args.pissa_init:
            raise ValueError("Please use scripts/pissa_init.py to initialize PiSSA for a quantized model.")

        if model_args.resize_vocab:
            raise ValueError("Cannot resize embedding layers of a quantized model.")

        if model_args.adapter_name_or_path is not None and finetuning_args.create_new_adapter:
            raise ValueError("Cannot create new adapter upon a quantized model.")

        if model_args.adapter_name_or_path is not None and len(model_args.adapter_name_or_path) != 1:
            raise ValueError("Quantized model only accepts a single adapter. Merge them first.")

    if data_args.template == "yi" and model_args.use_fast_tokenizer:
        logger.warning_rank0("We should use slow tokenizer for the Yi models. Change `use_fast_tokenizer` to False.")
        model_args.use_fast_tokenizer = False


def _check_extra_dependencies(
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    training_args: Optional["TrainingArguments"] = None,
) -> None:
    if model_args.use_unsloth:
        check_version("unsloth", mandatory=True)

    if model_args.enable_liger_kernel:
        check_version("liger-kernel", mandatory=True)

    if model_args.mixture_of_depths is not None:
        check_version("mixture-of-depth>=1.1.6", mandatory=True)

    if model_args.infer_backend == "vllm":
        check_version("vllm>=0.4.3,<=0.7.2")
        check_version("vllm", mandatory=True)

    if finetuning_args.use_galore:
        check_version("galore_torch", mandatory=True)

    if finetuning_args.use_apollo:
        check_version("apollo_torch", mandatory=True)

    if finetuning_args.use_badam:
        check_version("badam>=1.2.1", mandatory=True)

    if finetuning_args.use_adam_mini:
        check_version("adam-mini", mandatory=True)

    if finetuning_args.plot_loss:
        check_version("matplotlib", mandatory=True)

    if training_args is not None and training_args.predict_with_generate:
        check_version("jieba", mandatory=True)
        check_version("nltk", mandatory=True)
        check_version("rouge_chinese", mandatory=True)


def _parse_train_args(args: Optional[Union[Dict[str, Any], List[str]]] = None) -> _TRAIN_CLS:
    parser = HfArgumentParser(_TRAIN_ARGS)
    allow_extra_keys = is_env_enabled("ALLOW_EXTRA_ARGS")
    return _parse_args(parser, args, allow_extra_keys=allow_extra_keys)


def _parse_infer_args(args: Optional[Union[Dict[str, Any], List[str]]] = None) -> _INFER_CLS:
    parser = HfArgumentParser(_INFER_ARGS)
    allow_extra_keys = is_env_enabled("ALLOW_EXTRA_ARGS")
    return _parse_args(parser, args, allow_extra_keys=allow_extra_keys)


def _parse_eval_args(
    args: Optional[Union[Dict[str, Any], List[str]]] = None,
    allow_extra_keys: bool = False
) -> _EVAL_CLS:
    """
    Parse evaluation arguments.
    """
    parser = HfArgumentParser([ModelArguments, DataArguments, EvaluationArguments, FinetuningArguments])
    if isinstance(args, Dict):
        model_args, data_args, eval_args, finetuning_args = parser.parse_dict(
            args, allow_extra_keys=True  # 允许额外的参数
        )
    else:
        model_args, data_args, eval_args, finetuning_args = parser.parse_args_into_dataclasses()

    if isinstance(args, Dict) and "model_name_or_path" not in args:
        raise ValueError("Please specify `model_name_or_path` to load model.")

    return model_args, data_args, eval_args, finetuning_args


def _parse_cl_eval_args(args: Optional[Union[Dict[str, Any], List[str]]] = None) -> _CL_EVAL_CLS:
    """
    Parse continual learning evaluation arguments.
    """
    parser = HfArgumentParser([ModelArguments, DataArguments, CLEvaluationArguments, FinetuningArguments])
    allow_extra_keys = is_env_enabled("ALLOW_EXTRA_ARGS")
    return _parse_args(parser, args, allow_extra_keys=True)  # 允许额外的参数


def get_ray_args(args: Optional[Union[Dict[str, Any], List[str]]] = None) -> RayArguments:
    parser = HfArgumentParser(RayArguments)
    (ray_args,) = _parse_args(parser, args, allow_extra_keys=True)
    return ray_args


def get_train_args(args: Optional[Union[Dict[str, Any], List[str]]] = None) -> _TRAIN_CLS:
    model_args, data_args, training_args, finetuning_args, cl_finetuning_args, generating_args = _parse_train_args(args)
    return model_args, data_args, training_args, finetuning_args, cl_finetuning_args, generating_args


def get_infer_args(args: Optional[Union[Dict[str, Any], List[str]]] = None) -> _INFER_CLS:
    model_args, data_args, finetuning_args, generating_args = _parse_infer_args(args)

    _set_transformers_logging()

    if model_args.infer_backend == "vllm":
        if finetuning_args.stage != "sft":
            raise ValueError("vLLM engine only supports auto-regressive models.")

        if model_args.quantization_bit is not None:
            raise ValueError("vLLM engine does not support bnb quantization (GPTQ and AWQ are supported).")

        if model_args.rope_scaling is not None:
            raise ValueError("vLLM engine does not support RoPE scaling.")

        if model_args.adapter_name_or_path is not None and len(model_args.adapter_name_or_path) != 1:
            raise ValueError("vLLM only accepts a single adapter. Merge them first.")

    _verify_model_args(model_args, data_args, finetuning_args)
    _check_extra_dependencies(model_args, finetuning_args)

    if model_args.export_dir is not None and model_args.export_device == "cpu":
        model_args.device_map = {"": torch.device("cpu")}
        model_args.model_max_length = data_args.cutoff_len
    else:
        model_args.device_map = "auto"

    return model_args, data_args, finetuning_args, generating_args


def get_eval_args(args: Optional[Union[Dict[str, Any], List[str]]] = None) -> _EVAL_CLS:
    model_args, data_args, eval_args, finetuning_args = _parse_eval_args(args)

    _set_transformers_logging()

    if model_args.infer_backend == "vllm":
        raise ValueError("vLLM backend is only available for API, CLI and Web.")

    _verify_model_args(model_args, data_args, finetuning_args)
    _check_extra_dependencies(model_args, finetuning_args)

    model_args.device_map = "auto"

    transformers.set_seed(eval_args.seed)

    return model_args, data_args, eval_args, finetuning_args


def get_cl_eval_args(args: Optional[Union[Dict[str, Any], List[str]]] = None) -> _CL_EVAL_CLS:
    model_args, data_args, cl_eval_args, finetuning_args = _parse_cl_eval_args(args)

    _set_transformers_logging()

    if model_args.infer_backend == "vllm":
        raise ValueError("vLLM backend is only available for API, CLI and Web.")

    _verify_model_args(model_args, data_args, finetuning_args)
    _check_extra_dependencies(model_args, finetuning_args)

    model_args.device_map = "auto"

    transformers.set_seed(cl_eval_args.seed)

    return model_args, data_args, cl_eval_args, finetuning_args
