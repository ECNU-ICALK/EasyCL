"""
一键式持续学习训练和评估的实现
"""
import os
import json
import torch
import gc
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import asdict, dataclass, field
import logging
import sys
from pathlib import Path

# Attempt to import yaml and handle ImportError gracefully
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    # No need to log here, will log within _load_params if YAML file is provided without library

from llamafactory.hparams import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    FinetuningArguments,
    GeneratingArguments,
    get_train_args,
)
from easycl.hparams.cl_evaluation_args import CLEvaluationArguments
from easycl.hparams.cl_finetuning_args import CLFinetuningArguments
from llamafactory.train.tuner import run_exp
from llamafactory.extras.logging import get_logger
from llamafactory.hparams.parser import HfArgumentParser
from easycl.cl_workflow.evaluator import CLEvaluator
from easycl.cl_workflow.cl_eval.cl_metrics import CLMetricsCalculator
# Import BenchmarkHandler
from .benchmark_handler import BenchmarkHandler

logger = get_logger(__name__)

@dataclass
class CLTrainArguments:
    """持续学习训练参数"""
    train_params: str = field(
        default="",
        metadata={"help": "JSON格式的训练参数文件路径"}
    )
    output_dir: str = field(
        default="outputs",
        metadata={"help": "训练输出目录"}
    )

@dataclass
class CLEvalArguments:
    """持续学习评估参数"""
    eval_params: str = field(
        default="",
        metadata={"help": "JSON格式的评估参数文件路径"}
    )
    save_dir: str = field(
        default="eval_results",
        metadata={"help": "评估结果保存目录"}
    )

@dataclass
class CLWorkflowArguments:
    """持续学习工作流参数"""
    mode: str = field(
        default="train_only",
        metadata={
            "help": "运行模式：'train_only', 'eval_only', 'train_then_eval', 'full_workflow'"
        }
    )
    previewonly: bool = field(
        default=False,
        metadata={"help": "是否只预览命令而不执行"}
    )
    clean_dirs: bool = field(
        default=False,
        metadata={"help": "是否在运行前清空输出和评估目录"}
    )
    # --- Add Benchmark Arguments ---
    benchmark: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the benchmark to run."}
    )
    benchmark_order: Optional[str] = field(
        default=None,
        metadata={"help": "Specific task order defined within the benchmark's info file (e.g., 'order1')."}
    )
    benchmark_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing the benchmark definition and data."}
    )
    # --- End Benchmark Arguments ---

class CLCommandGenerator:
    """持续学习命令生成器"""

    def __init__(self, train_kwargs: Dict, eval_kwargs: Dict):
        self.train_kwargs = train_kwargs
        self.eval_kwargs = eval_kwargs
        # 从train_kwargs中解析任务列表
        self.tasks = self.train_kwargs.get("dataset", "").split(",")
        if not self.tasks or not self.tasks[0]:
            #logger.warning("未在训练参数中找到有效的dataset参数，tasks列表将为空")
            self.tasks = []

        # Load config only once
        self.config_path = os.path.join(os.path.dirname(__file__), "cl_params_config.json")
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except Exception as e:
            logger.error(f"无法加载参数配置文件: {self.config_path}. Error: {str(e)}")
            logger.error("将使用空的配置，可能导致参数管理错误！")
            self.config = {} # Use empty config to avoid crashing, but log error

    def _dict_to_args(self, args_dict: Dict) -> str:
        """将字典转换为命令行参数字符串

        Args:
            args_dict: 参数字典

        Returns:
            str: 命令行参数字符串
        """
        args = []
        for k, v in args_dict.items():
            # 跳过None值、注释参数和内部使用的参数
            if v is None or k.startswith("#") or k == "cl_method":
                continue

            # 处理布尔值
            if isinstance(v, bool):
                args.append(f"--{k} {str(v).lower()}")
            # 处理adapter_name_or_path特殊情况
            elif k == "adapter_name_or_path":
                # 对于adapter_name_or_path，如果是列表则转换为逗号分隔的字符串
                if isinstance(v, (list, tuple)):
                    args.append(f"--{k} {','.join(map(str, v))}")
                else:
                    args.append(f"--{k} {str(v)}")
            # 处理hidden_state_layers特殊情况
            elif k == "hidden_state_layers":
                # 将列表转换为空格分隔的字符串
                if isinstance(v, (list, tuple)):
                    args.append(f"--{k} {' '.join(map(str, v))}")
                else:
                    args.append(f"--{k} {str(v)}")
            # 处理列表或元组
            elif isinstance(v, (list, tuple)):
                # 如果是空列表，跳过该参数
                if not v:
                    continue
                # 如果是列表或元组，将其转换为逗号分隔的字符串
                args.append(f"--{k} {','.join(map(str, v))}")
            # 处理字典
            elif isinstance(v, dict):
                # 如果是空字典，跳过该参数
                if not v:
                    continue
                # 将字典转换为JSON字符串
                import json
                json_str = json.dumps(v, ensure_ascii=False).replace('"', '\\"')
                args.append(f'--{k} "{json_str}"')
            else:
                # 对于其他类型，直接转换为字符串
                args.append(f"--{k} {str(v)}")
        return " ".join(args)

    def _get_auto_managed_params(self, task_id: int, task: str, is_first_task: bool) -> Dict[str, Any]:
        """获取自动管理的、依赖于前序任务的参数 (Refactored based on new config)"""
        managed_args = {}
        # Use user-provided base output directory
        base_dir = self.train_kwargs["output_dir"]
        # Task-specific output dir (this is essential and always managed)
        task_output_dir = os.path.join(base_dir, f"task_{task_id}_{task}")
        managed_args["output_dir"] = task_output_dir # Always override output_dir for the task

        # Get CL method requirements and mappings from loaded config
        cl_method = self.train_kwargs.get("cl_method")

        # --- Task ID Management (Execute First for All Methods) ---
        if cl_method:
            method_requirements = self.config.get("cl_method_requirements", {}).get(cl_method, {})
            if method_requirements.get("needs_task_id"):
                managed_args["current_task_id"] = task
                if not is_first_task:
                    prev_task_id_param = "prev_task_id" # Default param name
                    # Check if there's a specific mapping for prev_task_id (unlikely but possible)
                    method_specific_mappings = self.config.get("task_output_mapping", {}).get(cl_method, {}).get("params", [])
                    mapping_dict = {m["param_name"]: m for m in method_specific_mappings} # For quick lookup
                    if prev_task_id_param in mapping_dict:
                         # Use mapped param name if specified, although source is always the previous task ID
                         # This part might be overly complex unless a method uses a different name like 'previous_task_identifier'
                         pass # Stick to 'prev_task_id' for simplicity unless explicitly needed otherwise
                    managed_args[prev_task_id_param] = self.tasks[task_id - 1]
                    logger.info_rank0(f"Task {task_id} ({task}): Setting {prev_task_id_param}={managed_args[prev_task_id_param]} for method {cl_method.upper()}")

        # Special handling for O-LoRA method
        if cl_method == "olora":
            if not is_first_task:
                # For O-LoRA, use the merged model from previous task as base model
                prev_task_dir_name = f"task_{task_id-1}_{self.tasks[task_id-1]}"
                prev_task_output_dir = os.path.join(base_dir, prev_task_dir_name)
                prev_merged_model_dir = os.path.join(prev_task_output_dir, "merged_model")
                
                # Set model_name_or_path to previous merged model
                managed_args["model_name_or_path"] = prev_merged_model_dir
                logger.info_rank0(f"Task {task_id}: O-LoRA method detected. Setting model_name_or_path={prev_merged_model_dir}")
                
                # Do NOT set adapter_name_or_path for O-LoRA
                # O-LoRA starts fresh from the merged base model
            
            # Continue with remaining parameter management for O-LoRA
            # Don't return early anymore

        # 如果没有指定cl_method且不是第一个任务，自动设置adapter_name_or_path
        if not cl_method and not is_first_task:
            # 获取前一个任务的输出目录
            prev_task_dir_name = f"task_{task_id-1}_{self.tasks[task_id-1]}"
            prev_task_output_dir = os.path.join(base_dir, prev_task_dir_name)

            # 设置adapter_name_or_path为前一个任务的输出目录
            managed_args["adapter_name_or_path"] = prev_task_output_dir
            logger.info_rank0(f"Task {task_id}: No CL method specified, but not first task. Setting adapter_name_or_path={prev_task_output_dir}")

            # 返回管理的参数
            return managed_args

        if not cl_method:
            # If no CL method specified, only manage output_dir and task_id (if needed)
            return managed_args

        method_requirements = self.config.get("cl_method_requirements", {}).get(cl_method, {})
        if not method_requirements:
            logger.warning(f"No requirements found for CL method '{cl_method}' in config. Skipping auto-param management.")
            return managed_args

        default_mappings = self.config.get("default_param_mappings", {})
        method_specific_mappings = self.config.get("task_output_mapping", {}).get(cl_method, {}).get("params", [])
        mapping_dict = {m["param_name"]: m for m in method_specific_mappings} # For quick lookup

        # --- First Task Special Params ---
        if is_first_task:
            first_task_special_params = self.config.get("first_task_special_params", {}).get(cl_method, {})
            for param_name, value in first_task_special_params.items():
                managed_args[param_name] = value
                logger.info_rank0(f"First Task ({task}): Setting special param {param_name}={value} for method {cl_method.upper()}")
            # No further processing needed for first task regarding previous task dependencies
            return managed_args

        # --- Subsequent Task Parameter Management (Not First Task) ---

        # Previous Task Directory (needed for model/adapter/data mappings)
        prev_task_dir_name = f"task_{task_id-1}_{self.tasks[task_id-1]}"
        prev_task_output_dir = os.path.join(base_dir, prev_task_dir_name)

        # --- Previous Model/Adapter Mapping ---
        if method_requirements.get("needs_prev_model"):
            target_param = "previous_task_model" # Default
            related_to = "output_dir"         # Default source

            # Check for method-specific mapping override
            found_specific_model_mapping = False
            for mapping in method_specific_mappings:
                # Check if this mapping relates to a model/adapter path
                # Heuristic: check if 'model' or 'adapter' is in the name or description, or if related_to is output_dir
                is_model_related_mapping = (
                    "model" in mapping["param_name"] or
                    "adapter" in mapping["param_name"] or
                    mapping.get("related_to") == "output_dir"
                )
                if is_model_related_mapping and mapping["param_name"] != "prev_task_dir": # Exclude LaMoL case for now
                    target_param = mapping["param_name"]
                    related_to = mapping["related_to"]
                    logger.info_rank0(f"Task {task_id}: Found specific mapping for model/adapter: {target_param} related to {related_to}")
                    found_specific_model_mapping = True
                    break # Assume only one primary model/adapter mapping per method

            if found_specific_model_mapping: # Handle specific mapping first
                if related_to == "output_dir":
                    managed_args[target_param] = prev_task_output_dir
                    logger.info_rank0(f"Task {task_id}: Setting specific {target_param}={prev_task_output_dir} for method {cl_method.upper()} based on task_output_mapping.")
                else:
                    logger.warning(f"Task {task_id}: Cannot map previous model/adapter via specific mapping for method {cl_method.upper()}. Unexpected related_to: {related_to}")
            else: # No specific mapping found in task_output_mapping for this method
                # Skip setting previous_task_model for O-LoRA as it uses model_name_or_path differently
                if cl_method != "olora":
                    managed_args["previous_task_model"] = prev_task_output_dir
                    logger.info_rank0(f"Task {task_id}: Setting previous_task_model={prev_task_output_dir} for method {cl_method.upper()} as default path reference.")

        # --- Handle adapter_name_or_path independently ---
        # 添加单独处理 adapter_name_or_path 的逻辑，基于 needs_adapter_name_or_path 标志
        if method_requirements.get("needs_adapter_name_or_path", False) and not is_first_task:
            # Skip adapter_name_or_path for O-LoRA as it starts fresh from merged base model
            if cl_method != "olora":
                # 检查是否有方法特定的adapter_name_or_path映射
                adapter_mapping = None
                for mapping in method_specific_mappings:
                    if mapping["param_name"] == "adapter_name_or_path" and mapping.get("related_to") == "output_dir":
                        adapter_mapping = mapping
                        break

                if adapter_mapping:
                    # 使用方法特定的映射
                    managed_args["adapter_name_or_path"] = prev_task_output_dir
                    logger.info_rank0(f"Task {task_id}: Setting adapter_name_or_path={prev_task_output_dir} for method {cl_method.upper()} based on method-specific mapping.")
                else:
                    # 没有找到方法特定的映射，但needs_adapter_name_or_path为True，使用默认映射
                    managed_args["adapter_name_or_path"] = prev_task_output_dir
                    logger.info_rank0(f"Task {task_id}: Setting adapter_name_or_path={prev_task_output_dir} for method {cl_method.upper()} based on needs_adapter_name_or_path=True.")

        # --- Previous Data Mapping ---
        if method_requirements.get("needs_prev_data"):
            target_param = "previous_task_data" # Default
            related_to = "dataset"           # Default source is previous dataset name/id

            # Check for method-specific mapping override
            if target_param in mapping_dict:
                related_to = mapping_dict[target_param].get("related_to", related_to)

            if related_to == "dataset":
                managed_args[target_param] = self.tasks[task_id - 1] # Use previous task name as identifier
                logger.info_rank0(f"Task {task_id}: Setting {target_param}={managed_args[target_param]} for method {cl_method.upper()}")
            else:
                logger.warning(f"Task {task_id}: Cannot map previous data for method {cl_method.upper()}. Unexpected related_to: {related_to}")

        # --- Shared Path Parameters ---
        # Ensure shared paths defined in the config are preserved from the original train_kwargs
        shared_paths_needed = method_requirements.get("needs_shared_paths", [])
        for shared_param in shared_paths_needed:
            if shared_param in self.train_kwargs:
                # Check if the shared param has a specific mapping rule (e.g., related_to itself)
                if shared_param in mapping_dict and mapping_dict[shared_param].get("related_to") == shared_param:
                    managed_args[shared_param] = self.train_kwargs[shared_param]
                    logger.info_rank0(f"Task {task_id}: Preserving shared path parameter {shared_param}={managed_args[shared_param]} for method {cl_method.upper()}")
                elif shared_param not in managed_args: # Avoid overwriting if already set by another rule
                    # If no specific mapping, assume it should be preserved if present in original args
                    managed_args[shared_param] = self.train_kwargs[shared_param]
                    logger.info_rank0(f"Task {task_id}: Preserving shared path parameter {shared_param}={managed_args[shared_param]} (from input args) for method {cl_method.upper()}")
            else:
                logger.warning(f"Task {task_id}: Method {cl_method.upper()} requires shared path '{shared_param}', but it was not found in input arguments.")

        # --- Handle LaMoL's prev_task_dir specifically ---
        if cl_method == "lamol" and "prev_task_dir" in mapping_dict:
            if mapping_dict["prev_task_dir"].get("related_to") == "output_dir":
                managed_args["prev_task_dir"] = prev_task_output_dir
                logger.info_rank0(f"Task {task_id}: Setting prev_task_dir={prev_task_output_dir} for LAMOL")


        return managed_args

    def generate_train_command(self, task_id: int, task: str) -> str:
        """生成训练命令 (Refactored based on new config)"""
        args = self.train_kwargs.copy()
        is_first_task = (task_id == 0)

        # 1. Set basic parameters (dataset)
        args["dataset"] = task

        # 2. Handle CL Method selection (assuming it's already in args if needed)
        cl_method = args.get("cl_method")
        method_requirements = {}
        if cl_method:
            method_requirements = self.config.get("cl_method_requirements", {}).get(cl_method, {})
            if not method_requirements:
                logger.warning(f"Config entry not found for specified cl_method: '{cl_method}'. Parameter handling might be incorrect.")

        # Ensure finetuning_type is set correctly for methods like iLora
        if cl_method == "ilora":
            args["finetuning_type"] = "lora"  # I-LORA requires LoRA

        # 3. Handle First Task Parameters
        if is_first_task:
            first_task_config = self.config.get("first_task_params", {})
            # Remove specified parameters
            for param in first_task_config.get("remove_params", []):
                if param in args:
                    del args[param]
                    logger.info_rank0(f"First Task ({task}): Removing parameter '{param}'.")

            # Force disable methods (remove their 'use_*' flag if present)
            for method_to_disable in first_task_config.get("force_disable_methods", []):
                use_flag = f"use_{method_to_disable}"
                if use_flag in args:
                    del args[use_flag]
                    logger.info_rank0(f"First Task ({task}): Forcing disable of method '{method_to_disable}' by removing '{use_flag}'.")
                # If the disabled method was the selected cl_method, warn and clear it
                if cl_method == method_to_disable:
                    logger.warning(f"First Task ({task}): Selected cl_method '{cl_method}' is in force_disable_methods. Clearing cl_method.")
                    cl_method = None
                    args.pop("cl_method", None)


        # 4. Handle Incremental Parameters (Subsequent Tasks Only)
        elif cl_method and not is_first_task: # Only process if not first task and a CL method is active
            incremental_params_config = self.config.get("incremental_params", {})
            default_mappings = self.config.get("default_param_mappings", {})

            # --- Automatic Replay List Handling ---
            method_requirements = self.config.get("cl_method_requirements", {}).get(cl_method, {})
            if method_requirements.get("needs_replay_list"):
                replay_list_config = default_mappings.get("replay_task_list", {})
                param_name = "replay_task_list" # Standard name
                source_type = replay_list_config.get("source", "accumulated_datasets") # Default to accumulated
                separator = replay_list_config.get("separator", ",") # Default separator

                if source_type == "accumulated_datasets":
                    previous_tasks_datasets = self.tasks[:task_id]
                    if previous_tasks_datasets:
                        args[param_name] = separator.join(previous_tasks_datasets)
                        logger.info_rank0(f"Task {task_id}: Auto-setting incremental param {param_name}={args[param_name]} for method {cl_method.upper()} based on needs_replay_list=True.")
                else:
                    logger.warning(f"Task {task_id}: Cannot auto-handle replay list for method {cl_method.upper()} with source type '{source_type}'.")
            # --- End Automatic Replay List Handling ---

            # --- Handle other explicitly defined incremental params (like EWC's previous_task_data) ---
            for method_in_config, inc_param_info in incremental_params_config.items():
                if method_in_config == "description": continue # Skip description key

                # Check if the current cl_method matches the method requiring this incremental param
                # AND ensure we are not re-processing the auto-handled replay_task_list
                if cl_method == method_in_config:
                    param_name = inc_param_info.get("param_name")
                    if not param_name or param_name == "replay_task_list": # Skip if no name or if it's the auto-handled one
                        continue

                    # Determine source and separator (check method specific, then default)
                    default_mapping_key = None
                    # Heuristic for data lists (like EWC)
                    if "data" in param_name:
                        default_mapping_key = "previous_task_data"
                    # Add more heuristics or make param_name match default mapping keys directly

                    default_map = default_mappings.get(default_mapping_key, {}) if default_mapping_key else {}
                    source_type = inc_param_info.get("source", default_map.get("source"))
                    separator = inc_param_info.get("separator", default_map.get("separator", ","))

                    # Example for accumulating data identifiers (like EWC)
                    if source_type == "accumulated_datasets": # Reusing this source type might be ambiguous, consider specific source?
                        # Logic for accumulating *data* identifiers, NOT task names like replay
                        # This might need adjustment based on how EWC needs its accumulated data list
                        # Assuming previous task's dataset ID is needed
                        previous_datasets = self.tasks[:task_id] # Still uses previous task names as identifiers
                        if previous_datasets:
                            args[param_name] = separator.join(previous_datasets)
                            logger.info_rank0(f"Task {task_id}: Setting incremental param {param_name}={args[param_name]} for method {cl_method.upper()} based on incremental_params config.")
                    # Add logic for other source_types if needed for other explicit incremental params
                    else:
                         logger.warning(f"Task {task_id}: Cannot handle explicitly defined incremental parameter '{param_name}' with source type '{source_type}' for method {cl_method.upper()}")


        # 5. Add Auto-Managed Parameters (Paths, IDs, First Task Specials)
        # This needs the possibly updated cl_method after first task handling
        current_cl_method = args.get("cl_method") # Get potentially cleared method
        effective_train_kwargs = self.train_kwargs.copy()
        effective_train_kwargs["cl_method"] = current_cl_method # Pass the correct method state

        # Create a temporary generator instance with potentially updated state if needed
        temp_command_generator = CLCommandGenerator(effective_train_kwargs, self.eval_kwargs)
        auto_managed_params = temp_command_generator._get_auto_managed_params(task_id, task, is_first_task)

        # Important: Update args, letting auto_managed_params override where necessary (like output_dir)
        args.update(auto_managed_params)

        # 6. Ensure necessary base parameters exist (stage, do_train)
        args.setdefault("stage", "sft")
        args.setdefault("do_train", True)

        # 7. Convert to command string
        # Make sure 'cl_method' itself is not passed as a command line arg if it was just for internal logic
        args_for_cli = args.copy()
        # args_for_cli.pop("cl_method", None) # Keep cl_method if needed by run_exp downstream? Check run_exp usage. Assuming run_exp doesn't need it directly as CLI arg.

        return f"easycl-cli cl_train {self._dict_to_args(args_for_cli)}"

    def generate_eval_command(self, task_output_dir: str, task_id: Optional[int], is_lora: bool, base_model_path: Optional[str] = None) -> str:
        """生成评估命令 (No changes needed based on request)"""
        args = self.eval_kwargs.copy()

        # 设置模型路径
        if is_lora:
            args["model_name_or_path"] = base_model_path
            # 确保adapter_name_or_path是字符串格式
            if isinstance(task_output_dir, (list, tuple)):
                args["adapter_name_or_path"] = ",".join(map(str, task_output_dir))
            else:
                args["adapter_name_or_path"] = str(task_output_dir)
        else:
            args["model_name_or_path"] = str(task_output_dir)
            if "adapter_name_or_path" in args:
                del args["adapter_name_or_path"]

        # 设置保存目录
        eval_base_dir = args.get("save_dir", "eval_results")
        if task_id is not None:
            args["save_dir"] = os.path.join(eval_base_dir, f"task_{task_id}")
        else:
            args["save_dir"] = os.path.join(eval_base_dir, "base")

        # 设置其他评估参数
        args.setdefault("batch_size", 16)
        args.setdefault("eval_mode", "single")
        args.setdefault("template", self.train_kwargs.get("template", ""))

        # 确保使用cl_tasks而不是dataset
        if "dataset" in args:
            args["cl_tasks"] = args.pop("dataset")
        elif "cl_tasks" not in args and "dataset" in self.train_kwargs:
            args["cl_tasks"] = self.train_kwargs["dataset"]

        # 移除旧的持续学习指标参数 (enable_transfer, enable_bwt, enable_fwt)
        # 新的 calculate_cl_metrics 参数会在 CLWorkflow 的 __init__ 中处理
        # CLEvaluationArguments 的 __post_init__ 会自动设置它
        args.pop("enable_transfer", None)
        args.pop("enable_bwt", None)
        args.pop("enable_fwt", None)

        # 移除不需要的参数 (注释或None值)
        for key in list(args.keys()):
            if key.startswith("#") or args[key] is None:
                args.pop(key)

        return f"easycl-cli cl_eval {self._dict_to_args(args)}"

    def generate_merge_command(self, task_id: int, task: str, task_output_dir: str) -> str:
        """为O-LoRA方法生成模型合并命令
        
        Args:
            task_id: 任务ID
            task: 任务名称
            task_output_dir: 当前任务的输出目录
            
        Returns:
            str: 模型合并命令
        """
        # Get base model path and template from train_kwargs
        base_model_path = self.train_kwargs.get("model_name_or_path")
        template = self.train_kwargs.get("template", "")
        
        # For subsequent tasks, use the merged model from previous task as base
        if task_id > 0:
            prev_task_dir_name = f"task_{task_id-1}_{self.tasks[task_id-1]}"
            prev_task_output_dir = os.path.join(self.train_kwargs["output_dir"], prev_task_dir_name)
            base_model_path = os.path.join(prev_task_output_dir, "merged_model")
        
        # Merged model will be saved in current task output directory
        export_dir = os.path.join(task_output_dir, "merged_model")
        
        # Construct merge command
        merge_args = {
            "model_name_or_path": base_model_path,
            "adapter_name_or_path": task_output_dir,
            "template": template,
            "finetuning_type": "lora",
            "export_dir": export_dir
        }
        
        # Convert to command string
        args_str = self._dict_to_args(merge_args)
        return f"llamafactory-cli export {args_str}"

class CLWorkflow:
    """持续学习工作流"""

    def __init__(
        self,
        workflow_args: CLWorkflowArguments,
        train_args: CLTrainArguments,
        eval_args: CLEvalArguments,
        **kwargs
    ):
        self.workflow_args = workflow_args
        self.train_args = train_args
        self.eval_args = eval_args
        self.current_task_id = None # Added to track current task within workflow
        self.is_benchmark_mode = False # Initialize benchmark mode flag

        # Load parameters: Prioritize benchmark args if provided
        if workflow_args.benchmark:
            if not workflow_args.benchmark_order or not workflow_args.benchmark_dir:
                raise ValueError("If 'benchmark' is specified, 'benchmark_order' and 'benchmark_dir' must also be provided.")
            logger.info("Benchmark mode activated.")
            self.is_benchmark_mode = True

            # Initialize handler (this also validates benchmark info)
            handler = BenchmarkHandler(
                benchmark_name=workflow_args.benchmark,
                benchmark_order=workflow_args.benchmark_order,
                benchmark_dir=workflow_args.benchmark_dir
            )

            # Load base train/eval args (can be empty if not provided)
            # _load_params should handle empty paths gracefully
            base_train_kwargs = self._load_params(train_args.train_params) if train_args.train_params else {}
            base_eval_kwargs = self._load_params(eval_args.eval_params) if eval_args.eval_params else {}

            # Get benchmark-specific configs, overriding base configs
            self.train_kwargs, self.eval_kwargs = handler.get_benchmark_configs(base_train_kwargs, base_eval_kwargs)
            self.tasks = handler.get_task_sequence() # Get tasks from benchmark handler

        else:
            # Standard mode: Load parameters from files
            logger.info("Standard workflow mode.")
            if self.workflow_args.mode != "eval_only":
                self.train_kwargs = self._load_params(train_args.train_params)
                if not self.train_kwargs:
                     raise ValueError("Training parameters file ('train_params') must be provided in non-eval_only mode.")
            else:
                self.train_kwargs = {} # No training in eval_only

            if self.workflow_args.mode != "train_only":
                self.eval_kwargs = self._load_params(eval_args.eval_params)
                if not self.eval_kwargs:
                     raise ValueError("Evaluation parameters file ('eval_params') must be provided in non-train_only mode.")
            else:
                self.eval_kwargs = {} # No evaluation in train_only

            # Parse tasks from loaded configs in standard mode
            if self.workflow_args.mode == "eval_only":
                self.tasks = self.eval_kwargs.get("cl_tasks", "").split(",")
                if not self.tasks or not self.tasks[0]:
                    raise ValueError("Evaluation config must contain 'cl_tasks' in eval_only mode.")
            elif self.train_kwargs:
                self.tasks = self.train_kwargs.get("dataset", "").split(",")
                if not self.tasks or not self.tasks[0]:
                    raise ValueError("Training config must contain 'dataset' in non-eval_only mode.")
            else:
                self.tasks = [] # Should not happen if validation passed

        # Normalize adapter paths after loading/overriding
        def _normalize_adapter_path(kwargs: Dict) -> None:
            if "adapter_name_or_path" in kwargs:
                val = kwargs["adapter_name_or_path"]
                if isinstance(val, list):
                    kwargs["adapter_name_or_path"] = ",".join(str(x).strip() for x in val)
        _normalize_adapter_path(self.train_kwargs)
        _normalize_adapter_path(self.eval_kwargs)

        # Validate combined/final parameters (happens after potential benchmark overrides)
        self._validate_params()

        # Set calculate_cl_metrics based on mode (after params are finalized)
        # Ensure eval_kwargs exists before modification
        if not self.eval_kwargs:
            self.eval_kwargs = {} # Initialize if empty (e.g., train_only mode)

        if self.workflow_args.mode == "full_workflow":
            if self.eval_kwargs.get("calculate_cl_metrics") is False:
                 logger.info_rank0("Workflow mode is 'full_workflow', but 'calculate_cl_metrics' was explicitly set to False in config. Overriding to True.")
            else:
                 logger.info_rank0("Workflow mode is 'full_workflow'. Automatically enabling CL metrics calculation.")
            self.eval_kwargs["calculate_cl_metrics"] = True
        else:
            # Only override to False if it was True, otherwise respect user's False or default (None)
            if self.eval_kwargs.get("calculate_cl_metrics") is True:
                 logger.info_rank0(f"Workflow mode is '{self.workflow_args.mode}', but 'calculate_cl_metrics' was explicitly set to True in config. Overriding to False as mode is not 'full_workflow'.")
                 self.eval_kwargs["calculate_cl_metrics"] = False
            elif "calculate_cl_metrics" not in self.eval_kwargs:
                 # Ensure it defaults to False if not specified and not full_workflow
                 self.eval_kwargs["calculate_cl_metrics"] = False
                 logger.info_rank0(f"Workflow mode is '{self.workflow_args.mode}'. Setting 'calculate_cl_metrics' to default False.")
        # End automatic setting

        # Task list is now set either by benchmark handler or standard config loading
        # Ensure eval 'cl_tasks' matches train 'dataset' IF NOT in benchmark mode
        if not self.is_benchmark_mode and self.train_kwargs and self.eval_kwargs:
            train_datasets = self.train_kwargs.get("dataset")
            eval_tasks = self.eval_kwargs.get("cl_tasks")
            if eval_tasks and eval_tasks != train_datasets:
                logger.warning(f"Eval config 'cl_tasks' ({eval_tasks}) differs from train 'dataset' ({train_datasets}). Using train 'dataset' for consistency.")
                self.eval_kwargs["cl_tasks"] = train_datasets
            elif not eval_tasks and train_datasets:
                self.eval_kwargs["cl_tasks"] = train_datasets # Set cl_tasks from train dataset

        # If cleaning dirs, do it after parameters are loaded and validated
        if self.workflow_args.clean_dirs and not self.workflow_args.previewonly:
            self._clean_directories()

        # Instantiate command generator with the final parameters
        self.command_generator = CLCommandGenerator(self.train_kwargs, self.eval_kwargs)

        # Setup evaluator and metrics calculator only if not previewing and if evaluation is involved
        if not self.workflow_args.previewonly and self.workflow_args.mode != "train_only":
            # Use train_kwargs as the primary source for model/data args unless in eval_only mode
            primary_args_source = self.eval_kwargs if self.workflow_args.mode == "eval_only" else self.train_kwargs

            # Check if essential keys exist before creating ModelArguments/DataArguments
            model_path_key = "model_name_or_path"
            dataset_dir_key = "dataset_dir"

            if model_path_key not in primary_args_source:
                # Try getting from eval_kwargs if eval_only, or raise error
                if self.workflow_args.mode == "eval_only" and model_path_key in self.eval_kwargs:
                    model_path = self.eval_kwargs[model_path_key]
                else:
                    raise ValueError(f"Missing required argument '{model_path_key}' in parameters for evaluation setup.")
            else:
                model_path = primary_args_source[model_path_key]

            # dataset_dir is more complex due to benchmark mode
            if self.is_benchmark_mode:
                dataset_dir = self.train_kwargs.get(dataset_dir_key) # Benchmark handler sets this in train_kwargs
            elif dataset_dir_key in primary_args_source:
                dataset_dir = primary_args_source[dataset_dir_key]
            else:
                # Fallback or default? Assume DataArguments can handle None or default
                dataset_dir = None
                #logger.warning(f"'{dataset_dir_key}' not found in primary parameters for evaluation setup. Using None.")

            model_args = ModelArguments(
                model_name_or_path=model_path
            )
            data_args = DataArguments(
                dataset_dir=dataset_dir
            )

            # Instantiate CLEvaluationArguments: Determine the initial task value.
            initial_task = self.eval_kwargs.get("task")
            if initial_task is None:
                if self.tasks:
                    initial_task = self.tasks[0]
                    logger.info_rank0(f"Evaluation parameter 'task' not found. Using the first task '{initial_task}' for initialization.")
                else:
                    raise ValueError("Evaluation requires at least one task. Please specify 'task' or 'cl_tasks' in parameters.")

            # Create CLEvaluationArguments instance
            cl_eval_args = CLEvaluationArguments(task=initial_task)

            # Populate remaining fields from the final eval_kwargs
            for key, value in self.eval_kwargs.items():
                if key == "task": continue # Skip task, already set
                if hasattr(cl_eval_args, key):
                    setattr(cl_eval_args, key, value)
                # else: logger.debug(f"Key '{key}' from final eval_kwargs not in CLEvaluationArguments.")

            # Ensure cl_tasks is explicitly set in cl_eval_args from self.tasks
            cl_eval_args.cl_tasks = ",".join(self.tasks)

            # Manually run post_init logic if needed
            if hasattr(cl_eval_args, "__post_init__"):
                try:
                    cl_eval_args.__post_init__()
                except Exception as e:
                    logger.error(f"Error running __post_init__ for CLEvaluationArguments: {e}")

            finetuning_args = FinetuningArguments()
            cl_finetuning_args = CLFinetuningArguments()

            # Populate args from combined configs (be careful with precedence)
            combined_args = {**self.train_kwargs, **self.eval_kwargs} # Eval potentially overrides train
            for key, value in combined_args.items():
                if hasattr(model_args, key) and getattr(model_args, key) is None:
                    setattr(model_args, key, value)
                if hasattr(data_args, key) and getattr(data_args, key) is None:
                    setattr(data_args, key, value)
                # cl_eval_args populated manually
                if hasattr(finetuning_args, key):
                    # Avoid overwriting defaults with None unless explicitly provided
                    if getattr(finetuning_args, key) is None or value is not None:
                         setattr(finetuning_args, key, value)
                if hasattr(cl_finetuning_args, key):
                     if getattr(cl_finetuning_args, key) is None or value is not None:
                         setattr(cl_finetuning_args, key, value)

            try:
                self.evaluator = CLEvaluator((model_args, data_args, cl_eval_args, finetuning_args, cl_finetuning_args))
                self.metrics_calculator = CLMetricsCalculator(self.tasks)
            except Exception as e:
                logger.error(f"Failed to initialize CLEvaluator or CLMetricsCalculator: {e}")
                logger.error("Evaluation and metrics calculation will be skipped.")
                self.evaluator = None
                self.metrics_calculator = None
        else:
            self.evaluator = None
            self.metrics_calculator = None

    def _validate_params(self):
        """验证参数的有效性 (Simplified: Removed default setting logic)"""
        logger.info_rank0("Validating parameters...")
        # Validate train parameters only if training is involved
        if self.workflow_args.mode != "eval_only":
            required_train_params = [
                "model_name_or_path",
                "dataset",
                "output_dir"
                # Removed specific training hparams like num_epochs, lr etc.
                # Assume they are correctly passed in the train_params file.
            ]
            missing_params = [p for p in required_train_params if p not in self.train_kwargs]
            if missing_params:
                raise ValueError(f"Train config missing required parameters: {', '.join(missing_params)}")

            # Check if LoRA params are present if LoRA is intended (e.g., for iLora)
            # This check might be better placed inside the specific method's logic if needed
            if self.train_kwargs.get("cl_method") == "ilora" or self.train_kwargs.get("finetuning_type") == "lora":
                required_lora_params = ["lora_rank", "lora_alpha", "lora_target"] # lora_dropout is optional
                missing_lora = [p for p in required_lora_params if p not in self.train_kwargs]
                if missing_lora:
                    raise ValueError(f"LoRA training requested but missing parameters: {', '.join(missing_lora)}")


            # === Reintroduce CL method detection based on use_* flags ===
            detected_cl_method = None
            enabled_methods = []
            config_path = os.path.join(os.path.dirname(__file__), "cl_params_config.json")
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                supported_methods = config.get("cl_methods_registry", {}).get("methods", [])

                for method in supported_methods:
                    use_flag = f"use_{method}"
                    if self.train_kwargs.get(use_flag, False):
                        enabled_methods.append(method)
                        if detected_cl_method is None: # Set the first detected method
                            detected_cl_method = method
                            logger.info_rank0(f"Detected enabled CL method flag '{use_flag}'. Setting cl_method='{method}'.")

                if len(enabled_methods) > 1:
                    logger.warning(
                        f"Multiple CL method flags enabled: {', '.join(enabled_methods)}. "
                        f"Using the first detected method: '{detected_cl_method}'."
                    )
                elif not detected_cl_method and "cl_method" not in self.train_kwargs:
                    logger.warning("No 'use_<method>' flag found true in train_kwargs, and 'cl_method' is not explicitly set.")

                # Set the detected cl_method in train_kwargs if found and not already explicitly set
                if detected_cl_method and "cl_method" not in self.train_kwargs:
                    self.train_kwargs["cl_method"] = detected_cl_method
                elif detected_cl_method and "cl_method" in self.train_kwargs and self.train_kwargs["cl_method"] != detected_cl_method:
                    logger.warning(f"Explicit 'cl_method' ('{self.train_kwargs['cl_method']}') differs from detected method based on 'use_{detected_cl_method}' flag. Using the explicit 'cl_method'.")

            except Exception as e:
                logger.error(f"Could not load or parse {config_path} to detect CL method: {e}")
                logger.warning("CL method detection skipped due to config loading error.")
            # === End of CL method detection ===

            # Check if a CL method is specified (either explicitly or detected) and if it's supported
            cl_method = self.train_kwargs.get("cl_method")
            supported_methods = [] # Define outside try block
            config_path = os.path.join(os.path.dirname(__file__), "cl_params_config.json")
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                supported_methods = config.get("cl_methods_registry", {}).get("methods", [])
            except Exception as e:
                 logger.error(f"Could not load or parse {config_path} to get supported methods: {e}")

            if cl_method:
                # Re-check support (might be redundant if detection worked, but safe)
                if supported_methods and cl_method not in supported_methods:
                    raise ValueError(f"Unsupported CL method specified or detected: '{cl_method}'. Supported methods from config: {supported_methods}")
                elif not supported_methods:
                     # Config loading failed earlier, can't verify
                     logger.warning(f"Cannot verify if CL method '{cl_method}' is supported due to earlier config loading issues.")
                else:
                    logger.info_rank0(f"Using CL method: '{cl_method}'.")
            # Removed the logic for checking multiple use_* flags and setting defaults


        # Validate eval parameters only if evaluation is involved
        if self.workflow_args.mode != "train_only":
            required_eval_params = ["save_dir"]
            if self.workflow_args.mode == "eval_only":
                # In eval_only, base model and tasks must be provided in eval config
                required_eval_params.extend(["model_name_or_path", "cl_tasks"])

            missing_params = [p for p in required_eval_params if p not in self.eval_kwargs]
            if missing_params:
                raise ValueError(f"Eval config missing required parameters: {', '.join(missing_params)}")

            # Perform checks previously in CLEvaluationArguments post_init if needed here
            # For example, checks related to eval_mode, multi_adapter_dir etc.
            # These are better handled when parsing CLEvaluationArguments


        logger.info_rank0("Parameter validation finished.")


    def _load_params(self, params_file: str) -> Dict:
        """加载参数文件 (Supports JSON and YAML)"""
        if not params_file:
            return {}

        file_ext = os.path.splitext(params_file)[1].lower()

        try:
            with open(params_file, "r", encoding="utf-8") as f:
                if file_ext == ".json":
                    return json.load(f)
                elif file_ext in [".yaml", ".yml"]:
                    if not YAML_AVAILABLE:
                         logger.error(f"Attempted to load YAML file '{params_file}' but PyYAML library is not installed. Please install it: pip install PyYAML")
                         raise ImportError("PyYAML library is required to load YAML configuration files.")
                    # Use safe_load to prevent arbitrary code execution
                    return yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported parameter file format: {file_ext}. Only .json, .yaml, .yml are supported.")
        except FileNotFoundError:
            logger.error(f"Parameter file not found: {params_file}")
            raise
        except (json.JSONDecodeError, yaml.YAMLError) as decode_error:
             logger.error(f"Error decoding {file_ext.upper()[1:]} from parameter file: {params_file}. Details: {decode_error}")
             raise
        except Exception as e:
            logger.error(f"Error loading parameter file {params_file}: {e}")
            raise

    def _get_commands(self) -> Tuple[List[str], List[str]]:
        """获取训练和评估命令列表 (Logic using command_generator remains the same)"""
        train_commands = []
        eval_commands = []

        # Generate train commands if needed
        if self.workflow_args.mode != "eval_only":
            cl_method = self.train_kwargs.get("cl_method")
            is_olora = (cl_method == "olora")
            
            for i, task in enumerate(self.tasks):
                # Update command generator's state if necessary (e.g., if it holds task-specific state)
                # Currently, command_generator seems stateless w.r.t tasks besides reading self.tasks
                train_cmd = self.command_generator.generate_train_command(i, task)
                train_commands.append(train_cmd)
                
                # For O-LoRA method, add merge command after each task (except if only one task)
                if is_olora and len(self.tasks) > 1:
                    task_output_dir = os.path.join(self.train_kwargs.get("output_dir", "outputs"), f"task_{i}_{task}")
                    merge_cmd = self.command_generator.generate_merge_command(i, task, task_output_dir)
                    train_commands.append(merge_cmd)

        # Generate eval commands if needed
        if self.workflow_args.mode != "train_only":
            # Determine if LoRA is used based on train_kwargs (even in eval_only mode, train_kwargs holds reference paths)
            # Check for finetuning_type or presence of lora args as indicators
            is_lora_potentially_used = (
                self.train_kwargs.get("finetuning_type") == "lora" or
                all(k in self.train_kwargs for k in ["lora_rank", "lora_target"]) or
                bool(self.train_kwargs.get("adapter_name_or_path")) # If adapter is explicitly given
            )
            base_model_path_for_eval = self.train_kwargs.get("model_name_or_path")

            # Special handling for O-LoRA evaluation
            cl_method = self.train_kwargs.get("cl_method")
            is_olora = (cl_method == "olora")

            # Evaluate base model if required by workflow mode
            if self.workflow_args.mode in ["full_workflow", "train_then_eval"]: # Evaluate base/initial state before task 0 training
                logger.info_rank0("Generating eval command for initial model state (before task 0)...")
                
                # Determine base model and adapter paths for the command
                # base_model_path_for_eval is already defined above from self.train_kwargs

                task_output_dir_for_command: Optional[str]
                base_model_path_for_command_arg: Optional[str] # Renamed to avoid conflict with outer scope base_model_path
                is_lora_for_command_gen: bool

                is_clmoe_base_eval = self.eval_kwargs.get("use_clmoe_eval", False)
                is_moelora_base_eval = self.eval_kwargs.get("use_moelora_eval", False) # New check

                if is_clmoe_base_eval or is_moelora_base_eval: # Combined condition
                    eval_method_name = "CL-MoE" if is_clmoe_base_eval else "MoE-LoRA"
                    logger.info_rank0(f"{eval_method_name} base evaluation detected. Adapter path will be taken from eval_kwargs.")
                    is_lora_for_command_gen = True # Both are adapter-based
                    task_output_dir_for_command = self.eval_kwargs.get("adapter_name_or_path") # This is the specific adapter path
                    base_model_path_for_command_arg = base_model_path_for_eval # This is the underlying base model
                    
                    if not task_output_dir_for_command:
                        # This error will ultimately be raised by CLEvalEvaluator, but good to log here.
                        logger.error( # Use standard logger.error for actual errors
                            f"{eval_method_name} base evaluation is enabled, but 'adapter_name_or_path' is missing in evaluation parameters. "
                            "Evaluation will likely fail."
                        )
                else:
                    # Standard logic for base model evaluation (could be full model or LoRA from train_kwargs)
                    adapter_from_train_kwargs = self.train_kwargs.get("adapter_name_or_path")
                    if adapter_from_train_kwargs:
                        # Base model is a LoRA model specified in train_kwargs
                        logger.info_rank0("Base model evaluation: Using adapter from train_kwargs.")
                        is_lora_for_command_gen = True
                        task_output_dir_for_command = adapter_from_train_kwargs
                        base_model_path_for_command_arg = base_model_path_for_eval
                    else:
                        # Base model is a full model (no adapter in train_kwargs)
                        logger.info_rank0("Base model evaluation: Using full model path from train_kwargs.")
                        is_lora_for_command_gen = False
                        task_output_dir_for_command = base_model_path_for_eval # Evaluating the base model directly
                        base_model_path_for_command_arg = None # No separate base model when evaluating a full model path

                base_eval_cmd = self.command_generator.generate_eval_command(
                    task_output_dir=task_output_dir_for_command,
                    task_id=None, 
                    is_lora=is_lora_for_command_gen,
                    base_model_path=base_model_path_for_command_arg
                )
                eval_commands.append(base_eval_cmd)

            # Evaluate after each task if required (not eval_only mode)
            if self.workflow_args.mode != "eval_only":
                for i, task in enumerate(self.tasks):
                    # For O-LoRA, use merged model path instead of adapter path
                    if is_olora:
                        # O-LoRA uses merged models for evaluation
                        task_output_dir = os.path.join(self.train_kwargs.get("output_dir", "outputs"), f"task_{i}_{task}", "merged_model")
                        is_task_output_lora = False  # Merged model is a full model, not LoRA
                        base_model_path_for_task_eval = None  # No separate base model needed
                    else:
                        # Standard path for other methods
                        task_output_dir = os.path.join(self.train_kwargs.get("output_dir", "outputs"), f"task_{i}_{task}")
                        # Determine if the output of this task is LoRA adapters or full model
                        # Assume LoRA if LoRA was potentially used during training phase
                        is_task_output_lora = is_lora_potentially_used
                        base_model_path_for_task_eval = base_model_path_for_eval if is_task_output_lora else None

                    eval_cmd = self.command_generator.generate_eval_command(
                        task_output_dir=task_output_dir, # Path to the adapter (if LoRA) or fine-tuned model
                        task_id=i,
                        is_lora=is_task_output_lora,
                        base_model_path=base_model_path_for_task_eval
                    )
                    eval_commands.append(eval_cmd)
            elif self.workflow_args.mode == "eval_only":
                # In eval_only, we evaluate the specified model/adapter on all cl_tasks
                logger.info_rank0("Generating eval command for the specified model/adapter in eval_only mode...")
                model_or_adapter_to_eval = self.eval_kwargs.get("adapter_name_or_path", self.eval_kwargs.get("model_name_or_path"))
                is_evaluating_lora = bool(self.eval_kwargs.get("adapter_name_or_path"))
                base_model_ref = self.eval_kwargs.get("model_name_or_path")

                eval_cmd = self.command_generator.generate_eval_command(
                    task_output_dir=model_or_adapter_to_eval,
                    task_id=None, # Evaluating a single checkpoint across tasks
                    is_lora=is_evaluating_lora,
                    base_model_path=base_model_ref if is_evaluating_lora else None
                )
                eval_commands.append(eval_cmd)


        return train_commands, eval_commands


    def preview_commands(self):
        """预览将要执行的命令 (No changes needed)"""
        # Use a temporary flag to prevent _get_commands from logging preview messages internally if called again by run()
        original_preview_flag = self.workflow_args.previewonly
        self.workflow_args.previewonly = True # Temporarily set flag for _get_commands

        train_commands, eval_commands = self._get_commands()

        self.workflow_args.previewonly = original_preview_flag # Restore flag

        if train_commands:
            print("\n--- Training Commands Preview ---")
            cl_method = self.train_kwargs.get("cl_method")
            is_olora = (cl_method == "olora")
            
            if is_olora and len(self.tasks) > 1:
                # For O-LoRA, show interleaved training and merge commands
                task_count = 0
                for i, cmd in enumerate(train_commands):
                    if "easycl-cli cl_train" in cmd:
                        task_count += 1
                        print(f"Task {task_count-1} Training: {cmd}\n")
                    elif "llamafactory-cli export" in cmd:
                        print(f"Task {task_count-1} Model Merge: {cmd}\n")
                    else:
                        print(f"Command {i+1}: {cmd}\n")
            else:
                # Standard preview for non-O-LoRA methods
                for i, cmd in enumerate(train_commands, 1):
                    print(f"Task {i-1}: {cmd}\n")

        if eval_commands:
            print("\n--- Evaluation Commands Preview ---")
            for i, cmd in enumerate(eval_commands):
                # Determine if it's base eval or task eval based on save_dir in command?
                # Or just print sequentially
                print(f"Eval Step {i}: {cmd}\n")

    def run(self):
        """运行持续学习工作流"""
        # Preview commands first, regardless of mode
        self.preview_commands()

        if self.workflow_args.previewonly:
            logger.info_rank0("Preview only mode. Exiting without execution.")
            return

        logger.info_rank0("Starting command execution...")

        # Execute workflow based on mode
        try:
            if self.workflow_args.mode == "train_only":
                self._run_train_only()
            elif self.workflow_args.mode == "eval_only":
                self._run_eval_only()
            elif self.workflow_args.mode == "train_then_eval":
                self._run_train_then_eval()
            elif self.workflow_args.mode == "full_workflow":
                # Full workflow currently implies train_then_eval logic
                # If different behavior is needed (e.g., eval after each train step), implement here
                self._run_train_then_eval() # Use train_then_eval logic for now
            else:
                raise ValueError(f"Unsupported workflow mode: {self.workflow_args.mode}")

            # Calculate CL metrics if evaluation was performed and requested
            if self.workflow_args.mode != "train_only" and self.eval_kwargs.get("calculate_cl_metrics", False):
                if self.metrics_calculator:
                    logger.info_rank0("Calculating CL metrics...")
                    metrics = self._calculate_cl_metrics()
                    if metrics:
                        logger.info_rank0("CL metrics calculation complete.")
                    else:
                        logger.warning("CL metrics calculation failed or produced no results.")
                else:
                    logger.warning("Metrics calculator not available. Skipping CL metrics calculation.")

        except Exception as e:
            logger.exception(f"An error occurred during workflow execution: {e}")
            # Consider adding cleanup or state saving here if needed upon failure


    def _run_command(self, cmd: str, step_info: str) -> None:
        """Helper to run a command and log/raise errors."""
        logger.info_rank0(f"{step_info}: Executing command: {cmd}")
        exit_code = os.system(cmd)
        if exit_code != 0:
            raise RuntimeError(f"{step_info}: Command failed with exit code {exit_code}: {cmd}")
        logger.info_rank0(f"{step_info}: Command finished successfully.")


    def _run_train_only(self):
        """运行训练模式"""
        train_commands, _ = self._get_commands()
        if train_commands:
            cl_method = self.train_kwargs.get("cl_method")
            is_olora = (cl_method == "olora")
            
            if is_olora and len(self.tasks) > 1:
                # For O-LoRA, train_commands contains interleaved training and merge commands
                # Pattern: [train_task0, merge_task0, train_task1, merge_task1, ...]
                task_count = 0
                for i, cmd in enumerate(train_commands):
                    if "easycl-cli cl_train" in cmd:
                        # This is a training command
                        task_count += 1
                        self._run_command(cmd, f"Training Task {task_count}/{len(self.tasks)}")
                    elif "llamafactory-cli export" in cmd:
                        # This is a merge command
                        self._run_command(cmd, f"Merging Task {task_count} Model")
                    else:
                        # Fallback for unexpected command types
                        self._run_command(cmd, f"Command {i+1}/{len(train_commands)}")
            else:
                # Standard execution for non-O-LoRA methods
                for i, cmd in enumerate(train_commands):
                    self._run_command(cmd, f"Training Task {i+1}/{len(self.tasks)}")
        else:
            logger.info_rank0("No training commands generated for train_only mode.")

    def _run_eval_only(self):
        """运行评估模式"""
        # 修改解包顺序：忽略训练命令，获取评估命令列表
        _, eval_commands = self._get_commands() # Ignore train commands
        if eval_commands:
            for i, cmd in enumerate(eval_commands):
                # Should typically only be one command in standard eval_only
                self._run_command(cmd, f"Evaluation Step {i+1}/{len(eval_commands)}")
        else:
            logger.info_rank0("No evaluation commands generated for eval_only mode.")
        # CL Metrics calculation happens in run()

            # 计算持续学习指标
            #self._calculate_cl_metrics()

    def _run_train_then_eval(self):
        """运行训练后评估模式"""
        train_commands, eval_commands = self._get_commands()

        # Execute training first
        if train_commands:
            cl_method = self.train_kwargs.get("cl_method")
            is_olora = (cl_method == "olora")
            
            if is_olora and len(self.tasks) > 1:
                # For O-LoRA, train_commands contains interleaved training and merge commands
                # Pattern: [train_task0, merge_task0, train_task1, merge_task1, ...]
                task_count = 0
                for i, cmd in enumerate(train_commands):
                    if "easycl-cli cl_train" in cmd:
                        # This is a training command
                        task_count += 1
                        self._run_command(cmd, f"Training Task {task_count}/{len(self.tasks)}")
                    elif "llamafactory-cli export" in cmd:
                        # This is a merge command
                        self._run_command(cmd, f"Merging Task {task_count} Model")
                    else:
                        # Fallback for unexpected command types
                        self._run_command(cmd, f"Command {i+1}/{len(train_commands)}")
            else:
                # Standard execution for non-O-LoRA methods
                for i, cmd in enumerate(train_commands):
                    self._run_command(cmd, f"Training Task {i+1}/{len(self.tasks)}")
        else:
            logger.info_rank0("No training commands generated for train_then_eval mode.")

        # Execute evaluation after all training
        if eval_commands:
            logger.info_rank0("Starting evaluation phase after training...")
            for i, cmd in enumerate(eval_commands):
                # Identify if it's base eval or task eval for logging clarity
                log_prefix = "Base Model Evaluation" if "base" in cmd else f"Task {i-1} Model Evaluation" if i>0 else "Evaluation Step" # Basic heuristic
                self._run_command(cmd, f"{log_prefix} ({i+1}/{len(eval_commands)})")
        else:
            logger.info_rank0("No evaluation commands generated for train_then_eval mode.")
        # CL Metrics calculation happens in run()

    def _run_full_workflow(self):
        """运行完整工作流 (Currently same as train_then_eval)"""
        # If "full_workflow" should mean evaluate after *each* task, the logic needs modification.
        # For now, treat it as train all, then evaluate all.
        logger.info_rank0("Running full workflow (Train All -> Eval All)...")
        self._run_train_then_eval()
        # CL Metrics calculation happens in run()

    def _calculate_cl_metrics(self):
        """计算持续学习指标 (Minor adjustments for clarity)"""
        if not self.evaluator or not self.metrics_calculator:
            logger.error("Evaluator or Metrics Calculator not initialized. Skipping CL metrics.")
            return None

        eval_base_dir = self.eval_kwargs.get("save_dir", "eval_results")
        # Base model results directory assumes a specific naming convention
        base_dir = os.path.join(eval_base_dir, "base")
        # Task results directories also assume a naming convention
        task_dirs = [os.path.join(eval_base_dir, f"task_{i}") for i in range(len(self.tasks))]

        # Ensure metrics_calculator has the correct tasks list
        if not hasattr(self.metrics_calculator, 'tasks') or not self.metrics_calculator.tasks:
            logger.warning("Metrics calculator tasks list missing. Setting from workflow.")
            self.metrics_calculator.tasks = self.tasks
        elif self.metrics_calculator.tasks != self.tasks:
            logger.warning("Metrics calculator tasks mismatch workflow tasks. Updating to workflow tasks.")
            self.metrics_calculator.tasks = self.tasks

        logger.info_rank0(f"Calculating CL metrics using results from:")
        logger.info_rank0(f"  Base results directory: {base_dir}")
        logger.info_rank0(f"  Task results directories: {task_dirs}")

        # Check if result directories exist
        if not os.path.exists(base_dir):
            logger.error(f"Base results directory not found: {base_dir}. Cannot calculate CL metrics.")
            return None
        missing_task_dirs = [d for d in task_dirs if not os.path.exists(d)]
        if missing_task_dirs:
            logger.error(f"Missing task results directories: {missing_task_dirs}. Cannot calculate CL metrics.")
            return None

        try:
            # Ensure calculate method exists and call it
            if hasattr(self.metrics_calculator, 'calculate'):
                metrics = self.metrics_calculator.calculate(
                    base_dir,
                    task_dirs
                )
                if metrics and isinstance(metrics, dict) and "error" not in metrics:
                    self._save_cl_metrics(metrics)
                    logger.info_rank0(f"CL metrics calculated and saved to: {os.path.join(eval_base_dir, 'cl_metrics.json')}")
                    return metrics
                elif isinstance(metrics, dict) and "error" in metrics:
                    logger.error(f"CL metrics calculation failed: {metrics.get('error', 'Unknown error')}")
                    return None
                else:
                    logger.error(f"CL metrics calculation returned unexpected result: {metrics}")
                    return None
            else:
                logger.error("Metrics calculator does not have a 'calculate' method.")
                return None
        except Exception as e:
            logger.exception(f"Error during CL metrics calculation: {e}")
            return None

    def _save_cl_metrics(self, metrics: Dict[str, Any]):
        """保存持续学习指标和详细结果 (No changes needed)"""
        eval_base_dir = self.eval_kwargs.get("save_dir", "eval_results")
        save_path = os.path.join(eval_base_dir, "cl_metrics.json")
        try:
            os.makedirs(eval_base_dir, exist_ok=True) # Ensure directory exists
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            logger.info_rank0(f"CL Metrics successfully saved to {save_path}")
            self._log_cl_metrics_summary(metrics, save_path) # Log summary after saving
        except Exception as e:
            logger.error(f"Failed to save CL metrics to {save_path}: {e}")


    def _log_cl_metrics_summary(self, metrics: Dict[str, Any], save_path: str):
        """Logs a formatted summary of the CL metrics."""
        logger.info_rank0("\n=== Continual Learning Evaluation Summary ===")

        # Task sequence and count
        logger.info_rank0(f"Task Sequence: {metrics.get('task_sequence', 'N/A')}")
        logger.info_rank0(f"Number of Tasks (N): {metrics.get('num_tasks', 'N/A')}")

        # Base model performance
        logger.info_rank0("\n--- Base Model Performance (R_0,i) ---")
        base_results = metrics.get("base_model_results", {})
        if base_results:
            for task, result in base_results.items():
                acc = result.get('accuracy', 'N/A')
                perf_str = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
                correct = result.get('correct', 'N/A')
                total = result.get('total', 'N/A')
                missing = " (Missing Data)" if result.get('missing') else ""
                logger.info_rank0(f"  Task '{task}': Accuracy = {perf_str}{missing} ({correct}/{total})")
        else:
            logger.info_rank0("  Base model results not available.")

        # Task result matrix
        logger.info_rank0("\n--- Task Evaluation Matrix (R_k,i) ---")
        matrix_results = metrics.get("task_results_matrix", {})
        if matrix_results:
            for k_str, task_evals in matrix_results.items(): # k_str = "after_task_k"
                task_k_index = k_str.split('_')[-1] # Extract k index
                logger.info_rank0(f"After Task {task_k_index} Training:")
                for task, result in task_evals.items():
                    acc = result.get('accuracy', 'N/A')
                    perf_str = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
                    correct = result.get('correct', 'N/A')
                    total = result.get('total', 'N/A')
                    missing = " (Missing Data)" if result.get('missing') else ""
                    logger.info_rank0(f"  -> Eval on Task '{task}': Accuracy = {perf_str}{missing} ({correct}/{total})")
        else:
            logger.info_rank0("  Task results matrix not available.")

        # CL Metrics Summary
        logger.info_rank0("\n--- Continual Learning Metrics ---")
        metrics_summary = metrics.get("metrics", {}).get("summary", {})
        if metrics_summary:
            last_acc = metrics_summary.get('last', 'N/A')
            avg_acc = metrics_summary.get('avg', 'N/A')
            bwt = metrics_summary.get('bwt', 'N/A')
            fwt = metrics_summary.get('fwt', 'N/A')
            logger.info_rank0(f"  Last Accuracy (Avg Acc on Last Task): {last_acc:.4f}" if isinstance(last_acc, float) else f"  Last Accuracy: {last_acc}")
            logger.info_rank0(f"  Average Accuracy (Avg Acc across all tasks after final training): {avg_acc:.4f}" if isinstance(avg_acc, float) else f"  Average Accuracy: {avg_acc}")
            logger.info_rank0(f"  Backward Transfer (BWT): {bwt:.4f}" if isinstance(bwt, float) else f"  Backward Transfer (BWT): {bwt}")
            logger.info_rank0(f"  Forward Transfer (FWT): {fwt:.4f}" if isinstance(fwt, float) else f"  Forward Transfer (FWT): {fwt}")
            if metrics.get('notes'):
                logger.info_rank0("  Notes:")
                for note in metrics['notes']:
                    logger.info_rank0(f"    - {note}")
        else:
            logger.info_rank0("  Metrics summary not available.")

        logger.info_rank0(f"\nDetailed results saved to: {save_path}")
        logger.info_rank0("==========================================")

    def _clean_directories(self):
        """清理输出和评估目录 (No changes needed)"""
        import shutil

        logger.info_rank0("Starting directory cleanup...")
        cleaned_dirs = []

        # Clean training output directory if training is involved
        if self.workflow_args.mode not in ["eval_only"]:
            output_dir = self.train_kwargs.get("output_dir")
            if output_dir and os.path.exists(output_dir):
                try:
                    logger.info_rank0(f"Attempting to clean training output directory: {output_dir}")
                    shutil.rmtree(output_dir)
                    logger.info_rank0(f"Successfully cleaned training output directory: {output_dir}")
                    cleaned_dirs.append(output_dir)
                except Exception as e:
                    logger.error(f"Error cleaning training output directory '{output_dir}': {e}")
            elif output_dir:
                logger.info_rank0(f"Training output directory '{output_dir}' not found, skipping cleanup.")
            else:
                logger.warning("Training output directory not configured, skipping cleanup.")

        # Clean evaluation results directory if evaluation is involved
        if self.workflow_args.mode not in ["train_only"]:
            save_dir = self.eval_args.save_dir # Use eval_args which should be correctly set
            if save_dir and os.path.exists(save_dir):
                try:
                    logger.info_rank0(f"Attempting to clean evaluation results directory: {save_dir}")
                    shutil.rmtree(save_dir)
                    logger.info_rank0(f"Successfully cleaned evaluation results directory: {save_dir}")
                    cleaned_dirs.append(save_dir)
                except Exception as e:
                    logger.error(f"Error cleaning evaluation results directory '{save_dir}': {e}")
            elif save_dir:
                logger.info_rank0(f"Evaluation results directory '{save_dir}' not found, skipping cleanup.")
            else:
                logger.warning("Evaluation results directory not configured, skipping cleanup.")

        if not cleaned_dirs:
            logger.info_rank0("No directories were cleaned.")
        logger.info_rank0("Directory cleanup finished.")

# CLTrainer class remains the same as it doesn't deal with config directly
class CLTrainer:
    """持续学习训练器"""

    def __init__(self, base_dir: str):
        """初始化持续学习训练器

        Args:
            base_dir: 基础输出目录
        """
        self.base_dir = base_dir
        # Runtime state might be less useful now config drives parameters
        # self.runtime_state_file = os.path.join(base_dir, "cl_runtime_state.json")

    def train_task(self, task: str, task_id: int, args: Dict[str, Any]) -> str:
        """训练单个任务 (Internally calls run_exp)

        Args:
            task: 任务名称
            task_id: 任务ID
            args: 训练参数 (already prepared by CLCommandGenerator logic)

        Returns:
            str: 任务输出目录 (as determined by args['output_dir'])
        """
        task_output_dir = args["output_dir"] # Get output dir from prepared args
        os.makedirs(task_output_dir, exist_ok=True)

        logger.info_rank0(f"Starting training for task {task_id} ({task}) with output to {task_output_dir}")

        # Directly call run_exp with the fully prepared arguments
        # run_exp likely parses these dictionary args internally
        try:
            # Assuming run_exp can accept a dictionary of arguments
            # If run_exp expects sys.argv style, we need to convert `args` back to that format
            # For now, assume run_exp(args_dict) works
            run_exp(args) # Pass the dictionary directly
            logger.info_rank0(f"Finished training for task {task_id} ({task}).")
        except Exception as e:
            logger.exception(f"Error during training execution (run_exp) for task {task_id} ({task}): {e}")
            raise # Re-raise the exception to halt the workflow

        # Saving runtime state might be less critical if config drives everything
        # self._save_runtime_state(...)

        return task_output_dir

    # _save_runtime_state and _load_runtime_state can be kept or removed if state management changes

# main function remains largely the same, parsing args and initiating the workflow
def main():
    """主函数"""
    parser = HfArgumentParser((
        CLWorkflowArguments,
        CLTrainArguments, # Specifies the input training config file
        CLEvalArguments   # Specifies the input evaluation config file
    ))

    # Check if a single JSON config file is provided for the whole workflow
    # This mode might need adjustment if train/eval params are always separate
    # Assuming standard CLI parsing for now
    workflow_args, train_args, eval_args = parser.parse_args_into_dataclasses()

    # Validate essential inputs based on mode
    if workflow_args.mode != "eval_only" and not train_args.train_params:
        raise ValueError(f"Mode '{workflow_args.mode}' requires a training parameter file specified via --train_params")
    if workflow_args.mode != "train_only" and not eval_args.eval_params:
        raise ValueError(f"Mode '{workflow_args.mode}' requires an evaluation parameter file specified via --eval_params")


    # Create and run the workflow
    try:
        workflow = CLWorkflow(workflow_args, train_args, eval_args)
        workflow.run() # run() now handles preview internally
    except ValueError as ve:
        logger.error(f"Configuration Error: {ve}")
        sys.exit(1)
    except RuntimeError as rte:
        logger.error(f"Runtime Error during workflow execution: {rte}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
