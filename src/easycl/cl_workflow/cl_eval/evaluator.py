"""持续学习评估器模块，用于自定义数据集的评估"""

import json
import os
import torch
import copy
import gc
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple, Union
from tqdm import tqdm
from transformers import HfArgumentParser

from llamafactory.eval.evaluator import Evaluator as BaseEvaluator
from llamafactory.hparams import GeneratingArguments
from llamafactory.model import load_model, load_tokenizer
from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.extras.misc import get_device_count
from llamafactory.extras.packages import is_vllm_available
from llamafactory.extras.constants import IGNORE_INDEX
from .adapters import (
    AlpacaEvalAdapter, 
    CustomDatasetAdapter
)
from easycl.hparams.parser import get_cl_eval_args
from easycl.cl.moe.moelora_loader import load_moelora_model
from easycl.cl.clmoe.clmoe_loader import load_clmoe_model
from llamafactory.extras.logging import get_logger # Import logger

logger = get_logger(__name__) # Initialize logger for this module

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

class CLEvalEvaluator(BaseEvaluator):
    """持续学习专用的评估器类"""
    
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        """初始化持续学习评估器"""
        self.model_args, self.data_args, self.eval_args, self.finetuning_args, self.cl_finetuning_args = get_cl_eval_args(args)
        # Parse GeneratingArguments separately from the input args dict
        parser = HfArgumentParser(GeneratingArguments)
        # parse_dict returns only the dataclass instance when parsing a single type
        self.generating_args = parser.parse_dict(args, allow_extra_keys=True)[0]
        
        self.tokenizer = load_tokenizer(self.model_args)["tokenizer"]
        # Set default padding side to left for generation consistency
        self.tokenizer.padding_side = "left" 
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        
        # Record the initial adapter path intended for single-adapter mode
        self.initial_adapter_path = self.model_args.adapter_name_or_path
        
        # 多adapter模式初始化
        self.using_multi_adapter = getattr(self.eval_args, "eval_mode", "") == "multi_adapter"
        
        # 确保multi_adapter_dir属性存在
        if self.using_multi_adapter and not hasattr(self.eval_args, "multi_adapter_dir"):
            # 尝试从args字典中获取multi_adapter_dir
            if isinstance(args, dict) and "multi_adapter_dir" in args:
                self.eval_args.multi_adapter_dir = args["multi_adapter_dir"]
                print(f"从参数字典中获取multi_adapter_dir: {self.eval_args.multi_adapter_dir}")
            else:
                # 设置默认值
                self.eval_args.multi_adapter_dir = "./multi_adapter_config"
                print(f"警告: 未找到multi_adapter_dir参数，使用默认值: {self.eval_args.multi_adapter_dir}")
        
        # vLLM相关初始化
        self.use_vllm = getattr(self.eval_args, "cleval_use_vllm", False)
        
        # 确保多adapter模式下不使用vLLM
        if self.using_multi_adapter and self.use_vllm:
            print("警告: 多adapter模式下不支持使用vLLM，已自动禁用vLLM")
            self.use_vllm = False
        
        if self.use_vllm:
            if not is_vllm_available():
                print("vLLM不可用，回退到标准推理")
                self.use_vllm = False
            else:
                print("初始化vLLM推理引擎")
                self._init_vllm_engine()
                print("vLLM推理引擎初始化完成")
        
        # 在非多adapter模式或需要基础模型进行推理时加载模型
        if not self.using_multi_adapter:
            # Check if cl-MoE evaluation is enabled for single adapter mode
            if getattr(self.eval_args, "use_clmoe_eval", False):
                print("信息: 检测到启用 cl-MoE 评估 (单Adapter模式)")
                if not self.model_args.adapter_name_or_path:
                     raise ValueError("错误: 启用 cl-MoE 评估，但未提供 adapter_name_or_path")
                # Ensure finetuning_type is 'lora' for cl-MoE
                if self.finetuning_args.finetuning_type != 'lora':
                     print("警告: 启用 cl-MoE 评估，但 finetuning_type 不是 'lora'，将强制设置为 'lora'")
                     self.finetuning_args.finetuning_type = 'lora'
                     # Ensure necessary cl-MoE args are present in finetuning_args if needed
                     if not hasattr(self.cl_finetuning_args, 'expert_num'):
                         # Attempt to fetch from args dict if available, else raise error or set default
                         if isinstance(args, dict) and 'expert_num' in args:
                              self.cl_finetuning_args.expert_num = args['expert_num']
                              print(f"信息: 从参数字典中获取 expert_num: {self.cl_finetuning_args.expert_num}")
                         else:
                              # Default or raise error, depending on expected behavior
                              self.cl_finetuning_args.expert_num = 2 # Example default
                              print(f"警告: 未找到 expert_num 参数，cl-MoE 加载器可能需要它。使用默认值: {self.cl_finetuning_args.expert_num}")
                         # Similarly for task_embedding_dim if needed by loader

                print(f"信息: 正在使用 load_clmoe_model 加载模型及 cl-MoE adapter: {self.model_args.adapter_name_or_path}")
                self.model = load_clmoe_model( # Use cl-MoE loader
                    tokenizer=self.tokenizer,
                    model_args=self.model_args,
                    finetuning_args=self.finetuning_args,
                    cl_finetuning_args=self.cl_finetuning_args,
                    is_trainable=False, # Evaluation mode
                    add_valuehead=False # No value head needed for evaluation
                )
                print("信息: cl-MoE 模型加载完成")
                self.current_adapter_path = self.model_args.adapter_name_or_path # Record the loaded adapter

            # Check if MoE-LoRA evaluation is enabled (only if cl-MoE is not)
            elif getattr(self.eval_args, "use_moelora_eval", False):
                print("信息: 检测到启用MoE-LoRA评估 (单Adapter模式)")
                # Directly use load_moelora_model
                if not self.model_args.adapter_name_or_path:
                     raise ValueError("错误: 启用MoE-LoRA评估，但未提供adapter_name_or_path")
                # Ensure finetuning_type is 'lora' for MoE-LoRA
                if self.finetuning_args.finetuning_type != 'lora':
                     print("警告: 启用MoE-LoRA评估，但finetuning_type不是'lora'，将强制设置为'lora'")
                     self.finetuning_args.finetuning_type = 'lora'
                     
                print(f"信息: 正在使用 load_moelora_model 加载模型及MoE-LoRA adapter: {self.model_args.adapter_name_or_path}")
                self.model = load_moelora_model(
                    tokenizer=self.tokenizer,
                    model_args=self.model_args,
                    finetuning_args=self.finetuning_args,
                    cl_finetuning_args=self.cl_finetuning_args,
                    is_trainable=False, # Evaluation mode
                    add_valuehead=False # No value head needed for evaluation
                )
                print("信息: MoE-LoRA模型加载完成")
                self.current_adapter_path = self.model_args.adapter_name_or_path # Record the loaded adapter
            else:
                # Standard single adapter mode (or base model loading if adapter path is None)
                print("信息: 标准单Adapter/基础模型模式评估")
                # Load model with or without adapter based on initial model_args.adapter_name_or_path
                self.model = load_model(self.tokenizer, self.model_args, self.finetuning_args)
                self.current_adapter_path = self.initial_adapter_path # Use initial path here
        elif self.using_multi_adapter:
            # 对于多adapter模式，先加载基础模型，不加载adapter
            self.original_adapter_path = self.model_args.adapter_name_or_path
            self.model_args.adapter_name_or_path = None
            self.model = load_model(self.tokenizer, self.model_args, self.finetuning_args)
            
            # 加载adapter配置
            self._load_multi_adapter_config()
        
        # 根据数据集类型选择适配器
        self.adapter = self._get_adapter()
        
        # 加载数据集选项配置
        self.dataset_options = self._load_dataset_options()
        
        # 初始化当前加载的adapter路径
        self.current_adapter_path = None

        # 预加载 dataset_info.json (如果存在)
        self.dataset_info = None
        task_dir = getattr(self.eval_args, "task_dir", "./data")
        dataset_info_path = os.path.join(task_dir, "dataset_info.json")
        fallback_info_path = os.path.join("./data", "dataset_info.json")

        # 优先检查 task_dir
        if os.path.exists(dataset_info_path):
            path_to_load = dataset_info_path
        # 如果 task_dir 不是 ./data 且 task_dir 中没找到，检查 ./data
        elif os.path.abspath(task_dir) != os.path.abspath("./data") and os.path.exists(fallback_info_path):
            path_to_load = fallback_info_path
            logger.warning(f"dataset_info.json not found in task_dir '{task_dir}', using fallback: {path_to_load}") # English log
        else:
            path_to_load = None
            # 在 evaluate_custom_dataset 中会处理找不到的情况，这里只记录日志
            if not os.path.exists(dataset_info_path):
                logger.warning(f"dataset_info.json not found in task directory: {dataset_info_path}") # English log
            if os.path.abspath(task_dir) != os.path.abspath("./data") and not os.path.exists(fallback_info_path):
                logger.warning(f"dataset_info.json also not found in fallback directory: {fallback_info_path}") # English log

        if path_to_load:
            try:
                with open(path_to_load, "r", encoding="utf-8") as f:
                    self.dataset_info = json.load(f)
                logger.info(f"Successfully preloaded dataset_info.json from {path_to_load}") # English log
            except Exception as e:
                logger.error(f"Failed to load or parse dataset_info.json from {path_to_load}: {e}") # English log
                # Keep self.dataset_info as None, error will be handled later if needed
    
    def _init_vllm_engine(self) -> None:
        """初始化vLLM推理引擎"""
        # 获取或设置默认值
        pipeline_parallel_size = getattr(self.eval_args, "vllm_pipeline_parallel_size", 1)
        image_max_pixels = getattr(self.eval_args, "vllm_image_max_pixels", 768 * 768)
        image_min_pixels = getattr(self.eval_args, "vllm_image_min_pixels", 32 * 32)
        
        if pipeline_parallel_size > get_device_count():
            raise ValueError("Pipeline parallel size应小于GPU数量")
        
        # 设置vLLM引擎参数
        engine_args = {
            "model": self.model_args.model_name_or_path,
            "trust_remote_code": True,
            "dtype": self.model_args.infer_dtype,
            "tensor_parallel_size": (get_device_count() // pipeline_parallel_size) or 1,
            "pipeline_parallel_size": pipeline_parallel_size,
            "disable_log_stats": True,
            "enable_lora": self.model_args.adapter_name_or_path is not None,
        }
        
        # 添加多模态支持
        if self.template.mm_plugin.__class__.__name__ != "BasePlugin":
            engine_args["limit_mm_per_prompt"] = {"image": 4, "video": 2}
        
        # 更新vLLM配置
        if hasattr(self.model_args, "vllm_config") and isinstance(self.model_args.vllm_config, dict):
            engine_args.update(self.model_args.vllm_config)
        
        # 创建vLLM引擎
        self.vllm_engine = LLM(**engine_args)
        
        # 设置采样参数
        self.sampling_params = SamplingParams(
            repetition_penalty=getattr(self.generating_args, "repetition_penalty", 1.0),
            temperature=getattr(self.generating_args, "temperature", 0.0),
            top_p=getattr(self.generating_args, "top_p", 1.0),
            top_k=getattr(self.generating_args, "top_k", -1),
            max_tokens=self.generating_args.max_new_tokens,
            stop_token_ids=self.template.get_stop_token_ids(self.tokenizer),
            skip_special_tokens=False,
        )
        
        # 保存图像处理参数
        self.vllm_image_max_pixels = image_max_pixels
        self.vllm_image_min_pixels = image_min_pixels
        
        # 如果使用LoRA，创建LoRA请求
        if self.model_args.adapter_name_or_path is not None:
            self.lora_request = LoRARequest("default", 1, self.model_args.adapter_name_or_path[0])
        else:
            self.lora_request = None

    def _load_multi_adapter_config(self) -> None:
        """加载多adapter配置文件"""
        # 获取配置文件路径
        multi_adapter_dir = self.eval_args.multi_adapter_dir
        config_path = os.path.join(multi_adapter_dir, "multiadapter_selected_config.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"多Adapter配置文件未找到: {config_path}")
        
        # 加载配置文件
        with open(config_path, "r", encoding="utf-8") as f:
            self.adapter_config = json.load(f)
        
        # 验证配置文件格式
        if "task_name" not in self.adapter_config or "adapters" not in self.adapter_config:
            raise ValueError(f"配置文件格式无效: {config_path}")
        
        # 验证任务名称是否匹配
        if self.adapter_config["task_name"] != self.eval_args.task:
            print(f"警告: 配置文件中的任务名称 ({self.adapter_config['task_name']}) 与当前评估任务 ({self.eval_args.task}) 不一致")
        
        # 创建adapter路径映射 (使用相对路径转换为绝对路径)
        self.adapter_paths = {}
        for adapter_name, adapter_info in self.adapter_config["adapters"].items():
            if "path" in adapter_info:
                # 将相对路径转换为绝对路径
                rel_path = adapter_info["path"]
                # 确认路径是否已经是绝对路径
                if os.path.isabs(rel_path):
                    abs_path = rel_path
                else:
                    abs_path = os.path.join(multi_adapter_dir, rel_path)
                self.adapter_paths[adapter_name] = abs_path
            else:
                raise ValueError(f"Adapter {adapter_name} 缺少路径信息")
        
        # 创建样本索引到adapter的映射
        self.index_to_adapter = {}
        for adapter_name, adapter_info in self.adapter_config["adapters"].items():
            if "indices" in adapter_info:
                adapter_path = self.adapter_paths[adapter_name]
                for idx in adapter_info["indices"]:
                    self.index_to_adapter[idx] = adapter_path
            else:
                raise ValueError(f"Adapter {adapter_name} 缺少样本索引信息")
        
        print(f"已加载 {len(self.adapter_paths)} 个Adapters 和 {len(self.index_to_adapter)} 个样本映射")

    def _load_adapter(self, adapter_path: str) -> None:
        """加载指定的adapter，通过重新加载整个模型实现"""
        # 如果已经加载了相同的adapter，则无需重新加载
        if self.current_adapter_path == adapter_path:
            print(f"信息: Adapter {adapter_path} 已加载，无需切换。")
            return

        print(f"信息: 准备加载新的 adapter: {adapter_path}")

        # 卸载当前模型以释放资源
        if hasattr(self, "model") and self.model is not None:
            print(f"信息: 卸载当前模型以加载 adapter: {adapter_path}")
            del self.model
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
            print(f"信息: 当前模型已卸载，CUDA 缓存已清理")

        # 创建model_args的深拷贝以避免修改原始配置
        temp_model_args = copy.deepcopy(self.model_args)
        
        # 将adapter_name_or_path设置为目标adapter路径
        # 关键修改：确保adapter_path被正确处理（本地路径或Hugging Face路径）
        if os.path.exists(adapter_path) or os.path.isdir(adapter_path):
            # 如果是本地路径，使用列表格式，与SSR中的处理方式一致
            temp_model_args.adapter_name_or_path = [adapter_path]
            print(f"信息: 检测到本地适配器路径: {adapter_path}")
        else:
            # 如果可能是Hugging Face路径，保持原样
            temp_model_args.adapter_name_or_path = adapter_path
        
        print(f"信息: 设置 adapter_name_or_path 为: {temp_model_args.adapter_name_or_path} (类型: {type(temp_model_args.adapter_name_or_path)})")
        
        # 设置finetuning_type为lora，确保正确加载
        # 如果adapter_path为空，则加载基础模型，无需设置finetuning_type
        temp_finetuning_args = copy.deepcopy(self.finetuning_args)
        if adapter_path:
            temp_finetuning_args.finetuning_type = "lora"
            print(f"信息: 设置 finetuning_type 为 lora 以加载 adapter: {adapter_path}")
        else:
            # 如果没有提供adapter路径，确保finetuning_type不是lora，防止意外加载
            temp_finetuning_args.finetuning_type = None
            print("信息: 未提供 adapter 路径，将加载基础模型")

        # 使用load_model重新加载模型和指定的adapter
        print(f"信息: 调用 load_model 加载模型及 adapter: {adapter_path if adapter_path else '基础模型'}")
        try:
            # load_model 需要 tokenizer, model_args, finetuning_args
            self.model = load_model(
                tokenizer=self.tokenizer, 
                model_args=temp_model_args, 
                finetuning_args=temp_finetuning_args
            )
            self.current_adapter_path = adapter_path
            print(f"信息: 成功加载模型及 adapter: {adapter_path if adapter_path else '基础模型'}")
            
            # 验证adapter是否实际激活 (如果加载了adapter)
            if adapter_path:
                if hasattr(self.model, "active_adapters") and self.model.active_adapters:
                    print(f"信息: 检测到激活的 adapters")
                elif hasattr(self.model, "active_adapter") and self.model.active_adapter:
                     # 兼容旧版peft
                     print(f"信息: 检测到激活的 adapter")
                else:
                    print(f"警告: 加载 adapter {adapter_path} 后，未检测到激活的 adapter。请检查模型或Peft版本。")

        except Exception as e:
            print(f"错误: 使用 load_model 加载 adapter {adapter_path} 时出错: {str(e)}")
            # 尝试恢复到某种状态，或者抛出异常
            self.current_adapter_path = None # 加载失败，重置状态
            raise RuntimeError(f"无法加载 adapter {adapter_path}") from e

    def _get_adapter(self):
        """根据数据集类型获取相应的适配器"""
        # 目前只支持自定义数据集格式
        # 可以根据 self.data_args.dataset_format 或任务名称来决定使用哪个adapter
        # if self.eval_args.task.lower() == "alpaca_eval":
        #     return AlpacaEvalAdapter()
        # else:
        #     return CustomDatasetAdapter()
        # 默认使用CustomDatasetAdapter
        return CustomDatasetAdapter()

    def _find_data_file(self, task_name: str, split_type: str) -> Optional[str]:
        """
        Finds the data file path for a given task and split type using dataset_info.json.

        Args:
            task_name: The full name of the task (e.g., "custom_task_abc").
            split_type: The type of split required ("dev" or "test").

        Returns:
            The absolute path to the data file, or None if not found.

        Raises:
            FileNotFoundError: If dataset_info.json cannot be found.
            ValueError: If no matching entry with 'file_name' is found in dataset_info.json.
        """
        # 1. 定位并加载 dataset_info.json (优先使用预加载的)
        if hasattr(self, "dataset_info") and self.dataset_info:
            dataset_info = self.dataset_info
            logger.debug("Using preloaded dataset_info.json") # English log
        else:
            task_dir = getattr(self.eval_args, "task_dir", "./data")
            dataset_info_path = os.path.join(task_dir, "dataset_info.json")
            if not os.path.exists(dataset_info_path):
                # Fallback check in ./data if task_dir is different
                if os.path.abspath(task_dir) != os.path.abspath("./data"):
                    fallback_info_path = os.path.join("./data", "dataset_info.json")
                    if os.path.exists(fallback_info_path):
                        dataset_info_path = fallback_info_path
                        logger.warning(f"dataset_info.json not found in task_dir '{task_dir}', using fallback: {fallback_info_path}") # English log
                    else:
                         raise FileNotFoundError(f"dataset_info.json not found in task directory: {task_dir} or fallback './data'") # English log
                else:
                    raise FileNotFoundError(f"dataset_info.json not found in task directory: {dataset_info_path}") # English log

            logger.debug(f"Loading dataset_info.json from: {dataset_info_path}") # English log
            try:
                with open(dataset_info_path, "r", encoding="utf-8") as f:
                    dataset_info = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load or parse dataset_info.json from {dataset_info_path}: {e}") # English log
                raise

        # 2. 使用完整任务名查找
        logger.info(f"Searching dataset_info.json for task_name='{task_name}' and split_type='{split_type}'") # English log

        # 3. 查找匹配的条目
        matches = []
        potential_keys = [
            f"{task_name}_{split_type}", # e.g., custom_task_abc_test
            task_name                   # e.g., custom_task_abc (match only if split field matches)
        ]
        logger.debug(f"Potential keys to search: {potential_keys}") # English log

        for key, entry in dataset_info.items():
            # 检查 key 是否匹配潜在名称
            key_matches_pattern = key in potential_keys
            # 检查 "split" 字段是否匹配 (如果 entry 中有 split 字段)
            split_field = entry.get("split")
            split_matches = split_field == split_type

            # 检查 'file_name' 字段是否存在
            file_name = entry.get("file_name")

            logger.debug(f"Checking entry: key='{key}', entry_split='{split_field}', target_split='{split_type}', file_name='{file_name}'") # English log

            # 条件1：key 完全匹配 (e.g., "custom_task_abc_test") 且有 file_name
            if key == f"{task_name}_{split_type}" and file_name:
                 matches.append({"key": key, "file_name": file_name, "priority": 1}) # Higher priority
                 logger.debug(f"Found priority match (key suffix): key='{key}', file_name='{file_name}'") # English log
                 continue # Found the most specific match, no need to check further for this entry

            # 条件2：key 是任务名 (e.g., "custom_task_abc") 且 split 字段匹配 且有 file_name
            if key == task_name and split_matches and file_name:
                matches.append({"key": key, "file_name": file_name, "priority": 0}) # Lower priority
                logger.debug(f"Found match (task name + split field): key='{key}', file_name='{file_name}'") # English log

        # 4. 处理查找结果
        if not matches:
            # 对于 'dev' split，允许找不到
            if split_type == "dev":
                logger.warning(f"No matching entry found in dataset_info.json for task '{task_name}' and split 'dev' with a 'file_name'. Proceeding without dev set.") # English log
                return None
            else:
                # 对于 'test' split，必须找到
                raise ValueError(f"No matching entry found in dataset_info.json for task '{task_name}' and split '{split_type}' with a 'file_name'.") # English log

        # 按优先级排序 (高优先级在前)
        matches.sort(key=lambda x: x["priority"], reverse=True)

        selected_match = matches[0] # 选择最高优先级的匹配项
        if len(matches) > 1:
             logger.warning(f"Multiple entries found in dataset_info.json for task '{task_name}' and split '{split_type}'. Prioritizing match with key '{selected_match['key']}'. Found matches: {matches}") # English log
        logger.info(f"Selected match: key='{selected_match['key']}', file_name='{selected_match['file_name']}'") # English log


        # 5. 构建并验证文件路径
        #    假设 file_name 是相对于 dataset_info.json 所在目录或 ./data 的路径
        task_dir = getattr(self.eval_args, "task_dir", "./data")
        file_path_in_task_dir = os.path.join(task_dir, selected_match["file_name"])
        file_path_in_data_dir = os.path.join("./data", selected_match["file_name"])

        if os.path.exists(file_path_in_task_dir):
            logger.info(f"Resolved data file path in task_dir: {file_path_in_task_dir}") # English log
            return file_path_in_task_dir
        elif os.path.abspath(task_dir) != os.path.abspath("./data") and os.path.exists(file_path_in_data_dir):
            logger.warning(f"File '{selected_match['file_name']}' not found in task_dir '{task_dir}', but found in fallback './data'. Using fallback path: {file_path_in_data_dir}") # English log
            return file_path_in_data_dir
        else:
            # 如果两个地方都找不到
            error_msg = f"Data file specified in dataset_info.json ('{selected_match['file_name']}') not found at expected path: {file_path_in_task_dir}"
            if os.path.abspath(task_dir) != os.path.abspath("./data"):
                 error_msg += f" or fallback path: {file_path_in_data_dir}"
            # 对于 'dev' split，找不到时返回 None 而不是抛出错误
            if split_type == "dev":
                 logger.warning(error_msg + ". Proceeding without dev set.") # English log
                 return None
            else:
                 raise FileNotFoundError(error_msg) # English log

    def _load_dataset_options(self) -> Dict:
        """加载数据集选项配置"""
        # 首先尝试使用命令行参数指定的路径
        if hasattr(self.eval_args, "dataset_options") and self.eval_args.dataset_options:
            options_path = self.eval_args.dataset_options
        else:
            # 如果未指定，则使用默认路径
            options_path = os.path.join("./data", "dataset_options.json")
        
        if os.path.exists(options_path):
            with open(options_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    @torch.inference_mode()
    def batch_inference(self, batch: List[Dict]) -> List[Dict]:
        """批量推理，支持多adapter模式"""
        # 如果使用vLLM且不是多adapter模式
        if self.use_vllm and not self.using_multi_adapter:
            return self._vllm_batch_inference(batch)
        
        # 多adapter模式处理
        if self.using_multi_adapter:
            print(f"\n=====================================")
            print(f"批量处理 {len(batch)} 个样本 (多Adapter模式)")
            print(f"=====================================")
            
            # 创建adapter到样本的分组
            adapter_to_examples = defaultdict(list)
            adapter_to_indices = defaultdict(list)
            adapter_names = defaultdict(str)
            
            # 根据配置将样本分配到对应的adapter
            for i, example in enumerate(batch):
                # 获取当前样本在全局测试集中的索引
                global_idx = example.get("index", -1)
                
                if global_idx == -1:
                    print(f"警告: 样本没有全局索引，无法找到对应的adapter，使用默认adapter")
                    # 如果样本没有索引，可以使用默认adapter或跳过
                    continue
                
                # 查找样本对应的adapter
                adapter_name = self.index_to_adapter.get(global_idx)
                if adapter_name is None:
                    print(f"警告: 样本索引 {global_idx} 未在配置中找到对应的adapter，跳过评估")
                    continue
                
                # 获取adapter路径
                adapter_path = self.adapter_paths.get(adapter_name)
                if not adapter_path:
                    print(f"错误: adapter {adapter_name} 未找到对应的路径")
                    continue
                
                # 将样本添加到对应adapter组
                adapter_to_examples[adapter_path].append(example)
                adapter_to_indices[adapter_path].append(i)
                adapter_names[adapter_path] = adapter_name  # 保存adapter名称用于日志
            
            # 打印分组信息
            print(f"\n样本按adapter分组情况:")
            for adapter_path in adapter_to_examples:
                print(f"  - Adapter '{adapter_names[adapter_path]}': {len(adapter_to_examples[adapter_path])} 个样本")
            
            # 预分配结果列表，确保结果顺序与输入一致
            outputs = [{"prediction": "", "is_correct": False, "match_type": "not_processed"}] * len(batch)
            
            # 获取所有adapter路径并按名称排序，确保顺序一致
            sorted_adapter_paths = sorted(adapter_to_examples.keys(), 
                                         key=lambda path: adapter_names[path])
            
            print(f"\n按顺序处理 {len(sorted_adapter_paths)} 个adapter:")
            for i, adapter_path in enumerate(sorted_adapter_paths):
                adapter_name = adapter_names[adapter_path]
                print(f"\n[{i+1}/{len(sorted_adapter_paths)}] 处理adapter: {adapter_name}")
                
                examples = adapter_to_examples[adapter_path]
                if not examples:
                    print(f"  - 跳过：没有需要处理的样本")
                    continue
                
                # 使用info级别打印当前使用的adapter
                print(f"  - 正在加载 adapter: {adapter_name}")
                print(f"  - 路径: {adapter_path}")
                print(f"  - 样本数量: {len(examples)}")
                
                # 使用新的方法加载当前adapter
                self._load_adapter(adapter_path)
                # 在处理当前adapter的样本前，先检查adapter是否成功激活
                if adapter_path:  # 只有在应当有adapter时才检查
                    # 检查adapter是否成功激活
                    is_adapter_active = (
                        hasattr(self.model, "active_adapters") and self.model.active_adapters or 
                        hasattr(self.model, "active_adapter") and self.model.active_adapter or
                        hasattr(self.model, "default") and self.model.active_adapter
                    )
                    
                    # 获取激活的adapter名称（兼容不同PEFT版本）
                    active_name = getattr(self.model, "active_adapters", None) or getattr(self.model, "active_adapter", None)
                    
                    if not is_adapter_active:
                        raise ValueError(
                            f"错误: 尝试使用adapter '{adapter_path}'，但模型中没有检测到激活的adapter。"
                            f"请检查adapter加载过程是否正确。"
                        )
                    
                    #print(f"  - 成功激活adapter: {active_name}")
                # 处理当前adapter的所有样本
                print(f"  - 开始处理样本...")
                sub_results = self._process_batch_with_current_adapter(examples)
                print(f"  - 样本处理完成")
                
                # 将结果放回原始位置
                for sub_idx, orig_idx in enumerate(adapter_to_indices[adapter_path]):
                    if sub_idx < len(sub_results):
                        outputs[orig_idx] = sub_results[sub_idx]
                
                # 打印处理结果摘要
                correct_count = sum(1 for r in sub_results if r.get("is_correct", False))
                accuracy = correct_count / len(sub_results) if sub_results else 0
                print(f"  - Adapter '{adapter_name}' 处理结果: 准确率 {accuracy:.2%} ({correct_count}/{len(sub_results)})")
            
            print("\n所有adapter处理完成")
            return outputs
        
        # 非多adapter模式，直接处理所有样本
        #print(f"\n处理 {len(batch)} 个样本 (标准模式)")

        # Check if an adapter was intended for this evaluator instance but is not currently active
        if self.initial_adapter_path: # Check if an adapter was specified at init
            is_adapter_active = hasattr(self.model, "active_adapter") and self.model.active_adapter is not None 
            if not is_adapter_active:
                raise ValueError(
                    f"错误: 评估器初始化时指定了Adapter '{self.initial_adapter_path}'，但在执行标准推理前检测到模型没有激活的Adapter。"
                    f" 请检查Adapter是否在初始化时正确加载。"
                )
            else:
                # Optional: Add a print statement confirming the check passed
                #print(f"信息: 检测到激活的Adapter ，符合预期 '{self.initial_adapter_path}'。继续标准推理。")
                pass

        return self._process_batch_with_current_adapter(batch)

    def _vllm_batch_inference(self, batch: List[Dict]) -> List[Dict]:
        """使用vLLM进行批量推理"""
        # 准备输入数据
        inputs = []
        for example in batch:
            # 格式化示例
            formatted = self.adapter.format_example(
                example,
                template=getattr(self, 'eval_template', ""),
                subject_name=getattr(self, 'subject_name', ""),
                dataset_options=self.dataset_options,
                dataset_name=self.eval_args.task,
                support_set=getattr(self, 'dev_examples', [])
            )
            
            if not formatted:
                continue
                
            # 编码输入
            input_ids, _ = self.template.encode_oneturn(
                tokenizer=self.tokenizer,
                messages=formatted
            )
            
            if not input_ids:
                continue
                
            # 处理多模态数据
            if example.get("images"):
                multi_modal_data = {
                    "image": self.template.mm_plugin._regularize_images(
                        example["images"],
                        image_max_pixels=self.vllm_image_max_pixels,
                        image_min_pixels=self.vllm_image_min_pixels
                    )
                }
            else:
                multi_modal_data = None
                
            inputs.append({
                "prompt_token_ids": input_ids,
                "multi_modal_data": multi_modal_data
            })
        
        if not inputs:
            return [{"prediction": "", "is_correct": False, "match_type": "format"}] * len(batch)
        
        # 使用vLLM进行推理
        try:
            results = self.vllm_engine.generate(
                inputs,
                self.sampling_params,
                lora_request=self.lora_request
            )
            
            # 处理结果
            outputs = []
            result_idx = 0
            
            for example in batch:
                if result_idx < len(results):
                    pred = results[result_idx].outputs[0].text
                    result_idx += 1
                    
                    # 处理预测结果
                    result = self.adapter.process_result(
                        pred,
                        example,
                        self.dataset_options,
                        self.eval_args.task
                    )
                    outputs.append(result)
                else:
                    outputs.append({"prediction": "", "is_correct": False, "match_type": "vllm_error"})
        except Exception as e:
            print(f"vLLM推理出错: {str(e)}")
            return [{"prediction": str(e)[:100], "is_correct": False, "match_type": "vllm_error"}] * len(batch)
        
        return outputs
    
    def _process_batch_with_current_adapter(self, batch: List[Dict]) -> List[Dict]:
        """使用当前加载的adapter处理一批样本"""
        # 格式化输入
        formatted_examples = []
        for example in batch:
            formatted = self.adapter.format_example(
                example,
                template=self.eval_template if hasattr(self, 'eval_template') else "",
                subject_name=self.subject_name if hasattr(self, 'subject_name') else "",
                dataset_options=self.dataset_options,
                dataset_name=self.eval_args.task,
                support_set=self.dev_examples if hasattr(self, 'dev_examples') else []
            )
            if formatted:  # 确保格式化结果不为空
                formatted_examples.append(formatted)

        if not formatted_examples:
            return [{"prediction": "", "is_correct": False, "match_type": "format"} for _ in batch]

        # 编码输入
        inputs = []
        for messages in formatted_examples:
            # 使用encode_oneturn方法进行编码
            input_ids, _ = self.template.encode_oneturn(
                tokenizer=self.tokenizer,
                messages=messages
            )
            if input_ids:  # 确保编码结果不为空
                inputs.append({
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids)
                })

        if not inputs:
            return [{"prediction": "", "is_correct": False, "match_type": "format"} for _ in batch]

        # 批量处理
        batch_inputs = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # 确保数据类型正确并移动到GPU
        batch_inputs = {
            k: v.to(dtype=torch.long, device=self.model.device)
            for k, v in batch_inputs.items()
        }

        # 生成答案
        outputs = []
        for i in range(0, len(inputs), self.eval_args.batch_size):
            batch_slice = {
                k: v[i:i + self.eval_args.batch_size]
                for k, v in batch_inputs.items()
            }
            
            # 验证输入
            if batch_slice["input_ids"].size(1) == 0:
                outputs.extend([{"prediction": "", "is_correct": False, "match_type": "format"}] * min(self.eval_args.batch_size, len(inputs) - i))
                continue

            # 使用 GeneratingArguments 配置生成参数
            generation_kwargs = {
                "max_new_tokens": self.generating_args.max_new_tokens,
                "num_beams": self.generating_args.num_beams,
                "do_sample": self.generating_args.do_sample,
                "temperature": self.generating_args.temperature,
                "top_p": self.generating_args.top_p,
                "top_k": self.generating_args.top_k,
                "repetition_penalty": self.generating_args.repetition_penalty,
                "length_penalty": self.generating_args.length_penalty,
                # Handle potential absence of early_stopping in older GeneratingArguments
                "early_stopping": getattr(self.generating_args, "early_stopping", False), 
                # Handle potential absence of no_repeat_ngram_size
                "no_repeat_ngram_size": getattr(self.generating_args, "no_repeat_ngram_size", 0), 
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            generated = self.model.generate(
                **batch_slice,
                **generation_kwargs
            )
            
            # 解码生成的文本
            predictions = []
            for gen in generated:
                # 只获取新生成的部分
                new_tokens = gen[batch_slice["input_ids"].size(1):]
                pred = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                predictions.append(pred.strip())
            
            # 处理每个预测结果
            for j, (pred, example) in enumerate(zip(predictions, batch[i:i + self.eval_args.batch_size])):
                # Truncate prediction at the first newline character
                if '\n' in pred:
                    pred = pred.split('\n', 1)[0].strip() # Split once and take the first part, strip again
                else:
                    pred = pred.strip() # Ensure stripping even if no newline

                result = self.adapter.process_result(
                    pred,
                    example,
                    self.dataset_options,
                    self.eval_args.task
                )
                outputs.append(result)

        # 确保输出长度与输入批次长度匹配
        if len(outputs) < len(batch):
            outputs.extend([{"prediction": "", "is_correct": False, "match_type": "format"}] * (len(batch) - len(outputs)))
        elif len(outputs) > len(batch):
            outputs = outputs[:len(batch)]

        return outputs
    
    def evaluate_custom_dataset(self) -> Dict:
        """评估自定义数据集"""
        # 不再需要提取 subject_name，使用完整的 task name
        # self.subject_name = self.eval_args.task.split("_")[0]

        # --- Updated data file lookup using dataset_info.json ---
        task_name = self.eval_args.task
        self.dev_examples = []
        dev_file_path = None # 初始化 dev_file_path

        try:
            # 尝试查找 dev 文件路径
            logger.info(f"Attempting to find 'dev' split file for task: {task_name}") # English log
            dev_file_path = self._find_data_file(task_name=task_name, split_type="dev")

            if dev_file_path and self.eval_args.n_shot > 0:
                logger.info(f"Loading {self.eval_args.n_shot}-shot examples from: {dev_file_path}") # English log
                try:
                    with open(dev_file_path, "r", encoding="utf-8") as f:
                        dev_data = json.load(f)
                        # Handle different potential structures (list or dict with "examples")
                        if isinstance(dev_data, dict) and "examples" in dev_data:
                            loaded_dev_examples = dev_data["examples"]
                        elif isinstance(dev_data, list):
                            loaded_dev_examples = dev_data
                        else:
                            logger.warning(f"Unexpected format in dev file: {dev_file_path}. Expected list or dict with 'examples' key.") # English log
                            loaded_dev_examples = []

                        self.dev_examples = loaded_dev_examples[:self.eval_args.n_shot]
                        logger.info(f"Successfully loaded {len(self.dev_examples)} examples for {self.eval_args.n_shot}-shot evaluation") # English log
                except Exception as e:
                    logger.error(f"Error loading or processing dev file {dev_file_path}: {e}") # English log
                    self.dev_examples = [] # Reset on error
            elif self.eval_args.n_shot > 0:
                 # _find_data_file already logs a warning if 'dev' is not found
                 logger.info(f"Proceeding with 0-shot evaluation as 'dev' file was not found or n_shot is 0.") # English log
            else: # n_shot is 0
                logger.info("Proceeding with 0-shot evaluation (n_shot=0).") # English log

            # 查找 test 文件路径 (必须找到)
            logger.info(f"Attempting to find 'test' split file for task: {task_name}") # English log
            test_file_path = self._find_data_file(task_name=task_name, split_type="test")

            if not test_file_path:
                 # _find_data_file should raise an error if test file is mandatory and not found
                 # This is a safeguard, but the error should ideally originate from _find_data_file
                 raise FileNotFoundError(f"Test file for task '{task_name}' could not be located via dataset_info.json.") # English log

            logger.info(f"Loading test examples from: {test_file_path}") # English log
            try:
                with open(test_file_path, "r", encoding="utf-8") as f:
                    test_data = json.load(f)
                    # Handle different potential structures
                    if isinstance(test_data, dict) and "examples" in test_data:
                        test_examples_raw = test_data["examples"]
                    elif isinstance(test_data, list):
                        test_examples_raw = test_data
                    else:
                        logger.error(f"Unexpected format in test file: {test_file_path}. Expected list or dict with 'examples' key.") # English log
                        return {} # Cannot proceed without test data
            except Exception as e:
                logger.error(f"Error loading or processing test file {test_file_path}: {e}") # English log
                return {} # Cannot proceed without test data

        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Failed to find required dataset files for task {task_name} using dataset_info.json: {e}") # English log
            # 如果找不到 test 文件或 dataset_info.json，则无法继续
            return {} # Return empty dict or re-raise error
        # --- End of updated data file lookup ---

        # 为测试样本添加索引（如果需要，用于多adapter模式下的样本分配）
        if self.using_multi_adapter:
             logger.debug("Adding indices to test examples for multi-adapter mode.") # English log
             self.test_examples = [
                 {**example, "index": i} for i, example in enumerate(test_examples_raw)
             ]
        else:
             self.test_examples = test_examples_raw

        # 创建保存目录
        os.makedirs(self.eval_args.save_dir, exist_ok=True)

        # 初始化适配器使用统计（仅用于日志）
        adapter_usage_stats = {
            "total_examples": len(self.test_examples),
            "processed_examples": 0,
            "per_adapter_counts": defaultdict(int)
        }

        if self.using_multi_adapter and self.use_vllm:
            print(f"\n警告: 多Adapter模式下不支持使用vLLM，已自动禁用vLLM")
            self.use_vllm = False

        # 运行评估
        correct = 0
        total = len(self.test_examples)
        results = []
        
        if self.using_multi_adapter:
            # 确保multi_adapter_dir属性存在
            if not hasattr(self.eval_args, "multi_adapter_dir"):
                # 尝试设置默认值
                self.eval_args.multi_adapter_dir = "./multi_adapter_config"
                print(f"警告: 未找到multi_adapter_dir参数，使用默认值: {self.eval_args.multi_adapter_dir}")
                
            print(f"\n信息: 使用多Adapter模式进行评估 (配置文件: {os.path.join(self.eval_args.multi_adapter_dir, 'multiadapter_selected_config.json')})")
            print(f"信息: 将按adapter名称顺序依次评估每个数据子集")
            
            try:
                # 验证多adapter配置是否已正确加载
                if not hasattr(self, "adapter_paths") or not self.adapter_paths:
                    print("警告: adapter_paths 属性未定义或为空，尝试重新加载配置")
                    self._load_multi_adapter_config()
                
                # 打印配置信息以便调试
                print(f"\n多Adapter配置信息:")
                print(f"  - 任务名称: {self.adapter_config.get('task_name', '未知')}")
                print(f"  - Adapter数量: {len(self.adapter_paths)}")
                for adapter_name, path in self.adapter_paths.items():
                    print(f"  - {adapter_name}: {path}")
                    sample_indices = [idx for idx, name in self.index_to_adapter.items() if name == adapter_name]
                    print(f"    负责 {len(sample_indices)} 个样本")
                
            except Exception as e:
                print(f"获取多Adapter配置信息时出错: {str(e)}")
            
            # 统计每个adapter负责的样本数
            adapter_sample_counts = defaultdict(int)
            for idx in range(len(self.test_examples)):
                adapter_name = self.index_to_adapter.get(idx)
                if adapter_name:
                    adapter_sample_counts[adapter_name] += 1
                    
            # 打印adapter分布情况
            print("\n信息: Adapter样本分布情况:")
            for adapter_name, count in sorted(adapter_sample_counts.items()):
                print(f"  - {adapter_name}: {count} 个样本 ({count/len(self.test_examples)*100:.1f}%)")
        
        # 批量处理
        for i in tqdm(range(0, total, self.eval_args.batch_size)):
            batch = self.test_examples[i:min(i + self.eval_args.batch_size, total)]
            
            # 执行批量推理
            batch_results = self.batch_inference(batch)
            
            # 统计结果
            results.extend(batch_results)
            correct += sum(1 for r in batch_results if r.get("is_correct", False))
            
            # 更新统计信息（仅用于日志）
            if self.using_multi_adapter:
                for example, result in zip(batch, batch_results):
                    if result.get("match_type") != "not_processed":
                        adapter_usage_stats["processed_examples"] += 1
                        # 对于已处理的样本，记录使用的adapter
                        global_idx = example.get("index", -1)
                        if global_idx != -1:
                            adapter_name = self.index_to_adapter.get(global_idx)
                            if adapter_name:
                                adapter_usage_stats["per_adapter_counts"][adapter_name] += 1

        # 计算准确率
        accuracy = correct / total if total > 0 else 0
        print(f"\nAccuracy: {accuracy:.2%} ({correct}/{total})")

        # 构造输出结果
        output = {
            "task": self.eval_args.task,
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
            "results": results
        }
        
        # 添加多adapter模式的信息
        if self.using_multi_adapter:
            # 转换为常规字典
            adapter_usage = {
                "total_examples": adapter_usage_stats["total_examples"],
                "processed_examples": adapter_usage_stats["processed_examples"],
                "per_adapter_counts": dict(adapter_usage_stats["per_adapter_counts"])
            }
            
            # 添加adapter使用百分比
            if adapter_usage["processed_examples"] > 0:
                adapter_usage["per_adapter_percentage"] = {
                    adapter_name: (count / adapter_usage["processed_examples"]) * 100
                    for adapter_name, count in adapter_usage["per_adapter_counts"].items()
                }
            
            # 记录处理信息到输出
            output["multi_adapter_usage"] = adapter_usage
            
            # 打印adapter使用统计
            print("\nAdapter使用统计:")
            print(f"  总样本数: {adapter_usage['total_examples']}")
            print(f"  已处理样本数: {adapter_usage['processed_examples']}")
            if adapter_usage["per_adapter_counts"]:
                print("  每个adapter处理的样本数:")
                for adapter_name, count in adapter_usage["per_adapter_counts"].items():
                    percentage = (count / adapter_usage["processed_examples"]) * 100 if adapter_usage["processed_examples"] > 0 else 0
                    print(f"    {adapter_name}: {count} ({percentage:.1f}%)")

        # 保存结果
        save_path = os.path.join(self.eval_args.save_dir, "results.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to {save_path}")
        
        return output