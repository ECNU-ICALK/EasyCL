from dataclasses import dataclass, field
from typing import Optional, List
from .evaluation_args import EvaluationArguments
import json
import os
from llamafactory.extras import logging
from dataclasses import field

logger = logging.get_logger(__name__)


@dataclass
class CLEvaluationArguments(EvaluationArguments):
#class CLEvaluationArguments:
    """继承基础评估参数，添加持续学习特有参数"""

    # MoE-LoRA 评估开关 (moved from CLEvaluationArguments)
    use_moelora_eval: bool = field(
        default=False,
        metadata={"help": "是否启用MoE-LoRA权重进行评估 (仅在 eval_mode='single' 时生效)"}
    )

    # CLIT-MoE 评估开关
    use_clitmoe_eval: bool = field(
        default=False,
        metadata={"help": "是否启用CLIT-MoE权重进行评估 (仅在 eval_mode='single' 时生效)"}
    )

    # CLEval vLLM 使用开关 (moved from CLEvaluationArguments)
    cleval_use_vllm: bool = field(
        default=False,
        metadata={"help": "是否在 CLEvalEvaluator 中使用 vLLM 进行推理 (仅在 eval_mode='single' 时生效)"}
    )
    
    cl_tasks: str = field(
        default="",
        metadata={"help": "评估任务列表，用逗号分隔，如'agnews,chnsenticorp,cmnli'"}
    )
    
    eval_mode: str = field(
        default="single",
        metadata={
            "help": "评估模式，可选'single'或'multi_adapter'",
            "choices": ["single", "multi_adapter"]
        }
    )
    
    cl_tuning_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "微调类型，当eval_mode为multi_adapter时必选，可选'lora'或'full_model'",
            "choices": ["lora", "full_model"]
        }
    )
    
    compared_adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "当eval_mode为compare且cl_tuning_type为lora时必选，用于指定微调后的adapter路径"}
    )
    
    compared_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "当eval_mode为compare且cl_tuning_type为full_model时必选，用于指定微调后的完整模型路径"}
    )
    
    # 新增多adapter相关参数
    multi_adapter_dir: Optional[str] = field(
        default=None,
        metadata={"help": "多adapter的存储路径，用于multi_adapter评估模式"}
    )
    
    # ABSCL Selector相关参数
    use_abscl_selector: bool = field(
        default=False,
        metadata={"help": "是否使用ABSCL选择器进行样本-Adapter的分配"}
    )
    
    abscl_selector_batch_size: int = field(
        default=16,
        metadata={"help": "ABSCL选择器处理样本的批量大小"}
    )
    
    # Dynamic ConPet Selector相关参数
    use_dynamic_conpet_selector: bool = field(
        default=False,
        metadata={"help": "是否使用Dynamic ConPet选择器进行样本-Adapter的分配"}
    )
    
    dynamic_conpet_selector_batch_size: int = field(
        default=16,
        metadata={"help": "Dynamic ConPet选择器处理样本的批量大小"}
    )
    
    # 持续学习指标相关参数
    calculate_cl_metrics: bool = field(
        default=False,
        metadata={"help": "是否计算持续学习指标 (Last, Avg, BWT, FWT)。如果 cl_tasks 包含多个任务，此项会自动设为 True。"}
    )
    
    # vLLM相关参数
    # cleval_use_vllm: bool = field( # MOVED TO EvaluationArguments
    #     default=False,
    #     metadata={"help": "是否使用vLLM进行推理，仅在非多adapter模式下有效"}
    # )
    
    vllm_pipeline_parallel_size: int = field(
        default=1,
        metadata={"help": "vLLM流水线并行大小，仅在cleval_use_vllm=True时有效"}
    )
    
    def __post_init__(self):
        # super().__post_init__() # Commented out to fix AttributeError

        
        # 自动设置 calculate_cl_metrics
        tasks = self.cl_tasks.split(",") if self.cl_tasks else []
        if len(tasks) > 1:
            if not self.calculate_cl_metrics:
                logger.info("检测到多个任务，将自动启用持续学习指标计算。")
                self.calculate_cl_metrics = True
        elif self.calculate_cl_metrics:
            logger.warning("只有一个或没有任务，持续学习指标计算将被禁用。")
            self.calculate_cl_metrics = False # 强制禁用

        
        # 新增多adapter模式的检查
        if self.eval_mode == "multi_adapter":
            if not self.multi_adapter_dir:
                raise ValueError("在multi_adapter模式下必须提供multi_adapter_dir")


        
        # 检查所有任务是否都有对应的配置
        if self.dataset_options:
            try:
                with open(self.dataset_options, "r") as f:
                    dataset_options = json.load(f)
                for task in tasks:
                    if task not in dataset_options:
                        # 允许缺少配置，但发出警告
                        logger.warning(f"任务 {task} 在 dataset_options 文件 '{self.dataset_options}' 中没有找到对应的配置。")
            except FileNotFoundError:
                logger.error(f"Dataset options file not found: {self.dataset_options}")
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from dataset options file: {self.dataset_options}")
            except Exception as e:
                logger.error(f"Error reading dataset options file {self.dataset_options}: {e}")
        
        # 检查选择器逻辑
        if self.use_abscl_selector and self.use_dynamic_conpet_selector:
            logger.warning("同时启用了ABSCL选择器和Dynamic ConPet选择器，将优先使用Dynamic ConPet选择器")