import os
from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict, Any

from datasets import DownloadMode


@dataclass
class EvaluationArguments:
    """评估参数配置"""

    task: str = field(
        default="mmlu_test",
        metadata={"help": "评估任务名称"}
    )

    lang: str = field(
        default="en",
        metadata={"help": "评估语言"}
    )

    task_dir: str = field(
        default="evaluation",
        metadata={"help": "数据集目录"}
    )

    dataset_options: Optional[str] = field(
        default=None,
        metadata={"help": "数据集选项配置文件路径"}
    )

    save_dir: Optional[str] = field(
        default=None,
        metadata={"help": "评估结果保存路径"}
    )

    n_shot: int = field(
        default=0,
        metadata={"help": "few-shot样例数量"}
    )

    batch_size: int = field(
        default=4,
        metadata={"help": "批处理大小"}
    )

    seed: int = field(
        default=42,
        metadata={"help": "随机种子"}
    )

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

    download_mode: str = field(
        default="reuse_dataset_if_exists",
        metadata={"help": "数据集下载模式"}
    )

    def __post_init__(self):
        pass  # 移除save_dir存在性检查
