import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union, Any
import json

from llamafactory.extras.logging import get_logger
from easycl.hparams import CLFinetuningArguments
def debugprint(*args, **kwargs):
    pass

logger = get_logger(__name__)

def is_distributed():
    """
    Check if we are running in a distributed environment
    """
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def get_rank():
    """
    Get the rank of the current process in distributed training
    Returns 0 if not in distributed environment
    """
    if is_distributed():
        return torch.distributed.get_rank()
    return 0

def is_main_process():
    """
    Check if the current process is the main process (rank 0)
    Always returns True if not in distributed environment
    """
    return get_rank() == 0

class DatasetClassifier(nn.Module):
    """
    Dataset classifier for distinguishing samples from different datasets
    """
    def __init__(self, hidden_size: int, num_datasets: int, dtype=None):
        """
        Initialize the dataset classifier

        Args:
            hidden_size: Dimension of model hidden states
            num_datasets: Number of datasets (including current dataset)
            dtype: Data type for classifier parameters (default: None, will use torch.float32)
        """
        super().__init__()
        rank = get_rank()
        debugprint(f"进程 rank={rank} 初始化 DatasetClassifier: hidden_size={hidden_size}, num_datasets={num_datasets}, dtype={dtype}")

        # 如果未指定数据类型，默认使用 float32
        if dtype is None:
            dtype = torch.float32
            debugprint(f"进程 rank={rank} 未指定数据类型，使用默认类型: {dtype}")

        # 使用指定的数据类型创建分类器
        self.classifier = nn.Linear(hidden_size, num_datasets, dtype=dtype)
        self.num_datasets = num_datasets
        debugprint(f"进程 rank={rank} 分类器权重类型: {self.classifier.weight.dtype}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            hidden_states: Model hidden states [batch_size, seq_len, hidden_size]

        Returns:
            logits: Classification logits [batch_size, num_datasets]
        """
        rank = get_rank()
        # Take the representation of the first token (usually [BOS] or special token)
        # as the sequence representation
        sequence_output = hidden_states[:, 0, :]  # [batch_size, hidden_size]
        debugprint(f"进程 rank={rank} DatasetClassifier forward: 输入 hidden_states 形状: {hidden_states.shape}, 类型: {hidden_states.dtype}, sequence_output 形状: {sequence_output.shape}")

        # 确保输入数据类型与分类器权重类型一致
        weight_dtype = self.classifier.weight.dtype
        if sequence_output.dtype != weight_dtype:
            debugprint(f"进程 rank={rank} DatasetClassifier forward: 输入数据类型 {sequence_output.dtype} 与权重类型 {weight_dtype} 不匹配，进行类型转换")
            sequence_output = sequence_output.to(weight_dtype)

        logits = self.classifier(sequence_output)  # [batch_size, num_datasets]
        debugprint(f"进程 rank={rank} DatasetClassifier forward: 输出 logits 形状: {logits.shape}, 类型: {logits.dtype}")
        return logits

def save_classifier(classifier: DatasetClassifier, save_path: str, dataset_names: List[str]) -> None:
    """
    Save classifier weights and configuration
    Only saves if running in the main process (rank 0)

    Args:
        classifier: Dataset classifier
        save_path: Save path
        dataset_names: List of dataset names
    """
    # Only save in the main process to avoid conflicts
    if not is_main_process():
        debugprint(f"非主进程 (rank={get_rank()})，跳过保存分类器")
        return

    debugprint(f"主进程 (rank=0) 尝试保存分类器到: {save_path}")
    os.makedirs(save_path, exist_ok=True)

    # Save classifier weights
    weights_path = os.path.join(save_path, "classifier.pt")
    debugprint(f"保存分类器权重到: {weights_path}")
    torch.save(classifier.state_dict(), weights_path)

    # Save configuration
    config = {
        "hidden_size": classifier.classifier.in_features,
        "num_datasets": classifier.num_datasets,
        "dataset_names": dataset_names
    }
    config_path = os.path.join(save_path, "classifier_config.json")
    debugprint(f"保存分类器配置到: {config_path}")
    debugprint(f"分类器配置内容: {config}")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Saved dataset classifier to {save_path} (rank 0)")

def load_classifier(load_path: str, hidden_size: int, new_num_datasets: Optional[int] = None, dtype: Optional[torch.dtype] = None) -> Tuple[DatasetClassifier, List[str]]:
    """
    Load classifier weights and configuration, optionally expand classification head
    In distributed setting, all processes will load the classifier

    Args:
        load_path: Load path
        hidden_size: Dimension of model hidden states
        new_num_datasets: New number of datasets, if classification head needs expansion
        dtype: Data type for classifier parameters (default: None, will use torch.float32)

    Returns:
        classifier: Dataset classifier
        dataset_names: List of dataset names
    """
    rank = get_rank()
    debugprint(f"进程 rank={rank} 尝试从以下路径加载分类器: {load_path}")
    config_path = os.path.join(load_path, "classifier_config.json")
    weights_path = os.path.join(load_path, "classifier.pt")

    # In distributed setting, we need to make sure all processes can access the files
    # Wait for all processes to reach this point
    if is_distributed():
        torch.distributed.barrier()

    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        if is_main_process():
            logger.warning(f"Classifier config or weights not found at {load_path}")
        debugprint(f"警告: 在 {load_path} 未找到分类器配置或权重 (rank={rank})")
        return None, []

    # Load configuration
    debugprint(f"进程 rank={rank} 从 {config_path} 加载配置")
    with open(config_path, "r") as f:
        config = json.load(f)
    debugprint(f"进程 rank={rank} 加载的配置: {config}")

    old_hidden_size = config["hidden_size"]
    old_num_datasets = config["num_datasets"]
    dataset_names = config["dataset_names"]

    # Check if hidden size matches
    if old_hidden_size != hidden_size:
        if is_main_process():
            logger.warning(f"Hidden size mismatch: saved {old_hidden_size}, current {hidden_size}")
        debugprint(f"警告: Hidden size 不匹配: 已保存 {old_hidden_size}, 当前 {hidden_size} (rank={rank})")

    # Create classifier
    if new_num_datasets is not None and new_num_datasets > old_num_datasets:
        # Need to expand classification head
        debugprint(f"进程 rank={rank} 需要扩展分类头: 从 {old_num_datasets} 到 {new_num_datasets} 个数据集")
        classifier = DatasetClassifier(hidden_size, new_num_datasets, dtype=dtype)
        debugprint(f"进程 rank={rank} 创建的分类器权重类型: {classifier.classifier.weight.dtype}")

        # Load old weights
        debugprint(f"进程 rank={rank} 从 {weights_path} 加载旧权重")
        old_state_dict = torch.load(weights_path, map_location="cpu")

        # Initialize new weights
        new_state_dict = classifier.state_dict()

        # Copy old weights (可能需要类型转换)
        debugprint(f"进程 rank={rank} 复制旧权重到新状态字典")
        old_weight = old_state_dict["classifier.weight"]
        old_bias = old_state_dict["classifier.bias"]

        # 确保数据类型匹配
        if old_weight.dtype != new_state_dict["classifier.weight"].dtype:
            debugprint(f"进程 rank={rank} 旧权重类型 {old_weight.dtype} 与新权重类型 {new_state_dict['classifier.weight'].dtype} 不匹配，进行类型转换")
            old_weight = old_weight.to(new_state_dict["classifier.weight"].dtype)
            old_bias = old_bias.to(new_state_dict["classifier.bias"].dtype)

        new_state_dict["classifier.weight"][:old_num_datasets] = old_weight
        new_state_dict["classifier.bias"][:old_num_datasets] = old_bias

        # Load updated weights
        classifier.load_state_dict(new_state_dict)

        if is_main_process():
            logger.info(f"Expanded classifier from {old_num_datasets} to {new_num_datasets} datasets")
        debugprint(f"进程 rank={rank} 已将分类器从 {old_num_datasets} 扩展到 {new_num_datasets} 个数据集")
    else:
        # Direct load
        debugprint(f"进程 rank={rank} 直接加载分类器，包含 {old_num_datasets} 个数据集")
        classifier = DatasetClassifier(hidden_size, old_num_datasets, dtype=dtype)
        debugprint(f"进程 rank={rank} 创建的分类器权重类型: {classifier.classifier.weight.dtype}")

        # 加载权重
        state_dict = torch.load(weights_path, map_location="cpu")

        # 确保数据类型匹配
        if state_dict["classifier.weight"].dtype != classifier.classifier.weight.dtype:
            debugprint(f"进程 rank={rank} 加载的权重类型 {state_dict['classifier.weight'].dtype} 与分类器权重类型 {classifier.classifier.weight.dtype} 不匹配，进行类型转换")
            for key in state_dict:
                state_dict[key] = state_dict[key].to(classifier.classifier.weight.dtype)

        classifier.load_state_dict(state_dict)
        if is_main_process():
            logger.info(f"Loaded classifier for {old_num_datasets} datasets")
        debugprint(f"进程 rank={rank} 已加载 {old_num_datasets} 个数据集的分类器")

    # Wait for all processes to finish loading
    if is_distributed():
        torch.distributed.barrier()

    return classifier, dataset_names

def compute_dataset_classification_loss(model_output: Dict[str, torch.Tensor],
                                       dataset_labels: torch.Tensor,
                                       classifier: DatasetClassifier) -> torch.Tensor:
    """
    Compute dataset classification loss

    Args:
        model_output: Model output containing hidden states
        dataset_labels: Dataset labels [batch_size]
        classifier: Dataset classifier

    Returns:
        loss: Classification loss
    """
    rank = get_rank()
    # Get model hidden states
    hidden_states = model_output.hidden_states[-1]  # Last layer hidden states [batch_size, seq_len, hidden_size]
    debugprint(f"进程 rank={rank} 计算分类损失: hidden_states 形状: {hidden_states.shape}, 类型: {hidden_states.dtype}, dataset_labels: {dataset_labels}")

    # 确保 dataset_labels 是 long 类型，这是 cross_entropy 所需的
    if dataset_labels.dtype != torch.long:
        debugprint(f"进程 rank={rank} 计算分类损失: dataset_labels 类型 {dataset_labels.dtype} 不是 long，进行类型转换")
        dataset_labels = dataset_labels.long()

    # Compute classification logits
    logits = classifier(hidden_states)  # [batch_size, num_datasets]
    debugprint(f"进程 rank={rank} 计算分类损失: logits 形状: {logits.shape}, 类型: {logits.dtype}")

    # Compute cross entropy loss
    loss = F.cross_entropy(logits, dataset_labels)
    debugprint(f"进程 rank={rank} 计算分类损失: 计算出的损失值: {loss.item()}")

    # In distributed training, we need to average the loss across all processes
    if is_distributed():
        # 确保 loss 是 float32 类型，以避免 all_reduce 时的精度问题
        if loss.dtype != torch.float32:
            loss = loss.float()
            debugprint(f"进程 rank={rank} 计算分类损失: 将损失转换为 float32 类型用于 all_reduce")

        # All-reduce the loss
        torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
        # Divide by world size to get the average
        world_size = torch.distributed.get_world_size()
        loss = loss / world_size
        debugprint(f"进程 rank={rank} 分布式平均后的损失值: {loss.item()}, world_size={world_size}")

    return loss