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

class DatasetClassifier(nn.Module):
    """
    Dataset classifier for distinguishing samples from different datasets
    """
    def __init__(self, hidden_size: int, num_datasets: int):
        """
        Initialize the dataset classifier
        
        Args:
            hidden_size: Dimension of model hidden states
            num_datasets: Number of datasets (including current dataset)
        """
        super().__init__()
        debugprint(f"初始化 DatasetClassifier: hidden_size={hidden_size}, num_datasets={num_datasets}")
        self.classifier = nn.Linear(hidden_size, num_datasets)
        self.num_datasets = num_datasets
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            hidden_states: Model hidden states [batch_size, seq_len, hidden_size]
            
        Returns:
            logits: Classification logits [batch_size, num_datasets]
        """
        # Take the representation of the first token (usually [BOS] or special token)
        # as the sequence representation
        sequence_output = hidden_states[:, 0, :]  # [batch_size, hidden_size]
        debugprint(f"DatasetClassifier forward: 输入 hidden_states 形状: {hidden_states.shape}, sequence_output 形状: {sequence_output.shape}")
        logits = self.classifier(sequence_output)  # [batch_size, num_datasets]
        debugprint(f"DatasetClassifier forward: 输出 logits 形状: {logits.shape}")
        return logits

def save_classifier(classifier: DatasetClassifier, save_path: str, dataset_names: List[str]) -> None:
    """
    Save classifier weights and configuration
    
    Args:
        classifier: Dataset classifier
        save_path: Save path
        dataset_names: List of dataset names
    """
    debugprint(f"尝试保存分类器到: {save_path}")
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

    # logger.info(f"Saved dataset classifier to {save_path}")
    debugprint(f"数据集分类器已保存到 {save_path}")

def load_classifier(load_path: str, hidden_size: int, new_num_datasets: Optional[int] = None) -> Tuple[DatasetClassifier, List[str]]:
    """
    Load classifier weights and configuration, optionally expand classification head
    
    Args:
        load_path: Load path
        hidden_size: Dimension of model hidden states
        new_num_datasets: New number of datasets, if classification head needs expansion
        
    Returns:
        classifier: Dataset classifier
        dataset_names: List of dataset names
    """
    debugprint(f"尝试从以下路径加载分类器: {load_path}")
    config_path = os.path.join(load_path, "classifier_config.json")
    weights_path = os.path.join(load_path, "classifier.pt")
    
    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        # logger.warning(f"Classifier config or weights not found at {load_path}")
        debugprint(f"警告: 在 {load_path} 未找到分类器配置或权重")
        return None, []
    
    # Load configuration
    debugprint(f"从 {config_path} 加载配置")
    with open(config_path, "r") as f:
        config = json.load(f)
    debugprint(f"加载的配置: {config}")

    old_hidden_size = config["hidden_size"]
    old_num_datasets = config["num_datasets"]
    dataset_names = config["dataset_names"]
    
    # Check if hidden size matches
    if old_hidden_size != hidden_size:
        # logger.warning(f"Hidden size mismatch: saved {old_hidden_size}, current {hidden_size}")
        debugprint(f"警告: Hidden size 不匹配: 已保存 {old_hidden_size}, 当前 {hidden_size}")
    
    # Create classifier
    if new_num_datasets is not None and new_num_datasets > old_num_datasets:
        # Need to expand classification head
        debugprint(f"需要扩展分类头: 从 {old_num_datasets} 到 {new_num_datasets} 个数据集")
        classifier = DatasetClassifier(hidden_size, new_num_datasets)
        
        # Load old weights
        debugprint(f"从 {weights_path} 加载旧权重")
        old_state_dict = torch.load(weights_path, map_location="cpu")
        
        # Initialize new weights
        new_state_dict = classifier.state_dict()
        
        # Copy old weights
        debugprint("复制旧权重到新状态字典")
        new_state_dict["classifier.weight"][:old_num_datasets] = old_state_dict["classifier.weight"]
        new_state_dict["classifier.bias"][:old_num_datasets] = old_state_dict["classifier.bias"]
        
        # Load updated weights
        classifier.load_state_dict(new_state_dict)
        
        # logger.info(f"Expanded classifier from {old_num_datasets} to {new_num_datasets} datasets")
        debugprint(f"已将分类器从 {old_num_datasets} 扩展到 {new_num_datasets} 个数据集")
    else:
        # Direct load
        debugprint(f"直接加载分类器，包含 {old_num_datasets} 个数据集")
        classifier = DatasetClassifier(hidden_size, old_num_datasets)
        classifier.load_state_dict(torch.load(weights_path, map_location="cpu"))
        # logger.info(f"Loaded classifier for {old_num_datasets} datasets")
        debugprint(f"已加载 {old_num_datasets} 个数据集的分类器")
    
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
    # Get model hidden states
    hidden_states = model_output.hidden_states[-1]  # Last layer hidden states [batch_size, seq_len, hidden_size]
    debugprint(f"计算分类损失: hidden_states 形状: {hidden_states.shape}, dataset_labels: {dataset_labels}")

    # Compute classification logits
    logits = classifier(hidden_states)  # [batch_size, num_datasets]
    debugprint(f"计算分类损失: logits 形状: {logits.shape}")

    # Compute cross entropy loss
    loss = F.cross_entropy(logits, dataset_labels)
    debugprint(f"计算分类损失: 计算出的损失值: {loss.item()}")

    return loss