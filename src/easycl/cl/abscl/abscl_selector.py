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
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict
from tqdm import tqdm
from transformers import Trainer, Seq2SeqTrainer, PreTrainedModel, TrainingArguments, BatchEncoding
from torch.utils.data import DataLoader, Dataset
import copy

from easycl.cl.abscl.abscl import ABSCLFeatureExtractor
from llamafactory.extras.logging import get_logger

logger = get_logger(__name__)

def mahalanobis_distance(feature: torch.Tensor, mean: torch.Tensor, cov_inv: torch.Tensor) -> float:
    """
    计算特征向量与均值之间的Mahalanobis距离
    
    Args:
        feature: 特征向量，形状 [hidden_size]
        mean: 均值向量，形状 [hidden_size]
        cov_inv: 协方差矩阵的逆，形状 [hidden_size, hidden_size]
        
    Returns:
        distance: Mahalanobis距离
    """
    # 确保所有张量在同一设备上
    device = feature.device
    mean = mean.to(device)
    cov_inv = cov_inv.to(device)
    
    diff = feature - mean  # [hidden_size]
    # 计算 diff^T * cov_inv * diff
    distance = torch.mm(torch.mm(diff.unsqueeze(0), cov_inv), diff.unsqueeze(1))
    return distance.item()

def find_adapters_in_dir(base_dir: str) -> Dict[str, str]:
    """
    扫描目录查找可用的Adapter，返回任务ID到Adapter相对路径的映射
    
    Args:
        base_dir: Adapter目录的基础路径
        
    Returns:
        adapters_map: {任务ID: 相对路径} 的字典
    """
    adapters_map = {}
    
    if not os.path.exists(base_dir):
        logger.warning(f"Adapter目录不存在: {base_dir}")
        return adapters_map
        
    # 扫描子目录，查找有效的Adapter目录
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        
        # 验证这是一个有效的Adapter目录
        adapter_config_path = os.path.join(item_path, "adapter_config.json")
        if os.path.isdir(item_path) and os.path.exists(adapter_config_path):
            # 尝试从目录名称或配置获取任务ID
            try:
                with open(adapter_config_path, 'r') as f:
                    config = json.load(f)
                task_id = config.get("task_id", item)  # 如果配置中没有task_id，则使用目录名
            except Exception as e:
                logger.warning(f"读取Adapter配置失败: {adapter_config_path}, 错误: {str(e)}")
                task_id = item  # 默认使用目录名作为任务ID
                
            # 记录任务ID到相对路径的映射
            adapters_map[task_id] = item
            logger.info(f"发现Adapter: {task_id} -> {item}")
    
    return adapters_map

def create_data_batch(sample: Dict[str, Any], tokenizer) -> BatchEncoding:
    """
    将单个样本转换为模型可处理的批次
    
    Args:
        sample: 数据样本
        tokenizer: 分词器
        
    Returns:
        batch: 模型输入批次
    """
    # 提取样本的instruction和input字段
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    
    # 组合成完整文本
    full_text = f"{instruction}\n{input_text}".strip()
    
    # 使用分词器处理
    encoded = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    
    return encoded

class SimpleDataset(Dataset):
    """
    简单的数据集包装器，用于将样本列表转换为DataLoader可用的格式
    """
    def __init__(self, samples, tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return create_data_batch(self.samples[idx], self.tokenizer)

def select_adapter(
    model_args,
    data_args,
    training_args,
    finetuning_args,
    dataset_path: str,
    multi_adapter_dir: str,
    task_name: str,
    device: Optional[torch.device] = None,
    batch_size: int = 16
) -> None:
    """
    根据测试数据，为每个样本选择最合适的Adapter，
    并生成multiadapter_selected_config.json配置文件
    
    Args:
        model_args: 模型参数
        data_args: 数据参数
        training_args: 训练参数
        finetuning_args: 微调参数
        dataset_path: 测试数据集路径
        multi_adapter_dir: 多Adapter目录路径
        task_name: 当前评估任务名称
        device: 计算设备
        batch_size: 批处理大小
    """
    # 确保输出目录存在
    os.makedirs(multi_adapter_dir, exist_ok=True)
    output_config_path = os.path.join(multi_adapter_dir, "multiadapter_selected_config.json")
    
    logger.info(f"开始进行ABSCL Adapter选择，输出配置将保存到: {output_config_path}")
    
    # 1. 加载统计信息(均值、协方差)
    stats_path = multi_adapter_dir
    stats_file = os.path.join(stats_path, "abscl_stats.pt")
    
    if not os.path.exists(stats_file):
        raise ValueError(f"未找到ABSCL统计文件: {stats_file}")
        
    logger.info(f"加载ABSCL特征统计信息: {stats_file}")
    stats = torch.load(stats_file)
    
    # 获取所有任务的均值和共享协方差矩阵的逆
    task_means = stats["task_means"]
    if not task_means:
        raise ValueError("统计信息中没有任务均值数据")
        
    cov_matrix = stats["cov_matrix"]
    if cov_matrix is None:
        raise ValueError("统计信息中没有协方差矩阵")
        
    # 计算协方差矩阵的逆
    # 添加小的正则化项，确保矩阵可逆
    logger.info("计算协方差矩阵的逆...")
    cov_matrix_np = cov_matrix.numpy()
    # 添加小的正则化项到对角线
    epsilon = 1e-5
    cov_matrix_np += np.eye(cov_matrix_np.shape[0]) * epsilon
    
    try:
        cov_inv_np = np.linalg.inv(cov_matrix_np)
        cov_inv = torch.tensor(cov_inv_np, dtype=torch.float32)
    except np.linalg.LinAlgError:
        logger.warning("协方差矩阵不可逆，尝试使用伪逆")
        cov_inv_np = np.linalg.pinv(cov_matrix_np)
        cov_inv = torch.tensor(cov_inv_np, dtype=torch.float32)
    
    # 2. 加载基础模型（无Adapter）
    # 创建finetuning_args副本并设置adapter_name_or_path为None
    finetuning_args_base = copy.deepcopy(finetuning_args)
    finetuning_args_base.adapter_name_or_path = None
    
    logger.info("加载基础模型（无Adapter）用于特征提取...")
    from llamafactory.model import load_model, load_tokenizer
    
    # 加载分词器
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    
    # 加载基础模型
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = load_model(tokenizer, model_args, finetuning_args_base)
    model.to(device)
    model.eval()
    
    # 3. 准备特征提取器
    logger.info("初始化特征提取器...")
    # 创建一个简单的Trainer对象，用于特征提取器
    dummy_training_args = TrainingArguments(
        output_dir=multi_adapter_dir,
        per_device_eval_batch_size=batch_size
    )
    dummy_trainer = Trainer(
        model=model,
        args=dummy_training_args
    )
    
    # 初始化特征提取器
    extractor = ABSCLFeatureExtractor(
        model=model,
        trainer=dummy_trainer,
        stats_path=stats_path,
        task_id="selector_temp",
        device=device
    )
    
    # 4. 扫描可用的Adapter
    logger.info(f"扫描可用Adapter: {multi_adapter_dir}")
    task_to_adapter = find_adapters_in_dir(multi_adapter_dir)
    
    if not task_to_adapter:
        raise ValueError(f"在{multi_adapter_dir}中未找到可用的Adapter")
        
    logger.info(f"找到{len(task_to_adapter)}个可用Adapter: {list(task_to_adapter.keys())}")
    
    # 5. 加载测试数据集
    logger.info(f"加载测试数据集: {dataset_path}")
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            # 处理可能的数据格式
            if isinstance(dataset, dict) and "examples" in dataset:
                samples = dataset["examples"]
            elif isinstance(dataset, list):
                samples = dataset
            else:
                samples = [dataset]  # 单个样本
    except Exception as e:
        raise ValueError(f"加载数据集失败: {str(e)}")
    
    logger.info(f"测试集包含{len(samples)}个样本")
    
    # 6. 为每个样本选择最佳Adapter
    logger.info("开始为每个样本选择最佳Adapter...")
    adapter_assignments = defaultdict(lambda: {"path": None, "indices": []})
    
    # 将样本分批处理，提高效率
    sample_dataset = SimpleDataset(samples, tokenizer)
    dataloader = DataLoader(
        sample_dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: {k: torch.cat([b[k] for b in batch]) for k in batch[0]}
    )
    
    # 移动协方差逆矩阵到正确设备
    cov_inv = cov_inv.to(device)
    # 将任务均值移到正确设备
    task_means = {k: v.to(device) for k, v in task_means.items()}
    
    sample_idx = 0
    for batch in tqdm(dataloader, desc="处理样本批次", disable=False):
        # 将batch移至设备
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 提取特征
        with torch.no_grad():
            hidden_states = extractor._forward_and_get_hidden_states(batch)
            
            if hidden_states is None:
                logger.warning(f"批次{sample_idx//batch_size}无法提取特征，使用默认Adapter")
                # 对批次中的每个样本使用默认Adapter（第一个可用的）
                default_task_id = list(task_to_adapter.keys())[0]
                default_adapter_path = task_to_adapter[default_task_id]
                
                for i in range(min(batch_size, len(samples) - sample_idx)):
                    current_idx = sample_idx + i
                    if current_idx < len(samples):
                        adapter_assignments[default_task_id]["path"] = default_adapter_path
                        adapter_assignments[default_task_id]["indices"].append(current_idx)
                
                sample_idx += min(batch_size, len(samples) - sample_idx)
                continue
            
            # 确保hidden_states在正确设备上
            if hidden_states.device != device:
                hidden_states = hidden_states.to(device)
                
            # 提取每个样本的特征
            if "attention_mask" in batch:
                # 找到每个样本中最后一个非填充token的位置
                seq_lengths = batch["attention_mask"].sum(dim=1) - 1
                batch_size_actual = hidden_states.size(0)
                
                # 获取每个样本的最后一个token的隐藏状态
                features = torch.stack(
                    [hidden_states[i, seq_lengths[i]] for i in range(batch_size_actual)]
                )
            else:
                # 如果没有attention_mask，就使用每个序列的最后一个token
                features = hidden_states[:, -1]
            
            # 为批次中的每个样本计算与各任务均值的距离
            for i in range(features.size(0)):
                if sample_idx >= len(samples):
                    break
                    
                feature = features[i]
                
                # 计算与每个任务均值的Mahalanobis距离
                min_distance = float('inf')
                best_task_id = None
                
                for task_id, mean in task_means.items():
                    # 确认任务ID在可用Adapter中
                    if task_id not in task_to_adapter:
                        continue
                        
                    distance = mahalanobis_distance(feature, mean, cov_inv)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_task_id = task_id
                
                # 如果没有找到最佳任务，使用第一个可用的
                if best_task_id is None:
                    best_task_id = list(task_to_adapter.keys())[0]
                    logger.warning(f"样本{sample_idx}无法找到最佳任务，使用默认任务: {best_task_id}")
                    
                # 记录样本到任务的分配
                adapter_path = task_to_adapter[best_task_id]
                adapter_assignments[best_task_id]["path"] = adapter_path
                adapter_assignments[best_task_id]["indices"].append(sample_idx)
                
                sample_idx += 1
    
    # 7. 生成配置文件
    logger.info("生成Adapter选择配置文件...")
    output_config = {
        "task_name": task_name,
        "adapters": dict(adapter_assignments)
    }
    
    with open(output_config_path, "w", encoding="utf-8") as f:
        json.dump(output_config, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Adapter选择配置已保存到: {output_config_path}")
    
    # 打印统计信息
    for task_id, data in adapter_assignments.items():
        logger.info(f"任务 {task_id}: 分配了 {len(data['indices'])} 个样本")
    
    # 释放资源
    del model
    del extractor
    torch.cuda.empty_cache()

def main():
    """
    命令行入口点
    """
    import argparse
    from llamafactory.hparams import get_eval_args
    
    parser = argparse.ArgumentParser(description="ABSCL Adapter选择器")
    parser.add_argument("--config-file", type=str, required=True, help="配置文件路径")
    parser.add_argument("--dataset-path", type=str, required=True, help="测试数据集路径")
    parser.add_argument("--multi-adapter-dir", type=str, help="多Adapter目录，可覆盖配置文件中的设置")
    parser.add_argument("--batch-size", type=int, default=16, help="特征提取批处理大小")
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config_file, "r") as f:
        config_dict = json.load(f)
        
    # 解析主要参数
    model_args, data_args, eval_args, finetuning_args = get_eval_args(config_dict)
    
    # 如果命令行提供了multi_adapter_dir，则覆盖配置
    if args.multi_adapter_dir:
        eval_args.multi_adapter_dir = args.multi_adapter_dir
        
    if not eval_args.multi_adapter_dir:
        raise ValueError("未指定multi_adapter_dir，可通过配置文件或--multi-adapter-dir参数提供")
        
    # 运行选择器
    select_adapter(
        model_args=model_args,
        data_args=data_args,
        training_args=eval_args,
        finetuning_args=finetuning_args,
        dataset_path=args.dataset_path,
        multi_adapter_dir=eval_args.multi_adapter_dir,
        task_name=eval_args.task,
        batch_size=args.batch_size
    )
    
if __name__ == "__main__":
    main()
