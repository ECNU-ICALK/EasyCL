import os
import torch
import numpy as np
import json
from tqdm import tqdm
import gc
from typing import Dict, List, Optional, Tuple, Any
from torch.utils.data import DataLoader
from transformers import Trainer
from peft import PeftModel
def debugprint(*args, **kwargs):
    pass
import torch.distributed as dist

from llamafactory.extras.logging import get_logger
from easycl.hparams import CLFinetuningArguments
from easycl.cl.distributed_utils import (
    is_distributed, is_main_process, get_rank, get_world_size,
    get_deepspeed_zero_stage, is_deepspeed_zero3_enabled,
    gather_parameters, all_reduce_tensor, broadcast_object
)

logger = get_logger(__name__)

class ABSCLFeatureExtractor:
    """
    Class for extracting and managing ABSCL feature statistics
    """

    def __init__(
        self,
        model: PeftModel,
        trainer: Trainer,
        stats_path: str,
        task_id: str,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.trainer = trainer
        self.stats_path = stats_path
        self.task_id = task_id
        self.device = device or trainer.args.device
        debugprint(f"ABSCL特征提取器初始化: 任务ID={task_id}, 统计路径={stats_path}, 设备={self.device}")
        logger.info_rank0(f"ABSCLFeatureExtractor initialized: task_id={task_id}, stats_path={stats_path}, device={self.device}") # 添加 logger

        # Create statistics directory
        os.makedirs(stats_path, exist_ok=True)
        logger.info_rank0(f"Feature statistics will be saved to: {stats_path}")

        # Load existing statistics
        self.stats = self._load_existing_stats()

        # Get feature dimension
        # Note: Assumes the model is a pretrained Transformers model
        if hasattr(model, "model"):
            # For PeftModel
            if hasattr(model.model, "config"):
                self.hidden_size = model.model.config.hidden_size
            else:
                # Try to get from base_model_prefix
                base_model = getattr(model.model, model.model.base_model_prefix, model.model)
                self.hidden_size = base_model.config.hidden_size
        else:
            # Get directly from model
            self.hidden_size = model.config.hidden_size

        logger.info_rank0(f"Feature dimension: {self.hidden_size}")

    def _load_existing_stats(self) -> Dict:
        """
        Load existing statistics
        """
        stats_file = os.path.join(self.stats_path, "abscl_stats.pt")

        # 只在主进程中加载统计信息，然后广播给其他进程
        if is_main_process():
            if os.path.exists(stats_file):
                logger.info_rank0(f"Loading existing statistics: {stats_file}")
                stats = torch.load(stats_file)

                # Log existing tasks
                logger.info_rank0(f"Existing task statistics: {list(stats['task_means'].keys())}")
            else:
                logger.info_rank0(f"No existing statistics found, creating new record")
                # Initialize statistics
                stats = {
                    "task_means": {},  # Store feature means for each task
                    "cov_matrix": None,  # Shared covariance matrix
                    "n_samples": 0,  # Total number of samples
                    "tasks_info": {}  # Additional information for each task
                }
        else:
            # 非主进程初始化为None，等待广播
            stats = None

        # 在分布式环境中广播统计信息
        if is_distributed():
            stats = broadcast_object(stats)

        return stats

    def _extract_features(self, dataloader: DataLoader) -> torch.Tensor:
        """
        Extract features from data

        Args:
            dataloader: Data loader

        Returns:
            features: Extracted feature matrix, shape [n_samples, hidden_size]
        """
        self.model.eval()  # Set to evaluation mode
        features = []

        logger.info_rank0("Starting feature extraction...") # 取消注释并使用英文
        debugprint("开始特征提取...")

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                # Move batch data to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                # Add hook to get hidden states from second-to-last layer
                hidden_states = self._forward_and_get_hidden_states(batch)

                if hidden_states is not None:
                    # For each sample, take the hidden state of the last token as feature
                    # Use attention mask to find the last non-padding token
                    if "attention_mask" in batch:
                        # Find position of last non-padding token for each sample
                        seq_lengths = batch["attention_mask"].sum(dim=1) - 1
                        batch_size = hidden_states.size(0)

                        # Get hidden state of last token for each sample
                        last_hidden = torch.stack(
                            [hidden_states[i, seq_lengths[i]] for i in range(batch_size)]
                        )
                    else:
                        # If no attention_mask, use last token of each sequence
                        last_hidden = hidden_states[:, -1]

                    features.append(last_hidden.cpu())

        if features:
            # Concatenate all features
            features = torch.cat(features, dim=0)
            logger.info_rank0(f"Feature extraction successful, shape: {features.shape}") # 取消注释并使用英文
            debugprint(f"特征提取成功，形状: {features.shape}")
            return features
        else:
            logger.info_rank0("Feature extraction failed, returning empty tensor") # error -> info
            debugprint("特征提取失败，返回空张量")
            return torch.zeros((0, self.hidden_size))

    def _forward_and_get_hidden_states(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        """
        Perform forward pass and get hidden states
        """
        # Remove unnecessary keys to avoid model errors
        batch_for_model = {k: v for k, v in batch.items()
                          if k in ["input_ids", "attention_mask", "token_type_ids"]}

        try:
            # 检查DeepSpeed ZeRO-3环境
            zero_stage = get_deepspeed_zero_stage(self.model)
            is_zero3 = zero_stage == 3

            debugprint(f"前向传播中检测到DeepSpeed ZeRO阶段: {zero_stage}")

            # 在ZeRO-3环境下，需要使用gather_parameters上下文管理器
            if is_zero3:
                logger.info_rank0("Using gather_parameters context for ZeRO-3 in forward pass")
                debugprint("在前向传播中使用ZeRO-3的gather_parameters上下文")
                # 在ZeRO-3环境下，使用gather_parameters上下文管理器临时收集完整参数
                with gather_parameters(self.model):
                    outputs = self.model(
                        **batch_for_model,
                        output_hidden_states=True,
                        return_dict=True
                    )
            else:
                # 非ZeRO-3环境，直接执行前向传播
                debugprint("在非ZeRO-3环境中执行标准前向传播")
                outputs = self.model(
                    **batch_for_model,
                    output_hidden_states=True,
                    return_dict=True
                )

            # Get hidden states from second-to-last layer
            if outputs.hidden_states is not None:
                # Usually, second-to-last layer is the layer before the final layer
                # hidden_states typically includes [embedding layer, middle layers..., final layer]
                debugprint(f"获取倒数第二层隐藏状态，隐藏状态总层数: {len(outputs.hidden_states)}")
                return outputs.hidden_states[-2]
            else:
                logger.warning_rank0("No hidden states in model outputs")
                debugprint("模型输出中没有隐藏状态")
                return None

        except Exception as e:
            logger.info_rank0(f"Error during forward pass: {str(e)}")
            debugprint(f"前向传播过程中出错: {str(e)}")
            return None

    def compute_feature_statistics(self, dataset) -> Dict:
        """
        Compute feature statistics

        Args:
            dataset: Dataset

        Returns:
            stats: Updated feature statistics
        """
        logger.info_rank0(f"Starting to compute feature statistics for task {self.task_id}")
        debugprint(f"开始计算任务 {self.task_id} 的特征统计信息")

        # Create data loader
        dataloader = self.trainer.get_train_dataloader()
        if dataset:
            # If specific dataset provided, create new data loader
            dataloader = DataLoader(
                dataset,
                batch_size=self.trainer.args.per_device_train_batch_size,
                collate_fn=self.trainer.data_collator,
                shuffle=False
            )

        # Extract features
        debugprint("开始从数据加载器中提取特征")
        features = self._extract_features(dataloader)

        if features.shape[0] == 0:
            logger.info_rank0("Could not extract features, skipping statistics computation")
            debugprint("无法提取特征，跳过统计计算")
            return self.stats

        # 在分布式环境中，需要收集所有进程的特征
        if is_distributed():
            debugprint("在分布式环境中收集特征统计信息")
            # 将特征张量移动到正确的设备
            features = features.to(self.device)
            debugprint(f"Features tensor moved to device: {features.device}")

            # 获取每个进程的特征数量
            local_count = torch.tensor([features.shape[0]], dtype=torch.long, device=self.device) # 确保在GPU上创建
            # 收集所有进程的特征数量
            # 注意: torch.distributed.all_reduce 要求张量在GPU上
            dist.all_reduce(local_count) # 直接使用 dist.all_reduce
            total_count = local_count.item()
            debugprint(f"分布式环境中的样本总数: {total_count}")

            # 计算本地特征的均值
            local_mean = torch.mean(features, dim=0) # 在GPU上计算
            local_n_samples = features.shape[0]
            debugprint(f"本地进程的样本数: {local_n_samples}, 本地均值设备: {local_mean.device}")

            # 在所有进程间同步均值 (确保 all_reduce_tensor 处理GPU张量)
            # 传递需要在GPU上的张量 local_mean * local_n_samples
            global_mean = all_reduce_tensor(local_mean * local_n_samples) / total_count
            debugprint(f"全局特征均值范围: [{torch.min(global_mean).item():.4f}, {torch.max(global_mean).item():.4f}], 设备: {global_mean.device}")

            # 计算本地特征的协方差贡献
            centered_feats = features - global_mean.unsqueeze(0) # 在GPU上计算
            local_cov = torch.matmul(centered_feats.t(), centered_feats) # 在GPU上计算
            debugprint(f"本地协方差贡献设备: {local_cov.device}")

            # 在所有进程间同步协方差 (确保 all_reduce_tensor 处理GPU张量)
            # 传递需要在GPU上的张量 local_cov
            global_cov = all_reduce_tensor(local_cov) / total_count
            debugprint(f"全局协方差矩阵形状: {global_cov.shape}, 设备: {global_cov.device}")

            # 使用全局统计信息更新
            task_mean = global_mean # 仍在GPU上
            task_cov = global_cov # 仍在GPU上
            n_samples = total_count
        else:
            # 非分布式环境，直接计算
            debugprint("在非分布式环境中直接计算特征统计信息")
            task_mean = torch.mean(features, dim=0)
            n_samples = features.shape[0]
            debugprint(f"样本数: {n_samples}")
            centered_feats = features - task_mean.unsqueeze(0)
            task_cov = torch.matmul(centered_feats.t(), centered_feats) / n_samples
            # 确保在非分布式时将 task_cov 移到 self.device (如果 features 不在 self.device)
            # 但这里 features 应该就是CPU，所以 task_cov 也是 CPU，后面 .cpu() 不会报错
            debugprint(f"特征均值范围: [{torch.min(task_mean).item():.4f}, {torch.max(task_mean).item():.4f}]")
            debugprint(f"协方差矩阵形状: {task_cov.shape}")

        # 更新统计信息
        debugprint(f"更新任务 {self.task_id} 的特征统计信息")
        # 保存时移回CPU
        self.stats["task_means"][self.task_id] = task_mean.cpu()
        self.stats["tasks_info"][self.task_id] = {
            "n_samples": n_samples
        }

        # 更新共享协方差矩阵
        if self.stats["cov_matrix"] is None:
            # 第一次计算协方差
            debugprint("首次计算协方差矩阵")
            self.stats["cov_matrix"] = task_cov.cpu()
        else:
            # 更新协方差矩阵（加权更新）
            total_samples = self.stats["n_samples"] + n_samples
            old_weight = self.stats["n_samples"] / total_samples
            new_weight = n_samples / total_samples
            debugprint(f"更新协方差矩阵 - 旧权重: {old_weight:.4f}, 新权重: {new_weight:.4f}")

            # 确保旧的协方差矩阵和新的都在CPU上进行计算和更新
            # 如果 self.stats["cov_matrix"] 是 None，则 task_cov 已经是CPU上的了
            # 如果不是 None，它已经是 CPU 上的了
            self.stats["cov_matrix"] = (old_weight * self.stats["cov_matrix"] +
                                        new_weight * task_cov.cpu()) # 确保 task_cov 移回CPU

        # 更新总样本数
        old_samples = self.stats["n_samples"]
        self.stats["n_samples"] += n_samples
        debugprint(f"更新总样本数: {old_samples} -> {self.stats['n_samples']}")

        # 记录一些统计信息
        logger.info_rank0(f"Task {self.task_id} feature mean range: [{torch.min(task_mean).item():.4f}, {torch.max(task_mean).item():.4f}]")
        logger.info_rank0(f"Sample count: {n_samples}, Total samples: {self.stats['n_samples']}")

        # 保存更新后的统计信息
        self.save_stats()

        return self.stats

    def save_stats(self) -> None:
        """
        Save feature statistics
        """
        debugprint("保存特征统计信息")
        # 只在主进程中保存统计信息
        if is_main_process():
            stats_file = os.path.join(self.stats_path, "abscl_stats.pt")
            torch.save(self.stats, stats_file)
            logger.info_rank0(f"Feature statistics saved to: {stats_file}")
            debugprint(f"特征统计信息已保存到: {stats_file}")

            # Also save some meta information in JSON format for easy viewing
            meta_info = {
                "tasks": list(self.stats["task_means"].keys()),
                "total_samples": int(self.stats["n_samples"]),
                "feature_dim": self.hidden_size,
                "tasks_info": {
                    task_id: {
                        "n_samples": int(info["n_samples"])
                    } for task_id, info in self.stats["tasks_info"].items()
                }
            }

            debugprint(f"元数据信息 - 任务数: {len(meta_info['tasks'])}, 总样本数: {meta_info['total_samples']}, 特征维度: {meta_info['feature_dim']}")

            meta_file = os.path.join(self.stats_path, "abscl_meta.json")
            with open(meta_file, "w") as f:
                json.dump(meta_info, f, indent=2)

            logger.info_rank0(f"Meta information saved to: {meta_file}")
            debugprint(f"元数据信息已保存到: {meta_file}")

        # 在分布式环境中等待主进程完成保存
        if is_distributed():
            debugprint("等待所有进程完成特征统计信息保存")
            torch.distributed.barrier()

def extract_feature_statistics(
    model: PeftModel,
    trainer: Trainer,
    task_id: str,
    finetuning_args=None,
    cl_finetuning_args=None,
    device: Optional[torch.device] = None,
    dataset=None
) -> None:
    """
    Extract feature statistics for ABSCL method

    Args:
        model: Model
        trainer: Trainer
        task_id: Task ID
        finetuning_args: Finetuning arguments
        cl_finetuning_args: CL finetuning arguments
        device: Device
        dataset: Dataset (optional)
    """
    logger.info_rank0(f"Entering extract_feature_statistics function, task_id: {task_id}")
    logger.info_rank0(f"Incoming cl_finetuning_args: {cl_finetuning_args}")
    debugprint(f"进入特征统计提取函数，任务ID: {task_id}")
    debugprint(f"当前CL微调参数: {cl_finetuning_args}")

    # 检查分布式环境
    is_dist = is_distributed()
    if is_dist:
        logger.info_rank0(f"Running in distributed environment with {get_world_size()} processes")
        debugprint(f"在分布式环境中运行，进程数: {get_world_size()}")

    if cl_finetuning_args is None:
        logger.warning_rank0("cl_finetuning_args not provided, skipping feature extraction")
        debugprint("未提供cl_finetuning_args参数，跳过特征提取")
        return

    # Get stats path from arguments
    stats_path = cl_finetuning_args.abscl_stats_path or os.path.join(
        os.path.dirname(trainer.args.output_dir), "abscl_stats"
    )
    logger.info_rank0(f"Statistics path (stats_path): {stats_path}")
    debugprint(f"统计路径 (stats_path): {stats_path}")

    # 在主进程中创建目录
    if is_main_process():
        os.makedirs(stats_path, exist_ok=True)

    # 在分布式环境中等待主进程创建目录
    if is_dist:
        torch.distributed.barrier()

    logger.info_rank0(f"Extracting feature statistics for task {task_id}")

    # 检查DeepSpeed ZeRO阶段
    zero_stage = get_deepspeed_zero_stage(model)
    logger.info_rank0(f"Detected DeepSpeed ZeRO Stage: {zero_stage}")
    debugprint(f"检测到DeepSpeed ZeRO阶段: {zero_stage}")

    feature_extractor = ABSCLFeatureExtractor(
        model=model,
        trainer=trainer,
        stats_path=stats_path,
        task_id=task_id,
        device=device
    )

    # Compute statistics
    feature_extractor.compute_feature_statistics(dataset)

    # Save statistics (已在compute_feature_statistics中调用)

    logger.info_rank0(f"Feature statistics extraction completed, saved to {stats_path}")
    debugprint(f"特征统计提取完成，已保存到 {stats_path}")

    # 在分布式环境中等待所有进程完成
    if is_dist:
        debugprint("等待所有分布式进程完成特征提取")
        torch.distributed.barrier()

    # Clean up to free memory
    debugprint("清理内存")
    del feature_extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        debugprint("已清空CUDA缓存")
