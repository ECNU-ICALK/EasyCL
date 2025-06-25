import os
import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING, Dict, Any, Optional, List, Union
from dataclasses import dataclass

from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from .dynamic_conpet import DatasetClassifier, compute_dataset_classification_loss, save_classifier, load_classifier
from .dynamic_conpet import is_distributed, get_rank, is_main_process
from llamafactory.extras.logging import get_logger
from easycl.hparams import CLFinetuningArguments
def debugprint(*args, **kwargs):
    pass

if TYPE_CHECKING:
    from transformers import ProcessorMixin

logger = get_logger(__name__)


class DynamicConPetTrainer(CustomSeq2SeqTrainer):
    """
    Dynamic ConPet trainer with dataset classification head training capability
    """
    def __init__(
        self,
        finetuning_args,
        cl_finetuning_args,
        processor: Optional["ProcessorMixin"] = None,
        gen_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize Dynamic ConPet Trainer"""
        debugprint("开始初始化 DynamicConPetTrainer")
        
        # Extract required parameters from kwargs
        self.dataset_classifier = kwargs.pop('dataset_classifier', None)
        self.dataset_names = kwargs.pop('dataset_names', [])
        self.dataset_indices_map = kwargs.pop('dataset_indices_map', {})

        # Store cl_finetuning_args
        self.cl_finetuning_args = cl_finetuning_args
        debugprint(f"获取到的 cl_finetuning_args: {self.cl_finetuning_args}")

        # Read classification loss weight from args, default to 1.0
        self.classification_loss_weight = getattr(self.cl_finetuning_args, "classification_loss_weight", 1.0)
        debugprint(f"分类损失权重设置为: {self.classification_loss_weight} (来自 cl_finetuning_args 或默认为 1.0)")

        super().__init__(
            finetuning_args=finetuning_args,
            processor=processor,
            gen_kwargs=gen_kwargs,
            **kwargs
        )

        # Log classifier information
        if self.dataset_classifier is not None:
            # logger.info(f"Initialized DynamicConPetTrainer with dataset classifier for {self.dataset_classifier.num_datasets} datasets")
            debugprint(f"已使用数据集分类器初始化 DynamicConPetTrainer，处理 {self.dataset_classifier.num_datasets} 个数据集")
            # logger.info(f"Dataset names: {self.dataset_names}")
            debugprint(f"数据集名称: {self.dataset_names}")
            # logger.info(f"Classification loss weight: {self.classification_loss_weight}")
            debugprint(f"分类损失权重: {self.classification_loss_weight}")
        else:
            debugprint("未使用数据集分类器初始化 DynamicConPetTrainer")
        debugprint("DynamicConPetTrainer 初始化完成")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override loss computation to add dataset classification loss
        Handles distributed training by properly synchronizing losses
        """
        rank = get_rank()
        debugprint(f"进程 rank={rank} 进入 compute_loss 方法")
        # Check for dataset labels
        dataset_labels = inputs.pop("dataset_labels", None)
        debugprint(f"进程 rank={rank} 提取到的 dataset_labels: {dataset_labels}")

        # Remove 'index' from inputs before passing to model
        inputs.pop("index", None)

        # Call original loss computation
        debugprint(f"进程 rank={rank} 调用原始模型的损失计算")
        original_outputs = model(**inputs, output_hidden_states=True)
        original_loss = original_outputs.loss
        debugprint(f"进程 rank={rank} 原始损失值: {original_loss.item() if original_loss is not None else 'None'}")

        # If no classifier or dataset labels, return only original loss
        if self.dataset_classifier is None or dataset_labels is None:
            debugprint(f"进程 rank={rank} 无分类器或数据集标签，仅返回原始损失")
            if return_outputs:
                return original_loss, original_outputs
            return original_loss

        # Compute dataset classification loss
        # compute_dataset_classification_loss already handles distributed reduction
        debugprint(f"进程 rank={rank} 计算数据集分类损失")
        classification_loss = compute_dataset_classification_loss(
            original_outputs,
            dataset_labels,
            self.dataset_classifier
        )
        debugprint(f"进程 rank={rank} 分类损失权重: {self.classification_loss_weight}")
        debugprint(f"进程 rank={rank} 计算出的分类损失: {classification_loss.item()}")

        # Combine losses
        total_loss = original_loss + self.classification_loss_weight * classification_loss
        debugprint(f"进程 rank={rank} 总损失 (原始损失 + 分类损失 * 权重): {total_loss.item()}")

        # Save to logs (only in main process to avoid duplicate logs)
        if is_main_process():
            log_data = {
                "original_loss": original_loss.detach().float().item(),
                "classification_loss": classification_loss.detach().float().item(),
                "total_loss": total_loss.detach().float().item()
            }
            debugprint(f"进程 rank={rank} 记录日志数据: {log_data}")
            self.log(log_data)

        debugprint(f"进程 rank={rank} compute_loss 方法结束")
        if return_outputs:
            return total_loss, original_outputs
        return total_loss

    def save_model(self, *args, **kwargs):
        """
        Save model and dataset classifier
        In distributed training, only the main process (rank 0) should save
        """
        rank = get_rank()
        debugprint(f"进程 rank={rank} 进入 save_model 方法")

        # Call original save method (CustomSeq2SeqTrainer already handles distributed saving)
        output = super().save_model(*args, **kwargs)
        debugprint(f"进程 rank={rank} 原始 save_model 返回: {output}")

        # Save dataset classifier (only in main process)
        if self.dataset_classifier is not None and self.args.should_save:
            classifier_save_path = os.path.join(self.args.output_dir, "dataset_classifier")
            debugprint(f"进程 rank={rank} 准备保存数据集分类器到: {classifier_save_path}")
            # save_classifier already checks if this is the main process
            save_classifier(self.dataset_classifier, classifier_save_path, self.dataset_names)
        else:
            debugprint(f"进程 rank={rank} 不保存数据集分类器 (可能未提供或 should_save 为 False)")

        # Wait for all processes to reach this point
        if is_distributed():
            torch.distributed.barrier()
            debugprint(f"进程 rank={rank} 在 save_model 结束时等待所有进程")

        debugprint(f"进程 rank={rank} save_model 方法结束")
        return output

    def get_batch_dataset_labels(self, batch_indices: List[int]) -> torch.Tensor:
        """
        Get dataset labels for batch indices

        Args:
            batch_indices: List of sample indices in the batch

        Returns:
            dataset_labels: Dataset label tensor
        """
        rank = get_rank()
        debugprint(f"进程 rank={rank} 进入 get_batch_dataset_labels, batch_indices: {batch_indices}")
        labels = []
        for idx in batch_indices:
            # Find which dataset the current index belongs to
            dataset_idx = -1 # Initialize with -1 to indicate not found initially
            found_dataset = False
            # Iterate through sorted map items to ensure correct range checking
            sorted_map_items = sorted(self.dataset_indices_map.items(), key=lambda item: item[0][0])
            debugprint(f"进程 rank={rank} 样本索引 {idx}, 查找数据集，数据集索引映射: {sorted_map_items}")
            for (start_idx, end_idx), ds_idx in sorted_map_items:
                if start_idx <= idx < end_idx:
                    dataset_idx = ds_idx
                    found_dataset = True
                    debugprint(f"进程 rank={rank} 样本索引 {idx} 属于数据集索引 {ds_idx} (范围 [{start_idx}, {end_idx}))")
                    break
            if not found_dataset:
                 # Default to the last dataset index if map is not empty, otherwise 0
                 default_idx = list(self.dataset_indices_map.values())[-1] if self.dataset_indices_map else 0
                 debugprint(f"进程 rank={rank} 警告: 样本索引 {idx} 未在 dataset_indices_map 中找到，默认使用索引 {default_idx}")
                 dataset_idx = default_idx

            labels.append(dataset_idx)

        result_tensor = torch.tensor(labels, device=self.args.device)
        debugprint(f"进程 rank={rank} 生成的 dataset_labels 张量: {result_tensor}")
        return result_tensor

    def _prepare_inputs(self, inputs):
        """
        Prepare inputs by adding dataset labels for each batch
        Works in both distributed and non-distributed settings
        """
        rank = get_rank()
        debugprint(f"进程 rank={rank} 进入 _prepare_inputs 方法")
        # Original input preparation
        prepared_inputs = super()._prepare_inputs(inputs)
        debugprint(f"进程 rank={rank} 完成原始 _prepare_inputs 调用")

        # Check if dataset labels need to be added
        if self.dataset_classifier is not None and self.dataset_indices_map:
            debugprint(f"进程 rank={rank} 需要添加数据集标签")
            # Get batch indices
            if "index" in prepared_inputs: # Check in prepared_inputs which is the result of super call
                batch_indices = prepared_inputs.pop("index")
                debugprint(f"进程 rank={rank} 提取到的批次索引: {batch_indices}")
                # Generate dataset labels
                dataset_labels = self.get_batch_dataset_labels(batch_indices)
                prepared_inputs["dataset_labels"] = dataset_labels
                debugprint(f"进程 rank={rank} 已添加 dataset_labels 到准备好的输入中")
            else:
                debugprint(f"进程 rank={rank} 警告: 在准备好的输入中未找到 'index'，无法添加数据集标签")
        else:
            debugprint(f"进程 rank={rank} 不需要添加数据集标签 (无分类器或无索引映射)")

        debugprint(f"进程 rank={rank} _prepare_inputs 方法结束")
        return prepared_inputs