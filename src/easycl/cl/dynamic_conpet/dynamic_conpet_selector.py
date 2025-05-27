import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import copy

from llamafactory.extras.logging import get_logger
from .dynamic_conpet import DatasetClassifier, load_classifier
#导入为空函数作为debugprint
def debugprint(*args, **kwargs):
    pass

logger = get_logger(__name__)

class SimpleDataset(Dataset):
    """
    Simple dataset wrapper for converting sample list to DataLoader compatible format
    """
    def __init__(self, samples, tokenizer):
        debugprint(f"初始化 SimpleDataset: 样本数量 = {len(samples)}")
        self.samples = samples
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        # Extract instruction and input fields from sample
        sample = self.samples[idx]
        debugprint(f"SimpleDataset 获取样本: 索引 = {idx}, 样本内容 = {sample}")
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        
        # Combine into full text
        full_text = f"{instruction}\n{input_text}".strip()
        debugprint(f"SimpleDataset: 合并后的文本 = '{full_text[:100]}...'")
        
        # Process with tokenizer
        encoded = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        debugprint(f"SimpleDataset: Tokenizer 输出键值 = {list(encoded.keys())}")
        
        # Add sample index for later identification
        encoded["index"] = idx
        
        return encoded

def find_adapters_in_dir(base_dir: str, exclude_shared: bool = True) -> Dict[str, str]:
    """
    Scan directory for available adapters, return mapping from task ID to adapter relative path
    
    Args:
        base_dir: Base path of adapter directory
        exclude_shared: Whether to exclude shared_adapter
        
    Returns:
        adapters_map: Dictionary of {task_id: relative_path}
    """
    adapters_map = {}
    debugprint(f"开始在目录 {base_dir} 中扫描适配器, exclude_shared={exclude_shared}")
    
    if not os.path.exists(base_dir):
        # logger.warning(f"Adapter directory does not exist: {base_dir}")
        debugprint(f"警告: 适配器目录不存在: {base_dir}")
        return adapters_map
        
    # Scan subdirectories for valid adapter directories
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        debugprint(f"检查项目: {item_path}")
        
        # Skip if excluding shared adapter and current item is shared_adapter
        if exclude_shared and item == "shared_adapter":
            debugprint(f"跳过共享适配器: {item}")
            continue
            
        # Verify this is a valid adapter directory
        adapter_config_path = os.path.join(item_path, "adapter_config.json")
        if os.path.isdir(item_path) and os.path.exists(adapter_config_path):
            # Use directory name as task ID (usually directory name is task ID)
            task_id = item
                
            # Record mapping from task ID to relative path
            adapters_map[task_id] = item
            # logger.info(f"Found adapter: {task_id} -> {item}")
            debugprint(f"找到适配器: {task_id} -> {item}")
        else:
            debugprint(f"跳过无效的适配器目录: {item_path}")
    
    debugprint(f"扫描完成，找到的适配器映射: {adapters_map}")
    return adapters_map

def select_adapter_dynamic_conpet(
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
    Use Dynamic ConPet classification head to select the most suitable adapter for each sample
    
    Args:
        model_args: Model parameters
        data_args: Data parameters
        training_args: Training parameters
        finetuning_args: Fine-tuning parameters
        dataset_path: Test dataset path
        multi_adapter_dir: Multi-adapter directory path
        task_name: Current evaluation task name
        device: Computation device (should be training_args.device for rank 0)
        batch_size: Batch size
    """
    debugprint("进入 select_adapter_dynamic_conpet 函数")

    # 仅在主进程 (rank 0) 或单进程 (local_rank 为 -1 或 0) 上执行
    # training_args.local_rank (来自于 eval_args) > 0 表示当前不是主进程
    if training_args.local_rank > 0:
        logger.info(f"Dynamic ConPet Selector: Skipping adapter selection on rank {training_args.local_rank} (not main process).")
        return

    # 使用传入的 device 参数作为当前操作设备
    # 这个 device 参数通常是 training_args.device (例如 cuda:0 for rank 0)
    current_device = device
    logger.info(f"Dynamic ConPet Selector: Running on device: {current_device} (local_rank: {training_args.local_rank})")
    if current_device is None: # 如果没有明确提供device，则基于可用性选择，但这通常应由调用者（如main）基于training_args.device设置
        current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.warning(f"Dynamic ConPet Selector: Device was None, fallback to {current_device}. Ensure training_args.device is correctly passed.")

    # Ensure output directory exists
    os.makedirs(multi_adapter_dir, exist_ok=True)
    output_config_path = os.path.join(multi_adapter_dir, "multiadapter_selected_config.json")
    
    # logger.info(f"Starting Dynamic ConPet adapter selection, output config will be saved to: {output_config_path}")
    debugprint(f"启动 Dynamic ConPet 适配器选择，输出配置将保存到: {output_config_path}")
    
    # 1. Load shared adapter and classifier
    shared_adapter_path = os.path.join(multi_adapter_dir, "shared_adapter")
    classifier_path = os.path.join(shared_adapter_path, "dataset_classifier")
    debugprint(f"共享适配器路径: {shared_adapter_path}")
    debugprint(f"分类器路径: {classifier_path}")
    
    if not os.path.exists(shared_adapter_path):
        raise ValueError(f"Shared adapter directory does not exist: {shared_adapter_path}")
    
    if not os.path.exists(classifier_path):
        raise ValueError(f"Classifier directory does not exist: {classifier_path}")
    
    # 2. Load base model (load shared adapter)
    # Create finetuning_args copy and set adapter_name_or_path to shared adapter
    finetuning_args_base = copy.deepcopy(finetuning_args)
    finetuning_args_base.adapter_name_or_path = [shared_adapter_path]
    
    # logger.info(f"Loading base model and shared adapter: {shared_adapter_path}")
    debugprint(f"加载基础模型和共享适配器: {shared_adapter_path}")
    from llamafactory.model import load_model, load_tokenizer
    
    # Load tokenizer
    debugprint("加载 tokenizer")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    
    # Load base model and shared adapter
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    debugprint(f"使用的设备: {device}")
    debugprint("加载模型")
    model = load_model(tokenizer, model_args, finetuning_args_base)
    model.to(current_device) # 确保模型在目标设备上
    model.eval()
    debugprint("模型加载完成并设置为评估模式")
    
    # 3. Load classifier
    # logger.info(f"Loading dataset classifier: {classifier_path}")
    debugprint(f"加载数据集分类器: {classifier_path}")
    dataset_classifier, dataset_names = load_classifier(
        classifier_path, 
        model.config.hidden_size
    )
    
    if dataset_classifier is None:
        raise ValueError(f"Failed to load dataset classifier: {classifier_path}")
    
    dataset_classifier.to(current_device) # 确保分类器在目标设备上
    dataset_classifier.eval()
    debugprint("数据集分类器加载完成并设置为评估模式")
    
    # logger.info(f"Classifier supports datasets: {dataset_names}")
    debugprint(f"分类器支持的数据集: {dataset_names}")
    
    # 4. Scan available adapters
    # logger.info(f"Scanning available adapters: {multi_adapter_dir}")
    debugprint(f"扫描可用适配器: {multi_adapter_dir}")
    task_to_adapter = find_adapters_in_dir(multi_adapter_dir)
    
    if not task_to_adapter:
        raise ValueError(f"No available adapters found in {multi_adapter_dir}")
    
    # logger.info(f"Found {len(task_to_adapter)} available adapters: {list(task_to_adapter.keys())}")
    debugprint(f"找到 {len(task_to_adapter)} 个可用适配器: {list(task_to_adapter.keys())}")
    
    # Check if there is an adapter named "current_task"
    current_task_id = None
    for task_id in task_to_adapter.keys():
        if task_id == "current_task" or task_id == task_name:
            current_task_id = task_id
            debugprint(f"找到当前任务适配器ID: {current_task_id}")
            break
    if current_task_id is None:
        debugprint("未找到名为 'current_task' 或与任务名匹配的适配器")
    
    # 5. Dataset name to task ID mapping
    dataset_to_adapter = {}
    debugprint("创建数据集名称到适配器任务ID的映射")
    
    # Create mapping from dataset names to adapter task IDs
    for idx, dataset_name in enumerate(dataset_names):
        # First check for exact match with adapter
        if dataset_name in task_to_adapter:
            dataset_to_adapter[idx] = dataset_name
            debugprint(f"数据集索引 {idx} ('{dataset_name}') 精确匹配到适配器任务ID: '{dataset_name}'")
        else:
            # If no exact match, try to find partial match
            matched = False
            for task_id in task_to_adapter.keys():
                # Check if task ID is substring of dataset name or vice versa
                if task_id in dataset_name or dataset_name in task_id:
                    dataset_to_adapter[idx] = task_id
                    matched = True
                    debugprint(f"数据集索引 {idx} ('{dataset_name}') 部分匹配到适配器任务ID: '{task_id}'")
                    break
            
            # If no match, use current_task (if available)
            if not matched:
                if current_task_id:
                    dataset_to_adapter[idx] = current_task_id
                    debugprint(f"数据集索引 {idx} ('{dataset_name}') 无匹配，使用当前任务适配器ID: '{current_task_id}'")
                else:
                    # If no current_task, use first available adapter
                    first_adapter_id = list(task_to_adapter.keys())[0]
                    dataset_to_adapter[idx] = first_adapter_id
                    debugprint(f"数据集索引 {idx} ('{dataset_name}') 无匹配且无当前任务适配器，使用第一个可用适配器ID: '{first_adapter_id}'")
    
    # logger.info(f"Dataset to adapter mapping: {dataset_to_adapter}")
    debugprint(f"数据集到适配器的最终映射: {dataset_to_adapter}")
    
    # 6. Load test dataset
    # logger.info(f"Loading test dataset: {dataset_path}")
    debugprint(f"加载测试数据集: {dataset_path}")
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            # Handle possible data formats
            if isinstance(dataset, dict) and "examples" in dataset:
                samples = dataset["examples"]
                debugprint("数据集格式为字典，包含 'examples' 键")
            elif isinstance(dataset, list):
                samples = dataset
                debugprint("数据集格式为列表")
            else:
                samples = [dataset]  # Single sample
                debugprint("数据集格式为单个样本，包装为列表")
    except Exception as e:
        raise ValueError(f"Failed to load dataset: {str(e)}")
    
    # logger.info(f"Test set contains {len(samples)} samples")
    debugprint(f"测试集包含 {len(samples)} 个样本")
    
    # 7. Select best adapter for each sample
    # logger.info("Starting best adapter selection for each sample...")
    debugprint("开始为每个样本选择最佳适配器...")
    adapter_assignments = defaultdict(lambda: {"path": None, "indices": []})
    
    # Process samples in batches for efficiency
    sample_dataset = SimpleDataset(samples, tokenizer)
    dataloader = DataLoader(
        sample_dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: {k: torch.cat([b[k] for b in batch]) if k != "index" else [b[k] for b in batch] for k in batch[0]}
    )
    debugprint(f"创建 DataLoader: batch_size={batch_size}")
    
    sample_idx = 0
    for batch_num, batch in enumerate(tqdm(dataloader, desc="Processing sample batches")):
        debugprint(f"处理批次 {batch_num + 1}/{len(dataloader)}")
        # Extract sample index list
        indices = batch.pop("index")
        debugprint(f"批次中的样本原始索引: {indices}")
        
        # Move batch to device
        batch = {k: v.to(current_device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass to get hidden states
        with torch.no_grad():
            debugprint("模型前向传播获取 hidden_states")
            outputs = model(**batch, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
            debugprint(f"获取到的 hidden_states 形状: {hidden_states.shape}")
            
            # Use classifier to predict dataset class
            debugprint("使用分类器预测数据集类别")
            # sequence_output = hidden_states[:, 0, :]  # Take first token representation # Redundant, classifier handles this
            logits = dataset_classifier(hidden_states.float())
            predicted_datasets = torch.argmax(logits, dim=-1)
            debugprint(f"分类器预测的 logits 形状: {logits.shape}, 预测的数据集索引: {predicted_datasets.tolist()}")
            
            # Assign adapter for each sample in batch
            for i, pred_dataset_idx in enumerate(predicted_datasets):
                pred_dataset_idx = pred_dataset_idx.item()
                original_idx = indices[i]
                debugprint(f"处理批次内样本 {i}: 原始索引={original_idx}, 预测数据集索引={pred_dataset_idx}")
                
                # Get corresponding adapter task ID based on predicted dataset
                if pred_dataset_idx in dataset_to_adapter:
                    adapter_id = dataset_to_adapter[pred_dataset_idx]
                    debugprint(f"  映射到适配器ID: '{adapter_id}' (来自 dataset_to_adapter)")
                elif current_task_id:
                    # If no mapping but have current_task, use it
                    adapter_id = current_task_id
                    debugprint(f"  无映射，使用当前任务适配器ID: '{current_task_id}'")
                else:
                    # Otherwise use first adapter
                    adapter_id = list(task_to_adapter.keys())[0]
                    debugprint(f"  无映射且无当前任务适配器，使用第一个可用适配器ID: '{adapter_id}'")
                
                # Record adapter assignment
                adapter_path = task_to_adapter[adapter_id]
                adapter_assignments[adapter_id]["path"] = adapter_path
                adapter_assignments[adapter_id]["indices"].append(original_idx)
                debugprint(f"  分配给适配器: ID='{adapter_id}', Path='{adapter_path}', 样本索引={original_idx}")
                
                sample_idx += 1
    debugprint(f"总共处理了 {sample_idx} 个样本")
    
    # 8. Generate configuration file
    # logger.info("Generating adapter selection configuration file...")
    debugprint("生成适配器选择配置文件...")
    output_config = {
        "task_name": task_name,
        "adapters": dict(adapter_assignments)
    }
    debugprint(f"最终配置内容: {json.dumps(output_config, indent=2, ensure_ascii=False)}")

    with open(output_config_path, "w", encoding="utf-8") as f:
        json.dump(output_config, f, indent=2, ensure_ascii=False)
        
    # logger.info(f"Adapter selection configuration saved to: {output_config_path}")
    debugprint(f"适配器选择配置已保存到: {output_config_path}")
    
    # Print statistics
    # logger.info("Adapter allocation statistics:")
    debugprint("适配器分配统计:")
    for task_id, data in adapter_assignments.items():
        # logger.info(f"Task {task_id}: Assigned {len(data['indices'])} samples")
        debugprint(f"  任务 {task_id}: 分配了 {len(data['indices'])} 个样本")
    
    # Free resources
    debugprint("释放模型和分类器资源")
    del model
    del dataset_classifier
    torch.cuda.empty_cache()
    debugprint("select_adapter_dynamic_conpet 函数结束")

def main():
    """
    Command line entry point
    """
    debugprint("Dynamic ConPet Selector 主程序入口")
    import argparse
    from llamafactory.hparams import get_eval_args
    
    parser = argparse.ArgumentParser(description="Dynamic ConPet Adapter Selector")
    parser.add_argument("--config-file", type=str, required=True, help="Configuration file path")
    parser.add_argument("--dataset-path", type=str, required=True, help="Test dataset path")
    parser.add_argument("--multi-adapter-dir", type=str, help="Multi-adapter directory, can override config file setting")
    parser.add_argument("--batch-size", type=int, default=16, help="Feature extraction batch size")
    
    args = parser.parse_args()
    debugprint(f"解析到的命令行参数: {args}")
    
    # Load configuration
    debugprint(f"从配置文件加载配置: {args.config_file}")
    with open(args.config_file, "r") as f:
        config_dict = json.load(f)
    debugprint(f"加载的配置字典 (部分): {list(config_dict.keys())}")
    
    # Parse main parameters
    debugprint("解析主要参数 (model_args, data_args, eval_args, finetuning_args)")
    model_args, data_args, eval_args, finetuning_args = get_eval_args(config_dict)

    # If multi_adapter_dir provided in command line, override config
    if args.multi_adapter_dir:
        debugprint(f"使用命令行提供的 multi_adapter_dir 覆盖配置: {args.multi_adapter_dir}")
        eval_args.multi_adapter_dir = args.multi_adapter_dir
        
    if not eval_args.multi_adapter_dir:
        raise ValueError("multi_adapter_dir not specified, provide through config file or --multi-adapter-dir parameter")
    debugprint(f"最终使用的 multi_adapter_dir: {eval_args.multi_adapter_dir}")
    debugprint(f"当前任务名称 (eval_args.task): {eval_args.task}")

    # Run selector
    debugprint("运行 select_adapter_dynamic_conpet")
    select_adapter_dynamic_conpet(
        model_args=model_args,
        data_args=data_args,
        training_args=eval_args, # Note: eval_args are used as training_args here
        finetuning_args=finetuning_args,
        dataset_path=args.dataset_path,
        multi_adapter_dir=eval_args.multi_adapter_dir,
        task_name=eval_args.task,
        device=eval_args.device, # 显式传递 eval_args.device
        batch_size=args.batch_size
    )
    debugprint("Dynamic ConPet Selector 主程序结束")

if __name__ == "__main__":
    main()