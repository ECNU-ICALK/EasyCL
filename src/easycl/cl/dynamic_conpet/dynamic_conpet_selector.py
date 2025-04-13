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

logger = get_logger(__name__)

class SimpleDataset(Dataset):
    """
    Simple dataset wrapper for converting sample list to DataLoader compatible format
    """
    def __init__(self, samples, tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        # Extract instruction and input fields from sample
        sample = self.samples[idx]
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        
        # Combine into full text
        full_text = f"{instruction}\n{input_text}".strip()
        
        # Process with tokenizer
        encoded = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        
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
    
    if not os.path.exists(base_dir):
        logger.warning(f"Adapter directory does not exist: {base_dir}")
        return adapters_map
        
    # Scan subdirectories for valid adapter directories
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        
        # Skip if excluding shared adapter and current item is shared_adapter
        if exclude_shared and item == "shared_adapter":
            continue
            
        # Verify this is a valid adapter directory
        adapter_config_path = os.path.join(item_path, "adapter_config.json")
        if os.path.isdir(item_path) and os.path.exists(adapter_config_path):
            # Use directory name as task ID (usually directory name is task ID)
            task_id = item
                
            # Record mapping from task ID to relative path
            adapters_map[task_id] = item
            logger.info(f"Found adapter: {task_id} -> {item}")
    
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
        device: Computation device
        batch_size: Batch size
    """
    # Ensure output directory exists
    os.makedirs(multi_adapter_dir, exist_ok=True)
    output_config_path = os.path.join(multi_adapter_dir, "multiadapter_selected_config.json")
    
    logger.info(f"Starting Dynamic ConPet adapter selection, output config will be saved to: {output_config_path}")
    
    # 1. Load shared adapter and classifier
    shared_adapter_path = os.path.join(multi_adapter_dir, "shared_adapter")
    classifier_path = os.path.join(shared_adapter_path, "dataset_classifier")
    
    if not os.path.exists(shared_adapter_path):
        raise ValueError(f"Shared adapter directory does not exist: {shared_adapter_path}")
    
    if not os.path.exists(classifier_path):
        raise ValueError(f"Classifier directory does not exist: {classifier_path}")
    
    # 2. Load base model (load shared adapter)
    # Create finetuning_args copy and set adapter_name_or_path to shared adapter
    finetuning_args_base = copy.deepcopy(finetuning_args)
    finetuning_args_base.adapter_name_or_path = [shared_adapter_path]
    
    logger.info(f"Loading base model and shared adapter: {shared_adapter_path}")
    from llamafactory.model import load_model, load_tokenizer
    
    # Load tokenizer
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    
    # Load base model and shared adapter
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = load_model(tokenizer, model_args, finetuning_args_base)
    model.to(device)
    model.eval()
    
    # 3. Load classifier
    logger.info(f"Loading dataset classifier: {classifier_path}")
    dataset_classifier, dataset_names = load_classifier(
        classifier_path, 
        model.config.hidden_size
    )
    
    if dataset_classifier is None:
        raise ValueError(f"Failed to load dataset classifier: {classifier_path}")
    
    dataset_classifier.to(device)
    dataset_classifier.eval()
    
    logger.info(f"Classifier supports datasets: {dataset_names}")
    
    # 4. Scan available adapters
    logger.info(f"Scanning available adapters: {multi_adapter_dir}")
    task_to_adapter = find_adapters_in_dir(multi_adapter_dir)
    
    if not task_to_adapter:
        raise ValueError(f"No available adapters found in {multi_adapter_dir}")
    
    logger.info(f"Found {len(task_to_adapter)} available adapters: {list(task_to_adapter.keys())}")
    
    # Check if there is an adapter named "current_task"
    current_task_id = None
    for task_id in task_to_adapter.keys():
        if task_id == "current_task" or task_id == task_name:
            current_task_id = task_id
            break
    
    # 5. Dataset name to task ID mapping
    dataset_to_adapter = {}
    
    # Create mapping from dataset names to adapter task IDs
    for idx, dataset_name in enumerate(dataset_names):
        # First check for exact match with adapter
        if dataset_name in task_to_adapter:
            dataset_to_adapter[idx] = dataset_name
        else:
            # If no exact match, try to find partial match
            matched = False
            for task_id in task_to_adapter.keys():
                # Check if task ID is substring of dataset name or vice versa
                if task_id in dataset_name or dataset_name in task_id:
                    dataset_to_adapter[idx] = task_id
                    matched = True
                    break
            
            # If no match, use current_task (if available)
            if not matched:
                if current_task_id:
                    dataset_to_adapter[idx] = current_task_id
                else:
                    # If no current_task, use first available adapter
                    dataset_to_adapter[idx] = list(task_to_adapter.keys())[0]
    
    logger.info(f"Dataset to adapter mapping: {dataset_to_adapter}")
    
    # 6. Load test dataset
    logger.info(f"Loading test dataset: {dataset_path}")
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            # Handle possible data formats
            if isinstance(dataset, dict) and "examples" in dataset:
                samples = dataset["examples"]
            elif isinstance(dataset, list):
                samples = dataset
            else:
                samples = [dataset]  # Single sample
    except Exception as e:
        raise ValueError(f"Failed to load dataset: {str(e)}")
    
    logger.info(f"Test set contains {len(samples)} samples")
    
    # 7. Select best adapter for each sample
    logger.info("Starting best adapter selection for each sample...")
    adapter_assignments = defaultdict(lambda: {"path": None, "indices": []})
    
    # Process samples in batches for efficiency
    sample_dataset = SimpleDataset(samples, tokenizer)
    dataloader = DataLoader(
        sample_dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: {k: torch.cat([b[k] for b in batch]) if k != "index" else [b[k] for b in batch] for k in batch[0]}
    )
    
    sample_idx = 0
    for batch in tqdm(dataloader, desc="Processing sample batches"):
        # Extract sample index list
        indices = batch.pop("index")
        
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass to get hidden states
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
            
            # Use classifier to predict dataset class
            sequence_output = hidden_states[:, 0, :]  # Take first token representation
            logits = dataset_classifier(hidden_states.float())
            predicted_datasets = torch.argmax(logits, dim=-1)
            
            # Assign adapter for each sample in batch
            for i, pred_dataset_idx in enumerate(predicted_datasets):
                pred_dataset_idx = pred_dataset_idx.item()
                
                # Get sample index in original dataset
                original_idx = indices[i]
                
                # Get corresponding adapter task ID based on predicted dataset
                if pred_dataset_idx in dataset_to_adapter:
                    adapter_id = dataset_to_adapter[pred_dataset_idx]
                elif current_task_id:
                    # If no mapping but have current_task, use it
                    adapter_id = current_task_id
                else:
                    # Otherwise use first adapter
                    adapter_id = list(task_to_adapter.keys())[0]
                
                # Record adapter assignment
                adapter_path = task_to_adapter[adapter_id]
                adapter_assignments[adapter_id]["path"] = adapter_path
                adapter_assignments[adapter_id]["indices"].append(original_idx)
                
                sample_idx += 1
    
    # 8. Generate configuration file
    logger.info("Generating adapter selection configuration file...")
    output_config = {
        "task_name": task_name,
        "adapters": dict(adapter_assignments)
    }
    
    with open(output_config_path, "w", encoding="utf-8") as f:
        json.dump(output_config, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Adapter selection configuration saved to: {output_config_path}")
    
    # Print statistics
    for task_id, data in adapter_assignments.items():
        logger.info(f"Task {task_id}: Assigned {len(data['indices'])} samples")
    
    # Free resources
    del model
    del dataset_classifier
    torch.cuda.empty_cache()

def main():
    """
    Command line entry point
    """
    import argparse
    from llamafactory.hparams import get_eval_args
    
    parser = argparse.ArgumentParser(description="Dynamic ConPet Adapter Selector")
    parser.add_argument("--config-file", type=str, required=True, help="Configuration file path")
    parser.add_argument("--dataset-path", type=str, required=True, help="Test dataset path")
    parser.add_argument("--multi-adapter-dir", type=str, help="Multi-adapter directory, can override config file setting")
    parser.add_argument("--batch-size", type=int, default=16, help="Feature extraction batch size")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config_file, "r") as f:
        config_dict = json.load(f)
        
    # Parse main parameters
    model_args, data_args, eval_args, finetuning_args = get_eval_args(config_dict)
    
    # If multi_adapter_dir provided in command line, override config
    if args.multi_adapter_dir:
        eval_args.multi_adapter_dir = args.multi_adapter_dir
        
    if not eval_args.multi_adapter_dir:
        raise ValueError("multi_adapter_dir not specified, provide through config file or --multi-adapter-dir parameter")
        
    # Run selector
    select_adapter_dynamic_conpet(
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