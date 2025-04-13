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
import torch
import numpy as np
import json
from tqdm import tqdm
import gc
from typing import Dict, List, Optional, Tuple, Any
from torch.utils.data import DataLoader
from transformers import Trainer
from peft import PeftModel

from llamafactory.extras.logging import get_logger
from easycl.hparams import CLFinetuningArguments

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
        
        if os.path.exists(stats_file):
            logger.info_rank0(f"Loading existing statistics: {stats_file}")
            stats = torch.load(stats_file)
            
            # Log existing tasks
            logger.info_rank0(f"Existing task statistics: {list(stats['task_means'].keys())}")
            return stats
        else:
            logger.info_rank0(f"No existing statistics found, creating new record")
            # Initialize statistics
            return {
                "task_means": {},  # Store feature means for each task
                "cov_matrix": None,  # Shared covariance matrix
                "n_samples": 0,  # Total number of samples
                "tasks_info": {}  # Additional information for each task
            }

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
        
        logger.info_rank0("Starting feature extraction...")
        
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
            logger.info_rank0(f"Feature extraction successful, shape: {features.shape}")
            return features
        else:
            logger.error_rank0("Feature extraction failed, returning empty tensor")
            return torch.zeros((0, self.hidden_size))
            
    def _forward_and_get_hidden_states(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        """
        Perform forward pass and get hidden states
        """
        # Remove unnecessary keys to avoid model errors
        batch_for_model = {k: v for k, v in batch.items() 
                          if k in ["input_ids", "attention_mask", "token_type_ids"]}
        
        try:
            # Perform forward pass with output_hidden_states=True to get all layer hidden states
            outputs = self.model(
                **batch_for_model,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Get hidden states from second-to-last layer
            if outputs.hidden_states is not None:
                # Usually, second-to-last layer is the layer before the final layer
                # hidden_states typically includes [embedding layer, middle layers..., final layer]
                return outputs.hidden_states[-2]
            else:
                logger.warning_rank0("No hidden states in model outputs")
                return None
                
        except Exception as e:
            logger.error_rank0(f"Error during forward pass: {str(e)}")
            return None

    def compute_feature_statistics(self, dataset) -> Dict:
        """
        Compute feature statistics
        
        Args:
            dataset: Dataset
            
        Returns:
            stats: Updated feature statistics
        """
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
        features = self._extract_features(dataloader)
        
        if features.shape[0] == 0:
            logger.error_rank0("Could not extract features, skipping statistics computation")
            return self.stats
        
        # Calculate feature mean for current task
        task_mean = torch.mean(features, dim=0)
        
        # Add current task mean to statistics
        self.stats["task_means"][self.task_id] = task_mean.cpu()
        self.stats["tasks_info"][self.task_id] = {
            "n_samples": features.shape[0]
        }
        
        # Update shared covariance matrix
        n_samples = features.shape[0]
        centered_feats = features - task_mean.unsqueeze(0)
        
        # Calculate covariance contribution from current data
        task_cov = torch.matmul(centered_feats.t(), centered_feats) / n_samples
        
        if self.stats["cov_matrix"] is None:
            # First time computing covariance
            self.stats["cov_matrix"] = task_cov.cpu()
        else:
            # Update covariance matrix (weighted update)
            total_samples = self.stats["n_samples"] + n_samples
            old_weight = self.stats["n_samples"] / total_samples
            new_weight = n_samples / total_samples
            
            self.stats["cov_matrix"] = (old_weight * self.stats["cov_matrix"] + 
                                        new_weight * task_cov.cpu())
        
        # Update total sample count
        self.stats["n_samples"] += n_samples
        
        # Log some statistics
        logger.info_rank0(f"Task {self.task_id} feature mean range: [{torch.min(task_mean).item():.4f}, {torch.max(task_mean).item():.4f}]")
        logger.info_rank0(f"Sample count: {n_samples}, Total samples: {self.stats['n_samples']}")
        
        # Save updated statistics
        self.save_stats()
        
        return self.stats
    
    def save_stats(self) -> None:
        """
        Save feature statistics
        """
        stats_file = os.path.join(self.stats_path, "abscl_stats.pt")
        torch.save(self.stats, stats_file)
        logger.info_rank0(f"Feature statistics saved to: {stats_file}")
        
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
        
        meta_file = os.path.join(self.stats_path, "abscl_meta.json")
        with open(meta_file, "w") as f:
            json.dump(meta_info, f, indent=2)
            
        logger.info_rank0(f"Meta information saved to: {meta_file}")

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
    if cl_finetuning_args is None:
        logger.warning_rank0("cl_finetuning_args not provided, skipping feature extraction")
        return
        
    # Get stats path from arguments
    stats_path = cl_finetuning_args.abscl_stats_path or os.path.join(
        os.path.dirname(trainer.args.output_dir), "abscl_stats"
    )
    
    logger.info_rank0(f"Extracting feature statistics for task {task_id}")
    feature_extractor = ABSCLFeatureExtractor(
        model=model,
        trainer=trainer,
        stats_path=stats_path,
        task_id=task_id,
        device=device
    )
    
    # Compute statistics
    feature_extractor.compute_feature_statistics(dataset)
    
    # Save statistics
    feature_extractor.save_stats()
    
    logger.info_rank0(f"Feature statistics extraction completed, saved to {stats_path}")
    
    # Clean up to free memory
    del feature_extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
