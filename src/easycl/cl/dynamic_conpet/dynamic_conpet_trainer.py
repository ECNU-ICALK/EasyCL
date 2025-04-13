import os
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass

from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from .dynamic_conpet import DatasetClassifier, compute_dataset_classification_loss, save_classifier, load_classifier
from llamafactory.extras.logging import get_logger
from easycl.hparams import CLFinetuningArguments

logger = get_logger(__name__)


class DynamicConPetTrainer(CustomSeq2SeqTrainer):
    """
    Dynamic ConPet trainer with dataset classification head training capability
    """
    def __init__(self, *args, **kwargs):
        """Initialize Dynamic ConPet Trainer"""
        # Extract required parameters from kwargs
        self.dataset_classifier = kwargs.pop('dataset_classifier', None)
        self.dataset_names = kwargs.pop('dataset_names', [])
        self.dataset_indices_map = kwargs.pop('dataset_indices_map', {})
        
        # Get cl_finetuning_args
        self.cl_finetuning_args = kwargs.pop("cl_finetuning_args", None)
        
        # Classification loss weight is always 1.0
        self.classification_loss_weight = 1.0
        
        super().__init__(*args, **kwargs)
        
        # Log classifier information
        if self.dataset_classifier is not None:
            logger.info(f"Initialized DynamicConPetTrainer with dataset classifier for {self.dataset_classifier.num_datasets} datasets")
            logger.info(f"Dataset names: {self.dataset_names}")
            logger.info(f"Classification loss weight: {self.classification_loss_weight}")
    
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        """
        Override loss computation to add dataset classification loss
        """
        # Check for dataset labels
        dataset_labels = inputs.pop("dataset_labels", None)
        
        # Call original loss computation
        original_outputs = model(**inputs, output_hidden_states=True)
        original_loss = original_outputs.loss
        
        # If no classifier or dataset labels, return only original loss
        if self.dataset_classifier is None or dataset_labels is None:
            if return_outputs:
                return original_loss, original_outputs
            return original_loss
        
        # Compute dataset classification loss
        classification_loss = compute_dataset_classification_loss(
            original_outputs, 
            dataset_labels, 
            self.dataset_classifier
        )
        
        # Combine losses
        total_loss = original_loss + self.classification_loss_weight * classification_loss
        
        # Save to logs
        self.log({
            "original_loss": original_loss.detach().float().item(),
            "classification_loss": classification_loss.detach().float().item(),
            "total_loss": total_loss.detach().float().item()
        })
        
        if return_outputs:
            return total_loss, original_outputs
        return total_loss
    
    def save_model(self, *args, **kwargs):
        """
        Save model and dataset classifier
        """
        # Call original save method
        output = super().save_model(*args, **kwargs)
        
        # Save dataset classifier
        if self.dataset_classifier is not None and self.args.should_save:
            classifier_save_path = os.path.join(self.args.output_dir, "dataset_classifier")
            save_classifier(self.dataset_classifier, classifier_save_path, self.dataset_names)
        
        return output
    
    def get_batch_dataset_labels(self, batch_indices: List[int]) -> torch.Tensor:
        """
        Get dataset labels for batch indices
        
        Args:
            batch_indices: List of sample indices in the batch
            
        Returns:
            dataset_labels: Dataset label tensor
        """
        labels = []
        for idx in batch_indices:
            # Find which dataset the current index belongs to
            dataset_idx = 0  # Default to current dataset (last one)
            for ds_idx, (start_idx, end_idx) in enumerate(self.dataset_indices_map.items()):
                if start_idx <= idx < end_idx:
                    dataset_idx = ds_idx
                    break
            labels.append(dataset_idx)
        
        return torch.tensor(labels, device=self.args.device)
    
    def _prepare_inputs(self, inputs):
        """
        Prepare inputs by adding dataset labels for each batch
        """
        # Original input preparation
        inputs = super()._prepare_inputs(inputs)
        
        # Check if dataset labels need to be added
        if self.dataset_classifier is not None and self.dataset_indices_map:
            # Get batch indices
            if "index" in inputs:
                batch_indices = inputs.pop("index")
                # Generate dataset labels
                dataset_labels = self.get_batch_dataset_labels(batch_indices)
                inputs["dataset_labels"] = dataset_labels
        
        return inputs