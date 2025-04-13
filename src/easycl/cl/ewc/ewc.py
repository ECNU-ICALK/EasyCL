import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
from torch.utils.data import Dataset, DataLoader
from llamafactory.extras.logging import get_logger
from easycl.hparams import CLFinetuningArguments

logger = get_logger(__name__)

class EWC:
    def __init__(self, model: nn.Module, lambda_ewc: float = 0.5):
        """Initialize EWC"""
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_dict = {}  # Store Fisher information
        self.param_dict = {}   # Store important parameters
        self.enabled = True    # EWC status flag
        
    def compute_fisher(self, dataloader: DataLoader, num_samples: int = 100) -> bool:
        """
        Compute Fisher Information Matrix
        Returns whether computation was successful
        """
        if not self.enabled:
            logger.warning("EWC is disabled. Skipping Fisher computation.")
            return False
            
        try:
            self.model.eval()
            device = next(self.model.parameters()).device
            
            # Initialize Fisher dictionary
            new_fisher_dict = {}
            for name, param in self.model.named_parameters():
                if 'lora_' in name:  # Only focus on LoRA parameters
                    new_fisher_dict[name] = torch.zeros_like(param.data)
            
            samples_processed = 0
            for batch_idx, batch in enumerate(dataloader):
                if samples_processed >= num_samples:
                    break
                    
                try:
                    # Move data to correct device
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    
                    self.model.zero_grad()
                    outputs = self.model(**batch)
                    
                    if not hasattr(outputs, 'loss'):
                        logger.warning(f"Batch {batch_idx}: Model outputs do not contain loss.")
                        continue
                        
                    loss = outputs.loss
                    loss.backward()
                    
                    for name, param in self.model.named_parameters():
                        if 'lora_' in name and param.grad is not None:
                            new_fisher_dict[name] += param.grad.data.pow(2)
                    
                    samples_processed += batch['input_ids'].size(0)
                    
                except Exception as e:
                    logger.warning(f"Error processing batch {batch_idx}: {str(e)}")
                    continue
                    
            # Normalize Fisher information
            if samples_processed > 0:
                for name in new_fisher_dict:
                    new_fisher_dict[name] /= samples_processed
                    
                self.fisher_dict = new_fisher_dict
                self.store_parameters()
                logger.info(f"Successfully computed Fisher information using {samples_processed} samples.")
                logger.info("EWC has been successfully enabled.")
                return True
                
            logger.warning("No samples were successfully processed.")
            return False
            
        except Exception as e:
            logger.error(f"Failed to compute Fisher information: {str(e)}")
            return False
            
    def store_parameters(self):
        """Store current important parameters"""
        self.param_dict = {}
        for name, param in self.model.named_parameters():
            if 'lora_' in name:
                self.param_dict[name] = param.data.clone()
                
    def ewc_loss(self) -> torch.Tensor:
        """Calculate EWC loss"""
        if not self.enabled or not self.fisher_dict or not self.param_dict:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
            
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for name, param in self.model.named_parameters():
            if name in self.fisher_dict:
                _loss = self.fisher_dict[name] * (param - self.param_dict[name]).pow(2)
                loss += _loss.sum()
                
        return self.lambda_ewc * loss
        
    def disable(self):
        """Disable EWC"""
        logger.info("Disabling EWC...")
        self.enabled = False
        self.fisher_dict = {}
        self.param_dict = {}
        
    def enable(self):
        """Enable EWC"""
        logger.info("Enabling EWC...")
        self.enabled = True

def get_ewc_trainer(trainer_cls):
    """Decorator for creating Trainer class with EWC support"""
    class EWCTrainer(trainer_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.ewc = EWC(self.model)
            
        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(**inputs)
            loss = outputs.loss
            
            # Add EWC loss
            ewc_loss = self.ewc.ewc_loss()
            loss += ewc_loss
            
            return (loss, outputs) if return_outputs else loss
            
    return EWCTrainer