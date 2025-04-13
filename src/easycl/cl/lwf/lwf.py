import torch
import torch.nn as nn
from typing import Optional, Dict
from llamafactory.extras.logging import get_logger
from easycl.hparams import CLFinetuningArguments
import traceback

logger = get_logger(__name__)

class LWF:
    def __init__(self, model: nn.Module, previous_task_model: Optional[nn.Module] = None, temperature: float = 2.0, alpha: float = 0.5):
        """Initialize LWF"""
        self.model = model
        self.temperature = temperature
        self.alpha = alpha
        self.enabled = True  # LWF status flag
        
        if previous_task_model is not None:
            self.previous_task_model = previous_task_model
        else:
            self.previous_task_model = None
            logger.warning("No previous task model provided. LWF will be disabled.")
            self.enabled = False

    def lwf_loss(self, logits: torch.Tensor, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate LWF loss"""
        if not self.enabled or self.previous_task_model is None:
            return torch.tensor(0.0, device=logits.device)
        
        try:
            # Get device information
            device = logits.device
            
            # Ensure input data is on the correct device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                previous_outputs = self.previous_task_model(**inputs)
                previous_logits = previous_outputs.logits
            
            # Ensure logits shapes match
            if logits.shape != previous_logits.shape:
                logger.warning(f"Shape mismatch: current logits {logits.shape}, previous logits {previous_logits.shape}")
                return torch.tensor(0.0, device=device)
            
            # Create mask for filtering padding
            labels = inputs.get('labels', None)
            if labels is None:
                logger.warning("No labels found in inputs")
                return torch.tensor(0.0, device=device)
            
            # Create attention mask
            attention_mask = labels.ne(-100)  # [batch_size, seq_len]
            
            # Get batch_size and vocab_size
            batch_size, seq_len, vocab_size = logits.shape
            
            # Reshape and mask processing
            current_logits = logits.view(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
            previous_logits = previous_logits.view(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
            flat_attention_mask = attention_mask.view(-1)  # [batch_size * seq_len]
            
            # Select only non-padding positions
            valid_current_logits = current_logits[flat_attention_mask]
            valid_previous_logits = previous_logits[flat_attention_mask]
            
            if valid_current_logits.shape[0] == 0:
                logger.warning("No valid tokens found for distillation")
                return torch.tensor(0.0, device=device)
            
            # Calculate soft labels
            soft_previous = nn.functional.softmax(valid_previous_logits / self.temperature, dim=-1)
            log_soft_current = nn.functional.log_softmax(valid_current_logits / self.temperature, dim=-1)
            
            # Calculate KL divergence loss
            distillation_loss = nn.functional.kl_div(
                log_soft_current,
                soft_previous,
                reduction='batchmean',
                log_target=False
            ) * (self.temperature ** 2)
            
            # Calculate original cross-entropy loss
            valid_labels = labels[attention_mask]
            ce_loss = nn.functional.cross_entropy(valid_current_logits, valid_labels)
            
            # Combine losses
            total_loss = (1 - self.alpha) * ce_loss + self.alpha * distillation_loss
            
            return total_loss
            
        except Exception as e:
            logger.error(f"Error in LWF loss calculation: {str(e)}")
            logger.error(traceback.format_exc())
            return torch.tensor(0.0, device=logits.device)

    def disable(self):
        """Disable LWF"""
        logger.info("Disabling LWF...")
        self.enabled = False
        self.previous_task_model = None
        
    def enable(self):
        """Enable LWF"""
        logger.info("Enabling LWF...")
        self.enabled = True