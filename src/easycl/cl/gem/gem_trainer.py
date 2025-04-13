# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
# Modifications copyright 2024 Your Name/Org (if applicable)
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

import torch
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union, Any

from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from llamafactory.extras.logging import get_logger

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from llamafactory.hparams import FinetuningArguments
    from easycl.hparams.cl_finetuning_args import CLFinetuningArguments

logger = get_logger(__name__)

class GEMSeq2SeqTrainer(CustomSeq2SeqTrainer):
    """
    Seq2SeqTrainer implementing Gradient Episodic Memory (GEM).
    Assumes input batches contain an 'is_memory' boolean field to distinguish
    between current task samples and episodic memory samples.
    Inherits from CustomSeq2SeqTrainer.
    """
    def __init__(self, finetuning_args: "FinetuningArguments", cl_finetuning_args: "CLFinetuningArguments", *args, **kwargs):
        # Pass finetuning_args to the parent class constructor
        super().__init__(finetuning_args=finetuning_args, *args, **kwargs)
        self.cl_finetuning_args = cl_finetuning_args
        if not cl_finetuning_args.use_gem:
             # Log a warning or potentially raise an error if GEM trainer is used without the flag
             logger.warning("GEMSeq2SeqTrainer initialized, but use_gem in cl_finetuning_args is False.")
             # Store a default or disable GEM specific logic
             self.gem_memory_strength = 0.0 # Effectively disables projection
        else:
            self.gem_memory_strength = cl_finetuning_args.gem_memory_strength
            logger.info(f"GEM Trainer initialized with memory strength: {self.gem_memory_strength}")

    def compute_loss(self, model: "PreTrainedModel", inputs: Dict[str, Union[torch.Tensor, Any]], return_outputs=False,num_items_in_batch=None) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[Dict]]]:
        """
        Computes the loss for the current task, applying GEM projection if necessary.

        Args:
            model: The model to compute the loss for.
            inputs: The input batch. Must contain an 'is_memory' tensor.
            return_outputs: Whether to return model outputs alongside the loss.

        Returns:
            The computed loss (potentially after gradient modification), and optionally the model outputs.
        """
        # If GEM is disabled (strength=0) or not use_gem flag, revert to standard loss computation
        if not hasattr(self, 'gem_memory_strength') or self.gem_memory_strength <= 0:
             return super().compute_loss(model, inputs, return_outputs=return_outputs)
             
        if "is_memory" not in inputs or not isinstance(inputs["is_memory"], torch.Tensor):
            raise ValueError(
                "GEMTrainer requires each batch sample to have an 'is_memory' boolean tensor field "
                "to distinguish current task vs. memory samples."
            )

        is_memory = inputs["is_memory"]
        current_mask = ~is_memory
        memory_mask = is_memory

        # Handle cases where a batch might only contain current or memory samples
        has_current_samples = torch.any(current_mask)
        has_memory_samples = torch.any(memory_mask)
        
        # If no current samples, only calculate memory loss for potential stats, but return 0 loss
        # as we don't optimize based on memory alone in this step.
        if not has_current_samples:
            logger.warning_once("Received a batch with only memory samples. Skipping GEM gradient calculation for this batch.")
            # Still might need to compute outputs if requested for evaluation etc.
            if return_outputs:
                # Need to decide what outputs to return - maybe just run forward pass on memory?
                # For simplicity, let's return 0 loss and None outputs if only memory present in training
                # This assumes return_outputs=False during standard training steps
                return (torch.tensor(0.0, device=model.device, requires_grad=True), None) if return_outputs else torch.tensor(0.0, device=model.device, requires_grad=True)
            else:
                 return torch.tensor(0.0, device=model.device, requires_grad=True)

        # Separate inputs
        # Ensure we handle metadata columns gracefully (like 'is_memory')
        current_inputs = {k: v[current_mask] for k, v in inputs.items() if isinstance(v, torch.Tensor)}        
        
        # Step 1: Compute current task loss and potentially outputs
        # We need the loss object itself for backpropagation
        loss_current_tuple = super().compute_loss(model, current_inputs, return_outputs=True) # Request outputs
        loss_current = loss_current_tuple[0]
        outputs_current = loss_current_tuple[1]

        # Step 2: Compute gradient for the current task
        g_current = self._get_grads(model, loss_current)
        if g_current is None:
             logger.warning_once("Could not compute gradients for the current task batch. Skipping GEM projection.")
             # Return the original loss and outputs if gradient calculation failed
             return (loss_current.detach(), outputs_current) if return_outputs else loss_current.detach()
             
        # If no memory samples in this batch, behave like normal training
        if not has_memory_samples:
            # Gradients are already on the parameters from _get_grads (due to retain_graph=True),
            # so just return the loss. The optimizer step will use these gradients.
            return (loss_current, outputs_current) if return_outputs else loss_current

        # Step 3: Compute memory loss and gradient (only if memory samples exist)
        memory_inputs = {k: v[memory_mask] for k, v in inputs.items() if isinstance(v, torch.Tensor)}    
        # We only need the loss value to compute gradients, no need for outputs here
        # Use .detach() on the loss to prevent graph connection back to current task computation if any shared layers exist? 
        # No, compute_loss should handle model state correctly. Use retain_graph in _get_grads.    
        loss_memory = super().compute_loss(model, memory_inputs, return_outputs=False)
        # Detach the memory loss before computing grads for projection? 
        # The gradient itself should be detached in _project_gem_qp. Loss needs grad_fn.
        g_memory = self._get_grads(model, loss_memory)
        
        if g_memory is None:
            logger.warning_once("Could not compute gradients for the memory samples in this batch. Skipping GEM projection.")
            # Proceed as if no memory samples were present
            return (loss_current, outputs_current) if return_outputs else loss_current

        # Step 4: Check for conflict and project if needed
        dot_product = torch.dot(g_current.detach(), g_memory.detach())

        if dot_product < 0:
            logger.debug(f"GEM conflict detected (dot product: {dot_product:.4f}). Projecting gradient.")
            g_projected = self._project_gem_qp(g_current, [g_memory]) # Pass memory grad as a list
            # Assign the projected gradient back to the model parameters
            self._assign_grads(model, g_projected)
            # The loss returned is still the original current task loss
            loss_to_return = loss_current 
        else:
            logger.debug(f"No GEM conflict (dot product: {dot_product:.4f}). Using original gradient.")
            # If no conflict, the gradients computed by _get_grads(loss_current) are already
            # associated with the parameters. We just need to return the loss.
            # No need to call backward() again here, as _get_grads did it.
            loss_to_return = loss_current

        # Return the original current task loss (and outputs if requested)
        # The optimizer will use the gradients currently assigned to model.parameters()
        # (which might be the original g_current or the projected g_projected)
        return (loss_to_return, outputs_current) if return_outputs else loss_to_return

    def _get_grads(self, model: "PreTrainedModel", loss: torch.Tensor) -> Optional[torch.Tensor]:
        """Computes and returns the flattened gradients for the given loss."""
        if loss == 0.0 or not loss.requires_grad:
             return None # Cannot compute gradients if loss is zero or doesn't require grad

        model.zero_grad() # Ensure grads are clean before backward
        
        # Need to handle potential DistributedDataParallel scenarios if applicable
        # For now, assume standard single/multi-GPU setup handled by Trainer
        try:
            loss.backward(retain_graph=True) # Retain graph is crucial for computing multiple grads
        except RuntimeError as e:
            logger.error(f"Error during backward pass: {e}. This might happen if the graph was already freed.")
            return None # Indicate failure
            
        grads = []
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                grads.append(param.grad.view(-1))
            # elif param.requires_grad and param.grad is None:
                # logger.warning_once(f"Parameter {name} requires grad but grad is None") 
                # pass # Potentially handle parameters that don't get gradients?
        
        if not grads:
            logger.warning_once("No gradients found for the model parameters.")
            return None
            
        # It's important gradients are on the same device for dot product / projection
        # Assuming all grads will be on the same device as the first one
        device = grads[0].device
        try:
            flat_grads = torch.cat(grads).to(device)
            return flat_grads
        except Exception as e:
            logger.error(f"Error concatenating gradients: {e}")
            return None

    def _assign_grads(self, model: "PreTrainedModel", flat_grad: torch.Tensor):
        """Assigns the flattened gradient `flat_grad` back to the model parameters."""
        pointer = 0
        for param in model.parameters():
            if param.requires_grad and param.grad is not None: # Check grad is not None
                numel = param.grad.numel()
                if pointer + numel > flat_grad.numel():
                    logger.error("Gradient assignment error: flat_grad is smaller than the total number of gradient elements.")
                    break
                # Ensure device compatibility before copying
                param.grad.copy_(flat_grad[pointer : pointer + numel].view_as(param.grad).to(param.device))
                pointer += numel
            # elif param.requires_grad and param.grad is None:
                # If grad was None initially, should we assign it? 
                # Generally no, stick to params that had grads.
                # pass
        if pointer != flat_grad.numel():
             logger.warning(f"Gradient assignment warning: Mismatch in number of elements. Assigned {pointer}, flat_grad had {flat_grad.numel()}")

    def _project_gem_qp(self, g_current: torch.Tensor, memory_grads: List[torch.Tensor], max_iter: int = 15) -> torch.Tensor:
        """
        Projects the current gradient `g_current` to satisfy GEM constraints using LBFGS.
        Minimizes ||g' - g_current||^2 s.t. dot(g', g_mem_i) >= 0 for all i.

        Args:
            g_current: The current task's gradient (flattened).
            memory_grads: A list of gradients from memory tasks (flattened).
            max_iter: Max iterations for LBFGS.

        Returns:
            The projected gradient g'.
        """
        g_current = g_current.detach() # Ensure we don't modify the original grad tensor directly
        # Initialize the projected gradient as a parameter, starting from the current gradient
        g_proj = torch.nn.Parameter(g_current.clone(), requires_grad=True)
        
        # Use LBFGS optimizer on the projected gradient parameter
        optimizer = torch.optim.LBFGS([g_proj], max_iter=max_iter, line_search_fn="strong_wolfe")
        
        # Get the penalty strength from the instance variable
        penalty_strength = self.gem_memory_strength
        
        # Define the closure for the optimizer
        def closure():
            optimizer.zero_grad()
            # Primary objective: minimize distance to original gradient
            loss = 0.5 * torch.norm(g_proj - g_current) ** 2
            
            # Constraint penalty: add penalty if dot product is negative
            for g_mem in memory_grads:
                # Ensure memory gradient is detached and on the correct device
                g_mem_detached = g_mem.detach().to(g_proj.device)
                dot_prod = torch.dot(g_proj, g_mem_detached)
                # Only apply penalty if the constraint is violated (dot_prod < 0)
                if dot_prod < 0:
                    # Add penalty proportional to the violation magnitude
                    loss = loss - penalty_strength * dot_prod # Maximize dot product towards >= 0
                    
            loss.backward()
            return loss

        # Perform the optimization
        optimizer.step(closure)
        
        # Return the optimized projected gradient, detached from computation graph
        return g_proj.detach()