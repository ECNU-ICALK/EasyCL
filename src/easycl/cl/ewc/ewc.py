import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, List, Optional, Union
from torch.utils.data import Dataset, DataLoader
from llamafactory.extras.logging import get_logger
from easycl.hparams import CLFinetuningArguments
from ..distributed_utils import (
    is_distributed, get_rank, get_world_size, is_main_process,
    get_local_rank, get_deepspeed_zero_stage, gather_parameters, all_reduce_tensor,broadcast_object
)
def debugprint(*args, **kwargs):
    pass
import traceback

logger = get_logger(__name__)

def _all_reduce_fisher_dict(fisher_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Applies all_reduce (SUM) to each tensor in the Fisher dictionary."""
    if not is_distributed():
        return fisher_dict

    reduced_dict = {}
    local_rank = get_local_rank()  # Get local rank for correct device placement
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
    for name, tensor in fisher_dict.items():
        try:
            # Ensure tensor is on the correct device for this rank's GPU
            tensor_on_device = tensor.to(device)
            debugprint(f"[Rank {get_rank()}] Reducing tensor {name} on device {tensor_on_device.device}")
            # Perform all_reduce (SUM operation is default for all_reduce_tensor)
            reduced_tensor = all_reduce_tensor(tensor_on_device, op=dist.ReduceOp.SUM)
            reduced_dict[name] = reduced_tensor.cpu()  # Move back to CPU after reduction
            debugprint(f"[Rank {get_rank()}] Reduced tensor {name} shape: {reduced_dict[name].shape}")
        except Exception as e:
            logger.error(f"[Rank {get_rank()}] Error reducing tensor {name}: {e}")
            logger.error(traceback.format_exc())
            # Fallback: return the local tensor if reduction fails
            reduced_dict[name] = tensor.cpu()

    # Barrier to ensure all processes finish reduction before proceeding
    if is_distributed():
        dist.barrier()
        debugprint(f"[Rank {get_rank()}] Finished _all_reduce_fisher_dict")
    return reduced_dict

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
            debugprint(f"[Rank {get_rank()}] Entering compute_fisher function")
            self.model.eval()
            device = next(self.model.parameters()).device
            rank = get_rank()
            world_size = get_world_size()

            # Get DeepSpeed stage
            stage = get_deepspeed_zero_stage(self.model)
            debugprint(f"[Rank {rank}] Computing Fisher information with DeepSpeed ZeRO Stage {stage}")

            # Initialize Fisher dictionary
            new_fisher_dict = {}
            debugprint(f"[Rank {rank}] Initialized empty new_fisher_dict")

            # 统一处理非ZeRO-3的情况：只在rank 0上计算Fisher矩阵
            if stage != 3:
                debugprint(f"[Rank {rank}] Entering non-ZeRO-3 logic (stage={stage})")
                # 只有rank 0需要初始化Fisher字典用于计算
                if rank == 0:
                    debugprint(f"[Rank {rank}] Initializing Fisher dictionary for computation (non-ZeRO-3)")
                    # 确保使用GPU设备初始化Fisher字典，而不是CPU
                    local_rank = get_local_rank()
                    compute_device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
                    debugprint(f"[Rank {rank}] Using device {compute_device} for Fisher computation")
                    for name, param in self.model.named_parameters():
                        if 'lora_' in name:  # Only focus on LoRA parameters
                            # 在计算设备上初始化Fisher字典，稍后再移动到CPU
                            new_fisher_dict[name] = torch.zeros_like(param.data, device=compute_device)
                            debugprint(f"[Rank {rank}] Initialized Fisher dict for {name} on {compute_device}")

                # 在非ZeRO-3环境下，只在rank 0上计算Fisher矩阵
                if rank == 0:
                    debugprint(f"[Rank {rank}] Starting Fisher computation loop on rank 0 only (non-ZeRO-3)")
                    samples_processed = 0
                    debugprint(f"[Rank {rank}] Entering dataloader loop (num_samples={num_samples})")
                    for batch_idx, batch in enumerate(dataloader):
                        debugprint(f"[Rank {rank}] Processing batch {batch_idx}")
                        if samples_processed >= num_samples:
                            debugprint(f"[Rank {rank}] Reached num_samples ({num_samples}), breaking loop.")
                            break

                        try:
                            # 确保使用正确的设备（GPU）进行计算
                            local_rank = get_local_rank()
                            compute_device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'

                            # 检查模型当前设备
                            model_device = next(self.model.parameters()).device
                            debugprint(f"[Rank {rank}] Model is on device: {model_device}, will compute on: {compute_device}")

                            # 保存原始模型设备，以便计算后恢复
                            original_device = model_device

                            # 将模型移动到计算设备
                            if model_device != compute_device:
                                debugprint(f"[Rank {rank}] Moving model from {model_device} to {compute_device}")
                                self.model.to(compute_device)
                                model_device = compute_device
                                debugprint(f"[Rank {rank}] Model moved to {model_device}")

                            # 将数据移动到计算设备（与模型相同的设备）
                            debugprint(f"[Rank {rank}] Moving batch {batch_idx} data to device {model_device}")
                            batch = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                            debugprint(f"[Rank {rank}] Data moved for batch {batch_idx}")

                            # 确保梯度清零
                            self.model.zero_grad()
                            debugprint(f"[Rank {rank}] Zeroed gradients for batch {batch_idx}")

                            # 前向传播
                            debugprint(f"[Rank {rank}] Starting forward pass for batch {batch_idx}")
                            try:
                                outputs = self.model(**batch)
                                debugprint(f"[Rank {rank}] Finished forward pass for batch {batch_idx}")
                            except RuntimeError as e:
                                if "expected scalar type" in str(e) or "device type" in str(e) or "Expected all tensors to be on the same device" in str(e):
                                    # 处理设备不匹配错误
                                    current_model_device = next(self.model.parameters()).device
                                    logger.warning(f"[Rank {rank}] Device mismatch in forward pass: {str(e)}")
                                    logger.warning(f"[Rank {rank}] Current model device: {current_model_device}")
                                    logger.warning(f"[Rank {rank}] Attempting to move batch to current model device")

                                    # 尝试将数据移动到当前模型设备
                                    batch = {k: v.to(current_model_device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                                    outputs = self.model(**batch)
                                    debugprint(f"[Rank {rank}] Forward pass succeeded after moving data to model device {current_model_device}")
                                else:
                                    # 重新抛出其他类型的错误
                                    raise
                            if not hasattr(outputs, 'loss'):
                                logger.warning(f"[Rank {rank}] Batch {batch_idx}: Model outputs do not contain loss.")
                                continue # Skip batch if loss is missing
                            loss = outputs.loss
                            debugprint(f"[Rank {rank}] Starting backward pass for batch {batch_idx}")
                            try:
                                loss.backward()
                                debugprint(f"[Rank {rank}] Finished backward pass for batch {batch_idx}")
                            except RuntimeError as e:
                                logger.error(f"[Rank {rank}] Error during backward pass: {str(e)}")
                                logger.error(traceback.format_exc())
                                continue  # 跳过这个批次

                            # 累积Fisher信息（平方梯度）
                            debugprint(f"[Rank {rank}] Accumulating gradients for batch {batch_idx}")
                            try:
                                current_model_device = next(self.model.parameters()).device
                                debugprint(f"[Rank {rank}] Current model device for gradient accumulation: {current_model_device}")

                                for name, param in self.model.named_parameters():
                                    if 'lora_' in name and param.grad is not None:
                                        # 先在当前设备上计算平方梯度
                                        grad_data = param.grad.data.detach().clone()
                                        grad_squared = grad_data.pow(2)

                                        # 检查Fisher字典中的张量设备
                                        fisher_device = new_fisher_dict[name].device
                                        debugprint(f"[Rank {rank}] Fisher dict for {name} is on {fisher_device}, grad is on {grad_squared.device}")

                                        # 将结果移动到Fisher字典的设备并累积
                                        if fisher_device != grad_squared.device:
                                            debugprint(f"[Rank {rank}] Moving grad_squared from {grad_squared.device} to {fisher_device}")
                                            new_fisher_dict[name] += grad_squared.to(fisher_device)
                                        else:
                                            new_fisher_dict[name] += grad_squared
                                debugprint(f"[Rank {rank}] Finished accumulating gradients for batch {batch_idx}")
                            except Exception as e:
                                logger.error(f"[Rank {rank}] Error accumulating gradients: {str(e)}")
                                logger.error(traceback.format_exc())
                                continue  # 跳过这个批次

                            samples_processed += batch['input_ids'].size(0)
                            debugprint(f"[Rank {rank}] samples_processed updated to {samples_processed} after batch {batch_idx}")

                        except Exception as e:
                            logger.warning(f"[Rank {rank}] Error processing batch {batch_idx}: {str(e)}")
                            logger.warning(traceback.format_exc())
                            continue # Continue to next batch on error
                        finally:
                            # 如果模型被移动过，将其移回原始设备
                            if 'original_device' in locals() and model_device != original_device:
                                debugprint(f"[Rank {rank}] Moving model back from {model_device} to {original_device}")
                                self.model.to(original_device)
                                debugprint(f"[Rank {rank}] Model moved back to {original_device}")

                    debugprint(f"[Rank {rank}] Exited dataloader loop")

                    # Normalize Fisher information
                    if samples_processed > 0:
                        debugprint(f"[Rank {rank}] Normalizing Fisher information (samples_processed={samples_processed})")
                        for name in new_fisher_dict:
                            new_fisher_dict[name] /= samples_processed
                        debugprint(f"[Rank {rank}] Finished normalizing Fisher information")

                        # 确保所有Fisher字典张量都在CPU上，以便安全广播
                        debugprint(f"[Rank {rank}] Moving Fisher dictionary to CPU for broadcast")
                        for name in new_fisher_dict:
                            if new_fisher_dict[name].device.type != 'cpu':
                                new_fisher_dict[name] = new_fisher_dict[name].cpu()
                        debugprint(f"[Rank {rank}] Completed Fisher computation with {samples_processed} samples")
                    else:
                        logger.warning(f"[Rank {rank}] No samples were successfully processed for Fisher computation.")
                        debugprint(f"[Rank {rank}] Returning False due to no samples processed")
                        return False

                # 使用broadcast_object广播Fisher字典，而不是all_reduce
                if is_distributed():
                    debugprint(f"[Rank {rank}] Preparing for broadcast. Barrier before broadcast.")
                    dist.barrier()  # 确保所有进程同步
                    debugprint(f"[Rank {rank}] Passed pre-broadcast barrier. Broadcasting Fisher dictionary from rank 0.")
                    new_fisher_dict = broadcast_object(new_fisher_dict, src=0)
                    debugprint(f"[Rank {rank}] broadcast_object called. Barrier after broadcast.")
                    dist.barrier()  # 再次同步
                    debugprint(f"[Rank {rank}] Passed post-broadcast barrier. Received Fisher dictionary broadcast.")
                else:
                    debugprint(f"[Rank {rank}] Not distributed, skipping broadcast.")

                # 所有rank都存储Fisher字典
                debugprint(f"[Rank {rank}] Storing received/computed Fisher dictionary.")
                self.fisher_dict = {k: v.cpu() for k, v in new_fisher_dict.items()}
                debugprint(f"[Rank {rank}] Fisher dictionary stored.")

                # 存储参数
                debugprint(f"[Rank {rank}] Calling store_parameters (non-ZeRO-3).")
                self.store_parameters()
                debugprint(f"[Rank {rank}] Returned from store_parameters (non-ZeRO-3).")

                if is_main_process():
                    logger.info(f"Successfully computed Fisher information on rank 0 and broadcasted to all {world_size} ranks.")
                    logger.info("EWC has been successfully enabled.")
                debugprint(f"[Rank {rank}] Returning True (non-ZeRO-3 success).")
                return True

            # ZeRO-3的情况保持原有逻辑
            else:
                debugprint(f"[Rank {rank}] Entering ZeRO-3 logic")
                # Initialize Fisher dictionary for ZeRO-3
                new_fisher_dict = {}
                debugprint(f"[Rank {rank}] Initialized empty new_fisher_dict (ZeRO-3)")

                samples_processed = 0
                for batch_idx, batch in enumerate(dataloader):
                    if samples_processed >= num_samples:
                        break

                    try:
                        # Move data to correct device
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                        self.model.zero_grad()

                        # Stage 3: Wrap forward and backward in gather_parameters
                        params_to_gather = list(self.model.parameters()) # Gather all parameters for forward/backward
                        debugprint(f"[Rank {rank}] Gathering parameters for forward/backward (Stage 3)")
                        with gather_parameters(self.model, params=params_to_gather):
                            debugprint(f"[Rank {rank}] Inside forward/backward gather context (Stage 3)")
                            # Forward pass
                            outputs = self.model(**batch)
                            if not hasattr(outputs, 'loss'):
                                logger.warning(f"[Rank {rank}] Batch {batch_idx}: Model outputs do not contain loss inside gather context.")
                                continue # Skip batch if loss is missing
                            loss = outputs.loss
                            # Backward pass to compute gradients
                            loss.backward()
                            debugprint(f"[Rank {rank}] Completed forward/backward inside gather context (Stage 3)")

                            # Check gradients inside the gather context
                            for name, param in self.model.named_parameters():
                                if 'lora_' in name:
                                    if hasattr(param, 'grad') and param.grad is not None and not param.grad.is_meta:
                                        if name not in new_fisher_dict:
                                            new_fisher_dict[name] = torch.zeros_like(param.data, device='cpu') # Initialize on CPU
                                        # Accumulate squared gradients, move grad to CPU
                                        grad_data = param.grad.data.detach().clone().cpu()
                                        new_fisher_dict[name] += grad_data.pow(2)
                                    elif hasattr(param, 'grad') and param.grad is None:
                                         pass # Expected for non-trainable params or params on other ranks
                                    elif hasattr(param, 'grad') and param.grad is not None and param.grad.is_meta:
                                         logger.warning(f"[Rank {rank}] Grad for {name} is meta inside gather context (Stage 3 Fisher). Skipping.")
                                    else:
                                         logger.warning(f"[Rank {rank}] Grad attribute missing for {name} inside gather context (Stage 3 Fisher).")

                        samples_processed += batch['input_ids'].size(0)

                    except RuntimeError as e:
                        if 'weight' in str(e) and 'must be 2-D' in str(e):
                             logger.error(f"[Rank {rank}] Still encountered embedding weight error in batch {batch_idx} even with gather_parameters: {str(e)}")
                             logger.error(traceback.format_exc())
                        else:
                            logger.warning(f"[Rank {rank}] Error processing batch {batch_idx} (ZeRO-3): {str(e)}")
                            logger.warning(traceback.format_exc())
                        continue # Continue to next batch on error
                    except Exception as e:
                         logger.warning(f"[Rank {rank}] Non-RuntimeError processing batch {batch_idx} (ZeRO-3): {str(e)}")
                         logger.warning(traceback.format_exc())
                         continue # Continue to next batch on error

                # Normalize Fisher information
                if samples_processed > 0:
                    for name in new_fisher_dict:
                        new_fisher_dict[name] /= samples_processed

                    # All-Reduce Fisher Matrix across all ranks for ZeRO-3
                    debugprint(f"[Rank {rank}] Calling _all_reduce_fisher_dict (ZeRO-3)")
                    new_fisher_dict = _all_reduce_fisher_dict(new_fisher_dict)
                    debugprint(f"[Rank {rank}] Returned from _all_reduce_fisher_dict (ZeRO-3)")

                    # Store the Fisher dictionary (on CPU to save GPU memory)
                    debugprint(f"[Rank {rank}] Storing reduced Fisher dictionary (ZeRO-3)")
                    self.fisher_dict = {k: v.cpu() for k, v in new_fisher_dict.items()}
                    debugprint(f"[Rank {rank}] Fisher dictionary stored (ZeRO-3)")

                    # Store parameters
                    debugprint(f"[Rank {rank}] Calling store_parameters (ZeRO-3).")
                    self.store_parameters()
                    debugprint(f"[Rank {rank}] Returned from store_parameters (ZeRO-3).")

                    if is_main_process():
                        logger.info(f"Successfully computed and reduced Fisher information using {samples_processed} samples across {get_world_size()} ranks.")
                        logger.info("EWC has been successfully enabled.")
                    debugprint(f"[Rank {rank}] Returning True (ZeRO-3 success).")
                    return True

                logger.warning(f"[Rank {rank}] No samples were successfully processed (ZeRO-3).")
                debugprint(f"[Rank {rank}] Returning False (ZeRO-3, no samples processed).")
                return False

        except Exception as e:
            logger.error(f"[Rank {get_rank()}] Failed to compute Fisher information: {str(e)}")
            logger.error(traceback.format_exc())
            debugprint(f"[Rank {get_rank()}] Returning False due to top-level exception in compute_fisher")
            return False

    def store_parameters(self):
        """Store current important parameters"""
        self.param_dict = {}
        rank = get_rank()
        world_size = get_world_size()

        # Get DeepSpeed stage
        stage = get_deepspeed_zero_stage(self.model)
        debugprint(f"[Rank {rank}] Storing parameters with DeepSpeed ZeRO Stage {stage}")

        # Stage 3: Use gather_parameters context
        if stage == 3:
            # Identify target parameters
            params_to_gather = {name: p for name, p in self.model.named_parameters() if 'lora_' in name}

            # Use gather_parameters context
            debugprint(f"[Rank {rank}] Gathering parameters for storage (Stage 3)")
            with gather_parameters(self.model, params=list(params_to_gather.values())):
                debugprint(f"[Rank {rank}] Inside gather_parameters context (Stage 3)")
                # Inside context, parameters are gathered on all ranks
                for name, param in params_to_gather.items():
                    if hasattr(param, 'is_meta') and param.is_meta:  # Check if param is still meta after gather
                        logger.warning(f"[Rank {rank}] Parameter {name} is still meta after gather in Stage 3. Skipping.")
                        continue
                    try:
                        self.param_dict[name] = param.data.detach().clone().cpu()
                        #debugprint(f"[Rank {rank}] Stored {name} (shape: {self.param_dict[name].shape}) to CPU (Stage 3)")
                    except Exception as e:
                        logger.error(f"[Rank {rank}] Failed to store param {name} in Stage 3 gather: {e}")
                        logger.error(traceback.format_exc())
        # 统一处理非ZeRO-3的情况：只在rank 0上存储参数，然后广播
        else:
            debugprint(f"[Rank {rank}] Storing parameters for non-ZeRO-3 (Stage {stage})")

            # 只在rank 0上存储参数
            if rank == 0:
                debugprint(f"[Rank {rank}] Storing parameters on rank 0 for non-ZeRO-3")
                for name, param in self.model.named_parameters():
                    if 'lora_' in name:
                        self.param_dict[name] = param.data.detach().clone().cpu()
                        debugprint(f"[Rank {rank}] Stored {name} (shape: {param.data.shape}) to CPU")

            # 使用broadcast_object广播参数字典
            if is_distributed():
                # 确保rank 0已完成参数存储
                dist.barrier()

                # 广播参数字典
                debugprint(f"[Rank {rank}] Broadcasting parameter dictionary from rank 0")
                self.param_dict = broadcast_object(self.param_dict, src=0)

                # 确保所有rank都接收到参数
                dist.barrier()
                debugprint(f"[Rank {rank}] Completed parameter dictionary broadcast")

        if is_main_process():
            logger.info(f"Stored {len(self.param_dict)} important parameters.")

    def ewc_loss(self) -> torch.Tensor:
        """Calculate EWC loss"""
        if not self.enabled or not self.fisher_dict or not self.param_dict:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        device = next(self.model.parameters()).device
        loss = torch.tensor(0.0, device=device)
        rank = get_rank()

        # Get DeepSpeed stage
        stage = get_deepspeed_zero_stage(self.model)
        debugprint(f"[Rank {rank}] Calculating EWC loss with DeepSpeed ZeRO Stage {stage}")

        # Stage 3: Use gather_parameters context
        if stage == 3:
            # Identify parameters needed
            params_to_gather = {name: p for name, p in self.model.named_parameters()
                               if name in self.fisher_dict and name in self.param_dict}

            # Use gather_parameters context
            debugprint(f"[Rank {rank}] Preparing to gather {len(params_to_gather)} parameters for EWC loss calculation (Stage 3)")
            if is_distributed():
                dist.barrier() # Sync before gather
            debugprint(f"[Rank {rank}] Entering gather_parameters context (Stage 3)")

            with gather_parameters(self.model, params=list(params_to_gather.values())):
                debugprint(f"[Rank {rank}] Inside gather_parameters context (Stage 3)")
                # Inside context, parameters are gathered
                for name, param in params_to_gather.items():
                    debugprint(f"[Rank {rank}] Processing parameter: {name} (Stage 3 gather)")
                    if hasattr(param, 'is_meta') and param.is_meta:  # Check if param is still meta after gather
                        logger.warning(f"[Rank {rank}] Parameter {name} is still meta after gather in Stage 3 ewc_loss. Skipping.")
                        continue
                    if name in self.fisher_dict and name in self.param_dict:
                        try:
                            debugprint(f"[Rank {rank}] Accessing fisher and param_star for {name}")
                            param_star = self.param_dict[name].to(device)  # Move stored param to current device
                            fisher = self.fisher_dict[name].to(device)  # Move fisher info to current device
                            debugprint(f"[Rank {rank}] Calculating loss component for {name}")
                            _loss_component = fisher * (param.data - param_star).pow(2)  # Use param.data
                            _loss_sum = _loss_component.sum()
                            loss += _loss_sum
                            debugprint(f"[Rank {rank}] Added loss component for {name}: {_loss_sum.item()}")
                        except Exception as e:
                            logger.error(f"[Rank {rank}] Error calculating EWC loss for {name} in Stage 3 gather: {e}")
                            logger.error(traceback.format_exc())
                    else:
                         debugprint(f"[Rank {rank}] Parameter {name} not found in fisher_dict or param_dict within gather context. Skipping.")

                debugprint(f"[Rank {rank}] Finished processing all parameters inside gather_parameters context (Stage 3)")
                if is_distributed():
                    dist.barrier() # Sync after processing all params inside gather
                debugprint(f"[Rank {rank}] Barrier after processing loop inside gather passed (Stage 3)")

            debugprint(f"[Rank {rank}] Exited gather_parameters context (Stage 3)")
            if is_distributed():
                 dist.barrier() # Sync after exiting gather
            debugprint(f"[Rank {rank}] Barrier after exiting gather passed (Stage 3)")

        # 统一处理非ZeRO-3的情况
        else:
            debugprint(f"[Rank {rank}] Calculating EWC loss for non-ZeRO-3 (Stage {stage})")

            # 所有rank都计算自己的参数部分的损失
            for name, param in self.model.named_parameters():
                if name in self.fisher_dict and name in self.param_dict:
                    try:
                        param_star = self.param_dict[name].to(device)
                        fisher = self.fisher_dict[name].to(device)
                        _loss = fisher * (param.data - param_star).pow(2)
                        _loss_sum = _loss.sum()
                        loss += _loss_sum
                        debugprint(f"[Rank {rank}] Added loss component for {name}: {_loss_sum.item()}")
                    except Exception as e:
                        logger.error(f"[Rank {rank}] Error calculating EWC loss for {name}: {e}")
                        logger.error(traceback.format_exc())

            # 在分布式环境中，需要对损失进行all-reduce
            if is_distributed():
                debugprint(f"[Rank {rank}] All-reducing EWC loss")
                loss = all_reduce_tensor(loss, op=dist.ReduceOp.SUM)
                debugprint(f"[Rank {rank}] All-reduced EWC loss: {loss.item()}")

        debugprint(f"[Rank {rank}] Final EWC loss contribution (before lambda): {loss.item()}")
        return self.lambda_ewc * 0.5 * loss

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