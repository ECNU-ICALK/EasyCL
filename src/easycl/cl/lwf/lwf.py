import torch
import torch.nn as nn
import os
import json
from typing import Optional, Dict, List, Any, Union, Tuple
from tqdm import tqdm
from llamafactory.extras.logging import get_logger
from easycl.hparams import CLFinetuningArguments
import traceback
def debugprint(*args, **kwargs):
    pass
from easycl.cl.distributed_utils import (
    is_distributed, get_rank, is_main_process, get_world_size,
    get_deepspeed_zero_stage, all_reduce_tensor
)

def gather_parameters(model, params: Optional[List[torch.nn.Parameter]] = None):
    """
    在 ZeRO-3 下临时 All-Gather 完整参数的上下文管理器。
    其他场景返回 nullcontext。
    """
    from contextlib import nullcontext
    if get_deepspeed_zero_stage(model) == 3:
        try:
            import deepspeed

            if params is None:
                params = list(model.parameters())

            debugprint(f"[rank {get_rank()}] Gather {len(params)} params (ZeRO-3)")
            if not params:
                debugprint(f"[rank {get_rank()}] Warning: No parameters found to gather for model {type(model).__name__}. Returning nullcontext.")
                return nullcontext()
            return deepspeed.zero.GatheredParameters(params, modifier_rank=None)
        except Exception as e:
            debugprint(f"[rank {get_rank()}] GatheredParameters 出错: {e}")
            return nullcontext()
    else:
        return nullcontext()

logger = get_logger(__name__)

class LWF:
    def __init__(self, model: nn.Module, previous_task_model: Optional[nn.Module] = None, temperature: float = 2.0, alpha: float = 0.5):
        """Initialize LWF with distributed training support"""
        self.model = model
        self.temperature = temperature
        self.alpha = alpha
        self.enabled = True  # LWF status flag
        self.cached_logits = {}  # Dictionary to store cached logits by sample id (only valid tokens)

        # 检测分布式环境
        self.is_distributed = is_distributed()
        self.rank = get_rank()
        self.is_main = is_main_process()
        self.world_size = get_world_size()

        # 检测DeepSpeed ZeRO阶段
        self.zero_stage = get_deepspeed_zero_stage(model)

        debugprint(f"[rank {self.rank}] LWF __init__ 已调用.")
        debugprint(f"[rank {self.rank}] 收到的 temperature: {temperature}")
        debugprint(f"[rank {self.rank}] 收到的 alpha: {alpha}")
        debugprint(f"[rank {self.rank}] 分布式环境: {self.is_distributed}, ZeRO-Stage: {self.zero_stage}")

        if previous_task_model is not None:
            self.previous_task_model = previous_task_model
            debugprint(f"[rank {self.rank}] 已提供并分配先前任务模型.")
        else:
            self.previous_task_model = None
            logger.warning("No previous task model provided. LWF will be disabled.")
            self.enabled = False
            debugprint(f"[rank {self.rank}] 未提供先前任务模型。LWF 已禁用.")
        debugprint(f"[rank {self.rank}] LWF 初始化完成。已启用: {self.enabled}")

    def lwf_loss(self, logits: torch.Tensor, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate LWF loss with distributed training support"""
        debugprint(f"[rank {self.rank}] LWF lwf_loss 已调用。LWF 启用状态: {self.enabled}, 温度: {self.temperature}, Alpha: {self.alpha}, 先前模型是否存在: {self.previous_task_model is not None}")
        if not self.enabled or self.previous_task_model is None:
            debugprint(f"[rank {self.rank}] LWF 已禁用或无先前模型，返回零损失。")
            return torch.tensor(0.0, device=logits.device)

        try:
            # Get device information
            device = logits.device

            # Ensure input data is on the correct device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            # Create a copy of inputs without 'index' for model forward pass
            model_inputs = {k: v for k, v in inputs.items() if k != "index"}
            debugprint(f"[rank {self.rank}] Removed 'index' from inputs for previous model forward pass")

            # 在ZeRO-3下，需要使用gather_parameters上下文管理器访问模型参数
            # 对于previous_task_model，我们需要确保在forward过程中能够访问完整参数
            if self.zero_stage == 3 and hasattr(self.previous_task_model, 'parameters'):
                debugprint(f"[rank {self.rank}] ZeRO-3环境下使用gather_parameters获取先前模型完整参数")
                # 使用gather_parameters上下文管理器
                with gather_parameters(self.previous_task_model):
                    with torch.no_grad():
                        previous_outputs = self.previous_task_model(**model_inputs)
                        previous_logits = previous_outputs.logits
            else:
                # 其他情况下直接前向传播
                with torch.no_grad():
                    previous_outputs = self.previous_task_model(**model_inputs)
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

            # 在分布式环境中同步损失
            if self.is_distributed:
                total_loss = all_reduce_tensor(total_loss)

            debugprint(f"[rank {self.rank}] 计算得到的 LWF total_loss: {total_loss}")
            return total_loss

        except Exception as e:
            logger.error(f"[rank {self.rank}] Error in LWF loss calculation: {str(e)}")
            logger.error(traceback.format_exc())
            return torch.tensor(0.0, device=logits.device)

    def disable(self):
        """Disable LWF"""
        logger.info("Disabling LWF...")
        self.enabled = False
        self.previous_task_model = None
        # Clear cached logits to free memory
        self.clear_cached_logits()

    def enable(self):
        """Enable LWF"""
        logger.info("Enabling LWF...")
        self.enabled = True

    def clear_cached_logits(self):
        """Clear all cached logits to free memory with distributed training support"""
        if hasattr(self, 'cached_logits') and self.cached_logits:
            num_cleared = len(self.cached_logits)
            self.cached_logits.clear()
            if self.is_main:
                logger.info(f"Cleared {num_cleared} cached logits to free memory")
            debugprint(f"[rank {self.rank}] Cleared {num_cleared} cached logits to free memory")
        else:
            if self.is_main:
                logger.info("No cached logits to clear")
            debugprint(f"[rank {self.rank}] No cached logits to clear")

        # Force garbage collection and clear CUDA cache for thorough memory cleanup
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        debugprint(f"[rank {self.rank}] Performed garbage collection and CUDA cache cleanup")

        # 在分布式环境中，确保所有进程都完成了清空操作
        if self.is_distributed:
            debugprint(f"[rank {self.rank}] Reached barrier before exiting clear_cached_logits.")
            import torch.distributed as dist
            dist.barrier()
            debugprint(f"[rank {self.rank}] Passed barrier after clear_cached_logits.")
            if self.is_main:
                logger.info("All processes have completed clearing cached logits.")

    def set_logits_dir(self, output_dir: str):
        """Set the directory for LWF (kept for compatibility)"""
        # No longer creating directory or storing files
        logger.info("LWF logits are now only cached in memory, not saved to disk.")

    def precompute_logits(self, dataloader, device):
        """
        Precompute logits for all samples in the dataset using the previous model
        Only caches the valid tokens (non-padding) to save memory
        Supports distributed training with different DeepSpeed ZeRO stages

        Args:
            dataloader: DataLoader containing the dataset
            device: Device to run the model on
        """
        if not self.enabled or self.previous_task_model is None:
            logger.warning("LWF is disabled or no previous model available. Cannot precompute logits.")
            return

        # 在分布式环境中，只在主进程显示总体进度
        if self.is_main:
            logger.info("Precomputing logits for all samples using previous model (memory-only)...")

        debugprint(f"[rank {self.rank}] 开始预计算logits，ZeRO-Stage: {self.zero_stage}")
        # Add detailed dataloader length log
        try:
            dataloader_len = len(dataloader)
            debugprint(f"[rank {self.rank}] Dataloader length (estimated): {dataloader_len}")
        except TypeError:
            debugprint(f"[rank {self.rank}] Dataloader does not support len().")
            dataloader_len = None # Set to None if len is not supported


        self.previous_task_model.eval()
        self.previous_task_model.to(device)
        debugprint(f"[rank {self.rank}] Previous task model moved to device: {device}") # Add this line


        # Create a progress bar (only on main process)
        progress_bar = None # Initialize progress_bar
        if self.is_main and dataloader_len is not None: # Only create progress bar if length is known
            progress_bar = tqdm(total=dataloader_len, desc="Precomputing logits")
        elif self.is_main:
            debugprint(f"[rank {self.rank}] Cannot create progress bar because dataloader length is unknown.")


        # 在ZeRO-3下，我们需要使用gather_parameters上下文管理器
        # 但由于我们在循环中多次调用模型，所以在每次forward时使用上下文管理器

        debugprint(f"[rank {self.rank}] Entering precompute_logits loop...") # Add this line
        processed_batches = 0 # Counter for processed batches
        with torch.no_grad():
            # Use iter and next to potentially get more info if dataloader hangs on first iteration
            dataloader_iter = iter(dataloader)
            while True:
                try:
                    debugprint(f"[rank {self.rank}] Attempting to get batch {processed_batches} from dataloader...")
                    batch = next(dataloader_iter)
                    debugprint(f"[rank {self.rank}] Successfully got batch {processed_batches}.")
                except StopIteration:
                    debugprint(f"[rank {self.rank}] Dataloader finished after {processed_batches} batches.")
                    break # Exit loop when dataloader is exhausted
                except Exception as e:
                    debugprint(f"[rank {self.rank}] Error getting batch {processed_batches} from dataloader: {e}")
                    logger.error(f"[rank {self.rank}] Dataloader error: {traceback.format_exc()}")
                    # Decide if we should break or continue depending on the error
                    break # Stop precomputation on dataloader error


                batch_idx = processed_batches # Use counter as batch index
                debugprint(f"[rank {self.rank}] Inside loop: batch_idx = {batch_idx}") # Add this line


                # Check if batch contains sample ids
                if "index" not in batch:
                    debugprint(f"[rank {self.rank}] Batch {batch_idx} missing 'index'. Skipping.") # Modified log
                    if self.is_main and progress_bar: progress_bar.update(1) # Update progress even if skipping
                    processed_batches += 1
                    continue

                # Move batch to device
                try:
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    debugprint(f"[rank {self.rank}] Batch {batch_idx} moved to device {device}.") # Add this line
                except Exception as e:
                    debugprint(f"[rank {self.rank}] Error moving batch {batch_idx} to device: {e}") # Add this line
                    if self.is_main and progress_bar: progress_bar.update(1) # Update progress on error
                    processed_batches += 1
                    continue


                # Get sample ids
                sample_ids = batch["index"].tolist()
                debugprint(f"[rank {self.rank}] Batch {batch_idx} sample IDs: {sample_ids}") # Add this line


                # Get labels for creating attention mask
                labels = batch.get('labels', None)
                if labels is None:
                    debugprint(f"[rank {self.rank}] Batch {batch_idx} missing 'labels'. Skipping.") # Modified log
                    if self.is_main and progress_bar: progress_bar.update(1) # Update progress even if skipping
                    processed_batches += 1
                    continue

                debugprint(f"[rank {self.rank}] Batch {batch_idx} performing forward pass...") # Add this line
                
                # Create a copy of batch without 'index' for model forward pass
                model_batch = {k: v for k, v in batch.items() if k != "index"}
                debugprint(f"[rank {self.rank}] Removed 'index' from batch for model forward pass")
                
                # Forward pass - 在ZeRO-3下使用自定义的 gather_parameters
                try:
                    if self.zero_stage == 3:
                        debugprint(f"[rank {self.rank}] Using gather_parameters utility for forward pass (ZeRO-3)")
                        # Revert back to using the gather_parameters function from distributed_utils
                        with gather_parameters(self.previous_task_model):
                            outputs = self.previous_task_model(**model_batch)
                    else:
                        debugprint(f"[rank {self.rank}] Performing direct forward pass (ZeRO < 3)")
                        outputs = self.previous_task_model(**model_batch)
                    debugprint(f"[rank {self.rank}] Batch {batch_idx} forward pass completed.")
                except Exception as e:
                    debugprint(f"[rank {self.rank}] Error during forward pass for batch {batch_idx}: {e}")
                    logger.error(f"[rank {self.rank}] Forward pass failed for batch {batch_idx}: {traceback.format_exc()}")
                    if self.is_main and progress_bar: progress_bar.update(1) # Update progress on error
                    processed_batches += 1
                    continue # Skip caching for this batch


                logits = outputs.logits
                debugprint(f"[rank {self.rank}] Batch {batch_idx} got logits shape: {logits.shape}") # Add this line


                # Create attention mask to identify valid tokens
                attention_mask = labels.ne(-100)  # [batch_size, seq_len]

                # Save valid logits for each sample
                debugprint(f"[rank {self.rank}] Batch {batch_idx} processing {len(sample_ids)} samples for caching...") # Add this line
                num_cached_in_batch = 0
                for i, sample_id in enumerate(sample_ids):
                    # Get valid positions for this sample
                    sample_mask = attention_mask[i]  # [seq_len]

                    if not sample_mask.any():
                        # debugprint(f"[rank {self.rank}] Sample {sample_id} has no valid tokens. Skipping.") # Keep this commented unless needed
                        continue

                    # Extract logits for this sample
                    sample_logits = logits[i]  # [seq_len, vocab_size]

                    # Only keep valid positions to save memory
                    # Use as_tuple=False for compatibility with older torch versions if needed
                    valid_positions = sample_mask.nonzero(as_tuple=False).squeeze(-1)
                    if valid_positions.numel() == 0: # Check if any valid positions found
                         # debugprint(f"[rank {self.rank}] Sample {sample_id} had mask but no nonzero positions?") # Keep commented unless needed
                         continue
                    valid_logits = sample_logits[valid_positions].cpu()  # Move to CPU to save memory


                    # Cache in memory (store both valid logits and their positions)
                    self.cached_logits[sample_id] = {
                        'logits': valid_logits,
                        'positions': valid_positions.cpu()
                    }
                    num_cached_in_batch += 1

                debugprint(f"[rank {self.rank}] Batch {batch_idx} caching completed. Cached {num_cached_in_batch} samples.") # Add this line


                # Increment batch counter
                processed_batches += 1

                # 只在主进程更新进度条
                if self.is_main and progress_bar:
                    progress_bar.update(1)

            # End of while True loop
            debugprint(f"[rank {self.rank}] Exited precompute_logits loop after processing {processed_batches} batches.") # Add this line


        # 只在主进程关闭进度条并显示完成信息
        if self.is_main and progress_bar:
            progress_bar.close()
            logger.info(f"Logits precomputation complete. Cached {len(self.cached_logits)} samples in memory across all batches.")
        elif self.is_main:
             logger.info(f"Logits precomputation loop finished. Cached {len(self.cached_logits)} samples in memory.")


        # 在分布式环境中，确保所有进程都完成了预计算
        if self.is_distributed:
            debugprint(f"[rank {self.rank}] Reached barrier before exiting precompute_logits.") # Add this line
            import torch.distributed as dist
            dist.barrier()
            debugprint(f"[rank {self.rank}] Passed barrier after precompute_logits.") # Add this line


        debugprint(f"[rank {self.rank}] 预计算logits完成，缓存了 {len(self.cached_logits)} 个样本")

    def get_cached_logits(self, sample_id: int, device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get cached logits for a sample

        Args:
            sample_id: ID of the sample
            device: Device to load the logits to

        Returns:
            Dictionary containing 'logits' and 'positions' tensors, or None if not found
        """
        # Get from memory cache
        if sample_id in self.cached_logits:
            cached_data = self.cached_logits[sample_id]
            return {
                'logits': cached_data['logits'].to(device),
                'positions': cached_data['positions'].to(device)
            }

        # Not found
        return None

    def lwf_loss_with_cached_logits(self, logits: torch.Tensor, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate LWF loss using cached logits with distributed training support

        Args:
            logits: Current model logits
            inputs: Input batch containing sample ids

        Returns:
            LWF loss tensor
        """
        if not self.enabled:
            return torch.tensor(0.0, device=logits.device)

        try:
            # Get device information
            device = logits.device

            # Check if inputs contain sample ids
            if "index" not in inputs:
                if self.is_main:
                    logger.warning("No sample ids found in inputs. Cannot use cached logits.")
                return torch.tensor(0.0, device=device)

            # Get sample ids
            sample_ids = inputs["index"].tolist()

            # Create mask for filtering padding
            labels = inputs.get('labels', None)
            if labels is None:
                if self.is_main:
                    logger.warning("No labels found in inputs")
                return torch.tensor(0.0, device=device)

            # Create attention mask
            attention_mask = labels.ne(-100)  # [batch_size, seq_len]

            # Get batch_size and vocab_size
            batch_size, seq_len, vocab_size = logits.shape

            # Reshape current logits
            current_logits = logits.view(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
            flat_attention_mask = attention_mask.view(-1)  # [batch_size * seq_len]

            # Select only non-padding positions for current logits
            valid_current_logits = current_logits[flat_attention_mask]
            valid_labels = labels[attention_mask]

            if valid_current_logits.shape[0] == 0:
                if self.is_main:
                    logger.warning("No valid tokens found for distillation")
                return torch.tensor(0.0, device=device)

            # Process each sample and collect valid previous logits
            valid_previous_logits_list = []

            for batch_idx, sample_id in enumerate(sample_ids):
                # Get cached data for this sample
                cached_data = self.get_cached_logits(sample_id, device)
                if cached_data is None:
                    if self.is_main:
                        logger.warning(f"No cached logits found for sample {sample_id}")
                    return torch.tensor(0.0, device=device)

                # Get the current sample's attention mask
                sample_mask = attention_mask[batch_idx]
                sample_valid_positions = sample_mask.nonzero().squeeze(-1)

                # Get the cached logits and their positions
                cached_logits = cached_data['logits']  # [cached_positions, vocab_size]
                cached_positions = cached_data['positions']  # [cached_positions]

                # Find the intersection of current valid positions and cached positions
                # This ensures we only use positions that are valid in both current and cached data
                for pos in sample_valid_positions:
                    pos_idx = (cached_positions == pos).nonzero()
                    if pos_idx.numel() > 0:
                        pos_idx = pos_idx.item()
                        valid_previous_logits_list.append(cached_logits[pos_idx])

            # Stack all valid previous logits
            if not valid_previous_logits_list:
                if self.is_main:
                    logger.warning("No matching valid positions found between current and cached logits")
                return torch.tensor(0.0, device=device)

            valid_previous_logits = torch.stack(valid_previous_logits_list, dim=0)

            # Ensure we have the same number of logits
            if valid_current_logits.shape[0] != valid_previous_logits.shape[0]:
                if self.is_main:
                    logger.warning(f"Mismatch in number of valid tokens: current {valid_current_logits.shape[0]}, cached {valid_previous_logits.shape[0]}")
                # Use the minimum number of tokens
                min_tokens = min(valid_current_logits.shape[0], valid_previous_logits.shape[0])
                valid_current_logits = valid_current_logits[:min_tokens]
                valid_previous_logits = valid_previous_logits[:min_tokens]
                valid_labels = valid_labels[:min_tokens]

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
            ce_loss = nn.functional.cross_entropy(valid_current_logits, valid_labels)

            # Combine losses
            total_loss = (1 - self.alpha) * ce_loss + self.alpha * distillation_loss

            # 在分布式环境中同步损失
            if self.is_distributed:
                total_loss = all_reduce_tensor(total_loss)

            debugprint(f"[rank {self.rank}] 计算得到的 LWF total_loss (使用内存缓存): {total_loss}")
            return total_loss

        except Exception as e:
            logger.error(f"[rank {self.rank}] Error in LWF loss calculation with cached logits: {str(e)}")
            logger.error(traceback.format_exc())
            return torch.tensor(0.0, device=logits.device)
