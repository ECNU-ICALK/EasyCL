import torch
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union, Any, Callable
from contextlib import nullcontext

def debugprint(*args, **kwargs):
    pass

from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from llamafactory.extras.logging import get_logger
from easycl.cl.distributed_utils import (
    is_distributed, get_rank, get_world_size, is_main_process,
    get_deepspeed_zero_stage, gather_parameters, synchronize_gradients,
    all_reduce_tensor, broadcast_object
)

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
    def __init__(self, finetuning_args: "FinetuningArguments", cl_finetuning_args: "CLFinetuningArguments", processor: Optional[Any] = None, *args, **kwargs):
        # Pass finetuning_args and processor to the parent class constructor
        super().__init__(finetuning_args=finetuning_args, processor=processor, *args, **kwargs)
        self.cl_finetuning_args = cl_finetuning_args
        # Use rank in debug print
        debugprint(f"[RANK {get_rank()}] GEM Trainer __init__: 传入的 cl_finetuning_args: {cl_finetuning_args}") # Debug print for CL args
        if not cl_finetuning_args.use_gem:
             # Log a warning or potentially raise an error if GEM trainer is used without the flag
             logger.warning("GEMSeq2SeqTrainer initialized, but use_gem in cl_finetuning_args is False.")
             # Store a default or disable GEM specific logic
             self.gem_memory_strength = 0.0 # Effectively disables projection
             debugprint(f"[RANK {get_rank()}] GEM Trainer __init__: use_gem 为 False, GEM 强度设置为 0.0") # Debug print for GEM disabled
        else:
            self.gem_memory_strength = cl_finetuning_args.gem_memory_strength
            logger.info(f"GEM Trainer initialized with memory strength: {self.gem_memory_strength}")
            debugprint(f"[RANK {get_rank()}] GEM Trainer __init__: use_gem 为 True, GEM 强度设置为: {self.gem_memory_strength}") # Debug print for GEM enabled

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
        rank = get_rank() # Get rank for logging
        debugprint(f"[RANK {rank}] GEM compute_loss: 方法入口, is_distributed: {is_distributed()}, 输入键: {list(inputs.keys())}, return_outputs: {return_outputs}") # Debug print at method start
        # If GEM is disabled (strength=0) or not use_gem flag, revert to standard loss computation
        if not hasattr(self, 'gem_memory_strength') or self.gem_memory_strength <= 0:
             debugprint(f"[RANK {rank}] GEM compute_loss: GEM 强度 <= 0 或未设置, 执行标准 compute_loss") # Debug print for standard path
             return super().compute_loss(model, inputs, return_outputs=return_outputs)

        if "is_memory" not in inputs or not isinstance(inputs["is_memory"], torch.Tensor):
            debugprint(f"[RANK {rank}] GEM compute_loss: 错误 - 输入中缺少 'is_memory' 字段") # Debug print for missing 'is_memory'
            raise ValueError(
                "GEMTrainer requires each batch sample to have an 'is_memory' boolean tensor field "
                "to distinguish current task vs. memory samples."
            )

        debugprint(f"[RANK {rank}] GEM compute_loss: 找到 'is_memory' 字段, 形状: {inputs['is_memory'].shape}") # Debug print for 'is_memory' found

        is_memory = inputs["is_memory"]
        current_mask = ~is_memory
        memory_mask = is_memory
        debugprint(f"[RANK {rank}] GEM compute_loss: current_mask 中 True 的数量: {current_mask.sum()}, memory_mask 中 True 的数量: {memory_mask.sum()}") # Debug print for mask counts

        # Handle cases where a batch might only contain current or memory samples
        has_current_samples = torch.any(current_mask)
        has_memory_samples = torch.any(memory_mask)
        debugprint(f"[RANK {rank}] GEM compute_loss: 是否有当前样本: {has_current_samples}, 是否有记忆样本: {has_memory_samples}") # Debug print for sample presence

        # If no current samples, only calculate memory loss for potential stats, but return 0 loss
        # as we don't optimize based on memory alone in this step.
        if not has_current_samples:
            logger.warning_once(f"[RANK {rank}] Received a batch with only memory samples. Skipping GEM gradient calculation for this batch.")
            
            effective_loss = torch.tensor(1e-9, device=model.device, requires_grad=True) # MODIFIED DEFAULT to small epsilon

            if has_memory_samples:
                # Try to compute a real loss on memory samples then zero it out
                # to ensure gradients are populated as zeros rather than None.
                # This helps avoid issues with optimizers like DeepSpeed during gradient norm calculation.
                memory_inputs_for_dummy_pass = {
                    k: v[memory_mask]
                    for k, v in inputs.items()
                    # Exclude 'is_memory' and ensure tensor is per-sample and compatible
                    if k != "is_memory" and isinstance(v, torch.Tensor) and v.shape[0] == inputs["is_memory"].shape[0]
                }
                # Filter out tensors that became empty after masking
                memory_inputs_for_dummy_pass = {
                    k: val_masked for k, val_masked in memory_inputs_for_dummy_pass.items() if val_masked.numel() > 0
                }

                if memory_inputs_for_dummy_pass.get("input_ids") is not None: # 'input_ids' is crucial
                    try:
                        debugprint(f"[RANK {rank}] GEM compute_loss: Memory-only batch. Performing forward pass on memory data to establish graph for zero gradients. Input keys: {list(memory_inputs_for_dummy_pass.keys())}")
                        # This compute_loss is from CustomSeq2SeqTrainer or its parent.
                        loss_from_mem_samples = super().compute_loss(model, memory_inputs_for_dummy_pass, return_outputs=False)

                        # Multiply by 0.0. If loss_from_mem_samples requires grad, the result should too.
                        # MODIFIED: Ensure effective_loss is a small epsilon while retaining graph properties
                        effective_loss_zero_contrib = loss_from_mem_samples * 0.0
                        effective_loss = effective_loss_zero_contrib + torch.tensor(1e-9, device=effective_loss_zero_contrib.device, requires_grad=False)
                        debugprint(f"[RANK {rank}] GEM compute_loss: Loss from memory samples (before zeroing attempt): {loss_from_mem_samples.item()}. Effective loss for backward: {effective_loss.item()}. Requires grad: {effective_loss.requires_grad}")

                        # Sanity check: ensure it still requires grad if original did and operation lost it
                        if loss_from_mem_samples.requires_grad and not effective_loss.requires_grad:
                            effective_loss = effective_loss.clone().detach().requires_grad_(True)
                            debugprint(f"[RANK {rank}] GEM compute_loss: Re-enabled requires_grad for epsilon-adjusted effective_loss.")
                    
                    except Exception as e:
                        logger.error(f"[RANK {rank}] Error during dummy forward pass on memory samples for zero-grad purpose: {e}. Defaulting to small epsilon tensor.")
                        debugprint(f"[RANK {rank}] GEM compute_loss: Error in dummy forward pass: {e}. Using default small epsilon tensor.")
                        # Fallback to the original simple tensor if dummy pass fails
                        effective_loss = torch.tensor(1e-9, device=model.device, requires_grad=True) # MODIFIED
                else:
                    debugprint(f"[RANK {rank}] GEM compute_loss: Memory-only batch, but 'input_ids' missing/empty for dummy pass. Using default small epsilon tensor.")
                    effective_loss = torch.tensor(1e-9, device=model.device, requires_grad=True) # MODIFIED
            else:
                # No current samples AND no memory samples (e.g., empty batch after filtering, or malformed initial batch)
                # effective_loss is already 1e-9 from the MODIFIED DEFAULT above.
                debugprint(f"[RANK {rank}] GEM compute_loss: Batch has neither current nor memory samples. Using default small epsilon tensor (already set).")
                # effective_loss = torch.tensor(1e-9, device=model.device, requires_grad=True) # This line is redundant due to modified default

            # if this is a memory-only batch (no current_samples),
            # the effective_loss is designed to produce zero gradients for model parameters.
            # However, DeepSpeed's BF16Optimizer asserts all_groups_norm > 0 if grad clipping is on.
            # To prevent this, we add a tiny, negligible loss component tied to one parameter,
            # ensuring at least one parameter has a non-zero gradient.
            # This applies if gradient clipping seems to be active (max_grad_norm > 0).
            # self.args should be accessible here as it's part of the Trainer class.
            if self.args.max_grad_norm > 0 and hasattr(model, "parameters"):
                param_perturbed = False
                for param in model.parameters(): # Iterate to find a parameter that requires grad
                    if param.requires_grad:
                        if effective_loss.requires_grad: # Ensure base effective_loss is valid
                            perturbation_coeff = 1e-12 # Use a very small coefficient
                            # Add a tiny loss component: param.sum() * very_small_number
                            # This makes param.grad slightly non-zero.
                            # .sum() handles various param shapes and makes it a scalar loss component.
                            try:
                                effective_loss = effective_loss + (param.sum() * perturbation_coeff)
                                debugprint(f"[RANK {get_rank()}] GEM compute_loss: Added tiny perturbation (coeff: {perturbation_coeff}) to effective_loss using one param to ensure non-zero grad norm for BF16 optimizer.")
                                param_perturbed = True
                                break # Perturb only one parameter
                            except RuntimeError as e:
                                # This might happen if param.sum() is not compatible with effective_loss device/dtype
                                logger.warning_once(f"[RANK {get_rank()}] GEM compute_loss: Error during perturbation: {e}. Skipping.")
                                break 
                        else:
                            logger.warning_once(f"[RANK {get_rank()}] GEM compute_loss: effective_loss does not require_grad before perturbation attempt in memory-only batch. Skipping perturbation.")
                            break # Skip if base effective_loss is faulty
                
                if not param_perturbed and any(p.requires_grad for p in model.parameters()):
                    logger.warning_once(f"[RANK {get_rank()}] GEM compute_loss: Could not apply perturbation in memory-only batch (no suitable param or effective_loss issue), grad norm might still be zero.")

            if return_outputs:
                # For outputs, since we are skipping current task, there are no "current" outputs.
                debugprint(f"[RANK {rank}] GEM compute_loss: Memory-only batch, returning effective_loss: {effective_loss.item()} and None outputs")
                return (effective_loss, None)
            else:
                debugprint(f"[RANK {rank}] GEM compute_loss: Memory-only batch, returning effective_loss: {effective_loss.item()}")
                return effective_loss

        # Separate inputs
        # Ensure we handle metadata columns gracefully (like 'is_memory')
        current_inputs = {
            k: v[current_mask] 
            for k, v in inputs.items() 
            if k != "is_memory" and isinstance(v, torch.Tensor) and v.shape[0] == inputs["is_memory"].shape[0]
        }
        debugprint(f"[RANK {rank}] GEM compute_loss: 分离出当前任务输入, 键: {list(current_inputs.keys())}") # Debug print for separated current inputs

        # Step 1: Compute current task loss and potentially outputs
        # We need the loss object itself for backpropagation
        debugprint(f"[RANK {rank}] GEM compute_loss: 开始计算当前任务损失") # Debug print before current loss calc
        loss_current_tuple = super().compute_loss(model, current_inputs, return_outputs=True) # Request outputs
        loss_current = loss_current_tuple[0]
        outputs_current = loss_current_tuple[1]
        debugprint(f"[RANK {rank}] GEM compute_loss: 计算得到当前任务损失: {loss_current.item() if isinstance(loss_current, torch.Tensor) else loss_current}") # Debug print after current loss calc

        # Step 2: Compute gradient for the current task
        debugprint(f"[RANK {rank}] GEM compute_loss: 开始计算当前任务梯度") # Debug print before current grad calc
        g_current = self._get_grads(model, loss_current)
        if g_current is None:
             logger.warning_once(f"[RANK {rank}] Could not compute gradients for the current task batch. Skipping GEM projection.")
             debugprint(f"[RANK {rank}] GEM compute_loss: 无法计算当前任务梯度, 跳过 GEM 投影") # Debug print for failed current grad calc
             # Return the original loss and outputs if gradient calculation failed
             return (loss_current.detach(), outputs_current) if return_outputs else loss_current.detach()
        debugprint(f"[RANK {rank}] GEM compute_loss: 计算得到当前任务梯度 g_current, 形状: {g_current.shape}") # Debug print after current grad calc
        if g_current is not None:
            g_current_norm = torch.norm(g_current).item()
            g_current_l1 = torch.sum(torch.abs(g_current)).item()
            debugprint(f"[RANK {rank}] GEM compute_loss: 当前任务梯度 g_current - L2范数: {g_current_norm:.4f}, L1范数: {g_current_l1:.4f}, 设备: {g_current.device}")

        # If no memory samples in this batch, behave like normal training
        if not has_memory_samples:
            # Gradients are already on the parameters from _get_grads (due to retain_graph=True),
            # so just return the loss. The optimizer step will use these gradients.
            debugprint(f"[RANK {rank}] GEM compute_loss: 无记忆样本, 返回原始当前任务损失和梯度") # Debug print for no memory samples
            return (loss_current, outputs_current) if return_outputs else loss_current

        # Step 3: Compute memory loss and gradient (only if memory samples exist)
        memory_inputs = {
            k: v[memory_mask] 
            for k, v in inputs.items() 
            if k != "is_memory" and isinstance(v, torch.Tensor) and v.shape[0] == inputs["is_memory"].shape[0]
        }
        debugprint(f"[RANK {rank}] GEM compute_loss: 分离出记忆任务输入, 键: {list(memory_inputs.keys())}") # Debug print for separated memory inputs
        loss_memory = super().compute_loss(model, memory_inputs, return_outputs=False)
        debugprint(f"[RANK {rank}] GEM compute_loss: 计算得到记忆任务损失: {loss_memory.item() if isinstance(loss_memory, torch.Tensor) else loss_memory}") # Debug print after memory loss calc
        # Detach the memory loss before computing grads for projection?
        # The gradient itself should be detached in _project_gem_qp. Loss needs grad_fn.
        debugprint(f"[RANK {rank}] GEM compute_loss: 开始计算记忆任务梯度") # Debug print before memory grad calc
        g_memory = self._get_grads(model, loss_memory)

        if g_memory is None:
            logger.warning_once(f"[RANK {rank}] Could not compute gradients for the memory samples in this batch. Skipping GEM projection.")
            debugprint(f"[RANK {rank}] GEM compute_loss: 无法计算记忆任务梯度, 跳过 GEM 投影") # Debug print for failed memory grad calc
            # Proceed as if no memory samples were present
            return (loss_current, outputs_current) if return_outputs else loss_current
        # Print memory gradient info only if it was successfully computed
        debugprint(f"[RANK {rank}] GEM compute_loss: 计算得到记忆任务梯度 g_memory, 形状: {g_memory.shape}") # Debug print after memory grad calc
        g_memory_norm = torch.norm(g_memory).item()
        g_memory_l1 = torch.sum(torch.abs(g_memory)).item()
        debugprint(f"[RANK {rank}] GEM compute_loss: 记忆任务梯度 g_memory - L2范数: {g_memory_norm:.4f}, L1范数: {g_memory_l1:.4f}, 设备: {g_memory.device}")

        # Step 4: Check for conflict and project if needed
        # Detach gradients before dot product to ensure it's just a value comparison
        g_current_detached = g_current.detach()
        g_memory_detached = g_memory.detach()
        dot_product = torch.dot(g_current_detached, g_memory_detached)

        # 在分布式环境中，确保所有进程计算相同的点积结果 - 移除！
        # 对于 ZeRO Stage 0 (DDP-like), backward() 之后梯度已经同步，
        # 点积结果在各 rank 上理论相同，不需要再次 all_reduce。
        # 显式 all_reduce 可能会干扰 DDP 的同步并导致死锁。
        # if is_distributed():
        #     debugprint(f"[RANK {rank}] GEM compute_loss: [REMOVED] 分布式环境中执行点积all-reduce")
        #     # dot_product = all_reduce_tensor(dot_product) # Removed this line

        debugprint(f"[RANK {rank}] GEM compute_loss: 计算得到本地梯度点积 (g_current . g_memory): {dot_product.item()}") # Debug print for dot product

        if dot_product < 0:
            logger.debug(f"[RANK {rank}] GEM conflict detected (dot product: {dot_product:.4f}). Projecting gradient.")
            debugprint(f"[RANK {rank}] GEM compute_loss: 检测到冲突 (点积 < 0), 准备进行 GEM 梯度投影") # Debug print for conflict detected
            # Note: Pass the original g_current (with grad info if needed by proj) and detached memory grads
            debugprint(f"[RANK {rank}] GEM compute_loss: 调用 _project_gem_qp") # Debug print before projection call
            g_projected = self._project_gem_qp(g_current, [g_memory_detached]) # Pass detached memory grad to QP
            debugprint(f"[RANK {rank}] GEM compute_loss: GEM 投影完成, 得到投影后的梯度 g_projected, 形状: {g_projected.shape}") # Debug print after projection
            # Assign the projected gradient back to the model parameters
            debugprint(f"[RANK {rank}] GEM compute_loss: 开始将投影后的梯度 g_projected 分配给模型参数") # Debug print before assigning projected grad
            self._assign_grads(model, g_projected)
            debugprint(f"[RANK {rank}] GEM compute_loss: 分配投影后的梯度 g_projected 完成") # Debug print after assigning projected grad
            # The loss returned is still the original current task loss
            loss_to_return = loss_current
        else:
            logger.debug(f"[RANK {rank}] No GEM conflict (dot product: {dot_product:.4f}). Using original gradient.")
            debugprint(f"[RANK {rank}] GEM compute_loss: 未检测到冲突 (点积 >= 0), 将使用原始当前任务梯度 g_current") # Debug print for no conflict
            # If no conflict, the gradients computed by _get_grads(loss_current) are already
            # associated with the parameters (due to retain_graph=True in _get_grads).
            # We just need to return the loss.
            loss_to_return = loss_current

        # Return the original current task loss (and outputs if requested)
        # The optimizer will use the gradients currently assigned to model.parameters()
        # (which might be the original g_current or the projected g_projected)
        debugprint(f"[RANK {rank}] GEM compute_loss: 方法出口, 返回损失: {loss_to_return.item() if isinstance(loss_to_return, torch.Tensor) else loss_to_return}") # Debug print at method exit
        return (loss_to_return, outputs_current) if return_outputs else loss_to_return

    def _get_grads(self, model: "PreTrainedModel", loss: torch.Tensor) -> Optional[torch.Tensor]:
        """Computes and returns the flattened gradients for the given loss."""
        rank = get_rank() # Get rank for logging
        debugprint(f"[RANK {rank}] GEM _get_grads: 方法入口, is_distributed: {is_distributed()}, 损失值: {loss.item() if isinstance(loss, torch.Tensor) and loss.numel() == 1 else loss}, 是否需要梯度: {loss.requires_grad if isinstance(loss, torch.Tensor) else '非张量'}") # Debug print at method start
        if loss == 0.0 or not loss.requires_grad:
             debugprint(f"[RANK {rank}] GEM _get_grads: 损失为 0 或不需要梯度, 返回 None") # Debug print for zero/no-grad loss
             return None # Cannot compute gradients if loss is zero or doesn't require grad

        # 检测DeepSpeed ZeRO阶段
        zero_stage = get_deepspeed_zero_stage(model)
        debugprint(f"[RANK {rank}] GEM _get_grads: 检测到 DeepSpeed ZeRO Stage: {zero_stage}")

        model.zero_grad() # Ensure grads are clean before backward
        debugprint(f"[RANK {rank}] GEM _get_grads: 执行 model.zero_grad()") # Debug print after zero_grad

        # Need to handle potential DistributedDataParallel scenarios if applicable
        try:
            debugprint(f"[RANK {rank}] GEM _get_grads: 开始执行 loss.backward(retain_graph=True)") # Debug print before backward
            loss.backward(retain_graph=True) # Retain graph is crucial for computing multiple grads
            debugprint(f"[RANK {rank}] GEM _get_grads: loss.backward(retain_graph=True) 执行成功") # Debug print after successful backward
        except RuntimeError as e:
            logger.error(f"[RANK {rank}] Error during backward pass: {e}. This might happen if the graph was already freed.")
            debugprint(f"[RANK {rank}] GEM _get_grads: loss.backward 发生错误: {e}") # Debug print for backward error
            return None # Indicate failure

        # 对于ZeRO-2，需要同步梯度 (ZeRO-0 handled by DDP within backward)
        if zero_stage == 2:
            debugprint(f"[RANK {rank}] GEM _get_grads: ZeRO-2 模式，执行梯度同步")
            synchronize_gradients(model)
            debugprint(f"[RANK {rank}] GEM _get_grads: ZeRO-2 模式，梯度同步完成")

        # 收集梯度
        grads = []
        debugprint(f"[RANK {rank}] GEM _get_grads: 开始收集参数梯度") # Debug print before collecting grads
        non_zero_grads_count = 0
        grads_l1_sum = 0.0

        # 对于ZeRO-3，需要使用gather_parameters上下文管理器
        gather_context = gather_parameters(model) if zero_stage == 3 else nullcontext()
        debugprint(f"[RANK {rank}] GEM _get_grads: ZeRO Stage {zero_stage}, 使用 gather context: {zero_stage == 3}")

        with gather_context:
            debugprint(f"[RANK {rank}] GEM _get_grads: 进入参数梯度收集循环 (ZeRO-3 使用 gather context)")
            for name, param in model.named_parameters(): # Iterate with names for potential logging
                if param.requires_grad and param.grad is not None:
                    # debugprint(f"[RANK {rank}] GEM _get_grads: 收集到参数 '{name}' 的梯度, 形状: {param.grad.shape}") # Verbose
                    grad_view = param.grad.view(-1)
                    grads.append(grad_view)
                    grad_l1 = torch.sum(torch.abs(grad_view)).item()
                    grads_l1_sum += grad_l1
                    if grad_l1 > 1e-9: # Check if grad is effectively non-zero
                        non_zero_grads_count += 1
                elif param.requires_grad and param.grad is None:
                    # logger.warning_once(f"Parameter {name} requires grad but grad is None")
                    # debugprint(f"[RANK {rank}] GEM _get_grads: 警告 - 参数 '{name}' 需要梯度但梯度为 None") # Debug print for None grad
                    pass # Potentially handle parameters that don't get gradients?
            debugprint(f"[RANK {rank}] GEM _get_grads: 参数梯度收集循环结束")

        if not grads:
            logger.warning_once(f"[RANK {rank}] No gradients found for the model parameters.")
            debugprint(f"[RANK {rank}] GEM _get_grads: 未找到任何梯度") # Debug print for no grads found
            return None

        # It's important gradients are on the same device for dot product / projection
        # Assuming all grads will be on the same device as the first one
        debugprint(f"[RANK {rank}] GEM _get_grads: 收集到 {len(grads)} 个梯度张量, 其中 {non_zero_grads_count} 个非零 (L1 > 1e-9), L1范数总和: {grads_l1_sum:.4f}") # Debug print after collecting grads
        device = grads[0].device
        debugprint(f"[RANK {rank}] GEM _get_grads: 收集到 {len(grads)} 个梯度张量, 目标设备: {device}") # Debug print before concatenating
        try:
            flat_grads = torch.cat(grads).to(device)
            debugprint(f"[RANK {rank}] GEM _get_grads: 梯度本地扁平化成功, 形状: {flat_grads.shape}, 设备: {flat_grads.device}")

            # 在分布式环境中，仅当 ZeRO stage > 0 时才需要手动 all-reduce 梯度
            # ZeRO-0 (DDP) 会在 backward() 中自动完成 all-reduce
            if zero_stage > 1: # Changed condition from is_distributed() and zero_stage > 0
                debugprint(f"[RANK {rank}] GEM _get_grads: ZeRO Stage {zero_stage} > 1, 在分布式环境中执行梯度 all-reduce")
                flat_grads = all_reduce_tensor(flat_grads)
                debugprint(f"[RANK {rank}] GEM _get_grads: 梯度 all-reduce 完成, 形状: {flat_grads.shape}, 设备: {flat_grads.device}")
            # else: # Optional: Log when skipping all_reduce for ZeRO-0
            #     debugprint(f"[RANK {rank}] GEM _get_grads: ZeRO Stage 0, 跳过手动梯度 all-reduce (由 DDP 处理)")

            flat_grads_norm = torch.norm(flat_grads).item()
            flat_grads_l1 = torch.sum(torch.abs(flat_grads)).item()
            debugprint(f"[RANK {rank}] GEM _get_grads: 最终扁平梯度 - 形状: {flat_grads.shape}, 设备: {flat_grads.device}, L2范数: {flat_grads_norm:.4f}, L1范数: {flat_grads_l1:.4f}") # Debug print after successful concatenation and potential all-reduce
            return flat_grads
        except Exception as e:
            logger.error(f"[RANK {rank}] Error concatenating or reducing gradients: {e}")
            debugprint(f"[RANK {rank}] GEM _get_grads: 梯度拼接或 reduce 时出错: {e}") # Debug print for concatenation/reduce error
            return None

    def _assign_grads(self, model: "PreTrainedModel", flat_grad: torch.Tensor):
        """Assigns the flattened gradient `flat_grad` back to the model parameters."""
        rank = get_rank() # Get rank for logging
        debugprint(f"[RANK {rank}] GEM _assign_grads: 方法入口, is_distributed: {is_distributed()}, 待分配的扁平梯度形状: {flat_grad.shape}") # Debug print at method start

        # 检测DeepSpeed ZeRO阶段
        zero_stage = get_deepspeed_zero_stage(model)
        debugprint(f"[RANK {rank}] GEM _assign_grads: 检测到 DeepSpeed ZeRO Stage: {zero_stage}")

        # 对于ZeRO-3，需要使用gather_parameters上下文管理器
        gather_context = gather_parameters(model) if zero_stage == 3 else nullcontext()
        debugprint(f"[RANK {rank}] GEM _assign_grads: ZeRO Stage {zero_stage}, 使用 gather context: {zero_stage == 3}")

        with gather_context:
            debugprint(f"[RANK {rank}] GEM _assign_grads: 进入参数梯度分配循环 (ZeRO-3 使用 gather context)")
            pointer = 0
            assigned_count = 0
            for name, param in model.named_parameters(): # Iterate with names for potential logging
                if param.requires_grad and param.grad is not None: # Check grad is not None
                    numel = param.grad.numel()
                    # debugprint(f"[RANK {rank}] GEM _assign_grads: 正在为参数 '{name}' (元素数量: {numel}) 分配梯度切片 (指针: {pointer})") # Verbose
                    if pointer + numel > flat_grad.numel():
                        logger.error(f"[RANK {rank}] Gradient assignment error: flat_grad is smaller than the total number of gradient elements for param '{name}'.")
                        debugprint(f"[RANK {rank}] GEM _assign_grads: 错误 - 扁平梯度大小不足以分配给参数 '{name}' (指针: {pointer}, 需要: {numel}, 总大小: {flat_grad.numel()})") # Debug print for size mismatch error
                        break
                    # Ensure device compatibility before copying
                    param.grad.copy_(flat_grad[pointer : pointer + numel].view_as(param.grad).to(param.device))
                    pointer += numel
                    assigned_count += 1
                # elif param.requires_grad and param.grad is None:
                    # If grad was None initially, should we assign it?
                    # Generally no, stick to params that had grads.
                    # debugprint(f"[RANK {rank}] GEM _assign_grads: 跳过参数 '{name}', 因为其原始梯度为 None") # Debug print for skipped None grad param
                    # pass
            debugprint(f"[RANK {rank}] GEM _assign_grads: 分配循环完成, 共分配 {assigned_count} 个参数梯度, 最终指针位置: {pointer}") # Debug print after loop
            if pointer != flat_grad.numel():
                 logger.warning(f"[RANK {rank}] Gradient assignment warning: Mismatch in number of elements. Assigned {pointer}, flat_grad had {flat_grad.numel()}")
                 debugprint(f"[RANK {rank}] GEM _assign_grads: 警告 - 指针最终位置 {pointer} 与扁平梯度大小 {flat_grad.numel()} 不匹配") # Debug print for final size mismatch

        # 对于ZeRO-2，需要同步梯度 (ZeRO-0 handled by DDP)
        if zero_stage == 2:
            debugprint(f"[RANK {rank}] GEM _assign_grads: ZeRO-2 模式，执行梯度同步")
            synchronize_gradients(model)
            debugprint(f"[RANK {rank}] GEM _assign_grads: ZeRO-2 模式，梯度同步完成")

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
        rank = get_rank() # Get rank for logging
        debugprint(f"[RANK {rank}] GEM _project_gem_qp: 方法入口, 当前梯度形状: {g_current.shape}, 记忆梯度数量: {len(memory_grads)}, 优化器最大迭代次数: {max_iter}") # Debug print at method start
        g_current = g_current.detach() # Ensure we don't modify the original grad tensor directly
        # Initialize the projected gradient as a parameter, starting from the current gradient
        g_proj = torch.nn.Parameter(g_current.clone(), requires_grad=True)
        debugprint(f"[RANK {rank}] GEM _project_gem_qp: 初始化投影梯度参数 (g_proj), 初始值来自 g_current, 形状: {g_proj.shape}") # Debug print after initializing g_proj

        # Use LBFGS optimizer on the projected gradient parameter
        optimizer = torch.optim.LBFGS([g_proj], max_iter=max_iter, line_search_fn="strong_wolfe")
        debugprint(f"[RANK {rank}] GEM _project_gem_qp: 初始化 LBFGS 优化器, max_iter={max_iter}") # Debug print after optimizer init

        # Get the penalty strength from the instance variable
        penalty_strength = self.gem_memory_strength
        debugprint(f"[RANK {rank}] GEM _project_gem_qp: 使用的惩罚强度 (gem_memory_strength): {penalty_strength}") # Debug print for penalty strength

        # Define the closure for the optimizer
        def closure():
            # Closure rank might be useful if called across processes, though LBFGS usually runs locally
            closure_rank = get_rank()
            debugprint(f"[RANK {closure_rank}] GEM _project_gem_qp closure: 开始计算损失和梯度") # Debug print at closure start
            optimizer.zero_grad()
            # Primary objective: minimize distance to original gradient
            loss = 0.5 * torch.norm(g_proj - g_current) ** 2
            debugprint(f"[RANK {closure_rank}] GEM _project_gem_qp closure: 基础距离损失 ||g' - g_current||^2 / 2: {loss.item()}") # Debug print for base loss

            # Constraint penalty: add penalty if dot product is negative
            for i, g_mem in enumerate(memory_grads):
                # Ensure memory gradient is detached and on the correct device
                g_mem_detached = g_mem.detach().to(g_proj.device)
                dot_prod = torch.dot(g_proj, g_mem_detached)
                debugprint(f"[RANK {closure_rank}] GEM _project_gem_qp closure: 计算与第 {i} 个记忆梯度的点积: {dot_prod.item()}") # Debug print for dot product in closure
                # Only apply penalty if the constraint is violated (dot_prod < 0)
                if dot_prod < 0:
                    penalty = -penalty_strength * dot_prod
                    debugprint(f"[RANK {closure_rank}] GEM _project_gem_qp closure: 点积 < 0, 添加惩罚: {penalty.item()} (强度: {penalty_strength})") # Debug print for applied penalty
                    # Add penalty proportional to the violation magnitude
                    loss = loss + penalty # Maximize dot product towards >= 0 (original paper minimizes -dot, so equivalent to adding -strength * dot when dot < 0)

            debugprint(f"[RANK {closure_rank}] GEM _project_gem_qp closure: 添加惩罚后的总损失: {loss.item()}") # Debug print for total loss in closure
            loss.backward()
            debugprint(f"[RANK {closure_rank}] GEM _project_gem_qp closure: loss.backward() 完成") # Debug print after backward in closure
            return loss

        # Perform the optimization
        debugprint(f"[RANK {rank}] GEM _project_gem_qp: 开始执行 optimizer.step(closure)") # Debug print before optimizer step
        optimizer.step(closure)
        debugprint(f"[RANK {rank}] GEM _project_gem_qp: optimizer.step(closure) 完成") # Debug print after optimizer step

        # Return the optimized projected gradient, detached from computation graph
        debugprint(f"[RANK {rank}] GEM _project_gem_qp: 方法出口, 返回优化后的投影梯度, 形状: {g_proj.detach().shape}") # Debug print at method exit
        return g_proj.detach()
