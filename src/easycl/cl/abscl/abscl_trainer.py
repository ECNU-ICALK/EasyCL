import os
import torch
import json
from typing import TYPE_CHECKING, Any, Dict, Optional
import safetensors.torch
from safetensors.torch import load_file as safe_load_file
import re
def debugprint(*args, **kwargs):
    pass
import torch.distributed as dist

from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from llamafactory.extras import logging
from easycl.cl.olora.olora import OLoRA
from peft import PeftModel
from easycl.hparams import CLFinetuningArguments
from easycl.cl.distributed_utils import (
    is_distributed, is_main_process, get_rank, get_world_size,
    get_deepspeed_zero_stage, is_deepspeed_zero3_enabled,
    gather_parameters, synchronize_gradients, all_reduce_tensor, broadcast_object
)

if TYPE_CHECKING:
    from transformers import ProcessorMixin

logger = logging.get_logger(__name__)

class ABSCLTrainer(CustomSeq2SeqTrainer):
    """
    ABSCL Trainer - Uses O-LoRA method to compute orthogonal constraints and L2 regularization
    """

    def __init__(
        self,
        finetuning_args,
        cl_finetuning_args,
        processor: Optional["ProcessorMixin"] = None,
        gen_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            finetuning_args=finetuning_args,
            processor=processor,
            gen_kwargs=gen_kwargs,
            **kwargs
        )
        debugprint(f"进入ABSCLTrainer初始化")
        logger.info_rank0(f"Entering ABSCLTrainer __init__") # 添加 logger
        logger.info_rank0(f"Incoming cl_finetuning_args: {cl_finetuning_args}") # 添加 logger
        debugprint(f"传入的CL微调参数: {cl_finetuning_args}")

        # Store cl_finetuning_args for later use
        self.cl_finetuning_args = cl_finetuning_args

        # Set loss weights
        self.orthogonal_lambda = cl_finetuning_args.abscl_orthogonal_lambda
        self.l2_lambda = cl_finetuning_args.abscl_shared_l2_lambda
        debugprint(f"正交约束权重设置为: {self.orthogonal_lambda}")
        logger.info_rank0(f"orthogonal_lambda set to: {self.orthogonal_lambda}") # 添加 logger
        debugprint(f"L2正则化权重设置为: {self.l2_lambda}")
        logger.info_rank0(f"l2_lambda set to: {self.l2_lambda}") # 添加 logger

        # Set task ID and paths
        self.task_id = cl_finetuning_args.current_task_id or "task"
        adapters_path = cl_finetuning_args.adapters_save_path or os.path.join(
            os.path.dirname(self.args.output_dir)
        )
        self.shared_adapter_path = os.path.join(adapters_path, "shared_adapter")
        debugprint(f"任务ID设置为: {self.task_id}")
        logger.info_rank0(f"task_id set to: {self.task_id}") # 添加 logger
        debugprint(f"适配器路径设置为: {adapters_path}")
        logger.info_rank0(f"adapters_path set to: {adapters_path}") # 添加 logger
        debugprint(f"共享适配器路径设置为: {self.shared_adapter_path}")
        logger.info_rank0(f"shared_adapter_path set to: {self.shared_adapter_path}") # 添加 logger

        # Use device string instead of device object
        device = self.args.device.type if hasattr(self.args.device, "type") else self.args.device

        # Create OLoRA instance for loss computation
        self.olora = OLoRA(
            model=self.model,
            orthogonal_lambda=self.orthogonal_lambda,
            l2_lambda=self.l2_lambda,
            olora_history_path=adapters_path,  # Use adapters_path as history path
            model_output_dir=self.args.output_dir,
            device=device,
            prev_task_id="shared_adapter"  # Set prev_task_id to shared_adapter
        )

        # Load shared adapter
        self._load_shared_adapter()

        debugprint(f"已配置ABSCL训练器(O-LoRA方法)")
        logger.info_rank0(f"Configured ABSCL trainer (O-LoRA method)") # 添加 logger
        debugprint(f"- 共享适配器路径: {self.shared_adapter_path}")
        logger.info_rank0(f"- Shared adapter path: {self.shared_adapter_path}") # 添加 logger
        debugprint(f"- 正交约束权重: {self.orthogonal_lambda}")
        logger.info_rank0(f"- Orthogonal constraint weight: {self.orthogonal_lambda}") # 添加 logger
        debugprint(f"- L2正则化权重: {self.l2_lambda}")
        logger.info_rank0(f"- L2 regularization weight: {self.l2_lambda}") # 添加 logger

    def _load_shared_adapter(self):
        """Load shared adapter as reference adapter for orthogonal constraint"""
        logger.info_rank0(f"Entering _load_shared_adapter, trying to load shared adapter: {self.shared_adapter_path}")
        debugprint(f"尝试加载共享适配器: {self.shared_adapter_path}")

        # 检查分布式环境
        is_dist = is_distributed()
        if is_dist:
            rank = get_rank()
            world_size = get_world_size()
            logger.info_rank0(f"Running in distributed environment with {world_size} processes")
            debugprint(f"[rank {rank}/{world_size-1}] 在分布式环境中运行，进程数: {world_size}")
        else:
            debugprint(f"在非分布式环境中运行")

        try:
            # 在分布式环境中，只需要在主进程中检查文件是否存在
            if is_main_process():
                # Check if shared adapter exists
                if not os.path.exists(self.shared_adapter_path):
                    logger.warning_rank0(f"Shared adapter path does not exist: {self.shared_adapter_path}")
                    debugprint(f"共享适配器路径不存在: {self.shared_adapter_path}")
                    exists_flag = False
                else:
                    # Verify shared adapter configuration files
                    config_path = os.path.join(self.shared_adapter_path, "adapter_config.json")
                    model_path = os.path.join(self.shared_adapter_path, "adapter_model.safetensors")
                    debugprint(f"检查共享适配器文件:")
                    debugprint(f"- 配置文件路径: {config_path}")
                    debugprint(f"- 权重文件路径: {model_path}")

                    if not os.path.exists(config_path):
                        logger.warning_rank0(f"Shared adapter config file does not exist: {config_path}")
                        debugprint(f"共享适配器配置文件不存在: {config_path}")
                        exists_flag = False
                    elif not os.path.exists(model_path):
                        logger.warning_rank0(f"Shared adapter weights file does not exist: {model_path}")
                        debugprint(f"共享适配器权重文件不存在: {model_path}")
                        exists_flag = False
                    else:
                        debugprint(f"共享适配器文件验证成功 - 配置文件和权重文件都存在")
                        exists_flag = True
            else:
                # 非主进程初始化为None，等待广播
                exists_flag = None
                if is_dist:
                    debugprint(f"[rank {rank}/{world_size-1}] 非主进程，等待广播 exists_flag")


            # 检查DeepSpeed ZeRO阶段
            zero_stage = get_deepspeed_zero_stage(self.model)
            logger.info_rank0(f"Detected DeepSpeed ZeRO Stage: {zero_stage}")
            debugprint(f"检测到DeepSpeed ZeRO阶段: {zero_stage}")

            # Set up adapters using same method as O-LoRA
            debugprint("使用O-LoRA方法设置适配器")
            # Load shared adapter as orthogonal reference
            debugprint("准备加载共享适配器作为正交参考")

            if is_dist:
                rank = get_rank()
                world_size = get_world_size()
                debugprint(f"[rank {rank}/{world_size-1}] 调用 olora.load_prev_adapter 前")

                # 打印当前模型的PEFT配置
                if hasattr(self.model, "peft_config"):
                    debugprint(f"[rank {rank}/{world_size-1}] 当前模型PEFT配置: {list(self.model.peft_config.keys())}")
                else:
                    debugprint(f"[rank {rank}/{world_size-1}] 当前模型没有PEFT配置")
            else:
                # 打印当前模型的PEFT配置
                if hasattr(self.model, "peft_config"):
                    debugprint(f"当前模型PEFT配置: {list(self.model.peft_config.keys())}")
                else:
                    debugprint(f"当前模型没有PEFT配置")

            # 注意：olora.load_prev_adapter内部已经处理了分布式同步，这里不需要额外的barrier
            debugprint(f"调用 olora.load_prev_adapter('shared_adapter') 开始")
            result = self.olora.load_prev_adapter("shared_adapter")
            debugprint(f"调用 olora.load_prev_adapter('shared_adapter') 结束，结果: {result}")

            # 确保所有进程获得相同的结果，但不需要额外的barrier
            if is_dist:
                rank = get_rank()
                world_size = get_world_size()
                debugprint(f"[rank {rank}/{world_size-1}] _load_shared_adapter: 共享适配器加载结果: {result}")
                # 广播加载结果，确保所有进程获得相同的结果
                # 注意：这里只广播布尔值，不会导致死锁
                debugprint(f"[rank {rank}/{world_size-1}] 广播加载结果前")
                result = broadcast_object(result, src=0)
                debugprint(f"[rank {rank}/{world_size-1}] 广播加载结果后: {result}")

                # 打印加载后模型的PEFT配置
                if hasattr(self.model, "peft_config"):
                    debugprint(f"[rank {rank}/{world_size-1}] 加载后模型PEFT配置: {list(self.model.peft_config.keys())}")
                else:
                    debugprint(f"[rank {rank}/{world_size-1}] 加载后模型没有PEFT配置")
            else:
                # 打印加载后模型的PEFT配置
                if hasattr(self.model, "peft_config"):
                    debugprint(f"加载后模型PEFT配置: {list(self.model.peft_config.keys())}")
                else:
                    debugprint(f"加载后模型没有PEFT配置")

            if result:
                logger.info_rank0(f"Successfully loaded shared adapter as orthogonal reference")
                debugprint("成功加载共享适配器作为正交参考")
                # 检查是否有merged_historical_weights
                if hasattr(self.olora, "merged_historical_weights") and self.olora.merged_historical_weights is not None:
                    debugprint(f"成功加载历史权重，包含 {len(self.olora.merged_historical_weights)} 个权重项")
                else:
                    debugprint("警告：虽然返回成功，但没有加载历史权重")
            else:
                logger.warning_rank0(f"Failed to load shared adapter")
                debugprint("加载共享适配器失败")

            return result
        except Exception as e:
            logger.info_rank0(f"Error loading shared adapter: {str(e)}")
            debugprint(f"加载共享适配器时出错: {str(e)}")
            return False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss using O-LoRA style loss computation"""
        logger.debug(f"Entering compute_loss, global_step: {self.state.global_step}")
        debugprint(f"计算损失函数，全局步骤: {self.state.global_step}")

        # 检测DeepSpeed ZeRO阶段
        zero_stage = get_deepspeed_zero_stage(model)
        is_zero3 = zero_stage == 3
        is_zero2 = zero_stage == 2
        logger.debug(f"Detected DeepSpeed ZeRO Stage: {zero_stage}")
        debugprint(f"检测到DeepSpeed ZeRO阶段: {zero_stage}")

        # 初始化辅助损失变量
        orthogonal_loss = torch.tensor(0.0, device=self.args.device)
        l2_loss = torch.tensor(0.0, device=self.args.device)

        # 创建模块到规范化路径的映射（用于ZeRO-3环境下的Forward Hooks）
        module_to_path_map = {}
        for name, mod in model.named_modules():
            # 移除可能的'module.'前缀
            clean_name = name.replace('module.', '')
            module_to_path_map[mod] = clean_name

        # 在ZeRO-3或ZeRO-2环境下使用Forward Hooks计算辅助损失
        if is_zero3 or is_zero2:
            logger.debug(f"Using Forward Hooks for auxiliary loss computation in ZeRO-{zero_stage}")
            debugprint(f"在ZeRO-{zero_stage}环境中使用Forward Hooks计算辅助损失")

            # 创建钩子列表
            hooks = []
            matched_weights_count = 0  # 添加计数器，记录成功匹配的历史权重数量

            # 注册正交损失的Forward Hook
            def orthogonal_hook_fn(module, inputs, outputs):
                nonlocal orthogonal_loss, matched_weights_count
                if hasattr(module, "lora_A") and hasattr(module.lora_A, "keys"):
                    # 检查是否存在default adapter
                    if 'default' in module.lora_A:
                        new_weight = module.lora_A['default'].weight

                        # 使用预先构建的映射获取规范化的模块路径
                        module_path = module_to_path_map.get(module)

                        if module_path and hasattr(self.olora, "merged_historical_weights"):
                            merged_a_key = f"{module_path}.merged_A"

                            if merged_a_key in self.olora.merged_historical_weights:
                                old_weight = self.olora.merged_historical_weights[merged_a_key].to(new_weight.device)

                                if new_weight.shape[1] == old_weight.shape[1]:
                                    dot_product = torch.mm(new_weight, old_weight.T)
                                    curr_loss = torch.abs(dot_product).sum()
                                    orthogonal_loss += curr_loss
                                    matched_weights_count += 1
                                else:
                                    logger.debug(f"Weight dimension mismatch - new: {new_weight.shape}, old: {old_weight.shape}")

            # 注册L2损失的Forward Hook
            def l2_hook_fn(module, inputs, outputs):
                nonlocal l2_loss
                if hasattr(module, "lora_A") and hasattr(module.lora_A, "keys"):
                    # 检查是否存在default adapter
                    if 'default' in module.lora_A:
                        # 计算A矩阵的L2损失
                        a_weight = module.lora_A['default'].weight
                        if a_weight.numel() > 0:
                            a_norm_sq = torch.sum(a_weight ** 2)
                            l2_loss += a_norm_sq

                        # 计算B矩阵的L2损失
                        if hasattr(module, "lora_B") and 'default' in module.lora_B:
                            b_weight = module.lora_B['default'].weight
                            if b_weight.numel() > 0:
                                b_norm_sq = torch.sum(b_weight ** 2)
                                l2_loss += b_norm_sq

            # 注册钩子到所有LoRA模块
            for name, module in model.named_modules():
                if hasattr(module, "lora_A") and hasattr(module.lora_A, "keys"):
                    # 注册正交损失钩子（如果有历史权重）
                    if hasattr(self.olora, "merged_historical_weights") and self.olora.merged_historical_weights is not None:
                        h1 = module.register_forward_hook(orthogonal_hook_fn)
                        hooks.append(h1)

                    # 注册L2损失钩子
                    h2 = module.register_forward_hook(l2_hook_fn)
                    hooks.append(h2)

            # 确保所有进程在执行前向传播前同步
            if is_distributed():
                rank = get_rank()
                world_size = get_world_size()
                print(f"[DEBUG] compute_loss: [rank {rank}/{world_size-1}] 即将进入barrier (前向传播前)")
                debugprint(f"[rank {rank}/{world_size-1}] compute_loss: 即将进入barrier (前向传播前)")

                # 添加每个进程的状态打印
                for i in range(world_size):
                    if rank == i:
                        print(f"[DEBUG] compute_loss: [rank {rank}/{world_size-1}] 正在等待进入前向传播前barrier")

                dist.barrier()

                print(f"[DEBUG] compute_loss: [rank {rank}/{world_size-1}] 已通过barrier，准备执行前向传播")
                debugprint(f"[rank {rank}/{world_size-1}] compute_loss: 所有进程已同步，准备执行前向传播")

            # 执行前向传播，触发钩子
            outputs = model(**inputs)
            base_loss = outputs.loss

            # 再次同步，确保所有进程都完成了前向传播
            if is_distributed():
                rank = get_rank()
                world_size = get_world_size()
                print(f"[DEBUG] compute_loss: [rank {rank}/{world_size-1}] 即将进入barrier (前向传播后)")
                debugprint(f"[rank {rank}/{world_size-1}] compute_loss: 即将进入barrier (前向传播后)")

                # 添加每个进程的状态打印
                for i in range(world_size):
                    if rank == i:
                        print(f"[DEBUG] compute_loss: [rank {rank}/{world_size-1}] 正在等待进入前向传播后barrier")

                dist.barrier()

                print(f"[DEBUG] compute_loss: [rank {rank}/{world_size-1}] 已通过barrier，前向传播已完成")
                debugprint(f"[rank {rank}/{world_size-1}] compute_loss: 所有进程已完成前向传播")

            # 应用损失权重
            orthogonal_loss = orthogonal_loss * self.orthogonal_lambda
            l2_loss = l2_loss * self.l2_lambda
            debugprint(f"应用损失权重后 - 正交损失: {orthogonal_loss.item():.4f}, L2损失: {l2_loss.item():.4f}")

            # 移除所有钩子
            for h in hooks:
                h.remove()
            debugprint(f"已移除所有Forward Hooks")

            logger.debug(f"ZeRO-{zero_stage} hook-based orthogonal_loss: {orthogonal_loss.item():.4f}, matched weights: {matched_weights_count}")
            logger.debug(f"ZeRO-{zero_stage} hook-based l2_loss: {l2_loss.item():.4f}")
            debugprint(f"ZeRO-{zero_stage}基于钩子的正交损失: {orthogonal_loss.item():.4f}, 匹配权重数: {matched_weights_count}")
            debugprint(f"ZeRO-{zero_stage}基于钩子的L2损失: {l2_loss.item():.4f}")
        else:
            # 非ZeRO-2/3环境，使用原来的方式计算损失
            debugprint("在标准环境中计算损失")
            outputs = model(**inputs)
            base_loss = outputs.loss

            # 计算正交损失
            debugprint("计算正交损失")
            orthogonal_loss = self.olora.compute_orthogonal_loss()
            logger.debug(f"Standard orthogonal_loss: {orthogonal_loss.item():.4f}")
            debugprint(f"标准正交损失: {orthogonal_loss.item():.4f}")

            # 计算L2损失
            debugprint("计算L2损失")
            l2_loss = self.olora.compute_l2_loss()
            logger.debug(f"Standard l2_loss: {l2_loss.item():.4f}")
            debugprint(f"标准L2损失: {l2_loss.item():.4f}")

        # 在ZeRO-2环境下同步梯度
        if is_zero2:
            debugprint("在ZeRO-2环境中同步梯度")
            synchronize_gradients(model)

        # 合并损失
        total_loss = base_loss + orthogonal_loss + l2_loss
        logger.debug(f"Total loss: {total_loss.item():.4f}")
        debugprint(f"总损失: {total_loss.item():.4f}, 基础损失: {base_loss.item():.4f}, 正交损失: {orthogonal_loss.item():.4f}, L2损失: {l2_loss.item():.4f}")

        # 保存当前损失值用于日志记录
        self._current_orthogonal_loss = orthogonal_loss.item()
        self._current_l2_loss = l2_loss.item()
        debugprint(f"已缓存当前损失值用于日志记录")

        # 每100步记录一次损失值
        if self.state.global_step % 100 == 0:
            logger.info_rank0(f"Step {self.state.global_step} losses - Base: {base_loss.item():.4f}, Orthogonal: {orthogonal_loss.item():.4f}, L2: {l2_loss.item():.4f}")
            debugprint(f"步骤 {self.state.global_step} 损失 - 基础: {base_loss.item():.4f}, 正交: {orthogonal_loss.item():.4f}, L2: {l2_loss.item():.4f}")

        # 更新输出指标
        if return_outputs:
            outputs.metrics = outputs.get("metrics", {})
            outputs.metrics.update({
                "orthogonal_loss": orthogonal_loss.item(),
                "l2_loss": l2_loss.item(),
            })

        return (total_loss, outputs) if return_outputs else total_loss

    def get_extra_losses(self, model=None):
        """Get extra loss values for logging"""
        if hasattr(self, '_current_orthogonal_loss') and hasattr(self, '_current_l2_loss'):
            # Use cached loss values
            return {
                "orthogonal_loss": self._current_orthogonal_loss,
                "shared_l2_loss": self._current_l2_loss,  # For compatibility with original code
            }
        elif model is not None:
            # 检测DeepSpeed ZeRO阶段
            zero_stage = get_deepspeed_zero_stage(model)
            is_zero3 = zero_stage == 3
            is_zero2 = zero_stage == 2

            if is_zero3 or is_zero2:
                # 在ZeRO-3或ZeRO-2环境下，使用与compute_loss相同的Forward Hooks方法计算损失
                logger.debug(f"Using Forward Hooks for extra loss computation in ZeRO-{zero_stage}")

                # 初始化辅助损失变量
                orthogonal_loss = torch.tensor(0.0, device=self.args.device)
                l2_loss = torch.tensor(0.0, device=self.args.device)

                # 创建模块到规范化路径的映射
                module_to_path_map = {}
                for name, mod in model.named_modules():
                    clean_name = name.replace('module.', '')
                    module_to_path_map[mod] = clean_name

                # 创建钩子列表
                hooks = []
                matched_weights_count = 0

                # 注册正交损失的Forward Hook
                def orthogonal_hook_fn(module, inputs, outputs):
                    nonlocal orthogonal_loss, matched_weights_count
                    if hasattr(module, "lora_A") and hasattr(module.lora_A, "keys"):
                        if 'default' in module.lora_A:
                            new_weight = module.lora_A['default'].weight
                            module_path = module_to_path_map.get(module)

                            if module_path and hasattr(self.olora, "merged_historical_weights"):
                                merged_a_key = f"{module_path}.merged_A"

                                if merged_a_key in self.olora.merged_historical_weights:
                                    old_weight = self.olora.merged_historical_weights[merged_a_key].to(new_weight.device)

                                    if new_weight.shape[1] == old_weight.shape[1]:
                                        dot_product = torch.mm(new_weight, old_weight.T)
                                        curr_loss = torch.abs(dot_product).sum()
                                        orthogonal_loss += curr_loss
                                        matched_weights_count += 1

                # 注册L2损失的Forward Hook
                def l2_hook_fn(module, inputs, outputs):
                    nonlocal l2_loss
                    if hasattr(module, "lora_A") and hasattr(module.lora_A, "keys"):
                        if 'default' in module.lora_A:
                            a_weight = module.lora_A['default'].weight
                            if a_weight.numel() > 0:
                                a_norm_sq = torch.sum(a_weight ** 2)
                                l2_loss += a_norm_sq

                            if hasattr(module, "lora_B") and 'default' in module.lora_B:
                                b_weight = module.lora_B['default'].weight
                                if b_weight.numel() > 0:
                                    b_norm_sq = torch.sum(b_weight ** 2)
                                    l2_loss += b_norm_sq

                # 注册钩子到所有LoRA模块
                for name, module in model.named_modules():
                    if hasattr(module, "lora_A") and hasattr(module.lora_A, "keys"):
                        if hasattr(self.olora, "merged_historical_weights") and self.olora.merged_historical_weights is not None:
                            h1 = module.register_forward_hook(orthogonal_hook_fn)
                            hooks.append(h1)

                        h2 = module.register_forward_hook(l2_hook_fn)
                        hooks.append(h2)

                # 创建一个简单的输入进行dummy forward pass
                dummy_input = {"input_ids": torch.ones((1, 1), dtype=torch.long, device=self.args.device)}

                # 确保所有进程使用相同的输入进行前向传播
                if is_distributed():
                    # 添加同步点，确保所有进程都准备好进行dummy forward
                    rank = get_rank()
                    world_size = get_world_size()
                    print(f"[DEBUG] get_extra_losses: [rank {rank}/{world_size-1}] 即将进入barrier (dummy forward前)")
                    debugprint(f"[rank {rank}/{world_size-1}] get_extra_losses: 即将进入barrier (dummy forward前)")

                    # 添加每个进程的状态打印
                    for i in range(world_size):
                        if rank == i:
                            print(f"[DEBUG] get_extra_losses: [rank {rank}/{world_size-1}] 正在等待进入dummy forward前barrier")

                    dist.barrier()

                    print(f"[DEBUG] get_extra_losses: [rank {rank}/{world_size-1}] 已通过barrier，准备执行dummy forward")
                    debugprint(f"[rank {rank}/{world_size-1}] get_extra_losses: 所有进程已同步，准备执行dummy forward")

                # 执行前向传播，触发钩子
                with torch.no_grad():
                    model(**dummy_input)

                # 再次同步，确保所有进程都完成了dummy forward
                if is_distributed():
                    rank = get_rank()
                    world_size = get_world_size()
                    print(f"[DEBUG] get_extra_losses: [rank {rank}/{world_size-1}] 即将进入barrier (dummy forward后)")
                    debugprint(f"[rank {rank}/{world_size-1}] get_extra_losses: 即将进入barrier (dummy forward后)")

                    # 添加每个进程的状态打印
                    for i in range(world_size):
                        if rank == i:
                            print(f"[DEBUG] get_extra_losses: [rank {rank}/{world_size-1}] 正在等待进入dummy forward后barrier")

                    dist.barrier()

                    print(f"[DEBUG] get_extra_losses: [rank {rank}/{world_size-1}] 已通过barrier，dummy forward已完成")
                    debugprint(f"[rank {rank}/{world_size-1}] get_extra_losses: 所有进程已完成dummy forward")

                # 应用损失权重
                orthogonal_loss = orthogonal_loss * self.orthogonal_lambda
                l2_loss = l2_loss * self.l2_lambda

                # 移除所有钩子
                for h in hooks:
                    h.remove()

                return {
                    "orthogonal_loss": orthogonal_loss.item(),
                    "shared_l2_loss": l2_loss.item(),  # For compatibility with original code
                }
            else:
                # 非ZeRO-2/3环境，使用原来的方式计算损失
                orthogonal_loss = self.olora.compute_orthogonal_loss()
                l2_loss = self.olora.compute_l2_loss()
                return {
                    "orthogonal_loss": orthogonal_loss.item(),
                    "shared_l2_loss": l2_loss.item(),  # For compatibility with original code
                }
        else:
            # No cache and no model, return zero values
            return {
                "orthogonal_loss": 0.0,
                "shared_l2_loss": 0.0,
            }
