import os
import copy
import torch
import gc
import random
from typing import TYPE_CHECKING, Optional
def debugprint(*args, **kwargs):
    pass
from accelerate.state import AcceleratorState, PartialState
from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from llamafactory.data.data_utils import merge_dataset
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.logging import get_logger
from llamafactory.extras.misc import calculate_tps, get_logits_processor
from llamafactory.extras.ploting import plot_loss
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.trainer_utils import create_modelcard_and_push
from llamafactory.train.sft.metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from .abscl_trainer import ABSCLTrainer
from .abscl import extract_feature_statistics  # Import feature extraction function
from easycl.cl.distributed_utils import (
    is_distributed, is_main_process, get_rank, get_world_size,
    get_deepspeed_zero_stage, is_deepspeed_zero3_enabled,
    gather_parameters, all_reduce_tensor, broadcast_object
)


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from llamafactory.hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments
    from easycl.hparams import CLFinetuningArguments


logger = get_logger(__name__)


def run_sft_abscl(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    cl_finetuning_args: "CLFinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    """
    Run sequence-to-sequence fine-tuning using the ABSCL method.
    This method trains two adapters: a shared adapter and a task-specific adapter.
    """
    debugprint("进入ABSCL微调运行函数")
    debugprint(f"传入的CL微调参数: {cl_finetuning_args}")

    # Load tokenizer and dataset (only need once)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # ============= Prepare training dataset for shared adapter using replay strategy =============
    if training_args.do_train:
        debugprint("\n" + "*" * 80)
        debugprint("*" + " " * 78 + "*")
        debugprint("*" + " " * 25 + "使用回放策略准备共享适配器数据集" + " " * 25 + "*")
        debugprint("*" + " " * 78 + "*")
        debugprint("*" + f" 回放比例: {cl_finetuning_args.replay_ratio}" + " " * (77 - len(f" 回放比例: {cl_finetuning_args.replay_ratio}")) + "*")
        debugprint("*" + " " * 78 + "*")
        debugprint("*" * 80 + "\n")
        logger.info_rank0("Preparing shared adapter dataset using replay strategy") # 添加 logger
        logger.info_rank0(f"Replay Ratio: {cl_finetuning_args.replay_ratio}") # 添加 logger

        merged_datasets = []
        replay_datasets_info = []

        # Load current task dataset
        current_dataset_module = get_dataset(
            template=template,
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            stage="sft",
            **tokenizer_module
        )
        if "train_dataset" in current_dataset_module:
            total_current_samples = len(current_dataset_module["train_dataset"])

            # Apply sampling to the current dataset, regardless of whether it's the first task
            max_current_samples = int(total_current_samples * cl_finetuning_args.replay_ratio)
            debugprint(f"当前任务数据集采样: 总样本数 {total_current_samples}, 回放比例 {cl_finetuning_args.replay_ratio}, 采样数量 {max_current_samples}")
            logger.info_rank0(f"Current task dataset sampling: Total samples {total_current_samples}, Replay ratio {cl_finetuning_args.replay_ratio}, Samples {max_current_samples}") # 添加 logger
            if max_current_samples < total_current_samples:
                current_indices = random.sample(range(total_current_samples), max_current_samples)
                current_dataset = current_dataset_module["train_dataset"].select(current_indices)
                debugprint(f"从当前任务中选择了 {max_current_samples}/{total_current_samples} 个样本")
                logger.info_rank0(f"Selected {max_current_samples}/{total_current_samples} samples from the current task") # 添加 logger
            else:
                current_dataset = current_dataset_module["train_dataset"]
                debugprint(f"使用当前任务的所有 {total_current_samples} 个样本")
                logger.info_rank0(f"Using all {total_current_samples} samples from the current task") # 添加 logger

            merged_datasets.append(current_dataset)
            replay_datasets_info.append({
                "name": "current_task",
                "dataset": data_args.dataset,
                "size_original": total_current_samples,
                "size_selected": len(current_dataset)
            })

        # Load replay task datasets
        if cl_finetuning_args.replay_task_list:
            original_dataset_dir = copy.deepcopy(data_args.dataset_dir)
            original_dataset = copy.deepcopy(data_args.dataset)
            replay_task_list = [task.strip() for task in cl_finetuning_args.replay_task_list.split(',')]
            debugprint(f"要回放的任务: {replay_task_list}")
            logger.info_rank0(f"Tasks to replay: {replay_task_list}") # 添加 logger
            maxsamples_list = (
                [int(x.strip()) for x in cl_finetuning_args.maxsamples_list.split(',')]
                if cl_finetuning_args.maxsamples_list else None
            )
            debugprint(f"回放任务的最大样本数列表: {maxsamples_list}")
            logger.info_rank0(f"Max samples list for replay tasks: {maxsamples_list}") # 添加 logger
            if cl_finetuning_args.previous_task_dataset:
                data_args.dataset_dir = cl_finetuning_args.previous_task_dataset
                debugprint(f"使用先前任务数据集目录进行回放: {data_args.dataset_dir}")
                logger.info_rank0(f"Using previous task dataset directory for replay: {data_args.dataset_dir}") # 添加 logger

            for task_idx, task_name in enumerate(replay_task_list):
                data_args.dataset = [task_name]
                debugprint(f"加载回放任务: {task_name}")
                logger.info_rank0(f"Loading replay task: {task_name}") # 添加 logger
                try:
                    replay_dataset_module = get_dataset(
                        template=template,
                        model_args=model_args,
                        data_args=data_args,
                        training_args=training_args,
                        stage="sft",
                        **tokenizer_module
                    )
                    if "train_dataset" in replay_dataset_module:
                        total_samples = len(replay_dataset_module["train_dataset"])
                        max_samples = (
                            min(maxsamples_list[task_idx], total_samples)
                            if maxsamples_list and task_idx < len(maxsamples_list)
                            else int(total_samples * cl_finetuning_args.replay_ratio)
                        )
                        debugprint(f"回放任务 {task_name}: 总样本数 {total_samples}, 回放比例 {cl_finetuning_args.replay_ratio}, 最大样本数 {max_samples}")
                        logger.info_rank0(f"Replay task {task_name}: Total samples {total_samples}, Replay ratio {cl_finetuning_args.replay_ratio}, Max samples {max_samples}") # 添加 logger
                        indices = random.sample(range(total_samples), max_samples)
                        replay_dataset = replay_dataset_module["train_dataset"].select(indices)
                        merged_datasets.append(replay_dataset)
                        replay_datasets_info.append({
                            "name": task_name,
                            "size_original": total_samples,
                            "size_selected": len(replay_dataset)
                        })
                        debugprint(f"任务 {task_name}: 为回放选择了 {len(replay_dataset)} 个样本")
                        logger.info_rank0(f"Task {task_name}: Selected {len(replay_dataset)} samples for replay") # 添加 logger
                except Exception as e:
                    logger.info_rank0(f"Failed to load replay task {task_name}: {str(e)}") # error -> info_rank0
                    debugprint(f"加载回放任务 {task_name} 失败: {str(e)}")
                    continue

            data_args.dataset_dir = original_dataset_dir
            data_args.dataset = original_dataset

        # Merge datasets
        if len(merged_datasets) > 0:
            dataset_module = {
                "train_dataset": merge_dataset(
                    merged_datasets,
                    data_args,
                    seed=training_args.seed
                )
            }
            if "eval_dataset" in current_dataset_module:
                dataset_module["eval_dataset"] = current_dataset_module["eval_dataset"]
        else:
            dataset_module = current_dataset_module
    else:
        dataset_module = get_dataset(
            template=template,
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            stage="sft",
            **tokenizer_module
        )

    # Save original output directory
    original_output_dir = training_args.output_dir

    # Ensure adapters_save_path exists
    adapters_save_path = cl_finetuning_args.adapters_save_path
    if not adapters_save_path:
        # If not specified, use output_dir as the default path
        adapters_save_path = os.path.join(original_output_dir, "adapters")

    # 在分布式环境中，只在主进程创建目录
    if is_main_process():
        os.makedirs(adapters_save_path, exist_ok=True)
        logger.info_rank0(f"Adapters will be saved to: {adapters_save_path}")

    # 在分布式环境中等待主进程创建目录
    if is_distributed():
        torch.distributed.barrier()

    # =========================== Train shared adapter ===========================
    adapter_name = "shared_adapter"
    adapter_output_dir = os.path.join(adapters_save_path, adapter_name)

    # Set output directory for shared adapter
    training_args = copy.deepcopy(training_args)  # Create a copy to avoid modifying the original object
    training_args.output_dir = adapter_output_dir
    training_args.overwrite_output_dir = True  # Overwrite existing adapter

    # 在分布式环境中，只在主进程创建目录
    if is_main_process():
        os.makedirs(training_args.output_dir, exist_ok=True)
        logger.info_rank0(f"Training shared adapter: {adapter_name}, Output directory: {training_args.output_dir}")

    # 在分布式环境中等待主进程创建目录
    if is_distributed():
        torch.distributed.barrier()

    # Create model argument copies for shared adapter
    model_args_copy = copy.deepcopy(model_args)
    finetuning_args_copy = copy.deepcopy(finetuning_args)
    cl_finetuning_args_copy = copy.deepcopy(cl_finetuning_args)

    # 在分布式环境中，只在主进程检查文件是否存在
    if is_main_process():
        # Check if a pre-trained shared adapter exists (check for adapter_config.json)
        shared_adapter_config_path = os.path.join(adapter_output_dir, "adapter_config.json")
        if os.path.exists(shared_adapter_config_path):
            # If a pre-trained shared adapter exists, load it
            adapter_exists = True
            logger.info_rank0(f"Loading pre-trained shared adapter: {adapter_output_dir}")
            debugprint(f"发现预训练的共享适配器配置文件: {shared_adapter_config_path}")
            debugprint(f"将加载已有的共享适配器: {adapter_output_dir}")
        else:
            # Otherwise, initialize the shared adapter randomly
            adapter_exists = False
            logger.info_rank0(f"Pre-trained shared adapter not found, initializing a new one")
            debugprint(f"未找到预训练的共享适配器配置文件: {shared_adapter_config_path}")
            debugprint(f"将初始化一个新的共享适配器")
    else:
        # 非主进程初始化为None，等待广播
        adapter_exists = None

    # 在分布式环境中广播检查结果
    if is_distributed():
        rank = get_rank()
        world_size = get_world_size()
        debugprint(f"[rank {rank}/{world_size-1}] 广播共享适配器存在状态前: adapter_exists = {adapter_exists}")
        adapter_exists = broadcast_object(adapter_exists)
        debugprint(f"[rank {rank}/{world_size-1}] 广播共享适配器存在状态后: adapter_exists = {adapter_exists}")
    else:
        debugprint(f"非分布式环境: adapter_exists = {adapter_exists}")

    # 根据检查结果设置adapter_name_or_path
    if adapter_exists:
        model_args_copy.adapter_name_or_path = [adapter_output_dir]
        debugprint(f"共享适配器存在，设置 model_args_copy.adapter_name_or_path = {model_args_copy.adapter_name_or_path}")
    else:
        model_args_copy.adapter_name_or_path = None
        debugprint(f"共享适配器不存在，设置 model_args_copy.adapter_name_or_path = None")

    # Load model (with shared adapter, if it exists)
    debugprint(f"开始加载模型，adapter_name_or_path = {model_args_copy.adapter_name_or_path}")
    model = load_model(tokenizer, model_args_copy, finetuning_args_copy, training_args.do_train)
    debugprint(f"模型加载完成，是否加载了共享适配器: {adapter_exists}")

    # 检查模型的PEFT配置
    if hasattr(model, "peft_config"):
        debugprint(f"模型PEFT配置: {list(model.peft_config.keys())}")
        for adapter_name, config in model.peft_config.items():
            debugprint(f"适配器 '{adapter_name}' 配置: {config}")
    else:
        debugprint("模型没有PEFT配置")

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args_copy.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args_copy.compute_dtype,
        **tokenizer_module,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False  # important for multimodal dataset

    # Metric utils
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args_copy.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Create callback copies for shared adapter
    current_callbacks = copy.deepcopy(callbacks) if callbacks is not None else []

    # Initialize Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args_copy,
        data_collator=data_collator,
        callbacks=current_callbacks,
        gen_kwargs=gen_kwargs,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    # Train shared adapter
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()  # Save to adapter_output_dir

        if finetuning_args_copy.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        if replay_datasets_info:
            for idx, ds_info in enumerate(replay_datasets_info):
                prefix = f"shared_adapter_dataset_{idx}"
                for key, value in ds_info.items():
                    # Ensure all values are basic types (string, number, etc.) that can be formatted
                    if isinstance(value, (str, int, float, bool)):
                        train_result.metrics[f"{prefix}_{key}"] = value
                    elif isinstance(value, list):
                        # Convert list to string
                        train_result.metrics[f"{prefix}_{key}"] = str(value)
                    else:
                        train_result.metrics[f"{prefix}_{key}"] = str(value)

            if cl_finetuning_args.replay_task_list:
                # Ensure it is a string, not a list
                if isinstance(replay_task_list, list):
                    task_list_str = ','.join(replay_task_list)
                else:
                    task_list_str = str(replay_task_list)
                train_result.metrics["shared_adapter_replay_task_list"] = task_list_str

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if is_main_process() and finetuning_args_copy.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])

    # Clean up memory for task-specific adapter training
    del model
    del trainer
    AcceleratorState._reset_state()   # 清掉全局 state
    PartialState._reset_state()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================== Train task-specific adapter ===========================
    if not cl_finetuning_args.current_task_id:
        logger.warning_rank0("No current_task_id specified, using 'task' as default task ID")
        task_id = "task"
        debugprint("未指定current_task_id，使用'task'作为默认任务ID")
    else:
        task_id = cl_finetuning_args.current_task_id
        debugprint(f"当前任务ID: {task_id}")
        logger.info_rank0(f"Current task ID: {task_id}") # 添加 logger

    # Reload the original, unsampled dataset for the task-specific adapter
    debugprint("\n" + "*" * 80)
    debugprint("*" + " " * 78 + "*")
    debugprint("*" + " " * 20 + "为任务适配器加载原始数据集" + " " * 20 + "*")
    debugprint("*" + " " * 78 + "*")
    debugprint("*" * 80 + "\n")
    logger.info_rank0("Loading original dataset for task adapter") # 添加 logger

    dataset_module = get_dataset(
        template=template,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        stage="sft",
        **tokenizer_module
    )

    if "train_dataset" in dataset_module:
        debugprint(f"使用完整原始数据集，包含 {len(dataset_module['train_dataset'])} 个样本进行任务适配器训练")
        logger.info_rank0(f"Using full original dataset with {len(dataset_module['train_dataset'])} samples for task adapter training") # 添加 logger

    adapter_name = task_id
    adapter_output_dir = os.path.join(adapters_save_path, adapter_name)

    # Set output directory for task-specific adapter
    training_args = copy.deepcopy(training_args)  # Recreate a copy
    training_args.output_dir = adapter_output_dir

    # 在分布式环境中，只在主进程创建目录
    if is_main_process():
        os.makedirs(training_args.output_dir, exist_ok=True)
        logger.info_rank0(f"Training task-specific adapter: {adapter_name}, Output directory: {training_args.output_dir}")

    # 在分布式环境中等待主进程创建目录
    if is_distributed():
        torch.distributed.barrier()

    # Create model argument copies for task-specific adapter
    model_args_copy = copy.deepcopy(model_args)
    finetuning_args_copy = copy.deepcopy(finetuning_args)

    # Ensure the shared adapter path exists
    shared_adapter_dir = os.path.join(adapters_save_path, "shared_adapter")
    logger.info_rank0(f"Checking shared adapter directory: {shared_adapter_dir}")

    # 在分布式环境中，只在主进程检查文件是否存在
    if is_main_process():
        if not os.path.exists(shared_adapter_dir) or not os.path.exists(os.path.join(shared_adapter_dir, "adapter_config.json")):
            shared_adapter_exists = False
            logger.warning_rank0(f"Warning: Shared adapter not found at {shared_adapter_dir}, will proceed but orthogonal loss might not be calculated")
        else:
            shared_adapter_exists = True
    else:
        # 非主进程初始化为None，等待广播
        shared_adapter_exists = None

    # 在分布式环境中广播检查结果
    if is_distributed():
        shared_adapter_exists = broadcast_object(shared_adapter_exists)

    # Set adapters_save_path in finetuning_args_copy
    finetuning_args_copy.adapters_save_path = adapters_save_path
    # debugprint(f"Set adapters_save_path for task adapter: {finetuning_args_copy.adapters_save_path}") # 使用英文
    logger.info_rank0(f"Set adapters_save_path for task adapter: {finetuning_args_copy.adapters_save_path}") # 添加 logger

    # Load only the base model, do not preload any adapters
    model_args_copy.adapter_name_or_path = None

    debugprint(f"加载基础模型并初始化新的任务特定适配器: {task_id}")
    logger.info_rank0(f"Loading base model and initializing new task-specific adapter: {task_id}") # 添加 logger

    # Load model (randomly initialize task-specific adapter)
    model = load_model(tokenizer, model_args_copy, finetuning_args_copy, training_args.do_train)

    # Check and log adapter configuration after creating the model
    debugprint("检查模型加载后的PEFT配置:")
    logger.info_rank0("Checking PEFT config after model loading:") # 添加 logger
    if hasattr(model, "peft_config"):
        for adapter_name_cfg, config in model.peft_config.items(): # Rename adapter_name to adapter_name_cfg to avoid conflict
            debugprint(f"适配器 '{adapter_name_cfg}' 配置: {config}")
            logger.info_rank0(f"Adapter '{adapter_name_cfg}' config: {config}") # 添加 logger
            if hasattr(config, "target_modules"):
                debugprint(f"适配器 '{adapter_name_cfg}' 目标模块: {config.target_modules}")
                logger.info_rank0(f"Adapter '{adapter_name_cfg}' target modules: {config.target_modules}") # 添加 logger

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args_copy.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args_copy.compute_dtype,
        **tokenizer_module,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False  # important for multimodal dataset

    # Metric utils
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args_copy.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Create callback copies for task-specific adapter
    current_callbacks = copy.deepcopy(callbacks) if callbacks is not None else []

    # Initialize Trainer
    trainer = ABSCLTrainer(
        model=model,
        args=training_args,
        cl_finetuning_args=cl_finetuning_args, # Use original cl_finetuning_args
        finetuning_args=finetuning_args_copy,
        data_collator=data_collator,
        callbacks=current_callbacks,
        gen_kwargs=gen_kwargs,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    # Train task-specific adapter
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()  # Save to adapter_output_dir

        if finetuning_args_copy.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        # Add logging for ABSCL extra losses
        extra_losses = trainer.get_extra_losses(model)
        train_result.metrics.update(extra_losses)

        # Log ABSCL-related parameters
        train_result.metrics["orthogonal_lambda"] = cl_finetuning_args.abscl_orthogonal_lambda
        train_result.metrics["shared_l2_lambda"] = cl_finetuning_args.abscl_shared_l2_lambda

        # Log used adapter information
        train_result.metrics["shared_adapter"] = os.path.join(adapters_save_path, "shared_adapter")
        train_result.metrics["task_adapter"] = os.path.join(adapters_save_path, adapter_name)

        # Log to console
        # debugprint(f"ABSCL Orthogonal Loss: {extra_losses['orthogonal_loss']}") # 使用英文
        # debugprint(f"ABSCL Shared L2 Loss: {extra_losses['shared_l2_loss']}") # 使用英文
        # debugprint(f"ABSCL Orthogonal Lambda: {cl_finetuning_args.abscl_orthogonal_lambda}") # 使用英文
        # debugprint(f"ABSCL Shared L2 Lambda: {cl_finetuning_args.abscl_shared_l2_lambda}") # 使用英文
        # debugprint(f"ABSCL Shared Adapter: {os.path.join(adapters_save_path, 'shared_adapter')}") # 使用英文
        # debugprint(f"ABSCL Task Adapter: {os.path.join(adapters_save_path, adapter_name)}") # 使用英文
        logger.info_rank0(f"ABSCL Orthogonal Loss: {extra_losses['orthogonal_loss']}") # 添加 logger
        logger.info_rank0(f"ABSCL Shared L2 Loss: {extra_losses['shared_l2_loss']}") # 添加 logger
        logger.info_rank0(f"ABSCL Orthogonal Lambda: {cl_finetuning_args.abscl_orthogonal_lambda}") # 添加 logger
        logger.info_rank0(f"ABSCL Shared L2 Lambda: {cl_finetuning_args.abscl_shared_l2_lambda}") # 添加 logger
        logger.info_rank0(f"ABSCL Shared Adapter: {os.path.join(adapters_save_path, 'shared_adapter')}") # 添加 logger
        logger.info_rank0(f"ABSCL Task Adapter: {os.path.join(adapters_save_path, adapter_name)}") # 添加 logger

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if is_main_process() and finetuning_args_copy.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy", "orthogonal_loss", "shared_l2_loss"])

        # ====================== Add feature extraction and statistics process ======================
        debugprint("\n" + "*" * 80)
        debugprint("*" + " " * 78 + "*")
        debugprint("*" + " " * 25 + "特征提取和统计处理" + " " * 24 + "*")
        debugprint("*" + " " * 78 + "*")
        debugprint("*" * 80 + "\n")
        logger.info_rank0("Feature extraction and statistics processing") # 添加 logger

        # Get statistics save path
        stats_path = cl_finetuning_args.abscl_stats_path # Use original cl_finetuning_args
        if not stats_path:
            stats_path = os.path.join(adapters_save_path, "abscl_stats")
            logger.info_rank0(f"abscl_stats_path not specified, using default path: {stats_path}")
        else:
            logger.info_rank0(f"Specified abscl_stats_path: {stats_path}")

        # 在分布式环境中，只在主进程创建目录
        if is_main_process():
            os.makedirs(stats_path, exist_ok=True)
            logger.info_rank0(f"Feature statistics will be saved to: {stats_path}")

        # 在分布式环境中等待主进程创建目录
        if is_distributed():
            torch.distributed.barrier()

        # Extract task-specific feature statistics
        try:
            extract_feature_statistics(
                model=model,
                trainer=trainer,
                task_id=task_id,
                finetuning_args=finetuning_args_copy,
                cl_finetuning_args=cl_finetuning_args,
                device=training_args.device,
                dataset=dataset_module.get("train_dataset")
            )
            debugprint(f"特征统计处理完成，结果保存在 {stats_path}")
            logger.info_rank0(f"Feature statistics processing complete, results saved in {stats_path}") # 添加 logger
        except Exception as e:
            logger.info_rank0(f"Error during feature extraction process: {str(e)}")
            debugprint(f"特征提取过程中出错: {str(e)}")

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluate task-specific adapter
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Generate predictions using task-specific adapter
    if training_args.do_predict:
        logger.warning_rank0("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    # Create model card
    create_modelcard_and_push(trainer, model_args_copy, data_args, training_args, finetuning_args_copy)

    # Clean up memory
    del model
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Log completion message
    debugprint(f"ABSCL训练完成。适配器已保存在 {adapters_save_path}")
    debugprint(f"  - 共享适配器: {os.path.join(adapters_save_path, 'shared_adapter')}")
    debugprint(f"  - 任务特定适配器: {os.path.join(adapters_save_path, task_id)}")
    logger.info_rank0(f"ABSCL training complete. Adapters saved in {adapters_save_path}") # 添加 logger
    logger.info_rank0(f"  - Shared adapter: {os.path.join(adapters_save_path, 'shared_adapter')}") # 添加 logger
    logger.info_rank0(f"  - Task-specific adapter: {os.path.join(adapters_save_path, task_id)}") # 添加 logger

    return
