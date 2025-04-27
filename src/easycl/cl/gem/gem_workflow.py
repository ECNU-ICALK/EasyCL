import os
import copy
from typing import TYPE_CHECKING, Optional, List, Dict, Any
import random
import torch

def debugprint(*args, **kwargs):
    pass

from llamafactory.data.data_utils import merge_dataset
from llamafactory.data import (
    SFTDataCollatorWith4DAttentionMask,
    get_dataset,
    get_template_and_fix_tokenizer
)
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.logging import get_logger
from llamafactory.extras.misc import calculate_tps, get_logits_processor
from llamafactory.extras.ploting import plot_loss
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.trainer_utils import create_modelcard_and_push
from llamafactory.train.sft.metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .gem_trainer import GEMSeq2SeqTrainer
from easycl.cl.distributed_utils import (
    is_distributed, is_main_process, get_rank, get_world_size,
    get_deepspeed_zero_stage, broadcast_object
)

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from llamafactory.hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments
    from easycl.hparams.cl_finetuning_args import CLFinetuningArguments

logger = get_logger(__name__)


def run_sft_gem(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    cl_finetuning_args: "CLFinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    debugprint("GEM run_sft_gem: 函数入口") # Debug print at function start
    # Load tokenizer and model
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module.get("processor", None) # Get processor, default to None if not found
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # Log replay settings
    if cl_finetuning_args.use_gem:
        debugprint(f"GEM run_sft_gem: GEM 模式已启用. 检查 cl_finetuning_args:") # Debug print GEM enabled
        debugprint(f"  use_gem: {cl_finetuning_args.use_gem}")
        debugprint(f"  gem_memory_strength: {cl_finetuning_args.gem_memory_strength}")
        debugprint(f"  replay_ratio: {cl_finetuning_args.replay_ratio}")
        debugprint(f"  maxsamples_list: {cl_finetuning_args.maxsamples_list}")
        debugprint(f"  replay_task_list: {cl_finetuning_args.replay_task_list}")
        debugprint(f"  previous_task_dataset: {cl_finetuning_args.previous_task_dataset}")

        logger.info("\n" + "*" * 80)
        logger.info("*" + " " * 78 + "*")
        logger.info("*" + " " * 28 + "GEM MODE ENABLED" + " " * 28 + "*")
        logger.info("*" + " " * 78 + "*")
        logger.info("*" + f" GEM Memory Strength: {cl_finetuning_args.gem_memory_strength}" + " " * (77 - len(f" GEM Memory Strength: {cl_finetuning_args.gem_memory_strength}")) + "*")
        logger.info("*" + f" Memory Ratio/Max Samples: {cl_finetuning_args.replay_ratio if cl_finetuning_args.maxsamples_list is None else 'List Specified'}" + " " * (77 - len(f" Memory Ratio/Max Samples: {cl_finetuning_args.replay_ratio if cl_finetuning_args.maxsamples_list is None else 'List Specified'}")) + "*")
        logger.info("*" + " " * 78 + "*")
        logger.info("*" * 80 + "\n")

        merged_datasets = []
        gem_datasets_info = []

        # 1. First load current task dataset
        current_dataset_module = get_dataset(
            template=template,
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            stage="sft",
            **tokenizer_module
        )
        debugprint(f"GEM run_sft_gem: 加载当前任务数据集完成, 数据集模块键: {list(current_dataset_module.keys())}") # Debug print after loading current dataset

        if "train_dataset" in current_dataset_module:
            current_dataset = current_dataset_module["train_dataset"].map(lambda example: {"is_memory": False})
            merged_datasets.append(current_dataset)
            logger.info(f"Loaded current dataset with {len(current_dataset)} samples (marked as is_memory=False)")
            debugprint(f"GEM run_sft_gem: 当前任务训练集大小: {len(current_dataset)}, 已添加到 merged_datasets") # Debug print for current train dataset size

            # Record current dataset info
            current_dataset_info = {
                "name": "current_task",
                "dataset": data_args.dataset,
                "size": len(current_dataset),
                "is_memory": False
            }
            gem_datasets_info.append(current_dataset_info)

        # 2. If replay_task_list is specified, load memory task data
        if cl_finetuning_args.replay_task_list and training_args.do_train:
            # Save current data directory and dataset configuration
            original_dataset_dir = copy.deepcopy(data_args.dataset_dir)
            original_dataset = copy.deepcopy(data_args.dataset)

            debugprint(f"GEM run_sft_gem: 开始加载记忆任务数据集, 任务列表: {cl_finetuning_args.replay_task_list}") # Debug print before memory loading loop
            # Parse task list for replay (as memory)
            memory_task_list = [task.strip() for task in cl_finetuning_args.replay_task_list.split(',')]

            # Parse max samples list (if provided)
            maxsamples_list = None
            if cl_finetuning_args.maxsamples_list:
                try:
                    maxsamples_list = [int(x.strip()) for x in cl_finetuning_args.maxsamples_list.split(',')]
                    if len(maxsamples_list) != len(memory_task_list):
                        logger.warning(f"Length of maxsamples_list ({len(maxsamples_list)}) doesn't match memory_task_list ({len(memory_task_list)}). Will use replay_ratio instead.")
                        maxsamples_list = None
                except ValueError:
                    logger.warning(f"Invalid format in maxsamples_list: {cl_finetuning_args.maxsamples_list}. Will use replay_ratio instead.")
                    maxsamples_list = None

            # If previous_task_dataset is provided, use it as data directory
            if cl_finetuning_args.previous_task_dataset:
                data_args.dataset_dir = cl_finetuning_args.previous_task_dataset
                logger.info(f"Using custom dataset directory for memory: {data_args.dataset_dir}")

            debugprint(f"GEM run_sft_gem: 最终使用的记忆任务列表: {memory_task_list}") # Debug print final memory task list
            logger.info(f"Memory task list: {memory_task_list}")
            if maxsamples_list:
                logger.info(f"Max samples per memory task: {maxsamples_list}")
                debugprint(f"GEM run_sft_gem: 将为每个记忆任务使用指定的最大样本数: {maxsamples_list}") # Debug print using max samples list
            else:
                logger.info(f"Using memory ratio: {cl_finetuning_args.replay_ratio}")
                debugprint(f"GEM run_sft_gem: 将使用记忆比率: {cl_finetuning_args.replay_ratio}") # Debug print using ratio

            for task_idx, task_name in enumerate(memory_task_list):
                # Set current memory task
                data_args.dataset = [task_name]

                debugprint(f"GEM run_sft_gem: 开始加载记忆任务 {task_idx+1}/{len(memory_task_list)}: {task_name}") # Debug print inside memory loading loop
                logger.info(f"Loading memory task {task_idx+1}/{len(memory_task_list)}: {task_name}")

                try:
                    # Load memory task dataset
                    memory_dataset_module = get_dataset(
                        template=template,
                        model_args=model_args,
                        data_args=data_args,
                        training_args=training_args,
                        stage="sft",
                        **tokenizer_module
                    )

                    if "train_dataset" in memory_dataset_module:
                        # Determine sample count
                        total_samples = len(memory_dataset_module["train_dataset"])

                        # Determine max samples based on maxsamples_list or replay_ratio
                        if maxsamples_list and task_idx < len(maxsamples_list):
                            max_samples = min(maxsamples_list[task_idx], total_samples)
                            logger.info(f"Using max samples from list: {max_samples}")
                            debugprint(f"GEM run_sft_gem: 任务 {task_name} - 使用 max samples from list: {max_samples}") # Debug print using max samples from list
                        else:
                            max_samples = int(total_samples * cl_finetuning_args.replay_ratio)
                            logger.info(f"Using ratio-based max samples: {max_samples}")
                            debugprint(f"GEM run_sft_gem: 任务 {task_name} - 使用比例计算最大样本数: {total_samples} * {cl_finetuning_args.replay_ratio} = {int(total_samples * cl_finetuning_args.replay_ratio)}") # Debug print ratio calculation

                        if max_samples < total_samples:
                            # 在分布式环境中，确保所有进程选择相同的样本
                            if is_distributed():
                                # 设置相同的随机种子以确保一致性
                                random_seed = training_args.seed + task_idx
                                random.seed(random_seed)
                                torch.manual_seed(random_seed)
                                debugprint(f"GEM run_sft_gem: 分布式环境中设置随机种子 {random_seed} 以确保一致的样本选择")

                                # 主进程选择样本并广播给其他进程
                                if is_main_process():
                                    indices = random.sample(range(total_samples), max_samples)
                                    debugprint(f"GEM run_sft_gem: 主进程选择了 {len(indices)} 个样本索引")
                                else:
                                    indices = None
                                    debugprint(f"GEM run_sft_gem: 非主进程等待接收样本索引")

                                # 广播索引到所有进程
                                indices = broadcast_object(indices)
                                debugprint(f"GEM run_sft_gem: 进程 {get_rank()} 接收到 {len(indices)} 个样本索引")
                            else:
                                # 非分布式环境，直接选择样本
                                indices = random.sample(range(total_samples), max_samples)

                            memory_dataset_raw = memory_dataset_module["train_dataset"].select(indices)
                            logger.info(f"Selected {max_samples}/{total_samples} samples from task {task_name}")
                            debugprint(f"GEM run_sft_gem: 任务 {task_name} - 从 {total_samples} 中随机选择了 {max_samples} 个样本") # Debug print sample selection
                        else:
                            memory_dataset_raw = memory_dataset_module["train_dataset"]
                            logger.info(f"Using all {total_samples} samples from task {task_name}")
                            debugprint(f"GEM run_sft_gem: 任务 {task_name} - 使用全部 {total_samples} 个样本") # Debug print using all samples

                        # Add 'is_memory': True flag to memory data
                        memory_dataset = memory_dataset_raw.map(lambda example: {"is_memory": True})
                        merged_datasets.append(memory_dataset)
                        logger.info(f"Marked {len(memory_dataset)} samples from {task_name} as is_memory=True")
                        debugprint(f"GEM run_sft_gem: 任务 {task_name} - 添加了 {len(memory_dataset)} 个记忆样本到 merged_datasets") # Debug print memory added to merge list

                        # Record memory dataset info
                        memory_dataset_info = {
                            "name": task_name,
                            "size_original": total_samples,
                            "size_selected": len(memory_dataset),
                            "is_memory": True
                        }
                        gem_datasets_info.append(memory_dataset_info)
                    else:
                        logger.warning(f"No training data found for memory task: {task_name}")

                except Exception as e:
                    debugprint(f"GEM run_sft_gem: 加载记忆任务 {task_name} 时出错: {str(e)}") # Debug print memory loading error
                    logger.error(f"Failed to load memory task {task_name}: {str(e)}")
                    continue

            # Restore original dataset configuration
            data_args.dataset_dir = original_dataset_dir
            data_args.dataset = original_dataset
            debugprint(f"GEM run_sft_gem: 记忆任务加载循环结束. 恢复原始数据集设置: {original_dataset}") # Debug print after memory loading loop

        # 3. Merge all datasets (Current + Memory)
        if len(merged_datasets) > 0: # Should always be at least 1 if current task data exists
            logger.info(f"Merging {len(merged_datasets)} datasets (current + memory) with strategy: concat")
            debugprint(f"GEM run_sft_gem: 开始合并 {len(merged_datasets)} 个数据集 (当前任务 + 记忆任务)") # Debug print before merge
            merged_data_args = copy.deepcopy(data_args)
            merged_data_args.mix_strategy = "concat"

            # 在分布式环境中，确保所有进程使用相同的随机种子
            if is_distributed():
                debugprint(f"GEM run_sft_gem: 分布式环境中设置数据集合并随机种子 {training_args.seed}")
                # 设置随机种子以确保一致性
                random.seed(training_args.seed)
                torch.manual_seed(training_args.seed)

            dataset_module = {}
            dataset_module["train_dataset"] = merge_dataset(
                merged_datasets,
                merged_data_args,
                seed=training_args.seed
            )
            debugprint(f"GEM run_sft_gem: 数据集合并完成, 合并后训练集大小: {len(dataset_module['train_dataset'])}") # Debug print after merge

            # Summarize merged dataset information
            total_samples = len(dataset_module["train_dataset"])
            logger.info("\n" + "#" * 80)
            logger.info("#" + " " * 78 + "#")
            logger.info("#" + " " * 22 + "GEM DATASET MERGE SUMMARY" + " " * 22 + "#")
            logger.info("#" + " " * 78 + "#")
            logger.info("#" + f" Total merged samples: {total_samples}" + " " * (77 - len(f" Total merged samples: {total_samples}")) + "#")

            for ds_info in gem_datasets_info:
                type_label = "Memory" if ds_info["is_memory"] else "Current"
                if ds_info["is_memory"]:
                    ds_status = f" {type_label} ({ds_info['name']}): {ds_info['size_selected']}/{ds_info['size_original']} samples"
                else:
                    ds_status = f" {type_label} ({ds_info['name']}): {ds_info['size']} samples"
                logger.info("#" + ds_status + " " * (77 - len(ds_status)) + "#")

            logger.info("#" + " " * 78 + "#")
            logger.info("#" * 80 + "\n")

            # Keep eval_dataset from current task
            if "eval_dataset" in current_dataset_module:
                dataset_module["eval_dataset"] = current_dataset_module["eval_dataset"]
        else:
            # Handle case where only current data exists (no memory tasks specified)
            logger.info("No memory tasks specified or loaded. Using only current task data.")
            dataset_module = current_dataset_module
            # Ensure 'is_memory' column exists even if only current data
            debugprint("GEM run_sft_gem: 无记忆任务, 仅使用当前任务数据") # Debug print no memory tasks
            if "train_dataset" in dataset_module:
                debugprint("GEM run_sft_gem: 为仅有的当前任务数据添加 is_memory=False 列") # Debug print adding is_memory for current only
                dataset_module["train_dataset"] = dataset_module["train_dataset"].map(lambda example: {"is_memory": False})

    else:
        debugprint("GEM run_sft_gem: GEM 模式未启用, 加载常规数据集") # Debug print GEM disabled branch
        # Load regular dataset when GEM is not enabled
        dataset_module = get_dataset(
            template=template,
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            stage="sft",
            **tokenizer_module
        )
        # Ensure 'is_memory' column exists and is False for standard SFT
        debugprint(f"GEM run_sft_gem: 常规数据集加载完成, 数据集模块键: {list(dataset_module.keys())}") # Debug print after loading regular dataset
        if "train_dataset" in dataset_module and training_args.do_train:
             logger.info("GEM not enabled, marking all training data as is_memory=False for compatibility.")
             debugprint("GEM run_sft_gem: GEM 未启用, 正在为训练数据添加 is_memory=False 列") # Debug print adding is_memory for non-GEM
             # Check if column already exists to avoid errors
             if "is_memory" not in dataset_module["train_dataset"].column_names:
                 dataset_module["train_dataset"] = dataset_module["train_dataset"].map(lambda example: {"is_memory": False})
             else: # If it somehow exists, ensure it's boolean False
                 logger.warning("'is_memory' column already exists in non-GEM mode. Ensuring values are False.")
                 dataset_module["train_dataset"] = dataset_module["train_dataset"].map(lambda example: {"is_memory": False})


    # Now load model (after dataset preparation since we might need to adjust it)
    model = load_model(
        tokenizer=tokenizer,
        model_args=model_args,
        finetuning_args=finetuning_args,
        is_trainable=training_args.do_train
    )
    debugprint(f"GEM run_sft_gem: 模型加载完成. 模型类型: {type(model).__name__}") # Debug print after model load

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    # Initialize data collator
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        tokenizer=tokenizer,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    debugprint("GEM run_sft_gem: 数据整理器 (Data Collator) 初始化完成") # Debug print after collator init

    # Set up training arguments
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams

    # Explicitly disable removing unused columns to keep 'is_memory' for GEM compute_loss
    if training_args.remove_unused_columns:
        logger.warning("Setting `remove_unused_columns=False` in training_args because the 'is_memory' column is required by GEM compute_loss.")
        # info_rank0("Setting `remove_unused_columns=False` in training_args because the 'is_memory' column is required by GEM compute_loss.") # Use info_rank0 if preferred
        debugprint("GEM run_sft_gem: Setting training_args.remove_unused_columns = False")
        training_args.remove_unused_columns = False

    # Set up metrics
    metric_kwargs = {}
    if training_args.predict_with_generate:
        metric_kwargs["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_kwargs["compute_metrics"] = ComputeAccuracy()
        metric_kwargs["preprocess_logits_for_metrics"] = eval_logit_processor

    # Set up generation arguments if needed
    if training_args.predict_with_generate or training_args.do_eval:
        gen_kwargs = generating_args.to_dict()
        gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
        gen_kwargs["logits_processor"] = get_logits_processor()
    else:
        gen_kwargs = None

    # Initialize the GEM trainer
    trainer = GEMSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        cl_finetuning_args=cl_finetuning_args,
        tokenizer=tokenizer,
        processor=processor, # Pass the processor here
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **metric_kwargs
    )
    debugprint("GEM run_sft_gem: GEMSeq2SeqTrainer 初始化完成") # Debug print after trainer init

    # Training
    if training_args.do_train:
        logger.info("*** Training ***")
        debugprint("GEM run_sft_gem: 开始训练") # Debug print before training
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        debugprint(f"GEM run_sft_gem: 训练完成. 训练结果指标: {train_result.metrics}") # Debug print after training
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

        # Calculate effective tokens per second if requested
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_second"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

    # Evaluation
    if training_args.do_eval and "eval_dataset" in dataset_module:
        if training_args.predict_with_generate:
            # Set left padding for generation
            tokenizer.padding_side = "left"

            debugprint("GEM run_sft_gem: 开始评估 (predict_with_generate=True)") # Debug print before eval (generate)
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            eval_dataset=dataset_module["eval_dataset"],
            metric_key_prefix="eval",
            **gen_kwargs
        )
        debugprint(f"GEM run_sft_gem: 评估完成. 评估指标: {metrics}") # Debug print after eval
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Model card and push to hub if requested
    if training_args.push_to_hub:
        debugprint("GEM run_sft_gem: 开始创建模型卡并推送到 Hub") # Debug print before push_to_hub
        create_modelcard_and_push(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            finetuning_args=finetuning_args,
            model_name=model_args.model_name_or_path.split("/")[-1],
        )

    # Plot training loss if enabled - 只在主进程执行
    if is_main_process() and finetuning_args.plot_loss:
        logger.info("Plotting training loss...")
        debugprint("GEM run_sft_gem: 开始绘制损失图") # Debug print before plotting loss
        plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    debugprint("GEM run_sft_gem: 函数结束") # Debug print at function end
    return trainer, tokenizer
