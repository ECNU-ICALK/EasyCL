import os
import copy
import torch
import gc
import random
import math
from typing import TYPE_CHECKING, Optional

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
from .dynamic_conpet_trainer import DynamicConPetTrainer
from .dynamic_conpet import DatasetClassifier, save_classifier, load_classifier
from accelerate.state import AcceleratorState, PartialState
from .dynamic_conpet import is_distributed, get_rank, is_main_process
def debugprint(*args, **kwargs):
    pass


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from llamafactory.hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments
    from easycl.hparams import CLFinetuningArguments


logger = get_logger(__name__)

# Helper function to add index to dataset examples
def add_indices_to_example(example, idx):
    example['index'] = idx
    return example

def run_sft_dynamic_conpet(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    cl_finetuning_args: "CLFinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    """
    Run sequence-to-sequence fine-tuning with Dynamic ConPet method
    This method will train two adapters: a shared adapter and a task-specific adapter
    """
    debugprint("进入 run_sft_dynamic_conpet 工作流")
    debugprint(f"传入的 model_args: {model_args}")
    debugprint(f"传入的 data_args: {data_args}")
    debugprint(f"传入的 training_args: {training_args}")
    debugprint(f"传入的 finetuning_args: {finetuning_args}")
    debugprint(f"传入的 cl_finetuning_args: {cl_finetuning_args}")
    debugprint(f"传入的 generating_args: {generating_args}")

    # Load tokenizer and dataset (only once)
    debugprint("加载 tokenizer 和模板")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module.get("processor", None)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # ============= Prepare current and historical datasets (1:1 ratio) =============
    if training_args.do_train:
        debugprint("\n" + "*" * 80)
        debugprint("*" + " " * 78 + "*")
        debugprint("*" + " " * 11 + "加载当前和历史数据集 (大约 1:1 比例)" + " " * 11 + "*")
        debugprint("*" + " " * 78 + "*")
        debugprint("*" * 80 + "\n")

        merged_datasets = []
        datasets_info = []

        # Load current task dataset
        debugprint("加载当前任务数据集")
        debugprint(f"当前任务数据集参数: dataset_dir={data_args.dataset_dir}, dataset={data_args.dataset}")
        current_dataset_module = get_dataset(
            template=template,
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            stage="sft",
            **tokenizer_module
        )

        current_dataset = None
        current_dataset_size = 0
        if "train_dataset" in current_dataset_module:
            # --- Add index to the current dataset ---
            current_dataset_raw = current_dataset_module["train_dataset"]
            current_dataset = current_dataset_raw.map(add_indices_to_example, with_indices=True, desc="Adding indices to current dataset")
            debugprint(f"已为当前数据集添加 'index' 列")
            # --- End of index addition ---

            current_dataset_size = len(current_dataset)
            debugprint(f"已加载当前数据集，包含 {current_dataset_size} 个样本")

            # Determine the name for the current task for classifier
            current_task_name_for_classifier = cl_finetuning_args.current_task_id
            if not current_task_name_for_classifier:
                current_task_name_for_classifier = "current_task" # Default if not provided
                debugprint(f"警告: cl_finetuning_args.current_task_id 未设置，分类器将使用默认名称 'current_task'")
            else:
                debugprint(f"分类器将使用当前任务名称: '{current_task_name_for_classifier}' (来自 cl_finetuning_args.current_task_id)")

            # Add to merged datasets
            merged_datasets.append(current_dataset) # Add the dataset with index
            datasets_info.append({
                "name": current_task_name_for_classifier, # Use determined name
                "dataset": data_args.dataset,
                "size": current_dataset_size
            })
            debugprint(f"添加当前任务数据集信息: {datasets_info[-1]}")
        else:
            debugprint("警告: 当前数据集模块中未找到 'train_dataset'")

        # Load historical task datasets
        historical_datasets = []
        if cl_finetuning_args.replay_task_list and current_dataset_size > 0:
            debugprint("开始加载历史任务数据集 (用于回放)")
            debugprint(f"回放任务列表: {cl_finetuning_args.replay_task_list}")
            original_dataset_dir = copy.deepcopy(data_args.dataset_dir)
            original_dataset = copy.deepcopy(data_args.dataset)
            replay_task_list = [task.strip() for task in cl_finetuning_args.replay_task_list.split(',')]
            debugprint(f"解析后的回放任务列表: {replay_task_list}")

            # Calculate number of samples to load for each historical task
            samples_per_task = math.ceil(current_dataset_size / len(replay_task_list))
            debugprint(f"每个历史任务需要加载的样本数 (目标): {samples_per_task}")

            maxsamples_list = None
            if cl_finetuning_args.maxsamples_list:
                maxsamples_list = [int(x.strip()) for x in cl_finetuning_args.maxsamples_list.split(',')]
                debugprint(f"使用提供的最大样本数列表: {maxsamples_list}")
            else:
                debugprint("未使用最大样本数列表，将根据比例计算")

            if cl_finetuning_args.previous_task_dataset:
                data_args.dataset_dir = cl_finetuning_args.previous_task_dataset
                debugprint(f"设置历史数据集目录: {data_args.dataset_dir}")
            else:
                debugprint("警告: 未提供 previous_task_dataset，可能无法加载历史数据")

            for task_idx, task_name in enumerate(replay_task_list):
                data_args.dataset = [task_name]
                debugprint(f"尝试加载历史任务: {task_name} (索引 {task_idx})")
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
                        # --- Add index to historical dataset ---
                        replay_dataset_raw = replay_dataset_module["train_dataset"]
                        replay_dataset_with_indices = replay_dataset_raw.map(
                            add_indices_to_example,
                            with_indices=True,
                            desc=f"Adding indices to historical dataset {task_name}"
                        )
                        debugprint(f"已为历史任务 {task_name} 添加 'index' 列")
                        # --- End of index addition ---

                        total_samples = len(replay_dataset_with_indices)
                        debugprint(f"任务 '{task_name}' 的总样本数 (带索引): {total_samples}")

                        # Determine number of samples to load
                        max_samples = (
                            min(maxsamples_list[task_idx], total_samples)
                            if maxsamples_list and task_idx < len(maxsamples_list)
                            else min(samples_per_task, total_samples)
                        )
                        debugprint(f"任务 '{task_name}' 确定加载的样本数: {max_samples}")

                        if max_samples < total_samples:
                            indices = random.sample(range(total_samples), max_samples)
                            replay_dataset_selected = replay_dataset_with_indices.select(indices)
                            debugprint(f"从历史任务 {task_name} 中选择了 {max_samples}/{total_samples} 个样本 (带索引)")
                        else:
                            replay_dataset_selected = replay_dataset_with_indices
                            debugprint(f"使用历史任务 {task_name} 的全部 {total_samples} 个样本 (带索引)")

                        historical_datasets.append(replay_dataset_selected) # Add the dataset with index
                        datasets_info.append({
                            "name": task_name,
                            "size": len(replay_dataset_selected)
                        })
                        debugprint(f"添加历史任务数据集信息: {datasets_info[-1]}")
                    else:
                         debugprint(f"警告: 历史任务 '{task_name}' 的模块中未找到 'train_dataset'")
                except Exception as e:
                    debugprint(f"错误: 加载历史任务 {task_name} 失败: {str(e)}")
                    continue

            data_args.dataset_dir = original_dataset_dir
            data_args.dataset = original_dataset
            debugprint("恢复原始的 data_args.dataset_dir 和 data_args.dataset")

            # Log total historical samples loaded
            if historical_datasets:
                total_historical_samples = sum(len(ds) for ds in historical_datasets)
                debugprint(f"总共加载的历史样本数 (带索引): {total_historical_samples}")

                # Add historical datasets to merge list
                merged_datasets.extend(historical_datasets)
        elif current_dataset_size <= 0:
             debugprint("当前任务数据集大小为 0，跳过历史数据集加载")
        else:
            debugprint("未提供回放任务列表 (replay_task_list)，不加载历史数据集")

        # Merge all datasets
        if len(merged_datasets) > 1:
            debugprint(f"准备合并 {len(merged_datasets)} 个数据集 (包含 'index' 列)")
            # NOTE: merge_dataset should preserve the 'index' column if present in all datasets
            train_dataset_merged = merge_dataset(
                merged_datasets,
                data_args,
                seed=training_args.seed
            )
            # Verify 'index' column exists after merge
            if "index" not in train_dataset_merged.column_names:
                debugprint("警告: 合并后的数据集中未找到 'index' 列! 这可能会导致后续错误。")
            else:
                debugprint("确认合并后的数据集中存在 'index' 列")

            dataset_module = {
                "train_dataset": train_dataset_merged
            }
            debugprint(f"最终合并的数据集大小: {len(dataset_module['train_dataset'])} 个样本")

            # Add evaluation dataset (from current task) - Evaluation dataset typically doesn't need index for loss calculation
            if "eval_dataset" in current_dataset_module:
                # Optionally add index to eval_dataset if needed elsewhere, but not strictly necessary for trainer
                # eval_dataset_with_indices = current_dataset_module["eval_dataset"].map(add_indices_to_example, with_indices=True, desc="Adding indices to eval dataset")
                # dataset_module["eval_dataset"] = eval_dataset_with_indices
                dataset_module["eval_dataset"] = current_dataset_module["eval_dataset"]
                debugprint(f"添加了来自当前任务的评估数据集，大小: {len(dataset_module['eval_dataset'])} 个样本")
            else:
                debugprint("当前任务模块中没有评估数据集")

        elif len(merged_datasets) == 1:
            dataset_module = {
                "train_dataset": current_dataset, # Use the current dataset with index
                 "eval_dataset": current_dataset_module.get("eval_dataset") # Keep original eval dataset
            }
            debugprint("仅使用当前数据集 (已添加 'index' 列)")
        else:
            debugprint("警告: 没有加载任何训练数据")
            dataset_module = current_dataset_module # Fallback, likely contains no train_dataset

    else:
        # Non-training mode: Load eval/predict dataset
        # Index is usually not needed here unless prediction saving requires original index
        debugprint("非训练模式 (do_train=False)，加载评估/预测数据集")
        dataset_module = get_dataset(
            template=template,
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            stage="sft",
            **tokenizer_module
        )
        # Optionally add index if needed for prediction analysis
        # if "test_dataset" in dataset_module:
        #     dataset_module["test_dataset"] = dataset_module["test_dataset"].map(add_indices_to_example, with_indices=True, desc="Adding indices to test dataset")
        # if "eval_dataset" in dataset_module:
        #     dataset_module["eval_dataset"] = dataset_module["eval_dataset"].map(add_indices_to_example, with_indices=True, desc="Adding indices to eval dataset")

        datasets_info = [] # No datasets_info needed if not training

    # Save original output directory
    original_output_dir = training_args.output_dir
    debugprint(f"原始输出目录: {original_output_dir}")

    # Ensure adapters_save_path exists
    rank = get_rank()
    adapters_save_path = cl_finetuning_args.adapters_save_path
    if not adapters_save_path:
        # If not specified, use output_dir as default path
        adapters_save_path = os.path.join(original_output_dir, "adapters")
        debugprint(f"进程 rank={rank} adapters_save_path 未指定，使用默认路径: {adapters_save_path}")

    # Only create directories in the main process
    if is_main_process():
        os.makedirs(adapters_save_path, exist_ok=True)
        debugprint(f"进程 rank=0 创建适配器保存目录: {adapters_save_path}")

    # Wait for directory creation to complete before proceeding
    if is_distributed():
        torch.distributed.barrier()
        debugprint(f"进程 rank={rank} 等待目录创建完成")

    debugprint(f"进程 rank={rank} 适配器将保存到: {adapters_save_path}")

    # =========================== Train shared adapter ===========================
    adapter_name = "shared_adapter"
    adapter_output_dir = os.path.join(adapters_save_path, adapter_name)
    debugprint(f"共享适配器名称: {adapter_name}")
    debugprint(f"共享适配器输出目录: {adapter_output_dir}")

    # Set shared adapter output directory
    training_args_shared = copy.deepcopy(training_args)  # Create copy for shared adapter
    training_args_shared.output_dir = adapter_output_dir
    training_args_shared.overwrite_output_dir = True  # Always overwrite shared adapter if re-training
    debugprint(f"为共享适配器设置的 training_args.output_dir: {training_args_shared.output_dir}")

    # Only create directories in the main process
    if is_main_process():
        os.makedirs(training_args_shared.output_dir, exist_ok=True)
        debugprint(f"进程 rank=0 创建共享适配器输出目录: {training_args_shared.output_dir}")

    # Wait for directory creation to complete before proceeding
    if is_distributed():
        torch.distributed.barrier()
        debugprint(f"进程 rank={rank} 等待共享适配器目录创建完成")

    debugprint(f"进程 rank={rank} 训练共享适配器: {adapter_name}, 输出目录: {training_args_shared.output_dir}")

    # Create model args copy for shared adapter
    model_args_shared = copy.deepcopy(model_args)
    finetuning_args_shared = copy.deepcopy(finetuning_args)

    # Check if pretrained shared adapter exists, check by adapter_config.json file
    shared_adapter_config_path = os.path.join(adapter_output_dir, "adapter_config.json")
    if os.path.exists(shared_adapter_config_path):
        # If pretrained shared adapter exists, load it
        model_args_shared.adapter_name_or_path = [adapter_output_dir]
        finetuning_args_shared.adapter_name_or_path = [adapter_output_dir]
        debugprint(f"从以下路径加载预训练的共享适配器: {adapter_output_dir} (rank0)")
    else:
        # Otherwise initialize new shared adapter (adapter_name_or_path should be None or empty list)
        model_args_shared.adapter_name_or_path = None
        finetuning_args_shared.adapter_name_or_path = None
        debugprint(f"未找到预训练的共享适配器 ({shared_adapter_config_path} 不存在)，将初始化一个新的共享适配器 (rank0)")

    # Load model (with shared adapter if exists, or initialize new one)
    debugprint("加载用于共享适配器训练的模型")
    model = load_model(tokenizer, model_args_shared, finetuning_args_shared, training_args_shared.do_train)
    debugprint("共享适配器模型加载完成")

    # Create dataset classifier
    dataset_classifier = None
    dataset_names = []
    dataset_indices_map = {}

    if training_args.do_train and len(datasets_info) > 0:
        debugprint("训练模式且有数据集信息，准备创建/加载数据集分类器")
        # Set dataset information
        num_datasets = len(datasets_info)
        dataset_names = [info["name"] for info in datasets_info]
        debugprint(f"数据集数量: {num_datasets}, 数据集名称: {dataset_names}")

        # Get model hidden size
        hidden_size = model.config.hidden_size
        debugprint(f"模型 hidden size: {hidden_size}")

        # Check if pretrained classifier exists
        classifier_path = os.path.join(adapter_output_dir, "dataset_classifier")
        classifier_config_file = os.path.join(classifier_path, "classifier_config.json")
        debugprint(f"检查预训练分类器路径: {classifier_path}")

        if os.path.exists(classifier_config_file):
            # Load and potentially expand classifier
            debugprint(f"进程 rank={rank} 从以下路径加载并可能扩展数据集分类器: {classifier_path}")

            # 获取模型的数据类型
            model_dtype = next(model.parameters()).dtype
            debugprint(f"进程 rank={rank} 模型参数数据类型: {model_dtype}")

            loaded_classifier, old_dataset_names = load_classifier(
                classifier_path,
                hidden_size,
                new_num_datasets=num_datasets,
                dtype=model_dtype
            )
            if loaded_classifier:
                dataset_classifier = loaded_classifier
                debugprint(f"加载的分类器原支持数据集: {old_dataset_names} (rank0)")
                debugprint(f"扩展/用于当前数据集: {dataset_names} (rank0)")
            else:
                 debugprint(f"进程 rank={rank} 警告: load_classifier 未能从 {classifier_path} 加载分类器，将创建一个新的。")
                 # 获取模型的数据类型
                 model_dtype = next(model.parameters()).dtype
                 debugprint(f"进程 rank={rank} 模型参数数据类型: {model_dtype}")

                 # 创建与模型相同数据类型的分类器
                 dataset_classifier = DatasetClassifier(hidden_size, num_datasets, dtype=model_dtype)
                 debugprint(f"进程 rank={rank} 因加载失败，创建了新的数据集分类器，包含 {num_datasets} 个数据集，数据类型: {model_dtype}")
        else:
            # Create new dataset classifier with the same dtype as model
            # 获取模型的数据类型
            model_dtype = next(model.parameters()).dtype
            debugprint(f"进程 rank={rank} 模型参数数据类型: {model_dtype}")

            # 创建与模型相同数据类型的分类器
            dataset_classifier = DatasetClassifier(hidden_size, num_datasets, dtype=model_dtype)
            debugprint(f"进程 rank={rank} 创建了新的数据集分类器，包含 {num_datasets} 个数据集，数据类型: {model_dtype}")

        # Move classifier to the correct device
        dataset_classifier.to(training_args_shared.device)
        debugprint(f"进程 rank={rank} 已将数据集分类器移动到设备: {training_args_shared.device}")

        # Synchronize classifier parameters across all processes if in distributed mode
        if is_distributed():
            # Broadcast the classifier parameters from rank 0 to all other processes
            for param in dataset_classifier.parameters():
                torch.distributed.broadcast(param.data, src=0)
            debugprint(f"进程 rank={rank} 已同步数据集分类器参数")

        # Build dataset index mapping for identifying which dataset a sample belongs to during training
        start_idx = 0
        current_dataset_indices_map = {}
        for idx, ds_info in enumerate(datasets_info):
            end_idx = start_idx + ds_info["size"]
            current_dataset_indices_map[idx] = (start_idx, end_idx)
            debugprint(f"数据集 '{ds_info['name']}' (索引 {idx}) 的样本索引范围: [{start_idx}, {end_idx})")
            start_idx = end_idx
        # Assign to the instance variable used by the trainer
        dataset_indices_map_for_trainer = {v: k for k, v in current_dataset_indices_map.items()}


        debugprint(f"数据集分类器已设置 {num_datasets} 个类别 (rank0)")
        for idx, name in enumerate(dataset_names):
            debugprint(f"  数据集 {idx}: {name} (rank0)")
    else:
        debugprint("非训练模式或无数据集信息，跳过数据集分类器设置")


    if getattr(model, "is_quantized", False) and not training_args.do_train:
        debugprint("模型已量化且非训练模式，设置 _hf_peft_config_loaded=True (hack)")
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    debugprint("创建 SFTDataCollatorWith4DAttentionMask (共享适配器)")
    data_collator = SFTDataCollatorWith4DAttentionMask(
        tokenizer=tokenizer,
        template=template,
        model=model if not training_args_shared.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args_shared.do_train else None,  # Use shared training args
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args_shared.generation_max_length = training_args_shared.generation_max_length or data_args.cutoff_len
    training_args_shared.generation_num_beams = data_args.eval_num_beams or training_args_shared.generation_num_beams
    training_args_shared.remove_unused_columns = False  # important for multimodal dataset
    debugprint(f"共享适配器训练设置: generation_max_length={training_args_shared.generation_max_length}, generation_num_beams={training_args_shared.generation_num_beams}")

    # Metric utils
    metric_module = {}
    if training_args_shared.predict_with_generate:
        debugprint("使用 ComputeSimilarity 作为评估指标 (predict_with_generate=True)")
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args_shared.compute_accuracy:
        debugprint("使用 ComputeAccuracy 作为评估指标 (compute_accuracy=True)")
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor
    else:
        debugprint("未指定计算指标的函数")

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()
    debugprint(f"生成参数 (gen_kwargs): {gen_kwargs}")

    # Create callbacks copy for shared adapter
    current_callbacks = copy.deepcopy(callbacks) if callbacks is not None else []
    debugprint(f"使用的回调函数数量 (共享适配器): {len(current_callbacks)}")

    # Initialize Trainer
    if dataset_classifier is not None and training_args.do_train:
        debugprint("初始化 DynamicConPetTrainer (共享适配器训练)")
        trainer = DynamicConPetTrainer(
            model=model,
            args=training_args_shared,
            finetuning_args=finetuning_args_shared,
            cl_finetuning_args=cl_finetuning_args,
            data_collator=data_collator,
            callbacks=current_callbacks,
            processor=processor,
            dataset_classifier=dataset_classifier,
            dataset_names=dataset_names,
            dataset_indices_map=dataset_indices_map_for_trainer,
            gen_kwargs=gen_kwargs,
            train_dataset=dataset_module.get("train_dataset"),
            eval_dataset=dataset_module.get("eval_dataset"),
            tokenizer=tokenizer,
            **metric_module,
        )
        debugprint("使用 DynamicConPetTrainer 并进行数据集分类 (rank0)")
    else:
        debugprint("初始化 CustomSeq2SeqTrainer (共享适配器训练，无分类器或非训练模式)")
        trainer = CustomSeq2SeqTrainer(
            model=model,
            args=training_args_shared,
            finetuning_args=finetuning_args_shared,
            data_collator=data_collator,
            callbacks=current_callbacks,
            processor=processor,
            gen_kwargs=gen_kwargs,
            train_dataset=dataset_module.get("train_dataset"),
            eval_dataset=dataset_module.get("eval_dataset"),
            tokenizer=tokenizer,
            **metric_module,
        )
        debugprint("使用标准的 CustomSeq2SeqTrainer (rank0)")

    # Train shared adapter
    if training_args_shared.do_train:
        debugprint("开始训练共享适配器")
        train_result = trainer.train(resume_from_checkpoint=training_args_shared.resume_from_checkpoint)
        debugprint(f"共享适配器训练完成，结果: {train_result.metrics}")
        debugprint("保存共享适配器模型")
        trainer.save_model()  # Save to adapter_output_dir (includes classifier if using DynamicConPetTrainer)

        if finetuning_args_shared.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )
            debugprint(f"计算得到的有效每秒 token 数: {train_result.metrics['effective_tokens_per_sec']}")

        if datasets_info:
            debugprint("记录共享适配器训练使用的数据集信息到 metrics")
            for idx, ds_info in enumerate(datasets_info):
                prefix = f"shared_adapter_dataset_{idx}"
                debugprint(f"记录数据集 {idx} ('{ds_info['name']}') 信息，前缀: {prefix}")
                for key, value in ds_info.items():
                    metric_key = f"{prefix}_{key}"
                    # Ensure all values are formattable basic types (string, number, etc.)
                    if isinstance(value, (str, int, float, bool)):
                        train_result.metrics[metric_key] = value
                    elif isinstance(value, list):
                        # Convert list to string
                        train_result.metrics[metric_key] = str(value)
                    else:
                        # Convert other types to string representation
                        train_result.metrics[metric_key] = str(value)
                    debugprint(f"  添加指标: {metric_key} = {train_result.metrics[metric_key]}")

            if cl_finetuning_args.replay_task_list:
                 # Ensure replay_task_list is stored as a string
                 if isinstance(cl_finetuning_args.replay_task_list, list):
                     task_list_str = ",".join(cl_finetuning_args.replay_task_list)
                 else:
                     task_list_str = str(cl_finetuning_args.replay_task_list)
                 train_result.metrics["shared_adapter_replay_task_list"] = task_list_str
                 debugprint(f"记录共享适配器回放任务列表: {task_list_str}")

        debugprint("记录共享适配器训练指标")
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        debugprint("保存共享适配器训练状态")
        trainer.save_state()
        if is_main_process() and finetuning_args_shared.plot_loss:
            debugprint("进程 rank=0 绘制共享适配器损失图")
            plot_loss(training_args_shared.output_dir, keys=["loss", "eval_loss", "eval_accuracy", "classification_loss", "total_loss"])
    else:
        debugprint("跳过共享适配器训练 (do_train=False)")

    # Clean up memory for task-specific adapter training
    debugprint("清理内存为任务特定适配器训练做准备")
    if is_distributed():
        torch.distributed.barrier()
    del model
    del trainer
    AcceleratorState._reset_state()   # 清掉全局 state
    PartialState._reset_state()       # 同时清 partial state               # 0.26+ 公共 API

    if 'dataset_classifier' in locals():
        del dataset_classifier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        debugprint("CUDA 缓存已清空")

    # =========================== Train task-specific adapter ===========================
    if not cl_finetuning_args.current_task_id:
        debugprint("警告: 未指定 current_task_id，使用 'current_task' 作为默认任务 ID (rank0)")
        task_id = "current_task"
    else:
        task_id = cl_finetuning_args.current_task_id
    debugprint(f"任务特定适配器 ID: {task_id}")

    # Use same dataset as shared adapter, no need to reload
    debugprint("\n" + "*" * 80)
    debugprint("*" + " " * 78 + "*")
    debugprint("*" + " " * 9 + "任务适配器使用与共享适配器相同的数据集" + " " * 9 + "*")
    debugprint("*" + " " * 78 + "*")
    debugprint("*" * 80 + "\n")

    adapter_name_task = task_id
    adapter_output_dir_task = os.path.join(adapters_save_path, adapter_name_task)
    debugprint(f"任务特定适配器名称: {adapter_name_task}")
    debugprint(f"任务特定适配器输出目录: {adapter_output_dir_task}")

    # Set task-specific adapter output directory
    training_args_task = copy.deepcopy(training_args_shared)  # Create new copy for task adapter
    training_args_task.output_dir = adapter_output_dir_task
    training_args_task.overwrite_output_dir = True
    debugprint(f"为任务适配器设置的 training_args.output_dir: {training_args_task.output_dir}")

    # Only create directories in the main process
    if is_main_process():
        os.makedirs(training_args_task.output_dir, exist_ok=True)
        debugprint(f"进程 rank=0 创建任务特定适配器输出目录: {training_args_task.output_dir}")

    # Wait for directory creation to complete before proceeding
    if is_distributed():
        torch.distributed.barrier()
        debugprint(f"进程 rank={rank} 等待任务特定适配器目录创建完成")

    debugprint(f"进程 rank={rank} 训练任务特定适配器: {adapter_name_task}, 输出目录: {training_args_task.output_dir}")

    # Create model args copy for task-specific adapter
    model_args_task = copy.deepcopy(model_args)
    finetuning_args_task = copy.deepcopy(finetuning_args)

    # Ensure shared adapter path exists (needed potentially for loading base model with shared weights)
    shared_adapter_dir = os.path.join(adapters_save_path, "shared_adapter")
    if not os.path.exists(shared_adapter_dir) or not os.path.exists(os.path.join(shared_adapter_dir, "adapter_config.json")):
        debugprint(f"警告: 未找到共享适配器: {shared_adapter_dir}，将继续，但如果下游期望共享适配器，可能会影响加载 (rank0)")

    # Only load base model, don't preload any adapters for the task-specific training stage
    model_args_task.adapter_name_or_path = None
    finetuning_args_task.adapter_name_or_path = None
    finetuning_args_task.adapter_name = adapter_name_task

    debugprint(f"加载基础模型并初始化新的任务特定适配器: {adapter_name_task} (rank0)")

    # Load model (this will load the base model and *initialize* a new LoRA adapter named adapter_name_task)
    model = load_model(tokenizer, model_args_task, finetuning_args_task, training_args_task.do_train)
    debugprint(f"任务适配器 '{adapter_name_task}' 的模型加载完成")

    # Check and log adapter configurations after model creation
    if hasattr(model, "peft_config") and model.peft_config:
        debugprint("检查模型加载后的 PEFT 配置:")
        for config_adapter_name, config in model.peft_config.items():
            debugprint(f"  适配器 '{config_adapter_name}' 配置: {config} (rank0)")
            if hasattr(config, "target_modules"):
                debugprint(f"  适配器 '{config_adapter_name}' 目标模块: {config.target_modules} (rank0)")
    else:
        debugprint("模型没有 PEFT 配置或配置为空")

    if getattr(model, "is_quantized", False) and not training_args_task.do_train:
        debugprint("模型已量化且非训练模式，设置 _hf_peft_config_loaded=True (hack)")
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    debugprint("创建 SFTDataCollatorWith4DAttentionMask (任务适配器)")
    data_collator = SFTDataCollatorWith4DAttentionMask(
        tokenizer=tokenizer,
        template=template,
        model=model if not training_args_task.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args_task.do_train else None,  # Use the same data collator instance
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    # Override the decoding parameters of Seq2SeqTrainer for task adapter
    training_args_task.generation_max_length = training_args_task.generation_max_length or data_args.cutoff_len
    training_args_task.generation_num_beams = data_args.eval_num_beams or training_args_task.generation_num_beams
    training_args_task.remove_unused_columns = False  # important for multimodal dataset             # 全局已初始化
    #training_args_task.deepspeed = None           # 不再传 plugin

    debugprint(f"任务适配器训练设置: generation_max_length={training_args_task.generation_max_length}, generation_num_beams={training_args_task.generation_num_beams}")

    # Metric utils (likely the same as for shared adapter)
    metric_module_task = {}
    if training_args_task.predict_with_generate:
        debugprint("使用 ComputeSimilarity 作为评估指标 (任务适配器, predict_with_generate=True)")
        metric_module_task["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args_task.compute_accuracy:
        debugprint("使用 ComputeAccuracy 作为评估指标 (任务适配器, compute_accuracy=True)")
        metric_module_task["compute_metrics"] = ComputeAccuracy()
        metric_module_task["preprocess_logits_for_metrics"] = eval_logit_processor
    else:
        debugprint("未指定计算指标的函数 (任务适配器)")

    # Keyword arguments for `model.generate` (likely the same)
    # gen_kwargs = ... # Already defined

    # Create callbacks copy for task-specific adapter
    current_callbacks_task = copy.deepcopy(callbacks) if callbacks is not None else []
    debugprint(f"使用的回调函数数量 (任务适配器): {len(current_callbacks_task)}")

    # Initialize Trainer for task-specific adapter (standard trainer, no classifier needed here)
    debugprint("初始化 CustomSeq2SeqTrainer (任务适配器训练)")
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args_task,
        finetuning_args=finetuning_args_task,
        data_collator=data_collator,
        callbacks=current_callbacks_task,
        processor=processor,
        gen_kwargs=gen_kwargs,
        train_dataset=dataset_module.get("train_dataset"),
        eval_dataset=dataset_module.get("eval_dataset"),
        tokenizer=tokenizer,
        **metric_module_task,
    )

    # Train task-specific adapter
    if training_args_task.do_train:
        debugprint("开始训练任务特定适配器")
        train_result = trainer.train(resume_from_checkpoint=training_args_task.resume_from_checkpoint)
        debugprint(f"任务特定适配器训练完成，结果: {train_result.metrics}")
        debugprint("保存任务特定适配器模型")
        trainer.save_model()  # Save to adapter_output_dir_task

        if finetuning_args_task.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )
            debugprint(f"计算得到的有效每秒 token 数 (任务适配器): {train_result.metrics['effective_tokens_per_sec']}")

        # Record used adapter information
        shared_adapter_full_path = os.path.join(adapters_save_path, "shared_adapter")
        task_adapter_full_path = os.path.join(adapters_save_path, adapter_name_task)
        train_result.metrics["shared_adapter"] = shared_adapter_full_path
        train_result.metrics["task_adapter"] = task_adapter_full_path
        debugprint(f"记录使用的适配器路径到 metrics: shared='{shared_adapter_full_path}', task='{task_adapter_full_path}'")

        # Add classifier related information if it exists
        classifier_path_final = os.path.join(shared_adapter_dir, "dataset_classifier")
        if os.path.exists(os.path.join(classifier_path_final, "classifier_config.json")):
            train_result.metrics["dataset_classifier"] = classifier_path_final
            debugprint(f"记录数据集分类器路径到 metrics: '{classifier_path_final}'")
        else:
            debugprint("未找到数据集分类器配置文件，不记录其路径")

        # Log to logger (replace with debugprint)
        debugprint(f"Dynamic ConPet 共享适配器: {shared_adapter_full_path}")
        debugprint(f"Dynamic ConPet 任务适配器: {task_adapter_full_path}")

        debugprint("记录任务适配器训练指标")
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        debugprint("保存任务适配器训练状态")
        trainer.save_state()
        if is_main_process() and finetuning_args_task.plot_loss:
            debugprint("进程 rank=0 绘制任务适配器损失图")
            plot_keys = [k for k in ["loss", "eval_loss", "eval_accuracy"] if k in train_result.metrics or hasattr(trainer.state, k)]
            if "dynamic_conpet_loss" in train_result.metrics: plot_keys.append("dynamic_conpet_loss")
            if "shared_l2_loss" in train_result.metrics: plot_keys.append("shared_l2_loss")
            if plot_keys:
                 plot_loss(training_args_task.output_dir, keys=plot_keys)
            else:
                 debugprint("进程 rank=0 警告: 没有可绘制的损失键")
    else:
        debugprint("跳过任务特定适配器训练 (do_train=False)")

    if training_args_task.predict_with_generate:
        debugprint("为生成设置 tokenizer padding_side = 'left'")
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluate task-specific adapter
    if training_args_task.do_eval:
        debugprint("开始评估任务特定适配器")
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        debugprint(f"任务特定适配器评估完成，指标: {metrics}")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    else:
        debugprint("跳过任务特定适配器评估 (do_eval=False)")

    # Generate predictions with task-specific adapter
    if training_args_task.do_predict:
        debugprint("警告: 批量生成可能非常慢。考虑使用 `scripts/vllm_infer.py` 代替。(rank0)")
        eval_dataset_for_predict = dataset_module.get("eval_dataset", dataset_module.get("test_dataset"))
        if eval_dataset_for_predict:
            debugprint(f"开始使用任务特定适配器在大小为 {len(eval_dataset_for_predict)} 的数据集上生成预测")
            predict_results = trainer.predict(eval_dataset_for_predict, metric_key_prefix="predict", **gen_kwargs)
            debugprint(f"任务特定适配器预测完成，指标: {predict_results.metrics}")
            trainer.log_metrics("predict", predict_results.metrics)
            trainer.save_metrics("predict", predict_results.metrics)
            debugprint("保存预测结果")
            trainer.save_predictions(eval_dataset_for_predict, predict_results, generating_args.skip_special_tokens)
        else:
             debugprint("警告: 未找到用于预测的数据集 (eval_dataset 或 test_dataset)")
    else:
        debugprint("跳过任务特定适配器预测 (do_predict=False)")

    # Create model card
    debugprint("创建模型卡并推送 (如果配置了)")
    create_modelcard_and_push(trainer, model_args_task, data_args, training_args_task, finetuning_args_task)

    # Clean up memory
    debugprint("清理任务特定适配器训练后的内存")
    del model
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        debugprint("CUDA 缓存已清空")

    # Restore original output directory (optional, maybe not needed)

    # Restore original output directory
    #logger.info_rank0(f"Dynamic ConPet training completed. Adapters saved to {adapters_save_path}")
    #logger.info_rank0(f"  - Shared adapter: {os.path.join(adapters_save_path, 'shared_adapter')}")
    #logger.info_rank0(f"  - Task-specific adapter: {os.path.join(adapters_save_path, task_id)}")

    return