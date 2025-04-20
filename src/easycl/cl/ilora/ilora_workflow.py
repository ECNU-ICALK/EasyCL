import os  # 添加此行导入os模块
import copy # Add copy
import random # Add random
from typing import TYPE_CHECKING, Optional

from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from llamafactory.data.data_utils import merge_dataset # Add merge_dataset
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.logging import get_logger
from llamafactory.extras.misc import calculate_tps, get_logits_processor
from llamafactory.extras.ploting import plot_loss
from llamafactory.model import load_tokenizer
from llamafactory.train.sft.metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from llamafactory.train.trainer_utils import create_modelcard_and_push
from easycl.cl.ilora.ilora_loader import load_ilora_model
from easycl.cl.ilora.ilora_trainer import ILORATrainer

def debugprint(*args, **kwargs):
    pass
if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from llamafactory.hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments
    from easycl.hparams.cl_finetuning_args import CLFinetuningArguments


logger = get_logger(__name__)


def run_sft_ilora(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    cl_finetuning_args: "CLFinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    """
    Runs the supervised fine-tuning process with I-LORA.
    
    This function is based on run_sft but uses I-LORA specific components.
    """
    debugprint("进入 run_sft_ilora 工作流") # 函数入口
    debugprint(f"  model_args: {model_args}")
    debugprint(f"  data_args: {data_args}")
    debugprint(f"  training_args: {training_args}")
    debugprint(f"  finetuning_args: {finetuning_args}")
    debugprint(f"  cl_finetuning_args: {cl_finetuning_args}") # 检查 CL 参数
    debugprint(f"  generating_args: {generating_args}")
    debugprint(f"  callbacks: {callbacks}")

    logger.info_rank0("Running SFT with I-LORA...")
    
    # Determine current task type (first task or subsequent task)
    is_first_task = cl_finetuning_args.prev_task_id is None
    if is_first_task:
        logger.info_rank0("Detected first task")
        debugprint("  任务类型判断：首任务")
    else:
        logger.info_rank0(f"Detected subsequent task, previous task ID: {cl_finetuning_args.prev_task_id}")
        debugprint(f"  任务类型判断：后续任务，prev_task_id: {cl_finetuning_args.prev_task_id}")
        
        # If previous_task_model is specified, print path for confirmation
        if cl_finetuning_args.previous_task_model:
            logger.info_rank0(f"Using specified previous task model path: {cl_finetuning_args.previous_task_model}")
            debugprint(f"  使用用户指定的前一个任务模型路径: {cl_finetuning_args.previous_task_model}")
            
            # Check for EMA adapter existence
            ema_path = os.path.join(cl_finetuning_args.previous_task_model, "ema")
            ema_adapter_path = os.path.join(cl_finetuning_args.previous_task_model, "ema_adapter")
            debugprint(f"    检查 EMA 适配器路径: {ema_path}, {ema_adapter_path}")
            
            if os.path.exists(ema_path):
                logger.info_rank0(f"Found previous task's EMA adapter: {ema_path}")
                debugprint(f"    找到前一个任务的 EMA 适配器: {ema_path}")
            elif os.path.exists(ema_adapter_path):
                logger.info_rank0(f"Found previous task's EMA adapter: {ema_adapter_path}")
                debugprint(f"    找到前一个任务的 EMA 适配器: {ema_adapter_path}")
            else:
                logger.warning_rank0(f"No EMA adapter found in {cl_finetuning_args.previous_task_model}")
                debugprint(f"    在 {cl_finetuning_args.previous_task_model} 中未找到 EMA 适配器")
        else:
            debugprint("  未指定前一个任务模型路径 (previous_task_model)")
    
    # Load tokenizer and get template
    debugprint("  加载 tokenizer 和 template")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    debugprint(f"  Tokenizer 加载完成 (类型: {type(tokenizer)})")
    
    # Get dataset with replay logic
    debugprint("  获取并处理数据集 (包含重放逻辑)")
    # 1. Load current task dataset first
    current_dataset_module = get_dataset(
        template=template,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        stage="sft",
        **tokenizer_module
    )
    debugprint(f"  当前任务数据集获取完成，包含键: {list(current_dataset_module.keys())}")

    merged_datasets = []
    replay_datasets_info = []
    dataset_module = {} # Initialize dataset_module

    # Add current training data if exists
    if "train_dataset" in current_dataset_module:
        current_train_dataset = current_dataset_module["train_dataset"]
        merged_datasets.append(current_train_dataset)
        logger.info_rank0(f"Loaded current dataset ({data_args.dataset}) with {len(current_train_dataset)} samples.")
        debugprint(f"  添加当前训练集到 merged_datasets (大小: {len(current_train_dataset)})")
        current_dataset_info = {
            "name": "current_task",
            "dataset": copy.deepcopy(data_args.dataset), # Store original dataset names
            "size": len(current_train_dataset)
        }
        replay_datasets_info.append(current_dataset_info)
    else:
        logger.warning_rank0("No training data found for the current task.")
        debugprint("  警告: 当前任务没有训练数据")

    # Keep eval dataset from current task
    if "eval_dataset" in current_dataset_module:
        dataset_module["eval_dataset"] = current_dataset_module["eval_dataset"]
        debugprint(f"  保留当前任务的评估集 (大小: {len(current_dataset_module['eval_dataset'])})")
    else:
        debugprint("  当前任务没有评估集")


    # 2. Check if replay is needed (replay_task_list is provided and we are training)
    if cl_finetuning_args.replay_task_list and training_args.do_train and merged_datasets: # Only replay if training and current data exists
        logger.info_rank0("*" * 80)
        logger.info_rank0("*" + " " * 78 + "*")
        logger.info_rank0("*" + " " * 28 + "REPLAY DATA LOADING ENABLED" + " " * 27 + "*")
        logger.info_rank0("*" + " " * 78 + "*")
        debugprint("  检测到 replay_task_list，开始加载重放数据")

        # Save current data directory and dataset configuration
        original_dataset_dir = copy.deepcopy(data_args.dataset_dir)
        original_dataset = copy.deepcopy(data_args.dataset)
        debugprint(f"  保存原始 data_args: dataset_dir='{original_dataset_dir}', dataset='{original_dataset}'")

        # Parse replay task list and maximum samples per task
        replay_task_list = [task.strip() for task in cl_finetuning_args.replay_task_list.split(',') if task.strip()]
        debugprint(f"  解析得到的 replay_task_list: {replay_task_list}")

        # Parse maximum samples list (if provided)
        maxsamples_list = None
        if cl_finetuning_args.maxsamples_list:
            try:
                maxsamples_list = [int(x.strip()) for x in cl_finetuning_args.maxsamples_list.split(',') if x.strip()]
                if len(maxsamples_list) != len(replay_task_list):
                    logger.warning_rank0(f"Length mismatch: maxsamples_list ({len(maxsamples_list)}) vs replay_task_list ({len(replay_task_list)}). Using replay_ratio ({cl_finetuning_args.replay_ratio}) instead.")
                    debugprint("  maxsamples_list 长度与 replay_task_list 不匹配，回退到 replay_ratio")
                    maxsamples_list = None
                else:
                    debugprint(f"  解析得到的 maxsamples_list: {maxsamples_list}")
            except ValueError:
                logger.warning_rank0(f"Invalid format in maxsamples_list: {cl_finetuning_args.maxsamples_list}. Using replay_ratio ({cl_finetuning_args.replay_ratio}) instead.")
                debugprint(f"  maxsamples_list 格式无效，回退到 replay_ratio")
                maxsamples_list = None
        else:
            debugprint(f"  未使用 maxsamples_list，将使用 replay_ratio: {cl_finetuning_args.replay_ratio}")
            if cl_finetuning_args.replay_ratio >= 0: # Only log ratio if it's meaningful
                logger.info_rank0(f"* Using replay ratio: {cl_finetuning_args.replay_ratio}" + " " * (77 - len(f"* Using replay ratio: {cl_finetuning_args.replay_ratio}")) + "*")

        # If previous_task_dataset is provided, use it as data directory for replay tasks
        replay_data_dir = original_dataset_dir # Default to original
        if cl_finetuning_args.previous_task_dataset:
             if os.path.exists(cl_finetuning_args.previous_task_dataset):
                 replay_data_dir = cl_finetuning_args.previous_task_dataset
                 logger.info_rank0(f"* Using custom dataset directory for replay tasks: {replay_data_dir}" + " " * (77 - len(f"* Using custom dataset directory for replay tasks: {replay_data_dir}")) + "*")
                 debugprint(f"  为重放任务设置 dataset_dir 为: {replay_data_dir}")
             else:
                 logger.warning_rank0(f"Specified previous_task_dataset path not found: {cl_finetuning_args.previous_task_dataset}. Using default dataset directory: {original_dataset_dir}")
                 debugprint(f"  指定的 previous_task_dataset 路径 '{cl_finetuning_args.previous_task_dataset}' 不存在，使用默认路径: {original_dataset_dir}")
                 # Keep replay_data_dir as original_dataset_dir

        logger.info_rank0(f"* Replay task list: {replay_task_list}" + " " * (77 - len(f"* Replay task list: {replay_task_list}")) + "*")
        if maxsamples_list:
            logger.info_rank0(f"* Max samples per task: {maxsamples_list}" + " " * (77 - len(f"* Max samples per task: {maxsamples_list}")) + "*")
        logger.info_rank0("*" + " " * 78 + "*")
        logger.info_rank0("*" * 80 + "*")

        # Temporarily change dataset_dir for loading replay tasks if needed
        data_args.dataset_dir = replay_data_dir

        for task_idx, task_name in enumerate(replay_task_list):
            # Set current replay task in data_args (dataset is now a list)
            data_args.dataset = [task_name]
            debugprint(f"  开始加载重放任务 {task_idx+1}/{len(replay_task_list)}: {task_name} (从目录: {data_args.dataset_dir})")
            logger.info_rank0(f"Loading replay task {task_idx+1}/{len(replay_task_list)}: {task_name} from {data_args.dataset_dir}")

            try:
                # Load replay task dataset
                replay_dataset_module = get_dataset(
                    template=template,
                    model_args=model_args,
                    data_args=data_args,
                    training_args=training_args,
                    stage="sft",
                    **tokenizer_module
                )

                if "train_dataset" in replay_dataset_module:
                    replay_train_data = replay_dataset_module["train_dataset"]
                    total_samples = len(replay_train_data)
                    debugprint(f"    加载成功，原始样本数: {total_samples}")

                    # Determine sample count
                    max_samples = 0
                    if maxsamples_list and task_idx < len(maxsamples_list):
                        max_samples = min(maxsamples_list[task_idx], total_samples)
                        debugprint(f"    使用 maxsamples_list[{task_idx}]={maxsamples_list[task_idx]}，确定最大样本数: {max_samples}")
                        logger.info_rank0(f"  Task {task_name}: Using max samples from list: {max_samples}")
                    elif cl_finetuning_args.replay_ratio >= 0: # Use ratio only if non-negative
                        current_task_size = len(merged_datasets[0]) # Size of the first dataset (current task)
                        max_samples = min(int(current_task_size * cl_finetuning_args.replay_ratio), total_samples)
                        debugprint(f"    使用 replay_ratio={cl_finetuning_args.replay_ratio} * 当前任务大小={current_task_size}，确定最大样本数: {max_samples} (上限为 {total_samples})")
                        logger.info_rank0(f"  Task {task_name}: Using ratio ({cl_finetuning_args.replay_ratio}) of current task size ({current_task_size}), resulting in max samples: {max_samples}")
                    else: # replay_ratio < 0 means take all
                         max_samples = total_samples
                         debugprint(f"    replay_ratio < 0，使用全部样本: {max_samples}")
                         logger.info_rank0(f"  Task {task_name}: Replay ratio is negative, using all {total_samples} samples.")


                    if 0 < max_samples < total_samples:
                        # Randomly select specified number of samples
                        indices = random.sample(range(total_samples), max_samples)
                        sampled_replay_dataset = replay_train_data.select(indices)
                        logger.info_rank0(f"  Task {task_name}: Selected {max_samples}/{total_samples} samples.")
                        debugprint(f"    随机采样 {max_samples} 个样本")
                    elif max_samples >= total_samples:
                        sampled_replay_dataset = replay_train_data
                        max_samples = total_samples # Ensure max_samples reflects actual number
                        logger.info_rank0(f"  Task {task_name}: Using all {total_samples} samples.")
                        debugprint(f"    使用全部 {total_samples} 个样本")
                    else: # max_samples <= 0
                        logger.info_rank0(f"  Task {task_name}: Skipping replay as max_samples is {max_samples}.")
                        debugprint(f"    最大样本数为 {max_samples}，跳过此任务的重放")
                        continue # Skip to next task if max_samples is 0

                    merged_datasets.append(sampled_replay_dataset)
                    debugprint(f"    将采样后的数据集 (大小: {len(sampled_replay_dataset)}) 添加到 merged_datasets")

                    # Record replay dataset info
                    replay_dataset_info = {
                        "name": task_name,
                        "size_original": total_samples,
                        "size_selected": len(sampled_replay_dataset)
                    }
                    replay_datasets_info.append(replay_dataset_info)

                else:
                    logger.warning_rank0(f"No training data found for replay task: {task_name}")
                    debugprint(f"    警告: 未找到重放任务 {task_name} 的训练数据")

            except Exception as e:
                logger.error(f"Failed to load or process replay task {task_name}: {str(e)}", exc_info=True)
                debugprint(f"    错误: 加载或处理重放任务 {task_name} 失败: {e}")
                continue # Skip to next task

        # Restore original dataset configuration *after* the loop
        data_args.dataset_dir = original_dataset_dir
        data_args.dataset = original_dataset
        debugprint(f"  恢复原始 data_args: dataset_dir='{original_dataset_dir}', dataset='{original_dataset}'")


        # 3. Merge datasets if replay occurred
        if len(merged_datasets) > 1:
            logger.info_rank0(f"Merging {len(merged_datasets)} datasets (current + {len(merged_datasets)-1} replay tasks) using 'concat' strategy.")
            debugprint(f"  开始合并 {len(merged_datasets)} 个数据集")
            # Use concat strategy to merge datasets
            merged_data_args = copy.deepcopy(data_args) # Use the restored data_args
            merged_data_args.mix_strategy = "concat"

            # Assign merged dataset to train_dataset
            dataset_module["train_dataset"] = merge_dataset(
                merged_datasets,
                merged_data_args,
                seed=training_args.seed
            )
            debugprint(f"  合并完成，最终训练集大小: {len(dataset_module['train_dataset'])}")

            # Summarize merged dataset information
            total_samples = len(dataset_module["train_dataset"])
            logger.info_rank0("*" + "#" * 78 + "*")
            logger.info_rank0("#" + " " * 78 + "#")
            logger.info_rank0("#" + " " * 25 + "DATASET MERGE SUMMARY" + " " * 25 + "#")
            logger.info_rank0("#" + " " * 78 + "#")
            logger.info_rank0("#" + f" Total merged training samples: {total_samples}" + " " * (77 - len(f" Total merged training samples: {total_samples}")) + "#")

            for ds_info in replay_datasets_info:
                if ds_info["name"] == "current_task":
                     current_ds_str = str(ds_info.get('dataset', 'N/A')) # Handle potential missing key
                     ds_status = f" Current task ({current_ds_str}): {ds_info['size']} samples"
                elif "size_selected" in ds_info:
                    # Replay dataset
                    ds_status = f" Replay task '{ds_info['name']}': {ds_info['size_selected']}/{ds_info['size_original']} samples selected"
                else: # Should not happen if recorded correctly
                     ds_status = f" Info for {ds_info['name']} missing details."
                logger.info_rank0("#" + ds_status + " " * (77 - len(ds_status)) + "#")

            logger.info_rank0("#" + " " * 78 + "#")
            logger.info_rank0("#" * 80 + "*")

        else: # Only current task data ended up in merged_datasets
            logger.info_rank0("No replay tasks were successfully added or loaded. Using only current task data for training.")
            debugprint("  没有成功执行重放或只有当前任务数据，直接使用当前训练集")
            if merged_datasets: # Should contain only current task data
                 dataset_module["train_dataset"] = merged_datasets[0]
            # eval_dataset is already handled

    else: # No replay_task_list provided or not in training mode or no current data
        if not training_args.do_train:
            debugprint("  不在训练模式，跳过重放数据加载")
        elif not cl_finetuning_args.replay_task_list:
            debugprint("  未提供 replay_task_list，跳过重放数据加载")
        elif not merged_datasets:
             debugprint("  当前任务无训练数据，跳过重放数据加载")
        else: # replay_task_list provided, but maybe only current task data loaded
            logger.info_rank0("Using only current task data as replay is not applicable or configured.")
            debugprint("  直接使用当前任务训练集 (重放未触发或仅当前任务)")

        # Assign the current dataset directly if it exists
        if merged_datasets: # Use the list which contains current data if loaded
            dataset_module["train_dataset"] = merged_datasets[0]
            debugprint("  确认训练集为当前任务数据")
        # eval_dataset is already handled

    # Ensure train_dataset exists if do_train is True
    if training_args.do_train and "train_dataset" not in dataset_module:
        logger.error("Training is enabled but no training dataset could be prepared (current or replay). Aborting.")
        raise ValueError("Training dataset is missing.")
    elif not training_args.do_train:
        # Ensure eval_dataset exists if do_eval or do_predict is True
         if (training_args.do_eval or training_args.do_predict) and "eval_dataset" not in dataset_module:
              logger.warning_rank0("Evaluation/Prediction is enabled but no evaluation dataset was loaded.")
              # Optionally raise error or proceed cautiously
              # raise ValueError("Evaluation/Prediction dataset is missing.")
         
    debugprint(f"  最终数据集模块键: {list(dataset_module.keys())}")
    if "train_dataset" in dataset_module:
        debugprint(f"    训练集大小: {len(dataset_module['train_dataset'])}")
    if "eval_dataset" in dataset_module:
        debugprint(f"    评估集大小: {len(dataset_module['eval_dataset'])}")
    # ---- End of Dataset Loading/Replay Logic ----

    # Infer previous task path before loading model (if needed for EMA)
    if not is_first_task and not cl_finetuning_args.previous_task_model:
        debugprint("  后续任务且未指定 previous_task_model，尝试自动推断路径")
        # Try to automatically infer previous task path
        possible_prev_paths = []
        
        # Infer based on current output path
        current_dir_parts = training_args.output_dir.split('/')
        if len(current_dir_parts) >= 2:
            # Assume path format is base_dir/task_X_name
            base_dir = '/'.join(current_dir_parts[:-1])
            prev_task_path = os.path.join(base_dir, f"task_0_{cl_finetuning_args.prev_task_id}")
            possible_prev_paths.append(prev_task_path)
            debugprint(f"    基于当前输出目录推断路径: {prev_task_path}")
        
        # Look in saves directory
        saves_base = os.path.join("saves", model_args.model_name_or_path.split('/')[-1], "lora") # 使用模型名称
        prev_task_saves = os.path.join(saves_base, f"task_0_{cl_finetuning_args.prev_task_id}")
        possible_prev_paths.append(prev_task_saves)
        debugprint(f"    基于 saves 目录推断路径: {prev_task_saves}")
        
        # Check which path exists
        found_path = False
        for path in possible_prev_paths:
            debugprint(f"    检查推断路径是否存在: {path}")
            if os.path.exists(path):
                logger.info_rank0(f"Automatically discovered previous task path: {path}")
                debugprint(f"    自动发现前一个任务路径: {path}")
                cl_finetuning_args.previous_task_model = path
                found_path = True
                break
                
        if not found_path:
            logger.warning_rank0(f"Could not automatically find previous task path, please specify --previous_task_model")
            debugprint("    未能自动找到前一个任务路径，请确认配置")
    
    # Load model with I-LORA adapter
    debugprint("  加载带有 I-LORA 适配器的模型，调用 load_ilora_model")
    model = load_ilora_model(
        tokenizer=tokenizer,
        model_args=model_args,
        finetuning_args=finetuning_args, # 注意这里传递的是 finetuning_args
        cl_finetuning_args=cl_finetuning_args, # 传递 CL 参数
        is_trainable=training_args.do_train  # 确保训练时 is_trainable 为 True
    )
    debugprint(f"  load_ilora_model 调用完成 (模型类型: {type(model)})")

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction
        debugprint("  应用 PEFT config 加载 hack 以兼容量化模型的预测")

    # Set up data collator
    debugprint("  设置 Data Collator (SFTDataCollatorWith4DAttentionMask)")
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )
    debugprint("  Data Collator 设置完成")

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False  # important for multimodal dataset
    debugprint("  已覆盖 Seq2SeqTrainer 的解码参数")
    debugprint(f"    generation_max_length: {training_args.generation_max_length}")
    debugprint(f"    generation_num_beams: {training_args.generation_num_beams}")
    debugprint(f"    remove_unused_columns: {training_args.remove_unused_columns}")

    # Set up metrics
    metric_module = {}
    if training_args.predict_with_generate:
        debugprint("  设置评估指标: ComputeSimilarity (用于生成)")
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        debugprint("  设置评估指标: ComputeAccuracy")
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor
    else:
        debugprint("  未设置评估指标")

    # Set up generation kwargs
    debugprint("  设置生成参数 (gen_kwargs)")
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()
    debugprint(f"  生成参数: {gen_kwargs}")

    # Initialize I-LORA trainer
    debugprint("  初始化 ILORATrainer")
    trainer = ILORATrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        cl_finetuning_args=cl_finetuning_args, # 传递 CL 参数
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs=gen_kwargs,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )
    debugprint("  ILORATrainer 初始化完成")

    # Training
    if training_args.do_train:
        debugprint("  开始训练 (调用 trainer.train)")
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        debugprint(f"  训练完成，结果: {train_result}")
        debugprint("  保存模型 (调用 trainer.save_model)")
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            effective_tps = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )
            train_result.metrics["effective_tokens_per_sec"] = effective_tps
            debugprint(f"  计算有效 Tokens/秒: {effective_tps}")

        # Print I-LORA specific training statistics
        if hasattr(model, "ilora") and model.ilora is not None:
            debugprint("  打印 I-LORA 训练统计信息")
            logger.info_rank0("=" * 50)
            logger.info_rank0("I-LORA Training Complete, Statistics:")
            avg_consistency_loss = train_result.metrics.get('avg_consistency_loss', 'N/A')
            avg_total_loss = train_result.metrics.get('avg_ilora_total_loss', train_result.metrics.get('train_loss','N/A')) # 回退到train_loss
            ema_alpha = train_result.metrics.get('ilora_ema_alpha', 'N/A')
            consistency_weight = train_result.metrics.get('ilora_consistency_weight', 'N/A')
            logger.info_rank0(f"Average Consistency Loss: {avg_consistency_loss}")
            logger.info_rank0(f"Average Total Loss: {avg_total_loss}")
            logger.info_rank0(f"EMA Smoothing Coefficient: {ema_alpha}")
            logger.info_rank0(f"Consistency Loss Weight: {consistency_weight}")
            debugprint(f"    平均一致性损失: {avg_consistency_loss}")
            debugprint(f"    平均总损失: {avg_total_loss}")
            debugprint(f"    EMA Alpha: {ema_alpha}")
            debugprint(f"    一致性权重: {consistency_weight}")
            
            # Add adapter save path information
            if cl_finetuning_args.save_ema_adapter:
                ema_adapter_path = cl_finetuning_args.ema_adapter_path or "ema_adapter"
                if not os.path.isabs(ema_adapter_path):
                    ema_adapter_path = os.path.join(training_args.output_dir, ema_adapter_path)
                logger.info_rank0(f"EMA Adapter saved to: {ema_adapter_path}")
                debugprint(f"    EMA 适配器保存路径: {ema_adapter_path}")
                
                # Add EMA adapter path to training metrics
                train_result.metrics["ema_adapter_path"] = ema_adapter_path
            else:
                debugprint("    未启用 EMA 适配器保存")
            
            # Add previous task path information to training metrics
            if cl_finetuning_args.previous_task_model is not None:
                train_result.metrics["previous_task_model"] = cl_finetuning_args.previous_task_model
                debugprint(f"    记录前一个任务模型路径: {cl_finetuning_args.previous_task_model}")
                
                # Add previous task's EMA adapter load path
                loaded_ema_path = "none"
                for possible_path in ["ema", "ema_adapter"]:
                    full_path = os.path.join(cl_finetuning_args.previous_task_model, possible_path)
                    if os.path.exists(full_path):
                        loaded_ema_path = full_path
                        break
                train_result.metrics["loaded_ema_adapter_path"] = loaded_ema_path
                debugprint(f"    记录加载的前一个任务 EMA 适配器路径: {loaded_ema_path}")
            else:
                debugprint("    没有前一个任务模型路径信息")
                
            if cl_finetuning_args.current_task_id is not None:
                logger.info_rank0(f"Current Task ID: {cl_finetuning_args.current_task_id}")
                train_result.metrics["current_task_id"] = cl_finetuning_args.current_task_id
                debugprint(f"    记录当前任务 ID: {cl_finetuning_args.current_task_id}")
                
            if cl_finetuning_args.prev_task_id is not None:
                train_result.metrics["prev_task_id"] = cl_finetuning_args.prev_task_id
                debugprint(f"    记录前一个任务 ID: {cl_finetuning_args.prev_task_id}")
                
            logger.info_rank0("=" * 50)
        else:
            debugprint("  模型没有 ilora 实例，跳过 I-LORA 统计信息打印")

        # Add replay info to metrics if it exists
        if replay_datasets_info:
             debugprint("  添加重放数据集信息到训练指标")
             for idx, ds_info in enumerate(replay_datasets_info):
                 prefix = f"dataset_{idx}"
                 for key, value in ds_info.items():
                      metric_key = f"{prefix}_{key}"
                      # Ensure value is suitable type for metrics (str, int, float, bool)
                      if isinstance(value, (str, int, float, bool)):
                          train_result.metrics[metric_key] = value
                      elif isinstance(value, list): # Convert list to string
                          train_result.metrics[metric_key] = ','.join(map(str, value))
                      else:
                          train_result.metrics[metric_key] = str(value) # Fallback to string
             # Add replay task list itself if used
             if cl_finetuning_args.replay_task_list:
                  # Ensure replay_task_list is logged as a string
                  task_list_str = ','.join([task.strip() for task in cl_finetuning_args.replay_task_list.split(',') if task.strip()])
                  train_result.metrics["replay_task_list"] = task_list_str
                  debugprint(f"    记录 replay_task_list: {task_list_str}")
             # Add replay ratio/maxsamples if used
             if cl_finetuning_args.replay_task_list: # Redundant check, but safe
                 if maxsamples_list:
                      train_result.metrics["replay_maxsamples_list"] = ','.join(map(str, maxsamples_list))
                 else:
                      train_result.metrics["replay_ratio"] = cl_finetuning_args.replay_ratio
                 if cl_finetuning_args.previous_task_dataset:
                      train_result.metrics["replay_data_dir"] = cl_finetuning_args.previous_task_dataset


        debugprint("  记录训练指标 (trainer.log_metrics('train'))")
        trainer.log_metrics("train", train_result.metrics)
        debugprint("  保存训练指标 (trainer.save_metrics('train'))")
        trainer.save_metrics("train", train_result.metrics)
        debugprint("  保存训练状态 (trainer.save_state())")
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            debugprint(f"  绘制损失曲线到目录: {training_args.output_dir}")
            # Add more loss keys for plotting
            plot_loss(
                training_args.output_dir,
                keys=["loss", "eval_loss", "task_loss", "consistency_loss",
                     "ilora_total_loss", "eval_accuracy"]
            )
        elif not finetuning_args.plot_loss:
            debugprint("  未启用损失曲线绘制")
    else:
        debugprint("  跳过训练 (training_args.do_train is False)")

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation
        debugprint("  为生成设置 tokenizer.padding_side = 'left'")

    # Evaluation
    if training_args.do_eval:
        debugprint("  开始评估 (调用 trainer.evaluate)")
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        debugprint(f"  评估完成，指标: {metrics}")
        debugprint("  记录评估指标 (trainer.log_metrics('eval'))")
        trainer.log_metrics("eval", metrics)
        debugprint("  保存评估指标 (trainer.save_metrics('eval'))")
        trainer.save_metrics("eval", metrics)
    else:
        debugprint("  跳过评估 (training_args.do_eval is False)")

    # Predict
    if training_args.do_predict:
        logger.warning_rank0("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        debugprint("  开始预测 (调用 trainer.predict)")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        debugprint(f"  预测完成，结果: {predict_results}")
        debugprint("  记录预测指标 (trainer.log_metrics('predict'))")
        trainer.log_metrics("predict", predict_results.metrics)
        debugprint("  保存预测指标 (trainer.save_metrics('predict'))")
        trainer.save_metrics("predict", predict_results.metrics)
        debugprint("  保存预测结果 (trainer.save_predictions)")
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)
    else:
        debugprint("  跳过预测 (training_args.do_predict is False)")

    # Create model card
    debugprint("  创建模型卡并推送 (create_modelcard_and_push)")
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)

    debugprint("退出 run_sft_ilora 工作流") # 函数出口