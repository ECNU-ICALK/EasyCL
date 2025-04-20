import os
import copy
from typing import Optional, List, TYPE_CHECKING

import torch
from transformers import Seq2SeqTrainingArguments

from llamafactory.data import (
    get_dataset, 
    get_template_and_fix_tokenizer
)
from llamafactory.data.data_utils import merge_dataset
from llamafactory.data.parser import get_dataset_list
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.logging import get_logger
from llamafactory.extras.misc import calculate_tps, get_logits_processor
from llamafactory.extras.ploting import plot_loss
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.trainer_utils import create_modelcard_and_push
from llamafactory.train.sft.metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from llamafactory.data import SFTDataCollatorWith4DAttentionMask

from .ssr import SSR
from .ssr_trainer import SSRTrainer

# Add debugprint import

def debugprint(*args, **kwargs):
    pass


if TYPE_CHECKING:
    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments
    from ...hparams.cl_finetuning_args import CLFinetuningArguments
    from transformers import TrainerCallback

logger = get_logger(__name__)


def run_sft_ssr(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    cl_finetuning_args: "CLFinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None
):
    """
    Run SSR (Self-Synthesized Rehearsal) method for continual learning
    """
    # Debug: Print entry point and CL args
    debugprint(f"进入 run_sft_ssr 函数")
    debugprint(f"CL Finetuning Args: {cl_finetuning_args}")

    # Check if this is the first task
    is_first_task = not cl_finetuning_args.prev_task_id
    debugprint(f"是否是第一个任务: {is_first_task}, 上一个任务ID: {cl_finetuning_args.prev_task_id}")
    
    # Load tokenizer
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    debugprint("Tokenizer 和 Template 已加载")
    
    # For the first task, skip pseudo-sample generation and train normally
    if is_first_task:
        logger.info_rank0("Current task is the first task, skipping pseudo-sample generation and proceeding with normal training")
        debugprint("当前是第一个任务，跳过伪样本生成")
        
        # Load current task dataset
        debugprint("开始加载当前任务数据集...")
        dataset_module = get_dataset(
            template=template,
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            stage="sft",
            **tokenizer_module
        )
        debugprint("当前任务数据集加载完成")
        debugprint(f"训练集大小: {len(dataset_module.get('train_dataset', []))}")
        debugprint(f"评估集: {list(dataset_module.get('eval_dataset', {}).keys()) if isinstance(dataset_module.get('eval_dataset'), dict) else bool(dataset_module.get('eval_dataset'))}")
        
        # Load model
        debugprint("开始加载模型...")
        model = load_model(
            tokenizer=tokenizer,
            model_args=model_args,
            finetuning_args=finetuning_args,
            is_trainable=training_args.do_train,
        )
        debugprint("模型加载完成")
        
        # Normal training process
        debugprint("创建 Data Collator...")
        data_collator = SFTDataCollatorWith4DAttentionMask(
            template=template,
            model=model if not training_args.predict_with_generate else None,
            pad_to_multiple_of=8 if training_args.do_train else None,
            label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
            block_diag_attn=model_args.block_diag_attn,
            attn_implementation=getattr(model.config, "_attn_implementation", None),
            compute_dtype=model_args.compute_dtype,
            **tokenizer_module,
        )
        debugprint("Data Collator 创建完成")
        
        # Training arguments configuration
        training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
        training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
        training_args.remove_unused_columns = False
        
        # Metric configuration
        metric_module = {}
        if training_args.predict_with_generate:
            metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
            debugprint("使用 ComputeSimilarity 作为评估指标")
        elif finetuning_args.compute_accuracy:
            metric_module["compute_metrics"] = ComputeAccuracy()
            metric_module["preprocess_logits_for_metrics"] = eval_logit_processor
            debugprint("使用 ComputeAccuracy 作为评估指标")
        else:
            debugprint("未配置评估指标")
        
        # Generation parameters configuration
        gen_kwargs = generating_args.to_dict(obey_generation_config=True)
        gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
        gen_kwargs["logits_processor"] = get_logits_processor()
        debugprint("生成参数配置完成")
        
        # Initialize trainer
        debugprint("开始初始化 SSRTrainer...")
        trainer = SSRTrainer(
            model=model,
            args=training_args,
            finetuning_args=finetuning_args,
            cl_finetuning_args=cl_finetuning_args,
            data_collator=data_collator,
            callbacks=callbacks,
            gen_kwargs=gen_kwargs,
            **dataset_module,
            **tokenizer_module,
            **metric_module,
        )
        debugprint("SSRTrainer 初始化完成")
        
        # Start training
        if training_args.do_train:
            debugprint("开始训练...")
            train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
            debugprint("训练完成")
            trainer.save_model()
            debugprint("模型已保存")
            
            if finetuning_args.include_effective_tokens_per_second:
                train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                    dataset_module["train_dataset"], train_result.metrics, stage="sft"
                )
                debugprint(f"计算有效 tokens/sec: {train_result.metrics['effective_tokens_per_sec']}")
                
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()
            debugprint("训练指标和状态已保存")
            
            if trainer.is_world_process_zero() and finetuning_args.plot_loss:
                debugprint("开始绘制损失曲线...")
                plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])
                debugprint("损失曲线绘制完成")
        
        # Prediction configuration
        if training_args.predict_with_generate:
            tokenizer.padding_side = "left"
            debugprint("设置 tokenizer padding_side 为 left 以进行生成预测")
        
        # Evaluation
        if training_args.do_eval:
            debugprint("开始评估...")
            metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
            debugprint(f"评估完成，指标: {metrics}")
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
            debugprint("评估指标已记录和保存")
        
        # Prediction
        if training_args.do_predict:
            logger.warning_rank0("Batch generation may be slow. Consider using `scripts/vllm_infer.py` instead.")
            debugprint("开始预测...")
            predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
            debugprint(f"预测完成，指标: {predict_results.metrics}")
            trainer.log_metrics("predict", predict_results.metrics)
            trainer.save_metrics("predict", predict_results.metrics)
            trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)
            debugprint("预测指标和结果已记录和保存")
        
        # Create model card
        debugprint("开始创建模型卡并推送...")
        create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
        debugprint("模型卡创建并推送完成")

        debugprint("run_sft_ssr 函数执行结束 (第一个任务)")
        return
    
    # For subsequent tasks, need to generate and process pseudo-samples
    logger.info_rank0(f"Current task ({cl_finetuning_args.current_task_id}) is a subsequent task, need to generate pseudo-samples for continual learning")
    debugprint("当前是后续任务，开始处理伪样本")
    
    # Initialize SSR method
    debugprint("开始初始化 SSR 方法...")
    ssr = SSR(model_args, data_args, finetuning_args, cl_finetuning_args)
    debugprint("SSR 方法初始化完成")
    
    # Load original dataset
    logger.info_rank0("Loading original dataset for training and pseudo-sample generation")
    debugprint("开始加载原始数据集...")
    orig_dataset_module = get_dataset(
        template=template,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        stage="sft",
        **tokenizer_module
    )
    debugprint("原始数据集加载完成")
    debugprint(f"原始训练集大小: {len(orig_dataset_module.get('train_dataset', []))}")
    
    # Check if dataset loaded successfully
    if "train_dataset" not in orig_dataset_module or len(orig_dataset_module["train_dataset"]) == 0:
        debugprint("错误: 无法加载有效的原始训练数据集")
        raise ValueError(f"Unable to load valid training dataset, please check dataset configuration")
    
    # Generate pseudo-samples using SSR method
    logger.info_rank0("Starting pseudo-sample generation...")
    debugprint("开始生成伪样本...")
    
    # Generate original pseudo-samples
    debugprint("调用 ssr.generate_pseudo_samples()")
    pseudo_samples = ssr.generate_pseudo_samples()
    debugprint(f"生成了 {len(pseudo_samples)} 个原始伪样本")
    
    # Refine pseudo-samples
    logger.info_rank0("Refining pseudo-samples...")
    debugprint("调用 ssr.refine_pseudo_samples()")
    refined_pseudo_samples = ssr.refine_pseudo_samples(pseudo_samples)
    debugprint(f"优化了 {len(refined_pseudo_samples)} 个伪样本")
    
    # Select diverse pseudo-samples
    logger.info_rank0("Selecting diverse pseudo-samples...")
    debugprint("调用 ssr.select_diverse_samples()")
    selected_pseudo_samples = ssr.select_diverse_samples(refined_pseudo_samples)
    debugprint(f"选择了 {len(selected_pseudo_samples)} 个多样化伪样本")
    
    # Save pseudo-samples
    logger.info_rank0("Saving pseudo-samples...")
    debugprint("调用 ssr.save_pseudo_samples()")
    pseudo_dir = ssr.save_pseudo_samples(
        selected_pseudo_samples, 
        cl_finetuning_args.current_task_id,
        cl_finetuning_args.prev_task_id
    )
    debugprint(f"伪样本已保存到目录: {pseudo_dir}")
    
    # Prepare two sets of data parameters and load
    data_args_orig = copy.deepcopy(data_args)  # Original data parameters
    data_args_pseudo = copy.deepcopy(data_args)  # Pseudo-sample data parameters
    debugprint("已创建原始数据参数和伪样本数据参数的副本")
    
    # Fix: Use complete directory path for pseudo-samples
    pseudo_samples_dir = cl_finetuning_args.pseudo_samples_dir
    if not os.path.isabs(pseudo_samples_dir):
        pseudo_samples_dir = os.path.join(os.getcwd(), pseudo_samples_dir)
    debugprint(f"伪样本绝对路径: {pseudo_samples_dir}")
    
    # Use pseudo-sample path from previous task ID
    # prev_pseudo_dir = os.path.join(pseudo_samples_dir, cl_finetuning_args.current_task_id)
    # Correct logic: pseudo_dir already contains the correct path for the current task's saved pseudo samples (including previous ones)
    current_pseudo_dir = pseudo_dir # Use the directory returned by save_pseudo_samples
    debugprint(f"当前任务使用的伪样本目录: {current_pseudo_dir}")
    
    # Set different data directories and dataset names
    # data_args_pseudo.dataset_dir = prev_pseudo_dir
    data_args_pseudo.dataset_dir = current_pseudo_dir # Set the dataset dir to where the combined pseudo samples were saved
    # data_args_pseudo.dataset = [f"pseudo_{cl_finetuning_args.current_task_id}"]
    # Correct logic: The saved file name is pseudo_{task_id}.json, and the dataset name inside dataset_info.json is pseudo_{task_id}
    data_args_pseudo.dataset = [f"pseudo_{cl_finetuning_args.current_task_id}"]
    debugprint(f"设置伪样本数据参数: dataset_dir={data_args_pseudo.dataset_dir}, dataset={data_args_pseudo.dataset}")
    
    # logger.info_rank0(f"Loading pseudo-sample dataset from path: {prev_pseudo_dir}")
    logger.info_rank0(f"Loading pseudo-sample dataset from path: {current_pseudo_dir}")
    
    # Load pseudo-sample dataset
    logger.info_rank0("Loading pseudo-sample dataset...")
    debugprint("开始加载伪样本数据集...")
    try:
        dataset_module_pseudo = get_dataset(
            template=template,
            model_args=model_args,
            data_args=data_args_pseudo,
            training_args=training_args,
            stage="sft",
            **tokenizer_module
        )
        debugprint("伪样本数据集加载完成")
        debugprint(f"伪样本训练集大小: {len(dataset_module_pseudo.get('train_dataset', []))}")
        debugprint(f"伪样本评估集: {list(dataset_module_pseudo.get('eval_dataset', {}).keys()) if isinstance(dataset_module_pseudo.get('eval_dataset'), dict) else bool(dataset_module_pseudo.get('eval_dataset'))}")
    except Exception as e:
        # Fix: Use correct logger method
        logger.warning_rank0(f"Failed to load pseudo-sample dataset: {e}")
        logger.warning_rank0("Will only use original dataset for training")
        debugprint(f"加载伪样本数据集失败: {e}，将只使用原始数据集")
        dataset_module_pseudo = {}
    
    # Merge training sets
    merged_module = {}
    debugprint("开始合并训练集...")
    if "train_dataset" in orig_dataset_module and "train_dataset" in dataset_module_pseudo:
        # Set merge strategy
        merged_data_args = copy.deepcopy(data_args)
        merged_data_args.mix_strategy = "concat"  # Simple concatenation strategy
        
        # Merge training sets
        train_datasets = [
            orig_dataset_module["train_dataset"],
            dataset_module_pseudo["train_dataset"]
        ]
        merged_module["train_dataset"] = merge_dataset(
            train_datasets,
            merged_data_args,
            seed=training_args.seed
        )
        
        debugprint(f"成功合并原始训练集 ({len(orig_dataset_module['train_dataset'])}) 和伪样本训练集 ({len(dataset_module_pseudo['train_dataset'])})，总大小: {len(merged_module['train_dataset'])}")
        logger.info_rank0(f"Successfully merged original dataset ({len(orig_dataset_module['train_dataset'])}) and pseudo-sample dataset ({len(dataset_module_pseudo['train_dataset'])}), total samples: {len(merged_module['train_dataset'])}")
    elif "train_dataset" in orig_dataset_module:
        merged_module["train_dataset"] = orig_dataset_module["train_dataset"]
        debugprint(f"只使用原始训练集，大小: {len(orig_dataset_module['train_dataset'])}")
        logger.warning_rank0(f"Only using original dataset ({len(orig_dataset_module['train_dataset'])}) for training")
    elif "train_dataset" in dataset_module_pseudo:
        merged_module["train_dataset"] = dataset_module_pseudo["train_dataset"]
        debugprint(f"只使用伪样本训练集，大小: {len(dataset_module_pseudo['train_dataset'])}")
        logger.warning_rank0(f"Only using pseudo-sample dataset ({len(dataset_module_pseudo['train_dataset'])}) for training")
    else:
        debugprint("错误: 原始数据集和伪样本数据集均无效")
        raise ValueError("Unable to load valid training dataset, please check dataset configuration")
        
    # Merge validation sets (if exist)
    eval_dataset = {}
    debugprint("开始合并评估集...")
    if "eval_dataset" in orig_dataset_module:
        if isinstance(orig_dataset_module["eval_dataset"], dict):
            eval_dataset.update(orig_dataset_module["eval_dataset"])
            debugprint(f"从原始数据集合并评估集 (字典类型): {list(orig_dataset_module['eval_dataset'].keys())}")
        else:
            eval_dataset["orig"] = orig_dataset_module["eval_dataset"]
            debugprint("从原始数据集合并评估集 (列表类型，键为 'orig')")

    if "eval_dataset" in dataset_module_pseudo:
        if isinstance(dataset_module_pseudo["eval_dataset"], dict):
            eval_dataset.update(dataset_module_pseudo["eval_dataset"])
            debugprint(f"从伪样本数据集合并评估集 (字典类型): {list(dataset_module_pseudo['eval_dataset'].keys())}")
        else:
            eval_dataset["pseudo"] = dataset_module_pseudo["eval_dataset"]
            debugprint("从伪样本数据集合并评估集 (列表类型，键为 'pseudo')")

    if eval_dataset:
        merged_module["eval_dataset"] = eval_dataset
        debugprint(f"合并后的评估集键: {list(eval_dataset.keys())}")
    else:
        debugprint("没有可用的评估集")
    
    # Load model with merged dataset
    debugprint("开始加载模型...")
    model = load_model(
        tokenizer=tokenizer,
        model_args=model_args,
        finetuning_args=finetuning_args,
        is_trainable=training_args.do_train,
    )
    debugprint("模型加载完成")
    
    # Normal training process
    debugprint("创建 Data Collator...")
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )
    debugprint("Data Collator 创建完成")
    
    # Training arguments configuration
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False
    
    # Metric configuration
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
        debugprint("使用 ComputeSimilarity 作为评估指标")
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor
        debugprint("使用 ComputeAccuracy 作为评估指标")
    else:
        debugprint("未配置评估指标")
    
    # Generation parameters configuration
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()
    debugprint("生成参数配置完成")
    
    # Initialize trainer
    debugprint("开始初始化 SSRTrainer...")
    trainer = SSRTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        cl_finetuning_args=cl_finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs=gen_kwargs,
        **merged_module,
        **tokenizer_module,
        **metric_module,
    )
    debugprint("SSRTrainer 初始化完成")
    
    # Start training
    if training_args.do_train:
        debugprint("开始训练...")
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        debugprint("训练完成")
        trainer.save_model()
        debugprint("模型已保存")
        
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                merged_module["train_dataset"], train_result.metrics, stage="sft"
            )
            debugprint(f"计算有效 tokens/sec: {train_result.metrics['effective_tokens_per_sec']}")
            
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        debugprint("训练指标和状态已保存")
        
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            debugprint("开始绘制损失曲线...")
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])
            debugprint("损失曲线绘制完成")
    
    # Prediction configuration
    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"
        debugprint("设置 tokenizer padding_side 为 left 以进行生成预测")
    
    # Evaluation
    if training_args.do_eval:
        debugprint("开始评估...")
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        debugprint(f"评估完成，指标: {metrics}")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        debugprint("评估指标已记录和保存")
    
    # Prediction
    if training_args.do_predict:
        logger.warning_rank0("Batch generation may be slow. Consider using `scripts/vllm_infer.py` instead.")
        debugprint("开始预测...")
        # Use merged_module.get("eval_dataset", {}) to handle cases where eval_dataset might be empty
        eval_pred_dataset = merged_module.get("eval_dataset", None)
        if not eval_pred_dataset:
             logger.warning_rank0("No evaluation dataset found for prediction.")
             debugprint("没有找到用于预测的评估数据集")
        else:
            predict_results = trainer.predict(eval_pred_dataset, metric_key_prefix="predict", **gen_kwargs)
            debugprint(f"预测完成，指标: {predict_results.metrics}")
            trainer.log_metrics("predict", predict_results.metrics)
            trainer.save_metrics("predict", predict_results.metrics)
            trainer.save_predictions(eval_pred_dataset, predict_results, generating_args.skip_special_tokens)
            debugprint("预测指标和结果已记录和保存")
    
    # Create model card
    debugprint("开始创建模型卡并推送...")
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
    debugprint("模型卡创建并推送完成")

    debugprint("run_sft_ssr 函数执行结束 (后续任务)")
