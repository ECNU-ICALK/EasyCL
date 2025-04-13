from typing import TYPE_CHECKING, Optional
import os
from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.logging import get_logger
from llamafactory.extras.misc import calculate_tps, get_logits_processor
from llamafactory.extras.ploting import plot_loss
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.trainer_utils import create_modelcard_and_push
from llamafactory.train.sft.metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
#从src/llamafactory/cl/olora/olora.py导入OLORA
from .olora import OLoRA
from .olora_trainer import OLoRATrainer
from llamafactory.hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments
from easycl.hparams import CLFinetuningArguments


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback


logger = get_logger(__name__)


def run_sft_olora(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    cl_finetuning_args: "CLFinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    """
    The main function to run O-LoRA fine-tuning.
    """
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

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

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False  # important for multimodal dataset

    # Metric utils
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Validate and initialize O-LoRA settings
    if cl_finetuning_args.use_olora:
        if not cl_finetuning_args.current_task_id:
            logger.warning(
                "No current_task_id provided for O-LoRA. "
                "Will try to extract from the output directory name."
            )
            # Try to extract task ID from output directory name
            try:
                cl_finetuning_args.current_task_id = os.path.basename(training_args.output_dir)
            except:
                raise ValueError(
                    "Could not determine current_task_id. "
                    "Please provide it explicitly for O-LoRA."
                )
                
        # Ensure O-LoRA history path exists
        os.makedirs(cl_finetuning_args.olora_history_path, exist_ok=True)
        
        logger.info("O-LoRA is enabled with following parameters:")
        logger.info(f"- Current task ID: {cl_finetuning_args.current_task_id}")
        logger.info(f"- Orthogonal lambda: {cl_finetuning_args.orthogonal_lambda}")
        logger.info(f"- L2 lambda: {cl_finetuning_args.l2_lambda}")
        logger.info(f"- History path: {cl_finetuning_args.olora_history_path}")

    # Initialize our Trainer
    trainer = OLoRATrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        cl_finetuning_args=cl_finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    # After loading the model, if O-LoRA is enabled, initialize and set up
    if cl_finetuning_args.use_olora:
        from .olora import OLoRA
        
        # Create O-LoRA instance
        olora = OLoRA(
            model=model,
            orthogonal_lambda=cl_finetuning_args.orthogonal_lambda,
            l2_lambda=cl_finetuning_args.l2_lambda,
            olora_history_path=cl_finetuning_args.olora_history_path,
            model_output_dir=training_args.output_dir,
            device=training_args.device.type if hasattr(training_args.device, "type") else training_args.device,
            prev_task_id=cl_finetuning_args.prev_task_id
        )
        
        # Load previous task's adapter parameters
        if olora.load_prev_adapter(cl_finetuning_args.prev_task_id):
            logger.info(f"Successfully loaded previous task adapter: {cl_finetuning_args.prev_task_id}")
        
        # Important: Set up adapters to enable orthogonal constraints
        current_adapter_name = "current"  # We use fixed name "current" for current task
        if olora.setup_adapters(current_adapter_name):
            logger.info(f"Successfully set up O-LoRA adapters with current adapter: {current_adapter_name}")
        else:
            logger.warning("Failed to set up O-LoRA adapters, orthogonal loss may not be computed correctly")
        
        # Attach O-LoRA instance to trainer for later use
        trainer.olora = olora

    # Training
    if training_args.do_train:
        # If current_task_name not specified, extract from dataset path
        if not cl_finetuning_args.current_task_id and data_args.dataset:
            # Get the filename of the last dataset
            last_dataset = data_args.dataset[-1]
            # Extract task name from filename and convert to uppercase
            task_name = os.path.splitext(os.path.basename(last_dataset))[0].upper()
            cl_finetuning_args.current_task_id = task_name
            logger.info(f"Extracted current task name: {task_name}")
        
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_second"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        # Add O-LoRA loss printing
        if cl_finetuning_args.use_olora:
            if hasattr(trainer, "olora"):
                orthogonal_loss = trainer.olora.compute_orthogonal_loss().item()
                l2_loss = trainer.olora.compute_l2_loss().item()
                train_result.metrics["orthogonal_loss"] = orthogonal_loss
                train_result.metrics["l2_loss"] = l2_loss
                logger.info(f"Final O-LoRA losses:")
                logger.info(f"- Orthogonal loss: {orthogonal_loss:.4f}")
                logger.info(f"- L2 loss: {l2_loss:.4f}")
            else:
                logger.warning("O-LoRA was enabled but trainer.olora is not found!")

        if cl_finetuning_args.use_olora:
            # Save final merged adapter after training
            trainer.olora.save_merged_adapter(cl_finetuning_args.current_task_id)
            
            # Record O-LoRA related metrics
            if "orthogonal_loss" in train_result.metrics:
                logger.info(f"Final orthogonal loss: {train_result.metrics['orthogonal_loss']:.4f}")
            if "l2_loss" in train_result.metrics:
                logger.info(f"Final L2 loss: {train_result.metrics['l2_loss']:.4f}")

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
