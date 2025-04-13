# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import PreTrainedModel

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.extras import logging
from llamafactory.extras.constants import V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from llamafactory.extras.misc import infer_optim_dtype
from llamafactory.extras.packages import is_ray_available
from llamafactory.hparams import get_infer_args, get_ray_args, get_train_args, read_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.callbacks import LogCallback, PissaConvertCallback, ReporterCallback
from llamafactory.train.dpo import run_dpo
from llamafactory.train.kto import run_kto
from llamafactory.train.ppo import run_ppo
from llamafactory.train.pt import run_pt
from llamafactory.train.rm import run_rm
from llamafactory.train.sft import run_sft
from llamafactory.train.trainer_utils import get_ray_trainer, get_swanlab_callback
from easycl.hparams.parser import get_cl_eval_args
from easycl.hparams.parser import get_train_args
from ..cl.ewc import run_sft_ewc
from ..cl.lwf import run_sft_lwf
from ..cl.replay import run_sft_replay
from ..cl.lamol import run_sft_lamol
from ..cl.olora import run_sft_olora
from ..cl.gem import run_sft_gem
from ..cl.ilora import run_sft_ilora
from ..cl.moe import run_sft_moelora
from ..cl.abscl import run_sft_abscl
from ..cl.dynamic_conpet import run_sft_dynamic_conpet
from ..cl.clmoe import run_sft_clitmoe
from ..cl.ssr import run_sft_ssr
from ..cl.pseudo_replay import run_sft_pseudo_replay

if is_ray_available():
    from ray.train.huggingface.transformers import RayTrainReportCallback


if TYPE_CHECKING:
    from transformers import TrainerCallback


logger = logging.get_logger(__name__)


def _training_function(config: Dict[str, Any]) -> None:
    args = config.get("args")
    callbacks: List[Any] = config.get("callbacks")
    model_args, data_args, training_args, finetuning_args,cl_finetuning_args, generating_args = get_train_args(args)

    callbacks.append(LogCallback())
    if finetuning_args.pissa_convert:
        callbacks.append(PissaConvertCallback())

    if finetuning_args.use_swanlab:
        callbacks.append(get_swanlab_callback(finetuning_args))

    # 添加EWC参数验证
    if cl_finetuning_args.use_ewc:
        if not cl_finetuning_args.previous_task_data:
            logger.warning("EWC is enabled but no previous task data provided. EWC will be disabled.")
            cl_finetuning_args.use_ewc = False

    # 添加LWF参数验证
    if cl_finetuning_args.use_lwf:
        if not cl_finetuning_args.previous_task_model:
            logger.warning("LWF is enabled but no previous task model provided. LWF will be disabled.")
            cl_finetuning_args.use_lwf = False

    # 添加Replay参数验证
    if cl_finetuning_args.use_replay:
        if not cl_finetuning_args.previous_task_dataset:
            logger.warning("Replay is enabled but no previous task dataset provided. Replay buffer will be initialized during training.")

    # 添加O-LoRA参数验证
    if cl_finetuning_args.use_olora:
        if not cl_finetuning_args.current_task_id:
            logger.warning(
                "No current_task_id provided for O-LoRA. "
                "Will try to extract from the output directory name."
            )
            try:
                cl_finetuning_args.current_task_id = os.path.basename(training_args.output_dir)
            except:
                logger.warning("Could not determine current_task_id. O-LoRA will be disabled.")
                cl_finetuning_args.use_olora = False

    # 添加GEM参数验证
    if cl_finetuning_args.use_gem:
        if finetuning_args.finetuning_type != "lora" and finetuning_args.finetuning_type != "full":
            logger.warning("GEM is currently only supported with LoRA or full parameter fine-tuning. GEM will be disabled.")
            cl_finetuning_args.use_gem = False

    # 添加MOELoRA参数验证
    if cl_finetuning_args.use_moe:
        if finetuning_args.finetuning_type != "lora":
            logger.warning("MOELoRA is only supported with LoRA fine-tuning. MOELoRA will be disabled.")
            cl_finetuning_args.use_moe = False
            
        if not cl_finetuning_args.current_task_id:
            logger.warning(
                "No current_task_id provided for MOELoRA. "
                "Will try to extract from the output directory name."
            )
            try:
                cl_finetuning_args.current_task_id = os.path.basename(training_args.output_dir)
            except:
                logger.warning("Could not determine current_task_id. MOELoRA will use default task ID 0.")
                cl_finetuning_args.current_task_id = "0"

    # 添加Dynamic ConPET参数验证
    if cl_finetuning_args.use_dynamic_conpet:
        if not cl_finetuning_args.current_task_id:
            logger.warning(
                "No current_task_id provided for Dynamic ConPET. "
                "This may affect proper functioning of task-specific adapters."
            )
        if finetuning_args.finetuning_type != "lora":
            logger.warning("Dynamic ConPET is only supported with LoRA fine-tuning. Dynamic ConPET will be disabled.")
            cl_finetuning_args.use_dynamic_conpet = False

    # 添加SSR参数验证
    if cl_finetuning_args.use_ssr:
        if not cl_finetuning_args.current_task_id:
            logger.warning(
                "No current_task_id provided for SSR. "
                "Will try to extract from the output directory name."
            )
            try:
                cl_finetuning_args.current_task_id = os.path.basename(training_args.output_dir)
            except:
                logger.warning("Could not determine current_task_id. SSR will use default task ID 'current_task'.")
                cl_finetuning_args.current_task_id = "current_task"
        
        if not os.path.exists(cl_finetuning_args.pseudo_samples_dir):
            logger.info_rank0(f"Creating pseudo samples directory: {cl_finetuning_args.pseudo_samples_dir}")

    # 添加ABSCL参数验证
    if cl_finetuning_args.use_abscl:
        if finetuning_args.finetuning_type != "lora":
            logger.warning("ABSCL is only supported with LoRA fine-tuning. ABSCL will be disabled.")
            cl_finetuning_args.use_abscl = False
        
        if not cl_finetuning_args.current_task_id:
            logger.warning(
                "No current_task_id provided for ABSCL. "
                "Will try to extract from the output directory name."
            )
            try:
                cl_finetuning_args.current_task_id = os.path.basename(training_args.output_dir)
            except:
                logger.warning("Could not determine current_task_id. ABSCL will use default domain ID.")
                cl_finetuning_args.current_task_id = "default_domain"

    # 添加CLIT-MoE参数验证
    if cl_finetuning_args.use_clit_moe:
        if finetuning_args.finetuning_type != "lora":
            logger.error("CLIT-MoE is only supported with LoRA fine-tuning. CLIT-MoE will be disabled.")
            cl_finetuning_args.use_clit_moe = False
        elif cl_finetuning_args.expert_num <= 1:
            logger.error("expert_num must be greater than 1 for CLIT-MoE. CLIT-MoE will be disabled.")
            cl_finetuning_args.use_clit_moe = False

    # 添加Pseudo Replay参数验证
    if cl_finetuning_args.use_pseudo_replay:
        if not cl_finetuning_args.current_task_id:
            logger.warning(
                "未提供current_task_id参数。"
                "尝试从输出目录名称中提取。"
            )
            try:
                cl_finetuning_args.current_task_id = os.path.basename(training_args.output_dir)
            except:
                logger.warning("无法确定current_task_id。将使用默认任务ID 'current_task'。")
                cl_finetuning_args.current_task_id = "current_task"
                
        if not os.path.exists(cl_finetuning_args.pseudo_samples_dir):
            os.makedirs(cl_finetuning_args.pseudo_samples_dir, exist_ok=True)
            logger.info_rank0(f"创建伪样本目录: {cl_finetuning_args.pseudo_samples_dir}")

    callbacks.append(ReporterCallback(model_args, data_args, finetuning_args, generating_args))

    if finetuning_args.stage == "pt":
        #easycl不支持pt，返回valueerror
        raise ValueError("PT is not supported in EasyCL.")
        
    elif finetuning_args.stage == "sft":
        if cl_finetuning_args.use_ewc:
            run_sft_ewc(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                finetuning_args=finetuning_args,
                generating_args=generating_args,
                cl_finetuning_args=cl_finetuning_args,
                callbacks=callbacks,
            )
        elif cl_finetuning_args.use_lwf:
            run_sft_lwf(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                finetuning_args=finetuning_args,
                generating_args=generating_args,
                cl_finetuning_args=cl_finetuning_args,
                callbacks=callbacks,
            )
        elif cl_finetuning_args.use_pseudo_replay:
            run_sft_pseudo_replay(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                finetuning_args=finetuning_args,
                generating_args=generating_args,
                cl_finetuning_args=cl_finetuning_args,
                callbacks=callbacks,
            )
            
        elif cl_finetuning_args.use_replay:
            run_sft_replay(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                finetuning_args=finetuning_args,
                generating_args=generating_args,
                cl_finetuning_args=cl_finetuning_args,
                callbacks=callbacks,
            )
        elif cl_finetuning_args.use_lamol:
            run_sft_lamol(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                finetuning_args=finetuning_args,
                generating_args=generating_args,
                cl_finetuning_args=cl_finetuning_args,
                callbacks=callbacks,
            )
        elif cl_finetuning_args.use_olora:
            run_sft_olora(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                finetuning_args=finetuning_args,
                generating_args=generating_args,
                cl_finetuning_args=cl_finetuning_args,
                callbacks=callbacks,
            )
        elif cl_finetuning_args.use_gem:
            run_sft_gem(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                finetuning_args=finetuning_args,
                generating_args=generating_args,
                cl_finetuning_args=cl_finetuning_args,
                callbacks=callbacks,
            )
        elif cl_finetuning_args.use_ilora:
            run_sft_ilora(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                finetuning_args=finetuning_args,
                generating_args=generating_args,
                cl_finetuning_args=cl_finetuning_args,
                callbacks=callbacks,
            )
        elif cl_finetuning_args.use_moe:
            run_sft_moelora(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                finetuning_args=finetuning_args,
                generating_args=generating_args,
                cl_finetuning_args=cl_finetuning_args,
                callbacks=callbacks,
            )
        elif cl_finetuning_args.use_dynamic_conpet:
            run_sft_dynamic_conpet(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                finetuning_args=finetuning_args,
                generating_args=generating_args,
                cl_finetuning_args=cl_finetuning_args,
                callbacks=callbacks,
            )
        elif cl_finetuning_args.use_abscl:
            run_sft_abscl(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                finetuning_args=finetuning_args,
                generating_args=generating_args,
                cl_finetuning_args=cl_finetuning_args,
                callbacks=callbacks,
            )
        elif cl_finetuning_args.use_clit_moe:
            run_sft_clitmoe(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                finetuning_args=finetuning_args,
                generating_args=generating_args,
                cl_finetuning_args=cl_finetuning_args,
                callbacks=callbacks,
            )
        elif cl_finetuning_args.use_ssr:
            run_sft_ssr(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                finetuning_args=finetuning_args,
                generating_args=generating_args,
                cl_finetuning_args=cl_finetuning_args,
                callbacks=callbacks,
            )
        else:
            run_sft(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                finetuning_args=finetuning_args,
                generating_args=generating_args,
                callbacks=callbacks,
            )
    elif finetuning_args.stage == "rm":
        #easycl不支持rm，返回valueerror
        raise ValueError("RM is not supported in EasyCL.")
        # run_rm(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "ppo":
        #easycl不支持ppo，返回valueerror
        raise ValueError("PPO is not supported in EasyCL.")
        # run_ppo(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "kto":
        #easycl不支持kto，返回valueerror
        raise ValueError("KTO is not supported in EasyCL.")
        # run_kto(model_args, data_args, training_args, finetuning_args, callbacks)
    else:
        raise ValueError(f"Unknown task: {finetuning_args.stage}.")

'''
def run_cl_eval(args: Optional[Dict[str, Any]] = None) -> None:
    """运行持续学习评估"""
    from ..eval.cl_runner import CLEvaluationRunner
    from ..hparams import get_eval_args
    
    model_args, data_args, eval_args, finetuning_args = get_eval_args(args)
    
    runner = CLEvaluationRunner(args)
    runner.setup_evaluator(args)
    
    if eval_args.eval_sequence:
        results = runner.evaluate_sequence(
            checkpoints=eval_args.checkpoints,
            tasks=eval_args.tasks,
            base_save_dir=eval_args.save_dir
        )
    else:
        results = runner.evaluate_checkpoint(
            checkpoint_path=model_args.model_name_or_path,
            tasks=eval_args.tasks,
            save_prefix="single_evaluation"
        )
        
    if eval_args.generate_report:
        report_path = os.path.join(eval_args.save_dir, "evaluation_report.md")
        runner.generate_report(results, report_path)
'''

def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List["TrainerCallback"]] = None) -> None:
    args = read_args(args)
    
    # Check if this is a CL evaluation request
    if isinstance(args, dict) and "cl_eval" in args:
        from ..cl.evaluation_handler import handle_cl_evaluation
        logger.info("Running continual learning evaluation from CLI")
        handle_cl_evaluation(args)
        return
        
    ray_args = get_ray_args(args)
    callbacks = callbacks or []
    if ray_args.use_ray:
        callbacks.append(RayTrainReportCallback())
        trainer = get_ray_trainer(
            training_function=_training_function,
            train_loop_config={"args": args, "callbacks": callbacks},
            ray_args=ray_args,
        )
        trainer.fit()
    else:
        _training_function(config={"args": args, "callbacks": callbacks})


def export_model(args: Optional[Dict[str, Any]] = None) -> None:
    model_args, data_args, finetuning_args, _ = get_infer_args(args)

    if model_args.export_dir is None:
        raise ValueError("Please specify `export_dir` to save model.")

    if model_args.adapter_name_or_path is not None and model_args.export_quantization_bit is not None:
        raise ValueError("Please merge adapters before quantizing the model.")

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args)  # must after fixing tokenizer to resize vocab

    if getattr(model, "quantization_method", None) is not None and model_args.adapter_name_or_path is not None:
        raise ValueError("Cannot merge adapters to a quantized model.")

    if not isinstance(model, PreTrainedModel):
        raise ValueError("The model is not a `PreTrainedModel`, export aborted.")

    if getattr(model, "quantization_method", None) is not None:  # quantized model adopts float16 type
        setattr(model.config, "torch_dtype", torch.float16)
    else:
        if model_args.infer_dtype == "auto":
            output_dtype = getattr(model.config, "torch_dtype", torch.float32)
            if output_dtype == torch.float32:  # if infer_dtype is auto, try using half precision first
                output_dtype = infer_optim_dtype(torch.bfloat16)
        else:
            output_dtype = getattr(torch, model_args.infer_dtype)

        setattr(model.config, "torch_dtype", output_dtype)
        model = model.to(output_dtype)
        logger.info_rank0(f"Convert model dtype to: {output_dtype}.")

    model.save_pretrained(
        save_directory=model_args.export_dir,
        max_shard_size=f"{model_args.export_size}GB",
        safe_serialization=(not model_args.export_legacy_format),
    )
    if model_args.export_hub_model_id is not None:
        model.push_to_hub(
            model_args.export_hub_model_id,
            token=model_args.hf_hub_token,
            max_shard_size=f"{model_args.export_size}GB",
            safe_serialization=(not model_args.export_legacy_format),
        )

    if finetuning_args.stage == "rm":
        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        if os.path.exists(os.path.join(vhead_path, V_HEAD_SAFE_WEIGHTS_NAME)):
            shutil.copy(
                os.path.join(vhead_path, V_HEAD_SAFE_WEIGHTS_NAME),
                os.path.join(model_args.export_dir, V_HEAD_SAFE_WEIGHTS_NAME),
            )
            logger.info_rank0(f"Copied valuehead to {model_args.export_dir}.")
        elif os.path.exists(os.path.join(vhead_path, V_HEAD_WEIGHTS_NAME)):
            shutil.copy(
                os.path.join(vhead_path, V_HEAD_WEIGHTS_NAME),
                os.path.join(model_args.export_dir, V_HEAD_WEIGHTS_NAME),
            )
            logger.info_rank0(f"Copied valuehead to {model_args.export_dir}.")

    try:
        tokenizer.padding_side = "left"  # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"
        tokenizer.save_pretrained(model_args.export_dir)
        if model_args.export_hub_model_id is not None:
            tokenizer.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token)

        if processor is not None:
            processor.save_pretrained(model_args.export_dir)
            if model_args.export_hub_model_id is not None:
                processor.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token)

    except Exception as e:
        logger.warning_rank0(f"Cannot save tokenizer, please copy the files manually: {e}.")

    with open(os.path.join(model_args.export_dir, "Modelfile"), "w", encoding="utf-8") as f:
        f.write(template.get_ollama_modelfile(tokenizer))
        logger.info_rank0(f"Saved ollama modelfile to {model_args.export_dir}.")


