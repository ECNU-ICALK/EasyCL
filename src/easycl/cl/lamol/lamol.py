import torch
from typing import Dict, List, Optional, Tuple, Union
import logging
from llamafactory.hparams import FinetuningArguments, ModelArguments, DataArguments
from easycl.hparams.cl_finetuning_args import CLFinetuningArguments
import json
import os
import glob
import datetime
import copy
import random
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    GenerationConfig
)
from llamafactory.model import load_model, load_tokenizer

logger = logging.getLogger(__name__)

class LAMOLGenerator:
    """
    LAMOL (pseudo-replay style) generator.
    Uses the previous task's model to generate pseudo samples based on 1-shot instruction prompts from the current task.
    """
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        finetuning_args: FinetuningArguments,
        cl_finetuning_args: CLFinetuningArguments,
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.finetuning_args = finetuning_args
        self.cl_finetuning_args = cl_finetuning_args

        # LAMOL specific arguments
        self.use_lamol = cl_finetuning_args.use_lamol
        self.lamol_show_gen = cl_finetuning_args.lamol_show_gen
        self.lamol_num_samples_per_task = cl_finetuning_args.lamol_num_samples_per_task
        self.lamol_generation_temperature = cl_finetuning_args.lamol_generation_temperature
        self.lamol_samples_dir = cl_finetuning_args.lamol_samples_dir
        # self.lamol_base_model_path = finetuning_args.lamol_base_model_path # Removed
        # Task IDs are needed for saving structure
        self.current_task_id = cl_finetuning_args.current_task_id
        self.prev_task_id = cl_finetuning_args.prev_task_id # Used implicitly by previous_task_model
        self.previous_task_model_path = cl_finetuning_args.previous_task_model # Reuse existing param

        if not self.current_task_id:
            logger.warning_rank0("`current_task_id` not provided for LAMOL, saving might be inconsistent.")

    def setup_previous_model(self) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
        """
        Load the previous task's model for pseudo sample generation.
        Uses finetuning_args.previous_task_model path.
        """
        if not self.previous_task_model_path:
            logger.warning_rank0("`previous_task_model` path not provided. Skipping LAMOL pseudo sample generation.")
            return None, None

        prev_model_args = copy.deepcopy(self.model_args)
        temp_finetuning_args = copy.deepcopy(self.finetuning_args)

        model_path_to_load = self.previous_task_model_path
        adapter_path_to_load = None

        # Determine how to load the previous model based on finetuning type
        if temp_finetuning_args.finetuning_type == "lora":
            logger.info_rank0(f"Loading previous task LoRA adapter from: {model_path_to_load}")
            # Keep the original base model path, just change the adapter
            adapter_path_to_load = [model_path_to_load]
            prev_model_args.adapter_name_or_path = adapter_path_to_load 
            # Important: Don't create a new adapter when loading for generation
            temp_finetuning_args.create_new_adapter = False
            # Use the base model path from the original model_args for LoRA
            model_identity = f"{prev_model_args.model_name_or_path} with adapter {model_path_to_load}"
        elif temp_finetuning_args.finetuning_type == "full":
            logger.info_rank0(f"Loading previous task full model from: {model_path_to_load}")
            # Load the full model from the specified path
            prev_model_args.model_name_or_path = model_path_to_load
            prev_model_args.adapter_name_or_path = None # Ensure no adapters are loaded
            model_identity = model_path_to_load
        else: # freeze tuning behaves like full tuning for loading previous state
             logger.info_rank0(f"Loading previous task model (freeze/full) from: {model_path_to_load}")
             prev_model_args.model_name_or_path = model_path_to_load
             prev_model_args.adapter_name_or_path = None
             model_identity = model_path_to_load

        # Load tokenizer associated with the previous model setup
        try:
            tokenizer_module = load_tokenizer(prev_model_args)
            tokenizer = tokenizer_module["tokenizer"]
        except Exception as e:
            logger.error(f"Failed to load tokenizer for previous model setup ({model_identity}): {e}")
            return None, None

        # Load the previous model itself
        try:
            prev_model = load_model(
                tokenizer,
                prev_model_args,
                temp_finetuning_args, # Use potentially modified finetuning args (e.g., create_new_adapter=False)
                is_trainable=False # Model is only for generation
            )
        except Exception as e:
             logger.error(f"Failed to load previous model ({model_identity}): {e}")
             return None, None

        logger.info_rank0(f"Loaded previous task model for LAMOL generation: {model_identity}")
        return prev_model, tokenizer

    def get_few_shot_example(self, dataset):
        """
        Get a single random example from the dataset for 1-shot prompting.
        """
        if not dataset or len(dataset) == 0:
             logger.warning_rank0("Dataset is empty, cannot get few-shot example.")
             return None

        index = random.randint(0, len(dataset) - 1)
        return dataset[index]

    def construct_prompt(self, example):
        """
        Construct the 1-shot prompt using only the instruction part.
        Expected format: "{instruction}\n\nInstruction:"
        """
        if not example:
            logger.warning_rank0("Cannot construct prompt from empty example.")
            return None

        instruction_text = ""
        # Try common instruction keys
        if "instruction" in example and example["instruction"]:
            instruction_text = example["instruction"]
        elif "query" in example and example["query"]:
            instruction_text = example["query"]
        elif "question" in example and example["question"]:
            instruction_text = example["question"]
        elif "messages" in example and isinstance(example["messages"], list):
            # Find the first user message
            for msg in example["messages"]:
                if msg.get("role") == "user" and msg.get("content"):
                    instruction_text = msg["content"]
                    break
        else:
             # Fallback: try the first non-empty string value
             for value in example.values():
                 if isinstance(value, str) and value:
                     instruction_text = value
                     break

        if not instruction_text:
            logger.warning_rank0(f"Could not extract instruction from example: {example}")
            return None

        # Format the prompt
        prompt = f"{instruction_text}\n\nInstruction:"
        return prompt

    def generate_pseudo_samples(self, dataset):
        """
        Generate pseudo samples using the *previous task's model* and 1-shot instruction prompts from the current task's data.
        """
        num_samples = self.lamol_num_samples_per_task
        if not dataset or len(dataset) == 0:
            logger.warning_rank0("Input dataset is empty. Skipping pseudo sample generation.")
            return []

        # Load the PREVIOUS task's model
        prev_model, tokenizer = self.setup_previous_model()

        # If loading failed, return empty list
        if prev_model is None or tokenizer is None:
             return []

        # Set up generation config
        generation_config = GenerationConfig(
            temperature=self.lamol_generation_temperature,
            do_sample=True, # Important for diversity
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=128 # Limit generated length
        )

        generated_texts = []
        logger.info_rank0(f"Starting generation of {num_samples} LAMOL pseudo samples using previous task model...")

        # Simplified generation loop
        for i in range(num_samples):
            if (i + 1) % 50 == 0:
                logger.info_rank0(f"Generated {i+1}/{num_samples} samples...")

            # Get a new 1-shot example for each generation from the CURRENT task's data
            few_shot_example = self.get_few_shot_example(dataset)
            if not few_shot_example:
                logger.warning_rank0("Failed to get few-shot example, stopping generation.")
                break

            prompt = self.construct_prompt(few_shot_example)
            if not prompt:
                logger.warning_rank0("Failed to construct prompt, skipping this sample.")
                continue

            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(prev_model.device)
            attention_mask = inputs["attention_mask"].to(prev_model.device)

            try:
                with torch.no_grad():
                    outputs = prev_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        generation_config=generation_config,
                        num_return_sequences=1,
                        return_dict_in_generate=True
                    )

                # Decode only the newly generated tokens
                new_tokens = outputs.sequences[0][input_ids.shape[1]:]
                generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                if generated_text:
                    original_instruction = prompt.split("\n\nInstruction:")[0]
                    generated_texts.append({
                        "instruction": original_instruction,
                        "generated_output": generated_text
                    })

            except Exception as e:
                logger.warning_rank0(f"Error during generation for sample {i+1}: {e}")
                continue

        # Release GPU memory
        del prev_model
        torch.cuda.empty_cache()

        pseudo_samples = self.parse_generated_texts(generated_texts)

        logger.info_rank0(f"Successfully generated {len(pseudo_samples)} valid LAMOL pseudo samples using previous task model.")
        return pseudo_samples

    def parse_generated_texts(self, generated_data: List[Dict[str, str]]):
        """
        Parse the generated instruction/output pairs into structured pseudo samples.
        Applies the lamol_show_gen prefix if enabled.
        """
        parsed_samples = []
        gen_prefix = "This is a generated sample. " if self.lamol_show_gen else ""

        for item in generated_data:
            instruction = item["instruction"]
            output = item["generated_output"] # The generated part is the 'output' in Alpaca format

            final_instruction = gen_prefix + instruction

            sample = {
                "instruction": final_instruction,
                "input": "", # No input in this generation style
                "output": output
            }
            parsed_samples.append(sample)

        # Deduplication
        unique_samples = []
        seen = set()
        for sample in parsed_samples:
            sample_str = json.dumps(sample, sort_keys=True)
            if sample_str not in seen:
                seen.add(sample_str)
                unique_samples.append(sample)

        logger.info_rank0(f"Parsed {len(generated_data)} generated texts into {len(unique_samples)} unique LAMOL samples.")
        return unique_samples

    def save_pseudo_samples(self, samples):
        """
        Save pseudo samples to the specified directory structure.
        """
        if not self.current_task_id:
             logger.error("Cannot save LAMOL samples without `current_task_id`.")
             return None

        output_dir = self.lamol_samples_dir
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)

        os.makedirs(output_dir, exist_ok=True)

        task_dir = os.path.join(output_dir, self.current_task_id)
        os.makedirs(task_dir, exist_ok=True)

        all_samples = samples

        pseudo_file_name = f"lamol_pseudo_{self.current_task_id}.json"
        pseudo_file_path = os.path.join(task_dir, pseudo_file_name)

        with open(pseudo_file_path, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)

        dataset_format = self.get_dataset_format()
        dataset_key = f"lamol_pseudo_{self.current_task_id}"
        dataset_info = {
            dataset_key: {
                "file_name": pseudo_file_name,
                **dataset_format
            }
        }

        dataset_info_path = os.path.join(task_dir, "dataset_info.json")
        with open(dataset_info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)

        logger.info_rank0(f"Saved {len(all_samples)} LAMOL pseudo samples for task {self.current_task_id} to {pseudo_file_path}")
        logger.info_rank0(f"Created dataset info at {dataset_info_path}")

        return task_dir

    def get_dataset_format(self):
        """
        Attempt to get format information from the original dataset specified in data_args.
        Defaults to Alpaca format if information cannot be retrieved.
        """
        try:
            from llamafactory.data.parser import _parse_dataset_info

            dataset_name = None
            if isinstance(self.data_args.dataset, list) and self.data_args.dataset:
                dataset_name = self.data_args.dataset[0]
            elif isinstance(self.data_args.dataset, str):
                 dataset_name = self.data_args.dataset

            if not dataset_name:
                 logger.warning_rank0("No dataset specified in data_args, defaulting format to Alpaca.")
                 return {"formatting": "alpaca", "split": "train"}

            dataset_info_path = os.path.join(self.data_args.dataset_dir, "dataset_info.json")
            if os.path.exists(dataset_info_path):
                 with open(dataset_info_path, 'r', encoding='utf-8') as f:
                      all_dataset_info = json.load(f)
                      if dataset_name in all_dataset_info:
                           src_info = _parse_dataset_info(all_dataset_info[dataset_name])
                           format_info = {"formatting": src_info.get("formatting", "alpaca")}
                           if "split" in src_info: format_info["split"] = src_info["split"]
                           if "columns" in src_info: format_info["columns"] = src_info["columns"]
                           if "tags" in src_info: format_info["tags"] = src_info["tags"]
                           logger.info_rank0(f"Using dataset format from {dataset_name}: {format_info}")
                           return format_info
                      else:
                           logger.warning_rank0(f"Dataset '{dataset_name}' not found in dataset_info.json, defaulting format.")
            else:
                 logger.warning_rank0(f"dataset_info.json not found in {self.data_args.dataset_dir}, defaulting format.")

        except ImportError:
            logger.warning_rank0("Could not import data parsing functions, defaulting format to Alpaca.")
        except Exception as e:
            logger.warning_rank0(f"Error retrieving dataset format: {e}. Defaulting format to Alpaca.")

        return {"formatting": "alpaca", "split": "train"}