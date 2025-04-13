import os
import json
import copy
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    GenerationConfig
)

from llamafactory.extras.logging import get_logger
from llamafactory.model import load_model, load_tokenizer
from llamafactory.hparams import (
    ModelArguments,
    DataArguments,
    FinetuningArguments,
)
from easycl.hparams import CLFinetuningArguments

logger = get_logger(__name__)

class PseudoReplay:
    """
    Pseudo Replay Implementation
    
    A simplified version of SSR (Selective Synthetic Replay) method for continual learning
    """
    
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        finetuning_args: FinetuningArguments,
        cl_finetuning_args: "CLFinetuningArguments"
    ):
        """Initialize PseudoReplay method"""
        self.model_args = model_args
        self.data_args = data_args
        self.finetuning_args = finetuning_args
        self.cl_finetuning_args = cl_finetuning_args
        
        # Pseudo Replay specific parameters
        self.use_pseudo_replay = cl_finetuning_args.use_pseudo_replay
        self.base_model_path = cl_finetuning_args.base_model_path
        self.num_samples_per_task = cl_finetuning_args.num_samples_per_task
        self.generation_temperature = cl_finetuning_args.generation_temperature
        self.pseudo_samples_dir = cl_finetuning_args.pseudo_samples_dir
        self.num_shots = cl_finetuning_args.num_shots
    
    def setup_base_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load base model for pseudo sample generation
        """
        # Create a copy of model arguments
        base_model_args = copy.deepcopy(self.model_args)
        
        # Use specified base model path
        if self.base_model_path:
            base_model_args.model_name_or_path = self.base_model_path
        
        # Remove all adapters, use pure base model for generation
        base_model_args.adapter_name_or_path = None
        
        # Load tokenizer
        tokenizer_module = load_tokenizer(base_model_args)
        tokenizer = tokenizer_module["tokenizer"]
        
        # Load base model - no training, disable adapters
        temp_finetuning_args = copy.deepcopy(self.finetuning_args)
        temp_finetuning_args.finetuning_type = "lora"  # Use lora type but don't load adapters
        base_model = load_model(
            tokenizer, 
            base_model_args, 
            temp_finetuning_args,
            is_trainable=False
        )
        
        logger.info_rank0(f"Loaded Pseudo Replay base model from: {base_model_args.model_name_or_path}")
        
        return base_model, tokenizer
    
    def get_few_shot_examples(self, dataset, num_shots=None):
        """
        Randomly select few-shot examples from dataset
        """
        if num_shots is None:
            num_shots = self.num_shots
        
        # Ensure dataset has enough samples
        if len(dataset) < num_shots:
            logger.warning_rank0(f"Dataset size ({len(dataset)}) is smaller than requested examples ({num_shots}), will use all samples")
            num_shots = len(dataset)
        
        # Randomly select samples
        indices = random.sample(range(len(dataset)), num_shots)
        examples = [dataset[i] for i in indices]
        
        return examples
    
    def construct_prompt(self, examples, template_type="alpaca"):
        """
        Construct few-shot prompt
        """
        instruction = 'Create task samples following examples below.\n\n'
        
        for example in examples:
            # Ensure compatibility with different dataset formats
            if "instruction" in example and "input" in example and "output" in example:
                # Alpaca format
                input_text = example["input"]
                input_text = "<noinput>" if input_text.lower() == "" else input_text
                
                instruction += (
                    f"Instruction: {example['instruction']}\n" +
                    f"Input: {input_text}\n" +
                    f"Output: {example['output']}\n\n"
                )
            elif "messages" in example:
                # ShareGPT format
                user_message = ""
                assistant_message = ""
                
                for msg in example["messages"]:
                    if msg["role"] == "user":
                        user_message = msg["content"]
                    elif msg["role"] == "assistant":
                        assistant_message = msg["content"]
                
                instruction += (
                    f"Instruction: Answer the following query\n" +
                    f"Input: {user_message}\n" +
                    f"Output: {assistant_message}\n\n"
                )
            else:
                # Generic format - best guess
                for key, value in example.items():
                    if isinstance(value, str):
                        # Assume first string field is instruction, second is input, third is output
                        if "instruction" not in locals():
                            instruction += f"Instruction: {value}\n"
                        elif "input_text" not in locals():
                            input_text = "<noinput>" if value == "" else value
                            instruction += f"Input: {input_text}\n"
                        elif "output" not in locals():
                            instruction += f"Output: {value}\n\n"
                            break
        
        instruction += 'Instruction:'
        
        return instruction
    
    def generate_pseudo_samples(self, dataset, num_samples=None):
        """
        Generate pseudo samples using base model
        """
        if num_samples is None:
            num_samples = self.num_samples_per_task
            
        # Load base model
        base_model, tokenizer = self.setup_base_model()
        
        # Select few-shot examples
        few_shot_examples = self.get_few_shot_examples(dataset)
        
        # Construct prompt
        prompt = self.construct_prompt(few_shot_examples)
        
        # Set generation config
        generation_config = GenerationConfig(
            temperature=self.generation_temperature,
            top_p=0.95,
            top_k=40,
            num_beams=1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Encode prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(base_model.device)
        
        # Generate pseudo samples
        logger.info_rank0(f"Starting to generate {num_samples} pseudo samples")
        generated_texts = []
        
        # Generate in batches to avoid GPU memory issues
        batch_size = min(5, num_samples)
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_samples - i*batch_size)
            if current_batch_size <= 0:
                break
                
            logger.info_rank0(f"Generating pseudo sample batch {i+1}/{num_batches}")
            
            # Generate multiple samples
            with torch.no_grad():
                outputs = base_model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    num_return_sequences=current_batch_size,
                    return_dict_in_generate=True
                )
            
            # Decode generated text
            for j in range(current_batch_size):
                if isinstance(outputs, dict) and "sequences" in outputs:
                    # Only take newly generated tokens
                    new_tokens = outputs.sequences[j][input_ids.shape[1]:]
                    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    
                    # Process and add generated text to result list
                    if generated_text:
                        generated_texts.append(prompt + " " + generated_text)
        
        # Free GPU memory
        del base_model
        torch.cuda.empty_cache()
        
        # Parse generated text into structured pseudo samples
        pseudo_samples = self.parse_generated_texts(generated_texts)
        
        logger.info_rank0(f"Successfully generated {len(pseudo_samples)} valid pseudo samples")
        
        return pseudo_samples
    
    def parse_generated_texts(self, generated_texts):
        """
        Parse generated text into structured pseudo samples
        """
        parsed_samples = []
        
        for text in generated_texts:
            try:
                # Split generated text to extract instruction, input and output
                parts = text.split("Output:", 1)
                if len(parts) != 2:
                    continue
                    
                output = parts[1].strip()
                remaining = parts[0].strip()
                
                # Split remaining part for input
                parts = remaining.split("Input:", 1)
                if len(parts) != 2:
                    continue
                    
                input_text = parts[1].strip()
                instruction_parts = parts[0].strip().split("Instruction:", 1)
                
                if len(instruction_parts) != 2:
                    continue
                    
                instruction = instruction_parts[1].strip()
                
                # Handle special tokens
                if input_text == "<noinput>":
                    input_text = ""
                
                # Validate parsed fields
                if not instruction:
                    continue
                
                # Create structured pseudo sample
                sample = {
                    "instruction": instruction,
                    "input": input_text,
                    "output": output
                }
                
                # Add to result list
                parsed_samples.append(sample)
                
            except Exception as e:
                logger.debug(f"Error parsing pseudo sample: {e}")
                continue
        
        # Ensure diversity of pseudo samples, remove exact duplicates
        unique_samples = []
        seen = set()
        
        for sample in parsed_samples:
            # Use string representation of sample as unique identifier
            sample_str = json.dumps(sample, sort_keys=True)
            if sample_str not in seen:
                seen.add(sample_str)
                unique_samples.append(sample)
        
        return unique_samples
    
    def save_pseudo_samples(self, samples, task_id, prev_task_id=None):
        """
        Save pseudo samples to specified directory
        """
        # Build save path
        output_dir = self.pseudo_samples_dir
        if not os.path.isabs(output_dir):
            # If relative path, relative to current working directory
            output_dir = os.path.join(os.getcwd(), output_dir)
            
        os.makedirs(output_dir, exist_ok=True)
        
        task_dir = os.path.join(output_dir, task_id)
        os.makedirs(task_dir, exist_ok=True)
        
        # If previous task's pseudo samples exist, load them first
        prev_samples = []
        if prev_task_id and prev_task_id != task_id:
            prev_task_dir = os.path.join(output_dir, prev_task_id)
            if os.path.exists(prev_task_dir):
                # Check previous task's pseudo sample files
                for prev_file in os.listdir(prev_task_dir):
                    if prev_file.startswith("pseudo_") and prev_file.endswith(".json"):
                        prev_file_path = os.path.join(prev_task_dir, prev_file)
                        try:
                            with open(prev_file_path, 'r', encoding='utf-8') as f:
                                prev_data = json.load(f)
                                prev_samples.extend(prev_data)
                        except Exception as e:
                            logger.warning_rank0(f"Error loading previous task's pseudo samples: {e}")
        
        # Merge current task's generated pseudo samples with historical ones
        all_samples = samples + prev_samples
        
        # Save merged pseudo samples
        pseudo_file_name = f"pseudo_{task_id}.json"
        pseudo_file_path = os.path.join(task_dir, pseudo_file_name)
        
        with open(pseudo_file_path, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)
        
        # Get format information from original dataset
        dataset_format = self.get_dataset_format()
        
        # Build dataset registration information
        dataset_info = {
            f"pseudo_{task_id}": {
                "file_name": pseudo_file_name,
                "formatting": dataset_format["formatting"],
                "split": "train"
            }
        }
        
        # Add other format-specific configurations to registration info
        if "columns" in dataset_format:
            dataset_info[f"pseudo_{task_id}"]["columns"] = dataset_format["columns"]
        if "tags" in dataset_format:
            dataset_info[f"pseudo_{task_id}"]["tags"] = dataset_format["tags"]
        
        # Save dataset registration information
        dataset_info_path = os.path.join(task_dir, "dataset_info.json")
        with open(dataset_info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
            
        logger.info_rank0(f"Saved merged pseudo samples from current and historical tasks to {pseudo_file_path}")
        logger.info_rank0(f"Current task pseudo samples: {len(samples)}")
        logger.info_rank0(f"Historical task pseudo samples: {len(prev_samples)}")
        logger.info_rank0(f"Total pseudo samples: {len(all_samples)}")
        
        return task_dir
    
    def get_dataset_format(self):
        """
        Get format information from original dataset for creating compatible dataset_info.json
        """
        try:
            from llamafactory.data.parser import get_dataset_list
            
            # Get current task's dataset information
            dataset_list = get_dataset_list(self.data_args.dataset, self.data_args.dataset_dir)
            if not dataset_list:
                # Use default alpaca format
                return {"formatting": "alpaca", "split": "train"}
            
            # Assume we use the first dataset as reference
            dataset_info_path = os.path.join(self.data_args.dataset_dir, "dataset_info.json")
            if os.path.exists(dataset_info_path):
                try:
                    with open(dataset_info_path, 'r', encoding='utf-8') as f:
                        dataset_info = json.load(f)
                        # Get current task's dataset name
                        dataset_name = self.data_args.dataset[0] if isinstance(self.data_args.dataset, list) else self.data_args.dataset
                        if dataset_name in dataset_info:
                            # Extract format information for this dataset
                            format_info = {}
                            src_info = dataset_info[dataset_name]
                            
                            # Copy basic format information
                            format_info["formatting"] = src_info.get("formatting", "alpaca")
                            format_info["split"] = src_info.get("split", "train")
                            
                            # Copy additional format information
                            if "columns" in src_info:
                                format_info["columns"] = src_info["columns"]
                            if "tags" in src_info:
                                format_info["tags"] = src_info["tags"]
                            
                            return format_info
                except Exception as e:
                    logger.warning_rank0(f"Error reading dataset_info.json: {e}")
        except Exception as e:
            logger.warning_rank0(f"Error getting dataset format information: {e}")
        
        # If unable to get format information, return default format
        return {"formatting": "alpaca", "split": "train"}
