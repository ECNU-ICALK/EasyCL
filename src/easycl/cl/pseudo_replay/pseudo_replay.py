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
    
    def _find_data_file(self, dataset_name: str, split_type: str = "train") -> Optional[str]:
        """
        Finds the data file path for a given dataset name and split type using dataset_info.json.
        Adapted from CLEvalEvaluator._find_data_file logic.

        Args:
            dataset_name: The name of the dataset (e.g., "alpaca_gpt4_en").
            split_type: The type of split required (usually "train").

        Returns:
            The absolute path to the data file, or None if not found or error occurs.
        """
        dataset_dir = self.data_args.dataset_dir
        dataset_info = None
        info_path = None
        base_dir_for_file = None # Directory where dataset_info.json was found

        # 1. Locate dataset_info.json
        potential_paths = [dataset_dir, os.path.join(os.getcwd(), "data")]
        for data_dir_path in potential_paths:
            current_info_path = os.path.join(data_dir_path, "dataset_info.json")
            if os.path.exists(current_info_path):
                info_path = current_info_path
                base_dir_for_file = data_dir_path # Remember the base dir
                logger.debug(f"Found dataset_info.json at: {info_path}") # English log
                break

        if not info_path:
            logger.warning_rank0(f"dataset_info.json not found in dataset_dir '{dataset_dir}' or fallback './data'. Cannot find raw file for few-shot examples.") # English log
            return None

        # 2. Load dataset_info.json
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                dataset_info = json.load(f)
        except Exception as e:
            logger.error_rank0(f"Failed to load or parse dataset_info.json from {info_path}: {e}. Cannot find raw file for few-shot examples.") # English log
            return None

        # 3. Find matching entry
        file_name = None
        # High priority: Check for key like "dataset_name_split"
        key_high_priority = f"{dataset_name}_{split_type}"
        if key_high_priority in dataset_info:
            entry = dataset_info[key_high_priority]
            if "file_name" in entry:
                file_name = entry["file_name"]
                logger.debug(f"Found high-priority match: key='{key_high_priority}', file_name='{file_name}'") # English log

        # Low priority: Check for key "dataset_name" and matching "split" field
        if file_name is None and dataset_name in dataset_info:
            entry = dataset_info[dataset_name]
            if "file_name" in entry and entry.get("split") == split_type:
                file_name = entry["file_name"]
                logger.debug(f"Found low-priority match: key='{dataset_name}', split='{split_type}', file_name='{file_name}'") # English log

        if not file_name:
            logger.warning_rank0(f"No matching entry with 'file_name' found in {info_path} for dataset '{dataset_name}' and split '{split_type}'. Cannot load few-shot examples.") # English log
            return None

        # 4. Construct and validate path (relative to the directory where dataset_info.json was found)
        full_path = os.path.join(base_dir_for_file, file_name)
        if os.path.exists(full_path):
            logger.debug(f"Resolved data file path: {full_path}") # English log
            return full_path
        else:
            # Try fallback ./data if dataset_info was found in dataset_dir and that wasn't ./data
            fallback_data_dir = os.path.join(os.getcwd(), "data")
            if os.path.abspath(base_dir_for_file) != os.path.abspath(fallback_data_dir):
                 fallback_path = os.path.join(fallback_data_dir, file_name)
                 if os.path.exists(fallback_path):
                     logger.warning_rank0(f"File '{file_name}' not found in '{base_dir_for_file}', but found in fallback './data'. Using fallback path: {fallback_path}") # English log
                     return fallback_path

            logger.error_rank0(f"Data file '{file_name}' specified in {info_path} not found at expected path: {full_path} (and fallback './data' if applicable). Cannot load few-shot examples.") # English log
            return None

    def get_few_shot_examples(self, num_shots=None):
        """
        Randomly select few-shot examples by reading the raw dataset file.
        Uses dataset_info.json to find the 'train' split file.
        """
        if num_shots is None:
            num_shots = self.num_shots
        
        # Determine the dataset name (use the first one if multiple are specified)
        if isinstance(self.data_args.dataset, list) and self.data_args.dataset:
            dataset_name = self.data_args.dataset[0]
        elif isinstance(self.data_args.dataset, str):
            dataset_name = self.data_args.dataset
        else:
            logger.warning_rank0("No dataset name found in data_args. Cannot load few-shot examples.") # English log
            return []

        logger.info_rank0(f"Attempting to load raw few-shot examples for dataset '{dataset_name}'") # English log

        # Find the raw data file path for the 'train' split
        data_file_path = self._find_data_file(dataset_name=dataset_name, split_type="train")

        if not data_file_path:
            logger.warning_rank0(f"Could not find raw data file for dataset '{dataset_name}'. Returning empty list for few-shot examples.") # English log
            return []

        # Load the raw data from the file
        try:
            with open(data_file_path, "r", encoding="utf-8") as f:
                # Handle both JSON and JSON Lines formats
                if data_file_path.endswith(".jsonl"):
                    raw_data = [json.loads(line) for line in f if line.strip()]
                elif data_file_path.endswith(".json"):
                    raw_data = json.load(f)
                    if not isinstance(raw_data, list): # Ensure it's a list of samples
                         logger.warning_rank0(f"Loaded JSON data from '{data_file_path}' is not a list. Cannot sample few-shot examples.")
                         return []
                else:
                    logger.warning_rank0(f"Unsupported file extension for raw data file: {data_file_path}. Expected .json or .jsonl. Returning empty list.")
                    return []
            logger.info_rank0(f"Successfully loaded {len(raw_data)} raw samples from {data_file_path}") # English log
        except Exception as e:
            logger.error_rank0(f"Error reading or parsing raw data file {data_file_path}: {e}") # English log
            return []

        # Ensure dataset has enough samples
        if not raw_data:
             logger.warning_rank0(f"Raw data file '{data_file_path}' is empty. Cannot sample few-shot examples.")
             return []
             
        if len(raw_data) < num_shots:
            logger.warning_rank0(f"Raw dataset size ({len(raw_data)}) is smaller than requested examples ({num_shots}), will use all samples") # English log
            num_shots = len(raw_data)

        # Randomly select samples
        try:
            indices = random.sample(range(len(raw_data)), num_shots)
            examples = [raw_data[i] for i in indices]
            logger.info_rank0(f"Selected {len(examples)} few-shot examples for dataset '{dataset_name}'") # English log
            return examples
        except Exception as e:
             logger.error_rank0(f"Error during random sampling from raw data: {e}") # English log
             return []
    
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
    
    def generate_pseudo_samples(self, num_samples=None):
        """
        Generate pseudo samples using base model
        """
        if num_samples is None:
            num_samples = self.num_samples_per_task
            
        # Load base model
        base_model, tokenizer = self.setup_base_model()
        base_model = base_model.cuda()
        tokenizer.padding_side = "left"
        # Select few-shot examples
        few_shot_examples = self.get_few_shot_examples()
        
        # If no few-shot examples could be loaded, handle gracefully (e.g., generate without few-shot prompt)
        if not few_shot_examples:
             logger.warning_rank0("No few-shot examples loaded. Proceeding with generation without few-shot prompt.")
             # Decide how to proceed: maybe use a generic prompt or raise an error
             # For now, let's try generating without the examples part of the prompt.
             # This might require adjusting construct_prompt or using a different prompt logic here.
             # Option 1: Use a generic instruction if construct_prompt fails
             prompt = "Instruction:" # Simplest fallback
             # Option 2: Modify construct_prompt to handle empty examples (Return just "Instruction:")
             # prompt = self.construct_prompt(few_shot_examples) # This might fail if examples is empty

             # Let's assume construct_prompt can handle empty examples gracefully or modify it.
             # For now, we will proceed, but this might need further adjustment
             # depending on construct_prompt's behavior with empty

        else:
             # Construct prompt using the loaded few-shot examples
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
        attention_mask = inputs["attention_mask"].to(base_model.device)
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
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    max_new_tokens=512,
                    num_return_sequences=current_batch_size,
                    return_dict_in_generate=True
                )
            
            # Decode generated text
            if isinstance(outputs, dict) and "sequences" in outputs:
                 sequences = outputs.sequences
                 # sequences shape: [current_batch_size, sequence_length]
                 prompt_length = input_ids.shape[1]
                 
                 for j in range(sequences.shape[0]): # Iterate through batch dimension
                    # Only take newly generated tokens
                    new_tokens = sequences[j][prompt_length:]

                    
                    # Use batch_decode for efficiency
                    # generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    
                    # Add generated text (including prompt initially for parsing logic) to result list
                    # We will decode later or let the parser handle it if needed.
                    # For now, just store the full sequence? Let's try decoding here.
                    full_decoded_text = tokenizer.decode(sequences[j], skip_special_tokens=True)

                    # Alternative: Decode only new tokens and prepend prompt
                    # generated_part = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    # full_decoded_text = prompt + generated_part # Approximate, prompt decoding might differ slightly

                    if full_decoded_text:
                         generated_texts.append(full_decoded_text)
   
        # Free GPU memory
        del base_model
        del tokenizer
        del inputs
        del input_ids
        del attention_mask
        if 'outputs' in locals(): del outputs
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
