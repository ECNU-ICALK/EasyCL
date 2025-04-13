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
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationConfig,
    AutoModel,
    AutoTokenizer
)

from ...extras import logging
from ...model import load_model, load_tokenizer
from ...hparams import ModelArguments, FinetuningArguments

logger = logging.get_logger(__name__)

class SSR:
    """
    Self-Synthesized Rehearsal (SSR) for continual learning with language models.
    
    SSR uses the pretrained model to generate synthetic examples from previous tasks,
    then refines these examples using the latest model adapted to those tasks.
    """
    
    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        model_args: Optional[ModelArguments] = None,
        finetuning_args: Optional[FinetuningArguments] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the SSR method.
        
        Args:
            model: The current model being trained
            model_args: Model arguments
            finetuning_args: Fine-tuning arguments
            device: The device to use for generation
        """
        self.model = model
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        
        # Placeholder for base model and previous model
        self.base_model = None
        self.previous_model = None
        self.feature_extractor = None
        
        # Store various adapter references
        self.adapter_history = []
        
        # Parameters for pseudo sample generation 
        self.current_task_id = finetuning_args.current_task_id if finetuning_args else None
        
        logger.info_rank0("Initialized SSR method for continual learning")
    
    def store_adapter_history(self, adapters: List[str]) -> None:
        """
        Store the history of adapters that have been merged.
        
        Args:
            adapters: List of adapter paths that were merged
        """
        if adapters:
            self.adapter_history.extend(adapters)
            logger.info_rank0(f"Stored {len(adapters)} adapter(s) in history")

    def process_resumed_adapter(self, adapter: str) -> None:
        """
        Process a resumed adapter.
        
        Args:
            adapter: Path to the resumed adapter
        """
        logger.info_rank0(f"Processing resumed adapter: {adapter}")
        # No specific processing needed for SSR, but keeping this method for consistency
    
    def modify_peft_config(self, peft_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modify the PEFT configuration for SSR if needed.
        
        Args:
            peft_kwargs: The original PEFT configuration
            
        Returns:
            The modified PEFT configuration
        """
        # No special PEFT config modification for SSR
        return peft_kwargs
    
    def post_process_model(self, model: PreTrainedModel) -> PreTrainedModel:
        """
        Post-process the model after PEFT initialization if needed.
        
        Args:
            model: The initialized model with adapters
            
        Returns:
            The post-processed model
        """
        # Store the reference to the model
        self.model = model
        return model
    
    def finalize_setup(self, model: PreTrainedModel) -> None:
        """
        Finalize the setup of SSR method.
        
        Args:
            model: The model with adapters
        """
        # Initialize base and previous models if needed
        if self.finetuning_args.base_model_path:
            logger.info_rank0(f"Loading base model from {self.finetuning_args.base_model_path} for pseudo-sample generation")
            # Base model is loaded when needed to save memory
        
        if self.finetuning_args.previous_task_model:
            logger.info_rank0(f"Previous task model path: {self.finetuning_args.previous_task_model}")
            # Previous model is loaded when needed to save memory
    
    def _load_base_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load the base model for pseudo-sample generation if it hasn't been loaded yet.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        if not self.base_model:
            # Create temporary model args for loading base model
            base_model_args = ModelArguments(
                model_name_or_path=self.finetuning_args.base_model_path or self.model_args.model_name_or_path,
                trust_remote_code=self.model_args.trust_remote_code,
                cache_dir=self.model_args.cache_dir,
            )
            
            # Load tokenizer
            tokenizer_module = load_tokenizer(base_model_args)
            tokenizer = tokenizer_module["tokenizer"]
            
            # Create temporary finetuning args to avoid adapter loading
            temp_finetuning_args = FinetuningArguments(
                finetuning_type="lora",  # Use same type but won't load adapters
                stage="sft",
            )
            
            # Load model without adapters
            self.base_model = load_model(
                tokenizer=tokenizer,
                model_args=base_model_args,
                finetuning_args=temp_finetuning_args,
                is_trainable=False,  # Not for training
            )
            
            # Move to device and set to eval mode
            self.base_model.to(self.device)
            self.base_model.eval()
            
            logger.info_rank0(f"Loaded base model for pseudo-sample generation")
            
            return self.base_model, tokenizer
        
        return self.base_model, None  # Tokenizer would need to be retrieved again
    
    def _load_previous_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load the previous task model for pseudo-sample refinement if it hasn't been loaded yet.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        if not self.previous_model and self.finetuning_args.previous_task_model:
            # Create temporary model args for loading previous model
            prev_model_args = ModelArguments(
                model_name_or_path=self.model_args.model_name_or_path,
                adapter_name_or_path=[self.finetuning_args.previous_task_model] if self.finetuning_args.finetuning_type == "lora" else None,
                trust_remote_code=self.model_args.trust_remote_code,
                cache_dir=self.model_args.cache_dir,
            )
            
            # Load tokenizer
            tokenizer_module = load_tokenizer(prev_model_args)
            tokenizer = tokenizer_module["tokenizer"]
            
            # Create temporary finetuning args
            temp_finetuning_args = FinetuningArguments(
                finetuning_type=self.finetuning_args.finetuning_type,
                stage="sft",
            )
            
            # Load model with previous task adapter
            self.previous_model = load_model(
                tokenizer=tokenizer,
                model_args=prev_model_args,
                finetuning_args=temp_finetuning_args,
                is_trainable=False,  # Not for training
            )
            
            # Move to device and set to eval mode
            self.previous_model.to(self.device)
            self.previous_model.eval()
            
            logger.info_rank0(f"Loaded previous task model for pseudo-sample refinement")
            
            return self.previous_model, tokenizer
        
        return self.previous_model, None  # Tokenizer would need to be retrieved again
    
    def _load_feature_extractor(self):
        """Load a feature extractor model for embedding generation"""
        if self.feature_extractor is None:
            try:
                # Try to load SimCSE model for better embeddings
                self.feature_extractor = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-base")
                self.feature_extractor_tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-base")
                self.feature_extractor.to(self.device)
                self.feature_extractor.eval()
                logger.info_rank0("Loaded SimCSE model for embeddings")
            except Exception as e:
                logger.warning_rank0(f"Failed to load SimCSE model: {e}. Will use base model for embeddings.")
                self.feature_extractor = None
        
        return self.feature_extractor
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts using feature extractor or base model.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        feature_extractor = self._load_feature_extractor()
        
        if feature_extractor is not None:
            # Use SimCSE model for better embeddings
            with torch.no_grad():
                inputs = self.feature_extractor_tokenizer(
                    texts, padding=True, truncation=True, return_tensors="pt"
                ).to(self.device)
                
                embeddings = feature_extractor(**inputs).last_hidden_state[:, 0, :]
                embeddings = embeddings.cpu().numpy()
        else:
            # Fallback to base model embeddings
            model, tokenizer = self._load_base_model()
            
            with torch.no_grad():
                embeddings = []
                # Process in batches to avoid OOM
                batch_size = 16
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    inputs = tokenizer(
                        batch_texts, padding=True, truncation=True, return_tensors="pt"
                    ).to(self.device)
                    
                    outputs = model(**inputs)
                    # Use [CLS] token or average of last hidden state
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.append(batch_embeddings)
                
                embeddings = np.vstack(embeddings)
        
        # Normalize embeddings
        embeddings = normalize(embeddings, axis=1)
        return embeddings
    
    def format_few_shot_examples(self, examples: List[Dict[str, str]]) -> str:
        """
        Format few-shot examples for ICL prompt.
        
        Args:
            examples: List of example dictionaries with 'instruction', 'input', and 'output' keys
            
        Returns:
            Formatted few-shot prompt
        """
        prompt = "Create task samples following examples below.\n\n"
        
        for example in examples:
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            input_text = "<noinput>" if input_text.lower() == "" else input_text
            output = example.get("output", "")
            
            prompt += f"Instruction: {instruction}\n"
            prompt += f"Input: {input_text}\n"
            prompt += f"Output: {output}\n\n"
        
        prompt += "Instruction:"
        return prompt
    
    def parse_generated_sample(self, text: str) -> Optional[Dict[str, str]]:
        """
        Parse generated text into instruction, input, and output components.
        
        Args:
            text: Generated text to parse
            
        Returns:
            Dictionary with 'instruction', 'input', and 'output' keys, or None if parsing fails
        """
        try:
            # First split by "Output:"
            parts = text.split("Output:", 1)
            if len(parts) != 2:
                return None
            
            output = parts[1].strip()
            remaining = parts[0].strip()
            
            # Split remaining by "Input:"
            parts = remaining.split("Input:", 1)
            if len(parts) != 2:
                return None
            
            input_text = parts[1].strip()
            instruction = parts[0].strip()
            
            # Remove "Instruction:" prefix if present
            if instruction.startswith("Instruction:"):
                instruction = instruction[len("Instruction:"):].strip()
            
            # Handle <noinput> placeholder
            if input_text.lower() == "<noinput>":
                input_text = ""
            
            # Validate components
            if not instruction or not output:
                return None
            
            return {
                "instruction": instruction,
                "input": input_text,
                "output": output
            }
        except Exception:
            return None
    
    def generate_pseudo_samples(
        self, 
        few_shot_examples: List[Dict[str, str]], 
        num_samples: int = 50,
        batch_size: int = 5
    ) -> List[Dict[str, str]]:
        """
        Generate pseudo samples using few-shot ICL.
        
        Args:
            few_shot_examples: List of few-shot examples to use
            num_samples: Number of pseudo samples to generate
            batch_size: Number of samples to generate in each batch
            
        Returns:
            List of generated pseudo samples
        """
        logger.info_rank0(f"Generating {num_samples} pseudo samples using {len(few_shot_examples)} few-shot examples")
        
        # Load base model and tokenizer
        model, tokenizer = self._load_base_model()
        
        # Format ICL prompt
        icl_prompt = self.format_few_shot_examples(few_shot_examples)
        
        # Set up generation config
        gen_config = GenerationConfig(
            max_new_tokens=512,
            temperature=self.finetuning_args.generation_temperature,
            top_p=0.95,
            do_sample=True,
        )
        
        pseudo_samples = []
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        # Generate in batches
        for i in range(num_batches):
            if len(pseudo_samples) >= num_samples:
                break
                
            logger.info_rank0(f"Generating batch {i+1}/{num_batches}...")
            
            # Encode prompt
            inputs = tokenizer(icl_prompt, return_tensors="pt").to(self.device)
            
            # Generate completions
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=gen_config,
                    return_dict_in_generate=True,
                    output_scores=False,
                    num_return_sequences=batch_size,
                )
            
            # Decode outputs
            for sequence in outputs.sequences:
                # Skip the prompt part in the generated text
                prompt_length = inputs.input_ids.shape[1]
                completion = tokenizer.decode(sequence[prompt_length:], skip_special_tokens=True)
                
                # Parse the completion into instruction, input, output format
                sample = self.parse_generated_sample(f"Instruction:{completion}")
                
                if sample:
                    pseudo_samples.append(sample)
            
            # Log progress
            logger.info_rank0(f"Generated {len(pseudo_samples)} valid samples so far")
        
        # Truncate to desired number
        pseudo_samples = pseudo_samples[:num_samples]
        
        logger.info_rank0(f"Successfully generated {len(pseudo_samples)} pseudo samples")
        return pseudo_samples
    
    def refine_pseudo_samples(self, pseudo_samples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Refine pseudo samples using the previous task model.
        
        Args:
            pseudo_samples: List of pseudo samples to refine
            
        Returns:
            List of refined pseudo samples
        """
        if not self.finetuning_args.previous_task_model:
            logger.warning_rank0("No previous task model provided, skipping refinement")
            return pseudo_samples
        
        logger.info_rank0(f"Refining {len(pseudo_samples)} pseudo samples with previous task model")
        
        # Load previous model and tokenizer
        model, tokenizer = self._load_previous_model()
        
        refined_samples = []
        batch_size = 4  # Process in small batches to avoid OOM
        
        # Set up generation config
        gen_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.7,  # Lower temperature for refinement
            top_p=0.95,
            do_sample=False,  # Use greedy decoding for refinement
        )
        
        # Process in batches
        for i in range(0, len(pseudo_samples), batch_size):
            batch = pseudo_samples[i:i+batch_size]
            batch_refined = []
            
            for sample in batch:
                instruction = sample["instruction"]
                input_text = sample["input"]
                
                # Format the prompt
                if input_text:
                    prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
                else:
                    prompt = f"Instruction: {instruction}\nInput: <noinput>\nOutput:"
                
                # Encode prompt
                inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
                
                # Generate refined output
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        generation_config=gen_config,
                        return_dict_in_generate=True,
                    )
                
                # Decode output and extract only the new part
                prompt_length = inputs.input_ids.shape[1]
                refined_output = tokenizer.decode(
                    outputs.sequences[0][prompt_length:],
                    skip_special_tokens=True
                )
                
                # Create refined sample
                refined_sample = {
                    "instruction": instruction,
                    "input": input_text,
                    "output": refined_output,
                    "original_output": sample["output"]  # Keep original for reference
                }
                
                batch_refined.append(refined_sample)
            
            refined_samples.extend(batch_refined)
            logger.info_rank0(f"Refined {len(refined_samples)}/{len(pseudo_samples)} samples")
        
        logger.info_rank0(f"Successfully refined {len(refined_samples)} pseudo samples")
        return refined_samples
    
    def cluster_and_select_samples(
        self, 
        pseudo_samples: List[Dict[str, str]], 
        n_clusters: int = 20,
        memory_size: int = 200
    ) -> List[Dict[str, str]]:
        """
        Cluster and select diverse pseudo samples.
        
        Args:
            pseudo_samples: List of pseudo samples to cluster
            n_clusters: Number of clusters to create
            memory_size: Maximum number of samples to keep
            
        Returns:
            List of selected diverse pseudo samples
        """
        if len(pseudo_samples) <= memory_size:
            logger.info_rank0(f"Number of samples ({len(pseudo_samples)}) less than memory size ({memory_size}), keeping all")
            return pseudo_samples
            
        logger.info_rank0(f"Clustering {len(pseudo_samples)} samples into {n_clusters} clusters")
        
        # Extract instruction texts for embedding
        instruction_texts = [
            f"{sample['instruction']} {sample['input']}" 
            for sample in pseudo_samples
        ]
        
        # Get embeddings
        embeddings = self._get_embeddings(instruction_texts)
        
        # Adjust number of clusters if needed
        actual_n_clusters = min(n_clusters, len(pseudo_samples))
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Count samples per cluster
        cluster_counts = np.bincount(cluster_labels, minlength=actual_n_clusters)
        
        # Calculate samples to select from each cluster proportionally
        samples_per_cluster = np.zeros(actual_n_clusters, dtype=int)
        remaining = memory_size
        
        # First, assign at least one sample to each non-empty cluster
        for i in range(actual_n_clusters):
            if cluster_counts[i] > 0:
                samples_per_cluster[i] = 1
                remaining -= 1
        
        # Distribute remaining samples proportionally
        if remaining > 0:
            props = cluster_counts / cluster_counts.sum()
            extras = np.floor(props * remaining).astype(int)
            samples_per_cluster += extras
            remaining -= extras.sum()
            
            # Distribute any remaining samples to largest clusters
            if remaining > 0:
                sorted_clusters = np.argsort(-cluster_counts)
                for i in range(int(remaining)):
                    samples_per_cluster[sorted_clusters[i]] += 1
        
        # Select samples from each cluster
        selected_samples = []
        
        for cluster_idx in range(actual_n_clusters):
            cluster_size = samples_per_cluster[cluster_idx]
            if cluster_size <= 0:
                continue
                
            # Get indices of samples in this cluster
            cluster_sample_indices = np.where(cluster_labels == cluster_idx)[0]
            
            if len(cluster_sample_indices) <= cluster_size:
                # Take all samples from this cluster
                selected_indices = cluster_sample_indices
            else:
                # Calculate distance to cluster center
                cluster_center = kmeans.cluster_centers_[cluster_idx]
                distances = np.linalg.norm(
                    embeddings[cluster_sample_indices] - cluster_center, 
                    axis=1
                )
                
                # Select samples closest to center
                closest_indices = np.argsort(distances)[:cluster_size]
                selected_indices = cluster_sample_indices[closest_indices]
            
            # Add selected samples
            for idx in selected_indices:
                selected_samples.append(pseudo_samples[idx])
        
        logger.info_rank0(f"Selected {len(selected_samples)} diverse samples from {len(pseudo_samples)} samples")
        return selected_samples
    
    def save_pseudo_samples(
        self, 
        pseudo_samples: List[Dict[str, str]], 
        task_id: str,
        output_dir: str
    ) -> None:
        """
        Save pseudo samples to disk.
        
        Args:
            pseudo_samples: List of pseudo samples to save
            task_id: ID of the task
            output_dir: Base directory to save pseudo samples
        """
        # Create directory if it doesn't exist
        task_dir = os.path.join(output_dir, task_id)
        os.makedirs(task_dir, exist_ok=True)
        
        # Save samples
        samples_path = os.path.join(task_dir, "samples.jsonl")
        with open(samples_path, "w", encoding="utf-8") as f:
            for sample in pseudo_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        # Save metadata
        metadata_path = os.path.join(task_dir, "metadata.json")
        metadata = {
            "task_id": task_id,
            "num_samples": len(pseudo_samples),
            "generation_timestamp": torch.cuda.current_device() if torch.cuda.is_available() else "cpu",
            "config": {
                "generation_temperature": self.finetuning_args.generation_temperature,
                "n_clusters": self.finetuning_args.n_clusters,
                "memory_size": self.finetuning_args.pseudo_sample_memory,
            }
        }
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info_rank0(f"Saved {len(pseudo_samples)} pseudo samples to {task_dir}")
    
    def load_pseudo_samples(self, task_id: str, output_dir: str) -> List[Dict[str, str]]:
        """
        Load pseudo samples from disk.
        
        Args:
            task_id: ID of the task
            output_dir: Base directory where pseudo samples are saved
            
        Returns:
            List of loaded pseudo samples
        """
        task_dir = os.path.join(output_dir, task_id)
        samples_path = os.path.join(task_dir, "samples.jsonl")
        
        if not os.path.exists(samples_path):
            logger.warning_rank0(f"No pseudo samples found at {samples_path}")
            return []
        
        pseudo_samples = []
        with open(samples_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line)
                pseudo_samples.append(sample)
        
        logger.info_rank0(f"Loaded {len(pseudo_samples)} pseudo samples from {task_dir}")
        return pseudo_samples
