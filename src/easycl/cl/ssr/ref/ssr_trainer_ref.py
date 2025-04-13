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
import random
from typing import Dict, List, Optional, Union, Any, Tuple

import torch
import numpy as np
from datasets import Dataset, concatenate_datasets
import jsonlines

from ...extras import logging
from ...train.sft.trainer import CustomSeq2SeqTrainer
from .ssr import SSR

logger = logging.get_logger(__name__)

class SSRTrainer(CustomSeq2SeqTrainer):
    """
    Trainer for Self-Synthesized Rehearsal (SSR) method that extends the base CustomSeq2SeqTrainer.
    This trainer handles the generation, refinement, and selection of pseudo samples for rehearsal.
    """
    
    def __init__(self, ssr_instance: Optional[SSR] = None, prev_task_datasets: Optional[Dict[str, Dataset]] = None, **kwargs):
        """
        Initialize the SSR trainer.
        
        Args:
            ssr_instance: SSR instance for managing pseudo samples
            prev_task_datasets: Dict of previous task datasets for few-shot examples
            **kwargs: Additional arguments to pass to the parent trainer
        """
        super().__init__(**kwargs)
        
        # Store SSR instance
        self.ssr = ssr_instance
        if self.ssr is None and hasattr(self.model, "ssr"):
            self.ssr = self.model.ssr
        
        # Store prev_task_datasets for few-shot examples
        self.prev_task_datasets = prev_task_datasets or {}
        
        # Initialize pseudo samples directory
        if not os.path.exists(self.finetuning_args.pseudo_samples_dir):
            try:
                # Try to create absolute path
                os.makedirs(self.finetuning_args.pseudo_samples_dir, exist_ok=True)
            except:
                # Fall back to path relative to output_dir
                self.finetuning_args.pseudo_samples_dir = os.path.join(
                    self.args.output_dir, self.finetuning_args.pseudo_samples_dir
                )
                os.makedirs(self.finetuning_args.pseudo_samples_dir, exist_ok=True)
        
        logger.info_rank0(f"SSR pseudo samples directory: {self.finetuning_args.pseudo_samples_dir}")
        
        # Check if we have a current task ID
        if not self.finetuning_args.current_task_id:
            # Try to extract from output dir if not provided
            self.finetuning_args.current_task_id = os.path.basename(self.args.output_dir)
            logger.info_rank0(f"Using directory name as current task ID: {self.finetuning_args.current_task_id}")
    
    def train(self, *args, **kwargs):
        """
        Extend the training process to include pseudo sample generation and rehearsal.
        
        Returns:
            TrainOutput from the parent trainer's train method
        """
        if not self.finetuning_args.use_ssr:
            logger.info_rank0("SSR not enabled, proceeding with regular training")
            return super().train(*args, **kwargs)
        
        # Check if we have previous tasks
        if self.finetuning_args.previous_task_model:
            # 1. Sample few-shot examples from previous task data
            few_shot_examples = self._get_few_shot_examples()
            
            if few_shot_examples:
                # 2. Generate pseudo samples using ICL
                pseudo_samples = self.generate_pseudo_samples(few_shot_examples)
                
                # 3. Refine pseudo samples with previous task model
                refined_samples = self.refine_pseudo_samples(pseudo_samples)
                
                # 4. Cluster and select diverse samples
                selected_samples = self.select_samples_for_rehearsal(refined_samples)
                
                # 5. Save pseudo samples for future tasks
                self.save_pseudo_samples(selected_samples)
            
            # 6. Load all historical pseudo samples for rehearsal
            rehearsal_dataset = self.prepare_rehearsal_dataset()
            
            # 7. Merge with current task data
            self.train_dataset = self.merge_current_and_rehearsal_data(rehearsal_dataset)
            
            logger.info_rank0(f"Final training dataset size: {len(self.train_dataset)}")
        else:
            logger.info_rank0("No previous task model specified, skipping pseudo sample generation")
        
        # 8. Train with the merged dataset
        return super().train(*args, **kwargs)
    
    def _get_few_shot_examples(self, num_shots: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get few-shot examples from previous task data.
        
        Args:
            num_shots: Number of examples to sample, defaults to self.finetuning_args.num_shots
            
        Returns:
            List of example dictionaries with 'instruction', 'input', and 'output' keys
        """
        if not num_shots:
            num_shots = self.finetuning_args.num_shots
        
        # First try to use prev_task_datasets if available
        if self.prev_task_datasets and len(self.prev_task_datasets) > 0:
            # Get the last previous task
            prev_task_id = list(self.prev_task_datasets.keys())[-1]
            prev_dataset = self.prev_task_datasets[prev_task_id]
            
            # Sample from the dataset
            if len(prev_dataset) > 0:
                # Random sample or cluster-based sample
                indices = random.sample(range(len(prev_dataset)), min(num_shots, len(prev_dataset)))
                examples = []
                
                for idx in indices:
                    item = prev_dataset[idx]
                    # Convert to standard format
                    example = {
                        "instruction": item.get("instruction", ""),
                        "input": item.get("input", ""),
                        "output": item.get("output", "")
                    }
                    examples.append(example)
                
                logger.info_rank0(f"Sampled {len(examples)} few-shot examples from previous task dataset")
                return examples
        
        # If no datasets, try to load from pseudo samples directory
        if self.finetuning_args.prev_task_id:
            task_dir = os.path.join(self.finetuning_args.pseudo_samples_dir, self.finetuning_args.prev_task_id)
            samples_path = os.path.join(task_dir, "samples.jsonl")
            
            if os.path.exists(samples_path):
                with jsonlines.open(samples_path) as f:
                    all_samples = list(f)
                
                # Sample a subset
                if len(all_samples) > num_shots:
                    examples = random.sample(all_samples, num_shots)
                else:
                    examples = all_samples
                
                logger.info_rank0(f"Sampled {len(examples)} few-shot examples from previous task pseudo samples")
                return examples
        
        # Fallback to dataset provided in previous_task_data if any
        if self.finetuning_args.previous_task_data:
            # This would require loading the dataset
            logger.warning_rank0("Loading few-shot examples from previous_task_data is not implemented yet")
            # TODO: Implement loading from previous_task_data
        
        logger.warning_rank0("No few-shot examples found, will generate without examples")
        return []
    
    def generate_pseudo_samples(self, few_shot_examples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Generate pseudo samples for rehearsal.
        
        Args:
            few_shot_examples: List of few-shot examples to use
            
        Returns:
            List of generated pseudo samples
        """
        # Check if we have SSR instance
        if not self.ssr:
            logger.error_rank0("No SSR instance found, cannot generate pseudo samples")
            return []
        
        # Number of samples to generate
        num_samples = min(500, self.finetuning_args.pseudo_sample_memory * 2)  # Generate more than needed for filtering
        
        # Generate pseudo samples
        pseudo_samples = self.ssr.generate_pseudo_samples(
            few_shot_examples=few_shot_examples,
            num_samples=num_samples,
            batch_size=5  # Generate in small batches
        )
        
        logger.info_rank0(f"Generated {len(pseudo_samples)} initial pseudo samples")
        return pseudo_samples
    
    def refine_pseudo_samples(self, pseudo_samples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Refine pseudo samples using the previous task model.
        
        Args:
            pseudo_samples: List of pseudo samples to refine
            
        Returns:
            List of refined pseudo samples
        """
        if not self.ssr:
            logger.error_rank0("No SSR instance found, cannot refine pseudo samples")
            return pseudo_samples
        
        # Refine the samples
        refined_samples = self.ssr.refine_pseudo_samples(pseudo_samples)
        
        logger.info_rank0(f"Refined {len(refined_samples)} pseudo samples")
        return refined_samples
    
    def select_samples_for_rehearsal(self, pseudo_samples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Select diverse pseudo samples for rehearsal using clustering.
        
        Args:
            pseudo_samples: List of refined pseudo samples to select from
            
        Returns:
            List of selected diverse pseudo samples
        """
        if not self.ssr:
            logger.error_rank0("No SSR instance found, cannot select pseudo samples")
            return pseudo_samples[:self.finetuning_args.pseudo_sample_memory]
        
        # Cluster and select samples
        selected_samples = self.ssr.cluster_and_select_samples(
            pseudo_samples=pseudo_samples,
            n_clusters=self.finetuning_args.n_clusters,
            memory_size=self.finetuning_args.pseudo_sample_memory
        )
        
        logger.info_rank0(f"Selected {len(selected_samples)} diverse pseudo samples for rehearsal")
        return selected_samples
    
    def save_pseudo_samples(self, pseudo_samples: List[Dict[str, str]]) -> None:
        """
        Save pseudo samples to disk for future tasks.
        
        Args:
            pseudo_samples: List of selected pseudo samples to save
        """
        if not self.ssr:
            logger.error_rank0("No SSR instance found, cannot save pseudo samples")
            return
        
        # Save the samples
        self.ssr.save_pseudo_samples(
            pseudo_samples=pseudo_samples,
            task_id=self.finetuning_args.current_task_id,
            output_dir=self.finetuning_args.pseudo_samples_dir
        )
        
        logger.info_rank0(f"Saved {len(pseudo_samples)} pseudo samples for task {self.finetuning_args.current_task_id}")
    
    def prepare_rehearsal_dataset(self) -> Dataset:
        """
        Load all historical pseudo samples and prepare the rehearsal dataset.
        
        Returns:
            Dataset containing all historical pseudo samples
        """
        # Get all subdirectories in the pseudo_samples_dir
        pseudo_samples_dir = self.finetuning_args.pseudo_samples_dir
        if not os.path.exists(pseudo_samples_dir):
            logger.warning_rank0(f"Pseudo samples directory {pseudo_samples_dir} does not exist")
            return Dataset.from_dict({"instruction": [], "input": [], "output": []})
        
        all_task_dirs = [
            d for d in os.listdir(pseudo_samples_dir) 
            if os.path.isdir(os.path.join(pseudo_samples_dir, d)) and d != self.finetuning_args.current_task_id
        ]
        
        if not all_task_dirs:
            logger.warning_rank0("No historical task directories found in pseudo samples directory")
            return Dataset.from_dict({"instruction": [], "input": [], "output": []})
        
        # Load samples from all task directories
        all_samples = []
        for task_id in all_task_dirs:
            task_dir = os.path.join(pseudo_samples_dir, task_id)
            samples_path = os.path.join(task_dir, "samples.jsonl")
            
            if os.path.exists(samples_path):
                with jsonlines.open(samples_path) as f:
                    task_samples = list(f)
                    all_samples.extend(task_samples)
                    logger.info_rank0(f"Loaded {len(task_samples)} pseudo samples from task {task_id}")
        
        if not all_samples:
            logger.warning_rank0("No historical pseudo samples found")
            return Dataset.from_dict({"instruction": [], "input": [], "output": []})
        
        # Limit total number of samples if needed
        cl_buffer_size = self.finetuning_args.cl_buffer_size
        if len(all_samples) > cl_buffer_size:
            logger.info_rank0(f"Limiting total number of rehearsal samples to {cl_buffer_size}")
            all_samples = random.sample(all_samples, cl_buffer_size)
        
        # Convert to dataset format
        all_instructions = [s["instruction"] for s in all_samples]
        all_inputs = [s["input"] for s in all_samples]
        all_outputs = [s["output"] for s in all_samples]
        
        rehearsal_dataset = Dataset.from_dict({
            "instruction": all_instructions,
            "input": all_inputs,
            "output": all_outputs
        })
        
        logger.info_rank0(f"Prepared rehearsal dataset with {len(rehearsal_dataset)} samples")
        return rehearsal_dataset
    
    def merge_current_and_rehearsal_data(self, rehearsal_dataset: Dataset) -> Dataset:
        """
        Merge current task data with rehearsal dataset.
        
        Args:
            rehearsal_dataset: Dataset containing historical pseudo samples
            
        Returns:
            Merged dataset for training
        """
        if len(rehearsal_dataset) == 0:
            logger.info_rank0("No rehearsal data to merge, using only current task data")
            return self.train_dataset
        
        # Check if datasets are compatible
        current_cols = set(self.train_dataset.column_names)
        rehearsal_cols = set(rehearsal_dataset.column_names)
        
        if not rehearsal_cols.issubset(current_cols):
            logger.warning_rank0(f"Rehearsal dataset columns {rehearsal_cols} not compatible with current task columns {current_cols}")
            
            # Try to convert rehearsal dataset to match current format
            try:
                # Get common columns
                common_cols = current_cols.intersection(rehearsal_cols)
                
                if not common_cols:
                    logger.error_rank0("No common columns between datasets, cannot merge")
                    return self.train_dataset
                
                # Use only common columns
                rehearsal_dataset = rehearsal_dataset.select_columns(list(common_cols))
                
                # Add missing columns with default values
                for col in current_cols - common_cols:
                    if "input_ids" in col or "attention_mask" in col or "labels" in col:
                        # Skip tokenized columns, they'll be generated during collation
                        continue
                    
                    # Add empty column
                    rehearsal_dataset = rehearsal_dataset.add_column(
                        col, [""] * len(rehearsal_dataset)
                    )
                
                logger.info_rank0(f"Adapted rehearsal dataset to match current task format")
            except Exception as e:
                logger.error_rank0(f"Failed to adapt rehearsal dataset: {e}")
                return self.train_dataset
        
        # Balance the datasets
        if len(rehearsal_dataset) > len(self.train_dataset):
            # Subsample rehearsal data to match current task data
            rehearsal_indices = random.sample(range(len(rehearsal_dataset)), len(self.train_dataset))
            rehearsal_dataset = rehearsal_dataset.select(rehearsal_indices)
            logger.info_rank0(f"Subsampled rehearsal dataset to {len(rehearsal_dataset)} samples")
        
        # Merge datasets
        merged_dataset = concatenate_datasets([self.train_dataset, rehearsal_dataset])
        
        logger.info_rank0(f"Merged current task data ({len(self.train_dataset)} samples) with " +
                        f"rehearsal data ({len(rehearsal_dataset)} samples) = {len(merged_dataset)} samples")
        
        # Shuffle the merged dataset
        merged_dataset = merged_dataset.shuffle(seed=42)
        
        return merged_dataset
