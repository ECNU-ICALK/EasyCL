import os
import json
import copy
import gc
import torch
import yaml
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Callable, Union
from dataclasses import asdict
from llamafactory.eval.evaluator import Evaluator
from .cl_eval import CLEvalEvaluator
from llamafactory.hparams import (
    ModelArguments,
    DataArguments,
    FinetuningArguments
)
from easycl.hparams import CLEvaluationArguments, CLFinetuningArguments
from llamafactory.extras import logging

logger = logging.get_logger(__name__)


class CLEvaluator:
    """Continuous Learning Evaluator"""
    
    @staticmethod
    def calculate_transfer(results: Dict[str, Dict[str, float]]) -> float:
        """Calculate transfer capability metric"""
        if not results:
            return 0.0
        accuracies = [task_result["accuracy"] for task_result in results.values()]
        return sum(accuracies) / len(accuracies)

    @staticmethod
    def calculate_bwt(results: Dict[str, Dict[str, float]], tasks: List[str]) -> float:
        """Calculate backward transfer metric"""
        if len(tasks) <= 1:
            return 0.0
        bwt = 0.0
        n = len(tasks)
        for i in range(n - 1):
            task = tasks[i]
            if task in results:
                bwt += results[task]["accuracy"]
        return bwt / (n - 1) if n > 1 else 0.0

    @staticmethod
    def calculate_fwt(results: Dict[str, Dict[str, float]], tasks: List[str]) -> float:
        """Calculate forward transfer metric"""
        if len(tasks) <= 1:
            return 0.0
        fwt = 0.0
        n = len(tasks)
        for i in range(1, n):
            task = tasks[i]
            if task in results:
                fwt += results[task]["accuracy"]
        return fwt / (n - 1) if n > 1 else 0.0

    def __init__(self, args: Tuple[ModelArguments, DataArguments, CLEvaluationArguments, FinetuningArguments, CLFinetuningArguments]):
        """Initialize continuous learning evaluator"""
        self.model_args, self.data_args, self.cl_eval_args, self.finetuning_args, self.cl_finetuning_args = args
        self.args_dict = {
            **asdict(self.model_args),
            **asdict(self.data_args),
            **asdict(self.cl_eval_args),
            **asdict(self.finetuning_args),
            **asdict(self.cl_finetuning_args)
        }
        self.tasks = self.cl_eval_args.cl_tasks.split(",") if self.cl_eval_args.cl_tasks else []
        self.dataset_options = self._load_dataset_options()
        
        # Handle multi-adapter mode
        self.using_multi_adapter = self.cl_eval_args.eval_mode == "multi_adapter"

    def _load_dataset_options(self) -> Dict:
        """Load dataset options configuration"""
        options_path_base = None
        if self.cl_eval_args.dataset_options:
            # If a specific path is provided, use it directly (assume it includes extension)
            options_path_base = self.cl_eval_args.dataset_options
            # Remove potential extension to try others
            options_path_base, _ = os.path.splitext(options_path_base)
        else:
            # Default base path without extension
            options_path_base = os.path.join("./data", "dataset_options")

        dataset_options = {}
        loaded = False
        for ext in ['.yaml', '.yml', '.json']:
            options_path = options_path_base + ext
            if os.path.exists(options_path):
                try:
                    with open(options_path, "r", encoding="utf-8") as f:
                        if ext in ['.yaml', '.yml']:
                            dataset_options = yaml.safe_load(f)
                        else:
                            dataset_options = json.load(f)
                    logger.info_rank0(f"Successfully loaded dataset options from: {options_path}")
                    loaded = True
                    break
                except yaml.YAMLError as e:
                    logger.error(f"Error decoding YAML from dataset options file: {options_path}. Error: {e}")
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON from dataset options file: {options_path}. Error: {e}")
                except Exception as e:
                    logger.error(f"Error loading dataset options file {options_path}: {e}")

        if not loaded:
             raise ValueError(f"Dataset options configuration file not found (tried .yaml, .yml, .json): {options_path_base}")

        # Verify all tasks have corresponding configurations
        for task in self.tasks:
            if task not in dataset_options:
                raise ValueError(f"Task {task} not found in dataset_options")

        return dataset_options

    def _run_abscl_selector(self, task: str, dataset_path: str) -> bool:
        """Run ABSCL selector to choose the most suitable Adapters for task dataset"""
        try:
            # Import ABSCL selector module
            from easycl.cl.abscl.abscl_selector import select_adapter
            logger.info(f"Using ABSCL selector to choose the most suitable Adapters for task: {task}")
            
            # Create copy to avoid modifying original parameters
            model_args_copy = copy.deepcopy(self.model_args)
            data_args_copy = copy.deepcopy(self.data_args)
            eval_args_copy = copy.deepcopy(self.cl_eval_args)
            finetuning_args_copy = copy.deepcopy(self.finetuning_args)
            
            # Set task name
            eval_args_copy.task = task
            
            # Run selector
            select_adapter(
                model_args=model_args_copy,
                data_args=data_args_copy,
                training_args=eval_args_copy,  # Note: in select_adapter, training_args is actually eval_args
                finetuning_args=finetuning_args_copy,
                dataset_path=dataset_path,
                multi_adapter_dir=self.cl_eval_args.multi_adapter_dir,
                task_name=task,
                batch_size=self.cl_eval_args.abscl_selector_batch_size
            )
            
            # Check if configuration file was successfully generated
            config_path = os.path.join(self.cl_eval_args.multi_adapter_dir, "multiadapter_selected_config.json")
            if not os.path.exists(config_path):
                logger.error(f"ABSCL selector did not generate configuration file: {config_path}")
                return False
                
            logger.info(f"ABSCL selector has successfully generated configuration file for task {task}")
            
            # Print configuration file contents for confirmation
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                    logger.info(f"Configuration file contains {len(config_data.get('adapters', {}))} adapters")
                    for adapter_name, adapter_info in config_data.get('adapters', {}).items():
                        sample_count = len(adapter_info.get('indices', []))
                        logger.info(f"  - Adapter '{adapter_name}': {sample_count} samples")
            except Exception as e:
                logger.warning(f"Error reading configuration file contents: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error running ABSCL selector: {str(e)}")
            return False

    def _run_dynamic_conpet_selector(self, task: str, dataset_path: str) -> bool:
        """Run Dynamic ConPet selector to choose the most suitable Adapters for task dataset"""
        try:
            # Import Dynamic ConPet selector module
            from easycl.cl.dynamic_conpet.dynamic_conpet_selector import select_adapter_dynamic_conpet
            logger.info(f"Using Dynamic ConPet selector for task: {task}")
            
            # Create copy to avoid modifying original parameters
            model_args_copy = copy.deepcopy(self.model_args)
            data_args_copy = copy.deepcopy(self.data_args)
            eval_args_copy = copy.deepcopy(self.cl_eval_args)
            finetuning_args_copy = copy.deepcopy(self.finetuning_args)
            
            # Set task name
            eval_args_copy.task = task
            
            # Run selector
            select_adapter_dynamic_conpet(
                model_args=model_args_copy,
                data_args=data_args_copy,
                training_args=eval_args_copy,  # Note: in selector, training_args is actually eval_args
                finetuning_args=finetuning_args_copy,
                dataset_path=dataset_path,
                multi_adapter_dir=self.cl_eval_args.multi_adapter_dir,
                task_name=task,
                batch_size=self.cl_eval_args.dynamic_conpet_selector_batch_size
            )
            
            # Check if configuration file was successfully generated
            config_path = os.path.join(self.cl_eval_args.multi_adapter_dir, "multiadapter_selected_config.json")
            if not os.path.exists(config_path):
                logger.error(f"Dynamic ConPet selector did not generate configuration file: {config_path}")
                return False
                
            logger.info(f"Dynamic ConPet selector has successfully generated configuration file for task {task}")
            
            # Print configuration file contents for confirmation
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                    logger.info(f"Configuration file contains {len(config_data.get('adapters', {}))} adapters")
                    for adapter_name, adapter_info in config_data.get('adapters', {}).items():
                        sample_count = len(adapter_info.get('indices', []))
                        logger.info(f"  - Adapter '{adapter_name}': {sample_count} samples")
            except Exception as e:
                logger.warning(f"Error reading configuration file contents: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error running Dynamic ConPet selector: {str(e)}")
            return False

    def evaluate_model(self, model_args_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate model performance on all tasks"""
        results = {}

        # Use the args held by the CLEvaluator instance, which should be benchmark-aware
        base_eval_args = {
            **asdict(self.model_args),
            **asdict(self.data_args),
            **asdict(self.cl_eval_args),
            **asdict(self.finetuning_args),
            **asdict(self.cl_finetuning_args)
        }
        # Remove keys that should not be simply copied if model_args_dict provided them
        # This part might need refinement depending on what model_args_dict actually contains
        # For now, assume base_eval_args is the most up-to-date source
        # logger.info_rank0(f"Using base eval args derived from CLEvaluator instance attributes.")


        for task in self.tasks:
            # Create eval_args dictionary for this task using the benchmark-aware base
            eval_args = copy.deepcopy(base_eval_args) # Use the potentially benchmark-updated args

            # Update task-specific parameters
            eval_args["task"] = task
            task_config = self.dataset_options.get(task) # Use .get for safety
            if task_config is None:
                 logger.error(f"Task {task} configuration not found in dataset_options. Skipping evaluation for this task.")
                 results[task] = {"error": "Task configuration not found."}
                 continue

            # Create independent save directory for each task
            # Ensure the base save_dir comes from the correct source (cl_eval_args)
            base_save_dir = self.cl_eval_args.save_dir
            task_save_dir = os.path.join(base_save_dir, task)
            os.makedirs(task_save_dir, exist_ok=True)
            eval_args["save_dir"] = task_save_dir # Set task-specific save dir

            logger.info_rank0(f"Evaluating task: {task}") # Use info_rank0
            logger.info_rank0(f"Task results will be saved in: {task_save_dir}") # Use info_rank0

            # Ensure multi_adapter_dir is correctly passed from cl_eval_args
            if self.using_multi_adapter:
                eval_args["multi_adapter_dir"] = self.cl_eval_args.multi_adapter_dir
                logger.info_rank0(f"Using multi-adapter mode for task evaluation: {task}") # Use info_rank0
                logger.info_rank0(f"Multi-adapter configuration directory: {eval_args['multi_adapter_dir']}") # Use info_rank0

            # Fix: Ensure adapter_name_or_path is in string format if needed
            if "adapter_name_or_path" in eval_args and isinstance(eval_args["adapter_name_or_path"], list):
                eval_args["adapter_name_or_path"] = ",".join(map(str, eval_args["adapter_name_or_path"])) # Ensure conversion to string
                logger.info_rank0(f"Converted adapter_name_or_path from list to comma-separated string: {eval_args['adapter_name_or_path']}") # Use info_rank0

            # Get test set path for potential selectors (Ensure correct base directory)
            # Use task_dir from cl_eval_args if available, otherwise data_args.dataset_dir
            primary_data_dir = self.cl_eval_args.task_dir if self.cl_eval_args.task_dir else self.data_args.dataset_dir
            if not primary_data_dir:
                 logger.warning(f"No task_dir or dataset_dir found for task {task}. Selectors might fail if they rely on dataset path.")
                 dataset_path = None # Indicate path is unknown
            else:
                 task_name_base = task.split("_")[0] # Assuming format like 'taskname_variant'
                 dataset_path = os.path.join(primary_data_dir, f"{task_name_base}_test.json")
                 logger.info_rank0(f"Expected test dataset path for selectors: {dataset_path}") # Use info_rank0


            # Run selector in multi-adapter mode
            if self.using_multi_adapter:
                if not dataset_path or not os.path.exists(dataset_path):
                    logger.warning(f"Test set file not found for task {task} at '{dataset_path}'. Cannot run selector.") # Use info_rank0
                else:
                    selector_run_attempted = False
                    selector_success = False # Default to failure unless a selector runs successfully

                    # Check Dynamic ConPet (if enabled)
                    if self.cl_eval_args.use_dynamic_conpet_selector:
                        selector_run_attempted = True
                        logger.info(f"Running Dynamic ConPet selector for task {task}...")
                        selector_success = self._run_dynamic_conpet_selector(task, dataset_path)
                        if not selector_success:
                            # If Dynamic ConPet fails, stop immediately
                            raise ValueError(f"Dynamic ConPet selector failed for task {task}. Evaluation aborted.")
                        else:
                            logger.info(f"Dynamic ConPet selector ran successfully, task {task} will use selected Adapters for evaluation")

                    # Only check ABSCL when Dynamic ConPet is not enabled and ABSCL is enabled
                    elif self.cl_eval_args.use_abscl_selector:
                        selector_run_attempted = True
                        logger.info(f"Running ABSCL selector for task {task}...")
                        selector_success = self._run_abscl_selector(task, dataset_path)
                        if not selector_success:
                            # If ABSCL fails, stop immediately
                            raise ValueError(f"ABSCL selector failed for task {task}. Evaluation aborted.")
                        else:
                            logger.info(f"ABSCL selector ran successfully, task {task} will use selected Adapters for evaluation")

                    # If multi-adapter mode is enabled but no selector is enabled
                    if not selector_run_attempted:
                         logger.warning(f"Multi-adapter mode is enabled but no selector is configured or enabled for task {task}. Will continue execution, but this may not be the expected behavior.")

                    # Ensure multi_adapter_dir is included in evaluation parameters
                    eval_args["multi_adapter_dir"] = self.cl_eval_args.multi_adapter_dir


            # Determine if it's a custom dataset or standard dataset
            task_name_base = task.split("_")[0] # Re-get base name
            if task_name_base in ["mmlu", "cmmlu", "ceval"]: # Standard datasets
                # Use original evaluator for standard datasets
                logger.info_rank0(f"Using standard evaluator for task: {task}") # Use info_rank0
                evaluator = Evaluator(eval_args) # Pass the updated eval_args
                evaluator.eval()
            else: # Custom datasets
                # Use continuous learning evaluator for custom datasets
                logger.info_rank0(f"Using continuous learning evaluator for task: {task}") # Use info_rank0

                # Ensure that cl_eval_args within eval_args has the correct task_dir
                # It should already be there from the initial deepcopy if CLEvaluator has it.
                # Double-check:
                if "task_dir" not in eval_args or eval_args["task_dir"] != self.cl_eval_args.task_dir:
                     eval_args["task_dir"] = self.cl_eval_args.task_dir # Ensure it's correct

                # Also ensure dataset_dir is present if needed by CLEvalEvaluator init
                if "dataset_dir" not in eval_args or eval_args["dataset_dir"] != self.data_args.dataset_dir:
                     eval_args["dataset_dir"] = self.data_args.dataset_dir

                evaluator = CLEvalEvaluator(eval_args) # Pass the correctly populated eval_args

                # Ensure correct attributes are set in multi-adapter mode
                if self.using_multi_adapter:
                    logger.info_rank0(f"Enabling multi-adapter mode evaluation for task {task}") # Use info_rank0
                    if not hasattr(evaluator, "using_multi_adapter") or not evaluator.using_multi_adapter:
                        logger.warning("Multi-adapter mode not properly enabled in CLEvalEvaluator, attempting manual enable")
                        evaluator.using_multi_adapter = True

                    # Ensure multi_adapter_dir is correctly passed
                    if not hasattr(evaluator.eval_args, "multi_adapter_dir") and "multi_adapter_dir" in eval_args:
                        evaluator.eval_args.multi_adapter_dir = eval_args["multi_adapter_dir"]
                        logger.info_rank0(f"Manually set evaluator's multi_adapter_dir: {evaluator.eval_args.multi_adapter_dir}") # Use info_rank0

                # Run evaluation
                evaluator.evaluate_custom_dataset()

            # Read evaluation results
            results_path = os.path.join(task_save_dir, "results.json")
            if os.path.exists(results_path):
                try:
                    with open(results_path, "r", encoding="utf-8") as f: # Add encoding
                        results[task] = json.load(f)
                except (IOError, json.JSONDecodeError) as e:
                    logger.error(f"Failed to read results file {results_path}: {e}")
                    results[task] = {"error": f"Failed to read results file: {e}"}
            else:
                logger.error(f"Results file not found after evaluation: {results_path}")
                results[task] = {"error": "Results file not found."}


            # Explicitly delete evaluator and free memory
            logger.info_rank0(f"Task {task} evaluation completed, releasing resources") # Use info_rank0
            del evaluator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Reset memory stats if needed, though often not necessary unless debugging memory issues
                # torch.cuda.reset_max_memory_allocated()
                # torch.cuda.reset_peak_memory_stats()

        return results

    def run(self) -> None:
        """Run continuous learning evaluation"""
        # Create main save directory
        os.makedirs(self.cl_eval_args.save_dir, exist_ok=True)

        # Record evaluation mode
        logger.info(f"Evaluation mode: {self.cl_eval_args.eval_mode}")
        if self.cl_eval_args.eval_mode == "multi_adapter":
            logger.info(f"Multi-adapter directory: {self.cl_eval_args.multi_adapter_dir}")
            
            # Check if multi-adapter configuration directory exists
            if not os.path.exists(self.cl_eval_args.multi_adapter_dir):
                logger.info(f"Creating multi-adapter configuration directory: {self.cl_eval_args.multi_adapter_dir}")
                os.makedirs(self.cl_eval_args.multi_adapter_dir, exist_ok=True)

        if self.cl_eval_args.eval_mode == "single":
            # Single model evaluation
            logger.info("Starting single model evaluation")
            base_results = self.evaluate_model(self.args_dict)
            final_results = {
                "individual_results": base_results
            }
        elif self.cl_eval_args.eval_mode == "multi_adapter":
            # Multi-adapter mode evaluation
            logger.info("Starting multi-adapter mode evaluation")
            multi_adapter_args = copy.deepcopy(self.args_dict)
            # Ensure multi_adapter_dir is included in parameters
            if "multi_adapter_dir" not in multi_adapter_args:
                multi_adapter_args["multi_adapter_dir"] = self.cl_eval_args.multi_adapter_dir
            results = self.evaluate_model(multi_adapter_args)
            final_results = {
                "multi_adapter_results": results
            }
        else:
            # Baseline-finetuned comparison mode
            original_save_dir = self.cl_eval_args.save_dir

            # Evaluate base model
            base_args = copy.deepcopy(self.args_dict)
            base_args["save_dir"] = os.path.join(original_save_dir, "base_model")
            os.makedirs(base_args["save_dir"], exist_ok=True)
            base_args["adapter_name_or_path"] = None
            base_model_results = self.evaluate_model(base_args)

            # Evaluate finetuned model
            finetuned_args = copy.deepcopy(self.args_dict)
            finetuned_args["save_dir"] = os.path.join(original_save_dir, "finetuned_model")
            os.makedirs(finetuned_args["save_dir"], exist_ok=True)
            if self.cl_eval_args.cl_tuning_type == "lora":
                if not self.cl_eval_args.compared_adapter_name_or_path:
                    raise ValueError("compared_adapter_name_or_path must be provided in compare mode when using lora type")
                finetuned_args["adapter_name_or_path"] = self.cl_eval_args.compared_adapter_name_or_path
            else:  # full_model
                if not self.cl_eval_args.compared_model_name_or_path:
                    raise ValueError("compared_model_name_or_path must be provided in compare mode when using full_model type")
                finetuned_args["model_name_or_path"] = self.cl_eval_args.compared_model_name_or_path
            
            finetuned_results = self.evaluate_model(finetuned_args)
            
            # Calculate improvements
            improvements = {}
            for task in self.tasks:
                if task in base_model_results and task in finetuned_results:
                    base_acc = base_model_results[task]["accuracy"]
                    finetuned_acc = finetuned_results[task]["accuracy"]
                    improvements[task] = {
                        "base_accuracy": base_acc,
                        "finetuned_accuracy": finetuned_acc,
                        "absolute_improvement": finetuned_acc - base_acc,
                        "relative_improvement": (finetuned_acc - base_acc) / base_acc * 100 if base_acc > 0 else 0
                    }

            final_results = {
                "base_model_results": base_model_results,
                "finetuned_model_results": finetuned_results,
                "improvements": improvements
            }

        # Add continuous learning metrics
        if self.cl_eval_args.calculate_cl_metrics:
            metrics = {}
            if self.cl_eval_args.eval_mode == "single":
                results_to_analyze = final_results.get("individual_results", {})
            elif self.cl_eval_args.eval_mode == "multi_adapter":
                results_to_analyze = final_results.get("multi_adapter_results", {})
            else:  # compare mode
                results_to_analyze = final_results.get("finetuned_model_results", {})
                
            # Calculate all metrics if calculate_cl_metrics is True
            metrics["transfer"] = self.calculate_transfer(results_to_analyze)
            metrics["bwt"] = self.calculate_bwt(results_to_analyze, self.tasks)
            metrics["fwt"] = self.calculate_fwt(results_to_analyze, self.tasks)
            final_results["cl_metrics"] = metrics

        # Save final results
        results_path = os.path.join(self.cl_eval_args.save_dir, "cl_results.json")
        with open(results_path, "w") as f:
            json.dump(final_results, f, indent=2)