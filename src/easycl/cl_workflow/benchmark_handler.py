import os
import json
import copy
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Attempt to import yaml and handle ImportError gracefully
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from llamafactory.extras.logging import get_logger

logger = get_logger(__name__)

class BenchmarkHandler:
    """Handles loading and processing of benchmark configurations."""

    def __init__(self, benchmark_name: str, benchmark_order: str, benchmark_dir: str):
        """
        Initializes the BenchmarkHandler.

        Args:
            benchmark_name: The name of the benchmark (e.g., 'testbench').
            benchmark_order: The specific task order key (e.g., 'order1').
            benchmark_dir: The root directory of the benchmark.
        """
        if not benchmark_name or not benchmark_order or not benchmark_dir:
            raise ValueError("Benchmark name, order, and directory must be provided.")

        self.benchmark_name = benchmark_name
        self.benchmark_order = benchmark_order
        # Ensure benchmark_dir is an absolute path for robustness
        self.benchmark_dir = os.path.abspath(benchmark_dir)
        self.benchmark_info = None

        logger.info(f"Initializing BenchmarkHandler for benchmark: '{self.benchmark_name}', order: '{self.benchmark_order}', dir: '{self.benchmark_dir}'")

        self._load_benchmark_info()
        self._validate_benchmark_info()

    def _load_benchmark_info(self):
        """Loads the benchmark_info.json file from the benchmark directory."""
        info_file_path = os.path.join(self.benchmark_dir, "benchmark_info.json")
        logger.info(f"Attempting to load benchmark info from: {info_file_path}")

        if not os.path.exists(info_file_path):
            logger.error(f"Benchmark info file not found: {info_file_path}")
            raise FileNotFoundError(f"Benchmark info file not found: {info_file_path}")

        try:
            with open(info_file_path, 'r', encoding='utf-8') as f:
                self.benchmark_info = json.load(f)
            logger.info(f"Successfully loaded benchmark info for '{self.benchmark_name}'")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {info_file_path}: {e}")
            raise ValueError(f"Invalid JSON format in benchmark info file: {info_file_path}") from e
        except Exception as e:
            logger.error(f"Failed to load benchmark info file {info_file_path}: {e}")
            raise

    def _validate_benchmark_info(self):
        """Validates the loaded benchmark information."""
        if not self.benchmark_info:
            raise ValueError("Benchmark info not loaded.")

        # Check top-level structure (should contain the benchmark name as key)
        if self.benchmark_name not in self.benchmark_info:
            logger.error(f"Benchmark name '{self.benchmark_name}' not found as a top-level key in {os.path.join(self.benchmark_dir, 'benchmark_info.json')}")
            raise ValueError(f"Benchmark name '{self.benchmark_name}' mismatch in info file.")

        benchmark_data = self.benchmark_info[self.benchmark_name]

        # Check for essential keys within the benchmark data
        required_keys = ["description", "orders"]
        for key in required_keys:
            if key not in benchmark_data:
                logger.error(f"Missing required key '{key}' in benchmark info for '{self.benchmark_name}'.")
                raise ValueError(f"Invalid benchmark info format: Missing key '{key}'")

        # Check if the requested order exists
        if self.benchmark_order not in benchmark_data["orders"]:
            available_orders = list(benchmark_data["orders"].keys())
            logger.error(f"Requested benchmark order '{self.benchmark_order}' not found for benchmark '{self.benchmark_name}'. Available orders: {available_orders}")
            raise ValueError(f"Benchmark order '{self.benchmark_order}' not defined.")

        # Check if the order contains a list of tasks
        if not isinstance(benchmark_data["orders"][self.benchmark_order], list) or not benchmark_data["orders"][self.benchmark_order]:
            logger.error(f"Benchmark order '{self.benchmark_order}' for '{self.benchmark_name}' does not contain a valid list of tasks.")
            raise ValueError(f"Invalid task list for order '{self.benchmark_order}'.")

        logger.info(f"Benchmark info for '{self.benchmark_name}' validated successfully.")

    def get_task_sequence(self) -> List[str]:
        """Returns the list of tasks for the specified benchmark order."""
        if not self.benchmark_info:
            raise RuntimeError("Benchmark info not loaded.") # Should not happen if init succeeded

        try:
            tasks = self.benchmark_info[self.benchmark_name]["orders"][self.benchmark_order]
            logger.info(f"Retrieved task sequence for order '{self.benchmark_order}': {tasks}")
            return tasks
        except KeyError:
            # This should be caught by validation, but added for safety
            logger.error(f"Failed to retrieve task sequence for benchmark '{self.benchmark_name}', order '{self.benchmark_order}'. Check benchmark_info.json structure.")
            raise ValueError("Could not retrieve task sequence.")

    def get_benchmark_configs(
        self,
        original_train_kwargs: Dict[str, Any],
        original_eval_kwargs: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Overrides original training and evaluation configurations with benchmark settings.

        Args:
            original_train_kwargs: The original training arguments dictionary.
            original_eval_kwargs: The original evaluation arguments dictionary.

        Returns:
            A tuple containing the modified training and evaluation kwargs.
        """
        logger.info("Applying benchmark configurations...")
        modified_train_kwargs = copy.deepcopy(original_train_kwargs)
        modified_eval_kwargs = copy.deepcopy(original_eval_kwargs)

        tasks = self.get_task_sequence()
        task_string = ",".join(tasks)

        # --- Override Task Lists ---
        modified_train_kwargs['dataset'] = task_string
        modified_eval_kwargs['cl_tasks'] = task_string
        logger.info(f"Overriding 'dataset' and 'cl_tasks' with benchmark tasks: {task_string}")

        # --- Override Training Path ---
        modified_train_kwargs['dataset_dir'] = self.benchmark_dir
        modified_train_kwargs['media_dir'] = self.benchmark_dir
        logger.info(f"Overriding 'dataset_dir' for training to: {self.benchmark_dir}")
        logger.info(f"Overriding 'media_dir' for training to: {self.benchmark_dir}")

        # --- Override Evaluation Paths ---
        # Set task_dir to the benchmark root. CLEvalEvaluator will handle subdirs.
        modified_eval_kwargs['task_dir'] = self.benchmark_dir
        modified_eval_kwargs['media_dir'] = self.benchmark_dir
        logger.info(f"Setting 'task_dir' for evaluation to benchmark root: {self.benchmark_dir}")
        logger.info(f"Setting 'media_dir' for evaluation to benchmark root: {self.benchmark_dir}")

        # --- Add Benchmark Markers ---
        modified_eval_kwargs['is_benchmark_mode'] = True
        modified_eval_kwargs['benchmark_dir_internal'] = self.benchmark_dir
        logger.info("Adding internal benchmark markers to evaluation args.")

        logger.info("Benchmark configurations applied.")
        return modified_train_kwargs, modified_eval_kwargs