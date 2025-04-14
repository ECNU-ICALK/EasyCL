import os
import json
import logging
from typing import List, Optional, Dict, Any

# Use the same logger as the main workflow module if possible, or create a new one
# Assuming get_logger is available or use standard logging
try:
    from llamafactory.extras.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class BenchmarkHandler:
    """
    Handles loading and parsing of benchmark definitions.
    Reads benchmark_info.json to determine task order and related paths.
    """

    def __init__(self, benchmark_name: str, benchmark_order: str, benchmark_dir: str):
        """
        Initializes the BenchmarkHandler.

        Args:
            benchmark_name: The name of the benchmark to run (e.g., "MyBenchmark").
            benchmark_order: The specific task order within the benchmark (e.g., "order1").
            benchmark_dir: The directory containing the benchmark definition (benchmark_info.json)
                           and associated data/options files.

        Raises:
            FileNotFoundError: If benchmark_dir or benchmark_info.json does not exist.
            ValueError: If benchmark_name or benchmark_order is invalid or not found in the definition.
            json.JSONDecodeError: If benchmark_info.json is not valid JSON.
        """
        self.benchmark_name = benchmark_name
        self.benchmark_order = benchmark_order
        self.benchmark_dir = benchmark_dir
        self._ordered_tasks: List[str] = []
        self._dataset_options_path: Optional[str] = None

        # Use info_rank0 if available, otherwise fallback to info
        log_func = getattr(logger, 'info_rank0', logger.info)
        log_func(f"Initializing BenchmarkHandler for benchmark '{benchmark_name}', order '{benchmark_order}' in directory '{benchmark_dir}'")

        if not os.path.isdir(benchmark_dir):
            raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")

        benchmark_info_path = os.path.join(benchmark_dir, "benchmark_info.json")
        if not os.path.isfile(benchmark_info_path):
            raise FileNotFoundError(f"Benchmark definition file not found: {benchmark_info_path}")

        # Load benchmark definition
        try:
            with open(benchmark_info_path, 'r', encoding='utf-8') as f:
                benchmark_data: Dict[str, Any] = json.load(f)
            log_func(f"Successfully loaded benchmark definition from {benchmark_info_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {benchmark_info_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error reading benchmark definition file {benchmark_info_path}: {e}")
            raise

        # Validate benchmark name
        if benchmark_name not in benchmark_data:
            valid_benchmarks = list(benchmark_data.keys())
            raise ValueError(f"Benchmark '{benchmark_name}' not found in {benchmark_info_path}. Available benchmarks: {valid_benchmarks}")

        benchmark_config = benchmark_data[benchmark_name]

        # Validate benchmark order
        orders = benchmark_config.get("orders", {})
        if benchmark_order not in orders:
            valid_orders = list(orders.keys())
            raise ValueError(f"Benchmark order '{benchmark_order}' not found for benchmark '{benchmark_name}'. Available orders: {valid_orders}")

        # Get ordered tasks
        self._ordered_tasks = orders[benchmark_order]
        if not isinstance(self._ordered_tasks, list) or not all(isinstance(task, str) for task in self._ordered_tasks):
            raise ValueError(f"Invalid task list format for benchmark '{benchmark_name}', order '{benchmark_order}'. Expected a list of strings.")
        if not self._ordered_tasks:
             raise ValueError(f"Task list for benchmark '{benchmark_name}', order '{benchmark_order}' cannot be empty.")

        log_func(f"Using task order for '{benchmark_name}/{benchmark_order}': {', '.join(self._ordered_tasks)}")

        # Find dataset_options file (check for yaml, yml, json)
        options_found = False
        for ext in ['.yaml', '.yml', '.json']:
            potential_path = os.path.join(self.benchmark_dir, f"dataset_options{ext}")
            if os.path.isfile(potential_path):
                self._dataset_options_path = potential_path
                log_func(f"Found dataset options file: {self._dataset_options_path}")
                options_found = True
                break

        if not options_found:
            logger.warning(f"No dataset_options file (.yaml, .yml, or .json) found in benchmark directory: {self.benchmark_dir}")


    def get_ordered_tasks(self) -> List[str]:
        """Returns the ordered list of task names for the specified benchmark and order."""
        return self._ordered_tasks

    def get_benchmark_dir(self) -> str:
        """Returns the path to the benchmark directory."""
        return self.benchmark_dir

    def get_dataset_options_path(self) -> Optional[str]:
        """
        Returns the path to the dataset_options file within the benchmark directory,
        if found (checks for .yaml, .yml, .json). Otherwise, returns None.
        """
        return self._dataset_options_path
