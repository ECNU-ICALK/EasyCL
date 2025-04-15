[中文](README_zh.md)

# EasyCL Continual Learning Workflow

This directory contains the implementation for the one-click continual learning (CL) workflow in EasyCL. It allows users to easily run sequential training and evaluation across multiple tasks using various CL strategies.

## Table of Contents

- [1. Workflow Modes](#1-workflow-modes)
- [2. Code Structure](#2-code-structure)
- [3. Example Usage](#3-example-usage)
- [4. CL Method Configuration](#4-cl-method-configuration)
- [5. CL Evaluation Metrics](#5-cl-evaluation-metrics)
- [6. CL Evaluation Hyperparameters](#6-cl-evaluation-hyperparameters)

## 1. Workflow Modes

The `cl_workflow` script supports several execution modes:

- **`train_only`**: Executes the training loop for each specified task sequentially based on the provided training configuration. Manages dependencies between tasks (e.g., passing model outputs, task IDs) automatically based on the chosen CL method.
- **`eval_only`**: Executes the evaluation loop based on the provided evaluation configuration. It evaluates a specified model (either a base model or a specific task's output) on the defined CL tasks.
- **`train_then_eval`**: First executes the full training loop for all tasks sequentially (same as `train_only`), and then performs evaluation after the final task is trained. It typically evaluates the initial model state and the model state after each task.
- **`full_workflow`**: Similar to `train_then_eval`, it runs the complete training sequence followed by the evaluation sequence. Additionally, if evaluation results are successfully generated for the base model and all task steps, it calculates and saves standard continual learning metrics (Last, Avg, BWT, FWT).

## 2. Code Structure

The main components within `src/easycl/cl_workflow` are:

- **`cl_train_and_eval.py`**: The main entry point for the workflow (`easycl-cli cl_workflow`). It parses arguments, loads configurations, generates the necessary training and evaluation commands using `CLCommandGenerator`, and orchestrates their execution based on the selected `mode`.
- **`evaluator.py`**: Contains the `CLEvaluator` class, responsible for managing the evaluation process across multiple CL tasks or specific evaluation modes (like 'single', 'multi_adapter'). It handles loading models/adapters and running evaluations for each task defined in `cl_tasks`.
- **`cl_params_config.json`**: A crucial JSON configuration file that defines:
  - Supported CL methods (`cl_methods_registry`)
  - Parameter requirements for each method (`cl_method_requirements`, e.g., whether it needs the previous model's path or a replay list)
  - Rules for automatic parameter mapping between tasks (`task_output_mapping`, `default_param_mappings`, e.g., automatically setting `previous_task_model` to the previous task's `output_dir`)
  - Rules for handling incremental parameters (`incremental_params`, e.g., accumulating task datasets for `replay_task_list`)
  - Special parameter handling for the very first task (`first_task_params`, `first_task_special_params`)
- **`cl_eval/` (Directory)**: Contains modules specifically related to the continual learning evaluation process:
  - **`cl_metrics.py`**: Implements `CLMetricsCalculator` to compute standard CL metrics (Last, Avg, BWT, FWT) from evaluation results across tasks
  - **`adapters.py`**: Provides adapters to handle different dataset formats (e.g., converting Alpaca format to MMLU style) during evaluation

## 3. Example Usage

You can run the workflow using the `easycl-cli cl_workflow` command. You need to provide configuration files for training and/or evaluation parameters in YAML format.

### Common Parameters

- **`--clean_dirs`**: Clean the output directory before executing the workflow. If specified, all existing output files will be deleted.
- **`--previewonly`**: Only preview the commands that would be executed without actually running them. This is useful for checking if the workflow configuration is correct.

### Train Only

```bash
easycl-cli cl_workflow --mode train_only --train_params ./example/train_examples/lora_example.yaml
```

**Preview Result**: Executes training commands sequentially for tasks defined in `train_config.json`, applying parameter management between tasks.

### Evaluate Only

```bash
easycl-cli cl_workflow --mode eval_only --eval_params ./example/eval_examples/lora_eval.yaml
```

**Preview Result**: Executes evaluation command(s) specified in `eval_config.json` (e.g., evaluating a specific fine-tuned model on `cl_tasks`).

### Train Then Evaluate

```bash
easycl-cli cl_workflow --mode train_then_eval \
    --train_params ./example/train_examples/lora_example.yaml \
    --eval_params ./example/eval_examples/lora_eval.yaml
```

**Preview Result**: Executes training commands sequentially, then executes evaluation commands (evaluating base model and model after each task).

### Full Workflow (Train, Evaluate, Calculate Metrics)

```bash
easycl-cli cl_workflow --mode full_workflow \
    --train_params ./example/train_examples/lora_example.yaml \
    --eval_params ./example/eval_examples/lora_eval.yaml
```

**Preview Result**: Executes training sequentially, then evaluates base/task models, and finally calculates and saves CL metrics (Last, Avg, BWT, FWT) to the evaluation output directory.

## 4. CL Method Configuration

Continual learning methods and their specific parameter requirements are managed through `cl_params_config.json`.

- **Registration**: CL methods are listed under `cl_methods_registry`. The workflow can also detect the intended CL method if a corresponding `use_<method_name>: true` flag is present in the training parameters.
- **Automatic Mapping**: The workflow automatically manages parameters that depend on previous tasks based on the rules in `cl_method_requirements` and the mappings defined in `task_output_mapping` and `default_param_mappings`. For example:
  - If `needs_prev_model` is true for a method, the workflow typically sets a parameter like `previous_task_model` or `adapter_name_or_path` to the `output_dir` of the preceding task
  - If `needs_replay_list` is true, the `replay_task_list` parameter is automatically populated with a comma-separated list of previous task dataset names
- **Incremental Mapping**: The `incremental_params` section defines how parameters accumulate information (e.g., how `previous_task_data` for EWC accumulates)
- **First Task**: `first_task_params` defines which dependency parameters are removed for the first task, and `first_task_special_params` sets specific initial values for certain method parameters (e.g., setting `orthogonal_lambda` to 0 for O-LoRA on the first task)

This configuration allows the workflow to adapt its behavior and parameter passing based on the selected CL strategy defined in the training config.

## 5. CL Evaluation Metrics

The workflow can calculate standard continual learning metrics using `CLMetricsCalculator` when `mode='full_workflow'` or if `calculate_cl_metrics: true` is set in the evaluation configuration.

### Last Accuracy (Last)

The average accuracy across all tasks *after* the model has finished training on the final (N-th) task. Measures the final overall performance.

\[ \text{Last} = \frac{1}{N} \sum_{i=1}^{N} R_{N,i} \]

Where \( R_{k,i} \) is the accuracy on task \( i \) after training on task \( k \), and \( N \) is the total number of tasks.

### Average Accuracy (Avg)

The average of the mean accuracy across *learned* tasks at each step of the CL process. Reflects the overall performance throughout the learning sequence.

\[ \text{Avg} = \frac{1}{N} \sum_{k=1}^{N} \left( \frac{1}{k} \sum_{i=1}^{k} R_{k,i} \right) \]

### Backward Transfer (BWT)

Measures the influence that learning a new task has on the performance of previously learned tasks (often indicates forgetting). It's the average difference between the accuracy on a task right after learning it versus after learning all subsequent tasks.

\[ \text{BWT} = \frac{1}{N-1} \sum_{i=1}^{N-1} (R_{N,i} - R_{i,i}) \]

A negative BWT typically indicates forgetting.

### Forward Transfer (FWT)

Measures the influence of previously learned knowledge on learning new tasks. It's the average difference between the accuracy on a task after learning preceding tasks versus the accuracy on that task using the initial base model (\( R_{0,i} \)).

\[ \text{FWT} = \frac{1}{N-1} \sum_{i=2}^{N} (R_{i-1,i} - R_{0,i}) \]

A positive FWT indicates that previous learning helped improve performance on new tasks compared to starting from scratch.

### Intransigence (IM)

*Note: This metric is currently **not** calculated* by the workflow, as it requires baseline results for each task trained individually (\( R_{k,k}^* \)), which are not generated by the standard sequential workflow.

### When Calculated

These metrics are computed and saved to `cl_metrics.json` in the main evaluation directory only if:

1. The workflow mode is `full_workflow`
2. Evaluation results (`results.json`) exist for the base model (in `eval_results/base/`) and for each task step (in `eval_results/task_k/`)

## 6. CL Evaluation Hyperparameters

Key hyperparameters controlling the CL evaluation process (primarily configured in the evaluation JSON file passed via `--eval_params`) include:

| Parameter | Type | Description |
|-----------|------|-------------|
| `cl_tasks` | String | Comma-separated list of task/dataset names to evaluate on (e.g., "task1,task2,task3") |
| `eval_mode` | String | Evaluation strategy:<br>- `single`: Evaluate a single model checkpoint across all `cl_tasks`<br>- `multi_adapter`: Evaluate using a multi-adapter setup<br>- `compare`: Evaluate both a base model and a specified fine-tuned model/adapter |
| `save_dir` | String | Base directory where evaluation results will be saved |
| `calculate_cl_metrics` | Boolean | Whether to calculate CL metrics (Last, Avg, BWT, FWT) |
| `cl_tuning_type` | String | Used in `compare` mode to specify if the compared model is LoRA (`lora`) or a full model (`full_model`) |
| `compared_adapter_name_or_path` | String | Path to the LoRA adapter to compare against the base model in `compare` mode with `cl_tuning_type='lora'` |
| `compared_model_name_or_path` | String | Path to the full model to compare against the base model in `compare` mode with `cl_tuning_type='full_model'` |
| `multi_adapter_dir` | String | Path to the directory containing multi-adapter configurations |
| `use_abscl_selector` | Boolean | Whether to use the ABSCL selector to choose adapters in `multi_adapter` mode |
| `use_dynamic_conpet_selector` | Boolean | Whether to use the Dynamic ConPet selector to choose adapters in `multi_adapter` mode |
| `abscl_selector_batch_size` | Integer | Batch size for the ABSCL selector |
| `dynamic_conpet_selector_batch_size` | Integer | Batch size for the Dynamic ConPet selector |

General evaluation arguments like `batch_size`, `n_shot` (from `EvaluationArguments`) are also applicable.
