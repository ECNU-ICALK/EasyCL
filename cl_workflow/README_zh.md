[English](README.md)

# LLaMA Factory 持续学习工作流

该目录包含 LLaMA Factory 中一键式持续学习 (CL) 工作流的实现。它允许用户使用各种 CL 策略，轻松地在多个任务上运行顺序训练和评估。

## 目录

- [1. 工作流模式](#1-工作流模式)
- [2. 代码结构](#2-代码结构)
- [3. 示例用法](#3-示例用法)
- [4. CL 方法配置](#4-cl-方法配置)
- [5. CL 评估指标](#5-cl-评估指标)
- [6. CL 评估超参数](#6-cl-评估超参数)

## 1. 工作流模式

`cl_workflow` 脚本支持多种执行模式：

- **`train_only`**: 根据提供的训练配置，顺序执行每个指定任务的训练循环。根据所选的 CL 方法，自动管理任务之间的依赖关系（例如，传递模型输出、任务 ID）。
- **`eval_only`**: 根据提供的评估配置，执行评估循环。它评估指定的模型（基础模型或特定任务的输出）在定义的 CL 任务上的表现。
- **`train_then_eval`**: 首先按顺序执行所有任务的完整训练循环（与 `train_only` 相同），然后在最后一个任务训练完成后执行评估。它通常会评估初始模型状态和每个任务后的模型状态。
- **`full_workflow`**: 类似于 `train_then_eval`，它运行完整的训练序列，然后运行评估序列。此外，如果基础模型和所有任务步骤的评估结果都成功生成，它会计算并保存标准的持续学习指标（Last, Avg, BWT, FWT）。

## 2. 代码结构

`src/llamafactory/cl_workflow` 内的主要组件包括：

- **`cl_train_and_eval.py`**: 工作流的主要入口点 (`llamafactory-cli cl_workflow`)。它解析参数，加载配置，使用 `CLCommandGenerator` 生成必要的训练和评估命令，并根据所选的 `mode` 来协调执行。
- **`evaluator.py`**: 包含 `CLEvaluator` 类，负责管理跨多个 CL 任务或特定评估模式（如 'single', 'multi_adapter'）的评估过程。它处理加载模型/适配器，并为 `cl_tasks` 中定义的每个任务运行评估。
- **`cl_params_config.json`**: 一个关键的 JSON 配置文件，定义了：
  - 支持的 CL 方法 (`cl_methods_registry`)
  - 每种方法的参数要求 (`cl_method_requirements`，例如，是否需要前一个模型的路径或重放列表）
  - 任务间自动参数映射的规则 (`task_output_mapping`, `default_param_mappings`，例如，自动将 `previous_task_model` 设置为前一个任务的 `output_dir`)
  - 处理增量参数的规则 (`incremental_params`，例如，为 `replay_task_list` 累积任务数据集）
  - 第一个任务的特殊参数处理 (`first_task_params`, `first_task_special_params`)
- **`cl_eval/` (目录)**: 包含与持续学习评估过程特别相关的模块：
  - **`cl_metrics.py`**: 实现 `CLMetricsCalculator`，用于根据跨任务的评估结果计算标准 CL 指标（Last, Avg, BWT, FWT）
  - **`adapters.py`**: 提供适配器以在评估期间处理不同的数据集格式（例如，将 Alpaca 格式转换为 MMLU 风格）

## 3. 示例用法

您可以使用 `llamafactory-cli cl_workflow` 命令运行工作流。您需要以 JSON 格式提供训练和/或评估参数的配置文件。

### 常用参数

- **`--clean_dir`**: 在执行工作流之前清理输出目录。如果指定此参数，将删除所有现有的输出文件。
- **`--preview_only`**: 仅预览将要执行的命令，而不实际执行它们。这对于检查工作流配置是否正确很有用。

### 仅训练

```bash
llamafactory-cli cl_workflow --mode train_only --train_params ./configs/train_config.json
```

**预览结果**: 顺序执行 `train_config_ewc.json` 中定义的任务的训练命令，并在任务之间应用参数管理。

### 仅评估

```bash
llamafactory-cli cl_workflow --mode eval_only --eval_params ./configs/eval_config.json
```

**预览结果**: 执行 `eval_config.json` 中指定的评估命令（例如，在 `cl_tasks` 上评估特定的微调模型）。

### 训练后评估

```bash
llamafactory-cli cl_workflow --mode train_then_eval \
    --train_params ./configs/train_config.json \
    --eval_params ./configs/eval_config.json
```

**预览结果**: 顺序执行训练命令，然后执行评估命令（评估基础模型和每个任务后的模型）。

### 完整工作流 (训练、评估、计算指标)

```bash
llamafactory-cli cl_workflow --mode full_workflow \
    --train_params ./configs/train_config.json \
    --eval_params ./configs/eval_config.json
```

**预览结果**: 顺序执行持续学习训练，然后评估基础/任务模型，最后计算 CL 指标（Last, Avg, BWT, FWT）并保存到评估输出目录。

## 4. CL 方法配置

持续学习方法及其特定的参数要求通过 `cl_params_config.json` 进行管理。

- **注册**: CL 方法在 `cl_methods_registry`下列出。如果训练参数中存在相应的 `use_<method_name>: true` 标志，工作流也可以检测到预期的 CL 方法。
- **自动映射**: 工作流根据 `cl_method_requirements` 中的规则以及 `task_output_mapping` 和 `default_param_mappings` 中定义的映射，自动管理依赖于先前任务的参数。例如：
  - 如果方法的 `needs_prev_model` 为 true，工作流通常会将 `previous_task_model` 或 `adapter_name_or_path` 等参数设置为前一个任务的 `output_dir`
  - 如果 `needs_replay_list` 为 true，`replay_task_list` 参数会自动填充为先前任务数据集名称的逗号分隔列表
- **增量映射**: `incremental_params` 部分定义了参数如何累积信息（例如，EWC 的 `previous_task_data` 如何累积）
- **首个任务**: `first_task_params` 定义了为第一个任务移除哪些依赖参数，而 `first_task_special_params` 为某些方法的参数设置了特定的初始值（例如，在第一个任务上为 O-LoRA 设置 `orthogonal_lambda` 为 0）

此配置允许工作流根据训练配置中定义的所选 CL 策略调整其行为和参数传递。

## 5. CL 评估指标

当 `mode='full_workflow'` 或在评估配置中设置了 `calculate_cl_metrics: true` 时，工作流可以使用 `CLMetricsCalculator` 计算标准的持续学习指标。

支持以下指标：

### 最终准确率 (Last)

模型在完成最后一个（第 N 个）任务的训练*后*，在所有任务上的平均准确率。衡量最终的整体性能。

\[ \text{Last} = \frac{1}{N} \sum_{i=1}^{N} R_{N,i} \]

其中 \( R_{k,i} \) 是在任务 \( k \) 上训练后在任务 \( i \) 上的准确率，\( N \) 是任务总数。

### 平均准确率 (Avg)

在 CL 过程的每一步中，模型在*已学习*任务上的平均准确率的均值。反映了整个学习序列中的整体性能。

\[ \text{Avg} = \frac{1}{N} \sum_{k=1}^{N} \left( \frac{1}{k} \sum_{i=1}^{k} R_{k,i} \right) \]

### 向后迁移 (BWT)

衡量学习新任务对先前学习任务性能的影响（通常表示遗忘）。它是任务刚学习完后的准确率与学习完所有后续任务后的准确率之间的平均差异。

\[ \text{BWT} = \frac{1}{N-1} \sum_{i=1}^{N-1} (R_{N,i} - R_{i,i}) \]

负的 BWT 通常表示遗忘。

### 向前迁移 (FWT)

衡量先前学到的知识对学习新任务的影响。它是学习了先前任务后在某个任务上的准确率与使用初始基础模型在该任务上的准确率（\( R_{0,i} \)）之间的平均差异。

\[ \text{FWT} = \frac{1}{N-1} \sum_{i=2}^{N} (R_{i-1,i} - R_{0,i}) \]

正的 FWT 表示先前的学习有助于提高在新任务上的性能（相比于从头开始）。

### 顽固性 (IM)

*注意：此指标目前**不会**被工作流计算*，因为它需要每个任务单独训练的基线结果 (\( R_{k,k}^* \))，而标准的顺序工作流不会生成这些结果。

### 计算时机

仅当满足以下条件时，这些指标才会被计算并保存到主评估目录下的 `cl_metrics.json` 文件中：

1. 工作流模式为 `full_workflow`
2. 基础模型（在 `eval_results/base/` 中）和每个任务步骤（在 `eval_results/task_k/` 中）的评估结果 (`results.json`) 都存在

## 6. CL 评估超参数

控制 CL 评估过程的关键超参数（主要在通过 `--eval_params` 传递的评估 JSON 文件中配置）包括：

| 参数 | 类型 | 描述 |
|------|------|------|
| `cl_tasks` | 字符串 | 需要评估的任务/数据集名称的逗号分隔列表（例如，"task1,task2,task3"） |
| `eval_mode` | 字符串 | 评估策略：<br>- `single`: 在所有 `cl_tasks` 上评估单个模型检查点<br>- `multi_adapter`: 对每个任务使用多适配器设置<br>- `compare`: 评估基础模型和指定的微调模型/适配器 |
| `save_dir` | 字符串 | 保存评估结果的基础目录 |
| `calculate_cl_metrics` | 布尔值 | 是否计算 CL 指标（Last, Avg, BWT, FWT） |
| `cl_tuning_type` | 字符串 | 在 `compare` 模式下使用，指定比较的模型是 LoRA (`lora`) 还是完整模型 (`full_model`) |
| `compared_adapter_name_or_path` | 字符串 | 在 `compare` 模式下且 `cl_tuning_type='lora'` 时，用于与基础模型进行比较的 LoRA 适配器的路径 |
| `compared_model_name_or_path` | 字符串 | 在 `compare` 模式下且 `cl_tuning_type='full_model'` 时，用于与基础模型进行比较的完整模型的路径 |
| `multi_adapter_dir` | 字符串 | 包含多适配器配置的目录路径 |
| `use_abscl_selector` | 布尔值 | 在 `multi_adapter` 模式下是否使用 ABSCL 选择器来选择适配器 |
| `use_dynamic_conpet_selector` | 布尔值 | 在 `multi_adapter` 模式下是否使用 Dynamic ConPet 选择器来选择适配器 |
| `abscl_selector_batch_size` | 整数 | ABSCL 选择器的批处理大小 |
| `dynamic_conpet_selector_batch_size` | 整数 | Dynamic ConPet 选择器的批处理大小 |

通用的评估参数，如 `batch_size`, `n_shot`（来自 `EvaluationArguments`）也适用。 