8<!-- Logo -->
<p align="center">
  <img src="assets/logo.png" alt="EasyCL Logo" style="width: 100%;" />
</p>

[ [English](README.md) | [中文](README_zh.md) ]

## Table of Contents

- [Introduction](#introduction)
- [Implemented Methods](#implemented-methods)
- [Installation](#installation)
- [Dataset Format Requirements](#dataset-format-requirements)
  - [Data Format](#data-format)
  - [Alpaca Format](#alpaca-format)
  - [Sharegpt Format](#sharegpt-format)
  - [Continuous Learning Evaluation](#continuous-learning-evaluation)
- [Workflow](#workflow)
  - [Train Only](#train-only)
  - [Evaluate Only](#evaluate-only)
  - [Train Then Evaluate](#train-then-evaluate)
  - [Full Workflow](#full-workflow-train-evaluate-calculate-metrics)
- [License](#license)

## Introduction

EasyCL is an extension of the LLaMA Factory framework, focusing on continual learning methods for large language models. It provides a comprehensive suite of tools and methods to address the problem of catastrophic forgetting in sequential learning tasks.

The framework integrates a variety of state-of-the-art continual learning techniques designed specifically for language models, allowing researchers and practitioners to easily implement, compare, and develop new methods.

For detailed implementation of the continual learning workflow, see [src/easycl/cl_workflow/README.md](src/easycl/cl_workflow/README.md).

## Implemented Methods

1. **Elastic Weight Consolidation (EWC)** - [View Implementation](src/easycl/cl/ewc/README.md) - [Overcoming catastrophic forgetting in neural networks](https://www.pnas.org/doi/pdf/10.1073/pnas.1611835114)

2. **Learning Without Forgetting (LWF)** - [View Implementation](src/easycl/cl/lwf/README.md) - [Learning without forgetting](https://ieeexplore.ieee.org/ielaam/34/8520726/8107520-aam.pdf)

3. **Experience Replay** - [View Implementation](src/easycl/cl/replay/README.md) - [Experience replay for continual learning](https://proceedings.neurips.cc/paper_files/paper/2019/file/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Paper.pdf)

4. **LAMOL (Language Modeling for Lifelong Language Learning)** - [View Implementation](src/easycl/cl/lamol/README.md) - [LAMOL: LAnguage MOdeling for Lifelong Language Learning](https://arxiv.org/pdf/1909.03329)

5. **O-LoRA (Orthogonal subspace learning)** - [View Implementation](src/easycl/cl/olora/README.md) - [Orthogonal subspace learning for language model continual learning](https://arxiv.org/pdf/2310.14152)

6. **Gradient Episodic Memory (GEM)** - [View Implementation](src/easycl/cl/gem/README.md) - [Gradient Episodic Memory for Continual Learning](https://proceedings.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf)

7. **I-LoRA (Interpolation-based LoRA)** - [View Implementation](src/easycl/cl/ilora/README.md) - [Analyzing and Reducing Catastrophic Forgetting in Parameter Efficient Tuning](https://arxiv.org/pdf/2402.18865)

8. **MOE-LoRA (Mixture of Experts with LoRA)** - [View Implementation](src/easycl/cl/moelora/README.md) - [CoIN: A Benchmark of Continual Instruction Tuning for Multimodel Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/file/6a45500d9eda640deed90d8a62742be5-Paper-Datasets_and_Benchmarks_Track.pdf)

9. **ABSCL (ABSA LLM-CL)** - [View Implementation](src/easycl/cl/abscl/README.md) - [Boosting Large Language Models with Continual Learning for Aspect-based Sentiment Analysis](https://arxiv.org/pdf/2405.05496)

10. **Dynamic ConPet** - [View Implementation](src/easycl/cl/dynamic_conpet/README.md) - [ConPET: Continual Parameter-Efficient Tuning for Large Language Models](https://arxiv.org/pdf/2309.14763)

11. **CLIT-MoE (Continual Learning with Task-specific MoE)** - [View Implementation](src/easycl/cl/clmoe/README.md) - [CL-MoE: Enhancing Multimodal Large Language Model with Dual Momentum Mixture-of-Experts for Continual Visual Question Answering](https://arxiv.org/pdf/2503.00413)

12. **Self-Synthesized Rehearsal (SSR)** - [View Implementation](src/easycl/cl/ssr/README.md) - [Mitigating catastrophic forgetting in large language models with self-synthesized rehearsal](https://arxiv.org/pdf/2403.01244)

13. **Pseudo Replay** - [View Implementation](src/easycl/cl/pseudo_replay/README.md) - [Experience replay for continual learning](https://proceedings.neurips.cc/paper_files/paper/2019/file/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Paper.pdf)

For more details about the continual learning methods, see [src/easycl/cl/README.md](src/easycl/cl/README.md).

## Installation

```bash
git clone https://github.com/ECNU-ICALK/EasyCL.git
cd EasyCL
pip install -e .
```
Note that if you already have LLaMA-Factory installed in your environment, you may need to uninstall the existing one and perform the installation again.

## Dataset Format Requirements

To use EasyCL, your datasets should conform to the LLaMA-Factory dataset format requirements:

### Data Format

The [dataset_info.json](dataset_info.json) file contains all available datasets. If you want to use a custom dataset, you **must** add a *dataset description* in the `dataset_info.json` file and use the dataset by modifying the `dataset: dataset_name` configuration.

Currently, we support datasets in both **alpaca** and **sharegpt** formats.

### Alpaca Format

#### Instruction Supervised Fine-tuning Dataset

In instruction supervised fine-tuning, content from the `instruction` column will be concatenated with content from the `input` column as the human instruction, which means the human instruction will be `instruction\input`. And content from the `output` column will be the model's response.

If specified, content from the `system` column will be used as the system prompt.

The `history` column is a list of string pairs, representing instructions and responses in each round of historical conversations. Note that in instruction supervised fine-tuning, responses in historical conversations will **also be used for model learning**.

```json
[
  {
    "instruction": "Human instruction (required)",
    "input": "Human input (optional)",
    "output": "Model response (required)",
    "system": "System prompt (optional)",
    "history": [
      ["First-round instruction (optional)", "First-round response (optional)"],
      ["Second-round instruction (optional)", "Second-round response (optional)"]
    ]
  }
]
```

For data in the above format, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "system": "system",
    "history": "history"
  }
}
```

### Sharegpt Format

#### Instruction Supervised Fine-tuning Dataset

Compared to the alpaca format, the sharegpt format supports **more role types**, such as human, gpt, observation, function, etc. They form a list of objects presented in the `conversations` column.

Note that human and observation must appear in odd positions, while gpt and function must appear in even positions.

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "Human instruction"
      },
      {
        "from": "function_call",
        "value": "Tool parameters"
      },
      {
        "from": "observation",
        "value": "Tool results"
      },
      {
        "from": "gpt",
        "value": "Model response"
      }
    ],
    "system": "System prompt (optional)",
    "tools": "Tool descriptions (optional)"
  }
]
```

For data in the above format, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "system": "system",
    "tools": "tools"
  }
}
```

### Continuous Learning Evaluation

If you need to use continuous learning evaluation, you need to register dataset options in `dataset_options.json`. Here is an example:

```json
"custom_dataset": {
  "options": ["Option1", "Option2", "Option3"],
  "description": "Custom dataset example with 3 options"
}
```

This configuration allows EasyCL to properly evaluate model performance on classification tasks during continuous learning.

## Workflow

### Train Only

```bash
easycl-cli cl_workflow --mode train_only --train_params ./configs/train_config.json
```

**Preview Result**: Executes training commands sequentially for tasks defined in `train_config_ewc.json`, applying parameter management between tasks.

### Evaluate Only

```bash
easycl-cli cl_workflow --mode eval_only --eval_params ./configs/eval_config.json
```

**Preview Result**: Executes evaluation command(s) specified in `eval_config.json` (e.g., evaluating a specific fine-tuned model on `cl_tasks`).

### Train Then Evaluate

```bash
easycl-cli cl_workflow --mode train_then_eval \
    --train_params ./configs/train_config_replay.json \
    --eval_params ./configs/eval_config.json
```

**Preview Result**: Executes training commands sequentially, then executes evaluation commands (evaluating base model and model after each task).

### Full Workflow (Train, Evaluate, Calculate Metrics)

```bash
easycl-cli cl_workflow --mode full_workflow \
    --train_params ./configs/train_config.json \
    --eval_params ./configs/eval_config.json
```

**Preview Result**: Executes training sequentially, then evaluates base/task models, and finally calculates and saves CL metrics (Last, Avg, BWT, FWT) to the evaluation output directory.

For detailed information about workflow configuration and CL metrics, see [src/easycl/cl_workflow/README.md](src/easycl/cl_workflow/README.md).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details. 
