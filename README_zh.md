<!-- Logo -->
<p align="center">
  <img src="assets/logo.png" alt="EasyCL Logo" style="width: 100%;" />
</p>

[ [English](README.md) | [中文](README_zh.md) ]

## 目录

- [简介](#简介)
- [已实现方法](#已实现方法)
- [安装](#安装)
- [数据集格式要求](#数据集格式要求)
  - [数据格式](#数据格式)
  - [Alpaca 格式](#alpaca-格式)
  - [Sharegpt 格式](#sharegpt-格式)
  - [持续学习评估](#持续学习评估)
- [工作流程](#工作流程)
  - [仅训练](#仅训练)
  - [仅评估](#仅评估)
  - [先训练后评估](#先训练后评估)
  - [完整工作流](#完整工作流训练评估计算指标)
- [许可证](#许可证)

## 简介

EasyCL 是 LLaMA Factory 框架的扩展，专注于大型语言模型的持续学习方法。它提供了一套全面的工具和方法，用于解决顺序学习任务中的灾难性遗忘问题。

该框架集成了各种专为语言模型设计的最先进持续学习技术，使研究人员和实践者能够轻松实现、比较和开发新方法。

有关持续学习工作流的详细实现，请参阅 [src/easycl/cl_workflow/README.md](src/easycl/cl_workflow/README.md)。

## 已实现方法

1. **弹性权重巩固 (EWC)** - [查看实现](src/easycl/cl/ewc/README.md) - [Overcoming catastrophic forgetting in neural networks](https://www.pnas.org/doi/pdf/10.1073/pnas.1611835114)

2. **无遗忘学习 (LWF)** - [查看实现](src/easycl/cl/lwf/README.md) - [Learning without forgetting](https://ieeexplore.ieee.org/ielaam/34/8520726/8107520-aam.pdf)

3. **经验回放 (Experience Replay)** - [查看实现](src/easycl/cl/replay/README.md) - [Experience replay for continual learning](https://proceedings.neurips.cc/paper_files/paper/2019/file/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Paper.pdf)

4. **LAMOL (语言建模的终身语言学习)** - [查看实现](src/easycl/cl/lamol/README.md) - [LAMOL: LAnguage MOdeling for Lifelong Language Learning](https://arxiv.org/pdf/1909.03329)

5. **O-LoRA (正交子空间学习)** - [查看实现](src/easycl/cl/olora/README.md) - [Orthogonal subspace learning for language model continual learning](https://arxiv.org/pdf/2310.14152)

6. **梯度情景记忆 (GEM)** - [查看实现](src/easycl/cl/gem/README.md) - [Gradient Episodic Memory for Continual Learning](https://proceedings.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf)

7. **I-LoRA (基于插值的 LoRA)** - [查看实现](src/easycl/cl/ilora/README.md) - [Analyzing and Reducing Catastrophic Forgetting in Parameter Efficient Tuning](https://arxiv.org/pdf/2402.18865)

8. **CLMoE (持续学习混合专家与 LoRA)** - [查看实现](src/easycl/cl/clmoe/README.md) - [CL-MoE: Enhancing Multimodal Large Language Model with Dual Momentum Mixture-of-Experts for Continual Visual Question Answering](https://arxiv.org/pdf/2503.00413)

9. **ABSCL (ABSA LLM-CL)** - [查看实现](src/easycl/cl/abscl/README.md) - [Boosting Large Language Models with Continual Learning for Aspect-based Sentiment Analysis](https://arxiv.org/pdf/2405.05496)

10. **动态 ConPet** - [查看实现](src/easycl/cl/dynamic_conpet/README.md) - [ConPET: Continual Parameter-Efficient Tuning for Large Language Models](https://arxiv.org/pdf/2309.14763)

11. **自合成排练 (SSR)** - [查看实现](src/easycl/cl/ssr/README.md) - [Mitigating catastrophic forgetting in large language models with self-synthesized rehearsal](https://arxiv.org/pdf/2403.01244)

12. **伪回放 (Pseudo Replay)** - [查看实现](src/easycl/cl/pseudo_replay/README.md) - [Experience replay for continual learning](https://proceedings.neurips.cc/paper_files/paper/2019/file/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Paper.pdf)

有关持续学习方法的更多详细信息，请参阅 [src/easycl/cl/README.md](src/easycl/cl/README.md)。

## 安装

```bash
git clone https://github.com/ECNU-ICALK/EasyCL.git
cd EasyCL
pip install -e . --no-deps
```
注意，如果你现在环境下已经安装了LLaMA-Factory或旧版EasyCL，你可能需要卸载现有的然后在重新执行一次安装。

## 数据集格式要求

要使用EasyCL，您的数据集应符合LLaMA-Factory的数据集格式要求：

### 数据格式

[dataset_info.json](dataset_info.json) 文件包含了所有可用的数据集。如果您希望使用自定义数据集，请**务必**在 `dataset_info.json` 文件中添加*数据集描述*，并通过修改 `dataset: 数据集名称` 配置来使用数据集。

目前我们支持 **alpaca** 格式和 **sharegpt** 格式的数据集。

### Alpaca 格式

#### 指令监督微调数据集

在指令监督微调时，`instruction` 列对应的内容会与 `input` 列对应的内容拼接后作为人类指令，即人类指令为 `instruction\ninput`。而 `output` 列对应的内容为模型回答。

如果指定，`system` 列对应的内容将被作为系统提示词。

`history` 列是由多个字符串二元组构成的列表，分别代表历史消息中每轮对话的指令和回答。注意在指令监督微调时，历史消息中的回答内容**也会被用于模型学习**。

```json
[
  {
    "instruction": "人类指令（必填）",
    "input": "人类输入（选填）",
    "output": "模型回答（必填）",
    "system": "系统提示词（选填）",
    "history": [
      ["第一轮指令（选填）", "第一轮回答（选填）"],
      ["第二轮指令（选填）", "第二轮回答（选填）"]
    ]
  }
]
```

对于上述格式的数据，`dataset_info.json` 中的*数据集描述*应为：

```json
"数据集名称": {
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

### Sharegpt 格式

#### 指令监督微调数据集

相比 alpaca 格式的数据集，sharegpt 格式支持**更多的角色种类**，例如 human、gpt、observation、function 等等。它们构成一个对象列表呈现在 `conversations` 列中。

注意其中 human 和 observation 必须出现在奇数位置，gpt 和 function 必须出现在偶数位置。

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "人类指令"
      },
      {
        "from": "function_call",
        "value": "工具参数"
      },
      {
        "from": "observation",
        "value": "工具结果"
      },
      {
        "from": "gpt",
        "value": "模型回答"
      }
    ],
    "system": "系统提示词（选填）",
    "tools": "工具描述（选填）"
  }
]
```

对于上述格式的数据，`dataset_info.json` 中的*数据集描述*应为：

```json
"数据集名称": {
  "file_name": "data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "system": "system",
    "tools": "tools"
  }
}
```

### 持续学习评估

如果需要使用持续学习评估，需要在`dataset_options.json`中注册数据集选项。以下是一个示例：

```json
"自定义数据集": {
  "options": ["选项1", "选项2", "选项3"],
  "description": "包含3个选项的自定义数据集示例"
}
```

这种配置允许EasyCL在持续学习过程中正确评估模型在分类任务上的性能。

## 工作流程

### 仅训练

```bash
easycl-cli cl_workflow --mode train_only --train_params ./configs/train_config.json
```

**预览结果**: 按顺序执行`train_config.json`中定义的任务训练命令，并在任务之间应用参数管理。

### 仅评估

```bash
easycl-cli cl_workflow --mode eval_only --eval_params ./configs/eval_config.json
```

**预览结果**: 执行`eval_config.json`中指定的评估命令（例如，在`cl_tasks`上评估特定的微调模型）。

### 先训练后评估

```bash
easycl-cli cl_workflow --mode train_then_eval \
    --train_params ./configs/train_config_replay.json \
    --eval_params ./configs/eval_config.json
```

**预览结果**: 按顺序执行训练命令，然后执行评估命令（评估基础模型和每个任务后的模型）。

### 完整工作流（训练、评估、计算指标）

```bash
easycl-cli cl_workflow --mode full_workflow \
    --train_params ./configs/train_config.json \
    --eval_params ./configs/eval_config.json
```

**预览结果**: 按顺序执行训练，然后评估基础/任务模型，最后计算并保存持续学习指标（Last、Avg、BWT、FWT）到评估输出目录。

有关工作流配置和持续学习指标的详细信息，请参阅 [src/easycl/cl_workflow/README.md](src/easycl/cl_workflow/README.md)。

## 许可证

本项目采用 Apache License 2.0 许可 - 详见 [LICENSE](LICENSE) 文件。