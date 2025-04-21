<!-- Logo -->
<p align="center">
  <img src="assets/logo.png" alt="EasyCL Logo" style="width: 100%;" />
</p>

[ [English](README.md) | [中文](README_zh.md) ]
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

##  状态概览

**注意:** 当前是开发版本，所以可能会遇到一些bug。如果你遇到bug，请在issue中向我提出，或者通过邮箱:caiyuxuanuestc@hotmail.com或微信damowangdongdong与我联系，非常感谢！

<details>
<summary>🚧 <strong>已知问题 / 即将推出的功能</strong></summary>

*   [待办] 对 MoE-LoRA 相关方法与原实现进行结果比较。
*   [功能] 计划添加对 [新方法/功能] 的支持。
*   优化 [特定过程] 中的内存使用。

</details>

<details>
<summary>✅ <strong>已解决问题 / 已完成功能</strong></summary>

*   [已解决] 评估未能正确依赖 `dataset_info.json` (2025-04-19)。
*   [已解决] 评估时使用了过于严格的生成参数（例如 MMLU 的参数）(2025-04-19)。
*   [已解决] 伪回放 (Pseudo Replay) 方法读取的是 tokenized 数据而非原始数据 (2025-04-20)。
*   [已解决] 梯度情景记忆 (GEM) 方法存在显存溢出问题 (2025-04-20)。
*   [已解决] 改善了O-Lora的逻辑，修复了维度不匹配问题 (2025-04-20)。
*   [已解决] 修复了伪样本生成相关方法的问题，并检查了所有现有方法的参数导入 (2025-04-20)。

</details>

## 目录

- [简介](#简介)
- [已实现方法](#已实现方法)
- [安装](#安装)
- [数据集格式要求](#数据集格式要求)
  - [数据格式](#数据格式)
  - [Alpaca 格式](#alpaca-格式)
  - [Sharegpt 格式](#sharegpt-格式)
  - [评估时的数据集要求](#评估时的数据集要求)
  - [持续学习评估](#持续学习评估)
- [Benchmark 适配](#benchmark-适配)
  - [创建自定义 Benchmark](#创建自定义-benchmark)
- [工作流程](#工作流程)
  - [仅训练](#仅训练)
  - [仅评估](#仅评估)
  - [先训练后评估](#先训练后评估)
  - [完整工作流（训练、评估、计算指标）](#完整工作流训练评估计算指标)
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

8. **CLMoE (双动量混合专家 LoRA)** - [查看实现](src/easycl/cl/clmoe/README.md) - [CL-MoE: Enhancing Multimodal Large Language Model with Dual Momentum Mixture-of-Experts for Continual Visual Question Answering](https://arxiv.org/pdf/2503.00413)

9. **MOE-LoRA (混合专家与低秩适应)** - [查看实现](src/easycl/cl/moe/README.md) - [CoIN: A Benchmark of Continual Instruction Tuning for Multimodal Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/file/6a45500d9eda640deed90d8a62742be5-Paper-Datasets_and_Benchmarks_Track.pdf)

10. **ABSCL (ABSA LLM-CL)** - [查看实现](src/easycl/cl/abscl/README.md) - [Boosting Large Language Models with Continual Learning for Aspect-based Sentiment Analysis](https://arxiv.org/pdf/2405.05496)

11. **动态 ConPet** - [查看实现](src/easycl/cl/dynamic_conpet/README.md) - [ConPET: Continual Parameter-Efficient Tuning for Large Language Models](https://arxiv.org/pdf/2309.14763)

12. **自合成排练 (SSR)** - [查看实现](src/easycl/cl/ssr/README.md) - [Mitigating catastrophic forgetting in large language models with self-synthesized rehearsal](https://arxiv.org/pdf/2403.01244)

13. **伪回放 (Pseudo Replay)** - [查看实现](src/easycl/cl/pseudo_replay/README.md) - [Experience replay for continual learning](https://proceedings.neurips.cc/paper_files/paper/2019/file/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Paper.pdf)

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

#### 评估时的数据集要求

EasyCL 的评估流程同样依赖 `dataset_info.json` 文件来定位和加载所需的数据集。当你运行评估命令并指定 `--cl_tasks <任务名称>` 时（例如 `--cl_tasks my_eval_task`），评估器会执行以下操作：

1.  **查找测试集 (Test Set)**: 评估器会在 `dataset_info.json` 中查找与 `<任务名称>_test` 匹配的条目（例如 `my_eval_task_test`），或者查找键为 `<任务名称>` 且 `split` 字段为 `"test"` 的条目。**测试集是评估所必需的。**
2.  **查找开发集 (Dev Set)**: 如果评估参数中设置了 `n_shot > 0`（即进行少样本评估），评估器会类似地查找与 `<任务名称>_dev` 匹配的条目（例如 `my_eval_task_dev`）或 `split` 字段为 `"dev"` 的条目，以加载少样本示例。**开发集对于零样本评估不是必需的。**

**示例:**

假设你的 `dataset_info.json` 包含以下条目：

```json
{
  "my_eval_task_dev": {
    "file_name": "my_data/my_eval_task_dev.json",
    "split": "dev",
    "columns": {
        "prompt": "instruction",
        "query": "input",
        "response": "output"
     }
  },
  "my_eval_task_test": {
    "file_name": "my_data/my_eval_task_test.json",
    "split": "test",
    "columns": {
        "prompt": "instruction",
        "query": "input",
        "response": "output"
     }
  }
  // ... 其他数据集 ...
}
```

当你运行 `easycl-cli cl_workflow --mode eval_only --eval_params <你的评估配置>.yaml`，并且该配置中指定了 `--cl_tasks my_eval_task` 时：
*   评估器会加载 `my_data/my_eval_task_test.json` 作为测试集。
*   如果配置中还指定了 `--n_shot 5`，评估器会加载 `my_data/my_eval_task_dev.json` 并从中选取前5个样本作为少样本示例。

**重要提示:**
*   请确保 `dataset_info.json` 中为需要评估的每个任务都定义了对应的 `test` 集条目，并提供了正确的 `file_name`。
*   如果需要进行少样本评估，请同时定义 `dev` 集条目。
*   `file_name` 指定的文件路径应相对于 `dataset_info.json` 所在的目录或项目根目录下的 `data` 目录。评估器会优先在 `task_dir` (如果指定) 或 `./data` 目录中查找。

### 持续学习评估

如果需要使用持续学习评估，需要在`dataset_options.json`中注册数据集选项。以下是一个示例：

```json
"自定义数据集": {
  "options": ["选项1", "选项2", "选项3"],
  "description": "包含3个选项的自定义数据集示例"
}
```

这种配置允许EasyCL在持续学习过程中正确评估模型在分类任务上的性能。

## Benchmark 适配

我们的框架可以自动实现 Benchmark 的训练以及评估，并支持多种任务顺序（Order）切换。这使得在标准数据集上复现和比较不同持续学习方法的效果变得更加容易。

我们目前已适配了以下三个常用的 Benchmark：

1.  **LFPT5** - [Lfpt5: A unified framework for lifelong few-shot language learning based on prompt tuning of t5](https://arxiv.org/pdf/2110.07298)
2.  **Large Number of Tasks Benchmark** - [Orthogonal subspace learning for language model continual learning](https://arxiv.org/pdf/2310.14152)
3.  **ABSACL_ATSC (Aspect-based Sentiment Analysis Continual Learning)** - [Adapting bert for continual learning of a sequence of aspect sentiment classification tasks](https://arxiv.org/pdf/2112.03271)

你可以使用如下命令来进行 Benchmark 评估（Benchmark 评估目前只支持在 `full_workflow` 模式下运行）：

```bash
easycl-cli cl_workflow --mode full_workflow \\
    --train_params ./example/train_examples/lora_example.yaml \\
    --eval_params ./example/eval_examples/lora_eval.yaml \\
    --benchmark ABSACL_ATSC --benchmark_order order1 --benchmark_dir ./benchmark/ABSACL_ATSC
```

**注意:**
*   运行 Benchmark 前，请确保对应的 Benchmark 数据已按要求存放于 `--benchmark_dir` 指定的目录下。
*   每个 Benchmark 都需要维护一个 `benchmark_info.json` 文件，用于注册 Benchmark 名称、定义不同的任务顺序 (order)，以及指定每个任务所需的数据集信息。
*   Benchmark 中涉及的数据集需要在benchmark目录的 `dataset_info.json` 和 `dataset_options.json`（如果需要评估分类任务）中进行注册。

### 创建自定义 Benchmark

如果你希望使用自己的 Benchmark，请遵循以下步骤：

1.  **准备数据集:**
    *   确保你的数据集符合 [数据格式要求](#数据格式要求) 中描述的 **Alpaca** 或 **ShareGPT** 格式。
    *   将每个任务的数据分别整理好。
2.  **组织 Benchmark 目录:**
    *   在 `benchmark` 目录下创建一个新的文件夹，以你的 Benchmark 名称命名（例如 `my_custom_benchmark`）。
    *   在该文件夹下，根据你的任务划分，存放相应的数据文件。
3.  **注册数据集信息:**
    *   在项目根目录的 `dataset_info.json` 文件中，为你的 Benchmark 中使用的每个数据集添加描述。参考 [数据格式](#数据格式) 部分的示例。
    *   请在项目根目录的 `dataset_options.json` 文件中注册数据集选项。参考 [持续学习评估](#持续学习评估) 部分的示例。
4.  **创建 `benchmark_info.json`:**
    *   在你创建的 Benchmark 目录下（例如 `benchmark/my_custom_benchmark`），创建一个 `benchmark_info.json` 文件。
    *   在此文件中，定义你的 Benchmark 名称、不同的任务顺序 (order)，并指定每个顺序下各个任务所对应的数据集名称（这些名称应与 `dataset_info.json` 中注册的名称一致）。可以参考现有 Benchmark（如 `benchmark/ABSACL_ATSC/benchmark_info.json`）的结构。
5.  **运行 Benchmark:**
    *   现在你可以使用 `easycl-cli` 命令，并通过 `--benchmark <你的Benchmark名称>` 和 `--benchmark_dir ./benchmark/<你的Benchmark目录>` 参数来运行你的自定义 Benchmark 了。

## 工作流程

为了方便实现命令行一键式训练，我们实现了命令行界面（Command-Line Interface）的训练，你可以使用多种模式进行训练和评估，他会按照src\easycl\cl_workflow\cl_params_config.json中的设置自动设置一些需要的参数映射。我们目前支持四种训练工作流程：仅训练，仅评估， 先训练后评估和完整工作流（训练、评估、计算指标） 。你可以使用--previewonly指令进行不运行命令的命令预览，并可以使用clean_dirs在运行命令前自动清理输出路径。

### 仅训练

```bash
easycl-cli cl_workflow --mode train_only --train_params ./example/train_examples/lora_example.yaml
```

**预览结果**: 按顺序执行`train_config.json`中定义的任务训练命令，并在任务之间应用参数管理。

### 仅评估

```bash
easycl-cli cl_workflow --mode eval_only --eval_params ./example/eval_examples/lora_eval.yaml
```

**预览结果**: 执行`eval_config.json`中指定的评估命令（例如，在`cl_tasks`上评估特定的微调模型）。

### 先训练后评估

```bash
easycl-cli cl_workflow --mode train_then_eval \
    --train_params ./example/train_examples/lora_example.yaml \
    --eval_params ./example/eval_examples/lora_eval.yaml
```

**预览结果**: 按顺序执行训练命令，然后执行评估命令（评估基础模型和每个任务后的模型）。

### 完整工作流（训练、评估、计算指标）

```bash
easycl-cli cl_workflow --mode full_workflow \
    --train_params ./example/train_examples/lora_example.yaml \
    --eval_params ./example/eval_examples/lora_eval.yaml
```

**预览结果**: 按顺序执行训练，然后评估基础/任务模型，最后计算并保存持续学习指标（Last、Avg、BWT、FWT）到评估输出目录。

有关工作流配置和持续学习指标的详细信息，请参阅 [src/easycl/cl_workflow/README.md](src/easycl/cl_workflow/README.md)。

## 许可证

本项目采用 Apache License 2.0 许可 - 详见 [LICENSE](LICENSE) 文件。

</copilot-edited-file>  ## 目录

- [简介](#简介)
- [已实现方法](#已实现方法)
- [安装](#安装)
- [数据集格式要求](#数据集格式要求)
  - [数据格式](#数据格式)
  - [Alpaca 格式](#alpaca-格式)
  - [Sharegpt 格式](#sharegpt-格式)
  - [评估时的数据集要求](#评估时的数据集要求)
  - [持续学习评估](#持续学习评估)
- [Benchmark 适配](#benchmark-适配)
  - [创建自定义 Benchmark](#创建自定义-benchmark)
- [工作流程](#工作流程)
  - [仅训练](#仅训练)
  - [仅评估](#仅评估)
  - [先训练后评估](#先训练后评估)
  - [完整工作流（训练、评估、计算指标）](#完整工作流训练评估计算指标)
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

8. **CLMoE (双动量混合专家 LoRA)** - [查看实现](src/easycl/cl/clmoe/README.md) - [CL-MoE: Enhancing Multimodal Large Language Model with Dual Momentum Mixture-of-Experts for Continual Visual Question Answering](https://arxiv.org/pdf/2503.00413)

9. **MOE-LoRA (混合专家与低秩适应)** - [查看实现](src/easycl/cl/moe/README.md) - [CoIN: A Benchmark of Continual Instruction Tuning for Multimodal Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/file/6a45500d9eda640deed90d8a62742be5-Paper-Datasets_and_Benchmarks_Track.pdf)

10. **ABSCL (ABSA LLM-CL)** - [查看实现](src/easycl/cl/abscl/README.md) - [Boosting Large Language Models with Continual Learning for Aspect-based Sentiment Analysis](https://arxiv.org/pdf/2405.05496)

11. **动态 ConPet** - [查看实现](src/easycl/cl/dynamic_conpet/README.md) - [ConPET: Continual Parameter-Efficient Tuning for Large Language Models](https://arxiv.org/pdf/2309.14763)

12. **自合成排练 (SSR)** - [查看实现](src/easycl/cl/ssr/README.md) - [Mitigating catastrophic forgetting in large language models with self-synthesized rehearsal](https://arxiv.org/pdf/2403.01244)

13. **伪回放 (Pseudo Replay)** - [查看实现](src/easycl/cl/pseudo_replay/README.md) - [Experience replay for continual learning](https://proceedings.neurips.cc/paper_files/paper/2019/file/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Paper.pdf)

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

#### 评估时的数据集要求

EasyCL 的评估流程同样依赖 `dataset_info.json` 文件来定位和加载所需的数据集。当你运行评估命令并指定 `--cl_tasks <任务名称>` 时（例如 `--cl_tasks my_eval_task`），评估器会执行以下操作：

1.  **查找测试集 (Test Set)**: 评估器会在 `dataset_info.json` 中查找与 `<任务名称>_test` 匹配的条目（例如 `my_eval_task_test`），或者查找键为 `<任务名称>` 且 `split` 字段为 `"test"` 的条目。**测试集是评估所必需的。**
2.  **查找开发集 (Dev Set)**: 如果评估参数中设置了 `n_shot > 0`（即进行少样本评估），评估器会类似地查找与 `<任务名称>_dev` 匹配的条目（例如 `my_eval_task_dev`）或 `split` 字段为 `"dev"` 的条目，以加载少样本示例。**开发集对于零样本评估不是必需的。**

**示例:**

假设你的 `dataset_info.json` 包含以下条目：

```json
{
  "my_eval_task_dev": {
    "file_name": "my_data/my_eval_task_dev.json",
    "split": "dev",
    "columns": {
        "prompt": "instruction",
        "query": "input",
        "response": "output"
     }
  },
  "my_eval_task_test": {
    "file_name": "my_data/my_eval_task_test.json",
    "split": "test",
    "columns": {
        "prompt": "instruction",
        "query": "input",
        "response": "output"
     }
  }

}
```

当你运行 `easycl-cli cl_workflow --mode eval_only --eval_params <你的评估配置>.yaml`，并且该配置中指定了 `--cl_tasks my_eval_task` 时：
*   评估器会加载 `my_data/my_eval_task_test.json` 作为测试集。
*   如果配置中还指定了 `--n_shot 5`，评估器会加载 `my_data/my_eval_task_dev.json` 并从中选取前5个样本作为少样本示例。

**重要提示:**
*   请确保 `dataset_info.json` 中为需要评估的每个任务都定义了对应的 `test` 集条目，并提供了正确的 `file_name`。
*   如果需要进行少样本评估，请同时定义 `dev` 集条目。
*   `file_name` 指定的文件路径应相对于 `dataset_info.json` 所在的目录或项目根目录下的 `data` 目录。评估器会优先在 `task_dir` (如果指定) 或 `./data` 目录中查找。

### 持续学习评估

如果需要使用持续学习评估，需要在`dataset_options.json`中注册数据集选项。以下是一个示例：

```json
"自定义数据集": {
  "options": ["选项1", "选项2", "选项3"],
  "description": "包含3个选项的自定义数据集示例"
}
```

这种配置允许EasyCL在持续学习过程中正确评估模型在分类任务上的性能。

## Benchmark 适配

我们的框架可以自动实现 Benchmark 的训练以及评估，并支持多种任务顺序（Order）切换。这使得在标准数据集上复现和比较不同持续学习方法的效果变得更加容易。

我们目前已适配了以下三个常用的 Benchmark：

1.  **LFPT5** - [Lfpt5: A unified framework for lifelong few-shot language learning based on prompt tuning of t5](https://arxiv.org/pdf/2110.07298)
2.  **Large Number of Tasks Benchmark** - [Orthogonal subspace learning for language model continual learning](https://arxiv.org/pdf/2310.14152)
3.  **ABSACL_ATSC (Aspect-based Sentiment Analysis Continual Learning)** - [Adapting bert for continual learning of a sequence of aspect sentiment classification tasks](https://arxiv.org/pdf/2112.03271)

你可以使用如下命令来进行 Benchmark 评估（Benchmark 评估目前只支持在 `full_workflow` 模式下运行）：

```bash
easycl-cli cl_workflow --mode full_workflow \\
    --train_params ./example/train_examples/lora_example.yaml \\
    --eval_params ./example/eval_examples/lora_eval.yaml \\
    --benchmark ABSACL_ATSC --benchmark_order order1 --benchmark_dir ./benchmark/ABSACL_ATSC
```

**注意:**
*   运行 Benchmark 前，请确保对应的 Benchmark 数据已按要求存放于 `--benchmark_dir` 指定的目录下。
*   每个 Benchmark 都需要维护一个 `benchmark_info.json` 文件，用于注册 Benchmark 名称、定义不同的任务顺序 (order)，以及指定每个任务所需的数据集信息。
*   Benchmark 中涉及的数据集需要在benchmark目录的 `dataset_info.json` 和 `dataset_options.json`（如果需要评估分类任务）中进行注册。

### 创建自定义 Benchmark

如果你希望使用自己的 Benchmark，请遵循以下步骤：

1.  **准备数据集:**
    *   确保你的数据集符合 [数据格式要求](#数据格式要求) 中描述的 **Alpaca** 或 **ShareGPT** 格式。
    *   将每个任务的数据分别整理好。
2.  **组织 Benchmark 目录:**
    *   在 `benchmark` 目录下创建一个新的文件夹，以你的 Benchmark 名称命名（例如 `my_custom_benchmark`）。
    *   在该文件夹下，根据你的任务划分，存放相应的数据文件。
3.  **注册数据集信息:**
    *   在项目根目录的 `dataset_info.json` 文件中，为你的 Benchmark 中使用的每个数据集添加描述。参考 [数据格式](#数据格式) 部分的示例。
    *   请在项目根目录的 `dataset_options.json` 文件中注册数据集选项。参考 [持续学习评估](#持续学习评估) 部分的示例。
4.  **创建 `benchmark_info.json`:**
    *   在你创建的 Benchmark 目录下（例如 `benchmark/my_custom_benchmark`），创建一个 `benchmark_info.json` 文件。
    *   在此文件中，定义你的 Benchmark 名称、不同的任务顺序 (order)，并指定每个顺序下各个任务所对应的数据集名称（这些名称应与 `dataset_info.json` 中注册的名称一致）。可以参考现有 Benchmark（如 `benchmark/ABSACL_ATSC/benchmark_info.json`）的结构。
5.  **运行 Benchmark:**
    *   现在你可以使用 `easycl-cli` 命令，并通过 `--benchmark <你的Benchmark名称>` 和 `--benchmark_dir ./benchmark/<你的Benchmark目录>` 参数来运行你的自定义 Benchmark 了。

## 工作流程

为了方便实现命令行一键式训练，我们实现了命令行界面（Command-Line Interface）的训练，你可以使用多种模式进行训练和评估，他会按照src\easycl\cl_workflow\cl_params_config.json中的设置自动设置一些需要的参数映射。我们目前支持四种训练工作流程：仅训练，仅评估， 先训练后评估和完整工作流（训练、评估、计算指标） 。你可以使用--previewonly指令进行不运行命令的命令预览，并可以使用clean_dirs在运行命令前自动清理输出路径。

### 仅训练

```bash
easycl-cli cl_workflow --mode train_only --train_params ./example/train_examples/lora_example.yaml
```

**预览结果**: 按顺序执行`train_config.json`中定义的任务训练命令，并在任务之间应用参数管理。

### 仅评估

```bash
easycl-cli cl_workflow --mode eval_only --eval_params ./example/eval_examples/lora_eval.yaml
```

**预览结果**: 执行`eval_config.json`中指定的评估命令（例如，在`cl_tasks`上评估特定的微调模型）。

### 先训练后评估

```bash
easycl-cli cl_workflow --mode train_then_eval \
    --train_params ./example/train_examples/lora_example.yaml \
    --eval_params ./example/eval_examples/lora_eval.yaml
```

**预览结果**: 按顺序执行训练命令，然后执行评估命令（评估基础模型和每个任务后的模型）。

### 完整工作流（训练、评估、计算指标）

```bash
easycl-cli cl_workflow --mode full_workflow \
    --train_params ./example/train_examples/lora_example.yaml \
    --eval_params ./example/eval_examples/lora_eval.yaml
```

**预览结果**: 按顺序执行训练，然后评估基础/任务模型，最后计算并保存持续学习指标（Last、Avg、BWT、FWT）到评估输出目录。

有关工作流配置和持续学习指标的详细信息，请参阅 [src/easycl/cl_workflow/README.md](src/easycl/cl_workflow/README.md)。

## 许可证

本项目采用 Apache License 2.0 许可 - 详见 [LICENSE](LICENSE) 文件。
