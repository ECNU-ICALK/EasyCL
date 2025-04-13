# 多Adapter选择与评估流程接口文档 (v2)

## 概述

本文档描述了持续学习评估中**解耦式**多Adapter选择与评估流程的接口规范。与旧版不同，新流程将**Adapter选择**步骤与**模型评估**步骤分离。

1.  **Adapter选择阶段**: 一个独立的脚本或流程（**选择器**）负责分析数据集，并决定每个数据样本应该由哪个Adapter进行处理。选择器的输出是一个JSON配置文件 (`multiadapter_selected_config.json`)。
2.  **模型评估阶段**: `CLEvalEvaluator`读取这个配置文件，根据其中的映射关系加载相应的Adapter，并对指定的数据子集进行评估，最后汇总结果。

这种解耦设计提供了更大的灵活性，允许用户实现各种复杂的Adapter选择策略，而不必修改核心评估器代码。

## 配置文件 (`multiadapter_selected_config.json`)

选择器脚本执行后，应在`multi_adapter_dir`目录下生成一个名为`multiadapter_selected_config.json`的文件。该文件是评估器执行多Adapter评估的唯一依据。

### 文件结构

```json
{
  "task_name": "name_of_the_evaluation_task",
  "adapters": {
    "adapter_name_1": {
      "path": "relative/path/from/multi_adapter_dir/to/adapter1",
      "indices": [0, 5, 10, 25, ...]
    },
    "adapter_name_2": {
      "path": "another/relative/path/to/adapter2",
      "indices": [1, 2, 7, 15, ...]
    },
    // ... 可以包含更多adapter条目
    "adapter_for_unassigned": { // 可选: 用于处理未被明确分配的样本
        "path": "path/to/default/or/fallback/adapter",
        "indices": [3, 4, 6, ...] // 包含所有未在上面列出的样本索引
    }
  }
}
```

### 字段说明

*   `task_name`: (字符串) 当前评估的任务名称，用于核对。
*   `adapters`: (字典) 包含所有在此次评估中使用的Adapter及其负责的数据样本。
    *   **Key**: (字符串) Adapter的唯一标识符（**逻辑名称**），例如可以基于其训练任务或特性命名，如 "math_adapter", "code_adapter", "task3_finetune"。**评估器将使用这个名称在日志中报告结果，而不是Adapter的路径。**
    *   **Value**: (字典) 包含该Adapter的具体信息。
        *   `path`: (字符串) 从`multi_adapter_dir`到该Adapter权重文件的**相对路径**。评估器将使用此路径加载Adapter。
        *   `indices`: (列表<整数>) 一个包含**原始测试集**中由该Adapter负责处理的**样本索引**（从0开始）的列表。**所有样本索引必须被分配，且不能重复分配** (除非使用可选的回退Adapter逻辑)。评估器将使用这些索引从原始数据集中提取子集。

**重要**: 配置文件需要确保覆盖测试集中的所有样本索引。如果某些样本没有被明确分配给某个特定Adapter，可以考虑：
a) 将它们分配给一个通用的或基础的Adapter。
b) 在配置文件中添加一个特殊的"default"或"unassigned"条目来处理这些样本。
c) 如果选择器无法处理某些样本，可以选择在配置文件中省略它们，评估器将跳过这些样本并记录警告，但这可能会导致评估结果不完整。

## Adapter选择器实现场景

选择器本身是一个独立的脚本或程序，其具体实现方式由用户决定。以下是两种常见的实现思路：

### 场景 1: 使用当前评估的基础模型进行选择

这种方法利用正在评估的基础模型（不加载任何Adapter）的能力来判断每个样本最适合由哪个专有Adapter处理。

*   **目标**: 让基础模型对每个样本进行分析（例如，分类到某个领域、判断任务类型、计算与某个Adapter主题的相关性等），然后根据分析结果将样本分配给最合适的Adapter。
*   **输入**:
    *   模型参数 (`model_args`)
    *   数据参数 (`data_args`)
    *   评估参数 (`eval_args`, 包含 `task` 和 `multi_adapter_dir`)
    *   微调参数 (`finetuning_args`)
    *   原始测试数据集文件路径 (例如从 `data_args` 或 `eval_args` 获取)
*   **流程**:
    1.  **加载资源**:
        *   解析传入的参数。
        *   加载**基础模型** (关键：确保`finetuning_args.adapter_name_or_path`设置为`None`或为空)。
        *   加载分词器 (`tokenizer`)。
        *   加载完整的测试数据集。
        *   扫描`multi_adapter_dir`以发现所有可用的Adapter及其路径，建立一个Adapter名称到路径的映射。
        *   *可选*: 加载在训练阶段保存的额外信息（例如，每个Adapter对应的任务描述、样本嵌入等），路径信息应定义在`finetuning_args`中并在训练时保存。
    2.  **样本分配**:
        *   遍历测试数据集中的每个样本及其索引 (`idx`, `sample`)。
        *   **设计选择逻辑**:
            *   格式化样本内容，构建一个适合基础模型进行**选择判断**的输入。这可能是一个分类问题 ("这个样本属于哪个领域？")、相似度计算问题 ("这个样本与哪个Adapter的描述最相似？") 或其他形式。
            *   使用基础模型进行推理。
            *   根据模型的输出（例如，预测的类别标签、最高概率、最相似的描述对应的Adapter）确定最适合处理该样本的Adapter**逻辑名称**。
        *   记录样本索引`idx`与选定的Adapter逻辑名称的映射关系。
    3.  **生成配置**:
        *   整理所有样本的分配结果。
        *   根据Adapter名称到路径的映射，构建`multiadapter_selected_config.json`文件的内容结构。
        *   将配置文件写入`multi_adapter_dir`。
*   **Mermaid流程图**:
    ```mermaid
    graph TD
        A[开始: 选择器脚本] --> B(加载参数);
        B --> C(加载基础模型 & Tokenizer);
        B --> D(加载测试数据集);
        B --> E(扫描 multi_adapter_dir 查找可用 Adapters);
        B --> F(可选: 加载额外选择信息);
        G{遍历数据集样本};
        C --> G;
        D --> G;
        E --> G;
        F --> G;
        G -- 对每个样本 --> H(格式化选择输入);
        H --> I(基础模型推理);
        I --> J(处理推理结果 -> 选择Adapter逻辑名称);
        J --> K(记录 样本索引 -> Adapter名称);
        K --> G;
        G -- 遍历结束 --> L(整理分配结果);
        L --> M(构建 JSON 配置结构);
        M --> N(写入 multiadapter_selected_config.json);
        N --> Z[结束];
    ```
*   **概念性Python代码**:
    ```python
    # -*- coding: utf-8 -*-
    # conceptual_base_model_selector.py
    # 注意: 这是高度简化的概念代码，需要根据实际模型和选择策略进行大量修改

    import json
    import os
    import torch
    import copy
    from collections import defaultdict
    # 假设可以从 llamafactory 正确导入所需模块
    from llamafactory.hparams import get_eval_args # 用于解析参数
    from llamafactory.model import load_model, load_tokenizer
    # 假设有数据加载函数
    # from your_data_utils import load_dataset_from_file, find_adapters_in_dir

    def run_base_model_selector(args_dict, dataset_path, output_config_path):
        # 1. 加载参数和资源
        model_args, data_args, eval_args, finetuning_args = get_eval_args(args_dict)
        multi_adapter_dir = eval_args.multi_adapter_dir

        # 加载基础模型 (无Adapter)
        finetuning_args_base = copy.deepcopy(finetuning_args)
        finetuning_args_base.adapter_name_or_path = None
        tokenizer_module = load_tokenizer(model_args)
        tokenizer = tokenizer_module["tokenizer"]
        # 确定设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(tokenizer, model_args, finetuning_args_base)
        model.to(device)
        model.eval()

        # 加载数据集
        # dataset = load_dataset_from_file(dataset_path) # 替换为实际数据加载逻辑
        with open(dataset_path, "r", encoding="utf-8") as f:
             dataset_data = json.load(f)
             dataset = dataset_data.get("examples", dataset_data)


        # 查找 Adapters (返回 {adapter_name: relative_path})
        # available_adapters = find_adapters_in_dir(multi_adapter_dir) # 替换为实际查找逻辑
        # 示例:
        available_adapters = {
             "adapter_math": "math_expert",
             "adapter_code": "code_expert",
             "adapter_general": "general_v1"
        }

        adapter_assignments = defaultdict(lambda: {"path": None, "indices": []})
        print(f"Starting adapter selection for {len(dataset)} samples...")

        # 2. 样本分配
        with torch.no_grad():
            for idx, sample in enumerate(dataset):
                # --- 选择逻辑核心 ---
                # 示例：假设基础模型能判断样本类型 (math, code, general)
                # 你需要根据你的模型和任务设计这部分
                instruction = sample.get("instruction", "")
                input_text = sample.get("input", "")
                full_text = f"{instruction}\n{input_text}".strip()

                # 构造适合分类的输入
                # 可能需要更复杂的prompt工程
                selection_prompt = f"Classify the domain of the following text: {full_text}"
                inputs = tokenizer(selection_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

                # 模型推理 (假设模型输出logits用于分类)
                outputs = model(**inputs) # 可能需要调整调用方式
                # 假设输出logits对应 ["math", "code", "general"] 类别
                predicted_class_idx = torch.argmax(outputs.logits, dim=-1).item()

                # 映射类别到Adapter名称
                selected_adapter_name = None
                if predicted_class_idx == 0 and "adapter_math" in available_adapters:
                    selected_adapter_name = "adapter_math"
                elif predicted_class_idx == 1 and "adapter_code" in available_adapters:
                    selected_adapter_name = "adapter_code"
                elif "adapter_general" in available_adapters: # Fallback or default
                     selected_adapter_name = "adapter_general"
                else:
                     print(f"Warning: Could not assign adapter for sample {idx}")
                     # 可以选择分配给第一个找到的 adapter 或特定 default adapter
                     if available_adapters:
                         selected_adapter_name = list(available_adapters.keys())[0]


                # 记录分配
                if selected_adapter_name:
                    adapter_path = available_adapters[selected_adapter_name]
                    adapter_assignments[selected_adapter_name]["path"] = adapter_path # 存储相对路径
                    adapter_assignments[selected_adapter_name]["indices"].append(idx)

                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx+1}/{len(dataset)} samples...")

        # 3. 生成配置
        output_data = {
            "task_name": eval_args.task,
            "adapters": dict(adapter_assignments) # 转换为普通字典
        }

        # 确保路径是相对 multi_adapter_dir 的
        for name, data in output_data["adapters"].items():
             # 假设 find_adapters_in_dir 返回的就是相对路径
             pass # 如果不是相对路径，需要在这里转换

        final_config_path = os.path.join(multi_adapter_dir, 'multiadapter_selected_config.json')
        with open(final_config_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Adapter selection config saved to: {final_config_path}")
        # --- Helper function stubs ---
        # def find_adapters_in_dir(base_dir):
        #    # Implement logic to scan base_dir for adapter folders
        #    # Return dict like {"adapter_name": "relative/path/to/adapter", ...}
        #    pass

    # Example of how to run this script (needs proper argument parsing)
    # if __name__ == "__main__":
    #     # Use argparse or similar to get args_dict, dataset_path
    #     args_dict = {...} # Load from command line or config file
    #     dataset_path = "path/to/your/task_test.json"
    #     run_base_model_selector(args_dict, dataset_path)

    ```

### 场景 2: 使用外部模型进行选择

这种方法使用一个**独立于**当前评估流程的模型来进行Adapter选择。这个外部模型可能更小、更快，或者专门为分类/路由任务训练过（例如，一个BERT或DeBERTa分类器）。

*   **目标**: 利用外部模型的专长（如文本分类）来决定每个样本应由哪个Adapter处理。
*   **输入**:
    *   数据参数 (`data_args`)
    *   评估参数 (`eval_args`, 包含 `task` 和 `multi_adapter_dir`)
    *   原始测试数据集文件路径
    *   外部选择器模型的标识符（例如Hugging Face Hub上的名称或本地路径）
*   **流程**:
    1.  **加载资源**:
        *   解析参数。
        *   加载**外部选择器模型**及其对应的分词器 (例如使用 `transformers.AutoModelForSequenceClassification.from_pretrained(...)`)。
        *   加载完整的测试数据集。
        *   扫描`multi_adapter_dir`查找可用Adapter及其路径。
        *   *可选*: 加载额外选择信息。
    2.  **样本分配**:
        *   遍历测试数据集中的每个样本及其索引 (`idx`, `sample`)。
        *   **设计选择逻辑**:
            *   格式化样本内容以适应**外部模型**的输入。
            *   使用外部模型进行推理。
            *   根据外部模型的输出（例如预测的类别）映射到最合适的Adapter逻辑名称。
        *   记录样本索引`idx`与选定的Adapter逻辑名称的映射。
    3.  **生成配置**:
        *   整理分配结果。
        *   构建`multiadapter_selected_config.json`文件。
        *   写入`multi_adapter_dir`。
*   **Mermaid流程图**:
    ```mermaid
    graph TD
        A[开始: 选择器脚本] --> B(加载参数);
        B --> C(加载外部选择器模型 & Tokenizer);
        B --> D(加载测试数据集);
        B --> E(扫描 multi_adapter_dir 查找可用 Adapters);
        B --> F(可选: 加载额外选择信息);
        G{遍历数据集样本};
        C --> G;
        D --> G;
        E --> G;
        F --> G;
        G -- 对每个样本 --> H(格式化样本输入给外部模型);
        H --> I(外部模型推理);
        I --> J(处理推理结果 -> 选择Adapter逻辑名称);
        J --> K(记录 样本索引 -> Adapter名称);
        K --> G;
        G -- 遍历结束 --> L(整理分配结果);
        L --> M(构建 JSON 配置结构);
        M --> N(写入 multiadapter_selected_config.json);
        N --> Z[结束];
    ```
*   **概念性Python代码**: 与场景1类似，主要区别在于加载和使用不同的模型 (`selector_model`, `selector_tokenizer`) 进行推理。

## 注意事项

1.  **选择器独立性**: 选择器脚本应独立运行，并且是评估流程的前置步骤。
2.  **配置准确性**: `multiadapter_selected_config.json` 文件的正确性至关重要，特别是Adapter路径和样本索引的准确性。路径应相对于`multi_adapter_dir`。
3.  **Adapter发现**: `find_adapters_in_dir`的逻辑需要明确，如何从目录结构或元数据中确定Adapter的**逻辑名称**和**相对路径**。
4.  **样本全覆盖**: 确保所有需要评估的样本都被分配到一个Adapter。处理未分配样本的策略应明确。
5.  **选择逻辑**: 核心在于设计有效的选择逻辑（步骤 H, I, J），这高度依赖具体任务和Adapter的特性。