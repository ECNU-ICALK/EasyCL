# ABSCL (ABSA LLM-CL)

\[ [English](README.md) | [中文](README_zh.md) \]

## 1. 核心思想

ABSCL (ABSA LLM-CL) 是一个专为方面级情感分析（ABSA）设计的持续学习方法。它训练两种类型的适配器：

1. **共享适配器（Shared Adapter）：** 使用重放策略进行训练，结合当前任务和先前任务的部分数据。该适配器旨在捕获跨任务的通用知识。
2. **任务特定适配器（Task-Specific Adapter）：** 仅使用当前任务的数据进行训练。

在训练任务特定适配器时，ABSCL 应用了两个受 O-LoRA 启发的约束：
* **正交约束：** 鼓励任务特定适配器的权重与共享适配器的权重保持正交，促进任务特定知识的分离。
* **L2正则化：** 在训练任务特定适配器时，对共享适配器的权重（作为参考加载）应用 L2 正则化，可能有助于防止共享知识库中的灾难性遗忘。

此外，ABSCL 在训练特定适配器后会提取每个任务的特征统计信息（平均向量和共享协方差矩阵）。这些统计信息后续可被 `abscl_selector.py` 用于基于马氏距离确定给定输入样本的最适合适配器（任务）。

## 2. 具体参数

以下参数是 ABSCL 方法特有的（参考 `finetuning_args`）：

* `--abscl_orthogonal_lambda`：（float，必需）任务特定适配器与共享适配器之间正交约束损失的权重。
* `--abscl_shared_l2_lambda`：（float，必需）在任务特定适配器训练期间应用于共享适配器权重的 L2 正则化损失的权重。
* `--abscl_stats_path`：（str，可选）保存和加载特征统计信息（平均向量和协方差矩阵）的路径。如果未提供，默认为 `adapters_save_path/abscl_stats`。
* `--current_task_id`：（str，必需）当前任务的标识符。用于命名任务特定适配器和存储其统计信息。
* `--adapters_save_path`：（str，必需）保存 `shared_adapter` 和任务特定适配器（由 `current_task_id` 命名）的基本目录。如果未设置 `abscl_stats_path`，特征统计信息也将相对于此路径存储。

*注意：诸如 `--replay_ratio`、`--replay_task_list`、`--maxsamples_list`、`--previous_task_dataset` 等参数在训练共享适配器时由 ABSCL 工作流程用于重放策略，但可能被视为通用重放参数。*

## 3. 文件说明

* `abscl_workflow.py`：编排主要的 ABSCL 训练过程。它处理重放数据准备、训练共享适配器、使用 ABSCL 约束训练任务特定适配器，并触发特征统计提取。
* `abscl_trainer.py`：定义 `ABSCLTrainer`，一个自定义的 Hugging Face `Trainer` 子类。它修改损失计算以包含正交性和共享 L2 正则化项，利用 O-LoRA 机制进行计算。
* `abscl.py`：包含 `ABSCLFeatureExtractor` 类，负责提取隐藏状态特征（具体来说，是倒数第二层的最后一个token的隐藏状态），并计算/更新每个任务的平均向量和共享协方差矩阵。
* `abscl_selector.py`：在使用 ABSCL 训练多个任务之后使用的脚本。它加载保存的特征统计信息和测试数据集，然后基于样本特征表示到任务平均值的马氏距离为每个测试样本分配最可能的任务适配器。

## 4. 引用

@article{ding2024boosting,
  title={Boosting large language models with continual learning for aspect-based sentiment analysis},
  author={Ding, Xuanwen and Zhou, Jie and Dou, Liang and Chen, Qin and Wu, Yuanbin and Chen, Chengcai and He, Liang},
  journal={arXiv preprint arXiv:2405.05496},
  year={2024}
}