[ [English](README.md) | [中文](README_zh.md) ]

# O-LoRA：语言模型持续学习的正交子空间学习

## 核心思想
O-LoRA 是一种持续学习方法，它通过在不同任务的适应矩阵之间引入正交约束来扩展 LoRA。这种方法在保持 LoRA 效率的同时，有助于防止灾难性遗忘。

## 方法特有参数
- `orthogonal_lambda` (float, 默认值=0.1)：正交约束损失的权重。
- `l2_lambda` (float, 默认值=0.01)：L2正则化损失的权重。
- `olora_history_path` (str, 默认值="olora_history")：存储O-LoRA历史文件的路径。
- `prev_task_id` (str, 可选)：用于正交约束的前一个任务ID。
- `current_task_id` (str, 必需)：当前任务的ID。

## 实现说明
- 所有的正交损失和L2损失计算都只使用 'default' adapter。
- 早期版本中使用的 'current' adapter 已被移除，因为它在实际计算中未被使用。

## 目录结构
- `olora.py`：O-LoRA的核心实现，包括正交损失计算和适配器管理。
- `olora_trainer.py`：将O-LoRA与HuggingFace的Trainer集成的自定义训练器类。
- `olora_workflow.py`：O-LoRA方法的训练工作流实现。
- `README.md`：英文文档。
- `README_zh.md`：中文文档。

## 引用
@article{wang2023orthogonal,
  title={Orthogonal subspace learning for language model continual learning},
  author={Wang, Xiao and Chen, Tianze and Ge, Qiming and Xia, Han and Bao, Rong and Zheng, Rui and Zhang, Qi and Gui, Tao and Huang, Xuanjing},
  journal={arXiv preprint arXiv:2310.14152},
  year={2023}
}