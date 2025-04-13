# 梯度情景记忆 (GEM)

[ [English](README.md) | [中文](README_zh.md) ]

## 主要思想
GEM是一种持续学习方法，通过使用情景记忆投影梯度来防止灾难性遗忘。它通过将当前任务的梯度投影到一个可行区域，确保梯度更新不会干扰过去任务的性能，从而维持或改善过去任务的表现。

## 方法特定参数
- `use_gem` (bool): 是否启用GEM方法。默认值：False
- `gem_memory_strength` (float): GEM投影约束的强度。较高的值会强制执行更强的约束。默认值：0.5
- `replay_ratio` (float): 从每个过去任务中保留的样本比例。默认值：0.1
- `replay_task_list` (str): 用作记忆的任务名称列表，以逗号分隔。默认值：None
- `maxsamples_list` (str): 每个记忆任务要保留的最大样本数列表，以逗号分隔。默认值：None
- `previous_task_dataset` (str): 包含先前任务数据集的目录路径。默认值：None

## 目录中的文件
- `gem_trainer.py`: 实现带有梯度投影机制的GEM训练器类
- `gem_workflow.py`: 管理GEM训练工作流，包括记忆管理和数据集合并
- `train_gem.json`: GEM训练的配置文件
- `README.md`: 英文文档
- `README_zh.md`: 中文文档

## 引用

@article{lopez2017gradient,
  title={Gradient episodic memory for continual learning},
  author={Lopez-Paz, David and Ranzato, Marc'Aurelio},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}