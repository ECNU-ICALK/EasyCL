[ [English](README.md) | [中文](README_zh.md) ]

# 不忘学习法 (Learning Without Forgetting, LWF)

## 主要思想
不忘学习法（LWF）是一种持续学习方法，它在学习新任务的同时保持对先前任务的知识。通过知识蒸馏技术，该方法能够在不需要访问旧任务数据的情况下，维持模型在先前学习任务上的性能。

## 参数
- `use_lwf` (bool): 是否启用LWF训练
- `lwf_temperature` (float): 知识蒸馏中用于软化概率分布的温度参数（默认值：2.0）
- `lwf_alpha` (float): 用于平衡蒸馏损失和交叉熵损失的权重参数（默认值：0.5）
- `previous_task_model` (str): 用于知识蒸馏的前一个任务模型的路径

## 目录结构
- `__init__.py`: 模块初始化文件
- `config.json`: LWF参数配置文件
- `lwf.py`: LWF方法的核心实现
- `lwf_trainer.py`: 包含LWF功能的自定义训练器类
- `lwf_workflow.py`: LWF的训练工作流实现
- `README.md`: 英文文档
- `README_zh.md`: 中文文档

## 引用
@article{li2017learning,
  title={Learning without forgetting},
  author={Li, Zhizhong and Hoiem, Derek},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={40},
  number={12},
  pages={2935--2947},
  year={2017},
  publisher={IEEE}
}