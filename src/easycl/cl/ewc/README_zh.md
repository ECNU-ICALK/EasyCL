[ [English](README.md) | [中文](README_zh.md) ]

# 弹性权重巩固 (EWC)

## 主要思想
EWC是一种持续学习方法，通过在损失函数中添加正则化项来防止灾难性遗忘。该正则化项惩罚对先前任务中重要参数的大幅改变，其中参数的重要性是通过Fisher信息矩阵来衡量的。

## 参数
- `use_ewc` (bool): 是否在训练过程中启用EWC。
- `ewc_lambda` (float): EWC正则化项的权重（默认值：0.5）。
- `ewc_num_samples` (int): 计算Fisher信息矩阵时使用的样本数量。
- `previous_task_data` (str): 前一个任务数据集的路径，用于计算Fisher信息。

## 文件
- `ewc.py`: EWC算法和Fisher信息计算的核心实现。
- `ewc_trainer.py`: 将EWC集成到训练过程中的自定义训练器类。
- `ewc_workflow.py`: 处理EWC初始化和任务转换的训练工作流。
- `config.json`: 包含EWC特定参数的配置文件。
- `__init__.py`: 模块初始化文件。

## 引用
@article{kirkpatrick2017overcoming,
  title={Overcoming catastrophic forgetting in neural networks},
  author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
  journal={Proceedings of the national academy of sciences},
  volume={114},
  number={13},
  pages={3521--3526},
  year={2017},
  publisher={National Academy of Sciences}
} 