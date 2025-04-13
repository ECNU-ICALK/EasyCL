[ [English](README.md) | [中文](README_zh.md) ]

# Dynamic ConPet：基于动态概念感知的参数高效持续学习方法

## 核心思想
Dynamic ConPet是一种新颖的持续学习方法，它结合了共享适配器和任务特定适配器，并配备动态数据集分类器。该方法使用共享适配器来捕获跨任务的通用知识，使用任务特定适配器进行专门学习。动态数据集分类器根据输入特征将其路由到合适的适配器，实现高效的知识迁移并防止灾难性遗忘。

## 参数
- `adapters_save_path` (str): 适配器保存路径（默认：None，将使用output_dir/adapters）
- `current_task_id` (str): 当前训练任务的ID
- `replay_task_list` (str): 用于重放的历史任务名称列表，以逗号分隔
- `maxsamples_list` (str): 每个历史任务使用的最大样本数列表，以逗号分隔
- `previous_task_dataset` (str): 包含历史任务数据集的目录路径

## 目录文件
- `dynamic_conpet.py`: 实现数据集分类器和Dynamic ConPet核心组件
- `dynamic_conpet_selector.py`: 处理基于数据集分类的适配器选择和路由
- `dynamic_conpet_trainer.py`: 扩展训练器类，添加Dynamic ConPet特定的训练逻辑
- `dynamic_conpet_workflow.py`: 管理完整的Dynamic ConPet训练工作流程
- `train_config.json`: 训练参数配置文件
- `__init__.py`: 包初始化文件

## 引用
@article{song2023conpet,
  title={Conpet: Continual parameter-efficient tuning for large language models},
  author={Song, Chenyang and Han, Xu and Zeng, Zheni and Li, Kuai and Chen, Chen and Liu, Zhiyuan and Sun, Maosong and Yang, Tao},
  journal={arXiv preprint arXiv:2309.14763},
  year={2023}
}