[ [English](README.md) | [中文](README_ZH.md) ]

# LAMOL: 基于语言建模的终身语言学习方法

## 主要思想
LAMOL是一种基于伪重放的持续学习方法，它使用语言模型来生成先前任务的伪样本。其核心思想是利用在先前任务上训练的模型生成能够捕获这些任务知识的合成数据，然后将这些数据与当前任务的数据结合进行训练。这种方法可以在不需要访问先前任务原始数据的情况下防止灾难性遗忘。

## 方法特定参数
- `use_lamol` (bool): 是否使用LAMOL方法进行持续学习。
- `lamol_show_gen` (bool): 是否添加表明样本为生成的前缀。
- `lamol_num_samples_per_task` (int): 每个任务生成的伪样本数量。
- `lamol_generation_temperature` (float): 伪样本生成的温度参数。
- `lamol_samples_dir` (str): 生成的伪样本保存目录。
- `previous_task_model` (str): 先前任务训练模型的路径。
- `current_task_id` (str): 当前任务的标识符。
- `prev_task_id` (str): 先前任务的标识符。

## 目录结构
- `lamol_trainer.py`: 实现了处理伪样本训练的LAMOL训练器类。
- `lamol_workflow.py`: 管理LAMOL训练工作流程和数据处理。
- `lamol.py`: 包含用于伪样本生成的核心LAMOL生成器类。
- `README.md`: 英文文档。
- `README_ZH.md`: 中文文档。

## 引用
@article{sun2019lamol,
  title={Lamol: Language modeling for lifelong language learning},
  author={Sun, Fan-Keng and Ho, Cheng-Hao and Lee, Hung-Yi},
  journal={arXiv preprint arXiv:1909.03329},
  year={2019}
}