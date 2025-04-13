# 经验回放

[ [English](README.md) | 中文 ]

## 主要思想
经验回放是一种持续学习方法，通过在训练新任务时回放部分先前任务的数据来维持模型在先前任务上的性能。这种方法通过定期复习和学习过去的经验来缓解灾难性遗忘问题。

## 方法特有参数
- `use_replay` (bool)：是否启用经验回放
- `replay_ratio` (float)：从每个先前任务中回放样本的比例（默认：1.0）
- `replay_task_list` (str)：要回放的先前任务列表，以逗号分隔
- `maxsamples_list` (str)：每个任务最大回放样本数列表，以逗号分隔
- `previous_task_dataset` (str)：包含先前任务数据集的目录

## 目录结构
- `replay_trainer.py`：经验回放训练器类，继承自CustomSeq2SeqTrainer
- `replay_workflow.py`：经验回放的主要工作流实现，包括数据集合并和训练逻辑
- `README.md`：英文文档
- `README_zh.md`：中文文档

## 引用
@article{rolnick2019experience,
  title={Experience replay for continual learning},
  author={Rolnick, David and Ahuja, Arun and Schwarz, Jonathan and Lillicrap, Timothy and Wayne, Gregory},
  journal={Advances in neural information processing systems},
  volume={32},
  year={2019}
}