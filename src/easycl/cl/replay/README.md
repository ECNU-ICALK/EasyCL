# Experience Replay

[ English | [中文](README_zh.md) ]

## Main Idea
Experience Replay is a continual learning method that maintains the model's performance on previous tasks by replaying a subset of data from previously learned tasks during the training of new tasks. This approach helps mitigate catastrophic forgetting by periodically reviewing and learning from past experiences.

## Method-specific Parameters
- `use_replay` (bool): Whether to enable experience replay
- `replay_ratio` (float): The ratio of samples to replay from each previous task (default: 1.0)
- `replay_task_list` (str): Comma-separated list of previous tasks to replay from
- `maxsamples_list` (str): Comma-separated list of maximum samples to replay from each task
- `previous_task_dataset` (str): Directory containing previous task datasets

## Directory Structure
- `replay_trainer.py`: Trainer class for experience replay, inheriting from CustomSeq2SeqTrainer
- `replay_workflow.py`: Main workflow implementation for experience replay, including dataset merging and training logic
- `README.md`: English documentation
- `README_zh.md`: Chinese documentation

## Citation
@article{rolnick2019experience,
  title={Experience replay for continual learning},
  author={Rolnick, David and Ahuja, Arun and Schwarz, Jonathan and Lillicrap, Timothy and Wayne, Gregory},
  journal={Advances in neural information processing systems},
  volume={32},
  year={2019}
}