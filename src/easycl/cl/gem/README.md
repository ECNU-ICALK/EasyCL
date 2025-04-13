# Gradient Episodic Memory (GEM)

[ [English](README.md) | [中文](README_zh.md) ]

## Main Idea
GEM is a continual learning method that prevents catastrophic forgetting by projecting gradients using episodic memory. It ensures that gradient updates for the current task do not interfere with past task performance by projecting them to a feasible region that maintains or improves past task performance.

## Method-Specific Parameters
- `use_gem` (bool): Whether to enable GEM method. Default: False
- `gem_memory_strength` (float): The strength of the GEM projection constraint. Higher values enforce stronger constraints. Default: 0.5
- `replay_ratio` (float): The ratio of samples to keep in memory from each past task. Default: 0.1
- `replay_task_list` (str): Comma-separated list of task names to use as memory. Default: None
- `maxsamples_list` (str): Comma-separated list of maximum samples to keep for each memory task. Default: None
- `previous_task_dataset` (str): Path to the directory containing previous task datasets. Default: None

## Files in Directory
- `gem_trainer.py`: Implements the GEM trainer class with gradient projection mechanism
- `gem_workflow.py`: Manages the GEM training workflow including memory management and dataset merging
- `train_gem.json`: Configuration file for GEM training
- `README.md`: English documentation
- `README_zh.md`: Chinese documentation

## Citation

@article{lopez2017gradient,
  title={Gradient episodic memory for continual learning},
  author={Lopez-Paz, David and Ranzato, Marc'Aurelio},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
