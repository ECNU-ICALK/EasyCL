[ [English](README.md) | [中文](README_zh.md) ]

# Learning Without Forgetting (LWF)

## Main Idea
Learning Without Forgetting (LWF) is a continual learning method that preserves knowledge from previous tasks while learning new ones. It achieves this by using knowledge distillation to maintain the model's performance on previously learned tasks without requiring access to the old task data.

## Parameters
- `use_lwf` (bool): Whether to enable LWF during training.
- `lwf_temperature` (float): Temperature parameter for softening probability distributions in knowledge distillation (default: 2.0).
- `lwf_alpha` (float): Weight parameter for balancing the distillation loss and cross-entropy loss (default: 0.5).
- `previous_task_model` (str): Path to the model from the previous task for knowledge distillation.

## Directory Structure
- `__init__.py`: Module initialization file
- `config.json`: Configuration file for LWF parameters
- `lwf.py`: Core implementation of the LWF method
- `lwf_trainer.py`: Custom trainer class with LWF functionality
- `lwf_workflow.py`: Training workflow implementation for LWF
- `README.md`: Documentation in English
- `README_zh.md`: Documentation in Chinese

## Citation
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
