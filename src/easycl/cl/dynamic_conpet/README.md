[ [English](README.md) | [中文](README_zh.md) ]

# Dynamic ConPet: Continual Learning with Dynamic Concept-aware Parameter Efficient Tuning

## Core Idea
Dynamic ConPet is a novel continual learning method that combines shared and task-specific adapters with a dynamic dataset classifier. It uses a shared adapter to capture common knowledge across tasks and task-specific adapters for specialized learning. The dynamic dataset classifier helps route inputs to appropriate adapters based on their characteristics, enabling efficient knowledge transfer and preventing catastrophic forgetting.

## Parameters
- `adapters_save_path` (str): Path to save the adapters (default: None, will use output_dir/adapters)
- `current_task_id` (str): ID of the current task being trained
- `replay_task_list` (str): Comma-separated list of historical task names for replay
- `maxsamples_list` (str): Comma-separated list of maximum samples to use from each historical task
- `previous_task_dataset` (str): Path to the directory containing historical task datasets

## Files in Directory
- `dynamic_conpet.py`: Implements the dataset classifier and core Dynamic ConPet components
- `dynamic_conpet_selector.py`: Handles adapter selection and routing based on dataset classification
- `dynamic_conpet_trainer.py`: Extends the trainer class with Dynamic ConPet specific training logic
- `dynamic_conpet_workflow.py`: Manages the complete Dynamic ConPet training workflow
- `train_config.json`: Configuration file for training parameters
- `__init__.py`: Package initialization file

## Citation

@article{song2023conpet,
  title={Conpet: Continual parameter-efficient tuning for large language models},
  author={Song, Chenyang and Han, Xu and Zeng, Zheni and Li, Kuai and Chen, Chen and Liu, Zhiyuan and Sun, Maosong and Yang, Tao},
  journal={arXiv preprint arXiv:2309.14763},
  year={2023}
}