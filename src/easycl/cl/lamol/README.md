[ [English](README.md) | [中文](README_ZH.md) ]

# LAMOL: LAnguage MOdeling for Lifelong Language Learning

## Main Idea
LAMOL is a pseudo-replay based continual learning method that uses language models to generate pseudo samples of previous tasks. The key idea is to use the model trained on previous tasks to generate synthetic data that captures the knowledge of those tasks, and then combine this data with the current task's data for training. This approach helps prevent catastrophic forgetting without requiring access to the original data from previous tasks.

## Method-Specific Parameters
- `use_lamol` (bool): Whether to use LAMOL method for continual learning.
- `lamol_show_gen` (bool): Whether to add a prefix indicating the sample is generated.
- `lamol_num_samples_per_task` (int): Number of pseudo samples to generate per task.
- `lamol_generation_temperature` (float): Temperature for pseudo sample generation.
- `lamol_samples_dir` (str): Directory to save generated pseudo samples.
- `previous_task_model` (str): Path to the model trained on previous task.
- `current_task_id` (str): Identifier for the current task.
- `prev_task_id` (str): Identifier for the previous task.

## Directory Structure
- `lamol_trainer.py`: Implements the LAMOL trainer class for handling training with pseudo samples.
- `lamol_workflow.py`: Manages the overall LAMOL training workflow and data processing.
- `lamol.py`: Contains the core LAMOL generator class for pseudo sample generation.
- `README.md`: English documentation.
- `README_ZH.md`: Chinese documentation.

## Citation
@article{sun2019lamol,
  title={Lamol: Language modeling for lifelong language learning},
  author={Sun, Fan-Keng and Ho, Cheng-Hao and Lee, Hung-Yi},
  journal={arXiv preprint arXiv:1909.03329},
  year={2019}
}
