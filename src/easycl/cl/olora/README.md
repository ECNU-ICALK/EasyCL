[ [English](README.md) | [中文](README_zh.md) ]

# O-LoRA: Orthogonal subspace learning for language model continual learning

## Core Idea
O-LoRA is a continual learning method that extends LoRA by introducing orthogonal constraints between different tasks' adaptation matrices. This approach helps prevent catastrophic forgetting while maintaining the efficiency of LoRA.

## Method-Specific Parameters
- `orthogonal_lambda` (float, default=0.1): The weight of orthogonal constraint loss.
- `l2_lambda` (float, default=0.01): The weight of L2 regularization loss.
- `olora_history_path` (str, default="olora_history"): Path to store O-LoRA history files.
- `prev_task_id` (str, optional): ID of the previous task for orthogonal constraint.
- `current_task_id` (str, required): ID of the current task.

## Directory Structure
- `olora.py`: Core implementation of O-LoRA, including orthogonal loss computation and adapter management.
- `olora_trainer.py`: Custom trainer class that integrates O-LoRA with HuggingFace's Trainer.
- `olora_workflow.py`: Training workflow implementation for O-LoRA method.
- `README.md`: English documentation.
- `README_zh.md`: Chinese documentation.

## Citation
@article{wang2023orthogonal,
  title={Orthogonal subspace learning for language model continual learning},
  author={Wang, Xiao and Chen, Tianze and Ge, Qiming and Xia, Han and Bao, Rong and Zheng, Rui and Zhang, Qi and Gui, Tao and Huang, Xuanjing},
  journal={arXiv preprint arXiv:2310.14152},
  year={2023}
}
