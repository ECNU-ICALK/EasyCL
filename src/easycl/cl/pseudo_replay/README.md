[ [中文](README_zh.md) | English ]

# Pseudo Replay

## Main Idea
Pseudo Replay is a simplified version of the SSR (Selective Synthetic Replay) method for continual learning. It uses a base model to generate pseudo samples for previous tasks to prevent catastrophic forgetting.

## Method-specific Parameters
- `use_pseudo_replay`: Whether to use pseudo replay method
- `base_model_path`: Path to the base model for generating pseudo samples
- `num_samples_per_task`: Number of pseudo samples to generate per task
- `generation_temperature`: Temperature for pseudo sample generation
- `pseudo_samples_dir`: Directory to store generated pseudo samples
- `num_shots`: Number of examples to use for few-shot generation

## Directory Structure
- `pseudo_replay.py`: Core implementation of the Pseudo Replay method
- `pseudo_replay_trainer.py`: Custom trainer for Pseudo Replay
- `pseudo_replay_workflow.py`: Training workflow for Pseudo Replay
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