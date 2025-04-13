# Self-Synthesized Rehearsal

[ English | [中文](README_zh.md) ]

## Main Idea
Self-Synthesized Rehearsal (SSR) is a continual learning method that leverages the model's own generation capability to create pseudo-samples from previous tasks. By generating and replaying these pseudo-samples during training on new tasks, SSR helps mitigate catastrophic forgetting without requiring access to the original data from previous tasks.

## Method-specific Parameters
- `use_ssr` (bool): Whether to enable SSR
- `base_model_path` (str): Path to the base model used for generating pseudo-samples
- `num_shots` (int): Number of examples to use for few-shot generation
- `generation_temperature` (float): Temperature for pseudo-sample generation
- `n_clusters` (int): Number of clusters for diverse sample selection
- `pseudo_sample_memory` (int): Maximum number of pseudo-samples to maintain
- `pseudo_samples_dir` (str): Directory to store generated pseudo-samples
- `previous_task_model` (str): Path to the model from the previous task for pseudo-sample refinement

## Directory Structure
- `ssr_trainer.py`: Trainer class for SSR, inheriting from CustomSeq2SeqTrainer
- `ssr_workflow.py`: Main workflow implementation for SSR, including pseudo-sample generation and training logic
- `ssr.py`: Core implementation of SSR method, including pseudo-sample generation, refinement, and selection
- `README.md`: English documentation
- `README_zh.md`: Chinese documentation

## Citation
@article{huang2024mitigating,
  title={Mitigating catastrophic forgetting in large language models with self-synthesized rehearsal},
  author={Huang, Jianheng and Cui, Leyang and Wang, Ante and Yang, Chengyi and Liao, Xinting and Song, Linfeng and Yao, Junfeng and Su, Jinsong},
  journal={arXiv preprint arXiv:2403.01244},
  year={2024}
}