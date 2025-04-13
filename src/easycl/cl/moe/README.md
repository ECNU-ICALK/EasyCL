[ [English](README.md) | [中文](README_zh.md) ]

# MOE-LoRA: Mixture of Experts with Low-Rank Adaptation

## Main Idea
MOE-LoRA combines the Mixture of Experts (MoE) architecture with Low-Rank Adaptation (LoRA) to enable more efficient and flexible model adaptation. It introduces multiple expert LoRA modules that can be dynamically selected based on the input task, allowing for better task-specific specialization while maintaining the parameter efficiency of LoRA.

## Method-specific Parameters
- `expert_num`: Number of expert modules in MOE-LoRA (default: 1, which falls back to standard LoRA)
- `task_embedding_dim`: Dimension of the task embedding used for expert routing (default: 64)

## Files in this Directory
- `moelora_adapter.py`: Implements the core MOE-LoRA adapter initialization and setup
- `moelora_loader.py`: Handles model loading with MOE-LoRA support
- `moelora_trainer.py`: Extends the trainer class to support MOE-LoRA training
- `moelora_workflow.py`: Implements the complete training workflow for MOE-LoRA

## Citation
@article{chen2024coin,
  title={CoIN: A Benchmark of Continual Instruction Tuning for Multimodel Large Language Models},
  author={Chen, Cheng and Zhu, Junchen and Luo, Xu and Shen, Hengtao and Song, Jingkuan and Gao, Lianli},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={57817--57840},
  year={2024}
}