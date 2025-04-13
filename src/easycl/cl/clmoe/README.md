[ [English](README.md) | [中文](README_zh.md) ]

# CLIT-MoE: dual momentum Mixture-of Experts

## Core Idea
CLIT-MoE is a novel continual learning method that combines task-specific expert routing with LoRA adapters, propose an MLLMs-based dual momentum Mixture-ofExperts (CL-MoE) framework for continual visual question answering (VQA). We integrate MLLMs with continual learning to utilize the rich commonsense knowledge in LLMs. We introduce a Dual-Router MoE (RMoE) strategy to select the global and local experts using task-level and instance-level routers, to robustly assign weights to the experts most appropriate for the task. Then, we design a dynamic Momentum MoE (MMoE) to update the parameters of experts dynamically based on the relationships between the experts and tasks/instances, so that the model can absorb new knowledge while maintaining existing knowledge.

## Parameters
- `expert_num` (int): Number of experts in the MoE layer (default: 8)
- `task_embedding_dim` (int): Dimension of task embedding vectors (default: 64)
- `use_clit_moe` (bool): Whether to enable CLIT-MoE adaptation (default: False)
- `prev_task_id` (str): ID of the previous task for parameter alignment
- `current_task_id` (str): ID of the current task being trained
- `previous_task_model` (str): Path to the previous task's model checkpoint

## Files in Directory
- `clitmoe_adapter.py`: Implements the CLIT-MoE adapter architecture and initialization
- `clitmoe_loader.py`: Handles model loading and configuration with CLIT-MoE support
- `clitmoe_trainer.py`: Extends the trainer class with CLIT-MoE specific training logic
- `clitmoe_workflow.py`: Manages the complete CLIT-MoE training workflow and parameter alignment
- `peft/`: Directory containing PEFT-related implementations for CLIT-MoE

## Citation
@article{huai2025cl,
  title={CL-MoE: Enhancing Multimodal Large Language Model with Dual Momentum Mixture-of-Experts for Continual Visual Question Answering},
  author={Huai, Tianyu and Zhou, Jie and Wu, Xingjiao and Chen, Qin and Bai, Qingchun and Zhou, Ze and He, Liang},
  journal={arXiv preprint arXiv:2503.00413},
  year={2025}
}