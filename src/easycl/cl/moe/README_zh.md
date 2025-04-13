[ [English](README.md) | [中文](README_zh.md) ]

# MOE-LoRA: 专家混合的低秩适应方法

## 主要思想
MOE-LoRA将专家混合（Mixture of Experts，MoE）架构与低秩适应（Low-Rank Adaptation，LoRA）相结合，实现更高效和灵活的模型适应。它引入了多个专家LoRA模块，这些模块可以根据输入任务动态选择，在保持LoRA参数效率的同时实现更好的任务特定专门化。

## 方法特有参数
- `expert_num`：MOE-LoRA中的专家模块数量（默认值：1，此时退化为标准LoRA）
- `task_embedding_dim`：用于专家路由的任务嵌入维度（默认值：64）

## 目录下的文件
- `moelora_adapter.py`：实现核心MOE-LoRA适配器的初始化和设置
- `moelora_loader.py`：处理支持MOE-LoRA的模型加载
- `moelora_trainer.py`：扩展训练器类以支持MOE-LoRA训练
- `moelora_workflow.py`：实现MOE-LoRA的完整训练工作流程

## 引用
@article{chen2024coin,
  title={CoIN: A Benchmark of Continual Instruction Tuning for Multimodel Large Language Models},
  author={Chen, Cheng and Zhu, Junchen and Luo, Xu and Shen, Hengtao and Song, Jingkuan and Gao, Lianli},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={57817--57840},
  year={2024}
}