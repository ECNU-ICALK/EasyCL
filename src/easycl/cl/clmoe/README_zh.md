[ [English](README.md) | [中文](README_zh.md) ]

# cl-MoE：双动量专家混合体

## 核心思想
cl-MoE 是一种新颖的持续学习方法，它将特定任务的专家路由与 LoRA 适配器相结合，为持续视觉问题解答（VQA）提出了一种基于 MLLMs 的双动量专家混合物（CL-MoE）框架。我们将 MLLMs 与持续学习相结合，以利用 LLMs 中丰富的常识性知识。我们引入了双路由 MoE（RMoE）策略，利用任务级和实例级路由器来选择全局和局部专家，从而稳健地为最适合任务的专家分配权重。然后，我们设计了动态动量模型（MMoE），根据专家与任务/实例之间的关系动态更新专家参数，从而使模型在吸收新知识的同时保持现有知识。

## 参数
- `expert_num` (int)：MoE层中的专家数量（默认：8）
- `task_embedding_dim` (int)：任务嵌入向量的维度（默认：64）
- `use_cl_moe` (bool)：是否启用cl-MoE适配（默认：False）
- `prev_task_id` (str)：用于参数对齐的前一个任务ID
- `current_task_id` (str)：当前正在训练的任务ID
- `previous_task_model` (str)：前一个任务的模型检查点路径

## 目录文件说明
- `clmoe_adapter.py`：实现cl-MoE适配器架构和初始化
- `clmoe_loader.py`：处理模型加载和cl-MoE支持的配置
- `clmoe_trainer.py`：扩展训练器类，添加cl-MoE特定的训练逻辑
- `clmoe_workflow.py`：管理完整的cl-MoE训练工作流和参数对齐
- `peft/`：包含cl-MoE的PEFT相关实现的目录

## 引用
@article{huai2025cl,
  title={CL-MoE: Enhancing Multimodal Large Language Model with Dual Momentum Mixture-of-Experts for Continual Visual Question Answering},
  author={Huai, Tianyu and Zhou, Jie and Wu, Xingjiao and Chen, Qin and Bai, Qingchun and Zhou, Ze and He, Liang},
  journal={arXiv preprint arXiv:2503.00413},
  year={2025}
}