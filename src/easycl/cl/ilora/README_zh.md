# 改进的LoRA (I-LORA)

[ [English](README.md) | [中文](README_zh.md) ]

## 主要思想
I-LORA 是一种持续学习方法，通过构建基于 LoRA 参数插值的双记忆体验重放框架，对 LoRA 进行了改进。

## 方法特定参数
- `use_ilora` (bool): 是否启用I-LORA方法。默认值：False
- `ema_alpha` (float): 更新稳定适配器的EMA平滑系数。默认值：0.999
- `consistency_weight` (float): 可塑适配器和稳定适配器之间一致性损失的权重。默认值：1.0
- `ilora_buffer_size` (int): 内存缓冲区存储样本的最大数量。默认值：500
- `selective_update` (bool): 是否仅在可塑模型表现更好时进行选择性更新。默认值：False
- `min_update_threshold` (float): 选择性更新的最小阈值。默认值：0.1
- `hidden_state_layers` (list): 计算一致性损失的隐藏状态层列表。默认值：[-1]
- `save_ema_adapter` (bool): 是否单独保存EMA适配器。默认值：True
- `ema_adapter_path` (str): 保存/加载EMA适配器的路径。默认值：None
- `previous_task_model` (str): 前一个任务模型的路径。默认值：None
- `current_task_id` (str): 当前任务的ID。默认值：None
- `prev_task_id` (str): 前一个任务的ID。默认值：None

## 目录中的文件
- `ilora.py`: I-LORA的核心实现，包括缓冲区管理和一致性损失计算
- `ilora_adapter.py`: I-LORA的适配器初始化和管理
- `ilora_trainer.py`: 包含I-LORA特定训练逻辑的自定义训练器
- `ilora_workflow.py`: I-LORA的训练工作流管理
- `ilora_loader.py`: I-LORA的模型加载工具
- `README.md`: 英文文档
- `README_zh.md`: 中文文档

## 引用
@article{ren2024analyzing,
  title={Analyzing and reducing catastrophic forgetting in parameter efficient tuning},
  author={Ren, Weijieying and Li, Xinlong and Wang, Lei and Zhao, Tianxiang and Qin, Wei},
  journal={arXiv preprint arXiv:2402.18865},
  year={2024}
}
