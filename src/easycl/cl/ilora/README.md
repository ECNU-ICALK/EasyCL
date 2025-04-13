# Interpolation-based LoRA  (I-LORA)

[ [English](README.md) | [中文](README_zh.md) ]

## Main Idea
I-LORA is a continual learning method that improves upon LoRA by constructing a dual-memory experience replay
framework based on LoRA parameter interpolations. 

## Method-Specific Parameters
- `use_ilora` (bool): Whether to enable I-LORA method. Default: False
- `ema_alpha` (float): EMA smoothing coefficient for updating stable adapter. Default: 0.999
- `consistency_weight` (float): Weight for the consistency loss between plastic and stable adapters. Default: 1.0
- `ilora_buffer_size` (int): Maximum number of samples to store in memory buffer. Default: 500
- `selective_update` (bool): Whether to selectively update only when plastic model performs better. Default: False
- `min_update_threshold` (float): Minimum threshold for selective updates. Default: 0.1
- `hidden_state_layers` (list): List of hidden state layers to compute consistency loss on. Default: [-1]
- `save_ema_adapter` (bool): Whether to save the EMA adapter separately. Default: True
- `ema_adapter_path` (str): Path to save/load the EMA adapter. Default: None
- `previous_task_model` (str): Path to the model from previous task. Default: None
- `current_task_id` (str): ID of the current task. Default: None
- `prev_task_id` (str): ID of the previous task. Default: None

## Files in Directory
- `ilora.py`: Core implementation of I-LORA including buffer management and consistency loss computation
- `ilora_adapter.py`: Adapter initialization and management for I-LORA
- `ilora_trainer.py`: Custom trainer with I-LORA specific training logic
- `ilora_workflow.py`: Training workflow management for I-LORA
- `ilora_loader.py`: Model loading utilities for I-LORA
- `README.md`: English documentation
- `README_zh.md`: Chinese documentation

## Citation
@article{ren2024analyzing,
  title={Analyzing and reducing catastrophic forgetting in parameter efficient tuning},
  author={Ren, Weijieying and Li, Xinlong and Wang, Lei and Zhao, Tianxiang and Qin, Wei},
  journal={arXiv preprint arXiv:2402.18865},
  year={2024}
}

