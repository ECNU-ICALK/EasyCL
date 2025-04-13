[ English | [中文](README_zh.md) ]

# 伪回放（Pseudo Replay）

## 主要思想
伪回放是SSR（选择性合成回放）方法的简化版本，用于持续学习。它使用基础模型为之前的任务生成伪样本，以防止灾难性遗忘。

## 方法特定参数
- `use_pseudo_replay`：是否使用伪回放方法
- `base_model_path`：用于生成伪样本的基础模型路径
- `num_samples_per_task`：每个任务生成的伪样本数量
- `generation_temperature`：生成伪样本时的温度参数
- `pseudo_samples_dir`：存储生成的伪样本的目录
- `num_shots`：用于少样本生成的示例数量

## 目录结构
- `pseudo_replay.py`：伪回放方法的核心实现
- `pseudo_replay_trainer.py`：伪回放的自定义训练器
- `pseudo_replay_workflow.py`：伪回放的训练工作流程
- `README.md`：英文文档
- `README_zh.md`：中文文档

## 引用
@article{rolnick2019experience,
  title={Experience replay for continual learning},
  author={Rolnick, David and Ahuja, Arun and Schwarz, Jonathan and Lillicrap, Timothy and Wayne, Gregory},
  journal={Advances in neural information processing systems},
  volume={32},
  year={2019}
}