# 自合成回放

[ [English](README.md) | 中文 ]

## 主要思想
自合成回放（Self-Synthesized Rehearsal，SSR）是一种持续学习方法，它利用模型自身的生成能力来创建先前任务的伪样本。通过在新任务训练期间生成和回放这些伪样本，SSR可以帮助缓解灾难性遗忘，而无需访问先前任务的原始数据。

## 方法特定参数
- `use_ssr` (bool): 是否启用SSR
- `base_model_path` (str): 用于生成伪样本的基础模型路径
- `num_shots` (int): 用于少样本生成的示例数量
- `generation_temperature` (float): 伪样本生成的温度参数
- `n_clusters` (int): 用于多样化样本选择的聚类数量
- `pseudo_sample_memory` (int): 维护的最大伪样本数量
- `pseudo_samples_dir` (str): 存储生成的伪样本的目录
- `previous_task_model` (str): 用于伪样本优化的前一任务模型路径

## 目录结构
- `ssr_trainer.py`: SSR训练器类，继承自CustomSeq2SeqTrainer
- `ssr_workflow.py`: SSR的主要工作流实现，包括伪样本生成和训练逻辑
- `ssr.py`: SSR方法的核心实现，包括伪样本生成、优化和选择
- `README.md`: 英文文档
- `README_zh.md`: 中文文档

## 引用
@article{huang2024mitigating,
  title={Mitigating catastrophic forgetting in large language models with self-synthesized rehearsal},
  author={Huang, Jianheng and Cui, Leyang and Wang, Ante and Yang, Chengyi and Liao, Xinting and Song, Linfeng and Yao, Junfeng and Su, Jinsong},
  journal={arXiv preprint arXiv:2403.01244},
  year={2024}
}