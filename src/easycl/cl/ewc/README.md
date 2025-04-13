[ [English](README.md) | [中文](README_zh.md) ]

# Elastic Weight Consolidation (EWC)

## Main Idea
EWC is a continual learning method that prevents catastrophic forgetting by adding a regularization term to the loss function. This term penalizes large changes to parameters that were important for previous tasks, where importance is measured using the Fisher Information Matrix.

## Parameters
- `use_ewc` (bool): Whether to enable EWC during training.
- `ewc_lambda` (float): The weight of the EWC regularization term (default: 0.5).
- `ewc_num_samples` (int): Number of samples to use when computing the Fisher Information Matrix.
- `previous_task_data` (str): Path to the dataset of the previous task, used for computing Fisher Information.

## Files
- `ewc.py`: Core implementation of the EWC algorithm and Fisher Information computation.
- `ewc_trainer.py`: Custom trainer class that integrates EWC into the training process.
- `ewc_workflow.py`: Training workflow that handles EWC initialization and task transitions.
- `config.json`: Configuration file containing EWC-specific parameters.
- `__init__.py`: Module initialization file.

## Citation
@article{kirkpatrick2017overcoming,
  title={Overcoming catastrophic forgetting in neural networks},
  author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
  journal={Proceedings of the national academy of sciences},
  volume={114},
  number={13},
  pages={3521--3526},
  year={2017},
  publisher={National Academy of Sciences}
}