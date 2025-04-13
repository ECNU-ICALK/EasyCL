# ABSCL (ABSA LLM-CL)

\[ [English](README.md) | [中文](README_zh.md) \]

## 1. Core Idea

ABSCL (ABSA LLM-CL) is a continual learning method designed for ABSA. It trains two types of adapters:

1.  **Shared Adapter:** Trained using a replay strategy, combining a subset of data from the current task and previous tasks. This adapter aims to capture general knowledge across tasks.
2.  **Task-Specific Adapter:** Trained using only the data from the current task.

During the training of the task-specific adapter, ABSCL applies two constraints inspired by O-LoRA:
*   **Orthogonality Constraint:** Encourages the task-specific adapter's weights to be orthogonal to the shared adapter's weights, promoting task-specific knowledge separation.
*   **L2 Regularization:** Applies L2 regularization specifically to the *shared* adapter's weights (loaded as a reference) while training the task-specific adapter, potentially preventing catastrophic forgetting in the shared knowledge base.

Additionally, ABSCL extracts feature statistics (mean vector and a shared covariance matrix) for each task after training its specific adapter. These statistics can be used later by `abscl_selector.py` to determine the most suitable adapter (task) for a given input sample based on Mahalanobis distance.

## 2. Specific Parameters

The following parameters are specific to the ABSCL method (refer to `finetuning_args`):

*   `--abscl_orthogonal_lambda`: (float, required) The weight for the orthogonality constraint loss between the task-specific adapter and the shared adapter.
*   `--abscl_shared_l2_lambda`: (float, required) The weight for the L2 regularization loss applied to the *shared* adapter weights during task-specific adapter training.
*   `--abscl_stats_path`: (str, optional) Path to save and load the feature statistics (mean vectors and covariance matrix). Defaults to `adapters_save_path/abscl_stats` if not provided.
*   `--current_task_id`: (str, required) An identifier for the current task. Used for naming the task-specific adapter and storing its statistics.
*   `--adapters_save_path`: (str, required) The base directory where the `shared_adapter` and task-specific adapters (named by `current_task_id`) will be saved. Feature statistics will also be stored relative to this path if `abscl_stats_path` is not set.

*Note: Parameters like `--replay_ratio`, `--replay_task_list`, `--maxsamples_list`, `--previous_task_dataset` are used by the ABSCL workflow for the replay strategy when training the shared adapter, but might be considered general replay parameters.*

## 3. File Descriptions

*   `abscl_workflow.py`: Orchestrates the main ABSCL training process. It handles the replay data preparation, trains the shared adapter, trains the task-specific adapter with ABSCL constraints, and triggers feature statistic extraction.
*   `abscl_trainer.py`: Defines the `ABSCLTrainer`, a custom Hugging Face `Trainer` subclass. It modifies the loss computation to include the orthogonality and shared L2 regularization terms, leveraging O-LoRA mechanisms for calculation.
*   `abscl.py`: Contains the `ABSCLFeatureExtractor` class responsible for extracting hidden state features (specifically, the last token's hidden state from the second-to-last layer) and computing/updating the per-task mean vectors and the shared covariance matrix.
*   `abscl_selector.py`: A script used *after* training multiple tasks with ABSCL. It loads the saved feature statistics and a test dataset, then assigns the most likely task adapter to each test sample based on the Mahalanobis distance of the sample's feature representation to the task means.

## 4. Citation

@article{ding2024boosting,
  title={Boosting large language models with continual learning for aspect-based sentiment analysis},
  author={Ding, Xuanwen and Zhou, Jie and Dou, Liang and Chen, Qin and Wu, Yuanbin and Chen, Chengcai and He, Liang},
  journal={arXiv preprint arXiv:2405.05496},
  year={2024}
}