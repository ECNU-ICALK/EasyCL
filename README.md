<!-- Logo (placeholder for future) -->
<!-- ![LLaMA Factory](assets/logo.png) -->

[ [English](README.md) | [中文](README_zh.md) ]

<p align="center">
  <a href="#introduction">📚 Introduction</a> •
  <a href="#implemented-methods">🔍 Implemented Methods</a> •
  <a href="#installation">⚙️ Installation</a> •
  <a href="#workflow">🚀 Workflow</a> •
  <a href="#license">📝 License</a>
</p>

## Introduction

EasyCL is an extension of the LLaMA Factory framework, focusing on continual learning methods for large language models. It provides a comprehensive suite of tools and methods to address the problem of catastrophic forgetting in sequential learning tasks.

The framework integrates a variety of state-of-the-art continual learning techniques designed specifically for language models, allowing researchers and practitioners to easily implement, compare, and develop new methods.

For detailed implementation of the continual learning workflow, see [src/llamafactory/cl_workflow/README.md](src/llamafactory/cl_workflow/README.md).

## Implemented Methods

1. **Elastic Weight Consolidation (EWC)** - [View Implementation](src/llamafactory/cl/ewc/README.md)
   - Adds regularization based on parameter importance
   - Uses Fisher Information Matrix to measure importance

2. **Learning Without Forgetting (LWF)** - [View Implementation](src/llamafactory/cl/lwf/README.md)
   - Preserves knowledge using knowledge distillation
   - Maintains performance without requiring old task data

3. **Experience Replay** - [View Implementation](src/llamafactory/cl/replay/README.md)
   - Replays a subset of data from previous tasks
   - Uses memory buffer to store past experiences

4. **LAMOL (Language Modeling for Lifelong Language Learning)** - [View Implementation](src/llamafactory/cl/lamol/README.md)
   - Generates pseudo samples of previous tasks
   - Combines generated samples with current task data

5. **O-LoRA (Orthogonal subspace learning)** - [View Implementation](src/llamafactory/cl/olora/README.md)
   - Extends LoRA with orthogonal constraints
   - Prevents interference between task-specific adaptations

6. **Gradient Episodic Memory (GEM)** - [View Implementation](src/llamafactory/cl/gem/README.md)
   - Projects gradients using episodic memory
   - Prevents interference with past task performance

7. **I-LoRA (Interpolation-based LoRA)** - [View Implementation](src/llamafactory/cl/ilora/README.md)
   - Uses dual-memory experience replay framework
   - Interpolates LoRA parameters with EMA updates

8. **MOE-LoRA (Mixture of Experts with LoRA)** - [View Implementation](src/llamafactory/cl/moelora/README.md)
   - Combines MoE architecture with LoRA for adaptation
   - Uses multiple expert modules for specialization

9. **ABSCL (ABSA LLM-CL)** - [View Implementation](src/llamafactory/cl/abscl/README.md)
   - Trains shared and task-specific adapters
   - Uses feature statistics for adapter selection

10. **Dynamic ConPet** - [View Implementation](src/llamafactory/cl/dynamic_conpet/README.md)
    - Combines shared and task-specific adapters
    - Uses classifier for dynamic routing

11. **CLIT-MoE (Continual Learning with Task-specific MoE)** - [View Implementation](src/llamafactory/cl/clitmoe/README.md)
    - Uses dual momentum Mixture-of-Experts
    - Implements task-level and instance-level routing

12. **Self-Synthesized Rehearsal (SSR)** - [View Implementation](src/llamafactory/cl/ssr/README.md)
    - Generates pseudo-samples from previous tasks
    - Uses clustering for diverse sample selection

13. **Pseudo Replay** - [View Implementation](src/llamafactory/cl/pseudo_replay/README.md)
    - Simplified version of SSR
    - Uses base model to generate samples for previous tasks

For more details about the continual learning methods, see [src/llamafactory/cl/README.md](src/llamafactory/cl/README.md).

## Installation

```bash
git clone https://github.com/ECNU-ICALK/EasyCL.git
cd EazyCL
pip install -e .
```
Note that if you already have LLaMA-Factory installed in your environment, you may need to uninstall the existing one and perform the installation again.


## Workflow

### Train Only

```bash
llamafactory-cli cl_workflow --mode train_only --train_params ./configs/train_config.json
```

**Preview Result**: Executes training commands sequentially for tasks defined in `train_config_ewc.json`, applying parameter management between tasks.

### Evaluate Only

```bash
llamafactory-cli cl_workflow --mode eval_only --eval_params ./configs/eval_config.json
```

**Preview Result**: Executes evaluation command(s) specified in `eval_config.json` (e.g., evaluating a specific fine-tuned model on `cl_tasks`).

### Train Then Evaluate

```bash
llamafactory-cli cl_workflow --mode train_then_eval \
    --train_params ./configs/train_config_replay.json \
    --eval_params ./configs/eval_config.json
```

**Preview Result**: Executes training commands sequentially, then executes evaluation commands (evaluating base model and model after each task).

### Full Workflow (Train, Evaluate, Calculate Metrics)

```bash
llamafactory-cli cl_workflow --mode full_workflow \
    --train_params ./configs/train_config.json \
    --eval_params ./configs/eval_config.json
```

**Preview Result**: Executes training sequentially, then evaluates base/task models, and finally calculates and saves CL metrics (Last, Avg, BWT, FWT) to the evaluation output directory.

For detailed information about workflow configuration and CL metrics, see [src/llamafactory/cl_workflow/README.md](src/llamafactory/cl_workflow/README.md).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details. 
