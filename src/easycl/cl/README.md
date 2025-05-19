[ [English](README.md) | [中文](README_zh.md) ]

# Continual Learning Methods

This directory contains implementations of various continual learning methods for large language models. Each method is designed to prevent catastrophic forgetting while learning new tasks.

## Implemented Methods

1. **Elastic Weight Consolidation (EWC)**
   - Prevents catastrophic forgetting by adding a regularization term to the loss function based on parameter importance.
   - Uses Fisher Information Matrix to measure parameter importance.

2. **Learning Without Forgetting (LWF)**
   - Preserves knowledge from previous tasks using knowledge distillation.
   - Maintains performance on previous tasks without requiring access to old task data.

3. **Experience Replay**
   - Maintains performance by replaying a subset of data from previous tasks.
   - Uses a memory buffer to store and replay past experiences.

4. **LAMOL (Language Modeling for Lifelong Language Learning)**
   - Uses language models to generate pseudo samples of previous tasks.
   - Combines generated samples with current task data for training.

5. **O-LoRA (Orthogonal subspace learning for language model continual learning)**
   - Extends LoRA with orthogonal constraints between different tasks' adaptation matrices.
   - Maintains efficiency while preventing catastrophic forgetting.

6. **Gradient Episodic Memory (GEM)**
   - Projects gradients using episodic memory to prevent interference with past task performance.
   - Ensures gradient updates maintain or improve past task performance.

7. **I-LoRA (Interpolation-based LoRA)**
   - Constructs a dual-memory experience replay framework based on LoRA parameter interpolations.
   - Uses EMA for stable adapter updates and consistency loss.

8. **MOE-LoRA (Mixture of Experts with Low-Rank Adaptation)**
   - Combines Mixture of Experts architecture with LoRA for efficient model adaptation.
   - Uses multiple expert LoRA modules for task-specific specialization.

9. **ABSCL (ABSA LLM-CL)**
   - Trains shared and task-specific adapters with orthogonality constraints.
   - Uses feature statistics for task-specific adapter selection.

10. **Dynamic ConPet**
    - Combines shared and task-specific adapters with dynamic dataset classification.
    - Uses a classifier to route inputs to appropriate adapters.

11. **CL-MoE (Continual Learning with Task-specific Mixture of Experts)**
    - Uses dual momentum Mixture-of-Experts for continual visual question answering.
    - Implements task-level and instance-level routing.

12. **Self-Synthesized Rehearsal (SSR)**
    - Generates pseudo-samples from previous tasks using the model's generation capability.
    - Uses clustering for diverse sample selection.

13. **Pseudo Replay**
    - Simplified version of SSR for continual learning.
    - Uses base model to generate pseudo samples for previous tasks.

## Workflow Integration

All methods are integrated into the training workflow through the `tuner.py` file. Each method has its own specific parameters that can be configured through the training arguments. The integration is done by:

1. Adding method-specific parameters to the training configuration
2. Implementing custom trainer classes for each method
3. Creating workflow files that handle the training process
4. Adding validation checks for required parameters

## Original Implementations

Below are the links to the original implementations of each method. If a method is marked as "NA", it means the original authors did not release their implementation code.

1. **LAMOL**
   - Original Implementation: [Link to be filled]
   - Note: Original implementation by the authors

2. **O-LoRA**
   - Original Implementation: [Link to be filled]
   - Note: Original implementation by the authors

3. **I-LoRA**
   - Original Implementation: [Link to be filled]
   - Note: Original implementation by the authors

4. **MOE-LoRA**
   - Original Implementation: [Link to be filled]
   - Note: Original implementation by the authors

5. **ABSCL**
   - Original Implementation: [Link to be filled]
   - Note: Original implementation by the authors

6. **Dynamic ConPet**
   - Original Implementation: [Link to be filled]
   - Note: Original implementation by the authors

7. **CL-MoE**
   - Original Implementation: [Link to be filled]
   - Note: Original implementation by the authors

8. **SSR**
   - Original Implementation: [Link to be filled]
   - Note: Original implementation by the authors

## GitHub Links

(To be filled by the user) 