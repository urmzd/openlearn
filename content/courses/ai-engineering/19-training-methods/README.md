# Choosing Training and Adaptation Methods

Every model you serve was made by a training pipeline, and most of the decisions that determine its quality, its cost, and its behavior were taken there rather than at inference time. This chapter is about that pipeline: how a base model is pretrained, how its memory and compute are budgeted and split across many GPUs, and how post-training (supervised fine-tuning, preference optimization, and reinforcement learning with verifiable rewards) turns a next-token predictor into an assistant or a specialist. It ends where most practitioners actually work: scoping a fine-tune for a real task, deciding whether one is needed at all, and estimating what it will cost.

The organizing idea is that every training job is three budgets. **FLOPs**, roughly $6ND$, are paid in GPU-hours. **Memory**, bytes per parameter plus activations, is paid in GPU count. **Data**, tokens of the right kind, is paid in people. Pretraining usually has a large compute and data budget; post-training can be limited by data quality, rollout generation, scoring, or optimization, depending on the recipe. Distributed training techniques trade memory for communication; parameter-efficient fine-tuning trades capacity for memory. If you can do the arithmetic for all three budgets in your head during a meeting, and you know which kind of data each post-training method consumes, you can scope any fine-tuning engagement and hold a credible conversation with the researchers who run the large runs.

The focused parts follow the order the pipeline runs: the stages and the people who own them, the mathematics of pretraining, memory and parallelism, post-training, parameter-efficient fine-tuning, scoping, and finally the same training step written in three frameworks. A companion [code directory](code/README.md) holds small runnable files that implement the formulas and assert that they hold, so you can check every number in the parts yourself.

> **Learning objectives**
> - Estimate the training FLOPs, GPU-hours, and dollar cost of a run from $N$, $D$, the hardware peak, and an assumed MFU, and identify the assumptions and unmodeled costs rather than treating the estimate as a hardware measurement.
> - Account for a common 16-byte-per-parameter AdamW state model, estimate activation memory, and choose among ZeRO/FSDP, tensor, pipeline, context, and expert parallelism by naming the dimension that does not fit.
> - Derive the DPO loss from the KL-constrained RLHF objective, and explain how GRPO removes the critic and why it needs prompts the model sometimes solves.
> - Estimate when LoRA or QLoRA can fit a one-GPU fine-tune, including activation memory and the remaining base-model computation.
> - Scope a fine-tune eval-first: decide whether to fine-tune, pick the method that matches the available data, and estimate the cost.

## Before you begin

This chapter assumes [Deep Learning](../02-deep-learning/README.md) (backpropagation, Adam, transformers), the policy-gradient material in [Reinforcement Learning](../04-reinforcement-learning/README.md), and the lifecycle overview in [Foundation Models and Architectures](../05-foundation-models/README.md). The framework basics, PyTorch versus JAX, DDP, the FSDP concept, and `jit`, are covered in [Training and Frameworks](../16-training-frameworks/README.md) in *AI Engineering*, and this chapter does not repeat them. The objectives lean on cross entropy, KL divergence, and the Bradley-Terry likelihood, which you can review in [Information Theory](../../mathematics/17-information-theory/README.md) and [Probability and Statistics](../../mathematics/13-probability-statistics/README.md) in *Mathematics*.

The best single references are free. Stanford's [*CS336: Language Modeling from Scratch*](https://stanford-cs336.github.io/) has lectures and assignments that build a tokenizer, a transformer, parallel training, scaling laws, data filtering, SFT, and RL. Hugging Face's [*The Ultra-Scale Playbook*](https://huggingface.co/spaces/nanotron/ultrascale-playbook) is the best single document on DP, TP, PP, CP, EP, and ZeRO with measured numbers. The full list of papers and supplementary material is under "Further reading" at the end of the chapter.

## In this chapter

Start with [how models learn](08-how-models-learn.md) for the high-level learning signals and training loop, then use the focused parts for the arithmetic and engineering decisions.

1. [Training lifecycle and learning signals](01-training-lifecycle.md): Identify the data and objective for each training stage.
2. [Pretraining compute and scaling](02-pretraining-compute-and-scaling.md): Estimate training compute and distinguish a cost model from a measurement.
3. [Memory, precision, and parallelism](03-memory-precision-and-parallelism.md): Account for model states and select the dimension to shard.
4. [Choosing post-training signals](04-choosing-post-training-signals.md): Match demonstrations, preferences, rewards, and teacher distributions to objectives.
5. [Parameter-efficient fine-tuning](05-parameter-efficient-fine-tuning.md): Explain the memory and computation saved by freezing weights.
6. [Evaluation and fine-tune scope](06-evaluation-and-fine-tune-scope.md): Choose an adaptation method from a measured failure and available data.
7. [The training step and validation](07-training-step-and-validation.md): Connect framework operations to the same optimization step and validate the result.
8. [How models learn](08-how-models-learn.md): supervised, self-supervised, preference, reward, and teacher signals; batch/step accounting and evaluation.
