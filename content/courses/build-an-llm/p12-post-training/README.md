# Post-training

A pretrained language model continues text; it does not yet answer questions in a reliable format, prefer good answers to merely plausible ones, or fit in a small deployment. This optional chapter follows the [TinyStories capstone](../capstones/01-tinystories.md) and adapts its small causal language model with four kinds of signal: supervised demonstrations, pairwise preferences, verifiable rewards, and a teacher's distribution. Together they are the recipe of the [post-trained capstone](../capstones/02-post-trained.md).

The sections build on each other. Supervised fine-tuning teaches the response format and produces the reference policy that the later objectives measure against. Direct preference optimization then learns from fixed pairs of chosen and rejected answers, with no reward model. Group relative policy optimization learns from the model's own rollouts when answers can be checked automatically. Distillation offers an alternative to both, training a student to match a teacher token by token.

> **Learning objectives**
> - Render chat data with the serving template, mask the loss to assistant tokens, and pack examples with document boundaries.
> - Derive DPO from the KL-regularized reward objective and implement DPO and IPO stably.
> - Implement GRPO's group-normalized advantages and clipped surrogate objective.
> - Implement forward and reverse KL distillation with temperature, and choose a direction.

## Before you begin

You should have a trained model from the [TinyStories capstone](../capstones/01-tinystories.md), and be comfortable with the cross-entropy of [Losses with a fused backward](../p00-foundations/03-losses-with-fused-backward.md), the adapters of [LoRA with PiSSA init and merge](../p06-objectives/06-lora-pissa-and-merge.md), and the sampling of [Sampling and logit processors](../p08-inference/01-sampling-and-logit-processors.md). From *Mathematics for Computing*, the chapter uses the KL divergence of [Entropy, cross-entropy, KL, JS, and the k3 estimator](../../mathematics-for-computing/11-information-theory/01-entropy-cross-entropy-and-kl.md).

## In this chapter

1. [Supervised fine-tuning: chat rendering, assistant-only loss, packing](01-supervised-fine-tuning.md): rendering conversations with the serving tokenizer's template, training only on assistant tokens, and packing short examples into rows without letting them attend to each other.
2. [Direct preference optimization and IPO](02-direct-preference-optimization.md): deriving a classification-style preference loss from the KL-regularized RLHF objective, and IPO's bounded alternative.
3. [Group relative policy optimization with verifiable rewards](03-grpo-with-verifiable-rewards.md): turning checker scores on sampled rollouts into a clipped policy-gradient update with a per-prompt baseline.
4. [Distillation with forward and reverse KL](04-distillation.md): matching a teacher's distribution in log space, the mass-covering and mode-seeking directions, and temperature scaling.

## Further reading

- The [Training and Post-Training](../../machine-learning/07-training-and-post-training/README.md) and [Reinforcement Learning](../../machine-learning/03-reinforcement-learning/README.md) chapters of *Machine Learning*.
