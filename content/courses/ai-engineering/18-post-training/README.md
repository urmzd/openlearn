# Post-training

A pretrained language model continues text; it does not yet answer questions in a reliable format, prefer good answers to merely plausible ones, or fit in a small deployment. This optional chapter follows the [TinyStories capstone](../capstones/01-tinystories.md) and adapts its small causal language model with four kinds of signal: supervised demonstrations, pairwise preferences, verifiable rewards, and a teacher's distribution. Together they are the recipe of the [post-trained capstone](../capstones/02-post-trained.md).

The methods are choices matched to the available learning signal. SFT learns from demonstrations; DPO learns from fixed preferences; PPO and GRPO learn from sampled outcomes; distillation transfers teacher behavior. For a conceptual reading order, start with [policy and data distributions](05-on-policy-and-off-policy.md), then [PPO and GAE](06-ppo-with-gae.md) and [reward design](07-rewards-and-preference-models.md), before the GRPO implementation. Follow the rollout and distillation engineering parts when moving beyond the scalar examples.

> **Learning objectives**
> - Render chat data with the serving template, mask the loss to assistant tokens, and pack examples with document boundaries.
> - Derive DPO from the KL-regularized reward objective and implement DPO and IPO stably.
> - Implement GRPO's group-normalized advantages and clipped surrogate objective.
> - Implement forward and reverse KL distillation with temperature, and choose a direction.
> - Derive PPO's critic targets and GAE, distinguish behavior from reference policies, and diagnose rollout/learner mismatch.
> - Audit rewards, normalization, policy lag, and distillation data before scaling the system.

## Before you begin

You should have a trained model from the [TinyStories capstone](../capstones/01-tinystories.md), and be comfortable with the cross-entropy of [Losses with a fused backward](../06-numerics-and-autograd/04-losses-with-fused-backward.md), the adapters of [LoRA: low-rank adaptation with PiSSA initialization and merging](../12-objectives-and-adaptation/06-lora-pissa-and-merge.md), and the sampling of [Sampling and logit processors](../21-inference/01-sampling-and-logit-processors.md). From *Mathematics*, the chapter uses the KL divergence of [Entropy, cross-entropy, KL, JS, and the k3 estimator](../../mathematics/17-information-theory/01-entropy-cross-entropy-and-kl.md).

## In this chapter

1. [Supervised fine-tuning: chat rendering, assistant-only loss, packing](01-supervised-fine-tuning.md): rendering conversations with the serving tokenizer's template, training only on assistant tokens, and packing short examples into rows without letting them attend to each other.
2. [Direct preference optimization and IPO](02-direct-preference-optimization.md): deriving a classification-style preference loss from the KL-regularized RLHF objective, and IPO's bounded alternative.
3. [Group relative policy optimization with verifiable rewards](03-grpo-with-verifiable-rewards.md): turning checker scores on sampled rollouts into a clipped policy-gradient update with a per-prompt baseline.
4. [Distillation with forward and reverse KL](04-distillation.md): matching a teacher's distribution in log space, the mass-covering and mode-seeking directions, and temperature scaling.
5. [On-policy, off-policy, offline, and online learning](05-on-policy-and-off-policy.md): behavior versus target distributions, importance sampling, support, and replay limits.
6. [PPO with a critic and GAE](06-ppo-with-gae.md): token rewards, terminal/truncation masks, actor/critic losses, and drift controls.
7. [Rewards and preference models](07-rewards-and-preference-models.md): comparison losses, verifiable outcomes, reward exploitation, and independent evaluation.
8. [Rollout systems and GRPO variants](08-rollout-systems-and-grpo-variants.md): actor/learner layouts, revision lag, token normalization, and useful learning throughput.
9. [Distillation data and deployment](09-distillation-data-and-deployment.md): sequence/logit/feature targets, student prefixes, vocabulary constraints, and compression validation.

10. [SFT recipes and data strategies](10-sft-recipes-and-data-strategies.md): completion/assistant/final-turn masking, mixtures, synthetic and rejection-sampled demonstrations, weighting, and parameter updates.
11. [Preference optimization variants](11-preference-optimization-variants.md): IPO, label-smoothed/robust DPO, KTO, ORPO, SimPO, online data, and the wider weighting/reference/loss families.

## Further reading

- The [Training and Post-Training](../19-training-methods/README.md) and [Reinforcement Learning](../04-reinforcement-learning/README.md) chapters of *AI Engineering*.
