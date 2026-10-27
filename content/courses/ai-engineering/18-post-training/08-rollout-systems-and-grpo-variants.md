# Rollout systems, GRPO variants, and training diagnostics

RL training repeatedly runs an inference workload inside the learning loop. Generation, scoring, optimization, and weight synchronization must fit together. A fast serving engine helps only when it supplies valid, fresh, correctly scored data to the learner.

> **Learning objectives**
> - Design synchronous and asynchronous actor/learner flows.
> - Explain group size, normalization, and length-weighting choices in GRPO.
> - Diagnose useful learning throughput rather than raw generation throughput.

## Connect learner and rollout workers

```text
versioned prompts → rollout actors at revision k → checker/reward model
                         ↓ behavior probabilities       ↓ scores
                     immutable trajectory records → learner updates
                         ↑                         ↓
                    new weights ← validated revision k+1
```

In a **synchronous** design, actors finish a round before the learner updates and publishes new weights. It simplifies revision accounting but can idle GPUs during scoring or updates. In an **asynchronous** design, actors collect while the learner updates. Queues improve utilization but introduce policy lag, mixed revisions, and backpressure. The learner should define a maximum acceptable lag and reject, refresh, or explicitly correct stale samples according to its algorithm.

**Colocation** places learner and actor on shared devices, often with weight swapping or time-sharing. It saves hardware and potentially transfer work but complicates memory scheduling. **Disaggregation** gives generation and learning separate pools, allowing different TP/DP layouts and independent sizing. It adds weight-transfer cost and consistency requirements. Publish a new revision only after all necessary actor shards agree; mixing layer weights from two revisions creates a different, unintended policy.

SGLang and vLLM are examples of rollout runtimes. A trainer supplies data flow, objective, backpropagation, optimizer, and checkpoint state. An inference server is not by itself a PPO or GRPO trainer. Compare their [local serving contracts](../20-inference-engines/14-local-serving-and-validation.md) on fixed tokens before integrating them with learning.

## From a scalar toy loss to token-level GRPO

For prompt $x$, sample $G$ completions $y_i$ from an old policy and assign outcome rewards $R_i$. Define group-normalized advantages $A_i$ and token ratios $\rho_{i,t}$. A common original-style surrogate averages each sequence before averaging sequences:

$$
J=\frac1G\sum_i\frac1{T_i}\sum_{t=1}^{T_i}
\left[\min(\rho_{i,t}A_i,\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i)-\beta k_{i,t}\right].
$$

$T_i$ counts valid generated actions according to the EOS/mask convention. Padding and prompt tokens do not participate. The same sequence advantage is broadcast across its generated tokens for outcome supervision; a process-reward variant requires a different credit construction. A whole-sequence probability ratio is not interchangeable with these per-token ratios.

The [original GRPO formulation, section 4.1](https://arxiv.org/html/2402.03300v3#S4) removes a learned value baseline, not the cost of rollouts or a separate reward model when the task uses one. The [earlier scalar implementation](03-grpo-with-verifiable-rewards.md) is a toy reduction over supplied entries; a production recipe must define the masks, token/sequence weighting, behavior distribution, and KL estimator explicitly.

## Group size and the baseline

For Bernoulli success probability $p$, the probability that a group contains both success and failure is

$$1-p^G-(1-p)^G.$$

At $p=0.5,G=4$ this is $0.875$. At $p=0.01,G=4$ it is about $0.0394$. Larger groups can reveal comparisons but cost more generation and do not fix an impossible task. Correlated samples can reduce diversity further; independence is an assumption of this calculation.

Subtracting an action-independent baseline preserves the expected policy gradient. The inclusive mean of a sampled group includes the current action's reward and is not action-independent. Without standard-deviation normalization, its expected gradient has a finite-group factor $(G-1)/G$ under independent samples. A leave-one-out baseline removes this particular self-inclusion factor for $G>1$. Dividing by sampled reward standard deviation adds further weighting and estimator effects. Avoid calling every group-normalized estimator unbiased.

An equal-reward group has zero centered policy advantages. Reference regularization or another objective can still update its tokens. Track zero-spread fraction by prompt difficulty and task, along with the denominator of discarded groups.

## Variants are objective changes

| Design choice | Motivation | What must be compared |
|---|---|---|
| Per-sequence token mean | Give each sequence equal total weight | Length bias and task quality by generated length |
| Global valid-token mean | Weight tokens equally across sequences | Longer responses receive more total weight |
| Fixed length denominator | Remove a sampled-length-dependent denominator | Gradient scale as actual lengths vary |
| Group/batch/no standard-deviation scaling | Change how easy/hard prompts contribute | Difficulty weighting, advantage variance, and stability |
| Asymmetric ratio clipping | Change exploration and update incentives | Ratio tails, entropy, drift, and success |
| Filtering all-equal groups | Spend updates on nonzero comparisons | Resampling cost and changed prompt distribution |
| Length-cap penalties or masking | Handle incomplete generations | Truncation rate, denominator, and length-conditioned success |

[DAPO](https://arxiv.org/abs/2503.14476) and [Dr. GRPO's analysis](https://arxiv.org/abs/2503.20783) motivate several such changes. The [TRL 0.23.1 GRPO reference](https://huggingface.co/docs/trl/v0.23.1/grpo_trainer) exposes choices including `loss_type` and reward scaling. A trainer's current default should not be assumed to reproduce the original paper. Record the resolved recipe, not just the algorithm name.

## Measure the complete learning system

Log rollout tokens/sec, accepted fresh tokens/sec, scorer timeouts, weight-sync time, queue age, optimizer tokens/sec, total GPU-hours, and success per compute budget. Record raw reward components, independent task success, length, truncation, entropy, reference KL, old/current KL, clip fraction, and per-group reward spread. Explain each metric's mask and denominator.

If generation throughput doubles but most data is stale, useful learning throughput can fall. If output lengths grow, reward may improve while successful answers per GPU-hour decline. If learner and actor probabilities disagree before an update, check sampling transforms, numerical execution, dropout, and model revisions before changing clipping.

## Key terms

- **Actor/learner architecture**: a system separating experience generation from gradient updates.
- **Zero-spread group**: a prompt group with identical rewards and zero centered advantages.
- **Recipe**: objective, masks, normalizations, update schedule, and generation contract taken together.

## Summary

The serving runtime, estimator, and scheduling policy form one learning system. Optimize accepted evidence and task improvement per total cost, while keeping algorithmic changes visible.

## Review questions

1. Why can global token averaging change the learned policy compared with sequence averaging?

   <details><summary>Answer</summary>It changes the relative weight assigned to long and short completions, so it changes the objective rather than merely implementation speed.</details>
2. Does removing the critic make GRPO automatically cheaper than PPO at the same quality?

   <details><summary>Answer</summary>No. More samples per prompt, scoring, synchronization, and different sample efficiency can outweigh critic-memory savings.</details>
