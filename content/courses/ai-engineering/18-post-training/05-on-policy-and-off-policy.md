# On-policy, off-policy, offline, and online learning

A rollout is evidence from a particular policy. Before using it in an update, identify the policy that sampled its actions, the policy being optimized, and the distribution of prompts or states. These distinctions determine whether the estimator is valid and whether old experience can be reused.

> **Learning objectives**
> - Separate behavior, target, old, and reference policies.
> - Explain why online data is not necessarily on-policy.
> - Recognize support mismatch and importance-weight variance.

## Name the distributions

| Name | Role | Changes when |
|---|---|---|
| Behavior policy $\mu$ | Actually generates recorded actions | A rollout worker receives new weights or its sampling transform changes |
| Target policy $\pi_\theta$ | Policy whose performance the learner seeks to improve | An optimizer update changes its parameters |
| Old policy $\pi_{\mathrm{old}}$ | Snapshot used to collect a PPO/GRPO rollout batch | A new rollout round begins |
| Reference policy $\pi_{\mathrm{ref}}$ | Anchor for a regularizer | The recipe explicitly refreshes it; commonly frozen after SFT |

**On-policy** methods collect fresh experience from the policy they are updating. PPO uses a fixed old-policy batch for a limited number of updates, so the target moves away from the behavior during those updates. It is conventionally an on-policy algorithm with bounded batch reuse, rather than a general replay-buffer algorithm.

**Off-policy** learning updates a target using experience from another behavior policy. Q-learning can learn a greedy target while an exploratory policy collects transitions. A distributed actor can be off-policy relative to the learner simply because its weights are stale.

**Offline** means learning from an existing fixed dataset without fresh environment interaction. **Online** means new data is collected during learning. An offline replay dataset is often off-policy; an online asynchronous learner can also be off-policy. DPO is typically an offline preference objective on fixed pairs, not a PPO importance-sampling update. Distillation's use of “on-policy” concerns whose generated prefixes supply its training states; the KL direction is an independent choice.

## Work an importance correction

For one state with actions $a,b$, let $\mu=[0.8,0.2]$, $\pi=[0.4,0.6]$, and action rewards $[1,3]$. Under $\mu$, an unweighted reward average tends to $1.4$. The target expectation is $2.2$. Importance weights $\pi/\mu=[0.5,3]$ recover it:

$$
\mathbb E_{a\sim\mu}\left[\frac{\pi(a)}{\mu(a)}r(a)\right]
=0.8(0.5)(1)+0.2(3)(3)=2.2.
$$

If $\mu(b)=0$ but $\pi(b)>0$, no recorded sample can tell us the reward of $b$. This is **support mismatch**; no finite weight can repair absent evidence. Small nonzero behavior probabilities also cause high-variance weights. Weight clipping reduces variance but introduces bias, so report clipping and the distribution of weights.

For weights $w_1,\ldots,w_n$, the diagnostic effective sample size is

$$\mathrm{ESS}=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}.$$

Weights $[1,1,1,1]$ give ESS 4; $[1,1,1,9]$ give $144/84\approx1.714$. This diagnoses weight concentration, not task quality or a proof of estimator accuracy.

## Why trajectories are harder

A trajectory ratio includes a product of action ratios. Products over long outputs can have extreme variance. Token-level PPO ratios are part of a local surrogate using fixed sampled states; they do not by themselves correct arbitrary changes in the full distribution of histories. Off-policy actor-critic algorithms use additional estimators and assumptions. [IMPALA's V-trace](https://arxiv.org/abs/1802.01561) is one example designed for actor/learner lag, not a switch that turns any stale PPO buffer into valid data.

For language-model rollouts, record the actual generation contract. Temperature, top-k, top-p, grammar constraints, repetition penalties, speculative verification, and even numerical differences can change the behavior distribution. If generation truncates an action to probability zero, an untruncated learner distribution has a support mismatch. A practical recipe may deliberately use an approximation; name it and measure its effect. Do not silently label raw unfiltered model log probabilities as the actual filtered behavior log probabilities.

## A rollout record that supports diagnosis

```text
prompt IDs and completion IDs; attention and completion masks
policy revision; tokenizer and chat-template revision
sampling transforms and seed; actual behavior token log probabilities
termination reason: EOS, length cap, tool failure, cancellation
reward components and checker revision; tool observations
collection time; learner revision at consumption
```

Hold old log probabilities fixed during PPO epochs. Recomputing the denominator from the changing learner makes every ratio one and removes the intended correction. A reference policy cannot substitute for the old behavior policy: regularization and importance correction have different jobs.

## Key terms

- **Support**: the actions or histories with nonzero probability under a distribution.
- **Policy lag**: the gap between the learner and the policy that produced a rollout.
- **Replay**: reusing stored experience, subject to the chosen algorithm's assumptions.

## Summary

Data freshness and policy agreement are separate properties. Record behavior probabilities, control lag, and use an estimator designed for the data you actually have.

## Review questions

1. A worker generates continuously but is 20 learner updates behind. Is that necessarily on-policy?

   <details><summary>Answer</summary>No. Collection is online, but actions come from a stale behavior policy. Bound lag or use a justified off-policy estimator.</details>
2. Why does a fixed seed not establish equality between rollout and training probabilities?

   <details><summary>Answer</summary>The seed controls random draws. Model revisions, tokenization, transforms, precision, and batch-dependent numerical computation can still differ.</details>

## Sources and checks

[Sutton and Barto, *Reinforcement Learning*, chapters 5–6](http://incompleteideas.net/book/the-book-2nd.html) supplies the behavior/target distinction; [PPO, section 3](https://arxiv.org/abs/1707.06347) specifies the limited reuse scheme. Recompute the finite-action expectation and ESS with [learning checks](../19-training-methods/code/learning_checks.py).
