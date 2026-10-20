# Group relative policy optimization with verifiable rewards

Some tasks have objective checks: an answer is in the right format or not, an arithmetic result is correct or not. For such tasks you do not need a human to label which of two responses is better, and you do not need a learned reward model; a checker can score every sample the model produces. Group relative policy optimization (GRPO) turns those scores into a policy-gradient update by sampling several completions of the same prompt and comparing them with each other.

This section derives each piece of the GRPO objective from what a policy-gradient update needs (a baseline, a scale, an importance ratio, a clip, and a KL penalty), works a group and a clipped step by hand, and implements the two functions that carry the math. It follows [Direct preference optimization and IPO](02-direct-preference-optimization.md), which learned from fixed pairs; GRPO learns from rollouts the model generates itself. Rollouts come from the serving engine of [tl-serve: OpenAI-compatible HTTP + SSE on tokio/hyper](../23-model-serving/06-openai-server-on-tokio.md) with the samplers of [Sampling and logit processors](../21-inference/01-sampling-and-logit-processors.md). The KL penalty is the one from [Entropy, cross-entropy, KL, JS, and the k3 estimator](../../mathematics/17-information-theory/01-entropy-cross-entropy-and-kl.md), and evaluating reward outcomes with honest uncertainty uses [LLN, CLT, confidence intervals, bootstrap](../../mathematics/13-probability-statistics/05-lln-clt-confidence-intervals-bootstrap.md), both in *Mathematics*. The solution is [`grpo.py`](../../../../course/solutions/python/tinyllm/post/grpo.py) (`group_advantages`, `grpo_loss`), and the contract is [`grpo.pyi`](../../../../course/contracts/py/tinyllm/post/grpo.pyi).

> **Learning objectives**
> - Compute within-group normalized advantages and explain why the baseline must be per prompt.
> - Evaluate the clipped surrogate objective by hand and explain why it takes the minimum.
> - Implement the GRPO loss stably, with a floored KL penalty.
> - Explain why a group with equal rewards produces no update, and what that implies for prompt difficulty.

## The objective

Write $r_i$ for the verifiable reward of sample $i$ in one prompt's group, $A_i$ for its normalized within-group advantage, $q_i$ for the new-to-old policy probability ratio, and $\varepsilon_c$ for the clipping range. The advantage is

$$A_i = \frac{r_i - \mathrm{mean}(r)}{\mathrm{std}(r) + \varepsilon},$$

the ratio is $q_i = \exp(\log p_{\text{new}} - \log p_{\text{old}})$, and the loss is the negative mean of the clipped surrogate plus a nonnegative KL penalty weighted by $\beta$:

$$\mathcal{L} = -\frac1n \sum_i \min\big(q_i A_i,\ \mathrm{clip}(q_i, 1-\varepsilon_c, 1+\varepsilon_c)\, A_i\big) + \beta \cdot \overline{\max(\mathrm{KL}, 0)}.$$

With identical rewards the centered advantages are zero, so the policy term and its gradient are zero.

## Why each piece is there

**A policy gradient needs a baseline.** The REINFORCE gradient $\mathbb{E}[r\, \nabla \log \pi(y)]$ is unbiased for any baseline subtracted from $r$, and a good baseline cuts its variance. PPO learns a value function for this; GRPO samples several completions for the same prompt and uses their mean reward, which needs no extra network.

**The baseline must be per prompt.** Prompts differ in difficulty. Normalizing across prompts would make every completion of an easy prompt look good and every completion of a hard prompt look bad, regardless of which completion was better. So advantages are computed group by group.

**Dividing by the standard deviation makes the step size reward-scale free.** A group with rewards in $\{0, 1\}$ and one with rewards in $\{0, 100\}$ then produce updates of the same size. The $\varepsilon$ keeps an all-equal group (std 0) at advantage 0 instead of $0/0$.

**Samples were drawn from the old policy, so the update uses an importance ratio.** $q = \pi_{\text{new}}/\pi_{\text{old}} = \exp(\log p_{\text{new}} - \log p_{\text{old}})$ reweights each sample to the current policy.

**Large ratios make the estimate unreliable, so the objective is clipped.** Taking the minimum of the raw and the clipped term gives a pessimistic bound: the objective never rewards pushing $q$ beyond $1 \pm \varepsilon_c$ in the direction the advantage favours, so the gradient there is zero, while moves in the harmful direction are never hidden by the clip.

**The KL term keeps the policy near the reference.** It is a penalty added to the loss, weighted by $\beta$. A KL estimate is nonnegative in expectation, so negative per-sample estimates are floored at zero.

## Working it by hand

**Advantages.** Rewards $[0, 1, 1]$ have mean $2/3$ and (population) standard deviation $\sqrt{2/9} \approx 0.471$. The advantages are $(-2/3)/0.471 \approx -1.414$ and $(1/3)/0.471 \approx 0.707$ twice: one negative and two positive values, summing to zero. If all three rewards equal 1, every advantage is zero and the policy gradient vanishes.

**The clipped objective.** Take two samples with $\log p_{\text{old}} = 0$ and $\log p_{\text{new}} = [\log 2, \log 0.5]$ (ratios 2 and 0.5), advantages $[1, -1]$, no KL, and $\varepsilon_c = 0.2$:

| sample | ratio $q$ | $qA$ | $\mathrm{clip}(q)A$ | min |
|---|---|---|---|---|
| 1 | 2 | 2 | 1.2 | 1.2 |
| 2 | 0.5 | -0.5 | -0.8 | -0.8 |

The mean objective is $(1.2 - 0.8)/2 = 0.2$, so the loss is $-0.2$. Both samples are clipped: the first because it already gained too much probability, the second because the minimum picks the more pessimistic value.

## Designing the API

`group_advantages(rewards, eps=1e-8)` takes one prompt's group of rewards (a non-empty vector) and returns its normalized advantages. `grpo_loss(logp_new, logp_old, advantages, kl, beta=0.0, clip=0.2)` takes equal-shaped arrays (one entry per sampled completion or token) and returns the scalar loss to minimize. The rollouts themselves come from the serving engine with its sampler; the caller records each rollout's seed and policy revision so that `logp_old` is reproducible and a run can be replayed.

## Building it step by step

Advantages are a few lines, but the input check matters as much as the arithmetic:

```python
def group_advantages(rewards, eps=1e-8):
    r = np.asarray(rewards, dtype=np.float64)
    if r.ndim != 1 or r.size == 0:
        raise ValueError("rewards must be a non-empty vector")   # one prompt's group, never a batch of groups
    # center on the group mean (the baseline), scale by the group's spread
    return (r - r.mean()) / (r.std() + eps)
```

Rejecting a 2-D array of several prompts' rewards guarantees that normalization cannot silently span prompts.

The loss validates shapes and hyperparameters, computes the ratio with an overflow guard, and takes the pessimistic surrogate:

```python
def grpo_loss(logp_new, logp_old, advantages, kl, beta=0.0, clip=0.2):
    new, old, adv, k = (np.asarray(x, dtype=np.float64) for x in (logp_new, logp_old, advantages, kl))
    if not (new.shape == old.shape == adv.shape == k.shape):
        raise ValueError("GRPO arrays must have equal shapes")
    if beta < 0 or not 0 < clip < 1:
        raise ValueError("beta must be nonnegative and clip in (0, 1)")
    # the importance ratio; the log difference is clipped so exp cannot overflow to inf
    ratio = np.exp(np.clip(new - old, -60.0, 60.0))
    # PPO's pessimistic surrogate: the smaller of the raw and the clipped term
    objective = np.minimum(ratio * adv, np.clip(ratio, 1.0 - clip, 1.0 + clip) * adv)
    # minimize: negative objective plus the floored KL penalty
    return float(-objective.mean() + beta * np.maximum(k, 0.0).mean())
```

With all advantages zero, both terms inside the minimum are zero for every ratio, so the loss is constant in `logp_new` and its gradient is zero; that is the equal-reward boundary. The full solution is [`grpo.py`](../../../../course/solutions/python/tinyllm/post/grpo.py).

### Why this design

Both functions are linear in the number of samples. The real cost of GRPO is the rollouts: $G$ completions per prompt, each a full generation.

The group baseline replaces a value network. A learned critic is a second model of the same size, trained alongside, with its own bias. The group mean is unbiased for that prompt and free, at the price of sampling several completions per prompt and getting no signal from a group whose rewards are all equal. The clipped minimum replaces a pure KL trust region, which would need a second-order step or a tuned penalty; the clip is one line and caps the per-sample incentive directly. And verifiable rewards replace a reward model: a checker (exact answer, valid format) cannot be gradient-hacked the way a learned reward can, but it only exists for tasks with checkable answers.

## What can go wrong

Three checks pin the implementation: the worked clipped objective must give $-0.2$; group advantages must have zero mean with the signs of the worked group; and with zero advantages the loss must not change with `logp_new`.

> **Warning:** Normalizing rewards across unrelated prompts leaks prompt difficulty into the advantage. `group_advantages` accepts one prompt's 1-D group only.

The other classic mistake is omitting clipping from the policy objective, which lets a single sample with a large ratio dominate the update; `grpo_loss` takes the minimum with the clipped ratio term. Finally, record rollouts and their seeds: without the policy revision that produced each sample, `logp_old` cannot be recomputed and a run cannot be reproduced.

## Where this leads

The [post-trained capstone](../capstones/02-post-trained.md) runs verifiable-reward training after SFT and DPO and reports reward, KL, clip fraction, and rollout revision. The next section, [Distillation with forward and reverse KL](04-distillation.md), offers an alternative recipe that learns from a teacher's distribution instead of from rewards. Production systems add asynchronous rollout pools, for more throughput with explicit rollout revisions, and calibrated rewards with tool execution, validated against reward hacking and distribution shift. Policy gradients and PPO are covered in depth in the [Reinforcement Learning](../04-reinforcement-learning/README.md) chapter of *AI Engineering*.

### Further reading

- Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (2024), which introduces GRPO.
- Schulman et al., *Proximal Policy Optimization Algorithms* (2017).

## Key terms

- **verifiable reward**: a reward computed by a deterministic checker rather than a learned model.
- **group**: the set of completions sampled for one prompt, whose rewards share a baseline.
- **advantage**: a sample's reward minus the group mean, divided by the group's standard deviation.
- **importance ratio**: $\pi_{\text{new}}/\pi_{\text{old}}$ for a sample drawn from the old policy.
- **clipped surrogate**: the minimum of $qA$ and $\mathrm{clip}(q, 1 \pm \varepsilon_c)A$, which removes the incentive to move a sample's probability beyond the clip range.

## Summary

- Normalize rewards within each prompt's sampled group; the group mean is a free, per-prompt baseline.
- Dividing by the group's standard deviation makes the step size independent of the reward scale.
- Clip the probability ratio and take the pessimistic minimum to limit each update.
- Verifiable rewards can be checked without a learned reward model, but groups with all-equal rewards carry no signal.
- Record rollouts and seeds so a run can be reproduced.

## Review questions

1. Why is the second sample in the worked example counted at $-0.8$ and not $-0.5$?

   <details><summary>Answer</summary>

   The minimum takes the pessimistic value. With a negative advantage the policy has already moved away from that sample (ratio 0.5), and the clip at $1 - \varepsilon_c = 0.8$ gives $-0.8 < -0.5$. Its gradient with respect to the ratio is then zero, so the update does not push further in that direction.

   </details>

2. What goes wrong if rewards from several prompts are normalized together?

   <details><summary>Answer</summary>

   The baseline becomes the batch mean, so prompt difficulty leaks into the advantage: all completions of easy prompts get positive advantages and all completions of hard prompts negative ones. The update then teaches "answer easy prompts", not "give the better answer to this prompt".

   </details>

3. Why does a group with all rewards equal produce no update, and is that a bug?

   <details><summary>Answer</summary>

   Every advantage is $0/(0 + \varepsilon) = 0$, so the objective is zero for every policy. It is correct: the group carries no information about which completion is better. It is also why GRPO needs prompts of the right difficulty, where some but not all completions succeed.

   </details>

4. Why clip `new - old` to $[-60, 60]$ before `exp`?

   <details><summary>Answer</summary>

   A large log-ratio would overflow `exp` to `inf`, and `inf * 0` advantage is NaN. At $\pm 60$ the ratio is already far outside $1 \pm \varepsilon_c$, so the clipped term decides the value anyway.

   </details>
