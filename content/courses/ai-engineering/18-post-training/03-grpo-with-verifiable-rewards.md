# Group relative policy optimization with verifiable rewards

Some tasks have objective checks: an answer is in the right format or not, an arithmetic result is correct or not. For such tasks you do not need a human to label which of two responses is better, and you do not need a learned reward model; a checker can score every sample the model produces. Group relative policy optimization (GRPO) turns those scores into a policy-gradient update by sampling several completions of the same prompt and comparing them with each other.

This section derives each piece of the GRPO objective from what a policy-gradient update needs (a baseline, a scale, an importance ratio, a clip, and a KL penalty), works a group and a clipped step by hand, and implements the two functions that carry the math. It follows [Direct preference optimization and IPO](02-direct-preference-optimization.md), which learned from fixed pairs; GRPO learns from rollouts the model generates itself. Rollouts come from the serving engine of [tl-serve: OpenAI-compatible HTTP + SSE on tokio/hyper](../23-model-serving/06-openai-server-on-tokio.md) with the samplers of [Sampling and logit processors](../21-inference/01-sampling-and-logit-processors.md). The KL penalty is the one from [Entropy, cross-entropy, KL, JS, and the k3 estimator](../../mathematics/17-information-theory/01-entropy-cross-entropy-and-kl.md), and evaluating reward outcomes with honest uncertainty uses [LLN, CLT, confidence intervals, bootstrap](../../mathematics/13-probability-statistics/05-lln-clt-confidence-intervals-bootstrap.md), both in *Mathematics*. The solution is [`grpo.py`](../../../../course/solutions/python/tinyllm/post/grpo.py) (`group_advantages`, `grpo_loss`), and the contract is [`grpo.pyi`](../../../../course/contracts/py/tinyllm/post/grpo.pyi).

> **Learning objectives**
> - Compute within-group normalized advantages and explain why the baseline must be per prompt.
> - Evaluate the clipped surrogate objective by hand and explain why it takes the minimum.
> - Implement the GRPO loss stably, with a floored KL penalty.
> - Explain why equal rewards produce zero centered policy advantages, and distinguish that from a KL-driven update.

## The objective

Write $r_i$ for the verifiable reward of sample $i$ in one prompt's group, $A_i$ for its normalized within-group advantage, $q_i$ for the new-to-old policy probability ratio, and $\varepsilon_c$ for the clipping range. The advantage is

$$A_i = \frac{r_i - \mathrm{mean}(r)}{\mathrm{std}(r) + \varepsilon},$$

the ratio is $q_i = \exp(\log p_{\text{new}} - \log p_{\text{old}})$, and the loss is the negative mean of the clipped surrogate plus a nonnegative KL penalty weighted by $\beta$:

$$\mathcal{L} = -\frac1n \sum_i \min\big(q_i A_i,\ \mathrm{clip}(q_i, 1-\varepsilon_c, 1+\varepsilon_c)\, A_i\big) + \beta \cdot \overline{\max(\mathrm{KL}, 0)}.$$

With identical rewards the centered advantages are zero, so the policy term and its gradient are zero.

## Why each piece is there

**A policy gradient can use a baseline to reduce variance.** Subtracting an action-independent baseline preserves the expected REINFORCE gradient. GRPO uses a sampled group mean; because that mean includes the current sample, the resulting estimator is not covered by the action-independent guarantee. Group standard-deviation normalization adds further weighting effects. See [the finite-group analysis](08-rollout-systems-and-grpo-variants.md#group-size-and-the-baseline). PPO commonly learns a value baseline, while GRPO avoids that learned critic.

**The baseline must be per prompt.** Prompts differ in difficulty. Normalizing across prompts would make every completion of an easy prompt look good and every completion of a hard prompt look bad, regardless of which completion was better. So advantages are computed group by group.

**Dividing by the standard deviation normalizes advantage scale within a group.** Scaling rewards by a positive constant gives approximately the same advantages when $\varepsilon$ is negligible. This does not guarantee an invariant optimizer step or balance against the KL term. The $\varepsilon$ keeps an all-equal group (std 0) at advantage 0 instead of $0/0$.

**Samples were drawn from the old policy, so the update uses an importance ratio.** $q = \pi_{\text{new}}/\pi_{\text{old}} = \exp(\log p_{\text{new}} - \log p_{\text{old}})$ reweights each sample to the current policy.

**Large ratios make the estimate unreliable, so the objective is clipped.** Taking the minimum of the raw and the clipped term gives a pessimistic bound: the objective never rewards pushing $q$ beyond $1 \pm \varepsilon_c$ in the direction the advantage favours, so the gradient there is zero, while moves in the harmful direction are never hidden by the clip.

**The KL term penalizes drift from the reference.** The toy API floors its supplied penalty entries at zero, but this is not a generally unbiased KL estimator. A signed log-ratio sample can be negative even when its expectation is nonnegative; flooring it changes that expectation. The original GRPO recipe uses the pointwise nonnegative estimator $k_3=e^{-d}+d-1$, where $d=\log\pi_\theta-\log\pi_{\mathrm{ref}}$. Its expectation equals forward KL when actions are sampled from the current policy with matching support; old-policy samples require the recipe's stated correction or approximation. Keep this distinction when replacing the toy penalty with a real training objective.

## Working it by hand

**Advantages.** Rewards $[0, 1, 1]$ have mean $2/3$ and (population) standard deviation $\sqrt{2/9} \approx 0.471$. The advantages are $(-2/3)/0.471 \approx -1.414$ and $(1/3)/0.471 \approx 0.707$ twice: one negative and two positive values, summing to zero. If all three rewards equal 1, every advantage is zero and the policy gradient vanishes.

**The clipped objective.** Take two samples with $\log p_{\text{old}} = \log 0.25$ and $\log p_{\text{new}} = [\log 0.5, \log 0.125]$ (ratios 2 and 0.5), advantages $[1, -1]$, no KL, and $\varepsilon_c = 0.2$:

| sample | ratio $q$ | $qA$ | $\mathrm{clip}(q)A$ | min |
|---|---|---|---|---|
| 1 | 2 | 2 | 1.2 | 1.2 |
| 2 | 0.5 | -0.5 | -0.8 | -0.8 |

The mean objective is $(1.2 - 0.8)/2 = 0.2$, so the loss is $-0.2$. Both samples are clipped: the first because it already gained too much probability, the second because the minimum picks the more pessimistic value.

## Designing the API

`group_advantages(rewards, eps=1e-8)` takes one prompt's group of rewards (a non-empty vector) and returns its normalized advantages. `grpo_loss(logp_new, logp_old, advantages, kl, beta=0.0, clip=0.2)` takes equal-shaped arrays (one entry per sampled completion or token) and returns the scalar loss to minimize. The rollouts come from the serving engine. Record actual behavior log probabilities, policy revision, seed, tokenizer/template, sampling transforms, and numerical execution settings. A seed and weight revision alone do not establish replay or equality with the learner's scoring path.

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

With all advantages zero, the policy surrogate is zero for every ratio. In this scalar helper, holding the supplied `kl` array fixed makes the result constant in `logp_new`; a real differentiable KL penalty can still produce an update. This helper returns a NumPy scalar, so it checks loss arithmetic rather than implementing an autograd training loop. The full solution is [`grpo.py`](../../../../course/solutions/python/tinyllm/post/grpo.py).

### Why this design

Both functions are linear in the number of samples. The real cost of GRPO is the rollouts: $G$ completions per prompt, each a full generation.

The group baseline avoids learning a value network; critics can otherwise use a separate backbone or a shared value head. The mean estimates expected prompt reward, but including each sample in its own baseline and normalizing by sampled spread affects the policy-gradient estimator. Sampling groups costs additional generation. The clipped minimum changes the sampled surrogate incentive; it does not enforce a hard KL trust region. A checker can replace a learned reward model on checkable tasks, but parser bugs, incomplete tests, and format-only rewards can still be exploited. [Reward design](07-rewards-and-preference-models.md) shows how to test that boundary.

## What can go wrong

Three checks pin the implementation: the worked clipped objective must give $-0.2$; group advantages must have zero mean with the signs of the worked group; and with zero advantages the loss must not change with `logp_new`.

> **Warning:** Normalizing rewards across unrelated prompts leaks prompt difficulty into the advantage. `group_advantages` accepts one prompt's 1-D group only.

The other classic mistake is omitting clipping from the policy objective, which lets a single sample with a large ratio dominate the update; `grpo_loss` takes the minimum with the clipped ratio term. Finally, keep immutable behavior probabilities with each rollout. Policy revision, sampling transforms, and numerical configuration are needed to interpret or attempt to reproduce them.

## Where this leads

The [post-trained capstone](../capstones/02-post-trained.md) runs verifiable-reward training after SFT and DPO and reports reward, KL, clip fraction, and rollout revision. The next section, [Distillation with forward and reverse KL](04-distillation.md), offers an alternative recipe that learns from a teacher's distribution instead of from rewards. Production systems add asynchronous rollout pools, for more throughput with explicit rollout revisions, and calibrated rewards with tool execution, validated against reward hacking and distribution shift. The worked [PPO and GAE section](06-ppo-with-gae.md) derives the critic-based alternative; [rollout systems and GRPO variants](08-rollout-systems-and-grpo-variants.md) adds token masks, length weighting, lag, and production diagnostics.

### Further reading

- Shao et al., [*DeepSeekMath*, section 4.1](https://arxiv.org/html/2402.03300v3#S4), which introduces GRPO.
- Schulman et al., [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347) (2017).

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
- Verifiable rewards can be checked without a learned reward model. Equal-reward groups carry no centered policy signal; KL or other objectives can still update them.
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

3. Why does a group with all rewards equal produce zero policy advantages, and can the model still update?

   <details><summary>Answer</summary>

   Every advantage is $0/(0 + \varepsilon) = 0$, so the clipped policy term is zero. A reference-KL penalty or other loss can still update the model. The group itself supplies no relative outcome signal, motivating prompts with a useful range of sampled outcomes.

   </details>

4. Why clip `new - old` to $[-60, 60]$ before `exp`?

   <details><summary>Answer</summary>

   A large log-ratio can overflow `exp`, and `inf * 0` is NaN. The helper clamps the log difference as a numerical guard. This is an additional approximation: the raw term still controls harmful moves, such as a negative advantage with a very large ratio, so clamping can change the loss and gradient. A real learner should diagnose extreme ratios rather than silently treating this guard as exact PPO clipping.

   </details>
