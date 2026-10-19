# Group relative policy optimization with verifiable rewards

## Overview

| | |
|---|---|
| **Module** | `L12.3` · optional build · Python · Pass 10 |
| **Solution** | [`grpo.py`](../../../../course/solutions/python/tinyllm/post/grpo.py): `group_advantages`, `grpo_loss` |
| **Contract** | [`grpo.pyi`](../../../../course/contracts/py/tinyllm/post/grpo.pyi) |
| **Needs** | `M11.1` KL, `M07.4` uncertainty, `L8.1` sampling, `L10.5` engine API |
| **Used by** | `C2` post-trained capstone |
| **Optional depth** | Policy gradients and PPO in `content/tracks/ml/03-reinforcement-learning` |

## Key Takeaways

- Normalize rewards within each prompt's sampled group.
- Clip the probability ratio to limit each update.
- Verifiable rewards can be checked without a learned reward model.
- Record rollouts and seeds so a run can be reproduced.

## How to read this chapter

Sections 1 to 3 state the objective and work a group and a clipped step by hand. Section 4 fixes the interface. Section 5 derives each piece of the objective from what a policy-gradient update needs, walks through `grpo.py`, and explains why each line is the right one. Try to predict each step of the solution before reading it.

## 1. Why now

Some tasks have objective checks, such as exact format or arithmetic correctness. Grouped rollouts let the model learn from those checks without a human preference label for each pair.

## 2. Principles

For rewards `r_i` in a group, use `A_i = (r_i - mean(r)) / (std(r) + ε)`. Let `q_i = exp(logp_new - logp_old)`. The clipped policy objective is the negative mean of `min(q_i A_i, clip(q_i, 1-εc, 1+εc) A_i)`, plus a nonnegative KL penalty. With identical rewards the centered advantages are zero, so the policy term and its gradient are zero.

| Symbol | Meaning |
|---|---|
| `r_i` | verifiable reward for sample `i` |
| `A_i` | normalized within-group advantage |
| `q_i` | new-to-old policy probability ratio |
| `εc` | clipping range |

## 3. Worked example by hand

**Advantages.** Rewards `[0, 1, 1]` have mean `2/3` and (population) standard deviation `sqrt(2/9) ≈ 0.471`. The advantages are `(-2/3)/0.471 ≈ -1.414` and `(1/3)/0.471 ≈ 0.707` twice: one negative and two positive values, summing to zero. If all three rewards equal 1, every advantage is zero and the policy gradient vanishes.

**The clipped objective.** Two samples with `logp_old = 0`, `logp_new = [log 2, log 0.5]` (ratios 2 and 0.5), advantages `[1, -1]`, no KL, `εc = 0.2`:

| sample | ratio `q` | `q A` | `clip(q) A` | min |
|---|---|---|---|---|
| 1 | 2 | 2 | 1.2 | 1.2 |
| 2 | 0.5 | -0.5 | -0.8 | -0.8 |

The mean objective is `(1.2 - 0.8)/2 = 0.2`, so the loss is `-0.2`. Both samples are clipped: the first because it already gained too much probability, the second because the minimum picks the more pessimistic value.

## 4. The interface

`group_advantages(rewards, eps=1e-8)` takes one prompt's group of rewards (a non-empty vector) and returns its normalized advantages. `grpo_loss(logp_new, logp_old, advantages, kl, beta=0.0, clip=0.2)` takes equal-shaped arrays (one entry per sampled completion or token) and returns the scalar loss to minimize. Rollouts themselves come from the `L10.5` engine with `L8.1` sampling; the caller records each rollout's seed and policy revision so `logp_old` is reproducible.

### What a correct solution must handle

| Case | Why it exists | Expected result |
|---|---|---|
| the hand GRPO objective | Pins the clipped objective | `-0.2` for the worked pair |
| grouped rewards are centered | Scopes normalization to one prompt | Group advantages have zero mean, signs as in section 3 |
| zero advantage, zero gradient | Checks equal-reward boundary | The loss does not change with `logp_new` |

## 5. The solution, step by step

### From first principles

1. **A policy gradient needs a baseline.** The REINFORCE gradient `E[r ∇ log π(y)]` is unbiased for any baseline subtracted from `r`, and a good baseline cuts its variance. PPO learns a value function for this; GRPO samples several completions for the same prompt and uses their mean reward, which needs no extra network.
2. **The baseline must be per prompt.** Prompts differ in difficulty. Normalizing across prompts would make every completion of an easy prompt look good and every completion of a hard prompt look bad, regardless of which completion was better. So advantages are computed group by group.
3. **Dividing by the standard deviation makes the step size reward-scale free.** A group with rewards in `{0, 1}` and one with rewards in `{0, 100}` then produce updates of the same size. The `ε` keeps an all-equal group (std 0) at advantage 0 instead of `0/0`.
4. **Samples were drawn from the old policy, so the update uses an importance ratio.** `q = π_new/π_old = exp(logp_new - logp_old)` reweights each sample to the current policy.
5. **Large ratios make the estimate unreliable, so the objective is clipped.** Taking the minimum of the raw and the clipped term gives a pessimistic bound: the objective never rewards pushing `q` beyond `1 ± εc` in the direction the advantage favours, so the gradient there is zero, while moves in the harmful direction are never hidden by the clip.
6. **The KL term keeps the policy near the reference.** It is a penalty added to the loss, weighted by `β`. A KL estimate is nonnegative in expectation, so negative per-sample estimates are floored at zero.

### The code, annotated

```python
def group_advantages(rewards, eps=1e-8):
    r = np.asarray(rewards, dtype=np.float64)
    if r.ndim != 1 or r.size == 0:
        raise ValueError("rewards must be a non-empty vector")   # one prompt's group, never a batch of groups
    # center on the group mean (the baseline), scale by the group's spread
    return (r - r.mean()) / (r.std() + eps)
```

The input check is the guard against pitfall 1: a 2-D array of several prompts' rewards is rejected, so normalization cannot silently span prompts.

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

With all advantages zero, both terms inside the minimum are zero for every ratio, so the loss is constant in `logp_new` and its gradient is zero, which is the equal-reward boundary.

Full solution: [`grpo.py`](../../../../course/solutions/python/tinyllm/post/grpo.py)

### Why this solution

- **Cost.** Both functions are linear in the number of samples. The real cost of GRPO is the rollouts: `G` completions per prompt, each a full generation.
- **Group baseline versus a value network.** A learned critic is a second model of the same size, trained alongside, with its own bias. The group mean is unbiased for that prompt and free, at the price of sampling several completions per prompt (and getting no signal from a group whose rewards are all equal).
- **Clipping versus a KL trust region alone.** A KL constraint needs a second-order step or a tuned penalty. The clipped minimum is one line and caps the per-sample incentive directly.
- **Verifiable rewards versus a reward model.** A checker (exact answer, valid format) cannot be gradient-hacked the way a learned reward can, but it only exists for tasks with checkable answers.

### Explain it back

**Why is the second sample in section 3 counted at `-0.8` and not `-0.5`?**

<details><summary>Answer</summary>

The minimum takes the pessimistic value. With a negative advantage the policy has already moved away from that sample (ratio 0.5), and the clip at `1 - εc = 0.8` gives `-0.8 < -0.5`. Its gradient with respect to the ratio is then zero, so the update does not push further in that direction.

</details>

**What goes wrong if rewards from several prompts are normalized together?**

<details><summary>Answer</summary>

The baseline becomes the batch mean, so prompt difficulty leaks into the advantage: all completions of easy prompts get positive advantages and all completions of hard prompts negative ones. The update then teaches "answer easy prompts", not "give the better answer to this prompt".

</details>

**Why does a group with all rewards equal produce no update, and is that a bug?**

<details><summary>Answer</summary>

Every advantage is `0/(0 + ε) = 0`, so the objective is zero for every policy. It is correct: the group carries no information about which completion is better. It is also why GRPO needs prompts of the right difficulty, where some but not all completions succeed.

</details>

**Why clip `new - old` to `[-60, 60]` before `exp`?**

<details><summary>Answer</summary>

A large log-ratio would overflow `exp` to `inf`, and `inf * 0` advantage is NaN. At ±60 the ratio is already far outside `1 ± εc`, so the clipped term decides the value anyway.

</details>

## 6. Pitfalls

| Pitfall | How the solution avoids it |
|---|---|
| Normalize rewards across unrelated prompts | `group_advantages` accepts one prompt's 1-D group only |
| Omit clipping from the policy objective | `grpo_loss` takes the minimum with the clipped ratio term |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M11.1` | Supplies the KL penalty. |
| Back | `M07.4` | Supports uncertainty-aware evaluation of reward outcomes. |
| Back | `L8.1` | Samples rollout tokens. |
| Back | `L10.5` | Provides the serving API and chat template. |
| Forward | `C2` | Runs verifiable reward training after SFT and DPO; reports reward, KL, clip fraction, and rollout revision. |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| grouped rollout objective | asynchronous rollout pools | More throughput and explicit rollout revisions | DeepSeekMath GRPO paper |
| verifiable reward | calibrated reward and tool execution | Validation against reward hacking and distribution shift | `content/tracks/ml/03-reinforcement-learning` |
