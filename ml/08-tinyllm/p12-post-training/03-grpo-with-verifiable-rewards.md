<!-- ol:module L12.3 -->
# Group relative policy optimization with verifiable rewards

## Overview

| | |
|---|---|
| **Module** | `L12.3` · optional build · Python · Pass 10 · 3 to 4 h |
| **You build** | `tinyllm/post/grpo.py`: grouped advantages, clipped objective, rollout client |
| **Contract** | [`grpo.pyi`](../../../course/contracts/py/tinyllm/post/grpo.pyi) |
| **Tests** | `course/tests/L12.3/` (why: hand objective, zero-advantage gradient, and toy reward improvement) |
| **Needs** | `M11.1` KL, `M07.4` uncertainty, `L8.1` sampling, `L10.5` engine API |
| **Used by** | `C2` post-trained capstone |
| **Milestone** | [MS-C2](../../../course/milestones/MS-C2.toml) |
| **Optional depth** | Policy gradients and PPO in `ml/03-reinforcement-learning` |

## Key Takeaways

- Normalize rewards within each prompt's sampled group.
- Clip the probability ratio to limit each update.
- Verifiable rewards can be checked without a learned reward model.
- Record rollouts and seeds so a run can be reproduced.

## How to work this chapter

```bash
ol start L12.3
ol tests L12.3
ol check L12.3
```

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

Rewards `[0, 1, 1]` have mean `2/3`; their advantages have one negative and two positive values, summing to zero. If all three rewards equal 1, every advantage is zero and the policy gradient vanishes. The tests use this exact group.

## 4. The interface

Implement group normalization and clipped loss. `test_hand_grpo_objective` checks the clipped arithmetic, `test_grouped_rewards_are_centered` checks normalization, and `test_zero_advantage_has_zero_policy_gradient` checks the zero-variance boundary.

| Test | Why it exists | Expected result |
|---|---|---|
| `test_hand_grpo_objective` | Pins the clipped objective | Matches the hand-computed scalar |
| `test_grouped_rewards_are_centered` | Scopes normalization to one prompt | Group advantages have zero mean |
| `test_zero_advantage_has_zero_policy_gradient` | Checks equal-reward boundary | Policy gradient is zero |

## 5. Pitfalls

| Pitfall | Caught by |
|---|---|
| Normalize rewards across unrelated prompts | `test_grouped_rewards_are_centered`; mutant `s01` |
| Omit clipping from the policy objective | `test_hand_grpo_objective`; mutant `s02` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M11.1` | Supplies the KL penalty. |
| Back | `M07.4` | Supports uncertainty-aware evaluation of reward outcomes. |
| Back | `L8.1` | Samples rollout tokens. |
| Back | `L10.5` | Provides the serving API and chat template. |
| Forward | `C2` | Runs verifiable reward training after SFT and DPO; reports reward, KL, clip fraction, and rollout revision. |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| grouped rollout objective | asynchronous rollout pools | More throughput and explicit rollout revisions | DeepSeekMath GRPO paper |
| verifiable reward | calibrated reward and tool execution | Validation against reward hacking and distribution shift | `ml/03-reinforcement-learning` |
