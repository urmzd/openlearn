<!-- ol:module L12.4 -->
# Distillation with forward and reverse KL

## Overview

| | |
|---|---|
| **Module** | `L12.4` · optional build · Python · Pass 10 · 2 to 3 h |
| **You build** | `tinyllm/post/distill.py`: forward and reverse KL objectives |
| **Contract** | [`distill.pyi`](../../../course/contracts/py/tinyllm/post/distill.pyi) |
| **Tests** | `course/tests/L12.4/` (why: hand distributions, divergence direction, and on-policy samples) |
| **Needs** | `M11.1` entropy and KL |
| **Used by** | `C2` alternative training recipe |
| **Milestone** | [MS-C2](../../../course/milestones/MS-C2.toml) |
| **Optional depth** | On-policy sequence-level distillation |

## Key Takeaways

- Forward KL penalizes teacher-supported modes the student misses.
- Reverse KL emphasizes modes the student already selects.
- Temperature changes both target entropy and gradient scale.
- On-policy distillation evaluates the teacher on student-generated prefixes.

## How to work this chapter

```bash
ol start L12.4
ol tests L12.4
ol check L12.4
```

## 1. Why now

The agent deployment can use a smaller student if it preserves useful behavior from a larger teacher. Token-level distillation gives a measurable alternative to preference-based training.

## 2. Principles

For teacher distribution `p` and student distribution `q`, forward KL is `D_KL(p||q)=Σ p_i log(p_i/q_i)`. Reverse KL is `D_KL(q||p)=Σ q_i log(q_i/p_i)`. Both are nonnegative and zero when the distributions agree. At temperature `T`, softmax uses logits divided by `T`; the common `T²` multiplier compensates for gradient scaling when training.

| Symbol | Meaning |
|---|---|
| `p` | teacher probabilities |
| `q` | student probabilities |
| `T` | distillation temperature |

## 3. Worked example by hand

For `p=[0.8,0.2]` and `q=[0.5,0.5]`, forward KL is `0.8 log(1.6)+0.2 log(0.4) ≈ 0.193`. Reverse KL is `0.5 log(0.625)+0.5 log(2.5) ≈ 0.223`. Tests fix these values and verify both directions become zero for equal distributions.

## 4. The interface

Implement stable log-softmax based KL and temperature handling. `test_hand_forward_kl` checks the worked distributions, `test_reverse_kl_modes` distinguishes divergence direction, and `test_on_policy_sampling` exercises extreme logits.

| Test | Why it exists | Expected result |
|---|---|---|
| `test_hand_forward_kl` | Pins forward KL arithmetic | Matches the worked distribution value |
| `test_reverse_kl_modes` | Makes divergence direction explicit | Reverse KL differs from forward KL on the fixture |
| `test_on_policy_sampling` | Checks teacher queries on student prefixes | Teacher logits align to sampled prefixes |

## 5. Pitfalls

| Pitfall | Caught by |
|---|---|
| Swap teacher and student in forward KL | `test_hand_forward_kl`; mutant `s01` |
| Use the student distribution as reverse-KL weights | `test_reverse_kl_modes`; mutant `s02` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M11.1` | Defines the forward and reverse KL measures. |
| Forward | `C2` | May use distillation instead of DPO or GRPO, evaluates on the same held-out suite, and records teacher identity. |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| token-level KL | on-policy sequence distillation | Transfers behavior on student-generated prefixes | Hugging Face TRL distillation recipes |
| temperature scaling | teacher/student calibration | Changes target entropy and gradient magnitude | Hinton et al., Distilling the Knowledge in a Neural Network |
