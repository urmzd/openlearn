<!-- ol:module L12.2 -->
# Direct preference optimization and IPO

## Overview

| | |
|---|---|
| **Module** | `L12.2` · optional build · Python · Pass 10 · 2 to 3 h |
| **You build** | `tinyllm/post/dpo.py`: DPO and IPO objectives |
| **Contract** | [`dpo.pyi`](../../../course/contracts/py/tinyllm/post/dpo.pyi) |
| **Tests** | `course/tests/L12.2/` (why: closed-form preference odds, reference equality, and IPO scaling) |
| **Needs** | `S-M10b` KL-regularized objective, `M11.1` KL |
| **Used by** | `C2` post-trained capstone |
| **Milestone** | [MS-C2](../../../course/milestones/MS-C2.toml) |
| **Optional depth** | IPO and odds-ratio preference optimization |

## Key Takeaways

- DPO optimizes preference odds relative to a frozen reference policy.
- The reference term prevents unbounded movement away from the SFT policy.
- IPO changes the loss shape and is not just a different learning rate.

## How to work this chapter

```bash
ol start L12.2
ol tests L12.2
ol check L12.2
```

## 1. Why now

SFT teaches response form, but does not encode which of two valid responses a user prefers. Paired responses provide a direct training signal without fitting a separate reward model.

## 2. Principles

For chosen and rejected completions, define `Δπ = log π(y+|x) - log π(y-|x)` and `Δref` similarly. DPO loss is `-log σ(β(Δπ-Δref))`. The reference policy is fixed. At policy equal to reference, the margin is zero and the per-pair loss is `log 2`.

| Symbol | Meaning |
|---|---|
| `π` | trainable policy |
| `πref` | frozen reference policy |
| `β` | preference strength / KL scale |
| `σ` | logistic sigmoid |

## 3. Worked example by hand

With `Δπ = 1.2`, `Δref = 0.2`, and `β = 0.1`, the scaled margin is 0.1, so the loss is `-log σ(0.1) ≈ 0.6444`. If both policies are equal, both deltas match and the loss is `log 2 ≈ 0.6931`.

## 4. The interface

Implement DPO and IPO over completion log probabilities. `test_hand_dpo_loss` checks the worked value, `test_equal_policy_reference_is_log_two` checks the equal-policy baseline, and `test_ipo_variant` checks IPO scaling. Never update the reference weights.

| Test | Why it exists | Expected result |
|---|---|---|
| `test_hand_dpo_loss` | Pins the DPO objective to the worked calculation | Loss near `0.6444` |
| `test_equal_policy_reference_is_log_two` | Verifies reference correction | Equal policies produce `log(2)` |
| `test_ipo_variant` | Checks the alternate IPO objective | Matches the hand-computed fixture |

## 5. Pitfalls

| Pitfall | Caught by |
|---|---|
| Reversing chosen and rejected | `test_hand_dpo_loss`; mutant `s01` |
| Omitting reference log probabilities | `test_hand_dpo_loss`; mutant `s02` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M11.1` | Defines the KL term that regularizes policy movement. |
| Back | `S-M10b` | Derives how KL regularization leads to reference-adjusted preference odds. |
| Forward | `C2` | Applies DPO to preference pairs after `L12.1` SFT. |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| paired preference objective | preference optimization pipelines | Annotator disagreement and pair provenance | Rafailov et al., Direct Preference Optimization |
| held-out preference check | online preference evaluation | Position controls and outcome measures beyond training loss | TRL preference trainer |
