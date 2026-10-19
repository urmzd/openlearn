# Direct preference optimization and IPO

## Overview

| | |
|---|---|
| **Module** | `L12.2` · optional build · Python · Pass 10 |
| **Solution** | [`dpo.py`](../../../../course/solutions/python/tinyllm/post/dpo.py): `dpo_loss`, `ipo_loss` |
| **Contract** | [`dpo.pyi`](../../../../course/contracts/py/tinyllm/post/dpo.pyi) |
| **Needs** | `S-M10b` KL-regularized objective, `M11.1` KL |
| **Used by** | `C2` post-trained capstone |
| **Optional depth** | IPO and odds-ratio preference optimization |

## Key Takeaways

- DPO optimizes preference odds relative to a frozen reference policy.
- The reference term prevents unbounded movement away from the SFT policy.
- IPO changes the loss shape and is not just a different learning rate.

## How to read this chapter

Sections 1 to 3 state the objective and compute it by hand. Section 4 fixes the interface. Section 5 derives DPO from the KL-regularized reward objective, walks through `dpo.py`, and explains why each line is written that way. Try to predict each step of the solution before reading it.

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

With `Δπ = 1.2`, `Δref = 0.2`, and `β = 0.1`, the scaled margin is 0.1, so the loss is `-log σ(0.1) = log(1 + e^-0.1) ≈ 0.6444`. If both policies are equal, both deltas match and the loss is `log 2 ≈ 0.6931`.

IPO on the same pair uses the unscaled margin `z = Δπ - Δref = 1.0` and the target `1/(2β) = 5`, so the loss is `(1 - 5)^2 = 16`.

## 4. The interface

`dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.1)` and `ipo_loss(...)` with the same arguments take completion log probabilities (summed over each completion's tokens), one entry per pair, and return the mean loss over pairs as a `float`. The reference log probabilities are inputs, computed once by the frozen reference model; the reference weights are never updated.

### What a correct solution must handle

| Case | Why it exists | Expected result |
|---|---|---|
| the hand DPO loss | Pins the DPO objective to the worked calculation | Loss near `0.6444` |
| policy equal to reference | Verifies reference correction | Equal policies produce `log(2)` |
| the IPO variant | Checks the alternate IPO objective | `16.0` for the worked pair |

## 5. The solution, step by step

### From first principles

1. **Start from the objective DPO replaces.** RLHF maximizes `E[r(x, y)] - β KL(π || πref)`. Its optimum has a closed form (the `S-M10b` derivation): `π*(y|x) = πref(y|x) exp(r(x, y)/β) / Z(x)`.
2. **Invert it.** Solving for the reward gives `r(x, y) = β log(π*(y|x)/πref(y|x)) + β log Z(x)`. The reward is a log-ratio against the reference, plus a term that depends only on the prompt.
3. **Use a pairwise preference model.** Bradley-Terry says `P(y+ beats y-) = σ(r(x, y+) - r(x, y-))`. In the difference, `β log Z(x)` cancels, which is the whole trick: no partition function, no reward model.
4. **What is left is the margin.** `r+ - r- = β((log π(y+) - log π(y-)) - (log πref(y+) - log πref(y-))) = β(Δπ - Δref)`. Maximizing the log-likelihood of the observed preferences is minimizing `-log σ(β(Δπ - Δref))`.
5. **Compute it stably.** `-log σ(z) = log(1 + e^-z)`, which overflows for large negative `z` if written literally. `logaddexp(0, -z)` computes the same value without overflow.
6. **IPO keeps the margin but changes the shape.** DPO's loss keeps decreasing as the margin grows, so on deterministic preferences it pushes the margin toward infinity. IPO regresses the margin to a finite target `1/(2β)` with a squared loss, which bounds how far the policy moves from the reference.

### The code, annotated

The shared margin. Shapes must match because each entry is one pair; broadcasting a scalar reference against a batch would silently give every pair the same reference:

```python
def _margin(pc, pr, rc, rr):
    pc, pr, rc, rr = (np.asarray(x, dtype=np.float64) for x in (pc, pr, rc, rr))
    if not (pc.shape == pr.shape == rc.shape == rr.shape):
        raise ValueError("preference log-probability shapes differ")
    # (policy chosen - policy rejected) - (reference chosen - reference rejected) = Δπ - Δref
    return (pc - pr) - (rc - rr)
```

DPO scales the margin by β and applies the logistic loss:

```python
def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.1):
    if beta <= 0:
        raise ValueError("beta must be positive")
    z = beta * _margin(policy_chosen, policy_rejected, reference_chosen, reference_rejected)
    # -log σ(z) = log(e^0 + e^-z), computed without overflow
    return float(np.mean(np.logaddexp(0.0, -z)))
```

IPO uses the unscaled margin and a squared distance to `1/(2β)`:

```python
def ipo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.1):
    if beta <= 0:
        raise ValueError("beta must be positive")
    z = _margin(policy_chosen, policy_rejected, reference_chosen, reference_rejected)
    return float(np.mean((z - 1.0 / (2.0 * beta)) ** 2))
```

Full solution: [`dpo.py`](../../../../course/solutions/python/tinyllm/post/dpo.py)

### Why this solution

- **Cost.** Both losses are $O(\text{pairs})$ once the four log probabilities are known. The expensive part is the forward passes: two through the policy, and two through the reference, which can be computed once per dataset and cached because the reference is frozen.
- **DPO versus a reward model plus PPO.** The reward model and the RL loop are replaced by one classification-style loss. The price is that DPO only learns from the pairs it is given; it cannot explore.
- **`logaddexp` versus `-log(sigmoid(z))`.** The literal form rounds `sigmoid(z)` to 0 for large negative `z` and returns `inf`. `logaddexp` is exact over the whole range.
- **IPO versus DPO.** IPO trades DPO's unbounded push for a fixed target margin. With `β = 0.1` the target is 5 nats, a concrete bound on how much more the policy prefers the chosen answer than the reference does.

### Explain it back

**Why does the loss equal `log 2` when the policy equals the reference?**

<details><summary>Answer</summary>

Then `Δπ = Δref`, the margin is 0, and `-log σ(0) = -log(1/2) = log 2`. The loss starts there and falls only as the policy prefers the chosen answer *more than the reference does*.

</details>

**Why does `log Z(x)` not appear anywhere in the code?**

<details><summary>Answer</summary>

It depends only on the prompt, so it is the same for the chosen and the rejected completion and cancels in the reward difference that Bradley-Terry uses.

</details>

**What happens if you drop the reference terms?**

<details><summary>Answer</summary>

The loss becomes `-log σ(β Δπ)`, which rewards any increase in the chosen-over-rejected log ratio, including the part the SFT model already had. Nothing ties the policy to the reference, so it can drift arbitrarily far; the worked example's loss would also change from 0.6444 to `-log σ(0.12) ≈ 0.6349`.

</details>

**Why is IPO's margin not multiplied by β?**

<details><summary>Answer</summary>

β enters through the target `1/(2β)` instead. Larger β means a smaller target margin, so a stronger tie to the reference, matching β's role as the KL weight in DPO.

</details>

## 6. Pitfalls

| Pitfall | How the solution avoids it |
|---|---|
| Reversing chosen and rejected | `_margin` subtracts rejected from chosen for both models in one place |
| Omitting reference log probabilities | the reference difference is part of `_margin`, shared by DPO and IPO |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M11.1` | Defines the KL term that regularizes policy movement. |
| Back | `S-M10b` | Derives how KL regularization leads to reference-adjusted preference odds. |
| Forward | `C2` | Applies DPO to preference pairs after `L12.1` SFT. |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| paired preference objective | preference optimization pipelines | Annotator disagreement and pair provenance | Rafailov et al., Direct Preference Optimization |
| held-out preference check | online preference evaluation | Position controls and outcome measures beyond training loss | TRL preference trainer |
