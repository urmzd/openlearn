<!-- ol:module S-M10b -->
# Optimization problem set, part b: constrained optimization, duality, and DPO

## Overview

| | |
|---|---|
| **Module** | `S-M10b` · solve · none · Pass 10 · 3 to 4 h |
| **You build** | answers in `solve/S-M10b.toml` for 18 SymPy-checked problems and four rubric derivations |
| **Contract** | none: a pen and paper set |
| **Tests** | `course/solve/S-M10b/key.toml` (hidden): typed answers, reject canaries, and proof rubrics |
| **Needs** | `M04.2` constrained optimization, `M11.1` KL; reading: [Optimization](README.md) |
| **Used by** | no call site; optional analysis behind L12.2 DPO |
| **Milestone** | `MS-P10` includes this optional solve set only when selected |
| **Optional depth** | Boyd and Vandenberghe, *Convex Optimization*, ch. 5; DPO derivation in Rafailov et al. |

## Key Takeaways

- KKT multipliers pair with the chosen inequality sign convention.
- Strong duality equates primal and dual optima under suitable convexity and feasibility assumptions.
- A KL-regularized policy optimum is proportional to the reference policy times exponentiated reward.
- The DPO log-ratio subtracts away the unknown normalization constant.

## How to work this chapter

```bash
ol start S-M10b
ol check S-M10b
```

## 1. Why now

The optional post-training sequence uses a KL penalty to keep an updated model near its SFT reference. These problems derive the optimization constraints and the closed-form policy that motivates DPO. They follow the earlier gradient and KL material and are not required for the core agent pass.

## 2. Principles

For a minimization problem with inequality `g_i(x) <= 0`, KKT multipliers satisfy `lambda_i >= 0`, stationarity, primal feasibility, and complementary slackness `lambda_i g_i(x)=0`. The Lagrangian is `L(x,lambda)=f(x)+Σ lambda_i g_i(x)`. For a KL-regularized reward objective, maximizing `E_pi[r] - beta D_KL(pi || pi_ref)` gives `pi*(y|x) = pi_ref(y|x) exp(r(x,y)/beta) / Z(x)`.

| Symbol | Meaning |
|---|---|
| `g_i(x)` | inequality constraint, nonpositive when feasible |
| `lambda_i` | nonnegative KKT multiplier |
| `pi_ref` | fixed reference policy |
| `beta` | KL strength, positive |
| `Z(x)` | normalizer over candidate completions |

## 3. Worked example

Minimize `x²` subject to `x >= 1`, written `1-x <= 0`. At `x=1`, stationarity gives `2x-lambda=0`, hence `lambda=2`. The primal value is 1. For two outcomes with equal reference probability, rewards `[1,0]`, and `beta=1`, the optimal policy has probabilities `[e/(1+e), 1/(1+e)]` and log odds 1. These values appear in q3-6 and q11-17.

## 4. Interface and tests

Enter exact fractions, intervals, vectors, booleans, or symbolic expressions in `solve/S-M10b.toml`. The checker evaluates typed expressions using SymPy. Four derivations are self-graded against the rubric. The problem set includes reject canaries for sign errors, infeasible multipliers, reversed odds, and a missing normalizer cancellation.

| Test | Why it exists | Expected result |
|---|---|---|
| KKT stationarity and feasibility | Finds sign convention mistakes | Feasible solution with zero residual |
| Dual value | Checks infimum over the primal variable | Gap zero in the convex fixture |
| DPO odds | Connects reward differences to policy/reference ratios | Difference equals reward gap over beta |

## 5. Pitfalls

| Pitfall | Caught by |
|---|---|
| Using a negative multiplier for `g<=0` | q4 and q6 |
| Forgetting feasibility along with stationarity | q10 canary |
| Maximizing over `x` when forming the dual | q12 |
| Keeping the unknown `Z(x)` in log odds | q18 rubric |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M04.2`, `M11.1` | constraints, gradients, and KL |
| Forward | `L12.2` | DPO uses the optimal policy's reference-adjusted odds |
| Forward | `C2` | optional post-training applies the preference objective |

## Going further

Strong duality needs assumptions such as convex objectives, convex constraints, and a strict feasible point. Real preference learning also has noisy labels and finite data, so a closed-form derivation is a guide to the objective rather than a guarantee of learned behavior.
