# Optimization problem set, part b: constrained optimization, duality, and DPO

## Overview

| | |
|---|---|
| **Module** | `S-M10b` · solve · none · Pass 10 · 3 to 4 h |
| **Solution** | worked answers to all 18 problems (17 exact answers and one short derivation), each under its problem in section 4 |
| **Contract** | none: a pen and paper set |
| **Needs** | `M04.2` constrained optimization, `M11.1` KL; reading: [Optimization](README.md) |
| **Used by** | no call site; optional analysis behind L12.2 DPO in Pass 10 |
| **Optional depth** | Boyd and Vandenberghe, *Convex Optimization*, ch. 5; DPO derivation in Rafailov et al. |

## Key Takeaways

- KKT multipliers pair with the chosen inequality sign convention.
- Strong duality equates primal and dual optima under suitable convexity and feasibility assumptions.
- A KL-regularized policy optimum is proportional to the reference policy times exponentiated reward.
- The DPO log-ratio subtracts away the unknown normalization constant.

## How to read this chapter

Sections 1 to 3 set up the KKT conditions, the dual, and the KL-regularized optimum, and work one example by hand. Section 4 states every problem with its worked answer folded underneath; attempt each on paper before opening it. The answers name the common wrong answer and why it fails, since sign conventions are where these derivations usually go wrong.

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

Minimize `x²` subject to `x >= 1`, written `1-x <= 0`. At `x=1`, stationarity gives `2x-lambda=0`, hence `lambda=2`. The primal value is 1. For two outcomes with equal reference probability, rewards `[1,0]`, and `beta=1`, the optimal policy has probabilities `[e/(1+e), 1/(1+e)]` and log odds 1. These values reappear in q3 to q6 and q11 to q17.

## 4. The problem set

For KKT questions, use the convention `g(x) <= 0` and multipliers `lambda >= 0`. Each answer below gives the result, the derivation, and the common wrong answer.

| What a correct answer must handle | Why it matters | Problems |
|---|---|---|
| KKT stationarity and feasibility | finds sign convention mistakes | q1 to q10 |
| Dual value | the dual is an infimum over the primal variable; the gap is zero in this convex example | q11 to q14 |
| DPO odds | connects reward differences to policy/reference ratios | q15 to q18 |

### Lagrange multipliers and KKT

**q1.** Minimize `x^2 + y^2` subject to `x + y = 2`. Give the minimizer. `[vector]`

<details><summary>Answer</summary>

**`[1, 1]`.** With $L = x^2 + y^2 + \lambda(x + y - 2)$, stationarity gives $2x + \lambda = 0$ and $2y + \lambda = 0$, so $x = y$; the constraint then gives $x = y = 1$. Geometrically, this is the point of the line $x + y = 2$ closest to the origin, where the gradient $(2x, 2y)$ is parallel to the constraint normal $(1, 1)$. A common mistake is `[0, 2]`: feasible, but $f = 4 > 2$.

</details>

**q2.** For the same problem, give the equality multiplier using `L=f+lambda*(x+y-2)`. `[number]`

<details><summary>Answer</summary>

**$\lambda = -2$.** From $2x + \lambda = 0$ at $x = 1$. An equality multiplier has no sign restriction; its sign depends only on how the constraint is written. A common mistake is $+2$, which is what the other convention $L = f - \lambda(x + y - 2)$ gives.

</details>

**q3.** Minimize `x^2` subject to `x >= 1`. Give the minimizer. `[number]`

<details><summary>Answer</summary>

**$x = 1$.** The unconstrained minimizer $x = 0$ is infeasible, and $x^2$ is increasing on $[1, \infty)$, so the constraint is active at the optimum. A common mistake is $-1$, which violates $x \ge 1$.

</details>

**q4.** Write the nonnegative multiplier for constraint `1-x <= 0` at the minimizer. `[number]`

<details><summary>Answer</summary>

**$\lambda = 2$.** $L = x^2 + \lambda(1 - x)$, so $\partial L/\partial x = 2x - \lambda = 0$ at $x = 1$ gives $\lambda = 2 \ge 0$. A common mistake is $-2$, from writing the constraint as $x - 1 \le 0$ or flipping the sign in $L$: a negative multiplier on a `g <= 0` constraint means the "optimum" is pushing the wrong way.

</details>

**q5.** For problem 3, give the Lagrangian gradient stationarity residual at `x=1`, `lambda=2`. `[number]`

<details><summary>Answer</summary>

**0.** $2x - \lambda = 2 - 2 = 0$. A common mistake is 4, reporting $2x + \lambda$ or the multiplier itself instead of the residual.

</details>

**q6.** Is complementary slackness satisfied at `x=1`, `lambda=2`? `[bool]`

<details><summary>Answer</summary>

**true.** $\lambda\,g(x) = 2(1 - 1) = 0$. The constraint is active, so a nonzero multiplier is allowed.

</details>

**q7.** Minimize `(x-3)^2` subject to `x <= 1`. Give the minimizer. `[number]`

<details><summary>Answer</summary>

**$x = 1$.** The unconstrained minimizer 3 is infeasible, and $(x - 3)^2$ decreases on $(-\infty, 1]$, so the optimum sits on the boundary. A common mistake is 3, ignoring the constraint.

</details>

**q8.** Give the nonnegative multiplier for `x-1 <= 0` in problem 7. `[number]`

<details><summary>Answer</summary>

**$\lambda = 4$.** $L = (x - 3)^2 + \lambda(x - 1)$, so $2(x - 3) + \lambda = 0$ at $x = 1$ gives $\lambda = 4$. Its size is the rate at which the optimal value would fall if the constraint were loosened: $\frac{d}{db}(b - 3)^2 = -4$ at $b = 1$. A common mistake is $-4$, a sign error in $g$.

</details>

**q9.** At a strict feasible point, what value must an inactive inequality multiplier take? `[number]`

<details><summary>Answer</summary>

**0.** Complementary slackness $\lambda_i g_i(x) = 0$ with $g_i(x) < 0$ forces $\lambda_i = 0$. An inactive constraint exerts no force on the optimum. A common mistake is 1, as if the multiplier had a default value.

</details>

**q10.** State whether `x=0, lambda=0` is feasible for `x >= 1`, with `g=1-x <= 0`. `[bool]`

<details><summary>Answer</summary>

**false.** $g(0) = 1 > 0$, so primal feasibility fails, even though stationarity of the unconstrained objective holds there ($2 \cdot 0 - 0 = 0$). This is why KKT is four conditions, not one: stationarity alone finds the unconstrained minimum.

</details>

### Duality

**q11.** Minimize `x^2` subject to `x >= 1`. Give the primal optimal value. `[number]`

<details><summary>Answer</summary>

**1.** The minimizer is $x = 1$ (q3), and $f(1) = 1$. A common mistake is 0, the unconstrained minimum, which is infeasible.

</details>

**q12.** For `L=x^2+lambda*(1-x)`, give the dual function `inf_x L`. `[expr in lam]`

<details><summary>Answer</summary>

**$\lambda - \lambda^2/4$.** For fixed $\lambda$, $L$ is a convex quadratic in $x$ minimized at $2x - \lambda = 0$, $x = \lambda/2$. Substituting: $\lambda^2/4 + \lambda - \lambda^2/2 = \lambda - \lambda^2/4$. A common mistake is $\lambda$ (dropping the $x$ terms); another is to maximize over $x$, which gives $+\infty$.

</details>

**q13.** Give the dual optimal value for problem 11. `[number]`

<details><summary>Answer</summary>

**1.** Maximize $d(\lambda) = \lambda - \lambda^2/4$ over $\lambda \ge 0$: $d'(\lambda) = 1 - \lambda/2 = 0$ at $\lambda = 2$, which is the KKT multiplier of q4, and $d(2) = 2 - 1 = 1$. A common mistake is 0, the value of $d$ at $\lambda = 0$.

</details>

**q14.** Is the duality gap zero? `[bool]`

<details><summary>Answer</summary>

**true.** Primal 1 (q11) equals dual 1 (q13). Weak duality always gives dual $\le$ primal; equality here follows from Slater's condition: the objective and constraint are convex and there is a strictly feasible point (for example $x = 2$, where $g = -1 < 0$).

</details>

### KL-regularized objective and DPO

**q15.** For rewards `[1,0]`, reference probabilities `[1/2,1/2]`, and `beta=1`, give optimal probabilities proportional to `p_ref * exp(reward/beta)`. `[vector]`

<details><summary>Answer</summary>

**$[e/(1 + e),\ 1/(1 + e)]$**, about $[0.731, 0.269]$. The unnormalized weights are $\frac12 e^1$ and $\frac12 e^0$; dividing by their sum $Z = \frac12(e + 1)$ gives the result. A common mistake is $[1/2, 1/2]$, which ignores the reward tilt and returns the reference policy.

</details>

**q16.** Give the log-odds of the optimal policy from problem 15. `[number]`

<details><summary>Answer</summary>

**1.** $\log\frac{e/(1+e)}{1/(1+e)} = \log e = 1$, which equals $(r_1 - r_2)/\beta$ plus the reference log odds $\log(\frac12/\frac12) = 0$. A common mistake is 0, which omits the reward difference.

</details>

**q17.** If chosen and rejected have equal policy/reference log odds, give the per-pair DPO loss. `[number]`

<details><summary>Answer</summary>

**$\log 2$.** The DPO loss is $-\log\sigma(\beta\,\Delta)$ with $\Delta$ the chosen log ratio minus the rejected log ratio. Equal log ratios give $\Delta = 0$, and $-\log\sigma(0) = -\log\frac12 = \log 2 \approx 0.693$. This is the loss at initialization, when the policy equals the reference. A common mistake is 0: the loss is $\mathrm{softplus}(-\beta\Delta)$, and $\mathrm{softplus}(0) = \log 2$, not 0.

</details>

**q18.** Derive in one or two sentences why subtracting the reference log ratio recovers the reward difference divided by beta at the optimum. `[proof]`

<details><summary>Answer</summary>

At the optimum, $\pi^*(y \mid x) = \pi_{\text{ref}}(y \mid x)\exp(r(x, y)/\beta)/Z(x)$, so $\log\frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} = \frac{r(x, y)}{\beta} - \log Z(x)$. Taking this log ratio for the chosen $y_w$ minus the rejected $y_l$ under the same prompt $x$, the shared $\log Z(x)$ cancels and leaves $\frac{r(x, y_w) - r(x, y_l)}{\beta}$.

The cancellation is the whole point: $Z(x)$ is a sum over every possible completion and cannot be computed, but DPO only ever needs differences for the same prompt, so it never has to.

</details>

## 5. Pitfalls

| Pitfall | How the solution avoids it |
|---|---|
| Using a negative multiplier for `g<=0` | writes every inequality as `g(x) <= 0` before forming `L = f + lambda g`, so stationarity gives `lambda >= 0`; see q4 and q8 |
| Forgetting feasibility along with stationarity | checks all four KKT conditions, not just the gradient; see q10 |
| Maximizing over `x` when forming the dual | the dual function is an infimum over `x`, then a maximum over `lambda >= 0`; see q12 and q13 |
| Keeping the unknown `Z(x)` in log odds | takes a difference of log ratios for the same prompt, so `log Z(x)` cancels; see q18 |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M04.2`, `M11.1` | constraints, gradients, and KL |
| Forward | `L12.2` | DPO uses the optimal policy's reference-adjusted odds |
| Forward | `C2` | optional post-training applies the preference objective |

## Going further

Strong duality needs assumptions such as convex objectives, convex constraints, and a strict feasible point. Real preference learning also has noisy labels and finite data, so a closed-form derivation is a guide to the objective rather than a guarantee of learned behavior.
