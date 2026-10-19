# Optimization problem set, part b: constrained optimization, duality, and DPO

Post-training a language model, in the optional post-training chapter of *Build Your Own LLM System*, uses a KL penalty to keep an updated model near its supervised fine-tuned reference. The method that chapter builds, [Direct preference optimization and IPO](../../build-an-llm/13-post-training/02-direct-preference-optimization.md), rests on one closed-form result: the policy that maximizes reward minus a KL penalty is the reference policy tilted by the exponentiated reward. This problem set derives that result and the constrained-optimization machinery behind it, so that when you meet the DPO loss you can see where every term comes from.

The set has 18 problems (17 exact answers and one short derivation) in three groups: Lagrange multipliers and the KKT conditions, duality, and the KL-regularized objective with DPO. It is optional pen-and-paper work. It assumes the constrained optimization and Lagrange material of the [Calculus 3 problem set](../05-calculus-3/03-problem-set.md) and the KL divergence of [Entropy, cross-entropy, KL, JS, and the k3 estimator](../12-information-theory/01-entropy-cross-entropy-and-kl.md). Attempt each problem on paper before opening its answer. The answers name the common wrong answer and why it fails, since sign conventions are where these derivations usually go wrong.

> **Learning objectives**
> - Write the KKT conditions for a problem with constraints $g_i(x) \le 0$ and solve small instances by hand, with multipliers of the correct sign.
> - Form the dual function as an infimum over the primal variable, maximize it, and check for a zero duality gap.
> - Derive the optimum of a KL-regularized reward objective, $\pi^* \propto \pi_{\text{ref}} \exp(r/\beta)$.
> - Explain why the DPO log ratio cancels the intractable normalizer $Z(x)$.

## The toolkit

For a minimization problem with inequality constraints $g_i(x) \le 0$, the Lagrangian is

$$L(x, \lambda) = f(x) + \sum_i \lambda_i\, g_i(x).$$

The KKT conditions at an optimum are four: stationarity ($\nabla_x L = 0$), primal feasibility ($g_i(x) \le 0$), dual feasibility ($\lambda_i \ge 0$), and complementary slackness ($\lambda_i\, g_i(x) = 0$). The sign of the multiplier pairs with the sign convention of the constraint, which is why every inequality in this set is written as $g(x) \le 0$ with $\lambda \ge 0$ before $L$ is formed. The dual function is $d(\lambda) = \inf_x L(x, \lambda)$, and the dual problem maximizes it over $\lambda \ge 0$. Weak duality says the dual optimum never exceeds the primal one; strong duality, which holds under suitable convexity and feasibility assumptions such as Slater's condition, makes them equal.

For a KL-regularized reward objective, maximizing $\mathbb{E}_{\pi}[r] - \beta\, D_{\mathrm{KL}}(\pi \,\|\, \pi_{\text{ref}})$ gives

$$\pi^*(y \mid x) = \frac{\pi_{\text{ref}}(y \mid x)\, \exp(r(x, y)/\beta)}{Z(x)}.$$

The symbols, briefly:

| Symbol | Meaning |
|---|---|
| $g_i(x)$ | inequality constraint, nonpositive when feasible |
| $\lambda_i$ | nonnegative KKT multiplier |
| $\pi_{\text{ref}}$ | fixed reference policy |
| $\beta$ | KL strength, positive |
| $Z(x)$ | normalizer over candidate completions |

## A problem worked in full

Minimize $x^2$ subject to $x \ge 1$, written $1 - x \le 0$. At $x = 1$, stationarity gives $2x - \lambda = 0$, hence $\lambda = 2$, and the primal value is 1. For the KL side, take two outcomes with equal reference probability, rewards $[1, 0]$, and $\beta = 1$: the optimal policy has probabilities $[e/(1+e),\ 1/(1+e)]$ and log odds 1. These values reappear in q3 to q6 and q11 to q17.

## The problems

For KKT questions, use the convention `g(x) <= 0` and multipliers `lambda >= 0`. Each answer gives the result, the derivation, and the common wrong answer. The first group (q1 to q10) catches sign-convention mistakes in stationarity and feasibility; the second (q11 to q14) builds the dual as an infimum over the primal variable and shows the gap is zero in this convex example; the third (q15 to q18) connects reward differences to policy-to-reference ratios.

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

## Mistakes to watch for

Four mistakes account for most wrong answers in this set:

| Pitfall | How to avoid it |
|---|---|
| Using a negative multiplier for `g<=0` | write every inequality as `g(x) <= 0` before forming `L = f + lambda g`, so stationarity gives `lambda >= 0` (q4, q8) |
| Forgetting feasibility along with stationarity | check all four KKT conditions, not just the gradient (q10) |
| Maximizing over `x` when forming the dual | the dual function is an infimum over `x`, then a maximum over `lambda >= 0` (q12, q13) |
| Keeping the unknown `Z(x)` in log odds | take a difference of log ratios for the same prompt, so `log Z(x)` cancels (q18) |

## Where this leads

The closed-form optimum of q15 and the cancellation of q18 are exactly what [Direct preference optimization and IPO](../../build-an-llm/13-post-training/02-direct-preference-optimization.md) uses: DPO trains on the optimal policy's reference-adjusted odds, and [Capstone 2: post-trained TinyStories model](../../build-an-llm/capstones/02-post-trained.md) applies the preference objective to a real model.

A caution about how far the derivation reaches. Strong duality needs assumptions such as convex objectives, convex constraints, and a strict feasible point. Real preference learning also has noisy labels and finite data, so a closed-form derivation is a guide to the objective rather than a guarantee of learned behavior.

### Further reading

- Boyd and Vandenberghe, [*Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/), chapter 5.
- Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023), for the DPO derivation.

## Summary

- KKT multipliers pair with the chosen inequality sign convention: with $g(x) \le 0$ and $L = f + \lambda g$, multipliers are nonnegative, and an inactive constraint has $\lambda = 0$.
- Strong duality equates primal and dual optima under suitable convexity and feasibility assumptions, such as Slater's condition.
- A KL-regularized policy optimum is proportional to the reference policy times exponentiated reward.
- The DPO log ratio subtracts away the unknown normalization constant $Z(x)$.
