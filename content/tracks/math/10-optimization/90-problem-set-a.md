# Optimization problem set, part a: convexity, GD rates, momentum, Adam

## Overview

| | |
|---|---|
| **Module** | `S-M10a` · solve · none · Pass 2 · 4 to 5 h |
| **Solution** | worked answers to all 17 problems (24 numeric or symbolic parts and 2 proofs), each under its problem in section 4 |
| **Contract** | none: a pen and paper set |
| **Needs** | `S-M05` (proof habits). Reading: the [Optimization topic](README.md), and gradients and Hessians from `S-M04` |
| **Used by** | no call site (a problem set). It is the analysis behind `M10.1` (gradient descent, Armijo, the condition number), `M10.2` (SGD with momentum), and `M10.3` (Adam and AdamW with bias correction); part b, `S-M10b`, is optional with `L12` |
| **Optional depth** | Boyd and Vandenberghe, *Convex Optimization* (free), ch. 3 and 9; Goh, "Why Momentum Really Works" (Distill, 2017); Kingma and Ba, "Adam" (2015), sections 2 and 3; Loshchilov and Hutter, "Decoupled Weight Decay Regularization" (2019) |

## Key Takeaways

- On a quadratic, gradient descent multiplies each eigen-direction by $1 - \eta\lambda_i$, so it converges exactly when $0 < \eta < 2/L$, and its best rate is $\frac{\kappa - 1}{\kappa + 1}$ (q5).
- The condition number $\kappa = L/\mu$ sets how many steps a digit of accuracy costs: about $\kappa \ln 10$ at $\eta = 1/L$ (q2, q6).
- Momentum averages gradients with weights $\beta^k$, so a constant gradient produces steps $1/(1 - \beta)$ times larger; on a quadratic its modes shrink by $\sqrt\beta$ per step (q10, q11).
- Adam's moments start at zero and are biased toward it by a factor $1 - \beta^t$; the bias correction removes exactly that, and the first step is $\eta$ times the sign of the gradient (q14, q15, q17).
- AdamW's decay multiplies the weights by $1 - \eta\lambda$ outside the adaptive step, so it is not rescaled by $\hat v$ (q16).

## How to read this chapter

Sections 1 and 2 explain why these facts matter and derive the tools from first principles; section 3 works two sibling problems by hand. Section 4 states every problem with its worked answer folded underneath: try each one on paper first, then open the answer and compare not just the result but the derivation. Each answer names the common wrong answer and why it is wrong, because those are exactly the mistakes that show up later as bugs in an optimizer.

---

## 1. Why now

The Pass 1 bigram was fitted by counting, with no optimizer at all. From Pass 2 on, every model is trained by an optimizer from the solutions: `M10.1` gradient descent with a line search, `M10.2` SGD with momentum, and `M10.3` AdamW, which trains everything from `L4.1` to the capstone. Those optimizers must match PyTorch's trajectories step by step, and an implementation goes wrong on exactly the details this set drills: a step size past $2/L$ that oscillates, momentum that is summed one step off, a bias correction applied to the wrong moment, weight decay coupled into the adaptive step. When a training run diverges in Pass 5, the first questions you will ask are the ones here: what is the curvature, what is the condition number, and is the learning rate under the stability limit?

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $f: \mathbb{R}^n \to \mathbb{R}$ | the objective (loss) | |
| $x^\star, f^\star$ | a minimizer and the minimum value | |
| $\nabla f, \nabla^2 f$ | gradient and Hessian | $n$, $n \times n$ |
| $L$ | smoothness: $\nabla^2 f \preceq L I$ (largest curvature) | scalar |
| $\mu$ | strong convexity: $\nabla^2 f \succeq \mu I$ (smallest curvature) | scalar |
| $\kappa = L/\mu$ | condition number | scalar $\ge 1$ |
| $\eta$ (`eta`) | step size, the learning rate | scalar |
| $\beta$ (`beta`), $\beta_1, \beta_2$ | momentum and Adam's moment decay rates | in $[0, 1)$ |
| $m_t, v_t$ | Adam's first and second moment estimates | like $x$ |
| $\lambda$ | AdamW weight decay | scalar |

### 2.1 Convexity

A function is **convex** if every chord lies on or above it: $f(tx + (1-t)y) \le t f(x) + (1-t) f(y)$ for $t \in [0, 1]$. For twice-differentiable $f$ this is equivalent to $f'' \ge 0$ in one variable, and to a positive semidefinite Hessian ($v^\top \nabla^2 f\, v \ge 0$ for every $v$) in several. Convexity matters because every local minimum of a convex function is global. Sums and nonnegative multiples of convex functions are convex (q4); $\log$ is concave; $\mathrm{LSE}$ is convex, so cross-entropy in the logits is convex, while the loss of a whole network in its weights is not.

### 2.2 Smoothness, strong convexity, and the condition number

$f$ is **$L$-smooth** when its gradient changes by at most $L$ per unit of distance, which for twice-differentiable $f$ means every Hessian eigenvalue is at most $L$; it is **$\mu$-strongly convex** when every eigenvalue is at least $\mu > 0$. For a quadratic $f(x) = \frac12 x^\top A x$ with symmetric $A$, $L$ and $\mu$ are the largest and smallest eigenvalues of $A$. The **condition number** $\kappa = L/\mu$ measures how elongated the level sets are: $\kappa = 1$ is a round bowl, $\kappa = 10^4$ a long narrow valley.

### 2.3 Gradient descent on a quadratic

In the eigenbasis of $A$, gradient descent $x_{t+1} = x_t - \eta A x_t$ acts on each coordinate separately: coordinate $i$ is multiplied by $1 - \eta\lambda_i$ every step. It converges from every start exactly when $\lvert 1 - \eta \lambda_i \rvert < 1$ for every $i$, that is $0 < \eta < 2/L$. The worst factor $\max_i \lvert 1 - \eta\lambda_i \rvert$ is smallest when the extreme eigenvalues balance, $1 - \eta\mu = -(1 - \eta L)$, giving $\eta = \frac{2}{L + \mu}$ and rate $\frac{\kappa - 1}{\kappa + 1}$. The safe default $\eta = 1/L$ gives rate $1 - 1/\kappa$. With rate $\rho$, reducing the error by a factor $\varepsilon$ takes $t \ge \ln(1/\varepsilon) / \ln(1/\rho)$ steps. Without strong convexity, an $L$-smooth convex $f$ still satisfies $f(x_t) - f^\star \le \frac{L\lVert x_0 - x^\star \rVert^2}{2t}$ at $\eta = 1/L$.

**Backtracking (Armijo) line search** avoids knowing $L$: start with $\alpha_0$ and shrink by $\rho$ until the step decreases $f$ by at least a fraction $c$ of what the linear model predicts, $f(x + \alpha d) \le f(x) + c\,\alpha \nabla f(x)^\top d$.

### 2.4 Momentum

Heavy-ball momentum keeps a velocity $v_{t+1} = \beta v_t + \nabla f(x_t)$ and steps $x_{t+1} = x_t - \eta v_{t+1}$. Unrolled, $v_{t+1} = \sum_{k=0}^{t} \beta^k \nabla f(x_{t-k})$: an exponentially weighted sum of past gradients, which for a constant gradient approaches $g/(1-\beta)$. On a one-dimensional quadratic with curvature $h$ the iterates satisfy the linear recurrence $x_{t+1} = (1 + \beta - \eta h)x_t - \beta x_{t-1}$; substituting $x_t = r^t$ gives a quadratic in $r$, and the iterates shrink like the larger root's modulus. When the roots are complex their product $\beta$ is the squared modulus, so both shrink by $\sqrt\beta$ per step, independent of $h$. Choosing $\beta$ so that this holds for every eigenvalue gives $\beta^\star = \left(\frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}\right)^2$ and a rate of about $1 - 1/\sqrt\kappa$ instead of $1 - 1/\kappa$.

### 2.5 Adam and AdamW

Adam keeps exponential averages of the gradient and of its elementwise square, $m_t$ and $v_t$, starting from 0. Because they start at 0, early averages are too small by the factor $1 - \beta^t$ (q17), and dividing by it gives $\hat m_t$ and $\hat v_t$. The step $\eta\, \hat m_t / (\sqrt{\hat v_t} + \epsilon)$ is per-coordinate: dividing by the root mean square of recent gradients makes it roughly $\eta$ in size whatever the gradient's scale. On the first step $\hat m_1 = g_1$ and $\hat v_1 = g_1^2$, so the step is $\eta \cdot \mathrm{sign}(g_1)$ when $\epsilon = 0$. **AdamW** adds weight decay as a separate multiplication of the weights by $1 - \eta\lambda$, instead of adding $\lambda\theta$ to the gradient where $\hat v$ would rescale it.

## 3. Worked example by hand

This is a sibling of q5 and q16, not one of the problems below.

**Step sizes on $f(x) = \frac12(2x_1^2 + 6x_2^2)$.** The Hessian is $\mathrm{diag}(2, 6)$, so $\mu = 2$, $L = 6$, $\kappa = 3$. Gradient descent converges for $0 < \eta < 2/6 = 1/3$. The best constant step is $\eta = 2/(6 + 2) = 1/4$, where the factors are $1 - 2/4 = 1/2$ and $1 - 6/4 = -1/2$: both coordinates shrink by $1/2$ per step, the rate $(\kappa - 1)/(\kappa + 1) = 2/4$. At $\eta = 1/L = 1/6$ the factors are $2/3$ and $0$, so the rate is $2/3 = 1 - 1/\kappa$: slower, though the stiff coordinate converges in one step. From $x_0 = (1, 1)$ with $\eta = 1/4$: $x_1 = (1/2, -1/2)$, $x_2 = (1/4, 1/4)$.

**One AdamW step.** $\theta_0 = 2$, $g_1 = -5$, $\eta = 1/10$, $\lambda = 1/2$, $\epsilon = 0$. Then $\hat m_1 = -5$, $\hat v_1 = 25$, the adaptive step is $-\eta \cdot (-5)/5 = +1/10$, and the decay is $-\eta\lambda\theta_0 = -1/10$. So $\theta_1 = 2 - 1/10 + 1/10 = 2$: this step's push upward and the decay cancel exactly.

## 4. The problem set

Each problem carries a tag for the kind of answer it wants (`[number]`, `[interval]`, `[expr in ...]`, `[bool]`, `[proof]`). Rates and steps are exact fractions. Open the answer only after you have one of your own.

### Convexity and the condition number

**q1.** Is the function convex on the given domain? (a) $f(x) = x^4$ on $\mathbb{R}$. (b) $f(x) = \log x$ on $(0, \infty)$. (c) $\mathrm{LSE}(x_1, x_2) = \log(e^{x_1} + e^{x_2})$ on $\mathbb{R}^2$. `[bool]`

**q2.** $f(x) = \frac12 x^\top A x$ with $A = \mathrm{diag}(1, 10)$. Give (a) the smoothness constant $L$, (b) the strong convexity constant $\mu$, and (c) the condition number $\kappa = L/\mu$. `[number]`

**q3.** On which interval is $f(x) = x^3 - 3x$ convex? Give the largest one. `[interval]`

**q4.** Prove: if $f$ and $g$ are convex on $\mathbb{R}^n$, then $f + g$ is convex. `[proof]`

### Gradient descent rates and step size

**q5.** Gradient descent $x_{t+1} = x_t - \eta \nabla f(x_t)$ with a constant step $\eta > 0$.
(a) For $f(x) = 5x^2$ (so $L = 10$), give the set of $\eta > 0$ for which $x_t \to 0$ from every start. `[interval]`
For $f(x) = \frac12 (x_1^2 + 10 x_2^2)$:
(b) give the step $\eta$ that minimizes the worst per-coordinate contraction $\max_i \lvert 1 - \eta \lambda_i \rvert$; `[number]`
(c) give that worst contraction factor; `[number]`
(d) give the worst contraction factor with $\eta = 1/L$. `[number]`

**q6.** The error shrinks by a factor $9/10$ per step. What is the smallest number of steps $t$ with $(9/10)^t \le 10^{-6}$? `[number]`

**q7.** Run gradient descent on $f(x) = x^2$ from $x_0 = 1$ with $\eta = 1/4$. Give $x_3$. `[number]`

**q8.** Backtracking (Armijo) line search accepts the first $\alpha \in \{\alpha_0, \rho\alpha_0, \rho^2\alpha_0, \dots\}$ with $f(x + \alpha d) \le f(x) + c\,\alpha\, \nabla f(x)^\top d$. For $f(x) = x^2$ at $x = 1$ with $d = -\nabla f(1)$, $\alpha_0 = 1$, $c = 1/2$, $\rho = 1/2$, which $\alpha$ is accepted? `[number]`

**q9.** For convex, $L$-smooth $f$, gradient descent with $\eta = 1/L$ satisfies $f(x_t) - f^\star \le \frac{L \lVert x_0 - x^\star \rVert^2}{2t}$. Give the bound for $L = 4$, $\lVert x_0 - x^\star \rVert = 3$, $t = 10$. `[number]`

### Momentum

Heavy-ball momentum: $v_{t+1} = \beta v_t + \nabla f(x_t)$, $x_{t+1} = x_t - \eta\, v_{t+1}$, with $v_0 = 0$.

**q10.** Suppose the gradient is a constant $g$. (a) Give $\lim_{t \to \infty} v_t$. `[expr in g, beta]` (b) For $\beta = 0.9$, by what factor is the long-run step larger than plain gradient descent's step $\eta g$? `[number]`

**q11.** On $f(x) = \frac{h}{2} x^2$ the iterates satisfy $x_{t+1} = (1 + \beta - \eta h)\, x_t - \beta\, x_{t-1}$.
(a) Give the monic characteristic polynomial in $r$ whose roots are the modes $x_t \propto r^t$. `[expr in r, beta, eta, h]`
(b) When its two roots are complex, both have modulus $\sqrt{\beta}$. Give that modulus for $\beta = 81/100$. `[number]`

**q12.** The best heavy-ball momentum on a quadratic with condition number $\kappa$ is $\beta^\star = \left(\frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}\right)^2$. Give $\beta^\star$ for $\kappa = 100$. `[number]`

**q13.** With $\beta = 1/2$ and gradient $1$ at every step, give $v_3$. `[number]`

### Adam

Adam keeps $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$ and $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$ from $m_0 = v_0 = 0$, corrects $\hat m_t = m_t / (1 - \beta_1^t)$ and $\hat v_t = v_t / (1 - \beta_2^t)$, and steps $\theta_t = \theta_{t-1} - \eta\, \hat m_t / (\sqrt{\hat v_t} + \epsilon)$. AdamW also decays the weights: $\theta_t = \theta_{t-1} - \eta \lambda \theta_{t-1} - \eta\, \hat m_t / (\sqrt{\hat v_t} + \epsilon)$.

**q14.** If every gradient equals the same $g$, give $m_t$. `[expr in g, beta1, t]`

**q15.** Take $\epsilon = 0$. On the first step the gradient is $g_1 = -3$ and $\eta = 1/100$. Give $\theta_1 - \theta_0$. `[number]`

**q16.** AdamW with $\epsilon = 0$, $\theta_0 = 1$, $g_1 = 4$, $\eta = 1/10$, $\lambda = 1/10$. Give $\theta_1$. `[number]`

**q17.** Prove that if every $g_i$ has the same expectation $E[g]$, then $E[m_t] = (1 - \beta_1^t)\, E[g]$, so $\hat m_t$ is an unbiased estimate of $E[g]$. `[proof]`

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Confusing where $f$ increases with where it is convex | a wrong region of convexity | q3 (canary [1, oo)) |
| Taking $\eta < 1/L$ as the stability limit, or including $\eta = 2/L$ | needlessly small learning rates, or a run that oscillates forever | q5 (canaries (0, 1/10) and (0, 1/5]) |
| Rate $1/\kappa$ instead of $1 - 1/\kappa$ | absurd step-count estimates | q5 (canary 1/10) |
| Rounding a step count down | one step short of the target accuracy | q6 (canary 131) |
| Dropping the factor 2 in $\nabla x^2 = 2x$ | half-speed descent | q7 (canary 27/64) |
| Accepting $\alpha_0$ without the Armijo test | a step that increases the loss | q8 (canary 1) |
| Summing momentum from the wrong index | velocity off by one term | q10 (canary), q13 (canaries 3/2 and 15/8) |
| Wrong sign of the $x_{t-1}$ term | a momentum analysis that predicts divergence | q11 (canary) |
| Forgetting the bias correction is about $m_t$'s start at 0 | first steps 10 times too small | q14 (canary g) |
| Decay inside the adaptive step, or omitted | AdamW trajectories that drift from torch's | q16 (canaries 9/10 and 99/100) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `S-M05` | proof habits for q4 and q17 |
| Forward | `M10.1` | `gradient_descent` and `armijo_step`, tested by the $(1 - 1/\kappa)^t$ bound of q5 |
| Forward | `M10.2` | `SGD` with momentum and Nesterov: q10 to q13 |
| Forward | `M10.3` | `AdamW` with bias correction and decoupled decay: q14 to q17 |
| Forward | `M10.4` | schedules change $\eta$ over time; clipping bounds the step |
| Forward | `L0.5` | the first training loop that calls your optimizer |
| Forward | `M10.5` | optional: the top Hessian eigenvalue and the edge of stability, where $\eta \approx 2/\lambda_{\max}$ |
