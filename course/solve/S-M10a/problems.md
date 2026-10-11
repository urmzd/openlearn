# S-M10a problems: convexity and conditioning, gradient descent rates, momentum, Adam

Answer every question in `solve/S-M10a.toml` (written by `ol start S-M10a`).
The tag after each question is its answer type: `[number]` is an exact value
(`9/11`, `81/121`), `[expr]` a formula in the named variables (write
$\beta$ as `beta`, $\beta_1$ as `beta1`, $\eta$ as `eta`), `[interval]` an
interval such as `[0, oo)` or `(0, 1/5)`, `[bool]` `true` or `false`, and
`[proof]` a file `solve/S-M10a/qN.md` graded against its rubric.

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
