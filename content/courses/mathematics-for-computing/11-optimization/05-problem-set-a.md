# Optimization problem set, part a: convexity, GD rates, momentum, Adam

Every model in this series after the count-based bigram is trained by an optimizer you build in this chapter: gradient descent with a line search, SGD with momentum, and AdamW, which trains everything from the encoder-decoder of *Build Your Own LLM System* to its capstone. Those optimizers must match PyTorch's trajectories step by step, and implementations go wrong on exactly the details this set drills: a step size past $2/L$ that oscillates, momentum that is summed one step off, a bias correction applied to the wrong moment, weight decay coupled into the adaptive step. When a training run diverges, the first questions you will ask are the ones here: what is the curvature, what is the condition number, and is the learning rate under the stability limit?

The set has 17 problems (24 numeric or symbolic parts and 2 proofs), grouped into convexity and the condition number, gradient descent rates and step size, momentum, and Adam. It is pen-and-paper work and supplies the analysis behind [Convexity, smoothness, gradient descent, and Armijo line search](01-gradient-descent-and-line-search.md), [The Optimizer protocol: SGD, momentum, Nesterov, weight decay](02-optimizer-protocol-and-sgd.md), and [Adam and AdamW: bias correction and decoupled weight decay](03-adam-and-adamw.md). You will want the proof habits of the [Discrete math 1 problem set](../06-discrete-math-1/03-problem-set.md) for q4 and q17, and the gradients and Hessians of the [Calculus 3 problem set](../05-calculus-3/03-problem-set.md).

Work each problem on paper first, then open the answer and compare not just the result but the derivation. Each answer names the common wrong answer and why it is wrong, because those are exactly the mistakes that show up later as bugs in an optimizer. Each problem also carries a tag for the kind of answer it wants (`[number]`, `[interval]`, `[expr in ...]`, `[bool]`, `[proof]`); rates and steps are exact fractions.

> **Learning objectives**
> - Decide convexity from second derivatives and Hessians, and compute $L$, $\mu$, and $\kappa$ for a quadratic.
> - Derive the stable step-size interval, the optimal constant step, and the step count for a target accuracy.
> - Analyze heavy-ball momentum through its velocity sum and its characteristic polynomial.
> - Compute Adam and AdamW steps by hand, and prove that bias correction makes the first moment unbiased.

## The toolkit

The problems use a small set of symbols, collected here:

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

### Convexity

A function is **convex** if every chord lies on or above it: $f(tx + (1-t)y) \le t f(x) + (1-t) f(y)$ for $t \in [0, 1]$. For twice-differentiable $f$ this is equivalent to $f'' \ge 0$ in one variable, and to a positive semidefinite Hessian ($v^\top \nabla^2 f\, v \ge 0$ for every $v$) in several. Convexity matters because every local minimum of a convex function is global. Sums and nonnegative multiples of convex functions are convex (q4); $\log$ is concave; $\mathrm{LSE}$ is convex, so cross-entropy in the logits is convex, while the loss of a whole network in its weights is not.

### Smoothness, strong convexity, and the condition number

$f$ is **$L$-smooth** when its gradient changes by at most $L$ per unit of distance, which for twice-differentiable $f$ means every Hessian eigenvalue is at most $L$; it is **$\mu$-strongly convex** when every eigenvalue is at least $\mu > 0$. For a quadratic $f(x) = \frac12 x^\top A x$ with symmetric $A$, $L$ and $\mu$ are the largest and smallest eigenvalues of $A$. The **condition number** $\kappa = L/\mu$ measures how elongated the level sets are: $\kappa = 1$ is a round bowl, $\kappa = 10^4$ a long narrow valley. It sets how many steps a digit of accuracy costs: about $\kappa \ln 10$ at $\eta = 1/L$.

### Gradient descent on a quadratic

In the eigenbasis of $A$, gradient descent $x_{t+1} = x_t - \eta A x_t$ acts on each coordinate separately: coordinate $i$ is multiplied by $1 - \eta\lambda_i$ every step. It converges from every start exactly when $\lvert 1 - \eta \lambda_i \rvert < 1$ for every $i$, that is $0 < \eta < 2/L$. The worst factor $\max_i \lvert 1 - \eta\lambda_i \rvert$ is smallest when the extreme eigenvalues balance, $1 - \eta\mu = -(1 - \eta L)$, giving $\eta = \frac{2}{L + \mu}$ and rate $\frac{\kappa - 1}{\kappa + 1}$. The safe default $\eta = 1/L$ gives rate $1 - 1/\kappa$. With rate $\rho$, reducing the error by a factor $\varepsilon$ takes $t \ge \ln(1/\varepsilon) / \ln(1/\rho)$ steps. Without strong convexity, an $L$-smooth convex $f$ still satisfies $f(x_t) - f^\star \le \frac{L\lVert x_0 - x^\star \rVert^2}{2t}$ at $\eta = 1/L$.

**Backtracking (Armijo) line search** avoids knowing $L$: start with $\alpha_0$ and shrink by $\rho$ until the step decreases $f$ by at least a fraction $c$ of what the linear model predicts, $f(x + \alpha d) \le f(x) + c\,\alpha \nabla f(x)^\top d$.

### Momentum

Heavy-ball momentum keeps a velocity $v_{t+1} = \beta v_t + \nabla f(x_t)$ and steps $x_{t+1} = x_t - \eta v_{t+1}$. Unrolled, $v_{t+1} = \sum_{k=0}^{t} \beta^k \nabla f(x_{t-k})$: an exponentially weighted sum of past gradients, which for a constant gradient approaches $g/(1-\beta)$, so steps are $1/(1 - \beta)$ times larger. On a one-dimensional quadratic with curvature $h$ the iterates satisfy the linear recurrence $x_{t+1} = (1 + \beta - \eta h)x_t - \beta x_{t-1}$; substituting $x_t = r^t$ gives a quadratic in $r$, and the iterates shrink like the larger root's modulus. When the roots are complex their product $\beta$ is the squared modulus, so both shrink by $\sqrt\beta$ per step, independent of $h$. Choosing $\beta$ so that this holds for every eigenvalue gives $\beta^\star = \left(\frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}\right)^2$ and a rate of about $1 - 1/\sqrt\kappa$ instead of $1 - 1/\kappa$.

### Adam and AdamW

Adam keeps exponential averages of the gradient and of its elementwise square, $m_t$ and $v_t$, starting from 0. Because they start at 0, early averages are too small by the factor $1 - \beta^t$ (q17), and dividing by it gives $\hat m_t$ and $\hat v_t$. The step $\eta\, \hat m_t / (\sqrt{\hat v_t} + \epsilon)$ is per-coordinate: dividing by the root mean square of recent gradients makes it roughly $\eta$ in size whatever the gradient's scale. On the first step $\hat m_1 = g_1$ and $\hat v_1 = g_1^2$, so the step is $\eta \cdot \mathrm{sign}(g_1)$ when $\epsilon = 0$. **AdamW** adds weight decay as a separate multiplication of the weights by $1 - \eta\lambda$, instead of adding $\lambda\theta$ to the gradient where $\hat v$ would rescale it.

## Two problems worked in full

Before the set itself, here are two siblings of q5 and q16, worked completely, so you can see the level of detail the answers expect.

Consider step sizes on $f(x) = \frac12(2x_1^2 + 6x_2^2)$. The Hessian is $\mathrm{diag}(2, 6)$, so $\mu = 2$, $L = 6$, $\kappa = 3$. Gradient descent converges for $0 < \eta < 2/6 = 1/3$. The best constant step is $\eta = 2/(6 + 2) = 1/4$, where the factors are $1 - 2/4 = 1/2$ and $1 - 6/4 = -1/2$: both coordinates shrink by $1/2$ per step, the rate $(\kappa - 1)/(\kappa + 1) = 2/4$. At $\eta = 1/L = 1/6$ the factors are $2/3$ and $0$, so the rate is $2/3 = 1 - 1/\kappa$: slower, though the stiff coordinate converges in one step. From $x_0 = (1, 1)$ with $\eta = 1/4$: $x_1 = (1/2, -1/2)$, $x_2 = (1/4, 1/4)$.

Now take one AdamW step with $\theta_0 = 2$, $g_1 = -5$, $\eta = 1/10$, $\lambda = 1/2$, $\epsilon = 0$. Then $\hat m_1 = -5$, $\hat v_1 = 25$, the adaptive step is $-\eta \cdot (-5)/5 = +1/10$, and the decay is $-\eta\lambda\theta_0 = -1/10$. So $\theta_1 = 2 - 1/10 + 1/10 = 2$: this step's push upward and the decay cancel exactly.

## The problems

The problems are grouped by topic. Each group builds on the toolkit above; open an answer only after you have one of your own.

### Convexity and the condition number

**q1.** Is the function convex on the given domain? (a) $f(x) = x^4$ on $\mathbb{R}$. (b) $f(x) = \log x$ on $(0, \infty)$. (c) $\mathrm{LSE}(x_1, x_2) = \log(e^{x_1} + e^{x_2})$ on $\mathbb{R}^2$. `[bool]`

<details><summary>Answer</summary>

**(a) true, (b) false, (c) true.**

1. (a) $f''(x) = 12x^2 \ge 0$ everywhere, so $f$ is convex. A common mistake is to answer false because $f''(0) = 0$: convexity needs $f'' \ge 0$, not $f'' > 0$ (that would be strict convexity, and $x^4$ is strictly convex anyway).
2. (b) $f''(x) = -1/x^2 < 0$ on $(0, \infty)$, so $\log$ is concave, not convex.
3. (c) The Hessian of $\mathrm{LSE}$ is $\mathrm{diag}(p) - pp^\top$ with $p = \mathrm{softmax}(x)$, the covariance matrix of a one-hot draw from $p$, so $v^\top H v = \mathrm{Var}_p(v_i) \ge 0$. This is why cross-entropy, which is $\mathrm{LSE}$ of the logits minus a linear term, is convex in the logits.

</details>

**q2.** $f(x) = \frac12 x^\top A x$ with $A = \mathrm{diag}(1, 10)$. Give (a) the smoothness constant $L$, (b) the strong convexity constant $\mu$, and (c) the condition number $\kappa = L/\mu$. `[number]`

<details><summary>Answer</summary>

**(a) $L = 10$, (b) $\mu = 1$, (c) $\kappa = 10$.**

The Hessian of $\frac12 x^\top A x$ is $A$ itself, and its eigenvalues are the diagonal entries 1 and 10. $L$ is the largest, $\mu$ the smallest, so $\kappa = L/\mu = 10$. A common mistake is $\kappa = 1/10$, which is $\mu/L$: the condition number is at least 1 by definition. This is the kind of quadratic on which the gradient descent of [Convexity, smoothness, gradient descent, and Armijo line search](01-gradient-descent-and-line-search.md) shows the $(1 - 1/\kappa)^t$ gap.

</details>

**q3.** On which interval is $f(x) = x^3 - 3x$ convex? Give the largest one. `[interval]`

<details><summary>Answer</summary>

**$[0, \infty)$.**

$f'(x) = 3x^2 - 3$ and $f''(x) = 6x$, which is nonnegative exactly when $x \ge 0$. A common mistake is $[1, \infty)$: that is where $f' \ge 0$, where $f$ is increasing, which says nothing about curvature. Another is $(-\infty, 0]$, the concave side.

</details>

**q4.** Prove: if $f$ and $g$ are convex on $\mathbb{R}^n$, then $f + g$ is convex. `[proof]`

<details><summary>Answer</summary>

**Proof.** Let $x, y \in \mathbb{R}^n$ and $t \in [0, 1]$. By convexity of $f$ and of $g$ at the same $x$, $y$, $t$:

$$f(tx + (1-t)y) \le t f(x) + (1-t) f(y), \qquad g(tx + (1-t)y) \le t g(x) + (1-t) g(y).$$

Adding the two inequalities gives

$$(f+g)(tx + (1-t)y) \le t (f+g)(x) + (1-t)(f+g)(y),$$

which is the definition of convexity for $f + g$. $\blacksquare$

What makes the proof work: both inequalities are taken at the same points and the same $t$, so the right-hand sides add term by term. The same argument with a factor $c \ge 0$ shows nonnegative multiples are convex; with $c < 0$ the inequality flips, which is why the condition is needed.

</details>

### Gradient descent rates and step size

**q5.** Gradient descent $x_{t+1} = x_t - \eta \nabla f(x_t)$ with a constant step $\eta > 0$.
(a) For $f(x) = 5x^2$ (so $L = 10$), give the set of $\eta > 0$ for which $x_t \to 0$ from every start. `[interval]`
For $f(x) = \frac12 (x_1^2 + 10 x_2^2)$:
(b) give the step $\eta$ that minimizes the worst per-coordinate contraction $\max_i \lvert 1 - \eta \lambda_i \rvert$; `[number]`
(c) give that worst contraction factor; `[number]`
(d) give the worst contraction factor with $\eta = 1/L$. `[number]`

<details><summary>Answer</summary>

**(a) $(0, 1/5)$, (b) $\eta = 2/11$, (c) $9/11$, (d) $9/10$.**

1. (a) $f'(x) = 10x$, so $x_{t+1} = (1 - 10\eta) x_t$. This converges from every start iff $\lvert 1 - 10\eta \rvert < 1$, i.e. $0 < \eta < 2/10 = 1/5$. A common mistake is $(0, 1/10)$, the condition $\eta \le 1/L$: it is sufficient but not the whole set. Another is $(0, 1/5]$: at $\eta = 2/L$ the factor is exactly $-1$ and the iterate flips sign forever without shrinking.
2. (b) The eigenvalues are $\mu = 1$ and $L = 10$. The worst factor $\max(\lvert 1 - \eta \rvert, \lvert 1 - 10\eta \rvert)$ is smallest where the two balance, $1 - \eta = -(1 - 10\eta)$, so $11\eta = 2$ and $\eta = 2/11$. Not $1/L$: that is the safe default, not the optimum.
3. (c) At $\eta = 2/11$ both factors have magnitude $1 - 2/11 = 9/11 = (\kappa - 1)/(\kappa + 1)$.
4. (d) At $\eta = 1/10$ the factors are $9/10$ and $0$, so the worst is $9/10 = 1 - 1/\kappa$. A common mistake is $1/10 = 1/\kappa$, which would predict absurdly fast convergence.

</details>

**q6.** The error shrinks by a factor $9/10$ per step. What is the smallest number of steps $t$ with $(9/10)^t \le 10^{-6}$? `[number]`

<details><summary>Answer</summary>

**132.**

Take logs: $t \ln(9/10) \le -6 \ln 10$, and since $\ln(9/10) < 0$ dividing flips the inequality: $t \ge 6 \ln 10 / \ln(10/9) \approx 13.8155 / 0.10536 \approx 131.13$. The smallest integer is 132. A common mistake is 131, rounding down: after 131 steps the error is still slightly above $10^{-6}$. This matches the rule of thumb from the toolkit above: at $\eta = 1/L$ with $\kappa = 10$, each digit costs about $\kappa \ln 10 \approx 23$ steps (the exact figure is $\ln 10 / \ln(10/9) \approx 21.9$).

</details>

**q7.** Run gradient descent on $f(x) = x^2$ from $x_0 = 1$ with $\eta = 1/4$. Give $x_3$. `[number]`

<details><summary>Answer</summary>

**$x_3 = 1/8$.**

$f'(x) = 2x$, so $x_{t+1} = x_t - \frac14 \cdot 2x_t = x_t / 2$. From $x_0 = 1$: $1/2$, $1/4$, $1/8$. A common mistake is $27/64$, which comes from using $f'(x) = x$ (dropping the 2), so the factor becomes $3/4$.

</details>

**q8.** Backtracking (Armijo) line search accepts the first $\alpha \in \{\alpha_0, \rho\alpha_0, \rho^2\alpha_0, \dots\}$ with $f(x + \alpha d) \le f(x) + c\,\alpha\, \nabla f(x)^\top d$. For $f(x) = x^2$ at $x = 1$ with $d = -\nabla f(1)$, $\alpha_0 = 1$, $c = 1/2$, $\rho = 1/2$, which $\alpha$ is accepted? `[number]`

<details><summary>Answer</summary>

**$\alpha = 1/2$.**

At $x = 1$: $f(1) = 1$, $\nabla f(1) = 2$, $d = -2$, and $\nabla f^\top d = -4$.

1. $\alpha = 1$: $x + \alpha d = -1$, $f = 1$. The test needs $1 \le 1 + \frac12 \cdot 1 \cdot (-4) = -1$. It fails: the step overshoots to the mirror point with no decrease.
2. $\alpha = 1/2$: $x + \alpha d = 0$, $f = 0$. The test needs $0 \le 1 + \frac12 \cdot \frac12 \cdot (-4) = 0$. It holds, so $1/2$ is accepted.

A common mistake is $\alpha = 1$, accepting $\alpha_0$ without testing it; another is $1/4$, backtracking once more after the test already passed.

</details>

**q9.** For convex, $L$-smooth $f$, gradient descent with $\eta = 1/L$ satisfies $f(x_t) - f^\star \le \frac{L \lVert x_0 - x^\star \rVert^2}{2t}$. Give the bound for $L = 4$, $\lVert x_0 - x^\star \rVert = 3$, $t = 10$. `[number]`

<details><summary>Answer</summary>

**$9/5$.**

$\frac{L \lVert x_0 - x^\star \rVert^2}{2t} = \frac{4 \cdot 9}{2 \cdot 10} = \frac{36}{20} = \frac95$. A common mistake is $18/5$, dropping the 2 in the denominator. Note the $1/t$ rate: without strong convexity, halving the gap costs twice as many steps, not a constant number.

</details>

### Momentum

Heavy-ball momentum: $v_{t+1} = \beta v_t + \nabla f(x_t)$, $x_{t+1} = x_t - \eta\, v_{t+1}$, with $v_0 = 0$.

**q10.** Suppose the gradient is a constant $g$. (a) Give $\lim_{t \to \infty} v_t$. `[expr in g, beta]` (b) For $\beta = 0.9$, by what factor is the long-run step larger than plain gradient descent's step $\eta g$? `[number]`

<details><summary>Answer</summary>

**(a) $g/(1 - \beta)$, (b) 10.**

1. (a) With $v_0 = 0$: $v_1 = g$, $v_2 = \beta g + g$, and in general $v_t = g(1 + \beta + \dots + \beta^{t-1}) = g\frac{1 - \beta^t}{1 - \beta} \to \frac{g}{1 - \beta}$. A common mistake is $\beta g/(1 - \beta)$, which sums from the second term only and forgets that the current gradient enters with weight 1.
2. (b) The long-run step is $\eta g/(1 - \beta)$, so the factor is $1/(1 - 0.9) = 10$. A common mistake is 9, which is $\beta/(1 - \beta)$. This factor is why a learning rate tuned for plain SGD must shrink by about $1 - \beta$ when momentum is turned on.

</details>

**q11.** On $f(x) = \frac{h}{2} x^2$ the iterates satisfy $x_{t+1} = (1 + \beta - \eta h)\, x_t - \beta\, x_{t-1}$.
(a) Give the monic characteristic polynomial in $r$ whose roots are the modes $x_t \propto r^t$. `[expr in r, beta, eta, h]`
(b) When its two roots are complex, both have modulus $\sqrt{\beta}$. Give that modulus for $\beta = 81/100$. `[number]`

<details><summary>Answer</summary>

**(a) $r^2 - (1 + \beta - \eta h) r + \beta$, (b) $9/10$.**

1. (a) Substitute $x_t = r^t$ into $x_{t+1} = (1 + \beta - \eta h)x_t - \beta x_{t-1}$ and divide by $r^{t-1}$: $r^2 = (1 + \beta - \eta h) r - \beta$, i.e. $r^2 - (1 + \beta - \eta h) r + \beta = 0$. A common mistake is $+\ldots - \beta$, the wrong sign on the $x_{t-1}$ term; the product of the roots would then be $-\beta$ and the analysis would predict a mode of modulus above 1. Another is $r^2 - (1 - \eta h) r$, plain gradient descent with a spurious root at 0.
2. (b) For a monic quadratic the product of the roots is the constant term, $\beta$. Complex roots are conjugates, so $r \bar r = \lvert r \rvert^2 = \beta$ and $\lvert r \rvert = \sqrt{81/100} = 9/10$. A common mistake is $81/100$, which is $\beta$ itself rather than its square root.

</details>

**q12.** The best heavy-ball momentum on a quadratic with condition number $\kappa$ is $\beta^\star = \left(\frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}\right)^2$. Give $\beta^\star$ for $\kappa = 100$. `[number]`

<details><summary>Answer</summary>

**$81/121$.**

$\sqrt{100} = 10$, so $\beta^\star = \left(\frac{10 - 1}{10 + 1}\right)^2 = \left(\frac{9}{11}\right)^2 = \frac{81}{121}$. A common mistake is $9/11$, forgetting the square; another is $99/101$, using $\kappa$ where $\sqrt\kappa$ belongs. The convergence rate is then $\sqrt{\beta^\star} = 9/11$, which for $\kappa = 100$ beats plain gradient descent's best rate of $99/101$ by a wide margin: that is the $\sqrt\kappa$ speed-up.

</details>

**q13.** With $\beta = 1/2$ and gradient $1$ at every step, give $v_3$. `[number]`

<details><summary>Answer</summary>

**$v_3 = 7/4$.**

$v_1 = \frac12 \cdot 0 + 1 = 1$, $v_2 = \frac12 \cdot 1 + 1 = \frac32$, $v_3 = \frac12 \cdot \frac32 + 1 = \frac74$. A common mistake is $3/2$ ($v_2$) or $15/8$ ($v_4$): an off-by-one in where the sum starts. With $v_0 = 0$, $v_t$ has exactly $t$ terms.

</details>

### Adam

Adam keeps $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$ and $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$ from $m_0 = v_0 = 0$, corrects $\hat m_t = m_t / (1 - \beta_1^t)$ and $\hat v_t = v_t / (1 - \beta_2^t)$, and steps $\theta_t = \theta_{t-1} - \eta\, \hat m_t / (\sqrt{\hat v_t} + \epsilon)$. AdamW also decays the weights: $\theta_t = \theta_{t-1} - \eta \lambda \theta_{t-1} - \eta\, \hat m_t / (\sqrt{\hat v_t} + \epsilon)$.

**q14.** If every gradient equals the same $g$, give $m_t$. `[expr in g, beta1, t]`

<details><summary>Answer</summary>

**$m_t = g(1 - \beta_1^t)$.**

Unroll: $m_t = (1 - \beta_1) \sum_{i=1}^{t} \beta_1^{t-i} g = g (1 - \beta_1)\frac{1 - \beta_1^t}{1 - \beta_1} = g(1 - \beta_1^t)$. A common mistake is $g$, which is the bias-corrected $\hat m_t$, not $m_t$. With $\beta_1 = 0.9$, $m_1 = 0.1 g$: without the correction the first step would be ten times too small, which is the bias that [Adam and AdamW](03-adam-and-adamw.md) removes.

</details>

**q15.** Take $\epsilon = 0$. On the first step the gradient is $g_1 = -3$ and $\eta = 1/100$. Give $\theta_1 - \theta_0$. `[number]`

<details><summary>Answer</summary>

**$\theta_1 - \theta_0 = 1/100$.**

1. $m_1 = (1 - \beta_1) g_1$ and $\hat m_1 = m_1/(1 - \beta_1) = g_1 = -3$.
2. $v_1 = (1 - \beta_2) g_1^2$ and $\hat v_1 = g_1^2 = 9$, so $\sqrt{\hat v_1} = 3$.
3. $\theta_1 - \theta_0 = -\eta\,\hat m_1/\sqrt{\hat v_1} = -\frac{1}{100} \cdot \frac{-3}{3} = +\frac{1}{100}$.

The first step is $-\eta\,\mathrm{sign}(g_1)$ whatever the size of the gradient. A common mistake is $-1/100$ (moving with the gradient instead of against it); another is $3/100$ (using the raw gradient size, as plain SGD would).

</details>

**q16.** AdamW with $\epsilon = 0$, $\theta_0 = 1$, $g_1 = 4$, $\eta = 1/10$, $\lambda = 1/10$. Give $\theta_1$. `[number]`

<details><summary>Answer</summary>

**$\theta_1 = 89/100$.**

1. The adaptive step is $-\eta\,\mathrm{sign}(g_1) = -\frac{1}{10}$, by the same argument as q15.
2. The decoupled decay is $-\eta\lambda\theta_0 = -\frac{1}{10}\cdot\frac{1}{10}\cdot 1 = -\frac{1}{100}$.
3. $\theta_1 = 1 - \frac{1}{100} - \frac{1}{10} = \frac{89}{100}$.

A common mistake is $9/10$ (forgot the decay) or $99/100$ (forgot the Adam step). Had the decay been added to the gradient instead ($g' = g + \lambda\theta$), Adam's normalization would have divided it by $\sqrt{\hat v}$ and the first step would still be exactly $-\eta$, the decay would vanish into the sign: that is why AdamW keeps it outside.

</details>

**q17.** Prove that if every $g_i$ has the same expectation $E[g]$, then $E[m_t] = (1 - \beta_1^t)\, E[g]$, so $\hat m_t$ is an unbiased estimate of $E[g]$. `[proof]`

<details><summary>Answer</summary>

**Proof.** From $m_0 = 0$ and $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$, induction on $t$ gives

$$m_t = (1 - \beta_1) \sum_{i=1}^{t} \beta_1^{t-i} g_i.$$

(Base case $t = 1$: $m_1 = (1 - \beta_1) g_1$. Step: $\beta_1 m_{t-1} + (1 - \beta_1) g_t = (1 - \beta_1)\left(\sum_{i=1}^{t-1} \beta_1^{t-i} g_i + g_t\right)$.) Taking expectations and using linearity with $E[g_i] = E[g]$ for every $i$:

$$E[m_t] = (1 - \beta_1) E[g] \sum_{k=0}^{t-1} \beta_1^{k} = (1 - \beta_1) E[g] \frac{1 - \beta_1^t}{1 - \beta_1} = (1 - \beta_1^t)\,E[g].$$

Dividing by the constant $1 - \beta_1^t$ gives $E[\hat m_t] = E[g]$, so $\hat m_t$ is unbiased. $\blacksquare$

The same argument with $g_i^2$ shows $E[\hat v_t] = E[g^2]$. The assumption that every $g_i$ has the same mean is what makes the geometric sum factor out; in training the mean drifts slowly, which is why the correction is exact only in this idealization but matters most in the first few hundred steps, where $\beta_2^t$ is still far from 0.

</details>

## Mistakes to watch for

The common wrong answers in the set fall into a handful of patterns, and each one has a counterpart bug in optimizer code:

| Pitfall | Symptom | How to avoid it |
|---|---|---|
| Confusing where $f$ increases with where it is convex | a wrong region of convexity | test $f'' \ge 0$, not $f' \ge 0$ (q3) |
| Taking $\eta < 1/L$ as the stability limit, or including $\eta = 2/L$ | needlessly small learning rates, or a run that oscillates forever | require $\lvert 1 - \eta\lambda \rvert < 1$ strictly, giving the open interval $(0, 2/L)$ (q5(a)) |
| Rate $1/\kappa$ instead of $1 - 1/\kappa$ | absurd step-count estimates | compute the factor $1 - \eta\lambda_i$ per eigenvalue (q5(d)) |
| Rounding a step count down | one step short of the target accuracy | take the ceiling of $\ln(1/\varepsilon)/\ln(1/\rho)$ (q6) |
| Dropping the factor 2 in $\nabla x^2 = 2x$ | half-speed descent | differentiate before iterating (q7) |
| Accepting $\alpha_0$ without the Armijo test | a step that increases the loss | evaluate the sufficient-decrease test at every candidate, including $\alpha_0$ (q8) |
| Summing momentum from the wrong index | velocity off by one term | start from $v_0 = 0$ so $v_t$ has exactly $t$ terms (q10, q13) |
| Wrong sign of the $x_{t-1}$ term | a momentum analysis that predicts divergence | derive the polynomial from the recurrence; the root product is $+\beta$ (q11) |
| Forgetting the bias correction is about $m_t$'s start at 0 | first steps 10 times too small | unroll $m_t$ to get the factor $1 - \beta_1^t$ (q14, q17) |
| Decay inside the adaptive step, or omitted | AdamW trajectories that drift from torch's | apply $-\eta\lambda\theta$ separately from $-\eta\hat m/\sqrt{\hat v}$ (q16) |

## Where this leads

The convexity and rate problems (q1 to q9) are the theory behind `gradient_descent` and `armijo_step` in [Convexity, smoothness, gradient descent, and Armijo line search](01-gradient-descent-and-line-search.md), whose convergence follows the $(1 - 1/\kappa)^t$ bound of q5. The momentum problems (q10 to q13) analyze `SGD` with momentum and Nesterov in [The Optimizer protocol](02-optimizer-protocol-and-sgd.md), and the Adam problems (q14 to q17) check `AdamW` with bias correction and decoupled decay in [Adam and AdamW](03-adam-and-adamw.md). [Learning-rate schedules (cosine, WSD, Noam) and gradient clipping](04-schedules-and-gradient-clipping.md) then changes $\eta$ over time and bounds the step, and [Training loop and the autograd bigram](../../build-an-llm/01-foundations/06-training-loop-and-the-autograd-bigram.md) in *Build Your Own LLM System* is the first training loop that calls these optimizers. Beyond this book, the top Hessian eigenvalue during training and the edge of stability, where $\eta \approx 2/\lambda_{\max}$, apply q5's analysis to real networks. Part b of the set, [Optimization problem set, part b](06-problem-set-b.md), turns to constrained optimization and the derivation behind DPO.

### Further reading

- Boyd and Vandenberghe, [*Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/) (free), chapters 3 and 9.
- Goh, [*Why Momentum Really Works*](https://distill.pub/2017/momentum/) (Distill, 2017).
- Kingma and Ba, ["Adam"](https://arxiv.org/abs/1412.6980) (2015), sections 2 and 3.
- Loshchilov and Hutter, ["Decoupled Weight Decay Regularization"](https://arxiv.org/abs/1711.05101) (2019).

## Summary

- On a quadratic, gradient descent multiplies each eigen-direction by $1 - \eta\lambda_i$, so it converges exactly when $0 < \eta < 2/L$, and its best rate is $\frac{\kappa - 1}{\kappa + 1}$ (q5).
- The condition number $\kappa = L/\mu$ sets how many steps a digit of accuracy costs: about $\kappa \ln 10$ at $\eta = 1/L$ (q2, q6).
- Momentum averages gradients with weights $\beta^k$, so a constant gradient produces steps $1/(1 - \beta)$ times larger; on a quadratic its modes shrink by $\sqrt\beta$ per step (q10, q11).
- Adam's moments start at zero and are biased toward it by a factor $1 - \beta^t$; the bias correction removes exactly that, and the first step is $\eta$ times the sign of the gradient (q14, q15, q17).
- AdamW's decay multiplies the weights by $1 - \eta\lambda$ outside the adaptive step, so it is not rescaled by $\hat v$ (q16).
