# Calculus 1 Problem Set

This problem set works through the pen-and-paper side of Calculus 1 that the code in this chapter depends on. [Derivatives and finite differences](01-derivatives-and-finite-differences.md) approximates derivatives by stopping a limit early, [Activation Functions and Their Derivatives](03-activation-functions.md) writes the derivative of every activation, and [Gradients and gradcheck](../10-calculus-3/01-gradients-and-gradcheck.md) turns those into `gradcheck`, the test that guards every backward pass in the autograd engine of *AI Engineering*. All of it assumes you can differentiate a formula by hand and know what a limit is, because a derivative is a limit and a finite difference is that limit stopped early. Training picks the weights that minimize a loss, which is optimization; the learning-rate schedules of [Learning-rate schedules and gradient clipping](../16-optimization/04-schedules-and-gradient-clipping.md) and the area under an ROC curve in [Logistic regression, ROC-AUC, and calibration](../13-probability-statistics/08-logistic-regression-roc-auc-calibration.md) are integrals.

The set has 57 problems in six groups: limits, derivative rules and the chain rule, implicit differentiation and related rates, optimization, integration and the fundamental theorem, and L'Hôpital's rule. Fifty-four have exact answers, and three (Problems 10, 25, and 57) ask for a written proof. It is best taken after the first three sections of this chapter; [Definite integrals: trapezoid and Simpson](04-definite-integrals-trapezoid-simpson.md) and the [Calculus 2 problem set](../08-calculus-2/03-problem-set.md) build on its integrals. The background it assumes, functions, exponentials, and logarithms, is covered in the [precalculus solve set](../06-precalculus/05-problem-set.md).

Each problem is followed by a collapsed worked answer: the result, the derivation, and, where there is one, the most common wrong answer and why it is wrong. Attempt each problem before opening its answer; the wrong answers are as instructive as the right ones.

> **Learning objectives**
> - Evaluate limits by rewriting indeterminate forms rather than substituting into them.
> - Differentiate any formula the autograd needs with the product, quotient, and chain rules, including the sigmoid, softplus, tanh, and SiLU.
> - Solve implicit-differentiation, related-rates, and optimization problems, checking every candidate.
> - Evaluate definite integrals with the fundamental theorem, including derivatives of integrals with variable limits.
> - Apply L'Hôpital's rule correctly and recognize when its hypothesis fails.

## The rules you will use

This review collects the definitions and rules the problems rely on, in the order the problems use them.

### Limits

$\lim_{x \to a} f(x) = L$ means that $f(x)$ gets arbitrarily close to $L$ as $x \ne a$ gets close to $a$. Precisely: for every $\varepsilon > 0$ there is a $\delta > 0$ such that $0 < |x - a| < \delta$ implies $|f(x) - L| < \varepsilon$. The value $f(a)$ plays no part, which is why $\frac{x^2 - 4}{x - 2}$ has a limit at 2 even though it is undefined there: for $x \ne 2$ it equals $x + 2$. Limits add, multiply, and divide (when the denominator's limit is not 0).

Three standard limits do most of the work: $\frac{\sin u}{u} \to 1$ as $u \to 0$, $(1 + \frac{a}{x})^x \to e^a$ as $x \to \infty$, and, for rational functions at infinity, only the highest powers matter. A **one-sided** limit restricts $x$ to one side of $a$. Forms like $0/0$, $\infty/\infty$, $\infty - \infty$, $0 \cdot \infty$, $1^\infty$, and $0^0$ are **indeterminate**: they say the problem needs rewriting (factor, multiply by a conjugate, take logarithms), not what the answer is.

### Derivative rules

The derivative $f'(x) = \frac{df}{dx}$ is the limit of the difference quotient, $\lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$, and the second derivative $f''$ is the derivative of $f'$. From that definition follow the rules you use instead of it: linearity; the **power rule** $\frac{d}{dx} x^n = n x^{n-1}$ for a constant $n$; $\frac{d}{dx} e^x = e^x$, $\frac{d}{dx} \ln x = \frac{1}{x}$, $\frac{d}{dx} \sin x = \cos x$, $\frac{d}{dx} \cos x = -\sin x$; the **product rule** $(fg)' = f'g + fg'$; the **quotient rule** $(f/g)' = (f'g - fg')/g^2$; and the **chain rule**, $\frac{d}{dx} f(g(x)) = f'(g(x))\, g'(x)$: the outer derivative evaluated at the inner function, times the inner derivative. The chain rule is the whole of backpropagation, as the chapter [Matrix calculus and autodiff](../14-matrix-calculus-and-autodiff/README.md) shows from [Dual numbers and forward-mode autodiff](../14-matrix-calculus-and-autodiff/01-dual-numbers-forward-mode.md) through [Matrix differentials and VJPs](../14-matrix-calculus-and-autodiff/03-matrix-differentials-and-vjps.md). A variable exponent, as in $x^x$, needs $x^x = e^{x \ln x}$ first; the power rule does not apply. Several problems use the sigmoid, $\sigma(z) = 1/(1 + e^{-z})$.

### Implicit differentiation and related rates

When $y$ is defined by an equation such as $x^2 + y^2 = 25$, differentiate both sides with respect to $x$, treating $y$ as a function of $x$ (so $\frac{d}{dx} y^2 = 2y\,y'$), and solve for $y'$. **Related rates** apply the same idea with time $t$: if two quantities are tied by an equation and both change with $t$, differentiating the equation in $t$ ties their rates.

### Optimization

At an interior minimum or maximum of a differentiable $f$, $f'(x) = 0$; such points are **critical points**. A global optimum on an interval is a critical point or an endpoint (or a limit at an open end), so list the candidates and compare their values. The second derivative classifies a critical point: $f'' > 0$ is a local minimum, $f'' < 0$ a local maximum. Gradient descent, in [Gradient descent and line search](../16-optimization/01-gradient-descent-and-line-search.md), finds the same points numerically when solving $f' = 0$ by hand is impossible.

### Integrals and the fundamental theorem

The definite integral $\int_a^b f(x)\,dx$, the signed area under $f$ from $a$ to $b$, is the limit of **Riemann sums** $\sum_i f(x_i)\,\Delta x$ over finer and finer partitions. The **fundamental theorem of calculus** (FTC) connects it to derivatives in two ways: if $F$ is an antiderivative of $f$ ($F' = f$), then $\int_a^b f = F(b) - F(a)$; and $\frac{d}{dx} \int_a^x f(t)\,dt = f(x)$ for continuous $f$. With a variable upper limit $u(x)$, the chain rule adds a factor: $\frac{d}{dx} \int_a^{u(x)} f(t)\,dt = f(u(x))\,u'(x)$. The **average value** of $f$ on $[a, b]$ is $\frac{1}{b - a}\int_a^b f$.

### L'Hôpital's rule

If $f(x) \to 0$ and $g(x) \to 0$ (or both $\to \pm\infty$) as $x \to a$, and $\lim f'(x)/g'(x)$ **exists**, then $\lim f(x)/g(x)$ equals it. It may need several applications ($\frac{e^x - 1 - x}{x^2}$ takes two). Products $0 \cdot \infty$ become quotients, and powers $0^0$ or $1^\infty$ become products after taking the logarithm. When the limit of $f'/g'$ does not exist, the rule says nothing, and the original limit may still exist (Problem 57).

## Two worked examples

Before the problems, here are two siblings of Problems 13 and 38, worked in full. They are not part of the set.

**A chain rule derivative.** Differentiate $f(x) = \ln(1 + e^{-x})$. The outer function is $\ln u$ with derivative $1/u$; the inner is $u = 1 + e^{-x}$ with derivative $-e^{-x}$ (the chain rule again, with inner function $-x$). So $f'(x) = \frac{-e^{-x}}{1 + e^{-x}}$. Multiply top and bottom by $e^{x}$: $f'(x) = \frac{-1}{e^x + 1} = -\sigma(-x)$. This is the derivative of the binary cross-entropy for a positive label, written in terms of the logit. Both $\frac{-e^{-x}}{1 + e^{-x}}$ and $\frac{-1}{e^x + 1}$ are the same answer written two ways; $\frac{1}{1 + e^{-x}}$, the sigmoid itself, is not (it has the wrong sign and the wrong argument).

**An optimization.** Maximize $g(s) = s(10 - 2s)^2$ for $0 \le s \le 5$, the volume of an open box folded from a $10 \times 10$ sheet. $g'(s) = (10 - 2s)^2 + s \cdot 2(10 - 2s)(-2) = (10 - 2s)(10 - 2s - 4s) = (10 - 2s)(10 - 6s)$. The critical points are $s = 5$ (a zero-volume box) and $s = 5/3$. Compare candidates: $g(0) = 0$, $g(5) = 0$, $g(5/3) = \frac{5}{3} \cdot \left(\frac{20}{3}\right)^2 = \frac{2000}{27}$. The maximizer is $s = 5/3$.

Now the problems. Numbers are exact ($e^2$, $2/\pi$, $\sqrt 7/2$, not decimals), and an expression answer may take any equivalent form. The tag after each problem says what kind of answer it wants.

## Problems on limits

**Problem 1.** $\displaystyle\lim_{x \to 2} \frac{x^2 - 4}{x - 2}$. `[number]`

<details><summary>Answer</summary>

**4.** Substituting gives $0/0$, a signal to rewrite. Factor: $\frac{(x - 2)(x + 2)}{x - 2} = x + 2$ for $x \ne 2$, and the limit never looks at $x = 2$, so it is $2 + 2 = 4$. A common mistake is 0, from substituting into the numerator only.

</details>

**Problem 2.** $\displaystyle\lim_{x \to 0} \frac{\sin 3x}{x}$. `[number]`

<details><summary>Answer</summary>

**3.** Write $\frac{\sin 3x}{x} = 3 \cdot \frac{\sin 3x}{3x}$ and use $\frac{\sin u}{u} \to 1$ with $u = 3x \to 0$. A common mistake is 1, applying the standard limit without matching its argument, which drops the factor 3.

</details>

**Problem 3.** $\displaystyle\lim_{x \to \infty} \frac{3x^2 + 2x}{5x^2 - 1}$. `[number]`

<details><summary>Answer</summary>

**3/5.** Divide top and bottom by $x^2$: $\frac{3 + 2/x}{5 - 1/x^2} \to \frac{3}{5}$. For rational functions at infinity only the highest powers matter. A common mistake is 0, treating the denominator as "winning" when the degrees are equal.

</details>

**Problem 4.** $\displaystyle\lim_{h \to 0} \frac{(1 + h)^3 - 1}{h}$. (Which derivative is this?) `[number]`

<details><summary>Answer</summary>

**3.** It is the difference quotient of $f(x) = x^3$ at $x = 1$, so the limit is $f'(1) = 3 \cdot 1^2 = 3$. Directly: $(1 + h)^3 - 1 = 3h + 3h^2 + h^3$, and dividing by $h$ gives $3 + 3h + h^2 \to 3$. The finite differences of [Derivatives and finite differences](01-derivatives-and-finite-differences.md) approximate exactly this limit with a fixed $h$. A common mistake is 1.

</details>

**Problem 5.** $\displaystyle\lim_{x \to 0} \frac{1 - \cos x}{x^2}$. `[number]`

<details><summary>Answer</summary>

**1/2.** Multiply top and bottom by $1 + \cos x$: $\frac{1 - \cos^2 x}{x^2(1 + \cos x)} = \left(\frac{\sin x}{x}\right)^2 \cdot \frac{1}{1 + \cos x} \to 1 \cdot \frac12$. (Or L'Hôpital twice: $\frac{\sin x}{2x} \to \frac12$.) A common mistake is 0, reading $1 - \cos x \to 0$ as the whole answer.

</details>

**Problem 6.** $\displaystyle\lim_{x \to \infty} \left(1 + \frac{2}{x}\right)^{x}$. `[number]`

<details><summary>Answer</summary>

**$e^2$.** The standard limit $(1 + \frac{a}{x})^x \to e^a$ with $a = 2$. To see it, take logs: $x \ln(1 + 2/x) = \frac{\ln(1 + 2/x)}{1/x}$, a $0/0$ form whose limit is 2 (L'Hôpital, or $\ln(1 + u) \approx u$), so the limit is $e^2$. Two common mistakes: 1, reading $1^\infty$ as 1 (it is indeterminate), and $e$, dropping the 2.

</details>

**Problem 7.** $\displaystyle\lim_{x \to 0^+} x \ln x$. `[number]`

<details><summary>Answer</summary>

**0.** A $0 \cdot (-\infty)$ form. Rewrite as $\frac{\ln x}{1/x}$, an $\infty/\infty$ form, and apply L'Hôpital: $\frac{1/x}{-1/x^2} = -x \to 0$. The factor $x$ goes to 0 faster than $\ln x$ goes to $-\infty$. A common mistake is $-1$.

</details>

**Problem 8.** $\displaystyle\lim_{x \to \infty} \left(\sqrt{x^2 + x} - x\right)$. `[number]`

<details><summary>Answer</summary>

**1/2.** An $\infty - \infty$ form. Multiply by the conjugate: $\frac{(x^2 + x) - x^2}{\sqrt{x^2 + x} + x} = \frac{x}{\sqrt{x^2 + x} + x} = \frac{1}{\sqrt{1 + 1/x} + 1} \to \frac{1}{2}$. A common mistake is 0, reading $\infty - \infty$ as 0.

</details>

**Problem 9.** $\displaystyle\lim_{x \to 0^-} \frac{|x|}{x}$ (from the left). `[number]`

<details><summary>Answer</summary>

**$-1$.** For $x < 0$, $|x| = -x$, so the quotient is $-1$ at every point the left-hand limit uses. A common mistake is 1, which is the right-hand limit; because the two one-sided limits differ, the two-sided limit does not exist.

</details>

**Problem 10.** Prove from the definition that $\displaystyle\lim_{x \to 3} (2x + 1) = 7$: for every $\varepsilon > 0$, give a $\delta > 0$ such that $0 < |x - 3| < \delta$ implies $|(2x + 1) - 7| < \varepsilon$. `[proof]`

<details><summary>Answer</summary>

**Proof (direct).** Let $\varepsilon > 0$ be given. For every real $x$, $|(2x + 1) - 7| = |2x - 6| = 2|x - 3|$. Choose $\delta = \varepsilon / 2$, which is positive. If $0 < |x - 3| < \delta$, then $|(2x + 1) - 7| = 2|x - 3| < 2\delta = \varepsilon$. Since $\varepsilon > 0$ was arbitrary, the definition of the limit holds, so $\lim_{x \to 3} (2x + 1) = 7$.

A complete proof states the goal, rewrites the distance as $2|x - 3|$, names $\delta$ explicitly (any positive $\delta \le \varepsilon/2$ works), and shows that this $\delta$ works for every such $x$.

</details>

## Problems on derivative rules and the chain rule

**Problem 11.** $\dfrac{d}{dx}\, x^3 \sin x$. `[expr in x]`

<details><summary>Answer</summary>

**$3x^2 \sin x + x^3 \cos x$.** Product rule, $(fg)' = f'g + fg'$, with $f = x^3$ and $g = \sin x$. A common mistake is $3x^2 \cos x$: differentiating each factor and multiplying, which is not a rule.

</details>

**Problem 12.** $\dfrac{d}{dx}\, \dfrac{e^x}{1 + x^2}$. `[expr in x]`

<details><summary>Answer</summary>

**$\dfrac{e^x (1 - 2x + x^2)}{(1 + x^2)^2}$**, which is $\dfrac{e^x (x - 1)^2}{(1 + x^2)^2}$. Quotient rule: $\frac{e^x (1 + x^2) - e^x \cdot 2x}{(1 + x^2)^2}$, then factor out $e^x$. Common mistakes: $+2x$ in the numerator (the quotient rule's minus sign flipped), and $e^x / (2x)$ (differentiating top and bottom separately).

</details>

**Problem 13.** The sigmoid is $\sigma(z) = \dfrac{1}{1 + e^{-z}}$. Give $\sigma'(z)$. `[expr in z]`

<details><summary>Answer</summary>

**$\dfrac{e^{-z}}{(1 + e^{-z})^2}$**, equal to $\sigma(z)(1 - \sigma(z))$. Chain rule on $(1 + e^{-z})^{-1}$: the outer derivative is $-(1 + e^{-z})^{-2}$, the inner derivative is $-e^{-z}$, and the two minus signs cancel. This is the sigmoid backward of [The op library and gradcheck_all](../../ai-engineering/06-numerics-and-autograd/03-op-library-and-gradcheck.md) and [Activation Functions and Their Derivatives](03-activation-functions.md). Common mistakes: a leading minus sign (the inner derivative's sign lost), and $\sigma(z)$ itself.

</details>

**Problem 14.** Softplus is $s(x) = \ln(1 + e^x)$. Give $s'(x)$. `[expr in x]`

<details><summary>Answer</summary>

**$\dfrac{e^x}{1 + e^x}$**, which is $\sigma(x)$. Chain rule: $\frac{1}{1 + e^x} \cdot e^x$. A common mistake is $\frac{1}{1 + e^x}$, which forgets the inner derivative $e^x$ and is $\sigma(-x)$.

</details>

**Problem 15.** $\tanh x = \dfrac{e^x - e^{-x}}{e^x + e^{-x}}$. Give $\dfrac{d}{dx} \tanh x$ in terms of exponentials. `[expr in x]`

<details><summary>Answer</summary>

**$\dfrac{4}{(e^x + e^{-x})^2}$.** Quotient rule: the numerator's derivative is $e^x + e^{-x}$ and the denominator's is $e^x - e^{-x}$, so
$\tanh' x = \frac{(e^x + e^{-x})^2 - (e^x - e^{-x})^2}{(e^x + e^{-x})^2}$. The difference of squares is $(a + b)^2 - (a - b)^2 = 4ab = 4e^x e^{-x} = 4$. This equals $1 - \tanh^2 x$. Common mistakes: $\frac{1}{(e^x + e^{-x})^2}$ (the 4 lost), and $1 - \tanh x$ (the square lost).

</details>

**Problem 16.** $\dfrac{d}{dx}\, \sin(x^2)$. `[expr in x]`

<details><summary>Answer</summary>

**$2x \cos(x^2)$.** Chain rule: outer $\cos$ evaluated at the inner $x^2$, times the inner derivative $2x$. Common mistakes: $\cos(x^2)$ (the inner derivative forgotten) and $2x \cos x$ (the outer derivative evaluated at $x$ instead of at $x^2$).

</details>

**Problem 17.** $\dfrac{d}{dx}\, \sqrt{1 + x^4}$. `[expr in x]`

<details><summary>Answer</summary>

**$\dfrac{2x^3}{\sqrt{1 + x^4}}$.** $\frac{d}{du}\sqrt u = \frac{1}{2\sqrt u}$, times the inner derivative $4x^3$: $\frac{4x^3}{2\sqrt{1 + x^4}}$. A common mistake is $\frac{1}{2\sqrt{1 + x^4}}$, without the inner derivative.

</details>

**Problem 18.** $\dfrac{d}{dx}\, \ln(\cos x)$ for $|x| < \pi/2$. `[expr in x]`

<details><summary>Answer</summary>

**$-\dfrac{\sin x}{\cos x}$**, that is $-\tan x$. Chain rule: $\frac{1}{\cos x} \cdot (-\sin x)$. A common mistake is $\frac{1}{\cos x}$, without the inner derivative.

</details>

**Problem 19.** $\dfrac{d}{dx}\, x^x$ for $x > 0$. (Write $x^x = e^{x \ln x}$.) `[expr in x]`

<details><summary>Answer</summary>

**$x^x(\ln x + 1)$.** $\frac{d}{dx} e^{x \ln x} = e^{x \ln x} \cdot \frac{d}{dx}(x \ln x) = x^x \left(\ln x + x \cdot \frac1x\right)$. A common mistake is $x \cdot x^{x - 1}$: the power rule, which holds only for a constant exponent.

</details>

**Problem 20.** $\dfrac{d}{dx}\, e^{-x^2/2}$, the shape of the normal density. `[expr in x]`

<details><summary>Answer</summary>

**$-x\, e^{-x^2/2}$.** Chain rule: $e^{u}$ with $u = -x^2/2$ and $u' = -x$. This is the $\varphi' = -x\varphi$ that the Gaussian integrals of the [Calculus 2 problem set](../08-calculus-2/03-problem-set.md) use. Common mistakes: $e^{-x^2/2}$ (no inner derivative) and $-\frac{x^2}{2} e^{-x^2/2}$ (the exponent multiplied in instead of its derivative).

</details>

**Problem 21.** SiLU (swish) is $\operatorname{silu}(x) = \dfrac{x}{1 + e^{-x}} = x\,\sigma(x)$. Give its derivative. `[expr in x]`

<details><summary>Answer</summary>

**$\dfrac{1}{1 + e^{-x}} + \dfrac{x\, e^{-x}}{(1 + e^{-x})^2}$**, that is $\sigma(x) + x\,\sigma(x)(1 - \sigma(x))$. Product rule on $x \cdot \sigma(x)$, using Problem 13 for $\sigma'$. This is the backward of SwiGLU's gate in [Gated MLPs: SwiGLU and GeGLU](../../ai-engineering/13-decoder-block/02-gated-mlps.md). A common mistake is to keep only the second term, $x\sigma'(x)$, forgetting that $x$ itself also changes.

</details>

**Problem 22.** $\dfrac{d^2}{dx^2}\, x e^{-x}$. `[expr in x]`

<details><summary>Answer</summary>

**$(x - 2)\, e^{-x}$.** First derivative by the product rule: $e^{-x} - x e^{-x} = (1 - x)e^{-x}$. Again: $-e^{-x} - (1 - x)e^{-x} = (x - 2)e^{-x}$. A common mistake is $(1 - x)e^{-x}$, which is only the first derivative.

</details>

**Problem 23.** $f(x) = (x^2 + 1)^5$. Give $f'(1)$. `[number]`

<details><summary>Answer</summary>

**160.** $f'(x) = 5(x^2 + 1)^4 \cdot 2x$, so $f'(1) = 5 \cdot 2^4 \cdot 2 = 160$. A common mistake is 80, which drops the inner derivative $2x$.

</details>

**Problem 24.** $\dfrac{d}{dx}\, \log_2 x$ for $x > 0$. `[expr in x]`

<details><summary>Answer</summary>

**$\dfrac{1}{x \ln 2}$.** Change of base: $\log_2 x = \frac{\ln x}{\ln 2}$, and $\frac{1}{\ln 2}$ is a constant. Common mistakes: $\frac1x$ (the derivative of $\ln$, not $\log_2$) and $\frac{\ln 2}{x}$ (the constant inverted).

</details>

**Problem 25.** Prove from the limit definition $f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$ that the derivative of $f(x) = x^2$ is $2x$. `[proof]`

<details><summary>Answer</summary>

**Proof (direct, from the definition).** Fix a real $x$. By definition, $f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$ when this limit exists. For $h \ne 0$,
$$\frac{(x + h)^2 - x^2}{h} = \frac{x^2 + 2xh + h^2 - x^2}{h} = \frac{h(2x + h)}{h} = 2x + h .$$
Cancelling $h$ is allowed because the limit as $h \to 0$ only looks at $h \ne 0$, so the difference quotient equals $2x + h$ at every point the limit uses. $2x + h$ is a polynomial in $h$, hence continuous, so $\lim_{h \to 0} (2x + h) = 2x$. Therefore $f'(x) = 2x$, for every real $x$.

The step most often skipped is the justification for cancelling $h$: it is exactly the fact that a limit ignores the point itself (see the review of limits above).

</details>

## Problems on implicit differentiation and related rates

**Problem 26.** The circle $x^2 + y^2 = 25$ passes through $(3, 4)$. Give the slope $dy/dx$ there. `[number]`

<details><summary>Answer</summary>

**$-3/4$.** Differentiate in $x$: $2x + 2y\,y' = 0$, so $y' = -x/y = -3/4$. A common mistake is $-4/3$, which is $dx/dy$.

</details>

**Problem 27.** $x y + y^3 = 2$ defines $y$ implicitly near $(1, 1)$. Give $dy/dx$ as a formula in $x$ and $y$. `[expr in x, y]`

<details><summary>Answer</summary>

**$-\dfrac{y}{x + 3y^2}$.** Differentiate in $x$: the product rule on $xy$ gives $y + x y'$, and the chain rule on $y^3$ gives $3y^2 y'$. So $y + (x + 3y^2) y' = 0$ and $y' = -y/(x + 3y^2)$; at $(1, 1)$ the slope is $-1/4$. Common mistakes: $-y/(3y^2)$ (differentiating $xy$ as $y$ alone, dropping the $x y'$ term of the product rule) and $+y/(x + 3y^2)$ (a sign slip).

</details>

**Problem 28.** A circle's radius grows at 2 cm/s. How fast (in cm²/s) does its area grow when the radius is 5 cm? `[number]`

<details><summary>Answer</summary>

**$20\pi$.** $A = \pi r^2$, so by the chain rule through time $\frac{dA}{dt} = 2\pi r \frac{dr}{dt} = 2\pi \cdot 5 \cdot 2 = 20\pi$. A common mistake is $10\pi$, which forgets $dr/dt = 2$.

</details>

**Problem 29.** A 10 m ladder leans on a wall. Its foot slides away from the wall at 1 m/s. When the foot is 6 m from the wall, give the rate (m/s) at which the top moves; a falling top has a negative rate. `[number]`

<details><summary>Answer</summary>

**$-3/4$.** With foot distance $x$ and top height $y$, $x^2 + y^2 = 100$; at $x = 6$, $y = 8$. Differentiate in $t$: $2x\dot x + 2y\dot y = 0$, so $\dot y = -x\dot x / y = -6 \cdot 1 / 8 = -3/4$. Common mistakes: $3/4$ (the sign dropped: the top falls) and $-4/3$ (the ratio inverted).

</details>

**Problem 30.** A sphere's volume grows at 100 cm³/s. Give $dr/dt$ (cm/s) when the radius is 5 cm. `[number]`

<details><summary>Answer</summary>

**$1/\pi$.** $V = \frac43 \pi r^3$, so $\frac{dV}{dt} = 4\pi r^2 \frac{dr}{dt}$. At $r = 5$: $100 = 4\pi \cdot 25 \cdot \frac{dr}{dt} = 100\pi \frac{dr}{dt}$, so $\frac{dr}{dt} = \frac{1}{\pi}$. A common wrong answer is $\frac{1}{4\pi}$, an arithmetic slip in evaluating $4\pi r^2$.

</details>

**Problem 31.** $e^y = x$ for $x > 0$. Differentiate implicitly and give $dy/dx$ as a formula in $x$ alone. `[expr in x]`

<details><summary>Answer</summary>

**$\dfrac{1}{x}$.** $e^y\, y' = 1$, so $y' = e^{-y} = 1/x$. This is the derivative of $\ln x$ (since $y = \ln x$), derived from that of $e^x$. A common mistake is $x$, which inverts the relation.

</details>

## Problems on optimization

**Problem 32.** $f(x) = x^2 - 6x + 11$. Give the $x$ that minimizes $f$. `[number]`

<details><summary>Answer</summary>

**3.** $f'(x) = 2x - 6 = 0$ at $x = 3$, and $f'' = 2 > 0$ makes it a minimum. Gradient descent, in [Gradient descent and line search](../16-optimization/01-gradient-descent-and-line-search.md), finds the same point numerically. A common mistake is $-3$, a sign slip solving $2x - 6 = 0$.

</details>

**Problem 33.** Give the minimum value of the $f$ of Problem 32. `[number]`

<details><summary>Answer</summary>

**2.** $f(3) = 9 - 18 + 11 = 2$ (or complete the square: $f(x) = (x - 3)^2 + 2$). A common mistake is 11, which is $f(0)$.

</details>

**Problem 34.** A rectangle has perimeter 20. Give its largest possible area. `[number]`

<details><summary>Answer</summary>

**25.** Sides $a$ and $10 - a$, area $A(a) = a(10 - a)$, $A'(a) = 10 - 2a = 0$ at $a = 5$: the square, with area 25. The endpoints $a = 0$ and $a = 10$ give 0. A common mistake is 100, a $10 \times 10$ square, whose perimeter is 40, not 20.

</details>

**Problem 35.** Give the set of critical points of $f(x) = x^3 - 3x$. `[set]`

<details><summary>Answer</summary>

**$\{-1, 1\}$.** $f'(x) = 3x^2 - 3 = 0$ gives $x = \pm 1$. Common mistakes: $\{0, \sqrt 3, -\sqrt 3\}$, which are the zeros of $f$, not of $f'$, and $\{1\}$, which misses the negative root.

</details>

**Problem 36.** Give the maximum of $f(x) = x e^{-x}$ over $x \ge 0$. `[number]`

<details><summary>Answer</summary>

**$e^{-1}$.** $f'(x) = (1 - x)e^{-x} = 0$ at $x = 1$. Candidates: $f(0) = 0$, $f(1) = e^{-1}$, and $f(x) \to 0$ as $x \to \infty$, so the maximum is $e^{-1}$. A common mistake is 1, which is the argmax (where the maximum occurs), not the maximum value.

</details>

**Problem 37.** Give the $c$ that minimizes $L(c) = (c - 1)^2 + (c - 2)^2 + (c - 6)^2$, a one-parameter least-squares fit. `[number]`

<details><summary>Answer</summary>

**3.** $L'(c) = 2(c - 1) + 2(c - 2) + 2(c - 6) = 6c - 18 = 0$ at $c = 3$, the mean of $1, 2, 6$. In general the mean minimizes squared error, which is why a constant predictor trained with MSE learns the mean. A common mistake is 2, the median, which minimizes absolute error instead.

</details>

**Problem 38.** An open box is folded from a $12 \times 12$ sheet by cutting a square of side $s$ from each corner. Give the $s$ that maximizes the volume. `[number]`

<details><summary>Answer</summary>

**2.** $V(s) = s(12 - 2s)^2$ on $0 \le s \le 6$. As in the worked box example, $V'(s) = (12 - 2s)(12 - 6s)$, zero at $s = 6$ and $s = 2$. Candidates: $V(0) = V(6) = 0$ and $V(2) = 2 \cdot 8^2 = 128$, so $s = 2$. Common mistakes: 6, the other root of $V' = 0$, a box of zero volume, and 3.

</details>

**Problem 39.** Give the shortest distance from the point $(0, 2)$ to the parabola $y = x^2$. `[number]`

<details><summary>Answer</summary>

**$\sqrt 7 / 2$.** Minimize the squared distance (same minimizer, simpler algebra): $D(x) = x^2 + (x^2 - 2)^2 = x^4 - 3x^2 + 4$. $D'(x) = 4x^3 - 6x = 2x(2x^2 - 3)$, zero at $x = 0$ and $x^2 = 3/2$. $D(0) = 4$ and $D = \frac94 - \frac92 + 4 = \frac74$ at $x^2 = \frac32$, so the distance is $\sqrt{7/4} = \sqrt 7/2$. Common mistakes: 2, the distance to the vertex $(0, 0)$ (a critical point, but a local maximum of $D$), and $7/4$, the squared distance.

</details>

## Problems on integration and the fundamental theorem

**Problem 40.** $\displaystyle\int_0^1 x^2\, dx$. `[number]`

<details><summary>Answer</summary>

**1/3.** $\left[\frac{x^3}{3}\right]_0^1 = \frac13$. A common mistake is $\frac12$, the integral of $x$.

</details>

**Problem 41.** $\displaystyle\int_0^\pi \sin x\, dx$. `[number]`

<details><summary>Answer</summary>

**2.** An antiderivative of $\sin$ is $-\cos$: $[-\cos x]_0^\pi = -(-1) - (-1) = 2$. A common mistake is 0, a sign slip in $-\cos$ (it is 0 over a full period $[0, 2\pi]$, not over $[0, \pi]$).

</details>

**Problem 42.** $\displaystyle\int_1^e \frac{1}{x}\, dx$. `[number]`

<details><summary>Answer</summary>

**1.** $[\ln x]_1^e = 1 - 0 = 1$. A common mistake is $e - 1$, integrating as if the integrand were 1.

</details>

**Problem 43.** $\displaystyle\int_0^2 (3x^2 - 2x + 1)\, dx$. `[number]`

<details><summary>Answer</summary>

**6.** $[x^3 - x^2 + x]_0^2 = 8 - 4 + 2 = 6$. A common wrong answer is 10, from mishandling the $-2x$ term (dropping it gives $8 + 2 = 10$).

</details>

**Problem 44.** Give the antiderivative $F$ of $2x\cos(x^2)$ with $F(0) = 0$. `[expr in x]`

<details><summary>Answer</summary>

**$\sin(x^2)$.** The integrand is the chain rule result of Problem 16 run backwards: $\frac{d}{dx}\sin(x^2) = 2x\cos(x^2)$, and $\sin(0) = 0$ fixes the constant at 0. Common mistakes: $2x \sin(x^2)$ (keeping the inner factor) and $\sin(x^2) + 1$ (the wrong constant, with $F(0) = 1$).

</details>

**Problem 45.** $\displaystyle\int_0^1 e^{2x}\, dx$. `[number]`

<details><summary>Answer</summary>

**$\dfrac{e^2 - 1}{2}$.** An antiderivative is $\frac12 e^{2x}$ (check: its derivative is $e^{2x}$). $\frac12(e^2 - 1)$. A common mistake is $e^2 - 1$, which forgets the $\frac12$ that undoes the inner derivative 2.

</details>

**Problem 46.** $\dfrac{d}{dx} \displaystyle\int_0^x \sqrt{1 + t^3}\, dt$ for $x \ge 0$. `[expr in x]`

<details><summary>Answer</summary>

**$\sqrt{1 + x^3}$.** FTC part 1: the derivative of an accumulated integral with respect to its upper limit is the integrand at that limit. No antiderivative is needed (and none exists in elementary terms). A common mistake is $\sqrt{1 + x^3} - 1$, subtracting the integrand's value at the lower limit as if evaluating $F(x) - F(0)$.

</details>

**Problem 47.** $\dfrac{d}{dx} \displaystyle\int_0^{x^2} \cos t\, dt$. `[expr in x]`

<details><summary>Answer</summary>

**$2x \cos(x^2)$.** FTC with a variable upper limit $u(x) = x^2$: $f(u(x))\,u'(x) = \cos(x^2) \cdot 2x$. Check: the integral is $\sin(x^2)$, whose derivative is the same. A common mistake is $\cos(x^2)$, forgetting the chain rule through the upper limit.

</details>

**Problem 48.** Give the average value of $\sin x$ over $[0, \pi]$. `[number]`

<details><summary>Answer</summary>

**$2/\pi$.** Average value $= \frac{1}{\pi - 0}\int_0^\pi \sin x\,dx = \frac{2}{\pi}$, using Problem 41. Common mistakes: 2 (the integral, not divided by the length $\pi$) and 0.

</details>

**Problem 49.** Give the area between $y = x$ and $y = x^2$ for $0 \le x \le 1$. `[number]`

<details><summary>Answer</summary>

**1/6.** On $[0, 1]$, $x \ge x^2$, so the area is $\int_0^1 (x - x^2)\,dx = \frac12 - \frac13 = \frac16$. A common mistake is $-\frac16$, subtracting the curves the wrong way round; an area between curves is the upper minus the lower.

</details>

**Problem 50.** $\displaystyle\int_0^1 \frac{1}{1 + x}\, dx$. `[number]`

<details><summary>Answer</summary>

**$\ln 2$.** $[\ln(1 + x)]_0^1 = \ln 2 - \ln 1 = \ln 2$. A common mistake is $\frac12$, the integrand's value at an endpoint.

</details>

**Problem 51.** Give the left Riemann sum of $f(x) = x$ on $[0, 1]$ with $n = 4$ equal pieces. `[number]`

<details><summary>Answer</summary>

**3/8.** Width $\frac14$, left endpoints $0, \frac14, \frac12, \frac34$: $\frac14\left(0 + \frac14 + \frac12 + \frac34\right) = \frac14 \cdot \frac32 = \frac38$. Common mistakes: $\frac58$ (the right sum, endpoints $\frac14, \ldots, 1$) and $\frac12$ (the exact integral). The trapezoid rule of [Definite integrals: trapezoid and Simpson](04-definite-integrals-trapezoid-simpson.md) averages the two sums, $\frac12(\frac38 + \frac58) = \frac12$, exact on this line.

</details>

## Problems on L'Hôpital's rule

**Problem 52.** $\displaystyle\lim_{x \to 0} \frac{e^x - 1 - x}{x^2}$. `[number]`

<details><summary>Answer</summary>

**1/2.** $0/0$; differentiate: $\frac{e^x - 1}{2x}$, still $0/0$; again: $\frac{e^x}{2} \to \frac12$. (Taylor: $e^x - 1 - x = \frac{x^2}{2} + \cdots$.) A common mistake is 1, stopping after one application and evaluating $\frac{e^x - 1}{x}$ instead.

</details>

**Problem 53.** $\displaystyle\lim_{x \to \infty} x^2 e^{-x}$. `[number]`

<details><summary>Answer</summary>

**0.** Write it as $\frac{x^2}{e^x}$, an $\infty/\infty$ form. Twice: $\frac{2x}{e^x}$, then $\frac{2}{e^x} \to 0$. Any power is eventually beaten by the exponential. A common mistake is $\infty$.

</details>

**Problem 54.** $\displaystyle\lim_{x \to 0} \frac{\sin x - x}{x^3}$. `[number]`

<details><summary>Answer</summary>

**$-1/6$.** $0/0$ three times: $\frac{\cos x - 1}{3x^2} \to \frac{-\sin x}{6x} \to \frac{-\cos x}{6} \to -\frac16$. (Taylor: $\sin x - x = -\frac{x^3}{6} + \cdots$.) A common mistake is $+\frac16$, a sign slip.

</details>

**Problem 55.** $\displaystyle\lim_{x \to 0^+} x^x$. `[number]`

<details><summary>Answer</summary>

**1.** A $0^0$ form: take the logarithm. $\ln x^x = x \ln x \to 0$ (Problem 7), so $x^x = e^{x \ln x} \to e^0 = 1$. A common mistake is 0, reading $0^0$ as 0.

</details>

**Problem 56.** $\displaystyle\lim_{x \to 1} \frac{\ln x}{x - 1}$. `[number]`

<details><summary>Answer</summary>

**1.** $0/0$; differentiate: $\frac{1/x}{1} \to 1$. It is also the derivative of $\ln$ at 1 written as a difference quotient. A common mistake is 0.

</details>

**Problem 57.** Show that $\displaystyle\lim_{x \to \infty} \frac{x + \sin x}{x} = 1$, and explain why L'Hôpital's rule cannot be used to get it. `[proof]`

<details><summary>Answer</summary>

**Proof (squeeze).** For $x > 0$, $\frac{x + \sin x}{x} = 1 + \frac{\sin x}{x}$. Since $|\sin x| \le 1$, we have $\left|\frac{\sin x}{x}\right| \le \frac{1}{x}$, and $\frac{1}{x} \to 0$ as $x \to \infty$. By the squeeze theorem ($-\frac1x \le \frac{\sin x}{x} \le \frac1x$), $\frac{\sin x}{x} \to 0$, so the limit is $1 + 0 = 1$.

**Why L'Hôpital does not apply.** The quotient has the form $\infty/\infty$, so one might differentiate: $\frac{(x + \sin x)'}{(x)'} = \frac{1 + \cos x}{1} = 1 + \cos x$. This oscillates between 0 and 2 and has no limit as $x \to \infty$. L'Hôpital's rule says that **if** $\lim f'/g'$ exists then $\lim f/g$ equals it; its hypothesis fails here, so it gives no information, even though $\lim f/g$ exists and equals 1, as proved above.

</details>

## Common mistakes

The worked answers keep returning to the same handful of errors. This table collects them with the problems where each one appears:

| Pitfall | Symptom | How the worked answers avoid it |
|---|---|---|
| Substituting into a $0/0$ form instead of simplifying first | a removable singularity "evaluated" as 0 or undefined | factor or multiply by the conjugate first (Problems 1, 8) |
| Treating $1^\infty$ as 1 | compound growth and $e$ disappear | take the logarithm and find the limit of the exponent (Problem 6) |
| Differentiating a product factor by factor | $\frac{d}{dx} x^3 \sin x$ given as $3x^2 \cos x$ | the product rule, both terms (Problems 11, 21) |
| Forgetting the inner derivative of the chain rule | backward passes off by the inner Jacobian; gradcheck fails | every chain rule answer writes the inner derivative explicitly (Problems 13, 16, 23, 45, 47) |
| Using the power rule on a variable exponent | $x^x$ differentiated as $x \cdot x^{x-1}$ | rewrite as $e^{x \ln x}$ first (Problem 19) |
| Reporting the argmax instead of the max, or keeping a degenerate critical point | the wrong number answered, or a zero-volume box chosen | list every candidate with its value and compare (Problems 36, 38) |
| Dropping the sign of a related rate | a falling ladder reported as rising | keep the sign from $\dot y = -x\dot x/y$ (Problem 29) |
| Applying L'Hôpital once too few times, or where $f'/g'$ has no limit | a wrong finite limit, or no answer to a limit that exists | re-check the form after each application (Problem 52); check that $\lim f'/g'$ exists before using it (Problem 57) |

> **Warning:** The forgotten inner derivative is the mistake that matters most in code. On paper it costs a point; in a backward pass it produces a gradient that is wrong by a factor and still trains, until gradcheck catches it.

## Where this leads

These problems feed directly into code. The limit of Problem 4 is what the finite differences of [Derivatives and finite differences](01-derivatives-and-finite-differences.md) approximate, trading truncation against rounding in the choice of step. The activation derivatives of Problems 13 to 15 and 21 appear in code, checked against PyTorch, in [Activation Functions and Their Derivatives](03-activation-functions.md). [Gradients and gradcheck](../10-calculus-3/01-gradients-and-gradcheck.md) compares an analytic derivative (the rules) with a numerical one (the limits). [Gradient descent and line search](../16-optimization/01-gradient-descent-and-line-search.md) finds the critical points of Problems 32 to 39 numerically. The [Calculus 2 problem set](../08-calculus-2/03-problem-set.md) continues with integration techniques, improper integrals, series, and differential equations.

### Further reading

- OpenStax, *Calculus Volume 1* (free), chapters 2 to 5.
- 3Blue1Brown, *Essence of Calculus*, episodes 1 to 9.

## Key terms

- **limit**: the value $L$ that $f(x)$ approaches as $x \ne a$ approaches $a$, defined by $\varepsilon$ and $\delta$.
- **indeterminate form**: an expression such as $0/0$ or $1^\infty$ that signals a limit must be rewritten.
- **chain rule**: $\frac{d}{dx} f(g(x)) = f'(g(x))\, g'(x)$.
- **critical point**: a point where $f'(x) = 0$, a candidate for an interior optimum.
- **related rates**: rates of change tied together by differentiating a constraint in time.
- **fundamental theorem of calculus**: $\int_a^b f = F(b) - F(a)$ for $F' = f$, and $\frac{d}{dx}\int_a^x f = f(x)$.
- **L'Hôpital's rule**: for $0/0$ or $\infty/\infty$ forms, $\lim f/g = \lim f'/g'$ when the latter exists.

## Summary

- A limit describes what $f(x)$ approaches, not $f(a)$ itself; $0/0$ and $\infty - \infty$ are signals to rewrite, never answers (Problems 1, 8).
- The product, quotient, and chain rules compute every derivative the autograd needs; the sigmoid, softplus, tanh, and SiLU derivatives are four lines each (Problems 13 to 15, 21).
- An optimum of a smooth function sits where $f' = 0$ or on the boundary, and you still have to check which candidate wins (Problems 36, 38).
- The fundamental theorem turns integrals into antiderivatives and makes $\frac{d}{dx}\int_a^{u(x)} f = f(u(x))\,u'(x)$ (Problems 46, 47).
- L'Hôpital's rule turns $0/0$ and $\infty/\infty$ into a limit of derivatives, and applies only when that limit exists (Problem 57).
