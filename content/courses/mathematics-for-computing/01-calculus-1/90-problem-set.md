# Calculus 1 problem set: limits, derivatives, rates, optimization, integrals, L'Hôpital

## Overview

| | |
|---|---|
| **Module** | `S-M01` · solve · none · Pass 2 · 6 to 8 h |
| **Solution** | worked answers to all 57 problems in section 4: 54 exact answers with derivations and common wrong answers, and 3 written proofs (q10, q25, q57) |
| **Contract** | none: a pen and paper set |
| **Needs** | no module. Reading: `S-M00` (functions, exponentials, logarithms) and the [Calculus 1 topic](README.md), sections 1 to 5 |
| **Used by** | no call site (a solve set). Take it after `M01.1` (finite differences), `M01.2` (Newton's method), and `M01.3` (activation derivatives) in Pass 2; `M01.4` and `S-M02` build on the integrals |
| **Optional depth** | OpenStax, *Calculus Volume 1* (free), ch. 2 to 5; 3Blue1Brown, *Essence of Calculus*, episodes 1 to 9 |

## Key Takeaways

- A limit describes what $f(x)$ approaches, not $f(a)$ itself; $0/0$ and $\infty - \infty$ are signals to rewrite, never answers (q1, q8).
- The product, quotient, and chain rules compute every derivative the autograd needs; the sigmoid, softplus, tanh, and SiLU derivatives are four lines each (q13 to q15, q21).
- An optimum of a smooth function sits where $f' = 0$ or on the boundary, and you still have to check which candidate wins (q36, q38).
- The fundamental theorem turns integrals into antiderivatives and makes $\frac{d}{dx}\int_a^{u(x)} f = f(u(x))\,u'(x)$ (q46, q47).
- L'Hôpital's rule turns $0/0$ and $\infty/\infty$ into a limit of derivatives, and only applies when that limit exists (q57).

## How to read this chapter

Section 2 reviews the rules the problems use, and section 3 works two sibling problems by hand. Section 4 lists all 57 problems, each followed by a collapsed worked answer: the result, the derivation, and the most common wrong answer with the reason it is wrong. Attempt each problem before opening its answer; the wrong answers are as instructive as the right ones.

---

## 1. Why now

In Pass 2 the system learns to learn. `M01.1` approximates derivatives with finite differences, `M01.3` writes the derivative of every activation function, and `M04.1` turns those into `gradcheck`, the test that guards every backward pass in `L0`. All of it assumes the reader can differentiate a formula by hand and know what a limit is, because a derivative is a limit and a finite difference is that limit stopped early. Training picks the weights that minimize a loss, which is optimization; the learning-rate schedules of `M10.4` and the area under an ROC curve (`M07.7`) are integrals. This set works through the pen and paper side of Calculus 1 that the code depends on.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $\lim_{x \to a} f(x) = L$ | $f(x)$ gets arbitrarily close to $L$ as $x \ne a$ gets close to $a$ | real |
| $\varepsilon, \delta$ | positive tolerances in the definition of a limit | positive reals |
| $f'(x)$, $\frac{df}{dx}$ | the derivative, $\lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$ | function |
| $f''(x)$ | the second derivative, the derivative of $f'$ | function |
| $\sigma(z)$ | the sigmoid, $1/(1 + e^{-z})$ | function |
| $F$ | an antiderivative of $f$: $F' = f$ | function |
| $\int_a^b f(x)\,dx$ | the definite integral, the signed area under $f$ from $a$ to $b$ | real |
| $t$ | time, in related-rate problems | real |

### 2.1 Limits

$\lim_{x \to a} f(x) = L$ means: for every $\varepsilon > 0$ there is a $\delta > 0$ such that $0 < |x - a| < \delta$ implies $|f(x) - L| < \varepsilon$. The value $f(a)$ plays no part, which is why $\frac{x^2 - 4}{x - 2}$ has a limit at 2 even though it is undefined there: for $x \ne 2$ it equals $x + 2$. Limits add, multiply, and divide (when the denominator's limit is not 0). Three standard limits do most of the work: $\frac{\sin u}{u} \to 1$ as $u \to 0$, $(1 + \frac{a}{x})^x \to e^a$ as $x \to \infty$, and for rational functions at infinity, only the highest powers matter. A **one-sided** limit restricts $x$ to one side of $a$. Forms like $0/0$, $\infty/\infty$, $\infty - \infty$, $0 \cdot \infty$, $1^\infty$, and $0^0$ are **indeterminate**: they say the problem needs rewriting (factor, multiply by a conjugate, take logarithms), not what the answer is.

### 2.2 Derivative rules

The derivative is the limit of the difference quotient. From that definition follow the rules you use instead of it: linearity; the **power rule** $\frac{d}{dx} x^n = n x^{n-1}$ for a constant $n$; $\frac{d}{dx} e^x = e^x$, $\frac{d}{dx} \ln x = \frac{1}{x}$, $\frac{d}{dx} \sin x = \cos x$, $\frac{d}{dx} \cos x = -\sin x$; the **product rule** $(fg)' = f'g + fg'$; the **quotient rule** $(f/g)' = (f'g - fg')/g^2$; and the **chain rule**, $\frac{d}{dx} f(g(x)) = f'(g(x))\, g'(x)$: the outer derivative evaluated at the inner function, times the inner derivative. The chain rule is the whole of backpropagation (`M08.1` to `M08.3`). A variable exponent, as in $x^x$, needs $x^x = e^{x \ln x}$ first; the power rule does not apply.

### 2.3 Implicit differentiation and related rates

When $y$ is defined by an equation such as $x^2 + y^2 = 25$, differentiate both sides with respect to $x$, treating $y$ as a function of $x$ (so $\frac{d}{dx} y^2 = 2y\,y'$), and solve for $y'$. **Related rates** apply the same idea with time: if two quantities are tied by an equation and both change with $t$, differentiating the equation in $t$ ties their rates.

### 2.4 Optimization

At an interior minimum or maximum of a differentiable $f$, $f'(x) = 0$; such points are **critical points**. A global optimum on an interval is a critical point or an endpoint (or a limit at an open end), so list the candidates and compare their values. The second derivative classifies a critical point: $f'' > 0$ is a local minimum, $f'' < 0$ a local maximum. Gradient descent (`M10.1`) finds the same points numerically when solving $f' = 0$ by hand is impossible.

### 2.5 Integrals and the fundamental theorem

The definite integral $\int_a^b f(x)\,dx$ is the limit of **Riemann sums** $\sum_i f(x_i)\,\Delta x$ over finer and finer partitions. The **fundamental theorem of calculus** (FTC) connects it to derivatives in two ways: if $F' = f$ then $\int_a^b f = F(b) - F(a)$; and $\frac{d}{dx} \int_a^x f(t)\,dt = f(x)$ for continuous $f$. With a variable upper limit $u(x)$, the chain rule adds a factor: $\frac{d}{dx} \int_a^{u(x)} f(t)\,dt = f(u(x))\,u'(x)$. The **average value** of $f$ on $[a, b]$ is $\frac{1}{b - a}\int_a^b f$.

### 2.6 L'Hôpital's rule

If $f(x) \to 0$ and $g(x) \to 0$ (or both $\to \pm\infty$) as $x \to a$, and $\lim f'(x)/g'(x)$ **exists**, then $\lim f(x)/g(x)$ equals it. It may need several applications ($\frac{e^x - 1 - x}{x^2}$ takes two). Products $0 \cdot \infty$ become quotients, and powers $0^0$ or $1^\infty$ become products after taking the logarithm. When the limit of $f'/g'$ does not exist, the rule says nothing, and the original limit may still exist (q57).

## 3. Worked example by hand

This is a sibling of q13 and q38, not one of the problems of section 4.

**A chain rule derivative.** Differentiate $f(x) = \ln(1 + e^{-x})$. Outer function $\ln u$ with derivative $1/u$; inner $u = 1 + e^{-x}$ with derivative $-e^{-x}$ (chain rule again, inner $-x$). So $f'(x) = \frac{-e^{-x}}{1 + e^{-x}}$. Multiply top and bottom by $e^{x}$: $f'(x) = \frac{-1}{e^x + 1} = -\sigma(-x)$. This is the derivative of the binary cross-entropy for a positive label, written in terms of the logit. Both $\frac{-e^{-x}}{1 + e^{-x}}$ and $\frac{-1}{e^x + 1}$ are the same answer written two ways; $\frac{1}{1 + e^{-x}}$, the sigmoid itself, is not (it has the wrong sign and the wrong argument).

**An optimization.** Maximize $g(s) = s(10 - 2s)^2$ for $0 \le s \le 5$ (an open box from a $10 \times 10$ sheet). $g'(s) = (10 - 2s)^2 + s \cdot 2(10 - 2s)(-2) = (10 - 2s)(10 - 2s - 4s) = (10 - 2s)(10 - 6s)$. The critical points are $s = 5$ (a zero-volume box) and $s = 5/3$. Compare candidates: $g(0) = 0$, $g(5) = 0$, $g(5/3) = \frac{5}{3} \cdot \left(\frac{20}{3}\right)^2 = \frac{2000}{27}$. The maximizer is $s = 5/3$.

## 4. The problem set, with worked answers

Each problem is followed by a collapsed answer: the result, the derivation, and, where there is one, the most common wrong answer and why it is wrong. Numbers are exact ($e^2$, $2/\pi$, $\sqrt 7/2$, not decimals), and an expression answer may take any equivalent form. The tag after each problem says what kind of answer it wants.

### Limits

**q1.** $\displaystyle\lim_{x \to 2} \frac{x^2 - 4}{x - 2}$. `[number]`

<details><summary>Answer</summary>

**4.** Substituting gives $0/0$, a signal to rewrite. Factor: $\frac{(x - 2)(x + 2)}{x - 2} = x + 2$ for $x \ne 2$, and the limit never looks at $x = 2$, so it is $2 + 2 = 4$. A common mistake is 0, from substituting into the numerator only.

</details>

**q2.** $\displaystyle\lim_{x \to 0} \frac{\sin 3x}{x}$. `[number]`

<details><summary>Answer</summary>

**3.** Write $\frac{\sin 3x}{x} = 3 \cdot \frac{\sin 3x}{3x}$ and use $\frac{\sin u}{u} \to 1$ with $u = 3x \to 0$. A common mistake is 1, applying the standard limit without matching its argument, which drops the factor 3.

</details>

**q3.** $\displaystyle\lim_{x \to \infty} \frac{3x^2 + 2x}{5x^2 - 1}$. `[number]`

<details><summary>Answer</summary>

**3/5.** Divide top and bottom by $x^2$: $\frac{3 + 2/x}{5 - 1/x^2} \to \frac{3}{5}$. For rational functions at infinity only the highest powers matter. A common mistake is 0, treating the denominator as "winning" when the degrees are equal.

</details>

**q4.** $\displaystyle\lim_{h \to 0} \frac{(1 + h)^3 - 1}{h}$. (Which derivative is this?) `[number]`

<details><summary>Answer</summary>

**3.** It is the difference quotient of $f(x) = x^3$ at $x = 1$, so the limit is $f'(1) = 3 \cdot 1^2 = 3$. Directly: $(1 + h)^3 - 1 = 3h + 3h^2 + h^3$, and dividing by $h$ gives $3 + 3h + h^2 \to 3$. `M01.1`'s finite differences approximate exactly this limit with a fixed $h$. A common mistake is 1.

</details>

**q5.** $\displaystyle\lim_{x \to 0} \frac{1 - \cos x}{x^2}$. `[number]`

<details><summary>Answer</summary>

**1/2.** Multiply top and bottom by $1 + \cos x$: $\frac{1 - \cos^2 x}{x^2(1 + \cos x)} = \left(\frac{\sin x}{x}\right)^2 \cdot \frac{1}{1 + \cos x} \to 1 \cdot \frac12$. (Or L'Hôpital twice: $\frac{\sin x}{2x} \to \frac12$.) A common mistake is 0, reading $1 - \cos x \to 0$ as the whole answer.

</details>

**q6.** $\displaystyle\lim_{x \to \infty} \left(1 + \frac{2}{x}\right)^{x}$. `[number]`

<details><summary>Answer</summary>

**$e^2$.** The standard limit $(1 + \frac{a}{x})^x \to e^a$ with $a = 2$. To see it, take logs: $x \ln(1 + 2/x) = \frac{\ln(1 + 2/x)}{1/x}$, a $0/0$ form whose limit is 2 (L'Hôpital, or $\ln(1 + u) \approx u$), so the limit is $e^2$. Two common mistakes: 1, reading $1^\infty$ as 1 (it is indeterminate), and $e$, dropping the 2.

</details>

**q7.** $\displaystyle\lim_{x \to 0^+} x \ln x$. `[number]`

<details><summary>Answer</summary>

**0.** A $0 \cdot (-\infty)$ form. Rewrite as $\frac{\ln x}{1/x}$, an $\infty/\infty$ form, and apply L'Hôpital: $\frac{1/x}{-1/x^2} = -x \to 0$. The factor $x$ goes to 0 faster than $\ln x$ goes to $-\infty$. A common mistake is $-1$.

</details>

**q8.** $\displaystyle\lim_{x \to \infty} \left(\sqrt{x^2 + x} - x\right)$. `[number]`

<details><summary>Answer</summary>

**1/2.** An $\infty - \infty$ form. Multiply by the conjugate: $\frac{(x^2 + x) - x^2}{\sqrt{x^2 + x} + x} = \frac{x}{\sqrt{x^2 + x} + x} = \frac{1}{\sqrt{1 + 1/x} + 1} \to \frac{1}{2}$. A common mistake is 0, reading $\infty - \infty$ as 0.

</details>

**q9.** $\displaystyle\lim_{x \to 0^-} \frac{|x|}{x}$ (from the left). `[number]`

<details><summary>Answer</summary>

**$-1$.** For $x < 0$, $|x| = -x$, so the quotient is $-1$ at every point the left-hand limit uses. A common mistake is 1, which is the right-hand limit; because the two one-sided limits differ, the two-sided limit does not exist.

</details>

**q10.** Prove from the definition that $\displaystyle\lim_{x \to 3} (2x + 1) = 7$: for every $\varepsilon > 0$, give a $\delta > 0$ such that $0 < |x - 3| < \delta$ implies $|(2x + 1) - 7| < \varepsilon$. `[proof]`

<details><summary>Answer</summary>

**Proof (direct).** Let $\varepsilon > 0$ be given. For every real $x$, $|(2x + 1) - 7| = |2x - 6| = 2|x - 3|$. Choose $\delta = \varepsilon / 2$, which is positive. If $0 < |x - 3| < \delta$, then $|(2x + 1) - 7| = 2|x - 3| < 2\delta = \varepsilon$. Since $\varepsilon > 0$ was arbitrary, the definition of the limit holds, so $\lim_{x \to 3} (2x + 1) = 7$.

A complete proof states the goal, rewrites the distance as $2|x - 3|$, names $\delta$ explicitly (any positive $\delta \le \varepsilon/2$ works), and shows that this $\delta$ works for every such $x$.

</details>

### Derivative rules and the chain rule

**q11.** $\dfrac{d}{dx}\, x^3 \sin x$. `[expr in x]`

<details><summary>Answer</summary>

**$3x^2 \sin x + x^3 \cos x$.** Product rule, $(fg)' = f'g + fg'$, with $f = x^3$ and $g = \sin x$. A common mistake is $3x^2 \cos x$: differentiating each factor and multiplying, which is not a rule.

</details>

**q12.** $\dfrac{d}{dx}\, \dfrac{e^x}{1 + x^2}$. `[expr in x]`

<details><summary>Answer</summary>

**$\dfrac{e^x (1 - 2x + x^2)}{(1 + x^2)^2}$**, which is $\dfrac{e^x (x - 1)^2}{(1 + x^2)^2}$. Quotient rule: $\frac{e^x (1 + x^2) - e^x \cdot 2x}{(1 + x^2)^2}$, then factor out $e^x$. Common mistakes: $+2x$ in the numerator (the quotient rule's minus sign flipped), and $e^x / (2x)$ (differentiating top and bottom separately).

</details>

**q13.** The sigmoid is $\sigma(z) = \dfrac{1}{1 + e^{-z}}$. Give $\sigma'(z)$. `[expr in z]`

<details><summary>Answer</summary>

**$\dfrac{e^{-z}}{(1 + e^{-z})^2}$**, equal to $\sigma(z)(1 - \sigma(z))$. Chain rule on $(1 + e^{-z})^{-1}$: the outer derivative is $-(1 + e^{-z})^{-2}$, the inner derivative is $-e^{-z}$, and the two minus signs cancel. This is the sigmoid backward of `L0.2` and `M01.3`. Common mistakes: a leading minus sign (the inner derivative's sign lost), and $\sigma(z)$ itself.

</details>

**q14.** Softplus is $s(x) = \ln(1 + e^x)$. Give $s'(x)$. `[expr in x]`

<details><summary>Answer</summary>

**$\dfrac{e^x}{1 + e^x}$**, which is $\sigma(x)$. Chain rule: $\frac{1}{1 + e^x} \cdot e^x$. A common mistake is $\frac{1}{1 + e^x}$, which forgets the inner derivative $e^x$ and is $\sigma(-x)$.

</details>

**q15.** $\tanh x = \dfrac{e^x - e^{-x}}{e^x + e^{-x}}$. Give $\dfrac{d}{dx} \tanh x$ in terms of exponentials. `[expr in x]`

<details><summary>Answer</summary>

**$\dfrac{4}{(e^x + e^{-x})^2}$.** Quotient rule: the numerator's derivative is $e^x + e^{-x}$ and the denominator's is $e^x - e^{-x}$, so
$\tanh' x = \frac{(e^x + e^{-x})^2 - (e^x - e^{-x})^2}{(e^x + e^{-x})^2}$. The difference of squares is $(a + b)^2 - (a - b)^2 = 4ab = 4e^x e^{-x} = 4$. This equals $1 - \tanh^2 x$. Common mistakes: $\frac{1}{(e^x + e^{-x})^2}$ (the 4 lost), and $1 - \tanh x$ (the square lost).

</details>

**q16.** $\dfrac{d}{dx}\, \sin(x^2)$. `[expr in x]`

<details><summary>Answer</summary>

**$2x \cos(x^2)$.** Chain rule: outer $\cos$ evaluated at the inner $x^2$, times the inner derivative $2x$. Common mistakes: $\cos(x^2)$ (the inner derivative forgotten) and $2x \cos x$ (the outer derivative evaluated at $x$ instead of at $x^2$).

</details>

**q17.** $\dfrac{d}{dx}\, \sqrt{1 + x^4}$. `[expr in x]`

<details><summary>Answer</summary>

**$\dfrac{2x^3}{\sqrt{1 + x^4}}$.** $\frac{d}{du}\sqrt u = \frac{1}{2\sqrt u}$, times the inner derivative $4x^3$: $\frac{4x^3}{2\sqrt{1 + x^4}}$. A common mistake is $\frac{1}{2\sqrt{1 + x^4}}$, without the inner derivative.

</details>

**q18.** $\dfrac{d}{dx}\, \ln(\cos x)$ for $|x| < \pi/2$. `[expr in x]`

<details><summary>Answer</summary>

**$-\dfrac{\sin x}{\cos x}$**, that is $-\tan x$. Chain rule: $\frac{1}{\cos x} \cdot (-\sin x)$. A common mistake is $\frac{1}{\cos x}$, without the inner derivative.

</details>

**q19.** $\dfrac{d}{dx}\, x^x$ for $x > 0$. (Write $x^x = e^{x \ln x}$.) `[expr in x]`

<details><summary>Answer</summary>

**$x^x(\ln x + 1)$.** $\frac{d}{dx} e^{x \ln x} = e^{x \ln x} \cdot \frac{d}{dx}(x \ln x) = x^x \left(\ln x + x \cdot \frac1x\right)$. A common mistake is $x \cdot x^{x - 1}$: the power rule, which holds only for a constant exponent.

</details>

**q20.** $\dfrac{d}{dx}\, e^{-x^2/2}$, the shape of the normal density. `[expr in x]`

<details><summary>Answer</summary>

**$-x\, e^{-x^2/2}$.** Chain rule: $e^{u}$ with $u = -x^2/2$ and $u' = -x$. This is the $\varphi' = -x\varphi$ that Gaussian integrals in `S-M02` use. Common mistakes: $e^{-x^2/2}$ (no inner derivative) and $-\frac{x^2}{2} e^{-x^2/2}$ (the exponent multiplied in instead of its derivative).

</details>

**q21.** SiLU (swish) is $\operatorname{silu}(x) = \dfrac{x}{1 + e^{-x}} = x\,\sigma(x)$. Give its derivative. `[expr in x]`

<details><summary>Answer</summary>

**$\dfrac{1}{1 + e^{-x}} + \dfrac{x\, e^{-x}}{(1 + e^{-x})^2}$**, that is $\sigma(x) + x\,\sigma(x)(1 - \sigma(x))$. Product rule on $x \cdot \sigma(x)$, using q13 for $\sigma'$. This is the backward of SwiGLU's gate (`L7.2`). A common mistake is to keep only the second term, $x\sigma'(x)$, forgetting that $x$ itself also changes.

</details>

**q22.** $\dfrac{d^2}{dx^2}\, x e^{-x}$. `[expr in x]`

<details><summary>Answer</summary>

**$(x - 2)\, e^{-x}$.** First derivative by the product rule: $e^{-x} - x e^{-x} = (1 - x)e^{-x}$. Again: $-e^{-x} - (1 - x)e^{-x} = (x - 2)e^{-x}$. A common mistake is $(1 - x)e^{-x}$, which is only the first derivative.

</details>

**q23.** $f(x) = (x^2 + 1)^5$. Give $f'(1)$. `[number]`

<details><summary>Answer</summary>

**160.** $f'(x) = 5(x^2 + 1)^4 \cdot 2x$, so $f'(1) = 5 \cdot 2^4 \cdot 2 = 160$. A common mistake is 80, which drops the inner derivative $2x$.

</details>

**q24.** $\dfrac{d}{dx}\, \log_2 x$ for $x > 0$. `[expr in x]`

<details><summary>Answer</summary>

**$\dfrac{1}{x \ln 2}$.** Change of base: $\log_2 x = \frac{\ln x}{\ln 2}$, and $\frac{1}{\ln 2}$ is a constant. Common mistakes: $\frac1x$ (the derivative of $\ln$, not $\log_2$) and $\frac{\ln 2}{x}$ (the constant inverted).

</details>

**q25.** Prove from the limit definition $f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$ that the derivative of $f(x) = x^2$ is $2x$. `[proof]`

<details><summary>Answer</summary>

**Proof (direct, from the definition).** Fix a real $x$. By definition, $f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$ when this limit exists. For $h \ne 0$,
$$\frac{(x + h)^2 - x^2}{h} = \frac{x^2 + 2xh + h^2 - x^2}{h} = \frac{h(2x + h)}{h} = 2x + h .$$
Cancelling $h$ is allowed because the limit as $h \to 0$ only looks at $h \ne 0$, so the difference quotient equals $2x + h$ at every point the limit uses. $2x + h$ is a polynomial in $h$, hence continuous, so $\lim_{h \to 0} (2x + h) = 2x$. Therefore $f'(x) = 2x$, for every real $x$.

The step most often skipped is the justification for cancelling $h$: it is exactly the fact that a limit ignores the point itself (section 2.1).

</details>

### Implicit differentiation and related rates

**q26.** The circle $x^2 + y^2 = 25$ passes through $(3, 4)$. Give the slope $dy/dx$ there. `[number]`

<details><summary>Answer</summary>

**$-3/4$.** Differentiate in $x$: $2x + 2y\,y' = 0$, so $y' = -x/y = -3/4$. A common mistake is $-4/3$, which is $dx/dy$.

</details>

**q27.** $x y + y^3 = 2$ defines $y$ implicitly near $(1, 1)$. Give $dy/dx$ as a formula in $x$ and $y$. `[expr in x, y]`

<details><summary>Answer</summary>

**$-\dfrac{y}{x + 3y^2}$.** Differentiate in $x$: the product rule on $xy$ gives $y + x y'$, and the chain rule on $y^3$ gives $3y^2 y'$. So $y + (x + 3y^2) y' = 0$ and $y' = -y/(x + 3y^2)$; at $(1, 1)$ the slope is $-1/4$. Common mistakes: $-y/(3y^2)$ (differentiating $xy$ as $y$ alone, dropping the $x y'$ term of the product rule) and $+y/(x + 3y^2)$ (a sign slip).

</details>

**q28.** A circle's radius grows at 2 cm/s. How fast (in cm²/s) does its area grow when the radius is 5 cm? `[number]`

<details><summary>Answer</summary>

**$20\pi$.** $A = \pi r^2$, so by the chain rule through time $\frac{dA}{dt} = 2\pi r \frac{dr}{dt} = 2\pi \cdot 5 \cdot 2 = 20\pi$. A common mistake is $10\pi$, which forgets $dr/dt = 2$.

</details>

**q29.** A 10 m ladder leans on a wall. Its foot slides away from the wall at 1 m/s. When the foot is 6 m from the wall, give the rate (m/s) at which the top moves; a falling top has a negative rate. `[number]`

<details><summary>Answer</summary>

**$-3/4$.** With foot distance $x$ and top height $y$, $x^2 + y^2 = 100$; at $x = 6$, $y = 8$. Differentiate in $t$: $2x\dot x + 2y\dot y = 0$, so $\dot y = -x\dot x / y = -6 \cdot 1 / 8 = -3/4$. Common mistakes: $3/4$ (the sign dropped: the top falls) and $-4/3$ (the ratio inverted).

</details>

**q30.** A sphere's volume grows at 100 cm³/s. Give $dr/dt$ (cm/s) when the radius is 5 cm. `[number]`

<details><summary>Answer</summary>

**$1/\pi$.** $V = \frac43 \pi r^3$, so $\frac{dV}{dt} = 4\pi r^2 \frac{dr}{dt}$. At $r = 5$: $100 = 4\pi \cdot 25 \cdot \frac{dr}{dt} = 100\pi \frac{dr}{dt}$, so $\frac{dr}{dt} = \frac{1}{\pi}$. A common wrong answer is $\frac{1}{4\pi}$, an arithmetic slip in evaluating $4\pi r^2$.

</details>

**q31.** $e^y = x$ for $x > 0$. Differentiate implicitly and give $dy/dx$ as a formula in $x$ alone. `[expr in x]`

<details><summary>Answer</summary>

**$\dfrac{1}{x}$.** $e^y\, y' = 1$, so $y' = e^{-y} = 1/x$. This is the derivative of $\ln x$ (since $y = \ln x$), derived from that of $e^x$. A common mistake is $x$, which inverts the relation.

</details>

### Optimization

**q32.** $f(x) = x^2 - 6x + 11$. Give the $x$ that minimizes $f$. `[number]`

<details><summary>Answer</summary>

**3.** $f'(x) = 2x - 6 = 0$ at $x = 3$, and $f'' = 2 > 0$ makes it a minimum. Gradient descent (`M10.1`) finds the same point numerically. A common mistake is $-3$, a sign slip solving $2x - 6 = 0$.

</details>

**q33.** Give the minimum value of the $f$ of q32. `[number]`

<details><summary>Answer</summary>

**2.** $f(3) = 9 - 18 + 11 = 2$ (or complete the square: $f(x) = (x - 3)^2 + 2$). A common mistake is 11, which is $f(0)$.

</details>

**q34.** A rectangle has perimeter 20. Give its largest possible area. `[number]`

<details><summary>Answer</summary>

**25.** Sides $a$ and $10 - a$, area $A(a) = a(10 - a)$, $A'(a) = 10 - 2a = 0$ at $a = 5$: the square, with area 25. The endpoints $a = 0$ and $a = 10$ give 0. A common mistake is 100, a $10 \times 10$ square, whose perimeter is 40, not 20.

</details>

**q35.** Give the set of critical points of $f(x) = x^3 - 3x$. `[set]`

<details><summary>Answer</summary>

**$\{-1, 1\}$.** $f'(x) = 3x^2 - 3 = 0$ gives $x = \pm 1$. Common mistakes: $\{0, \sqrt 3, -\sqrt 3\}$, which are the zeros of $f$, not of $f'$, and $\{1\}$, which misses the negative root.

</details>

**q36.** Give the maximum of $f(x) = x e^{-x}$ over $x \ge 0$. `[number]`

<details><summary>Answer</summary>

**$e^{-1}$.** $f'(x) = (1 - x)e^{-x} = 0$ at $x = 1$. Candidates: $f(0) = 0$, $f(1) = e^{-1}$, and $f(x) \to 0$ as $x \to \infty$, so the maximum is $e^{-1}$. A common mistake is 1, which is the argmax (where the maximum occurs), not the maximum value.

</details>

**q37.** Give the $c$ that minimizes $L(c) = (c - 1)^2 + (c - 2)^2 + (c - 6)^2$, a one-parameter least-squares fit. `[number]`

<details><summary>Answer</summary>

**3.** $L'(c) = 2(c - 1) + 2(c - 2) + 2(c - 6) = 6c - 18 = 0$ at $c = 3$, the mean of $1, 2, 6$. In general the mean minimizes squared error, which is why a constant predictor trained with MSE learns the mean. A common mistake is 2, the median, which minimizes absolute error instead.

</details>

**q38.** An open box is folded from a $12 \times 12$ sheet by cutting a square of side $s$ from each corner. Give the $s$ that maximizes the volume. `[number]`

<details><summary>Answer</summary>

**2.** $V(s) = s(12 - 2s)^2$ on $0 \le s \le 6$. As in section 3, $V'(s) = (12 - 2s)(12 - 6s)$, zero at $s = 6$ and $s = 2$. Candidates: $V(0) = V(6) = 0$ and $V(2) = 2 \cdot 8^2 = 128$, so $s = 2$. Common mistakes: 6, the other root of $V' = 0$, a box of zero volume, and 3.

</details>

**q39.** Give the shortest distance from the point $(0, 2)$ to the parabola $y = x^2$. `[number]`

<details><summary>Answer</summary>

**$\sqrt 7 / 2$.** Minimize the squared distance (same minimizer, simpler algebra): $D(x) = x^2 + (x^2 - 2)^2 = x^4 - 3x^2 + 4$. $D'(x) = 4x^3 - 6x = 2x(2x^2 - 3)$, zero at $x = 0$ and $x^2 = 3/2$. $D(0) = 4$ and $D = \frac94 - \frac92 + 4 = \frac74$ at $x^2 = \frac32$, so the distance is $\sqrt{7/4} = \sqrt 7/2$. Common mistakes: 2, the distance to the vertex $(0, 0)$ (a critical point, but a local maximum of $D$), and $7/4$, the squared distance.

</details>

### Integration and the fundamental theorem

**q40.** $\displaystyle\int_0^1 x^2\, dx$. `[number]`

<details><summary>Answer</summary>

**1/3.** $\left[\frac{x^3}{3}\right]_0^1 = \frac13$. A common mistake is $\frac12$, the integral of $x$.

</details>

**q41.** $\displaystyle\int_0^\pi \sin x\, dx$. `[number]`

<details><summary>Answer</summary>

**2.** An antiderivative of $\sin$ is $-\cos$: $[-\cos x]_0^\pi = -(-1) - (-1) = 2$. A common mistake is 0, a sign slip in $-\cos$ (it is 0 over a full period $[0, 2\pi]$, not over $[0, \pi]$).

</details>

**q42.** $\displaystyle\int_1^e \frac{1}{x}\, dx$. `[number]`

<details><summary>Answer</summary>

**1.** $[\ln x]_1^e = 1 - 0 = 1$. A common mistake is $e - 1$, integrating as if the integrand were 1.

</details>

**q43.** $\displaystyle\int_0^2 (3x^2 - 2x + 1)\, dx$. `[number]`

<details><summary>Answer</summary>

**6.** $[x^3 - x^2 + x]_0^2 = 8 - 4 + 2 = 6$. A common wrong answer is 10, from mishandling the $-2x$ term (dropping it gives $8 + 2 = 10$).

</details>

**q44.** Give the antiderivative $F$ of $2x\cos(x^2)$ with $F(0) = 0$. `[expr in x]`

<details><summary>Answer</summary>

**$\sin(x^2)$.** The integrand is the chain rule result of q16 run backwards: $\frac{d}{dx}\sin(x^2) = 2x\cos(x^2)$, and $\sin(0) = 0$ fixes the constant at 0. Common mistakes: $2x \sin(x^2)$ (keeping the inner factor) and $\sin(x^2) + 1$ (the wrong constant, with $F(0) = 1$).

</details>

**q45.** $\displaystyle\int_0^1 e^{2x}\, dx$. `[number]`

<details><summary>Answer</summary>

**$\dfrac{e^2 - 1}{2}$.** An antiderivative is $\frac12 e^{2x}$ (check: its derivative is $e^{2x}$). $\frac12(e^2 - 1)$. A common mistake is $e^2 - 1$, which forgets the $\frac12$ that undoes the inner derivative 2.

</details>

**q46.** $\dfrac{d}{dx} \displaystyle\int_0^x \sqrt{1 + t^3}\, dt$ for $x \ge 0$. `[expr in x]`

<details><summary>Answer</summary>

**$\sqrt{1 + x^3}$.** FTC part 1: the derivative of an accumulated integral with respect to its upper limit is the integrand at that limit. No antiderivative is needed (and none exists in elementary terms). A common mistake is $\sqrt{1 + x^3} - 1$, subtracting the integrand's value at the lower limit as if evaluating $F(x) - F(0)$.

</details>

**q47.** $\dfrac{d}{dx} \displaystyle\int_0^{x^2} \cos t\, dt$. `[expr in x]`

<details><summary>Answer</summary>

**$2x \cos(x^2)$.** FTC with a variable upper limit $u(x) = x^2$: $f(u(x))\,u'(x) = \cos(x^2) \cdot 2x$. Check: the integral is $\sin(x^2)$, whose derivative is the same. A common mistake is $\cos(x^2)$, forgetting the chain rule through the upper limit.

</details>

**q48.** Give the average value of $\sin x$ over $[0, \pi]$. `[number]`

<details><summary>Answer</summary>

**$2/\pi$.** Average value $= \frac{1}{\pi - 0}\int_0^\pi \sin x\,dx = \frac{2}{\pi}$, using q41. Common mistakes: 2 (the integral, not divided by the length $\pi$) and 0.

</details>

**q49.** Give the area between $y = x$ and $y = x^2$ for $0 \le x \le 1$. `[number]`

<details><summary>Answer</summary>

**1/6.** On $[0, 1]$, $x \ge x^2$, so the area is $\int_0^1 (x - x^2)\,dx = \frac12 - \frac13 = \frac16$. A common mistake is $-\frac16$, subtracting the curves the wrong way round; an area between curves is the upper minus the lower.

</details>

**q50.** $\displaystyle\int_0^1 \frac{1}{1 + x}\, dx$. `[number]`

<details><summary>Answer</summary>

**$\ln 2$.** $[\ln(1 + x)]_0^1 = \ln 2 - \ln 1 = \ln 2$. A common mistake is $\frac12$, the integrand's value at an endpoint.

</details>

**q51.** Give the left Riemann sum of $f(x) = x$ on $[0, 1]$ with $n = 4$ equal pieces. `[number]`

<details><summary>Answer</summary>

**3/8.** Width $\frac14$, left endpoints $0, \frac14, \frac12, \frac34$: $\frac14\left(0 + \frac14 + \frac12 + \frac34\right) = \frac14 \cdot \frac32 = \frac38$. Common mistakes: $\frac58$ (the right sum, endpoints $\frac14, \ldots, 1$) and $\frac12$ (the exact integral). The trapezoid rule of `M01.4` averages the two sums, $\frac12(\frac38 + \frac58) = \frac12$, exact on this line.

</details>

### L'Hôpital's rule

**q52.** $\displaystyle\lim_{x \to 0} \frac{e^x - 1 - x}{x^2}$. `[number]`

<details><summary>Answer</summary>

**1/2.** $0/0$; differentiate: $\frac{e^x - 1}{2x}$, still $0/0$; again: $\frac{e^x}{2} \to \frac12$. (Taylor: $e^x - 1 - x = \frac{x^2}{2} + \cdots$.) A common mistake is 1, stopping after one application and evaluating $\frac{e^x - 1}{x}$ instead.

</details>

**q53.** $\displaystyle\lim_{x \to \infty} x^2 e^{-x}$. `[number]`

<details><summary>Answer</summary>

**0.** Write it as $\frac{x^2}{e^x}$, an $\infty/\infty$ form. Twice: $\frac{2x}{e^x}$, then $\frac{2}{e^x} \to 0$. Any power is eventually beaten by the exponential. A common mistake is $\infty$.

</details>

**q54.** $\displaystyle\lim_{x \to 0} \frac{\sin x - x}{x^3}$. `[number]`

<details><summary>Answer</summary>

**$-1/6$.** $0/0$ three times: $\frac{\cos x - 1}{3x^2} \to \frac{-\sin x}{6x} \to \frac{-\cos x}{6} \to -\frac16$. (Taylor: $\sin x - x = -\frac{x^3}{6} + \cdots$.) A common mistake is $+\frac16$, a sign slip.

</details>

**q55.** $\displaystyle\lim_{x \to 0^+} x^x$. `[number]`

<details><summary>Answer</summary>

**1.** A $0^0$ form: take the logarithm. $\ln x^x = x \ln x \to 0$ (q7), so $x^x = e^{x \ln x} \to e^0 = 1$. A common mistake is 0, reading $0^0$ as 0.

</details>

**q56.** $\displaystyle\lim_{x \to 1} \frac{\ln x}{x - 1}$. `[number]`

<details><summary>Answer</summary>

**1.** $0/0$; differentiate: $\frac{1/x}{1} \to 1$. It is also the derivative of $\ln$ at 1 written as a difference quotient. A common mistake is 0.

</details>

**q57.** Show that $\displaystyle\lim_{x \to \infty} \frac{x + \sin x}{x} = 1$, and explain why L'Hôpital's rule cannot be used to get it. `[proof]`

<details><summary>Answer</summary>

**Proof (squeeze).** For $x > 0$, $\frac{x + \sin x}{x} = 1 + \frac{\sin x}{x}$. Since $|\sin x| \le 1$, we have $\left|\frac{\sin x}{x}\right| \le \frac{1}{x}$, and $\frac{1}{x} \to 0$ as $x \to \infty$. By the squeeze theorem ($-\frac1x \le \frac{\sin x}{x} \le \frac1x$), $\frac{\sin x}{x} \to 0$, so the limit is $1 + 0 = 1$.

**Why L'Hôpital does not apply.** The quotient has the form $\infty/\infty$, so one might differentiate: $\frac{(x + \sin x)'}{(x)'} = \frac{1 + \cos x}{1} = 1 + \cos x$. This oscillates between 0 and 2 and has no limit as $x \to \infty$. L'Hôpital's rule says that **if** $\lim f'/g'$ exists then $\lim f/g$ equals it; its hypothesis fails here, so it gives no information, even though $\lim f/g$ exists and equals 1, as proved above.

</details>

## 5. Pitfalls

| Pitfall | Symptom | How the worked answers avoid it |
|---|---|---|
| Substituting into a $0/0$ form instead of simplifying first | a removable singularity "evaluated" as 0 or undefined | factor or multiply by the conjugate first (q1, q8) |
| Treating $1^\infty$ as 1 | compound growth and $e$ disappear | take the logarithm and find the limit of the exponent (q6) |
| Differentiating a product factor by factor | $\frac{d}{dx} x^3 \sin x$ given as $3x^2 \cos x$ | the product rule, both terms (q11, q21) |
| Forgetting the inner derivative of the chain rule | backward passes off by the inner Jacobian; gradcheck fails | every chain rule answer writes the inner derivative explicitly (q13, q16, q23, q45, q47) |
| Using the power rule on a variable exponent | $x^x$ differentiated as $x \cdot x^{x-1}$ | rewrite as $e^{x \ln x}$ first (q19) |
| Reporting the argmax instead of the max, or keeping a degenerate critical point | the wrong number answered, or a zero-volume box chosen | list every candidate with its value and compare (q36, q38) |
| Dropping the sign of a related rate | a falling ladder reported as rising | keep the sign from $\dot y = -x\dot x/y$ (q29) |
| Applying L'Hôpital once too few times, or where $f'/g'$ has no limit | a wrong finite limit, or no answer to a limit that exists | re-check the form after each application (q52); check that $\lim f'/g'$ exists before using it (q57) |
## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `S-M00` | functions, exponentials, logarithms, and the trigonometry these problems differentiate |
| Forward | `M01.1` | finite differences approximate the limit of q4; the step size trades truncation against rounding |
| Forward | `M01.3` | the activation derivatives of q13 to q15 and q21, in code, checked against torch |
| Forward | `M04.1` | gradcheck compares an analytic derivative (the rules) with a numerical one (the limits) |
| Forward | `M10.1` | gradient descent finds the critical points of q32 to q39 numerically |
| Forward | `S-M02` | integration techniques, improper integrals, series, and ODEs |
