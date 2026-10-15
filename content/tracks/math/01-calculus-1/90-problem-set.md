<!-- ol:module S-M01 -->
# Calculus 1 problem set: limits, derivatives, rates, optimization, integrals, L'Hôpital

## Overview

| | |
|---|---|
| **Module** | `S-M01` · solve · none · Pass 2 · 6 to 8 h |
| **You build** | answers in `solve/S-M01.toml` (54 checked by SymPy) and 3 proofs in `solve/S-M01/q10.md`, `q25.md`, `q57.md` (self-graded against their rubrics) |
| **Contract** | none: a pen and paper set |
| **Tests** | `course/solve/S-M01/key.toml` (hidden): typed answers plus reject canaries; the problems are in `course/solve/S-M01/problems.md` and in section 4 |
| **Needs** | no module. Reading: `S-M00` (functions, exponentials, logarithms) and the [Calculus 1 topic](README.md), sections 1 to 5 |
| **Used by** | no call site (a solve set). Take it after `M01.1` (finite differences), `M01.2` (Newton's method), and `M01.3` (activation derivatives) in Pass 2; `M01.4` and `S-M02` build on the integrals |
| **Milestone** | `MS-P2` (the Pass 2 gate runs `ol check` on every solve part of the pass) |
| **Optional depth** | OpenStax, *Calculus Volume 1* (free), ch. 2 to 5; 3Blue1Brown, *Essence of Calculus*, episodes 1 to 9 |

## Key Takeaways

- A limit describes what $f(x)$ approaches, not $f(a)$ itself; $0/0$ and $\infty - \infty$ are signals to rewrite, never answers (q1, q8).
- The product, quotient, and chain rules compute every derivative your autograd needs; the sigmoid, softplus, tanh, and SiLU derivatives are four lines each (q13 to q15, q21).
- An optimum of a smooth function sits where $f' = 0$ or on the boundary, and you still have to check which candidate wins (q36, q38).
- The fundamental theorem turns integrals into antiderivatives and makes $\frac{d}{dx}\int_a^{u(x)} f = f(u(x))\,u'(x)$ (q46, q47).
- L'Hôpital's rule turns $0/0$ and $\infty/\infty$ into a limit of derivatives, and only applies when that limit exists (q57).

## How to work this chapter

```bash
ol start S-M01             # writes solve/S-M01.toml and the three proof files
ol check S-M01             # SymPy checks the answers, then asks each proof rubric (y/n)
ol check S-M01 --regrade   # ask the rubrics again after you change a proof
```

---

## 1. Why now

In Pass 2 your system learns to learn. `M01.1` approximates derivatives with finite differences, `M01.3` writes the derivative of every activation function, and `M04.1` turns those into `gradcheck`, the test that guards every backward pass in `L0`. All of it assumes you can differentiate a formula by hand and know what a limit is, because a derivative is a limit and a finite difference is that limit stopped early. Training picks the weights that minimize a loss, which is optimization; the learning-rate schedules of `M10.4` and the area under an ROC curve (`M07.7`) are integrals. This set checks the pen and paper side of Calculus 1 before your code depends on it.

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

This is a sibling of q13 and q38, not one of the graded problems.

**A chain rule derivative.** Differentiate $f(x) = \ln(1 + e^{-x})$. Outer function $\ln u$ with derivative $1/u$; inner $u = 1 + e^{-x}$ with derivative $-e^{-x}$ (chain rule again, inner $-x$). So $f'(x) = \frac{-e^{-x}}{1 + e^{-x}}$. Multiply top and bottom by $e^{x}$: $f'(x) = \frac{-1}{e^x + 1} = -\sigma(-x)$. This is the derivative of the binary cross-entropy for a positive label, written in terms of the logit. In `solve/` it would be `answer = "-exp(-x)/(1 + exp(-x))"`; `answer = "-1/(exp(x) + 1)"` passes too, because SymPy checks equivalence, and `answer = "1/(1 + exp(-x))"` fails.

**An optimization.** Maximize $g(s) = s(10 - 2s)^2$ for $0 \le s \le 5$ (an open box from a $10 \times 10$ sheet). $g'(s) = (10 - 2s)^2 + s \cdot 2(10 - 2s)(-2) = (10 - 2s)(10 - 2s - 4s) = (10 - 2s)(10 - 6s)$. The critical points are $s = 5$ (a zero-volume box) and $s = 5/3$. Compare candidates: $g(0) = 0$, $g(5) = 0$, $g(5/3) = \frac{5}{3} \cdot \left(\frac{20}{3}\right)^2 = \frac{2000}{27}$. The maximizer is $s = 5/3$.

## 4. The problem set

Write each answer in `solve/S-M01.toml`:

```toml
[q1]
answer = "4"
[q13]
answer = "exp(-z)/(1 + exp(-z))^2"
[q35]
answer = "{-1, 1}"
[q10]
proof = "S-M01/q10.md"
```

Numbers are exact: `exp(2)`, `2/pi`, `sqrt(7)/2`; `0.5` fails where `1/2` is expected. An `[expr]` answer may take any equivalent form. Write $e$ as `E` or `exp(1)` and $\pi$ as `pi`.

<!-- ol:problems S-M01 -->

### Limits

**q1.** $\displaystyle\lim_{x \to 2} \frac{x^2 - 4}{x - 2}$. `[number]`

**q2.** $\displaystyle\lim_{x \to 0} \frac{\sin 3x}{x}$. `[number]`

**q3.** $\displaystyle\lim_{x \to \infty} \frac{3x^2 + 2x}{5x^2 - 1}$. `[number]`

**q4.** $\displaystyle\lim_{h \to 0} \frac{(1 + h)^3 - 1}{h}$. (Which derivative is this?) `[number]`

**q5.** $\displaystyle\lim_{x \to 0} \frac{1 - \cos x}{x^2}$. `[number]`

**q6.** $\displaystyle\lim_{x \to \infty} \left(1 + \frac{2}{x}\right)^{x}$. `[number]`

**q7.** $\displaystyle\lim_{x \to 0^+} x \ln x$. `[number]`

**q8.** $\displaystyle\lim_{x \to \infty} \left(\sqrt{x^2 + x} - x\right)$. `[number]`

**q9.** $\displaystyle\lim_{x \to 0^-} \frac{|x|}{x}$ (from the left). `[number]`

**q10.** Prove from the definition that $\displaystyle\lim_{x \to 3} (2x + 1) = 7$: for every $\varepsilon > 0$, give a $\delta > 0$ such that $0 < |x - 3| < \delta$ implies $|(2x + 1) - 7| < \varepsilon$. `[proof]`

### Derivative rules and the chain rule

**q11.** $\dfrac{d}{dx}\, x^3 \sin x$. `[expr in x]`

**q12.** $\dfrac{d}{dx}\, \dfrac{e^x}{1 + x^2}$. `[expr in x]`

**q13.** The sigmoid is $\sigma(z) = \dfrac{1}{1 + e^{-z}}$. Give $\sigma'(z)$. `[expr in z]`

**q14.** Softplus is $s(x) = \ln(1 + e^x)$. Give $s'(x)$. `[expr in x]`

**q15.** $\tanh x = \dfrac{e^x - e^{-x}}{e^x + e^{-x}}$. Give $\dfrac{d}{dx} \tanh x$ in terms of exponentials. `[expr in x]`

**q16.** $\dfrac{d}{dx}\, \sin(x^2)$. `[expr in x]`

**q17.** $\dfrac{d}{dx}\, \sqrt{1 + x^4}$. `[expr in x]`

**q18.** $\dfrac{d}{dx}\, \ln(\cos x)$ for $|x| < \pi/2$. `[expr in x]`

**q19.** $\dfrac{d}{dx}\, x^x$ for $x > 0$. (Write $x^x = e^{x \ln x}$.) `[expr in x]`

**q20.** $\dfrac{d}{dx}\, e^{-x^2/2}$, the shape of the normal density. `[expr in x]`

**q21.** SiLU (swish) is $\operatorname{silu}(x) = \dfrac{x}{1 + e^{-x}} = x\,\sigma(x)$. Give its derivative. `[expr in x]`

**q22.** $\dfrac{d^2}{dx^2}\, x e^{-x}$. `[expr in x]`

**q23.** $f(x) = (x^2 + 1)^5$. Give $f'(1)$. `[number]`

**q24.** $\dfrac{d}{dx}\, \log_2 x$ for $x > 0$. `[expr in x]`

**q25.** Prove from the limit definition $f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$ that the derivative of $f(x) = x^2$ is $2x$. `[proof]`

### Implicit differentiation and related rates

**q26.** The circle $x^2 + y^2 = 25$ passes through $(3, 4)$. Give the slope $dy/dx$ there. `[number]`

**q27.** $x y + y^3 = 2$ defines $y$ implicitly near $(1, 1)$. Give $dy/dx$ as a formula in $x$ and $y$. `[expr in x, y]`

**q28.** A circle's radius grows at 2 cm/s. How fast (in cm²/s) does its area grow when the radius is 5 cm? `[number]`

**q29.** A 10 m ladder leans on a wall. Its foot slides away from the wall at 1 m/s. When the foot is 6 m from the wall, give the rate (m/s) at which the top moves; a falling top has a negative rate. `[number]`

**q30.** A sphere's volume grows at 100 cm³/s. Give $dr/dt$ (cm/s) when the radius is 5 cm. `[number]`

**q31.** $e^y = x$ for $x > 0$. Differentiate implicitly and give $dy/dx$ as a formula in $x$ alone. `[expr in x]`

### Optimization

**q32.** $f(x) = x^2 - 6x + 11$. Give the $x$ that minimizes $f$. `[number]`

**q33.** Give the minimum value of the $f$ of q32. `[number]`

**q34.** A rectangle has perimeter 20. Give its largest possible area. `[number]`

**q35.** Give the set of critical points of $f(x) = x^3 - 3x$. `[set]`

**q36.** Give the maximum of $f(x) = x e^{-x}$ over $x \ge 0$. `[number]`

**q37.** Give the $c$ that minimizes $L(c) = (c - 1)^2 + (c - 2)^2 + (c - 6)^2$, a one-parameter least-squares fit. `[number]`

**q38.** An open box is folded from a $12 \times 12$ sheet by cutting a square of side $s$ from each corner. Give the $s$ that maximizes the volume. `[number]`

**q39.** Give the shortest distance from the point $(0, 2)$ to the parabola $y = x^2$. `[number]`

### Integration and the fundamental theorem

**q40.** $\displaystyle\int_0^1 x^2\, dx$. `[number]`

**q41.** $\displaystyle\int_0^\pi \sin x\, dx$. `[number]`

**q42.** $\displaystyle\int_1^e \frac{1}{x}\, dx$. `[number]`

**q43.** $\displaystyle\int_0^2 (3x^2 - 2x + 1)\, dx$. `[number]`

**q44.** Give the antiderivative $F$ of $2x\cos(x^2)$ with $F(0) = 0$. `[expr in x]`

**q45.** $\displaystyle\int_0^1 e^{2x}\, dx$. `[number]`

**q46.** $\dfrac{d}{dx} \displaystyle\int_0^x \sqrt{1 + t^3}\, dt$ for $x \ge 0$. `[expr in x]`

**q47.** $\dfrac{d}{dx} \displaystyle\int_0^{x^2} \cos t\, dt$. `[expr in x]`

**q48.** Give the average value of $\sin x$ over $[0, \pi]$. `[number]`

**q49.** Give the area between $y = x$ and $y = x^2$ for $0 \le x \le 1$. `[number]`

**q50.** $\displaystyle\int_0^1 \frac{1}{1 + x}\, dx$. `[number]`

**q51.** Give the left Riemann sum of $f(x) = x$ on $[0, 1]$ with $n = 4$ equal pieces. `[number]`

### L'Hôpital's rule

**q52.** $\displaystyle\lim_{x \to 0} \frac{e^x - 1 - x}{x^2}$. `[number]`

**q53.** $\displaystyle\lim_{x \to \infty} x^2 e^{-x}$. `[number]`

**q54.** $\displaystyle\lim_{x \to 0} \frac{\sin x - x}{x^3}$. `[number]`

**q55.** $\displaystyle\lim_{x \to 0^+} x^x$. `[number]`

**q56.** $\displaystyle\lim_{x \to 1} \frac{\ln x}{x - 1}$. `[number]`

**q57.** Show that $\displaystyle\lim_{x \to \infty} \frac{x + \sin x}{x} = 1$, and explain why L'Hôpital's rule cannot be used to get it. `[proof]`

<!-- /ol:problems -->

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Substituting into a $0/0$ form instead of simplifying first | a removable singularity "evaluated" as 0 or undefined | q1 (canary 0), q8 (canary 0) |
| Treating $1^\infty$ as 1 | compound growth and $e$ disappear | q6 (canary 1) |
| Differentiating a product factor by factor | $\frac{d}{dx} x^3 \sin x$ given as $3x^2 \cos x$ | q11, q21 (canaries) |
| Forgetting the inner derivative of the chain rule | backward passes off by the inner Jacobian; gradcheck fails | q13, q16, q23, q45, q47 (canaries) |
| Using the power rule on a variable exponent | $x^x$ differentiated as $x \cdot x^{x-1}$ | q19 (canary) |
| Reporting the argmax instead of the max, or keeping a degenerate critical point | the wrong number answered, or a zero-volume box chosen | q36 (canary 1), q38 (canary 6) |
| Dropping the sign of a related rate | a falling ladder reported as rising | q29 (canary 3/4) |
| Applying L'Hôpital once too few times, or where $f'/g'$ has no limit | a wrong finite limit, or no answer to a limit that exists | q52 (canary 1), q57 (proof) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `S-M00` | functions, exponentials, logarithms, and the trigonometry these problems differentiate |
| Forward | `M01.1` | finite differences approximate the limit of q4; the step size trades truncation against rounding |
| Forward | `M01.3` | the activation derivatives of q13 to q15 and q21, in code, checked against torch |
| Forward | `M04.1` | gradcheck compares an analytic derivative (your rules) with a numerical one (your limits) |
| Forward | `M10.1` | gradient descent finds the critical points of q32 to q39 numerically |
| Forward | `S-M02` | integration techniques, improper integrals, series, and ODEs |
