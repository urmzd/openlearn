# Calculus 2 problem set: integration techniques, the Gaussian, series, Taylor, polar, ODEs

## Overview

| | |
|---|---|
| **Module** | `S-M02` · solve · none · Pass 2 · 6 to 8 h |
| **Solution** | a worked answer after every problem in section 4: 48 exact answers and 4 proofs (q12, q18, q30, q40; model proofs also in [`course/solutions/solve/S-M02/`](../../../../course/solutions/solve/S-M02/)) |
| **Contract** | none: a pen and paper set |
| **Needs** | no module. Reading: `S-M01` (derivatives and the fundamental theorem) and the [Calculus 2 topic](README.md) |
| **Used by** | no call site (a solve set). Take it after `M02.1` (Taylor series in code) and `M02.2` (the EMA as a geometric series) in Pass 2; `M07.0` and `M07.3` read its Gaussian integrals, which are the solve-only `M02.3` |
| **Optional depth** | OpenStax, *Calculus Volume 2* (free), ch. 3 to 7; Trefethen, *Approximation Theory and Approximation Practice*, ch. 1 to 3, for why a few Taylor terms after range reduction are enough |

## Key Takeaways

- Integration by parts is the product rule integrated; substitution is the chain rule integrated (q1 to q11, q12).
- The Gaussian integral $\int e^{-x^2} = \sqrt{\pi}$ gives the normal density its constant, and half the second moment of a standard normal sits on each side of 0, which is the ReLU factor in Kaiming initialization (q16, q17).
- A series converges when its partial sums do; terms going to 0 is necessary, not sufficient (q20, q30).
- A Taylor polynomial plus a Lagrange remainder is an approximation with a guarantee; range reduction keeps the remainder small (q35, q38).
- A first-order ODE describes a rate; the logistic equation's solution is the sigmoid, and Euler's method is one step of gradient descent on gradient flow (q48, q51, q52).

## How to read this chapter

Sections 1 to 3 review the techniques and work two examples by hand. Section 4 is the problem set: every problem is followed by a collapsed worked answer with the final result, the derivation, and the common wrong answers and why they are wrong. Try each problem before opening its answer.

---

## 1. Why now

The Pass 2 code leans on this calculus in three places. `M02.1` computes $e^x$ and $\operatorname{erf}$ from Taylor polynomials, and its tests check a Lagrange remainder bound; `M09.6` later puts the same polynomial in C. `M02.2` treats the exponential moving average as a geometric series and derives Adam's bias correction (`M10.3`) from its partial sum. And `M07.0` and `M07.3` draw normal random numbers and pick initialization scales from integrals of the normal density, which only exist because $\int e^{-x^2}$ converges to $\sqrt{\pi}$. This set practises the hand techniques behind those modules: integrating by parts and by substitution, improper integrals, convergence, Taylor remainders, curves, and the simplest differential equations.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $u, v$ | functions in integration by parts | functions |
| $\int_a^\infty f$ | improper integral, $\lim_{b \to \infty} \int_a^b f$ | real or divergent |
| $\varphi(z)$ | standard normal density, $e^{-z^2/2}/\sqrt{2\pi}$ | function |
| $\sum_{n} a_n$, $S_N$ | a series and its partial sum $S_N = \sum_{n \le N} a_n$ | real |
| $R$ | radius of convergence of a power series $\sum c_n x^n$ | nonnegative real |
| $T_n(x)$ | degree $n$ Taylor polynomial at 0, $\sum_{k \le n} f^{(k)}(0)\, x^k / k!$ | polynomial |
| $R_n(x)$ | remainder $f(x) - T_n(x)$ | real |
| $r, \theta$ | polar coordinates: $x = r\cos\theta$, $y = r\sin\theta$ | reals |
| $y(t)$, $y'$ | an unknown function of time and its derivative | function |
| $h$ | the step size of Euler's method | positive real |

### 2.1 Integration techniques

**Substitution** reverses the chain rule: $\int f(g(x))\,g'(x)\,dx = \int f(u)\,du$ with $u = g(x)$; change the limits with it. **Integration by parts** reverses the product rule: $\int_a^b u v' = [uv]_a^b - \int_a^b u' v$; pick $u$ to get simpler when differentiated ($x$, $\ln x$). **Partial fractions** split a rational function into simple pieces: $\frac{1}{x^2 - 1} = \frac{1}{2}\left(\frac{1}{x - 1} - \frac{1}{x + 1}\right)$. **Trigonometric identities** such as $\sin^2 x = \frac{1 - \cos 2x}{2}$ and substitutions such as $x = 2\sin\theta$ remove square roots.

### 2.2 Improper integrals and the Gaussian

An integral over an infinite range, or of a function that blows up at an endpoint, is defined as a limit, and it **converges** when the limit is finite: $\int_1^\infty x^{-2} = 1$ but $\int_1^\infty x^{-1} = \infty$; $\int_0^1 x^{-1/2} = 2$ although the integrand is unbounded. The **Gaussian integral** $I = \int_{-\infty}^\infty e^{-x^2} dx$ has no elementary antiderivative, but $I^2$ is a double integral over the plane, and in polar coordinates ($dx\,dy = r\,dr\,d\theta$) it becomes $\int_0^{2\pi}\int_0^\infty e^{-r^2} r\,dr\,d\theta = \pi$. Substituting $x = z/\sqrt{2}$ gives $\int e^{-z^2/2} dz = \sqrt{2\pi}$, the constant of the normal density. By symmetry, $\int_0^\infty z^2 \varphi(z)\,dz$ is half of $\int_{-\infty}^\infty z^2 \varphi(z)\,dz = 1$.

### 2.3 Series and convergence

A series $\sum a_n$ **converges** to $S$ when its partial sums $S_N \to S$. Tests: a **geometric** series $\sum_{n \ge 0} r^n = \frac{1}{1 - r}$ for $|r| < 1$; the **$p$-series** $\sum n^{-p}$ converges exactly for $p > 1$ (compare with $\int_1^\infty x^{-p}$, the **integral test**); the **ratio test** gives convergence when $|a_{n+1}/a_n| \to L < 1$; an **alternating** series with terms decreasing to 0 converges. A **power series** $\sum c_n x^n$ converges for $|x| < R$ and diverges for $|x| > R$; each endpoint $x = \pm R$ needs its own test.

### 2.4 Taylor polynomials and the remainder

$T_n(x) = \sum_{k=0}^{n} \frac{f^{(k)}(0)}{k!} x^k$ matches $f$ and its first $n$ derivatives at 0. **Lagrange's form** of the remainder says $f(x) - T_n(x) = \frac{f^{(n+1)}(c)}{(n+1)!} x^{n+1}$ for some $c$ between 0 and $x$, so a bound on $f^{(n+1)}$ bounds the error. The error grows like $|x|^{n+1}$, so implementations shrink $x$ first: **range reduction** writes $x = k\ln 2 + r$ with $|r| \le \frac{1}{2}\ln 2$ and computes $e^x = 2^k e^r$, where a degree 6 polynomial in $r$ already reaches float32 precision (`M02.1`, `M09.6`).

### 2.5 Parametric curves and polar coordinates

A curve $x(t), y(t)$ has slope $\frac{dy}{dx} = \frac{y'(t)}{x'(t)}$ and arc length $\int \sqrt{x'(t)^2 + y'(t)^2}\,dt$. In polar coordinates the area swept by $r(\theta)$ is $\int \frac{1}{2} r^2\,d\theta$, and multiplying an equation by $r$ converts it with $r^2 = x^2 + y^2$, $r\cos\theta = x$, $r\sin\theta = y$.

### 2.6 First-order differential equations

$y' = -k y$ has the solution $y(t) = y(0)\,e^{-kt}$ (separate variables: $dy/y = -k\,dt$). A **linear** equation $y' + p\,y = q(t)$ is solved with the integrating factor $e^{\int p}$. The **logistic** equation $y' = y(1 - y)$ separates by partial fractions and gives the sigmoid. **Euler's method** steps $y_{k+1} = y_k + h\,f(t_k, y_k)$. Gradient descent with learning rate $h$ is exactly Euler's method on the **gradient flow** $x' = -\nabla f(x)$, and `M02.4` shows momentum is Euler on the heavy-ball equation.

## 3. Worked example by hand

These are siblings of q1 and q17, not problems of the set.

**By parts, twice.** Compute $\int_0^1 x^2 e^{-x}\,dx$. Take $u = x^2$, $v' = e^{-x}$, so $u' = 2x$, $v = -e^{-x}$: the integral is $[-x^2 e^{-x}]_0^1 + 2\int_0^1 x e^{-x}\,dx = -e^{-1} + 2\int_0^1 x e^{-x}\,dx$. Again with $u = x$: $\int_0^1 x e^{-x} = [-x e^{-x}]_0^1 + \int_0^1 e^{-x} = -e^{-1} + (1 - e^{-1}) = 1 - 2e^{-1}$. Total: $-e^{-1} + 2 - 4e^{-1} = 2 - 5e^{-1} \approx 0.1606$. Keep it exact as $2 - 5e^{-1}$; the decimal is only an approximation.

**A Gaussian moment.** $\mathbb{E}[|Z|] = 2\int_0^\infty z\,\varphi(z)\,dz$ for a standard normal $Z$. Substitute $u = z^2/2$, $du = z\,dz$: $\int_0^\infty z e^{-z^2/2} dz = \int_0^\infty e^{-u} du = 1$. So $\mathbb{E}[|Z|] = \frac{2}{\sqrt{2\pi}} = \sqrt{2/\pi} \approx 0.798$.

## 4. The problem set, with worked answers

Answers are exact (for example $\frac12\ln\frac32$, not 0.2027). The tag after each problem says what kind of answer is expected.

### Integration techniques

**q1.** $\displaystyle\int_0^1 x e^x\, dx$ (by parts). `[number]`

<details><summary>Answer</summary>

**$1$.** By parts with $u = x$, $v' = e^x$: $\int_0^1 x e^x\,dx = [x e^x]_0^1 - \int_0^1 e^x\,dx = e - (e - 1) = 1$. A common mistake is $e - 1$, which keeps only $\int e^x$ and drops the boundary term $[xe^x]$.

</details>

**q2.** $\displaystyle\int_1^e \ln x\, dx$. `[number]`

<details><summary>Answer</summary>

**$1$.** By parts with $u = \ln x$, $v' = 1$: $\int_1^e \ln x\,dx = [x\ln x]_1^e - \int_1^e 1\,dx = e - (e - 1) = 1$. Answering $e$ forgets to subtract $\int_1^e 1\,dx$.

</details>

**q3.** $\displaystyle\int_0^{\pi/2} x \cos x\, dx$. `[number]`

<details><summary>Answer</summary>

**$\pi/2 - 1$.** By parts with $u = x$, $v' = \cos x$: $[x\sin x]_0^{\pi/2} - \int_0^{\pi/2}\sin x\,dx = \frac{\pi}{2} - 1$, since $\int_0^{\pi/2}\sin x\,dx = [-\cos x]_0^{\pi/2} = 1$. A common mistake is $\pi/2 + 1$, from taking the antiderivative of $\sin$ as $+\cos$.

</details>

**q4.** $\displaystyle\int_0^1 \frac{x}{1 + x^2}\, dx$ (substitution). `[number]`

<details><summary>Answer</summary>

**$\frac12\ln 2$.** With $u = 1 + x^2$, $du = 2x\,dx$, so $x\,dx = \frac12 du$ and the limits become 1 and 2: $\frac12\int_1^2 \frac{du}{u} = \frac12\ln 2$. Answering $\ln 2$ loses the $\frac12$ from $du = 2x\,dx$.

</details>

**q5.** $\displaystyle\int_0^{\pi/2} \sin^2 x\, dx$. `[number]`

<details><summary>Answer</summary>

**$\pi/4$.** Use $\sin^2 x = \frac{1 - \cos 2x}{2}$: $\int_0^{\pi/2} \frac{1 - \cos 2x}{2}\,dx = \left[\frac{x}{2} - \frac{\sin 2x}{4}\right]_0^{\pi/2} = \frac{\pi}{4}$. Answering $\pi/2$ forgets the halving in the identity.

</details>

**q6.** $\displaystyle\int_2^3 \frac{1}{x^2 - 1}\, dx$ (partial fractions). `[number]`

<details><summary>Answer</summary>

**$\frac12\ln\frac32$.** Partial fractions: $\frac{1}{x^2 - 1} = \frac12\left(\frac{1}{x - 1} - \frac{1}{x + 1}\right)$. Then $\frac12[\ln(x - 1) - \ln(x + 1)]_2^3 = \frac12[(\ln 2 - \ln 4) - (\ln 1 - \ln 3)] = \frac12\ln\frac{2 \cdot 3}{4} = \frac12\ln\frac32$. Common mistakes: $\ln\frac32$ (the $\frac12$ of the decomposition lost) and $\frac12\ln 2$ (an evaluation slip at the limits).

</details>

**q7.** Give the antiderivative $F$ of $x^2 e^x$ with $F(0) = 0$. `[expr in x]`

<details><summary>Answer</summary>

**$F(x) = e^x(x^2 - 2x + 2) - 2$.** By parts twice: $\int x^2 e^x = x^2 e^x - 2\int x e^x = x^2 e^x - 2(xe^x - e^x) = e^x(x^2 - 2x + 2) + C$. The condition $F(0) = 2 + C = 0$ gives $C = -2$. A common mistake is to stop at $e^x(x^2 - 2x + 2)$, which has $F(0) = 2$; $x^3 e^x/3$ treats $e^x$ as a constant.

</details>

**q8.** $\displaystyle\int_0^1 x \sqrt{1 - x^2}\, dx$. `[number]`

<details><summary>Answer</summary>

**$1/3$.** With $u = 1 - x^2$, $du = -2x\,dx$: $\int_0^1 x\sqrt{1 - x^2}\,dx = \frac12\int_0^1 \sqrt{u}\,du = \frac12 \cdot \frac23 = \frac13$ (the sign of $du$ is absorbed by swapping the limits $1 \to 0$). Answering $2/3$ loses the $\frac12$.

</details>

**q9.** $\displaystyle\int_0^\pi x \sin x\, dx$. `[number]`

<details><summary>Answer</summary>

**$\pi$.** By parts with $u = x$, $v' = \sin x$: $[-x\cos x]_0^\pi + \int_0^\pi \cos x\,dx = \pi + 0 = \pi$. Answering 0 drops the boundary term $[-x\cos x]_0^\pi = \pi$.

</details>

**q10.** $\displaystyle\int_0^1 x^3 e^{x^2}\, dx$. `[number]`

<details><summary>Answer</summary>

**$1/2$.** With $u = x^2$, $du = 2x\,dx$, write $x^3\,dx = x^2 \cdot x\,dx = \frac12 u\,du$. Then $\frac12\int_0^1 u e^u\,du = \frac12[(u - 1)e^u]_0^1 = \frac12(0 - (-1)) = \frac12$. Answering 1 loses the $\frac12$.

</details>

**q11.** $\displaystyle\int_0^1 \frac{1}{\sqrt{4 - x^2}}\, dx$ (substitute $x = 2\sin\theta$). `[number]`

<details><summary>Answer</summary>

**$\pi/6$.** With $x = 2\sin\theta$, $dx = 2\cos\theta\,d\theta$ and $\sqrt{4 - x^2} = 2\cos\theta$, so the integrand becomes $d\theta$. The limits are $\theta = 0$ and $\theta = \arcsin\frac12 = \frac{\pi}{6}$. Answering $\pi/3$ uses $\arccos\frac12$ for the upper limit.

</details>

**q12.** Derive the integration by parts formula $\int_a^b u\,v'\,dx = \big[u v\big]_a^b - \int_a^b u'\,v\,dx$ from the product rule and the fundamental theorem of calculus. `[proof]`

<details><summary>Answer</summary>

**Claim.** If $u$ and $v$ are differentiable on $[a, b]$ with $u'v$ and $uv'$ continuous (hence integrable) there, then $\int_a^b u\,v'\,dx = [uv]_a^b - \int_a^b u'\,v\,dx$.

**Proof.** By the product rule, $(uv)' = u'v + uv'$ on $[a, b]$. Integrate both sides from $a$ to $b$. The left side is the integral of the derivative of $uv$, so by the fundamental theorem of calculus it equals $u(b)v(b) - u(a)v(a) = [uv]_a^b$. The right side splits by linearity: $\int_a^b u'v\,dx + \int_a^b uv'\,dx$, both of which exist by the continuity assumption. So $[uv]_a^b = \int_a^b u'v\,dx + \int_a^b uv'\,dx$, and subtracting $\int_a^b u'v\,dx$ gives the claim.

A complete proof states the product rule, applies the fundamental theorem to the left side, uses linearity to split the right side, and names the integrability assumption. Model proof also in [`q12.md`](../../../../course/solutions/solve/S-M02/q12.md).

</details>

### Improper integrals and the Gaussian

**q13.** $\displaystyle\int_1^\infty \frac{1}{x^2}\, dx$. `[number]`

<details><summary>Answer</summary>

**$1$.** $\int_1^b x^{-2}\,dx = 1 - \frac1b \to 1$. It converges because $p = 2 > 1$; answering $\infty$ confuses it with $\int_1^\infty x^{-1}$.

</details>

**q14.** $\displaystyle\int_0^\infty e^{-2x}\, dx$. `[number]`

<details><summary>Answer</summary>

**$1/2$.** $\int_0^b e^{-2x}\,dx = \frac{1 - e^{-2b}}{2} \to \frac12$. Answering 2 multiplies by the rate instead of dividing.

</details>

**q15.** $\displaystyle\int_0^1 \frac{1}{\sqrt{x}}\, dx$ (improper at 0). `[number]`

<details><summary>Answer</summary>

**$2$.** $\int_\delta^1 x^{-1/2}\,dx = 2 - 2\sqrt\delta \to 2$ as $\delta \to 0^+$. The integrand is unbounded at 0, but the integral converges ($p = \frac12 < 1$ near 0). Calling it $\infty$ is the common mistake.

</details>

**q16.** $\displaystyle\int_{-\infty}^{\infty} e^{-x^2}\, dx$. `[number]`

<details><summary>Answer</summary>

**$\sqrt\pi$.** This is the Gaussian integral of 2.2 and q18: $I^2 = \int_0^{2\pi}\int_0^\infty e^{-r^2} r\,dr\,d\theta = \pi$. It matters because the normalizer of the normal density (`M07.0`) is this integral rescaled. Common mistakes: $\sqrt{2\pi}$, which is $\int e^{-x^2/2}$, and $\pi$, which is $I^2$.

</details>

**q17.** $Z$ is a standard normal with density $\varphi(z) = e^{-z^2/2}/\sqrt{2\pi}$. Give $\mathbb{E}[\max(Z, 0)^2] = \displaystyle\int_0^\infty z^2 \varphi(z)\, dz$, the second moment of a ReLU of a standard normal. `[number]`

<details><summary>Answer</summary>

**$1/2$.** $z^2\varphi(z)$ is even, so the half-line integral is half of the full one, and $\int_{-\infty}^\infty z^2\varphi(z)\,dz = \operatorname{Var}(Z) = 1$. So $\mathbb{E}[\max(Z, 0)^2] = \frac12$. This is the factor Kaiming initialization (`M07.3`) undoes with gain $\sqrt 2$: a ReLU keeps half the second moment. Common mistakes: 1 (the full second moment) and $1/\sqrt{2\pi}$ (the value $\int_0^\infty z\varphi(z)\,dz$, a first moment).

</details>

**q18.** Prove $\displaystyle\int_{-\infty}^{\infty} e^{-x^2}\, dx = \sqrt{\pi}$ by squaring the integral and changing to polar coordinates. `[proof]`

<details><summary>Answer</summary>

**Claim.** $I = \int_{-\infty}^\infty e^{-x^2}\,dx = \sqrt\pi$.

**Proof.** First, $I$ converges: $0 < e^{-x^2} \le e^{1 - 2|x|}$ for all $x$ (since $x^2 \ge 2|x| - 1$), and $\int e^{1 - 2|x|}\,dx$ is finite. Because the integral converges absolutely, the product of two copies is a double integral over the plane (Fubini):
$$I^2 = \int_{-\infty}^\infty e^{-x^2}\,dx \int_{-\infty}^\infty e^{-y^2}\,dy = \iint_{\mathbb{R}^2} e^{-(x^2 + y^2)}\,dx\,dy .$$
Change to polar coordinates $x = r\cos\theta$, $y = r\sin\theta$. The Jacobian determinant is $r$, so $dx\,dy = r\,dr\,d\theta$, and $x^2 + y^2 = r^2$:
$$I^2 = \int_0^{2\pi}\int_0^\infty e^{-r^2}\,r\,dr\,d\theta .$$
The inner integral, with $s = r^2$, $ds = 2r\,dr$, is $\frac12\int_0^\infty e^{-s}\,ds = \frac12$. The outer integral of $\frac12$ over $[0, 2\pi)$ is $\pi$. So $I^2 = \pi$, and since the integrand is positive, $I > 0$ and $I = \sqrt\pi$.

The steps a proof must contain: the double integral (with convergence), the area element $r\,dr\,d\theta$, the evaluation of both integrals, and the sign argument for taking the positive root. Model proof also in [`q18.md`](../../../../course/solutions/solve/S-M02/q18.md).

</details>

### Series and convergence tests

**q19.** Does $\displaystyle\sum_{n=1}^\infty \frac{1}{n^2}$ converge? `[bool]`

<details><summary>Answer</summary>

**Yes.** It is a $p$-series with $p = 2 > 1$; by the integral test it is bounded by $1 + \int_1^\infty x^{-2}\,dx = 2$. (Its sum is $\pi^2/6$.)

</details>

**q20.** Does $\displaystyle\sum_{n=1}^\infty \frac{1}{n}$ converge? `[bool]`

<details><summary>Answer</summary>

**No.** The harmonic series diverges (proof in q30), even though its terms go to 0. Terms tending to 0 is necessary for convergence, not sufficient.

</details>

**q21.** $\displaystyle\sum_{n=0}^\infty \left(\frac{2}{3}\right)^n$. `[number]`

<details><summary>Answer</summary>

**$3$.** Geometric with $r = \frac23$, starting at $n = 0$: $\frac{1}{1 - 2/3} = 3$. Answering 2 starts the sum at $n = 1$ (that sum is $\frac{r}{1 - r} = 2$). The EMA of `M02.2` is a truncated series of this kind.

</details>

**q22.** $\displaystyle\sum_{n=1}^\infty \frac{1}{n(n + 1)}$ (telescoping). `[number]`

<details><summary>Answer</summary>

**$1$.** $\frac{1}{n(n + 1)} = \frac1n - \frac{1}{n + 1}$, so the partial sum telescopes to $S_N = 1 - \frac{1}{N + 1} \to 1$. Answering $\frac12$ stops at the first term.

</details>

**q23.** $\displaystyle\sum_{n=1}^\infty \frac{n}{2^n}$. `[number]`

<details><summary>Answer</summary>

**$2$.** Differentiate $\sum_{n \ge 0} x^n = \frac{1}{1 - x}$ and multiply by $x$: $\sum_{n \ge 1} n x^n = \frac{x}{(1 - x)^2}$. At $x = \frac12$: $\frac{1/2}{1/4} = 2$. Answering 1 sums $\sum (1/2)^n$ instead.

</details>

**q24.** Does $\displaystyle\sum_{n=1}^\infty \frac{n!}{n^n}$ converge? (Ratio test.) `[bool]`

<details><summary>Answer</summary>

**Yes.** Ratio test: $\frac{a_{n+1}}{a_n} = \frac{(n + 1)!}{(n + 1)^{n+1}} \cdot \frac{n^n}{n!} = \left(\frac{n}{n + 1}\right)^n = \left(1 + \frac1n\right)^{-n} \to \frac1e < 1$.

</details>

**q25.** Does the alternating series $\displaystyle\sum_{n=1}^\infty \frac{(-1)^{n+1}}{n}$ converge? `[bool]`

<details><summary>Answer</summary>

**Yes.** By the alternating series test: the terms $\frac1n$ decrease to 0, so the partial sums converge (conditionally; the absolute series is harmonic).

</details>

**q26.** Give the sum $\displaystyle\sum_{n=1}^\infty \frac{(-1)^{n+1}}{n}$. `[number]`

<details><summary>Answer</summary>

**$\ln 2$.** The series is $\ln(1 + x) = x - \frac{x^2}{2} + \frac{x^3}{3} - \dots$ at $x = 1$, which is allowed because the series converges there (q25) and Abel's theorem carries the value to the endpoint. Common mistakes: $-\ln 2$ (sign of the first term) and 1 (the first term alone).

</details>

**q27.** Give the radius of convergence of $\displaystyle\sum_{n=1}^\infty \frac{x^n}{n}$. `[number]`

<details><summary>Answer</summary>

**$1$.** Ratio test on $\frac{x^n}{n}$: $\left|\frac{x^{n+1}/(n + 1)}{x^n/n}\right| = |x|\frac{n}{n + 1} \to |x|$, so the series converges for $|x| < 1$ and diverges for $|x| > 1$. The $\frac1n$ does not change the radius; $\infty$ is the radius of $\sum x^n/n!$.

</details>

**q28.** Give the interval of convergence of $\displaystyle\sum_{n=1}^\infty \frac{x^n}{n}$, endpoints included or not. `[interval]`

<details><summary>Answer</summary>

**$[-1, 1)$.** The radius is 1 (q27). At $x = 1$ the series is harmonic and diverges; at $x = -1$ it is $\sum (-1)^n/n = -\ln 2$, which converges by the alternating series test. Common mistakes: $(-1, 1)$, which drops the convergent left endpoint, and $[-1, 1]$, which keeps the divergent right one.

</details>

**q29.** Give the set of real $p$ for which $\displaystyle\sum_{n=1}^\infty \frac{1}{n^p}$ converges. `[interval]`

<details><summary>Answer</summary>

**$p \in (1, \infty)$.** The integral test compares with $\int_1^\infty x^{-p}\,dx$, finite exactly when $p > 1$. Including $p = 1$ (answering $[1, \infty)$) includes the harmonic series.

</details>

**q30.** Prove that $\displaystyle\sum_{n=1}^\infty \frac{1}{n}$ diverges. `[proof]`

<details><summary>Answer</summary>

**Claim.** $\sum_{n=1}^\infty \frac1n$ diverges.

**Proof (grouping).** Let $S_N = \sum_{n=1}^N \frac1n$. Group the terms after the first into blocks $B_k = \{2^{k-1} + 1, \dots, 2^k\}$ for $k = 1, 2, \dots$. Block $B_k$ has $2^{k-1}$ terms, and each $n \in B_k$ satisfies $n \le 2^k$, so $\frac1n \ge \frac{1}{2^k}$. Hence each block sums to at least $2^{k-1} \cdot 2^{-k} = \frac12$. For $N = 2^m$,
$$S_{2^m} \ge 1 + \frac{m}{2} .$$
The partial sums increase (all terms are positive) and $S_{2^m} \to \infty$, so they are unbounded and have no finite limit. The series diverges, even though $\frac1n \to 0$.

The integral test is an equally good route: $S_N \ge \int_1^{N+1} \frac{dx}{x} = \ln(N + 1)$, since $\frac1n \ge \int_n^{n+1}\frac{dx}{x}$. Either proof needs a lower bound that grows without limit, a justification of each inequality, and the conclusion that unbounded partial sums mean divergence. Model proof also in [`q30.md`](../../../../course/solutions/solve/S-M02/q30.md).

</details>

### Taylor series and remainders

**q31.** Give the degree 3 Taylor polynomial of $e^x$ at 0. `[expr in x]`

<details><summary>Answer</summary>

**$1 + x + \frac{x^2}{2} + \frac{x^3}{6}$.** Every derivative of $e^x$ at 0 is 1, so the coefficients are $\frac{1}{k!}$. Common mistakes: $1 + x + x^2 + x^3$ (factorials forgotten) and stopping at degree 2.

</details>

**q32.** Give the Maclaurin series of $\ln(1 + x)$ through the $x^3$ term. `[expr in x]`

<details><summary>Answer</summary>

**$x - \frac{x^2}{2} + \frac{x^3}{3}$.** Integrate $\frac{1}{1 + x} = 1 - x + x^2 - \dots$ term by term from 0. Answering $x + \frac{x^2}{2} + \frac{x^3}{3}$ loses the alternating signs (that is $-\ln(1 - x)$).

</details>

**q33.** Give the coefficient of $x^5$ in the Maclaurin series of $\sin x$. `[number]`

<details><summary>Answer</summary>

**$\frac{1}{120}$.** $\sin x = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \dots$, and the signs alternate starting with $+$, so the $x^5$ term is $+\frac{1}{120}$. Common mistakes: $-\frac{1}{120}$ (sign) and $\frac15$ (dividing by 5 instead of $5!$).

</details>

**q34.** Give the degree 4 Taylor polynomial of $\cos x$ at 0. `[expr in x]`

<details><summary>Answer</summary>

**$1 - \frac{x^2}{2} + \frac{x^4}{24}$.** The derivatives of $\cos$ at 0 cycle $1, 0, -1, 0, 1$, divided by $k!$. Answering $\frac{x^4}{4}$ divides by 4 instead of $4! = 24$.

</details>

**q35.** With the Lagrange remainder, bound $|e^x - T_3(x)|$ for $|x| \le 1/2$, where $T_3$ is the degree 3 Taylor polynomial at 0. Use $e^c \le e^{1/2}$ for the unknown point $c$ and give the bound. `[number]`

<details><summary>Answer</summary>

**$\frac{e^{1/2}}{384}$.** The Lagrange remainder of degree 3 is $R_3(x) = \frac{e^c}{4!}x^4$ with $c$ between 0 and $x$. For $|x| \le \frac12$, $e^c \le e^{1/2}$ and $x^4 \le \frac{1}{16}$, so $|R_3| \le \frac{e^{1/2}}{24 \cdot 16} = \frac{e^{1/2}}{384} \approx 0.0043$. `M02.1`'s exp polynomial is checked against exactly this kind of bound. Common mistakes: a remainder of the wrong degree (such as $\frac{e^{1/2}}{96}$), and $\frac{1}{384}$, which drops the $e^c$ factor and is too optimistic.

</details>

**q36.** Give the smallest $n$ with $\dfrac{1}{(n + 1)!} < 10^{-6}$. `[number]`

<details><summary>Answer</summary>

**$n = 9$.** $9! = 362\,880$, so $\frac{1}{9!} \approx 2.8 \times 10^{-6}$ (too big), while $10! = 3\,628\,800$ gives $\frac{1}{10!} \approx 2.8 \times 10^{-7} < 10^{-6}$. So $n + 1 = 10$ and $n = 9$. Answering 10 reports $n + 1$.

</details>

**q37.** Give the degree 3 Taylor polynomial of $\dfrac{1}{1 - x}$ at 0. `[expr in x]`

<details><summary>Answer</summary>

**$1 + x + x^2 + x^3$.** The geometric series $\frac{1}{1 - x} = \sum x^n$; every derivative at 0 is $k!$, so every coefficient is 1. Answering $1 - x + x^2 - x^3$ gives the series of $\frac{1}{1 + x}$.

</details>

**q38.** Range reduction writes $x = k \ln 2 + r$ with $k$ the integer nearest to $x / \ln 2$, so that $e^x = 2^k e^r$ with $|r| \le \frac{1}{2}\ln 2$. For $x = 5$, give $r$. `[number]`

<details><summary>Answer</summary>

**$r = 5 - 7\ln 2$.** $5/\ln 2 \approx 7.213$, so the nearest integer is $k = 7$ and $r = 5 - 7\ln 2 \approx 0.148$, within $\frac12\ln 2 \approx 0.347$. `M02.1`'s `exp_range_reduced` and the C `tl_expf` (`M09.6`) use exactly this reduction. Common mistakes: $5 - 8\ln 2 \approx -0.545$ (rounding up, so $|r| > \frac12\ln 2$) and $5/\ln 2 - 7$ (the remainder in units of $\ln 2$, not in $x$).

</details>

**q39.** $\operatorname{erf}(x) = \dfrac{2}{\sqrt{\pi}} \displaystyle\int_0^x e^{-t^2}\, dt$. Give the first three nonzero terms of its Maclaurin series. `[expr in x]`

<details><summary>Answer</summary>

**$\frac{2}{\sqrt\pi}\left(x - \frac{x^3}{3} + \frac{x^5}{10}\right)$.** Expand $e^{-t^2} = 1 - t^2 + \frac{t^4}{2} - \dots$ and integrate from 0 to $x$: $x - \frac{x^3}{3} + \frac{x^5}{10}$, then multiply by $\frac{2}{\sqrt\pi}$. `M02.1`'s `erf_series` feeds `M01.3`'s exact GELU. Common mistakes: $\frac{x^5}{5}$ (the $\frac12$ of $\frac{t^4}{2}$ lost) and dropping the $\frac{2}{\sqrt\pi}$.

</details>

**q40.** Prove that $\left|\sin x - \left(x - \dfrac{x^3}{6}\right)\right| \le \dfrac{|x|^5}{120}$ for every real $x$. `[proof]`

<details><summary>Answer</summary>

**Claim.** For every real $x$, $\left|\sin x - \left(x - \frac{x^3}{6}\right)\right| \le \frac{|x|^5}{120}$.

**Proof.** Let $f = \sin$. Its derivatives at 0 are $0, 1, 0, -1, 0$ for orders 0 to 4, so the degree 4 Taylor polynomial is $T_4(x) = x - \frac{x^3}{6}$ (the $x^4$ coefficient is 0, so it equals the degree 3 polynomial). Taylor's theorem with the Lagrange remainder gives, for each $x \ne 0$, a point $c$ between 0 and $x$ with
$$\sin x - T_4(x) = \frac{f^{(5)}(c)}{5!}x^5, \qquad f^{(5)} = \cos .$$
Since $|\cos c| \le 1$ and $5! = 120$, $|\sin x - T_4(x)| \le \frac{|x|^5}{120}$. At $x = 0$ both sides are 0. So the inequality holds for every real $x$.

The key step is using the degree 4 remainder: treating $x - \frac{x^3}{6}$ as only a degree 3 polynomial gives the weaker bound $\frac{x^4}{24}$. Model proof also in [`q40.md`](../../../../course/solutions/solve/S-M02/q40.md).

</details>

### Parametric curves and polar coordinates

**q41.** The curve $x = t^2$, $y = t^3$. Give $dy/dx$ at $t = 1$. `[number]`

<details><summary>Answer</summary>

**$3/2$.** $\frac{dy}{dx} = \frac{y'(t)}{x'(t)} = \frac{3t^2}{2t} = \frac{3t}{2}$, which is $\frac32$ at $t = 1$. Answering $\frac23$ computes $dx/dy$.

</details>

**q42.** Give the arc length of $x = \cos t$, $y = \sin t$ for $0 \le t \le \pi/2$. `[number]`

<details><summary>Answer</summary>

**$\pi/2$.** The speed is $\sqrt{\sin^2 t + \cos^2 t} = 1$, so the length is the parameter range, a quarter of the unit circle. Answering 1 gives the radius, not the length of the arc.

</details>

**q43.** Give the arc length of $x = t^2$, $y = t^3$ for $0 \le t \le 1$. `[number]`

<details><summary>Answer</summary>

**$\frac{13\sqrt{13} - 8}{27}$.** The speed is $\sqrt{(2t)^2 + (3t^2)^2} = t\sqrt{4 + 9t^2}$. With $u = 4 + 9t^2$, $du = 18t\,dt$: $\frac{1}{18}\int_4^{13}\sqrt u\,du = \frac{1}{18}\cdot\frac23\left(13^{3/2} - 8\right) = \frac{13\sqrt{13} - 8}{27} \approx 1.440$. Answering $\frac{13\sqrt{13}}{27}$ drops the lower limit $4^{3/2} = 8$.

</details>

**q44.** Give the area enclosed by the polar curve $r = 2\cos\theta$, $-\pi/2 \le \theta \le \pi/2$. `[number]`

<details><summary>Answer</summary>

**$\pi$.** $\frac12\int_{-\pi/2}^{\pi/2} 4\cos^2\theta\,d\theta = 2 \cdot \frac{\pi}{2} = \pi$. Check: $r = 2\cos\theta$ is the circle $(x - 1)^2 + y^2 = 1$ of radius 1. Answering $2\pi$ forgets the $\frac12$ in $\frac12 r^2\,d\theta$.

</details>

**q45.** Rewrite the polar curve $r = 2\sin\theta$ as an equation in $x$ and $y$. `[equation in x, y]`

<details><summary>Answer</summary>

**$x^2 + y^2 = 2y$.** Multiply by $r$: $r^2 = 2r\sin\theta$, then substitute $r^2 = x^2 + y^2$ and $r\sin\theta = y$. It is the circle of radius 1 centred at $(0, 1)$. Answering $x^2 + y^2 = 2x$ converts $r = 2\cos\theta$ instead.

</details>

**q46.** Give the area enclosed by the cardioid $r = 1 + \cos\theta$, $0 \le \theta \le 2\pi$. `[number]`

<details><summary>Answer</summary>

**$3\pi/2$.** $\frac12\int_0^{2\pi}(1 + \cos\theta)^2\,d\theta = \frac12\int_0^{2\pi}(1 + 2\cos\theta + \cos^2\theta)\,d\theta = \frac12(2\pi + 0 + \pi) = \frac{3\pi}{2}$. Answering $3\pi$ forgets the $\frac12$.

</details>

### First-order differential equations

**q47.** Solve $y' = -2y$ with $y(0) = 3$. `[expr in t]`

<details><summary>Answer</summary>

**$y = 3e^{-2t}$.** Separating, $y = Ce^{-2t}$, and $y(0) = 3$ fixes $C = 3$. Answering $e^{-2t}$ ignores the initial condition.

</details>

**q48.** Solve the logistic equation $y' = y(1 - y)$ with $y(0) = 1/2$. `[expr in t]`

<details><summary>Answer</summary>

**$y = \frac{1}{1 + e^{-t}}$.** Separate with partial fractions: $\frac{dy}{y(1 - y)} = \left(\frac1y + \frac{1}{1 - y}\right)dy = dt$, so $\ln\frac{y}{1 - y} = t + C$. $y(0) = \frac12$ gives $C = 0$, and solving $\frac{y}{1 - y} = e^t$ gives the sigmoid, the same function as in `L0.2` and `M01.3`. Common mistakes: $\frac{e^t}{2}$ (solving $y' = y$) and $\frac{1}{1 + e^t}$ (a sign slip, which decreases instead of increasing).

</details>

**q49.** Solve $y' + y = t$ with $y(0) = 0$. `[expr in t]`

<details><summary>Answer</summary>

**$y = t - 1 + e^{-t}$.** The integrating factor is $e^t$: $(ye^t)' = te^t$, so $ye^t = (t - 1)e^t + C$ and $y = t - 1 + Ce^{-t}$. $y(0) = -1 + C = 0$ gives $C = 1$. Answering $t - 1$ keeps only the particular solution and misses $y(0) = 0$.

</details>

**q50.** A quantity decays by $y' = -k y$ and halves every 10 time units. Give $k$. `[number]`

<details><summary>Answer</summary>

**$k = \frac{\ln 2}{10}$.** $y(10) = y(0)e^{-10k} = \frac12 y(0)$, so $10k = \ln 2$. Common mistakes: $\frac{1}{20}$ (treating halving as linear decay) and $10\ln 2$ (inverting the relation).

</details>

**q51.** Euler's method on $y' = y$, $y(0) = 1$, with step $h = 1/2$: give the approximation of $y(1)$ after two steps. `[number]`

<details><summary>Answer</summary>

**$9/4$.** Each Euler step on $y' = y$ multiplies by $1 + h$: $y_1 = 1 + \frac12 = \frac32$, $y_2 = \frac32 \cdot \frac32 = \frac94$. The exact value $e \approx 2.718$ is not the Euler approximation; 2 is one step of size 1.

</details>

**q52.** Gradient flow on $f(x) = x^2/2$ is $x'(t) = -f'(x) = -x$. Starting from $x(0) = 4$, give the time at which $x(t) = 1$. `[number]`

<details><summary>Answer</summary>

**$t = \ln 4$.** The solution is $x(t) = 4e^{-t}$, and $4e^{-t} = 1$ gives $t = \ln 4 \approx 1.386$. Answering 3 assumes constant speed 1; the speed $|x|$ shrinks as $x$ approaches the minimum, which is why gradient descent slows near it.

</details>

## 5. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Losing the constant from $du = g'(x)\,dx$ | substitution answers off by a factor of 2 | writes $du$ out in full before substituting (q4, q6, q10) |
| Treating an unbounded integrand as a divergent integral | $\int_0^1 x^{-1/2}$ reported as infinite | takes the limit $\delta \to 0^+$ before deciding (q15) |
| Confusing the normalizers of $e^{-x^2}$ and $e^{-x^2/2}$ | normal densities off by $\sqrt{2}$ | keeps the exponent's scale: $\int e^{-x^2} = \sqrt\pi$, $\int e^{-x^2/2} = \sqrt{2\pi}$ (q16) |
| Concluding convergence from terms that go to 0 | the harmonic series called convergent | bounds the partial sums from below (q20, q30) |
| Testing only the interior of an interval of convergence | an endpoint included or dropped wrongly | tests each endpoint with its own series test (q28, q29) |
| Using the remainder of the wrong degree, or dropping its derivative factor | an error bound that is too optimistic | uses the degree $n + 1$ derivative and keeps the $e^c$ bound (q35) |
| Rounding the range-reduction quotient the wrong way | $|r| > \frac{1}{2}\ln 2$, and the polynomial loses accuracy | rounds $x/\ln 2$ to the nearest integer (q38) |
| Ignoring the initial condition of an ODE | a family of solutions instead of one | applies $y(0)$ to fix the constant (q47, q49) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `S-M01` | derivatives, antiderivatives, and the fundamental theorem |
| Forward | `M02.1` | Taylor polynomials with range reduction for `exp` and `erf`, tested against the Lagrange bound of q35 |
| Forward | `M02.2` | the EMA as a truncated geometric series (q21) and its bias correction |
| Forward | `M07.0` | the normal density's constant (q16) behind Box-Muller normals |
| Forward | `M07.3` | the ReLU second moment of q17 sets Kaiming's gain $\sqrt{2}$ |
| Forward | `M02.4` | Euler's method and gradient flow (q51, q52) explain momentum |
| Forward | `S-M04` | multiple integrals and the change of variables behind q18 |
