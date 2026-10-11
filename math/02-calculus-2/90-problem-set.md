<!-- ol:module S-M02 -->
# Calculus 2 problem set: integration techniques, the Gaussian, series, Taylor, polar, ODEs

## Overview

| | |
|---|---|
| **Module** | `S-M02` · solve · none · Pass 2 · 6 to 8 h |
| **You build** | answers in `solve/S-M02.toml` (48 checked by SymPy) and 4 proofs in `solve/S-M02/q12.md`, `q18.md`, `q30.md`, `q40.md` (self-graded against their rubrics) |
| **Contract** | none: a pen and paper set |
| **Tests** | `course/solve/S-M02/key.toml` (hidden): typed answers plus reject canaries; the problems are in `course/solve/S-M02/problems.md` and in section 4 |
| **Needs** | no module. Reading: `S-M01` (derivatives and the fundamental theorem) and the [Calculus 2 topic](README.md) |
| **Used by** | no call site (a solve set). Take it after `M02.1` (Taylor series in code) and `M02.2` (the EMA as a geometric series) in Pass 2; `M07.0` and `M07.3` read its Gaussian integrals, which are the solve-only `M02.3` |
| **Milestone** | `MS-P2` (the Pass 2 gate runs `ol check` on every solve part of the pass) |
| **Optional depth** | OpenStax, *Calculus Volume 2* (free), ch. 3 to 7; Trefethen, *Approximation Theory and Approximation Practice*, ch. 1 to 3, for why a few Taylor terms after range reduction are enough |

## Key Takeaways

- Integration by parts is the product rule integrated; substitution is the chain rule integrated (q1 to q11, q12).
- The Gaussian integral $\int e^{-x^2} = \sqrt{\pi}$ gives the normal density its constant, and half the second moment of a standard normal sits on each side of 0, which is the ReLU factor in Kaiming initialization (q16, q17).
- A series converges when its partial sums do; terms going to 0 is necessary, not sufficient (q20, q30).
- A Taylor polynomial plus a Lagrange remainder is an approximation with a guarantee; range reduction keeps the remainder small (q35, q38).
- A first-order ODE describes a rate; the logistic equation's solution is the sigmoid, and Euler's method is one step of gradient descent on gradient flow (q48, q51, q52).

## How to work this chapter

```bash
ol start S-M02             # writes solve/S-M02.toml and the four proof files
ol check S-M02             # SymPy checks the answers, then asks each proof rubric (y/n)
ol check S-M02 --regrade   # ask the rubrics again after you change a proof
```

---

## 1. Why now

Your Pass 2 code leans on this calculus in three places. `M02.1` computes $e^x$ and $\operatorname{erf}$ from Taylor polynomials, and its tests check a Lagrange remainder bound; `M09.6` later puts the same polynomial in C. `M02.2` treats the exponential moving average as a geometric series and derives Adam's bias correction (`M10.3`) from its partial sum. And `M07.0` and `M07.3` draw normal random numbers and pick initialization scales from integrals of the normal density, which only exist because $\int e^{-x^2}$ converges to $\sqrt{\pi}$. This set checks the hand techniques behind those modules: integrating by parts and by substitution, improper integrals, convergence, Taylor remainders, curves, and the simplest differential equations.

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

This is a sibling of q1 and q17, not one of the graded problems.

**By parts, twice.** Compute $\int_0^1 x^2 e^{-x}\,dx$. Take $u = x^2$, $v' = e^{-x}$, so $u' = 2x$, $v = -e^{-x}$: the integral is $[-x^2 e^{-x}]_0^1 + 2\int_0^1 x e^{-x}\,dx = -e^{-1} + 2\int_0^1 x e^{-x}\,dx$. Again with $u = x$: $\int_0^1 x e^{-x} = [-x e^{-x}]_0^1 + \int_0^1 e^{-x} = -e^{-1} + (1 - e^{-1}) = 1 - 2e^{-1}$. Total: $-e^{-1} + 2 - 4e^{-1} = 2 - 5e^{-1} \approx 0.1606$. In `solve/` this is `answer = "2 - 5*exp(-1)"`; `answer = "0.1606"` fails as inexact.

**A Gaussian moment.** $\mathbb{E}[|Z|] = 2\int_0^\infty z\,\varphi(z)\,dz$ for a standard normal $Z$. Substitute $u = z^2/2$, $du = z\,dz$: $\int_0^\infty z e^{-z^2/2} dz = \int_0^\infty e^{-u} du = 1$. So $\mathbb{E}[|Z|] = \frac{2}{\sqrt{2\pi}} = \sqrt{2/\pi} \approx 0.798$.

## 4. The problem set

Write each answer in `solve/S-M02.toml`:

```toml
[q3]
answer = "pi/2 - 1"
[q19]
answer = "true"
[q28]
answer = "[-1, 1)"
[q45]
answer = "x^2 + y^2 = 2*y"
[q12]
proof = "S-M02/q12.md"
```

Numbers are exact (`log(3/2)/2`, not `0.2027`). Taylor polynomials are checked symbolically, in any term order. Write $e$ as `E` or `exp(1)` and $\pi$ as `pi`.

<!-- ol:problems S-M02 -->

### Integration techniques

**q1.** $\displaystyle\int_0^1 x e^x\, dx$ (by parts). `[number]`

**q2.** $\displaystyle\int_1^e \ln x\, dx$. `[number]`

**q3.** $\displaystyle\int_0^{\pi/2} x \cos x\, dx$. `[number]`

**q4.** $\displaystyle\int_0^1 \frac{x}{1 + x^2}\, dx$ (substitution). `[number]`

**q5.** $\displaystyle\int_0^{\pi/2} \sin^2 x\, dx$. `[number]`

**q6.** $\displaystyle\int_2^3 \frac{1}{x^2 - 1}\, dx$ (partial fractions). `[number]`

**q7.** Give the antiderivative $F$ of $x^2 e^x$ with $F(0) = 0$. `[expr in x]`

**q8.** $\displaystyle\int_0^1 x \sqrt{1 - x^2}\, dx$. `[number]`

**q9.** $\displaystyle\int_0^\pi x \sin x\, dx$. `[number]`

**q10.** $\displaystyle\int_0^1 x^3 e^{x^2}\, dx$. `[number]`

**q11.** $\displaystyle\int_0^1 \frac{1}{\sqrt{4 - x^2}}\, dx$ (substitute $x = 2\sin\theta$). `[number]`

**q12.** Derive the integration by parts formula $\int_a^b u\,v'\,dx = \big[u v\big]_a^b - \int_a^b u'\,v\,dx$ from the product rule and the fundamental theorem of calculus. `[proof]`

### Improper integrals and the Gaussian

**q13.** $\displaystyle\int_1^\infty \frac{1}{x^2}\, dx$. `[number]`

**q14.** $\displaystyle\int_0^\infty e^{-2x}\, dx$. `[number]`

**q15.** $\displaystyle\int_0^1 \frac{1}{\sqrt{x}}\, dx$ (improper at 0). `[number]`

**q16.** $\displaystyle\int_{-\infty}^{\infty} e^{-x^2}\, dx$. `[number]`

**q17.** $Z$ is a standard normal with density $\varphi(z) = e^{-z^2/2}/\sqrt{2\pi}$. Give $\mathbb{E}[\max(Z, 0)^2] = \displaystyle\int_0^\infty z^2 \varphi(z)\, dz$, the second moment of a ReLU of a standard normal. `[number]`

**q18.** Prove $\displaystyle\int_{-\infty}^{\infty} e^{-x^2}\, dx = \sqrt{\pi}$ by squaring the integral and changing to polar coordinates. `[proof]`

### Series and convergence tests

**q19.** Does $\displaystyle\sum_{n=1}^\infty \frac{1}{n^2}$ converge? `[bool]`

**q20.** Does $\displaystyle\sum_{n=1}^\infty \frac{1}{n}$ converge? `[bool]`

**q21.** $\displaystyle\sum_{n=0}^\infty \left(\frac{2}{3}\right)^n$. `[number]`

**q22.** $\displaystyle\sum_{n=1}^\infty \frac{1}{n(n + 1)}$ (telescoping). `[number]`

**q23.** $\displaystyle\sum_{n=1}^\infty \frac{n}{2^n}$. `[number]`

**q24.** Does $\displaystyle\sum_{n=1}^\infty \frac{n!}{n^n}$ converge? (Ratio test.) `[bool]`

**q25.** Does the alternating series $\displaystyle\sum_{n=1}^\infty \frac{(-1)^{n+1}}{n}$ converge? `[bool]`

**q26.** Give the sum $\displaystyle\sum_{n=1}^\infty \frac{(-1)^{n+1}}{n}$. `[number]`

**q27.** Give the radius of convergence of $\displaystyle\sum_{n=1}^\infty \frac{x^n}{n}$. `[number]`

**q28.** Give the interval of convergence of $\displaystyle\sum_{n=1}^\infty \frac{x^n}{n}$, endpoints included or not. `[interval]`

**q29.** Give the set of real $p$ for which $\displaystyle\sum_{n=1}^\infty \frac{1}{n^p}$ converges. `[interval]`

**q30.** Prove that $\displaystyle\sum_{n=1}^\infty \frac{1}{n}$ diverges. `[proof]`

### Taylor series and remainders

**q31.** Give the degree 3 Taylor polynomial of $e^x$ at 0. `[expr in x]`

**q32.** Give the Maclaurin series of $\ln(1 + x)$ through the $x^3$ term. `[expr in x]`

**q33.** Give the coefficient of $x^5$ in the Maclaurin series of $\sin x$. `[number]`

**q34.** Give the degree 4 Taylor polynomial of $\cos x$ at 0. `[expr in x]`

**q35.** With the Lagrange remainder, bound $|e^x - T_3(x)|$ for $|x| \le 1/2$, where $T_3$ is the degree 3 Taylor polynomial at 0. Use $e^c \le e^{1/2}$ for the unknown point $c$ and give the bound. `[number]`

**q36.** Give the smallest $n$ with $\dfrac{1}{(n + 1)!} < 10^{-6}$. `[number]`

**q37.** Give the degree 3 Taylor polynomial of $\dfrac{1}{1 - x}$ at 0. `[expr in x]`

**q38.** Range reduction writes $x = k \ln 2 + r$ with $k$ the integer nearest to $x / \ln 2$, so that $e^x = 2^k e^r$ with $|r| \le \frac{1}{2}\ln 2$. For $x = 5$, give $r$. `[number]`

**q39.** $\operatorname{erf}(x) = \dfrac{2}{\sqrt{\pi}} \displaystyle\int_0^x e^{-t^2}\, dt$. Give the first three nonzero terms of its Maclaurin series. `[expr in x]`

**q40.** Prove that $\left|\sin x - \left(x - \dfrac{x^3}{6}\right)\right| \le \dfrac{|x|^5}{120}$ for every real $x$. `[proof]`

### Parametric curves and polar coordinates

**q41.** The curve $x = t^2$, $y = t^3$. Give $dy/dx$ at $t = 1$. `[number]`

**q42.** Give the arc length of $x = \cos t$, $y = \sin t$ for $0 \le t \le \pi/2$. `[number]`

**q43.** Give the arc length of $x = t^2$, $y = t^3$ for $0 \le t \le 1$. `[number]`

**q44.** Give the area enclosed by the polar curve $r = 2\cos\theta$, $-\pi/2 \le \theta \le \pi/2$. `[number]`

**q45.** Rewrite the polar curve $r = 2\sin\theta$ as an equation in $x$ and $y$. `[equation in x, y]`

**q46.** Give the area enclosed by the cardioid $r = 1 + \cos\theta$, $0 \le \theta \le 2\pi$. `[number]`

### First-order differential equations

**q47.** Solve $y' = -2y$ with $y(0) = 3$. `[expr in t]`

**q48.** Solve the logistic equation $y' = y(1 - y)$ with $y(0) = 1/2$. `[expr in t]`

**q49.** Solve $y' + y = t$ with $y(0) = 0$. `[expr in t]`

**q50.** A quantity decays by $y' = -k y$ and halves every 10 time units. Give $k$. `[number]`

**q51.** Euler's method on $y' = y$, $y(0) = 1$, with step $h = 1/2$: give the approximation of $y(1)$ after two steps. `[number]`

**q52.** Gradient flow on $f(x) = x^2/2$ is $x'(t) = -f'(x) = -x$. Starting from $x(0) = 4$, give the time at which $x(t) = 1$. `[number]`

<!-- /ol:problems -->

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Losing the constant from $du = g'(x)\,dx$ | substitution answers off by a factor of 2 | q4 (canary log(2)), q6 (canary log(3/2)), q10 (canary 1) |
| Treating an unbounded integrand as a divergent integral | $\int_0^1 x^{-1/2}$ reported as infinite | q15 (canary oo) |
| Confusing the normalizers of $e^{-x^2}$ and $e^{-x^2/2}$ | normal densities off by $\sqrt{2}$ | q16 (canary sqrt(2*pi)) |
| Concluding convergence from terms that go to 0 | the harmonic series called convergent | q20, q30 (proof) |
| Testing only the interior of an interval of convergence | an endpoint included or dropped wrongly | q28 (canaries), q29 (canary [1, oo)) |
| Using the remainder of the wrong degree, or dropping its derivative factor | an error bound that is too optimistic | q35 (canaries) |
| Rounding the range-reduction quotient the wrong way | $|r| > \frac{1}{2}\ln 2$, and the polynomial loses accuracy | q38 (canary 5 - 8*log(2)) |
| Ignoring the initial condition of an ODE | a family of solutions instead of one | q47, q49 (canaries) |

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
