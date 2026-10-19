# Calculus 2

Second-semester calculus extends integration to harder problems and then pivots to a deep idea: a function can be represented as an infinite sum. That idea is not a curiosity. It is how computers approximate transcendental functions such as $e^x$, $\sin x$, and $\operatorname{erf}$, and it is how you analyze the long-run behavior of algorithms and iterative methods, from the average cost of hashing to the smoothed loss curve of a training run.

This chapter follows that arc in two built sections and one problem set. The first section turns Taylor series into working code: a polynomial for $e^x$ with a proven error bound, range reduction so the bound holds for every input, and a series for erf that avoids catastrophic cancellation. The second section develops convergence of series and the geometric series, and uses them to explain why an exponential moving average starts out biased toward zero and how to correct it, the same correction Adam applies to its moments. The problem set then practises the hand techniques behind both: integration by parts and substitution, the Gaussian integral, convergence tests, Taylor remainders, polar curves, and first-order differential equations.

> **Learning objectives**
> - Choose an integration technique (parts, substitution, partial fractions, trigonometric substitution) and carry it out.
> - Decide whether a series converges using the divergence, geometric, $p$-series, integral, comparison, ratio, root, and alternating series tests.
> - Approximate a function with a Taylor polynomial, bound the error with the Lagrange remainder, and use range reduction to keep that error small.
> - Explain the exponential moving average as a geometric series and derive its bias correction.
> - Solve separable and first-order linear differential equations, and recognize gradient descent as Euler's method.

## Before you begin

You should be comfortable with limits, derivatives, and definite integrals from [Calculus 1](../02-calculus-1/README.md), in particular [Derivatives and finite differences](../02-calculus-1/01-derivatives-and-finite-differences.md). The code sections also call two helpers from [Precalculus](../01-precalculus/README.md): `horner` from [Polynomials, Horner, and stable quadratic roots](../01-precalculus/04-polynomials-horner-and-stable-roots.md) and the geometric-sequence helpers from [Sequences, geometric series, and frequency ladders](../01-precalculus/03-sequences-and-frequency-ladders.md). The companion textbook is OpenStax, [*Calculus Volume 2*](https://openstax.org/details/books/calculus-volume-2) (free, CC BY-NC-SA 4.0).

## In this chapter

1. [Taylor series, remainder bounds, and range reduction](01-taylor-series.md): derives Taylor polynomials and the Lagrange remainder, then builds `exp_range_reduced` ($e^x = 2^k e^r$ with a Cody-Waite reduction) and a cancellation-free `erf_series`, the float64 references for the exact GELU and the C `expf`.
2. [Series convergence, the EMA as a geometric series, and bias correction](02-series-and-the-ema.md): defines convergence, sums the geometric series exactly, and builds an `EMA` class whose debiased value is exact for a constant input, the correction behind Adam.
3. [Calculus 2 problem set](03-problem-set.md): 52 pen-and-paper problems with worked answers on integration techniques, the Gaussian, series, Taylor polynomials, polar curves, and ODEs.

## The mathematics at a glance

The sections above go deep on Taylor series and the geometric series. The rest of second-semester calculus is summarized here, with a worked example for each topic, so that you have the whole toolkit in view. The OpenStax chapter numbers point to fuller treatments and exercise sets.

### Integration techniques

*Calculus Volume 2*, sections 1.1 to 1.5, covers four techniques. **Integration by parts**, $\int u\,dv = uv - \int v\,du$, handles products of two kinds of function; a common heuristic for choosing $u$ is LIATE (logarithmic, inverse trigonometric, algebraic, trigonometric, exponential), taking the earliest kind present. **Trigonometric integrals** reduce powers with identities like $\sin^2 x = \frac{1 - \cos 2x}{2}$. **Trigonometric substitution** removes square roots: for $\sqrt{a^2 - x^2}$ let $x = a\sin\theta$, for $\sqrt{a^2 + x^2}$ let $x = a\tan\theta$, and for $\sqrt{x^2 - a^2}$ let $x = a\sec\theta$. **Partial fractions** decompose $P(x)/Q(x)$ into simpler fractions when $\deg P < \deg Q$.

Two short examples show the first and last of these. For $\int x e^x\,dx$, take $u = x$ and $dv = e^x\,dx$, so $du = dx$ and $v = e^x$; the result is $x e^x - \int e^x\,dx = x e^x - e^x + C = e^x(x - 1) + C$. For $\int \frac{dx}{x^2 - 1}$, write $\frac{1}{(x-1)(x+1)} = \frac{A}{x-1} + \frac{B}{x+1}$; solving gives $A = \frac12$, $B = -\frac12$, and the integral is $\frac12\ln|x-1| - \frac12\ln|x+1| + C$.

### Applications of integration

Integrals measure more than area (sections 2.1 to 2.4). The **arc length** of $y = f(x)$ on $[a, b]$ is $L = \int_a^b \sqrt{1 + f'(x)^2}\,dx$; the **surface area of revolution** is $S = 2\pi\int_a^b f(x)\sqrt{1 + f'(x)^2}\,dx$; and the **center of mass** of a lamina is $\bar x = M_y / m$ with $M_y = \int x\,\rho(x)\,f(x)\,dx$.

For example, the arc length of $y = x^{3/2}$ from $x = 0$ to $x = 4$ uses $f'(x) = \frac32 x^{1/2}$, so $L = \int_0^4 \sqrt{1 + \frac{9x}{4}}\,dx$. With $u = 1 + \frac{9x}{4}$, $du = \frac94\,dx$, this is $L = \frac49 \cdot \frac23\left[u^{3/2}\right]_1^{10} = \frac{8}{27}\left(10\sqrt{10} - 1\right)$.

### Parametric curves and polar coordinates

A **parametric curve** $x = f(t)$, $y = g(t)$ has slope $\frac{dy}{dx} = \frac{dy/dt}{dx/dt}$ and arc length $L = \int_a^b \sqrt{(dx/dt)^2 + (dy/dt)^2}\,dt$: at each instant the speed is the hypotenuse of the two velocity components (sections 3.1 to 3.4). **Polar coordinates** use $x = r\cos\theta$, $y = r\sin\theta$, and the area swept by $r(\theta)$ from $\alpha$ to $\beta$ is $A = \frac12\int_\alpha^\beta r(\theta)^2\,d\theta$.

For example, one petal of $r = \cos 2\theta$ spans $\theta \in [-\frac\pi4, \frac\pi4]$, so its area is $\frac12\int_{-\pi/4}^{\pi/4}\cos^2 2\theta\,d\theta = \frac12 \cdot \frac\pi4 = \frac\pi8$.

### Sequences and series

A **sequence** $\{a_n\}$ converges if $\lim_{n\to\infty} a_n$ exists and is finite; a **series** $\sum_{n=1}^\infty a_n$ converges if its sequence of partial sums does (sections 5.1 to 5.6). A **power series** $\sum c_n (x - a)^n$ has a radius of convergence $R$, and the **Taylor series** $f(x) = \sum_{n=0}^\infty \frac{f^{(n)}(a)}{n!}(x - a)^n$ is the power series built from derivatives.

The convergence tests form a decision tree, and each has a one-line intuition:

- **Divergence test**: if $\lim a_n \ne 0$, the series diverges. Terms must shrink to zero, but that is necessary, not sufficient.
- **Ratio test**: if $\lim |a_{n+1}/a_n| = L < 1$ the series converges absolutely; if $L > 1$ it diverges. It compares the series to a geometric one.
- **Root test**: if $\lim |a_n|^{1/n} = L$, the same conclusions hold as for the ratio test.
- **Integral test**: if $f$ is positive, continuous, and decreasing with $a_n = f(n)$, then $\sum a_n$ and $\int_1^\infty f(x)\,dx$ converge or diverge together.
- **Comparison test**: bound the series by a known geometric or $p$-series.
- **Alternating series test**: if $|a_n|$ decreases to 0, the alternating series converges.

A few series are worth knowing by heart. The geometric series $\sum r^n = \frac{1}{1 - r}$ for $|r| < 1$; the $p$-series $\sum 1/n^p$ converges if and only if $p > 1$; the harmonic series $\sum 1/n$ diverges ($p = 1$); and $e^x = \sum \frac{x^n}{n!}$, $\sin x = \sum \frac{(-1)^n x^{2n+1}}{(2n+1)!}$, $\cos x = \sum \frac{(-1)^n x^{2n}}{(2n)!}$.

For example, $\sum_{n=1}^\infty \frac{n^2}{2^n}$ yields to the ratio test: $\left|\frac{a_{n+1}}{a_n}\right| = \frac{(n+1)^2}{2n^2} \to \frac12 < 1$, so the series converges.

### Introduction to differential equations

An **ordinary differential equation** relates a function to its derivatives (sections 4.1 to 4.3). A **separable** equation $\frac{dy}{dx} = f(x)\,g(y)$ is solved by separating the variables and integrating both sides. A **first-order linear** equation $\frac{dy}{dx} + P(x)\,y = Q(x)$ is solved with the integrating factor $\mu(x) = e^{\int P(x)\,dx}$.

For example, $\frac{dy}{dx} = 2xy$ with $y(0) = 1$ separates to $\frac{dy}{y} = 2x\,dx$; integrating gives $\ln|y| = x^2 + C$, the initial condition gives $C = 0$, and the solution is $y = e^{x^2}$.

### Choosing a technique

When you face a new problem, the first decision is which tool to reach for. This table collects the cues:

| Technique | When to use | Key formula or idea |
|---|---|---|
| Integration by parts | product of two kinds of function | LIATE priority for choosing $u$ |
| Trigonometric substitution | $\sqrt{a^2 \pm x^2}$ or $\sqrt{x^2 - a^2}$ | substitute $x = a\sin\theta$, $a\tan\theta$, or $a\sec\theta$ |
| Partial fractions | rational functions $P(x)/Q(x)$ | factor the denominator, decompose |
| Ratio test | series with factorials or exponentials | compute $\lim \lvert a_{n+1}/a_n \rvert$ |
| Comparison test | series similar to a known $p$-series or geometric series | bound by a known convergent or divergent series |
| Taylor expansion | approximating $f(x)$ near a point | $f(x) = \sum f^{(n)}(a)(x-a)^n/n!$ |
| Separable ODE | $dy/dx = f(x)\,g(y)$ | separate and integrate both sides |

### Where this shows up in computing

These ideas recur well beyond this book. The harmonic series appears in average-case hashing analysis and the coupon-collector problem. Taylor series are how libraries compute $\sin$, $\cos$, and $\exp$ in floating point. The radius of convergence of a power series governs whether iterative algorithms converge, and convergence tests in general tell you whether an iterative method terminates. Differential equations model dynamic systems, from PID controllers to epidemics, and Fourier series in signal processing build on the same foundations.

## Further reading

- OpenStax, [*Calculus Volume 2*](https://openstax.org/details/books/calculus-volume-2) (free). Good exercise sets for each topic above: integration techniques, sections 1.1 (exercises 1 to 15), 1.3 (115 to 128), 1.4 (155 to 170), with challenge problems in 1.5 (200 to 210); applications, 2.1 (1 to 10), 2.2 (50 to 60), 2.4 (100 to 110, challenge 115 to 120); parametric and polar, 3.1 (1 to 12), 3.3 (110 to 122), 3.4 (150 to 162, challenge 168 to 175); series, 5.1 (1 to 14), 5.3 (120 to 138), 5.4 (175 to 190), 5.6 (260 to 275, challenge 280 to 288); differential equations, 4.1 (1 to 10), 4.3 (100 to 115, challenge 120 to 125).
- [Paul's Online Math Notes](https://tutorial.math.lamar.edu), for worked examples of every technique.
- [3Blue1Brown, calculus](https://www.3blue1brown.com/topics/calculus), for geometric intuition about series and Taylor polynomials.
