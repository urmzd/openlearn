# S-M02 problems: integration techniques, improper integrals, series, Taylor, parametric and polar, ODEs

Answer every question in `solve/S-M02.toml` (written by `ol start S-M02`).
The tag after each question is its answer type. `[number]` is an exact
value (`pi/2 - 1`, `log(3/2)/2`, `sqrt(pi)`, `exp(1/2)/384`); a decimal
fails. `[expr in x]` is a formula in the named variables. `[bool]` is
`true` or `false`. `[interval]` is a union of intervals such as `[-1, 1)`
or `(1, oo)`. `[equation]` is one equation with a single `=`, in any
equivalent form. `[proof]` is a file `solve/S-M02/qN.md`, graded against its
rubric. Natural log is `log`, $e$ is `E` or `exp(1)`, and $\pi$ is `pi`.

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
