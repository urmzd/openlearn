# S-M01 problems: limits, derivatives, rates, optimization, integrals, L'Hôpital

Answer every question in `solve/S-M01.toml` (written by `ol start S-M01`).
The tag after each question is its answer type. `[number]` is an exact
value (`3/5`, `exp(2)`, `2/pi`, `log(2)`, `sqrt(7)/2`); a decimal fails.
`[expr in x]` is a formula in the named variables (`2*x*cos(x^2)`).
`[set]` is `{a, b}`. `[proof]` is a file `solve/S-M01/qN.md`, graded
against its rubric. Natural log is `log`, $e$ is `E` or `exp(1)`, and
$\pi$ is `pi`.

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
