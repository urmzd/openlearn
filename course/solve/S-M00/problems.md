# S-M00 problems: functions, exponents and logarithms, trigonometry and Euler, series, polynomials, inequalities

Answer every question in `solve/S-M00.toml` (written by `ol start S-M00`).
The tag after each question is its answer type. `[number]` is a value:
give it **exactly** (`5/3`, `sqrt(3)/2`, `log(2)`, `5*pi/6`, `-7/5 + 24*I/5`)
unless the question says "decimal ok", where 7 significant digits are
enough. `[expr in x]` is a formula in the named variables (`(x + 7)/3`;
write $\theta$ as `t`). `[interval]` is a union of intervals such as
`(-oo, 2) U [3, 5)`. `[set]` is `{a, b}`, `[vector]` is `[a, b, c]`, and
`[bool]` is `true` or `false`. Natural log is `log`, $e$ is `E`, and
$i$ is `I`.

### Functions and inverses

**q1.** $f(x) = 3x - 7$. Give $f^{-1}(x)$. `[expr in x]`

**q2.** $f(x) = e^{2x + 1}$. Give $f^{-1}(x)$ for $x > 0$. `[expr in x]`

**q3.** $g(x) = \dfrac{2x + 1}{x - 3}$ for $x \ne 3$. Give $g^{-1}(x)$. `[expr in x]`

**q4.** $f(x) = x^2$ and $g(x) = x + 1$. Give $(f \circ g)(x) = f(g(x))$. `[expr in x]`

**q5.** The sigmoid is $\sigma(z) = 1 / (1 + e^{-z})$. Give its inverse, the logit $\sigma^{-1}(p)$, for $0 < p < 1$. `[expr in p]`

**q6.** Give the range of $\sigma$ (the set of values it takes over all real $z$). `[interval]`

**q7.** Give the domain of $h(x) = \ln(4 - x^2)$ (the real $x$ where it is defined). `[interval]`

**q8.** Is $f(x) = x^3 + x$ one-to-one on all of $\mathbb{R}$? (Hint: is it monotonic?) `[bool]`

**q9.** $f(x) = 2^x$ and $g(x) = \log_2 x$. Compute $f(g(16)) + g(f(3))$. `[number]`

**q10.** Softplus is $s(x) = \ln(1 + e^x)$. Give $s^{-1}(y)$ for $y > 0$. `[expr in y]`

### Exponents, logarithms, and units of information

**q11.** $\log_2 8 + \log_2 4$. `[number]`

**q12.** $\log_8 32$. `[number]`

**q13.** Simplify $e^{3 \ln x}$ for $x > 0$. `[expr in x]`

**q14.** Solve $2^x = 1000$ for $x$. `[number]`, decimal ok

**q15.** How many bits is one nat? `[number]`, decimal ok

**q16.** A model's mean negative log-likelihood is $2.0$ nats per token on a text of $1000$ tokens and $4000$ bytes. Give its bits per byte. `[number]`, decimal ok

**q17.** A model's perplexity is $e^{\text{mean NLL in nats}} = 8$ per token. Give its mean NLL in bits per token. `[number]`

**q18.** Write $a = \ln 2$ and $b = \ln 3$. Express $\ln 72$ in $a$ and $b$. `[expr in a, b]`

**q19.** Solve $\log_3 x + \log_3 (x - 8) = 2$. `[number]`

**q20.** Solve $4^x = 8$. `[number]`

**q21.** A quantity decays as $N(t) = N_0 e^{-kt}$ with $k > 0$. After what time is it half of $N_0$? `[expr in k]`

**q22.** How many decimal digits does $2^{100}$ have? (Use $\log_{10} 2 \approx 0.30103$.) `[number]`

### Trigonometry, complex numbers, and Euler's formula

**q23.** $\sin(\pi/6)$. `[number]`

**q24.** $\cos(2\pi/3)$. `[number]`

**q25.** Write $\cos a \cos b - \sin a \sin b$ as one trigonometric function of $a$ and $b$. `[expr in a, b]`

**q26.** Rotate the point $(1, 0)$ counterclockwise by $\pi/3$. Give the image. `[vector]`

**q27.** $R(\pi/2)\,R(\pi/3) = R(\theta)$ with $\theta \in [0, 2\pi)$, where $R(\theta)$ is the rotation by $\theta$. Give $\theta$. `[number]`

**q28.** $(1 + i)^8$. `[number]`

**q29.** $|3 + 4i|$. `[number]`

**q30.** Rotate $3 + 4i$ by the angle $t$ with $\cos t = 3/5$ and $\sin t = 4/5$: compute $(3 + 4i)(\cos t + i \sin t)$. `[number]`

**q31.** $e^{i\pi/2}$. `[number]`

**q32.** Express $\cos t$ using only exponentials $e^{it}$ and $e^{-it}$. `[expr in t]`

**q33.** For the unit complex numbers $u = e^{ia}$ and $v = e^{ib}$, simplify $\operatorname{Re}(u\,\bar{v})$, where $\bar{v}$ is the complex conjugate. `[expr in a, b]`

**q34.** A RoPE dimension pair rotates by $\omega p$ radians at position $p$, with $\omega = 0.01$. After how many positions does the pattern repeat (the period)? `[number]`

### Sequences and geometric series

**q35.** $1 + \tfrac12 + \tfrac14 + \tfrac18 + \cdots$ (forever). `[number]`

**q36.** Give $\sum_{j=0}^{n-1} r^j$ in closed form for $r \ne 1$. `[expr in r, n]`

**q37.** The sum of the first 10 terms of $3, 6, 12, 24, \ldots$ `[number]`

**q38.** The RoPE ladder of a 128-dimensional head with base $10000$ is $\omega_i = 10000^{-2i/128}$ for $i = 0, \ldots, 63$. Give the ratio $\omega_{i+1}/\omega_i$. `[number]`

**q39.** $1 + 2 + \cdots + 100$. `[number]`

**q40.** An exponential moving average $m_t = \beta m_{t-1} + (1 - \beta) g_t$ starts at $m_0 = 0$ with $\beta = 0.9$ and sees $g_t = 1$ at every step. Give $m_3$. `[number]`

**q41.** In q40, $m_3 = (1 - \beta)(g_3 + \beta g_2 + \beta^2 g_1)$. Give the weight on the oldest gradient $g_1$. `[number]`

**q42.** $\lim_{n \to \infty} \dfrac{2n + 1}{n + 3}$. `[number]`

**q43.** Write $0.272727\ldots$ (27 repeating) as a fraction. `[number]`

**q44.** ALiBi with 4 heads uses the slopes $2^{-2}, 2^{-4}, 2^{-6}, 2^{-8}$. Give their sum. `[number]`

### Polynomials

**q45.** Evaluate $p(x) = 2 - 3x + x^2 + 4x^3$ at $x = 2$ (Horner's rule makes it three multiply-adds). `[number]`

**q46.** Divide $x^3 - 6x^2 + 11x - 6$ by $x - 1$. Give the quotient. `[expr in x]`

**q47.** Give all roots of $x^3 - 6x^2 + 11x - 6$. `[set]`

**q48.** Give the roots of $2x^2 + 3x - 2$. `[set]`

**q49.** Give the smaller root of $x^2 - 10^6 x + 1$ exactly (radicals allowed) or to 15 significant digits. `[number]`

**q50.** Without solving, give the product of the roots of $3x^2 - 7x + 2$. `[number]`

**q51.** How many real roots does $x^2 + x + 1$ have? `[number]`

**q52.** The Taylor polynomial of $e^x$ of degree 3 around 0 is $c_0 + c_1 x + c_2 x^2 + c_3 x^3$. Give $[c_0, c_1, c_2, c_3]$. `[vector]`

### Inequalities

**q53.** Solve $|x - 3| < 2$. `[interval]`

**q54.** Solve $x^2 - 5x + 6 > 0$. `[interval]`

**q55.** The smallest integer $n$ with $2^n > 1000$ (the bits needed to number 1000 items). `[number]`

**q56.** Solve $\dfrac{x - 1}{x + 2} \ge 0$. `[interval]`

**q57.** For which probabilities $p \in (0, 1)$ is the surprisal $-\log_2 p$ greater than 3 bits? `[interval]`

**q58.** Solve $e^x \le 2$. `[interval]`

**q59.** The smallest value of $x + 1/x$ over $x > 0$. `[number]`

**q60.** Solve $\ln x < x - 1$ over $x > 0$. `[interval]`
