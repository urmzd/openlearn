# Problem set B: condition numbers, error bounds, Newton, polynomial approximation, and low precision

Inference code is full of numbers that should agree but were computed differently: the sampler's float64 sums against the Rust engine's, a quantized matmul against float32, a float16 KV cache against a float32 one. The C kernels then compute `rsqrt` by Newton's method and `expf` by range reduction and a polynomial, and must be proved accurate to a few ulps. [Error analysis, condition numbers, and tolerance budgets](03-error-analysis-condition-numbers-and-tolerance-budgets.md) and [FP8 and microscaling formats](04-fp8-and-microscaling-formats.md) turn the bounds and the formats into code. This set derives them by hand first, so that a failing tolerance or a kernel that is off by one ulp is something you can explain rather than tune.

The set has five parts: condition numbers (q1 to q6), sum and dot error bounds (q7 to q12), fixed-point and Newton convergence (q13 to q17), polynomial approximation (q18 to q23), and low-precision arithmetic (q24 to q28). Three of the problems, q6, q12, and q17, are short proofs. q14 to q17 prepare [Fixed-point iteration and fast inverse square root in C](05-fixed-point-iteration-and-rsqrt.md), and q18 to q20 prepare [Polynomial approximation and range reduction: expf in C](06-polynomial-approximation-and-expf.md); reading the set beside those sections and the two above works well.

It assumes the representation and rounding of [Problem set A](90-problem-set-a.md) and Newton's method from [Newton's method](../01-calculus-1/02-newtons-method.md). No code is needed. Work each problem before opening its answer; each answer gives the final value, the derivation, and the common wrong answer with the reason it is wrong.

> **Learning objectives**
> - Compute relative and matrix condition numbers and use them to bound how much a problem amplifies input errors.
> - Derive and apply $\gamma_k$ bounds for recursive, pairwise, and statistically typical summation.
> - Determine the convergence rate of a fixed-point iteration and of Newton's method at simple and double roots.
> - Choose a polynomial degree from the Lagrange remainder after range reduction, and explain why Chebyshev nodes minimize the interpolation node product.
> - Round values into E4M3, E5M2, and FP4, and choose an MX block scale.

## The results every problem uses

The symbols used in the problems:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $u = 2^{-p}$ | unit roundoff, $p$ significand bits with the implicit 1 | float |
| $\gamma_k = ku/(1 - ku)$ | accumulated relative error of $k$ roundings | float |
| $\kappa_f(x) = \lvert x f'(x)/f(x)\rvert$ | relative condition number of $f$ at $x$ | float |
| $\kappa_2(A) = \sigma_{\max}/\sigma_{\min}$ | 2-norm condition number | float |
| $g$, $x^*$ | an iteration map $x_{k+1} = g(x_k)$ and its fixed point $g(x^*) = x^*$ | function, float |
| $e_k$ | the error after $k$ steps | float |
| $T_n$ | the Chebyshev polynomial of degree $n$, $T_n(\cos\theta) = \cos n\theta$ | polynomial |
| $X$ | an MX block scale, a power of two | float |

### Conditioning

A relative change $\epsilon$ in the input changes the output of $f$ by about $\kappa_f(x)\,\epsilon$ (first-order Taylor). For $Ax = b$, $\lVert\delta x\rVert/\lVert x\rVert \le \kappa_2(A)\, \lVert\delta b\rVert/\lVert b\rVert$, with $\lVert A\rVert_2 = \sigma_{\max}$ and $\lVert A^{-1}\rVert_2 = 1/\sigma_{\min}$. The singular values of $A$ are the square roots of the eigenvalues of $A^\top A$, not the eigenvalues of $A$ (they coincide only for symmetric positive semidefinite $A$).

### Error bounds

Under $\mathrm{fl}(a \circ b) = (a \circ b)(1 + \delta)$, $\lvert\delta\rvert \le u$, a dot product of length $k$ in any order satisfies $\lvert\mathrm{fl}(x^\top y) - x^\top y\rvert \le \gamma_k\, \lvert x\rvert^\top\lvert y\rvert$. To first order $\gamma_k \approx ku$. A balanced (pairwise) tree puts each term through $\lceil\log_2 k\rceil$ additions instead of up to $k - 1$. Rounding errors with independent random signs add like a random walk, so the typical error is about $\sqrt{k}\,u$; the worst case is $ku$.

### Iterations

If $g$ is differentiable near a fixed point $x^*$, then $e_{k+1} = g(x_k) - g(x^*) \approx g'(x^*)\, e_k$: linear convergence with rate $\lvert g'(x^*)\rvert$ when it is below 1. Newton's iteration for a root of $f$ is $g(x) = x - f(x)/f'(x)$; at a simple root $g'(x^*) = 0$ and $e_{k+1} \approx \tfrac{g''(x^*)}{2} e_k^2$, which is quadratic. At a double root $f'(x^*) = 0$ too and the rate degrades to linear.

### Polynomial approximation

Taylor's theorem with the Lagrange remainder gives $e^r = \sum_{j=0}^{n} r^j/j! + R_n$ with $\lvert R_n\rvert \le e^{\lvert r\rvert}\lvert r\rvert^{n+1}/(n+1)!$. Range reduction $x = k\ln 2 + r$, $k = \mathrm{round}(x/\ln 2)$, keeps $\lvert r\rvert \le \ln 2/2$, so a short polynomial suffices, and $e^x = 2^k e^r$ is an exponent adjustment. Horner's rule $a_0 + r(a_1 + r(a_2 + \dots))$ uses $n$ multiplications for degree $n$. Interpolating at $n + 1$ nodes leaves an error proportional to $\prod_i (x - x_i)$; the Chebyshev nodes (zeros of $T_{n+1}$) make that product the monic $T_{n+1}/2^n$, whose maximum on $[-1, 1]$ is $2^{-n}$, the smallest possible.

### Low precision

A format with $M$ fraction bits and bias $B$ has quantum $2^{\max(\lfloor\log_2 v\rfloor, 1 - B) - M}$ near $v$; rounding is to the nearest multiple, ties to the even significand. The formats used here are E4M3 ($M = 3$, $B = 7$, max 448), E5M2 ($M = 2$, $B = 15$, max 57344, top exponent reserved), and FP4 E2M1 (values 0, 0.5, 1, 1.5, 2, 3, 4, 6). An MX block scale is $X = 2^{\lfloor\log_2 a_{\max}\rfloor - e_{\max}}$, with $e_{\max}$ the exponent of the element format's largest normal.

## Three worked examples

These are siblings of q1, q14, and q24, worked in full before you start.

The conditioning of $\sqrt{x}$: $f'(x) = 1/(2\sqrt{x})$, so $\kappa = \lvert x \cdot \tfrac{1}{2\sqrt{x}} / \sqrt{x}\rvert = \tfrac12$ for every $x > 0$. Square roots halve relative errors.

One Newton step for $1/\sqrt{a}$, with $a = 2$ and $y_0 = 0.7$: $y_1 = 0.7\,(3 - 2 \cdot 0.49)/2 = 0.7 \cdot 2.02/2 = 0.707$. The relative error drops from $1 - \sqrt{2} \cdot 0.7 = 0.01005$ to $1 - \sqrt{2}\cdot 0.707 = 0.000151$, close to $\tfrac32 (0.01005)^2 = 0.000152$: quadratic.

Rounding 2.75 to E4M3: $\lfloor\log_2 2.75\rfloor = 1$, the quantum is $2^{1-3} = 0.25$, and $2.75/0.25 = 11$ exactly, so the value is 2.75 and the code is $((-2 + 3 + 7 - 1) \ll 3) + 11 = 56 + 11 = 67$. To decode: $e_f = 8$, $f = 3$, $(8 + 3)\cdot 2^{8 - 7 - 3} = 2.75$.

## The problems

The problems are grouped by topic. The tag after each question is its answer type.

### Condition numbers

**q1.** Give the relative condition number of $f(x) = x - 1$ at $x = 1.001$. `[number]`

<details><summary>Answer</summary>

$1001$. $f'(x) = 1$, so $\kappa_f = \lvert x \cdot 1 / (x - 1)\rvert = 1.001 / 0.001 = 1001$: a relative error of $10^{-6}$ in $x$ becomes about $10^{-3}$ in $x - 1$. This is cancellation seen as conditioning. A common mistake is $1$, the absolute condition $\lvert f'(x)\rvert$; $1/1001$ inverts the ratio.

</details>

**q2.** Give the relative condition number of $f(x) = e^x$ at $x = 50$. `[number]`

<details><summary>Answer</summary>

$50$. $x f'(x)/f(x) = x e^x / e^x = x$. A common mistake is $e^{50}$, the absolute condition $\lvert f'(x)\rvert$; relative errors are amplified by $x$, not by $e^x$.

</details>

**q3.** Give $\kappa_2$ of $\mathrm{diag}(10, 1/10)$. `[number]`

<details><summary>Answer</summary>

$100$. For a diagonal matrix the singular values are the absolute diagonal entries, $10$ and $1/10$, so $\kappa_2 = 10/(1/10) = 100$. A common mistake is $10$, taking the largest entry alone.

</details>

**q4.** Give $\kappa_2$ of $A = \begin{pmatrix} 3 & 0 \\ 4 & 5 \end{pmatrix}$. `[number]`

<details><summary>Answer</summary>

$3$. $A^\top A = \begin{pmatrix} 25 & 20 \\ 20 & 25 \end{pmatrix}$ has eigenvalues $45$ and $5$, so the singular values are $\sqrt{45}$ and $\sqrt 5$ and $\kappa_2 = \sqrt{45/5} = 3$. A common mistake is $5/3$, the ratio of $A$'s eigenvalues (its diagonal, since it is triangular): $A$ is not symmetric, so eigenvalues are not singular values. $9$ is the ratio of $A^\top A$'s eigenvalues, the square of the answer.

</details>

**q5.** $\kappa_2(A) = 1000$ and $b$ moves by a relative amount $\lVert\delta b\rVert/\lVert b\rVert = 10^{-6}$. Give the bound on the relative change $\lVert\delta x\rVert/\lVert x\rVert$ of the solution of $Ax = b$. `[number]`

<details><summary>Answer</summary>

$10^{-3}$. $\lVert\delta x\rVert/\lVert x\rVert \le \kappa_2(A) \cdot \lVert\delta b\rVert/\lVert b\rVert = 1000 \cdot 10^{-6}$. A common mistake is $10^{-9}$, dividing by the condition number instead of multiplying.

</details>

**q6.** Prove that if $Ax = b$ and $A(x + \delta x) = b + \delta b$ with $A$ invertible and $b \ne 0$, then $\lVert\delta x\rVert/\lVert x\rVert \le \kappa_2(A)\, \lVert\delta b\rVert/\lVert b\rVert$ in the 2-norm. `[proof]`

<details><summary>Answer</summary>

**Proof.** Subtracting $Ax = b$ from $A(x + \delta x) = b + \delta b$ gives $A\,\delta x = \delta b$, so $\delta x = A^{-1}\delta b$. By the definition of the induced norm,

$$\lVert\delta x\rVert \le \lVert A^{-1}\rVert\,\lVert\delta b\rVert.$$

Also $\lVert b\rVert = \lVert Ax\rVert \le \lVert A\rVert\,\lVert x\rVert$. Since $b \ne 0$, $x \ne 0$, and dividing gives $1/\lVert x\rVert \le \lVert A\rVert/\lVert b\rVert$. Multiplying the two non-negative bounds,

$$\frac{\lVert\delta x\rVert}{\lVert x\rVert} \le \lVert A\rVert\,\lVert A^{-1}\rVert\,\frac{\lVert\delta b\rVert}{\lVert b\rVert}.$$

In the 2-norm, $\lVert A\rVert_2 = \sigma_{\max}(A)$, and the singular values of $A^{-1}$ are the reciprocals of those of $A$, so $\lVert A^{-1}\rVert_2 = 1/\sigma_{\min}(A)$. The factor is therefore $\sigma_{\max}/\sigma_{\min} = \kappa_2(A)$. $\blacksquare$

A complete proof has four steps: $A\,\delta x = \delta b$; the two norm bounds on $\lVert\delta x\rVert$ and $1/\lVert x\rVert$; their product; and the identification of $\lVert A\rVert_2\lVert A^{-1}\rVert_2$ with $\sigma_{\max}/\sigma_{\min}$.

</details>

### Sum and dot error bounds

**q7.** Give $\gamma_3$ for float32 exactly. `[number]`

<details><summary>Answer</summary>

$\gamma_3 = \dfrac{3u}{1 - 3u} = \dfrac{3 \cdot 2^{-24}}{1 - 3 \cdot 2^{-24}} = \dfrac{3}{2^{24} - 3}$ (multiply numerator and denominator by $2^{24}$), with float32's $u = 2^{-24}$. A common mistake is $3 \cdot 2^{-24}$, the first-order value without the denominator; $3/(2^{23} - 3)$ uses machine epsilon $2^{-23}$ instead of $u$.

</details>

**q8.** A float32 dot product of length $k = 4096$ has $\lvert x\rvert \cdot \lvert y\rvert = 1$. Give the first-order bound $k u$ on its error. `[number]`

<details><summary>Answer</summary>

$2^{-12}$. $ku \cdot \lvert x\rvert^\top\lvert y\rvert = 4096 \cdot 2^{-24} \cdot 1 = 2^{12 - 24}$. A common mistake is $2^{-11}$, from using $\epsilon = 2^{-23}$ for $u$.

</details>

**q9.** The same dot product with the sum done pairwise: give the first-order bound $(\lceil \log_2 k\rceil + 1)\, u$ (one rounding per product, one per tree level). `[number]`

<details><summary>Answer</summary>

$13 \cdot 2^{-24}$. $\lceil\log_2 4096\rceil = 12$ tree levels, each one rounding, plus one rounding for each product: $(12 + 1)u$. A common mistake is $12 \cdot 2^{-24}$, which forgets that the products round too; $2^{-12}$ is the recursive (non-pairwise) bound of q8.

</details>

**q10.** In bf16, give the smallest $k$ for which $k u \ge 1$, where the bound $\gamma_k$ says nothing. `[number]`

<details><summary>Answer</summary>

$256$. bf16 has 8 significant bits, so $u = 2^{-8}$ and $ku \ge 1$ first at $k = 2^8$. A common mistake is $128$, from $\epsilon = 2^{-7}$.

</details>

**q11.** The statistical rule of thumb replaces $k u$ by $\sqrt{k}\, u$ (errors with random signs add like a random walk). Give it for float32 and $k = 4096$. `[number]`

<details><summary>Answer</summary>

$2^{-18}$. $\sqrt{4096}\,u = 64 \cdot 2^{-24} = 2^{6 - 24}$, sixty-four times smaller than the worst case $2^{-12}$ of q8. A common mistake is $2^{-12}$, which is the worst case $ku$, not the statistical rule.

</details>

**q12.** Prove that recursive summation of three numbers satisfies $\lvert \mathrm{fl}(\mathrm{fl}(x_1 + x_2) + x_3) - (x_1 + x_2 + x_3)\rvert \le \gamma_2 (\lvert x_1\rvert + \lvert x_2\rvert + \lvert x_3\rvert)$ under the model $\mathrm{fl}(a + b) = (a + b)(1 + \delta)$, $\lvert\delta\rvert \le u$. `[proof]`

<details><summary>Answer</summary>

**Proof.** Assume $2u < 1$ so $\gamma_2 = 2u/(1 - 2u)$ is defined. The first addition gives $s_1 = \mathrm{fl}(x_1 + x_2) = (x_1 + x_2)(1 + \delta_1)$ and the second $s_2 = \mathrm{fl}(s_1 + x_3) = (s_1 + x_3)(1 + \delta_2)$, with $\lvert\delta_1\rvert, \lvert\delta_2\rvert \le u$. Expanding,

$$s_2 = (x_1 + x_2)(1 + \delta_1)(1 + \delta_2) + x_3(1 + \delta_2),$$

so the error is

$$s_2 - (x_1 + x_2 + x_3) = (x_1 + x_2)\bigl[(1 + \delta_1)(1 + \delta_2) - 1\bigr] + x_3\,\delta_2.$$

Bound each factor. $\lvert(1 + \delta_1)(1 + \delta_2) - 1\rvert = \lvert\delta_1 + \delta_2 + \delta_1\delta_2\rvert \le 2u + u^2$, and this is at most $\gamma_2$ because $\gamma_2 - (2u + u^2) = \frac{2u - (2u + u^2)(1 - 2u)}{1 - 2u} = \frac{u^2(3 + 2u)}{1 - 2u} \ge 0$. Also $\lvert\delta_2\rvert \le u \le \gamma_2$. By the triangle inequality,

$$\lvert s_2 - (x_1 + x_2 + x_3)\rvert \le \gamma_2\lvert x_1 + x_2\rvert + \gamma_2\lvert x_3\rvert \le \gamma_2(\lvert x_1\rvert + \lvert x_2\rvert + \lvert x_3\rvert). \quad\blacksquare$$

The key observation is that $x_1$ and $x_2$ pass through two roundings and $x_3$ through one, and both counts are covered by $\gamma_2$.

</details>

### Fixed point and Newton convergence

**q13.** The iteration $x_{k+1} = \cos x_k$ converges to the fixed point $x^* \approx 0.739085$. Give its asymptotic linear rate $\lvert g'(x^*)\rvert = \sin x^*$ to at least 7 significant digits. `[number, rtol 1e-6]`

<details><summary>Answer</summary>

$\sin x^* \approx 0.6736120$. $g(x) = \cos x$, so $g'(x) = -\sin x$ and the rate is $\lvert g'(x^*)\rvert = \sin(0.7390851332) = 0.6736120291832148$: each step multiplies the error by about $0.674$, so one decimal digit takes about $\ln 10 / \ln(1/0.674) \approx 5.8$ steps. A common mistake is to quote the fixed point $0.739$ instead of the rate; four digits ($0.6736$) are not enough for the requested precision.

</details>

**q14.** Newton's method for $f(y) = 1/y^2 - a$ gives $y_{k+1} = y_k (3 - a y_k^2)/2$, the update [Fixed-point iteration and fast inverse square root in C](05-fixed-point-iteration-and-rsqrt.md) uses for `rsqrt`. With $a = 4$ and $y_0 = 0.4$, give $y_1$ exactly. `[number]`

<details><summary>Answer</summary>

$y_1 = 59/125$. $a y_0^2 = 4 \cdot 4/25 = 16/25$, $3 - 16/25 = 59/25$, and $y_1 = \frac{2}{5} \cdot \frac{59}{25} \cdot \frac12 = \frac{59}{125} = 0.472$, approaching $1/\sqrt{4} = 1/2$. A common mistake is $1/2$, the limit rather than one step (and the decimal $0.472$ is the same number, but the exact fraction is asked).

</details>

**q15.** Let $e_k = 1 - \sqrt{a}\, y_k$ be the relative error of that iteration. Using the identity of q17, give $e_1$ exactly when $e_0 = 2^{-4}$. `[number]`

<details><summary>Answer</summary>

$e_1 = 47/8192$. With $e_0 = 1/16$: $\frac{3 e_0^2 - e_0^3}{2} = \frac{3/256 - 1/4096}{2} = \frac{48/4096 - 1/4096}{2} = \frac{47}{8192} \approx 2^{-7.45}$. A common mistake is $3/512$, keeping only the leading term $\frac32 e_0^2$ and dropping $-\frac12 e_0^3$.

</details>

**q16.** (a) Give the order of convergence of Newton's method at a simple root. (b) At a double root Newton converges only linearly; give the rate (the ratio $e_{k+1}/e_k$ in the limit). `[number]`

<details><summary>Answer</summary>

(a) $2$. At a simple root, $g'(x^*) = 0$ and $e_{k+1} \approx \frac{g''(x^*)}{2} e_k^2$.

(b) $1/2$. Near a double root, $f(x) = (x - x^*)^2 h(x)$ with $h(x^*) \ne 0$. Then $f/f' \approx (x - x^*)/2$, so $x_{k+1} - x^* \approx (x_k - x^*) - (x_k - x^*)/2$: the error halves every step. A common mistake is 1 (no convergence) or 2 (still quadratic).

</details>

**q17.** Prove that the iteration of q14 satisfies $e_{k+1} = (3 e_k^2 - e_k^3)/2$, and conclude that it converges quadratically. `[proof]`

<details><summary>Answer</summary>

**Proof.** From the definition, $y_k = (1 - e_k)/\sqrt{a}$, so $a y_k^2 = (1 - e_k)^2$. Substituting into the update and multiplying by $\sqrt{a}$,

$$\sqrt{a}\,y_{k+1} = \frac{(1 - e_k)\bigl(3 - (1 - e_k)^2\bigr)}{2}.$$

Expand: $3 - (1 - e_k)^2 = 2 + 2e_k - e_k^2$, and

$$(1 - e_k)(2 + 2e_k - e_k^2) = 2 + 2e_k - e_k^2 - 2e_k - 2e_k^2 + e_k^3 = 2 - 3e_k^2 + e_k^3.$$

So $\sqrt{a}\,y_{k+1} = 1 - (3e_k^2 - e_k^3)/2$, and reading off $e_{k+1} = 1 - \sqrt{a}\,y_{k+1}$,

$$e_{k+1} = \frac{3e_k^2 - e_k^3}{2}.$$

**Quadratic convergence.** If $\lvert e_0\rvert \le 1/2$, then $\lvert e_{k+1}\rvert \le \lvert e_k\rvert^2 (3 + \lvert e_k\rvert)/2 \le \tfrac{7}{4}\lvert e_k\rvert^2 \le \tfrac{7}{8}\lvert e_k\rvert$, so $e_k \to 0$ and $y_k \to 1/\sqrt{a}$. Then $e_{k+1}/e_k^2 = (3 - e_k)/2 \to 3/2$, a finite non-zero constant: the convergence is quadratic, and the number of correct bits roughly doubles each step (q15: $e_0 = 2^{-4}$ gives $e_1 \approx 2^{-7.4}$). $\blacksquare$

</details>

### Polynomial approximation

**q18.** `expf` reduces $x = k \ln 2 + r$ with $k = \mathrm{round}(x / \ln 2)$, so $e^x = 2^k e^r$. Give $r$ for $x = 10$, exactly. `[number]`

<details><summary>Answer</summary>

$r = 10 - 14\ln 2 \approx 0.29594$. $10/\ln 2 = 14.427$, the nearest integer is $k = 14$, and $r = x - k\ln 2$. A common mistake is $10 - 15\ln 2 \approx -0.397$, which rounds $14.427$ up and leaves $r$ outside $[-\ln 2/2, \ln 2/2]$.

</details>

**q19.** Give the half-width of the interval the reduced argument $r$ lies in. `[number]`

<details><summary>Answer</summary>

$\ln 2 / 2 \approx 0.3466$. Rounding $x/\ln 2$ to the nearest integer leaves a remainder of at most $1/2$ in units of $\ln 2$. A common mistake is $\ln 2$, the bound for truncation (floor), not rounding.

</details>

**q20.** On $\lvert r\rvert \le \ln 2 / 2$, the Taylor polynomial of $e^r$ of degree $n$ has the Lagrange bound $e^{\ln 2/2} (\ln 2/2)^{n+1}/(n+1)!$. Give the smallest $n$ for which it is below $2^{-24}$. `[number]`

<details><summary>Answer</summary>

$n = 7$. With $h = \ln 2/2 \approx 0.3466$ and $e^h = \sqrt 2$: for $n = 6$ the bound is $\sqrt2 \cdot h^7/7! \approx 1.69 \times 10^{-7}$, above $2^{-24} \approx 5.96 \times 10^{-8}$; for $n = 7$ it is $\sqrt2 \cdot h^8/8! \approx 7.3 \times 10^{-9}$, below. This is why [Polynomial approximation and range reduction: expf in C](06-polynomial-approximation-and-expf.md) uses degree 7. A common mistake is 6, which looks close but is about three times too large; 8 is more than needed.

</details>

**q21.** How many multiplications does Horner's rule use for a polynomial of degree 7? `[number]`

<details><summary>Answer</summary>

$7$. Horner's form $a_0 + r(a_1 + r(a_2 + \dots + r\,a_7))$ has one multiplication per degree. A common mistake is $28 = 1 + 2 + \dots + 7$, computing every power $r^j$ from scratch.

</details>

**q22.** Give the three Chebyshev nodes on $[-1, 1]$ (the zeros of $T_3(x) = 4x^3 - 3x$). `[set]`

<details><summary>Answer</summary>

$\{-\sqrt3/2, 0, \sqrt3/2\}$. $4x^3 - 3x = x(4x^2 - 3) = 0$ gives $x = 0$ or $x = \pm\sqrt{3}/2$; equivalently $x_j = \cos\frac{(2j + 1)\pi}{6}$ for $j = 0, 1, 2$. Common mistakes are the equispaced $\{-1, 0, 1\}$ or $\{-1/2, 0, 1/2\}$.

</details>

**q23.** Give $\max_{x \in [-1, 1]} \lvert (x - x_0)(x - x_1)(x - x_2)\rvert$ for those three nodes. `[number]`

<details><summary>Answer</summary>

$1/4$. The node product is the monic polynomial with those roots, $x^3 - \tfrac34 x = T_3(x)/4$, and $\lvert T_3\rvert \le 1$ on $[-1, 1]$ with equality at the endpoints, so the maximum is $1/4 = 2^{-2}$, the $2^{-n}$ of the Chebyshev bound above. A common mistake is $2/(3\sqrt3) \approx 0.385$, the maximum of $\lvert x^3 - x\rvert$ for the equispaced nodes $-1, 0, 1$ (attained at $x = 1/\sqrt3$).

</details>

### Low-precision arithmetic

**q24.** Round $5.3$ to e4m3 (OCP E4M3: bias 7, 3 fraction bits) with round to nearest, ties to even. (a) Give the value. (b) Give the code as an unsigned integer (sign bit, then 4 exponent bits, then 3 fraction bits). `[number]`

<details><summary>Answer</summary>

(a) $11/2 = 5.5$. $\lfloor\log_2 5.3\rfloor = 2$, so the quantum is $2^{2 - 3} = 0.5$; $5.3/0.5 = 10.6$ rounds to $n = 11$, and $11 \cdot 0.5 = 5.5$.

(b) $75$. The code is $((q_e + M + B - 1) \ll M) + n = ((-1 + 3 + 7 - 1) \ll 3) + 11 = 64 + 11 = 75$, binary `0 1001 011`: sign 0, exponent field 9, fraction 3, value $(1 + 3/8) \cdot 2^{9 - 7} = 5.5$.

A common mistake in (a) is $21/4 = 5.25$, which uses a grid twice as fine as E4M3's in $[4, 8)$, or $5$, which truncates. In (b), $11$ is the significand count $n$, not the code, and $83$ puts the exponent field one too high.

</details>

**q25.** (a) Give the largest finite e5m2 value (bias 15, 2 fraction bits, the top exponent field reserved for infinity and NaN). (b) Give the smallest positive e4m3 value. `[number]`

<details><summary>Answer</summary>

(a) $57344 = 1.75 \cdot 2^{15}$. The largest finite exponent field is $30$ (31 is reserved), $30 - 15 = 15$, and the largest significand with 2 fraction bits is $1.11_2 = 1.75$. A common mistake is $61440 = 1.875 \cdot 2^{15}$, which uses 3 fraction bits; $448$ is E4M3's maximum.

(b) $2^{-9}$. The smallest subnormal is one unit of the subnormal quantum $2^{1 - B - M} = 2^{1 - 7 - 3}$. A common mistake is $2^{-6}$, the smallest normal.

</details>

**q26.** Give the unit roundoff of e4m3. `[number]`

<details><summary>Answer</summary>

$2^{-4}$. E4M3 has $3 + 1 = 4$ significant bits, so $u = 2^{-4}$. A common mistake is $2^{-3}$, the gap at 1 (machine epsilon), twice $u$.

</details>

**q27.** An MX block in fp4 E2M1 (largest value 6, $e_{\max} = 2$) has largest magnitude 13. (a) Give its shared scale $X = 2^{\lfloor \log_2 13 \rfloor - e_{\max}}$. (b) Give the dequantized value of the element 13. `[number]`

<details><summary>Answer</summary>

(a) $X = 2$. $\lfloor\log_2 13\rfloor = 3$, so $X = 2^{3 - 2} = 2$. (b) $12$. $13 / 2 = 6.5$ is above FP4's largest value 6, so it saturates to 6, and $6 \cdot 2 = 12$. A common mistake in (a) is $13/6$, a per-tensor floating scale; MX scales are powers of two. In (b), $13$ assumes no rounding and $14 = 7 \cdot 2$ assumes FP4 has a 7.

</details>

**q28.** Which fp8 format covers the wider range of magnitudes? (a) e4m3 (b) e5m2 `[choice]`

<details><summary>Answer</summary>

(b) E5M2. Its five exponent bits span magnitudes from $2^{-16}$ (smallest subnormal) to $57344$, about 32 binades, against E4M3's $2^{-9}$ to $448$, about 18. E4M3 spends the bit on precision instead, which is why weights and activations use E4M3 and gradients often use E5M2.

</details>

## Common mistakes

The wrong answers flagged in the problems are the same mistakes that break real tolerances and kernels. Each maps to a problem:

| Pitfall | Symptom | How to avoid it |
|---|---|---|
| The absolute condition $\lvert f'(x)\rvert$ for the relative one | $e^{50}$ instead of 50 | q1, q2: multiply by $x/f(x)$ |
| Eigenvalues of $A$ for singular values | $5/3$ instead of 3 for a non-symmetric matrix | q4: take square roots of the eigenvalues of $A^\top A$ |
| $\epsilon = 2u$ for $u$ | every bound off by 2; bf16 void at 128 | q7, q8, q10: $u = 2^{-p}$ with $p$ counting the implicit bit |
| Forgetting that products round too | the pairwise bound one $u$ short | q9: one rounding per tree level plus one per product |
| Quoting the fixed point instead of the rate | 0.739 for 0.674 | q13: the rate is $\lvert g'(x^*)\rvert = \sin x^*$ |
| Keeping only the leading term of Newton's error | $3/512$ instead of $47/8192$ | q15: use the exact identity of q17 |
| Rounding $x/\ln 2$ up instead of to nearest | a reduced argument outside $[-\ln 2/2, \ln 2/2]$ | q18: $k$ is the nearest integer |
| Equispaced interpolation nodes | a larger node product (0.385 vs 0.25) | q22, q23: the zeros of $T_{n+1}$ |
| E5M2's or the significand count's grid for E4M3's code | wrong value or code | q24: quantum $2^{\lfloor\log_2 v\rfloor - 3}$, code $((q_e + M + B - 1) \ll M) + n$ |
| A per-tensor float scale where MX wants a power of two | a scale of 13/6 | q27: $X = 2^{\lfloor\log_2 a_{\max}\rfloor - e_{\max}}$ |

## Where this leads

q1 to q12 become code in [Error analysis, condition numbers, and tolerance budgets](03-error-analysis-condition-numbers-and-tolerance-budgets.md) (`relative_condition`, `cond`, `gamma`, `dot_error_bound`), and q24 to q28 become the fp8, FP4, and MX conversions of [FP8 and microscaling formats](04-fp8-and-microscaling-formats.md). [Fixed-point iteration and fast inverse square root in C](05-fixed-point-iteration-and-rsqrt.md) implements q14 to q17's Newton update and relies on its quadratic convergence, and [Polynomial approximation and range reduction: expf in C](06-polynomial-approximation-and-expf.md) implements q18 to q21's range reduction, degree, and Horner evaluation. In *Build Your Own LLM System*, the quantization error budgets of [Quantization: int8, int4 groups, fp8, and the KV cache](../../build-an-llm/p08-inference/05-quantization.md) combine q8's bound with q24's rounding error.

### Further reading

- Higham, *Accuracy and Stability of Numerical Algorithms*, chapters 2 to 4 and 7.
- Trefethen, *Approximation Theory and Approximation Practice* (SIAM, 2013), chapters 1 to 4.
- Muller, *Elementary Functions: Algorithms and Implementation* (3rd ed., 2016), chapters 3 and 11.

## Key terms

- **Relative condition number**: $\lvert x f'(x)/f(x)\rvert$, the factor by which $f$ amplifies relative input errors.
- **$\gamma_k$**: $ku/(1 - ku)$, the bound on the relative error accumulated over $k$ roundings.
- **Linear convergence**: the error shrinks by a constant factor $\lvert g'(x^*)\rvert < 1$ each step.
- **Quadratic convergence**: the error is proportional to the square of the previous error, so correct digits double.
- **Chebyshev nodes**: the zeros of $T_{n+1}$, which minimize the maximum of the interpolation node product on $[-1, 1]$.
- **MX block scale**: a shared power of two that places a block's largest magnitude in the element format's top binade.

## Summary

- The relative condition number $\kappa_f(x) = \lvert x f'(x) / f(x)\rvert$ and the matrix condition number $\kappa_2(A) = \sigma_{\max}/\sigma_{\min}$ say how much the problem amplifies input errors, whatever the algorithm (q1 to q6).
- Every rounding multiplies by $(1 + \delta)$, $\lvert\delta\rvert \le u$, and $k$ of them stay within $\gamma_k = ku/(1 - ku)$; pairwise summation cuts $k$ to $\log_2 k$, and random signs typically give $\sqrt{k}\,u$ (q7 to q12).
- Fixed-point iteration converges linearly at the rate $\lvert g'(x^*)\rvert$; Newton converges quadratically at a simple root, so the correct digits double each step (q13 to q17).
- `expf` reduces $x$ to $r \in [-\ln 2/2, \ln 2/2]$ and evaluates a short polynomial; the Lagrange remainder fixes the degree, and Chebyshev nodes minimize the interpolation error's node product (q18 to q23).
- fp8 and MX formats are the same rounding with few bits: E4M3 has precision $2^{-4}$ and range to 448, E5M2 trades a bit of precision for range to 57344, and an MX block scale is the power of two that puts the block's maximum in the top binade (q24 to q28).
