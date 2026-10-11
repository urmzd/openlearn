# S-M09b problems: condition numbers, error bounds, Newton, polynomial approximation, low precision

Answer every question in `solve/S-M09b.toml` (written by `ol start S-M09b`).
The tag after each question is its answer type: `[number]` is an exact value
(`3/16777213`, `13*2^-24`, `10 - 14*log(2)`) unless the question gives a
tolerance, `[set]` a set `{1, 2}`, `[choice]` one letter, and `[proof]` a
file `solve/S-M09b/qN.md` graded against its rubric.

Notation: $u = 2^{-p}$ is the unit roundoff of a format with $p$
significand bits including the implicit one (float32 $2^{-24}$, bf16
$2^{-8}$, e4m3 $2^{-4}$); $\gamma_k = ku/(1 - ku)$; $\kappa_f(x) = \lvert x f'(x) / f(x)\rvert$
is the relative condition number of a scalar function; $\kappa_2(A) = \sigma_{\max}/\sigma_{\min}$
is the 2-norm condition number of a matrix.

### Condition numbers

**q1.** Give the relative condition number of $f(x) = x - 1$ at $x = 1.001$. `[number]`

**q2.** Give the relative condition number of $f(x) = e^x$ at $x = 50$. `[number]`

**q3.** Give $\kappa_2$ of $\mathrm{diag}(10, 1/10)$. `[number]`

**q4.** Give $\kappa_2$ of $A = \begin{pmatrix} 3 & 0 \\ 4 & 5 \end{pmatrix}$. `[number]`

**q5.** $\kappa_2(A) = 1000$ and $b$ moves by a relative amount $\lVert\delta b\rVert/\lVert b\rVert = 10^{-6}$. Give the bound on the relative change $\lVert\delta x\rVert/\lVert x\rVert$ of the solution of $Ax = b$. `[number]`

**q6.** Prove that if $Ax = b$ and $A(x + \delta x) = b + \delta b$ with $A$ invertible and $b \ne 0$, then $\lVert\delta x\rVert/\lVert x\rVert \le \kappa_2(A)\, \lVert\delta b\rVert/\lVert b\rVert$ in the 2-norm. `[proof]`

### Sum and dot error bounds

**q7.** Give $\gamma_3$ for float32 exactly. `[number]`

**q8.** A float32 dot product of length $k = 4096$ has $\lvert x\rvert \cdot \lvert y\rvert = 1$. Give the first-order bound $k u$ on its error. `[number]`

**q9.** The same dot product with the sum done pairwise: give the first-order bound $(\lceil \log_2 k\rceil + 1)\, u$ (one rounding per product, one per tree level). `[number]`

**q10.** In bf16, give the smallest $k$ for which $k u \ge 1$, where the bound $\gamma_k$ says nothing. `[number]`

**q11.** The statistical rule of thumb replaces $k u$ by $\sqrt{k}\, u$ (errors with random signs add like a random walk). Give it for float32 and $k = 4096$. `[number]`

**q12.** Prove that recursive summation of three numbers satisfies $\lvert \mathrm{fl}(\mathrm{fl}(x_1 + x_2) + x_3) - (x_1 + x_2 + x_3)\rvert \le \gamma_2 (\lvert x_1\rvert + \lvert x_2\rvert + \lvert x_3\rvert)$ under the model $\mathrm{fl}(a + b) = (a + b)(1 + \delta)$, $\lvert\delta\rvert \le u$. `[proof]`

### Fixed point and Newton convergence

**q13.** The iteration $x_{k+1} = \cos x_k$ converges to the fixed point $x^* \approx 0.739085$. Give its asymptotic linear rate $\lvert g'(x^*)\rvert = \sin x^*$ to at least 7 significant digits. `[number, rtol 1e-6]`

**q14.** Newton's method for $f(y) = 1/y^2 - a$ gives $y_{k+1} = y_k (3 - a y_k^2)/2$, the update `M09.5` uses for `rsqrt`. With $a = 4$ and $y_0 = 0.4$, give $y_1$ exactly. `[number]`

**q15.** Let $e_k = 1 - \sqrt{a}\, y_k$ be the relative error of that iteration. Using the identity of q17, give $e_1$ exactly when $e_0 = 2^{-4}$. `[number]`

**q16.** (a) Give the order of convergence of Newton's method at a simple root. (b) At a double root Newton converges only linearly; give the rate (the ratio $e_{k+1}/e_k$ in the limit). `[number]`

**q17.** Prove that the iteration of q14 satisfies $e_{k+1} = (3 e_k^2 - e_k^3)/2$, and conclude that it converges quadratically. `[proof]`

### Polynomial approximation

**q18.** `expf` reduces $x = k \ln 2 + r$ with $k = \mathrm{round}(x / \ln 2)$, so $e^x = 2^k e^r$. Give $r$ for $x = 10$, exactly. `[number]`

**q19.** Give the half-width of the interval the reduced argument $r$ lies in. `[number]`

**q20.** On $\lvert r\rvert \le \ln 2 / 2$, the Taylor polynomial of $e^r$ of degree $n$ has the Lagrange bound $e^{\ln 2/2} (\ln 2/2)^{n+1}/(n+1)!$. Give the smallest $n$ for which it is below $2^{-24}$. `[number]`

**q21.** How many multiplications does Horner's rule use for a polynomial of degree 7? `[number]`

**q22.** Give the three Chebyshev nodes on $[-1, 1]$ (the zeros of $T_3(x) = 4x^3 - 3x$). `[set]`

**q23.** Give $\max_{x \in [-1, 1]} \lvert (x - x_0)(x - x_1)(x - x_2)\rvert$ for those three nodes. `[number]`

### Low-precision arithmetic

**q24.** Round $5.3$ to e4m3 (OCP E4M3: bias 7, 3 fraction bits) with round to nearest, ties to even. (a) Give the value. (b) Give the code as an unsigned integer (sign bit, then 4 exponent bits, then 3 fraction bits). `[number]`

**q25.** (a) Give the largest finite e5m2 value (bias 15, 2 fraction bits, the top exponent field reserved for infinity and NaN). (b) Give the smallest positive e4m3 value. `[number]`

**q26.** Give the unit roundoff of e4m3. `[number]`

**q27.** An MX block in fp4 E2M1 (largest value 6, $e_{\max} = 2$) has largest magnitude 13. (a) Give its shared scale $X = 2^{\lfloor \log_2 13 \rfloor - e_{\max}}$. (b) Give the dequantized value of the element 13. `[number]`

**q28.** Which fp8 format covers the wider range of magnitudes? (a) e4m3 (b) e5m2 `[choice]`
