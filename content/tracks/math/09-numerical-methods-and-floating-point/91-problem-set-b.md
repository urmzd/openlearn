<!-- ol:module S-M09b -->
# Floating point problem set, part b: condition numbers, error bounds, Newton, polynomial approximation, low precision

## Overview

| | |
|---|---|
| **Module** | `S-M09b` · solve · none · Pass 6 · 4 to 5 h |
| **You build** | answers in `solve/S-M09b.toml` (29 checked by SymPy) and 3 proofs in `solve/S-M09b/q6.md`, `q12.md`, `q17.md` (self-graded against their rubrics) |
| **Contract** | none: a pen and paper set |
| **Tests** | `course/solve/S-M09b/key.toml` (hidden): typed answers plus reject canaries; the problems are in `course/solve/S-M09b/problems.md` and in section 4 |
| **Needs** | no code. Reading: `S-M09a` (representation and rounding), `M09.3` (the error bounds and condition numbers as code), `M09.4` (the fp8 and MX formats as code), `M01.2` (Newton's method) |
| **Used by** | no call site (a solve set). Do it beside `M09.3` and `M09.4`; q14 to q17 prepare `M09.5` (Newton `rsqrt` in C) and q18 to q20 prepare `M09.6` (`expf` in C) |
| **Milestone** | `MS-P6` (the Pass 6 gate runs `ol check` on every solve part of the pass) |
| **Optional depth** | Higham, *Accuracy and Stability of Numerical Algorithms*, ch. 2 to 4 and 7; Trefethen, *Approximation Theory and Approximation Practice* (SIAM, 2013), ch. 1 to 4; Muller, *Elementary Functions: Algorithms and Implementation* (3rd ed., 2016), ch. 3 and 11 |

## Key Takeaways

- The relative condition number $\kappa_f(x) = \lvert x f'(x) / f(x)\rvert$ and the matrix condition number $\kappa_2(A) = \sigma_{\max}/\sigma_{\min}$ say how much the problem amplifies input errors, whatever the algorithm (q1 to q6).
- Every rounding multiplies by $(1 + \delta)$, $\lvert\delta\rvert \le u$, and $k$ of them stay within $\gamma_k = ku/(1 - ku)$; pairwise summation cuts $k$ to $\log_2 k$, and random signs typically give $\sqrt{k}\,u$ (q7 to q12).
- Fixed-point iteration converges linearly at the rate $\lvert g'(x^*)\rvert$; Newton converges quadratically at a simple root, so the correct digits double each step (q13 to q17).
- `expf` reduces $x$ to $r \in [-\ln 2/2, \ln 2/2]$ and evaluates a short polynomial; the Lagrange remainder fixes the degree, and Chebyshev nodes minimize the interpolation error's node product (q18 to q23).
- fp8 and MX formats are the same rounding with few bits: E4M3 has precision $2^{-4}$ and range to 448, E5M2 trades a bit of precision for range to 57344, and an MX block scale is the power of two that puts the block's maximum in the top binade (q24 to q28).

## How to work this chapter

```bash
ol start S-M09b             # writes solve/S-M09b.toml and the proof files
ol check S-M09b             # SymPy checks the answers, then asks each proof rubric (y/n)
ol check S-M09b --regrade   # ask the rubrics again after you change a proof
```

---

## 1. Why now

Part 8 compares numbers that should agree but were computed differently: your sampler's float64 sums against the Rust engine's, a quantized matmul against float32, a float16 KV cache against a float32 one. Part 9 then writes `rsqrt` by Newton's method and `expf` by range reduction and a polynomial, in C, and must prove them accurate to a few ulps. `M09.3` and `M09.4` turn the bounds and the formats into code; this set makes you derive them by hand first, so a failing tolerance or a kernel that is off by one ulp is something you can explain rather than tune.

## 2. Principles

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

### 2.1 Conditioning

A relative change $\epsilon$ in the input changes the output of $f$ by about $\kappa_f(x)\,\epsilon$ (first-order Taylor). For $Ax = b$, $\lVert\delta x\rVert/\lVert x\rVert \le \kappa_2(A)\, \lVert\delta b\rVert/\lVert b\rVert$, with $\lVert A\rVert_2 = \sigma_{\max}$ and $\lVert A^{-1}\rVert_2 = 1/\sigma_{\min}$. The singular values of $A$ are the square roots of the eigenvalues of $A^\top A$, not the eigenvalues of $A$ (they coincide only for symmetric positive semidefinite $A$).

### 2.2 Error bounds

Under $\mathrm{fl}(a \circ b) = (a \circ b)(1 + \delta)$, $\lvert\delta\rvert \le u$, a dot product of length $k$ in any order satisfies $\lvert\mathrm{fl}(x^\top y) - x^\top y\rvert \le \gamma_k\, \lvert x\rvert^\top\lvert y\rvert$. To first order $\gamma_k \approx ku$. A balanced (pairwise) tree puts each term through $\lceil\log_2 k\rceil$ additions instead of up to $k - 1$. Rounding errors with independent random signs add like a random walk, so the typical error is about $\sqrt{k}\,u$; the worst case is $ku$.

### 2.3 Iterations

If $g$ is differentiable near a fixed point $x^*$, then $e_{k+1} = g(x_k) - g(x^*) \approx g'(x^*)\, e_k$: linear convergence with rate $\lvert g'(x^*)\rvert$ when it is below 1. Newton's iteration for a root of $f$ is $g(x) = x - f(x)/f'(x)$; at a simple root $g'(x^*) = 0$ and $e_{k+1} \approx \tfrac{g''(x^*)}{2} e_k^2$: quadratic. At a double root $f'(x^*) = 0$ too and the rate degrades to linear.

### 2.4 Polynomial approximation

Taylor's theorem with the Lagrange remainder: $e^r = \sum_{j=0}^{n} r^j/j! + R_n$ with $\lvert R_n\rvert \le e^{\lvert r\rvert}\lvert r\rvert^{n+1}/(n+1)!$. Range reduction $x = k\ln 2 + r$, $k = \mathrm{round}(x/\ln 2)$, keeps $\lvert r\rvert \le \ln 2/2$, so a short polynomial suffices, and $e^x = 2^k e^r$ is an exponent adjustment. Horner's rule $a_0 + r(a_1 + r(a_2 + \dots))$ uses $n$ multiplications for degree $n$. Interpolating at $n + 1$ nodes leaves an error proportional to $\prod_i (x - x_i)$; the Chebyshev nodes (zeros of $T_{n+1}$) make that product the monic $T_{n+1}/2^n$, whose maximum on $[-1, 1]$ is $2^{-n}$, the smallest possible.

### 2.5 Low precision

A format with $M$ fraction bits and bias $B$ has quantum $2^{\max(\lfloor\log_2 v\rfloor, 1 - B) - M}$ near $v$; rounding is to the nearest multiple, ties to the even significand. E4M3 ($M = 3$, $B = 7$, max 448), E5M2 ($M = 2$, $B = 15$, max 57344, top exponent reserved), FP4 E2M1 (values 0, 0.5, 1, 1.5, 2, 3, 4, 6). An MX block scale is $X = 2^{\lfloor\log_2 a_{\max}\rfloor - e_{\max}}$, with $e_{\max}$ the exponent of the element format's largest normal.

## 3. Worked example by hand

These are siblings of q1, q14, and q24, not graded problems.

**Conditioning of $\sqrt{x}$.** $f'(x) = 1/(2\sqrt{x})$, so $\kappa = \lvert x \cdot \tfrac{1}{2\sqrt{x}} / \sqrt{x}\rvert = \tfrac12$ for every $x > 0$: square roots halve relative errors.

**One Newton step for $1/\sqrt{a}$.** With $a = 2$ and $y_0 = 0.7$: $y_1 = 0.7\,(3 - 2 \cdot 0.49)/2 = 0.7 \cdot 2.02/2 = 0.707$. The relative error drops from $1 - \sqrt{2} \cdot 0.7 = 0.01005$ to $1 - \sqrt{2}\cdot 0.707 = 0.000151$, close to $\tfrac32 (0.01005)^2 = 0.000152$: quadratic.

**2.75 to E4M3.** $\lfloor\log_2 2.75\rfloor = 1$, quantum $2^{1-3} = 0.25$, $2.75/0.25 = 11$ exactly: value 2.75, code $((-2 + 3 + 7 - 1) \ll 3) + 11 = 56 + 11 = 67$. Decode: $e_f = 8$, $f = 3$, $(8 + 3)\cdot 2^{8 - 7 - 3} = 2.75$.

## 4. The problem set

Write each answer in `solve/S-M09b.toml`; lettered parts are their own tables:

```toml
[q1]
answer = "1001"
[q7]
answer = "3/(2^24 - 3)"
[q16.b]
answer = "1/2"
[q6]
proof = "S-M09b/q6.md"
```

Give exact values as integers, fractions, powers of two, or expressions in `log`; only q13 accepts a decimal, within a relative tolerance of $10^{-6}$.

<!-- ol:problems S-M09b -->

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

<!-- /ol:problems -->

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| The absolute condition $\lvert f'(x)\rvert$ for the relative one | $e^{50}$ instead of 50 | q2 (canary exp(50)) |
| Eigenvalues of $A$ for singular values | $5/3$ instead of 3 for a non-symmetric matrix | q4 (canary 5/3) |
| $\epsilon = 2u$ for $u$ | every bound off by 2; bf16 void at 128 | q7, q8, q10 (canaries with 2^-23, 2^-11, 128) |
| Forgetting that products round too | the pairwise bound one $u$ short | q9 (canary 12*2^-24) |
| Quoting the fixed point instead of the rate | 0.739 for 0.674 | q13 (canary) |
| Keeping only the leading term of Newton's error | $3/512$ instead of $47/8192$ | q15 (canary) |
| Rounding $x/\ln 2$ up instead of to nearest | a reduced argument outside $[-\ln 2/2, \ln 2/2]$ | q18 (canary 10 - 15*log(2)) |
| Equispaced interpolation nodes | a larger node product (0.385 vs 0.25) | q22, q23 (canaries) |
| E5M2's or the significand count's grid for E4M3's code | wrong value or code | q24 (canaries 21/4, 11, 83) |
| A per-tensor float scale where MX wants a power of two | a scale of 13/6 | q27 (canary 13/6) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `S-M09a` | representation, rounding, and cancellation |
| Back | `M01.2` | Newton's method as code, behind q14 to q17 |
| Forward | `M09.3` | q1 to q12 as code: `relative_condition`, `cond`, `gamma`, `dot_error_bound` |
| Forward | `M09.4` | q24 to q28 as code: fp8, FP4, and MX scales |
| Forward | `M09.5` | `rsqrt` in C: q14 to q17's Newton update and its quadratic convergence |
| Forward | `M09.6` | `expf` in C: q18 to q21's range reduction, degree, and Horner evaluation |
| Forward | `L8.5` | quantization error budgets combine q8 with q24's rounding error |
