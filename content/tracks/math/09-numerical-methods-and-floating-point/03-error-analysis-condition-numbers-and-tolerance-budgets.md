<!-- ol:module M09.3 -->
# Error analysis, condition numbers, tolerance budgets

## Overview

| | |
|---|---|
| **Module** | `M09.3` · build · Python · Pass 6 · 3 to 4 h |
| **You build** | `python/tinyllm/num/tolerance.py`: `unit_roundoff`, `gamma`, `sum_error_bound`, `dot_error_bound`, `matmul_error_bound`, `assert_close_bounded`, `bound_ratio`, `cond`, `relative_condition`, `fd_error_model`, `optimal_fd_step` |
| **Contract** | [`course/contracts/py/tinyllm/num/tolerance.pyi`](../../../../course/contracts/py/tinyllm/num/tolerance.pyi) |
| **Tests** | `course/tests/M09.3/` (what they check: section 4) |
| **Needs** | `M03.5` (`svd`: `cond` takes its singular values) · `M01.1` (`central_diff`: the tests measure a real difference against your step) · reading: `M09.1` the unit roundoff, `M09.2` summation order, `S-M09a` |
| **Used by** | `L8.5`'s `output_error_bound` budgets quantization error with `matmul_error_bound(W, x^T, dtype, dA=abs(W - dequantize(q)))` (at most `scale/2`) · optional `L9.1` checks its standalone C matmul against fixture values within `matmul_error_bound` · your own differential tests from rung R5 on |
| **Milestone** | `MS-P6` (Pass 6 gate: every math module of the pass checks green) |
| **Optional depth** | Higham, *Accuracy and Stability of Numerical Algorithms* (SIAM, 2nd ed.), ch. 2 to 4 and 7; Trefethen and Bau, *Numerical Linear Algebra*, lectures 12 to 15; Higham and Mary, "A New Approach to Probabilistic Rounding Error Analysis" (SIAM J. Sci. Comput., 2019) |

## Key Takeaways

- One model of rounding, $\mathrm{fl}(a \circ b) = (a \circ b)(1 + \delta)$ with $\lvert\delta\rvert \le u$, chained through $k$ operations, gives $\gamma_k = ku/(1 - ku)$, and with it a bound $\gamma_k \lvert x\rvert \cdot \lvert y\rvert$ on a dot product's error that holds for every input and every summation order (`test_dot_bound_holds_and_is_not_loose`, `test_matmul_bound_covers_any_order`).
- The bound scales with $\lvert x\rvert \cdot \lvert y\rvert$, not with $\lvert x \cdot y\rvert$: when terms cancel, the relative error of the result can be huge while the algorithm is perfectly fine (`test_matmul_bound_covers_any_order`).
- The condition number measures the problem, not the algorithm: $\kappa_2(A) = \sigma_{\max}/\sigma_{\min}$ bounds how much a relative change in $b$ moves the solution of $Ax = b$, and the bound is reached (`test_cond_bounds_the_amplification`).
- A tolerance budget adds what you introduce on purpose (quantization moves each weight by up to half a step) to what rounding adds; a test that only allows rounding fails a correct quantized kernel (`test_matmul_bound_budgets_a_perturbation`).
- Finite differences trade truncation (falls with $h$) against rounding (grows as $1/h$); the optimal step is $\approx 2\sqrt{u}$ forward and $(3u)^{1/3}$ central (`test_fd_model_and_optimal_step`, `test_optimal_step_on_real_differences`).

## How to work this chapter

```bash
ol start M09.3              # stubs tolerance.py into your repo
ol tests M09.3              # read the test catalog first: rung R0, you write no tests here
ol check M09.3              # exit code is the verdict
ol check M09.3 --ref-deps   # only if you skipped M03.5 or M01.1
ol diff  M09.3              # after passing: your code against the reference
```

---

## 1. Why now

Part 8 and Part 9 compare numbers that are not equal and must not be. Your C matmul (`L9.1`) sums in tiles, numpy sums pairwise in blocks, and the two agree only to rounding; your int8 and int4 kernels (`L9.5`) start from weights that were moved on purpose (`L8.5`). A tolerance chosen by eye fails one of two ways. Too tight, and a correct kernel fails on a long row (the error of a dot product grows with its length). Too loose, and a kernel that drops one term of 257 passes. So far the course tests have used the frozen `tests/_lib/close.py`, whose $\sqrt{K}$ rule is a statistical rule of thumb. Before you write kernels whose tests you own, you derive the rigorous version: where the error comes from, how big it can be for a given input, and how much of it is the problem's fault rather than the algorithm's.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $p$ | significand bits including the implicit 1: 53 (f64), 24 (f32), 11 (f16), 8 (bf16), 4 (e4m3), 3 (e5m2) | integer |
| $u = 2^{-p}$ | unit roundoff: the largest relative error of one rounding | float |
| $\mathrm{fl}(\cdot)$ | the floating-point result of an expression | float |
| $\delta_i$ | the relative error of one rounding, $\lvert\delta_i\rvert \le u$ | float |
| $\gamma_k = \dfrac{ku}{1 - ku}$ | the accumulated relative error of $k$ roundings, for $ku < 1$ | float |
| $x, y \in \mathbb{R}^k$ | the vectors of a dot product | `float[k]` |
| $\lvert x\rvert$ | elementwise absolute value | `float[k]` |
| $A, B$ | matrices, $A$ is $m \times k$, $B$ is $k \times n$ | `float[m, k]`, `float[k, n]` |
| $\sigma_{\max}, \sigma_{\min}$ | the largest and smallest singular values (`M03.5`) | float |
| $\kappa_2(A) = \sigma_{\max} / \sigma_{\min}$ | the 2-norm condition number | float $\ge 1$ |
| $\kappa_f(x) = \lvert x f'(x) / f(x)\rvert$ | the relative condition number of a scalar function | float $\ge 0$ |
| $h$ | a finite-difference step | float $> 0$ |

### 2.1 The rounding model

IEEE arithmetic rounds the exact result of each operation to the nearest float (`M09.1`), so for $\circ \in \{+, -, \times, /\}$

$$\mathrm{fl}(a \circ b) = (a \circ b)(1 + \delta), \qquad \lvert\delta\rvert \le u = 2^{-p}.$$

$u$ is half the gap between 1 and the next float: the gap is $2^{1-p}$ (machine epsilon, `np.finfo(np.float32).eps` $= 2^{-23}$), and rounding to the nearest float errs by at most half of it, $2^{-24}$ for float32. Counting the implicit bit matters: bf16 stores 7 fraction bits but keeps 8 significant bits, so its $u$ is $2^{-8}$.

### 2.2 Products of $(1 + \delta)$ and $\gamma_k$

A value that passes through $k$ roundings carries a factor $\prod_{i=1}^k (1 + \delta_i)$. If $ku < 1$,

$$\prod_{i=1}^{k} (1 + \delta_i) = 1 + \theta_k, \qquad \lvert\theta_k\rvert \le \gamma_k = \frac{ku}{1 - ku}.$$

(Induction: $(1 + \theta_{k-1})(1 + \delta_k) = 1 + \theta_{k-1} + \delta_k + \theta_{k-1}\delta_k$, and $\gamma_{k-1} + u + \gamma_{k-1} u \le \gamma_k$.) To first order $\gamma_k \approx ku$; the denominator makes the bound rigorous and says it is void once $ku \ge 1$, which for bf16 happens at $k = 256$.

Now a dot product summed left to right: $s_1 = \mathrm{fl}(x_1 y_1)$, $s_i = \mathrm{fl}(s_{i-1} + \mathrm{fl}(x_i y_i))$. The term $x_1 y_1$ goes through one multiplication and $k - 1$ additions, $x_i y_i$ for $i \ge 2$ through one multiplication and $k - i + 1$ additions; every term goes through at most $k$ roundings, so

$$\lvert \mathrm{fl}(x^\top y) - x^\top y\rvert \le \gamma_k \sum_i \lvert x_i y_i\rvert = \gamma_k\, \lvert x\rvert^\top \lvert y\rvert.$$

The same count holds for any order of the additions (pairwise, blocked, tiled): each term meets at most $k - 1$ additions. That is why one bound covers numpy's BLAS, your tiled C kernel (`L9.1`), and a plain loop. A sum without products ($y = 1$, exact multiplications) needs $\gamma_{k-1}$: one term alone is exact.

### 2.3 Why $\lvert x\rvert \cdot \lvert y\rvert$ and not $\lvert x \cdot y\rvert$

The roundings are relative to the partial sums, and the partial sums can be large even when the result is small. $[1, 10^{-8}, -1] \cdot [1, 1, 1]$ in float32 computes $1 + 10^{-8} = 1$, then $1 - 1 = 0$: the exact result $10^{-8}$ is lost completely, a 100% relative error, and that is correct float32 arithmetic. The bound $\gamma_3 (1 + 10^{-8} + 1) \approx 3.6 \times 10^{-7}$ allows it. A bound proportional to $\lvert x^\top y\rvert = 10^{-8}$ would call a correct kernel broken. For a matrix product the elementwise bound is $\gamma_k (\lvert A\rvert\, \lvert B\rvert)_{ij}$.

### 2.4 Tolerance budgets

A test compares an implementation against a reference. Every source of difference must be in the allowance, and nothing else:

$$\text{allowed}_{ij} = \underbrace{(dA\, \lvert B\rvert)_{ij}}_{\text{weights moved by } \le dA} + \underbrace{\gamma_k \big((\lvert A\rvert + dA)\, \lvert B\rvert\big)_{ij}}_{\text{rounding of the moved product}}.$$

With quantization to a grid of step $s$ (`L8.5`), each weight moves by at most $dA = s/2$. With $dA = 0$ it is the plain rounding bound optional `L9.1` checks its standalone C matmul against. `bound_ratio` reports $\max_{ij} \lvert\text{error}\rvert / \text{allowed}$: at most 1 passes, and how close to 1 it is tells you whether the test can still catch a bug. `assert_close_bounded` multiplies the dot bound by a `slack` (default 4) because the "expected" side is itself computed in floating point (in float64, or in float32 by a different order).

The worst case is rarely reached: rounding errors have random signs and partially cancel, so the typical error grows like $\sqrt{k}\, u$, not $k u$ (Higham and Mary 2019). The frozen `tests/_lib/close.py` that grades your modules uses that statistical rule: tolerances times $\sqrt{K}$. It is tighter and almost always right; the bound here is looser and always right. Use the bound when a false failure would be expensive to debug.

### 2.5 Condition numbers

Error analysis separates two questions. The **backward error** of an algorithm asks: for which nearby input is my computed output the exact answer? The **condition number** of the problem asks: how much does the exact answer move when the input moves? The forward error is at most their product.

For a scalar function, a relative change $\epsilon$ in $x$ changes $f(x)$ by $f'(x)\, x \epsilon$, a relative change of

$$\kappa_f(x) = \left\lvert \frac{x f'(x)}{f(x)} \right\rvert.$$

$\sqrt{x}$ has $\kappa = 1/2$: it halves relative errors. $x - 1$ near $x = 1$ has $\kappa = \lvert x/(x - 1)\rvert$, which is $10^8$ at $1 + 10^{-8}$: the cancellation of `S-M09a` is ill-conditioning, a property of subtraction near equal numbers, not of any algorithm.

For a linear system $Ax = b$ and a perturbation $A(x + \delta x) = b + \delta b$: $A\, \delta x = \delta b$, so $\lVert\delta x\rVert \le \lVert A^{-1}\rVert\, \lVert\delta b\rVert$, and $\lVert b\rVert \le \lVert A\rVert\, \lVert x\rVert$. Multiplying,

$$\frac{\lVert\delta x\rVert}{\lVert x\rVert} \le \lVert A\rVert\, \lVert A^{-1}\rVert\, \frac{\lVert\delta b\rVert}{\lVert b\rVert} = \kappa_2(A)\, \frac{\lVert\delta b\rVert}{\lVert b\rVert}$$

in the 2-norm, where $\lVert A\rVert_2 = \sigma_{\max}$ and $\lVert A^{-1}\rVert_2 = 1/\sigma_{\min}$. Equality holds when $b$ lies along the top left singular vector and $\delta b$ along the bottom one. Eigenvalues are not a substitute: $\begin{pmatrix} 1 & 10^3 \\ 0 & 1 \end{pmatrix}$ has both eigenvalues 1 and $\kappa_2 \approx 10^6$. A matrix whose $\sigma_{\min}$ is below $\max(m, n) \cdot 2^{-52} \sigma_{\max}$ is singular to working precision, and `cond` returns $\infty$ rather than a meaningless $10^{17}$.

### 2.6 The optimal finite-difference step

`M01.1`'s forward difference $(f(x + h) - f(x))/h$ has truncation error $\tfrac{h}{2}\lvert f''\rvert$ (Taylor) and rounding error up to $2u\lvert f\rvert / h$ (two evaluations, each off by $u\lvert f\rvert$, divided by $h$). The total

$$E_1(h) = \frac{h}{2} D + \frac{2uF}{h}, \qquad \frac{dE_1}{dh} = \frac{D}{2} - \frac{2uF}{h^2} = 0 \;\Rightarrow\; h^* = 2\sqrt{uF/D},$$

with $F = \lvert f\rvert$ and $D = \lvert f''\rvert$. The central difference $(f(x + h) - f(x - h))/(2h)$ has truncation $\tfrac{h^2}{6}\lvert f'''\rvert$ and rounding $uF/h$:

$$E_2(h) = \frac{h^2}{6} D + \frac{uF}{h}, \qquad \frac{D h}{3} - \frac{uF}{h^2} = 0 \;\Rightarrow\; h^* = (3uF/D)^{1/3}.$$

In float64 with unit scales that is about $2.1 \times 10^{-8}$ and $6.9 \times 10^{-6}$, and the best achievable errors are about $10^{-8}$ and $10^{-11}$: the central difference wins by three orders of magnitude, which is why the frozen gradcheck uses it.

## 3. Worked example by hand

**A dot product that loses two terms.** $x = [1, 10^{-8}, 10^{-8}]$, $y = [1, 1, 1]$, all float32. Recursive summation: $1 + 10^{-8}$ rounds to 1 (the gap at 1 is $2^{-23} \approx 1.19 \times 10^{-7}$, and $10^{-8}$ is less than half of it), and so does the next addition. The computed result is 1; the exact result is $1.00000002$; the error is $2 \times 10^{-8}$.

The bound: $u = 2^{-24} \approx 5.96 \times 10^{-8}$, $\gamma_3 = 3u/(1 - 3u) \approx 1.7881 \times 10^{-7}$, $\lvert x\rvert \cdot \lvert y\rvert = 1.00000002$, so the allowance is $1.7881 \times 10^{-7}$. The error is about 0.11 of the allowance: inside, as it must be.

**An ill-conditioned system.** $A = \begin{pmatrix} 1 & 1 \\ 1 & 1.0001 \end{pmatrix}$ is symmetric, so its singular values are its eigenvalues, $\lambda = \tfrac12\big(2.0001 \pm \sqrt{4 + 10^{-8}}\big)$: about $2.00005$ and $4.99988 \times 10^{-5}$. So $\kappa_2(A) \approx 40002$. With $b = [2, 2.0001]$ the solution is $x = [1, 1]$. Move $b$ to $[2, 2.0002]$, a relative change of $10^{-4} / \lVert b\rVert = 10^{-4}/2.8285 = 3.54 \times 10^{-5}$; the solution becomes $[0, 2]$, a relative change of $\lVert[-1, 1]\rVert / \lVert[1, 1]\rVert = 1$. The amplification is $1 / 3.54 \times 10^{-5} \approx 28284$, below 40002 as the theory says. No algorithm can do better than this problem allows.

These are the first cases in section 4: `test_hand_example` and `test_hand_example_condition`.

## 4. The interface

```python
# python/tinyllm/num/tolerance.py
def unit_roundoff(dtype: str) -> float                                   # "f32" -> 2^-24
def gamma(k: int, dtype: str) -> float                                   # k u / (1 - k u)
def sum_error_bound(k: int, dtype: str, abs_sum) -> NDArray              # gamma_(k-1) * sum|x|
def dot_error_bound(k: int, dtype: str, abs_dot) -> NDArray              # gamma_k * |x|.|y|
def matmul_error_bound(A, B, dtype: str, dA=0.0) -> NDArray              # dA|B| + gamma_k (|A| + dA)|B|
def assert_close_bounded(actual, expected, k: int, dtype: str, abs_dot, slack: float = 4.0) -> None
def bound_ratio(actual, expected, bound) -> float                        # max |error| / bound
def cond(A) -> float                                                     # sigma_max / sigma_min via M03.5
def relative_condition(f, df, x: float) -> float                         # |x f'(x) / f(x)|
def fd_error_model(h, order, dtype, f_scale=1.0, deriv_scale=1.0) -> float
def optimal_fd_step(order, dtype, f_scale=1.0, deriv_scale=1.0) -> float
```

`dtype` is one of `"f64"`, `"f32"`, `"f16"`, `"bf16"`, `"e4m3"`, `"e5m2"` or the numpy names. These helpers decide only this module's verdict: course tests elsewhere assert through the frozen `tests/_lib/close.py` (D35), and you use these in your own tests.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit | section 3's dot product, $u$, $\gamma_3$, the bound | you and the test agree on the definitions |
| `test_hand_example_condition` | unit | section 3's system: $\kappa_2 \approx 40002$, amplification 28284 | the meaning of a condition number |
| `test_unit_roundoff_table` | unit | $u$ for six formats and the numpy names; $u = \epsilon/2$ | every bound scales with it |
| `test_gamma_edges` | boundary | $\gamma_0 = 0$, exact formula, void at $ku \ge 1$ | bf16 sums of 256 terms have no bound |
| `test_dot_bound_holds_and_is_not_loose` | property | 1000 float32 dots hold the bound; median bound/error under 100 | a tolerance that never false-fails and still catches |
| `test_sum_bound_holds` | property | float16 running sums within $\gamma_{k-1}\sum\lvert x\rvert$; $k = 1$ is exact | the count of roundings |
| `test_bounds_are_elementwise_arrays` | unit | array in, float64 array out | kernel tests pass whole matrices |
| `test_matmul_bound_covers_any_order` | property | BLAS, long sequential sums, and cancellation stay inside | `L9.1` tiles in its own order |
| `test_matmul_bound_budgets_a_perturbation` | property | quantized weights pass with $dA = s/2$, fail without | `L8.5`'s quantization budget |
| `test_assert_close_bounded_passes_and_fails` | unit | a correct dot passes, one dropped term fails, slack scales | the helper is a verdict |
| `test_assert_close_bounded_special_values` | boundary | NaN, infinities, zero bounds, shape mismatch | overflowed kernels compare sanely |
| `test_bound_ratio` | unit | the worst element, $0/0 = 0$, $x/0 = \infty$ | optional C parity reports one number per operation |
| `test_cond_matches_numpy` | golden | LAPACK on square, tall, wide, Hilbert, non-normal | an independent implementation agrees |
| `test_cond_edges` | boundary | identity, scale invariance, orthogonal, singular gives $\infty$ | no meaningless $10^{17}$ |
| `test_cond_bounds_the_amplification` | property | amplification $\le \kappa_2$, reached at the singular directions | the theorem of section 2.5 |
| `test_relative_condition` | unit | $\sqrt{x}$, $\log$, $x - 1$ near 1, zeros | cancellation is conditioning |
| `test_fd_model_and_optimal_step` | unit | both models, both optimal steps, minimality | `gradcheck` steps |
| `test_optimal_step_on_real_differences` | property | `M01.1`'s central difference is best near $h^*$ | the model predicts real behavior |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. using $\epsilon = 2^{1-p}$ for $u$, or forgetting the implicit bit | every tolerance off by 2 | `test_unit_roundoff_table` (mutants `s01`, `s03`) |
| 2. $\gamma_k = ku$ without the denominator | a bound that stays finite where it is void | `test_gamma_edges` (mutant `s02`) |
| 3. the wrong count of roundings, or $\lvert A B\rvert$ for $\lvert A\rvert\lvert B\rvert$ | a correct kernel fails on long or cancelling rows | `test_dot_bound_holds_and_is_not_loose` (mutant `s04`), `test_matmul_bound_covers_any_order` (mutant `s20`) |
| 4. a budget without the deliberate perturbation | a correct int4 kernel fails its test | `test_matmul_bound_budgets_a_perturbation` (mutants `s06`, `s07`) |
| 5. ignoring the slack on the expected side | flaky failures at long $k$ | `test_assert_close_bounded_passes_and_fails` (mutant `s09`) |
| 6. treating a numerically singular matrix as finite | $\kappa = 10^{17}$ reported as a number | `test_cond_edges` (mutant `s11`) |
| 7. eigenvalues instead of singular values | non-normal matrices look well conditioned | `test_cond_matches_numpy` (mutant `s12`) |
| 8. the absolute condition $\lvert f'/f\rvert$ | a scale-dependent number | `test_relative_condition` (mutant `s14`) |
| 9. $\sqrt{u}$ for the central difference | a step 100 times too small, ten times the error | `test_fd_model_and_optimal_step` (mutant `s16`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M03.5` | `svd` gives the singular values `cond` divides |
| Back | `M01.1` | `central_diff` is the difference whose step section 2.6 optimizes |
| Back | `M09.1` | $u$ and the formats |
| Forward | `L8.5` | `output_error_bound(w, q, x)`: `matmul_error_bound(W, x^T, "f32", dA=abs(W - dequantize(q)))`, the quantization step (at most `scale/2`) plus the f32 rounding |
| Forward | `L9.1` | standalone C matmul checked against fixture values within `matmul_error_bound` |
| Forward | `L9.1` | the tiled matmul's own differential test, if you write it with these bounds |

If you skip this module, `L8.5` still needs its error budget; optional `L9.1` can be studied later.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `assert_close_bounded` | PyTorch `torch.testing.assert_close` | per-dtype default tolerances, NaN and device handling (a fixed table, not a bound) | `torch/testing/_comparison.py` |
| `dot_error_bound` | probabilistic error analysis | $\lambda\sqrt{k}\,u$ bounds that hold with probability $1 - \delta$ | Higham and Mary (2019) |
| `cond` | `numpy.linalg.cond`, LAPACK `xGECON` | estimates $\kappa_1$ in $O(n^2)$ from an LU factorization instead of an SVD | LAPACK `dgecon.f` |
| `optimal_fd_step` | JAX and PyTorch gradcheck | central differences at a fixed $\epsilon \approx 10^{-6}$ in float64 | `torch/autograd/gradcheck.py` |
