<!-- ol:module M02.1 -->
# Taylor series, remainder bounds, range reduction

## Overview

| | |
|---|---|
| **Module** | `M02.1` · build · Python · Pass 2 · 3 to 4 h |
| **You build** | `python/tinyllm/num/taylor.py`: `exp_taylor_coeffs`, `exp_range_reduced` (exp as $2^k$ times a polynomial), `erf_series` (erf from a series with no cancellation) |
| **Contract** | [`course/contracts/py/tinyllm/num/taylor.pyi`](../../../../course/contracts/py/tinyllm/num/taylor.pyi) |
| **Tests** | `course/tests/M02.1/test_taylor.py` (what they check: section 4) |
| **Needs** | `M00.4` `horner`, which evaluates the polynomial (or `--ref-deps`). Reading: `M01.1` (derivatives and their error terms) |
| **Used by** | `M01.3` computes the exact GELU with `erf_series` · later `M09.6` ports `exp_range_reduced` to C as `tl_expf` for softmax and SiLU kernels, and `M08.1`'s `dual_erf` uses the same series |
| **Milestone** | `MS-P2` (the Pass 2 gate) |
| **Optional depth** | OpenStax, *Calculus Volume 2* (free), sections 6.3 and 6.4 (Taylor and Maclaurin series, the remainder); Muller, *Elementary Functions: Algorithms and Implementation*, ch. 11 (range reduction, Cody and Waite); Abramowitz and Stegun 7.1.6 (the erf series used here) |

## Key Takeaways

- The **Taylor polynomial** of degree $n$ matches $f$ and its first $n$ derivatives at a point; for $e^x$ at 0 its coefficients are $1/i!$ (`test_coefficients_are_inverse_factorials`).
- The **Lagrange remainder** bounds the error exactly: $|e^r - p_n(r)| \le e^{\max(r, 0)} |r|^{n+1}/(n+1)!$, tiny when $|r|$ is small and useless when it is not (`test_lagrange_bound_holds`).
- **Range reduction** makes $|r|$ small for every input: $e^x = 2^k e^r$ with $k = \operatorname{round}(x/\ln 2)$ and $|r| \le \frac{\ln 2}{2}$; degree 6 is then float32-accurate (`test_hand_example`, `test_degree_six_is_float32_accurate`).
- Subtracting $k \ln 2$ in **two parts** (Cody and Waite) keeps $r$ exact to the last bit even for $k = 1000$ (`test_full_precision_across_the_range`).
- A series can be correct and still useless in floating point: the textbook alternating series for erf cancels catastrophically at $x = 5$; the series with all terms of one sign does not (`test_erf_no_cancellation_at_x_5`).

## How to work this chapter

```bash
ol start M02.1              # stubs python/tinyllm/num/taylor.py into your repo
ol tests M02.1              # read the test catalog first: rung R0, you write no tests here
ol check M02.1              # exit code is the verdict
ol check M02.1 --ref-deps   # only if your M00.4 is not passing yet
ol diff  M02.1              # after passing: your code against the reference
```

---

## 1. Why now

A CPU can add and multiply; it cannot compute $e^x$. Until now numpy has done it for you, and from Pass 6 on your C kernels must do it themselves: softmax (`L9.2`) exponentiates every logit, SiLU (`L9.6`) every activation. The way every math library computes $e^x$ is a polynomial on a small interval plus an exact rescaling, and the polynomial comes from the Taylor series, with an error bound that tells you which degree is enough. The same tools give erf, which your exact GELU (`M01.3`) needs next. This module builds both in Python, where you can watch the error bound hold point by point, so that `M09.6` has a trusted reference to port.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $f^{(i)}(a)$ | the $i$-th derivative of $f$ at $a$ ($f^{(0)} = f$) | `float` |
| $i!$ | $1 \cdot 2 \cdots i$, with $0! = 1$ | integer |
| $p_n(x)$ | the degree-$n$ Taylor polynomial of $f$ at $a$ | `float` |
| $R_n(x)$ | the remainder $f(x) - p_n(x)$ | `float` |
| $x$ | the input of `exp_range_reduced` | float64 array |
| $k$ | $\operatorname{round}(x / \ln 2)$, an integer | int |
| $r$ | the reduced argument $x - k\ln 2$, $\lvert r \rvert \le \frac{\ln 2}{2}$ | float64 array |
| $t_n$ | the $n$-th term of the erf series | float64 array |
| $N$ | the number of erf terms summed (`terms`) | `int` |

### 2.1 Polynomials that match derivatives

Near a point $a$, which polynomial of degree $n$ is the best stand-in for $f$? The one whose value and first $n$ derivatives at $a$ equal $f$'s. Write $p(x) = \sum_{i=0}^{n} c_i (x - a)^i$. Differentiating $i$ times and setting $x = a$ kills every term but one: $p^{(i)}(a) = i!\, c_i$. Matching $p^{(i)}(a) = f^{(i)}(a)$ gives

$$p_n(x) = \sum_{i=0}^{n} \frac{f^{(i)}(a)}{i!} (x - a)^i .$$

For $f = e^x$ at $a = 0$ every derivative is $e^0 = 1$, so $p_n(x) = 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \cdots + \frac{x^n}{n!}$, coefficients $1/i!$. `exp_taylor_coeffs(n)` returns them in **ascending** order of power, $[1/0!, \ldots, 1/n!]$, the order `M00.4`'s `horner` takes (numpy's `polyval` wants the reverse). Computing $c_i = c_{i-1}/i$ avoids forming $i!$, which overflows float64 at $i = 171$.

### 2.2 The remainder

**Taylor's theorem** (Lagrange form): if $f$ has $n + 1$ continuous derivatives, then for some $\xi$ between $a$ and $x$,

$$R_n(x) = f(x) - p_n(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!}(x - a)^{n+1} .$$

You do not know $\xi$, but you can bound $|f^{(n+1)}|$ between $a$ and $x$, and that bounds the error. (`M01.1` used the first terms of this formula to find the error of finite differences; this is the proven version.) For $e^r$ at 0, $f^{(n+1)}(\xi) = e^\xi$ with $\xi$ between 0 and $r$, so relative to $e^r$:

$$\frac{|e^r - p_n(r)|}{e^r} = \frac{e^{\xi - r}\,|r|^{n+1}}{(n+1)!} \le e^{|r| - r}\,\frac{|r|^{n+1}}{(n+1)!} .$$

(If $r > 0$ then $\xi \le r$ and $e^{\xi - r} \le 1$; if $r < 0$ then $\xi \le 0$ and $e^{\xi - r} \le e^{-r} = e^{|r|}$.) `test_lagrange_bound_holds` checks this inequality at 4001 points for every degree from 1 to 12. The bound shrinks factorially in $n$ but grows like $|r|^{n+1}$: at $r = 0.35$ degree 6 gives $1.2 \times 10^{-7}$; at $r = 10$ degree 6 gives 2000. A Taylor polynomial is only good near its center.

### 2.3 Range reduction

Every real $x$ can be written $x = k \ln 2 + r$ with $k$ an integer and $|r| \le \frac{\ln 2}{2} \approx 0.3466$: take $k = \operatorname{round}(x / \ln 2)$. Then

$$e^x = e^{k\ln 2} \cdot e^r = 2^k\, e^r ,$$

and multiplying by $2^k$ is exact in binary floating point: it only changes the exponent field (`np.ldexp(p, k)`). So $e^x$ costs one reduction, one polynomial on $|r| \le 0.3466$, and one exponent adjustment, and the remainder bound of section 2.2 holds for **every** $x$. With degree 6 the relative error is at most $e^{2 \cdot 0.3466}\,0.3466^7/7! = 2.4 \times 10^{-7}$, about two float32 units in the last place: the target of `tl_expf`. Rounding $k$ down (floor) instead of to nearest leaves $r \in [0, \ln 2)$, doubles the largest $|r|$, and multiplies the bound by $2^7 = 128$.

**Subtract $k \ln 2$ exactly.** $\ln 2$ is irrational, so the float64 constant `LN2` is off by up to $\varepsilon/4$ relative. Multiplied by $k = 1000$ that error lands in $r$ as $10^{-13}$, and $e^r$ inherits it: 400 times the float64 rounding level. Cody and Waite split the constant: `LN2_HI` holds the first 32 bits of $\ln 2$ with the low 21 bits zero, so $k \cdot$ `LN2_HI` is exact for $|k| < 2^{21}$, and `LN2_LO` $= \ln 2 -$ `LN2_HI` is tiny. Then $r = (x - k\,\mathrm{LN2\_HI}) - k\,\mathrm{LN2\_LO}$ is accurate to the last bit.

**The edges.** $e^x$ overflows float64 above $x \approx 709.78$ and is below the smallest subnormal under $x \approx -745.1$. Clipping $x$ to $[-750, 710]$ first changes no result and keeps $k$ small; then $+\infty \mapsto \infty$, $-\infty \mapsto 0$, and nan is passed through separately.

### 2.4 A series for erf without cancellation

$\operatorname{erf}(x) = \frac{2}{\sqrt\pi}\int_0^x e^{-t^2}dt$. Integrating the Taylor series of $e^{-t^2}$ term by term gives the textbook series

$$\operatorname{erf}(x) = \frac{2}{\sqrt\pi} \sum_{n \ge 0} \frac{(-1)^n x^{2n+1}}{n!\,(2n+1)} ,$$

which converges for every $x$ but alternates in sign. At $x = 5$ its largest term is about $6 \times 10^{8}$, the sum is below 1, and float64 keeps 16 significant digits of each term: the cancellation leaves about 7 correct digits of the answer. The fix is a different series for the same function. The Taylor series of $e^{x^2}\operatorname{erf}(x)$ has only positive coefficients:

$$\operatorname{erf}(x) = \frac{2}{\sqrt\pi}\, e^{-x^2} \sum_{n \ge 0} t_n, \qquad t_n = \frac{2^n x^{2n+1}}{1 \cdot 3 \cdot 5 \cdots (2n+1)}, \qquad \frac{t_{n+1}}{t_n} = \frac{2x^2}{2n+3} .$$

Every term has the sign of $x$, so nothing cancels, and each term comes from the previous one by one multiplication. **The tail bound**: once the ratio $\rho = \frac{2x^2}{2N+3}$ is below 1, the ratios keep falling, so the omitted terms are at most a geometric series, $\sum_{n \ge N} t_n \le \frac{t_N}{1 - \rho}$ (`M02.2` sums geometric series). `test_erf_tail_bound_holds` checks it.

**Cutoff.** For $|x| \ge 6$, $1 - \operatorname{erf}(x) < 2.2 \times 10^{-17}$, below half the gap between 1 and the float before it, so erf rounds to exactly $\pm 1$. The series would need hundreds of terms there, and near $|x| = 27$ its partial sums overflow while $e^{-x^2}$ underflows to 0, giving $0 \cdot \infty = $ nan. So `erf_series` returns $\operatorname{sign}(x)$ for $|x| \ge 6$. With 120 or more terms the result is within $3 \times 10^{-15}$ of erf everywhere.

## 3. Worked example by hand

**$e^1$ with degree 3.** $k = \operatorname{round}(1/0.693147) = \operatorname{round}(1.4427) = 1$, so $r = 1 - \ln 2 = 0.3068528$.

$$p_3(r) = 1 + r + \frac{r^2}{2} + \frac{r^3}{6} = 1 + 0.3068528 + 0.0470793 + 0.0048155 = 1.3587476 ,$$

$$2^1 \cdot p_3(r) = 2.7174952 \quad \text{vs.} \quad e = 2.7182818 .$$

The relative error is $2.9 \times 10^{-4}$, and the Lagrange bound (here $r > 0$, so the factor is 1) is $r^4/4! = 0.0088658/24 = 3.7 \times 10^{-4}$: the bound holds with little room to spare. Horner's rule evaluates the same polynomial as $1 + r(1 + r(\frac12 + r \cdot \frac16))$. This is `test_hand_example`.

**$\operatorname{erf}(0.5)$ with three terms.** $x^2 = 0.25$, ratio $2x^2 = 0.5$ over $2n + 1$: $t_0 = 0.5$, $t_1 = 0.5 \cdot 0.5/3 = 0.0833333$, $t_2 = 0.0833333 \cdot 0.5/5 = 0.0083333$, sum $0.5916667$. Prefactor $\frac{2}{\sqrt\pi} e^{-0.25} = 1.1283792 \times 0.7788008 = 0.8787826$. Result $0.8787826 \times 0.5916667 = 0.5199464$, against $\operatorname{erf}(0.5) = 0.5204999$. The next term $t_3 = 0.000595$ times the prefactor is $0.000523$, and the tail bound $t_3/(1 - 0.5/9)$ times the prefactor, $0.000554$, covers the actual gap of $0.000553$. This is `test_erf_hand_example`.

## 4. The interface

```python
ERF_CUTOFF: float  # 6.0
def exp_taylor_coeffs(n: int) -> NDArray: ...                  # [1/0!, ..., 1/n!]
def exp_range_reduced(x: ArrayLike, deg: int = 6) -> NDArray: ...
def erf_series(x: ArrayLike, terms: int) -> NDArray: ...
```

All arithmetic is float64 and results have the shape of `x`. `exp_range_reduced` handles $\pm\infty$, nan, overflow, and underflow without warnings. Negative `n`, `deg`, or `terms < 1` raise `ValueError`.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit, smoke | section 3: $2.7174952$, error $2.9 \times 10^{-4}$ inside the bound | you and the tests agree on reduction and polynomial |
| `test_coefficients_are_inverse_factorials` | unit | $[1, 1, \frac12, \frac16, \frac1{24}]$, length $n + 1$, $1/30!$ | the coefficient order `horner` and `M09.6` use |
| `test_lagrange_bound_holds` | property | the bound of section 2.2 at 4001 points, degrees 1 to 12 | the remainder bound is the design's property test |
| `test_degree_six_is_float32_accurate` | golden | relative error at most $2.4 \times 10^{-7}$ on $[-87, 88]$ | `tl_expf`'s accuracy budget |
| `test_full_precision_across_the_range` | golden | degree 20 within $10^{-15}$ near $\pm 700$ | the two-part reduction |
| `test_special_values_and_extremes` | boundary | $\pm\infty$, nan, overflow, underflow, subnormal, no warnings | masked logits are $-\infty$ |
| `test_shape_and_scalars` | unit | scalars, matrices, integer input, `deg < 0` raises | callers pass every shape |
| `test_erf_hand_example` | unit, smoke | section 3: 0.5199464 | the series and its ratio |
| `test_erf_matches_math_erf` | golden | 160 terms within $3 \times 10^{-15}$ of `math.erf` on $[-7, 7]$ | `gelu_erf` must match PyTorch |
| `test_erf_no_cancellation_at_x_5` | boundary | $x = 3, 4, 5, -5.5$ to $3 \times 10^{-15}$ | the alternating series fails here |
| `test_erf_cutoff_and_specials` | boundary | $\pm 1$ at and beyond 6, $\pm\infty$, nan, 0 | GELU feeds $x/\sqrt2$ up to 28 |
| `test_erf_tail_bound_holds` | property | the geometric tail bound at many $N$ and $x$ | the remainder for erf |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. coefficients in the wrong order for the evaluator (numpy's `polyval` is descending, `horner` ascending) | $p(r)$ evaluates the reversed polynomial; at degree 3, $e^1 \approx 0.89$ | `test_hand_example`, `test_lagrange_bound_holds` (mutant `s01`) |
| 2. rounding $k$ down instead of to nearest | $r$ up to 0.69, error $8 \times 10^{-6}$ at degree 6 | `test_degree_six_is_float32_accurate` (mutant `s04`) |
| 3. one rounded $\ln 2$ in the reduction | relative error $8 \times 10^{-14}$ near $x = 700$ | `test_full_precision_across_the_range` (mutant `s05`) |
| 4. the alternating erf series | 7 correct digits at $x = 5$ | `test_erf_no_cancellation_at_x_5` (mutant `s09`) |
| 5. no cutoff for large $\lvert x \rvert$ | nan at $x = 28$, wrong values beyond 6 | `test_erf_cutoff_and_specials` (mutant `s10`) |
| adding the low part of $\ln 2$ instead of subtracting it | relative error $4 \times 10^{-10}$ per unit of $k$ | `test_hand_example` (mutant `s02`) |
| coefficients $1/(i+1)!$ | every value wrong | `test_coefficients_are_inverse_factorials` (mutant `s03`) |
| nan passed through the clip as 0 | $e^{\mathrm{nan}} = 1$ | `test_special_values_and_extremes` (mutant `s06`) |
| an off-by-one in the erf ratio or term count | the hand example's third term is wrong | `test_erf_hand_example` (mutants `s07`, `s08`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M00.4` | `horner(exp_taylor_coeffs(deg), r)` evaluates the polynomial |
| Back | `M01.1` | derivatives and the first terms of Taylor's formula (reading) |
| Forward | `M01.3` | `gelu_erf` and `dgelu_erf` compute $\Phi(x)$ with `erf_series(x / sqrt(2), 160)` |
| Forward | `M08.1` | `dual_erf` differentiates erf in forward mode |
| Forward | `M09.2` | stable softmax and logsumexp: why $e^x$ overflows at 709.78 |
| Forward | `M09.6` | `tl_expf` in C: the same reduction and a degree-6 polynomial in float32, checked against your Python |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `exp_range_reduced` | glibc `exp` and `expf` | a 128-entry table ($e^{j/128}$) so $r$ is even smaller and the polynomial shorter; correctly rounded in most cases | glibc `sysdeps/ieee754/dbl-64/e_exp.c`, `sysdeps/ieee754/flt-32/e_expf.c` |
| Taylor coefficients | minimax polynomials (Remez algorithm) | the polynomial with the smallest maximum error on the interval, not the best at one point: one or two degrees fewer for the same accuracy | Sollya; `M09.6` |
| two-part $\ln 2$ | Payne and Hanek reduction | exact reduction of $\sin$ and $\cos$ for arguments like $10^{300}$, where $\pi$ needs a thousand bits | Muller, *Elementary Functions*, ch. 11 |
| `erf_series` | SLEEF, CUDA `erff` | branch-free vectorized approximations for whole SIMD lanes | SLEEF `src/libm/sleefsimdsp.c` |
