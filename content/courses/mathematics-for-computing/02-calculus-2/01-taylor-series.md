# Taylor series, remainder bounds, range reduction

## Overview

| | |
|---|---|
| **Module** | `M02.1` · build · Python · Pass 2 · 1.5 to 2 h reading |
| **Solution** | [`course/solutions/python/tinyllm/num/taylor.py`](../../../../course/solutions/python/tinyllm/num/taylor.py): `exp_taylor_coeffs`, `exp_range_reduced` (exp as $2^k$ times a polynomial), `erf_series` (erf from a series with no cancellation) |
| **Contract** | [`course/contracts/py/tinyllm/num/taylor.pyi`](../../../../course/contracts/py/tinyllm/num/taylor.pyi) |
| **Needs** | `M00.4` `horner`, which evaluates the polynomial. Reading: `M01.1` (derivatives and their error terms) |
| **Used by** | `M01.3` computes the exact GELU with `erf_series` · later `M09.6` ports `exp_range_reduced` to C as `tl_expf` for softmax and SiLU kernels, and `M08.1`'s `dual_erf` uses the same series |
| **Optional depth** | OpenStax, *Calculus Volume 2* (free), sections 6.3 and 6.4 (Taylor and Maclaurin series, the remainder); Muller, *Elementary Functions: Algorithms and Implementation*, ch. 11 (range reduction, Cody and Waite); Abramowitz and Stegun 7.1.6 (the erf series used here) |

## Key Takeaways

- The **Taylor polynomial** of degree $n$ matches $f$ and its first $n$ derivatives at a point; for $e^x$ at 0 its coefficients are $1/i!$.
- The **Lagrange remainder** bounds the error exactly: $|e^r - p_n(r)| \le e^{\max(r, 0)} |r|^{n+1}/(n+1)!$, tiny when $|r|$ is small and useless when it is not.
- **Range reduction** makes $|r|$ small for every input: $e^x = 2^k e^r$ with $k = \operatorname{round}(x/\ln 2)$ and $|r| \le \frac{\ln 2}{2}$; degree 6 is then float32-accurate.
- Subtracting $k \ln 2$ in **two parts** (Cody and Waite) keeps $r$ exact to the last bit even for $k = 1000$.
- A series can be correct and still useless in floating point: the textbook alternating series for erf cancels catastrophically at $x = 5$; the series with all terms of one sign does not.

## How to read this chapter

Sections 1 and 2 derive Taylor polynomials, the Lagrange remainder, range reduction for $e^x$, and a cancellation-free series for erf from first principles. Section 3 works both by hand. Section 4 fixes the interface, and section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

A CPU can add and multiply; it cannot compute $e^x$. Until now numpy has done it, and from Pass 6 on the C kernels must do it themselves: softmax (`L9.2`) exponentiates every logit, SiLU (`L9.6`) every activation. The way every math library computes $e^x$ is a polynomial on a small interval plus an exact rescaling, and the polynomial comes from the Taylor series, with an error bound that tells you which degree is enough. The same tools give erf, which the exact GELU (`M01.3`) needs next. This module builds both in Python, where you can watch the error bound hold point by point, so that `M09.6` has a trusted reference to port.

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

(If $r > 0$ then $\xi \le r$ and $e^{\xi - r} \le 1$; if $r < 0$ then $\xi \le 0$ and $e^{\xi - r} \le e^{-r} = e^{|r|}$.) The inequality holds at every point and every degree, so it is a property any correct implementation can be checked against. The bound shrinks factorially in $n$ but grows like $|r|^{n+1}$: at $r = 0.35$ degree 6 gives $1.2 \times 10^{-7}$; at $r = 10$ degree 6 gives 2000. A Taylor polynomial is only good near its center.

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

Every term has the sign of $x$, so nothing cancels, and each term comes from the previous one by one multiplication. **The tail bound**: once the ratio $\rho = \frac{2x^2}{2N+3}$ is below 1, the ratios keep falling, so the omitted terms are at most a geometric series, $\sum_{n \ge N} t_n \le \frac{t_N}{1 - \rho}$ (`M02.2` sums geometric series).

**Cutoff.** For $|x| \ge 6$, $1 - \operatorname{erf}(x) < 2.2 \times 10^{-17}$, below half the gap between 1 and the float before it, so erf rounds to exactly $\pm 1$. The series would need hundreds of terms there, and near $|x| = 27$ its partial sums overflow while $e^{-x^2}$ underflows to 0, giving $0 \cdot \infty = $ nan. So `erf_series` returns $\operatorname{sign}(x)$ for $|x| \ge 6$. With 120 or more terms the result is within $3 \times 10^{-15}$ of erf everywhere.

## 3. Worked example by hand

**$e^1$ with degree 3.** $k = \operatorname{round}(1/0.693147) = \operatorname{round}(1.4427) = 1$, so $r = 1 - \ln 2 = 0.3068528$.

$$p_3(r) = 1 + r + \frac{r^2}{2} + \frac{r^3}{6} = 1 + 0.3068528 + 0.0470793 + 0.0048155 = 1.3587476 ,$$

$$2^1 \cdot p_3(r) = 2.7174952 \quad \text{vs.} \quad e = 2.7182818 .$$

The relative error is $2.9 \times 10^{-4}$, and the Lagrange bound (here $r > 0$, so the factor is 1) is $r^4/4! = 0.0088658/24 = 3.7 \times 10^{-4}$: the bound holds with little room to spare. Horner's rule evaluates the same polynomial as $1 + r(1 + r(\frac12 + r \cdot \frac16))$.

**$\operatorname{erf}(0.5)$ with three terms.** $x^2 = 0.25$, ratio $2x^2 = 0.5$ over $2n + 1$: $t_0 = 0.5$, $t_1 = 0.5 \cdot 0.5/3 = 0.0833333$, $t_2 = 0.0833333 \cdot 0.5/5 = 0.0083333$, sum $0.5916667$. Prefactor $\frac{2}{\sqrt\pi} e^{-0.25} = 1.1283792 \times 0.7788008 = 0.8787826$. Result $0.8787826 \times 0.5916667 = 0.5199464$, against $\operatorname{erf}(0.5) = 0.5204999$. The next term $t_3 = 0.000595$ times the prefactor is $0.000523$, and the tail bound $t_3/(1 - 0.5/9)$ times the prefactor, $0.000554$, covers the actual gap of $0.000553$.

## 4. The interface

```python
ERF_CUTOFF: float  # 6.0
def exp_taylor_coeffs(n: int) -> NDArray: ...                  # [1/0!, ..., 1/n!]
def exp_range_reduced(x: ArrayLike, deg: int = 6) -> NDArray: ...
def erf_series(x: ArrayLike, terms: int) -> NDArray: ...
```

All arithmetic is float64 and results have the shape of `x`. `exp_range_reduced` handles $\pm\infty$, nan, overflow, and underflow without warnings. Negative `n`, `deg`, or `terms < 1` raise `ValueError`.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked $e^1$ | unit | section 3: $2.7174952$, error $2.9 \times 10^{-4}$ inside the bound | everyone agrees on reduction and polynomial |
| inverse-factorial coefficients | unit | $[1, 1, \frac12, \frac16, \frac1{24}]$, length $n + 1$, $1/30!$ | the coefficient order `horner` and `M09.6` use |
| the Lagrange bound | property | the bound of section 2.2 at 4001 points, degrees 1 to 12 | the remainder bound is the design's central property |
| degree 6 is float32-accurate | golden | relative error at most $2.4 \times 10^{-7}$ on $[-87, 88]$ | `tl_expf`'s accuracy budget |
| full precision across the range | golden | degree 20 within $10^{-15}$ near $\pm 700$ | the two-part reduction |
| special values and extremes | boundary | $\pm\infty$, nan, overflow, underflow, subnormal, no warnings | masked logits are $-\infty$ |
| shapes and scalars | unit | scalars, matrices, integer input, `deg < 0` raises | callers pass every shape |
| the hand-worked $\operatorname{erf}(0.5)$ | unit | section 3: 0.5199464 | the series and its ratio |
| erf against `math.erf` | golden | 160 terms within $3 \times 10^{-15}$ of `math.erf` on $[-7, 7]$ | `gelu_erf` must match PyTorch |
| no cancellation at $x = 5$ | boundary | $x = 3, 4, 5, -5.5$ to $3 \times 10^{-15}$ | the alternating series fails here |
| erf cutoff and special values | boundary | $\pm 1$ at and beyond 6, $\pm\infty$, nan, 0 | GELU feeds $x/\sqrt2$ up to 28 |
| the erf tail bound | property | the geometric tail bound at many $N$ and $x$ | the remainder for erf |

## 5. The solution, step by step

### From first principles

1. **Coefficients by recurrence, ascending.** $c_i = 1/i!$ (section 2.1), and $c_i = c_{i-1}/i$ builds them with one division each and never forms $i!$, which overflows at 171. Ascending order is what `horner` takes.
2. **Reduce first, so the bound is small everywhere.** The remainder grows like $|r|^{n+1}$ (section 2.2), so the polynomial must only ever see small arguments. Writing $x = k\ln 2 + r$ with $k = \operatorname{round}(x/\ln 2)$ bounds $|r| \le \frac{\ln 2}{2}$ for every $x$, and $e^x = 2^k e^r$ with an exact rescaling.
3. **Round to nearest, not down.** Nearest rounding centres $r$ on 0; flooring puts it in $[0, \ln 2)$, doubles its largest size, and multiplies the degree-6 bound by $2^7$.
4. **Subtract $k\ln 2$ in two exact pieces.** With `LN2_HI` short enough that $k \cdot$ `LN2_HI` has no rounding, $x - k \cdot \text{LN2\_HI}$ is (nearly) exact, and the small correction $k \cdot$ `LN2_LO` contributes only its own tiny rounding (section 2.3).
5. **Rescale by changing the exponent.** `np.ldexp(p, k)` multiplies by $2^k$ without rounding; where the true result is beyond float64 it gives $\infty$ or 0, which is the correct answer.
6. **Keep the edges out of the arithmetic.** Clip $x$ to $[-750, 710]$ so $k$ stays small and $\pm\infty$ map to the right limits; take NaN out before clipping (where it would turn into a number) and put it back at the end.
7. **erf from the all-positive series.** Section 2.4: $t_0 = x$, $t_{n+1} = t_n \cdot \frac{2x^2}{2n+3}$, summed, times $\frac{2}{\sqrt\pi}e^{-x^2}$. Every term has the sign of $x$, so the sum has no cancellation.
8. **Cut off where erf is $\pm 1$ in float64.** For $|x| \ge 6$ return $\operatorname{sign}(x)$, and feed 0 into the series for those elements so the loop never overflows (section 2.4).

### The code, annotated

```python
ERF_CUTOFF = 6.0  # erfc(6) = 2.2e-17 < half an ulp of 1.0: erf(x) rounds to sign(x)
LN2 = math.log(2.0)
# Cody and Waite: ln 2 split so that k * LN2_HI is exact for |k| < 2**21
# (LN2_HI has 21 trailing zero bits) and LN2_LO carries the rest.
LN2_HI = 6.93147180369123816490e-01
LN2_LO = 1.90821492927058770002e-10


def exp_taylor_coeffs(n: int) -> NDArray:
    if n < 0:
        raise ValueError(f"degree n must be >= 0, got {n}")
    c = np.empty(int(n) + 1, dtype=np.float64)
    c[0] = 1.0
    for i in range(1, int(n) + 1):
        c[i] = c[i - 1] / i  # 1/i! = (1/(i-1)!) / i, no factorial overflow
    return c
```

`LN2` is only used to *choose* $k$; an error in it can at worst pick a $k$ one off at a rounding tie, which still leaves $|r|$ within the bound. The subtraction uses the split constants.

```python
def exp_range_reduced(x: ArrayLike, deg: int = 6) -> NDArray:
    if deg < 0:
        raise ValueError(f"deg must be >= 0, got {deg}")
    x = np.asarray(x, dtype=np.float64)
    nan = np.isnan(x)                          # remembered, restored at the end
    # exp(-750) is below the smallest subnormal and exp(710) above the
    # largest double, so clipping changes no result and keeps k small.
    xc = np.clip(np.where(nan, 0.0, x), -750.0, 710.0)
    k = np.rint(xc / LN2)  # round half to even: |r| <= ln(2) / 2
    r = (xc - k * LN2_HI) - k * LN2_LO         # two-part reduction (step 4)
    # Ascending coefficients, the order horner (M00.4) takes; numpy.polyval
    # wants them descending.
    p = horner(exp_taylor_coeffs(deg), r)      # e^r on |r| <= 0.3466
    with np.errstate(over="ignore", under="ignore"):
        out = np.ldexp(p, k.astype(np.int32))  # times 2^k, exact; inf or 0 at the edges
    return np.where(nan, np.nan, out)
```

Trace the hand example: $x = 1$, `xc / LN2` is $1.4427$, so $k = 1$; $r = 1 - \ln 2 = 0.3068528$; with `deg=3`, `horner` returns $1.3587476$; `ldexp` doubles it to $2.7174952$. For $x = +\infty$ the clip gives 710, $k = 1024$, and `ldexp` overflows to $\infty$ (quietly, inside `errstate`); for $x = -\infty$ the clip gives $-750$ and `ldexp` underflows to 0. The parentheses in `(xc - k * LN2_HI) - k * LN2_LO` matter: the large, exact cancellation happens first, and the small correction is subtracted from the already small result.

```python
def erf_series(x: ArrayLike, terms: int) -> NDArray:
    if terms < 1:
        raise ValueError(f"terms must be >= 1, got {terms}")
    x = np.asarray(x, dtype=np.float64)
    big = np.abs(x) >= ERF_CUTOFF              # includes +-inf
    xs = np.where(big | np.isnan(x), 0.0, x)   # the loop only sees safe inputs
    x2 = xs * xs
    # t_0 = x, t_{n+1} = t_n * 2 x^2 / (2n + 3): every term has the sign of x.
    term = xs.copy()
    total = xs.copy()
    for n in range(1, int(terms)):
        term = term * (2.0 * x2) / (2 * n + 1)  # t_n from t_{n-1}: one multiply, one divide
        total = total + term
    out = (2.0 / math.sqrt(math.pi)) * np.exp(-x2) * total
    out = np.where(big, np.sign(x), out)       # erf is +-1 to float64 beyond the cutoff
    return np.where(np.isnan(x), np.nan, out)
```

The loop index `n` here is the index of the term being produced, so its ratio is $\frac{2x^2}{2n + 1}$; that is the same recurrence as $\frac{t_{n+1}}{t_n} = \frac{2x^2}{2n+3}$ with the index shifted by one. `terms` counts $t_0, \ldots, t_{\text{terms}-1}$. For the hand example (`terms=3`, $x = 0.5$): $t_1 = 0.5 \cdot 0.5/3$, $t_2 = t_1 \cdot 0.5/5$, total $0.5916667$, times $0.8787826$ gives $0.5199464$.

Full solution: [`taylor.py`](../../../../course/solutions/python/tinyllm/num/taylor.py).

### Why this solution

- **Cost.** `exp_range_reduced` is $O(\text{deg})$ per element: one division, one `rint`, two multiply-subtracts, `deg` Horner steps, and one `ldexp`. `erf_series` is $O(\text{terms})$ per element; 160 terms is expensive but this is a float64 reference, not the production `erf` (Going further).
- **Range reduction over a higher degree.** Without reduction, a Taylor polynomial for $e^x$ on $[-87, 88]$ would need hundreds of terms and would still lose digits to huge alternating terms for negative $x$. Reduction makes degree 6 enough for float32 everywhere.
- **Two-part constant over extended precision.** A double-double $\ln 2$ or `np.longdouble` would also work but is slower and not portable to the C port; the Cody-Waite split costs one extra multiply-subtract and is exact for every $|k| < 2^{21}$, far beyond the $|k| \le 1083$ the clip allows.
- **All-positive series over the alternating one.** Both converge, but in floating point only the positive one is accurate for $|x|$ beyond about 3 (section 2.4). Choosing the representation that does not cancel is the same lesson as `M00.4`'s quadratic formula.
- **Tradeoffs accepted.** Taylor coefficients are not optimal on an interval: a minimax polynomial (Going further) reaches the same accuracy with a degree or two fewer. Taylor is kept because its remainder bound is provable by hand, which makes it the right reference for `M09.6`. The erf series near the cutoff also needs many terms ($2x^2 = 72$ at $x = 6$, so the terms keep growing until $n \approx 35$), which is why the contract uses 120 or more.

### Explain it back

**1. Why does `exp_range_reduced` round $k$ to nearest instead of taking the floor?**

<details><summary>Answer</summary>

Rounding to nearest puts $r = x - k\ln 2$ in $[-\frac{\ln 2}{2}, \frac{\ln 2}{2}]$; the floor puts it in $[0, \ln 2)$, twice as far from the expansion point at worst. The degree-6 remainder scales like $|r|^7$, so doubling $|r|$ multiplies the bound by $2^7 = 128$; the measured degree-6 error rises to about $8 \times 10^{-6}$, far outside float32 accuracy.

</details>

**2. What goes wrong with `r = xc - k * LN2` near $x = 700$, and why does the two-part form fix it?**

<details><summary>Answer</summary>

`LN2` differs from $\ln 2$ by about $2.3 \times 10^{-17}$. Multiplied by $k \approx 1000$ that is an absolute error of a few times $10^{-14}$ in $r$, and $e^r$ inherits it as a relative error, hundreds of times the float64 rounding level. `LN2_HI` has 21 trailing zero bits, so $k \cdot \text{LN2\_HI}$ is exact for $|k| < 2^{21}$ and so is $x - k \cdot \text{LN2\_HI}$ in the cases that matter; the remaining $k \cdot \text{LN2\_LO}$ is about $10^{-7}$, and its rounding error is about $10^{-23}$, negligible.

</details>

**3. Why does the function replace NaN with 0 before clipping and restore it at the end, instead of letting NaN flow through?**

<details><summary>Answer</summary>

NaN must come out as NaN, but the steps in between are not NaN-safe: `np.rint(nan)` cast to `int32` gives an arbitrary integer and an "invalid value" warning. Computing on a harmless 0 and overwriting those positions with `np.where(nan, np.nan, out)` guarantees the right output and no warnings. Letting NaN become a number is the pitfall where $e^{\mathrm{nan}}$ comes out as 1.

</details>

**4. Why does `erf_series` zero out $|x| \ge 6$ before the loop rather than just replacing the result afterwards?**

<details><summary>Answer</summary>

`np.where` cannot prevent the loop from computing on those elements. For $x = 28$ the partial sums grow past $10^{308}$ and overflow while $e^{-x^2}$ underflows to 0, giving $0 \cdot \infty = $ NaN, with warnings, before the final `where` discards it. Feeding 0 into the series for those elements keeps every intermediate finite; the answer $\operatorname{sign}(x)$ is then substituted.

</details>

**5. The section 3 example shows a relative error of $2.9 \times 10^{-4}$ against a bound of $3.7 \times 10^{-4}$. Why is the bound always above the error, and why is it so tight here?**

<details><summary>Answer</summary>

The Lagrange remainder is exact for some $\xi$ between 0 and $r$: the error is $\frac{e^\xi r^4}{4!}$, and bounding $e^{\xi - r} \le 1$ (since $r > 0$) can only make it larger. It is tight because the next Taylor term $r^4/4!$ dominates the remainder when $r$ is small: the omitted terms after it are a factor of about $r/5 \approx 0.06$ smaller each.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. coefficients in the wrong order for the evaluator (numpy's `polyval` is descending, `horner` ascending) | $p(r)$ evaluates the reversed polynomial; at degree 3, $e^1 \approx 0.89$ | `exp_taylor_coeffs` returns ascending coefficients and passes them to `horner`; step 1 |
| 2. rounding $k$ down instead of to nearest | $r$ up to 0.69, error $8 \times 10^{-6}$ at degree 6 | `np.rint`; step 3 |
| 3. one rounded $\ln 2$ in the reduction | relative error $8 \times 10^{-14}$ near $x = 700$ | `(xc - k * LN2_HI) - k * LN2_LO`; step 4 |
| 4. the alternating erf series | 7 correct digits at $x = 5$ | the all-positive series times $e^{-x^2}$; step 7 |
| 5. no cutoff for large $\lvert x \rvert$ | nan at $x = 28$, wrong values beyond 6 | `ERF_CUTOFF = 6.0`, those inputs zeroed in the loop and replaced by $\operatorname{sign}(x)$; step 8 |
| adding the low part of $\ln 2$ instead of subtracting it | relative error $4 \times 10^{-10}$ per unit of $k$ | both parts are subtracted |
| coefficients $1/(i+1)!$ | every value wrong | `c[0] = 1.0`, then `c[i] = c[i - 1] / i` |
| nan passed through the clip as 0 | $e^{\mathrm{nan}} = 1$ | the NaN mask is restored at the end; step 6 |
| an off-by-one in the erf ratio or term count | the hand example's third term is wrong | ratio `2.0 * x2 / (2 * n + 1)` for term `n`, loop over `range(1, terms)` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M00.4` | `horner(exp_taylor_coeffs(deg), r)` evaluates the polynomial |
| Back | `M01.1` | derivatives and the first terms of Taylor's formula (reading) |
| Forward | `M01.3` | `gelu_erf` and `dgelu_erf` compute $\Phi(x)$ with `erf_series(x / sqrt(2), 160)` |
| Forward | `M08.1` | `dual_erf` differentiates erf in forward mode |
| Forward | `M09.2` | stable softmax and logsumexp: why $e^x$ overflows at 709.78 |
| Forward | `M09.6` | `tl_expf` in C: the same reduction and a degree-6 polynomial in float32, checked against this Python |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `exp_range_reduced` | glibc `exp` and `expf` | a 128-entry table ($e^{j/128}$) so $r$ is even smaller and the polynomial shorter; correctly rounded in most cases | glibc `sysdeps/ieee754/dbl-64/e_exp.c`, `sysdeps/ieee754/flt-32/e_expf.c` |
| Taylor coefficients | minimax polynomials (Remez algorithm) | the polynomial with the smallest maximum error on the interval, not the best at one point: one or two degrees fewer for the same accuracy | Sollya; `M09.6` |
| two-part $\ln 2$ | Payne and Hanek reduction | exact reduction of $\sin$ and $\cos$ for arguments like $10^{300}$, where $\pi$ needs a thousand bits | Muller, *Elementary Functions*, ch. 11 |
| `erf_series` | SLEEF, CUDA `erff` | branch-free vectorized approximations for whole SIMD lanes | SLEEF `src/libm/sleefsimdsp.c` |
