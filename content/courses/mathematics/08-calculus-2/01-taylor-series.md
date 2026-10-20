# Taylor series, remainder bounds, and range reduction

A CPU can add and multiply; it cannot compute $e^x$. Up to this point numpy has done that work for you, but the optional C kernels later in *AI Engineering* must do it themselves: [softmax in C](../../ai-engineering/22-kernels/04-softmax-three-pass-and-online.md) exponentiates every logit, and the [SiLU kernel](../../ai-engineering/22-kernels/08-elementwise-kernels.md) exponentiates every activation. The way every math library computes $e^x$ is a polynomial on a small interval plus an exact rescaling. The polynomial comes from the Taylor series, and the Taylor series comes with an error bound that tells you which degree is enough.

The same tools give you erf, which the exact GELU in [Activation Functions and Their Derivatives](../07-calculus-1/03-activation-functions.md) needs. This section builds both in Python, where you can watch the error bound hold point by point, so that the C port in [Polynomial approximation and range reduction: expf in C](../15-numerical-methods-and-floating-point/06-polynomial-approximation-and-expf.md) has a trusted reference to check against.

You will lean on two earlier sections. [Polynomials, Horner, and stable quadratic roots](../06-precalculus/04-polynomials-horner-and-stable-roots.md) supplies the `horner` evaluator that runs every polynomial here, and [Derivatives and finite differences](../07-calculus-1/01-derivatives-and-finite-differences.md) used the first few terms of Taylor's formula informally; this section proves the full version. The finished code is in [`course/solutions/python/tinyllm/num/taylor.py`](../../../../course/solutions/python/tinyllm/num/taylor.py), and its interface is fixed by the contract [`course/contracts/py/tinyllm/num/taylor.pyi`](../../../../course/contracts/py/tinyllm/num/taylor.pyi).

> **Learning objectives**
> - Derive the degree-$n$ Taylor polynomial of a function at a point and compute the coefficients of $e^x$ at 0 without forming factorials.
> - State the Lagrange remainder and use it to bound the error of $e^r \approx p_n(r)$ at any point and degree.
> - Explain range reduction $e^x = 2^k e^r$, why $k$ is rounded to nearest, and why $k \ln 2$ is subtracted in two parts.
> - Recognize catastrophic cancellation in an alternating series and replace it with an all-positive series for erf, with a geometric tail bound.
> - Implement `exp_range_reduced` and `erf_series` so they handle infinities, NaN, overflow, and underflow without warnings.

## Polynomials that match derivatives

The notation used throughout the section is collected in one place first, since several symbols do double duty as code variables.

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

Near a point $a$, which polynomial of degree $n$ is the best stand-in for $f$? The natural answer is the one whose value and first $n$ derivatives at $a$ equal $f$'s. Write $p(x) = \sum_{i=0}^{n} c_i (x - a)^i$. Differentiating $i$ times and setting $x = a$ kills every term but one, leaving $p^{(i)}(a) = i!\, c_i$. Matching $p^{(i)}(a) = f^{(i)}(a)$ then fixes every coefficient:

$$p_n(x) = \sum_{i=0}^{n} \frac{f^{(i)}(a)}{i!} (x - a)^i .$$

For $f = e^x$ at $a = 0$ every derivative is $e^0 = 1$, so $p_n(x) = 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \cdots + \frac{x^n}{n!}$, with coefficients $1/i!$. The function `exp_taylor_coeffs(n)` returns them in **ascending** order of power, $[1/0!, \ldots, 1/n!]$, which is the order `horner` takes (numpy's `polyval` wants the reverse). Computing $c_i = c_{i-1}/i$ avoids forming $i!$, which overflows float64 at $i = 171$.

## The remainder tells you how wrong you are

A polynomial that matches derivatives at one point is only useful if you know how far it drifts away from that point. **Taylor's theorem** in Lagrange form answers exactly that: if $f$ has $n + 1$ continuous derivatives, then for some $\xi$ between $a$ and $x$,

$$R_n(x) = f(x) - p_n(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!}(x - a)^{n+1} .$$

You do not know $\xi$, but you can bound $|f^{(n+1)}|$ between $a$ and $x$, and that bounds the error. (The finite-difference analysis in [Derivatives and finite differences](../07-calculus-1/01-derivatives-and-finite-differences.md) used the first terms of this formula; this is the proven version.) For $e^r$ at 0, $f^{(n+1)}(\xi) = e^\xi$ with $\xi$ between 0 and $r$, so relative to $e^r$:

$$\frac{|e^r - p_n(r)|}{e^r} = \frac{e^{\xi - r}\,|r|^{n+1}}{(n+1)!} \le e^{|r| - r}\,\frac{|r|^{n+1}}{(n+1)!} .$$

The inequality on the right follows from two cases. If $r > 0$ then $\xi \le r$ and $e^{\xi - r} \le 1$; if $r < 0$ then $\xi \le 0$ and $e^{\xi - r} \le e^{-r} = e^{|r|}$. The bound holds at every point and every degree, so it is a property any correct implementation can be checked against.

Read the bound carefully and you see both its power and its limit. It shrinks factorially in $n$ but grows like $|r|^{n+1}$: at $r = 0.35$, degree 6 gives $1.2 \times 10^{-7}$; at $r = 10$, degree 6 gives 2000. A Taylor polynomial is only good near its center. The whole trick of computing $e^x$, then, is to make sure the polynomial never sees a large argument.

## Range reduction: keep the argument small

Every real $x$ can be written $x = k \ln 2 + r$ with $k$ an integer and $|r| \le \frac{\ln 2}{2} \approx 0.3466$: take $k = \operatorname{round}(x / \ln 2)$. Then

$$e^x = e^{k\ln 2} \cdot e^r = 2^k\, e^r ,$$

and multiplying by $2^k$ is exact in binary floating point, because it only changes the exponent field (`np.ldexp(p, k)`). So $e^x$ costs one reduction, one polynomial on $|r| \le 0.3466$, and one exponent adjustment, and the remainder bound of the previous section holds for **every** $x$. With degree 6 the relative error is at most $e^{2 \cdot 0.3466}\,0.3466^7/7! = 2.4 \times 10^{-7}$, about two float32 units in the last place. That is the accuracy target of `tl_expf` in the C port.

The choice of rounding matters. Rounding $k$ down (floor) instead of to nearest leaves $r \in [0, \ln 2)$, doubles the largest $|r|$, and multiplies the bound by $2^7 = 128$.

### Subtracting $k \ln 2$ exactly

The reduction itself is a subtraction, and subtractions are where floating point bites. $\ln 2$ is irrational, so the float64 constant `LN2` is off by up to $\varepsilon/4$ relative. Multiplied by $k = 1000$ that error lands in $r$ as $10^{-13}$, and $e^r$ inherits it: 400 times the float64 rounding level.

Cody and Waite's fix is to split the constant. `LN2_HI` holds the first 32 bits of $\ln 2$ with the low 21 bits zero, so $k \cdot$ `LN2_HI` is exact for $|k| < 2^{21}$, and `LN2_LO` $= \ln 2 -$ `LN2_HI` is tiny. Then $r = (x - k\,\mathrm{LN2\_HI}) - k\,\mathrm{LN2\_LO}$ is accurate to the last bit.

### The edges of the range

$e^x$ overflows float64 above $x \approx 709.78$ and falls below the smallest subnormal under $x \approx -745.1$. Clipping $x$ to $[-750, 710]$ first changes no result and keeps $k$ small; then $+\infty \mapsto \infty$, $-\infty \mapsto 0$, and NaN is passed through separately.

## A series for erf without cancellation

The second function is erf, defined by $\operatorname{erf}(x) = \frac{2}{\sqrt\pi}\int_0^x e^{-t^2}dt$. Integrating the Taylor series of $e^{-t^2}$ term by term gives the textbook series

$$\operatorname{erf}(x) = \frac{2}{\sqrt\pi} \sum_{n \ge 0} \frac{(-1)^n x^{2n+1}}{n!\,(2n+1)} ,$$

which converges for every $x$ but alternates in sign. That is a problem in floating point. At $x = 5$ its largest term is about $6 \times 10^{8}$, the sum is below 1, and float64 keeps 16 significant digits of each term, so the cancellation leaves about 7 correct digits of the answer.

The fix is a different series for the same function. The Taylor series of $e^{x^2}\operatorname{erf}(x)$ has only positive coefficients:

$$\operatorname{erf}(x) = \frac{2}{\sqrt\pi}\, e^{-x^2} \sum_{n \ge 0} t_n, \qquad t_n = \frac{2^n x^{2n+1}}{1 \cdot 3 \cdot 5 \cdots (2n+1)}, \qquad \frac{t_{n+1}}{t_n} = \frac{2x^2}{2n+3} .$$

Every term has the sign of $x$, so nothing cancels, and each term comes from the previous one by one multiplication.

This series also has a clean **tail bound**. Once the ratio $\rho = \frac{2x^2}{2N+3}$ is below 1, the ratios keep falling, so the omitted terms are at most a geometric series: $\sum_{n \ge N} t_n \le \frac{t_N}{1 - \rho}$. (Geometric series are the subject of the next section, [Series convergence, the EMA as a geometric series, and bias correction](02-series-and-the-ema.md).)

Finally, there is a cutoff. For $|x| \ge 6$, $1 - \operatorname{erf}(x) < 2.2 \times 10^{-17}$, below half the gap between 1 and the float before it, so erf rounds to exactly $\pm 1$. The series would need hundreds of terms there, and near $|x| = 27$ its partial sums overflow while $e^{-x^2}$ underflows to 0, giving $0 \cdot \infty = $ NaN. So `erf_series` returns $\operatorname{sign}(x)$ for $|x| \ge 6$. With 120 or more terms the result is within $3 \times 10^{-15}$ of erf everywhere.

## Working it by hand

Before writing code, run both algorithms on paper with low degree so that every intermediate number is visible.

Start with $e^1$ at degree 3. First reduce: $k = \operatorname{round}(1/0.693147) = \operatorname{round}(1.4427) = 1$, so $r = 1 - \ln 2 = 0.3068528$. Then evaluate the polynomial:

$$p_3(r) = 1 + r + \frac{r^2}{2} + \frac{r^3}{6} = 1 + 0.3068528 + 0.0470793 + 0.0048155 = 1.3587476 ,$$

$$2^1 \cdot p_3(r) = 2.7174952 \quad \text{vs.} \quad e = 2.7182818 .$$

The relative error is $2.9 \times 10^{-4}$, and the Lagrange bound (here $r > 0$, so the factor is 1) is $r^4/4! = 0.0088658/24 = 3.7 \times 10^{-4}$. The bound holds with little room to spare. Horner's rule evaluates the same polynomial as $1 + r(1 + r(\frac12 + r \cdot \frac16))$.

Now take $\operatorname{erf}(0.5)$ with three terms. Here $x^2 = 0.25$, so the numerator of the ratio is $2x^2 = 0.5$, divided by $2n + 1$: $t_0 = 0.5$, $t_1 = 0.5 \cdot 0.5/3 = 0.0833333$, $t_2 = 0.0833333 \cdot 0.5/5 = 0.0083333$, for a sum of $0.5916667$. The prefactor is $\frac{2}{\sqrt\pi} e^{-0.25} = 1.1283792 \times 0.7788008 = 0.8787826$. The result is $0.8787826 \times 0.5916667 = 0.5199464$, against $\operatorname{erf}(0.5) = 0.5204999$. The next term $t_3 = 0.000595$ times the prefactor is $0.000523$, and the tail bound $t_3/(1 - 0.5/9)$ times the prefactor, $0.000554$, covers the actual gap of $0.000553$.

## Designing the interface

The module exposes one constant and three functions:

```python
ERF_CUTOFF: float  # 6.0
def exp_taylor_coeffs(n: int) -> NDArray: ...                  # [1/0!, ..., 1/n!]
def exp_range_reduced(x: ArrayLike, deg: int = 6) -> NDArray: ...
def erf_series(x: ArrayLike, terms: int) -> NDArray: ...
```

All arithmetic is float64, and results have the shape of `x`. `exp_range_reduced` handles $\pm\infty$, NaN, overflow, and underflow without warnings. A negative `n` or `deg`, or `terms < 1`, raises `ValueError`.

## Building it step by step

The design follows directly from the principles above, one decision at a time.

1. **Coefficients by recurrence, ascending.** $c_i = 1/i!$, and $c_i = c_{i-1}/i$ builds them with one division each and never forms $i!$, which overflows at 171. Ascending order is what `horner` takes.
2. **Reduce first, so the bound is small everywhere.** The remainder grows like $|r|^{n+1}$, so the polynomial must only ever see small arguments. Writing $x = k\ln 2 + r$ with $k = \operatorname{round}(x/\ln 2)$ bounds $|r| \le \frac{\ln 2}{2}$ for every $x$, and $e^x = 2^k e^r$ with an exact rescaling.
3. **Round to nearest, not down.** Nearest rounding centres $r$ on 0; flooring puts it in $[0, \ln 2)$, doubles its largest size, and multiplies the degree-6 bound by $2^7$.
4. **Subtract $k\ln 2$ in two exact pieces.** With `LN2_HI` short enough that $k \cdot$ `LN2_HI` has no rounding, $x - k \cdot \text{LN2\_HI}$ is (nearly) exact, and the small correction $k \cdot$ `LN2_LO` contributes only its own tiny rounding.
5. **Rescale by changing the exponent.** `np.ldexp(p, k)` multiplies by $2^k$ without rounding; where the true result is beyond float64 it gives $\infty$ or 0, which is the correct answer.
6. **Keep the edges out of the arithmetic.** Clip $x$ to $[-750, 710]$ so $k$ stays small and $\pm\infty$ map to the right limits; take NaN out before clipping (where it would turn into a number) and put it back at the end.
7. **erf from the all-positive series.** $t_0 = x$, $t_{n+1} = t_n \cdot \frac{2x^2}{2n+3}$, summed, times $\frac{2}{\sqrt\pi}e^{-x^2}$. Every term has the sign of $x$, so the sum has no cancellation.
8. **Cut off where erf is $\pm 1$ in float64.** For $|x| \ge 6$ return $\operatorname{sign}(x)$, and feed 0 into the series for those elements so the loop never overflows.

### The constants and the coefficients

The listing opens with the constants and the coefficient builder:

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

`LN2` is only used to *choose* $k$. An error in it can at worst pick a $k$ one off at a rounding tie, which still leaves $|r|$ within the bound. The subtraction uses the split constants.

### The exponential

With the coefficients in hand, `exp_range_reduced` is the reduction, the polynomial, and the rescaling, wrapped in the edge handling:

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

Trace the hand example through it. With $x = 1$, `xc / LN2` is $1.4427$, so $k = 1$; $r = 1 - \ln 2 = 0.3068528$; with `deg=3`, `horner` returns $1.3587476$; `ldexp` doubles it to $2.7174952$. For $x = +\infty$ the clip gives 710, $k = 1024$, and `ldexp` overflows to $\infty$ (quietly, inside `errstate`); for $x = -\infty$ the clip gives $-750$ and `ldexp` underflows to 0. The parentheses in `(xc - k * LN2_HI) - k * LN2_LO` matter: the large, exact cancellation happens first, and the small correction is subtracted from the already small result.

### The error function

`erf_series` applies the same edge discipline to the all-positive series:

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

The full solution is [`taylor.py`](../../../../course/solutions/python/tinyllm/num/taylor.py).

### Why this design

`exp_range_reduced` costs $O(\text{deg})$ per element: one division, one `rint`, two multiply-subtracts, `deg` Horner steps, and one `ldexp`. `erf_series` costs $O(\text{terms})$ per element; 160 terms is expensive, but this is a float64 reference, not the production `erf` (see Further reading).

Range reduction beats simply raising the degree. Without reduction, a Taylor polynomial for $e^x$ on $[-87, 88]$ would need hundreds of terms and would still lose digits to huge alternating terms for negative $x$. Reduction makes degree 6 enough for float32 everywhere.

The two-part constant beats extended precision. A double-double $\ln 2$ or `np.longdouble` would also work, but it is slower and does not port to C; the Cody-Waite split costs one extra multiply-subtract and is exact for every $|k| < 2^{21}$, far beyond the $|k| \le 1083$ the clip allows.

The all-positive series beats the alternating one. Both converge, but in floating point only the positive one is accurate for $|x|$ beyond about 3. Choosing the representation that does not cancel is the same lesson as the stable quadratic formula in [Polynomials, Horner, and stable quadratic roots](../06-precalculus/04-polynomials-horner-and-stable-roots.md).

Some tradeoffs are accepted knowingly. Taylor coefficients are not optimal on an interval: a minimax polynomial reaches the same accuracy with a degree or two fewer. Taylor is kept because its remainder bound is provable by hand, which makes it the right reference for the C port. The erf series near the cutoff also needs many terms ($2x^2 = 72$ at $x = 6$, so the terms keep growing until $n \approx 35$), which is why the contract uses 120 or more.

## What can go wrong

A correct implementation must survive a specific set of checks. Each one exists because something downstream depends on it:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked $e^1$ | unit | $2.7174952$, error $2.9 \times 10^{-4}$ inside the bound | everyone agrees on reduction and polynomial |
| inverse-factorial coefficients | unit | $[1, 1, \frac12, \frac16, \frac1{24}]$, length $n + 1$, $1/30!$ | the coefficient order `horner` and the C port use |
| the Lagrange bound | property | the remainder bound at 4001 points, degrees 1 to 12 | the remainder bound is the design's central property |
| degree 6 is float32-accurate | golden | relative error at most $2.4 \times 10^{-7}$ on $[-87, 88]$ | `tl_expf`'s accuracy budget |
| full precision across the range | golden | degree 20 within $10^{-15}$ near $\pm 700$ | the two-part reduction |
| special values and extremes | boundary | $\pm\infty$, NaN, overflow, underflow, subnormal, no warnings | masked logits are $-\infty$ |
| shapes and scalars | unit | scalars, matrices, integer input, `deg < 0` raises | callers pass every shape |
| the hand-worked $\operatorname{erf}(0.5)$ | unit | 0.5199464 | the series and its ratio |
| erf against `math.erf` | golden | 160 terms within $3 \times 10^{-15}$ of `math.erf` on $[-7, 7]$ | `gelu_erf` must match PyTorch |
| no cancellation at $x = 5$ | boundary | $x = 3, 4, 5, -5.5$ to $3 \times 10^{-15}$ | the alternating series fails here |
| erf cutoff and special values | boundary | $\pm 1$ at and beyond 6, $\pm\infty$, NaN, 0 | GELU feeds $x/\sqrt2$ up to 28 |
| the erf tail bound | property | the geometric tail bound at many $N$ and $x$ | the remainder for erf |

The mistakes that break these checks are worth knowing by their symptoms, because each produces a recognizable wrong number.

> **Warning:** Coefficients in the wrong order for the evaluator (numpy's `polyval` is descending, `horner` ascending) evaluate the reversed polynomial; at degree 3 you get $e^1 \approx 0.89$. `exp_taylor_coeffs` returns ascending coefficients and passes them to `horner`.

Rounding $k$ down instead of to nearest lets $r$ reach 0.69 and pushes the degree-6 error to $8 \times 10^{-6}$; `np.rint` prevents it. Using one rounded $\ln 2$ in the reduction gives a relative error of $8 \times 10^{-14}$ near $x = 700$; the two-part `(xc - k * LN2_HI) - k * LN2_LO` fixes it. Adding the low part of $\ln 2$ instead of subtracting it gives a relative error of $4 \times 10^{-10}$ per unit of $k$; both parts must be subtracted. Coefficients $1/(i+1)!$ make every value wrong; the code sets `c[0] = 1.0` and then `c[i] = c[i - 1] / i`. Letting NaN pass through the clip as 0 yields $e^{\mathrm{nan}} = 1$; the NaN mask is restored at the end.

On the erf side, the alternating series gives only 7 correct digits at $x = 5$, which is why the code uses the all-positive series times $e^{-x^2}$. Having no cutoff for large $|x|$ gives NaN at $x = 28$ and wrong values beyond 6; `ERF_CUTOFF = 6.0` zeroes those inputs in the loop and replaces them by $\operatorname{sign}(x)$. An off-by-one in the erf ratio or term count shows up as a wrong third term in the hand example; the ratio must be `2.0 * x2 / (2 * n + 1)` for term `n`, with the loop over `range(1, terms)`.

## Where this leads

These functions are used immediately and then again much later. Back in [Activation Functions and Their Derivatives](../07-calculus-1/03-activation-functions.md), `gelu_erf` and `dgelu_erf` compute $\Phi(x)$ with `erf_series(x / sqrt(2), 160)`. [Dual numbers and forward-mode autodiff](../14-matrix-calculus-and-autodiff/01-dual-numbers-forward-mode.md) differentiates erf in forward mode with `dual_erf`, which uses the same series. [Stable numerics: logsumexp, softmax, and compensated sums](../15-numerical-methods-and-floating-point/02-stable-numerics.md) explains why $e^x$ overflows at 709.78 and how stable softmax avoids it. And [Polynomial approximation and range reduction: expf in C](../15-numerical-methods-and-floating-point/06-polynomial-approximation-and-expf.md) builds `tl_expf` with the same reduction and a degree-6 polynomial in float32, checked against this Python.

### Further reading

- OpenStax, *Calculus Volume 2* (free), sections 6.3 and 6.4: Taylor and Maclaurin series and the remainder.
- Muller, *Elementary Functions: Algorithms and Implementation*, ch. 11: range reduction, Cody and Waite, and Payne and Hanek reduction, which reduces $\sin$ and $\cos$ exactly for arguments like $10^{300}$, where $\pi$ needs a thousand bits.
- Abramowitz and Stegun 7.1.6: the erf series used here.
- glibc `exp` and `expf` (`sysdeps/ieee754/dbl-64/e_exp.c`, `sysdeps/ieee754/flt-32/e_expf.c`): a 128-entry table of $e^{j/128}$ makes $r$ even smaller and the polynomial shorter, and the results are correctly rounded in most cases.
- Minimax polynomials via the Remez algorithm (the Sollya tool): the polynomial with the smallest maximum error on the interval rather than the best at one point, one or two degrees fewer for the same accuracy. The C port discusses this further.
- SLEEF (`src/libm/sleefsimdsp.c`) and CUDA `erff`: branch-free vectorized approximations for whole SIMD lanes.

## Key terms

- **Taylor polynomial**: the degree-$n$ polynomial whose value and first $n$ derivatives at a point match those of $f$.
- **Lagrange remainder**: the exact error $f^{(n+1)}(\xi)(x-a)^{n+1}/(n+1)!$ of a Taylor polynomial, for some unknown $\xi$ between $a$ and $x$.
- **Range reduction**: rewriting $e^x = 2^k e^r$ so the polynomial only ever sees a small argument $|r| \le \frac{\ln 2}{2}$.
- **Cody-Waite reduction**: subtracting $k \ln 2$ as two pieces, a short exact high part and a tiny low part, so $r$ is accurate to the last bit.
- **Catastrophic cancellation**: loss of significant digits when large terms of opposite sign nearly cancel.
- **Tail bound**: an upper bound on the sum of the omitted terms of a series, here a geometric series.

## Summary

- The Taylor polynomial of degree $n$ matches $f$ and its first $n$ derivatives at a point; for $e^x$ at 0 its coefficients are $1/i!$, built by $c_i = c_{i-1}/i$.
- The Lagrange remainder bounds the error exactly: $|e^r - p_n(r)| \le e^{\max(r, 0)} |r|^{n+1}/(n+1)!$, tiny when $|r|$ is small and useless when it is not.
- Range reduction makes $|r|$ small for every input: $e^x = 2^k e^r$ with $k = \operatorname{round}(x/\ln 2)$ and $|r| \le \frac{\ln 2}{2}$; degree 6 is then float32-accurate.
- Subtracting $k \ln 2$ in two parts (Cody and Waite) keeps $r$ exact to the last bit even for $k = 1000$.
- A series can be correct and still useless in floating point: the alternating series for erf cancels catastrophically at $x = 5$; the series with all terms of one sign does not.

## Review questions

1. Why does `exp_range_reduced` round $k$ to nearest instead of taking the floor?

   <details><summary>Answer</summary>

   Rounding to nearest puts $r = x - k\ln 2$ in $[-\frac{\ln 2}{2}, \frac{\ln 2}{2}]$; the floor puts it in $[0, \ln 2)$, twice as far from the expansion point at worst. The degree-6 remainder scales like $|r|^7$, so doubling $|r|$ multiplies the bound by $2^7 = 128$; the measured degree-6 error rises to about $8 \times 10^{-6}$, far outside float32 accuracy.

   </details>

2. What goes wrong with `r = xc - k * LN2` near $x = 700$, and why does the two-part form fix it?

   <details><summary>Answer</summary>

   `LN2` differs from $\ln 2$ by about $2.3 \times 10^{-17}$. Multiplied by $k \approx 1000$ that is an absolute error of a few times $10^{-14}$ in $r$, and $e^r$ inherits it as a relative error, hundreds of times the float64 rounding level. `LN2_HI` has 21 trailing zero bits, so $k \cdot \text{LN2\_HI}$ is exact for $|k| < 2^{21}$ and so is $x - k \cdot \text{LN2\_HI}$ in the cases that matter; the remaining $k \cdot \text{LN2\_LO}$ is about $10^{-7}$, and its rounding error is about $10^{-23}$, negligible.

   </details>

3. Why does the function replace NaN with 0 before clipping and restore it at the end, instead of letting NaN flow through?

   <details><summary>Answer</summary>

   NaN must come out as NaN, but the steps in between are not NaN-safe: `np.rint(nan)` cast to `int32` gives an arbitrary integer and an "invalid value" warning. Computing on a harmless 0 and overwriting those positions with `np.where(nan, np.nan, out)` guarantees the right output and no warnings. Letting NaN become a number is the pitfall where $e^{\mathrm{nan}}$ comes out as 1.

   </details>

4. Why does `erf_series` zero out $|x| \ge 6$ before the loop rather than just replacing the result afterwards?

   <details><summary>Answer</summary>

   `np.where` cannot prevent the loop from computing on those elements. For $x = 28$ the partial sums grow past $10^{308}$ and overflow while $e^{-x^2}$ underflows to 0, giving $0 \cdot \infty = $ NaN, with warnings, before the final `where` discards it. Feeding 0 into the series for those elements keeps every intermediate finite; the answer $\operatorname{sign}(x)$ is then substituted.

   </details>

5. The hand-worked $e^1$ example shows a relative error of $2.9 \times 10^{-4}$ against a bound of $3.7 \times 10^{-4}$. Why is the bound always above the error, and why is it so tight here?

   <details><summary>Answer</summary>

   The Lagrange remainder is exact for some $\xi$ between 0 and $r$: the error is $\frac{e^\xi r^4}{4!}$, and bounding $e^{\xi - r} \le 1$ (since $r > 0$) can only make it larger. It is tight because the next Taylor term $r^4/4!$ dominates the remainder when $r$ is small: the omitted terms after it are a factor of about $r/5 \approx 0.06$ smaller each.

   </details>
