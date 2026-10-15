<!-- ol:module M00.4 -->
# Polynomials, Horner, stable quadratic roots

## Overview

| | |
|---|---|
| **Module** | `M00.4` · build · Python · Pass 2 · 2 to 3 h |
| **You build** | `python/tinyllm/num/poly.py`: `horner`, `quadratic_roots` |
| **Contract** | [`course/contracts/py/tinyllm/num/poly.pyi`](../../../../course/contracts/py/tinyllm/num/poly.pyi) |
| **Tests** | `course/tests/M00.4/test_poly.py` (what they check: section 4) |
| **Needs** | nothing to call. Reading: `M00.1` (exponents), `lang.01` (numpy arrays) |
| **Used by** | `M02.1` evaluates its range-reduced Taylor polynomial for $e^x$ with `horner` · later `M09.6` ports that code to C as `tl_expf` |
| **Milestone** | `MS-P2` (the Pass 2 gate) |
| **Optional depth** | OpenStax, *College Algebra 2e* (free), ch. 5 (polynomial functions); Higham, *Accuracy and Stability of Numerical Algorithms*, ch. 5 (Horner's error bound); Press et al., *Numerical Recipes*, section 5.6 (quadratic and cubic equations) |

## Key Takeaways

- A polynomial is its list of **coefficients**; in this course they are stored in **ascending** order of power, so `coeffs[k]` multiplies $x^k$ (`test_coefficients_are_ascending`).
- **Horner's rule** evaluates a degree-$n$ polynomial with $n$ multiplications and $n$ additions by nesting, $c_0 + x(c_1 + x(c_2 + \cdots))$, and its error is bounded by $\gamma_{2n}\sum_k |c_k||x|^k$ (`test_hand_example_horner`, `test_horner_within_error_bound`).
- On integer coefficients and points it is **exact** as long as every partial result stays below $2^{53}$ (`test_horner_exact_on_integer_polynomials`).
- The textbook quadratic formula **cancels** when $b^2 \gg 4ac$: the small root comes out with few or no correct digits. Computing $q = -(b + \operatorname{sign}(b)\sqrt{b^2 - 4ac})/2$ and the roots $q/a$ and $c/q$ never subtracts nearly equal numbers (`test_hand_example_roots`, `test_roots_golden_cancellation`).
- Two edge cases break naive code: $\operatorname{sign}(0)$ must count as $+1$, and $b^2$ overflows for $|b| \approx 10^{200}$ unless the coefficients are scaled first (`test_b_zero`, `test_huge_b_does_not_overflow`).

## How to work this chapter

```bash
ol start M00.4              # stubs python/tinyllm/num/poly.py into your repo
ol tests M00.4              # read the test catalog first: rung R0, you write no tests here
ol check M00.4              # exit code is the verdict
ol diff  M00.4              # after passing: your code against the reference
```

---

## 1. Why now

Every activation function and softmax in your models calls $e^x$, and so far numpy has computed it for you. In Pass 6 your C kernels (`L9.2` softmax, `L9.6` SiLU) need their own `tl_expf`, and `M09.6` builds it the way every math library does: reduce $x$ to a small range, then evaluate a **polynomial** that approximates $e^x$ there. `M02.1` first derives those polynomials (Taylor series) and evaluates them in Python. Both evaluate with Horner's rule, the subject of this module, and both depend on how evaluation order changes the rounding error. The second half of the module is the smallest example of the course's numerical theme: a formula that is correct on paper and wrong in floating point. The quadratic formula you learned in school loses the small root of $x^2 - 10^8 x + 1$ entirely, and a two-line rewrite fixes it.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $n$ | the degree: the highest power with a nonzero coefficient | `int` |
| $c_0, \ldots, c_n$ | coefficients, ascending: $c_k$ multiplies $x^k$ | `Sequence[float]` |
| $p(x)$ | the polynomial $\sum_{k=0}^n c_k x^k$ | `NDArray` |
| $r$ | a root: a number with $p(r) = 0$ | `float` |
| $a, b, c$ | the coefficients of the quadratic $ax^2 + bx + c$, $a \ne 0$ | `float` |
| $D$ | the discriminant, $b^2 - 4ac$ | `float` |
| $x_1, x_2$ | the two roots, $x_1 \le x_2$ | `float` |
| $q$ | the stable intermediate $-\tfrac12(b + \operatorname{sign}(b)\sqrt D)$ | `float` |
| $u$ | float64 unit roundoff, $2^{-53} \approx 1.1 \times 10^{-16}$ | |
| $\gamma_k$ | $ku/(1 - ku)$, the bound on $k$ rounded operations | `float` |

### 2.1 Polynomials and their roots

A **polynomial** of degree $n$ is $p(x) = c_0 + c_1 x + \cdots + c_n x^n$ with $c_n \ne 0$. It is determined by its coefficient list, which this course writes in ascending order: $[1, 2, 3]$ is $1 + 2x + 3x^2$. (numpy's `np.polyval` takes the opposite, descending order; `numpy.polynomial.polynomial.polyval` takes ascending. Mixing the two is the first pitfall.)

A **root** is an $r$ with $p(r) = 0$. The factor theorem says $p(r) = 0$ exactly when $(x - r)$ divides $p$: $p(x) = (x - r)\,s(x)$ for a polynomial $s$ of degree $n - 1$. Dividing out one root at a time shows a degree-$n$ polynomial has at most $n$ roots. Example: $x^3 - 6x^2 + 11x - 6$ is 0 at $x = 1$, and dividing by $x - 1$ leaves $x^2 - 5x + 6 = (x - 2)(x - 3)$.

### 2.2 Horner's rule

Evaluating $\sum c_k x^k$ term by term computes every power separately. Factoring $x$ out repeatedly gives the nested form

$$p(x) = c_0 + x\Bigl(c_1 + x\bigl(c_2 + \cdots + x(c_{n-1} + x\,c_n)\bigr)\Bigr),$$

evaluated from the inside out:

```text
acc = 0
for k = n, n-1, ..., 0:      # highest power first
    acc = acc * x + c[k]
```

That is $n$ multiplications and $n$ additions (the first step multiplies 0), the fewest possible for a general polynomial, and one fused multiply-add per coefficient in C. Horner's intermediate values are also the coefficients of the quotient $p(x)/(x - r)$ when you evaluate at $x = r$, with the remainder $p(r)$ last: this is "synthetic division". Starting from `acc = 0` makes the empty coefficient list the zero polynomial, and works the same for a scalar $x$ or a whole array of points.

### 2.3 How accurate is Horner?

Each step rounds twice, once for the multiplication and once for the addition, each with relative error at most $u$. Following those errors through the $n$ steps (Higham, ch. 5) bounds the computed value $\hat p$:

$$|\hat p(x) - p(x)| \le \gamma_{2n} \sum_{k=0}^n |c_k|\,|x|^k, \qquad \gamma_{2n} = \frac{2nu}{1 - 2nu} \approx 2nu .$$

The right side is the size of the largest terms, not of the result. When terms of opposite sign nearly cancel, near a root, the *relative* error of $\hat p$ can be large: that comes from the polynomial being ill-conditioned there, and no evaluation order fixes it. When all the inputs are integers and every partial result `acc` stays below $2^{53}$, no step rounds at all and Horner is **exact**; the tests check both facts.

### 2.4 The quadratic formula

For $ax^2 + bx + c = 0$ with $a \ne 0$, divide by $a$ and complete the square:

$$\left(x + \frac{b}{2a}\right)^2 = \frac{b^2 - 4ac}{4a^2} \quad\Longrightarrow\quad x = \frac{-b \pm \sqrt{D}}{2a}, \qquad D = b^2 - 4ac .$$

$D > 0$ gives two real roots, $D = 0$ a double root $-b/2a$, $D < 0$ none (two complex roots, which `quadratic_roots` rejects). Expanding $a(x - x_1)(x - x_2)$ and matching coefficients gives **Vieta's formulas**, true for any quadratic:

$$x_1 + x_2 = -\frac{b}{a}, \qquad x_1 x_2 = \frac{c}{a} .$$

### 2.5 Cancellation and the stable formula

When $b^2 \gg 4|ac|$, $\sqrt D$ is very close to $|b|$. One of the two signs in $-b \pm \sqrt D$ then subtracts two nearly equal numbers. Each has a rounding error of about $u|b|$, the difference is tiny, and the error is a large fraction of it: **catastrophic cancellation**. The other sign adds two numbers of the same sign and is accurate. So compute only the safe one,

$$q = -\tfrac12\left(b + \operatorname{sign}(b)\sqrt D\right), \qquad x = \frac{q}{a}, \qquad x = \frac{c}{q},$$

where the second root comes from Vieta: $x_1 x_2 = c/a$ and $x_1 = q/a$ give $x_2 = c/(a x_1) = c/q$. Neither step subtracts. Two details make it robust. $\operatorname{sign}(b)$ must be $+1$ when $b = 0$ (`numpy.sign(0)` is 0, which makes $q = 0$ for $x^2 - 4$); then $q = 0$ only when $b = 0$ and $D = 0$, the double root 0. And $b^2$ overflows to infinity once $|b| > 10^{154}$: dividing $a$, $b$, $c$ by the same power of two first (exact in binary floating point, and the roots do not change) keeps every intermediate in range. Return the roots sorted, $x_1 \le x_2$.

## 3. Worked example by hand

**Horner.** $p(x) = 1 + 2x + 3x^2$, coefficients $[1, 2, 3]$, at $x = 2$:

| step | $k$ | `acc * x + c[k]` | `acc` |
|---|---|---|---|
| 1 | 2 | $0 \cdot 2 + 3$ | 3 |
| 2 | 1 | $3 \cdot 2 + 2$ | 8 |
| 3 | 0 | $8 \cdot 2 + 1$ | **17** |

Check: $1 + 4 + 12 = 17$. Two multiplications, two additions. At $x = 0, 1, -1$ the values are $1, 6, 2$: `test_hand_example_horner`.

**A friendly quadratic.** $x^2 - 5x + 6$: $D = 25 - 24 = 1$, $b = -5 < 0$ so $\operatorname{sign}(b) = -1$, $q = -\tfrac12(-5 - 1) = 3$. Roots $q/a = 3$ and $c/q = 6/3 = 2$, sorted $(2, 3)$; Vieta: $2 + 3 = 5 = -b/a$, $2 \cdot 3 = 6 = c/a$.

**The cancellation case.** $x^2 - 10^8 x + 1$. Exactly, $D = 10^{16} - 4$ and the roots are $5 \cdot 10^7 \pm \sqrt{25 \cdot 10^{14} - 1}$, about $10^8$ and $10^{-8}$. In float64:

- $\sqrt{10^{16} - 4} = 10^8 \sqrt{1 - 4 \cdot 10^{-16}} \approx 10^8 - 2 \cdot 10^{-8}$. Near $10^8$ consecutive float64 values are $1.49 \times 10^{-8}$ apart, so it rounds to $10^8 - 1.49 \times 10^{-8}$.
- Textbook small root: $(10^8 - (10^8 - 1.49 \times 10^{-8}))/2 = 7.45 \times 10^{-9}$. The true value is $1.00 \times 10^{-8}$: **25% wrong**, with every digit lost to the subtraction.
- Stable: $q = -\tfrac12(-10^8 - (10^8 - 1.49 \times 10^{-8})) = 10^8$ to 16 digits, so $x = q/a = 10^8$ and $x = c/q = 1/10^8 = 10^{-8}$, correct to the last digit.

Both quadratics are `test_hand_example_roots`.

## 4. The interface

```python
# python/tinyllm/num/poly.py
def horner(coeffs: Sequence[float], x: ArrayLike) -> NDArray: ...      # ascending coeffs, float64, x's shape
def quadratic_roots(a: float, b: float, c: float) -> tuple[float, float]: ...   # x1 <= x2, no cancellation
```

`quadratic_roots` raises `ValueError` when $a = 0$ (a linear equation), when $D < 0$ (complex roots), and when a coefficient is NaN or infinite.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_horner` | unit, smoke | section 3: $p(2) = 17$, and $p$ at 0, 1, $-1$ | you and the tests agree on the order |
| `test_hand_example_roots` | unit, smoke | $(2, 3)$, and $(10^{-8}, 10^8)$ to float64 precision | the cancellation case by hand |
| `test_coefficients_are_ascending` | boundary | $[1, 0]$ is 1, $[0, 1]$ is $x$, $2 + x^3$ at 3 is 29 | `M02.1` and `M09.6` store Taylor coefficients ascending |
| `test_horner_exact_on_integer_polynomials` | property | random integer polynomials agree with Python's exact integers | no rounding when nothing needs rounding |
| `test_horner_within_error_bound` | property | real coefficients stay within $\gamma_{2n}\sum|c_k||x|^k$ of exact rational arithmetic | section 2.3's bound holds |
| `test_horner_golden` | golden | Taylor polynomials of $\exp$ and $\cos$ and $(x-1)\cdots(x-5)$ against mpmath | the polynomials `M02.1` and `M09.6` evaluate |
| `test_horner_shapes` | boundary | scalars, 2-D arrays, float32 input, no coefficients, one coefficient | called on whole arrays of points |
| `test_roots_golden_cancellation` | golden | 15 quadratics with $b^2 \gg 4ac$ against mpmath at 80 digits | both roots right to float64 precision |
| `test_b_zero` | boundary | $x^2 - 4$ and $2x^2 - 8$ give $(-2, 2)$ | $\operatorname{sign}(0)$ |
| `test_special_roots` | unit | a zero root, a double root, $a < 0$, $a \ne 1$ | the second root is $c/q$ |
| `test_roots_ascending` | unit | $x_1 \le x_2$ for every sign pattern | callers rely on the order |
| `test_vieta_relations` | property | $x_1 + x_2 = -b/a$ and $x_1 x_2 = c/a$ on 200 random quadratics | an independent check of both roots |
| `test_huge_b_does_not_overflow` | boundary | $x^2 + 10^{200}x + 1$ gives $(-10^{200}, -10^{-200})$ | scaling before squaring |
| `test_rejects_non_quadratics` | boundary | $a = 0$, $D < 0$, NaN, and infinity raise `ValueError` | errors, not NaN pairs |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. reading the coefficients in descending order (the `np.polyval` convention) | $[1, 2, 3]$ evaluates $x^2 + 2x + 3$; Taylor polynomials come out reversed | `test_coefficients_are_ascending` (mutant `s01`) |
| 2. the textbook formula $(-b \pm \sqrt D)/2a$ | the small root of $x^2 - 10^8x + 1$ is 25% wrong; with $b = 10^{15}$ it is 0 | `test_hand_example_roots`, `test_roots_golden_cancellation` (mutant `s02`) |
| 3. $\operatorname{sign}(0) = 0$ | $x^2 - 4$ returns $(0, 0)$, or divides by zero | `test_b_zero` (mutant `s03`) |
| 4. squaring $b$ unscaled | $x^2 + 10^{200}x + 1$ returns $(-\infty, 0)$ | `test_huge_b_does_not_overflow` (mutant `s08`) |
| 5. taking the second root as $c/x_1$ | correct only when $a = 1$; $2x^2 - 10x + 12$ is wrong | `test_special_roots` (mutant `s04`) |
| 6. returning the roots in the order computed | $x_1 > x_2$ whenever $b < 0$ | `test_roots_ascending` (mutant `s05`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M00.1` | exponents and powers (reading) |
| Back | `lang.01` | numpy arrays and broadcasting (reading) |
| Forward | `M02.1` | `exp_range_reduced` evaluates the Taylor polynomial of $e^x$ with `horner` after range reduction |
| Forward | `M09.6` | `tl_expf` in C: range reduction, then a degree-5 or 6 polynomial by the same Horner loop, within 4 ulp |
| Forward | `M09.2` | stable rewrites of formulas that cancel, the theme section 2.5 starts (reading) |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `horner` | Cephes `polevl` and `p1evl` | the Horner loop behind many C math libraries, with descending coefficients and an implicit leading 1 | Cephes `polevl.c` |
| `horner` in an `expf` | musl and glibc `expf` | a short polynomial after range reduction, evaluated with Estrin-style pairing for instruction-level parallelism | musl `src/math/expf.c` |
| `horner` over arrays | `numpy.polynomial.polynomial.polyval` | ascending coefficients like yours, plus fitting, roots, and Chebyshev bases | numpy `numpy/polynomial/` |
| `quadratic_roots` | *Numerical Recipes* section 5.6; Kahan's notes on the discriminant | the same $q$ trick, and an extra-precise $b^2 - 4ac$ for nearly double roots | Kahan, "On the Cost of Floating-Point Computation Without Extra-Precise Arithmetic" (2004) |
