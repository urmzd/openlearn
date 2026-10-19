# Polynomials, Horner, stable quadratic roots

## Overview

| | |
|---|---|
| **Module** | `M00.4` · build · Python · Pass 2 · 1 to 2 h reading |
| **Solution** | [`course/solutions/python/tinyllm/num/poly.py`](../../../../course/solutions/python/tinyllm/num/poly.py): `horner`, `quadratic_roots` |
| **Contract** | [`course/contracts/py/tinyllm/num/poly.pyi`](../../../../course/contracts/py/tinyllm/num/poly.pyi) |
| **Needs** | nothing to call. Reading: `M00.1` (exponents), `lang.01` (numpy arrays) |
| **Used by** | `M02.1` evaluates its range-reduced Taylor polynomial for $e^x$ with `horner` · later `M09.6` ports that code to C as `tl_expf` |
| **Optional depth** | OpenStax, *College Algebra 2e* (free), ch. 5 (polynomial functions); Higham, *Accuracy and Stability of Numerical Algorithms*, ch. 5 (Horner's error bound); Press et al., *Numerical Recipes*, section 5.6 (quadratic and cubic equations) |

## Key Takeaways

- A polynomial is its list of **coefficients**; in this course they are stored in **ascending** order of power, so `coeffs[k]` multiplies $x^k$.
- **Horner's rule** evaluates a degree-$n$ polynomial with $n$ multiplications and $n$ additions by nesting, $c_0 + x(c_1 + x(c_2 + \cdots))$, and its error is bounded by $\gamma_{2n}\sum_k |c_k||x|^k$.
- On integer coefficients and points it is **exact** as long as every partial result stays below $2^{53}$.
- The textbook quadratic formula **cancels** when $b^2 \gg 4ac$: the small root comes out with few or no correct digits. Computing $q = -(b + \operatorname{sign}(b)\sqrt{b^2 - 4ac})/2$ and the roots $q/a$ and $c/q$ never subtracts nearly equal numbers.
- Two edge cases break naive code: $\operatorname{sign}(0)$ must count as $+1$, and $b^2$ overflows for $|b| \approx 10^{200}$ unless the coefficients are scaled first.

## How to read this chapter

Sections 1 and 2 build Horner's rule and the stable quadratic formula from first principles, including why the school formula fails in floating point. Section 3 works both by hand. Section 4 fixes the interface, and section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

Every activation function and softmax in the models calls $e^x$, and so far numpy has computed it for you. In Pass 6 the C kernels (`L9.2` softmax, `L9.6` SiLU) need their own `tl_expf`, and `M09.6` builds it the way every math library does: reduce $x$ to a small range, then evaluate a **polynomial** that approximates $e^x$ there. `M02.1` first derives those polynomials (Taylor series) and evaluates them in Python. Both evaluate with Horner's rule, the subject of this module, and both depend on how evaluation order changes the rounding error. The second half of the module is the smallest example of the course's numerical theme: a formula that is correct on paper and wrong in floating point. The quadratic formula you learned in school loses the small root of $x^2 - 10^8 x + 1$ entirely, and a two-line rewrite fixes it.

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

The right side is the size of the largest terms, not of the result. When terms of opposite sign nearly cancel, near a root, the *relative* error of $\hat p$ can be large: that comes from the polynomial being ill-conditioned there, and no evaluation order fixes it. When all the inputs are integers and every partial result `acc` stays below $2^{53}$, no step rounds at all and Horner is **exact**.

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

Check: $1 + 4 + 12 = 17$. Two multiplications, two additions. At $x = 0, 1, -1$ the values are $1, 6, 2$.

**A friendly quadratic.** $x^2 - 5x + 6$: $D = 25 - 24 = 1$, $b = -5 < 0$ so $\operatorname{sign}(b) = -1$, $q = -\tfrac12(-5 - 1) = 3$. Roots $q/a = 3$ and $c/q = 6/3 = 2$, sorted $(2, 3)$; Vieta: $2 + 3 = 5 = -b/a$, $2 \cdot 3 = 6 = c/a$.

**The cancellation case.** $x^2 - 10^8 x + 1$. Exactly, $D = 10^{16} - 4$ and the roots are $5 \cdot 10^7 \pm \sqrt{25 \cdot 10^{14} - 1}$, about $10^8$ and $10^{-8}$. In float64:

- $\sqrt{10^{16} - 4} = 10^8 \sqrt{1 - 4 \cdot 10^{-16}} \approx 10^8 - 2 \cdot 10^{-8}$. Near $10^8$ consecutive float64 values are $1.49 \times 10^{-8}$ apart, so it rounds to $10^8 - 1.49 \times 10^{-8}$.
- Textbook small root: $(10^8 - (10^8 - 1.49 \times 10^{-8}))/2 = 7.45 \times 10^{-9}$. The true value is $1.00 \times 10^{-8}$: **25% wrong**, with every digit lost to the subtraction.
- Stable: $q = -\tfrac12(-10^8 - (10^8 - 1.49 \times 10^{-8})) = 10^8$ to 16 digits, so $x = q/a = 10^8$ and $x = c/q = 1/10^8 = 10^{-8}$, correct to the last digit.

## 4. The interface

```python
# python/tinyllm/num/poly.py
def horner(coeffs: Sequence[float], x: ArrayLike) -> NDArray: ...      # ascending coeffs, float64, x's shape
def quadratic_roots(a: float, b: float, c: float) -> tuple[float, float]: ...   # x1 <= x2, no cancellation
```

`quadratic_roots` raises `ValueError` when $a = 0$ (a linear equation), when $D < 0$ (complex roots), and when a coefficient is NaN or infinite.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked Horner example | unit | section 3: $p(2) = 17$, and $p$ at 0, 1, $-1$ | everyone agrees on the order |
| the hand-worked quadratics | unit | $(2, 3)$, and $(10^{-8}, 10^8)$ to float64 precision | the cancellation case by hand |
| ascending coefficients | boundary | $[1, 0]$ is 1, $[0, 1]$ is $x$, $2 + x^3$ at 3 is 29 | `M02.1` and `M09.6` store Taylor coefficients ascending |
| exact on integer polynomials | property | random integer polynomials agree with Python's exact integers | no rounding when nothing needs rounding |
| within Horner's error bound | property | real coefficients stay within $\gamma_{2n}\sum|c_k||x|^k$ of exact rational arithmetic | section 2.3's bound holds |
| reference polynomials | golden | Taylor polynomials of $\exp$ and $\cos$ and $(x-1)\cdots(x-5)$ against mpmath | the polynomials `M02.1` and `M09.6` evaluate |
| input shapes | boundary | scalars, 2-D arrays, float32 input, no coefficients, one coefficient | called on whole arrays of points |
| quadratics that cancel | golden | 15 quadratics with $b^2 \gg 4ac$ against mpmath at 80 digits | both roots right to float64 precision |
| $b = 0$ | boundary | $x^2 - 4$ and $2x^2 - 8$ give $(-2, 2)$ | $\operatorname{sign}(0)$ |
| special roots | unit | a zero root, a double root, $a < 0$, $a \ne 1$ | the second root is $c/q$ |
| ascending roots | unit | $x_1 \le x_2$ for every sign pattern | callers rely on the order |
| Vieta's relations | property | $x_1 + x_2 = -b/a$ and $x_1 x_2 = c/a$ on 200 random quadratics | an independent check of both roots |
| huge $b$ | boundary | $x^2 + 10^{200}x + 1$ gives $(-10^{200}, -10^{-200})$ | scaling before squaring |
| non-quadratics | boundary | $a = 0$, $D < 0$, NaN, and infinity raise `ValueError` | errors, not NaN pairs |

## 5. The solution, step by step

### From first principles

1. **Store coefficients ascending, evaluate descending.** `coeffs[k]` multiplies $x^k$ (section 2.1), but Horner's nesting (section 2.2) starts from the innermost bracket, which holds the highest coefficient $c_n$. So the loop walks the list in reverse.
2. **Start from zero, not from $c_n$.** With `acc = 0`, the first step computes $0 \cdot x + c_n = c_n$, so no special case is needed for the first coefficient, and an empty list returns the zero polynomial. Making `acc` an array of zeros shaped like $x$ lets one loop evaluate at every point at once.
3. **Reject what has no two real roots.** Non-finite coefficients, $a = 0$ (linear), and $D < 0$ (complex) all raise, so a caller never receives a NaN pair it has to recognise.
4. **Scale before squaring.** $b^2$ overflows for $|b| > 10^{154}$. Multiplying $a, b, c$ by the same nonzero number does not move the roots, and multiplying by a power of two is exact in binary floating point. Scaling by $2^{-e}$, where $2^e$ is just above the largest coefficient magnitude, puts the largest coefficient in $[\tfrac12, 1)$, so $b^2$ and $4ac$ cannot overflow.
5. **Compute only the sign of $\pm$ that adds.** With $\operatorname{sign}(b)$ chosen to match $b$, the sum $b + \operatorname{sign}(b)\sqrt D$ adds two numbers of the same sign, so $q$ carries full relative precision (section 2.5). $b = 0$ must take $+1$: then $q = -\tfrac12\sqrt D$, which is nonzero whenever $D > 0$.
6. **Get the other root from Vieta, not from the dangerous sign.** $q/a$ is one root; since $x_1 x_2 = c/a$, the other is $c/(a \cdot q/a) = c/q$. A division does not cancel.
7. **Handle the one remaining zero.** $q = 0$ only if $b = 0$ and $D = 0$, which forces $c = 0$: the equation is $ax^2 = 0$, with the double root 0. Return it rather than dividing $c/q = 0/0$.
8. **Sort.** Which of $q/a$ and $c/q$ is smaller depends on the signs of $a$, $b$, $c$, so the result is sorted explicitly.

### The code, annotated

```python
def horner(coeffs: Sequence[float], x: ArrayLike) -> NDArray:
    c = [float(v) for v in coeffs]          # accepts any sequence, computes in float64
    xs = np.asarray(x, dtype=np.float64)    # float32 or int input is promoted; shape kept
    acc = np.zeros_like(xs)  # the empty sum: the zero polynomial
    # Innermost bracket first: start at the highest power and work down.
    for ck in reversed(c):
        acc = acc * xs + ck                 # one multiply and one add per coefficient
    return acc
```

`reversed(c)` is the whole difference between ascending and descending storage (pitfall 1). On the section 3 example, `reversed([1, 2, 3])` yields 3, 2, 1, and `acc` goes $3, 8, 17$. Because `acc` and `xs` are arrays, the same four lines evaluate a scalar, a vector of points, or a 2-D grid; `np.zeros_like` of a 0-d array is a 0-d array, so a scalar `x` gives a 0-d result.

```python
def quadratic_roots(a: float, b: float, c: float) -> tuple[float, float]:
    a, b, c = float(a), float(b), float(c)
    if not all(math.isfinite(v) for v in (a, b, c)):
        raise ValueError(f"coefficients must be finite, got {a}, {b}, {c}")
    if a == 0.0:
        raise ValueError("a == 0: the equation is linear, not quadratic")
    # Scale by a power of two (exact) so that b * b cannot overflow when |b|
    # is near 1e200; the roots of s a x^2 + s b x + s c are the same.
    e = math.frexp(max(abs(a), abs(b), abs(c)))[1]   # max = m * 2^e with 0.5 <= m < 1
    a, b, c = math.ldexp(a, -e), math.ldexp(b, -e), math.ldexp(c, -e)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        raise ValueError(f"discriminant {disc} < 0: the roots are complex")
    # q = -(b + sign(b) sqrt(disc)) / 2 adds two numbers of the same sign, so
    # nothing cancels. sign(0) is taken as +1: numpy's sign(0) = 0 would
    # make q = 0 for x^2 - 4.
    sign = 1.0 if b >= 0.0 else -1.0
    q = -0.5 * (b + sign * math.sqrt(disc))
    if q == 0.0:  # b == 0 and disc == 0, so c == 0: the double root 0
        return (0.0, 0.0)
    # One root is q / a; Vieta (x1 x2 = c / a) gives the other as c / q.
    r1, r2 = q / a, c / q
    return (r1, r2) if r1 <= r2 else (r2, r1)
```

`math.frexp(v)` returns $(m, e)$ with $v = m \cdot 2^e$ and $\tfrac12 \le m < 1$, and `math.ldexp(v, -e)` multiplies by $2^{-e}$ by adjusting the exponent field only, so the scaling introduces no rounding. The roots come out at the original scale without unscaling, because $q/a$ and $c/q$ are ratios of scaled quantities: for $x^2 + 10^{200}x + 1$, $b$ becomes about $0.65$ and $a = c \approx 2^{-665}$; $4ac$ underflows to 0, which is harmless because it is negligible next to $b^2$; $q \approx -0.65$, $q/a = -10^{200}$ and $c/q = -10^{-200}$.

Trace the cancellation case of section 3, $x^2 - 10^8x + 1$: $b < 0$ so `sign` is $-1$, $q = -\tfrac12(b - \sqrt D)$ adds two negative numbers, giving $q$ equivalent to $10^8$ at the original scale; `q / a` is $10^8$ and `c / q` is $10^{-8}$, both to full precision, and sorting returns $(10^{-8}, 10^8)$.

Full solution: [`poly.py`](../../../../course/solutions/python/tinyllm/num/poly.py).

### Why this solution

- **Cost.** `horner` does $n$ multiplications and $n$ additions per point, the minimum for a general polynomial, and $O(1)$ extra memory per point. Computing each $x^k$ separately costs more multiplications and rounds more, and is no more accurate.
- **No numpy `polyval`.** `np.polyval` uses descending coefficients, the opposite of the convention `M02.1` and `M09.6` share; a hand-written loop also maps one-to-one onto the C loop in `tl_expf`.
- **The stable formula over higher precision.** One could evaluate the textbook formula in extended precision. That only postpones the problem to larger $b$, is not available in the C port, and costs more. The $q$ rewrite is exact algebra and fixes the cancellation at every scale.
- **Scaling over special-casing.** Testing for large $|b|$ and switching formulas needs a threshold; scaling by a power of two always works and is exact.
- **Tradeoffs accepted.** Scaling by the *largest* coefficient can underflow a coefficient more than about 600 orders of magnitude smaller: $10^{200}x^2 + 10^{-200} = 0$ has no real roots, but its scaled $c$ underflows to 0 and the function returns $(0, 0)$. No caller has coefficients spanning that range. The stable formula also does not make $D$ itself accurate: for nearly double roots ($b^2 \approx 4ac$) the discriminant cancels and the roots keep only about half their digits, which is the ill-conditioning of the problem, not of the formula (Kahan's notes in Going further compute $D$ in extra precision).

### Explain it back

**1. Why does `horner` start with `acc = 0` rather than `acc = c[n]`?**

<details><summary>Answer</summary>

The first iteration computes $0 \cdot x + c_n = c_n$ exactly, so starting from 0 gives the same result without indexing the last element. It also makes the empty coefficient list return zeros of the right shape, and builds `acc` with the shape of `x` from the start, so scalars and arrays take the same path.

</details>

**2. Why does the stable formula compute the second root as $c/q$ and not $c/x_1$?**

<details><summary>Answer</summary>

Vieta's product is $x_1 x_2 = c/a$, so $x_2 = c/(a x_1)$. With $x_1 = q/a$, that is $c/q$. Writing $c/x_1$ drops the factor $a$ and is right only when $a = 1$: for $2x^2 - 10x + 12$, $q = 6$, $x_1 = 3$, and $c/x_1 = 4$ instead of the correct $c/q = 2$.

</details>

**3. What goes wrong with `np.sign(b)` in place of the explicit `1.0 if b >= 0.0 else -1.0`?**

<details><summary>Answer</summary>

`np.sign(0.0)` is 0, so for $b = 0$ the formula becomes $q = -\tfrac12(0 + 0 \cdot \sqrt D) = 0$, and $c/q$ divides by zero for any equation like $x^2 - 4$. Taking $+1$ gives $q = -\tfrac12\sqrt{16} = -2$ and the roots $q/a = -2$ and $c/q = 2$.

</details>

**4. Why scale by a power of two, and why does the code not need to scale the roots back?**

<details><summary>Answer</summary>

Multiplying by $2^{-e}$ changes only the exponent field, so $a$, $b$, $c$ are scaled with no rounding error, and the scaled equation has exactly the same roots. Both root formulas are ratios, $q/a$ and $c/q$, where $q$ is linear in the coefficients, so the scale factor cancels and the roots come out at the original scale.

</details>

**5. Horner is exact on integer inputs while partial results stay below $2^{53}$. Why does that bound matter, and why is the relative error large near a root even though Horner is "accurate"?**

<details><summary>Answer</summary>

Every integer up to $2^{53}$ is a float64, so while `acc * x + c` stays in that range neither operation rounds. Near a root, the result is a small difference of large terms; Horner's error bound is proportional to $\sum |c_k||x|^k$, the size of those terms, not to $|p(x)|$. A small absolute error divided by a near-zero value is a large relative error, and that comes from the polynomial's conditioning, so no evaluation order removes it.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. reading the coefficients in descending order (the `np.polyval` convention) | $[1, 2, 3]$ evaluates $x^2 + 2x + 3$; Taylor polynomials come out reversed | iterates `reversed(c)` over ascending storage; step 1 |
| 2. the textbook formula $(-b \pm \sqrt D)/2a$ | the small root of $x^2 - 10^8x + 1$ is 25% wrong; with $b = 10^{15}$ it is 0 | computes only $q$, which adds same-sign numbers; step 5 |
| 3. $\operatorname{sign}(0) = 0$ | $x^2 - 4$ returns $(0, 0)$, or divides by zero | `sign = 1.0 if b >= 0.0 else -1.0` |
| 4. squaring $b$ unscaled | $x^2 + 10^{200}x + 1$ returns $(-\infty, 0)$ | scales all three coefficients by $2^{-e}$ with `frexp`/`ldexp`; step 4 |
| 5. taking the second root as $c/x_1$ | correct only when $a = 1$; $2x^2 - 10x + 12$ is wrong | uses Vieta's $c/q$; step 6 |
| 6. returning the roots in the order computed | $x_1 > x_2$ whenever $b < 0$ | sorts the pair before returning; step 8 |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M00.1` | exponents and powers (reading) |
| Back | `lang.01` | numpy arrays and broadcasting (reading) |
| Forward | `M02.1` | `exp_range_reduced` evaluates the Taylor polynomial of $e^x$ with `horner` after range reduction |
| Forward | `M09.6` | `tl_expf` in C: range reduction, then a degree-5 or 6 polynomial by the same Horner loop, within 4 ulp |
| Forward | `M09.2` | stable rewrites of formulas that cancel, the theme section 2.5 starts (reading) |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `horner` | Cephes `polevl` and `p1evl` | the Horner loop behind many C math libraries, with descending coefficients and an implicit leading 1 | Cephes `polevl.c` |
| `horner` in an `expf` | musl and glibc `expf` | a short polynomial after range reduction, evaluated with Estrin-style pairing for instruction-level parallelism | musl `src/math/expf.c` |
| `horner` over arrays | `numpy.polynomial.polynomial.polyval` | ascending coefficients like the solution's, plus fitting, roots, and Chebyshev bases | numpy `numpy/polynomial/` |
| `quadratic_roots` | *Numerical Recipes* section 5.6; Kahan's notes on the discriminant | the same $q$ trick, and an extra-precise $b^2 - 4ac$ for nearly double roots | Kahan, "On the Cost of Floating-Point Computation Without Extra-Precise Arithmetic" (2004) |
