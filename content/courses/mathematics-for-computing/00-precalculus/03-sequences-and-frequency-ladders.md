# Sequences, geometric series, frequency ladders

## Overview

| | |
|---|---|
| **Module** | `M00.3` · build · Python · Pass 2 · 1 to 2 h reading |
| **Solution** | [`course/solutions/python/tinyllm/num/series.py`](../../../../course/solutions/python/tinyllm/num/series.py): `geometric`, `geometric_sum` (with the helper `_ones_sum`), `rope_inv_freq`, `alibi_slopes` |
| **Contract** | [`course/contracts/py/tinyllm/num/series.pyi`](../../../../course/contracts/py/tinyllm/num/series.pyi) |
| **Needs** | nothing to call. Reading: `M00.1` (powers, `log1p` and `expm1`), `lang.01` |
| **Used by** | `M02.2` the EMA's weights and bias correction call `geometric` and `geometric_sum` · later `L7.3` RoPE `inv_freq` and `L7.4` ALiBi slopes and YaRN's per-frequency ramps · `M10.4` learning-rate schedules (reading) · later: `L5.4` |
| **Optional depth** | OpenStax, *Precalculus 2e* (free), ch. 11 (sequences and series); Press, Smith, and Lewis, "Train Short, Test Long: Attention with Linear Biases" (ALiBi, 2022), section 3; Su et al., "RoFormer" (2021), section 3.3 |

## Key Takeaways

- A **geometric sequence** multiplies by the same ratio $r$ at every step: $a, ar, ar^2, \ldots$.
- Its first $n$ terms add up to $a(1 - r^n)/(1 - r)$, by subtracting $r$ times the sum from the sum.
- Near $r = 1$ that formula **cancels**: it subtracts two nearly equal numbers and loses half its digits. Rewriting it with $\operatorname{expm1}$ and $\operatorname{log1p}$ keeps them all; Adam's bias correction lives exactly there.
- RoPE's **inverse frequencies** $b^{-2i/d}$ form a geometric ladder from one radian per token down to about $1/b$, so pairs of dimensions see position at every scale from a few tokens to tens of thousands.
- **ALiBi slopes** are a geometric ladder of $2^{-8/n}$; head counts that are not powers of two interleave a second ladder.

## How to read this chapter

Sections 1 and 2 build sequences, their sums, and the two position ladders from first principles, including why the textbook sum fails near $r = 1$. Section 3 works every function by hand. Section 4 fixes the interface, and section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

`M00.2` gave you rotations, and RoPE rotates each pair of a query by an angle proportional to the token's position. The open question is how fast each pair should turn. If every pair turned by 1 radian per token, positions $2\pi \approx 6.3$ tokens apart would look identical; if every pair turned slowly, neighbours would be indistinguishable. RoPE answers with a ladder of speeds, a **geometric sequence**, and so does ALiBi, the other position scheme in `L7.4`. The same mathematics appears in training: the moving averages of Adam (`M02.2`, `M10.3`) weight past gradients by a geometric sequence, and correct for its partial sum. The SmolLM2 loader (`L7.9`) reads `rope_theta = 100000` from a `config.json` and must turn it into exactly the frequencies Hugging Face computes. This module builds the sequences, their sums (accurately, including the hard case), and both ladders.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $a$ | the first term of a sequence | `float` |
| $r$ | the common ratio | `float` |
| $n$ | the number of terms | `int` |
| $a_j$ | term $j$, counted from $j = 0$ | `float` |
| $S_n$ | the sum of the first $n$ terms, $\sum_{j=0}^{n-1} a r^j$ | `float` |
| $\beta$ | the decay of an exponential moving average, $0 < \beta < 1$ | `float` |
| $g_t, m_t$ | the value seen at step $t$ and the moving average after it | `float` |
| $d$ | the rotary dimension (`d_rot`): the number of coordinates RoPE rotates, even | `int` |
| $b$ | the RoPE base (`rope_theta` in a `config.json`) | `float` |
| $\omega_i$ | inverse frequency of pair $i$, $b^{-2i/d}$, in radians per position | `float[d/2]` |
| $\lambda_i$ | wavelength of pair $i$, $2\pi/\omega_i$, in positions | `float` |
| $H$ | the number of attention heads | `int` |
| $s_h$ | the ALiBi slope of head $h$ | `float[H]` |
| $\varepsilon$ | float64 unit roundoff, $2^{-53} \approx 1.1 \times 10^{-16}$ | |

### 2.1 Sequences

A **sequence** is a list of numbers indexed by position: $a_0, a_1, a_2, \ldots$. Two families matter here. An **arithmetic** sequence adds the same step, $a_j = a + jd$ (positions $0, 1, 2, \ldots$ are one). A **geometric** sequence multiplies by the same ratio, $a_j = a r^j$: $1, \tfrac12, \tfrac14, \ldots$ has $a = 1$, $r = \tfrac12$. With $0 < r < 1$ the terms shrink toward 0; with $r > 1$ they grow without bound; with $r < 0$ they alternate in sign. Compute term $j$ as `a * r**j`, not by multiplying the previous term by $r$: a running product rounds once per step and its error grows with $j$.

### 2.2 The geometric sum

Let $S_n = a + ar + \cdots + ar^{n-1}$. Multiply by $r$: $rS_n = ar + \cdots + ar^{n-1} + ar^n$. Subtracting, every term but two cancels:

$$S_n - rS_n = a - ar^n \quad\Longrightarrow\quad S_n = a\,\frac{1 - r^n}{1 - r}, \qquad r \ne 1,$$

and $S_n = na$ when $r = 1$ (every term is $a$). When $|r| < 1$, $r^n$ approaches 0 as $n$ grows, so the infinite sum converges: $a + ar + ar^2 + \cdots = a/(1 - r)$. That is why $1 + \tfrac12 + \tfrac14 + \cdots = 2$, and why a repeating decimal is a fraction: $0.2727\ldots = 27/100 + 27/100^2 + \cdots = (27/100)/(1 - 1/100) = 3/11$.

### 2.3 The exponential moving average is a geometric series

An exponential moving average (EMA) with decay $\beta$ updates $m_t = \beta m_{t-1} + (1 - \beta) g_t$, starting from $m_0 = 0$. Unrolling,

$$m_t = (1 - \beta)\left(g_t + \beta g_{t-1} + \beta^2 g_{t-2} + \cdots + \beta^{t-1} g_1\right),$$

so the weights are a geometric sequence $(1 - \beta)\beta^j$, newest first, and they add up to $(1 - \beta)\frac{1 - \beta^t}{1 - \beta} = 1 - \beta^t$, not 1. Early on the average is biased toward its starting value 0: after one step with $\beta = 0.999$, $m_1 = 0.001\,g_1$. Adam divides by $1 - \beta^t$ to undo exactly this (`M02.2`, `M10.3`). With $\beta = 0.999$ the ratio is $1 - 10^{-3}$, close to 1, which is the hard case of section 2.6.

### 2.4 Frequency ladders: RoPE

RoPE rotates pair $i$ of a query or key at position $p$ by the angle $p\,\omega_i$ (`M00.2`, section 2.5), with

$$\omega_i = b^{-2i/d}, \qquad i = 0, 1, \ldots, \tfrac d2 - 1 .$$

This is a geometric sequence with first term $b^0 = 1$ and ratio $b^{-2/d}$. Pair 0 turns one radian per token and repeats every $\lambda_0 = 2\pi \approx 6.3$ positions; the last pair turns $b^{-(d-2)/d}$ radians per token, slightly faster than $1/b$, and repeats only after about $2\pi b$ positions. Like the hands of a clock (seconds, minutes, hours), fast pairs resolve neighbouring tokens and slow pairs tell far-apart ones apart, and the geometric spacing gives every scale the same number of pairs. The exponent steps by $2/d$ because there are $d/2$ pairs spread over the exponents from 0 to almost 1. Models with **partial rotary** embeddings rotate only the first $d$ coordinates of each head; the formula is the same with that $d$. The SmolLM2-135M `config.json` says `hidden_size = 576` over `num_attention_heads = 9` (a head size of 64) and `rope_theta = 100000`; `rope_inv_freq(64, 1e5)` must match what Hugging Face computes from those (in float32, to float32 precision), or the loaded checkpoint sees the wrong positions.

### 2.5 Frequency ladders: ALiBi

ALiBi (Press et al.) adds no rotation at all. Head $h$ subtracts $s_h$ times the distance between query and key from each attention score, so far tokens are penalized linearly, and each head has its own slope. For $H$ a power of two the slopes are the geometric sequence with first term and ratio $2^{-8/H}$:

$$s_h = 2^{-8h/H}, \qquad h = 1, \ldots, H,$$

from $2^{-8/H}$ (steep: the head looks at the last few tokens) down to $2^{-8} = 1/256$ (flat: it looks hundreds of tokens back). For other $H$, with $p$ the largest power of two below $H$, the paper's code takes the $p$ slopes for $p$ heads, then fills the remaining $H - p$ heads with every other slope of the $2p$-head ladder, $2^{-8 \cdot 1/(2p)}, 2^{-8 \cdot 3/(2p)}, \ldots$, which fall between the existing ones. Find $p$ with integer arithmetic (`1 << (H.bit_length() - 1)`), never with a floating-point $\log_2$.

### 2.6 Floating point near $r = 1$

For $r$ close to 1 the closed form divides one small difference by another. $1 - r$ is computed exactly (subtracting nearby floats is exact), but $r^n$ is rounded to about $\varepsilon$ relative error first, and $1 - r^n$ then subtracts two numbers that agree in most of their digits, keeping the rounding error and losing the rest. With $r = 1 - 2^{-40}$ and $n = 10$, $1 - r^n \approx 9 \times 10^{-12}$ carries an absolute error near $10^{-16}$: about 5 correct digits out of 16.

The cure is to never form $r^n$ near 1. Since $r^n = e^{n \ln r}$,

$$\frac{1 - r^n}{1 - r} = \frac{\operatorname{expm1}(n \cdot \operatorname{log1p}(r - 1))}{r - 1},$$

where $\operatorname{log1p}(u) = \ln(1 + u)$ and $\operatorname{expm1}(v) = e^v - 1$ are computed accurately for small arguments (`math.log1p`, `math.expm1`; `M00.1`). Every quantity in this form is small and accurate, and the result keeps nearly all 16 digits. For $r > 0$ this form is used everywhere; for $r \le 0$ the closed form has no cancellation ($1 - r \ge 1$). When the sum overflows (huge $r$ and $n$), the answer is $\pm\infty$, by the sign of the last term.

## 3. Worked example by hand

**A geometric sum.** $1 + \tfrac12 + \tfrac14 + \tfrac18$: $a = 1$, $r = \tfrac12$, $n = 4$.

$$S_4 = \frac{1 - (1/2)^4}{1 - 1/2} = \frac{15/16}{1/2} = \frac{15}{8} = 1.875 ,$$

which agrees with adding $1 + 0.5 + 0.25 + 0.125$. With $a = 3$ every term triples, so $S_4 = 45/8$.

**The RoPE ladder for $d = 8$, $b = 10000$.** Four pairs, exponents $-2i/8$:

| $i$ | exponent | $\omega_i = 10000^{\text{exponent}}$ | wavelength $2\pi/\omega_i$ |
|---|---|---|---|
| 0 | 0 | 1 | 6.28 positions |
| 1 | $-1/4$ | $0.1$ | 62.8 |
| 2 | $-1/2$ | $0.01$ | 628 |
| 3 | $-3/4$ | $0.001$ | 6283 |

($10000^{1/4} = 10$ because $10^4 = 10000$.) The ratio is $0.1 = 10000^{-2/8}$.

**ALiBi with 6 heads.** $6$ is not a power of two, so $p = 4$. The 4-head ladder has first term and ratio $2^{-8/4} = 2^{-2}$: $2^{-2}, 2^{-4}, 2^{-6}, 2^{-8} = \tfrac14, \tfrac1{16}, \tfrac1{64}, \tfrac1{256}$. The 8-head ladder is $2^{-1}, 2^{-2}, \ldots, 2^{-8}$; every other entry from the first is $2^{-1}, 2^{-3}, 2^{-5}, \ldots$, and the two extra heads take the first two:

$$s = \left[\tfrac14, \tfrac1{16}, \tfrac1{64}, \tfrac1{256}, \tfrac12, \tfrac18\right].$$

The extra slopes $\tfrac12$ and $\tfrac18$ sit between the existing ones, so the six heads still cover the range evenly.

## 4. The interface

```python
# python/tinyllm/num/series.py
def geometric(a: float, r: float, n: int) -> NDArray: ...        # [a, a r, ..., a r^(n-1)]
def geometric_sum(a: float, r: float, n: int) -> float: ...      # accurate near r = 1
def rope_inv_freq(d_rot: int, base: float) -> NDArray: ...       # base ** (-2 i / d_rot), [d_rot/2]
def alibi_slopes(n_heads: int) -> NDArray: ...                   # Press et al., any head count
```

Everything is float64. Bad counts (negative, fractional, or `True`), an odd or non-positive `d_rot`, a base that is not finite and above 1, and a head count below 1 raise `ValueError`.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked geometric sum | unit | section 3: $15/8$, $45/8$, and the four terms | everyone agrees on what $n$ counts |
| the hand-worked RoPE ladder | unit | $d = 8$, $b = 10000$ gives $[1, 0.1, 0.01, 0.001]$ | the exponent is $-2i/d$ |
| the hand-worked six-head ALiBi slopes | unit | the 6-head slopes of section 3 | the non-power-of-two rule |
| edge-case terms | unit | negative and unit ratios, and $n = 0$ (empty) | term 0 is $a$ |
| sums against exact rationals | differential | 66 sums against exact `Fraction` arithmetic on the same inputs | any correct formula passes, any wrong count fails |
| sums near $r = 1$ keep their digits | boundary | $r = 1 \pm 2^{-40}$ and $1 - 10^{-9}$: relative error below $10^{-12}$ | pitfall 1 |
| the EMA bias-correction identity | property | $\sum_{j<t} (1 - \beta)\beta^j = 1 - \beta^t$ for $\beta$ up to $0.999$ | Adam's bias correction (`M02.2`) |
| sums that overflow | boundary | $\pm\infty$ for sums beyond float64, by the sign of the last term | no `OverflowError` escapes |
| bad term counts | boundary | $n = -1$, $2.5$, `True` raise `ValueError` | counts are whole numbers |
| Hugging Face RoPE frequencies | golden | Hugging Face's default RoPE for SmolLM2-135M, Llama 2, Llama 3, and a partial rotary size, to float32 precision | exactly what `L7.9` loads |
| the RoPE ladder's shape | property | first rung 1, constant ratio $b^{-2/d}$, last rung above $1/b$ | the ladder's shape for any $d$ and $b$ |
| bad RoPE arguments | boundary | odd or zero $d$, bases $\le 1$ or infinite raise `ValueError` | a flat or climbing ladder is a bug |
| ALiBi for powers of two | unit | 8 heads: $1/2, \ldots, 1/256$; 16 heads: $2^{-j/2}$; 1 head: $1/256$ | the power-of-two rule |
| BLOOM's ALiBi slopes | golden | BLOOM's `build_alibi_tensor` for 20 head counts from 1 to 128 | twelve of them are not powers of two |
| bad head counts | boundary | 0, $-4$, $2.5$ heads raise `ValueError` | a model has whole heads |

## 5. The solution, step by step

### From first principles

1. **Terms are independent powers.** Term $j$ is $ar^j$ by definition (section 2.1), so each term can be computed directly from $j$. Computing it from the previous term would add one rounding per step; computing $r^j$ directly rounds once. So `geometric` evaluates `a * r**j` over the vector $j = 0, \ldots, n-1$, and $n = 0$ is the empty vector.
2. **The sum factors as $a$ times the sum of ones.** $S_n = a(1 + r + \cdots + r^{n-1})$, so all the numerical care goes into one helper for $1 + r + \cdots + r^{n-1}$, and `geometric_sum` multiplies by $a$ once at the end.
3. **Two exact cases come first.** $n = 0$ sums nothing ($0$); $r = 1$ makes every term 1 ($n$). Both avoid dividing $0/0$.
4. **For $r > 0$, never form $r^n$.** Section 2.6 shows that $1 - r^n$ cancels near $r = 1$. Writing $r^n = e^{n \ln r}$ and $\ln r = \operatorname{log1p}(r - 1)$ turns the numerator into $\operatorname{expm1}(n \operatorname{log1p}(r-1))$. The denominator $r - 1$ is exact for $r$ near 1 (two floats within a factor of 2 of each other subtract exactly), so every piece carries nearly full precision. Far from 1 the form is just as accurate, so there is no threshold to tune: one formula for all $r > 0$.
5. **For $r \le 0$, the closed form is safe.** $\operatorname{log1p}(r - 1)$ is undefined for $r \le 0$, but there $1 - r \ge 1$, so the denominator is never small, and the textbook formula does no harmful cancellation (except $r$ near $-1$, which the contract does not promise).
6. **Overflow has a known answer.** If $|r| > 1$ and $n$ is large, the sum is dominated by its last term $r^{n-1}$, so the result is $\pm\infty$ with that term's sign: $+$ for $r > 0$, and for $r < 0$, $+$ when $n - 1$ is even. Python's `math.expm1` and `float ** int` raise `OverflowError` instead of returning infinity, so the helper catches it and returns the infinity.
7. **RoPE is a geometric ladder with a fixed exponent grid.** With $d/2$ pairs and $\omega_i = b^{-2i/d}$, the solution builds $i = 0, \ldots, d/2 - 1$ as floats and raises $b$ to $-2i/d$ in one vectorised step. The arguments must make a decreasing ladder: $d$ even and at least 2 (there must be whole pairs), $b > 1$ and finite.
8. **ALiBi is two ladders.** Find $p$, the largest power of two $\le H$, exactly with bit arithmetic. The first $p$ slopes are `geometric(start, start, p)` with $\text{start} = 2^{-8/p}$, which gives $\text{start}^{1}, \ldots, \text{start}^{p}$. If $H > p$, build the $2p$-head ladder the same way, take every other entry from the first (`[0::2]`), and keep the first $H - p$ of them, exactly the six-head computation of section 3.

### The code, annotated

Argument checking is shared by every function that takes a count. `True` is an `int` in Python, so it must be rejected explicitly:

```python
def _check_n(n: int) -> int:
    # bool first: isinstance(True, int) is True, but a count of True makes no sense.
    # int(n) != n rejects 2.5 while accepting 3.0.
    if isinstance(n, bool) or int(n) != n or n < 0:
        raise ValueError(f"n must be a non-negative integer, got {n!r}")
    return int(n)


def geometric(a: float, r: float, n: int) -> NDArray:
    n = _check_n(n)
    # a * r**j for each j, not a running product: a running product rounds at
    # every step and its error grows with j.
    return float(a) * np.power(float(r), np.arange(n, dtype=np.float64))
```

`np.arange(n)` starts at 0, so term 0 is $a \cdot r^0 = a$ (pitfall 6), and it stops at $n - 1$, so there are exactly $n$ terms (pitfall 5).

The sum of ones carries all the floating-point reasoning:

```python
def _ones_sum(r: float, n: int) -> float:
    """1 + r + ... + r^(n-1), accurate for every r > 0, including r near 1."""
    if n == 0:
        return 0.0                      # the empty sum
    if r == 1.0:
        return float(n)                 # n ones; the closed form would be 0/0
    try:
        if r > 0.0:
            # (1 - r^n) / (1 - r) = expm1(n ln r) / (r - 1). With r near 1 the
            # textbook numerator 1 - r**n subtracts two nearly equal numbers;
            # log1p and expm1 never form that difference.
            d = r - 1.0                 # exact when r is near 1
            return math.expm1(n * math.log1p(d)) / d
        # r <= 0: 1 - r >= 1 and the closed form has no cancellation to fear,
        # except r^n near 1 (r near -1), which the contract does not promise.
        return (1.0 - r**n) / (1.0 - r)
    except OverflowError:
        # |r| > 1 and n large: the last term r^(n-1) dominates, and so does its sign.
        return math.inf if (r > 0.0 or (n - 1) % 2 == 0) else -math.inf


def geometric_sum(a: float, r: float, n: int) -> float:
    n = _check_n(n)
    return float(a) * _ones_sum(float(r), n)
```

Both branches compute the same quantity, $(1 - r^n)/(1 - r)$; the $r > 0$ branch only flips both signs, $(e^{n\ln r} - 1)/(r - 1)$, so that the numerator is an `expm1`. For the EMA of section 2.3, `geometric_sum(1 - beta, beta, t)` is then $(1-\beta)\cdot\frac{1-\beta^t}{1-\beta} = 1 - \beta^t$ with all its digits, even at $\beta = 0.999$.

RoPE's ladder is one line once the arguments are validated:

```python
def rope_inv_freq(d_rot: int, base: float) -> NDArray:
    # d_rot must split into whole pairs; a base <= 1 would give a flat or
    # climbing ladder, and an infinite base a ladder of zeros.
    if isinstance(d_rot, bool) or int(d_rot) != d_rot or d_rot < 2 or d_rot % 2:
        raise ValueError(f"d_rot must be a positive even integer, got {d_rot!r}")
    base = float(base)
    if not (base > 1.0 and math.isfinite(base)):
        raise ValueError(f"base must be finite and > 1, got {base}")
    d_rot = int(d_rot)
    # One frequency per pair i = 0 .. d_rot/2 - 1; the exponent steps by 2/d_rot.
    i = np.arange(d_rot // 2, dtype=np.float64)
    return base ** (-2.0 * i / d_rot)
```

`i` starts at 0, so the first rung is exactly $b^0 = 1$ (pitfall 3), and the factor 2 in the exponent puts the last rung at $b^{-(d-2)/d}$, just above $1/b$ (pitfall 2). The condition is written `not (base > 1.0 and ...)` rather than `base <= 1.0 or ...` so that a NaN base, for which every comparison is false, is rejected too.

ALiBi builds its two ladders from `geometric`:

```python
def alibi_slopes(n_heads: int) -> NDArray:
    if isinstance(n_heads, bool) or int(n_heads) != n_heads or n_heads < 1:
        raise ValueError(f"n_heads must be a positive integer, got {n_heads!r}")
    n = int(n_heads)
    p = 1 << (n.bit_length() - 1)  # the largest power of two <= n (exact, no log2)
    start = 2.0 ** (-8.0 / p)
    slopes = geometric(start, start, p)   # start^1, ..., start^p = 2^-8/p ... 2^-8
    if p == n:
        return slopes
    # The extra n - p heads take every other slope of the 2p-head ladder,
    # starting from the first, so they fall between the existing ones.
    start2 = 2.0 ** (-8.0 / (2 * p))
    extra = geometric(start2, start2, 2 * p)[0::2][: n - p]
    return np.concatenate([slopes, extra])
```

For $H = 6$: `bit_length()` of 6 (binary `110`) is 3, so $p = 4$; `slopes` is $[2^{-2}, 2^{-4}, 2^{-6}, 2^{-8}]$; the 8-head ladder is $2^{-1}, \ldots, 2^{-8}$, `[0::2]` keeps $2^{-1}, 2^{-3}, 2^{-5}, 2^{-7}$, and `[:2]` keeps $\tfrac12, \tfrac18$, matching section 3.

Full solution: [`series.py`](../../../../course/solutions/python/tinyllm/num/series.py).

### Why this solution

- **Cost.** Every function is $O(n)$ time and memory for the $n$ values it returns; `geometric_sum` is $O(1)$, a handful of library calls regardless of $n$. Summing the terms in a loop would be $O(n)$ and, for $n$ in the millions (an EMA after a million steps), would also accumulate $n$ roundings.
- **One formula for all $r > 0$.** An alternative is to switch to the textbook form when $|r - 1|$ is "large enough". That needs a threshold, and any threshold either leaves a band of cancellation or is unnecessary: the `expm1`/`log1p` form is accurate everywhere for $r > 0$, so there is nothing to switch.
- **Exact power-of-two detection.** `math.log2(n)` is exact for powers of two in CPython, but `math.log(n, 2)` and other routes divide two rounded logarithms and can land just below an integer (for example `math.log(1000, 10)` is `2.9999999999999996`), and `floor` then picks the wrong ladder. Bit length has no rounding at all.
- **float64 throughout.** Hugging Face computes RoPE frequencies in float32; computing in float64 and comparing at float32 precision means the solution is at least as accurate as what it must reproduce, and later code can cast down.
- **Tradeoff accepted.** For $r$ near $-1$ (for example $r = -1 + 10^{-12}$ with even $n$), $1 - r^n$ cancels and the sum loses digits. No caller uses negative ratios near $-1$, so the contract leaves it out rather than adding a branch nobody exercises.

### Explain it back

**1. Why is the denominator computed as `r - 1.0` with no special care, when the numerator needed `expm1` and `log1p`?**

<details><summary>Answer</summary>

Subtracting two floats that are within a factor of 2 of each other is exact (Sterbenz's lemma), so for $r$ near 1 the value $r - 1$ has no rounding error at all. The danger is only in subtracting a *rounded* quantity: $r^n$ is already rounded before $1 - r^n$ is formed, and the subtraction exposes that rounding. The denominator subtracts the exact input $r$.

</details>

**2. Why does the `expm1` branch apply only to $r > 0$?**

<details><summary>Answer</summary>

It rewrites $r^n$ as $e^{n \ln r}$, and $\ln r$ is undefined for $r \le 0$ (`math.log1p(r - 1)` with $r - 1 \le -1$ raises). It is also unnecessary there: $1 - r \ge 1$, so the denominator is never small and dividing does not amplify the numerator's rounding.

</details>

**3. What is the sum for $r = 10^{10}$, $n = 40$, and $a = 1$, and why does the solution not raise?**

<details><summary>Answer</summary>

The true sum is about $10^{390}$, beyond float64's maximum of about $1.8 \times 10^{308}$, so the right float answer is $+\infty$. `math.expm1(40 * math.log1p(1e10))` raises `OverflowError` rather than returning infinity; the helper catches it and returns the sign of the last term $r^{39}$, positive because $r > 0$. For $r = -10^{10}$ the last term $r^{39}$ is negative ($n - 1 = 39$ is odd), giving $-\infty$.

</details>

**4. Why does `alibi_slopes` call `geometric(start, start, p)` rather than `geometric(1.0, start, p)`?**

<details><summary>Answer</summary>

ALiBi's first slope is $2^{-8/p}$, not 1: the ladder is $\text{start}^{1}, \ldots, \text{start}^{p}$, ending at $2^{-8}$. `geometric(1.0, start, p)` would give $\text{start}^0, \ldots, \text{start}^{p-1}$, one rung too high everywhere, with a first head that penalises distance by a full unit per token.

</details>

**5. What would break if `rope_inv_freq` used the exponent $-i/d$?**

<details><summary>Answer</summary>

The ladder would end at about $b^{-1/2}$ instead of $1/b$: for $b = 10^5$ the slowest pair would repeat every $2\pi \cdot 316 \approx 2000$ positions instead of about $6 \times 10^5$, so distant tokens alias, and every pretrained checkpoint, whose attention weights were learned with $-2i/d$, would see rotated queries and keys it was never trained on.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. the textbook closed form near $r = 1$ | Adam's bias correction with $\beta_2 = 0.999$ correct to 7 digits instead of 16 | uses $\operatorname{expm1}(n\operatorname{log1p}(r-1))/(r-1)$ for every $r > 0$; step 4 |
| 2. the RoPE exponent $-i/d$ instead of $-2i/d$ | frequencies fall only to $b^{-1/2}$: long-range pairs turn far too fast and checkpoints from HF load with the wrong positions | `base ** (-2.0 * i / d_rot)`; step 7 |
| 3. starting the ladder at $i = 1$ | every frequency shifted one rung; the 1-radian pair is missing | `i = np.arange(d_rot // 2)` starts at 0 |
| 4. ignoring the non-power-of-two rule, or taking the odd entries | 12-head and 40-head models get slopes no published model uses | second ladder of $2p$ heads, `[0::2][: n - p]`; step 8 |
| 5. one term too many | $S_n$ includes $ar^n$; every bias correction is off | `np.arange(n)` stops at $n - 1$; the closed form uses $r^n$ in the numerator for exactly $n$ terms |
| 6. starting the sequence at $ar$ | `geometric` returns $ar, \ldots, ar^n$; ALiBi slopes shift by one rung | exponents start at $j = 0$; ALiBi passes `start` as the first term explicitly |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M00.1` | powers, $e$, and `log1p`/`expm1` (reading) |
| Back | `lang.01` | numpy `arange` and `power` (reading) |
| Forward | `L5.4` | a call site of this module |
| Forward | `L7.3` | `rope_inv_freq(d_rot, rope_theta)` builds RoPE's angles, `positions * inv_freq` |
| Forward | `L7.4` | ALiBi biases from `alibi_slopes`; YaRN and NTK scaling bend the same ladder per frequency |
| Forward | `M02.2` | the EMA's weights, `geometric(1 - beta, beta, t)`, and its bias correction, `geometric_sum(1 - beta, beta, t)` |
| Forward | `M10.4` | step-decay schedules are geometric sequences of learning rates (reading) |
| Forward | `M00.2` | the ladder feeds the angles that `rotate_pairs` applies (reading) |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `rope_inv_freq` | Hugging Face `ROPE_INIT_FUNCTIONS` and `compute_default_rope_parameters` | one function per scaling kind (`linear`, `dynamic`, `yarn`, `llama3`, `longrope`) that bends this ladder for longer contexts | `transformers/modeling_rope_utils.py` |
| `alibi_slopes` | BLOOM and MPT attention biases | slopes built once per model, biases fused into the attention kernel | `transformers/models/bloom/modeling_bloom.py` (`build_alibi_tensor`) |
| `geometric_sum` near $r = 1$ | PyTorch `torch.optim.Adam` | the bias corrections `1 - beta1 ** step` and `1 - beta2 ** step` (in float32, where the cancellation is worse) | `torch/optim/adam.py` |
| `log1p` and `expm1` | C99 `log1p`, `expm1` | the accurate small-argument functions every math library ships | `man 3 expm1` |
