# Sequences, geometric series, and frequency ladders

The [previous section](02-trig-rotations-and-eulers-formula.md) gave you rotations, and RoPE rotates each pair of a query by an angle proportional to the token's position. That leaves an open question: how fast should each pair turn? If every pair turned by 1 radian per token, positions $2\pi \approx 6.3$ tokens apart would look identical; if every pair turned slowly, neighbours would be indistinguishable. RoPE's answer is a ladder of speeds, a **geometric sequence**, and so is the answer of ALiBi, the other position scheme in [Context extension](../../build-an-llm/08-modern-block/04-context-extension.md) in *Build Your Own LLM System*.

The same mathematics turns up in training. The moving averages of Adam weight past gradients by a geometric sequence and correct for its partial sum, which is the subject of [Series and the EMA](../03-calculus-2/02-series-and-the-ema.md) and [Adam and AdamW](../11-optimization/03-adam-and-adamw.md). And when the [Llama-family loader](../../build-an-llm/08-modern-block/09-llama-family-model.md) reads `rope_theta = 100000` from SmolLM2's `config.json`, it must turn it into exactly the frequencies Hugging Face computes. This section builds geometric sequences and their sums, accurately, including the hard case near a ratio of 1 where the textbook formula loses half its digits, and then both position ladders.

You need powers, $e$, and the functions `log1p` and `expm1` from [Exponents, logarithms, and units of information](01-exponents-logs-and-units-of-information.md), and numpy's `arange` and `power` from [Python and numpy](../../programming-languages/02-language-and-tool-primers/01-python-and-numpy.md) in *Programming Languages*. The finished code is [`course/solutions/python/tinyllm/num/series.py`](../../../../course/solutions/python/tinyllm/num/series.py), which provides `geometric`, `geometric_sum` (with the helper `_ones_sum`), `rope_inv_freq`, and `alibi_slopes`; its interface is fixed by the contract [`course/contracts/py/tinyllm/num/series.pyi`](../../../../course/contracts/py/tinyllm/num/series.pyi).

> **Learning objectives**
> - Derive the partial sum $a(1 - r^n)/(1 - r)$ of a geometric sequence and its infinite limit for $|r| < 1$.
> - Show that an exponential moving average weights its inputs by a geometric sequence whose weights sum to $1 - \beta^t$.
> - Explain why the closed-form sum cancels near $r = 1$ and rewrite it with $\operatorname{expm1}$ and $\operatorname{log1p}$ so it keeps its digits.
> - Build RoPE's inverse-frequency ladder $b^{-2i/d}$ and ALiBi's head slopes, including the rule for head counts that are not powers of two.

## Sequences

The symbols used throughout are collected here first:

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

A **sequence** is a list of numbers indexed by position: $a_0, a_1, a_2, \ldots$. Two families matter here. An **arithmetic** sequence adds the same step, $a_j = a + jd$; the positions $0, 1, 2, \ldots$ are one. A **geometric** sequence multiplies by the same ratio, $a_j = a r^j$: $1, \tfrac12, \tfrac14, \ldots$ has $a = 1$ and $r = \tfrac12$. With $0 < r < 1$ the terms shrink toward 0; with $r > 1$ they grow without bound; with $r < 0$ they alternate in sign.

Compute term $j$ as `a * r**j`, not by multiplying the previous term by $r$. A running product rounds once per step, and its error grows with $j$.

## The geometric sum

Let $S_n = a + ar + \cdots + ar^{n-1}$. Multiply by $r$ to get $rS_n = ar + \cdots + ar^{n-1} + ar^n$. Subtracting, every term but two cancels:

$$S_n - rS_n = a - ar^n \quad\Longrightarrow\quad S_n = a\,\frac{1 - r^n}{1 - r}, \qquad r \ne 1,$$

and $S_n = na$ when $r = 1$, since every term is $a$. When $|r| < 1$, $r^n$ approaches 0 as $n$ grows, so the infinite sum converges: $a + ar + ar^2 + \cdots = a/(1 - r)$. That is why $1 + \tfrac12 + \tfrac14 + \cdots = 2$, and why a repeating decimal is a fraction: $0.2727\ldots = 27/100 + 27/100^2 + \cdots = (27/100)/(1 - 1/100) = 3/11$.

## The exponential moving average is a geometric series

The first place this sum does real work is the exponential moving average (EMA). With decay $\beta$, it updates $m_t = \beta m_{t-1} + (1 - \beta) g_t$, starting from $m_0 = 0$. Unrolling the recursion,

$$m_t = (1 - \beta)\left(g_t + \beta g_{t-1} + \beta^2 g_{t-2} + \cdots + \beta^{t-1} g_1\right),$$

so the weights are a geometric sequence $(1 - \beta)\beta^j$, newest first, and they add up to $(1 - \beta)\frac{1 - \beta^t}{1 - \beta} = 1 - \beta^t$, not 1. Early on, the average is biased toward its starting value 0: after one step with $\beta = 0.999$, $m_1 = 0.001\,g_1$. Adam divides by $1 - \beta^t$ to undo exactly this. With $\beta = 0.999$ the ratio is $1 - 10^{-3}$, close to 1, and that is precisely the case where the closed form is numerically dangerous, as we will see shortly.

## Frequency ladders: RoPE

RoPE rotates pair $i$ of a query or key at position $p$ by the angle $p\,\omega_i$, with

$$\omega_i = b^{-2i/d}, \qquad i = 0, 1, \ldots, \tfrac d2 - 1 .$$

This is a geometric sequence with first term $b^0 = 1$ and ratio $b^{-2/d}$. Pair 0 turns one radian per token and repeats every $\lambda_0 = 2\pi \approx 6.3$ positions; the last pair turns $b^{-(d-2)/d}$ radians per token, slightly faster than $1/b$, and repeats only after about $2\pi b$ positions. Like the hands of a clock (seconds, minutes, hours), fast pairs resolve neighbouring tokens and slow pairs tell far-apart ones apart, and the geometric spacing gives every scale the same number of pairs. The exponent steps by $2/d$ because there are $d/2$ pairs spread over the exponents from 0 to almost 1.

Models with **partial rotary** embeddings rotate only the first $d$ coordinates of each head; the formula is the same with that $d$. The SmolLM2-135M `config.json` says `hidden_size = 576` over `num_attention_heads = 9`, a head size of 64, and `rope_theta = 100000`. `rope_inv_freq(64, 1e5)` must match what Hugging Face computes from those (in float32, to float32 precision), or the loaded checkpoint sees the wrong positions.

## Frequency ladders: ALiBi

ALiBi (Press et al.) adds no rotation at all. Head $h$ subtracts $s_h$ times the distance between query and key from each attention score, so far tokens are penalized linearly, and each head has its own slope. For $H$ a power of two, the slopes are the geometric sequence with first term and ratio $2^{-8/H}$:

$$s_h = 2^{-8h/H}, \qquad h = 1, \ldots, H,$$

running from $2^{-8/H}$ (steep: the head looks at the last few tokens) down to $2^{-8} = 1/256$ (flat: it looks hundreds of tokens back).

For other $H$, let $p$ be the largest power of two below $H$. The paper's code takes the $p$ slopes for $p$ heads, then fills the remaining $H - p$ heads with every other slope of the $2p$-head ladder, $2^{-8 \cdot 1/(2p)}, 2^{-8 \cdot 3/(2p)}, \ldots$, which fall between the existing ones. Find $p$ with integer arithmetic (`1 << (H.bit_length() - 1)`), never with a floating-point $\log_2$.

## Floating point near a ratio of 1

Back to the EMA's hard case. For $r$ close to 1, the closed form divides one small difference by another. $1 - r$ is computed exactly, because subtracting nearby floats is exact. But $r^n$ is first rounded to about $\varepsilon$ relative error, and $1 - r^n$ then subtracts two numbers that agree in most of their digits, keeping the rounding error and losing the rest. With $r = 1 - 2^{-40}$ and $n = 10$, $1 - r^n \approx 9 \times 10^{-12}$ carries an absolute error near $10^{-16}$: about 5 correct digits out of 16.

The cure is to never form $r^n$ near 1. Since $r^n = e^{n \ln r}$,

$$\frac{1 - r^n}{1 - r} = \frac{\operatorname{expm1}(n \cdot \operatorname{log1p}(r - 1))}{r - 1},$$

where $\operatorname{log1p}(u) = \ln(1 + u)$ and $\operatorname{expm1}(v) = e^v - 1$ are computed accurately for small arguments by `math.log1p` and `math.expm1`. Every quantity in this form is small and accurate, and the result keeps nearly all 16 digits. For $r > 0$ this form is used everywhere; for $r \le 0$ the closed form has no cancellation, because $1 - r \ge 1$. When the sum overflows (huge $r$ and $n$), the answer is $\pm\infty$, with the sign of the last term.

## Working it by hand

Start with a geometric sum: $1 + \tfrac12 + \tfrac14 + \tfrac18$, so $a = 1$, $r = \tfrac12$, $n = 4$.

$$S_4 = \frac{1 - (1/2)^4}{1 - 1/2} = \frac{15/16}{1/2} = \frac{15}{8} = 1.875 ,$$

which agrees with adding $1 + 0.5 + 0.25 + 0.125$. With $a = 3$ every term triples, so $S_4 = 45/8$.

Next, the RoPE ladder for $d = 8$ and $b = 10000$. There are four pairs, with exponents $-2i/8$:

| $i$ | exponent | $\omega_i = 10000^{\text{exponent}}$ | wavelength $2\pi/\omega_i$ |
|---|---|---|---|
| 0 | 0 | 1 | 6.28 positions |
| 1 | $-1/4$ | $0.1$ | 62.8 |
| 2 | $-1/2$ | $0.01$ | 628 |
| 3 | $-3/4$ | $0.001$ | 6283 |

($10000^{1/4} = 10$ because $10^4 = 10000$.) The ratio is $0.1 = 10000^{-2/8}$.

Finally, ALiBi with 6 heads. Since 6 is not a power of two, $p = 4$. The 4-head ladder has first term and ratio $2^{-8/4} = 2^{-2}$: $2^{-2}, 2^{-4}, 2^{-6}, 2^{-8} = \tfrac14, \tfrac1{16}, \tfrac1{64}, \tfrac1{256}$. The 8-head ladder is $2^{-1}, 2^{-2}, \ldots, 2^{-8}$; every other entry from the first is $2^{-1}, 2^{-3}, 2^{-5}, \ldots$, and the two extra heads take the first two:

$$s = \left[\tfrac14, \tfrac1{16}, \tfrac1{64}, \tfrac1{256}, \tfrac12, \tfrac18\right].$$

The extra slopes $\tfrac12$ and $\tfrac18$ sit between the existing ones, so the six heads still cover the range evenly.

## Designing the interface

Four functions cover the sequences and both ladders:

```python
# python/tinyllm/num/series.py
def geometric(a: float, r: float, n: int) -> NDArray: ...        # [a, a r, ..., a r^(n-1)]
def geometric_sum(a: float, r: float, n: int) -> float: ...      # accurate near r = 1
def rope_inv_freq(d_rot: int, base: float) -> NDArray: ...       # base ** (-2 i / d_rot), [d_rot/2]
def alibi_slopes(n_heads: int) -> NDArray: ...                   # Press et al., any head count
```

Everything is float64. Bad counts (negative, fractional, or `True`), an odd or non-positive `d_rot`, a base that is not finite and above 1, and a head count below 1 raise `ValueError`.

## Building it step by step

Eight observations fix the implementation.

1. **Terms are independent powers.** Term $j$ is $ar^j$ by definition, so each term can be computed directly from $j$. Computing it from the previous term would add one rounding per step; computing $r^j$ directly rounds once. So `geometric` evaluates `a * r**j` over the vector $j = 0, \ldots, n-1$, and $n = 0$ is the empty vector.
2. **The sum factors as $a$ times the sum of ones.** $S_n = a(1 + r + \cdots + r^{n-1})$, so all the numerical care goes into one helper for $1 + r + \cdots + r^{n-1}$, and `geometric_sum` multiplies by $a$ once at the end.
3. **Two exact cases come first.** $n = 0$ sums nothing ($0$); $r = 1$ makes every term 1 ($n$). Both avoid dividing $0/0$.
4. **For $r > 0$, never form $r^n$.** $1 - r^n$ cancels near $r = 1$. Writing $r^n = e^{n \ln r}$ and $\ln r = \operatorname{log1p}(r - 1)$ turns the numerator into $\operatorname{expm1}(n \operatorname{log1p}(r-1))$. The denominator $r - 1$ is exact for $r$ near 1 (two floats within a factor of 2 of each other subtract exactly), so every piece carries nearly full precision. Far from 1 the form is just as accurate, so there is no threshold to tune: one formula for all $r > 0$.
5. **For $r \le 0$, the closed form is safe.** $\operatorname{log1p}(r - 1)$ is undefined for $r \le 0$, but there $1 - r \ge 1$, so the denominator is never small and the textbook formula does no harmful cancellation (except for $r$ near $-1$, which the contract does not promise).
6. **Overflow has a known answer.** If $|r| > 1$ and $n$ is large, the sum is dominated by its last term $r^{n-1}$, so the result is $\pm\infty$ with that term's sign: $+$ for $r > 0$, and for $r < 0$, $+$ when $n - 1$ is even. Python's `math.expm1` and `float ** int` raise `OverflowError` instead of returning infinity, so the helper catches it and returns the infinity.
7. **RoPE is a geometric ladder on a fixed exponent grid.** With $d/2$ pairs and $\omega_i = b^{-2i/d}$, the solution builds $i = 0, \ldots, d/2 - 1$ as floats and raises $b$ to $-2i/d$ in one vectorised step. The arguments must make a decreasing ladder: $d$ even and at least 2 (there must be whole pairs), and $b > 1$ and finite.
8. **ALiBi is two ladders.** Find $p$, the largest power of two $\le H$, exactly with bit arithmetic. The first $p$ slopes are `geometric(start, start, p)` with $\text{start} = 2^{-8/p}$, which gives $\text{start}^{1}, \ldots, \text{start}^{p}$. If $H > p$, build the $2p$-head ladder the same way, take every other entry from the first (`[0::2]`), and keep the first $H - p$ of them, exactly the six-head computation above.

### Terms and argument checking

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

`np.arange(n)` starts at 0, so term 0 is $a \cdot r^0 = a$, and it stops at $n - 1$, so there are exactly $n$ terms.

### The accurate sum

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

Both branches compute the same quantity, $(1 - r^n)/(1 - r)$; the $r > 0$ branch only flips both signs, $(e^{n\ln r} - 1)/(r - 1)$, so that the numerator is an `expm1`. For the EMA, `geometric_sum(1 - beta, beta, t)` is then $(1-\beta)\cdot\frac{1-\beta^t}{1-\beta} = 1 - \beta^t$ with all its digits, even at $\beta = 0.999$.

### The RoPE ladder

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

`i` starts at 0, so the first rung is exactly $b^0 = 1$, and the factor 2 in the exponent puts the last rung at $b^{-(d-2)/d}$, just above $1/b$. The condition is written `not (base > 1.0 and ...)` rather than `base <= 1.0 or ...` so that a NaN base, for which every comparison is false, is rejected too.

### The ALiBi slopes

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

For $H = 6$, the `bit_length()` of 6 (binary `110`) is 3, so $p = 4$. `slopes` is $[2^{-2}, 2^{-4}, 2^{-6}, 2^{-8}]$; the 8-head ladder is $2^{-1}, \ldots, 2^{-8}$, `[0::2]` keeps $2^{-1}, 2^{-3}, 2^{-5}, 2^{-7}$, and `[:2]` keeps $\tfrac12, \tfrac18$, matching the hand computation.

The full solution is [`series.py`](../../../../course/solutions/python/tinyllm/num/series.py).

### Why this design

Every function is $O(n)$ in time and memory for the $n$ values it returns, and `geometric_sum` is $O(1)$, a handful of library calls regardless of $n$. Summing the terms in a loop would be $O(n)$ and, for $n$ in the millions (an EMA after a million steps), would also accumulate $n$ roundings.

The sum uses one formula for all $r > 0$. The alternative is to switch to the textbook form when $|r - 1|$ is "large enough", but that needs a threshold, and any threshold either leaves a band of cancellation or is unnecessary: the `expm1`/`log1p` form is accurate everywhere for $r > 0$, so there is nothing to switch.

Power-of-two detection is exact. `math.log2(n)` is exact for powers of two in CPython, but `math.log(n, 2)` and similar routes divide two rounded logarithms and can land just below an integer (`math.log(1000, 10)` is `2.9999999999999996`), and `floor` then picks the wrong ladder. Bit length has no rounding at all.

Everything is float64. Hugging Face computes RoPE frequencies in float32; computing in float64 and comparing at float32 precision means the solution is at least as accurate as what it must reproduce, and later code can cast down.

One tradeoff is accepted deliberately. For $r$ near $-1$ (for example $r = -1 + 10^{-12}$ with even $n$), $1 - r^n$ cancels and the sum loses digits. No caller uses negative ratios near $-1$, so the contract leaves the case out rather than adding a branch nobody exercises.

## What can go wrong

A correct implementation must pass the following checks, each tied to something that depends on it:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked geometric sum | unit | $15/8$, $45/8$, and the four terms | everyone agrees on what $n$ counts |
| the hand-worked RoPE ladder | unit | $d = 8$, $b = 10000$ gives $[1, 0.1, 0.01, 0.001]$ | the exponent is $-2i/d$ |
| the hand-worked six-head ALiBi slopes | unit | the 6-head slopes computed above | the non-power-of-two rule |
| edge-case terms | unit | negative and unit ratios, and $n = 0$ (empty) | term 0 is $a$ |
| sums against exact rationals | differential | 66 sums against exact `Fraction` arithmetic on the same inputs | any correct formula passes, any wrong count fails |
| sums near $r = 1$ keep their digits | boundary | $r = 1 \pm 2^{-40}$ and $1 - 10^{-9}$: relative error below $10^{-12}$ | the cancellation near $r = 1$ |
| the EMA bias-correction identity | property | $\sum_{j<t} (1 - \beta)\beta^j = 1 - \beta^t$ for $\beta$ up to $0.999$ | Adam's bias correction |
| sums that overflow | boundary | $\pm\infty$ for sums beyond float64, by the sign of the last term | no `OverflowError` escapes |
| bad term counts | boundary | $n = -1$, $2.5$, `True` raise `ValueError` | counts are whole numbers |
| Hugging Face RoPE frequencies | golden | Hugging Face's default RoPE for SmolLM2-135M, Llama 2, Llama 3, and a partial rotary size, to float32 precision | exactly what the Llama-family loader loads |
| the RoPE ladder's shape | property | first rung 1, constant ratio $b^{-2/d}$, last rung above $1/b$ | the ladder's shape for any $d$ and $b$ |
| bad RoPE arguments | boundary | odd or zero $d$, bases $\le 1$ or infinite raise `ValueError` | a flat or climbing ladder is a bug |
| ALiBi for powers of two | unit | 8 heads: $1/2, \ldots, 1/256$; 16 heads: $2^{-j/2}$; 1 head: $1/256$ | the power-of-two rule |
| BLOOM's ALiBi slopes | golden | BLOOM's `build_alibi_tensor` for 20 head counts from 1 to 128 | twelve of them are not powers of two |
| bad head counts | boundary | 0, $-4$, $2.5$ heads raise `ValueError` | a model has whole heads |

The mistake with the most consequence is the textbook formula itself.

> **Warning:** Using the closed form $(1 - r^n)/(1 - r)$ directly near $r = 1$ makes Adam's bias correction with $\beta_2 = 0.999$ correct to 7 digits instead of 16. The solution uses $\operatorname{expm1}(n\operatorname{log1p}(r-1))/(r-1)$ for every $r > 0$.

The RoPE ladder has two classic errors. Using the exponent $-i/d$ instead of $-2i/d$ makes the frequencies fall only to $b^{-1/2}$, so long-range pairs turn far too fast and Hugging Face checkpoints load with the wrong positions; the solution writes `base ** (-2.0 * i / d_rot)`. Starting the ladder at $i = 1$ shifts every frequency one rung and loses the 1-radian pair; `np.arange(d_rot // 2)` starts at 0. For ALiBi, ignoring the non-power-of-two rule, or taking the odd entries of the second ladder instead of the even ones, gives 12-head and 40-head models slopes no published model uses; the solution builds the $2p$-head ladder and takes `[0::2][: n - p]`.

The last two are off-by-one errors in the sequence itself. One term too many makes $S_n$ include $ar^n$ and throws off every bias correction; `np.arange(n)` stops at $n - 1$, and the closed form uses $r^n$ in the numerator for exactly $n$ terms. Starting the sequence at $ar$ makes `geometric` return $ar, \ldots, ar^n$ and shifts the ALiBi slopes by one rung; the exponents start at $j = 0$, and ALiBi passes `start` as the first term explicitly.

## Where this leads

In this book, [Series and the EMA](../03-calculus-2/02-series-and-the-ema.md) builds the EMA's weights as `geometric(1 - beta, beta, t)` and its bias correction as `geometric_sum(1 - beta, beta, t)`, and [learning-rate schedules](../11-optimization/04-schedules-and-gradient-clipping.md) include step-decay schedules that are geometric sequences of learning rates. In *Build Your Own LLM System*, [RoPE](../../build-an-llm/08-modern-block/03-rope.md) calls `rope_inv_freq(d_rot, rope_theta)` to build its angles as `positions * inv_freq`, which feed the `rotate_pairs` of the [previous section](02-trig-rotations-and-eulers-formula.md); [Context extension](../../build-an-llm/08-modern-block/04-context-extension.md) takes ALiBi biases from `alibi_slopes` and bends the same ladder per frequency for YaRN and NTK scaling; and the [positional encodings](../../build-an-llm/06-transformer-2017/04-positional-encodings.md) of the 2017 Transformer use a ladder of the same shape.

### Further reading

- OpenStax, *Precalculus 2e* (free), chapter 11 on sequences and series.
- Press, Smith, and Lewis, "Train Short, Test Long: Attention with Linear Biases" (ALiBi, 2022), section 3.
- Su et al., "RoFormer" (2021), section 3.3.
- Hugging Face `ROPE_INIT_FUNCTIONS` and `compute_default_rope_parameters` in `transformers/modeling_rope_utils.py`: one function per scaling kind (`linear`, `dynamic`, `yarn`, `llama3`, `longrope`) that bends this ladder for longer contexts.
- BLOOM's `build_alibi_tensor` in `transformers/models/bloom/modeling_bloom.py`: the slopes built once per model, with the biases fused into the attention kernel in BLOOM and MPT.
- PyTorch `torch.optim.Adam` in `torch/optim/adam.py`: the bias corrections `1 - beta1 ** step` and `1 - beta2 ** step`, computed in float32, where the cancellation is worse.
- C99 `log1p` and `expm1` (`man 3 expm1`): the accurate small-argument functions every math library ships.

## Key terms

- **Geometric sequence**: a sequence that multiplies by the same ratio $r$ at every step, $a_j = ar^j$.
- **Geometric sum**: the sum of the first $n$ terms, $a(1 - r^n)/(1 - r)$ for $r \ne 1$ and $na$ for $r = 1$.
- **Exponential moving average**: $m_t = \beta m_{t-1} + (1 - \beta) g_t$, whose weights are a geometric sequence summing to $1 - \beta^t$.
- **Cancellation**: the loss of significant digits when subtracting two nearly equal rounded numbers.
- **`log1p` / `expm1`**: $\ln(1 + u)$ and $e^v - 1$, computed accurately for small arguments.
- **Inverse frequency**: RoPE's rotation speed $\omega_i = b^{-2i/d}$ for pair $i$, in radians per position.
- **ALiBi slope**: the per-head factor $s_h$ by which ALiBi penalizes query-key distance.

## Summary

- A geometric sequence multiplies by the same ratio $r$ at every step: $a, ar, ar^2, \ldots$.
- Its first $n$ terms add up to $a(1 - r^n)/(1 - r)$, found by subtracting $r$ times the sum from the sum.
- Near $r = 1$ that formula cancels and loses half its digits; rewriting it with $\operatorname{expm1}$ and $\operatorname{log1p}$ keeps them all, and Adam's bias correction lives exactly there.
- RoPE's inverse frequencies $b^{-2i/d}$ form a geometric ladder from one radian per token down to about $1/b$, so pairs of dimensions see position at every scale from a few tokens to tens of thousands.
- ALiBi slopes are a geometric ladder of $2^{-8/n}$; head counts that are not powers of two interleave a second ladder.

## Review questions

1. Why is the denominator computed as `r - 1.0` with no special care, when the numerator needed `expm1` and `log1p`?
   <details><summary>Answer</summary>Subtracting two floats that are within a factor of 2 of each other is exact (Sterbenz's lemma), so for $r$ near 1 the value $r - 1$ has no rounding error at all. The danger is only in subtracting a <em>rounded</em> quantity: $r^n$ is already rounded before $1 - r^n$ is formed, and the subtraction exposes that rounding. The denominator subtracts the exact input $r$.</details>
2. Why does the `expm1` branch apply only to $r > 0$?
   <details><summary>Answer</summary>It rewrites $r^n$ as $e^{n \ln r}$, and $\ln r$ is undefined for $r \le 0$ (<code>math.log1p(r - 1)</code> with $r - 1 \le -1$ raises). It is also unnecessary there: $1 - r \ge 1$, so the denominator is never small and dividing does not amplify the numerator's rounding.</details>
3. What is the sum for $r = 10^{10}$, $n = 40$, and $a = 1$, and why does the solution not raise?
   <details><summary>Answer</summary>The true sum is about $10^{390}$, beyond float64's maximum of about $1.8 \times 10^{308}$, so the right float answer is $+\infty$. <code>math.expm1(40 * math.log1p(1e10))</code> raises <code>OverflowError</code> rather than returning infinity; the helper catches it and returns the sign of the last term $r^{39}$, positive because $r > 0$. For $r = -10^{10}$ the last term $r^{39}$ is negative ($n - 1 = 39$ is odd), giving $-\infty$.</details>
4. Why does `alibi_slopes` call `geometric(start, start, p)` rather than `geometric(1.0, start, p)`?
   <details><summary>Answer</summary>ALiBi's first slope is $2^{-8/p}$, not 1: the ladder is $\text{start}^{1}, \ldots, \text{start}^{p}$, ending at $2^{-8}$. <code>geometric(1.0, start, p)</code> would give $\text{start}^0, \ldots, \text{start}^{p-1}$, one rung too high everywhere, with a first head that penalises distance by a full unit per token.</details>
5. What would break if `rope_inv_freq` used the exponent $-i/d$?
   <details><summary>Answer</summary>The ladder would end at about $b^{-1/2}$ instead of $1/b$: for $b = 10^5$ the slowest pair would repeat every $2\pi \cdot 316 \approx 2000$ positions instead of about $6 \times 10^5$, so distant tokens alias, and every pretrained checkpoint, whose attention weights were learned with $-2i/d$, would see rotated queries and keys it was never trained on.</details>
6. Write $0.2727\ldots$ as a fraction using the infinite geometric sum.
   <details><summary>Answer</summary>It is $27/100 + 27/100^2 + \cdots$, a geometric series with $a = 27/100$ and $r = 1/100$, so its sum is $(27/100)/(1 - 1/100) = 27/99 = 3/11$.</details>
