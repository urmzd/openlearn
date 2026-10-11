<!-- ol:module M00.3 -->
# Sequences, geometric series, frequency ladders

## Overview

| | |
|---|---|
| **Module** | `M00.3` · build · Python · Pass 2 · 2 to 3 h |
| **You build** | `python/tinyllm/num/series.py`: `geometric`, `geometric_sum`, `rope_inv_freq`, `alibi_slopes` |
| **Contract** | [`course/contracts/py/tinyllm/num/series.pyi`](../../course/contracts/py/tinyllm/num/series.pyi) |
| **Tests** | `course/tests/M00.3/test_series.py` (what they check: section 4) |
| **Needs** | nothing to call. Reading: `M00.1` (powers, `log1p` and `expm1`), `lang.01` |
| **Used by** | `M02.2` the EMA's weights and bias correction call `geometric` and `geometric_sum` · later `L7.3` RoPE `inv_freq` and `L7.4` ALiBi slopes and YaRN's per-frequency ramps · `M10.4` learning-rate schedules (reading) · later: `L5.4` |
| **Milestone** | `MS-P2` (the Pass 2 gate) |
| **Optional depth** | OpenStax, *Precalculus 2e* (free), ch. 11 (sequences and series); Press, Smith, and Lewis, "Train Short, Test Long: Attention with Linear Biases" (ALiBi, 2022), section 3; Su et al., "RoFormer" (2021), section 3.3 |

## Key Takeaways

- A **geometric sequence** multiplies by the same ratio $r$ at every step: $a, ar, ar^2, \ldots$ (`test_geometric_terms`).
- Its first $n$ terms add up to $a(1 - r^n)/(1 - r)$, by subtracting $r$ times the sum from the sum (`test_hand_example_geometric_sum`, `test_sum_matches_exact_rationals`).
- Near $r = 1$ that formula **cancels**: it subtracts two nearly equal numbers and loses half its digits. Rewriting it with $\operatorname{expm1}$ and $\operatorname{log1p}$ keeps them all; Adam's bias correction lives exactly there (`test_sum_near_one_keeps_its_digits`, `test_ema_bias_correction_identity`).
- RoPE's **inverse frequencies** $b^{-2i/d}$ form a geometric ladder from one radian per token down to about $1/b$, so pairs of dimensions see position at every scale from a few tokens to tens of thousands (`test_rope_inv_freq_golden_hf`).
- **ALiBi slopes** are a geometric ladder of $2^{-8/n}$; head counts that are not powers of two interleave a second ladder (`test_hand_example_alibi_six_heads`, `test_alibi_golden_hf`).

## How to work this chapter

```bash
ol start M00.3              # stubs python/tinyllm/num/series.py into your repo
ol tests M00.3              # read the test catalog first: rung R0, you write no tests here
ol check M00.3              # exit code is the verdict
ol diff  M00.3              # after passing: your code against the reference
```

---

## 1. Why now

`M00.2` gave you rotations, and RoPE rotates each pair of a query by an angle proportional to the token's position. The open question is how fast each pair should turn. If every pair turned by 1 radian per token, positions $2\pi \approx 6.3$ tokens apart would look identical; if every pair turned slowly, neighbours would be indistinguishable. RoPE answers with a ladder of speeds, a **geometric sequence**, and so does ALiBi, the other position scheme in `L7.4`. The same mathematics appears in training: the moving averages of Adam (`M02.2`, `M10.3`) weight past gradients by a geometric sequence, and correct for its partial sum. Your SmolLM2 loader (`L7.9`) will read `rope_theta = 100000` from a `config.json` and must turn it into exactly the frequencies Hugging Face computes. This module builds the sequences, their sums (accurately, including the hard case), and both ladders.

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

This is a geometric sequence with first term $b^0 = 1$ and ratio $b^{-2/d}$. Pair 0 turns one radian per token and repeats every $\lambda_0 = 2\pi \approx 6.3$ positions; the last pair turns $b^{-(d-2)/d}$ radians per token, slightly faster than $1/b$, and repeats only after about $2\pi b$ positions. Like the hands of a clock (seconds, minutes, hours), fast pairs resolve neighbouring tokens and slow pairs tell far-apart ones apart, and the geometric spacing gives every scale the same number of pairs. The exponent steps by $2/d$ because there are $d/2$ pairs spread over the exponents from 0 to almost 1. Models with **partial rotary** embeddings rotate only the first $d$ coordinates of each head; the formula is the same with that $d$. The SmolLM2-135M `config.json` says `hidden_size = 576` over `num_attention_heads = 9` (a head size of 64) and `rope_theta = 100000`; your `rope_inv_freq(64, 1e5)` must match what Hugging Face computes from those, which is what the golden test checks.

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

which agrees with adding $1 + 0.5 + 0.25 + 0.125$. With $a = 3$ every term triples, so $S_4 = 45/8$. These are `test_hand_example_geometric_sum`.

**The RoPE ladder for $d = 8$, $b = 10000$.** Four pairs, exponents $-2i/8$:

| $i$ | exponent | $\omega_i = 10000^{\text{exponent}}$ | wavelength $2\pi/\omega_i$ |
|---|---|---|---|
| 0 | 0 | 1 | 6.28 positions |
| 1 | $-1/4$ | $0.1$ | 62.8 |
| 2 | $-1/2$ | $0.01$ | 628 |
| 3 | $-3/4$ | $0.001$ | 6283 |

($10000^{1/4} = 10$ because $10^4 = 10000$.) The ratio is $0.1 = 10000^{-2/8}$: `test_hand_example_rope_ladder`.

**ALiBi with 6 heads.** $6$ is not a power of two, so $p = 4$. The 4-head ladder has first term and ratio $2^{-8/4} = 2^{-2}$: $2^{-2}, 2^{-4}, 2^{-6}, 2^{-8} = \tfrac14, \tfrac1{16}, \tfrac1{64}, \tfrac1{256}$. The 8-head ladder is $2^{-1}, 2^{-2}, \ldots, 2^{-8}$; every other entry from the first is $2^{-1}, 2^{-3}, 2^{-5}, \ldots$, and the two extra heads take the first two:

$$s = \left[\tfrac14, \tfrac1{16}, \tfrac1{64}, \tfrac1{256}, \tfrac12, \tfrac18\right],$$

`test_hand_example_alibi_six_heads`. The extra slopes $\tfrac12$ and $\tfrac18$ sit between the existing ones, so the six heads still cover the range evenly.

## 4. The interface

```python
# python/tinyllm/num/series.py
def geometric(a: float, r: float, n: int) -> NDArray: ...        # [a, a r, ..., a r^(n-1)]
def geometric_sum(a: float, r: float, n: int) -> float: ...      # accurate near r = 1
def rope_inv_freq(d_rot: int, base: float) -> NDArray: ...       # base ** (-2 i / d_rot), [d_rot/2]
def alibi_slopes(n_heads: int) -> NDArray: ...                   # Press et al., any head count
```

Everything is float64. Bad counts (negative, fractional, or `True`), an odd or non-positive `d_rot`, a base that is not finite and above 1, and a head count below 1 raise `ValueError`.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_geometric_sum` | unit, smoke | section 3: $15/8$, $45/8$, and the four terms | you and the tests agree on what $n$ counts |
| `test_hand_example_rope_ladder` | unit, smoke | $d = 8$, $b = 10000$ gives $[1, 0.1, 0.01, 0.001]$ | the exponent is $-2i/d$ |
| `test_hand_example_alibi_six_heads` | unit, smoke | the 6-head slopes of section 3 | the non-power-of-two rule |
| `test_geometric_terms` | unit | negative and unit ratios, and $n = 0$ (empty) | term 0 is $a$ |
| `test_sum_matches_exact_rationals` | differential | 66 sums against exact `Fraction` arithmetic on the same inputs | any correct formula passes, any wrong count fails |
| `test_sum_near_one_keeps_its_digits` | boundary | $r = 1 \pm 2^{-40}$ and $1 - 10^{-9}$: relative error below $10^{-12}$ | pitfall 1 |
| `test_ema_bias_correction_identity` | property | $\sum_{j<t} (1 - \beta)\beta^j = 1 - \beta^t$ for $\beta$ up to $0.999$ | Adam's bias correction (`M02.2`) |
| `test_sum_overflows_to_infinity` | boundary | $\pm\infty$ for sums beyond float64, by the sign of the last term | no `OverflowError` escapes |
| `test_bad_n_rejected` | boundary | $n = -1$, $2.5$, `True` raise `ValueError` | counts are whole numbers |
| `test_rope_inv_freq_golden_hf` | golden | Hugging Face's default RoPE for SmolLM2-135M, Llama 2, Llama 3, and a partial rotary size | exactly what `L7.9` loads |
| `test_rope_ladder_is_geometric` | property | first rung 1, constant ratio $b^{-2/d}$, last rung above $1/b$ | the ladder's shape for any $d$ and $b$ |
| `test_rope_bad_arguments` | boundary | odd or zero $d$, bases $\le 1$ or infinite raise `ValueError` | a flat or climbing ladder is a bug |
| `test_alibi_powers_of_two_exact` | unit | 8 heads: $1/2, \ldots, 1/256$; 16 heads: $2^{-j/2}$; 1 head: $1/256$ | the power-of-two rule |
| `test_alibi_golden_hf` | golden | BLOOM's `build_alibi_tensor` for 20 head counts from 1 to 128 | twelve of them are not powers of two |
| `test_alibi_bad_head_count` | boundary | 0, $-4$, $2.5$ heads raise `ValueError` | a model has whole heads |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. the textbook closed form near $r = 1$ | Adam's bias correction with $\beta_2 = 0.999$ correct to 7 digits instead of 16; tests at $10^{-12}$ fail | `test_sum_near_one_keeps_its_digits` (mutant `s03`) |
| 2. the RoPE exponent $-i/d$ instead of $-2i/d$ | frequencies fall only to $b^{-1/2}$: long-range pairs turn far too fast and checkpoints from HF load with the wrong positions | `test_hand_example_rope_ladder` (mutant `s04`) |
| 3. starting the ladder at $i = 1$ | every frequency shifted one rung; the 1-radian pair is missing | `test_rope_inv_freq_golden_hf` (mutant `s05`) |
| 4. ignoring the non-power-of-two rule, or taking the odd entries | 12-head and 40-head models get slopes no published model uses | `test_hand_example_alibi_six_heads` (mutants `s08`, `s09`, `s10`) |
| 5. one term too many | $S_n$ includes $ar^n$; every bias correction is off | `test_hand_example_geometric_sum` (mutant `s02`) |
| 6. starting the sequence at $ar$ | `geometric` returns $ar, \ldots, ar^n$; ALiBi slopes shift by one rung | `test_geometric_terms` (mutant `s01`) |

## 6. Where it's used next
| Forward | `L5.4` | Registered call site uses this module. |

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M00.1` | powers, $e$, and `log1p`/`expm1` (reading) |
| Back | `lang.01` | numpy `arange` and `power` (reading) |
| Forward | `L7.3` | `rope_inv_freq(d_rot, rope_theta)` builds RoPE's angles, `positions * inv_freq` |
| Forward | `L7.4` | ALiBi biases from `alibi_slopes`; YaRN and NTK scaling bend the same ladder per frequency |
| Forward | `M02.2` | the EMA's weights, `geometric(1 - beta, beta, t)`, and its bias correction, `geometric_sum(1 - beta, beta, t)` |
| Forward | `M10.4` | step-decay schedules are geometric sequences of learning rates (reading) |
| Forward | `M00.2` | the ladder feeds the angles that `rotate_pairs` applies (reading) |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `rope_inv_freq` | Hugging Face `ROPE_INIT_FUNCTIONS` and `compute_default_rope_parameters` | one function per scaling kind (`linear`, `dynamic`, `yarn`, `llama3`, `longrope`) that bends this ladder for longer contexts | `transformers/modeling_rope_utils.py` |
| `alibi_slopes` | BLOOM and MPT attention biases | slopes built once per model, biases fused into the attention kernel | `transformers/models/bloom/modeling_bloom.py` (`build_alibi_tensor`) |
| `geometric_sum` near $r = 1$ | PyTorch `torch.optim.Adam` | the bias corrections `1 - beta1 ** step` and `1 - beta2 ** step` (in float32, where the cancellation is worse) | `torch/optim/adam.py` |
| `log1p` and `expm1` | C99 `log1p`, `expm1` | the accurate small-argument functions every math library ships | `man 3 expm1` |
