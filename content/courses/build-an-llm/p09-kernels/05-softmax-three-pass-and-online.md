<!-- ol:module L9.2 -->
# Softmax in C: three-pass and online two-pass

## Overview

| | |
|---|---|
| **Module** | `L9.2` · side · C · Pass 6 · 2 to 3 h |
| **You build** | `c/src/kernels/softmax.c`: `tl_softmax_f32` (three passes over each row: max, exponentiate and sum, normalize) and `tl_softmax_online_f32` (two passes: a running max with a rescaled running sum, then normalize) |
| **Contract** | [`course/contracts/c/include/tinyllm/softmax.h`](../../../../course/contracts/c/include/tinyllm/softmax.h) · rules: [`c/ABI.md`](../../../../course/contracts/c/ABI.md) |
| **Tests** | `course/tests/L9.2/`: `test_softmax.c` (C, under ASan and UBSan, both kernels through one table) and shared file fixtures (what they check: section 4) |
| **Needs** | `rt.02` the error slot and the loader · `M09.6` your `tl_expf` · `M09.2` your Python `softmax`, the specification (or `--ref-deps`). Reading: `S-M05` (loop invariants) |
| **Used by** | `L9.3` and `L9.4` build their C-side attention oracle from `tl_softmax_f32`, and apply the online update of this chapter to tiles of keys |
| **Milestone** | `MS-L9` (the C backend generates the same tokens as numpy) |
| **Optional depth** | Milakov and Gimelshein, "Online normalizer calculation for softmax" (2018); Dao et al., "FlashAttention" (2022), section 3.1; Higham, *Accuracy and Stability of Numerical Algorithms*, chapter 4 (summation) |

## Key Takeaways

- **Subtract the row maximum first.** $\mathrm{softmax}(x) = \mathrm{softmax}(x - c)$ for any constant $c$, and with $c = \max_j x_j$ every exponent is at most 0, so rows of $\pm 10^4$ are exact instead of $\infty / \infty$ (`large_logits_stay_finite`).
- **The online version needs one read of $x$ for the statistics.** It keeps a running maximum $m$ and a running sum $s$, and rescales $s$ by $e^{m_{\text{old}} - m_{\text{new}}}$ whenever the maximum grows; the invariant "$s$ is the sum against the current $m$" holds after every element (`online_rescales_when_the_max_grows`).
- **Masked entries are $-\infty$ and get exactly 0;** a row with every entry masked gives zeros, not $0/0$, and the online pass never evaluates $-\infty - (-\infty)$ (`masked_entries_and_fully_masked_rows`).
- **Each row is a distribution:** non-negative, summing to 1 within about $n \cdot 2^{-24}$ for $n$ columns, and the two versions agree element by element (`rows_sum_to_one_and_versions_agree`).
- **Your C kernels agree with your Python `softmax`** across row lengths from 1 to 4096 and scales from 0.01 to $10^4$ (`test_matches_your_m09_2_softmax`).

## How to work this chapter

```bash
ol start L9.2              # stubs c/src/kernels/softmax.c into your repo
ol tests L9.2              # read the test catalog first
ol check L9.2              # exit code is the verdict
ol check L9.2 --ref-deps   # only if rt.02, M09.6, or M09.2 is not passing yet
ol parity softmax.online   # the golden parity suite against a float64 oracle
ol diff  L9.2              # after passing: your code against the reference
```

---

## 1. Why now

Your Python sampler and every attention layer you wrote in Parts 5 to 7 call `softmax` from `M09.2`, in numpy. Part 9 moves the forward pass into C so that the Rust engine (the standalone Rust engine) can run it, and attention is where most of the softmax work is: one row of scores per query, per head, per layer, per step. A C softmax that overflows on a large score, turns a fully masked row into NaN, or reads its row three times when two would do, shows up later as a garbage token or as a slow decode. This module writes the row softmax in C twice. The three-pass version is the textbook. The online version is the one idea FlashAttention (`L9.3`) and paged attention (`L9.4`) are built on: you can normalize a row you are still reading, as long as you remember what maximum you normalized against.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x \in \mathbb{R}^n$ | one row of scores (logits) | `float[cols]` |
| $n$ | the row length, `cols` | `int64_t` |
| $y = \mathrm{softmax}(x)$ | $y_j = e^{x_j} / \sum_i e^{x_i}$ | `float[cols]` |
| $m$ | the row maximum, $\max_j x_j$ (or the running maximum) | scalar |
| $s$ | the normalizer $\sum_j e^{x_j - m}$ (or the running sum) | scalar |
| $m_j, s_j$ | running maximum and sum after reading $x_0, \dots, x_j$ | scalars |
| $u$ | float32 unit roundoff, $2^{-24} \approx 6 \times 10^{-8}$ | |

### 2.1 Shift invariance and overflow

For any constant $c$,

$$\frac{e^{x_j - c}}{\sum_i e^{x_i - c}} = \frac{e^{-c} e^{x_j}}{e^{-c} \sum_i e^{x_i}} = \frac{e^{x_j}}{\sum_i e^{x_i}} ,$$

so subtracting a constant changes nothing mathematically (`M09.2` proved this in Python). Numerically it changes everything: float32 overflows above about $3.4 \times 10^{38} = e^{88.7}$, so $e^{100}$ is $\infty$ and a row containing it computes $\infty / \infty = \mathrm{NaN}$. With $c = m = \max_j x_j$, every exponent $x_j - m \le 0$, every term is in $[0, 1]$, the largest term is exactly $e^0 = 1$, and the sum is at least 1. Nothing overflows, and the division never divides by something tiny.

### 2.2 Three passes

The direct algorithm reads the row three times:

1. $m = \max_j x_j$.
2. $e_j = e^{x_j - m}$ (stored into $y_j$), $s = \sum_j e_j$.
3. $y_j = e_j \cdot (1 / s)$.

Computing $1/s$ once and multiplying is cheaper than $n$ divisions and costs at most one extra rounding per element.

### 2.3 The online update

Can the maximum and the sum come from one read? Suppose that after reading $x_0, \dots, x_j$ we hold

$$m_j = \max_{i \le j} x_i, \qquad s_j = \sum_{i \le j} e^{x_i - m_j} .$$

Read $x_{j+1}$. If it does not exceed $m_j$, the maximum stays and the new term joins the sum: $m_{j+1} = m_j$, $s_{j+1} = s_j + e^{x_{j+1} - m_j}$. If it does, every old term was computed against the wrong maximum. Since $e^{x_i - m_{j+1}} = e^{x_i - m_j} \, e^{m_j - m_{j+1}}$, the whole old sum rescales by one factor:

$$m_{j+1} = x_{j+1}, \qquad s_{j+1} = s_j \, e^{m_j - m_{j+1}} + e^{0} = s_j \, e^{m_j - m_{j+1}} + 1 .$$

Both branches preserve the two equations above, and they hold trivially for the empty prefix with $m = -\infty$, $s = 0$. By induction (`S-M05`), after the last element $m$ is the row maximum and $s$ the normalizer, so the second pass writes $y_j = e^{x_j - m} / s$. That is two reads of $x$ and one write of $y$, against three reads and two writes for the three-pass version: for a row that does not fit in cache, a third less memory traffic.

The same update **merges two partial results**: a prefix summarized by $(m_a, s_a)$ and a block summarized by $(m_b, s_b)$ combine to $m = \max(m_a, m_b)$, $s = s_a e^{m_a - m} + s_b e^{m_b - m}$. FlashAttention applies exactly this to tiles of keys, with the weighted sum of values carried along and rescaled by the same factor.

### 2.4 Masked entries and NaN

Attention blocks a key by giving its score $-\infty$, and $e^{-\infty} = 0$, so a masked entry contributes nothing and gets weight exactly 0. Two cases need code:

- **A fully masked row** (a padded query, a window that excludes everything): $m = -\infty$ and $s = \sum 0 = 0$, so the textbook gives $0/0 = \mathrm{NaN}$. The contract says zeros: such a query attends to nothing.
- **$-\infty$ in the online pass before any finite value**: the update would compute $e^{x_j - m} = e^{-\infty - (-\infty)} = e^{\mathrm{NaN}}$. A $-\infty$ entry adds 0 whatever $m$ is, so the kernel simply skips it.

NaN is different: it means a bug upstream, and the contract makes it visible. No comparison with NaN is true, so NaN never becomes the maximum; it enters the sum, the sum becomes NaN, and the whole row becomes NaN. Rows are independent, so the next row is exact.

### 2.5 Rounding

Each term is in $[0, 1]$, the sum is accumulated in float32 left to right, and each of the $n$ additions rounds with relative error at most $u$. The computed $s$ is within about $n u$ of the true sum (relative), and the outputs then sum to 1 within about $n u$ plus one rounding per element. The property test allows $2 n u + 10^{-7}$. `tl_expf` (`M09.6`) itself is within 4 ulp, which the same bound absorbs. The three-pass and online versions round differently (the online sum is built from rescaled partial sums), so they agree to float32 tolerance, not bit for bit.

## 3. Worked example by hand

$x = [1, 2, 3]$, one row.

**Three passes.** Pass 1: $m = 3$. Pass 2: $e = [e^{-2}, e^{-1}, e^{0}] = [0.1353353, 0.3678794, 1]$, $s = 1.5032147$. Pass 3: $1/s = 0.6652410$, so

$$y = [0.1353353, 0.3678794, 1] \times 0.6652410 = [0.0900306, 0.2447285, 0.6652410] .$$

**Online.** Start $m = -\infty$, $s = 0$.

| $j$ | $x_j$ | Branch | $s$ after | $m$ after |
|---|---|---|---|---|
| 0 | 1 | $1 > -\infty$: $s = 0 \cdot e^{-\infty} + 1$ | 1 | 1 |
| 1 | 2 | $2 > 1$: $s = 1 \cdot e^{-1} + 1$ | 1.3678794 | 2 |
| 2 | 3 | $3 > 2$: $s = 1.3678794 \cdot e^{-1} + 1 = 0.5032147 + 1$ | 1.5032147 | 3 |

The same $m$ and $s$ as the three-pass version, and pass 2 writes the same $y$. These are the numbers of the first test, `hand_example`, which runs both kernels, and of `test_hand_example`.

## 4. The interface

```c
/* tinyllm/softmax.h */
tl_status tl_softmax_f32(const float *x, float *y, int64_t rows, int64_t cols);         /* 3-pass */
tl_status tl_softmax_online_f32(const float *x, float *y, int64_t rows, int64_t cols);  /* 2-pass */
/* Row-major, contiguous; y may equal x (in place). A row of all -inf gives zeros;
   NaN propagates to its row. TL_EINVAL for a negative dimension, or a NULL pointer
   when rows * cols > 0. */
```

From Python, declare the two symbols on your loader and pass `f32_ptr` buffers, as for `tl_matmul_f32`.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example` | unit, smoke | section 3, both kernels | you and the tests agree on the definition |
| `online_rescales_when_the_max_grows` | unit | an increasing row (rescale every step) and a decreasing one (never) | the update FlashAttention applies to tiles |
| `large_logits_stay_finite` | boundary | $[10^4, 10^4] \to [0.5, 0.5]$ and $[-10^4, 0] \to [0, 1]$ exactly | trained logits and unscaled scores |
| `masked_entries_and_fully_masked_rows` | boundary | $-\infty$ gets 0; an all-$-\infty$ row gives zeros; $-\infty$ first in the online pass | causal and padding masks |
| `nan_propagates_to_its_row_only` | boundary | a NaN row is NaN, the next row exact | bugs stay visible and local |
| `in_place` | unit | `y == x` | normalizing a score buffer in place |
| `bad_arguments_are_einval` | boundary | negative dims and NULL buffers give `TL_EINVAL`; empty work with NULL is fine | errors instead of crashes |
| `rows_sum_to_one_and_versions_agree` | property, differential | 200 random rows: non-negative, sum to 1 within $2nu$, zeros on masks, the two kernels agree | the definition, for every length up to 501 |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| multiplying by $s$ instead of $1/s$ | rows sum to $s^2$, not 1 | `hand_example` (mutant `s01`) |
| a wrong normalizer in the online pass 2 | rows do not sum to 1 | `hand_example` (mutant `s02`) |
| not rescaling the running sum when the maximum grows | an increasing row looks almost uniform | `online_rescales_when_the_max_grows` (mutant `s03`) |
| rescaling by $e^{m_{\text{new}} - m_{\text{old}}}$ (the sign flipped) | the running sum explodes | `online_rescales_when_the_max_grows` (mutant `s04`) |
| not subtracting the maximum | $e^{10^4} = \infty$, and $\infty / \infty$ = NaN | `large_logits_stay_finite` (mutants `s05`, `s06`) |
| no special case for a fully masked row | $0/0$: NaN poisons the whole batch through the next matmul | `masked_entries_and_fully_masked_rows` (mutants `s07`, `s09`) |
| updating the online state with a $-\infty$ entry while $m = -\infty$ | $e^{\mathrm{NaN}}$ in the running sum | `masked_entries_and_fully_masked_rows` (mutant `s08`) |
| zero-filling a row that is all NaN | a broken forward pass looks like a masked row | `nan_propagates_to_its_row_only` (mutant `s10`) |
| reading `x[j]` after writing `y[j]` | wrong in place, right otherwise | `in_place` (mutant `s11`) |
| skipping the NULL check | a crash instead of `TL_EINVAL` | `bad_arguments_are_einval` (mutant `s12`) |
| computing `x + r * cols` when `x` is NULL and `cols` is 0 | UBSan: "applying zero offset to null pointer"; return before touching pointers when there is no work | `bad_arguments_are_einval` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `rt.02` | the error slot behind `TL_EINVAL` the Python tests use |
| Back | `M09.6` | `tl_expf`, your exponential: every $e^{x}$ in this file |
| Back | `M09.2` | the Python `softmax` that is the specification |
| Forward | `L9.3` | FlashAttention: the online update over tiles of keys, with the output rescaled by the same factor; its C tests build the naive attention oracle from `tl_softmax_f32` |
| Forward | `L9.4` | paged attention for decode: the online update over KV blocks; its C oracle uses `tl_softmax_f32` too |

If you skip this module, `ol check L9.3` stops with `BLOCKED ... needs L9.2`: build it, or pass `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| online two-pass softmax | FlashAttention's tile update | the same rescaling applied to the output accumulator, so attention never stores the score matrix | Dao et al. (2022), algorithm 1 |
| scalar loops | PyTorch `softmax` CPU kernel | vectorized max, exp, and sum with SIMD, a vectorized polynomial exp | `aten/src/ATen/native/cpu/SoftMaxKernel.cpp` |
| one row per loop | llama.cpp `ggml_soft_max` | fused scale and mask (ALiBi, causal) in the same pass | `ggml/src/ggml-cpu/ops.cpp` |
| float32 sum | cuDNN and Triton fused softmax | one block per row, a tree reduction for the max and the sum | Triton tutorial "Fused Softmax" |
