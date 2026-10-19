# Softmax in C: three-pass and online two-pass

## Overview

| | |
|---|---|
| **Module** | `L9.2` · side · C · Pass 6 |
| **Solution** | [`softmax.c`](../../../../course/solutions/c/src/kernels/softmax.c): `tl_softmax_f32` (three passes over each row: max, exponentiate and sum, normalize) and `tl_softmax_online_f32` (two passes: a running max with a rescaled running sum, then normalize) |
| **Contract** | [`course/contracts/c/include/tinyllm/softmax.h`](../../../../course/contracts/c/include/tinyllm/softmax.h) · rules: [`c/ABI.md`](../../../../course/contracts/c/ABI.md) |
| **Needs** | `rt.02` the error slot and the loader · `M09.6` `tl_expf` · `M09.2` the Python `softmax`, the specification. Reading: `S-M05` (loop invariants) |
| **Used by** | `L9.3` and `L9.4` build their C-side attention oracle from `tl_softmax_f32`, and apply the online update of this chapter to tiles of keys; with them, the C backend generates the same tokens as numpy |
| **Optional depth** | Milakov and Gimelshein, "Online normalizer calculation for softmax" (2018); Dao et al., "FlashAttention" (2022), section 3.1; Higham, *Accuracy and Stability of Numerical Algorithms*, chapter 4 (summation) |

## Key Takeaways

- **Subtract the row maximum first.** $\mathrm{softmax}(x) = \mathrm{softmax}(x - c)$ for any constant $c$, and with $c = \max_j x_j$ every exponent is at most 0, so rows of $\pm 10^4$ are exact instead of $\infty / \infty$.
- **The online version needs one read of $x$ for the statistics.** It keeps a running maximum $m$ and a running sum $s$, and rescales $s$ by $e^{m_{\text{old}} - m_{\text{new}}}$ whenever the maximum grows; the invariant "$s$ is the sum against the current $m$" holds after every element.
- **Masked entries are $-\infty$ and get exactly 0;** a row with every entry masked gives zeros, not $0/0$, and the online pass never evaluates $-\infty - (-\infty)$.
- **Each row is a distribution:** non-negative, summing to 1 within about $n \cdot 2^{-24}$ for $n$ columns, and the two versions agree element by element.
- **The C kernels agree with `M09.2`'s Python `softmax`** across row lengths from 1 to 4096 and scales from 0.01 to $10^4$.

## How to read this chapter

Sections 1 to 3 build both kernels from first principles (shift invariance, the three passes, the online update and its invariant, masks and NaN, rounding) and run both on one row by hand. Section 4 fixes the interface. Section 5 walks through the solution and explains why each choice is the right one. Try to predict each step before reading it.

---

## 1. Why now

The Python sampler and every attention layer of Parts 5 to 7 call `softmax` from `M09.2`, in numpy. Part 9 moves the forward pass into C so that the Rust engine (the standalone Rust engine) can run it, and attention is where most of the softmax work is: one row of scores per query, per head, per layer, per step. A C softmax that overflows on a large score, turns a fully masked row into NaN, or reads its row three times when two would do, shows up later as a garbage token or as a slow decode. This module writes the row softmax in C twice. The three-pass version is the textbook. The online version is the one idea FlashAttention (`L9.3`) and paged attention (`L9.4`) are built on: you can normalize a row you are still reading, as long as you remember what maximum you normalized against.

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

Each term is in $[0, 1]$, the sum is accumulated in float32 left to right, and each of the $n$ additions rounds with relative error at most $u$. The computed $s$ is within about $n u$ of the true sum (relative), and the outputs then sum to 1 within about $n u$ plus one rounding per element. A sound tolerance is $2 n u + 10^{-7}$. `tl_expf` (`M09.6`) itself is within 4 ulp, which the same bound absorbs. The three-pass and online versions round differently (the online sum is built from rescaled partial sums), so they agree to float32 tolerance, not bit for bit.

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

The same $m$ and $s$ as the three-pass version, and pass 2 writes the same $y$. Both kernels must reproduce these numbers.

## 4. The interface

```c
/* tinyllm/softmax.h */
tl_status tl_softmax_f32(const float *x, float *y, int64_t rows, int64_t cols);         /* 3-pass */
tl_status tl_softmax_online_f32(const float *x, float *y, int64_t rows, int64_t cols);  /* 2-pass */
/* Row-major, contiguous; y may equal x (in place). A row of all -inf gives zeros;
   NaN propagates to its row. TL_EINVAL for a negative dimension, or a NULL pointer
   when rows * cols > 0. */
```

From Python, the two symbols are declared on the ctypes loader and called with `f32_ptr` buffers, as for `tl_matmul_f32`.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| The hand example | unit, smoke | section 3, both kernels | the definition is pinned by numbers you can check by hand |
| The online sum rescales when the max grows | unit | an increasing row (rescale every step) and a decreasing one (never) | the update FlashAttention applies to tiles |
| Large logits stay finite | boundary | $[10^4, 10^4] \to [0.5, 0.5]$ and $[-10^4, 0] \to [0, 1]$ exactly | trained logits and unscaled scores |
| Masked entries and fully masked rows | boundary | $-\infty$ gets 0; an all-$-\infty$ row gives zeros; $-\infty$ first in the online pass | causal and padding masks |
| NaN stays in its row | boundary | a NaN row is NaN, the next row exact | bugs stay visible and local |
| In place | unit | `y == x` | normalizing a score buffer in place |
| Bad arguments | boundary | negative dims and NULL buffers give `TL_EINVAL`; empty work with NULL is fine | errors instead of crashes |
| Rows sum to one, and the versions agree | property, differential | 200 random rows: non-negative, sum to 1 within $2nu$, zeros on masks, the two kernels agree | the definition, for every length up to 501 |

## 5. The solution, step by step

### From first principles

1. **The output must not overflow, whatever the logits.** Shift invariance (section 2.1) lets us subtract any constant; the row maximum is the one constant that bounds every exponent by 0 and makes the sum at least 1.
2. **The three-pass version follows directly**: find $m$, exponentiate and sum, scale by $1/s$. Storing $e_j$ into $y_j$ in pass 2 avoids a second exponential in pass 3.
3. **To fuse the first two passes, keep an invariant**: $s$ is the sum of the terms seen so far *against the current maximum*. When the maximum grows, the one factor $e^{m_{\text{old}} - m_{\text{new}}}$ restores the invariant (section 2.3). The result is two reads of $x$.
4. **$-\infty$ entries must be skipped explicitly in the online pass**, because they contribute 0 but would compute $-\infty - (-\infty)$ while $m$ is still $-\infty$.
5. **A fully masked row must give zeros**, so detect it ($m$ still $-\infty$) and write zeros instead of $0/0$.
6. **NaN must stay visible and stay in its row**: no special case beyond letting it flow into the sum.
7. **In place must work**: in every loop, read `x[j]` before writing `y[j]`.

### The code, annotated

Both kernels share the argument check, and both return before any pointer arithmetic when there is no work:

```c
static tl_status check_args(const char *fn, const float *x, const float *y, int64_t rows, int64_t cols) {
    if (rows < 0 || cols < 0) { tl_set_last_error(fn); return TL_EINVAL; }
    /* NULL is an error only if there is something to read or write. */
    if (rows > 0 && cols > 0 && (x == NULL || y == NULL)) { tl_set_last_error(fn); return TL_EINVAL; }
    return TL_OK;
}
// in each kernel, right after the check:
if (rows == 0 || cols == 0) return TL_OK; /* no work; and NULL + 0 is undefined behavior */
```

The three-pass kernel, steps 2, 5 and 6:

```c
for (int64_t r = 0; r < rows; r++) {
    const float *xr = x + r * cols;
    float *yr = y + r * cols;
    /* Pass 1: the maximum. NaN never wins a > comparison, so it is
     * skipped here and propagates through the sum below (or through the
     * masked-row branch when no entry is finite). */
    float m = -INFINITY, fill = 0.0f;
    for (int64_t j = 0; j < cols; j++) {
        if (xr[j] > m) m = xr[j];
        else if (isnan(xr[j])) fill = xr[j];
    }
    if (m == -INFINITY) { /* every key masked: zeros, not 0/0 (or NaN if one was NaN) */
        for (int64_t j = 0; j < cols; j++) yr[j] = fill;
        continue;
    }
    /* Pass 2: exponentiate and sum. Reading xr[j] before writing yr[j]
     * keeps y == x (in place) correct. */
    float s = 0.0f;
    for (int64_t j = 0; j < cols; j++) {
        float e = tl_expf(xr[j] - m);    /* exponent <= 0: in [0, 1] */
        yr[j] = e;
        s += e;                          /* s >= 1: the max term is e^0 */
    }
    /* Pass 3: normalize, one division per row. */
    float inv = 1.0f / s;
    for (int64_t j = 0; j < cols; j++) yr[j] *= inv;
}
```

The `fill` variable is what keeps a row of $-\infty$ and NaN from looking like a masked row: it becomes NaN if any entry was NaN, so such a row is all NaN, not all zeros.

The online kernel, steps 3 and 4:

```c
for (int64_t r = 0; r < rows; r++) {
    const float *xr = x + r * cols;
    float *yr = y + r * cols;
    /* Pass 1, one read of x. Invariant after element j:
     *   m = max(x_0..x_j),  s = sum_{i <= j} e^(x_i - m). */
    float m = -INFINITY, s = 0.0f;           /* the empty prefix */
    for (int64_t j = 0; j < cols; j++) {
        float v = xr[j];
        if (v == -INFINITY) continue; /* adds e^-inf = 0; and -inf - -inf would be NaN */
        if (v > m) {
            s = s * tl_expf(m - v) + 1.0f;   /* rescale the old sum, add e^0 for v */
            m = v;
        } else {
            s += tl_expf(v - m); /* also the path a NaN takes: s becomes NaN */
        }
    }
    if (m == -INFINITY && s == 0.0f) {       /* nothing finite and nothing NaN */
        for (int64_t j = 0; j < cols; j++) yr[j] = 0.0f;
        continue;
    }
    /* Pass 2, the second read of x. */
    float inv = 1.0f / s;
    for (int64_t j = 0; j < cols; j++) yr[j] = tl_expf(xr[j] - m) * inv;
}
```

On the first finite element, `m` is $-\infty$, so `tl_expf(m - v)` is $e^{-\infty} = 0$ and `s` becomes $0 \cdot 0 + 1 = 1$: the initialisation needs no special case. On the worked example, the loop goes through $s = 1, 1.3678794, 1.5032147$ and $m = 1, 2, 3$, exactly the table of section 3. The rescale factor is $e^{m - v}$ with $m$ the *old* maximum, an exponent below 0, so it can only shrink the sum.

Full solution: [`softmax.c`](../../../../course/solutions/c/src/kernels/softmax.c); contract: [`softmax.h`](../../../../course/contracts/c/include/tinyllm/softmax.h).

### Why this solution

- **Cost.** Both are $O(n)$ per row. The three-pass kernel does $n$ exponentials, three reads, and two writes; the online kernel does up to $2n$ exponentials (one in pass 1, one in pass 2, plus a rescale whenever the maximum grows) but two reads and one write. For rows that stay in cache the three-pass kernel is often faster; for rows streamed from memory, and for tiles in FlashAttention, the online form wins because it never needs the whole row at once.
- **Against no shift (or a fixed shift).** Unshifted exponentials overflow above about 88.7 and underflow to an all-zero row below about $-103$; any fixed shift fails on some row. The row maximum is the one shift that always works.
- **Against `expf` from libm.** `tl_expf` (`M09.6`) has a known error bound (4 ulp) and identical results on every platform, which the cross-language comparisons need.
- **Against dividing each element by $s$.** $n$ divisions cost more than one reciprocal and $n$ multiplies; the extra rounding per element is within the $2nu$ tolerance.
- **Tradeoff accepted.** The two kernels round differently and agree only to float32 tolerance; the sum is accumulated left to right in float32 rather than pairwise, which keeps the code simple and the error within $nu$ for the row lengths used.

### Explain it back

1. In the online pass, why is the rescale factor $e^{m_{\text{old}} - v}$ and not $e^{v - m_{\text{old}}}$?
   <details><summary>Answer</summary>Each old term is $e^{x_i - m_{\text{old}}}$ and must become $e^{x_i - v}$. The ratio is $e^{m_{\text{old}} - v}$, which is below 1 because $v > m_{\text{old}}$. The flipped sign would multiply the sum by a factor above 1 every time the maximum grows, and it would explode.</details>
2. Why does the online kernel skip $-\infty$ entries instead of letting the update handle them?
   <details><summary>Answer</summary>While $m$ is still $-\infty$, the else branch would compute `tl_expf(-inf - (-inf))`, which is $e^{\mathrm{NaN}}$, and the sum would become NaN for a perfectly valid row such as $[-\infty, 1]$. A $-\infty$ entry contributes 0 regardless, so skipping it is exact.</details>
3. How does the first finite element initialise the state without a special case?
   <details><summary>Answer</summary>With $m = -\infty$ and $s = 0$, any finite $v$ takes the `v > m` branch: $s = 0 \cdot e^{-\infty - v} + 1 = 1$, $m = v$. That is exactly the invariant for a one-element prefix.</details>
4. Why does the three-pass kernel track `fill` in pass 1?
   <details><summary>Answer</summary>A row whose only non-$-\infty$ entries are NaN keeps $m = -\infty$ (NaN never wins a comparison), so it would take the masked-row branch and become zeros, hiding an upstream bug. `fill` becomes NaN in that case, so the row is written as NaN.</details>
5. Why is the online version the basis of FlashAttention, though it does more exponentials?
   <details><summary>Answer</summary>Its state $(m, s)$ summarises any prefix and merges with another block's state by the same rescaling. Attention can then process keys tile by tile, carrying $(m, s)$ and a rescaled output accumulator, without ever storing the full row of scores; the memory traffic saved outweighs the extra exponentials.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| multiplying by $s$ instead of $1/s$ | rows sum to $s^2$, not 1 | computes `inv = 1.0f / s` and multiplies by `inv` |
| a wrong normalizer in the online pass 2 | rows do not sum to 1 | pass 2 uses the final $(m, s)$ of pass 1 |
| not rescaling the running sum when the maximum grows | an increasing row looks almost uniform | `s = s * tl_expf(m - v) + 1.0f` on every new maximum |
| rescaling by $e^{m_{\text{new}} - m_{\text{old}}}$ (the sign flipped) | the running sum explodes | the factor is `tl_expf(m - v)` with `m` the old maximum |
| not subtracting the maximum | $e^{10^4} = \infty$, and $\infty / \infty$ = NaN | every exponent is `x - m` |
| no special case for a fully masked row | $0/0$: NaN poisons the whole batch through the next matmul | `m == -INFINITY` writes zeros |
| updating the online state with a $-\infty$ entry while $m = -\infty$ | $e^{\mathrm{NaN}}$ in the running sum | `continue` on `v == -INFINITY` |
| zero-filling a row that is all NaN | a broken forward pass looks like a masked row | `fill` carries NaN in the three-pass kernel; `s == 0.0f` fails for NaN in the online one |
| reading `x[j]` after writing `y[j]` | wrong in place, right otherwise | each loop reads `xr[j]` before writing `yr[j]` |
| skipping the NULL check | a crash instead of `TL_EINVAL` | `check_args` refuses NULL when there is work |
| computing `x + r * cols` when `x` is NULL and `cols` is 0 | UBSan: "applying zero offset to null pointer" | returns before touching pointers when there is no work |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `rt.02` | the error slot behind `TL_EINVAL` |
| Back | `M09.6` | `tl_expf`: every $e^{x}$ in this file |
| Back | `M09.2` | the Python `softmax` that is the specification |
| Forward | `L9.3` | FlashAttention: the online update over tiles of keys, with the output rescaled by the same factor; its C tests build the naive attention oracle from `tl_softmax_f32` |
| Forward | `L9.4` | paged attention for decode: the online update over KV blocks; its C oracle uses `tl_softmax_f32` too |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| online two-pass softmax | FlashAttention's tile update | the same rescaling applied to the output accumulator, so attention never stores the score matrix | Dao et al. (2022), algorithm 1 |
| scalar loops | PyTorch `softmax` CPU kernel | vectorized max, exp, and sum with SIMD, a vectorized polynomial exp | `aten/src/ATen/native/cpu/SoftMaxKernel.cpp` |
| one row per loop | llama.cpp `ggml_soft_max` | fused scale and mask (ALiBi, causal) in the same pass | `ggml/src/ggml-cpu/ops.cpp` |
| float32 sum | cuDNN and Triton fused softmax | one block per row, a tree reduction for the max and the sum | Triton tutorial "Fused Softmax" |
