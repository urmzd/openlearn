# Softmax in C: three-pass and online two-pass

The Python sampler and every attention layer you built earlier in this book call the numpy `softmax` from [Stable numerics: logsumexp, softmax, and compensated sums](../../mathematics/15-numerical-methods-and-floating-point/02-stable-numerics.md) in *Mathematics*. This chapter moves the forward pass into C, and attention is where most of the softmax work is: one row of scores per query, per head, per layer, per step. A C softmax that overflows on a large score, turns a fully masked row into NaN, or reads its row three times when two would do, shows up later as a garbage token or a slow decode.

This section writes the row softmax in C twice. The three-pass version is the textbook algorithm. The online version is the one idea that [FlashAttention](05-flash-attention-forward.md) and [paged attention](06-paged-attention-decode.md) are built on: you can normalize a row you are still reading, as long as you remember which maximum you normalized against. By the end you will be able to derive the online update, prove its invariant, and explain every special case that keeps masked rows at zero and NaN visible.

You need the error slot from [C runtime support and an arena allocator with marks](01-arena-allocator-with-marks.md) and `tl_expf` from [Polynomial approximation and range reduction: expf in C](../../mathematics/15-numerical-methods-and-floating-point/06-polynomial-approximation-and-expf.md). The Python `softmax` in the stable-numerics section is the specification these kernels must match, and the loop-invariant arguments in the [discrete math problem set](../../mathematics/11-discrete-math-1/03-problem-set/README.md) are useful background. The contract is [`softmax.h`](../../../../course/contracts/c/include/tinyllm/softmax.h) under [`c/ABI.md`](../../../../course/contracts/c/ABI.md), and the reference solution is [`softmax.c`](../../../../course/solutions/c/src/kernels/softmax.c): `tl_softmax_f32` (three passes over each row: max, exponentiate and sum, normalize) and `tl_softmax_online_f32` (two passes: a running max with a rescaled running sum, then normalize).

> **Learning objectives**
> - Use shift invariance to compute softmax without overflow for logits as large as $\pm 10^4$.
> - Derive the online update for the running maximum and sum, and prove its invariant by induction.
> - Merge two partial softmax summaries, the step FlashAttention applies to tiles.
> - Handle masked entries, fully masked rows, and NaN exactly as the contract requires.
> - Bound the rounding error of a row and choose a sound test tolerance.

## Shift invariance and overflow

The section uses this notation:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x \in \mathbb{R}^n$ | one row of scores (logits) | `float[cols]` |
| $n$ | the row length, `cols` | `int64_t` |
| $y = \mathrm{softmax}(x)$ | $y_j = e^{x_j} / \sum_i e^{x_i}$ | `float[cols]` |
| $m$ | the row maximum, $\max_j x_j$ (or the running maximum) | scalar |
| $s$ | the normalizer $\sum_j e^{x_j - m}$ (or the running sum) | scalar |
| $m_j, s_j$ | running maximum and sum after reading $x_0, \dots, x_j$ | scalars |
| $u$ | float32 unit roundoff, $2^{-24} \approx 6 \times 10^{-8}$ | |

For any constant $c$,

$$\frac{e^{x_j - c}}{\sum_i e^{x_i - c}} = \frac{e^{-c} e^{x_j}}{e^{-c} \sum_i e^{x_i}} = \frac{e^{x_j}}{\sum_i e^{x_i}} ,$$

so subtracting a constant changes nothing mathematically; the stable-numerics section proved this in Python. Numerically it changes everything. float32 overflows above about $3.4 \times 10^{38} = e^{88.7}$, so $e^{100}$ is $\infty$ and a row containing it computes $\infty / \infty = \mathrm{NaN}$. With $c = m = \max_j x_j$, every exponent $x_j - m \le 0$, every term is in $[0, 1]$, the largest term is exactly $e^0 = 1$, and the sum is at least 1. Nothing overflows, and the division never divides by something tiny.

## Three passes

The direct algorithm reads the row three times. First it finds $m = \max_j x_j$. Then it computes $e_j = e^{x_j - m}$, stores each one into $y_j$, and accumulates $s = \sum_j e_j$. Finally it writes $y_j = e_j \cdot (1 / s)$. Computing $1/s$ once and multiplying is cheaper than $n$ divisions and costs at most one extra rounding per element.

## The online update

Three reads raise an obvious question: can the maximum and the sum come from one read? Suppose that after reading $x_0, \dots, x_j$ you hold

$$m_j = \max_{i \le j} x_i, \qquad s_j = \sum_{i \le j} e^{x_i - m_j} .$$

Now read $x_{j+1}$. If it does not exceed $m_j$, the maximum stays and the new term joins the sum: $m_{j+1} = m_j$, $s_{j+1} = s_j + e^{x_{j+1} - m_j}$. If it does exceed $m_j$, every old term was computed against the wrong maximum. But since $e^{x_i - m_{j+1}} = e^{x_i - m_j} \, e^{m_j - m_{j+1}}$, the whole old sum rescales by one factor:

$$m_{j+1} = x_{j+1}, \qquad s_{j+1} = s_j \, e^{m_j - m_{j+1}} + e^{0} = s_j \, e^{m_j - m_{j+1}} + 1 .$$

Both branches preserve the two equations above, and they hold trivially for the empty prefix with $m = -\infty$, $s = 0$. By induction, after the last element $m$ is the row maximum and $s$ the normalizer, so a second pass writes $y_j = e^{x_j - m} / s$. That is two reads of $x$ and one write of $y$, against three reads and two writes for the three-pass version: for a row that does not fit in cache, a third less memory traffic.

The same update also **merges two partial results**. A prefix summarized by $(m_a, s_a)$ and a block summarized by $(m_b, s_b)$ combine to $m = \max(m_a, m_b)$, $s = s_a e^{m_a - m} + s_b e^{m_b - m}$. FlashAttention applies exactly this to tiles of keys, carrying along the weighted sum of values and rescaling it by the same factor.

## Masked entries and NaN

Attention blocks a key by giving its score $-\infty$, and $e^{-\infty} = 0$, so a masked entry contributes nothing and gets weight exactly 0. Two cases still need code.

The first is a **fully masked row**, such as a padded query or a window that excludes everything. Then $m = -\infty$ and $s = \sum 0 = 0$, and the textbook gives $0/0 = \mathrm{NaN}$. The contract says zeros instead: such a query attends to nothing. The second is **$-\infty$ in the online pass before any finite value**: the update would compute $e^{x_j - m} = e^{-\infty - (-\infty)} = e^{\mathrm{NaN}}$. A $-\infty$ entry adds 0 whatever $m$ is, so the kernel simply skips it.

NaN is different: it means a bug upstream, and the contract makes it visible. No comparison with NaN is true, so NaN never becomes the maximum; it enters the sum, the sum becomes NaN, and the whole row becomes NaN. Rows are independent, so the next row is exact.

## Rounding

Each term is in $[0, 1]$, the sum is accumulated in float32 left to right, and each of the $n$ additions rounds with relative error at most $u$. The computed $s$ is therefore within about $n u$ of the true sum (relative), and the outputs sum to 1 within about $n u$ plus one rounding per element. A sound tolerance is $2 n u + 10^{-7}$. `tl_expf` itself is within 4 ulp, which the same bound absorbs. The three-pass and online versions round differently, because the online sum is built from rescaled partial sums, so they agree to float32 tolerance, not bit for bit.

## Working it by hand

Take one row, $x = [1, 2, 3]$.

With three passes, pass 1 gives $m = 3$. Pass 2 gives $e = [e^{-2}, e^{-1}, e^{0}] = [0.1353353, 0.3678794, 1]$ and $s = 1.5032147$. Pass 3 computes $1/s = 0.6652410$, so

$$y = [0.1353353, 0.3678794, 1] \times 0.6652410 = [0.0900306, 0.2447285, 0.6652410] .$$

Online, start with $m = -\infty$, $s = 0$ and read the elements one at a time:

| $j$ | $x_j$ | Branch | $s$ after | $m$ after |
|---|---|---|---|---|
| 0 | 1 | $1 > -\infty$: $s = 0 \cdot e^{-\infty} + 1$ | 1 | 1 |
| 1 | 2 | $2 > 1$: $s = 1 \cdot e^{-1} + 1$ | 1.3678794 | 2 |
| 2 | 3 | $3 > 2$: $s = 1.3678794 \cdot e^{-1} + 1 = 0.5032147 + 1$ | 1.5032147 | 3 |

You arrive at the same $m$ and $s$ as the three-pass version, and pass 2 writes the same $y$. Both kernels must reproduce these numbers.

## The interface

```c
/* tinyllm/softmax.h */
tl_status tl_softmax_f32(const float *x, float *y, int64_t rows, int64_t cols);         /* 3-pass */
tl_status tl_softmax_online_f32(const float *x, float *y, int64_t rows, int64_t cols);  /* 2-pass */
/* Row-major, contiguous; y may equal x (in place). A row of all -inf gives zeros;
   NaN propagates to its row. TL_EINVAL for a negative dimension, or a NULL pointer
   when rows * cols > 0. */
```

From Python, the two symbols are declared on the ctypes loader and called with `f32_ptr` buffers, as for `tl_matmul_f32`.

## Building it step by step

The output must not overflow, whatever the logits. Shift invariance lets you subtract any constant, and the row maximum is the one constant that bounds every exponent by 0 and makes the sum at least 1. The three-pass version follows directly: find $m$, exponentiate and sum, scale by $1/s$. Storing $e_j$ into $y_j$ in pass 2 avoids a second exponential in pass 3.

To fuse the first two passes, keep an invariant: $s$ is the sum of the terms seen so far *against the current maximum*. When the maximum grows, the one factor $e^{m_{\text{old}} - m_{\text{new}}}$ restores the invariant, and the result is two reads of $x$. Then come the special cases. $-\infty$ entries must be skipped explicitly in the online pass, because they contribute 0 but would compute $-\infty - (-\infty)$ while $m$ is still $-\infty$. A fully masked row must give zeros, so detect it ($m$ still $-\infty$) and write zeros instead of $0/0$. NaN must stay visible and stay in its row, which needs no special case beyond letting it flow into the sum. And in place must work, so in every loop you read `x[j]` before writing `y[j]`.

### Argument checks

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

### The three-pass kernel

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

### The online kernel

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

On the first finite element, `m` is $-\infty$, so `tl_expf(m - v)` is $e^{-\infty} = 0$ and `s` becomes $0 \cdot 0 + 1 = 1$: the initialization needs no special case. On the worked example, the loop goes through $s = 1, 1.3678794, 1.5032147$ and $m = 1, 2, 3$, exactly the table above. The rescale factor is $e^{m - v}$ with $m$ the *old* maximum, an exponent below 0, so it can only shrink the sum.

The full solution is [`softmax.c`](../../../../course/solutions/c/src/kernels/softmax.c), against the contract [`softmax.h`](../../../../course/contracts/c/include/tinyllm/softmax.h).

### Why this design

Both kernels are $O(n)$ per row. The three-pass kernel does $n$ exponentials, three reads, and two writes; the online kernel does up to $2n$ exponentials (one in pass 1, one in pass 2, plus a rescale whenever the maximum grows) but only two reads and one write. For rows that stay in cache the three-pass kernel is often faster; for rows streamed from memory, and for tiles in FlashAttention, the online form wins because it never needs the whole row at once.

The alternatives each fail somewhere. With no shift, or a fixed shift, unshifted exponentials overflow above about 88.7 and underflow to an all-zero row below about $-103$, and any fixed shift fails on some row; the row maximum is the one shift that always works. `expf` from libm has no common error bound or cross-platform results, whereas `tl_expf` has a known bound (4 ulp) and identical results on every platform, which the cross-language comparisons need. Dividing each element by $s$ costs $n$ divisions instead of one reciprocal and $n$ multiplies, and the extra rounding per element of the reciprocal approach is within the $2nu$ tolerance. The tradeoff accepted is that the two kernels round differently and agree only to float32 tolerance, and that the sum is accumulated left to right in float32 rather than pairwise, which keeps the code simple and the error within $nu$ for the row lengths used.

## What can go wrong

Every rule above has a test that pins it:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| The hand example | unit, smoke | the worked example, both kernels | the definition is pinned by numbers you can check by hand |
| The online sum rescales when the max grows | unit | an increasing row (rescale every step) and a decreasing one (never) | the update FlashAttention applies to tiles |
| Large logits stay finite | boundary | $[10^4, 10^4] \to [0.5, 0.5]$ and $[-10^4, 0] \to [0, 1]$ exactly | trained logits and unscaled scores |
| Masked entries and fully masked rows | boundary | $-\infty$ gets 0; an all-$-\infty$ row gives zeros; $-\infty$ first in the online pass | causal and padding masks |
| NaN stays in its row | boundary | a NaN row is NaN, the next row exact | bugs stay visible and local |
| In place | unit | `y == x` | normalizing a score buffer in place |
| Bad arguments | boundary | negative dims and NULL buffers give `TL_EINVAL`; empty work with NULL is fine | errors instead of crashes |
| Rows sum to one, and the versions agree | property, differential | 200 random rows: non-negative, sum to 1 within $2nu$, zeros on masks, the two kernels agree | the definition, for every length up to 501 |

Beyond these tests, the C kernels must agree with the Python `softmax` from *Mathematics* across row lengths from 1 to 4096 and scales from 0.01 to $10^4$.

The arithmetic mistakes are easy to make and easy to spot once you know them. Multiplying by $s$ instead of $1/s$ makes rows sum to $s^2$, not 1; compute `inv = 1.0f / s` and multiply by it. Using the wrong normalizer in the online pass 2 means rows do not sum to 1; pass 2 must use the final $(m, s)$ of pass 1. Not rescaling the running sum when the maximum grows makes an increasing row look almost uniform; the solution does `s = s * tl_expf(m - v) + 1.0f` on every new maximum.

> **Warning:** Flipping the sign of the rescale, $e^{m_{\text{new}} - m_{\text{old}}}$, makes the running sum explode. The factor is `tl_expf(m - v)` with `m` the old maximum.

Not subtracting the maximum gives $e^{10^4} = \infty$ and $\infty / \infty = \mathrm{NaN}$; every exponent must be `x - m`.

The special cases are where most real bugs live. Without a special case for a fully masked row, $0/0$ produces NaN that poisons the whole batch through the next matmul; `m == -INFINITY` writes zeros. Updating the online state with a $-\infty$ entry while $m = -\infty$ puts $e^{\mathrm{NaN}}$ in the running sum; `continue` on `v == -INFINITY`. Zero-filling a row that is all NaN makes a broken forward pass look like a masked row; `fill` carries NaN in the three-pass kernel, and `s == 0.0f` fails for NaN in the online one.

Finally, memory and argument handling. Reading `x[j]` after writing `y[j]` is wrong in place and right otherwise, so the bug hides; each loop reads `xr[j]` before writing `yr[j]`. Skipping the NULL check gives a crash instead of `TL_EINVAL`; `check_args` refuses NULL when there is work. And computing `x + r * cols` when `x` is NULL and `cols` is 0 trips UBSan ("applying zero offset to null pointer"); return before touching pointers when there is no work.

## Where this leads

These kernels rest on the error slot behind `TL_EINVAL` from [the runtime support layer](01-arena-allocator-with-marks.md), on `tl_expf` for every $e^x$ in the file, and on the Python `softmax` from [Stable numerics](../../mathematics/15-numerical-methods-and-floating-point/02-stable-numerics.md) as the specification. Looking forward, [FlashAttention forward in C](05-flash-attention-forward.md) applies the online update over tiles of keys, with the output rescaled by the same factor, and its C tests build the naive attention oracle from `tl_softmax_f32`. [Paged attention for decode in C](06-paged-attention-decode.md) applies the online update over KV blocks, and its C oracle uses `tl_softmax_f32` too. With them, the C backend generates the same tokens as numpy.

Production softmax kernels extend the same ideas:

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| online two-pass softmax | FlashAttention's tile update | the same rescaling applied to the output accumulator, so attention never stores the score matrix | Dao et al. (2022), algorithm 1 |
| scalar loops | PyTorch `softmax` CPU kernel | vectorized max, exp, and sum with SIMD, a vectorized polynomial exp | `aten/src/ATen/native/cpu/SoftMaxKernel.cpp` |
| one row per loop | llama.cpp `ggml_soft_max` | fused scale and mask (ALiBi, causal) in the same pass | `ggml/src/ggml-cpu/ops.cpp` |
| float32 sum | cuDNN and Triton fused softmax | one block per row, a tree reduction for the max and the sum | Triton tutorial "Fused Softmax" |

### Further reading

- Milakov and Gimelshein, "Online normalizer calculation for softmax" (2018).
- Dao et al., "FlashAttention" (2022), section 3.1.
- Higham, *Accuracy and Stability of Numerical Algorithms*, chapter 4 (summation).

## Key terms

- **shift invariance**: $\mathrm{softmax}(x) = \mathrm{softmax}(x - c)$ for any constant $c$.
- **normalizer**: the sum $s = \sum_j e^{x_j - m}$ that each exponential is divided by.
- **online softmax**: computing the maximum and the normalizer in one read by rescaling the running sum whenever the maximum grows.
- **rescale factor**: $e^{m_{\text{old}} - m_{\text{new}}}$, which converts a sum computed against the old maximum to one against the new.
- **fully masked row**: a row whose entries are all $-\infty$; by contract its softmax is all zeros.
- **unit roundoff**: $u = 2^{-24}$ for float32, the maximum relative error of one rounding.

## Summary

- Subtract the row maximum first. $\mathrm{softmax}(x) = \mathrm{softmax}(x - c)$ for any constant $c$, and with $c = \max_j x_j$ every exponent is at most 0, so rows of $\pm 10^4$ are exact instead of $\infty / \infty$.
- The online version needs one read of $x$ for the statistics. It keeps a running maximum $m$ and a running sum $s$, and rescales $s$ by $e^{m_{\text{old}} - m_{\text{new}}}$ whenever the maximum grows; the invariant "$s$ is the sum against the current $m$" holds after every element.
- Masked entries are $-\infty$ and get exactly 0; a row with every entry masked gives zeros, not $0/0$, and the online pass never evaluates $-\infty - (-\infty)$.
- Each row is a distribution: non-negative, summing to 1 within about $n \cdot 2^{-24}$ for $n$ columns, and the two versions agree element by element.
- The C kernels agree with the Python `softmax` across row lengths from 1 to 4096 and scales from 0.01 to $10^4$.

## Review questions

1. In the online pass, why is the rescale factor $e^{m_{\text{old}} - v}$ and not $e^{v - m_{\text{old}}}$?
   <details><summary>Answer</summary>Each old term is $e^{x_i - m_{\text{old}}}$ and must become $e^{x_i - v}$. The ratio is $e^{m_{\text{old}} - v}$, which is below 1 because $v > m_{\text{old}}$. The flipped sign would multiply the sum by a factor above 1 every time the maximum grows, and it would explode.</details>
2. Why does the online kernel skip $-\infty$ entries instead of letting the update handle them?
   <details><summary>Answer</summary>While $m$ is still $-\infty$, the else branch would compute `tl_expf(-inf - (-inf))`, which is $e^{\mathrm{NaN}}$, and the sum would become NaN for a perfectly valid row such as $[-\infty, 1]$. A $-\infty$ entry contributes 0 regardless, so skipping it is exact.</details>
3. How does the first finite element initialize the state without a special case?
   <details><summary>Answer</summary>With $m = -\infty$ and $s = 0$, any finite $v$ takes the `v > m` branch: $s = 0 \cdot e^{-\infty - v} + 1 = 1$, $m = v$. That is exactly the invariant for a one-element prefix.</details>
4. Why does the three-pass kernel track `fill` in pass 1?
   <details><summary>Answer</summary>A row whose only non-$-\infty$ entries are NaN keeps $m = -\infty$ (NaN never wins a comparison), so it would take the masked-row branch and become zeros, hiding an upstream bug. `fill` becomes NaN in that case, so the row is written as NaN.</details>
5. Why is the online version the basis of FlashAttention, though it does more exponentials?
   <details><summary>Answer</summary>Its state $(m, s)$ summarizes any prefix and merges with another block's state by the same rescaling. Attention can then process keys tile by tile, carrying $(m, s)$ and a rescaled output accumulator, without ever storing the full row of scores; the memory traffic saved outweighs the extra exponentials.</details>
6. Two blocks of a row are summarized by $(m_a, s_a) = (2, 1.5)$ and $(m_b, s_b) = (3, 1)$. What is the merged summary?
   <details><summary>Answer</summary>$m = \max(2, 3) = 3$ and $s = 1.5 \, e^{2 - 3} + 1 \cdot e^{0} = 1.5 \times 0.3678794 + 1 \approx 1.5518192$.</details>
