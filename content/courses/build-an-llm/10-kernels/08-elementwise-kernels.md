# Elementwise kernels in C: RMSNorm, RoPE, SiLU-mul, embedding, add, argmax

A transformer forward pass is mostly matmuls and attention, but between them sit six small operations: embedding lookup, normalization, rotary position encoding, gated activation, residual addition, and greedy selection. None of them is hard to write, and each is fast as long as it touches every element once. What makes them worth a section is that each has one detail that silently changes the model's output when you get it wrong: where $\varepsilon$ goes, which entries RoPE pairs, how SiLU behaves at large gates, and how argmax breaks ties. Get one wrong and the numbers still look plausible; the tokens are simply different.

This section implements all six as standalone C routines, with the Python definitions from earlier in the book as the behavioral reference. They complete the C kernel set of this chapter, whose point is that a C backend can generate the same tokens as numpy. The Python and Rust engine paths keep their own implementations; these routines are useful as independent, carefully specified examples.

The specifications are `RMSNorm` from [Pre-LN and RMSNorm](../08-modern-block/01-pre-ln-rmsnorm.md) and `rope_cos_sin` and `apply_rope` from [RoPE](../08-modern-block/03-rope.md); [Gated MLPs: SwiGLU and GeGLU](../08-modern-block/02-gated-mlps.md) is good background for SiLU-mul. The kernels call `tl_rsqrtf` from [Fast inverse square root in standalone C](../../mathematics-for-computing/10-numerical-methods-and-floating-point/05-fixed-point-iteration-and-rsqrt.md) and `tl_expf` from [Polynomial approximation and range reduction](../../mathematics-for-computing/10-numerical-methods-and-floating-point/06-polynomial-approximation-and-expf.md), both in *Mathematics for Computing*, and the shared support layer from [C runtime support and an arena allocator with marks](01-arena-allocator-with-marks.md). The contract is [`elementwise.h`](../../../../course/contracts/c/include/tinyllm/elementwise.h) under [`c/ABI.md`](../../../../course/contracts/c/ABI.md), and the reference solution is [`elementwise.c`](../../../../course/solutions/c/src/kernels/elementwise.c): `tl_rmsnorm_f32`, `tl_rope_f32` (half and interleaved layouts, partial rotary, attention scaling), `tl_silu_mul_f32`, `tl_embedding_f32`, `tl_add_f32`, and `tl_argmax_f32`. Try to predict each line of the solution before you read it.

> **Learning objectives**
> - Implement RMSNorm with $\varepsilon$ inside the root, in place, with each row's bits independent of the batch.
> - Implement RoPE for both the half and interleaved layouts, with partial rotary and attention scaling, reading each pair before writing it.
> - Write SiLU in the one form that saturates to the right limit at both ends without NaN.
> - Implement embedding, residual add, and an argmax whose tie and NaN rules match the sampling contract.

## RMSNorm

The section uses this notation:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x \in \mathbb{R}^{d}$ | one row (a token's hidden state) | `float[d]` |
| $w \in \mathbb{R}^{d}$ | the RMSNorm gain | `float[d]` |
| $\varepsilon$ | a small constant that keeps the root away from 0 | `float` |
| $\overline{x^2}$ | $\frac{1}{d}\sum_i x_i^2$, the mean of squares | scalar |
| $T, H, D$ | tokens, heads, head width of a RoPE input `[T, H, D]` | `int64_t` |
| $r$ | `d_rot`, the rotated width (even, $\le D$) | `int64_t` |
| $p_t$ | the absolute position of token $t$ | `int32_t` |
| $\omega_i$ | `inv_freq[i]`, the frequency of pair $i$, $0 \le i < r/2$ | `float[r/2]` |
| $\theta_{t,i} = p_t\,\omega_i$ | the rotation angle of pair $i$ at token $t$ | `float` |
| $\mu$ | `attn_scaling`, YaRN's factor on cos and sin (1 otherwise) | `float` |
| $\sigma(g)$ | the logistic sigmoid $1/(1 + e^{-g})$ | |

RMSNorm rescales a row to unit root mean square, then applies a learned gain:

$$y_i = \frac{x_i}{\sqrt{\overline{x^2} + \varepsilon}}\, w_i, \qquad \overline{x^2} = \frac{1}{d}\sum_{i=0}^{d-1} x_i^2 .$$

In C that is one pass to accumulate $\sum_i x_i^2$ in float32 in increasing $i$, one call to `tl_rsqrtf` (Newton's method) for $1/\sqrt{\overline{x^2} + \varepsilon}$, and one pass to write $x_i \cdot \text{inv} \cdot w_i$. Three properties follow. Because the order of the sum is fixed per row, a row's result is the same whether the call normalizes 1 row or 64; that is batch invariance, ABI rule 10. Because the sum is complete before the first write, `y == x` works. And $\varepsilon$ goes inside the root: a zero row then gives $0 \cdot (1/\sqrt{\varepsilon}) = 0$, never $0/0$, and Llama's checkpoints were trained that way.

## RoPE

RoPE rotates pairs of a query or key vector by an angle proportional to its position. Pair $i$ of token $t$ is rotated by $\theta_{t,i} = p_t \omega_i$:

$$\begin{pmatrix} a' \\ b' \end{pmatrix} = \mu \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} a \\ b \end{pmatrix} = \begin{pmatrix} \mu(a\cos\theta - b\sin\theta) \\ \mu(a\sin\theta + b\cos\theta) \end{pmatrix} .$$

A rotation keeps the pair's length (times $\mu$), and the dot product of a query rotated by $\theta_q$ and a key rotated by $\theta_k$ depends only on $\theta_q - \theta_k$, so attention sees relative position.

Three details of the contract decide whether you match a given checkpoint. The first is **the layout**, which says which entries form pair $i$. Layout 0 ("half", used by HF Llama and SmolLM2) pairs $(x_i, x_{i + r/2})$; layout 1 ("interleaved", used by Meta's code and the RoPE paper) pairs $(x_{2i}, x_{2i+1})$. Both are the same rotation on a permuted vector, which is why HF's conversion script permutes the rows of `q_proj` and `k_proj`. The second is **partial rotary**: only the first $r$ entries of each head rotate, and entries $r$ to $D - 1$ pass through. The third is **the angle**, which is formed once per (token, pair) in double precision and rounded to float. That equals float32$(p_t) \cdot$ float32$(\omega_i)$ for every position below $2^{24}$, the same angle the Python code computes. One angle serves all $H$ heads, so the loop order is token, pair, head.

Both members of a pair must be read before either is written; computing $b'$ from the new $a'$ is a different, and wrong, map.

## SiLU-mul

Llama's MLP computes $\mathrm{down}(\mathrm{silu}(\mathrm{gate}\,x) \odot \mathrm{up}\,x)$, and the middle step is elementwise:

$$y_i = \mathrm{silu}(g_i)\, u_i, \qquad \mathrm{silu}(g) = g\,\sigma(g) = \frac{g}{1 + e^{-g}} .$$

Written this way it is safe at both ends. For $g \to +\infty$, $e^{-g} \to 0$ and $y \to g u$. For $g \to -\infty$, $e^{-g}$ overflows to $+\infty$ (`tl_expf` saturates above 88.7), and $g / \infty = -0$, the right limit. The algebraically equal $g e^{g} / (1 + e^{g})$ computes $\infty / \infty = \mathrm{NaN}$ at $g = 100$.

## Embedding, add, and argmax

The remaining three are short. **Embedding** copies row `ids[t]` of a `[V, d]` table, so the source starts at element `ids[t] * d`. The contract makes the caller check ids against $V$, since the kernel cannot know $V$. **Add** is the residual connection, $y = a + b$, usually in place. **Argmax** picks the greedy token (temperature 0 in the sampling contract, `spec/sampling.md`): the largest value, the **lowest index** among equal values (a strict `>` while scanning forward), NaN entries skipped, and $-1$ when there is no number at all, so the caller reports an error instead of emitting token 0.

## Working it by hand

Each kernel has a small example that pins its definition.

For RMSNorm, take $x = [3, 4]$, $w = [1, 0.5]$, $\varepsilon = 0$. Then $\overline{x^2} = (9 + 16)/2 = 12.5$, $\sqrt{12.5} = 3.5355339$, and $\text{inv} = 0.2828427$, so $y = [3 \cdot 0.2828427 \cdot 1,\ 4 \cdot 0.2828427 \cdot 0.5] = [0.8485281, 0.5656854]$.

For RoPE in the half layout, take $x = [1, 2, 3, 4]$, $D = r = 4$, $\omega = [1, 0.01]$, position 1, and $\mu = 1$. The pairs are $(x_0, x_2) = (1, 3)$ at $\theta = 1$ and $(x_1, x_3) = (2, 4)$ at $\theta = 0.01$:

| Pair | $\cos\theta$ | $\sin\theta$ | $a' = a\cos - b\sin$ | $b' = a\sin + b\cos$ |
|---|---|---|---|---|
| $(1, 3)$, $\theta = 1$ | 0.5403023 | 0.8414710 | $0.5403023 - 2.5244130 = -1.9841107$ | $0.8414710 + 1.6209069 = 2.4623779$ |
| $(2, 4)$, $\theta = 0.01$ | 0.9999500 | 0.0099998 | $1.9999000 - 0.0399993 = 1.9599007$ | $0.0199997 + 3.9998000 = 4.0197997$ |

Written back to positions 0, 2 and 1, 3, the result is $[-1.9841107, 1.9599007, 2.4623779, 4.0197997]$. With the interleaved layout the pairs are $(1, 2)$ and $(3, 4)$, and the result is $[-1.1426397, 1.9220756, 2.9598507, 4.0297995]$.

For SiLU-mul, take gate $[0, 1, -1]$ and up $[5, 2, 3]$. Then $\mathrm{silu}(0) = 0$, $\mathrm{silu}(1) = 1/(1 + e^{-1}) = 0.7310586$, and $\mathrm{silu}(-1) = -1/(1 + e) = -0.2689414$; times up, $[0, 1.4621172, -0.8068243]$.

For argmax, take $[2, 7, 7, -1]$. The value 7 first appears at index 1, and index 2 is not strictly greater, so the answer is 1.

A correct implementation reproduces all four to float32 precision.

## The interface

```c
/* tinyllm/elementwise.h: void, the caller guarantees valid pointers and sizes */
void    tl_rmsnorm_f32(const float *x, const float *w, float *y, int64_t rows, int64_t d, float eps);
void    tl_rope_f32(float *x, const int32_t *pos, int64_t T, int64_t H, int64_t D, int64_t d_rot,
                    const float *inv_freq, float attn_scaling, int layout /* 0 half, 1 interleaved */);
void    tl_silu_mul_f32(const float *gate, const float *up, float *y, int64_t n);
void    tl_embedding_f32(const float *table, const int32_t *ids, float *out, int64_t n, int64_t d);
void    tl_add_f32(const float *a, const float *b, float *y, int64_t n);
int32_t tl_argmax_f32(const float *x, int64_t n);   /* lowest index on ties; NaN skipped; -1 if none */
```

From Python, declare each with `restype None` (or `c_int32` for argmax) on the loader.

## Building it step by step

Every kernel here reads each element once and does $O(1)$ work with it, so the only performance requirement is a single pass with no temporaries. Correctness is the hard part, because each kernel has one detail that silently changes the model.

RMSNorm needs the full sum of squares before any output, so it is two passes over one row: accumulate, then scale. Since the first pass finishes before the second writes, `y == x` is safe. Keeping the accumulator per row and summing in increasing $i$ makes each row independent of the batch, and putting $\varepsilon$ inside the root is the trained definition and makes a zero row give zeros. RoPE's angle depends on (token, pair), not on the head, so computing cos and sin once per (token, pair) and looping heads innermost does the transcendental work $T \cdot r/2$ times instead of $T \cdot H \cdot r/2$. The layout only changes which two indices form pair $i$; the rotation is the same. And a rotation is a map of the pair as a whole, so both entries must be read into locals before either is overwritten.

SiLU must be written in the one form whose overflow lands on the right limit, $g / (1 + e^{-g})$: at $g \to -\infty$ the denominator becomes $+\infty$ and the quotient $-0$. Argmax's tie rule and NaN rule are part of the sampling contract, so that greedy outputs match Python and Rust exactly: strict `>`, skip NaN, $-1$ when nothing is a number.

### RMSNorm

The accumulator `ss` is declared inside the row loop, the division by $d$ happens before adding $\varepsilon$, and `tl_rsqrtf` gives the reciprocal root directly:

```c
void tl_rmsnorm_f32(const float *x, const float *w, float *y, int64_t rows, int64_t d, float eps) {
    if (rows <= 0 || d <= 0) return;
    for (int64_t r = 0; r < rows; r++) {
        const float *xr = x + r * d;
        float *yr = y + r * d;
        float ss = 0.0f;                                 /* per row: no batch dependence */
        for (int64_t i = 0; i < d; i++) ss += xr[i] * xr[i];   /* fixed order, increasing i */
        float inv = tl_rsqrtf(ss / (float)d + eps);      /* mean of squares, eps INSIDE the root */
        for (int64_t i = 0; i < d; i++) yr[i] = xr[i] * inv * w[i]; /* safe when y == x */
    }
}
```

On the worked example's row, `ss = 25`, `ss / d = 12.5`, and `inv = 0.2828427`.

### RoPE

The angle is formed in double and rounded once, so it equals the float32 product Python computes; `attn_scaling` is folded into `c` and `s`; and the layout picks the pair indices:

```c
    const int64_t half = d_rot / 2;                       /* pairs live inside d_rot only */
    for (int64_t t = 0; t < T; t++) {
        for (int64_t i = 0; i < half; i++) {
            float angle = (float)((double)pos[t] * (double)inv_freq[i]); /* one rounding */
            float c = (float)cos((double)angle) * attn_scaling;           /* mu folded in */
            float s = (float)sin((double)angle) * attn_scaling;
            for (int64_t h = 0; h < H; h++) {                 /* one angle, every head */
                float *xh = x + (t * H + h) * D;              /* [T, H, D] */
                int64_t ia = layout ? 2 * i : i;              /* interleaved: (2i, 2i+1) */
                int64_t ib = layout ? 2 * i + 1 : i + half;   /* half: (i, i + r/2) */
                float a = xh[ia], b = xh[ib];                 /* both read before either write */
                xh[ia] = a * c - b * s;
                xh[ib] = a * s + b * c;
            }
        }
    }
```

Entries from `d_rot` to `D - 1` are never indexed, because the largest index is `d_rot - 1` in both layouts. The partial-rotary tail therefore passes through untouched, and `inv_freq` is read only for `i < d_rot / 2`.

### SiLU-mul, embedding, and add

These are one line each; what matters is the form of the SiLU and the row offset in the embedding:

```c
        y[i] = g / (1.0f + tl_expf(-g)) * up[i];  /* g -> -inf: g / inf = -0, never NaN */

        memcpy(out + t * d, table + (int64_t)ids[t] * d, (size_t)d * sizeof(float)); /* row ids[t] */

    for (int64_t i = 0; i < n; i++) y[i] = a[i] + b[i];   /* y may be a: element i read first */
```

The cast `(int64_t)ids[t]` before multiplying matters: `ids[t] * d` in 32 bits would overflow for large vocabularies times width.

### Argmax

`best` starts at $-1$ (no number seen), NaN is skipped before comparing, and the comparison is strict:

```c
int32_t tl_argmax_f32(const float *x, int64_t n) {
    int32_t best = -1;                       /* -1: empty or all NaN, caller reports the error */
    for (int64_t i = 0; i < n; i++) {
        if (isnan(x[i])) continue;           /* a NaN never wins, even in first place */
        if (best < 0 || x[i] > x[best]) best = (int32_t)i;  /* strict >: ties to lowest index */
    }
    return best;
}
```

A row of all $-\infty$ returns 0: $-\infty$ is a number, so the first entry becomes `best` and nothing is strictly greater. The full solution is [`elementwise.c`](../../../../course/solutions/c/src/kernels/elementwise.c).

### Why this design

Every kernel is $O(n)$ time and $O(1)$ extra space. RMSNorm reads each row twice, which is cheap because the row fits in cache, and RoPE does $T \cdot r/2$ cos and sin evaluations.

Each choice has a rejected alternative. $x / (\sqrt{\overline{x^2}} + \varepsilon)$ agrees on ordinary rows, but on tiny rows the result differs from what the model was trained with, and the zero-row case is no longer the clean $0 \cdot \varepsilon^{-1/2}$. A precomputed cos/sin table per position is what production does and saves the transcendentals, but it needs memory sized by the maximum context and a cache policy; here the angle per (token, pair) is cheap relative to the head loop, and computing it directly keeps the kernel stateless. Computing `sigmoid(g)` as `1 / (1 + e^{-g})` and then multiplying by `g` is also safe, but the algebraically equal $g e^g / (1 + e^g)$ is not ($\infty / \infty$ at $g = 100$); the chosen form has one division and a correct limit at both ends. And argmax returns $-1$ rather than 0 on an empty or all-NaN row because token 0 is a real token, and returning it would hide a numerical failure upstream as a plausible output.

## What can go wrong

The tests pin each kernel's one critical detail:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand examples | unit | the worked examples for RMSNorm, RoPE, SiLU-mul, argmax | the definitions are the agreed ones |
| $\varepsilon$ inside the root | boundary | $[10^{-3}, 10^{-3}]$ with $\varepsilon = 10^{-6}$ gives 0.7071; a zero row gives zeros | tiny-norm rows, Llama's trained form |
| independent, in-place RMSNorm rows | property | every row alone equals the row in a batch of up to 12, bit for bit; `y == x` | batched decode in [continuous batching](../11-serving/03-continuous-batching.md) |
| interleaved RoPE, position 0 | unit | the interleaved worked example; position 0 is the identity | loading Meta-layout weights |
| partial rotary and scaling | unit | the tail past `d_rot` untouched; $\mu = 2$ doubles the pair; `[T, H, D]` indexing | GPT-NeoX style partial rotary, YaRN |
| RoPE is a rotation | property | every pair keeps its length, random shapes and positions, both layouts | no half-updated pairs, no wrong partners |
| SiLU saturates without NaN | boundary | gates $\pm 100$, $-10^4$ give the right limits | large activations in trained MLPs |
| embedding gathers rows | unit | repeated and out-of-order ids | the first op of the forward |
| add in place | unit | $y = a + b$ with `y == a` | the residual stream |
| argmax ties, NaN, empty | boundary | ties to the lowest index, NaN skipped, $-1$ for all NaN or empty, all $-\infty$ gives 0 | greedy decoding parity |

RMSNorm has four classic mistakes. Putting $\varepsilon$ outside the root, $x / (\sqrt{\overline{x^2}} + \varepsilon)$, is correct on ordinary rows and wrong on tiny ones, so it drifts from the trained model; use `tl_rsqrtf(ss / d + eps)`. Not dividing the sum of squares by $d$ shrinks every output by $\sqrt{d}$; write `ss / (float)d`. Forgetting the gain $w$ gives the right shape at the wrong scale; multiply by `w[i]`. Declaring the sum of squares once per call instead of per row normalizes row 2 by rows 1 and 2 together, so results depend on the batch; declare `float ss = 0.0f` inside the row loop.

> **Warning:** Swapping the RoPE layouts gives plausible numbers and the wrong attention pattern on HF weights. Layout 0 pairs $(i, i + r/2)$; layout 1 pairs $(2i, 2i+1)$.

RoPE has more. Rotating by $-\theta$ (a sign flipped) mirrors relative positions and degrades the model's outputs; the solution computes $a' = ac - bs$, $b' = as + bc$. Computing $b'$ from the new $a'$ changes lengths, so it is not a rotation; read `a` and `b` into locals first. Rotating all $D$ entries when `d_rot < D` scrambles the pass-through tail and reads `inv_freq` past its end; index pairs only within `d_rot`. Ignoring `attn_scaling` makes YaRN-extended models lose their temperature correction; fold it into `c` and `s`.

The last group covers the other kernels. Writing SiLU as a plain sigmoid, or with $e^{g}$ instead of $e^{-g}$, gives the wrong MLP, or $0$ instead of $g u$ at large gates; write `g / (1 + tl_expf(-g)) * up`. Reading the embedding row at `ids[t]` instead of `ids[t] * d` makes every token read a slice of row 0 or 1; use `table + (int64_t)ids[t] * d`. An argmax with `>=` sends ties to the last index, so greedy output differs from Python; use strict `>`. An argmax without the NaN check lets a NaN in first place win forever; skip with `isnan` before comparing. And returning 0 for an empty or all-NaN row makes the engine emit token 0 instead of reporting the error; start `best` at $-1$.

## Where this leads

These kernels rest on the shared status, error, and allocator support from [the runtime support layer](01-arena-allocator-with-marks.md), on `tl_rsqrtf` for the $1/\sqrt{\cdot}$ of RMSNorm and `tl_expf` for the $e^{-g}$ of SiLU from *Mathematics for Computing*, and on the Python specifications in [Pre-LN and RMSNorm](../08-modern-block/01-pre-ln-rmsnorm.md) and [RoPE](../08-modern-block/03-rope.md). Looking forward, the Rust forward pass in the serving part of the book calls the same six operations in its own independent Rust implementation, starting with the [model runner](../11-serving/02-model-runner-and-sampler.md).

Production systems fuse these operations to save memory traffic:

| This solution | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| separate RMSNorm and add | fused add + RMSNorm (vLLM `fused_add_rms_norm`) | one pass that adds the residual and normalizes, halving memory traffic | vLLM `csrc/layernorm_kernels.cu` |
| RoPE with `cos` per call | cached cos/sin tables and fused QK rotation | precomputed tables per position, rotation fused into the QKV projection epilogue | llama.cpp `ggml_rope_ext`; FlashInfer `rope.cuh` |
| scalar SiLU | vectorized SwiGLU with a polynomial exp | SIMD over 8 or 16 lanes | ggml `ggml_vec_swiglu_f32` |
| argmax | fused sampling kernels | argmax and top-k inside the final matmul's epilogue | FlashInfer `sampling.cuh` |

### Further reading

- Zhang and Sennrich, "Root Mean Square Layer Normalization" (2019).
- Su et al., "RoFormer" (2021), section 3.4.
- Shazeer, "GLU Variants Improve Transformer" (2020).

## Key terms

- **elementwise kernel**: a kernel that reads each element once and does constant work with it, so it is bound by memory bandwidth.
- **RMSNorm**: normalization by $1/\sqrt{\overline{x^2} + \varepsilon}$ followed by a learned per-channel gain.
- **RoPE layout**: the rule for which two entries of a head form a rotated pair: half $(i, i + r/2)$ or interleaved $(2i, 2i+1)$.
- **partial rotary**: rotating only the first $r$ entries of each head and passing the rest through.
- **SiLU**: $g\,\sigma(g) = g / (1 + e^{-g})$, the activation in SwiGLU.
- **greedy argmax**: selecting the largest logit, with ties broken to the lowest index and NaN skipped.

## Summary

- Each kernel touches every element once, so it is bound by memory bandwidth, not arithmetic: one pass, no temporaries, and outputs that may alias their first input.
- RMSNorm puts $\varepsilon$ inside the root, $x / \sqrt{\overline{x^2} + \varepsilon} \cdot w$, and its sum of squares runs in a fixed order per row, so a row's bits never depend on its batch.
- RoPE is a rotation of pairs, and which entries form a pair is the layout: $(x_i, x_{i + r/2})$ for HF Llama, $(x_{2i}, x_{2i+1})$ for Meta's code; both members are read before either is written.
- SiLU saturates without NaN when written $g / (1 + e^{-g})$: at $g = -10^4$ the exponential overflows to $+\infty$ and the quotient is $-0$.
- Greedy argmax breaks ties to the lowest index and skips NaN, the rule the Python sampler and the Rust engine share.

## Review questions

1. Why is `y == x` safe for RMSNorm but would not be for a one-pass "running" normalization?
   <details><summary>Answer</summary>The first loop finishes the sum of squares before the second loop writes anything, so overwriting $x_i$ cannot change `inv`. A one-pass scheme that wrote outputs while still accumulating would read some already-normalized values.</details>
2. Why does the solution loop over heads innermost in RoPE?
   <details><summary>Answer</summary>The angle $p_t \omega_i$ does not depend on the head, so cos and sin are computed once per (token, pair) and reused across all $H$ heads.</details>
3. What map do you get if $b'$ is computed from the new $a'$?
   <details><summary>Answer</summary>$b' = a'\sin\theta + b\cos\theta$, which is not the rotation: the pair's length changes and the dot products no longer depend only on the position difference.</details>
4. Why is SiLU at $g = -10^4$ equal to $-0$ and not NaN in this code?
   <details><summary>Answer</summary>`tl_expf(10^4)` saturates to $+\infty$, so the denominator is $+\infty$ and $g / \infty = -0$. Then $-0 \cdot u$ is a signed zero, which is the correct limit of $g\,\sigma(g)$.</details>
5. Why must argmax use strict `>`?
   <details><summary>Answer</summary>Scanning forward, `>=` would let a later equal value take over, sending ties to the highest index. The sampling contract says lowest index, and every engine must agree to produce the same greedy tokens.</details>
6. In the RMSNorm test with $x = [10^{-3}, 10^{-3}]$ and $\varepsilon = 10^{-6}$, why is each output (with $w = 1$) about 0.7071 rather than 1?
   <details><summary>Answer</summary>$\overline{x^2} = 10^{-6}$, so the root is $\sqrt{10^{-6} + 10^{-6}} = \sqrt{2} \times 10^{-3}$ and $y = 10^{-3} / (\sqrt{2} \times 10^{-3}) = 1/\sqrt{2} \approx 0.7071$. With $\varepsilon$ outside the root the answer would be $10^{-3}/(10^{-3} + 10^{-6}) \approx 0.999$, which is why this test distinguishes the two forms.</details>
