# Elementwise kernels in C: RMSNorm, RoPE, SiLU-mul, embedding, add, argmax

## Overview

| | |
|---|---|
| **Module** | `L9.6` · side · C · Pass 6 · 3 to 4 h |
| **Solution** | [`c/src/kernels/elementwise.c`](../../../../course/solutions/c/src/kernels/elementwise.c): `tl_rmsnorm_f32`, `tl_rope_f32` (half and interleaved layouts, partial rotary, attention scaling), `tl_silu_mul_f32`, `tl_embedding_f32`, `tl_add_f32`, `tl_argmax_f32` |
| **Contract** | [`course/contracts/c/include/tinyllm/elementwise.h`](../../../../course/contracts/c/include/tinyllm/elementwise.h) · rules: [`c/ABI.md`](../../../../course/contracts/c/ABI.md) |
| **Needs** | `rt.02` the loader · `M09.5` `tl_rsqrtf` · `M09.6` `tl_expf` · `L7.1` `RMSNorm` · `L7.3` `rope_cos_sin` and `apply_rope`. Reading: `L7.2` (SwiGLU) |
| **Used by** | These standalone C routines are useful as independent examples; the Python and Rust engine paths retain their own implementations. With `L9.3` to `L9.5` they complete the C kernel set of Pass 6, whose point is that a C backend generates the same tokens as numpy |
| **Optional depth** | Zhang and Sennrich, "Root Mean Square Layer Normalization" (2019); Su et al., "RoFormer" (2021), section 3.4; Shazeer, "GLU Variants Improve Transformer" (2020) |

## Key Takeaways

- **Each kernel touches every element once,** so it is bound by memory bandwidth, not arithmetic: one pass, no temporaries, and outputs that may alias their first input.
- **RMSNorm puts $\varepsilon$ inside the root,** $x / \sqrt{\overline{x^2} + \varepsilon} \cdot w$, and its sum of squares runs in a fixed order per row, so a row's bits never depend on its batch.
- **RoPE is a rotation of pairs,** and which entries form a pair is the layout: $(x_i, x_{i + r/2})$ for HF Llama, $(x_{2i}, x_{2i+1})$ for Meta's code; both members are read before either is written.
- **SiLU saturates without NaN** when written $g / (1 + e^{-g})$: at $g = -10^4$ the exponential overflows to $+\infty$ and the quotient is $-0$.
- **Greedy argmax breaks ties to the lowest index and skips NaN,** the rule the Python sampler and the Rust engine share.

## How to read this chapter

Sections 1 to 3 state each operation, the one detail of each that silently changes the model, and work all four by hand. Section 4 fixes the C interface and lists what a correct implementation must handle. Section 5 walks through the solution, [`elementwise.c`](../../../../course/solutions/c/src/kernels/elementwise.c), kernel by kernel. Try to predict each line of the solution before reading it.

---

## 1. Why now

Elementwise and row-wise operations include embedding lookup, normalization, rotary position encoding, gated activation, residual addition, and greedy selection. Each operation has details that silently change model output when implemented incorrectly: where $\varepsilon$ goes, which entries RoPE pairs, how SiLU behaves at large gates, and how argmax breaks ties. This optional module implements the C versions as standalone routines, with the Python definitions serving as the behavioral reference.

## 2. Principles

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

### 2.1 RMSNorm

RMSNorm (`L7.1`) rescales a row to unit root mean square, then applies a learned gain:

$$y_i = \frac{x_i}{\sqrt{\overline{x^2} + \varepsilon}}\, w_i, \qquad \overline{x^2} = \frac{1}{d}\sum_{i=0}^{d-1} x_i^2 .$$

In C that is one pass to accumulate $\sum_i x_i^2$ in float32 in increasing $i$, one call to `tl_rsqrtf` (`M09.5`, Newton's method) for $1/\sqrt{\overline{x^2} + \varepsilon}$, and one pass to write $x_i \cdot \text{inv} \cdot w_i$. Because the order of the sum is fixed per row, a row's result is the same whether the call normalizes 1 row or 64 (batch invariance, `c/ABI.md` rule 10). Because the sum is complete before the first write, `y == x` works. $\varepsilon$ goes inside the root: a zero row then gives $0 \cdot (1/\sqrt{\varepsilon}) = 0$, never $0/0$, and Llama's checkpoints were trained that way.

### 2.2 RoPE

RoPE (`L7.3`) rotates pairs of a query or key vector by an angle proportional to its position. Pair $i$ of token $t$ is rotated by $\theta_{t,i} = p_t \omega_i$:

$$\begin{pmatrix} a' \\ b' \end{pmatrix} = \mu \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} a \\ b \end{pmatrix} = \begin{pmatrix} \mu(a\cos\theta - b\sin\theta) \\ \mu(a\sin\theta + b\cos\theta) \end{pmatrix} .$$

A rotation keeps the pair's length (times $\mu$), and the dot product of a query rotated by $\theta_q$ and a key rotated by $\theta_k$ depends only on $\theta_q - \theta_k$: attention sees relative position. Three details of the contract:

- **The layout** says which entries form pair $i$: `layout 0` ("half", HF Llama, SmolLM2) pairs $(x_i, x_{i + r/2})$; `layout 1` ("interleaved", Meta's code, the RoPE paper) pairs $(x_{2i}, x_{2i+1})$. Both are the same rotation on a permuted vector, which is why HF's conversion script permutes the rows of `q_proj` and `k_proj`.
- **Partial rotary**: only the first $r$ entries of each head rotate; entries $r$ to $D - 1$ pass through.
- **The angle** is formed once per (token, pair) in double precision and rounded to float, which equals float32$(p_t) \cdot$ float32$(\omega_i)$ for every position below $2^{24}$: the same angle the Python code computes. One angle serves all $H$ heads, so the loop order is token, pair, head.

Both members of a pair must be read before either is written; computing $b'$ from the new $a'$ is a different (wrong) map.

### 2.3 SiLU-mul (SwiGLU)

Llama's MLP (`L7.2`) computes $\mathrm{down}(\mathrm{silu}(\mathrm{gate}\,x) \odot \mathrm{up}\,x)$. The middle step is elementwise:

$$y_i = \mathrm{silu}(g_i)\, u_i, \qquad \mathrm{silu}(g) = g\,\sigma(g) = \frac{g}{1 + e^{-g}} .$$

Written this way it is safe at both ends. For $g \to +\infty$, $e^{-g} \to 0$ and $y \to g u$. For $g \to -\infty$, $e^{-g}$ overflows to $+\infty$ (`tl_expf` saturates above 88.7), and $g / \infty = -0$, the right limit. The algebraically equal $g e^{g} / (1 + e^{g})$ computes $\infty / \infty = \mathrm{NaN}$ at $g = 100$.

### 2.4 Embedding, add, argmax

**Embedding** copies row `ids[t]` of a `[V, d]` table: the source starts at element `ids[t] * d`. The contract makes the caller check ids against $V$ (the kernel cannot know $V$). **Add** is the residual connection, $y = a + b$, usually in place. **Argmax** picks the greedy token (`spec/sampling.md`, temperature 0): the largest value, the **lowest index** among equal values (a strict `>` while scanning forward), NaN entries skipped, and $-1$ when there is no number at all, so the caller reports an error instead of emitting token 0.

## 3. Worked example by hand

**RMSNorm** of $x = [3, 4]$, $w = [1, 0.5]$, $\varepsilon = 0$: $\overline{x^2} = (9 + 16)/2 = 12.5$, $\sqrt{12.5} = 3.5355339$, $\text{inv} = 0.2828427$, so $y = [3 \cdot 0.2828427 \cdot 1,\ 4 \cdot 0.2828427 \cdot 0.5] = [0.8485281, 0.5656854]$.

**RoPE** (half layout) of $x = [1, 2, 3, 4]$, $D = r = 4$, $\omega = [1, 0.01]$, position 1, $\mu = 1$. Pairs are $(x_0, x_2) = (1, 3)$ at $\theta = 1$ and $(x_1, x_3) = (2, 4)$ at $\theta = 0.01$:

| Pair | $\cos\theta$ | $\sin\theta$ | $a' = a\cos - b\sin$ | $b' = a\sin + b\cos$ |
|---|---|---|---|---|
| $(1, 3)$, $\theta = 1$ | 0.5403023 | 0.8414710 | $0.5403023 - 2.5244130 = -1.9841107$ | $0.8414710 + 1.6209069 = 2.4623779$ |
| $(2, 4)$, $\theta = 0.01$ | 0.9999500 | 0.0099998 | $1.9999000 - 0.0399993 = 1.9599007$ | $0.0199997 + 3.9998000 = 4.0197997$ |

Written back to positions 0, 2 and 1, 3: $[-1.9841107, 1.9599007, 2.4623779, 4.0197997]$. With the interleaved layout the pairs are $(1, 2)$ and $(3, 4)$ and the result is $[-1.1426397, 1.9220756, 2.9598507, 4.0297995]$.

**SiLU-mul** of gate $[0, 1, -1]$ and up $[5, 2, 3]$: $\mathrm{silu}(0) = 0$, $\mathrm{silu}(1) = 1/(1 + e^{-1}) = 0.7310586$, $\mathrm{silu}(-1) = -1/(1 + e) = -0.2689414$; times up: $[0, 1.4621172, -0.8068243]$.

**Argmax** of $[2, 7, 7, -1]$: 7 first appears at index 1, and index 2 is not strictly greater, so the answer is 1.

A correct implementation reproduces all four to float32 precision.

## 4. The interface

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

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand examples | unit | section 3 for RMSNorm, RoPE, SiLU-mul, argmax | the definitions are the agreed ones |
| $\varepsilon$ inside the root | boundary | $[10^{-3}, 10^{-3}]$ with $\varepsilon = 10^{-6}$ gives 0.7071; a zero row gives zeros | tiny-norm rows, Llama's trained form |
| independent, in-place RMSNorm rows | property | every row alone equals the row in a batch of up to 12, bit for bit; `y == x` | batched decode in `L10.2` |
| interleaved RoPE, position 0 | unit | the interleaved worked example; position 0 is the identity | loading Meta-layout weights |
| partial rotary and scaling | unit | the tail past `d_rot` untouched; $\mu = 2$ doubles the pair; `[T, H, D]` indexing | GPT-NeoX style partial rotary, YaRN |
| RoPE is a rotation | property | every pair keeps its length, random shapes and positions, both layouts | no half-updated pairs, no wrong partners |
| SiLU saturates without NaN | boundary | gates $\pm 100$, $-10^4$ give the right limits | large activations in trained MLPs |
| embedding gathers rows | unit | repeated and out-of-order ids | the first op of the forward |
| add in place | unit | $y = a + b$ with `y == a` | the residual stream |
| argmax ties, NaN, empty | boundary | ties to the lowest index, NaN skipped, $-1$ for all NaN or empty, all $-\infty$ gives 0 | greedy decoding parity |

## 5. The solution, step by step

### 5.1 From first principles

1. **Every kernel here reads each element once and does O(1) work with it**, so the only performance requirement is a single pass with no temporaries. Correctness is the hard part: each kernel has one detail that silently changes the model.
2. **RMSNorm needs the full sum of squares before any output**, so it is two passes over one row: accumulate, then scale. Since the first pass finishes before the second writes, `y == x` is safe. Keeping the accumulator per row and summing in increasing $i$ makes each row independent of the batch. Putting $\varepsilon$ inside the root is the trained definition and makes a zero row give zeros.
3. **RoPE's angle depends on (token, pair), not on the head**, so computing cos and sin once per (token, pair) and looping heads innermost does the transcendental work $T \cdot r/2$ times instead of $T \cdot H \cdot r/2$. The layout only changes which two indices form pair $i$; the rotation is the same.
4. **A rotation is a map of the pair as a whole**, so both entries must be read into locals before either is overwritten.
5. **SiLU must be written in the one form whose overflow lands on the right limit**: $g / (1 + e^{-g})$. At $g \to -\infty$ the denominator becomes $+\infty$ and the quotient $-0$.
6. **Argmax's tie rule and NaN rule are part of the sampling contract**, so greedy outputs match Python and Rust exactly: strict `>`, skip NaN, $-1$ when nothing is a number.

### 5.2 The code, annotated

RMSNorm: the accumulator `ss` is declared inside the row loop, the division by $d$ happens before adding $\varepsilon$, and `tl_rsqrtf` gives the reciprocal root directly.

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

On section 3's row, `ss = 25`, `ss / d = 12.5`, and `inv = 0.2828427`.

RoPE: the angle is formed in double and rounded once, so it equals the float32 product Python computes; `attn_scaling` is folded into `c` and `s`; the layout picks the pair indices.

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

Entries from `d_rot` to `D - 1` are never indexed (the largest index is `d_rot - 1` in both layouts), so the partial-rotary tail passes through untouched and `inv_freq` is read only for `i < d_rot / 2`.

SiLU-mul, embedding, and add are one line each; what matters is the form of the SiLU and the row offset in the embedding.

```c
        y[i] = g / (1.0f + tl_expf(-g)) * up[i];  /* g -> -inf: g / inf = -0, never NaN */

        memcpy(out + t * d, table + (int64_t)ids[t] * d, (size_t)d * sizeof(float)); /* row ids[t] */

    for (int64_t i = 0; i < n; i++) y[i] = a[i] + b[i];   /* y may be a: element i read first */
```

The cast `(int64_t)ids[t]` before multiplying matters: `ids[t] * d` in 32 bits would overflow for large vocabularies times width.

Argmax: `best` starts at $-1$ (no number seen), NaN is skipped before comparing, and the comparison is strict.

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

All $-\infty$ returns 0: $-\infty$ is a number, so the first entry becomes `best` and nothing is strictly greater.

Full solution: [`elementwise.c`](../../../../course/solutions/c/src/kernels/elementwise.c).

### 5.3 Why this solution

- **Cost.** Every kernel is $O(n)$ time and $O(1)$ extra space; RMSNorm reads each row twice (it fits in cache), RoPE does $T \cdot r/2$ cos and sin evaluations.
- **Why not $x / (\sqrt{\overline{x^2}} + \varepsilon)$?** It agrees on ordinary rows, but on tiny rows the result differs from what the model was trained with, and the zero row case is no longer the clean $0 \cdot \varepsilon^{-1/2}$.
- **Why not precompute a cos/sin table?** A table per position is what production does (see Going further) and saves the transcendentals, but it needs memory sized by the maximum context and a cache policy. Here the angle per (token, pair) is cheap relative to the head loop, and computing it directly keeps the kernel stateless.
- **Why not `sigmoid(g) * g` via `1 / (1 + e^{-g})` then multiply?** That form is also safe, but the algebraically equal $g e^g / (1 + e^g)$ is not ($\infty / \infty$ at $g = 100$). The chosen form has one division and a correct limit at both ends.
- **Why return $-1$ instead of 0 from argmax on an empty or all-NaN row?** Token 0 is a real token; returning it would hide a numerical failure upstream as a plausible output.

### 5.4 Explain it back

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

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| $\varepsilon$ outside the root, $x / (\sqrt{\overline{x^2}} + \varepsilon)$ | correct on ordinary rows, wrong on tiny ones; drifts from the trained model | `tl_rsqrtf(ss / d + eps)` |
| the sum of squares not divided by $d$ | every output shrinks by $\sqrt{d}$ | `ss / (float)d` |
| forgetting the gain $w$ | right shape, wrong scale | `xr[i] * inv * w[i]` |
| the sum of squares declared once per call, not per row | row 2 is normalized by rows 1 and 2 together; results depend on the batch | `float ss = 0.0f` inside the row loop |
| the layouts swapped | plausible numbers, wrong attention pattern on HF weights | `layout 0` pairs $(i, i + r/2)$, `layout 1` pairs $(2i, 2i+1)$ |
| rotating by $-\theta$ (a sign flipped) | relative positions mirrored; the model's outputs degrade | $a' = ac - bs$, $b' = as + bc$ |
| computing $b'$ from the new $a'$ | lengths change; not a rotation | `a` and `b` read into locals first |
| rotating all $D$ entries when `d_rot < D` | the pass-through tail is scrambled (and `inv_freq` is read past its end) | pairs indexed only within `d_rot` |
| ignoring `attn_scaling` | YaRN-extended models lose their temperature correction | folded into `c` and `s` |
| SiLU as a sigmoid, or with $e^{g}$ instead of $e^{-g}$ | wrong MLP; or $0$ instead of $g u$ at large gates | `g / (1 + tl_expf(-g)) * up` |
| embedding row at `ids[t]` instead of `ids[t] * d` | every token reads a slice of row 0 or 1 | `table + (int64_t)ids[t] * d` |
| argmax with `>=` | ties go to the last index; greedy output differs from Python | strict `>` |
| argmax without the NaN check | a NaN in the first place wins forever | `isnan` skip before comparing |
| returning 0 for an empty or all-NaN row | the engine emits token 0 instead of reporting the error | `best` starts at $-1$ |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `rt.02` | the shared status, error, and allocator support |
| Back | `M09.5` | `tl_rsqrtf`: the $1/\sqrt{\cdot}$ of RMSNorm |
| Back | `M09.6` | `tl_expf`: the $e^{-g}$ of SiLU |
| Back | `L7.1` | `RMSNorm`, the specification of `tl_rmsnorm_f32` |
| Back | `L7.3` | `rope_cos_sin` and `apply_rope`, the specification of `tl_rope_f32` |
| Forward | the standalone Rust engine | the Rust forward calls the same six kernels in its independent Rust implementation |

## Going further

| This solution | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| separate RMSNorm and add | fused add + RMSNorm (vLLM `fused_add_rms_norm`) | one pass that adds the residual and normalizes, halving memory traffic | vLLM `csrc/layernorm_kernels.cu` |
| RoPE with `cos` per call | cached cos/sin tables and fused QK rotation | precomputed tables per position, rotation fused into the QKV projection epilogue | llama.cpp `ggml_rope_ext`; FlashInfer `rope.cuh` |
| scalar SiLU | vectorized SwiGLU with a polynomial exp | SIMD over 8 or 16 lanes | ggml `ggml_vec_swiglu_f32` |
| argmax | fused sampling kernels | argmax and top-k inside the final matmul's epilogue | FlashInfer `sampling.cuh` |
