<!-- ol:module L9.6 -->
# Elementwise kernels in C: RMSNorm, RoPE, SiLU-mul, embedding, add, argmax

## Overview

| | |
|---|---|
| **Module** | `L9.6` · side · C · Pass 6 · 3 to 4 h |
| **You build** | `c/src/kernels/elementwise.c`: `tl_rmsnorm_f32`, `tl_rope_f32` (half and interleaved layouts, partial rotary, attention scaling), `tl_silu_mul_f32`, `tl_embedding_f32`, `tl_add_f32`, `tl_argmax_f32` |
| **Contract** | [`course/contracts/c/include/tinyllm/elementwise.h`](../../../course/contracts/c/include/tinyllm/elementwise.h) · rules: [`c/ABI.md`](../../../course/contracts/c/ABI.md) |
| **Tests** | `course/tests/L9.6/`: `test_elementwise.c` (C, under ASan and UBSan) and shared file fixtures (what they check: section 4) |
| **Needs** | `rt.02` the loader · `M09.5` your `tl_rsqrtf` · `M09.6` your `tl_expf` · `L7.1` your `RMSNorm` · `L7.3` your `rope_cos_sin` and `apply_rope` (or `--ref-deps`). Reading: `L7.2` (SwiGLU) |
| **Used by** | These standalone C routines are useful as independent examples; the Python and Rust engine paths retain their own implementations. |
| **Milestone** | `MS-L9` (the C backend generates the same tokens as numpy) |
| **Optional depth** | Zhang and Sennrich, "Root Mean Square Layer Normalization" (2019); Su et al., "RoFormer" (2021), section 3.4; Shazeer, "GLU Variants Improve Transformer" (2020) |

## Key Takeaways

- **Each kernel touches every element once,** so it is bound by memory bandwidth, not arithmetic: one pass, no temporaries, and outputs that may alias their first input (`rmsnorm_rows_are_independent_and_in_place`, `add_in_place`).
- **RMSNorm puts $\varepsilon$ inside the root,** $x / \sqrt{\overline{x^2} + \varepsilon} \cdot w$, and its sum of squares runs in a fixed order per row, so a row's bits never depend on its batch (`rmsnorm_eps_is_inside_the_root`).
- **RoPE is a rotation of pairs,** and which entries form a pair is the layout: $(x_i, x_{i + r/2})$ for HF Llama, $(x_{2i}, x_{2i+1})$ for Meta's code; both members are read before either is written (`rope_interleaved_layout_and_position_zero`, `rope_is_a_rotation`).
- **SiLU saturates without NaN** when written $g / (1 + e^{-g})$: at $g = -10^4$ the exponential overflows to $+\infty$ and the quotient is $-0$ (`silu_mul_saturates_without_nan`).
- **Greedy argmax breaks ties to the lowest index and skips NaN,** the rule the Python sampler and the Rust engine share (`argmax_ties_nan_and_empty`).

## How to work this chapter

```bash
ol start L9.6              # stubs c/src/kernels/elementwise.c into your repo
ol tests L9.6              # read the test catalog first
ol check L9.6              # exit code is the verdict
ol check L9.6 --ref-deps   # only if rt.02, M09.5, M09.6, L7.1, or L7.3 is not passing yet
ol diff  L9.6              # after passing: your code against the reference
```

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

In C that is one pass to accumulate $\sum_i x_i^2$ in float32 in increasing $i$, one call to your `tl_rsqrtf` (`M09.5`, Newton's method) for $1/\sqrt{\overline{x^2} + \varepsilon}$, and one pass to write $x_i \cdot \text{inv} \cdot w_i$. Because the order of the sum is fixed per row, a row's result is the same whether the call normalizes 1 row or 64 (batch invariance, `c/ABI.md` rule 10). Because the sum is complete before the first write, `y == x` works. $\varepsilon$ goes inside the root: a zero row then gives $0 \cdot (1/\sqrt{\varepsilon}) = 0$, never $0/0$, and Llama's checkpoints were trained that way.

### 2.2 RoPE

RoPE (`L7.3`) rotates pairs of a query or key vector by an angle proportional to its position. Pair $i$ of token $t$ is rotated by $\theta_{t,i} = p_t \omega_i$:

$$\begin{pmatrix} a' \\ b' \end{pmatrix} = \mu \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} a \\ b \end{pmatrix} = \begin{pmatrix} \mu(a\cos\theta - b\sin\theta) \\ \mu(a\sin\theta + b\cos\theta) \end{pmatrix} .$$

A rotation keeps the pair's length (times $\mu$), and the dot product of a query rotated by $\theta_q$ and a key rotated by $\theta_k$ depends only on $\theta_q - \theta_k$: attention sees relative position. Three details of the contract:

- **The layout** says which entries form pair $i$: `layout 0` ("half", HF Llama, SmolLM2) pairs $(x_i, x_{i + r/2})$; `layout 1` ("interleaved", Meta's code, the RoPE paper) pairs $(x_{2i}, x_{2i+1})$. Both are the same rotation on a permuted vector, which is why HF's conversion script permutes the rows of `q_proj` and `k_proj`.
- **Partial rotary**: only the first $r$ entries of each head rotate; entries $r$ to $D - 1$ pass through.
- **The angle** is formed once per (token, pair) in double precision and rounded to float, which equals float32$(p_t) \cdot$ float32$(\omega_i)$ for every position below $2^{24}$: the same angle your Python computes. One angle serves all $H$ heads, so the loop order is token, pair, head.

Both members of a pair must be read before either is written; computing $b'$ from the new $a'$ is a different (wrong) map.

### 2.3 SiLU-mul (SwiGLU)

Llama's MLP (`L7.2`) computes $\mathrm{down}(\mathrm{silu}(\mathrm{gate}\,x) \odot \mathrm{up}\,x)$. The middle step is elementwise:

$$y_i = \mathrm{silu}(g_i)\, u_i, \qquad \mathrm{silu}(g) = g\,\sigma(g) = \frac{g}{1 + e^{-g}} .$$

Written this way it is safe at both ends. For $g \to +\infty$, $e^{-g} \to 0$ and $y \to g u$. For $g \to -\infty$, $e^{-g}$ overflows to $+\infty$ (your `tl_expf` saturates above 88.7), and $g / \infty = -0$, the right limit. The algebraically equal $g e^{g} / (1 + e^{g})$ computes $\infty / \infty = \mathrm{NaN}$ at $g = 100$.

### 2.4 Embedding, add, argmax

**Embedding** copies row `ids[t]` of a `[V, d]` table: the source starts at element `ids[t] * d`. The contract makes the caller check ids against $V$ (the kernel cannot know $V$). **Add** is the residual connection, $y = a + b$, usually in place. **Argmax** picks the greedy token (`spec/sampling.md`, temperature 0): the largest value, the **lowest index** among equal values (a strict `>` while scanning forward), NaN entries skipped, and $-1$ when there is no number at all, so the caller reports an error instead of emitting token 0.

## 3. Worked example by hand

**RMSNorm** of $x = [3, 4]$, $w = [1, 0.5]$, $\varepsilon = 0$: $\overline{x^2} = (9 + 16)/2 = 12.5$, $\sqrt{12.5} = 3.5355339$, $\text{inv} = 0.2828427$, so $y = [3 \cdot 0.2828427 \cdot 1,\ 4 \cdot 0.2828427 \cdot 0.5] = [0.8485281, 0.5656854]$.

**RoPE** (half layout) of $x = [1, 2, 3, 4]$, $D = r = 4$, $\omega = [1, 0.01]$, position 1, $\mu = 1$. Pairs are $(x_0, x_2) = (1, 3)$ at $\theta = 1$ and $(x_1, x_3) = (2, 4)$ at $\theta = 0.01$:

| Pair | $\cos\theta$ | $\sin\theta$ | $a' = a\cos - b\sin$ | $b' = a\sin + b\cos$ |
|---|---|---|---|---|
| $(1, 3)$, $\theta = 1$ | 0.5403023 | 0.8414710 | $0.5403023 - 2.5244130 = -1.9841107$ | $0.8414710 + 1.6209069 = 2.4623779$ |
| $(2, 4)$, $\theta = 0.01$ | 0.9999500 | 0.0099998 | $1.9999000 - 0.0399993 = 1.9599007$ | $0.0199997 + 3.9998000 = 4.0197997$ |

Written back to positions 0, 2 and 1, 3: $[-1.9841107, 1.9599007, 2.4623779, 4.0197997]$. With the interleaved layout the pairs are $(1, 2)$ and $(3, 4)$ and the result is $[-1.1426397, 1.9220756, 2.9598507, 4.0297995]$ (`rope_interleaved_layout_and_position_zero`).

**SiLU-mul** of gate $[0, 1, -1]$ and up $[5, 2, 3]$: $\mathrm{silu}(0) = 0$, $\mathrm{silu}(1) = 1/(1 + e^{-1}) = 0.7310586$, $\mathrm{silu}(-1) = -1/(1 + e) = -0.2689414$; times up: $[0, 1.4621172, -0.8068243]$.

**Argmax** of $[2, 7, 7, -1]$: 7 first appears at index 1, and index 2 is not strictly greater, so the answer is 1.

All four are the first test, `hand_example`; RMSNorm and argmax repeat through the shared fixture in `test_hand_example`.

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

From Python, declare each with `restype None` (or `c_int32` for argmax) on your loader.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example` | unit, smoke | section 3 for RMSNorm, RoPE, SiLU-mul, argmax | you and the tests agree on the definitions |
| `rmsnorm_eps_is_inside_the_root` | boundary | $[10^{-3}, 10^{-3}]$ with $\varepsilon = 10^{-6}$ gives 0.7071; a zero row gives zeros | tiny-norm rows, Llama's trained form |
| `rmsnorm_rows_are_independent_and_in_place` | property | every row alone equals the row in a batch of up to 12, bit for bit; `y == x` | batched decode in `L10.2` |
| `rope_interleaved_layout_and_position_zero` | unit | the interleaved worked example; position 0 is the identity | loading Meta-layout weights |
| `rope_partial_rotary_and_scaling` | unit | the tail past `d_rot` untouched; $\mu = 2$ doubles the pair; `[T, H, D]` indexing | GPT-NeoX style partial rotary, YaRN |
| `rope_is_a_rotation` | property | every pair keeps its length, random shapes and positions, both layouts | no half-updated pairs, no wrong partners |
| `silu_mul_saturates_without_nan` | boundary | gates $\pm 100$, $-10^4$ give the right limits | large activations in trained MLPs |
| `embedding_gathers_rows` | unit | repeated and out-of-order ids | the first op of the forward |
| `add_in_place` | unit | $y = a + b$ with `y == a` | the residual stream |
| `argmax_ties_nan_and_empty` | boundary | ties to the lowest index, NaN skipped, $-1$ for all NaN or empty, all $-\infty$ gives 0 | greedy decoding parity |
| `hand_example` | unit, smoke | RMSNorm and argmax of section 3 in the C test harness | checks standalone routines |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| $\varepsilon$ outside the root, $x / (\sqrt{\overline{x^2}} + \varepsilon)$ | correct on ordinary rows, wrong on tiny ones; drifts from the trained model | `rmsnorm_eps_is_inside_the_root` (mutant `s01`) |
| the sum of squares not divided by $d$ | every output shrinks by $\sqrt{d}$ | `rmsnorm_eps_is_inside_the_root` (mutant `s02`) |
| forgetting the gain $w$ | right shape, wrong scale | `hand_example` (mutant `s03`) |
| the sum of squares declared once per call, not per row | row 2 is normalized by rows 1 and 2 together; results depend on the batch | `rmsnorm_rows_are_independent_and_in_place` (mutant `s15`) |
| the layouts swapped | plausible numbers, wrong attention pattern on HF weights | `rope_interleaved_layout_and_position_zero` (mutant `s04`) |
| rotating by $-\theta$ (a sign flipped) | relative positions mirrored; the model's outputs degrade | `rope_interleaved_layout_and_position_zero` (mutant `s05`) |
| computing $b'$ from the new $a'$ | lengths change; not a rotation | `rope_is_a_rotation` (mutant `s06`) |
| rotating all $D$ entries when `d_rot < D` | the pass-through tail is scrambled (and `inv_freq` is read past its end) | `rope_partial_rotary_and_scaling` (mutant `s07`) |
| ignoring `attn_scaling` | YaRN-extended models lose their temperature correction | `rope_partial_rotary_and_scaling` (mutant `s08`) |
| SiLU as a sigmoid, or with $e^{g}$ instead of $e^{-g}$ | wrong MLP; or $0$ instead of $g u$ at large gates | `silu_mul_saturates_without_nan` (mutants `s09`, `s10`) |
| embedding row at `ids[t]` instead of `ids[t] * d` | every token reads a slice of row 0 or 1 | `embedding_gathers_rows` (mutant `s11`) |
| argmax with `>=` | ties go to the last index; greedy output differs from Python | `argmax_ties_nan_and_empty` (mutant `s12`) |
| argmax without the NaN check | a NaN in the first place wins forever | `argmax_ties_nan_and_empty` (mutant `s13`) |
| returning 0 for an empty or all-NaN row | the engine emits token 0 instead of reporting the error | `argmax_ties_nan_and_empty` (mutant `s14`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `rt.02` | the shared status, error, and allocator support |
| Back | `M09.5` | `tl_rsqrtf`: the $1/\sqrt{\cdot}$ of RMSNorm |
| Back | `M09.6` | `tl_expf`: the $e^{-g}$ of SiLU |
| Back | `L7.1` | `RMSNorm`, the specification of `tl_rmsnorm_f32` |
| Back | `L7.3` | `rope_cos_sin` and `apply_rope`, the specification of `tl_rope_f32` |
| Forward | the standalone Rust engine | the Rust forward calls the same six kernels in its independent Rust implementation |


## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| separate RMSNorm and add | fused add + RMSNorm (vLLM `fused_add_rms_norm`) | one pass that adds the residual and normalizes, halving memory traffic | vLLM `csrc/layernorm_kernels.cu` |
| RoPE with `cos` per call | cached cos/sin tables and fused QK rotation | precomputed tables per position, rotation fused into the QKV projection epilogue | llama.cpp `ggml_rope_ext`; FlashInfer `rope.cuh` |
| scalar SiLU | vectorized SwiGLU with a polynomial exp | SIMD over 8 or 16 lanes | ggml `ggml_vec_swiglu_f32` |
| argmax | fused sampling kernels | argmax and top-k inside the final matmul's epilogue | FlashInfer `sampling.cuh` |
