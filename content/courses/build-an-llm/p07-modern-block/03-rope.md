# RoPE (half and interleaved layouts, partial rotary)

## Overview

| | |
|---|---|
| **Module** | `L7.3` · build · Python · Pass 5 · 2 h reading |
| **Solution** | [`rope.py`](../../../../course/solutions/python/tinyllm/modern/rope.py): `rope_cos_sin`, `apply_rope`, `RopeSpec`; model oracle tests in [`test_rope.py`](../../../../course/solutions/learner-tests/L7.3/python/tests/l7-3-rope/test_rope.py) |
| **Contract** | [`course/contracts/py/tinyllm/modern/rope.pyi`](../../../../course/contracts/py/tinyllm/modern/rope.pyi) |
| **Needs** | `L0.2` `F.concat`, `F.stack`, `F.reshape` · `L0.1` `Tensor` and its indexing · `M00.2` `rotate_pairs` · reading: `M00.3` (the frequency ladder) |
| **Used by** | `L7.5` rotates queries and keys · `L7.6` the decoupled rope key of MLA · `L7.7` windowed attention · `L7.9` every Llama attention layer, on the way to SmolLM2-135M logits that match Hugging Face · `L8.2` decoding at the right positions · later: `L9.6` the C `tl_rope_f32`, `L13.2`, `L13.3` |
| **Optional depth** | Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021), sections 3.2 to 3.4; EleutherAI blog, "Rotary Embeddings: A Relative Revolution" (2021) |

## Key Takeaways

- RoPE rotates pair $i$ of a query or key at position $p$ by the angle $p\,\theta_i$; rotations keep lengths and add angles, so a score depends only on the offset between positions.
- Two layouts choose which entries form a pair: HF's **half** layout pairs $x_i$ with $x_{i + r/2}$, Meta's **interleaved** layout pairs $x_{2i}$ with $x_{2i+1}$; they are one rotation on a permuted vector.
- **Partial rotary** rotates only the first $r$ entries and passes the rest through.
- The angle is one float32 product, as in Hugging Face; lower precision destroys long positions.
- The backward is the inverse rotation.

## How to read this chapter

Sections 1 to 3 build the rotation from first principles and work it by hand in both layouts. Section 4 fixes the interface and the cases a correct solution must handle. Section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

The 2017 transformer from `L5` adds a sinusoidal or learned vector to each token embedding (`L5.4`), so position enters once, at the bottom, mixed into the content. SmolLM2 has no position embedding at all: its checkpoint has no `wpe` table, and a model that ignores position cannot tell "dog bites man" from "man bites dog". The position goes in **inside every attention layer**, by rotating the queries and keys. Load SmolLM2 without it and the logits are garbage; rotate the wrong pairs and they are subtly wrong. This module builds the rotation exactly as Hugging Face does, in both layouts real checkpoints use.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $q, k \in \mathbb{R}^{d_h}$ | one head's query and key | `float32[d_h]` |
| $p$, $m$, $n$ | token positions | `int` $\ge 0$ |
| $r$ | `rotary_dim`: how many entries rotate, even, $r \le d_h$ | `int` |
| $\theta_i$ | inverse frequency of pair $i$, $i = 0 .. r/2 - 1$ (`inv_freq`) | `float[r/2]` |
| $R(\alpha)$ | the 2D rotation $\begin{pmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{pmatrix}$ | $2 \times 2$ |
| $R_p$ | rotate pair $i$ by $p\,\theta_i$ for every $i$ | $d_h \times d_h$ |
| $s$ | `attention_scaling` (YaRN, `L7.4`) | `float` |

### 2.1 Rotations and relative position

`M00.2` showed that $R(\alpha)$ keeps lengths, $R(\alpha)^\top = R(-\alpha)$, and $R(\alpha) R(\beta) = R(\alpha + \beta)$. RoPE splits $q$ into $r/2$ pairs and rotates pair $i$ of the token at position $p$ by $p\,\theta_i$. For a query at $m$ and a key at $n$, pair by pair:

$$\langle R(m\theta_i) q_i, R(n\theta_i) k_i \rangle = q_i^\top R(m\theta_i)^\top R(n\theta_i)\, k_i = q_i^\top R((n - m)\theta_i)\, k_i .$$

The score depends on $n - m$ only. Attention gets relative position for free, inside the dot product, and at no parameter cost. The frequencies $\theta_i = \text{base}^{-2i/r}$ form `M00.3`'s geometric ladder: pair 0 turns once per $2\pi$ positions, the last pair barely moves over the whole context, so fast pairs resolve nearby order and slow pairs distinguish far positions.

### 2.2 Two layouts

Which entries form a pair is a convention fixed by how the weights were trained:

- **half** (`rotate_half`, HF Llama, Mistral, Qwen, SmolLM2, gpt-oss): pair $i$ is $(x_i, x_{i + r/2})$.
- **interleaved** (the RoFormer paper, Meta's original Llama code): pair $i$ is $(x_{2i}, x_{2i+1})$, read as one complex number $x_{2i} + i\,x_{2i+1}$ multiplied by $e^{i p \theta_i}$.

Rotating pair $(a, b)$ by $\alpha$ gives $(a\cos\alpha - b\sin\alpha,\ a\sin\alpha + b\cos\alpha)$ in either layout. Reorder the entries $(a_0, b_0, a_1, b_1, \dots)$ into $(a_0, a_1, \dots, b_0, b_1, \dots)$ and the interleaved rotation becomes the half rotation. Hugging Face converts Meta's checkpoints by permuting the rows of `q_proj` and `k_proj` exactly this way, so their projections produce half-layout vectors directly.

### 2.3 Computing the angles

`rope_cos_sin(positions, inv_freq, s)` returns one cosine and one sine **per pair**, shape `positions.shape + (r/2,)`. The angle $p\,\theta_i$ is computed as Hugging Face does: one float32 product of `float32(p)` and `float32(θ_i)`. At position 65535 the float32 spacing is about 0.004, so the angle itself is known only to a few thousandths of a radian; computing it in float16 (spacing 32 at that position, and 65535 overflows) destroys it. Both tables are multiplied by $s$, so the query and the key each grow by $s$ and every score by $s^2$: YaRN's temperature (`L7.4`).

### 2.4 Partial rotary

GPT-NeoX and Phi rotate only the first $r < d_h$ entries of each head and leave the other $d_h - r$ unchanged: those entries carry content with no position at all. `apply_rope(x, cos, sin, layout, rotary_dim=r)` rotates `x[..., :r]` and concatenates `x[..., r:]` back unchanged.

### 2.5 Backward

$R_p$ is linear and orthogonal, so the gradient with respect to $x$ is $R_p^\top g = R_{-p}\, g$: the upstream gradient rotated back, which is `apply_rope(g, cos, -sin)`. Building `apply_rope` from indexing, products with constant tables, `F.concat`, and `F.stack` gives exactly that through autograd.

## 3. Worked example by hand

$d_h = r = 4$, $\theta = (1, 0.01)$, position $p = 1$, $x = (1, 0, 0, 1)$. Angles: $1$ and $0.01$ radians. $\cos 1 = 0.540302$, $\sin 1 = 0.841471$, $\cos 0.01 = 0.999950$, $\sin 0.01 = 0.010000$.

**Half layout.** Pair 0 is $(x_0, x_2) = (1, 0)$, rotated by 1: $(0.540302, 0.841471)$. Pair 1 is $(x_1, x_3) = (0, 1)$, rotated by 0.01: $(-0.010000, 0.999950)$. Writing pair 0 back to entries 0 and 2 and pair 1 to entries 1 and 3: $(0.540302, -0.010000, 0.841471, 0.999950)$.

**Interleaved layout.** Pair 0 is $(x_0, x_1) = (1, 0)$: $(0.540302, 0.841471)$. Pair 1 is $(x_2, x_3) = (0, 1)$: $(-0.010000, 0.999950)$. Output $(0.540302, 0.841471, -0.010000, 0.999950)$.

Same two rotations, different entries.

## 4. The interface

```python
def rope_cos_sin(positions, inv_freq, attention_scaling=1.0) -> tuple[NDArray, NDArray]: ...  # float32 [..., r/2]
def apply_rope(x: Tensor, cos, sin, layout="half", rotary_dim=None) -> Tensor: ...         # [..., d_h]
@dataclass
class RopeSpec: inv_freq: NDArray; attention_scaling: float; layout: str; rotary_dim: int
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand example | unit | section 3 in both layouts | the solution and the reader agree on the pairs |
| cos/sin tables | golden | HF's tables up to position 65535 | long contexts in `L7.4` |
| tables with attention scaling | golden | YaRN's scaled tables | `rope_scaling` configs in `L7.9` |
| half layout | golden | HF's `apply_rotary_pos_emb`, per-row positions | SmolLM2 logits in `L7.9` |
| interleaved layout | golden | Meta's complex-number rotation | Meta-layout checkpoints |
| partial rotary | golden | GPT-NeoX with `rotary_dim` 8 of 16 | partial-rotary models |
| scores depend only on relative position | property | shifted pairs give equal scores | the reason RoPE exists |
| length preserved, position 0 is the identity | property | norms kept; position 0 unchanged | no scale drift with position |
| layouts are a permutation of each other | property | the HF conversion permutation | loading Meta weights |
| interleaved matches `rotate_pairs` | differential | agrees with `M00.2` | one definition of a rotation |
| gradients in both layouts and partial | gradcheck | float64 central differences | `q_proj`, `k_proj` learn through RoPE |
| the gradient is the inverse rotation | property | backward = rotation by $-p\theta$ | the C backward later |
| `RopeSpec` fields | unit | the spec drives `apply_rope` | `L7.5`, `L7.6` take a `RopeSpec` |
| validation | boundary | odd or oversized `rotary_dim`, bad tables, bad positions | wiring bugs fail loudly |

### A model oracle test suite

[`test_rope.py`](../../../../course/solutions/learner-tests/L7.3/python/tests/l7-3-rope/test_rope.py) shows how to test this module independently of Hugging Face. Its oracle is complex multiplication in numpy float64: form $a + ib$ from each pair (by the layout's rule), multiply by $e^{i p \theta}$, and split back. It compares `apply_rope` for both layouts and partial rotary, checks the hand example, that `attention_scaling` multiplies both tables, that large positions stay precise, that the gradient is the inverse rotation, and that bad input raises. A complex-number oracle is a good choice because it is a second, independent definition of the same rotation: a slip in the real-valued formula (a swapped sign, a wrong pairing) cannot also appear in $z \cdot e^{i\alpha}$.

## 5. The solution, step by step

### 5.1 From first principles

1. A score must depend on $n - m$ only (section 2.1). Rotations compose by adding angles, so rotating each pair by an angle linear in position, $p\,\theta_i$, gives exactly that. Nothing about the rotation is learned, so the module has no parameters.
2. The angles depend on positions and frequencies, never on the data, so they can be computed once as two tables, $\cos(p\theta_i)$ and $\sin(p\theta_i)$, shape `positions.shape + (r/2,)`. One entry per pair, not per element: both elements of a pair share an angle.
3. The tables must match Hugging Face bit for bit for long positions (section 2.3), which fixes the arithmetic: one float32 product `float32(p) * float32(θ_i)`, then cos and sin.
4. YaRN scales every score by $s^2$. Scaling both tables by $s$ scales $q$ and $k$ each by $s$, so the score picks up $s^2$ and no other code changes.
5. The rotation itself is $(a\cos\alpha - b\sin\alpha,\ a\sin\alpha + b\cos\alpha)$ in both layouts (section 2.2). The layouts differ only in where $a$ and $b$ come from, so the solution is one formula and two ways of slicing.
6. The backward must be the inverse rotation (section 2.5). If the forward is written only from slicing, elementwise products with constant tables, `F.concat`, `F.stack`, and `F.reshape`, autograd composes their backwards into exactly $R_{-p}$. No custom backward is needed, and none can be wrong.
7. Partial rotary means "the first $r$ entries rotate, the rest pass through", so it is one slice and one concat around the full-rotary case.

### 5.2 The code, annotated

**The tables.** Validation first, because the tables are built from integers that come from callers (a decoding loop, a packed batch) and a float position like 2.5 is always a wiring bug.

```python
def rope_cos_sin(positions, inv_freq, attention_scaling=1.0):
    p = np.asarray(positions)
    # positions must be whole numbers >= 0; a fractional or negative one is a bug upstream
    if p.size and (p.dtype.kind not in "iuf" or np.any(p != np.round(p))):
        raise ValueError(f"positions must be integers, got {p.dtype}")
    if p.size and p.min() < 0:
        raise ValueError(f"positions must be >= 0, got min {p.min()}")
    f = np.asarray(inv_freq)
    if f.ndim != 1:
        raise ValueError(f"inv_freq must be 1-D, got shape {f.shape}")
    # One float32 product per (position, pair), as HF's rotary embedding does
    # (positions.float() * inv_freq.float()); then cos and sin of it.
    # The trailing [..., None] broadcasts any positions shape against r/2 pairs.
    angle = p.astype(np.float32)[..., None] * f.astype(np.float32)
    s = np.float32(attention_scaling)
    # scale BOTH tables: q and k each grow by s, every score by s^2 (YaRN)
    return (np.cos(angle) * s).astype(np.float32), (np.sin(angle) * s).astype(np.float32)
```

`positions` can be any shape: `[T]` for one sequence, `[B, T]` for per-row positions in a packed or left-padded batch, `[1]` for one decoding step. The output shape `positions.shape + (r/2,)` then broadcasts against `x[..., r/2]` without any reshaping by the caller.

**Checking `rotary_dim` and the tables.**

```python
def _rotary_dim(x, cos, sin, rotary_dim):
    dh = x.shape[-1]
    r = dh if rotary_dim is None else int(rotary_dim)
    # pairs need an even count; you cannot rotate more entries than the head has
    if r < 2 or r % 2 or r > dh:
        raise ValueError(...)
    for name, t in (("cos", cos), ("sin", sin)):
        # one angle per pair, and the table must broadcast over x's leading axes
        if t.ndim < 1 or t.shape[-1] != r // 2:
            raise ValueError(...)
        try:
            np.broadcast_shapes(t.shape, x.shape[:-1] + (r // 2,))
        except ValueError:
            raise ValueError(...) from None
    return r
```

The broadcast check matters because numpy would otherwise happily broadcast a `[T, r/2]` table against an `[H, T, r/2]` slice in the wrong axis order whenever two sizes coincide, and the error would surface as wrong logits rather than an exception.

**The rotation.**

```python
def apply_rope(x, cos, sin, layout="half", rotary_dim=None):
    if layout not in ("half", "interleaved"):
        raise ValueError(...)
    c, s = np.asarray(cos), np.asarray(sin)
    r = _rotary_dim(x, c, s, rotary_dim)
    dh, h = x.shape[-1], r // 2
    if layout == "half":
        # pair i is (x[i], x[i + r/2]): a is the first half, b the second
        a, b = x[..., :h], x[..., h:r]
        # rotated a goes back to the first half, rotated b to the second
        rot = F.concat([a * c - b * s, b * c + a * s], axis=-1)
    else:
        # pair i is (x[2i], x[2i + 1]); rotate, then interleave back
        a, b = x[..., 0:r:2], x[..., 1:r:2]
        # stack on a new last axis gives [..., r/2, 2]: (a'_0, b'_0), (a'_1, b'_1), ...
        pairs = F.stack([a * c - b * s, a * s + b * c], axis=-1)
        # flattening that axis restores the interleaved order a'_0, b'_0, a'_1, ...
        rot = F.reshape(pairs, x.shape[:-1] + (r,))
    if r == dh:
        return rot
    # partial rotary: entries r.. pass through untouched
    return F.concat([rot, x[..., r:]], axis=-1)
```

Check it against section 3. Half layout: $a = (1, 0)$, $b = (0, 1)$, $c = (0.540302, 0.999950)$, $s = (0.841471, 0.010000)$. Then $a c - b s = (0.540302, -0.010000)$ and $b c + a s = (0.841471, 0.999950)$, concatenated: $(0.540302, -0.010000, 0.841471, 0.999950)$, the hand answer. Interleaved: $a = (x_0, x_2) = (1, 0)$, $b = (x_1, x_3) = (0, 1)$, the same two rotated pairs, and the stack-then-reshape writes them as $(0.540302, 0.841471, -0.010000, 0.999950)$.

Every operation on `x` is a slice, a product with a constant numpy table, a sum, a concat, a stack, or a reshape. Each of those has a backward in `L0.2`, and their composition is the transpose of the rotation, which is the rotation by $-p\theta$. Both halves of each pair reach the output through `x`'s own slices, so both receive gradient.

**The spec.**

```python
@dataclass
class RopeSpec:
    """How an attention layer rotates its queries and keys (L7.5, L7.6)."""
    inv_freq: NDArray
    attention_scaling: float
    layout: str
    rotary_dim: int
```

`RopeSpec` bundles everything a layer needs so that `L7.9` can read `config.json` once (`rope_theta`, `rope_scaling`, `partial_rotary_factor`) and hand the same spec to every attention layer.

Full solution: [`rope.py`](../../../../course/solutions/python/tinyllm/modern/rope.py).

### 5.3 Why this solution

- **Cost.** The tables are $O(T \cdot r/2)$ to build and are shared by every head and layer; `apply_rope` is $O(\text{numel}(x))$ elementwise work. RoPE adds nothing to the parameter count and is negligible next to the $O(T^2 d_h)$ scores.
- **Pair-wise formula, not a $d_h \times d_h$ matrix.** Building $R_p$ explicitly costs $O(d_h^2)$ per position and a matmul per token; the matrix is block-diagonal with $2 \times 2$ blocks, so the four products per pair are all the work there is.
- **Real arithmetic, not complex numbers.** Meta's code views pairs as complex numbers, which is elegant for the interleaved layout but awkward for the half layout and for autograd built on real tensors. The real formula serves both layouts with the same backward.
- **float32 angles, cached tables.** Computing angles in float16 or bfloat16 is cheaper but wrong past a few thousand positions (section 2.3). Matching HF's single float32 product is what makes the golden comparisons bit-close at 65535.
- **Accepted tradeoff.** The solution builds new tensors with `concat`/`stack` rather than rotating in place, which costs a copy of $q$ and $k$ per layer. Production kernels rotate in place (see Going further); here clarity and a free, correct backward are worth more.

### 5.4 Explain it back

1. Why does the score between a rotated query and key depend only on $n - m$?
   <details><summary>Answer</summary>Pair by pair the score is $q_i^\top R(m\theta_i)^\top R(n\theta_i) k_i$. $R^\top = R(-\cdot)$ and rotations add angles, so this is $q_i^\top R((n-m)\theta_i) k_i$: the absolute positions cancel.</details>
2. Why do `rope_cos_sin` tables have $r/2$ columns and not $r$?
   <details><summary>Answer</summary>Both entries of a pair rotate by the same angle, so there is one angle per pair. The half layout uses the same table for $a$ and $b$; HF's own tables repeat it to width $r$ only because its `rotate_half` multiplies the whole vector at once.</details>
3. Why multiply both cos and sin by `attention_scaling`, rather than just one?
   <details><summary>Answer</summary>Each rotated entry is a linear combination of $\cos$ and $\sin$ terms, so scaling both scales the whole rotated vector by $s$. Applied to $q$ and $k$, every score grows by $s^2$, which is YaRN's temperature. Scaling only cos would distort the rotation itself, not just its length.</details>
4. Why does the solution need no hand-written backward?
   <details><summary>Answer</summary>The forward is built only from differentiable primitives whose backwards already exist (slicing, products with constants, concat, stack, reshape). Their composition is the transpose of the orthogonal rotation, i.e. the rotation by $-p\theta$, which is the correct gradient.</details>
5. What breaks if the solution used the interleaved slicing on a SmolLM2 checkpoint?
   <details><summary>Answer</summary>SmolLM2 was trained (and HF-converted) for the half layout, so its `q_proj`/`k_proj` rows place pair $i$ at entries $i$ and $i + r/2$. Interleaved slicing rotates the wrong entries together; the model still runs, scores are still relative, but every attention pattern is wrong.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. adjacent pairs where the checkpoint expects half pairs | the model loads and every attention pattern is wrong | `layout` is explicit and defaults to `"half"`; the half branch slices `[:h]` and `[h:r]` (step 5) |
| 2. rotating by $-p\theta$ (a sign slip in sin) | relative scores still look fine; the weights disagree | the rotated first entry is `a * c - b * s`, the second `b * c + a * s`, as in the hand example |
| 3. partial rotary on the wrong entries | GPT-NeoX-style models degrade quietly | rotates only `x[..., :r]` and concatenates `x[..., r:]` back unchanged (step 7) |
| 4. angles in float16 | positions past 2048 round to even numbers; 65535 overflows | one float32 product `float32(p) * float32(θ)` (step 3) |
| scaling only the cosine | YaRN scores are not $s^2$ times larger | multiplies both tables by `s` (step 4) |
| reading half of each pair as a constant | `k_proj` learns from half its gradient | both `a` and `b` are slices of the tensor `x`, so autograd routes gradient to both (step 6) |
| leaving the interleaved output in half order | the next layer reads permuted features | stacks on a new last axis and reshapes, restoring $(a'_0, b'_0, a'_1, \dots)$ |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L0.2` | `F.concat`, `F.stack`, `F.reshape` give the backward |
| Back | `L0.1` | `Tensor` slicing |
| Back | `M00.2` | the 2D rotation and `rotate_pairs` |
| Forward | `L7.5` | rotates $q$ and $k$ with a `RopeSpec` before the scores |
| Forward | `L7.4` | its scaled `inv_freq` and `attention_scaling` feed `rope_cos_sin` |
| Forward | `L7.6` | MLA rotates only a decoupled rope part of each key |
| Forward | `L7.7` | sliding-window attention rotates with the same tables |
| Forward | `L7.9` | `config.json`'s `rope_theta` and `rope_scaling` build the spec |
| Forward | `L8.2` | decoding passes each new token's absolute position |
| Forward | `L9.6` | `tl_rope_f32` in C must match this |
| Forward | `L13.2` | `rope_cos_sin` per axis and `apply_rope` build 2D RoPE and M-RoPE |
| Forward | `L13.3` | `pos='rope2d'` rotates queries and keys with `apply_rope` |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `rope_cos_sin` | HF `LlamaRotaryEmbedding` | caches tables, recomputes them for dynamic scaling | `transformers/models/llama/modeling_llama.py` |
| `apply_rope` | vLLM `RotaryEmbedding` (`rotary_embedding` kernel) | in place on q and k in one fused CUDA kernel, `is_neox_style` selects the layout | vLLM `csrc/pos_encoding_kernels.cu` |
| partial rotary | GPT-NeoX `rotary_pct`, Phi `partial_rotary_factor` | the fraction from the config | HF GPT-NeoX and Phi modeling files |
| position handling | multimodal RoPE (M-RoPE, Qwen2-VL) | separate time, height, width sections of the pairs | Qwen2-VL report |
