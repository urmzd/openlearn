<!-- ol:module L7.3 -->
# RoPE (half and interleaved layouts, partial rotary)

## Overview

| | |
|---|---|
| **Module** | `L7.3` · build · Python · Pass 5 · 3 h, plus your graded tests (rung R5) |
| **You build** | `python/tinyllm/modern/rope.py`: `rope_cos_sin`, `apply_rope`, `RopeSpec`; and your own oracle tests in `python/tests/l7-3-rope/` |
| **Contract** | [`course/contracts/py/tinyllm/modern/rope.pyi`](../../../course/contracts/py/tinyllm/modern/rope.pyi) |
| **Tests** | `course/tests/L7.3/test_rope.py` (what they check: section 4), golden values from transformers 5.19.0 (`LlamaRotaryEmbedding`, `apply_rotary_pos_emb`, GPT-NeoX partial rotary) and Meta's complex-number `apply_rotary_emb` in `course/fixtures/L7.3/rope_hf.npz` (`course/oracle/L7.3/rope_hf.py`); your tests are graded by mutation, threshold 0.80 with every pitfall fault required |
| **Needs** | `L0.2` `F.concat`, `F.stack`, `F.reshape` · `L0.1` `Tensor` and its indexing · `M00.2` `rotate_pairs` (the tests compare against it) · reading: `M00.3` (the frequency ladder) (or `--ref-deps`) |
| **Used by** | `L7.5` rotates queries and keys · `L7.6` the decoupled rope key of MLA · `L7.7` windowed attention · `L7.9` every Llama attention layer · `L8.2` decoding at the right positions · later: `L9.6` the C `tl_rope_f32` |
| **Milestone** | `MS-L7` (SmolLM2-135M logits match Hugging Face) |
| **Optional depth** | Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021), sections 3.2 to 3.4; EleutherAI blog, "Rotary Embeddings: A Relative Revolution" (2021) |

## Key Takeaways

- RoPE rotates pair $i$ of a query or key at position $p$ by the angle $p\,\theta_i$; rotations keep lengths and add angles, so a score depends only on the offset between positions (`test_scores_depend_only_on_relative_position`).
- Two layouts choose which entries form a pair: HF's **half** layout pairs $x_i$ with $x_{i + r/2}$, Meta's **interleaved** layout pairs $x_{2i}$ with $x_{2i+1}$; they are one rotation on a permuted vector (`test_layouts_are_a_permutation_of_each_other`).
- **Partial rotary** rotates only the first $r$ entries and passes the rest through (`test_partial_rotary_golden`).
- The angle is one float32 product, as in Hugging Face; lower precision destroys long positions (`test_cos_sin_table_golden`).
- The backward is the inverse rotation (`test_gradient_is_the_inverse_rotation`).

## How to work this chapter

```bash
ol start L7.3              # stubs rope.py; prints your test path and rung (R5)
ol tests L7.3              # the course tests
# write your oracle tests in python/tests/l7-3-rope/, then:
ol check L7.3              # course tests and the mutation grade of your tests
ol diff  L7.3              # after passing: your code against the reference
```

---

## 1. Why now

Your 2017 transformer adds a sinusoidal or learned vector to each token embedding (`L5.4`), so position enters once, at the bottom, mixed into the content. SmolLM2 has no position embedding at all: its checkpoint has no `wpe` table, and a model that ignores position cannot tell "dog bites man" from "man bites dog". The position goes in **inside every attention layer**, by rotating the queries and keys. Load SmolLM2 without it and the logits are garbage; rotate the wrong pairs and they are subtly wrong. This module builds the rotation exactly as Hugging Face does, in both layouts real checkpoints use.

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

Same two rotations, different entries. This is `test_hand_example`.

## 4. The interface

```python
def rope_cos_sin(positions, inv_freq, attention_scaling=1.0) -> tuple[NDArray, NDArray]: ...  # float32 [..., r/2]
def apply_rope(x: Tensor, cos, sin, layout="half", rotary_dim=None) -> Tensor: ...         # [..., d_h]
@dataclass
class RopeSpec: inv_freq: NDArray; attention_scaling: float; layout: str; rotary_dim: int
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit | section 3 in both layouts | you and the test agree on the pairs |
| `test_cos_sin_table_golden` | golden | HF's tables up to position 65535 | long contexts in `L7.4` |
| `test_cos_sin_attention_scaling_golden` | golden | YaRN's scaled tables | `rope_scaling` configs in `L7.9` |
| `test_half_layout_golden` | golden | HF's `apply_rotary_pos_emb`, per-row positions | SmolLM2 in MS-L7 |
| `test_interleaved_layout_golden` | golden | Meta's complex-number rotation | Meta-layout checkpoints |
| `test_partial_rotary_golden` | golden | GPT-NeoX with `rotary_dim` 8 of 16 | partial-rotary models |
| `test_scores_depend_only_on_relative_position` | property | shifted pairs give equal scores | the reason RoPE exists |
| `test_rotation_preserves_length_and_position_zero_is_identity` | property | norms kept; position 0 unchanged | no scale drift with position |
| `test_layouts_are_a_permutation_of_each_other` | property | the HF conversion permutation | loading Meta weights |
| `test_interleaved_matches_rotate_pairs` | differential | agrees with `M00.2` | one definition of a rotation |
| `test_gradcheck_both_layouts_and_partial` | gradcheck | float64 central differences | `q_proj`, `k_proj` learn through RoPE |
| `test_gradient_is_the_inverse_rotation` | property | backward = rotation by $-p\theta$ | the C backward later |
| `test_rope_spec_fields` | unit | the spec drives `apply_rope` | `L7.5`, `L7.6` take a `RopeSpec` |
| `test_validation` | boundary | odd or oversized `rotary_dim`, bad tables, bad positions | wiring bugs fail loudly |

### Your graded tests (rung R5)

Your oracle is complex multiplication in numpy float64: form $a + ib$ from each pair (by the layout's rule), multiply by $e^{i p \theta}$, and split back. Compare `apply_rope` for both layouts and partial rotary, check the hand example, that `attention_scaling` multiplies both tables, that large positions stay precise, that the gradient is the inverse rotation, and that bad input raises. Import only `tinyllm.modern.rope`, `tinyllm.autograd.tensor`, and `tinyllm.autograd.functional`. `ol check L7.3` requires 0.80 with every pitfall fault killed.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. adjacent pairs where the checkpoint expects half pairs | the model loads and every attention pattern is wrong | `test_half_layout_golden` (mutant `s01`) |
| 2. rotating by $-p\theta$ (a sign slip in sin) | relative scores still look fine; the weights disagree | `test_hand_example`, `test_cos_sin_table_golden` (mutant `s02`) |
| 3. partial rotary on the wrong entries | GPT-NeoX-style models degrade quietly | `test_partial_rotary_golden` (mutant `s03`) |
| 4. angles in float16 | positions past 2048 round to even numbers; 65535 overflows | `test_cos_sin_table_golden`, `test_scores_depend_only_on_relative_position` (mutant `s04`) |
| scaling only the cosine | YaRN scores are not $s^2$ times larger | `test_cos_sin_attention_scaling_golden` (mutant `s05`) |
| reading half of each pair as a constant | `k_proj` learns from half its gradient | `test_gradcheck_both_layouts_and_partial` (mutant `s06`) |
| leaving the interleaved output in half order | the next layer reads permuted features | `test_interleaved_layout_golden` (mutant `s07`) |

## 6. Where it's used next

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

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `rope_cos_sin` | HF `LlamaRotaryEmbedding` | caches tables, recomputes them for dynamic scaling | `transformers/models/llama/modeling_llama.py` |
| `apply_rope` | vLLM `RotaryEmbedding` (`rotary_embedding` kernel) | in place on q and k in one fused CUDA kernel, `is_neox_style` selects the layout | vLLM `csrc/pos_encoding_kernels.cu` |
| partial rotary | GPT-NeoX `rotary_pct`, Phi `partial_rotary_factor` | the fraction from the config | HF GPT-NeoX and Phi modeling files |
| position handling | multimodal RoPE (M-RoPE, Qwen2-VL) | separate time, height, width sections of the pairs | Qwen2-VL report |
