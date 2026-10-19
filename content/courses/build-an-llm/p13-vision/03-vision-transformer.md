# Vision transformer encoder

## Overview

| | |
|---|---|
| **Module** | `L13.3` · build · Python · Pass 12 |
| **Solution** | [`vit.py`](../../../../course/solutions/python/tinyllm/vision/vit.py): `ViTConfig` (with `from_hf`), `PatchEmbed`, `VisionBlock`, `ViT` (`embed`, `encode`, `forward`), `interpolate_pos_embed`, `bucketed_position_ids` |
| **Contract** | [`course/contracts/py/tinyllm/vision/vit.pyi`](../../../../course/contracts/py/tinyllm/vision/vit.pyi) |
| **Needs** | `L13.1` `Conv2d` · `L13.2` `rope2d_cos_sin` · `L7.3` `apply_rope` · `L5.3` `MultiHeadAttention` · `L5.1` `scaled_dot_product_attention` · `L0.4` `Linear`, `LayerNorm`, `Embedding`, `ModuleList`, `Module` · `L0.2` ops · `L0.1` `Tensor` · `M12.3` `resize_planar_f32` · `M07.3` `normal_init` · `M06.3` `PCG32`. Reading: `M12.2` `patchify` |
| **Used by** | `L13.5` (the image tower of the dual encoder) · `L13.7` (the vision tower of every VLM architecture, tapped through `hidden`; a tiny VLM's logits match Hugging Face within $10^{-5}$ only if these hidden states do) |
| **Optional depth** | Dosovitskiy et al., "An image is worth 16x16 words" (2021), section 3.1; Radford et al., "Learning transferable visual models from natural language supervision" (2021), for CLIP's tower; Zhai et al., "Sigmoid loss for language image pre-training" (2023), for SigLIP's; Dehghani et al., "Patch n' Pack: NaViT" (2023), for fractional positions and padded patches |

## Key Takeaways

- A convolution whose kernel equals its stride is one linear map per patch: `PatchEmbed` equals cutting the image into patches, flattening each in $(c, u, v)$ order, and multiplying by the flattened kernel.
- One configurable encoder reproduces four published towers (ViT, CLIP, SigLIP, Idefics3) at every hidden state within $10^{-5}$; they differ only in a CLS token, a patch bias, where the extra LayerNorms sit, and the activation.
- Without positions the outputs permute with the patches, so position information has to be added: a learned table (resized bicubically for a new grid, or bucketed by fractional coordinate), a fixed 2D sin-cos table, or 2D RoPE inside attention.
- Padding is hidden on the key side of attention, so a real patch's output is the same whatever the padded patches hold.
- `hidden` holds the $L + 1$ block inputs and outputs before the final norm; `L13.7` picks one with `vision_feature_layer`.

## How to read this chapter

Section 1 says why the system needs an image encoder now. Section 2 derives the patch embedding, the block, the four tower variants, the three position schemes, and padding from first principles, and section 3 works bucketed positions and a sin-cos row by hand. Section 4 fixes the interface. Section 5 walks through `vit.py` and explains why each switch and each line is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

`L13.1` built convolution and `L13.2` built 2D rotary positions, but the transformer from Parts 5 and 7 still only reads token ids. This module makes an image into a sequence that the same blocks can read. Nothing in the blocks is new: attention from `L5.3`, LayerNorm and MLPs from Part 0. What is new is the front (patches and a position scheme) and matching published checkpoints exactly. Four towers that look alike differ in small switches (a CLS token, a patch bias, a pre-LayerNorm, `quick_gelu`), and a wrong switch still loads and still runs: it just produces features the language model was never trained on. `L13.5` trains this tower contrastively and `L13.7` puts it in front of a language model, where LLaVA reads `hidden[-2]`, so every hidden state has to be right, not only the last.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x$ | pixel values | `float32[N, C, H, W]` |
| $p$ | patch size | int |
| $(H_p, W_p)$ | patch grid, $(H/p, W/p)$, row-major | ints |
| $T$ | sequence length, $H_p W_p$ (plus 1 with a CLS token) | int |
| $d$, $L$ | width and number of blocks | ints |
| $n$ | table side, `image_size / p` | int |
| $h_\ell$ | hidden state after block $\ell$ | `float32[N, T, d]` |

### 2.1 Patch embedding is one linear map

`PatchEmbed` is `L13.1`'s `Conv2d(C, d, k = p, stride = p)`, optionally without a bias. With $k = s = p$ the output at grid cell $(i, j)$ is
$$y_{o,i,j} = b_o + \sum_{c=0}^{C-1} \sum_{u=0}^{p-1} \sum_{v=0}^{p-1} W_{o,c,u,v}\; x_{c,\ ip+u,\ jp+v}.$$
The window of cell $(i, j)$ covers rows $ip$ to $ip + p - 1$ and columns $jp$ to $jp + p - 1$. Because the stride equals the kernel, the windows of different cells are disjoint and together cover the image, so each pixel is read exactly once. Flatten the window of cell $(i, j)$ in $(c, u, v)$ order into a vector $z_{ij} \in \mathbb{R}^{Cp^2}$ (this is `M12.2`'s `patchify`), and flatten the kernel the same way into $\tilde W \in \mathbb{R}^{d \times Cp^2}$. The triple sum is then the dot product $y_{\cdot,i,j} = \tilde W z_{ij} + b$: one linear layer applied to every patch. It has $dCp^2 + d$ parameters, and the sequence is $[N, d, H_p, W_p]$ reshaped to $[N, H_p W_p, d]$ in row-major order.

### 2.2 The blocks, and four towers from one config

Block $i$ is pre-LN:
$$h \leftarrow h + \text{attn}(\text{LN}_1(h)), \qquad h \leftarrow h + W_2\,\text{act}(W_1\,\text{LN}_2(h) + b_1) + b_2.$$
The residual stream is never normalized inside a block, so an optional final `post_layernorm` normalizes it once at the end. Three activations appear in checkpoints: `gelu` is the exact $x\,\Phi(x)$; `gelu_tanh` is $\tfrac{1}{2}x\,(1 + \tanh(\sqrt{2/\pi}\,(x + 0.044715x^3)))$; `quick_gelu` is $x\,\sigma(1.702x)$, where the constant scales a logistic curve onto the normal CDF $\Phi$ (the two differ by at most about $0.01$). The three are close but not interchangeable at $10^{-5}$.

`ViTConfig.from_hf` reads a Hugging Face vision config and sets the switches by `model_type`:

| `model_type` | CLS | patch bias | `pre_layrnorm` | `post_layernorm` | `act` | `ln_eps` |
|---|---|---|---|---|---|---|
| `vit` | yes | yes | no | yes | `gelu` | $10^{-12}$ |
| `clip_vision_model` | yes | no | yes | no | `quick_gelu` | $10^{-5}$ |
| `siglip_vision_model` | no | yes | no | yes | `gelu_tanh` | $10^{-6}$ |
| `idefics3_vision`, `smolvlm_vision` | no | yes | no | yes | `gelu_tanh` | $10^{-6}$ |

The last row also sets `bucketed` (section 2.4). The keys `layer_norm_eps` and `hidden_act` in the config override the defaults, and `num_channels` defaults to 3. Parameter names follow Hugging Face's CLIP and SigLIP towers (`embeddings.patch_embedding.weight`, `embeddings.class_embedding`, `embeddings.position_embedding.weight`, `pre_layrnorm` with HF's spelling, `encoder.layers.<i>.self_attn.q_proj`, `layer_norm1`, `mlp.fc1`, `mlp.fc2`, `layer_norm2`, `post_layernorm`), so those checkpoints load by name.

`encode` returns `(last, hidden)`. With `output_hidden_states`, `hidden[0]` is the input of block 0 (the output of `embed`, after `pre_layrnorm` if there is one) and `hidden[i]` is the output of block $i - 1$, all before `post_layernorm`: $L + 1$ entries. Otherwise `hidden` is `[]`.

### 2.3 Positions

**Why they are needed.** Let $\pi$ permute the tokens. In attention, output $i$ is $\sum_j \alpha_{ij} v_j$ with $\alpha_{ij} = \text{softmax}_j(q_i \cdot k_j / \sqrt{d_h})$, and $q_i, k_j, v_j$ depend on tokens $i$ and $j$ alone. Permuting the inputs relabels $i$ and $j$ the same way, and a sum over all $j$ does not care about order, so the outputs are permuted by $\pi$ and nothing else changes. LayerNorm and the MLP act on each token separately. By induction over blocks, $f(\pi x) = \pi f(x)$: a tower with no positions cannot tell a picture from a shuffled one. `ViTConfig.pos` selects one of three remedies.

**`learned`.** A table of $P = n^2$ rows (plus one first row for CLS when `cls_token`), drawn with `normal_init` at std 0.02 and added to the patch vectors; the CLS vector gets row 0. For a grid other than $n \times n$ (and without `bucketed`), `interpolate_pos_embed` resizes the patch rows: view them as an image $[d, n, n]$ with one channel per width dimension, resample every channel to $(H_p, W_p)$, and flatten back row-major, keeping the CLS row untouched. The resampling is `M12.3`'s `resize_planar_f32(.., "bicubic")`, which is torch's `F.interpolate` with `align_corners=False`: output index $o$ reads the source coordinate $(o + 0.5)\,n/H_p - 0.5$ through the Keys cubic with $a = -0.75$ on its 4 neighbours, edge samples repeated. The resized table is a constant: no gradient flows to the stored table through it.

**`sincos2d`.** A fixed table, nothing to learn. With $q = d/4$ and $\omega_k = 10000^{-k/q}$ for $k < q$, the patch at row $r$, column $c$ gets
$$[\sin(r\omega_0..\omega_{q-1}),\ \cos(r\omega_0..\omega_{q-1}),\ \sin(c\omega_0..\omega_{q-1}),\ \cos(c\omega_0..\omega_{q-1})].$$
The first half of the width encodes the row and the second half the column, each as sines then cosines. It needs $d \bmod 4 = 0$, and a CLS token gets no addition.

**`rope2d`.** Nothing is added. Inside every block the queries and keys of each head are rotated by `L13.2`'s `rope2d_cos_sin(grid, d_head, rope_base, rope_merge)` with `L7.3`'s `apply_rope` in the `"half"` layout. For one rotary pair, $\langle R(\theta_q)\,q,\ R(\theta_k)\,k \rangle = \langle q,\ R(\theta_k - \theta_q)\,k \rangle$ because rotations are orthogonal, so the score sees only the difference of angles. The first half of the pairs uses row angles and the second half column angles, so scores depend on (row offset, column offset). Both $q$ and $k$ must be rotated for the difference to appear. It needs $d_\text{head} \bmod 4 = 0$; a CLS token is not rotated; and with `rope_merge` $m > 1$ the sequence is in `L13.2`'s block-major order (Qwen2-VL), which the caller of `encode` provides.

### 2.4 Padded patches and bucketed positions

Idefics3 and SmolVLM batch images whose real content covers only part of the grid. `patch_mask` (`bool[N, Hp, Wp]`, True for a real patch) marks it. The mask is applied to keys: with $m_j \in \{0, 1\}$,
$$\alpha_{ij} = \frac{m_j\, e^{s_{ij}}}{\sum_l m_l\, e^{s_{il}}},$$
so a padded token has weight 0 in every query's sum. The output of a real token $i$ is then a function of real tokens only. Patch embedding is per patch (section 2.1), LayerNorm and the MLP are per token, so by induction a real patch's output equals what it would be if the padding were not there, provided its position is the same in both cases. Masking the query side instead would not do this: padded keys would still leak into real queries. The outputs at padded positions are unspecified.

The position has to be the same in both cases, and it has to work for any aspect ratio, so `bucketed` towers do not index the table by grid cell. Let $h$ and $w$ be the numbers of real rows and columns. Patch $(i, j)$ sits at fractional coordinates $(i/h, j/w) \in [0, 1)$, computed in float32 and clamped to $1 - 10^{-6}$. The table side $n$ cuts $[0, 1)$ at the boundaries $k/n$ for $k = 1..n-1$, and $\text{bucket}(x)$ counts the boundaries that are $\le x$. Then
$$\text{id}(i, j) = \text{bucket}(i/h)\cdot n + \text{bucket}(j/w),$$
and masked patches get 0. For a full grid, $h = n$ and $i/n \ge k/n$ exactly when $k \le i$, so $\text{bucket}(i/n) = i$ and the ids are $0..n^2 - 1$: the ordinary table. A smaller image spreads over the whole table instead of using one corner of it.

## 3. Worked example by hand

A table of side $n = 4$ (16 rows, boundaries $1/4, 2/4, 3/4$) and an image whose real patches are $h = 3$ rows by $w = 2$ columns inside a $4 \times 4$ grid.

| Coordinate | Fraction | Boundaries $\le$ it | Bucket |
|---|---|---|---|
| row 0 | $0$ | none | 0 |
| row 1 | $1/3 = 0.333$ | $1/4$ | 1 |
| row 2 | $2/3 = 0.667$ | $1/4, 2/4$ | 2 |
| column 0 | $0$ | none | 0 |
| column 1 | $1/2$ | $1/4, 2/4$ | 2 |

Column 1 lands exactly on the boundary $2/4$; a boundary counts when it is $\le x$, so it goes to the upper bucket. With $\text{id} = 4\,\text{bucket}_\text{row} + \text{bucket}_\text{col}$:

$$\begin{bmatrix} 0 & 2 \\ 4 & 6 \\ 8 & 10 \end{bmatrix}$$

for the real patches, and 0 for the ten padded ones (the whole fourth row reads `[0, 0, 0, 0]`). The image never touches table row 3 or table columns 1 and 3. A full $4 \times 4$ mask gives $0, 1, \dots, 15$.

A second check, for `sincos2d` with $d = 8$: $q = 2$, $\omega = (10000^0, 10000^{-1/2}) = (1, 0.01)$. The patch at $(r, c) = (1, 2)$ gets $[\sin 1, \sin 0.01, \cos 1, \cos 0.01, \sin 2, \sin 0.02, \cos 2, \cos 0.02] = [0.8415, 0.0100, 0.5403, 1.0000, 0.9093, 0.0200, -0.4161, 0.9998]$.

## 4. The interface

```python
@dataclass
class ViTConfig:
    image_size: int; patch_size: int; n_channels: int
    d: int; n_layers: int; n_heads: int; d_ff: int
    pos: Literal["learned", "sincos2d", "rope2d"] = "learned"
    cls_token: bool = False; pre_norm: bool = False
    norm: Literal["layernorm"] = "layernorm"
    act: Literal["gelu", "gelu_tanh", "quick_gelu"] = "gelu"
    ln_eps: float = 1e-6; post_layernorm: bool = True; patch_bias: bool = True
    bucketed: bool = False; rope_base: float = 10000.0; rope_merge: int = 1
    @classmethod
    def from_hf(cls, config: dict) -> "ViTConfig": ...
class PatchEmbed(Conv2d):
    def __init__(self, cfg: ViTConfig, rng: object = None) -> None: ...
class VisionBlock(Module):
    def forward(self, x, mask=None, rope=None) -> Tensor: ...            # [N, T, d] -> [N, T, d]
class ViT(Module):
    def __init__(self, cfg: ViTConfig, rng: object = None) -> None: ...
    def embed(self, pixel_values, patch_mask=None) -> Tensor: ...        # [N, C, H, W] -> [N, T, d]
    def encode(self, h, grid_hw, patch_mask=None, output_hidden_states=False) -> tuple[Tensor, list[Tensor]]: ...
    def forward(self, pixel_values, patch_mask=None, output_hidden_states=False) -> tuple[Tensor, list[Tensor]]: ...
def interpolate_pos_embed(table, old_grid, new_grid) -> NDArray: ...     # float32 [(1 +) nh*nw, d]
def bucketed_position_ids(patch_mask, n_side) -> NDArray: ...            # int64 [..., Hp*Wp]
```

`ViT(cfg)` raises `ValueError` when `n_heads` does not divide `d`, the patch size does not divide `image_size`, or `d` (for `sincos2d`) or `d_head` (for `rope2d`) is not a multiple of 4. `embed` raises it when $H$ or $W$ is not a multiple of $p$, the channel count is wrong, or `patch_mask` is not `[N, Hp, Wp]`. With `rng=None` the parameters come from `PCG32(0).substream("init")`.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example: bucketed positions | unit, smoke | section 3's ids, the full-grid identity, a batch of masks | SmolVLM's positions for any aspect ratio |
| towers match HF | golden | `vit`, `clip`, `siglip`, `idefics3`: every hidden state and the last one within `rtol=1e-5, atol=1e-5` of HF (real patches only for `idefics3`) | any layer `L13.7` taps is HF's |
| position interpolation matches HF | golden | a $4 \times 4$ table serving a $6 \times 5$ grid (31 tokens) matches HF; same-size resize is the identity; the CLS row is bit-exact | running a tower off its training resolution |
| rope2d tower matches Qwen2-VL | golden | `encode` with `pos="rope2d"`, `rope_merge=2` on a $4 \times 6$ grid matches Qwen2-VL's vision blocks | `L13.7`'s `qwen2vl` architecture |
| patch embedding is patchify plus linear | differential | the conv equals patches times the flattened kernel plus bias; no bias when `patch_bias=False` | section 2.1; CLIP's biasless stem |
| hidden states and names | unit | HF parameter names in order, a 17-row table with CLS, $L + 1$ hidden states, `hidden[0] == embed(x)`, `[]` when not asked | loading by name; `vision_feature_layer` |
| sin-cos positions | unit | section 3's sin-cos rows at three grid cells within $10^{-6}$ | the fixed table's layout |
| outputs permute with patches without positions | property | zeroed table: shuffled patches give shuffled outputs | why positions exist |
| padded patches never change real outputs | property | a $3 \times 2$ image padded to $4 \times 4$ with zeros or fives equals the unpadded run on the real patches | mixed-size batches in `L13.7` |
| gradient check | gradcheck | backward to the pixels and the position table against central differences | `L13.5` trains this tower |
| `from_hf` and bad configs | boundary | per-kind switches and overrides; unsupported `model_type`, missing sizes, impossible shapes, wrong inputs all raise `ValueError` | failing at construction, not mid-forward |

## 5. The solution, step by step

### From first principles

1. **An image must become a sequence of $d$-vectors.** By section 2.1 a stride-$p$ conv with a $p \times p$ kernel is exactly one linear map per patch, so `PatchEmbed` is `L13.1`'s `Conv2d` with `k = stride = p`; reshaping its `[N, d, Hp, Wp]` output gives `[N, Hp Wp, d]` in row-major order.
2. **The blocks are the language model's blocks.** Pre-LN attention and MLP with residuals, built from `MultiHeadAttention`'s projections and `scaled_dot_product_attention`. The only additions are a key mask and an optional rotation of $q$ and $k$.
3. **Four towers differ in five switches, so one config with switches.** A CLS token, a patch bias, a pre-LayerNorm, a post-LayerNorm, the activation, and the LayerNorm epsilon. `from_hf` maps `model_type` to those switches, so loading a checkpoint is choosing a row of section 2.2's table.
4. **Positions are needed (permutation equivariance), and each scheme lives in exactly one place.** Learned and sin-cos tables are *added* in `embed`; 2D RoPE *rotates* inside attention in `encode`. Off-size grids resize the learned table bicubically; bucketed towers index it by fractional coordinate.
5. **Padding must be invisible.** Mask keys, not queries (section 2.4); give a real patch the same position it would have unpadded (bucketing by the real $h, w$).
6. **Every hidden state must be right, not only the last.** `hidden` records the input of block 0 and each block's output before `post_layernorm`, so `hidden[-2]` is the penultimate block as Hugging Face defines it.

### The code, annotated

The switch table is the whole difference between the four towers:

```python
HF_KINDS = {
    "vit": dict(cls_token=True, pre_norm=False, post_layernorm=True, patch_bias=True, act="gelu", ln_eps=1e-12),
    "clip_vision_model": dict(cls_token=True, pre_norm=True, post_layernorm=False, patch_bias=False, act="quick_gelu", ln_eps=1e-5),
    "siglip_vision_model": dict(cls_token=False, pre_norm=False, post_layernorm=True, patch_bias=True, act="gelu_tanh", ln_eps=1e-6),
    "idefics3_vision": dict(..., bucketed=True),   # as siglip, plus bucketed positions
    "smolvlm_vision": dict(..., bucketed=True),
}
HF_ACTS = {"gelu": "gelu", "gelu_pytorch_tanh": "gelu_tanh", "gelu_new": "gelu_tanh", "quick_gelu": "quick_gelu"}
```

`from_hf` starts from the row for `model_type`, raises on missing sizes, then lets `layer_norm_eps` and `hidden_act` in the config override the defaults. The activations:

```python
def _act(x: Tensor, kind: str) -> Tensor:
    if kind == "gelu":
        return F.gelu(x)                          # exact x Phi(x)
    if kind == "gelu_tanh":
        return F.gelu(x, approximate="tanh")
    if kind == "quick_gelu":
        return x * F.sigmoid(x * 1.702)           # 1.702 maps the logistic onto Phi
    raise ValueError(f"unknown activation {kind!r}")
```

The two position tables. `sincos2d_table` is section 2.3's formula, rows first, sines before cosines:

```python
def sincos2d_table(grid_hw, d):
    gh, gw = grid_hw
    q = d // 4
    omega = 1.0 / 10000.0 ** (np.arange(q) / q)
    r, c = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")   # row-major, like the patches
    ar = r.reshape(-1, 1) * omega
    ac = c.reshape(-1, 1) * omega
    return np.concatenate([np.sin(ar), np.cos(ar), np.sin(ac), np.cos(ac)], axis=1).astype(np.float32)
```

`interpolate_pos_embed` treats the table as a $d$-channel image:

```python
has_cls = t.shape[0] == oh * ow + 1
cls_row, grid = (t[:1], t[1:]) if has_cls else (t[:0], t)   # the CLS row is never resized
d = t.shape[1]
img = grid.reshape(oh, ow, d).transpose(2, 0, 1)            # [d, oh, ow]: one channel per width dim
out = resize_planar_f32(img, (nh, nw), "bicubic")           # torch's bicubic, a = -0.75
rows = out.transpose(1, 2, 0).reshape(nh * nw, d)           # back to row-major rows
return np.concatenate([cls_row, rows], axis=0).astype(np.float32)
```

`bucketed_position_ids` reproduces Idefics3's float32 arithmetic and section 2.4's rule:

```python
bounds = (np.arange(1, n_side) * (1.0 / n_side)).astype(np.float32)   # k / n for k = 1..n-1
out = np.zeros((n, hp * wp), dtype=np.int64)                          # padded patches stay 0
for b in range(n):
    h = int(m[b, :, 0].sum())          # real rows (the mask is a top-left rectangle)
    w = int(m[b, 0, :].sum())          # real columns
    if h == 0 or w == 0:
        continue
    step_h = np.float32(1.0) / np.float32(h)
    step_w = np.float32(1.0) / np.float32(w)
    fh = np.minimum(np.arange(hp, dtype=np.float32) * step_h, np.float32(1.0 - 1e-6))
    fw = np.minimum(np.arange(wp, dtype=np.float32) * step_w, np.float32(1.0 - 1e-6))
    bh = np.searchsorted(bounds, fh, side="right")   # side="right": a boundary equal to x counts
    bw = np.searchsorted(bounds, fw, side="right")
    ids = (bh[:, None] * n_side + bw[None, :]).reshape(-1)
    keep = m[b].reshape(-1)
    out[b, keep] = ids[keep]
```

The block. Attention is `MultiHeadAttention`'s projections around `scaled_dot_product_attention`, with RoPE on the patches only:

```python
def attend(self, x, mask, rope):
    a = self.self_attn
    q = a.split_heads(a.q_proj(x))
    k = a.split_heads(a.k_proj(x))
    v = a.split_heads(a.v_proj(x))
    if rope is not None:
        cos, sin, s = rope                        # s = 1 when a CLS token sits in front

        def rot(t):
            # A CLS token in front keeps its vector: rotate patches only.
            r = apply_rope(t[:, :, s:], cos, sin, "half")
            return F.concat([t[:, :, :s], r], axis=2) if s else r

        q, k = rot(q), rot(k)                     # both: the score then sees only angle differences
    ctx, _ = scaled_dot_product_attention(q, k, v, mask=mask)
    return a.out_proj(a.merge_heads(ctx))

def forward(self, x, mask=None, rope=None):
    h = x + self.attend(self.layer_norm1(x), mask, rope)   # pre-LN: the residual stream is never normalized
    return h + self.mlp(self.layer_norm2(h))
```

`embed` builds the sequence: patches, then the position scheme, then CLS, then the optional pre-LayerNorm:

```python
pe = e.patch_embedding(x)  # [N, d, Hp, Wp]
h = F.transpose(F.reshape(pe, (n, cfg.d, gh * gw)), 1, 2)  # [N, Hp*Wp, d], row-major
n_side = cfg.image_size // p
if cfg.pos == "learned":
    w = e.position_embedding.weight
    s = 1 if cfg.cls_token else 0                # row 0 belongs to CLS
    if cfg.bucketed:
        mask = np.ones((n, gh, gw), bool) if patch_mask is None else np.asarray(patch_mask, bool)
        h = h + F.embedding(w[s:], bucketed_position_ids(mask, n_side))   # a lookup: gradients reach the table
    elif (gh, gw) == (n_side, n_side):
        h = h + w[s:]                            # the training grid: the table as is
    else:
        h = h + interpolate_pos_embed(w.data[s:], (n_side, n_side), (gh, gw))   # .data: a constant
elif cfg.pos == "sincos2d":
    h = h + sincos2d_table((gh, gw), cfg.d)
if cfg.cls_token:
    cls = F.reshape(e.class_embedding, (1, 1, cfg.d)) + np.zeros((n, 1, cfg.d), dtype=h.data.dtype)
    if cfg.pos == "learned":
        cls = cls + e.position_embedding.weight[0:1]   # CLS gets its own position row
    h = F.concat([cls, h], axis=1)
if self.pre_layrnorm is not None:
    h = self.pre_layrnorm(h)                     # CLIP only
```

`encode` builds the key mask and the rotary tables once and runs the blocks:

```python
mask = None
if patch_mask is not None:
    keep = np.asarray(patch_mask, bool).reshape(n, -1)
    if s:
        keep = np.concatenate([np.ones((n, 1), bool), keep], axis=1)   # CLS is always real
    # ... raise if keep does not cover every token
    # Keys only: a padded patch is invisible to every query.
    mask = keep[:, None, None, :]                # [N, 1 (heads), 1 (queries), T (keys)]
rope = None
if cfg.pos == "rope2d":
    cos, sin = rope2d_cos_sin(grid_hw, cfg.d // cfg.n_heads, cfg.rope_base, cfg.rope_merge)
    rope = (cos, sin, s)
hidden = [h] if output_hidden_states else []     # entry 0: the input of block 0
for layer in self.encoder.layers:
    h = layer(h, mask, rope)
    if output_hidden_states:
        hidden.append(h)                         # before post_layernorm
if self.post_layernorm is not None:
    h = self.post_layernorm(h)
return h, hidden
```

`forward` is `embed` followed by `encode` with the grid computed from the pixel size. Keeping them separate lets `L13.7` feed `encode` a sequence that a processor has already reordered (Qwen2-VL's block-major patches).

Full solution: [`vit.py`](../../../../course/solutions/python/tinyllm/vision/vit.py)

### Why this solution

- **Cost.** Patch embedding is $O(N\,T\,C p^2 d)$; each block is $O(N\,T^2 d + N\,T\,d\,d_\text{ff})$, so attention's $T^2$ dominates for large grids, which is why patch size and token merging matter downstream.
- **One configurable encoder versus four classes.** The four published towers share every line except five switches. One class with a switch table means one tested code path and checkpoints that load by Hugging Face's names; four classes would mean four copies of the same block to keep in sync.
- **Masking keys versus dropping padded tokens.** Dropping tokens would give each image its own sequence length and break batching. A key mask keeps the batch rectangular and makes real outputs exactly independent of padding.
- **Bucketing versus resizing for Idefics3.** Resizing changes the table per aspect ratio and is a constant (no gradient); bucketing is a lookup into the trained table, so it trains, and it is what those checkpoints were trained with.
- **Accepted tradeoff.** `interpolate_pos_embed` reads `w.data`, so an off-resolution forward gives no gradient to the table. That matches inference use; training at a new resolution would need a differentiable resize.

### Explain it back

**Why can a tower with no positions not tell an image from its shuffled patches?**

<details><summary>Answer</summary>

Attention's output for token $i$ is a sum over all keys, which does not depend on their order, and LayerNorm and the MLP act per token. So permuting the input tokens permutes the outputs the same way and changes nothing else. A pooled or CLS output is then identical for both images.

</details>

**Why is the padding mask applied to keys and not to queries?**

<details><summary>Answer</summary>

A real query $i$'s output is a weighted sum over keys. Masking keys sets every padded key's weight to zero, so real outputs see only real tokens. Masking queries would only stop computing outputs *for* padded positions; real queries would still attend to padded keys, and their outputs would change with the padding's values.

</details>

**In section 3, why does column 1 of a 2-column image get bucket 2 and not 1?**

<details><summary>Answer</summary>

Its fraction is $1/2$, exactly the boundary $2/4$. The rule counts boundaries $\le x$, which is `searchsorted(..., side="right")`, so $1/4$ and $2/4$ both count. With `side="left"` it would get bucket 1 and read a different table row from what Idefics3 trained with.

</details>

**Why does `hidden` start with the input of block 0, and why before `post_layernorm`?**

<details><summary>Answer</summary>

That is Hugging Face's `hidden_states` convention, $L + 1$ entries, and VLMs index into it (`hidden[-2]` in LLaVA). Leaving out entry 0 shifts every index by one block. The final norm belongs to the tower's output, not to the per-layer states VLMs were trained to read.

</details>

**Why does `PatchEmbed` need no code of its own beyond calling `Conv2d`?**

<details><summary>Answer</summary>

With kernel equal to stride the windows are disjoint and cover the image, so the conv is one linear map per patch in $(c, u, v)$ order (section 2.1). `L13.1`'s `Conv2d` with `k = stride = p` already computes exactly that, with its backward.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. the CLS token gets no position row | ViT and CLIP states off from block 0 on | `embed` adds `position_embedding.weight[0:1]` to the CLS vector |
| 2. the padding mask hides queries, not keys | real patches read the padding; outputs change with the fill value | `encode` builds the mask as `keep[:, None, None, :]`, the key axis |
| 3. a coordinate on a bucket boundary goes to the lower bucket | column $1/2$ reads table column 1, not 2 | `np.searchsorted(bounds, f, side="right")` counts boundaries $\le x$ |
| 4. the position table resized bilinearly | off-resolution outputs drift from HF | `interpolate_pos_embed` calls `resize_planar_f32(..., "bicubic")` |
| 5. 2D RoPE rotates the queries only | scores depend on absolute position | `attend` applies `rot` to both `q` and `k` |
| 6. `quick_gelu` without its 1.702 | CLIP states off by far more than $10^{-5}$ | `_act` computes `x * F.sigmoid(x * 1.702)` |
| 7. hidden states leave out the block-0 input | $L$ entries, so `hidden[-2]` is one block early | `encode` starts the list with `[h]` before the first block |
| 8. the sin-cos table puts cosines first | a valid-looking table no checkpoint uses | `sincos2d_table` concatenates `sin, cos` per axis, rows first |
| 9. accepting a config with missing sizes | a `KeyError` or a wrong default later | `from_hf` lists every missing key and raises |
| 10. accepting a patch size that does not divide the image | a ragged grid | `ViT.__init__` and `embed` raise on a remainder |
| 11. ignoring `layer_norm_eps` | ViT's $10^{-12}$ silently becomes the default | `from_hf` overrides `ln_eps` from the config |

Two more to keep in mind: the bicubic here is torch's ($a = -0.75$, no antialias), not Pillow's ($a = -0.5$) that `L13.4` uses on pixels; and CLIP's `post_layernorm` in Hugging Face is applied to the pooled CLS vector only, which is why `clip_vision_model` turns it off for the hidden states.

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L13.1` | `PatchEmbed` subclasses `Conv2d` with kernel = stride = $p$ |
| Back | `L13.2`, `L7.3` | `rope2d_cos_sin` builds the angles, `apply_rope` rotates queries and keys |
| Back | `L5.3`, `L5.1` | the block uses `MultiHeadAttention`'s projections and `scaled_dot_product_attention` with a key mask |
| Back | `L0.4`, `L0.2`, `L0.1` | `Linear`, `LayerNorm`, `Embedding`, `ModuleList`; autograd ops on `Tensor` |
| Back | `M12.3` | `resize_planar_f32` resizes the position table |
| Back | `M07.3`, `M06.3` | `normal_init` for the CLS vector and the table; `PCG32(0).substream("init")` |
| Back | `M12.2` | `patchify`, the reshape behind section 2.1 (reading) |
| Forward | `L13.5` | the image tower of the CLIP and SigLIP dual encoder; its text tower stacks `VisionBlock` too |
| Forward | `L13.7` | the vision tower of every VLM architecture; `vision_feature_layer` indexes `hidden` |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `ViT` | HF `SiglipVisionModel`, `CLIPVisionModel` | SDPA and flash attention, a pooling head, fp16 | `transformers/models/siglip/modeling_siglip.py` |
| `ViT` | `timm` `VisionTransformer` | register tokens, LayerScale, drop path, dozens of pretrained variants | `timm/models/vision_transformer.py` |
| `bucketed_position_ids`, `patch_mask` | NaViT, SigLIP 2 NaFlex | native aspect ratio with packed variable-length sequences | Dehghani et al. (2023); Tschannen et al. (2025) |
| `pos="rope2d"` | Qwen2-VL, EVA-02 | rotary positions at any resolution, no table to resize | `transformers/models/qwen2_vl/modeling_qwen2_vl.py` |
| `interpolate_pos_embed` | DINOv2, HF `interpolate_pos_encoding` | multi-resolution training so the resized table is in distribution | Oquab et al. (2023) |
