<!-- ol:module L13.3 -->
# Vision transformer encoder

## Overview

| | |
|---|---|
| **Module** | `L13.3` · build · Python · Pass 12 · 5 to 7 h |
| **You build** | `python/tinyllm/vision/vit.py`: `ViTConfig` (with `from_hf`), `PatchEmbed`, `VisionBlock`, `ViT` (`embed`, `encode`, `forward`), `interpolate_pos_embed`, `bucketed_position_ids` |
| **Contract** | [`course/contracts/py/tinyllm/vision/vit.pyi`](../../../../course/contracts/py/tinyllm/vision/vit.pyi) |
| **Tests** | `course/tests/L13.3/test_vit.py` (what they check: section 4), golden values in `course/fixtures/L13.3/vit_hf.npz` from `course/oracle/L13.3/vit_hf.py` (tiny random HF `ViTModel`, `CLIPVisionModel`, `SiglipVisionModel`, `Idefics3VisionTransformer`, and Qwen2-VL vision blocks, transformers 5.19.0) |
| **Needs** | `L13.1` `Conv2d` · `L13.2` `rope2d_cos_sin` · `L7.3` `apply_rope` · `L5.3` `MultiHeadAttention` · `L5.1` `scaled_dot_product_attention` · `L0.4` `Linear`, `LayerNorm`, `Embedding`, `ModuleList`, `Module` · `L0.2` ops · `L0.1` `Tensor` · `M12.3` `resize_planar_f32` · `M07.3` `normal_init` · `M06.3` `PCG32` (or `--ref-deps`). Reading: `M12.2` `patchify` |
| **Used by** | `L13.5` (the image tower of the dual encoder) · `L13.7` (the vision tower of every VLM architecture, tapped through `hidden`) |
| **Milestone** | `MS-L13` step 4 (tiny VLM logits within $10^{-5}$; they start from this tower's hidden states) |
| **Optional depth** | Dosovitskiy et al., "An image is worth 16x16 words" (2021), section 3.1; Radford et al., "Learning transferable visual models from natural language supervision" (2021), for CLIP's tower; Zhai et al., "Sigmoid loss for language image pre-training" (2023), for SigLIP's; Dehghani et al., "Patch n' Pack: NaViT" (2023), for fractional positions and padded patches |

## Key Takeaways

- A convolution whose kernel equals its stride is one linear map per patch: `PatchEmbed` equals cutting the image into patches, flattening each in $(c, u, v)$ order, and multiplying by the flattened kernel (`test_patch_embed_is_patchify_plus_linear`).
- One configurable encoder reproduces four published towers (ViT, CLIP, SigLIP, Idefics3) at every hidden state within $10^{-5}$; they differ only in a CLS token, a patch bias, where the extra LayerNorms sit, and the activation (`test_towers_match_hf`).
- Without positions the outputs permute with the patches, so position information has to be added: a learned table (resized bicubically for a new grid, or bucketed by fractional coordinate), a fixed 2D sin-cos table, or 2D RoPE inside attention (`test_outputs_permute_with_patches_without_positions`, `test_position_interpolation_matches_hf`, `test_sincos2d_positions`, `test_rope2d_tower_matches_qwen2vl`).
- Padding is hidden on the key side of attention, so a real patch's output is the same whatever the padded patches hold (`test_padded_patches_never_change_real_outputs`).
- `hidden` holds the $L + 1$ block inputs and outputs before the final norm; `L13.7` picks one with `vision_feature_layer` (`test_hidden_states_and_names`).

## How to work this chapter

```bash
ol start L13.3              # stubs python/tinyllm/vision/vit.py into your repo
ol tests L13.3              # read the test catalog first: rung R0, you write no tests here
ol check L13.3              # exit code is the verdict
ol check L13.3 --ref-deps   # only if one of your Needs modules is not passing yet
ol diff  L13.3              # after passing: your code against the reference
```

---

## 1. Why now

`L13.1` gave you convolution and `L13.2` gave you 2D rotary positions, but the transformer from Parts 5 and 7 still only reads token ids. This module makes an image into a sequence that the same blocks can read. Nothing in the blocks is new: attention from `L5.3`, LayerNorm and MLPs from Part 0. What is new is the front (patches and a position scheme) and matching published checkpoints exactly. Four towers that look alike differ in small switches (a CLS token, a patch bias, a pre-LayerNorm, `quick_gelu`), and a wrong switch still loads and still runs: it just produces features the language model was never trained on. `L13.5` trains this tower contrastively and `L13.7` puts it in front of a language model, where LLaVA reads `hidden[-2]`, so every hidden state has to be right, not only the last.

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

for the real patches, and 0 for the ten padded ones (the whole fourth row reads `[0, 0, 0, 0]`). The image never touches table row 3 or table columns 1 and 3. A full $4 \times 4$ mask gives $0, 1, \dots, 15$ (`test_hand_example_bucketed_positions`).

A second check, for `sincos2d` with $d = 8$: $q = 2$, $\omega = (10000^0, 10000^{-1/2}) = (1, 0.01)$. The patch at $(r, c) = (1, 2)$ gets $[\sin 1, \sin 0.01, \cos 1, \cos 0.01, \sin 2, \sin 0.02, \cos 2, \cos 0.02] = [0.8415, 0.0100, 0.5403, 1.0000, 0.9093, 0.0200, -0.4161, 0.9998]$ (`test_sincos2d_positions`).

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_bucketed_positions` | unit, smoke | section 3's ids, the full-grid identity, a batch of masks | SmolVLM's positions for any aspect ratio |
| `test_towers_match_hf` | golden | `vit`, `clip`, `siglip`, `idefics3`: every hidden state and the last one within `rtol=1e-5, atol=1e-5` of HF (real patches only for `idefics3`) | any layer `L13.7` taps is HF's |
| `test_position_interpolation_matches_hf` | golden | a $4 \times 4$ table serving a $6 \times 5$ grid (31 tokens) matches HF; same-size resize is the identity; the CLS row is bit-exact | running a tower off its training resolution |
| `test_rope2d_tower_matches_qwen2vl` | golden | `encode` with `pos="rope2d"`, `rope_merge=2` on a $4 \times 6$ grid matches Qwen2-VL's vision blocks | `L13.7`'s `qwen2vl` architecture |
| `test_patch_embed_is_patchify_plus_linear` | differential | the conv equals patches times the flattened kernel plus bias; no bias when `patch_bias=False` | section 2.1; CLIP's biasless stem |
| `test_hidden_states_and_names` | unit | HF parameter names in order, a 17-row table with CLS, $L + 1$ hidden states, `hidden[0] == embed(x)`, `[]` when not asked | loading by name; `vision_feature_layer` |
| `test_sincos2d_positions` | unit | section 3's sin-cos rows at three grid cells within $10^{-6}$ | the fixed table's layout |
| `test_outputs_permute_with_patches_without_positions` | property | zeroed table: shuffled patches give shuffled outputs | why positions exist |
| `test_padded_patches_never_change_real_outputs` | property | a $3 \times 2$ image padded to $4 \times 4$ with zeros or fives equals the unpadded run on the real patches | mixed-size batches in `L13.7` |
| `test_gradcheck_vit` | gradcheck | backward to the pixels and the position table against central differences | `L13.5` trains this tower |
| `test_from_hf_and_bad_configs` | boundary | per-kind switches and overrides; unsupported `model_type`, missing sizes, impossible shapes, wrong inputs all raise `ValueError` | failing at construction, not mid-forward |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. the CLS token gets no position row | ViT and CLIP states off from block 0 on | `test_towers_match_hf` (mutant `s01`) |
| 2. the padding mask hides queries, not keys | real patches read the padding; outputs change with the fill value | `test_padded_patches_never_change_real_outputs`, `test_towers_match_hf` (mutant `s02`) |
| 3. a coordinate on a bucket boundary goes to the lower bucket | column $1/2$ reads table column 1, not 2 | `test_hand_example_bucketed_positions` (mutant `s03`) |
| 4. the position table resized bilinearly | off-resolution outputs drift from HF | `test_position_interpolation_matches_hf` (mutant `s04`) |
| 5. 2D RoPE rotates the queries only | scores depend on absolute position | `test_rope2d_tower_matches_qwen2vl` (mutant `s05`) |
| 6. `quick_gelu` without its 1.702 | CLIP states off by far more than $10^{-5}$ | `test_towers_match_hf` (mutant `s06`) |
| 7. hidden states leave out the block-0 input | $L$ entries, so `hidden[-2]` is one block early | `test_hidden_states_and_names`, `test_towers_match_hf` (mutant `s07`) |
| 8. the sin-cos table puts cosines first | a valid-looking table no checkpoint uses | `test_sincos2d_positions` (mutant `s08`) |
| accepting a config with missing sizes | a `KeyError` or a wrong default later | `test_from_hf_and_bad_configs` (mutant `m01`) |
| accepting a patch size that does not divide the image | a ragged grid | `test_from_hf_and_bad_configs` (mutant `m02`) |
| ignoring `layer_norm_eps` | ViT's $10^{-12}$ silently becomes the default | `test_from_hf_and_bad_configs` (mutant `m03`) |

Two more that no single mutant names: the bicubic here is torch's ($a = -0.75$, no antialias), not Pillow's ($a = -0.5$) that `L13.4` uses on pixels; and CLIP's `post_layernorm` in Hugging Face is applied to the pooled CLS vector only, which is why `clip_vision_model` turns it off for the hidden states.

## 6. Where it's used next

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

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `ViT` | HF `SiglipVisionModel`, `CLIPVisionModel` | SDPA and flash attention, a pooling head, fp16 | `transformers/models/siglip/modeling_siglip.py` |
| `ViT` | `timm` `VisionTransformer` | register tokens, LayerScale, drop path, dozens of pretrained variants | `timm/models/vision_transformer.py` |
| `bucketed_position_ids`, `patch_mask` | NaViT, SigLIP 2 NaFlex | native aspect ratio with packed variable-length sequences | Dehghani et al. (2023); Tschannen et al. (2025) |
| `pos="rope2d"` | Qwen2-VL, EVA-02 | rotary positions at any resolution, no table to resize | `transformers/models/qwen2_vl/modeling_qwen2_vl.py` |
| `interpolate_pos_embed` | DINOv2, HF `interpolate_pos_encoding` | multi-resolution training so the resized table is in distribution | Oquab et al. (2023) |
