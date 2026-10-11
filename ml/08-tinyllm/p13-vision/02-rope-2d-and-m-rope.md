<!-- ol:module L13.2 -->
# 2D RoPE and M-RoPE

## Overview

| | |
|---|---|
| **Module** | `L13.2` · build · Python · Pass 12 · 3 to 4 h |
| **You build** | `python/tinyllm/vision/rope2d.py`: `patch_coords`, `rope2d_cos_sin` (axial 2D RoPE for patches), `mrope_position_ids`, `mrope_cos_sin`, `apply_mrope` (Qwen2-VL's M-RoPE) |
| **Contract** | [`course/contracts/py/tinyllm/vision/rope2d.pyi`](../../../course/contracts/py/tinyllm/vision/rope2d.pyi) |
| **Tests** | `course/tests/L13.2/test_rope2d.py` (what they check: section 4), golden values from `course/oracle/L13.2/mrope_hf.py` (transformers 5.19.0 Qwen2-VL: `get_rope_index`, `Qwen2VLRotaryEmbedding`, `apply_rotary_pos_emb`, `get_vision_position_ids`, `Qwen2VLVisionRotaryEmbedding`) in `course/fixtures/L13.2/mrope_hf.npz` |
| **Needs** | `L7.3` `rope_cos_sin`, `apply_rope` · `L0.1` `Tensor` (or `--ref-deps`). Reading: [`M00.2`](../../../math/00-precalculus/02-trig-rotations-and-eulers-formula.md) (rotations compose by adding angles) |
| **Used by** | `L13.3` (the ViT's `pos='rope2d'`) · later `L13.7` (the `qwen2vl` architecture uses both halves) |
| **Milestone** | `MS-L13` (step 4's `tiny-qwen2vl` logits depend on these position ids and rotations) |
| **Optional depth** | Su et al., "RoFormer: Enhanced transformer with rotary position embedding" (2021); Wang et al., "Qwen2-VL: Enhancing vision-language model's perception of the world at any resolution" (2024), section 2.1; Heo et al., "Rotary position embedding for vision transformer" (2024) |

## Key Takeaways

- Axial 2D RoPE gives half the rotary pairs to the row and half to the column, so a score depends only on $(\Delta h, \Delta w)$ (`test_2d_scores_depend_only_on_offsets`).
- Both axes share one ladder of $r/4$ frequencies computed over the half-width $r/2$, and patches are ordered block-major under a merge (`test_vision_rope_matches_hf`).
- In M-RoPE every token has three ids $(t, h, w)$; text sets all three to its position, and the token after an image starts $\max(H/m, W/m)$ later, not one per image token (`test_hand_example`, `test_mrope_position_ids_match_hf`).
- `mrope_section` assigns consecutive pairs of one frequency ladder to the axes $t, h, w$; when $t = h = w$ the result is exactly `L7.3`'s 1D RoPE (`test_apply_mrope_matches_hf`, `test_mrope_reduces_to_1d_rope_on_text`).

## How to work this chapter

```bash
ol start L13.2              # stubs python/tinyllm/vision/rope2d.py into your repo
ol tests L13.2              # read the test catalog first: rung R0, you write no tests here
ol check L13.2              # exit code is the verdict
ol check L13.2 --ref-deps   # only if your L7.3 or L0.1 is not passing yet
ol diff  L13.2              # after passing: your code against the reference
```

---

## 1. Why now

`L7.3`'s RoPE knows one coordinate: a token's index in the sequence. `L13.3` is about to cut an image into a grid of patches and feed them to attention as a flat list. Numbered $p = hW + w$, the patch directly below another is $W$ positions away, and that distance changes with every image width: a model trained at one resolution reads garbage offsets at another, which is exactly what any-resolution models such as Qwen2-VL cannot afford. A language model that reads images has a second problem: an image of a thousand tokens would consume a thousand 1D positions between the words before it and the words after it. This module fixes both. Axial 2D RoPE rotates by row and by column separately (the ViT's `pos='rope2d'`), and Qwen2-VL's M-RoPE gives every token of the language model three position ids, so an image occupies a small square of positions. `L13.7`'s `qwen2vl` architecture needs both, bit-compatible with Hugging Face.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $r$ | rotary width (`d_rot`), $r/2$ pairs | int, multiple of 4 for 2D |
| $\omega_i$ | frequency of pair $i$ (`inv_freq`) | `float[r/2]` (`float32[r/4]` for 2D) |
| $(h, w)$ | patch row and column | `int64[H·W, 2]` |
| $m$ | spatial merge: $m \times m$ patches become one LM token | int |
| $(t, h, w)$ | M-RoPE position ids of each token | `int64[3, T]` |
| $s$ | position at which an image starts | int |
| cos, sin | one value per pair (`L7.3`'s convention) | `float32[·, r/2]` |

### 2.1 One rotation, one coordinate

Write pair $i$ of a query as the complex number $q_i = a + \mathrm{i}b$. Rotating $(a, b)$ by angle $\theta$ gives $(a\cos\theta - b\sin\theta,\ a\sin\theta + b\cos\theta)$, which is $q_i e^{\mathrm{i}\theta}$ (`M00.2`). The dot product of two real pairs is $\operatorname{Re}(q_i \bar k_i)$. If the query at position $p$ rotates by $p\,\omega_i$ and the key at $p'$ by $p'\omega_i$, the pair contributes
$$\operatorname{Re}\big(q_i e^{\mathrm{i}p\omega_i}\ \overline{k_i e^{\mathrm{i}p'\omega_i}}\big) = \operatorname{Re}\big(q_i \bar k_i\, e^{\mathrm{i}(p - p')\omega_i}\big),$$
because conjugation negates the key's angle and angles add. The score is the sum over pairs, so it depends on positions only through $p - p'$. That argument is per pair and never asks where a pair's angle comes from. Everything below is a choice of which coordinate drives which pair.

### 2.2 Axial 2D RoPE for patches

Give the first $r/4$ pairs the row and the last $r/4$ pairs the column. For $i < r/4$,
$$\theta_i = h\,\omega_i, \qquad \theta_{r/4 + i} = w\,\omega_i, \qquad \omega_i = \text{base}^{-2i/(r/2)}.$$
By 2.1 the score of patch $(h, w)$ against $(h', w')$ is
$$\sum_{i < r/4} \operatorname{Re}\big(q_i \bar k_i\, e^{\mathrm{i}(h - h')\omega_i}\big) + \sum_{i < r/4} \operatorname{Re}\big(q_{r/4+i}\, \bar k_{r/4+i}\, e^{\mathrm{i}(w - w')\omega_i}\big),$$
a function of $(h - h', w - w')$ only, whatever the image size. The ladder is `L7.3`'s ladder for a head of width $r/2$, used twice: each axis gets the full range of frequencies from 1 down toward $1/\text{base}$. Spreading one ladder over the whole width $r$ would give neither axis the right range. $r$ must be a multiple of 4. The tables are `float32[H·W, r/2]`, rows first then columns, each half computed by `rope_cos_sin`, and `apply_rope(x, cos, sin, "half")` applies them, where pair $j$ is $(x_j, x_{j + r/2})$. This is Qwen2-VL's vision tower.

**Patch order.** Row $n$ of the table belongs to the $n$-th patch, so the order is part of the contract. With `merge = 1` it is row-major. With `merge = m` the grid is cut into $m \times m$ blocks, blocks in row-major order and row-major inside each block. Qwen2-VL orders patches this way so one block's $m^2$ patches are adjacent and its `PatchMerger` can concatenate them into one token.

### 2.3 M-RoPE position ids

In the language model each token gets three ids. A text token at 1D position $p$ gets $(p, p, p)$. An image of $t \times H \times W$ patches arrives as $t \cdot (H/m) \cdot (W/m)$ image tokens, one per merge block. If it starts at position $s$, token $(t_i, h_i, w_i)$ of that grid, in row-major $(t, h, w)$ order, gets
$$(s + t_i,\ s + h_i,\ s + w_i),$$
and the next token starts at $s + \max(H/m, W/m)$. Two things follow. All ids are offset by $s$, so by 2.1 the image looks the same wherever it sits in the prompt. And the image advances the position by its side length, not its token count: a $32 \times 32$ token image costs 32 positions instead of 1024, while the text after it still has ids larger than every row and column id the image used. For a single frame ($t = 1$) that is the image's largest id plus one. The $i$-th run of image tokens belongs to row $i$ of `image_grid_thw`; two images may be adjacent, so runs are split by the expected lengths, not by gaps.

### 2.4 Sections: three axes on one ladder

The LM keeps `L7.3`'s single ladder $\omega_0, \dots, \omega_{r/2-1}$ over the whole head. `mrope_section` cuts the $r/2$ pairs into consecutive sections; section $j$ takes its position from axis $j \bmod 3$ (time, row, column, time, ...). With $a(i)$ the axis of the section holding pair $i$,
$$\theta_i = \text{pos}[a(i)]\ \omega_i.$$
Qwen2-VL uses sections $[16, 24, 24]$ on a head of 128: time gets the 16 fastest pairs, rows the next 24, columns the slowest 24. Unlike 2.2, each pair keeps its own frequency and only its coordinate changes. So when $t = h = w = p$ every pair's angle is $p\,\omega_i$: M-RoPE on text is `L7.3`'s RoPE, bit for bit, and a `qwen2vl` model with no image is a plain Qwen2 language model. The sections must sum to $r/2$.

`apply_mrope` rotates in the "half" layout through `apply_rope`. The map is linear in $x$ and orthogonal, so its backward is the transpose: the upstream gradient rotated by $-\theta$.

## 3. Worked example by hand

**Position ids.** The sequence is two text tokens, one image of $1 \times 4 \times 4$ patches with $m = 2$, then two text tokens: `[5, 6, IMG, IMG, IMG, IMG, 8, 9]`. The image needs $1 \cdot 2 \cdot 2 = 4$ tokens. Text takes positions 0 and 1, so $s = 2$; the grid $(h_i, w_i)$ runs $(0,0), (0,1), (1,0), (1,1)$; the next token starts at $2 + \max(2, 2) = 4$.

| Token | 5 | 6 | IMG | IMG | IMG | IMG | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| $t$ | 0 | 1 | 2 | 2 | 2 | 2 | 4 | 5 |
| $h$ | 0 | 1 | 2 | 2 | 3 | 3 | 4 | 5 |
| $w$ | 0 | 1 | 2 | 3 | 2 | 3 | 4 | 5 |

Counting image tokens instead would put token 8 at position 6.

**Patch order.** `patch_coords((2, 4), 2)` has two $2 \times 2$ blocks side by side: $(0,0), (0,1), (1,0), (1,1)$, then $(0,2), (0,3), (1,2), (1,3)$. Without a merge, `patch_coords((2, 3))` is row-major: $(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)$.

**One rotation.** Take $r = 4$ on a $2 \times 2$ grid. The ladder over the half-width 2 has one frequency, $\omega_0 = 10000^0 = 1$, so the patch at $(h, w)$ has $\cos = [\cos h, \cos w]$. For patch $(1, 0)$: $\cos = [0.5403, 1]$, $\sin = [0.8415, 0]$. In the "half" layout $x = [1, 1, 0, 0]$ holds the pairs $(x_0, x_2) = (1, 0)$ and $(x_1, x_3) = (1, 0)$. The row pair rotates by 1 rad to $(0.5403, 0.8415)$ and the column pair by 0, so the result is $[0.5403, 1, 0.8415, 0]$.

## 4. The interface

```python
def patch_coords(grid_hw: tuple[int, int], merge: int = 1) -> NDArray: ...          # int64 [H*W, 2]
def rope2d_cos_sin(grid_hw, d_rot: int, base: float = 10000.0, merge: int = 1) -> tuple[NDArray, NDArray]: ...
                                                                                     # float32 [H*W, d_rot/2] each
def mrope_position_ids(input_ids, image_grid_thw, image_token_id: int, spatial_merge: int = 2) -> NDArray: ...
                                                                                     # int64 [3, T]
def mrope_cos_sin(pos3, inv_freq, mrope_section: Sequence[int]) -> tuple[NDArray, NDArray]: ...   # float32 [T, r/2]
def apply_mrope(x: Tensor, pos3, inv_freq, mrope_section: Sequence[int]) -> Tensor: ...           # [..., T, r]
```

`image_grid_thw` is `[n_images, 3]` in patches $(t, H, W)$. `ValueError` for: $H$ or $W$ below 1, a merge that does not divide both, `d_rot` not a positive multiple of 4, image tokens that do not split into runs of exactly the expected lengths (too few, or a grid with no run), `pos3` not `[3, T]`, sections that do not sum to `len(inv_freq)`, or an `x` whose last axis is not `2 * len(inv_freq)`.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit, smoke | section 3's id table and both patch orders, exactly | you and the test agree on the resume rule |
| `test_mrope_position_ids_match_hf` | golden | `get_rope_index` on a tall image, a wide one, and a two-frame one directly after it, integer-exact | `L13.7` feeds these ids to every attention layer |
| `test_apply_mrope_matches_hf` | golden | queries and keys rotated with sections $[2, 3, 3]$ within $10^{-5}$ of HF's `apply_rotary_pos_emb` | the section-to-axis mapping |
| `test_vision_rope_matches_hf` | golden | block-major positions exact; cos and sin within $10^{-6}$ of the vision rotary tables (HF repeats them to the full head); rotated $q$, $k$ within $10^{-5}$ | `L13.3`'s `rope2d` ViT |
| `test_2d_scores_depend_only_on_offsets` | property | shifting both patches by the same $(dh, dw)$ changes a score by less than $10^{-4}$; a row step differs from a column step | resolution-independent attention |
| `test_mrope_reduces_to_1d_rope_on_text` | property | text-only ids are $(p, p, p)$ and the output equals `L7.3`'s with tolerance 0 | a `qwen2vl` model without images is Qwen2 |
| `test_gradcheck_apply_mrope` | gradcheck | the backward to $x$ against frozen central differences | M-RoPE sits inside trained attention |
| `test_rejects_bad_inputs` | boundary | every `ValueError` listed above | wrong splits fail loudly, not silently |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. text after an image resuming after its token count | every later token rotated too far; long prompts run out of positions | `test_hand_example`, `test_mrope_position_ids_match_hf` (mutant `s01`) |
| 2. sections taking their axes in reverse order | columns and time swapped; text-only prompts still look right | `test_apply_mrope_matches_hf` (mutant `s02`) |
| 3. the 2D ladder spread over the whole rotary width | each axis gets the wrong frequencies; pretrained towers break | `test_vision_rope_matches_hf` (mutant `s03`) |
| 4. row coordinates left row-major under a merge | tables no longer line up with the merged patches | `test_hand_example`, `test_vision_rope_matches_hf` (mutant `s04`) |
| 5. image time ids not offset by the start position | every image looks like it sits at time 0 | `test_hand_example`, `test_mrope_position_ids_match_hf` (mutant `s05`) |
| 6. columns before rows in the 2D tables | the image is read transposed | `test_vision_rope_matches_hf`, `test_2d_scores_depend_only_on_offsets` (mutant `s06`) |
| accepting a rotary width that is only even | tables whose width no longer matches the pairs they rotate | `test_rejects_bad_inputs` (mutant `m01`) |
| accepting sections that sum to less than $r/2$ | trailing pairs with no axis | `test_rejects_bad_inputs` (mutant `m02`) |
| not checking that the merge divides the grid | a fractional token grid, silently floored | `test_rejects_bad_inputs` (mutant `m03`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L7.3` | `rope_cos_sin` computes every table (float32 angle, then cos and sin); `apply_rope` in the "half" layout does every rotation |
| Back | `L0.1` | `apply_mrope` takes and returns a `Tensor`, differentiable in $x$ |
| Back | `M00.2` | rotations compose by adding angles: the whole of section 2.1 (reading) |
| Forward | `L13.3` | the ViT's `pos='rope2d'` rotates queries and keys with `rope2d_cos_sin` |
| Forward | `L13.7` | the `qwen2vl` architecture: `rope2d_cos_sin(..., merge=2)` in the vision tower, `mrope_position_ids` and `apply_mrope` in the language model (with B14's VLM group) |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `mrope_position_ids` | HF `Qwen2VLModel.get_rope_index` | batches with padding and attention masks, video runs | `transformers/models/qwen2_vl/modeling_qwen2_vl.py` |
| `apply_mrope` | HF `Qwen2VLRotaryEmbedding` with `apply_rotary_pos_emb` | cos and sin cached per batch, fp16 | the same file |
| `rope2d_cos_sin` | HF `Qwen2VLVisionRotaryEmbedding` | tables for a batch of images of different sizes in one pass | the same file |
| M-RoPE's time axis | Qwen2.5-VL | time ids tied to absolute seconds, so video frame rate can vary | Bai et al., "Qwen2.5-VL technical report" (2025) |
| axial 2D RoPE | RoPE-Mixed, EVA-02 | learned mixed-axis frequencies, so diagonal offsets get their own rotations | Heo et al. (2024); Fang et al., "EVA-02" (2023) |
