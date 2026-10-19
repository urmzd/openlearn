# 2D RoPE and M-RoPE

## Overview

| | |
|---|---|
| **Module** | `L13.2` · build · Python · Pass 12 |
| **Solution** | [`rope2d.py`](../../../../course/solutions/python/tinyllm/vision/rope2d.py): `patch_coords`, `rope2d_cos_sin` (axial 2D RoPE for patches), `mrope_position_ids`, `mrope_cos_sin`, `apply_mrope` (Qwen2-VL's M-RoPE) |
| **Contract** | [`course/contracts/py/tinyllm/vision/rope2d.pyi`](../../../../course/contracts/py/tinyllm/vision/rope2d.pyi) |
| **Needs** | `L7.3` `rope_cos_sin`, `apply_rope` · `L0.1` `Tensor`. Reading: [`M00.2`](../../../tracks/math/00-precalculus/02-trig-rotations-and-eulers-formula.md) (rotations compose by adding angles) |
| **Used by** | `L13.3` (the ViT's `pos='rope2d'`) · later `L13.7` (the `qwen2vl` architecture uses both halves; a `tiny-qwen2vl` model's logits depend on these position ids and rotations) |
| **Optional depth** | Su et al., "RoFormer: Enhanced transformer with rotary position embedding" (2021); Wang et al., "Qwen2-VL: Enhancing vision-language model's perception of the world at any resolution" (2024), section 2.1; Heo et al., "Rotary position embedding for vision transformer" (2024) |

## Key Takeaways

- Axial 2D RoPE gives half the rotary pairs to the row and half to the column, so a score depends only on $(\Delta h, \Delta w)$.
- Both axes share one ladder of $r/4$ frequencies computed over the half-width $r/2$, and patches are ordered block-major under a merge.
- In M-RoPE every token has three ids $(t, h, w)$; text sets all three to its position, and the token after an image starts $\max(H/m, W/m)$ later, not one per image token.
- `mrope_section` assigns consecutive pairs of one frequency ladder to the axes $t, h, w$; when $t = h = w$ the result is exactly `L7.3`'s 1D RoPE.

## How to read this chapter

Section 1 says why one position coordinate is no longer enough. Section 2 derives axial 2D RoPE and M-RoPE from the one fact that rotations compose by adding angles, and section 3 works the position ids and one rotation by hand. Section 4 fixes the interface. Section 5 walks through `rope2d.py` and explains why each line is the right one. Try to predict each step of the solution before reading it.

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

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example | unit, smoke | section 3's id table and both patch orders, exactly | the resume rule is pinned |
| M-RoPE position ids match HF | golden | `get_rope_index` on a tall image, a wide one, and a two-frame one directly after it, integer-exact | `L13.7` feeds these ids to every attention layer |
| `apply_mrope` matches HF | golden | queries and keys rotated with sections $[2, 3, 3]$ within $10^{-5}$ of HF's `apply_rotary_pos_emb` | the section-to-axis mapping |
| vision RoPE matches HF | golden | block-major positions exact; cos and sin within $10^{-6}$ of the vision rotary tables (HF repeats them to the full head); rotated $q$, $k$ within $10^{-5}$ | `L13.3`'s `rope2d` ViT |
| 2D scores depend only on offsets | property | shifting both patches by the same $(dh, dw)$ changes a score by less than $10^{-4}$; a row step differs from a column step | resolution-independent attention |
| M-RoPE reduces to 1D RoPE on text | property | text-only ids are $(p, p, p)$ and the output equals `L7.3`'s with tolerance 0 | a `qwen2vl` model without images is Qwen2 |
| gradient check, `apply_mrope` | gradcheck | the backward to $x$ against central differences | M-RoPE sits inside trained attention |
| bad inputs rejected | boundary | every `ValueError` listed above | wrong splits fail loudly, not silently |

## 5. The solution, step by step

### From first principles

1. **Reuse `L7.3` for every table.** Section 2.1 shows that RoPE's relative-position property holds pair by pair, whatever coordinate drives the pair. So the solution never writes a new rotation: it decides, for each pair, which coordinate and which frequency to use, and hands the resulting angles to `L7.3`'s `rope_cos_sin` and `apply_rope`.
2. **The patch order is data, so compute it explicitly.** Row $n$ of every table belongs to patch $n$. `patch_coords` returns the $(h, w)$ of each patch in the order the vision tower sees them, and both 2D tables are built from it, so order and tables cannot disagree.
3. **Block-major order is a reshape and a transpose.** A row-major $H \times W$ grid reshaped to $[H/m, m, W/m, m]$ has axes (block row, row in block, block column, column in block). Swapping the middle two gives (block row, block column, row in block, column in block), which flattened is block-major.
4. **For 2D, one ladder over the half-width, used twice.** Each axis must get the full range of frequencies (section 2.2), so the ladder is `L7.3`'s ladder for width $r/2$, and the table is the row table next to the column table.
5. **For M-RoPE ids, one scan with a running position.** Text increments the position by 1; an image run fills a $t \times (H/m) \times (W/m)$ grid offset by the current position and then advances it by $\max(H/m, W/m)$. Runs are consumed by their expected length, so adjacent images split correctly.
6. **For M-RoPE tables, compute every axis at every frequency, then select.** `rope_cos_sin` on the `[3, T]` ids gives `[3, T, r/2]`; pair $i$ then reads axis $a(i)$. With $t = h = w$ the three slices are identical, so the selection returns exactly the 1D table.

### The code, annotated

`patch_coords` is step 3:

```python
def patch_coords(grid_hw, merge=1):
    h, w = (int(v) for v in grid_hw)
    if h < 1 or w < 1 or merge < 1 or h % merge or w % merge:
        raise ValueError(...)                          # a merge must tile the grid exactly
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")   # row-major
    if merge > 1:
        # Blocks of m x m patches, block-major: [H/m, m, W/m, m] -> [H/m, W/m, m, m].
        shape = (h // merge, merge, w // merge, merge)
        rows = rows.reshape(shape).transpose(0, 2, 1, 3)
        cols = cols.reshape(shape).transpose(0, 2, 1, 3)   # the same permutation for both coordinates
    return np.stack([rows.ravel(), cols.ravel()], axis=-1).astype(np.int64)
```

`rope2d_cos_sin` is step 4. The ladder is built in float32 to match the Hugging Face vision tower bit for bit:

```python
def rope2d_cos_sin(grid_hw, d_rot, base=10000.0, merge=1):
    if d_rot < 4 or d_rot % 4:
        raise ValueError(...)                          # r/2 pairs split evenly between two axes
    xy = patch_coords(grid_hw, merge)
    half = d_rot // 2  # pairs; each axis owns half of them
    # One ladder for both axes, over the half-width: base^(-2i / half), i.e. r/4 frequencies
    inv = 1.0 / (base ** (np.arange(0, half, 2, dtype=np.float32) / np.float32(half)))
    ch, sh = rope_cos_sin(xy[:, 0], inv)               # rows drive the first r/4 pairs
    cw, sw = rope_cos_sin(xy[:, 1], inv)               # columns the last r/4
    return np.concatenate([ch, cw], axis=-1), np.concatenate([sh, sw], axis=-1)
```

`mrope_position_ids` is step 5, with every malformed split rejected:

```python
pos = np.zeros((3, len(ids)), dtype=np.int64)
cur, i, img = 0, 0, 0                                  # next position, token index, image index
while i < len(ids):
    if ids[i] != image_token_id:
        pos[:, i] = cur                                # text: (p, p, p)
        cur += 1
        i += 1
        continue
    # ... raise if there is no grid left for this run
    t, gh, gw = (int(v) for v in grids[img])
    # ... raise if the merge does not divide the grid
    lh, lw = gh // m, gw // m                          # the token grid after merging
    n = t * lh * lw
    if i + n > len(ids) or np.any(ids[i : i + n] != image_token_id):
        raise ValueError(...)                          # the run must be exactly n image tokens
    tt, hh, ww = np.meshgrid(np.arange(t), np.arange(lh), np.arange(lw), indexing="ij")
    pos[0, i : i + n] = cur + tt.ravel()               # every axis offset by the start position
    pos[1, i : i + n] = cur + hh.ravel()
    pos[2, i : i + n] = cur + ww.ravel()
    # Text resumes past the image: its widest spatial extent.
    cur += max(lh, lw)
    i += n                                             # consume exactly this image's tokens
    img += 1
if img != len(grids):
    raise ValueError(...)                              # a grid with no run
```

`mrope_cos_sin` is step 6:

```python
cos, sin = rope_cos_sin(p, f)  # [3, T, r/2]: every axis at every frequency
axis = np.repeat(np.arange(len(sec)) % 3, sec)  # which axis drives pair i: t, h, w, t, ...
cols = np.arange(len(f))
# fancy-index (axis[i], :, i) for each pair i -> [r/2, T], transposed to [T, r/2]
return cos[axis, :, cols].T.copy(), sin[axis, :, cols].T.copy()
```

`apply_mrope` checks the width and calls `apply_rope(x, cos, sin, "half")`, so the backward is `L7.3`'s: the gradient rotated by $-\theta$.

Full solution: [`rope2d.py`](../../../../course/solutions/python/tinyllm/vision/rope2d.py)

### Why this solution

- **Cost.** Tables are $O(T \cdot r)$, built once per image or prompt; `mrope_cos_sin` computes three times the needed table and discards two thirds, which is cheap next to attention and keeps the code a single gather. The id scan is $O(T)$.
- **Composing `L7.3` versus new rotation code.** Every rotation goes through the already-tested `apply_rope`, so the only new correctness questions are which coordinate and frequency each pair gets, which is exactly what the golden comparisons pin.
- **Axial versus flattened 1D positions.** Flattened positions make the vertical neighbour $W$ away, so offsets change with image width. Axial RoPE makes the score a function of $(\Delta h, \Delta w)$ at every resolution, at the price of no special treatment for diagonal offsets (the RoPE-Mixed variant learns those).
- **Resume by side length versus token count.** Token count would burn $H W/m^2$ positions per image; side length burns $\max(H, W)/m$ and still keeps every later text id above every image id.

### Explain it back

**Why does a text token after a $1 \times 4 \times 4$ image with $m = 2$ starting at $s = 2$ get position 4 and not 6?**

<details><summary>Answer</summary>

The image's token grid is $2 \times 2$, so its row and column ids run from 2 to 3. Text resumes at $s + \max(2, 2) = 4$, one past the largest id the image used. Counting its 4 tokens would give 6, wasting positions and rotating every later token further than the pretrained model expects.

</details>

**Why does M-RoPE on text-only input equal 1D RoPE exactly, with tolerance 0?**

<details><summary>Answer</summary>

Each pair keeps its own frequency $\omega_i$ and only chooses which axis supplies the position. For text all three axes hold $p$, so every pair's angle is $p\,\omega_i$, the same float32 computation `L7.3` does, and the selection copies values without arithmetic.

</details>

**Why is the 2D ladder built over $r/2$ instead of $r$?**

<details><summary>Answer</summary>

Each axis owns $r/4$ pairs and should see frequencies from 1 down toward $1/\text{base}$, like a head of width $r/2$. A ladder over $r$ would give the row axis only the fast half and the column axis only the slow half of the range, and would not match pretrained vision towers.

</details>

**What does the block-major order buy, and what would break without it?**

<details><summary>Answer</summary>

It puts each $m \times m$ block's patches next to each other, so the `PatchMerger` can concatenate $m^2$ consecutive rows into one token. If the coordinates stayed row-major while the patches were block-major, row $n$ of the table would rotate a different patch from the one at position $n$.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. text after an image resuming after its token count | every later token rotated too far; long prompts run out of positions | `mrope_position_ids` advances `cur` by `max(lh, lw)`, not `n` |
| 2. sections taking their axes in reverse order | columns and time swapped; text-only prompts still look right | `axis = np.repeat(np.arange(len(sec)) % 3, sec)`: time, row, column in order |
| 3. the 2D ladder spread over the whole rotary width | each axis gets the wrong frequencies; pretrained towers break | `rope2d_cos_sin` builds `inv` over `half = d_rot // 2` |
| 4. row coordinates left row-major under a merge | tables no longer line up with the merged patches | `patch_coords` transposes rows and columns together into block-major order |
| 5. image time ids not offset by the start position | every image looks like it sits at time 0 | all three axes get `cur + ...` |
| 6. columns before rows in the 2D tables | the image is read transposed | the tables concatenate `[ch, cw]`: rows first |
| 7. accepting a rotary width that is only even | tables whose width no longer matches the pairs they rotate | `rope2d_cos_sin` raises unless `d_rot % 4 == 0` |
| 8. accepting sections that sum to less than $r/2$ | trailing pairs with no axis | `mrope_cos_sin` raises unless `sum(sec) == len(f)` |
| 9. not checking that the merge divides the grid | a fractional token grid, silently floored | `patch_coords` and `mrope_position_ids` raise on a remainder |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L7.3` | `rope_cos_sin` computes every table (float32 angle, then cos and sin); `apply_rope` in the "half" layout does every rotation |
| Back | `L0.1` | `apply_mrope` takes and returns a `Tensor`, differentiable in $x$ |
| Back | `M00.2` | rotations compose by adding angles: the whole of section 2.1 (reading) |
| Forward | `L13.3` | the ViT's `pos='rope2d'` rotates queries and keys with `rope2d_cos_sin` |
| Forward | `L13.7` | the `qwen2vl` architecture: `rope2d_cos_sin(..., merge=2)` in the vision tower, `mrope_position_ids` and `apply_mrope` in the language model |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `mrope_position_ids` | HF `Qwen2VLModel.get_rope_index` | batches with padding and attention masks, video runs | `transformers/models/qwen2_vl/modeling_qwen2_vl.py` |
| `apply_mrope` | HF `Qwen2VLRotaryEmbedding` with `apply_rotary_pos_emb` | cos and sin cached per batch, fp16 | the same file |
| `rope2d_cos_sin` | HF `Qwen2VLVisionRotaryEmbedding` | tables for a batch of images of different sizes in one pass | the same file |
| M-RoPE's time axis | Qwen2.5-VL | time ids tied to absolute seconds, so video frame rate can vary | Bai et al., "Qwen2.5-VL technical report" (2025) |
| axial 2D RoPE | RoPE-Mixed, EVA-02 | learned mixed-axis frequencies, so diagonal offsets get their own rotations | Heo et al. (2024); Fang et al., "EVA-02" (2023) |
