# 2D RoPE and M-RoPE

The RoPE of [RoPE (half and interleaved layouts, partial rotary)](../p07-modern-block/03-rope.md) knows one coordinate: a token's index in the sequence. [Vision transformer encoder](03-vision-transformer.md) is about to cut an image into a grid of patches and feed them to attention as a flat list. Numbered $p = hW + w$, the patch directly below another is $W$ positions away, and that distance changes with every image width. A model trained at one resolution reads garbage offsets at another, which is exactly what any-resolution models such as Qwen2-VL cannot afford.

A language model that reads images has a second problem: an image of a thousand tokens would consume a thousand 1D positions between the words before it and the words after it. This section fixes both. Axial 2D RoPE rotates by row and by column separately (the ViT's `pos='rope2d'`), and Qwen2-VL's M-RoPE gives every token of the language model three position ids, so an image occupies a small square of positions. The planned vision-language model section's `qwen2vl` architecture needs both, bit-compatible with Hugging Face: a tiny Qwen2-VL model's logits depend on these position ids and rotations.

Everything here rests on the one fact that rotations compose by adding angles, covered in [Trig, unit circle, 2D rotations, complex numbers, Euler's formula](../../mathematics-for-computing/00-precalculus/02-trig-rotations-and-eulers-formula.md) in *Mathematics for Computing*. You will reuse `rope_cos_sin` and `apply_rope` from the RoPE section and the `Tensor` from [Tensor, broadcasting backward, and no_grad](../p00-foundations/01-tensor-and-broadcasting-backward.md). The full solution is [`rope2d.py`](../../../../course/solutions/python/tinyllm/vision/rope2d.py) and its contract is [`course/contracts/py/tinyllm/vision/rope2d.pyi`](../../../../course/contracts/py/tinyllm/vision/rope2d.pyi).

> **Learning objectives**
> - Show that RoPE's relative-position property holds pair by pair, whatever coordinate drives each pair.
> - Build axial 2D RoPE tables for a patch grid, in block-major order under a spatial merge.
> - Compute Qwen2-VL's M-RoPE position ids for a prompt mixing text and images, including where text resumes after an image.
> - Map `mrope_section` onto one frequency ladder and explain why M-RoPE on text is exactly 1D RoPE.

## Notation

The symbols below recur throughout.

| Symbol | Meaning | Type / shape |
|---|---|---|
| $r$ | rotary width (`d_rot`), $r/2$ pairs | int, multiple of 4 for 2D |
| $\omega_i$ | frequency of pair $i$ (`inv_freq`) | `float[r/2]` (`float32[r/4]` for 2D) |
| $(h, w)$ | patch row and column | `int64[H·W, 2]` |
| $m$ | spatial merge: $m \times m$ patches become one LM token | int |
| $(t, h, w)$ | M-RoPE position ids of each token | `int64[3, T]` |
| $s$ | position at which an image starts | int |
| cos, sin | one value per pair (the RoPE section's convention) | `float32[·, r/2]` |

## One rotation, one coordinate

Write pair $i$ of a query as the complex number $q_i = a + \mathrm{i}b$. Rotating $(a, b)$ by angle $\theta$ gives $(a\cos\theta - b\sin\theta,\ a\sin\theta + b\cos\theta)$, which is $q_i e^{\mathrm{i}\theta}$. The dot product of two real pairs is $\operatorname{Re}(q_i \bar k_i)$. If the query at position $p$ rotates by $p\,\omega_i$ and the key at $p'$ by $p'\omega_i$, the pair contributes
$$\operatorname{Re}\big(q_i e^{\mathrm{i}p\omega_i}\ \overline{k_i e^{\mathrm{i}p'\omega_i}}\big) = \operatorname{Re}\big(q_i \bar k_i\, e^{\mathrm{i}(p - p')\omega_i}\big),$$
because conjugation negates the key's angle and angles add. The score is the sum over pairs, so it depends on positions only through $p - p'$.

Notice that this argument is per pair and never asks where a pair's angle comes from. Everything that follows in this section is just a choice of which coordinate drives which pair.

## Axial 2D RoPE for patches

The first choice is for the vision tower. Give the first $r/4$ pairs the row and the last $r/4$ pairs the column. For $i < r/4$,
$$\theta_i = h\,\omega_i, \qquad \theta_{r/4 + i} = w\,\omega_i, \qquad \omega_i = \text{base}^{-2i/(r/2)}.$$
By the per-pair argument, the score of patch $(h, w)$ against $(h', w')$ is
$$\sum_{i < r/4} \operatorname{Re}\big(q_i \bar k_i\, e^{\mathrm{i}(h - h')\omega_i}\big) + \sum_{i < r/4} \operatorname{Re}\big(q_{r/4+i}\, \bar k_{r/4+i}\, e^{\mathrm{i}(w - w')\omega_i}\big),$$
a function of $(h - h', w - w')$ only, whatever the image size.

The ladder is the 1D ladder for a head of width $r/2$, used twice: each axis gets the full range of frequencies from 1 down toward $1/\text{base}$. Spreading one ladder over the whole width $r$ would give neither axis the right range. $r$ must be a multiple of 4. The tables are `float32[H·W, r/2]`, rows first then columns, each half computed by `rope_cos_sin`, and `apply_rope(x, cos, sin, "half")` applies them, where pair $j$ is $(x_j, x_{j + r/2})$. This is Qwen2-VL's vision tower.

There is one more piece of contract hidden in the tables: row $n$ belongs to the $n$-th patch, so the patch order matters. With `merge = 1` it is row-major. With `merge = m` the grid is cut into $m \times m$ blocks, blocks in row-major order and row-major inside each block. Qwen2-VL orders patches this way so one block's $m^2$ patches are adjacent and its `PatchMerger` can concatenate them into one token.

## M-RoPE position ids

The second choice is for the language model, where each token gets three ids. A text token at 1D position $p$ gets $(p, p, p)$. An image of $t \times H \times W$ patches arrives as $t \cdot (H/m) \cdot (W/m)$ image tokens, one per merge block. If it starts at position $s$, token $(t_i, h_i, w_i)$ of that grid, in row-major $(t, h, w)$ order, gets
$$(s + t_i,\ s + h_i,\ s + w_i),$$
and the next token starts at $s + \max(H/m, W/m)$.

Two things follow. All ids are offset by $s$, so by the per-pair argument the image looks the same wherever it sits in the prompt. And the image advances the position by its side length, not its token count: a $32 \times 32$ token image costs 32 positions instead of 1024, while the text after it still has ids larger than every row and column id the image used. For a single frame ($t = 1$) that is the image's largest id plus one. The $i$-th run of image tokens belongs to row $i$ of `image_grid_thw`; two images may be adjacent, so runs are split by the expected lengths, not by gaps.

## Sections: three axes on one ladder

Having three ids per token, the language model must decide which pairs read which id. It keeps a single ladder $\omega_0, \dots, \omega_{r/2-1}$ over the whole head. `mrope_section` cuts the $r/2$ pairs into consecutive sections; section $j$ takes its position from axis $j \bmod 3$ (time, row, column, time, ...). With $a(i)$ the axis of the section holding pair $i$,
$$\theta_i = \text{pos}[a(i)]\ \omega_i.$$
Qwen2-VL uses sections $[16, 24, 24]$ on a head of 128: time gets the 16 fastest pairs, rows the next 24, columns the slowest 24.

Unlike axial 2D RoPE, each pair keeps its own frequency and only its coordinate changes. So when $t = h = w = p$ every pair's angle is $p\,\omega_i$: M-RoPE on text is 1D RoPE, bit for bit, and a `qwen2vl` model with no image is a plain Qwen2 language model. The sections must sum to $r/2$.

`apply_mrope` rotates in the "half" layout through `apply_rope`. The map is linear in $x$ and orthogonal, so its backward is the transpose: the upstream gradient rotated by $-\theta$.

## Working it by hand

Start with position ids. The sequence is two text tokens, one image of $1 \times 4 \times 4$ patches with $m = 2$, then two text tokens: `[5, 6, IMG, IMG, IMG, IMG, 8, 9]`. The image needs $1 \cdot 2 \cdot 2 = 4$ tokens. Text takes positions 0 and 1, so $s = 2$; the grid $(h_i, w_i)$ runs $(0,0), (0,1), (1,0), (1,1)$; the next token starts at $2 + \max(2, 2) = 4$. The resulting ids are:

| Token | 5 | 6 | IMG | IMG | IMG | IMG | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| $t$ | 0 | 1 | 2 | 2 | 2 | 2 | 4 | 5 |
| $h$ | 0 | 1 | 2 | 2 | 3 | 3 | 4 | 5 |
| $w$ | 0 | 1 | 2 | 3 | 2 | 3 | 4 | 5 |

Counting image tokens instead would put token 8 at position 6.

Next, patch order. `patch_coords((2, 4), 2)` has two $2 \times 2$ blocks side by side: $(0,0), (0,1), (1,0), (1,1)$, then $(0,2), (0,3), (1,2), (1,3)$. Without a merge, `patch_coords((2, 3))` is row-major: $(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)$.

Finally, one rotation. Take $r = 4$ on a $2 \times 2$ grid. The ladder over the half-width 2 has one frequency, $\omega_0 = 10000^0 = 1$, so the patch at $(h, w)$ has $\cos = [\cos h, \cos w]$. For patch $(1, 0)$: $\cos = [0.5403, 1]$, $\sin = [0.8415, 0]$. In the "half" layout $x = [1, 1, 0, 0]$ holds the pairs $(x_0, x_2) = (1, 0)$ and $(x_1, x_3) = (1, 0)$. The row pair rotates by 1 rad to $(0.5403, 0.8415)$ and the column pair by 0, so the result is $[0.5403, 1, 0.8415, 0]$.

## Designing the API

Five functions cover both halves:

```python
def patch_coords(grid_hw: tuple[int, int], merge: int = 1) -> NDArray: ...          # int64 [H*W, 2]
def rope2d_cos_sin(grid_hw, d_rot: int, base: float = 10000.0, merge: int = 1) -> tuple[NDArray, NDArray]: ...
                                                                                     # float32 [H*W, d_rot/2] each
def mrope_position_ids(input_ids, image_grid_thw, image_token_id: int, spatial_merge: int = 2) -> NDArray: ...
                                                                                     # int64 [3, T]
def mrope_cos_sin(pos3, inv_freq, mrope_section: Sequence[int]) -> tuple[NDArray, NDArray]: ...   # float32 [T, r/2]
def apply_mrope(x: Tensor, pos3, inv_freq, mrope_section: Sequence[int]) -> Tensor: ...           # [..., T, r]
```

`image_grid_thw` is `[n_images, 3]` in patches $(t, H, W)$. The functions raise `ValueError` for $H$ or $W$ below 1, a merge that does not divide both, `d_rot` not a positive multiple of 4, image tokens that do not split into runs of exactly the expected lengths (too few, or a grid with no run), `pos3` not `[3, T]`, sections that do not sum to `len(inv_freq)`, or an `x` whose last axis is not `2 * len(inv_freq)`.

## Building it step by step

The design follows from six decisions.

1. **Reuse the 1D RoPE for every table.** The per-pair argument shows that RoPE's relative-position property holds whatever coordinate drives the pair. So the solution never writes a new rotation: it decides, for each pair, which coordinate and which frequency to use, and hands the resulting angles to `rope_cos_sin` and `apply_rope`.
2. **The patch order is data, so compute it explicitly.** Row $n$ of every table belongs to patch $n$. `patch_coords` returns the $(h, w)$ of each patch in the order the vision tower sees them, and both 2D tables are built from it, so order and tables cannot disagree.
3. **Block-major order is a reshape and a transpose.** A row-major $H \times W$ grid reshaped to $[H/m, m, W/m, m]$ has axes (block row, row in block, block column, column in block). Swapping the middle two gives (block row, block column, row in block, column in block), which flattened is block-major.
4. **For 2D, one ladder over the half-width, used twice.** Each axis must get the full range of frequencies, so the ladder is the one for width $r/2$, and the table is the row table next to the column table.
5. **For M-RoPE ids, one scan with a running position.** Text increments the position by 1; an image run fills a $t \times (H/m) \times (W/m)$ grid offset by the current position and then advances it by $\max(H/m, W/m)$. Runs are consumed by their expected length, so adjacent images split correctly.
6. **For M-RoPE tables, compute every axis at every frequency, then select.** `rope_cos_sin` on the `[3, T]` ids gives `[3, T, r/2]`; pair $i$ then reads axis $a(i)$. With $t = h = w$ the three slices are identical, so the selection returns exactly the 1D table.

`patch_coords` implements step 3:

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

`rope2d_cos_sin` implements step 4. The ladder is built in float32 to match the Hugging Face vision tower bit for bit:

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

`mrope_position_ids` implements step 5, with every malformed split rejected:

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

`mrope_cos_sin` implements step 6:

```python
cos, sin = rope_cos_sin(p, f)  # [3, T, r/2]: every axis at every frequency
axis = np.repeat(np.arange(len(sec)) % 3, sec)  # which axis drives pair i: t, h, w, t, ...
cols = np.arange(len(f))
# fancy-index (axis[i], :, i) for each pair i -> [r/2, T], transposed to [T, r/2]
return cos[axis, :, cols].T.copy(), sin[axis, :, cols].T.copy()
```

`apply_mrope` checks the width and calls `apply_rope(x, cos, sin, "half")`, so the backward is the 1D RoPE backward: the gradient rotated by $-\theta$. The full solution is [`rope2d.py`](../../../../course/solutions/python/tinyllm/vision/rope2d.py).

### Why this design

Tables are $O(T \cdot r)$, built once per image or prompt. `mrope_cos_sin` computes three times the needed table and discards two thirds, which is cheap next to attention and keeps the code a single gather; the id scan is $O(T)$. Because every rotation goes through the already-tested `apply_rope`, the only new correctness questions are which coordinate and frequency each pair gets, and those are exactly what the golden comparisons pin.

The two design choices that matter most are axial positions and resuming by side length. Flattened 1D positions make the vertical neighbour $W$ away, so offsets change with image width; axial RoPE makes the score a function of $(\Delta h, \Delta w)$ at every resolution, at the price of no special treatment for diagonal offsets (the RoPE-Mixed variant learns those). Resuming by token count would burn $HW/m^2$ positions per image; resuming by side length burns $\max(H, W)/m$ and still keeps every later text id above every image id.

## What can go wrong

The checks below pin the behaviour against Hugging Face and against the mathematical properties derived above.

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example | unit, smoke | the hand-worked id table and both patch orders, exactly | the resume rule is pinned |
| M-RoPE position ids match HF | golden | `get_rope_index` on a tall image, a wide one, and a two-frame one directly after it, integer-exact | the planned VLM feeds these ids to every attention layer |
| `apply_mrope` matches HF | golden | queries and keys rotated with sections $[2, 3, 3]$ within $10^{-5}$ of HF's `apply_rotary_pos_emb` | the section-to-axis mapping |
| vision RoPE matches HF | golden | block-major positions exact; cos and sin within $10^{-6}$ of the vision rotary tables (HF repeats them to the full head); rotated $q$, $k$ within $10^{-5}$ | the `rope2d` ViT |
| 2D scores depend only on offsets | property | shifting both patches by the same $(dh, dw)$ changes a score by less than $10^{-4}$; a row step differs from a column step | resolution-independent attention |
| M-RoPE reduces to 1D RoPE on text | property | text-only ids are $(p, p, p)$ and the output equals 1D RoPE's with tolerance 0 | a `qwen2vl` model without images is Qwen2 |
| gradient check, `apply_mrope` | gradcheck | the backward to $x$ against central differences | M-RoPE sits inside trained attention |
| bad inputs rejected | boundary | every `ValueError` listed in the API | wrong splits fail loudly, not silently |

> **Warning:** Resuming text after an image's token count instead of its side length rotates every later token too far, and long prompts run out of positions. `mrope_position_ids` advances `cur` by `max(lh, lw)`, not `n`.

Several mistakes leave text-only prompts looking right while breaking images, which is why the golden tests use images. Sections that take their axes in reverse order swap columns and time; `axis = np.repeat(np.arange(len(sec)) % 3, sec)` keeps time, row, column in order. Image time ids not offset by the start position make every image look like it sits at time 0; all three axes get `cur + ...`.

On the vision side, spreading the 2D ladder over the whole rotary width gives each axis the wrong frequencies and breaks pretrained towers, so `rope2d_cos_sin` builds `inv` over `half = d_rot // 2`. Leaving the coordinates row-major under a merge means the tables no longer line up with the merged patches; `patch_coords` transposes rows and columns together into block-major order. Putting columns before rows reads the image transposed; the tables concatenate `[ch, cw]`, rows first.

The last group is validation. Accepting a rotary width that is only even produces tables whose width no longer matches the pairs they rotate, so `rope2d_cos_sin` raises unless `d_rot % 4 == 0`. Accepting sections that sum to less than $r/2$ leaves trailing pairs with no axis, so `mrope_cos_sin` raises unless `sum(sec) == len(f)`. And a merge that does not divide the grid would silently floor to a fractional token grid, so `patch_coords` and `mrope_position_ids` raise on a remainder.

## Where this leads

The next section, [Vision transformer encoder](03-vision-transformer.md), uses `rope2d_cos_sin` for the ViT's `pos='rope2d'` to rotate queries and keys. The planned vision-language model section's `qwen2vl` architecture will use both halves: `rope2d_cos_sin(..., merge=2)` in the vision tower, and `mrope_position_ids` with `apply_mrope` in the language model.

Production code extends these pieces in a few ways. Hugging Face's `Qwen2VLModel.get_rope_index` handles batches with padding and attention masks, and video runs; `Qwen2VLRotaryEmbedding` with `apply_rotary_pos_emb` caches cos and sin per batch in fp16; and `Qwen2VLVisionRotaryEmbedding` builds tables for a batch of images of different sizes in one pass, all in `transformers/models/qwen2_vl/modeling_qwen2_vl.py`. Qwen2.5-VL ties time ids to absolute seconds so video frame rate can vary. RoPE-Mixed and EVA-02 learn mixed-axis frequencies so diagonal offsets get their own rotations.

### Further reading

- Su et al., "RoFormer: Enhanced transformer with rotary position embedding" (2021).
- Wang et al., "Qwen2-VL: Enhancing vision-language model's perception of the world at any resolution" (2024), section 2.1.
- Heo et al., "Rotary position embedding for vision transformer" (2024).
- Bai et al., "Qwen2.5-VL technical report" (2025).
- Fang et al., "EVA-02" (2023).

## Key terms

- **axial 2D RoPE**: rotary embedding where half the pairs rotate by the patch row and half by the column, so scores depend on $(\Delta h, \Delta w)$.
- **block-major order**: patch ordering in which each $m \times m$ merge block's patches are consecutive, blocks in row-major order.
- **M-RoPE**: Qwen2-VL's multimodal RoPE, giving each token three position ids $(t, h, w)$.
- **`mrope_section`**: the split of a head's pairs into consecutive sections, each driven by the time, row, or column id.
- **spatial merge**: combining $m \times m$ adjacent patches into one language-model token.

## Summary

- Axial 2D RoPE gives half the rotary pairs to the row and half to the column, so a score depends only on $(\Delta h, \Delta w)$.
- Both axes share one ladder of $r/4$ frequencies computed over the half-width $r/2$, and patches are ordered block-major under a merge.
- In M-RoPE every token has three ids $(t, h, w)$; text sets all three to its position, and the token after an image starts $\max(H/m, W/m)$ later, not one per image token.
- `mrope_section` assigns consecutive pairs of one frequency ladder to the axes $t, h, w$; when $t = h = w$ the result is exactly 1D RoPE.

## Review questions

1. Why does a text token after a $1 \times 4 \times 4$ image with $m = 2$ starting at $s = 2$ get position 4 and not 6?

   <details><summary>Answer</summary>

   The image's token grid is $2 \times 2$, so its row and column ids run from 2 to 3. Text resumes at $s + \max(2, 2) = 4$, one past the largest id the image used. Counting its 4 tokens would give 6, wasting positions and rotating every later token further than the pretrained model expects.

   </details>

2. Why does M-RoPE on text-only input equal 1D RoPE exactly, with tolerance 0?

   <details><summary>Answer</summary>

   Each pair keeps its own frequency $\omega_i$ and only chooses which axis supplies the position. For text all three axes hold $p$, so every pair's angle is $p\,\omega_i$, the same float32 computation 1D RoPE does, and the selection copies values without arithmetic.

   </details>

3. Why is the 2D ladder built over $r/2$ instead of $r$?

   <details><summary>Answer</summary>

   Each axis owns $r/4$ pairs and should see frequencies from 1 down toward $1/\text{base}$, like a head of width $r/2$. A ladder over $r$ would give the row axis only the fast half and the column axis only the slow half of the range, and would not match pretrained vision towers.

   </details>

4. What does the block-major order buy, and what would break without it?

   <details><summary>Answer</summary>

   It puts each $m \times m$ block's patches next to each other, so the `PatchMerger` can concatenate $m^2$ consecutive rows into one token. If the coordinates stayed row-major while the patches were block-major, row $n$ of the table would rotate a different patch from the one at position $n$.

   </details>
