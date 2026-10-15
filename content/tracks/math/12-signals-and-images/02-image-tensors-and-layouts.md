<!-- ol:module M12.2 -->
# Image tensors and layouts

## Overview

| | |
|---|---|
| **Module** | `M12.2` · build · Python · Pass 12 · 2 h |
| **You build** | `python/tinyllm/image/layout.py`: `to_layout` (NCHW and NHWC, with the bytes really moved), `strides_of`, `rescale_normalize` (u8 HWC to normalized f32 CHW), `to_grayscale_pil` (Pillow's `"L"` rule bit for bit), `patchify` and `unpatchify` |
| **Contract** | [`course/contracts/py/tinyllm/image/layout.pyi`](../../../../course/contracts/py/tinyllm/image/layout.pyi) |
| **Tests** | `course/tests/M12.2/test_layout.py` (what they check: section 4), Pillow golden pixels from `course/oracle/M12.2/grayscale_pil.py` in `course/fixtures/M12.2/grayscale_pil.npz` |
| **Needs** | no module code. Reading: `M03.1` (row-major layout), `lang.01` (numpy views, copies, and strides) |
| **Used by** | later `L13.3` (`patchify` in the ViT patch embedding) · `L13.6` (`rescale_normalize` in every image processor) · `data.10` (`to_grayscale_pil` for perceptual hashes); they join the registry with B14's L13 and data groups |
| **Milestone** | `MS-P12` (the multimodal gate) |
| **Optional depth** | PyTorch, "Channels Last Memory Format" tutorial (free); ITU-R Recommendation BT.601 (luma coefficients); Dosovitskiy et al., "An image is worth 16x16 words" (ICLR 2021), section 3.1 |

## Key Takeaways

- A layout is a memory order, not a shape: the logical axes stay $(N, C, H, W)$ and the strides say how far apart neighbours are; channels last makes the channel stride the itemsize (`test_hand_example_strides`, `test_strides_match_numpy_for_any_shape`).
- A transpose is a view; converting a layout means copying the bytes into the new order (`test_to_layout_moves_the_bytes`).
- Model inputs are $(x \cdot \text{scale} - \mu_c) / \sigma_c$ in float32 CHW (`test_rescale_normalize_hand_values`).
- Pillow's grayscale is luma in 16-bit fixed point, which differs from the textbook thousandths on about 0.05% of pixels (`test_hand_example_grayscale`, `test_grayscale_matches_pil`).
- Patchify is a reshape and a transpose, and with $(c, i, j)$ order it turns a stride-$p$ conv into one matmul (`test_patchify_is_a_stride_p_conv`).

## How to work this chapter

```bash
ol start M12.2              # stubs python/tinyllm/image/layout.py into your repo
ol tests M12.2              # read the test catalog first: rung R0, you write no tests here
ol check M12.2              # exit code is the verdict
ol diff  M12.2              # after passing: your code against the reference
```

---

## 1. Why now

Text came into the system as integer token ids. Images come in as bytes from a decoder, uint8 with the three channels of each pixel side by side (HWC), and the vision towers of `L13` want float32 planes, one channel after another (CHW), normalized with the statistics the checkpoint was trained on. Every processor in `L13.6` does this conversion, and if it gets the axis order or the normalization order wrong the model still runs and quietly returns nonsense: there is no shape error to warn you. The data pipeline in `data.10` hashes grayscale thumbnails to find near-duplicate images, and a grayscale value one level off from Pillow's flips hash bits. This module pins down these conventions once, before any model code depends on them.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $N, C, H, W$ | batch, channels, height, width | ints |
| NCHW | memory order batch, channel, row, column | layout |
| NHWC | memory order batch, row, column, channel ("channels last") | layout |
| $e$ | bytes per element (itemsize) | int |
| $\text{st}_a$ | stride of logical axis $a$, in bytes | int |
| $\mu_c$, $\sigma_c$ | per-channel mean and standard deviation | floats |
| $p$ | patch size | int |
| $L$ | grayscale (luma) value | uint8 |

### 2.1 Shape and strides

An array is a block of memory plus, for each axis, a stride: the number of bytes to step to move one index along that axis. Element $(n, c, h, w)$ lives at offset $n\,\text{st}_N + c\,\text{st}_C + h\,\text{st}_H + w\,\text{st}_W$. For a contiguous NCHW tensor the last axis is fastest:
$$(\text{st}_N, \text{st}_C, \text{st}_H, \text{st}_W) = e\,(CHW,\; HW,\; W,\; 1).$$
Stored channels last (NHWC) but still indexed by the same logical axes, the channel becomes the fastest axis:
$$(\text{st}_N, \text{st}_C, \text{st}_H, \text{st}_W) = e\,(HWC,\; 1,\; WC,\; C).$$
This is exactly what PyTorch reports for `memory_format=torch.channels_last`: the shape stays NCHW, only the strides change. `strides_of` returns the strides per logical axis, not in memory order; that distinction is the whole point.

### 2.2 Changing layout

`x.transpose(0, 2, 3, 1)` on an NCHW array does not move a byte: it returns a view with permuted strides. Code that reads memory directly (a C kernel, a safetensors writer, a GPU upload) sees the old order. `to_layout` therefore returns `np.ascontiguousarray` of the transposed view, which copies. When source and destination agree it still copies, so a caller who writes to the result never corrupts the original.

### 2.3 Rescale and normalize

A model trained on CLIP preprocessing expects each channel scaled to $[0, 1]$ and standardized:
$$y_c = \frac{x_c \cdot \text{scale} - \mu_c}{\sigma_c}, \qquad \text{scale} = 1/255,$$
with CLIP's $\mu = (0.4815, 0.4578, 0.4082)$ and $\sigma = (0.2686, 0.2613, 0.2758)$. The order is fixed: subtracting the mean before scaling gives values 255 times too far from zero. `rescale_normalize` computes in float64, rounds once to float32, and returns CHW.

### 2.4 Grayscale the way Pillow does it

ITU-R 601-2 luma is $L = 0.299 R + 0.587 G + 0.114 B$. Pillow evaluates it in 16-bit fixed point with rounding:
$$L = \big(19595 R + 38470 G + 7471 B + 32768\big) \gg 16,$$
where $19595/65536 = 0.29900$, $38470/65536 = 0.58701$, $7471/65536 = 0.11400$ (to five places). The textbook integer version, $(299R + 587G + 114B + 500) \operatorname{div} 1000$, rounds the same exact value differently whenever the luma sits on a half: about 0.05% of RGB triples. Pillow's own documentation prints the decimal formula; the code is the fixed-point one, and only the code matters for bit-exact hashes.

### 2.5 Patches

A ViT with patch size $p$ cuts a $C \times H \times W$ image into $(H/p)(W/p)$ squares and flattens each into a vector of $C p^2$ numbers. With a reshape to $[C, H/p, p, W/p, p]$ and a transpose to $[H/p, W/p, C, p, p]$, patches come out row-major over the grid, each flattened in $(c, i, j)$ order. That order is the order a KCRS conv weight flattens in, so a conv with kernel and stride $p$ is `patchify(x, p) @ w.reshape(K, -1).T`: `L13.3` relies on this identity. `unpatchify` reverses both steps.

## 3. Worked example by hand

A float32 image with $C, H, W = 2, 3, 4$ ($e = 4$ bytes):

| Layout | $\text{st}_C$ | $\text{st}_H$ | $\text{st}_W$ |
|---|---|---|---|
| CHW | $3 \cdot 4 \cdot 4 = 48$ | $4 \cdot 4 = 16$ | 4 |
| HWC | 4 | $4 \cdot 2 \cdot 4 = 32$ | $2 \cdot 4 = 8$ |

In HWC, stepping one channel moves 4 bytes (the next value of the same pixel), and stepping one row moves past $W \cdot C = 8$ values. This is `test_hand_example_strides`.

Grayscale of pure blue at 250: the exact luma is $0.114 \cdot 250 = 28.5$. The thousandths rule gives $(28500 + 500) \operatorname{div} 1000 = 29$; Pillow gives $(7471 \cdot 250 + 32768) \gg 16 = 1900518 \gg 16 = 28$, because $7471/65536$ is slightly less than $0.114$. Pure red, green, blue at 255 give 76, 150, 29 under both rules. This is `test_hand_example_grayscale`.

Patches of the $2 \times 4 \times 4$ image `arange(32)` with $p = 2$: patch 1 is grid position $(0, 1)$, the top-right $2 \times 2$ block of channel 0, $(2, 3, 6, 7)$, followed by the same block of channel 1, $(18, 19, 22, 23)$ (`test_hand_example_patchify`).

## 4. The interface

```python
def to_layout(x, src: Layout, dst: Layout) -> NDArray: ...          # contiguous copy
def strides_of(shape, layout: Layout, itemsize: int) -> tuple: ...   # per logical axis
def rescale_normalize(img_u8_hwc, scale, mean, std) -> NDArray: ...  # f32 CHW
def to_grayscale_pil(img_u8_hwc) -> NDArray: ...                     # uint8 HW
def patchify(x_chw, p) -> NDArray: ...                               # [(H/p)(W/p), C p p]
def unpatchify(patches, c, h, w, p) -> NDArray: ...
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_strides` | unit, smoke | section 3's stride table and numpy's strides of the same views | the layout vocabulary |
| `test_strides_match_numpy_for_any_shape` | property | channels-last strides for random shapes and itemsizes | what a C kernel indexes with |
| `test_to_layout_moves_the_bytes` | property | C-contiguous result in the new order; same-layout still copies | raw-memory readers see the right order |
| `test_rescale_normalize_hand_values` | unit | CLIP normalization, CHW float32 | every `L13.6` processor |
| `test_hand_example_grayscale` | unit | section 3's grayscale values, including the 28 vs 29 pixel | the rounding rule |
| `test_grayscale_matches_pil` | golden | Pillow `convert("L")` on 1152 pixels, a third of them chosen where the rules disagree | `data.10`'s hashes |
| `test_hand_example_patchify` | unit | section 3's patches | the ViT patch order |
| `test_patchify_is_a_stride_p_conv` | differential | patchify then matmul equals a stride-$p$ conv; unpatchify inverts | `L13.3`'s PatchEmbed |
| `test_rejects_bad_inputs` | boundary | bad layouts, zero dims, zero std, patch size not dividing; inputs untouched | pipeline bugs fail at the boundary |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. returning the transposed view | numpy indexing looks right, raw memory is in the old order | `test_to_layout_moves_the_bytes` (mutant `s01`) |
| 2. channels-last strides listed in memory order | strides for the wrong axes; a kernel walks off the row | `test_hand_example_strides`, `test_strides_match_numpy_for_any_shape` (mutant `s02`) |
| 3. the textbook thousandths luma | one level off on 0.05% of pixels, hash bits flip | `test_hand_example_grayscale`, `test_grayscale_matches_pil` (mutant `s03`) |
| 4. leaving the normalized image HWC | the model reads rows as channels | `test_rescale_normalize_hand_values` (mutant `s04`) |
| 5. patches flattened $(i, j, c)$ | PatchEmbed weights meet the wrong pixels | `test_hand_example_patchify`, `test_patchify_is_a_stride_p_conv` (mutant `s05`) |
| 6. subtracting the mean before rescaling | inputs about 255 times too large | `test_rescale_normalize_hand_values` (mutant `s06`) |
| converting to the same layout without a copy | writing the result corrupts the caller's array | `test_to_layout_moves_the_bytes` (mutant `s07`) |
| accepting a zero-size dimension | strides of an empty tensor pass silently | `test_rejects_bad_inputs` (mutant `m01`) |
| accepting a zero std | infinities in the model input | `test_rejects_bad_inputs` (mutant `m02`) |
| accepting float pixels for grayscale | a float image truncates silently | `test_grayscale_matches_pil` (mutant `m03`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M03.1` | row-major layout and strides of a matrix (reading) |
| Back | `lang.01` | numpy views versus copies (reading) |
| Forward | `M12.3` | resizes the HWC uint8 images this module defines (reading) |
| Forward | `L13.3` | `patchify` inside the ViT patch embedding (with B14's L13 group) |
| Forward | `L13.6` | `rescale_normalize` after resizing, in every image processor |
| Forward | `data.10` | `to_grayscale_pil` before pHash and dHash |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `strides_of`, `to_layout` | `torch.Tensor.contiguous(memory_format=torch.channels_last)` | layout-aware kernels that skip the copy when the next op prefers NHWC | `c10/core/MemoryFormat.h` |
| `rescale_normalize` | HF `BaseImageProcessor.rescale` and `normalize` | channel-first or channel-last inference, per-processor defaults from `preprocessor_config.json` | `transformers/image_transforms.py` |
| `to_grayscale_pil` | Pillow `Image.convert("L")` | the C implementation, including the `L24` macro | `src/libImaging/Convert.c` |
| `patchify` | `einops.rearrange(x, 'c (h p1) (w p2) -> (h w) (c p1 p2)')` | the same reshape written as a pattern | einops documentation |
