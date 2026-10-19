# Image tensors and layouts

Text came into the system as integer token ids. Images come in as bytes from a decoder: uint8, with the three channels of each pixel side by side (HWC). Vision towers want float32 planes, one channel after another (CHW), normalized with the statistics the checkpoint was trained on. Every image processor does this conversion, and if it gets the axis order or the normalization order wrong the model still runs and quietly returns nonsense. There is no shape error to warn anyone. A data pipeline that hashes grayscale thumbnails to find near-duplicate images has the same problem in miniature: a grayscale value one level off from Pillow's flips hash bits.

This section pins down these conventions once, before any model code depends on them. You will see what a memory layout really is (a set of strides, not a shape), why changing layout means copying bytes, how model inputs are rescaled and normalized, how Pillow computes grayscale bit for bit, and why cutting an image into ViT patches is just a reshape. It builds on row-major layout from [Vectors, matrices, row-major layout, and matmul in Python](../03-linear-algebra/01-vectors-matrices-and-matmul.md) and on NumPy views, copies, and strides from [Python and numpy](../../programming-languages/02-language-and-tool-primers/01-python-and-numpy.md) in *Programming Languages*.

The reference implementation is [`python/tinyllm/image/layout.py`](../../../../course/solutions/python/tinyllm/image/layout.py), with its interface in [`course/contracts/py/tinyllm/image/layout.pyi`](../../../../course/contracts/py/tinyllm/image/layout.pyi). It provides `to_layout` (NCHW and NHWC, with the bytes really moved), `strides_of`, `rescale_normalize` (u8 HWC to normalized f32 CHW), `to_grayscale_pil` (Pillow's `"L"` rule bit for bit), and `patchify` and `unpatchify`.

> **Learning objectives**
> - Compute the per-axis strides of an NCHW tensor stored channels first or channels last.
> - Explain why a transpose is a view and a layout change is a copy, and when the difference matters.
> - Apply $(x \cdot \text{scale} - \mu_c)/\sigma_c$ in the right order and produce float32 CHW.
> - Reproduce Pillow's fixed-point grayscale and explain where it differs from the textbook rule.
> - Cut an image into ViT patches in $(c, i, j)$ order and relate patchify to a stride-$p$ convolution.

## Notation

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

## Shape and strides

An array is a block of memory plus, for each axis, a stride: the number of bytes to step to move one index along that axis. Element $(n, c, h, w)$ lives at offset $n\,\text{st}_N + c\,\text{st}_C + h\,\text{st}_H + w\,\text{st}_W$. For a contiguous NCHW tensor the last axis is fastest:
$$(\text{st}_N, \text{st}_C, \text{st}_H, \text{st}_W) = e\,(CHW,\; HW,\; W,\; 1).$$
Stored channels last (NHWC) but still indexed by the same logical axes, the channel becomes the fastest axis:
$$(\text{st}_N, \text{st}_C, \text{st}_H, \text{st}_W) = e\,(HWC,\; 1,\; WC,\; C).$$
This is exactly what PyTorch reports for `memory_format=torch.channels_last`: the shape stays NCHW, only the strides change. A layout is a memory order, not a shape. `strides_of` returns the strides per logical axis, not in memory order; that distinction is the whole point.

## Changing layout

If a layout is just strides, it is tempting to change it by permuting them. `x.transpose(0, 2, 3, 1)` on an NCHW array does exactly that and does not move a byte: it returns a view with permuted strides. Code that reads memory directly (a C kernel, a safetensors writer, a GPU upload) sees the old order. `to_layout` therefore returns `np.ascontiguousarray` of the transposed view, which copies. When source and destination agree it still copies, so a caller who writes to the result never corrupts the original.

## Rescale and normalize

With the bytes in the right order, the values must be put on the scale the model was trained on. A model trained on CLIP preprocessing expects each channel scaled to $[0, 1]$ and standardized:
$$y_c = \frac{x_c \cdot \text{scale} - \mu_c}{\sigma_c}, \qquad \text{scale} = 1/255,$$
with CLIP's $\mu = (0.4815, 0.4578, 0.4082)$ and $\sigma = (0.2686, 0.2613, 0.2758)$. The order is fixed: subtracting the mean before scaling gives values 255 times too far from zero. `rescale_normalize` computes in float64, rounds once to float32, and returns CHW.

## Grayscale the way Pillow does it

Some consumers want one channel, not three. ITU-R 601-2 luma is $L = 0.299 R + 0.587 G + 0.114 B$. Pillow evaluates it in 16-bit fixed point with rounding:
$$L = \big(19595 R + 38470 G + 7471 B + 32768\big) \gg 16,$$
where $19595/65536 = 0.29900$, $38470/65536 = 0.58701$, $7471/65536 = 0.11400$ (to five places). The textbook integer version, $(299R + 587G + 114B + 500) \operatorname{div} 1000$, rounds the same exact value differently whenever the luma sits on a half: about 0.05% of RGB triples. Pillow's own documentation prints the decimal formula; the code is the fixed-point one, and only the code matters for bit-exact hashes.

## Patches

The last conversion is the one a ViT performs first. A ViT with patch size $p$ cuts a $C \times H \times W$ image into $(H/p)(W/p)$ squares and flattens each into a vector of $C p^2$ numbers. With a reshape to $[C, H/p, p, W/p, p]$ and a transpose to $[H/p, W/p, C, p, p]$, patches come out row-major over the grid, each flattened in $(c, i, j)$ order. That order is the order a KCRS conv weight flattens in (see [Convolution as a linear map](01-convolution-as-a-linear-map.md)), so a conv with kernel and stride $p$ is `patchify(x, p) @ w.reshape(K, -1).T`. The [Vision transformer encoder](../../build-an-llm/p13-vision/03-vision-transformer.md) in *Build Your Own LLM System* relies on this identity. `unpatchify` reverses both steps.

## Working it by hand

Take a float32 image with $C, H, W = 2, 3, 4$ ($e = 4$ bytes). Its strides in the two layouts are:

| Layout | $\text{st}_C$ | $\text{st}_H$ | $\text{st}_W$ |
|---|---|---|---|
| CHW | $3 \cdot 4 \cdot 4 = 48$ | $4 \cdot 4 = 16$ | 4 |
| HWC | 4 | $4 \cdot 2 \cdot 4 = 32$ | $2 \cdot 4 = 8$ |

In HWC, stepping one channel moves 4 bytes (the next value of the same pixel), and stepping one row moves past $W \cdot C = 8$ values.

Now grayscale pure blue at 250. The exact luma is $0.114 \cdot 250 = 28.5$. The thousandths rule gives $(28500 + 500) \operatorname{div} 1000 = 29$; Pillow gives $(7471 \cdot 250 + 32768) \gg 16 = 1900518 \gg 16 = 28$, because $7471/65536$ is slightly less than $0.114$. Pure red, green, and blue at 255 give 76, 150, and 29 under both rules.

Finally, patch the $2 \times 4 \times 4$ image `arange(32)` with $p = 2$. Patch 1 is grid position $(0, 1)$: the top-right $2 \times 2$ block of channel 0, $(2, 3, 6, 7)$, followed by the same block of channel 1, $(18, 19, 22, 23)$.

## Designing the interface

```python
def to_layout(x, src: Layout, dst: Layout) -> NDArray: ...          # contiguous copy
def strides_of(shape, layout: Layout, itemsize: int) -> tuple: ...   # per logical axis
def rescale_normalize(img_u8_hwc, scale, mean, std) -> NDArray: ...  # f32 CHW
def to_grayscale_pil(img_u8_hwc) -> NDArray: ...                     # uint8 HW
def patchify(x_chw, p) -> NDArray: ...                               # [(H/p)(W/p), C p p]
def unpatchify(patches, c, h, w, p) -> NDArray: ...
```

Every conversion returns a fresh, contiguous array, so callers can write to results freely. `strides_of` accepts a 3-D (CHW) or 4-D shape and always reports strides per logical axis. Inputs that cannot be converted meaningfully (an unknown layout name, a zero dimension, a zero standard deviation, a patch size that does not divide the image, float pixels passed to grayscale) are rejected at the boundary, and inputs are never modified.

## Building it step by step

Each function follows from one observation about memory or arithmetic.

1. **A layout is a permutation of axes in memory.** NCHW and NHWC differ only in which logical axis is fastest. So converting between them is a fixed axis permutation, `(0, 2, 3, 1)` one way and `(0, 3, 1, 2)` back, and nothing else.
2. **A permutation alone moves no bytes.** NumPy's `transpose` returns a view with permuted strides. Anything that reads raw memory needs the bytes in the new order, so the conversion must end in a contiguous copy. Same-layout conversion copies too, so the result never aliases the input.
3. **Strides follow from the memory order, then get reported per logical axis.** In memory order, the stride of each axis is the product of the sizes of the faster axes, times the itemsize. For NHWC the memory order is $(N, H, W, C)$, giving $(HWC, WC, C, 1)$; reported in the logical order $(N, C, H, W)$ that is $(HWC, 1, WC, C)$.
4. **Normalization is per channel, then the layout changes.** $(x \cdot \text{scale} - \mu_c)/\sigma_c$ broadcasts naturally over HWC, where the channel is the last axis, so compute it there, in float64, then transpose to CHW, make it contiguous, and round to float32 once.
5. **Bit-exact grayscale means integer arithmetic.** Pillow's rule is integer multiply-add, add half of $2^{16}$, shift right by 16. Reproducing it in floats would invite the same half-way disagreements as the thousandths rule. Widen to int64 first: $255 \cdot 65536$ overflows a uint8 and even an int16.
6. **Patchify is a reshape, a transpose, and a reshape.** Splitting $H$ into $(H/p, p)$ and $W$ into $(W/p, p)$ is a free reshape of a row-major array. Ordering the axes as (grid row, grid column, channel, in-patch row, in-patch column) and merging the first two and last three gives patches row-major over the grid, each flattened $(c, i, j)$. Unpatchify is the inverse permutation.

The layout change is the first two steps in code:

```python
_PERM = {("NCHW", "NHWC"): (0, 2, 3, 1), ("NHWC", "NCHW"): (0, 3, 1, 2)}


def to_layout(x: NDArray, src: Layout, dst: Layout) -> NDArray:
    # ... validate layout names and ndim == 4 ...
    if src == dst:
        # Still a copy: a caller who writes to the result must not change x.
        return np.ascontiguousarray(x).copy()
    # transpose only relabels the axes (a view with permuted strides); the
    # contiguous copy is what actually moves the bytes into the new order.
    return np.ascontiguousarray(x.transpose(_PERM[(src, dst)]))
```

The `.copy()` in the same-layout branch matters because `np.ascontiguousarray` returns its input unchanged when it is already contiguous.

The stride computation is step 3:

```python
def strides_of(shape, layout, itemsize):
    # ... validate layout, 3 or 4 dims, all >= 1, itemsize >= 1 ...
    dims = tuple(shape) if len(shape) == 4 else (1, *shape)   # treat CHW as N = 1
    n, c, h, w = dims
    if layout == "NCHW":
        st = (c * h * w, h * w, w, 1)
    else:
        # Channels last: memory order N, H, W, C, reported per logical axis.
        st = (h * w * c, 1, w * c, c)
    st = tuple(itemsize * s for s in st)
    return st if len(shape) == 4 else st[1:]                  # drop N again for CHW
```

These are exactly the strides NumPy reports for `to_layout(x, "NCHW", "NHWC").transpose(0, 3, 1, 2)`: a channels-last array viewed with NCHW axes.

Normalization is step 4:

```python
def rescale_normalize(img_u8_hwc, scale, mean, std):
    # ... validate HWC, one mean and std per channel, every std > 0 ...
    # Scale first, then subtract: mu and sigma are in [0, 1] units.
    y = (img.astype(np.float64) * scale - m) / s  # float64, then one rounding
    # HWC -> CHW, contiguous, then float32: the layout the vision tower reads.
    return np.ascontiguousarray(y.transpose(2, 0, 1)).astype(np.float32)
```

`m` and `s` have shape `(C,)`, so they broadcast against the last axis of the HWC image: each channel gets its own statistics with no explicit loop.

Grayscale is step 5, in integers:

```python
def to_grayscale_pil(img_u8_hwc):
    # Float or non-RGB input is rejected: truncating a float image silently
    # would give a "grayscale" that matches nothing.
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError(...)
    r, g, b = (img[..., i].astype(np.int64) for i in range(3))   # widen before multiplying
    # ITU-R 601-2 luma in 16-bit fixed point, rounded: Pillow's L24 macro.
    return ((r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16).astype(np.uint8)
```

The three weights sum to $65536$, so white $(255, 255, 255)$ maps to $(255 \cdot 65536 + 32768) \gg 16 = 255$ and the result always fits in a uint8.

And patches are step 6:

```python
def patchify(x_chw, p):
    # ... validate CHW, p >= 1, p divides H and W ...
    gh, gw = h // p, w // p
    # [C, gh, p, gw, p] -> [gh, gw, C, p, p]: patches row-major over the grid,
    # each flattened in (c, i, j) order like a KCRS conv weight.
    return (
        x.reshape(c, gh, p, gw, p).transpose(1, 3, 0, 2, 4).reshape(gh * gw, c * p * p)
    )


def unpatchify(patches, c, h, w, p):
    # ... validate p, divisibility, and the patches' shape ...
    # The inverse: [gh, gw, C, p, p] -> [C, gh, p, gw, p] -> [C, H, W].
    return a.reshape(gh, gw, c, p, p).transpose(2, 0, 3, 1, 4).reshape(c, h, w)
```

Because each patch is flattened $(c, i, j)$, a stride-$p$ conv with weight `w` of shape `[K, C, p, p]` is `patchify(x, p) @ w.reshape(K, -1).T`, with output `[(H/p)(W/p), K]`, one token embedding per patch.

The full solution is [`layout.py`](../../../../course/solutions/python/tinyllm/image/layout.py).

## Why this design

Every function is $O(CHW)$: one pass to compute and at most one copy to reorder. Patchify's final reshape copies (the transposed array is not contiguous), which is the same cost as any layout change.

The functions copy rather than return views. Views are cheaper, but a layout change that leaves memory in the old order defeats the purpose for every raw-memory consumer, and views alias the caller's array. Correctness of the bytes is the point.

Grayscale is integer, not float. A float formula with Pillow's coefficients would usually agree, but "usually" is not bit-exact: a perceptual hash that differs from Pillow's in one bit for 0.05% of images is a duplicate detector that disagrees with every other tool.

Normalization computes in float64 and casts once. Normalizing in float32 would round twice (after the scale and after the division); computing in float64 and rounding once makes the result the correctly rounded value of the formula, which is what a golden comparison against another implementation needs.

Patchify is a reshape rather than a conv. The two are equal by construction, but the reshape is a pure data movement with no arithmetic, and it makes the patch order explicit, which is the thing that must match the pretrained weights.

## What can go wrong

A correct implementation reproduces the hand-worked stride table and agrees with NumPy's strides of the same views, and its channels-last strides match NumPy for random shapes and itemsizes, because that is what a C kernel indexes with. `to_layout` must produce a C-contiguous result in the new order, and the same-layout case must still copy, so raw-memory readers see the right order. `rescale_normalize` must give CLIP normalization by hand as CHW float32. Grayscale must reproduce the hand-worked values, including the 28-versus-29 pixel, and match Pillow's `convert("L")` on 1152 pixels, a third of them chosen where the two rules disagree. Patches must match the hand example; patchify followed by a matmul must equal a stride-$p$ conv, and unpatchify must invert patchify. Bad layouts, zero dimensions, zero standard deviations, and patch sizes that do not divide the image must be rejected with inputs untouched, so pipeline bugs fail at the boundary.

Here are the mistakes those checks are designed to catch:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| returning the transposed view | numpy indexing looks right, raw memory is in the old order | `np.ascontiguousarray(x.transpose(...))` (step 2) |
| channels-last strides listed in memory order | strides for the wrong axes; a kernel walks off the row | reports per logical axis: `(h*w*c, 1, w*c, c)` (step 3) |
| the textbook thousandths luma | one level off on 0.05% of pixels, hash bits flip | Pillow's fixed-point weights `19595, 38470, 7471` with `+ 0x8000 >> 16` |
| leaving the normalized image HWC | the model reads rows as channels | transposes `(2, 0, 1)` to CHW before the cast |
| patches flattened $(i, j, c)$ | PatchEmbed weights meet the wrong pixels | transposes to `[gh, gw, C, p, p]` before flattening (step 6) |
| subtracting the mean before rescaling | inputs about 255 times too large | computes `x * scale - m`, then divides by `s` |
| converting to the same layout without a copy | writing the result corrupts the caller's array | the same-layout branch ends in `.copy()` |
| accepting a zero-size dimension | strides of an empty tensor pass silently | `strides_of` rejects any dimension below 1 |
| accepting a zero std | infinities in the model input | `rescale_normalize` requires every std to be positive |
| accepting float pixels for grayscale | a float image truncates silently | `to_grayscale_pil` requires `uint8` RGB |

> **Warning:** Most of these bugs produce no error at all. A model fed HWC data, $(i, j, c)$ patches, or unscaled pixels runs normally and returns confident nonsense.

## Where this leads

The next section, [Resampling and interpolation](03-resampling-and-interpolation.md), resizes the HWC uint8 images defined here. In *Build Your Own LLM System*, the [Vision transformer encoder](../../build-an-llm/p13-vision/03-vision-transformer.md) uses `patchify` inside its patch embedding, and [Image preprocessing to the specification](../../build-an-llm/p13-vision/06-image-preprocessing.md) calls `rescale_normalize` after resizing in every image processor. The image deduplication stage of the [corpus pipeline](../../data-engineering/05-corpus-pipeline/README.md) in *Data Engineering* calls `to_grayscale_pil` before computing pHash and dHash.

Each piece has a production counterpart:

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `strides_of`, `to_layout` | `torch.Tensor.contiguous(memory_format=torch.channels_last)` | layout-aware kernels that skip the copy when the next op prefers NHWC | `c10/core/MemoryFormat.h` |
| `rescale_normalize` | HF `BaseImageProcessor.rescale` and `normalize` | channel-first or channel-last inference, per-processor defaults from `preprocessor_config.json` | `transformers/image_transforms.py` |
| `to_grayscale_pil` | Pillow `Image.convert("L")` | the C implementation, including the `L24` macro | `src/libImaging/Convert.c` |
| `patchify` | `einops.rearrange(x, 'c (h p1) (w p2) -> (h w) (c p1 p2)')` | the same reshape written as a pattern | einops documentation |

### Further reading

- PyTorch, "Channels Last Memory Format" tutorial (free).
- ITU-R Recommendation BT.601 (luma coefficients).
- Dosovitskiy et al., "An image is worth 16x16 words" (ICLR 2021), section 3.1.

## Key terms

- **Stride**: the number of bytes to step in memory to advance one index along an axis.
- **Channels last (NHWC)**: a memory order in which the channel is the fastest axis; the logical shape can stay NCHW.
- **View**: an array that shares memory with another and differs only in shape, strides, or offset.
- **Luma**: the weighted sum $0.299R + 0.587G + 0.114B$ used as grayscale; Pillow computes it in 16-bit fixed point.
- **Patchify**: cutting an image into $p \times p$ squares, each flattened into a vector of $C p^2$ numbers.

## Summary

- A layout is a memory order, not a shape: the logical axes stay $(N, C, H, W)$ and the strides say how far apart neighbours are; channels last makes the channel stride the itemsize.
- A transpose is a view; converting a layout means copying the bytes into the new order.
- Model inputs are $(x \cdot \text{scale} - \mu_c) / \sigma_c$ in float32 CHW.
- Pillow's grayscale is luma in 16-bit fixed point, which differs from the textbook thousandths on about 0.05% of pixels.
- Patchify is a reshape and a transpose, and with $(c, i, j)$ order it turns a stride-$p$ conv into one matmul.

## Review questions

1. Why does `strides_of` for NHWC return $(HWC, 1, WC, C)$ times the itemsize and not $(HWC, WC, C, 1)$?
   <details><summary>Answer</summary>
   The second tuple is the strides in memory order $(N, H, W, C)$. The function reports per logical axis $(N, C, H, W)$, which is how PyTorch's channels-last tensors present themselves: shape NCHW, channel stride equal to one element. Returning memory order would assign the row stride to the channel axis.
   </details>
2. `to_layout(x, "NCHW", "NCHW")` copies. Why not return `x`?
   <details><summary>Answer</summary>
   Every other call returns a fresh array, so callers can write to the result. Returning `x` for the same-layout case would make one case alias the input, and a caller writing into the "converted" image would silently modify the original.
   </details>
3. Why widen to int64 in `to_grayscale_pil`?
   <details><summary>Answer</summary>
   $255 \cdot 38470 \approx 9.8 \times 10^6$ does not fit in uint8 or int16, and the sum of the three products plus $2^{15}$ reaches $255 \cdot 65536 + 32768 \approx 1.67 \times 10^7$. NumPy keeps uint8 arithmetic in uint8 and wraps, so the multiplication must happen in a wider type.
   </details>
4. Why must patches be flattened $(c, i, j)$ and not $(i, j, c)$?
   <details><summary>Answer</summary>
   A pretrained patch embedding is a conv weight of shape `[K, C, p, p]`, which flattens row-major in $(c, i, j)$ order. Only patches in the same order make `patchify(x, p) @ w.reshape(K, -1).T` equal the conv. With $(i, j, c)$, every shape still matches and the weights multiply the wrong pixels.
   </details>
5. What happens if the mean is subtracted before scaling?
   <details><summary>Answer</summary>
   $\mu_c$ and $\sigma_c$ are in $[0, 1]$ units. Applied to raw pixels in $[0, 255]$, $(x - \mu_c)/\sigma_c$ gives values up to about $255/0.27 \approx 950$ instead of roughly $\pm 2$: about 255 times too far from zero. The model sees inputs far outside its training range and returns nonsense with no error.
   </details>
