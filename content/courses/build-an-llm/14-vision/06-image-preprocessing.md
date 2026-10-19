# Image preprocessing to the specification

The vision tower expects RGB tensors, but requests carry compressed image bytes. Between the two sit orientation, transparency, and a processor-specific geometry recipe, and a silent mistake anywhere in that gap gives plausible features that disagree with training. A model's pixel recipe is part of what its weights mean: channel order, crop position, and normalization all matter, and none of them raises an error when wrong.

This section makes that boundary explicit and gives both the model and a future corpus pipeline one implementation to call. [Contrastive image-text: CLIP and SigLIP](05-contrastive-image-text.md) already consumes it through `DualEncoder.encode_images`; the planned vision-language model section and a planned data-engineering corpus step will use it too. Together with the rest of this chapter it lets the vision tower consume real encoded images.

You will build on the rescale, normalize, and HWC-to-CHW conversion of [Image tensors and layouts](../../mathematics-for-computing/13-signals-and-images/02-image-tensors-and-layouts.md), the Pillow-compatible kernels of [Resampling and interpolation](../../mathematics-for-computing/13-signals-and-images/03-resampling-and-interpolation.md), and the FNV-1a hashing of [Modular arithmetic, hashing, and PCG32](../../mathematics-for-computing/07-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md), all in *Mathematics for Computing*, plus the splitters and token counts of [Dynamic resolution and tiling](04-dynamic-resolution-and-tiling.md). The full solution is [`preprocess.py`](../../../../course/solutions/python/tinyllm/vision/preprocess.py); the contracts are the Python stub [`preprocess.pyi`](../../../../course/contracts/py/tinyllm/vision/preprocess.pyi) and the [image preprocessing specification](../../../../course/contracts/formats/image-preprocess.md).

> **Learning objectives**
> - Load a Hugging Face processor config faithfully and reject anything you cannot reproduce exactly.
> - Decode untrusted image bytes safely, checking the pixel budget before decompression.
> - Apply EXIF orientation, palette expansion, and exact alpha compositing in your own code.
> - Reproduce the CLIP, SigLIP, Idefics3, and AnyRes geometry and normalization recipes.
> - Compute a media identity hash from the original bytes, separate from the preprocessing revision.

## Decoding untrusted bytes

Decoding is the one untrusted, expensive step, so it is where the pipeline starts. Read width and height before allocating decoded pixels. Compare their product with the configured pixel budget, then reject unsupported color spaces and bit depths before calling the decoder. Pillow silently reduces true-color 16-bit PNG to 8-bit RGB, so checking its exposed mode alone is insufficient: inspect PNG's IHDR bit-depth byte too. Decode one frame and retain its EXIF orientation.

The division of labour is deliberate. Container decoding is the library boundary; orientation, compositing, and transforms are the solution's own code, so their semantics are fixed and inspectable.

## Orientation, palette, and alpha

Orientation is an index transform. Cases 2 and 4 reverse columns and rows; case 3 reverses both. Cases 5 through 8 exchange height and width through a transpose or quarter-turn. Apply these before resizing and return contiguous owned storage, not a negative-stride view into a decoder buffer.

After orientation, expand palette and gray values to RGB. Composite alpha over white with wide integer arithmetic, rounding half up only after adding the foreground and background contributions.

## Geometry by processor family

With clean RGB pixels in hand, the geometry depends on which processor the model was trained with. Load the named Hugging Face processor's defaults, then apply config overrides. Read missing token geometry from the adjacent processor config. Do not guess unknown processor types or silently treat an unsupported resampler as bilinear.

The families differ in a few ways. CLIP resizes the shortest edge then center-crops; SigLIP resizes directly. Idefics3 puts the global image after its tiles; AnyRes puts it before them. The splitting and token counts come from the tiling section, the interpolation from the resampling section of the mathematics book, and rescale, normalize, and the HWC-to-CHW conversion from its image layouts section.

A centered crop starts at `floor((input-output)/2)`. Negative starts mean zero padding. For an odd padding difference the extra pixel is at the top or left.

## Normalization and masks

Keep the arithmetic stages in order: multiply by the rescale factor, subtract the channel mean, divide by the channel standard deviation. Store float32 output. A frame mask covers the frames actually returned; cross-image batch padding belongs to the caller.

## Media identity

The last piece is identity for caching and provenance. Hash the original media bytes, and keep the preprocessing revision and model identity in separate cache-key fields. FNV-1a starts at 14695981039346656037, XORs each byte, multiplies by 1099511628211, and reduces modulo $2^{64}$. Prefix image bytes with 1 and audio bytes with 2. The pixel-only fixture interface prefixes canonical RGB bytes with their dimensions; reshaping identical bytes must not preserve identity.

## Working it by hand

Start with normalization alone. For an already-sized 2 by 2 SigLIP image, no interpolation changes the pixels. Take RGB `[0,127,255]`, factor `1/255`, means `[0.1,0.2,0.3]`, and standard deviations `[0.5,1,2]`. The normalized pixel is approximately `[-0.2,0.29803922,0.35]`. A correct implementation reproduces every pixel of this small image and gives identical output when the same sized uint8 pixels are preprocessed again.

Next, geometry. For an Idefics3 image of 9 by 17, longest edge 24 and tile size 8, the resized short edge is `int(24*9/17)=12`. Growing to tile multiples yields 16 by 24: 2 rows by 3 columns. Append the global image: seven frames, each producing four tokens, for 28 tokens. Low detail retains one frame and four tokens. The HF processor produces the same count and frame order.

Finally, the per-pixel transforms. An alpha-128 red pixel over white becomes `[255,127,127]`. An orientation-6 image rotates clockwise: its old bottom-left pixel becomes its new top-left. All eight orientations agree with Pillow's own orientation implementation, even though the solution never imports that transform.

## The contract

The public entry points are `ProcessorConfig` (with `from_hf`, `validate`, and `tiling`), `decode_image`, `media_hash`, `preprocess_pixels`, and `preprocess`; their signatures are in the [Python contract](../../../../course/contracts/py/tinyllm/vision/preprocess.pyi), and the pixel-level rules are in the [pixel specification](../../../../course/contracts/formats/image-preprocess.md). The Hugging Face reference outputs are generated with the PIL backend, so their resize convention is the one taught in the resampling section. Expected pixels allow one uint8 unit of error; normalized values allow that error divided by `255*min(std)`.

## Building it step by step

Eight principles determine the implementation.

1. **The weights define the input distribution.** The tower was trained on tensors produced by one specific processor. Any difference in channel order, crop offset, kernel, or normalization constants shifts the input distribution without raising an error. So the recipe must be read from the model's `preprocessor_config.json`, not guessed, and anything the solution cannot reproduce exactly must be rejected.
2. **Decoding is the only untrusted, expensive step.** A few kilobytes of PNG can declare a 50000 by 50000 image. Dimensions live in the header, so the budget check `width*height <= max_pixels` must happen after `Image.open` (header only) and before `image.load()` (decompression).
3. **Pillow is a container parser, not the recipe.** It decompresses pixels, but its `mode` hides some facts (16-bit PNG arrives as 8-bit RGB). Every transform that changes meaning (orientation, palette, alpha, resize) is written in the solution, so its semantics are fixed and inspectable.
4. **Orientation is a pure index permutation**, so it is exact and must happen before any interpolation; resizing first and rotating after would change which source pixels are blended for non-square targets.
5. **Compositing must be exact.** Over white, $c' = (c\alpha + 255(255-\alpha))/255$. Both products reach 65025, so the arithmetic needs a type wider than uint8, and rounding happens once, after the sum (the alpha-128 red pixel gives $65152/255 = 255.5 \to 255$ and $32512/255 = 127.5 \to 127$ with `(x + 127) // 255`).
6. **Geometry follows the processor family.** Single-frame processors resize (shortest edge or exact size) then optionally center-crop; Idefics3 and AnyRes reuse the tiling splitters. Only the frame order differs: Idefics3 appends the global image, AnyRes prepends it.
7. **Normalization order is fixed**: rescale, subtract mean, divide by std, then HWC to CHW in float32. `rescale_normalize` from the mathematics book already does exactly this.
8. **Identity uses the original bytes.** Two encodings of the same pixels are different media for caching and provenance, so the runtime hash covers the encoded file, prefixed by a kind byte so an image and an audio file with the same bytes cannot collide.

### Configuration: defaults, then overrides, then validation

`from_hf` starts from per-processor defaults, layers the JSON on top, fills token geometry from the adjacent `processor_config.json` only when the main file is silent, and converts Hugging Face's integer resample codes. An unknown code becomes the string `"unsupported"` so that `validate` rejects it rather than silently using bilinear.

```python
name = (
    raw.get("image_processor_type", "CLIPImageProcessor")
    .removesuffix("Pil")
    .removesuffix("Fast")
)
if name not in (
    "CLIPImageProcessor", "SiglipImageProcessor", "Idefics3ImageProcessor",
    "SmolVLMImageProcessor", "LlavaImageProcessor", "LlavaNextImageProcessor",
):
    raise ValueError(f"unsupported image_processor_type: {name!r}")  # never guess
# ... per-family defaults (SigLIP: direct 224x224 resize, no crop, mean=std=0.5)
companion = path.with_name("processor_config.json")
if companion.is_file() and companion != path:
    extra = json.loads(companion.read_text())
    # ...
    for key in ("image_seq_len", "scale_factor", "patch_size"):
        if key in extra and key not in raw:  # explicit preprocessor values win
            raw[key] = extra[key]
known = {f.name for f in fields(cls)}
cfg = cls(**(defaults | {k: v for k, v in raw.items() if k in known}))
if isinstance(cfg.resample, int) and not isinstance(cfg.resample, bool):
    cfg.resample = {
        0: "nearest", 1: "lanczos3", 2: "bilinear", 3: "bicubic",
    }.get(cfg.resample, "unsupported")  # validate() will reject this
# ... scalar mean/std broadcast to three channels, int sizes to dicts,
# image_seq_len derived from (feature side // (patch_size*scale_factor))**2
cfg.validate()
```

`validate` is called again at the top of every public entry point, because a `ProcessorConfig` is a mutable dataclass and may have been built by hand.

### Decoding: header checks before decompression

```python
with Image.open(io.BytesIO(data)) as image:          # parses the header only
    if image.width * image.height > max_pixels:
        raise ValueError(f"image exceeds max_pixels={max_pixels}")
    # Pillow exposes 16-bit RGB PNG as mode RGB after discarding
    # low bits, so inspect the IHDR depth before decompression.
    if image.format == "PNG" and len(data) > 24 and data[24] == 16:
        raise ValueError("unsupported 16-bit PNG")
    if image.mode not in ("1", "L", "LA", "P", "RGB", "RGBA"):
        raise ValueError(f"unsupported image mode: {image.mode}")
    orientation = image.getexif().get(0x0112)
    if orientation is not None and (
        not isinstance(orientation, int) or orientation not in range(1, 9)
    ):
        raise ValueError(f"invalid EXIF orientation: {orientation!r}")
    image.load()                                       # first allocation of pixels
    return _orient(_rgb(image), orientation)
```

Byte 24 is the IHDR bit depth: 8 bytes of PNG signature, 4 of chunk length, 4 of chunk type, then width (4) and height (4). The whole block runs with `DecompressionBombWarning` promoted to an error, and Pillow's `OSError` and `SyntaxError` become `ValueError`, so every malformed input fails the same way.

### Orientation: eight index permutations, then an owned copy

```python
elif orientation == 6:
    result = np.rot90(pixels, -1, axes=(0, 1))   # clockwise: old bottom-left -> top-left
elif orientation == 7:
    result = pixels.transpose(1, 0, 2)[::-1, ::-1]  # transverse
elif orientation == 8:
    result = np.rot90(pixels, 1, axes=(0, 1))    # counterclockwise
# ...
# Views with negative strides would alias the decoder buffer; copy to C order.
return np.array(result, dtype=np.uint8, order="C", copy=True)
```

### Palette, gray, and alpha to RGB on white

`_rgb` expands `P` through an RGBA palette (with `tRNS` transparency either a single index or a per-index table), repeats gray to three channels, and then composites any alpha:

```python
if alpha is not None:
    # Exact integer round-half-up of compositing over white. uint32
    # avoids overflow for both products and their sum.
    opacity = np.asarray(alpha, dtype=np.uint32)[..., None]
    rgb = (rgb.astype(np.uint32) * opacity + 255 * (255 - opacity) + 127) // 255
return np.asarray(rgb, dtype=np.uint8)
```

### Single-frame geometry

The shortest edge is scaled to the target, the long edge follows with truncation (matching Hugging Face), and an optional `longest_edge` cap rescales both. The crop starts at `floor((input-output)/2)`; a negative start means the image is smaller than the crop and the remainder stays zero.

```python
def _resize_size(hw, size):
    if "height" in size:                      # SigLIP-style direct resize
        return size["height"], size["width"]
    h, w = hw
    short, long = min(h, w), max(h, w)
    new_short = size["shortest_edge"]
    new_long = int(new_short * long / short)  # truncation, as HF does
    maximum = size.get("longest_edge")
    if maximum is not None and new_long > maximum:
        new_short, new_long = max(1, int(maximum * new_short / new_long)), maximum
    return (new_short, new_long) if h <= w else (new_long, new_short)

def _center_crop(a, size):
    h, w = a.shape[:2]
    oh, ow = size
    # floor: an odd crop drops the extra row at the bottom/right,
    # an odd pad puts the extra zero row at the top/left
    top, left = (h - oh) // 2, (w - ow) // 2
    out = np.zeros((oh, ow, 3), dtype=np.uint8)  # zero padding when top/left < 0
    sy, sx = max(top, 0), max(left, 0)
    dy, dx = max(-top, 0), max(-left, 0)
    nh, nw = min(h - sy, oh - dy), min(w - sx, ow - dx)
    out[dy : dy + nh, dx : dx + nw] = a[sy : sy + nh, sx : sx + nw]
    return out
```

### The pipeline

`preprocess_pixels` picks the frames by processor family, bounding each output allocation before it is made, then normalizes all frames with one shared function:

```python
split = cfg.do_image_splitting if detail == "auto" else detail == "high"
# ...
if cfg.kind == "idefics3":
    target = idefics3_resize_size((h, w), tiling.longest_edge, cfg.tile_size, split)
    # Include the global tile in the allocation bound, before resizing.
    output_area = target[0] * target[1] + cfg.tile_size**2
    # ... raise if output_area > cfg.max_pixels
    tiles, base, grid = idefics3_split(
        a, tiling.longest_edge, cfg.tile_size, split, kernel=cfg.resample
    )
    frames = tiles + [base]                       # Idefics3: global image last
elif cfg.kind == "llava_next":
    # ... bound th*tw + bh*bw, then anyres_tiles when split
    base = resize(a, (bh, bw), cfg.resample)
    frames = [_single(frame, cfg) for frame in [base] + tiles]  # AnyRes: global first
else:
    frames = [_single(a, cfg)]
values = np.stack([
    rescale_normalize(                            # rescale, -mean, /std, HWC->CHW
        frame,
        cfg.rescale_factor if cfg.do_rescale else 1.0,
        cfg.image_mean if cfg.do_normalize else (0.0,) * 3,
        cfg.image_std if cfg.do_normalize else (1.0,) * 3,
    )
    for frame in frames
])
# Pixel-only fixtures have no encoded file. Frame the canonical bytes
# with dimensions so reshapes of the same flat bytes cannot alias.
identity = b"RGB8" + struct.pack("<II", h, w) + a.tobytes(order="C")
```

The mask is `np.ones((len(frames), H, W))`: it covers only the frames this image produced, because padding a batch of images to a common frame count is the caller's job. `n_tokens` comes from the tiling section's `image_token_count`, so token counting and tiling cannot drift apart.

### The encoded entry point and the hash

`preprocess` decodes, reuses the pixel path, then overwrites the pixel-derived hash with the hash of the original bytes:

```python
def media_hash(kind, data):
    """D44: FNV-1a-64 of kind byte (1=image, 2=audio) followed by bytes."""
    if kind not in ("image", "audio"):
        raise ValueError(f"unsupported media kind: {kind!r}")
    return fnv1a64(bytes([1 if kind == "image" else 2]) + data)

for data in images:
    item = preprocess_pixels([decode_image(data, cfg.max_pixels)], cfg, detail)[0]
    item.media_hash = media_hash("image", data)  # identity follows encoded bytes
    result.append(item)
```

The `D44` in the docstring names the media-identity decision in the specification. The full solution is [`preprocess.py`](../../../../course/solutions/python/tinyllm/vision/preprocess.py), and splitting and token counts are in [`tiling.py`](../../../../course/solutions/python/tinyllm/vision/tiling.py).

### Why this design

Decoding and every transform are $O(HW)$ in time and memory. The budget checks bound $HW$ (including resized and tiled outputs) before any large allocation, so the worst case is set by configuration, not by input.

Using Pillow's `ImageOps.exif_transpose`, `convert("RGB")`, and `resize` would be shorter, but `convert("RGB")` drops alpha onto black, and the transforms would be a library's choices rather than a written specification. A float compositing formula would round differently at exactly-half values. Checking the budget after `load()` is too late: the allocation has already happened.

The tradeoffs accepted are explicit. Unsupported modes (CMYK, 16-bit) and unknown processor types are rejected instead of approximated, so some valid images fail loudly. Only the first frame of an animated container is used.

## What can go wrong

A correct implementation is checked on the cases below.

| Case | Kind of check | Contract |
|---|---|---|
| Hand example on an already-sized image | unit, smoke | Hand arithmetic and already-sized identity |
| Frozen PNG/JPEG decodes | golden | PNG/JPEG decode, EXIF, grayscale, palette and alpha |
| Unsupported modes and over-budget headers | unit | Reject bad headers before decompression |
| Corrupt bytes and invalid orientation tags | unit | Explicit malformed-input errors |
| Media-hash domain bytes | unit | The specification's domain bytes and exact FNV arithmetic |
| HF processor outputs | golden | Four kernels, processor layouts and normalization |
| Encoded vs pixel interface, batches | property | Encoded and pixel interfaces agree on tensors |
| `detail` overrides | unit | Low/high/auto splitting and image-token counts |
| Bad configs, oversized inputs and outputs | boundary | Fail before invalid arithmetic or allocation |
| Companion `processor_config.json` geometry | unit | Processor-config geometry and explicit precedence |

The typical mistakes all produce plausible output, which is what makes them dangerous. Ignoring EXIF, or swapping clockwise and counterclockwise, makes rectangular photos face the wrong way; `_orient` maps all eight tags explicitly (6 is `rot90(-1)`, 8 is `rot90(1)`) before any resize. Multiplying alpha in uint8 or compositing on black changes the color of transparent edges; `_rgb` composites over white in uint32 with one rounding after the sum. Normalizing before rescaling leaves inputs numerically plausible but in disagreement with training; all frames go through `rescale_normalize`, which applies rescale, mean, and std in that order.

> **Warning:** Decoding before checking dimensions lets a small compressed file cause a huge allocation. `decode_image` checks `width*height` against `max_pixels` before `image.load()`.

Two mistakes concern ordering and identity. Putting the global image on the wrong end of the frame list makes the spatial markers point at the wrong feature sequence; the solution uses `tiles + [base]` for Idefics3 and `[base] + tiles` for AnyRes. Reusing decoded-byte hashes for encoded media means runtime identities no longer follow the specification's rule; `preprocess` sets `media_hash("image", data)` from the original bytes.

## Where this leads

[Contrastive image-text: CLIP and SigLIP](05-contrastive-image-text.md) consumes the complete single-frame pipeline through `DualEncoder.encode_images`. The planned vision-language model section will use it for image inputs and placeholder expansion, and a planned data-engineering corpus step will use its decode and orientation before filtering.

### Further exploration

- Compare every supported resize kernel on a thin image and explain the edge rounding.
- Read [`tiling.py`](../../../../course/solutions/python/tinyllm/vision/tiling.py) and explain why disabling Idefics3's first resize still permits splitting.
- Using the [pixel specification](../../../../course/contracts/formats/image-preprocess.md), trace an RGBA pixel through white compositing and EXIF orientation.

## Key terms

- **pixel budget**: the configured maximum `width*height`, checked from the header before decompression.
- **EXIF orientation**: a tag (1 to 8) saying how stored pixels must be flipped or rotated for display.
- **compositing over white**: blending a transparent pixel with a white background, $c' = (c\alpha + 255(255-\alpha))/255$.
- **processor config**: the `preprocessor_config.json` that fixes a model's resize, crop, and normalization recipe.
- **media hash**: an FNV-1a-64 hash of a kind byte followed by the original encoded bytes.

## Summary

- A model's pixel recipe is part of its weights' meaning: channel order, crop position, and normalization matter.
- Container decoding is the library boundary; orientation, compositing, and transforms are the solution's own code.
- Check the pixel budget between reading the header and decompressing, and reject what cannot be reproduced exactly.
- Normalize in a fixed order (rescale, subtract mean, divide by std), and order frames by processor family.
- Hash original media bytes; keep preprocessing revision and model identity in separate cache-key fields.

## Review questions

1. Why must the pixel budget be checked between `Image.open` and `image.load()`?

   <details><summary>Answer</summary>

   `open` reads only the header, which already holds width and height; `load` decompresses and allocates. Checking earlier is impossible and checking later lets a tiny file trigger a huge allocation.

   </details>

2. Why is the alpha formula computed in uint32 and rounded with `+ 127` after the sum?

   <details><summary>Answer</summary>

   Each product is up to 255 by 255, which overflows uint8 and uint16 once summed. Rounding once after adding the foreground and background contributions gives the exact round-half-up of the true composite; rounding each term separately can be off by one.

   </details>

3. Why does `preprocess` replace the hash computed by `preprocess_pixels`?

   <details><summary>Answer</summary>

   The pixel path has no file, so it hashes dimensions plus canonical RGB bytes. At runtime the identity must follow the encoded bytes, as the specification requires: a PNG and a JPEG of the same picture are different inputs, and the encoded hash can be computed without decoding.

   </details>

4. What breaks if the global image is put at the wrong end of the frame list?

   <details><summary>Answer</summary>

   Nothing raises, but the prompt's spatial markers (from the tiling section) name tile positions in a fixed order. With the global frame on the wrong end, every marker points at the features of a different frame.

   </details>

5. Why does `_orient` copy the result instead of returning the view?

   <details><summary>Answer</summary>

   Flips and rotations are views with negative strides into the decoder's buffer. Downstream code (resize kernels, `tobytes` for hashing) assumes contiguous, owned uint8 storage; a copy makes that true and detaches the array from Pillow.

   </details>
