# Image preprocessing, revision 1

`preprocess_rev` is a positive u32. Increment it when changing a pixel recipe.
Encoder cache keys include this revision separately from the D44 media hash.
Python decodes containers with Pillow, and Rust uses the image crate. Neither
implementation delegates the learning exercises in this specification to a
third-party image transform pipeline.

## Decode and orientation

Read dimensions before decompression and enforce `max_pixels`. Reject CMYK,
16-bit PNG (including true-color PNG that decoders may down-convert), malformed
containers and truncated pixel streams. Use the first frame of animated files.
Apply EXIF tag 0x0112 before resizing: 1 identity, 2 horizontal flip, 3 half-turn,
4 vertical flip, 5 transpose, 6 clockwise quarter-turn, 7 transpose then both
flips, 8 counterclockwise quarter-turn. A present tag outside 1 through 8 fails.

Expand grayscale and palette entries to RGB. For opacity `a` and channel `c`,
white compositing is `(c*a + 255*(255-a) + 127) // 255`, evaluated in a wide
integer type. Honor palette and color-key transparency. Result: owned,
contiguous uint8 HWC RGB. Decode parity is exact for PNG and within 2 LSB for
JPEG against committed synthetic pixels.

## Pixel pipeline

- CLIP and LLaVA: resize the shorter edge, truncate the longer edge toward
  zero, then center-crop. Explicit height/width sizes resize directly. An
  optional longest edge caps a shorter-edge resize while keeping aspect ratio.
- SigLIP: resize to explicit height and width, without a center crop by default.
- A crop starts at `floor((input-output)/2)`. If the image is smaller than the
  crop, pad with zeros; an odd extra padding pixel goes on the top or left.
- Idefics3/SmolVLM: L13.4 longest-edge resize, tile-multiple resize, row-major
  tiles, then global image. `do_resize=false` skips the first resize only.
- LLaVA-NeXT: L13.4 AnyRes fit and centered zero padding, base image first,
  then row-major tiles. Apply configured per-frame resize and crop afterward.
- `detail=low` disables splitting, `high` enables it, and `auto` follows
  `do_image_splitting`. Single-frame processors ignore the splitting switch.

Use M12.3 nearest, bilinear, bicubic (`a=-0.5`) or Lanczos3. Coordinates are
`(dst+0.5)*in/out-0.5`. Downsampling stretches filter support, normalizes each
output's weights, and uses the PIL horizontal-then-vertical uint8 rounding.
HF enum values are 0, 2, 3 and 1 respectively; BOX and HAMMING are unsupported.

After geometry, apply `x*rescale_factor`, then `(x-mean)/std`, and return float32
CHW. Either arithmetic stage can be disabled independently. Means and positive
standard deviations have three finite RGB entries. Config defaults follow the
named HF processor; unknown JSON keys are ignored but unsupported values fail.
The companion `processor_config.json` supplies missing token geometry. Explicit
preprocessor values take precedence. Full feature selection retains additional
base tokens; default selection removes the CLS token.

## Shapes, tokens and identity

Return one `ImageInputs` per original image. `pixel_values` is contiguous
`[frames,3,H,W]`; `pixel_mask` is boolean `[frames,H,W]`. Only real frames are
returned, so masks are all true. Geometry padding inside a frame is part of
its input, not missing batch data. Batch assembly may add padded frames later.
`grid` is `(rows,cols)`, or `(0,0)` for one image. Preserve oriented original
`(H,W)` for AnyRes feature unpadding. `n_tokens` is L13.4's normative count,
including AnyRes newline features. It is not always frame count times patches.

The runtime byte-input identity is `fnv1a64(0x01 || original_file_bytes)`.
The predecoded fixture interface has no original file: its identity is
`fnv1a64(0x01 || "RGB8" || u32_le(height) || u32_le(width) || RGB_bytes)`.
Audio uses domain byte 0x02. Hashes identify content; they are not integrity
proofs. Pixel and encoded identities intentionally differ. Do not include
resize settings in this hash: model identity, detail and revision are separate
cache-key fields.

Input and intermediate allocations are bounded by `max_pixels`. Idefics3 also
bounds the configured longest-edge square conservatively, including its 4096
cap. Empty batches return empty lists; preprocessing never modifies inputs.

Independent HF PIL fixtures cover CLIP, SigLIP, Idefics3 and LLaVA-NeXT.
Resize parity allows 1 LSB; normalized parity allows `1/(255*min(std))`.
