<!-- ol:module L13.6 -->
# Image preprocessing to the specification

## Overview

| | |
|---|---|
| **Module** | `L13.6` · build · Python · Pass 12 · 4 to 6 h |
| **You build** | `python/tinyllm/vision/preprocess.py` |
| **Contract** | `course/contracts/py/tinyllm/vision/preprocess.pyi` |
| **Tests** | `course/tests/L13.6/test_decode.py`, `course/tests/L13.6/test_preprocess.py` |
| **Needs** | `M12.2`, `M12.3`, `L13.4`, `M06.3` |
| **Used by** | `L13.5`; Planned: `L13.7` and `data.10`. Those integrations remain future work. |
| **Milestone** | `MS-L13` |
| **Optional depth** | Compare every supported resize kernel on a thin image and explain edge rounding. |

## Key Takeaways

- A model's pixel recipe is part of its weights' meaning: channel order, crop position and normalization matter.
- Container decoding is the library boundary. Orientation, compositing and transforms are learner code.
- Hash original media bytes; keep preprocessing revision and model identity in separate cache-key fields.

## How to work this chapter

```bash
ol start L13.6
ol tests L13.6
ol check L13.6 --ref-deps
ol diff L13.6
```

## 1. Why now

The vision tower expects RGB tensors, but requests carry compressed image bytes.
The gap includes orientation, transparency and a processor-specific geometry
recipe. A silent mistake gives plausible features that disagree with training.
This module makes that boundary explicit and gives both the model and future
corpus pipeline one implementation to call.

## 2. Principles

Read width and height before allocating decoded pixels. Compare their product
with the configured pixel budget, then reject unsupported color spaces and bit
depths before calling the decoder. Pillow silently reduces true-color 16-bit
PNG to 8-bit RGB, so checking its exposed mode alone is insufficient: inspect
PNG's IHDR bit-depth byte too. Decode one frame and retain its EXIF orientation.

Orientation is an index transform. Cases 2 and 4 reverse columns and rows;
case 3 reverses both. Cases 5 through 8 exchange height and width through a
transpose or quarter-turn. Apply these before resizing and return contiguous
owned storage, not a negative-stride view into a decoder buffer. Expand palette
and gray values to RGB. Composite alpha over white with wide integer arithmetic,
rounding half up only after adding foreground and background contributions.

Load the named HF processor's defaults, then apply config overrides. Read missing
token geometry from the adjacent processor config. Do not guess unknown
processor types or silently treat an unsupported resampler as bilinear.
CLIP resizes the shortest edge then center-crops; SigLIP resizes directly.
Idefics3 puts the global image after its tiles; AnyRes puts it before them.
Use L13.4 for splitting and token counts, M12.3 for interpolation, and M12.2
for rescale/normalize and the HWC-to-CHW conversion.

A centered crop starts at `floor((input-output)/2)`. Negative starts mean zero
padding. For an odd padding difference the extra pixel is at the top or left.
Keep arithmetic stages in order: multiply by rescale factor, subtract channel
mean, divide by channel standard deviation. Store float32 output. A frame mask
covers actual returned frames; cross-image batch padding belongs to the caller.

FNV-1a starts at 14695981039346656037, XORs each byte, multiplies by
1099511628211, and reduces modulo 2^64. Prefix image bytes with 1 and audio
bytes with 2. The pixel-only fixture interface prefixes canonical RGB bytes
with their dimensions; reshaping identical bytes must not preserve identity.

## 3. Worked example

For an already-sized 2 by 2 SigLIP image, no interpolation changes the pixels.
Take RGB `[0,127,255]`, factor `1/255`, means `[0.1,0.2,0.3]`, and standard
deviations `[0.5,1,2]`. The normalized pixel is approximately
`[-0.2,0.29803922,0.35]`. `test_hand_example_identity` checks every pixel of
this small image and checks that preprocessing the same sized uint8 pixels
again gives identical output.

For an Idefics3 image of 9 by 17, longest edge 24 and tile size 8, the resized
short edge is `int(24*9/17)=12`. Growing to tile multiples yields 16 by 24:
2 rows by 3 columns. Append the global image: seven frames, each producing
four tokens, for 28 tokens. Low detail retains one frame and four tokens.
`test_detail_overrides` proves the count and ordering alongside the HF fixture.

An alpha-128 red pixel over white becomes `[255,127,127]`. An orientation-6
image rotates clockwise: its old bottom-left pixel becomes its new top-left.
The frozen decode fixture checks all eight orientations against Pillow's
independent orientation implementation, without importing that transform into
the learner code.

## 4. The interface and tests

| Test | Kind | Contract |
|---|---|---|
| `test_hand_example_identity` | unit, smoke | Hand arithmetic and already-sized identity |
| `test_frozen_decode` | golden | PNG/JPEG decode, EXIF, grayscale, palette and alpha |
| `test_unsupported_before_load`, `test_budget_before_load` | unit | Reject bad headers before decompression |
| `test_corrupt_inputs`, `test_invalid_orientation` | unit | Explicit malformed-input errors |
| `test_media_hash_domain` | unit | D44 domain bytes and exact FNV arithmetic |
| `test_hf_processors` | golden | Four kernels, processor layouts and normalization |
| `test_encoded_identity_and_batch` | property | Encoded and pixel interfaces agree on tensors |
| `test_detail_overrides` | unit | Low/high/auto splitting and image-token counts |
| `test_bad_config`, `test_input_and_allocation_bounds` | boundary | Fail before invalid arithmetic or allocation |
| `test_companion_geometry` | unit | Processor-config geometry and explicit precedence |

Read the [pixel specification](../../../../course/contracts/formats/image-preprocess.md)
and the [Python contract](../../../../course/contracts/py/tinyllm/vision/preprocess.pyi).
The HF fixture is generated without any learner imports, using the PIL backend
so its resize convention is the same one taught in M12.3. Expected pixels allow
one uint8 unit of error; normalized values allow that error divided by
`255*min(std)`. Metadata records the generator versions and manifest digest.

## 5. Pitfalls

| Mistake | Symptom | Test |
|---|---|---|
| Ignore EXIF or swap clockwise and counterclockwise | Rectangular photos face the wrong way | `test_hand_example_identity` | unit, smoke | Hand arithmetic and already-sized identity |
| `test_frozen_decode` |
| Multiply alpha in uint8 or composite on black | Transparent edges change color | `test_hand_example_identity` | unit, smoke | Hand arithmetic and already-sized identity |
| `test_frozen_decode` |
| Decode before checking dimensions | A small compressed file causes a huge allocation | `test_budget_before_load` |
| Normalize before rescaling | Inputs remain numerically plausible but disagree with training | `test_hand_example_identity` |
| Put the global image on the wrong end | Spatial markers point at the wrong feature sequence | `test_hf_processors` |
| Reuse decoded-byte hashes for encoded media | Runtime identities no longer follow D44 | `test_encoded_identity_and_batch` |

## 6. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `M12.2` | Rescale/normalize and CHW conversion. |
| Back | `M12.3` | Learner-written PIL-compatible resize kernels. |
| Back | `L13.4` | Dynamic resolution, frame ordering and token counts. |
| Back | `M06.3` | FNV-1a content identifiers. |
| Forward | `L13.7` | Planned VLM image inputs and placeholder expansion. |
| Forward | `data.10` | Planned corpus decode and orientation before filtering. |

| Forward | `L13.5` | `DualEncoder.encode_images` consumes the complete single-frame image pipeline. |

## Going further

| Resource | Exercise |
|---|---|
| [Independent oracle](../../../../course/oracle/L13.6/preprocess_hf.py) | Explain why disabling Idefics3's first resize still permits splitting. |
| [Decode oracle](../../../../course/oracle/L13.6/decode_pil.py) | Trace an RGBA pixel through white compositing and EXIF orientation. |
