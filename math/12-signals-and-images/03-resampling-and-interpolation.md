<!-- ol:module M12.3 -->
# Resampling and interpolation

## Overview

| | |
|---|---|
| **Module** | `M12.3` · build · Python · Pass 12 · 4 to 5 h |
| **You build** | `python/tinyllm/sig/resample.py`: `kernel_weights` (nearest, bilinear, bicubic, Lanczos3, with antialias), `resize` (Pillow's fixed-point passes bit for bit, or a float mode), `resize_planar_f32` (`F.interpolate` semantics for position tables), and `resample_audio` (torchaudio's windowed-sinc resampler) |
| **Contract** | [`course/contracts/py/tinyllm/sig/resample.pyi`](../../course/contracts/py/tinyllm/sig/resample.pyi) |
| **Tests** | `course/tests/M12.3/test_resample.py` (what they check: section 4), golden values from `course/oracle/M12.3/resize_pil.py` (Pillow 12.3, torchvision 0.29, `F.interpolate`) and `course/oracle/M12.3/resample_torchaudio.py` (torchaudio 2.11) in `course/fixtures/M12.3/` |
| **Needs** | `M12.1` `conv1d_direct` (or `--ref-deps`). Reading: `M12.2` (uint8 HWC images), `M02.1` (series and the sinc), `M12.5` (aliasing, as a concept) |
| **Used by** | later `L13.6` (`resize` in every image processor) · `L13.3` (`interpolate_pos_embed` through `resize_planar_f32`) · `L14.1` (`resample_audio` to 16 kHz) · `data.10` (hash thumbnails); they join the registry with B14's L13, L14, and data groups |
| **Milestone** | `MS-P12` (the multimodal gate) |
| **Optional depth** | Keys, "Cubic convolution interpolation for digital image processing" (1981); Smith, *Digital Audio Resampling Home Page* (free, the bandlimited interpolation derivation); Pillow `src/libImaging/Resample.c` |

## Key Takeaways

- Resampling is interpolation (a kernel turns samples back into a continuous signal) followed by sampling on the new grid; every output is a weighted sum of a few inputs, with weights that sum to 1 (`test_weights_sum_to_one`).
- Shrinking without first removing detail the new grid cannot hold aliases; antialiasing stretches the kernel by the scale factor, which low-pass filters in the same pass (`test_hand_example_bilinear_antialias`, `test_antialias_prevents_aliasing`).
- Pillow runs separable passes, horizontal then vertical, in 22-bit fixed point and rounds to uint8 between them; matching it bit for bit means copying all three choices (`test_pil_mode_is_pillow_bit_for_bit`).
- `F.interpolate` is a different convention (sample at $(o + 0.5)\,\text{in}/\text{out} - 0.5$, clamp at the edge, bicubic $a = -0.75$), and it is the one HF position tables are resized with (`test_planar_matches_interpolate`).
- Audio resampling is the same idea in 1-D: a bank of windowed-sinc low-pass filters applied as one strided conv (`test_resample_audio_matches_torchaudio`, `test_band_limited_round_trip`).

## How to work this chapter

```bash
ol start M12.3              # stubs python/tinyllm/sig/resample.py into your repo
ol tests M12.3              # read the test catalog first: rung R0, you write no tests here
ol check M12.3              # exit code is the verdict
ol check M12.3 --ref-deps   # only if your M12.1 is not passing yet
ol diff  M12.3              # after passing: your code against the reference
```

---

## 1. Why now

Every image a vision model sees is resized first: CLIP wants 224 by 224, SigLIP 384, Idefics3 splits large images into 512-pixel tiles. The checkpoints were trained on images resized by Pillow, so `L13.6` must reproduce Pillow's output, and the preprocessing parity rule (DESIGN D48) allows at most one unit in the last place. A naive bilinear resize that samples at pixel corners, or shrinks without antialiasing, produces images that look fine and shift model outputs measurably. Audio has the same problem in time: Whisper hears 16 kHz, recordings arrive at 44.1 or 48 kHz, and `L14.1` must bring them down without folding high frequencies into the speech band.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $n_\text{in}$, $n_\text{out}$ | input and output length along one axis | ints |
| $\text{scale} = n_\text{in} / n_\text{out}$ | input samples per output sample | float |
| $c_x = (x + 0.5)\,\text{scale}$ | center of output $x$ in input coordinates | float |
| $k(t)$ | interpolation kernel, support $a_k$ | function |
| $f_s = \max(\text{scale}, 1)$ | antialias stretch | float |
| $w_{x, j}$ | weight of input $j$ in output $x$ | float |
| $a$ | bicubic (Keys) parameter | float |
| $\text{sr}$ | sample rate | Hz |
| $\operatorname{sinc}(t) = \sin(\pi t)/(\pi t)$ | the ideal low-pass kernel | function |

### 2.1 Interpolation kernels

Interpolation evaluates a continuous image between the pixels as $\hat{x}(u) = \sum_j x[j]\, k(u - j)$. Nearest picks one pixel. Bilinear uses the triangle $k(t) = \max(0, 1 - |t|)$ (support 1). Bicubic uses Keys' cubic, support 2:
$$k(t) = \begin{cases} (a+2)|t|^3 - (a+3)|t|^2 + 1 & |t| < 1 \\ a|t|^3 - 5a|t|^2 + 8a|t| - 4a & 1 \le |t| < 2 \\ 0 & \text{otherwise}\end{cases}$$
Pillow uses $a = -0.5$ (the value that makes the cubic reproduce quadratics exactly); PyTorch's non-antialiased bicubic uses $a = -0.75$. Lanczos3 is a windowed sinc, $\operatorname{sinc}(t)\operatorname{sinc}(t/3)$ on $[-3, 3)$, the closest of the four to the ideal low-pass filter. All four are 1 at 0 and 0 at the other integers, so resizing to the same size changes nothing.

### 2.2 Weights per output pixel

Output pixel $x$ sits at the center of its footprint, $c_x = (x + 0.5)\,\text{scale}$ in input coordinates, where input pixel $j$ is centered at $j + 0.5$. Its taps are the inputs within the kernel's support of that center, from $\text{lo} = \max(\operatorname{int}(c_x - a_k f_s + 0.5), 0)$ to $\text{hi} = \min(\operatorname{int}(c_x + a_k f_s + 0.5), n_\text{in})$, with weights
$$w_{x, j} = \frac{k\big((j - c_x + 0.5)/f_s\big)}{\sum_{j'=\text{lo}}^{\text{hi}-1} k\big((j' - c_x + 0.5)/f_s\big)} .$$
Dividing by the sum of the kept taps, not by the kernel's total area, renormalizes at the border where taps fall off the image: a flat image stays flat up to its edge.

### 2.3 Antialiasing is a stretched kernel

Shrinking by a factor `scale` lowers the Nyquist frequency (`M12.5`) of the new grid by that factor. Detail between the two limits does not vanish: it aliases into false low-frequency patterns (moire). The fix is to low-pass filter first, and stretching the kernel by $f_s = \max(\text{scale}, 1)$ does exactly that: a kernel $f_s$ times wider has a frequency response $f_s$ times narrower. Pillow always does it; torchvision does it with `antialias=True`. Enlarging needs no filter, so $f_s = 1$. The float mode of `resize` is these weights without rounding, which is torchvision's antialiased float resize.

### 2.4 Separable passes and Pillow's arithmetic

All four kernels are separable, so a 2-D resize is a horizontal pass on every row and then a vertical pass on every column, and a pass whose size does not change is skipped. Pillow does integer arithmetic for 8-bit images: each weight becomes $\operatorname{round}(w \cdot 2^{22})$ (halves away from zero), each output accumulates $2^{21} + \sum_j x[j]\, w^\text{fixed}_j$, and the result is $\operatorname{clip}(\text{acc} \gg 22, 0, 255)$. The intermediate image after the horizontal pass is uint8 again. Pass order, the rounding of the weights, and the rounding between passes each change a few pixels by one level, and together they are what "matches Pillow" means. Nearest is special in Pillow: it copies input $\operatorname{int}(s_x)$ where $s_0 = \text{scale}/2$ and $s_{x+1} = s_x + \text{scale}$ is accumulated in float64.

### 2.5 `F.interpolate`, a different convention

PyTorch's non-antialiased interpolation maps output $o$ to the source coordinate $(o + 0.5)\,\text{in}/\text{out} - 0.5$ (or $o\,(\text{in} - 1)/(\text{out} - 1)$ with `align_corners=True`), takes the 2 (bilinear) or 4 (bicubic, $a = -0.75$) neighbours of that coordinate, and reads the edge pixel for neighbours outside the image instead of renormalizing. It never stretches the kernel. HF's `interpolate_pos_embed` resizes a ViT's position table this way, so `resize_planar_f32` implements it; with `antialias=True` it uses the Pillow weights in float, as PyTorch's antialiased path does.

### 2.6 Band-limited audio resampling

The Whittaker-Shannon formula reconstructs a band-limited signal exactly from its samples with sinc kernels. Resampling from rate $\text{sr}_\text{in}$ to $\text{sr}_\text{out}$ evaluates that reconstruction, low-passed below the smaller Nyquist rate, at the new instants. With $g = \gcd$, $\text{orig} = \text{sr}_\text{in}/g$ and $\text{new} = \text{sr}_\text{out}/g$, the output pattern repeats every $\text{orig}$ input samples, so torchaudio precomputes $\text{new}$ filters (one per output phase), each a sinc at cutoff $\min(\text{orig}, \text{new}) \cdot \text{rolloff}$ tapered by a Hann window over $\pm$`lowpass_filter_width` zero crossings. Applying them is one `conv1d_direct` with $\text{new}$ output channels and stride $\text{orig}$ (`M12.1`), after zero-padding the signal so each filter is centered on its output instant, and the phases interleave back into time. The output has $\lceil \text{new} \cdot T / \text{orig} \rceil$ samples.

## 3. Worked example by hand

Shrink the row $(10, 20, 30, 40)$ to 2 pixels with bilinear antialiasing. $\text{scale} = 2$, so $f_s = 2$ and the triangle's support grows from 1 to 2.

Output 0: $c_0 = 0.5 \cdot 2 = 1.0$; taps from $\operatorname{int}(1 - 2 + 0.5) \to 0$ to $\operatorname{int}(1 + 2 + 0.5) = 3$, so pixels 0, 1, 2 at kernel arguments $(j - 1 + 0.5)/2 = -0.25, 0.25, 0.75$. The triangle gives $0.75, 0.75, 0.25$, sum $1.75$, so the weights are $3/7, 3/7, 1/7$: the window was cut at the left border and renormalized. Output 1 mirrors it: pixels 1, 2, 3 with $1/7, 3/7, 3/7$.

In float, output 0 is $(3 \cdot 10 + 3 \cdot 20 + 30)/7 = 17.14$ and output 1 is $(20 + 90 + 120)/7 = 32.86$. Pillow's fixed point: $3/7 \cdot 2^{22} = 1797558.86 \to 1797559$ and $1/7 \cdot 2^{22} = 599186.29 \to 599186$; output 0 accumulates $2^{21} + 10 \cdot 1797559 + 20 \cdot 1797559 + 30 \cdot 599186 = 73999502$, and $73999502 \gg 22 = 17$. Output 1 gives 33. So the row becomes $(17, 33)$. Without antialiasing the support stays 1: output 0 has taps at $\pm 0.5$, weights $0.5, 0.5$, and simply averages pixels 0 and 1. This is `test_hand_example_bilinear_antialias`.

## 4. The interface

```python
def kernel_weights(in_size, out_size, kernel, antialias=True, cubic_a=-0.5) -> tuple[NDArray, NDArray]: ...
def resize(img, size, kernel, antialias=True, mode='pil', cubic_a=-0.5) -> NDArray: ...
def resize_planar_f32(x_chw, size, kernel, align_corners=False, antialias=False) -> NDArray: ...
def resample_audio(x, sr_in, sr_out, lowpass_filter_width=6, rolloff=0.99) -> NDArray: ...
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_bilinear_antialias` | unit, smoke | section 3's weights $3/7, 3/7, 1/7$ and Pillow's $(17, 33)$; the non-antialiased weights | the definitions |
| `test_pil_mode_is_pillow_bit_for_bit` | golden | six images, four sizes, four kernels: every pixel equal to Pillow 12.3 | `L13.6` preprocessing parity (stricter than its 1-LSB budget) |
| `test_float_mode_matches_torchvision` | golden | float bilinear and bicubic within $10^{-4}$ of torchvision's antialiased resize | float pipelines |
| `test_planar_matches_interpolate` | golden | seven `F.interpolate` cases incl. the ViT position table, `align_corners`, and antialias, within $10^{-5}$ | `L13.3`'s `interpolate_pos_embed` |
| `test_resample_audio_matches_torchaudio` | golden | 44.1k, 48k, 22.05k, 8k to 16k, a wide filter, stereo, within $10^{-5}$ | `L14.1`'s 16 kHz input |
| `test_weights_sum_to_one` | property | every row of every kernel sums to 1; a flat image stays flat | no brightness change at the border |
| `test_identity_at_equal_size` | property | same-size resize and same-rate resampling are exact copies | no accidental blur |
| `test_band_limited_round_trip` | property | 16k to 48k and back stays within $2 \cdot 10^{-3}$ | the filters keep the band |
| `test_antialias_prevents_aliasing` | property | a checkerboard shrunk by 3 aliases to black and white without antialias, stays gray with it | why the stretch exists |
| `test_rejects_bad_arguments` | boundary | unknown kernel or mode, empty output, float pixels in 'pil' mode, Lanczos in interpolate mode, bad rates | caller bugs fail early |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. never stretching the kernel when shrinking | aliasing; moire on fine textures | `test_antialias_prevents_aliasing`, `test_pil_mode_is_pillow_bit_for_bit` (mutant `s01`) |
| 2. dividing by the kernel's area instead of the kept taps | edges darken or brighten | `test_weights_sum_to_one`, `test_hand_example_bilinear_antialias` (mutant `s02`) |
| 3. vertical pass before horizontal | one-level differences on many pixels | `test_pil_mode_is_pillow_bit_for_bit` (mutant `s03`) |
| 4. PyTorch's $a = -0.75$ in Pillow mode | bicubic pixels slightly sharper than Pillow | `test_pil_mode_is_pillow_bit_for_bit` (mutant `s04`) |
| 5. sampling at pixel corners, $c_x = x \cdot \text{scale}$ | the image shifts by half a pixel | `test_hand_example_bilinear_antialias`, `test_float_mode_matches_torchvision` (mutant `s05`) |
| 6. truncating fixed-point weights instead of rounding | rare one-level differences | `test_pil_mode_is_pillow_bit_for_bit` (mutant `s06`) |
| 7. not centering the audio filters (padding only on the right) | resampled audio delayed by `width` samples | `test_resample_audio_matches_torchaudio`, `test_band_limited_round_trip` (mutant `s07`) |
| 8. Pillow's $a = -0.5$ in `F.interpolate` mode | position tables differ from HF | `test_planar_matches_interpolate` (mutant `s08`) |
| dropping the $+0.5$ in the tap range | the last tap of each window missing | `test_pil_mode_is_pillow_bit_for_bit` (mutant `m01`) |
| accepting an output size of 0 | empty weights instead of an error | `test_identity_at_equal_size` (mutant `m02`) |
| accepting float pixels in 'pil' mode | silent truncation | `test_rejects_bad_arguments` (mutant `m03`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M12.1` | `resample_audio` is one strided `conv1d_direct` over the zero-padded signal |
| Back | `M12.2` | images are uint8 HWC (reading) |
| Back | `M02.1` | the sinc and its series (reading) |
| Forward | `L13.6` | `resize` in every image processor, checked against Pillow (with B14's L13 group) |
| Forward | `L13.3` | `interpolate_pos_embed` resizes position tables with `resize_planar_f32` |
| Forward | `L14.1` | `resample_audio` brings every clip to 16 kHz |
| Forward | `data.10` | thumbnails for perceptual hashes |
| Forward | `S-M12` | sampling, aliasing, and interpolation kernels by hand |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `resize` (pil mode) | Pillow `ImagingResample` | SIMD fixed-point passes, `reducing_gap` (a fast box reduce before the filter) | `src/libImaging/Resample.c` |
| `resize` (float mode) | `torch._C._nn._upsample_bicubic2d_aa` | the same weights vectorized on CPU and CUDA | `aten/src/ATen/native/UpSample.h` |
| `resize` in Rust | `fast_image_resize` (the engine, `L15.1`) | SIMD convolution with Pillow-compatible kernels | `fast_image_resize` crate docs |
| `resample_audio` | `soxr`, `rubato` | polyphase filters with better stopband, streaming input | libsoxr; `rubato` crate (`L15.1`) |
