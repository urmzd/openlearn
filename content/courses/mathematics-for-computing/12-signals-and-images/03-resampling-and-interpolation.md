# Resampling and interpolation

Every image a vision model sees is resized first: CLIP wants 224 by 224, SigLIP 384, and Idefics3 splits large images into 512-pixel tiles. The checkpoints were trained on images resized by Pillow, so a reimplementation must reproduce Pillow's output, and a preprocessing parity rule of at most one unit in the last place leaves no room for a different algorithm. A naive bilinear resize that samples at pixel corners, or shrinks without antialiasing, produces images that look fine and shift model outputs measurably. Audio has the same problem in time. Whisper hears 16 kHz, recordings arrive at 44.1 or 48 kHz, and the front end must bring them down without folding high frequencies into the speech band.

This section builds resampling from interpolation kernels, through per-pixel footprint weights and antialiasing, to Pillow's fixed-point passes, PyTorch's `F.interpolate` convention, and the polyphase sinc resampler that torchaudio uses for audio. You will work one shrink by hand, number by number, and then see why each rounding and offset in the solution is there. It uses `conv1d_direct` from [Convolution as a linear map](01-convolution-as-a-linear-map.md) and the uint8 HWC images of [Image tensors and layouts](02-image-tensors-and-layouts.md). The sinc and its series appear in [Taylor series, remainder bounds, and range reduction](../02-calculus-2/01-taylor-series.md), and aliasing, used here as a concept, is treated properly in [Sampling, Nyquist, windows, and the STFT](05-sampling-windows-and-stft.md).

The reference implementation is [`python/tinyllm/sig/resample.py`](../../../../course/solutions/python/tinyllm/sig/resample.py), with its interface in [`course/contracts/py/tinyllm/sig/resample.pyi`](../../../../course/contracts/py/tinyllm/sig/resample.pyi). It provides `kernel_weights` (nearest, bilinear, bicubic, Lanczos3, with antialias), `resize` (Pillow's fixed-point passes bit for bit, or a float mode), `resize_planar_f32` (`F.interpolate` semantics for position tables), and `resample_audio` (torchaudio's windowed-sinc resampler).

> **Learning objectives**
> - Write the nearest, bilinear, Keys bicubic, and Lanczos3 kernels and compute the normalized weights of any output pixel.
> - Explain why shrinking aliases and how stretching the kernel by $\max(\text{scale}, 1)$ prevents it.
> - Reproduce Pillow's resize bit for bit: pass order, 22-bit fixed-point weights, and rounding between passes.
> - Distinguish `F.interpolate`'s half-pixel, edge-clamped convention from Pillow's.
> - Build a polyphase windowed-sinc audio resampler as one strided convolution.

## Notation

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

## Interpolation kernels

Resampling is interpolation followed by sampling: a kernel turns the samples back into a continuous signal, and the new grid samples it. Interpolation evaluates a continuous image between the pixels as $\hat{x}(u) = \sum_j x[j]\, k(u - j)$. Nearest picks one pixel. Bilinear uses the triangle $k(t) = \max(0, 1 - |t|)$ (support 1). Bicubic uses Keys' cubic, support 2:
$$k(t) = \begin{cases} (a+2)|t|^3 - (a+3)|t|^2 + 1 & |t| < 1 \\ a|t|^3 - 5a|t|^2 + 8a|t| - 4a & 1 \le |t| < 2 \\ 0 & \text{otherwise}\end{cases}$$
Pillow uses $a = -0.5$ (the value that makes the cubic reproduce quadratics exactly); PyTorch's non-antialiased bicubic uses $a = -0.75$. Lanczos3 is a windowed sinc, $\operatorname{sinc}(t)\operatorname{sinc}(t/3)$ on $[-3, 3)$, the closest of the four to the ideal low-pass filter. All four are 1 at 0 and 0 at the other integers, so resizing to the same size changes nothing.

## Weights per output pixel

A kernel defines the continuous image; the next step is to decide where each output pixel samples it and which inputs it touches. Output pixel $x$ sits at the center of its footprint, $c_x = (x + 0.5)\,\text{scale}$ in input coordinates, where input pixel $j$ is centered at $j + 0.5$. Its taps are the inputs within the kernel's support of that center, from $\text{lo} = \max(\operatorname{int}(c_x - a_k f_s + 0.5), 0)$ to $\text{hi} = \min(\operatorname{int}(c_x + a_k f_s + 0.5), n_\text{in})$, with weights
$$w_{x, j} = \frac{k\big((j - c_x + 0.5)/f_s\big)}{\sum_{j'=\text{lo}}^{\text{hi}-1} k\big((j' - c_x + 0.5)/f_s\big)} .$$
Every output is a weighted sum of a few inputs, with weights that sum to 1. Dividing by the sum of the kept taps, not by the kernel's total area, renormalizes at the border where taps fall off the image: a flat image stays flat up to its edge.

## Antialiasing is a stretched kernel

Shrinking by a factor `scale` lowers the Nyquist frequency of the new grid by that factor. Detail between the two limits does not vanish: it aliases into false low-frequency patterns (moire). The fix is to low-pass filter first, and stretching the kernel by $f_s = \max(\text{scale}, 1)$ does exactly that in the same pass: a kernel $f_s$ times wider has a frequency response $f_s$ times narrower. Pillow always does it; torchvision does it with `antialias=True`. Enlarging needs no filter, so $f_s = 1$. The float mode of `resize` is these weights without rounding, which is torchvision's antialiased float resize.

## Separable passes and Pillow's arithmetic

All four kernels are separable, so a 2-D resize is a horizontal pass on every row and then a vertical pass on every column, and a pass whose size does not change is skipped. Pillow does integer arithmetic for 8-bit images: each weight becomes $\operatorname{round}(w \cdot 2^{22})$ (halves away from zero), each output accumulates $2^{21} + \sum_j x[j]\, w^\text{fixed}_j$, and the result is $\operatorname{clip}(\text{acc} \gg 22, 0, 255)$. The intermediate image after the horizontal pass is uint8 again. Pass order, the rounding of the weights, and the rounding between passes each change a few pixels by one level, and together they are what "matches Pillow" means.

Nearest is special in Pillow: it copies input $\operatorname{int}(s_x)$ where $s_0 = \text{scale}/2$ and $s_{x+1} = s_x + \text{scale}$ is accumulated in float64.

## `F.interpolate`, a different convention

Not every resize in a model pipeline goes through Pillow. PyTorch's non-antialiased interpolation maps output $o$ to the source coordinate $(o + 0.5)\,\text{in}/\text{out} - 0.5$ (or $o\,(\text{in} - 1)/(\text{out} - 1)$ with `align_corners=True`), takes the 2 (bilinear) or 4 (bicubic, $a = -0.75$) neighbours of that coordinate, and reads the edge pixel for neighbours outside the image instead of renormalizing. It never stretches the kernel. HF's `interpolate_pos_embed` resizes a ViT's position table this way, so `resize_planar_f32` implements it; with `antialias=True` it uses the Pillow weights in float, as PyTorch's antialiased path does.

## Band-limited audio resampling

Audio resampling is the same idea in one dimension, with the ideal kernel. The Whittaker-Shannon formula reconstructs a band-limited signal exactly from its samples with sinc kernels. Resampling from rate $\text{sr}_\text{in}$ to $\text{sr}_\text{out}$ evaluates that reconstruction, low-passed below the smaller Nyquist rate, at the new instants.

With $g = \gcd$, $\text{orig} = \text{sr}_\text{in}/g$ and $\text{new} = \text{sr}_\text{out}/g$, the output pattern repeats every $\text{orig}$ input samples. So torchaudio precomputes $\text{new}$ filters (one per output phase), each a sinc at cutoff $\min(\text{orig}, \text{new}) \cdot \text{rolloff}$ tapered by a Hann window over $\pm$`lowpass_filter_width` zero crossings. Applying them is one `conv1d_direct` with $\text{new}$ output channels and stride $\text{orig}$, after zero-padding the signal so each filter is centered on its output instant, and the phases interleave back into time. The output has $\lceil \text{new} \cdot T / \text{orig} \rceil$ samples.

## Working it by hand

Shrink the row $(10, 20, 30, 40)$ to 2 pixels with bilinear antialiasing. $\text{scale} = 2$, so $f_s = 2$ and the triangle's support grows from 1 to 2.

For output 0, $c_0 = 0.5 \cdot 2 = 1.0$. The taps run from $\operatorname{int}(1 - 2 + 0.5) \to 0$ to $\operatorname{int}(1 + 2 + 0.5) = 3$, so pixels 0, 1, 2 at kernel arguments $(j - 1 + 0.5)/2 = -0.25, 0.25, 0.75$. The triangle gives $0.75, 0.75, 0.25$, sum $1.75$, so the weights are $3/7, 3/7, 1/7$: the window was cut at the left border and renormalized. Output 1 mirrors it: pixels 1, 2, 3 with $1/7, 3/7, 3/7$.

In float, output 0 is $(3 \cdot 10 + 3 \cdot 20 + 30)/7 = 17.14$ and output 1 is $(20 + 90 + 120)/7 = 32.86$. Pillow's fixed point rounds the weights first: $3/7 \cdot 2^{22} = 1797558.86 \to 1797559$ and $1/7 \cdot 2^{22} = 599186.29 \to 599186$. Output 0 accumulates $2^{21} + 10 \cdot 1797559 + 20 \cdot 1797559 + 30 \cdot 599186 = 73999502$, and $73999502 \gg 22 = 17$. Output 1 gives 33. So the row becomes $(17, 33)$.

Without antialiasing the support stays 1: output 0 has taps at $\pm 0.5$, weights $0.5, 0.5$, and simply averages pixels 0 and 1.

## Designing the interface

```python
def kernel_weights(in_size, out_size, kernel, antialias=True, cubic_a=-0.5) -> tuple[NDArray, NDArray]: ...
def resize(img, size, kernel, antialias=True, mode='pil', cubic_a=-0.5) -> NDArray: ...
def resize_planar_f32(x_chw, size, kernel, align_corners=False, antialias=False) -> NDArray: ...
def resample_audio(x, sr_in, sr_out, lowpass_filter_width=6, rolloff=0.99) -> NDArray: ...
```

`kernel_weights` is exposed on its own because it is the whole definition of a 1-D resampler; the other three functions apply it (or a variant) along axes. `resize` defaults to Pillow mode with Pillow's $a = -0.5$, and Pillow mode accepts only uint8. `resize_planar_f32` follows `F.interpolate`, so it does not offer Lanczos. `resample_audio` accepts mono `[T]` or multichannel `[channels, T]` audio. Unknown kernels or modes, empty outputs, float pixels in Pillow mode, Lanczos in interpolate mode, and bad rates are all rejected.

## Building it step by step

The solution follows from six observations.

1. **Every 1-D resampler is a sparse matrix, stored by rows.** Each output is a weighted sum of a short run of consecutive inputs. So one axis of a resize is fully described by, for each output, a start index and a short vector of weights. `kernel_weights` returns exactly that pair, `(start [out], weights [out, taps])`, and one helper applies it along any axis. Everything else is choosing the weights.
2. **The weights come from the footprint geometry.** Output $x$ is centered at $(x + 0.5)\,\text{scale}$; input $j$ is centered at $j + 0.5$, so the kernel argument is $(j + 0.5 - c_x)/f_s$. The taps are the inputs within $a_k f_s$ of the center, clipped to the image, and the weights are divided by their own sum so they total 1 even where the window was clipped.
3. **Antialiasing is one number, $f_s$.** Stretching the kernel by $f_s = \max(\text{scale}, 1)$ widens the support and the argument scaling together. With `antialias=False`, $f_s = 1$ and the same code gives the plain interpolating kernel.
4. **Pillow equality means reproducing Pillow's arithmetic, not its math.** The weights are the same real numbers, but Pillow converts them to 22-bit fixed point with round-half-away-from-zero, accumulates integers starting from half a unit ($2^{21}$), shifts, clips, and stores uint8 after each pass, horizontal first. Each of those is a separate source of one-level differences, so the solution copies all of them.
5. **`F.interpolate` is a different family of weights, not a variant of these.** It maps to source coordinate $(o + 0.5)\,\text{in}/\text{out} - 0.5$, uses a fixed 2 or 4 taps from the floor of that coordinate, and clamps indices to the edge rather than renormalizing. That needs its own tap function and an apply that clamps, but the same "start plus weights" representation.
6. **Audio resampling is a polyphase filter bank, which is a strided conv.** With $\text{orig}$ input and $\text{new}$ output samples per period, output phase $j$ always uses the same filter, shifted by $\text{orig}$ inputs per period. So $\text{new}$ filters stacked as output channels and applied with stride $\text{orig}$ through `conv1d_direct` compute every output; interleaving the channels restores time order.

In the listings below, argument validation is elided as `...`; the bodies are as in the solution.

First, the kernels, evaluated at offsets measured in input pixels:

```python
def _filter(kernel: str, x: NDArray, cubic_a: float) -> NDArray:
    if kernel == "bilinear":
        ax = np.abs(x)
        return np.where(ax < 1.0, 1.0 - ax, 0.0)              # the triangle
    if kernel == "bicubic":
        a = cubic_a
        ax = np.abs(x)
        near = ((a + 2.0) * ax - (a + 3.0)) * ax * ax + 1.0     # Keys, |x| < 1 (Horner form)
        far = (((ax - 5.0) * ax + 8.0) * ax - 4.0) * a          # Keys, 1 <= |x| < 2
        return np.where(ax < 1.0, near, np.where(ax < 2.0, far, 0.0))
    if kernel == "lanczos3":
        # sinc(x) sinc(x / 3) on [-3, 3), half open as in Pillow.
        return np.where((x >= -3.0) & (x < 3.0), np.sinc(x) * np.sinc(x / 3.0), 0.0)
    raise ValueError(f"unknown kernel {kernel!r}")
```

`np.sinc` is the normalized $\sin(\pi x)/(\pi x)$, so Lanczos3 is 1 at 0 and 0 at every other integer, as interpolation requires.

Next, the weight table, one row per output (steps 2 and 3):

```python
def kernel_weights(in_size, out_size, kernel, antialias=True, cubic_a=-0.5):
    # ... sizes >= 1; "nearest" returns Pillow's accumulated index with weight 1 ...
    scale = in_size / out_size
    # Antialias: when shrinking, stretch the kernel by the scale so it also
    # low-passes everything finer than the output grid can hold.
    fscale = max(scale, 1.0) if antialias else 1.0
    support = _SUPPORT[kernel] * fscale                 # 1, 2 or 3 input pixels, stretched
    taps = int(math.ceil(support)) * 2 + 1              # enough columns for any window
    start = np.empty(out_size, dtype=np.int64)
    weights = np.zeros((out_size, taps))
    for xx in range(out_size):
        center = (xx + 0.5) * scale                     # pixel centers, not corners
        lo = max(int(center - support + 0.5), 0)        # +0.5: Pillow's rounding of the range
        hi = min(int(center + support + 0.5), in_size)
        # Argument: distance from input center (j + 0.5) to output center, unstretched.
        k = _filter(
            kernel, (np.arange(lo, hi) - center + 0.5) * (1.0 / fscale), cubic_a
        )
        total = k.sum()
        # Taps past the border are dropped and the rest renormalized, so a
        # flat image stays flat right up to the edge.
        weights[xx, : hi - lo] = k / total if total != 0.0 else k
        start[xx] = lo
    return start, weights
```

In the hand example, output 0 has `center = 1.0`, `support = 2`, `lo = int(-0.5) = 0` (truncation toward zero, then the `max`), `hi = int(3.5) = 3`, and kernel arguments $(-0.5, 0.5, 1.5) \cdot \tfrac12$: weights $3/7, 3/7, 1/7$.

Applying a table along one axis takes one gather and multiply-add per tap column:

```python
def _apply_axis(x, start, weights, axis):
    """sum_t x[start + t] * weights[:, t] along `axis` (taps past the end weigh 0)."""
    xm = np.moveaxis(x, axis, -1)
    n = xm.shape[-1]
    out = np.zeros(
        xm.shape[:-1] + (len(start),), dtype=np.result_type(xm.dtype, weights.dtype)
    )
    for t in range(weights.shape[1]):
        # Unused tap columns have weight 0, so clamping their index is harmless.
        idx = np.minimum(start + t, n - 1)
        out += xm[..., idx] * weights[:, t]
    return np.moveaxis(out, -1, axis)
```

The result dtype follows the inputs, so the same function runs in int64 for Pillow's fixed point and in float64 for the float mode.

One Pillow pass, in fixed point (step 4):

```python
def _pil_pass(img, out_size, axis, kernel, antialias, cubic_a):
    start, w = kernel_weights(img.shape[axis], out_size, kernel, antialias, cubic_a)
    if kernel == "nearest":
        return np.take(img, start, axis=axis)
    # Coefficients to fixed point, rounding half away from zero.
    kk = np.trunc(w * (1 << PRECISION_BITS) + np.where(w < 0, -0.5, 0.5)).astype(
        np.int64
    )
    # Integer accumulation, starting at half a unit so the shift rounds.
    acc = _apply_axis(img.astype(np.int64), start, kk, axis) + (
        1 << (PRECISION_BITS - 1)
    )
    # Bicubic and Lanczos have negative lobes, so the sum can leave [0, 255].
    return np.clip(acc >> PRECISION_BITS, 0, 255).astype(np.uint8)
```

`PRECISION_BITS = 22` is Pillow's choice for 8-bit images: $32 - 8 - 2$, that is 8 bits for the pixel value and 2 bits of headroom for overshoot from negative lobes, inside a 32-bit accumulator. Negative weights round away from zero too, which is why there is a `np.where` and not just `+ 0.5`.

The two passes run in Pillow's order:

```python
    if mode == "pil":
        # ... uint8 required ...
        y = a
        # Horizontal first, then vertical, rounding to uint8 in between; a pass
        # whose size does not change is skipped, as Pillow does.
        if out_w != a.shape[1]:
            y = _pil_pass(y, out_w, 1, kernel, antialias, cubic_a)
        if out_h != a.shape[0]:
            y = _pil_pass(y, out_h, 0, kernel, antialias, cubic_a)
        return y.copy() if y is a else y         # same size: still a fresh array
```

The float mode runs the same two passes on float64 with the unrounded weights, which is torchvision's antialiased resize.

For `F.interpolate` (step 5), the bilinear taps are the floor of the source coordinate and the next pixel:

```python
    else:
        src = (o + 0.5) * (in_size / out_size) - 0.5   # half-pixel convention
    if kernel == "bilinear":
        src = np.maximum(src, 0.0)  # linear clamps the source; cubic does not
        i0 = np.floor(src).astype(np.int64)
        lam = src - i0
        return i0, np.stack([1.0 - lam, lam], axis=1)
```

Bicubic takes the four neighbours `i0 - 1 .. i0 + 2` with $a = -0.75$, and `_apply_axis_clamped` reads `np.clip(start + t, 0, n - 1)`, so a tap off the edge reads the edge pixel (no renormalization). With `antialias=True`, `resize_planar_f32` instead uses `kernel_weights` in float, as PyTorch's antialiased path does.

Finally, audio (step 6): the polyphase sinc bank, then one strided conv.

```python
def _sinc_kernel(orig, new, width_taps, rolloff):
    base = min(orig, new) * rolloff  # the cutoff, in units of the lower rate
    width = math.ceil(width_taps * orig / base)          # filter half-length, in input samples
    idx = np.arange(-width, width + orig, dtype=np.float64)[None, :] / orig
    # Row j: input offsets relative to output phase j's instant, j/new periods in.
    t = (np.arange(0, -new, -1, dtype=np.float64)[:, None] / new + idx) * base
    t = np.clip(t, -width_taps, width_taps)              # +-width_taps zero crossings
    window = np.cos(t * math.pi / width_taps / 2) ** 2  # Hann over the clipped support
    t = t * math.pi
    safe = np.where(t == 0.0, 1.0, t)
    k = np.where(t == 0.0, 1.0, np.sin(safe) / safe)    # sinc, with sinc(0) = 1
    return k * window * (base / orig), width             # unit DC gain at the lower rate


def resample_audio(x, sr_in, sr_out, lowpass_filter_width=6, rolloff=0.99):
    # ... validate rates, filter, and shape ...
    if sr_in == sr_out:
        return a.copy()
    g = math.gcd(sr_in, sr_out)
    orig, new = sr_in // g, sr_out // g                  # 44100 -> 16000: 441 -> 160
    kern, width = _sinc_kernel(orig, new, lowpass_filter_width, rolloff)
    wav = a.reshape(-1, a.shape[-1])                     # mono or [channels, T]
    length = wav.shape[-1]
    # Pad both sides so every filter is centered on its output instant.
    padded = np.pad(wav, ((0, 0), (width, width + orig)))
    # One strided conv1d: phase j of the output block is kernel row j.
    # The result is [B, new, blocks]; interleave the phases back into time.
    y = conv1d_direct(padded[:, None, :], kern[:, None, :], None, stride=orig)
    y = y.transpose(0, 2, 1).reshape(wav.shape[0], -1)
    target = math.ceil(new * length / orig)
    y = y[:, :target]                                    # drop the tail the padding added
    return y[0] if a.ndim == 1 else y
```

The `transpose(0, 2, 1)` turns `[B, phase, block]` into `[B, block, phase]`, so after the reshape the samples are in time order: block 0 phase 0, block 0 phase 1, and so on.

The full solution is [`resample.py`](../../../../course/solutions/python/tinyllm/sig/resample.py).

## Why this design

A separable resize costs $O(\text{taps})$ per output pixel per pass, about $2 a_k f_s$: cheap when enlarging, and growing with the shrink factor when antialiasing, which is the price of not aliasing. The audio resampler costs $O(\text{taps})$ per output sample with taps $\approx 2 \cdot \text{width} + \text{orig}$, and a period of $\text{orig}$ inputs is reused for every block.

Separable passes matter because a 2-D kernel window would cost $\text{taps}^2$ per pixel instead of $2 \cdot \text{taps}$. Every kernel here is a product of 1-D kernels, so separability is exact in real arithmetic; it is the rounding between passes that makes the order matter, which is why Pillow's order is copied.

The solution reproduces Pillow's integer path instead of matching it in float because the checkpoints were trained on Pillow's uint8 output. A float resize rounded once at the end differs from Pillow by one level on some pixels, enough to fail an exact comparison and to shift model outputs slightly. Fixed point with the same rounding is the only way to be bit-exact.

For audio, a polyphase conv beats an FFT resampler. FFT resampling assumes a periodic signal and smears the end into the start; the windowed-sinc bank is local, streams naturally, and is torchaudio's default, so Whisper's training audio was resampled this way.

The tradeoffs accepted are Python loops over outputs in `kernel_weights` and over taps in `_apply_axis`. The weight table is computed once per size pair, and the per-tap loop is vectorized over every row, so neither dominates at image sizes.

## What can go wrong

A correct implementation must first reproduce the hand example: weights $3/7, 3/7, 1/7$, Pillow's $(17, 33)$, and the non-antialiased weights. Pillow mode must equal Pillow 12.3 on every pixel across six images, four sizes, and four kernels, which is stricter than the one-unit preprocessing budget. Float mode must be within $10^{-4}$ of torchvision's antialiased bilinear and bicubic resize. Planar mode must be within $10^{-5}$ of seven `F.interpolate` cases, including a ViT position table, `align_corners`, and antialias. Audio must be within $10^{-5}$ of torchaudio for 44.1k, 48k, 22.05k, and 8k to 16k, with a wide filter, and in stereo.

Several properties must hold too. Every row of every kernel's weights sums to 1, so a flat image stays flat with no brightness change at the border. A same-size resize and a same-rate resampling are exact copies, with no accidental blur. A band-limited round trip from 16k to 48k and back stays within $2 \cdot 10^{-3}$, showing the filters keep the band. And a checkerboard shrunk by 3 aliases to black and white without antialiasing but stays gray with it, which is why the stretch exists. Bad arguments are rejected so caller bugs fail early.

The common mistakes and their symptoms:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| never stretching the kernel when shrinking | aliasing; moire on fine textures | `fscale = max(scale, 1.0)` widens the support and scales the argument (step 3) |
| dividing by the kernel's area instead of the kept taps | edges darken or brighten | divides by `k.sum()` over the taps inside the image |
| vertical pass before horizontal | one-level differences on many pixels | `resize` runs the width pass, then the height pass |
| PyTorch's $a = -0.75$ in Pillow mode | bicubic pixels slightly sharper than Pillow | `cubic_a` defaults to $-0.5$ for `resize` |
| sampling at pixel corners, $c_x = x \cdot \text{scale}$ | the image shifts by half a pixel | `center = (xx + 0.5) * scale` and the `+ 0.5` in the kernel argument |
| truncating fixed-point weights instead of rounding | rare one-level differences | `np.trunc(w * 2**22 ± 0.5)`, half away from zero |
| not centering the audio filters (padding only on the right) | resampled audio delayed by `width` samples | pads `width` on the left and `width + orig` on the right |
| Pillow's $a = -0.5$ in `F.interpolate` mode | position tables differ from HF | `_torch_axis` hard-codes $a = -0.75$ |
| dropping the $+0.5$ in the tap range | the last tap of each window missing | `int(center ± support + 0.5)` for `lo` and `hi` |
| accepting an output size of 0 | empty weights instead of an error | `kernel_weights` and `resize` reject sizes below 1 |
| accepting float pixels in 'pil' mode | silent truncation | Pillow mode requires `uint8` |

> **Warning:** Every one of these produces a plausible-looking image or waveform. Only a pixel-for-pixel comparison against the reference library reveals them.

## Where this leads

In *Build Your Own LLM System*, [Image preprocessing to the specification](../../build-an-llm/p13-vision/06-image-preprocessing.md) calls `resize` in every image processor and checks it against Pillow; the [Vision transformer encoder](../../build-an-llm/p13-vision/03-vision-transformer.md) resizes position tables in `interpolate_pos_embed` with `resize_planar_f32`; [Dynamic resolution and tiling](../../build-an-llm/p13-vision/04-dynamic-resolution-and-tiling.md) uses `resize` to fit the base image and each tile grid; and the [audio frontend](../../build-an-llm/p14-speech/01-audio-frontend.md) brings every clip to 16 kHz with `resample_audio`. The image deduplication stage of the [corpus pipeline](../../data-engineering/05-corpus-pipeline/README.md) in *Data Engineering* makes its hash thumbnails with `resize`. The chapter's [problem set](90-problem-set.md) works sampling, aliasing, and interpolation kernels by hand.

Each piece has a production counterpart:

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `resize` (pil mode) | Pillow `ImagingResample` | SIMD fixed-point passes, `reducing_gap` (a fast box reduce before the filter) | `src/libImaging/Resample.c` |
| `resize` (float mode) | `torch._C._nn._upsample_bicubic2d_aa` | the same weights vectorized on CPU and CUDA | `aten/src/ATen/native/UpSample.h` |
| `resize` in Rust | `fast_image_resize` (used by the Rust serving engine) | SIMD convolution with Pillow-compatible kernels | `fast_image_resize` crate docs |
| `resample_audio` | `soxr`, `rubato` | polyphase filters with better stopband, streaming input | libsoxr; `rubato` crate |

### Further reading

- Keys, "Cubic convolution interpolation for digital image processing" (1981).
- Smith, *Digital Audio Resampling Home Page* (free; the bandlimited interpolation derivation).
- Pillow, `src/libImaging/Resample.c`.

## Key terms

- **Interpolation kernel**: a function $k(t)$, 1 at 0 and 0 at other integers, that turns samples into a continuous signal via $\sum_j x[j]\,k(u - j)$.
- **Keys bicubic**: the piecewise cubic kernel with parameter $a$; Pillow uses $a = -0.5$, PyTorch $a = -0.75$.
- **Lanczos3**: the windowed sinc $\operatorname{sinc}(t)\operatorname{sinc}(t/3)$ on $[-3, 3)$.
- **Aliasing**: detail finer than a grid can hold reappearing as false low-frequency patterns after sampling.
- **Antialiasing**: low-pass filtering before sampling; for resizing, stretching the kernel by $\max(\text{scale}, 1)$.
- **Polyphase filter bank**: one filter per output phase, applied with a stride equal to the input period.

## Summary

- Resampling is interpolation (a kernel turns samples back into a continuous signal) followed by sampling on the new grid; every output is a weighted sum of a few inputs, with weights that sum to 1.
- Shrinking without first removing detail the new grid cannot hold aliases; antialiasing stretches the kernel by the scale factor, which low-pass filters in the same pass.
- Pillow runs separable passes, horizontal then vertical, in 22-bit fixed point and rounds to uint8 between them; matching it bit for bit means copying all three choices.
- `F.interpolate` is a different convention (sample at $(o + 0.5)\,\text{in}/\text{out} - 0.5$, clamp at the edge, bicubic $a = -0.75$), and it is the one HF position tables are resized with.
- Audio resampling is the same idea in 1-D: a bank of windowed-sinc low-pass filters applied as one strided conv.

## Review questions

1. Why is the kernel argument $(j - c_x + 0.5)/f_s$ and not $(j - c_x)/f_s$?
   <details><summary>Answer</summary>
   Input pixel $j$ covers $[j, j + 1)$ and is centered at $j + 0.5$, while $c_x$ is already a center in the same continuous coordinates. The distance between the two centers is $j + 0.5 - c_x$. Dropping the $0.5$ measures from pixel corners and shifts the image by half a pixel.
   </details>
2. What does dividing by `total`, the sum of the kept taps, buy at the border?
   <details><summary>Answer</summary>
   Near the border part of the window falls outside the image and its taps are dropped. Without renormalizing, the remaining weights sum to less than 1 and edge pixels get darker (or brighter, for kernels with negative lobes). Dividing by the kept sum makes every row sum to 1, so a flat image stays flat up to the edge.
   </details>
3. Why does `_pil_pass` clip to $[0, 255]$, when a weighted average of pixels should stay in range?
   <details><summary>Answer</summary>
   Bicubic and Lanczos weights are partly negative, so the output is not a convex combination: next to a sharp edge it overshoots (ringing) below 0 or above 255. Pillow clips, and so must any bit-exact reproduction.
   </details>
4. Why does the audio path need $\text{new}$ different filters instead of one?
   <details><summary>Answer</summary>
   Within one period of $\text{orig}$ input samples there are $\text{new}$ output instants, each at a different fractional offset from the input grid. A sinc centered at a different offset is a different set of tap values. After one period the pattern repeats, so $\text{new}$ filters cover every output, applied with stride $\text{orig}$.
   </details>
5. What would padding the signal only on the right do to the resampled audio?
   <details><summary>Answer</summary>
   Each filter is centered `width` samples into its window. Without `width` zeros on the left, output $k$'s window would start at its own instant, so the filter's center would land `width` input samples later: the output is delayed by `width` samples, and a round trip no longer returns the original.
   </details>
