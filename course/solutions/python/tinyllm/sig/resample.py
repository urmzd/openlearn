"""tinyllm.sig.resample (M12.3): resampling and interpolation.

Resizing an image is sampling a continuous image you never had: interpolate
between the pixels with a kernel (nearest, bilinear, bicubic, Lanczos3),
then sample the result on the new grid. Shrinking needs one more step. The
new grid is coarser, so detail finer than its Nyquist limit would alias into
moire unless it is removed first; Pillow (and torchvision with
antialias=True) stretch the kernel by the scale factor, which low-pass
filters and interpolates in one pass. Both run separably, one axis at a
time, and Pillow rounds to uint8 between the horizontal and the vertical
pass. The processors in L13.6 must match Pillow within one unit of the last
place, so this module reproduces its fixed-point arithmetic exactly.

Audio has the same problem in one dimension: torchaudio's sinc resampler is
a bank of windowed-sinc low-pass filters applied as one strided convolution,
which is how `resample_audio` uses M12.1's `conv1d_direct`.

Contract: contracts/py/tinyllm/sig/resample.pyi.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from tinyllm.sig.conv import conv1d_direct

Kernel = Literal["nearest", "bilinear", "bicubic", "lanczos3"]
_SUPPORT = {"bilinear": 1.0, "bicubic": 2.0, "lanczos3": 3.0}
PRECISION_BITS = 22  # Pillow's 8-bit fixed point: 32 - 8 - 2


def _filter(kernel: str, x: NDArray, cubic_a: float) -> NDArray:
    """The interpolation kernel evaluated at offsets x (in input pixels)."""
    if kernel == "bilinear":
        ax = np.abs(x)
        return np.where(ax < 1.0, 1.0 - ax, 0.0)
    if kernel == "bicubic":
        a = cubic_a
        ax = np.abs(x)
        near = ((a + 2.0) * ax - (a + 3.0)) * ax * ax + 1.0
        far = (((ax - 5.0) * ax + 8.0) * ax - 4.0) * a
        return np.where(ax < 1.0, near, np.where(ax < 2.0, far, 0.0))
    if kernel == "lanczos3":
        # sinc(x) sinc(x / 3) on [-3, 3), half open as in Pillow.
        return np.where((x >= -3.0) & (x < 3.0), np.sinc(x) * np.sinc(x / 3.0), 0.0)
    raise ValueError(f"unknown kernel {kernel!r}")


def _nearest_index(in_size: int, out_size: int) -> NDArray:
    """Pillow's NEAREST source index: (int)(scale/2 + x*scale), accumulated."""
    scale = in_size / out_size
    idx = np.empty(out_size, dtype=np.int64)
    xo = scale * 0.5
    for x in range(out_size):
        # Pillow adds the scale once per pixel (ImagingScaleAffine), so the
        # rounding of this running sum, not of (x + 0.5) * scale, decides ties.
        idx[x] = min(int(xo), in_size - 1)
        xo += scale
    return idx


def kernel_weights(
    in_size: int,
    out_size: int,
    kernel: Kernel,
    antialias: bool = True,
    cubic_a: float = -0.5,
) -> tuple[NDArray, NDArray]:
    if in_size < 1 or out_size < 1:
        raise ValueError(
            f"kernel_weights: sizes must be >= 1, got {in_size} -> {out_size}"
        )
    if kernel == "nearest":
        return _nearest_index(in_size, out_size), np.ones((out_size, 1))
    if kernel not in _SUPPORT:
        raise ValueError(f"kernel_weights: unknown kernel {kernel!r}")
    scale = in_size / out_size
    # Antialias: when shrinking, stretch the kernel by the scale so it also
    # low-passes everything finer than the output grid can hold.
    fscale = max(scale, 1.0) if antialias else 1.0
    support = _SUPPORT[kernel] * fscale
    taps = int(math.ceil(support)) * 2 + 1
    start = np.empty(out_size, dtype=np.int64)
    weights = np.zeros((out_size, taps))
    for xx in range(out_size):
        center = (xx + 0.5) * scale
        lo = max(int(center - support + 0.5), 0)
        hi = min(int(center + support + 0.5), in_size)
        k = _filter(
            kernel, (np.arange(lo, hi) - center + 0.5) * (1.0 / fscale), cubic_a
        )
        total = k.sum()
        # Taps past the border are dropped and the rest renormalized, so a
        # flat image stays flat right up to the edge.
        weights[xx, : hi - lo] = k / total if total != 0.0 else k
        start[xx] = lo
    return start, weights


def _apply_axis(x: NDArray, start: NDArray, weights: NDArray, axis: int) -> NDArray:
    """sum_t x[start + t] * weights[:, t] along `axis` (taps past the end weigh 0)."""
    xm = np.moveaxis(x, axis, -1)
    n = xm.shape[-1]
    out = np.zeros(
        xm.shape[:-1] + (len(start),), dtype=np.result_type(xm.dtype, weights.dtype)
    )
    for t in range(weights.shape[1]):
        idx = np.minimum(start + t, n - 1)
        out += xm[..., idx] * weights[:, t]
    return np.moveaxis(out, -1, axis)


def _pil_pass(
    img: NDArray, out_size: int, axis: int, kernel: str, antialias: bool, cubic_a: float
) -> NDArray:
    """One separable Pillow pass on uint8 data, in 22-bit fixed point."""
    start, w = kernel_weights(img.shape[axis], out_size, kernel, antialias, cubic_a)
    if kernel == "nearest":
        return np.take(img, start, axis=axis)
    # Coefficients to fixed point, rounding half away from zero.
    kk = np.trunc(w * (1 << PRECISION_BITS) + np.where(w < 0, -0.5, 0.5)).astype(
        np.int64
    )
    acc = _apply_axis(img.astype(np.int64), start, kk, axis) + (
        1 << (PRECISION_BITS - 1)
    )
    return np.clip(acc >> PRECISION_BITS, 0, 255).astype(np.uint8)


def resize(
    img: NDArray,
    size: tuple[int, int],
    kernel: Kernel,
    antialias: bool = True,
    mode: Literal["pil", "float"] = "pil",
    cubic_a: float = -0.5,
) -> NDArray:
    a = np.asarray(img)
    if a.ndim not in (2, 3):
        raise ValueError(f"resize: need HW or HWC, got shape {a.shape}")
    out_h, out_w = size
    if out_h < 1 or out_w < 1:
        raise ValueError(f"resize: output size must be positive, got {size}")
    if kernel not in ("nearest", "bilinear", "bicubic", "lanczos3"):
        raise ValueError(f"resize: unknown kernel {kernel!r}")
    if mode == "pil":
        if a.dtype != np.uint8:
            raise ValueError(f"resize: mode 'pil' needs uint8 pixels, got {a.dtype}")
        y = a
        # Horizontal first, then vertical, rounding to uint8 in between; a pass
        # whose size does not change is skipped, as Pillow does.
        if out_w != a.shape[1]:
            y = _pil_pass(y, out_w, 1, kernel, antialias, cubic_a)
        if out_h != a.shape[0]:
            y = _pil_pass(y, out_h, 0, kernel, antialias, cubic_a)
        return y.copy() if y is a else y
    if mode != "float":
        raise ValueError(f"resize: mode is 'pil' or 'float', got {mode!r}")
    y = a.astype(np.float64)
    for axis, n_out in ((1, out_w), (0, out_h)):
        if n_out != y.shape[axis]:
            start, w = kernel_weights(y.shape[axis], n_out, kernel, antialias, cubic_a)
            y = _apply_axis(y, start, w, axis)
    return y


def _torch_axis(
    in_size: int, out_size: int, kernel: str, align_corners: bool
) -> tuple[NDArray, NDArray]:
    """F.interpolate (antialias=False) taps along one axis: start index, weights."""
    o = np.arange(out_size, dtype=np.float64)
    if kernel == "nearest":
        if align_corners:
            raise ValueError(
                "resize_planar_f32: align_corners does not apply to nearest"
            )
        scale = np.float32(in_size) / np.float32(out_size)
        idx = np.floor(o.astype(np.float32) * scale).astype(np.int64)
        return np.minimum(idx, in_size - 1), np.ones((out_size, 1))
    if align_corners:
        scale = (in_size - 1) / (out_size - 1) if out_size > 1 else 0.0
        src = o * scale
    else:
        src = (o + 0.5) * (in_size / out_size) - 0.5
    if kernel == "bilinear":
        src = np.maximum(src, 0.0)  # linear clamps the source; cubic does not
        i0 = np.floor(src).astype(np.int64)
        lam = src - i0
        return i0, np.stack([1.0 - lam, lam], axis=1)
    if kernel == "bicubic":
        i0 = np.floor(src).astype(np.int64)
        t = src - i0
        a = -0.75  # PyTorch's non-antialiased bicubic, not Pillow's -0.5
        w = np.stack(
            [
                _filter("bicubic", t + 1.0, a),
                _filter("bicubic", t, a),
                _filter("bicubic", 1.0 - t, a),
                _filter("bicubic", 2.0 - t, a),
            ],
            axis=1,
        )
        return i0 - 1, w
    raise ValueError(f"resize_planar_f32: {kernel!r} needs antialias=True")


def _apply_axis_clamped(
    x: NDArray, start: NDArray, weights: NDArray, axis: int
) -> NDArray:
    """Like _apply_axis, but taps outside [0, n) read the edge sample."""
    xm = np.moveaxis(x, axis, -1)
    n = xm.shape[-1]
    out = np.zeros(xm.shape[:-1] + (len(start),))
    for t in range(weights.shape[1]):
        out += xm[..., np.clip(start + t, 0, n - 1)] * weights[:, t]
    return np.moveaxis(out, -1, axis)


def resize_planar_f32(
    x_chw: NDArray,
    size: tuple[int, int],
    kernel: Kernel,
    align_corners: bool = False,
    antialias: bool = False,
) -> NDArray:
    x = np.asarray(x_chw)
    if x.ndim != 3:
        raise ValueError(f"resize_planar_f32: need CHW, got shape {x.shape}")
    out_h, out_w = size
    if out_h < 1 or out_w < 1:
        raise ValueError(f"resize_planar_f32: output size must be positive, got {size}")
    y = x.astype(np.float64)
    if antialias:
        if align_corners:
            raise ValueError("resize_planar_f32: antialias does not take align_corners")
        for axis, n_out in ((2, out_w), (1, out_h)):
            if n_out != y.shape[axis]:
                start, w = kernel_weights(y.shape[axis], n_out, kernel, True)
                y = _apply_axis(y, start, w, axis)
        return y.astype(np.float32)
    for axis, n_out in ((2, out_w), (1, out_h)):
        start, w = _torch_axis(y.shape[axis], n_out, kernel, align_corners)
        y = _apply_axis_clamped(y, start, w, axis)
    return y.astype(np.float32)


def _sinc_kernel(
    orig: int, new: int, width_taps: int, rolloff: float
) -> tuple[NDArray, int]:
    """torchaudio's sinc_interp_hann polyphase bank: [new, taps], and the pad width."""
    base = min(orig, new) * rolloff  # the cutoff, in units of the lower rate
    width = math.ceil(width_taps * orig / base)
    idx = np.arange(-width, width + orig, dtype=np.float64)[None, :] / orig
    t = (np.arange(0, -new, -1, dtype=np.float64)[:, None] / new + idx) * base
    t = np.clip(t, -width_taps, width_taps)
    window = np.cos(t * math.pi / width_taps / 2) ** 2  # Hann over the clipped support
    t = t * math.pi
    safe = np.where(t == 0.0, 1.0, t)
    k = np.where(t == 0.0, 1.0, np.sin(safe) / safe)
    return k * window * (base / orig), width


def resample_audio(
    x: NDArray,
    sr_in: int,
    sr_out: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
) -> NDArray:
    a = np.asarray(x, dtype=np.float64)
    if sr_in < 1 or sr_out < 1 or lowpass_filter_width < 1 or not 0.0 < rolloff <= 1.0:
        raise ValueError(
            f"resample_audio: bad rates or filter ({sr_in}, {sr_out}, {lowpass_filter_width}, {rolloff})"
        )
    if a.ndim not in (1, 2):
        raise ValueError(
            f"resample_audio: need [T] or [channels, T], got shape {a.shape}"
        )
    if sr_in == sr_out:
        return a.copy()
    g = math.gcd(sr_in, sr_out)
    orig, new = sr_in // g, sr_out // g
    kern, width = _sinc_kernel(orig, new, lowpass_filter_width, rolloff)
    wav = a.reshape(-1, a.shape[-1])
    length = wav.shape[-1]
    padded = np.pad(wav, ((0, 0), (width, width + orig)))
    # One strided conv1d: phase j of the output block is kernel row j.
    # The result is [B, new, blocks]; interleave the phases back into time.
    y = conv1d_direct(padded[:, None, :], kern[:, None, :], None, stride=orig)
    y = y.transpose(0, 2, 1).reshape(wav.shape[0], -1)
    target = math.ceil(new * length / orig)
    y = y[:, :target]
    return y[0] if a.ndim == 1 else y
