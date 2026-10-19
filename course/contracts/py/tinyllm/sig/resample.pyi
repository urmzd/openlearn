# contracts/py/tinyllm/sig/resample.pyi (M12.3): resampling and interpolation
# chapter: content/courses/mathematics-for-computing/13-signals-and-images/03-resampling-and-interpolation.md
#
# Image resizing follows Pillow's Resample.c: separable passes, horizontal
# then vertical, a pass skipped when its size does not change. For output
# pixel x of a pass from n_in to n_out samples, scale = n_in / n_out,
# center = (x + 0.5) scale, and with antialias the kernel is stretched by
# fs = max(scale, 1) (so support = kernel_support * fs; without antialias
# fs = 1). Taps run over input indices j in [lo, hi), lo = max(int(center -
# support + 0.5), 0), hi = min(int(center + support + 0.5), n_in), with
# weight k((j - center + 0.5) * (1 / fs)), renormalized to sum to 1 (int()
# truncates toward zero, as in C). Kernels: bilinear 1 - |t| on |t| < 1;
# bicubic (Keys) with parameter a on |t| < 2; lanczos3 sinc(t) sinc(t / 3)
# on -3 <= t < 3. Nearest copies input pixel min(int(s_x), n_in - 1) with
# s_0 = scale / 2 and s_{x+1} = s_x + scale accumulated in float64, as
# Pillow does; it never antialiases. Inputs are never modified.
from typing import Literal

from numpy.typing import NDArray

Kernel = Literal["nearest", "bilinear", "bicubic", "lanczos3"]
PRECISION_BITS: int  # 22: Pillow's fixed-point fraction bits for 8-bit images

def kernel_weights(in_size: int, out_size: int, kernel: Kernel, antialias: bool = True, cubic_a: float = -0.5) -> tuple[NDArray, NDArray]:
    """(start int64 [out_size], weights float64 [out_size, taps]): output x
    is sum over t of weights[x, t] * in[start[x] + t]. taps = 2 ceil(support)
    + 1 for bilinear, bicubic, lanczos3 (entries past hi - lo are 0) and 1 for
    nearest (weight 1 at the nearest index). Every row sums to 1. ValueError
    for a size < 1 or an unknown kernel."""

def resize(img: NDArray, size: tuple[int, int], kernel: Kernel, antialias: bool = True,
           mode: Literal["pil", "float"] = "pil", cubic_a: float = -0.5) -> NDArray:
    """Resize HW or HWC img to size = (out_h, out_w).
    mode 'pil': img is uint8, the result is uint8 equal to Pillow's
    Image.resize((out_w, out_h), kernel) (cubic_a -0.5, antialias True): each
    pass converts the weights to fixed point, round(w * 2**22) with halves
    away from zero, accumulates sum in[j] * w_fixed + 2**21 in integers, and
    stores clip(acc >> 22, 0, 255) as uint8 before the next pass.
    mode 'float': any real dtype, computed in float64 with the same weights
    and no rounding (torchvision's antialiased float resize for bilinear and
    bicubic), returned as float64.
    ValueError for an unknown kernel or mode, a size < 1, a non-uint8 image
    in 'pil' mode, or an image that is not 2-D or 3-D."""

def resize_planar_f32(x_chw: NDArray, size: tuple[int, int], kernel: Kernel, align_corners: bool = False, antialias: bool = False) -> NDArray:
    """torch.nn.functional.interpolate(x[None], size, mode=kernel,
    align_corners, antialias)[0] for float CHW x, returned as float32 (computed
    in float64). antialias=False: nearest copies index min(floor(o *
    float32(in/out)), in - 1); bilinear and bicubic map output o to the source
    coordinate (o + 0.5) in/out - 0.5 (align_corners False; bilinear clamps it
    at 0) or o (in - 1)/(out - 1) (align_corners True), bicubic uses Keys with
    a = -0.75 on the 4 neighbours, and taps outside the input read the edge
    sample. antialias=True: the Pillow weights above (bicubic a = -0.5) in
    float64. ValueError for lanczos3 without antialias, align_corners with
    nearest or with antialias, a size < 1, or x not 3-D."""

def resample_audio(x: NDArray, sr_in: int, sr_out: int, lowpass_filter_width: int = 6, rolloff: float = 0.99) -> NDArray:
    """torchaudio.functional.resample(x, sr_in, sr_out, lowpass_filter_width,
    rolloff) with resampling_method='sinc_interp_hann', for float x [T] or
    [channels, T], float64 result. With g = gcd(sr_in, sr_out), orig = sr_in/g,
    new = sr_out/g, cutoff base = min(orig, new) rolloff and width =
    ceil(lowpass_filter_width orig / base): a bank of new windowed-sinc filters
    of 2 width + orig taps, applied as one conv1d with stride orig over x
    zero-padded by width on the left and width + orig on the right, then
    trimmed to ceil(new T / orig) samples. A copy of x when sr_in == sr_out.
    ValueError for a rate or width < 1, rolloff outside (0, 1], or x not
    1-D or 2-D."""
