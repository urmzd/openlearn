"""FP8 E4M3/E5M2, FP4 E2M1, and MX block scaling with E8M0 scales (M09.4).

A format with M fraction bits and exponent bias B holds, in the binade of
floor(log2 |v|) (or the subnormal range below 2^(1 - B)), the multiples of
the quantum 2^qe, qe = max(floor(log2 |v|), 1 - B) - M. Rounding v is
rounding |v| / 2^qe to an integer n with ties to even; both the division and
the rounding are exact in float64 (a power-of-two scale, then numpy's rint),
so this file works on values, while the C twin (c/src/numerics/lowp.c) works
on bits. The code is ((qe + M + B - 1) << M) + n, which carries into the
exponent field by itself.

Contract: contracts/py/tinyllm/num/lowp.pyi.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

# name: (fraction bits M, bias B, largest finite magnitude code)
_FMT = {"e4m3": (3, 7, 0x7E), "e5m2": (2, 15, 0x7B), "fp4_e2m1": (1, 1, 0x7)}
_MAX = {"e4m3": 448.0, "e5m2": 57344.0, "fp4_e2m1": 6.0}
_EMAX = {"fp4_e2m1": 2, "fp8_e4m3": 8}


def _round_codes(a: NDArray, fmt: str) -> NDArray:
    """int64 magnitude codes of the finite float64 magnitudes a >= 0, rounded
    to nearest even, before any overflow handling (may exceed the max code)."""
    M, B, _ = _FMT[fmt]
    _, e = np.frexp(a)  # a = f * 2^e with f in [0.5, 1): floor(log2 a) = e - 1
    qe = np.maximum(e.astype(np.int64) - 1, 1 - B) - M
    n = np.rint(np.ldexp(a, -qe)).astype(np.int64)  # exact scale, then ties to even
    code = ((qe + M + B - 1) << M) + n
    return np.where(n == 0, 0, code)


def _decode(mag: NDArray, fmt: str) -> NDArray:
    """float64 values of magnitude codes (no specials)."""
    M, B, _ = _FMT[fmt]
    mag = mag.astype(np.int64)
    ef, mant = mag >> M, mag & ((1 << M) - 1)
    sub = np.ldexp(mant.astype(np.float64), 1 - B - M)
    nor = np.ldexp(((1 << M) + mant).astype(np.float64), (ef - B - M).astype(np.int64))
    return np.where(ef == 0, sub, nor)


def _check_fp8(fmt: str) -> None:
    if fmt not in ("e4m3", "e5m2"):
        raise ValueError(f"fmt must be 'e4m3' or 'e5m2', got {fmt!r}")


def _encode_fp8(v: NDArray, fmt: str) -> NDArray:
    """uint8 codes of float64 values v (any shape), with the edge rules."""
    _check_fp8(fmt)
    v = np.asarray(v, dtype=np.float64)
    nan, inf = np.isnan(v), np.isinf(v)
    a = np.where(nan | inf, 0.0, np.abs(v))
    code = np.minimum(_round_codes(a, fmt), _FMT[fmt][2])  # saturate finite overflow
    if fmt == "e4m3":
        code = np.where(inf, 0x7E, code)  # no infinity: saturate
    else:
        code = np.where(inf, 0x7C, code)
    sign = np.where(np.signbit(v), 0x80, 0)
    return np.where(nan, 0x7F, code | sign).astype(np.uint8)


def fp8_max(fmt: str) -> float:
    _check_fp8(fmt)
    return _MAX[fmt]


def f32_to_fp8_bits(x: ArrayLike, fmt: str) -> NDArray:
    return _encode_fp8(np.asarray(x, dtype=np.float32).astype(np.float64), fmt)


def fp8_bits_to_f32(codes: ArrayLike, fmt: str) -> NDArray:
    _check_fp8(fmt)
    c = np.asarray(codes)
    if c.size and (c.min() < 0 or c.max() > 255):
        raise ValueError("fp8 codes must lie in 0..255")
    c = c.astype(np.int64)
    mag = c & 0x7F
    v = _decode(mag, fmt)
    if fmt == "e4m3":
        v = np.where(mag == 0x7F, np.nan, v)
    else:
        v = np.where(mag == 0x7C, np.inf, np.where(mag > 0x7C, np.nan, v))
    return np.where(c & 0x80, -v, v).astype(np.float32)


def fp8_scale(amax: float, fmt: str) -> float:
    if not (math.isfinite(amax) and amax >= 0):
        raise ValueError(f"amax must be finite and >= 0, got {amax!r}")
    return 1.0 if amax == 0 else amax / fp8_max(fmt)


def _check_scale(scale: float) -> None:
    if not (math.isfinite(scale) and scale > 0):
        raise ValueError(f"scale must be finite and > 0, got {scale!r}")


def quantize_fp8(x: ArrayLike, fmt: str, scale: float) -> NDArray:
    _check_scale(scale)
    return _encode_fp8(np.asarray(x, dtype=np.float64) / scale, fmt)


def dequantize_fp8(q: ArrayLike, fmt: str, scale: float) -> NDArray:
    _check_scale(scale)
    return (fp8_bits_to_f32(q, fmt).astype(np.float64) * scale).astype(np.float32)


def f32_to_e2m1_bits(x: ArrayLike) -> NDArray:
    v = np.asarray(x, dtype=np.float64)
    if not np.isfinite(v).all():
        raise ValueError("fp4 has no NaN or infinity")
    code = np.minimum(_round_codes(np.abs(v), "fp4_e2m1"), 0x7)
    return (code | np.where(np.signbit(v), 0x8, 0)).astype(np.uint8)


def e2m1_bits_to_f32(codes: ArrayLike) -> NDArray:
    c = np.asarray(codes)
    if c.size and (c.min() < 0 or c.max() > 15):
        raise ValueError("fp4 codes must lie in 0..15")
    c = c.astype(np.int64)
    v = _decode(c & 0x7, "fp4_e2m1")
    return np.where(c & 0x8, -v, v).astype(np.float32)


def e8m0_to_f32(codes: ArrayLike) -> NDArray:
    c = np.asarray(codes)
    if c.size and (c.min() < 0 or c.max() > 255):
        raise ValueError("e8m0 codes must lie in 0..255")
    c = c.astype(np.int64)
    v = np.ldexp(np.ones(c.shape), c - 127)
    return np.where(c == 255, np.nan, v).astype(np.float32)


def _check_elem(elem: str) -> str:
    if elem not in _EMAX:
        raise ValueError(f"elem must be 'fp4_e2m1' or 'fp8_e4m3', got {elem!r}")
    return "fp4_e2m1" if elem == "fp4_e2m1" else "e4m3"


def e8m0_scale_code(amax: float, elem: str) -> int:
    _check_elem(elem)
    if not (math.isfinite(amax) and amax >= 0):
        raise ValueError(f"amax must be finite and >= 0, got {amax!r}")
    a = 1.0 if amax == 0 else amax
    e = math.frexp(a)[1] - 1  # floor(log2 a), exact
    return int(min(max(e - _EMAX[elem] + 127, 0), 254))


def _blocks(x: NDArray, block: int) -> NDArray:
    """x viewed as [..., n / block, block]."""
    if block < 1:
        raise ValueError(f"block must be >= 1, got {block}")
    if x.ndim < 1 or x.shape[-1] % block:
        raise ValueError(f"the last axis ({x.shape[-1:] or 'none'}) must be a multiple of block {block}")
    return x.reshape(x.shape[:-1] + (x.shape[-1] // block, block))


def mx_quantize(x: ArrayLike, block: int = 32, elem: str = "fp4_e2m1") -> tuple[NDArray, NDArray]:
    fmt = _check_elem(elem)
    v = np.asarray(x, dtype=np.float64)
    if not np.isfinite(v).all():
        raise ValueError("mx_quantize needs finite values")
    blk = _blocks(v, block)
    amax = np.max(np.abs(blk), axis=-1)
    scales = np.vectorize(lambda a: e8m0_scale_code(float(a), elem), otypes=[np.int64])(amax)
    q = np.ldexp(blk, (127 - scales)[..., None])  # x / X, exact: X is a power of two
    if fmt == "fp4_e2m1":
        codes = f32_to_e2m1_bits(q)
    else:
        codes = _encode_fp8(q, "e4m3")
    return codes.reshape(v.shape), scales.astype(np.uint8)


def mx_dequantize(codes: ArrayLike, scales_e8m0: ArrayLike, block: int, elem: str) -> NDArray:
    fmt = _check_elem(elem)
    c = np.asarray(codes)
    s = np.asarray(scales_e8m0).astype(np.int64)
    blk = _blocks(c, block)
    if s.shape != blk.shape[:-1]:
        raise ValueError(f"scales shape {s.shape} does not match {blk.shape[:-1]} blocks")
    vals = e2m1_bits_to_f32(blk) if fmt == "fp4_e2m1" else fp8_bits_to_f32(blk, "e4m3")
    out = vals.astype(np.float64) * e8m0_to_f32(s).astype(np.float64)[..., None]
    return out.astype(np.float32).reshape(c.shape)
