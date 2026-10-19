"""tinyllm.num.fp (M09.1): IEEE 754 binary32 anatomy, ulp, and round to
nearest even into bfloat16 and binary16.

A binary32 float is 32 bits: 1 sign bit s, 8 exponent bits E (biased by 127),
and 23 mantissa bits M. A normal number (0 < E < 255) is
(-1)^s * (1 + M / 2^23) * 2^(E - 127); E = 0 holds zero and the subnormals
(-1)^s * (M / 2^23) * 2^-126; E = 255 holds infinity (M = 0) and NaN (M != 0).
bfloat16 keeps binary32's sign and exponent and the top 7 mantissa bits;
binary16 has 5 exponent bits (bias 15) and 10 mantissa bits. Rounding a
float32 to either is integer arithmetic on its bits through a uint32 view.

Contract: contracts/py/tinyllm/num/fp.pyi. The C conversions of M09.4 follow
the same edge rules (tinyllm/numerics.h).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

# dtype -> (precision p in bits including the implicit 1, minimum normal exponent)
_FORMATS = {"f32": (24, -126), "f16": (11, -14), "bf16": (8, -126)}
BF16_NAN = 0x7FC0  # the quiet NaN every bf16 conversion produces


def _f32(x: ArrayLike) -> NDArray:
    """x as a fresh C-contiguous float32 array of the same shape (float64
    inputs are rounded to nearest even by the cast; overflow to inf is the
    IEEE result, not an error)."""
    with np.errstate(over="ignore"):
        return np.array(x, dtype=np.float32, order="C")


def decompose_f32(x: float) -> tuple[int, int, int]:
    bits = int(_f32(x).reshape(()).view(np.uint32))
    return bits >> 31, (bits >> 23) & 0xFF, bits & 0x7FFFFF


def compose_f32(sign: int, exponent: int, mantissa: int) -> float:
    if sign not in (0, 1) or not 0 <= exponent <= 255 or not 0 <= mantissa < 1 << 23:
        raise ValueError(f"compose_f32: fields out of range: {sign}, {exponent}, {mantissa}")
    bits = np.array((sign << 31) | (exponent << 23) | mantissa, dtype=np.uint32)
    return float(bits.view(np.float32))


def ulp(x: ArrayLike, dtype: Literal["f32", "f16", "bf16"]) -> NDArray:
    if dtype not in _FORMATS:
        raise ValueError(f"ulp: dtype must be one of {sorted(_FORMATS)}, got {dtype!r}")
    p, emin = _FORMATS[dtype]
    a = np.abs(np.asarray(x, dtype=np.float64))
    # frexp writes a = m * 2^k with m in [0.5, 1), so floor(log2 a) = k - 1
    # exactly (log2 itself rounds near powers of two).
    _, k = np.frexp(a)
    e = np.maximum(k.astype(np.int64) - 1, emin)  # subnormals share emin's spacing
    out = np.ldexp(1.0, e - (p - 1))
    out = np.where(a == 0.0, np.ldexp(1.0, emin - (p - 1)), out)
    return np.where(np.isfinite(a), out, np.nan)


def f32_to_bf16_bits(x: ArrayLike) -> NDArray:
    f = _f32(x)
    b = f.view(np.uint32).astype(np.uint64)  # room for the carry
    # Round to nearest, ties to even, on the 16 bits being dropped: add just
    # under half (0x7FFF), plus 1 when the kept part is odd, then truncate.
    # A carry out of the mantissa bumps the exponent, which is exactly the
    # next binade; a carry out of the largest finite value lands on inf.
    lsb = (b >> 16) & 1
    r = ((b + 0x7FFF + lsb) >> 16).astype(np.uint16)
    return np.where(np.isnan(f), np.uint16(BF16_NAN), r)


def bf16_bits_to_f32(u16: ArrayLike) -> NDArray:
    u = np.array(u16, dtype=np.uint16, order="C").astype(np.uint32)
    return (u << 16).view(np.float32)


def round_to_bf16(x: ArrayLike) -> NDArray:
    return bf16_bits_to_f32(f32_to_bf16_bits(x))


def _f16_bits(f: NDArray) -> NDArray:
    """binary16 codes (as int64) of a float32 array, NaN mapped to 0x7E00."""
    b = f.view(np.uint32).astype(np.int64)
    sign = (b >> 31) << 15
    exp = (b >> 23) & 0xFF
    man = b & 0x7FFFFF
    # Write |f| = sig * 2^(E - 23) with an integer significand sig < 2^24.
    sig = np.where(exp > 0, man | 0x800000, man)
    E = np.where(exp > 0, exp - 127, -126)
    # binary16 spaces its values 2^(Eh - 10) apart, Eh = max(E, -14): below
    # -14 the spacing stops shrinking (subnormals). So the binary16 value is
    # q * 2^(Eh - 10) with q = sig / 2^shift, shift = 13 + (Eh - E) >= 13.
    Eh = np.maximum(E, -14)
    shift = np.minimum(13 + (Eh - E), 25)  # 25 already rounds everything to 0
    q = sig >> shift
    rem = sig & ((np.int64(1) << shift) - 1)
    half = np.int64(1) << (shift - 1)
    up = (rem > half) | ((rem == half) & ((q & 1) == 1))  # ties to even
    q = q + up
    # Normal codes are ((Eh + 15) << 10) | (q - 1024) = ((Eh + 14) << 10) + q,
    # and the same formula gives the subnormals (Eh = -14, q < 1024). A q that
    # rounded up to 2048 or 1024 carries into the exponent on its own.
    code = ((Eh + 14) << 10) + q
    code = np.where(code >= 0x7C00, 0x7C00, code)  # overflow (and inf) -> inf
    code = np.where(exp == 0xFF, np.where(man != 0, 0x7E00, 0x7C00), code)
    return np.where((exp == 0xFF) & (man != 0), 0x7E00, sign | code)


def round_to_fp16(x: ArrayLike) -> NDArray:
    f = _f32(x)
    h = _f16_bits(f)
    e16 = (h >> 10) & 0x1F
    m16 = h & 0x3FF
    mag = np.where(e16 == 0, np.ldexp(m16.astype(np.float64), -24),
                   np.ldexp((m16 + 1024).astype(np.float64), e16 - 25))
    mag = np.where(e16 == 0x1F, np.where(m16 != 0, np.nan, np.inf), mag)
    out = np.where((h >> 15) == 1, -mag, mag)
    return out.astype(np.float32)
