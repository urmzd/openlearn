# contracts/py/tinyllm/num/lowp.pyi (M09.4; the C half is tinyllm/numerics.h, M09.4 section)
# chapter: content/courses/mathematics/15-numerical-methods-and-floating-point/04-fp8-and-microscaling-formats.md
#
# Low-precision formats, emulated exactly in numpy:
#
#   e4m3      OCP FP8 E4M3 (E4M3FN): 1 sign, 4 exponent (bias 7), 3 fraction
#             bits; no infinity; 0x7F and 0xFF are NaN; max finite 448.
#   e5m2      OCP FP8 E5M2: 1, 5 (bias 15), 2; IEEE-like: 0x7C is +inf,
#             0x7D to 0x7F NaN; max finite 57344.
#   fp4_e2m1  OCP FP4 E2M1 (an MX element type): 1, 2 (bias 1), 1; values
#             0, 0.5, 1, 1.5, 2, 3, 4, 6; no infinity or NaN. Codes 0..15.
#   e8m0      OCP E8M0, the MX block scale: an unsigned exponent, value
#             2^(code - 127) for codes 0..254; 255 is NaN; no zero.
#
# Rounding is to nearest with ties to even, once, from the exact input
# value. Encoding a float32 follows the same edge rules as the C
# conversions in tinyllm/numerics.h:
#   e4m3  |x| > 448 and +-inf saturate to +-448 (0x7E, 0xFE); NaN gives 0x7F;
#         |x| <= 2^-10 gives +-0 with the sign kept.
#   e5m2  a finite x that rounds above 57344 saturates to +-57344 (0x7B,
#         0xFB); +-inf stays +-inf (0x7C, 0xFC); NaN gives 0x7F.
#   fp4   |x| above 6 saturates to +-6 (0x7, 0xF).
# Decoding is exact for every code.
#
# MX (OCP Microscaling Formats v1.0): a block of `block` consecutive values
# along the last axis shares one e8m0 scale X = 2^(floor(log2(amax)) - emax)
# with amax the block's largest magnitude and emax the exponent of the
# element format's largest normal (fp4_e2m1: 2, as 6 = 1.5 * 2^2; e4m3: 8,
# as 448 = 1.75 * 2^8); each element is x / X rounded to the element format
# with saturation. An all-zero block takes the scale amax = 1 would give.
from typing import Literal

from numpy.typing import ArrayLike, NDArray

def fp8_max(fmt: Literal["e4m3", "e5m2"]) -> float:
    """The largest finite value: 448.0 (e4m3) or 57344.0 (e5m2).
    ValueError for another format."""

def f32_to_fp8_bits(x: ArrayLike, fmt: Literal["e4m3", "e5m2"]) -> NDArray:
    """uint8 codes of float32(x) (the input is first converted to float32),
    by the rules above. Same shape as x. Bit-identical to tl_f32_to_e4m3 and
    tl_f32_to_e5m2."""

def fp8_bits_to_f32(codes: ArrayLike, fmt: Literal["e4m3", "e5m2"]) -> NDArray:
    """float32 values of uint8 codes: exact; NaN codes give NaN, e5m2's
    0x7C and 0xFC give +-inf. ValueError for a code outside 0..255."""

def fp8_scale(amax: float, fmt: Literal["e4m3", "e5m2"]) -> float:
    """Per-tensor scale amax / fp8_max(fmt), so that x / scale maps the
    largest magnitude onto the largest code; 1.0 when amax == 0.
    ValueError for a negative or non-finite amax."""

def quantize_fp8(x: ArrayLike, fmt: Literal["e4m3", "e5m2"], scale: float) -> NDArray:
    """uint8 codes of x / scale: the quotient is formed in float64 and rounded
    once to the format (no intermediate float32 rounding), with the rules
    above. ValueError unless scale is finite and > 0."""

def dequantize_fp8(q: ArrayLike, fmt: Literal["e4m3", "e5m2"], scale: float) -> NDArray:
    """float32 of fp8_bits_to_f32(q, fmt) * scale, the product formed in
    float64 and rounded once to float32. ValueError unless scale is finite
    and > 0."""

def f32_to_e2m1_bits(x: ArrayLike) -> NDArray:
    """uint8 fp4 codes (0..15, sign in bit 3) of x's float64 value, ties to
    even, saturating at +-6. ValueError for NaN or an infinity (fp4 has
    neither)."""

def e2m1_bits_to_f32(codes: ArrayLike) -> NDArray:
    """float32 values of fp4 codes 0..15. ValueError for a code above 15."""

def e8m0_to_f32(codes: ArrayLike) -> NDArray:
    """float32 2^(code - 127) (code 0 is 2^-127, a float32 subnormal); 255
    gives NaN. ValueError for a code outside 0..255."""

def e8m0_scale_code(amax: float, elem: Literal["fp4_e2m1", "fp8_e4m3"]) -> int:
    """The MX scale code of a block whose largest magnitude is amax:
    floor(log2(amax)) - emax(elem) + 127, clamped to 0..254; amax == 0 uses
    amax = 1. ValueError for a negative or non-finite amax or another elem."""

def mx_quantize(
    x: ArrayLike, block: int = 32, elem: Literal["fp4_e2m1", "fp8_e4m3"] = "fp4_e2m1"
) -> tuple[NDArray, NDArray]:
    """(codes, scales): codes uint8 with x's shape (fp4 codes 0..15, or e4m3
    codes), scales uint8 e8m0 codes of shape x.shape[:-1] + (n / block,)
    for a last axis of length n. Element i of a block with scale X is
    x_i / X rounded to elem (the quotient is exact: X is a power of two).
    ValueError unless x has at least one axis, n is a multiple of block,
    block >= 1, and every value is finite; or for another elem."""

def mx_dequantize(
    codes: ArrayLike, scales_e8m0: ArrayLike, block: int, elem: Literal["fp4_e2m1", "fp8_e4m3"]
) -> NDArray:
    """float32 elem value * 2^(scale code - 127), elementwise per block:
    the inverse map of mx_quantize up to rounding. ValueError when the
    shapes do not match (scales' last axis * block == codes' last axis)."""
