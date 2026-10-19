# contracts/py/tinyllm/num/fp.pyi (M09.1)
# chapter: content/courses/mathematics-for-computing/10-numerical-methods-and-floating-point/01-ieee-754.md
#
# IEEE 754 binary32 anatomy, the unit in the last place, and round to nearest
# even into bfloat16 and binary16, done on the bits (a uint32 view), with the
# same edge rules as tinyllm/numerics.h:
#   bf16  upper 16 bits after rounding; NaN gives 0x7FC0 (quiet, sign
#         dropped); finite values that round past the largest bf16 give +-inf.
#   f16   IEEE binary16 with subnormals; overflow gives +-inf; NaN gives NaN.
# Array arguments accept anything numpy converts; float inputs are first
# converted to float32 (itself a round to nearest even).
from typing import Literal

from numpy.typing import ArrayLike, NDArray

def decompose_f32(x: float) -> tuple[int, int, int]:
    """(sign, biased exponent, mantissa) of float32(x): sign in {0, 1},
    exponent in [0, 255], mantissa in [0, 2^23). 1.0 -> (0, 127, 0)."""

def compose_f32(sign: int, exponent: int, mantissa: int) -> float:
    """The float32 with those fields, as a Python float: the inverse of
    decompose_f32 for every value except NaN (exponent 255 with a non-zero
    mantissa gives some NaN). ValueError for a field out of range."""

def ulp(x: ArrayLike, dtype: Literal["f32", "f16", "bf16"]) -> NDArray:
    """float64 array: the gap from |x| to the next larger magnitude in dtype,
    2^(e - p + 1) with e = floor(log2 |x|) clamped below at the format's
    minimum normal exponent and p its precision (24, 11, 8). ulp(0) is the
    smallest subnormal; inf and NaN give NaN. ulp(1, 'f32') == 2^-23.
    ValueError for an unknown dtype."""

def f32_to_bf16_bits(x: ArrayLike) -> NDArray:
    """uint16 bfloat16 codes of float32(x), round to nearest, ties to even."""

def bf16_bits_to_f32(u16: ArrayLike) -> NDArray:
    """float32 values of bfloat16 codes: the code shifted into the upper half
    of a float32. Exact for all 65536 codes."""

def round_to_bf16(x: ArrayLike) -> NDArray:
    """float32 array: bf16_bits_to_f32(f32_to_bf16_bits(x)), NaN stays NaN.
    Idempotent and monotone (non-decreasing)."""

def round_to_fp16(x: ArrayLike) -> NDArray:
    """float32 array: float32(x) rounded to the nearest binary16 value, ties
    to even, with binary16 subnormals and overflow to +-inf; NaN stays NaN.
    Bit-identical to numpy's float32 -> float16 -> float32 conversion."""
