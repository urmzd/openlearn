# contracts/py/tinyllm/io/safetensors.pyi (L0.0 v0: F32 only; L0.6 takes the unit over and adds every dtype)
# chapter: content/courses/ai-engineering/06-numerics-and-autograd/07-safetensors-checkpoints-and-token-streams.md
# chapter: content/courses/ai-engineering/06-numerics-and-autograd/01-byte-bigram.md (v0)
#
# The file format and the canonical writer rules are formats/safetensors.md.
# The v0 signatures are unchanged: the Pass 1 CLI verbs keep working.
#
# Course dtypes and the numpy arrays that carry them:
#
#   F32      float32            F16    float16
#   BF16     float32 values     I8     int8
#            (or uint16 bits)   U8     uint8
#   F8_E4M3  float32 values     I32    int32
#            (or uint8 codes)
#
# numpy has no bfloat16 or float8 type. The writer rounds float32 values to
# BF16 and F8_E4M3 to nearest, ties to even (BF16 through
# tinyllm.num.fp.f32_to_bf16_bits, M09.1); the reader decodes both back to
# float32, which holds every BF16 and F8_E4M3 value exactly. F8_E4M3 is the
# "fn" variant: bias 7, no infinities, S.1111.111 is NaN, largest 448.
from typing import Mapping, Optional

from numpy.typing import NDArray

DTYPE_SIZES: dict[str, int]  # {"F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "I8": 1, "U8": 1, "I32": 4}
E4M3_MAX: float  # 448.0

def save_safetensors(
    path: str,
    tensors: Mapping[str, NDArray],
    meta: Mapping[str, str],
    dtypes: Optional[Mapping[str, str]] = None,
) -> None:
    """Write tensors and metadata to path, byte-identical to the pinned
    `safetensors` library (formats/safetensors.md, "Rules a canonical writer
    follows": by dtype in the order F32, I32, BF16, F16, F8_E4M3, I8, U8, then
    by name). An empty meta omits __metadata__. Tensor bytes are row-major
    little-endian whatever the array's memory order or byte order.
    A tensor's dtype is dtypes[name] when given, else its numpy dtype's
    (float32 F32, float16 F16, int8 I8, uint8 U8, int32 I32). dtypes may also
    say F16 for float32 values (rounded, ties to even), BF16 for float32
    values or uint16 bits, and F8_E4M3 for float32 values or uint8 codes.
    ValueError for any other numpy dtype or pairing (float64 is never
    converted silently), an unknown dtype name, a dtypes key that is not a
    tensor, an F8_E4M3 value with |x| > 448 or infinite, a tensor named
    "__metadata__", or a meta key or value that is not a str."""

def read_header(path: str) -> tuple[dict[str, tuple[str, tuple[int, ...]]], dict[str, str]]:
    """({name: (dtype, shape)}, meta) after checking all five reader rules of
    formats/safetensors.md against the header and the file size, without
    reading the tensor bytes. ValueError for any rule violation (a dtype
    outside DTYPE_SIZES is unknown)."""

def load_safetensors(path: str) -> tuple[dict[str, NDArray], dict[str, str]]:
    """Read path into (tensors, meta), checking the five reader rules.
    Arrays are native-endian, C-contiguous, writable copies: F32 float32,
    F16 float16, BF16 and F8_E4M3 decoded to float32 (exact), I8 int8,
    U8 uint8, I32 int32. ValueError for any rule violation."""
