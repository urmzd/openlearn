# contracts/py/tinyllm/infer/quant.pyi (L8.5): weight and KV quantization
# chapter: content/courses/build-an-llm/p08-inference/05-quantization.md
#
# Every scheme is symmetric (no zero point) and stores small integers or
# low-precision codes plus scales; dequantizing multiplies them back. Rounding
# to an integer is round half to even (np.rint) everywhere.
#
#   int8 per channel   W[r, c] ~ q[r, c] * s[r],  s[r] = amax(row r) / 127 (float32),
#                      q = rint(W / s) in [-127, 127]; a zero row has s = 0, q = 0
#   int4 group         W[r, c] ~ q[r, c] * s[r, c // group], q in [-8, 7],
#                      s = float16(amax(group) / 7), and q = rint(W / s) computed
#                      with the float16 scale ACTUALLY stored, so |W - q*s| <= s/2;
#                      (weights are float32 and every quotient W / s is formed in
#                      float32 before rounding, here and for int8)
#                      a group whose scale is 0 (all zero, or underflow) has q = 0.
#                      Packed per formats/safetensors.md: byte b of row r holds
#                      column 2b in its low nibble and 2b + 1 in its high nibble,
#                      each in 4-bit two's complement (the L9.5 kernel layout)
#   fp8 per channel    codes of W[r] / s[r] in OCP E4M3 (or E5M2) through M09.4's
#                      tinyllm.num.lowp, s[r] = float32(amax(row r)) / float32(fp8 max),
#                      1.0 for a zero row
#   mxfp4              M09.4's mx_quantize: blocks of 32 along a row share one
#                      E8M0 power-of-two scale, elements are FP4 E2M1 codes
#   KV fp8             formats/kv-block.md format v2: one float32 scale per head
#                      of a [n_kv_heads, T, d_head] slab, s = float32(amax) /
#                      float32(448) (1.0 when amax is 0), codes E4M3 of x / s
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence, Union

from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.module import Module

@dataclass
class Q8Tensor:
    q: NDArray  # int8 [out, in]
    scales: NDArray  # float32 [out]

@dataclass
class Q4Tensor:
    packed: NDArray  # uint8 [out, in / 2], low nibble = even column
    scales: NDArray  # float16 [out, in / group]
    group: int
    shape: tuple[int, int]  # (out, in)

@dataclass
class FP8Tensor:
    codes: NDArray  # uint8 [out, in]
    scales: NDArray  # float32 [out]
    fmt: str  # "e4m3" or "e5m2"

@dataclass
class MXTensor:
    codes: NDArray  # uint8 element codes [out, in]
    scales: NDArray  # uint8 E8M0 codes [out, in / block]
    block: int
    elem: str  # "fp4_e2m1" or "fp8_e4m3"

@dataclass
class KVQuant:
    codes: NDArray  # uint8 E4M3 [n_kv_heads, T, d_head]
    scales: NDArray  # float32 [n_kv_heads]

Quantized = Union[Q8Tensor, Q4Tensor, FP8Tensor, MXTensor, KVQuant]
SCHEMES: tuple[str, ...]  # ("int8", "q4_g32", "fp8_e4m3", "mxfp4")

def quantize_int8_per_channel(w: ArrayLike) -> tuple[NDArray, NDArray]:
    """(q int8 [out, in], scales float32 [out]) for a 2-D weight, as above.
    ValueError for another rank or a non-finite value."""

def pack_int4(q: ArrayLike) -> NDArray:
    """uint8 [out, in / 2] from int values in [-8, 7] of shape [out, in] (in
    even). ValueError for an odd width or a value out of range."""

def unpack_int4(packed: ArrayLike) -> NDArray:
    """int8 [out, 2 * packed.shape[1]]: the inverse of pack_int4."""

def quantize_int4_group(w: ArrayLike, group: int = 32) -> Q4Tensor:
    """The int4 group scheme above. ValueError unless w is 2-D and finite,
    group is even and >= 2, and in is a multiple of group; ValueError when a
    scale overflows float16."""

def quantize_fp8_per_channel(w: ArrayLike, fmt: Literal["e4m3", "e5m2"] = "e4m3") -> FP8Tensor:
    """The fp8 scheme above. ValueError for another rank or format."""

def quantize_mx(w: ArrayLike, block: int = 32, elem: Literal["fp4_e2m1", "fp8_e4m3"] = "fp4_e2m1") -> MXTensor:
    """lowp.mx_quantize over the rows of a 2-D weight."""

def quantize_kv_fp8(x: ArrayLike) -> KVQuant:
    """The KV fp8 scheme above for one [n_kv_heads, T, d_head] slab.
    ValueError for another rank or a non-finite value."""

def dequantize(q: Union[Quantized, tuple[NDArray, NDArray]]) -> NDArray:
    """float32 values of any quantized form above (an int8 (q, scales)
    tuple included): q * scale, per scheme. TypeError for anything else."""

def nbytes(q: Union[Quantized, tuple[NDArray, NDArray]]) -> int:
    """Bytes stored: codes or packed values plus scales."""

class QuantLinear(Module):
    q: Any  # the quantized weight (any form dequantize accepts)
    bias: Optional[NDArray]  # float32 [out], not a parameter
    in_f: int
    out_f: int

    def __init__(self, q: Any, bias: Optional[ArrayLike] = None) -> None:
        """A Linear whose weight is stored quantized. It holds no parameters:
        it is for inference."""

    def forward(self, x: Tensor) -> Tensor:
        """x @ dequantize(q)^T + bias over the last axis: [..., in_f] -> [..., out_f]."""

def quantize_model(model: Module, scheme: str, skip: Sequence[str] = ("lm_head",)) -> Module:
    """Replace, in place, every Linear of `model` (L0.4's tinyllm.nn.layers.Linear)
    by a QuantLinear holding its weight quantized with `scheme` ("int8",
    "q4_g32", "fp8_e4m3", or "mxfp4"); a Linear whose dotted name, or any
    component of it, is in `skip` stays as it is. Returns `model`. ValueError
    for an unknown scheme."""

def export_q4(model: Module) -> tuple[dict[str, NDArray], dict[str, str]]:
    """The tensors and metadata of a `*.q4.safetensors` file (formats/
    safetensors.md): for each QuantLinear holding a Q4Tensor at dotted name
    P, "P.weight.qweight" (uint8) and "P.weight.scales" (float16), plus
    "P.bias" when it has one; every parameter of the model under its own
    name; metadata {"format": "tinyllm", "quant": "int4-g<group>-sym"}.
    ValueError when the model holds no Q4Tensor or mixes group sizes."""

def output_error_bound(w: ArrayLike, q: Any, x: ArrayLike, dtype: str = "f32") -> NDArray:
    """float64 [n, out]: the elementwise error budget of QuantLinear(q) on
    the rows of x [n, in] against the exact x @ w^T (no bias): M09.3's
    matmul_error_bound(w, x^T, dtype, dA)^T with dA = |w - dequantize(q)|,
    the perturbation the quantization made (at most scale / 2 per element
    for int8 and int4), plus the rounding of the in-term dot products in
    dtype. ValueError unless w is a finite 2-D [out, in], x is [n, in], and
    q dequantizes to w's shape."""

def quant_ppl(
    model: Module,
    scheme: str,
    ids: ArrayLike,
    ctx_len: int,
    stride: int,
    skip: Sequence[str] = ("lm_head",),
) -> dict[str, float]:
    """The perplexity cost of a scheme: L6.7's eval_ppl(model, ids, ctx_len,
    stride) before and after quantize_model(scheme, skip) of a deep copy
    (`model` itself is left unquantized): {"ppl", "ppl_quant", "delta" =
    ppl_quant - ppl, "ratio" = ppl_quant / ppl}, the numbers MS-L8's
    budgets read. ValueError as eval_ppl and quantize_model raise it."""
