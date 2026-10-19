"""Weight and KV quantization (L8.5).

Contract: contracts/py/tinyllm/infer/quant.pyi; the int4 byte layout in
contracts/formats/safetensors.md (the L9.5 kernel reads it as written here);
the KV fp8 layout in contracts/formats/kv-block.md (format v2). FP8 and MX
encodings come from M09.4's tinyllm.num.lowp, which matches the C
conversions bit for bit.

Every scheme is symmetric: a value is an integer (or a low-precision code)
times a scale shared by a row, a group, or a head. Choosing the scale is the
whole design: it maps the largest magnitude onto the largest code, so
nothing clips, and the rounding error of each element is at most half a
step.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd.tensor import Tensor
from tinyllm.eval.lm import eval_ppl
from tinyllm.nn.layers import Linear
from tinyllm.nn.module import Module
from tinyllm.num.lowp import (
    dequantize_fp8,
    fp8_max,
    mx_dequantize,
    mx_quantize,
    quantize_fp8,
)
from tinyllm.num.tolerance import matmul_error_bound


@dataclass
class Q8Tensor:
    q: NDArray
    scales: NDArray


@dataclass
class Q4Tensor:
    packed: NDArray
    scales: NDArray
    group: int
    shape: tuple[int, int]


@dataclass
class FP8Tensor:
    codes: NDArray
    scales: NDArray
    fmt: str


@dataclass
class MXTensor:
    codes: NDArray
    scales: NDArray
    block: int
    elem: str


@dataclass
class KVQuant:
    codes: NDArray
    scales: NDArray


Quantized = Union[Q8Tensor, Q4Tensor, FP8Tensor, MXTensor, KVQuant]
SCHEMES = ("int8", "q4_g32", "fp8_e4m3", "mxfp4")


def _matrix(w: ArrayLike, fn: str) -> NDArray:
    a = np.asarray(w, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"{fn}: want a 2-D weight [out, in], got shape {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{fn}: the weight holds a NaN or an infinity")
    return a


def quantize_int8_per_channel(w: ArrayLike) -> tuple[NDArray, NDArray]:
    a = _matrix(w, "quantize_int8_per_channel")
    scales = (np.abs(a).max(axis=1) / np.float32(127.0)).astype(np.float32)
    safe = np.where(scales > 0, scales, np.float32(1.0))
    q = np.rint(a / safe[:, None])
    q = np.where(scales[:, None] > 0, q, 0.0)
    return np.clip(q, -127, 127).astype(np.int8), scales


def pack_int4(q: ArrayLike) -> NDArray:
    v = np.asarray(q)
    if v.ndim != 2 or v.shape[1] % 2:
        raise ValueError(f"pack_int4: want [out, in] with in even, got shape {v.shape}")
    if v.size and (v.min() < -8 or v.max() > 7):
        raise ValueError("pack_int4: values must be in [-8, 7]")
    n = (v.astype(np.int16) & 0xF).astype(np.uint8)  # 4-bit two's complement
    # the even column goes in the low nibble, the odd one in the high nibble
    return (n[:, 0::2] | (n[:, 1::2] << 4)).astype(np.uint8)


def unpack_int4(packed: ArrayLike) -> NDArray:
    p = np.asarray(packed, dtype=np.uint8)
    out = np.empty((p.shape[0], 2 * p.shape[1]), dtype=np.int8)
    for col, nib in ((0, p & 0xF), (1, p >> 4)):
        s = nib.astype(np.int8)
        out[:, col::2] = np.where(s >= 8, s - 16, s)  # sign-extend bit 3
    return out


def quantize_int4_group(w: ArrayLike, group: int = 32) -> Q4Tensor:
    a = _matrix(w, "quantize_int4_group")
    rows, cols = a.shape
    if group < 2 or group % 2 or cols % group:
        raise ValueError(
            f"quantize_int4_group: group {group} must be even and divide in = {cols}"
        )
    g = a.reshape(rows, cols // group, group)
    amax = np.abs(g).max(axis=2)
    with np.errstate(over="ignore"):
        s16 = (amax / np.float32(7.0)).astype(np.float16)  # the scale that is stored
    if not np.isfinite(s16).all():
        raise ValueError(
            "quantize_int4_group: a scale overflows float16 (|w| above about 4.6e5)"
        )
    s = s16.astype(np.float32)[:, :, None]  # quantize against the stored scale
    q = np.where(s > 0, np.rint(g / np.where(s > 0, s, 1.0)), 0.0)
    q = np.clip(q, -8, 7).astype(np.int8).reshape(rows, cols)
    return Q4Tensor(pack_int4(q), s16, int(group), (rows, cols))


def quantize_fp8_per_channel(
    w: ArrayLike, fmt: Literal["e4m3", "e5m2"] = "e4m3"
) -> FP8Tensor:
    a = _matrix(w, "quantize_fp8_per_channel")
    top = np.float32(fp8_max(fmt))
    amax = np.abs(a).max(axis=1)
    scales = np.where(amax > 0, amax / top, np.float32(1.0)).astype(np.float32)
    codes = np.empty(a.shape, dtype=np.uint8)
    for r in range(a.shape[0]):
        codes[r] = quantize_fp8(a[r], fmt, float(scales[r]))
    return FP8Tensor(codes, scales, fmt)


def quantize_mx(
    w: ArrayLike, block: int = 32, elem: Literal["fp4_e2m1", "fp8_e4m3"] = "fp4_e2m1"
) -> MXTensor:
    a = _matrix(w, "quantize_mx")
    codes, scales = mx_quantize(a, block, elem)
    return MXTensor(
        np.asarray(codes, np.uint8), np.asarray(scales, np.uint8), int(block), elem
    )


def quantize_kv_fp8(x: ArrayLike) -> KVQuant:
    a = np.asarray(x, dtype=np.float32)
    if a.ndim != 3:
        raise ValueError(
            f"quantize_kv_fp8: want [n_kv_heads, T, d_head], got shape {a.shape}"
        )
    if not np.isfinite(a).all():
        raise ValueError("quantize_kv_fp8: the slab holds a NaN or an infinity")
    flat = np.abs(a).reshape(a.shape[0], -1)
    amax = flat.max(axis=1) if a.size else np.zeros(a.shape[0], np.float32)  # per head
    e4m3_max = np.float32(448.0)
    scales = np.where(amax > 0, amax / e4m3_max, np.float32(1.0)).astype(np.float32)
    codes = np.empty(a.shape, dtype=np.uint8)
    for h in range(a.shape[0]):
        codes[h] = quantize_fp8(a[h], "e4m3", float(scales[h]))
    return KVQuant(codes, scales)


def dequantize(q: Union[Quantized, tuple[NDArray, NDArray]]) -> NDArray:
    if isinstance(q, tuple):
        q = Q8Tensor(*q)
    if isinstance(q, Q8Tensor):
        return (
            q.q.astype(np.float32) * np.asarray(q.scales, np.float32)[:, None]
        ).astype(np.float32)
    if isinstance(q, Q4Tensor):
        vals = unpack_int4(q.packed).astype(np.float32)
        s = np.repeat(np.asarray(q.scales).astype(np.float32), q.group, axis=1)
        return vals * s
    if isinstance(q, FP8Tensor):
        rows = [
            dequantize_fp8(q.codes[r], q.fmt, float(q.scales[r]))
            for r in range(q.codes.shape[0])
        ]
        return (
            np.stack(rows).astype(np.float32)
            if rows
            else np.zeros(q.codes.shape, np.float32)
        )
    if isinstance(q, MXTensor):
        return np.asarray(
            mx_dequantize(q.codes, q.scales, q.block, q.elem), dtype=np.float32
        )
    if isinstance(q, KVQuant):
        out = np.empty(q.codes.shape, dtype=np.float32)
        for h in range(q.codes.shape[0]):
            out[h] = dequantize_fp8(q.codes[h], "e4m3", float(q.scales[h]))
        return out
    raise TypeError(f"dequantize: not a quantized tensor: {type(q).__name__}")


def nbytes(q: Union[Quantized, tuple[NDArray, NDArray]]) -> int:
    if isinstance(q, tuple):
        return int(sum(np.asarray(x).nbytes for x in q))
    data = (
        q.packed
        if isinstance(q, Q4Tensor)
        else (q.q if isinstance(q, Q8Tensor) else q.codes)
    )
    if isinstance(q, MXTensor) and q.elem == "fp4_e2m1":
        # two fp4 codes per byte once packed
        return int(data.size // 2 + np.asarray(q.scales).nbytes)
    return int(np.asarray(data).nbytes + np.asarray(q.scales).nbytes)


class QuantLinear(Module):
    def __init__(self, q: Any, bias: Optional[ArrayLike] = None) -> None:
        super().__init__()
        w = dequantize(q)
        self.q = q
        self.out_f, self.in_f = int(w.shape[0]), int(w.shape[1])
        self._w_t = np.ascontiguousarray(w.T)  # [in, out], dequantized once
        self.bias = (
            None
            if bias is None
            else np.asarray(bias, dtype=np.float32).reshape(self.out_f)
        )

    def forward(self, x: Tensor) -> Tensor:
        y = x @ self._w_t
        return y + self.bias if self.bias is not None else y


def _quantize_weight(w: NDArray, scheme: str) -> Any:
    if scheme == "int8":
        return Q8Tensor(*quantize_int8_per_channel(w))
    if scheme == "q4_g32":
        return quantize_int4_group(w, 32)
    if scheme == "fp8_e4m3":
        return quantize_fp8_per_channel(w, "e4m3")
    if scheme == "mxfp4":
        return quantize_mx(w, 32, "fp4_e2m1")
    raise ValueError(f"unknown scheme {scheme!r}; one of {SCHEMES}")


def quantize_model(
    model: Module, scheme: str, skip: Sequence[str] = ("lm_head",)
) -> Module:
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; one of {SCHEMES}")
    found = dict(model.named_modules())
    for name, mod in list(found.items()):
        if not isinstance(mod, Linear) or not name:
            continue
        if name in skip or any(part in skip for part in name.split(".")):
            continue
        parent_name, _, attr = name.rpartition(".")
        bias = None if mod.bias is None else mod.bias.data
        setattr(
            found[parent_name],
            attr,
            QuantLinear(_quantize_weight(mod.weight.data, scheme), bias),
        )
    return model


def export_q4(model: Module) -> tuple[dict[str, NDArray], dict[str, str]]:
    tensors: dict[str, NDArray] = {}
    groups = set()
    for name, mod in model.named_modules():
        if isinstance(mod, QuantLinear) and isinstance(mod.q, Q4Tensor):
            groups.add(mod.q.group)
            tensors[f"{name}.weight.qweight"] = mod.q.packed.astype(np.uint8)
            tensors[f"{name}.weight.scales"] = np.asarray(
                mod.q.scales, dtype=np.float16
            )
            if mod.bias is not None:
                tensors[f"{name}.bias"] = mod.bias.astype(np.float32)
    if len(groups) != 1:
        raise ValueError(
            f"export_q4: want Q4 weights of one group size, found {sorted(groups) or 'none'}"
        )
    for name, p in model.named_parameters():
        tensors[name] = np.asarray(p.data)
    return tensors, {"format": "tinyllm", "quant": f"int4-g{groups.pop()}-sym"}


def output_error_bound(
    w: ArrayLike, q: Any, x: ArrayLike, dtype: str = "f32"
) -> NDArray:
    a = _matrix(w, "output_error_bound").astype(np.float64)
    xs = np.asarray(x, dtype=np.float64)
    if xs.ndim != 2 or xs.shape[1] != a.shape[1]:
        raise ValueError(
            f"output_error_bound: want x [n, in = {a.shape[1]}], got shape {xs.shape}"
        )
    d = dequantize(q).astype(np.float64)
    if d.shape != a.shape:
        raise ValueError(
            f"output_error_bound: q holds {d.shape}, the weight is {a.shape}"
        )
    # The quantization moved W by |W - W'| elementwise; M09.3 adds that
    # perturbation to the rounding of the k-term dot products in dtype.
    return matmul_error_bound(a, xs.T, dtype, np.abs(d - a)).T


def quant_ppl(
    model: Module,
    scheme: str,
    ids: ArrayLike,
    ctx_len: int,
    stride: int,
    skip: Sequence[str] = ("lm_head",),
) -> dict[str, float]:
    base = eval_ppl(model, ids, ctx_len, stride)
    # quantize_model works in place: quantize a copy, so the caller keeps
    # the float model it passed in.
    quant = quantize_model(copy.deepcopy(model), scheme, skip)
    got = eval_ppl(quant, ids, ctx_len, stride)
    return {
        "ppl": float(base["ppl"]),
        "ppl_quant": float(got["ppl"]),
        "delta": float(got["ppl"] - base["ppl"]),
        "ratio": float(got["ppl"] / base["ppl"]),
    }
