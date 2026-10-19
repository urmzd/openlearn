"""safetensors for every course dtype (L0.6, which took this unit over from L0.0).

The layout, the reader rules, and the canonical writer rules are
formats/safetensors.md: an 8-byte little-endian header length N, N bytes of
compact JSON padded with spaces to a multiple of 8, then every tensor's bytes
back to back, little-endian and row-major. L0.0 wrote and read F32 only; this
version adds F16, BF16, F8_E4M3, I8, U8, and I32. numpy has no bfloat16 or
float8 type, so those two travel as float32 values (or as their raw bits) and
are converted here: BF16 through M09.1's round-to-nearest-even converters,
F8_E4M3 through the 256-entry code table built below.

Contract: contracts/py/tinyllm/io/safetensors.pyi.
"""

from __future__ import annotations

import json
import math
import os
from typing import Mapping, Optional

import numpy as np
from numpy.typing import NDArray

from tinyllm.num.fp import bf16_bits_to_f32, f32_to_bf16_bits

MAX_HEADER = 100_000_000

# Bytes per element of every dtype the course reads and writes.
DTYPE_SIZES = {"F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "I8": 1, "U8": 1, "I32": 4}

# The canonical writer's dtype order (formats/safetensors.md, writer rule 1),
# restricted to the course dtypes: larger alignment first.
_ORDER = ["F32", "I32", "BF16", "F16", "F8_E4M3", "I8", "U8"]

# numpy dtype of an array that maps to a safetensors dtype with no `dtypes` entry.
_INFER = {
    np.dtype(np.float32): "F32",
    np.dtype(np.float16): "F16",
    np.dtype(np.int8): "I8",
    np.dtype(np.uint8): "U8",
    np.dtype(np.int32): "I32",
}

# Little-endian storage type of each dtype (BF16 and F8_E4M3 as raw bits).
_STORE = {"F32": "<f4", "F16": "<f2", "BF16": "<u2", "F8_E4M3": "u1", "I8": "i1", "U8": "u1", "I32": "<i4"}


def _e4m3_table() -> NDArray:
    """float32 value of each of the 256 F8_E4M3 codes (the "fn" variant:
    sign, 4 exponent bits with bias 7, 3 mantissa bits; no infinities;
    S.1111.111 is NaN)."""
    out = np.empty(256, dtype=np.float32)
    for c in range(256):
        s, e, m = c >> 7, (c >> 3) & 0xF, c & 0x7
        if e == 0xF and m == 0x7:
            v = math.nan
        elif e == 0:
            v = (m / 8.0) * 2.0**-6  # subnormal: no implicit 1, exponent fixed at 1 - bias
        else:
            v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        out[c] = -v if s else v
    return out


E4M3_MAX = 448.0
_E4M3_CACHE: list = []  # the table, built on first use


def _e4m3() -> NDArray:
    """The code table, built once."""
    if not _E4M3_CACHE:
        _E4M3_CACHE.append(_e4m3_table())
    return _E4M3_CACHE[0]


def e4m3_to_f32(codes: NDArray) -> NDArray:
    """float32 values of F8_E4M3 codes (a uint8 array), exact."""
    return _e4m3()[np.asarray(codes, dtype=np.uint8)]


def f32_to_e4m3(x: NDArray) -> NDArray:
    """F8_E4M3 codes of float32 values, rounded to nearest, ties to even.
    NaN gives 0x7F, -0.0 gives 0x80. ValueError for |x| > 448 or infinity:
    the format has no infinity, so a value out of range needs a scale first."""
    f = np.asarray(x, dtype=np.float32)
    a = np.abs(f).astype(np.float64)
    finite = ~np.isnan(f)
    if np.any(a[finite] > E4M3_MAX):
        raise ValueError(f"F8_E4M3 holds |x| <= {E4M3_MAX:g}; got {float(a[finite].max()):g} (scale it first)")
    pos = _e4m3()[:0x7F].astype(np.float64)  # codes 0x00..0x7E: 0, ..., 448, increasing
    hi = np.clip(np.searchsorted(pos, a, side="left"), 0, len(pos) - 1)
    lo = np.clip(hi - 1, 0, None)
    d_lo, d_hi = a - pos[lo], pos[hi] - a
    # Nearest code; on an exact tie the even one (its last mantissa bit is 0,
    # and consecutive codes alternate that bit, across exponents too).
    pick = np.where(d_lo < d_hi, lo, np.where(d_hi < d_lo, hi, np.where(lo % 2 == 0, lo, hi)))
    code = pick.astype(np.uint8) | np.where(np.signbit(f), 0x80, 0).astype(np.uint8)
    return np.where(finite, code, np.uint8(0x7F)).astype(np.uint8)


def _encode(name: str, arr: NDArray, dtype: Optional[str]) -> tuple[str, bytes]:
    """(safetensors dtype, little-endian row-major bytes) of one tensor."""
    k = arr.dtype.newbyteorder("=")  # a big-endian float32 is still a float32
    if dtype is None:
        dtype = _INFER.get(k)
        if dtype is None:
            raise ValueError(
                f"tensor {name!r}: numpy dtype {arr.dtype} has no safetensors dtype; "
                "convert it (float32, float16, int8, uint8, int32) or name one in dtypes"
            )
    if dtype not in DTYPE_SIZES:
        raise ValueError(f"tensor {name!r}: unknown dtype {dtype!r} (one of {sorted(DTYPE_SIZES)})")
    if dtype == "BF16":
        if k == np.uint16:
            bits = arr  # raw bits, written as is
        elif k == np.float32:
            bits = f32_to_bf16_bits(arr)
        else:
            raise ValueError(f"tensor {name!r}: BF16 takes float32 values or uint16 bits, got {k}")
        out = np.asarray(bits, dtype=np.uint16)
    elif dtype == "F8_E4M3":
        if k == np.uint8:
            out = arr
        elif k == np.float32:
            out = f32_to_e4m3(arr)
        else:
            raise ValueError(f"tensor {name!r}: F8_E4M3 takes float32 values or uint8 codes, got {k}")
    elif dtype == "F16":
        if k not in (np.float16, np.float32):
            raise ValueError(f"tensor {name!r}: F16 takes float16 or float32 values, got {k}")
        out = arr.astype(np.float16)  # float32 -> float16 rounds to nearest, ties to even
    else:
        want = {"F32": np.float32, "I8": np.int8, "U8": np.uint8, "I32": np.int32}[dtype]
        if k != want:
            raise ValueError(f"tensor {name!r}: {dtype} takes {np.dtype(want)} arrays, got {k}")
        out = arr
    # Little-endian, row-major, whatever the array's byte order or memory layout.
    return dtype, np.asarray(out).astype(_STORE[dtype], copy=False).tobytes(order="C")


def save_safetensors(
    path: str,
    tensors: Mapping[str, NDArray],
    meta: Mapping[str, str],
    dtypes: Optional[Mapping[str, str]] = None,
) -> None:
    dtypes = dict(dtypes or {})
    for k, v in meta.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(f"metadata must map str to str, got {k!r}: {v!r}")
    unknown = sorted(set(dtypes) - set(tensors))
    if unknown:
        raise ValueError(f"dtypes names tensors that are not being written: {unknown}")
    encoded = []
    for name in tensors:
        if not isinstance(name, str) or name == "__metadata__":
            raise ValueError(f"bad tensor name {name!r}")
        arr = np.asarray(tensors[name])
        dt, data = _encode(name, arr, dtypes.get(name))
        encoded.append((_ORDER.index(dt), name.encode("utf-8"), name, dt, arr.shape, data))
    # Writer rule 1: by dtype (larger alignment first), then by name as UTF-8 bytes.
    encoded.sort(key=lambda e: (e[0], e[1]))
    header: dict = {}
    if meta:
        header["__metadata__"] = {k: meta[k] for k in sorted(meta, key=lambda s: s.encode("utf-8"))}
    offset = 0
    for _, _, name, dt, shape, data in encoded:
        header[name] = {"dtype": dt, "shape": [int(d) for d in shape], "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    text = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    text += b" " * (-len(text) % 8)
    with open(path, "wb") as f:
        f.write(len(text).to_bytes(8, "little"))
        f.write(text)
        for e in encoded:
            f.write(e[5])


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict:
    """json object hook: a repeated key is an error, not last-one-wins."""
    out: dict = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate key {k!r} in the header")
        out[k] = v
    return out


def _parse(path: str, head: bytes, size: int) -> tuple[int, list, dict[str, str]]:
    """Check reader rules 1 to 5 against the header bytes and the file size.
    Returns (N, [(begin, end, name, dtype, shape)] sorted by begin, metadata)."""
    # Rule 1: the length prefix and the header fit in the file.
    if size < 8 or len(head) < 8:
        raise ValueError(f"{path}: {size} bytes is too short for the 8-byte header length")
    n = int.from_bytes(head[:8], "little")
    if n > MAX_HEADER or 8 + n > size:
        raise ValueError(f"{path}: header length {n} does not fit in a {size}-byte file")
    # Rule 2: UTF-8 JSON object, no duplicate keys.
    try:
        header = json.loads(head[8 : 8 + n].decode("utf-8"), object_pairs_hook=_no_duplicates)
    except UnicodeDecodeError as e:
        raise ValueError(f"{path}: header is not UTF-8: {e}") from None
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: header is not JSON: {e}") from None
    if not isinstance(header, dict):
        raise ValueError(f"{path}: header must be a JSON object")
    # Rule 3: metadata maps strings to strings.
    meta = header.pop("__metadata__", {})
    if not isinstance(meta, dict) or not all(isinstance(v, str) for v in meta.values()):
        raise ValueError(f"{path}: __metadata__ must map strings to strings")
    # Rule 4: known dtype, non-negative integer dims, sizes agree.
    entries = []
    for name, e in header.items():
        if not isinstance(e, dict) or set(e) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"{path}: tensor {name!r} needs exactly dtype, shape, data_offsets")
        dt, shape, offs = e["dtype"], e["shape"], e["data_offsets"]
        if dt not in DTYPE_SIZES:
            raise ValueError(f"{path}: tensor {name!r} has dtype {dt!r}, not one of {sorted(DTYPE_SIZES)}")
        if not isinstance(shape, list) or not all(type(d) is int and d >= 0 for d in shape):
            raise ValueError(f"{path}: tensor {name!r} has a bad shape {shape!r}")
        if (
            not isinstance(offs, list)
            or len(offs) != 2
            or not all(type(x) is int for x in offs)
            or not 0 <= offs[0] <= offs[1]
        ):
            raise ValueError(f"{path}: tensor {name!r} has bad data_offsets {offs!r}")
        if offs[1] - offs[0] != math.prod(shape) * DTYPE_SIZES[dt]:
            raise ValueError(f"{path}: tensor {name!r}: {offs[1] - offs[0]} bytes for {dt} shape {shape}")
        entries.append((offs[0], offs[1], name, dt, shape))
    # Rule 5: sorted by begin, the tensors tile the data buffer exactly.
    entries.sort(key=lambda e: (e[0], e[1]))
    pos = 0
    for begin, end, name, _, _ in entries:
        if begin != pos:
            raise ValueError(f"{path}: tensor {name!r} begins at {begin}, expected {pos} (gap or overlap)")
        pos = end
    if 8 + n + pos != size:
        raise ValueError(f"{path}: the data buffer holds {size - 8 - n} bytes, the tensors cover {pos}")
    return n, entries, dict(meta)


def read_header(path: str) -> tuple[dict[str, tuple[str, tuple[int, ...]]], dict[str, str]]:
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) == 8:
            n = int.from_bytes(head, "little")
            # Read the header only when rule 1 holds: a corrupt length must
            # not make the reader allocate gigabytes.
            if n <= MAX_HEADER and 8 + n <= size:
                head += f.read(n)
    _, entries, meta = _parse(path, head, size)
    return {name: (dt, tuple(shape)) for _, _, name, dt, shape in entries}, meta


def load_safetensors(path: str) -> tuple[dict[str, NDArray], dict[str, str]]:
    with open(path, "rb") as f:
        buf = f.read()
    n, entries, meta = _parse(path, buf, len(buf))
    base = 8 + n
    tensors = {}
    for begin, end, name, dt, shape in entries:
        raw = np.frombuffer(buf, dtype=_STORE[dt], count=(end - begin) // DTYPE_SIZES[dt], offset=base + begin)
        if dt == "BF16":
            arr = bf16_bits_to_f32(raw)
        elif dt == "F8_E4M3":
            arr = e4m3_to_f32(raw)
        else:
            arr = raw.astype(np.dtype(_STORE[dt]).newbyteorder("="))  # native order, a writable copy
        # order="C" keeps a scalar's shape () (ascontiguousarray would make it (1,)).
        tensors[name] = np.array(arr, order="C", copy=True).reshape(shape)
    return tensors, meta
