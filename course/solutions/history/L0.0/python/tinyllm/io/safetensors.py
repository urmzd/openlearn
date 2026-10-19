"""safetensors v0: write and read F32 tensors (L0.0; L0.6 adds every dtype).

The layout and the canonical writer rules are formats/safetensors.md:
an 8-byte little-endian header length N, N bytes of compact JSON padded with
spaces to a multiple of 8, then every tensor's bytes back to back.

Contract: contracts/py/tinyllm/io/safetensors.pyi.
"""

from __future__ import annotations

import json
import math

import numpy as np
from numpy.typing import NDArray

MAX_HEADER = 100_000_000


def save_safetensors(path: str, tensors: dict[str, NDArray], meta: dict[str, str]) -> None:
    for k, v in meta.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(f"metadata must map str to str, got {k!r}: {v!r}")
    header: dict = {}
    if meta:
        header["__metadata__"] = {k: meta[k] for k in sorted(meta, key=lambda s: s.encode("utf-8"))}
    blobs: list[bytes] = []
    offset = 0
    # One dtype in v0, so the canonical order is simply by name, as UTF-8 bytes.
    for name in sorted(tensors, key=lambda s: s.encode("utf-8")):
        if not isinstance(name, str) or name == "__metadata__":
            raise ValueError(f"bad tensor name {name!r}")
        arr = np.asarray(tensors[name])
        if arr.dtype.kind != "f" or arr.dtype.itemsize != 4:
            raise ValueError(f"tensor {name!r}: dtype {arr.dtype} is not F32 (contract v0 writes F32 only)")
        # Little-endian, row-major, whatever the array's byte order or memory layout.
        data = arr.astype("<f4", copy=False).tobytes(order="C")
        header[name] = {"dtype": "F32", "shape": [int(d) for d in arr.shape],
                        "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blobs.append(data)
    text = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    text += b" " * (-len(text) % 8)
    with open(path, "wb") as f:
        f.write(len(text).to_bytes(8, "little"))
        f.write(text)
        for data in blobs:
            f.write(data)


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict:
    """json object hook: a repeated key is an error, not last-one-wins."""
    out: dict = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate key {k!r} in the header")
        out[k] = v
    return out


def load_safetensors(path: str) -> tuple[dict[str, NDArray], dict[str, str]]:
    with open(path, "rb") as f:
        buf = f.read()
    # Rule 1: the length prefix and the header fit in the file.
    if len(buf) < 8:
        raise ValueError(f"{path}: {len(buf)} bytes is too short for the 8-byte header length")
    n = int.from_bytes(buf[:8], "little")
    if n > MAX_HEADER or 8 + n > len(buf):
        raise ValueError(f"{path}: header length {n} does not fit in a {len(buf)}-byte file")
    # Rule 2: UTF-8 JSON object, no duplicate keys.
    try:
        header = json.loads(buf[8:8 + n].decode("utf-8"), object_pairs_hook=_no_duplicates)
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
    # Rule 4: known dtype (v0: F32 only), non-negative integer dims, sizes agree.
    entries = []
    for name, e in header.items():
        if not isinstance(e, dict) or set(e) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"{path}: tensor {name!r} needs exactly dtype, shape, data_offsets")
        if e["dtype"] != "F32":
            raise ValueError(f"{path}: tensor {name!r} has dtype {e['dtype']!r}; contract v0 reads F32 only")
        shape, offs = e["shape"], e["data_offsets"]
        if not isinstance(shape, list) or not all(type(d) is int and d >= 0 for d in shape):
            raise ValueError(f"{path}: tensor {name!r} has a bad shape {shape!r}")
        if (not isinstance(offs, list) or len(offs) != 2 or not all(type(x) is int for x in offs)
                or not 0 <= offs[0] <= offs[1]):
            raise ValueError(f"{path}: tensor {name!r} has bad data_offsets {offs!r}")
        if offs[1] - offs[0] != math.prod(shape) * 4:
            raise ValueError(f"{path}: tensor {name!r}: {offs[1] - offs[0]} bytes for shape {shape}")
        entries.append((offs[0], offs[1], name, shape))
    # Rule 5: sorted by begin, the tensors tile the data buffer exactly.
    entries.sort()
    pos = 0
    for begin, end, name, _ in entries:
        if begin != pos:
            raise ValueError(f"{path}: tensor {name!r} begins at {begin}, expected {pos} (gap or overlap)")
        pos = end
    if 8 + n + pos != len(buf):
        raise ValueError(f"{path}: the data buffer holds {len(buf) - 8 - n} bytes, the tensors cover {pos}")
    base = 8 + n
    tensors = {}
    for begin, end, name, shape in entries:
        arr = np.frombuffer(buf, dtype="<f4", count=(end - begin) // 4, offset=base + begin)
        tensors[name] = arr.astype(np.float32).reshape(shape)
    return tensors, dict(meta)
