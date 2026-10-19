#!/usr/bin/env python3
"""Stdlib-only safetensors inspector.

The safetensors layout (https://github.com/huggingface/safetensors#format):

    [ 8 bytes ]  N = header length, unsigned 64-bit little-endian
    [ N bytes ]  UTF-8 JSON header:
                 {"name": {"dtype": "BF16", "shape": [4096, 4096],
                           "data_offsets": [begin, end]},
                  "__metadata__": {"format": "pt"}}
    [ rest    ]  raw tensor bytes, row-major, little-endian.
                 data_offsets are relative to the start of this buffer.

Because the header is tiny and at the front, you can inspect a checkpoint
without loading it, and even inspect a remote checkpoint with two HTTP Range
requests (no download of the weights).

Usage:
    python safetensors_inspect.py model.safetensors
    python safetensors_inspect.py model.safetensors.index.json
    python safetensors_inspect.py model.safetensors --tensor lm_head.weight --limit 8
    python safetensors_inspect.py --url https://huggingface.co/<repo>/resolve/main/model.safetensors
    python safetensors_inspect.py --selftest
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import tempfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

# Element width in bits, mirroring Dtype::bitsize() in safetensors/src/tensor.rs.
DTYPE_BITS = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "F8_E4M3": 8,
    "F8_E5M2": 8,
    "F8_E8M0": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2FNUZ": 8,
    "F4": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
    "I16": 16,
    "U16": 16,
    "F16": 16,
    "BF16": 16,
    "I32": 32,
    "U32": 32,
    "F32": 32,
    "I64": 64,
    "U64": 64,
    "F64": 64,
    "C64": 64,
}

# Dtypes often used as *containers* for packed low-bit weights (GPTQ/AWQ
# qweight in I32, MXFP4 blocks in U8). Element count != parameter count.
PACKED_CONTAINERS = {"U8", "I32", "U32"}

MAX_HEADER = 100 * 1024 * 1024  # the reference implementation also caps this


# --------------------------------------------------------------------------
# Header parsing
# --------------------------------------------------------------------------


class Source:
    """Random-access byte reader over a local file or an HTTP(S) URL."""

    def __init__(self, target: str):
        self.target = target
        self.is_url = target.startswith(("http://", "https://"))

    def read(self, offset: int, length: int) -> bytes:
        if length == 0:
            return b""
        if not self.is_url:
            with open(self.target, "rb") as f:
                f.seek(offset)
                data = f.read(length)
        else:
            req = urllib.request.Request(self.target)
            req.add_header("Range", f"bytes={offset}-{offset + length - 1}")
            token = os.environ.get("HF_TOKEN")
            if token:
                req.add_header("Authorization", f"Bearer {token}")
            with urllib.request.urlopen(req, timeout=60) as resp:  # follows redirects
                data = resp.read()
        if len(data) != length:
            raise ValueError(
                f"short read: wanted {length} bytes at {offset}, got {len(data)}"
            )
        return data


def parse_header(src: Source) -> tuple[int, dict]:
    """Return (header_len N, header dict). Data buffer starts at 8 + N."""
    (n,) = struct.unpack("<Q", src.read(0, 8))
    if n > MAX_HEADER:
        raise ValueError(f"header length {n} is implausible; not a safetensors file?")
    raw = src.read(8, n)
    if not raw.startswith(b"{"):
        raise ValueError("header must start with '{' (0x7B)")
    header = json.loads(raw.decode("utf-8"))
    return n, header


def validate(header: dict) -> list[str]:
    """Check the invariants the spec requires: sizes match shapes, no holes."""
    problems = []
    spans = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype, shape, (begin, end) = info["dtype"], info["shape"], info["data_offsets"]
        if dtype not in DTYPE_BITS:
            problems.append(f"{name}: unknown dtype {dtype}")
            continue
        expected = math.ceil(math.prod(shape) * DTYPE_BITS[dtype] / 8)
        if end - begin != expected:
            problems.append(
                f"{name}: {end - begin} bytes but {dtype}{shape} needs {expected}"
            )
        spans.append((begin, end, name))
    spans.sort()
    cursor = 0
    for begin, end, name in spans:
        if begin != cursor:
            problems.append(f"{name}: gap or overlap at offset {cursor} -> {begin}")
        cursor = end
    return problems


# --------------------------------------------------------------------------
# Decoding raw bytes (F32 / F16 / BF16 / ints) without numpy
# --------------------------------------------------------------------------

STRUCT_CODES = {
    "F32": "f",
    "F64": "d",
    "F16": "e",
    "I8": "b",
    "U8": "B",
    "I16": "h",
    "U16": "H",
    "I32": "i",
    "U32": "I",
    "I64": "q",
    "U64": "Q",
}


def decode(dtype: str, raw: bytes) -> list:
    if dtype == "BF16":
        # bfloat16 is the top 16 bits of an IEEE float32: shift left 16.
        halves = struct.unpack(f"<{len(raw) // 2}H", raw)
        return [struct.unpack("<f", struct.pack("<I", h << 16))[0] for h in halves]
    code = STRUCT_CODES.get(dtype)
    if code is None:
        raise ValueError(f"decoding {dtype} is not implemented (raw bytes only)")
    count = len(raw) // struct.calcsize(code)
    return list(struct.unpack(f"<{count}{code}", raw))


def read_tensor(src: Source, n: int, info: dict, limit: int | None = None) -> list:
    begin, end = info["data_offsets"]
    if (
        limit is not None
        and info["dtype"] in DTYPE_BITS
        and DTYPE_BITS[info["dtype"]] >= 8
    ):
        end = min(end, begin + limit * DTYPE_BITS[info["dtype"]] // 8)
    raw = src.read(8 + n + begin, end - begin)
    return decode(info["dtype"], raw)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1000 or unit == "TB":
            return f"{nbytes:.2f} {unit}" if unit != "B" else f"{int(nbytes)} B"
        nbytes /= 1000
    return f"{nbytes:.2f} TB"


def summarize(header: dict, show: int) -> dict:
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    elems = {k: math.prod(v["shape"]) for k, v in tensors.items()}
    nbytes = {
        k: v["data_offsets"][1] - v["data_offsets"][0] for k, v in tensors.items()
    }
    by_dtype = Counter()
    for k, v in tensors.items():
        by_dtype[v["dtype"]] += elems[k]
    if show:
        width = min(70, max((len(k) for k in tensors), default=10))
        print(f"  {'tensor':<{width}} {'dtype':<8} {'shape':<22} {'bytes':>12}")
        for k in sorted(tensors)[:show]:
            v = tensors[k]
            print(
                f"  {k:<{width}.{width}} {v['dtype']:<8} {v['shape']!s:<22} {human(nbytes[k]):>12}"
            )
        if len(tensors) > show:
            print(f"  ... {len(tensors) - show} more (use --show N)")
    return {
        "tensors": len(tensors),
        "elements": sum(elems.values()),
        "bytes": sum(nbytes.values()),
        "by_dtype": by_dtype,
    }


def report_file(target: str, show: int, tensor: str | None, limit: int) -> dict:
    src = Source(target)
    n, header = parse_header(src)
    print(f"== {target}")
    print(f"  header: {n} bytes, data starts at byte {8 + n}")
    meta = header.get("__metadata__")
    if meta:
        print(f"  __metadata__: {meta}")
    stats = summarize(header, show)
    problems = validate(header)
    print(
        f"  tensors={stats['tensors']}  elements={stats['elements']:,}  data={human(stats['bytes'])}"
    )
    print(
        "  elements by dtype: "
        + ", ".join(f"{d}={c:,}" for d, c in stats["by_dtype"].most_common())
    )
    packed = PACKED_CONTAINERS & set(stats["by_dtype"])
    if packed:
        print(
            f"  note: {sorted(packed)} may hold packed low-bit weights; elements != params"
        )
    print("  validation: " + ("OK" if not problems else "; ".join(problems[:5])))
    if tensor:
        if tensor not in header:
            raise SystemExit(f"tensor {tensor!r} not in this file")
        values = read_tensor(src, n, header[tensor], limit)
        print(
            f"  {tensor}[:{len(values)}] = {[round(x, 6) if isinstance(x, float) else x for x in values]}"
        )
    return stats


def report_index(path: str, show: int) -> None:
    index = json.loads(Path(path).read_text())
    weight_map: dict[str, str] = index["weight_map"]
    shards: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        shards[shard].append(name)
    total = index.get("metadata", {}).get("total_size")
    print(f"== {path}")
    print(
        f"  {len(weight_map)} tensors across {len(shards)} shards; metadata.total_size={total and human(total)}"
    )
    for shard in sorted(shards):
        print(f"  {shard}: {len(shards[shard])} tensors")
    base = Path(path).parent
    present = [s for s in sorted(shards) if (base / s).exists()]
    if not present:
        print("  (shard files not present locally; index-only view)")
        return
    grand = 0
    for shard in present:
        header = parse_header(Source(str(base / shard)))[1]
        missing = set(shards[shard]) - set(header)
        extra = set(header) - set(shards[shard]) - {"__metadata__"}
        if missing or extra:
            print(
                f"  MISMATCH in {shard}: missing={sorted(missing)[:3]} extra={sorted(extra)[:3]}"
            )
        grand += report_file(str(base / shard), show, None, 0)["bytes"]
    if total is not None and len(present) == len(shards):
        print(
            f"  sum of shard data = {human(grand)}  (index says {human(total)}; equal: {grand == total})"
        )


# --------------------------------------------------------------------------
# Self-test: write a safetensors file by hand, then parse it back
# --------------------------------------------------------------------------


def write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, list[int], bytes]],
    metadata: dict[str, str] | None = None,
) -> None:
    header: dict = {}
    if metadata:
        header["__metadata__"] = metadata
    offset = 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)  # pad so the data buffer is 8-byte aligned
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _, _, raw in tensors.values():
            f.write(raw)


def bf16_bytes(values: list[float]) -> bytes:
    # truncate float32 to its top 16 bits (round-toward-zero is fine for a test)
    return b"".join(
        struct.pack("<H", struct.unpack("<I", struct.pack("<f", v))[0] >> 16)
        for v in values
    )


def selftest() -> None:
    w = [0.5, -1.25, 3.0, 2.0, -0.125, 7.75]
    tensors = {
        "model.embed_tokens.weight": ("F32", [2, 3], struct.pack("<6f", *w)),
        "model.norm.weight": ("BF16", [4], bf16_bytes([1.0, 2.0, -3.0, 0.5])),
        "model.layers.0.mlp.qweight": ("I8", [3], struct.pack("<3b", -1, 0, 127)),
    }
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        f1 = d / "model-00001-of-00002.safetensors"
        f2 = d / "model-00002-of-00002.safetensors"
        write_safetensors(f1, dict(list(tensors.items())[:2]), {"format": "pt"})
        write_safetensors(f2, dict(list(tensors.items())[2:]))
        src = Source(str(f1))
        n, header = parse_header(src)
        assert n % 8 == 0, "header should be padded"
        assert header["__metadata__"] == {"format": "pt"}
        assert header["model.embed_tokens.weight"]["data_offsets"] == [0, 24]
        assert header["model.norm.weight"]["data_offsets"] == [24, 32]
        assert validate(header) == []
        got = read_tensor(src, n, header["model.embed_tokens.weight"])
        assert got == w, got
        assert read_tensor(src, n, header["model.norm.weight"]) == [1.0, 2.0, -3.0, 0.5]
        # a corrupt header must be caught
        bad = dict(header)
        bad["model.norm.weight"] = {
            "dtype": "BF16",
            "shape": [5],
            "data_offsets": [24, 32],
        }
        assert validate(bad), "validator should flag a shape/size mismatch"
        index = {
            "metadata": {"total_size": 24 + 8 + 3},
            "weight_map": {
                "model.embed_tokens.weight": f1.name,
                "model.norm.weight": f1.name,
                "model.layers.0.mlp.qweight": f2.name,
            },
        }
        (d / "model.safetensors.index.json").write_text(json.dumps(index))
        report_file(str(f1), 10, "model.embed_tokens.weight", 6)
        report_index(str(d / "model.safetensors.index.json"), 0)
    print("selftest OK")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "path", nargs="?", help=".safetensors file or model.safetensors.index.json"
    )
    p.add_argument(
        "--url", help="inspect a remote file via HTTP Range requests (HF_TOKEN honored)"
    )
    p.add_argument("--show", type=int, default=20, help="tensors to list (default 20)")
    p.add_argument("--tensor", help="decode this tensor's raw bytes into a list")
    p.add_argument(
        "--limit", type=int, default=16, help="max elements to decode (default 16)"
    )
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)
    if a.selftest:
        selftest()
    elif a.url:
        report_file(a.url, a.show, a.tensor, a.limit)
    elif a.path and a.path.endswith(".json"):
        report_index(a.path, a.show)
    elif a.path:
        report_file(a.path, a.show, a.tensor, a.limit)
    else:
        p.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
