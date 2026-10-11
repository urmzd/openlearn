"""Course tests for L0.0: the safetensors v0 writer and reader
(tinyllm/io/safetensors.py, F32 only).

The golden files in course/fixtures/L0.0/safetensors/ were written by the
pinned `safetensors` library (course/oracle/L0.0/safetensors_golden.py);
cases.json holds the inputs that produced them. Your writer must produce the
same bytes, which is what lets the Rust engine (L10.0) and any other
safetensors reader load your checkpoint.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.pcg32 import PCG32
from tinyllm.io.safetensors import load_safetensors, save_safetensors

GOLDEN = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "L0.0" / "safetensors"


def cases() -> list[dict]:
    return json.loads((GOLDEN / "cases.json").read_text())["cases"]


def inputs(case: dict) -> dict[str, np.ndarray]:
    return {
        t["name"]: np.array(t["values"], dtype=np.float32).reshape(t["shape"])
        for t in case["tensors"]
    }


def written(
    tmp_path: Path, tensors: dict, meta: dict, name: str = "out.safetensors"
) -> bytes:
    p = tmp_path / name
    save_safetensors(str(p), tensors, meta)
    return p.read_bytes()


def split(blob: bytes) -> tuple[int, dict, bytes]:
    """(N, parsed header, data buffer) of a file, by the layout in formats/safetensors.md."""
    n = int.from_bytes(blob[:8], "little")
    return n, json.loads(blob[8 : 8 + n].decode("utf-8")), blob[8 + n :]


def raw_file(header: dict | bytes, data: bytes) -> bytes:
    text = (
        header
        if isinstance(header, bytes)
        else json.dumps(header, separators=(",", ":")).encode()
    )
    text += b" " * (-len(text) % 8)
    return len(text).to_bytes(8, "little") + text + data


# --- the writer -------------------------------------------------------------------


def test_worked_example_bytes(tmp_path):
    # WHY: the 120-byte file worked out by hand in formats/safetensors.md and
    #      the chapter: N = 96 (93 header bytes plus 3 spaces), then 1.0, 2.0,
    #      3.0, 4.0 as little-endian float32. Padding is spaces, never zeros.
    # KIND: unit
    # CATCHES: s14
    # CHAPTER: L0.0 section 3, Worked example by hand
    blob = written(
        tmp_path,
        {"w": np.array([[1, 2], [3, 4]], dtype=np.float32)},
        {"format": "tinyllm"},
    )
    assert len(blob) == 120
    assert blob[:8] == bytes.fromhex("6000000000000000")
    assert blob[8:104] == (
        b'{"__metadata__":{"format":"tinyllm"},"w":{"dtype":"F32","shape":[2,2],'
        b'"data_offsets":[0,16]}}   '
    )
    assert blob[104:] == bytes.fromhex("0000803f000000400000404000008040")
    assert blob == (GOLDEN / "worked_example.safetensors").read_bytes()


def test_matches_library_bytes(tmp_path):
    # WHY: byte-identical to the pinned library on every golden case: tensors
    #      sorted by name, __metadata__ first and omitted when empty, compact
    #      JSON with raw UTF-8, offsets relative to the data buffer, no padding
    #      when the header is already a multiple of 8, a scalar and an empty
    #      tensor, -0.0 and a subnormal kept bit for bit.
    # KIND: golden
    # CATCHES: s12, s13, s17, s18
    # CHAPTER: L0.0 section 2, Principles (the safetensors layout)
    for case in cases():
        got = written(
            tmp_path, inputs(case), dict(case["meta"]), case["name"] + ".safetensors"
        )
        want = (GOLDEN / f"{case['name']}.safetensors").read_bytes()
        assert got == want, (
            f"{case['name']}: your bytes differ from the library's\n got {got!r}\nwant {want!r}"
        )


def test_bytes_are_little_endian_row_major(tmp_path):
    # WHY: the file stores row-major little-endian float32 whatever the array
    #      in memory looks like. A transposed view is column-major in memory
    #      and a '>f4' array is big-endian; writing either buffer as is
    #      scrambles the checkpoint for every reader.
    # KIND: boundary
    # CATCHES: s15, s16
    # CHAPTER: L0.0 section 5, Pitfalls, item 7
    a = np.arange(6, dtype=np.float32).reshape(2, 3) / 4
    view = a.T  # shape (3, 2), Fortran-ordered view of a
    _, hdr, data = split(written(tmp_path, {"t": view}, {}))
    assert hdr["t"]["shape"] == [3, 2]
    assert data == np.ascontiguousarray(view).astype("<f4").tobytes()
    _, _, data = split(written(tmp_path, {"t": a.astype(">f4")}, {}))
    assert data == a.astype("<f4").tobytes()


def test_save_rejects_non_f32(tmp_path):
    # WHY: contract v0 writes F32 only. L0.6 adds supported numeric types, so
    #      int64 and bool check that unknown arrays are not silently relabeled.
    # KIND: boundary
    # CATCHES: s19
    # CHAPTER: L0.0 section 4, The interface
    for bad in (
        np.zeros(2, dtype=np.float64),
        np.zeros(2, dtype=np.int64),
        np.zeros(2, dtype=np.bool_),
    ):
        with pytest.raises(ValueError):
            save_safetensors(str(tmp_path / "x.safetensors"), {"t": bad}, {})
    with pytest.raises(ValueError):
        save_safetensors(
            str(tmp_path / "x.safetensors"),
            {"__metadata__": np.zeros(1, np.float32)},
            {},
        )
    with pytest.raises(ValueError):
        save_safetensors(
            str(tmp_path / "x.safetensors"), {"t": np.zeros(1, np.float32)}, {"n": 1}
        )


# --- the reader -------------------------------------------------------------------


def test_load_reads_library_files():
    # WHY: every file the library wrote loads back to the exact inputs and
    #      metadata, so a checkpoint from any writer that follows the format
    #      page reads correctly.
    # KIND: golden
    # CATCHES: m06, m07
    # CHAPTER: L0.0 section 2, Principles (the safetensors layout)
    for case in cases():
        tensors, meta = load_safetensors(str(GOLDEN / f"{case['name']}.safetensors"))
        want = inputs(case)
        assert meta == case["meta"]
        assert sorted(tensors) == sorted(want)
        for name, arr in want.items():
            got = tensors[name]
            assert got.dtype == np.float32 and got.shape == arr.shape, name
            assert got.tobytes() == arr.tobytes(), (
                name
            )  # bitwise: -0.0 and subnormals included


def test_roundtrip(tmp_path):
    # WHY: load(save(x)) == x for random shapes and values, with a writable
    #      native float32 result the model can use directly.
    # KIND: property
    # CATCHES: m06
    # CHAPTER: L0.0 section 4, The interface
    rng = PCG32(seed=int(os.environ.get("OL_SEED", "0")))
    for trial in range(20):
        tensors = {}
        for i in range(1 + rng.below(4)):
            shape = tuple(rng.below(4) for _ in range(rng.below(3)))
            tensors[f"t{rng.below(100)}.{i}"] = rng.normal_array(shape).astype(
                np.float32
            )
        meta = {"format": "tinyllm"} if trial % 2 else {}
        p = tmp_path / f"r{trial}.safetensors"
        save_safetensors(str(p), tensors, meta)
        back, m2 = load_safetensors(str(p))
        assert m2 == meta
        assert sorted(back) == sorted(tensors)
        for k, v in tensors.items():
            assert back[k].shape == v.shape and back[k].tobytes() == v.tobytes()
            assert back[k].flags.writeable and back[k].flags.c_contiguous


def test_load_rejects_bad_files(tmp_path):
    # WHY: a reader that trusts the header reads garbage or out of bounds.
    #      Each file breaks one rule of formats/safetensors.md: a gap between
    #      tensors, trailing bytes, a size that disagrees with the shape, a
    #      header longer than the file, a duplicate key, an unknown dtype, and
    #      supported F16 with offsets inconsistent with its declared shape.
    # KIND: boundary
    # CATCHES: s20, s21
    # CHAPTER: L0.0 section 5, Pitfalls, item 8
    four = np.zeros(1, dtype="<f4").tobytes()
    bad = {
        "gap": raw_file(
            {
                "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                "b": {"dtype": "F32", "shape": [1], "data_offsets": [8, 12]},
            },
            four * 3,
        ),
        "trailing": raw_file(
            {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}, four * 2
        ),
        "size": raw_file(
            {"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}, four
        ),
        "short": (1000).to_bytes(8, "little") + b"{}      ",
        "duplicate": raw_file(
            b'{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},'
            b'"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}',
            four,
        ),
        "f64": raw_file(
            {"a": {"dtype": "F64", "shape": [1], "data_offsets": [0, 8]}}, four * 2
        ),
        "f16_bad_size": raw_file(
            {"a": {"dtype": "F16", "shape": [2], "data_offsets": [0, 2]}}, four
        ),
        "array_header": raw_file(b"[]", b""),
        "metadata_not_str": raw_file(
            {
                "__metadata__": {"a": 1},
                "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            },
            four,
        ),
    }
    for name, blob in bad.items():
        p = tmp_path / f"{name}.safetensors"
        p.write_bytes(blob)
        with pytest.raises(ValueError):
            load_safetensors(str(p))
            pytest.fail(f"{name}: loaded a file that breaks the format")


def test_load_caps_the_header_length(tmp_path):
    # WHY: rule 1 caps N at 100,000,000 so a corrupt length cannot make the
    #      reader allocate and parse gigabytes. This file is otherwise valid
    #      (`{}` padded with spaces, exactly 8 + N bytes), so only the cap
    #      rejects it.
    # KIND: boundary
    # CHAPTER: L0.0 section 5, Pitfalls, item 8
    n = 100_000_001
    p = tmp_path / "huge.safetensors"
    with open(p, "wb") as f:
        f.write(n.to_bytes(8, "little") + b"{}")
        f.write(b" " * (n - 2))
    with pytest.raises(ValueError):
        load_safetensors(str(p))
