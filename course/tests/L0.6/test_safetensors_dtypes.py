"""Course tests for L0.6: safetensors for every course dtype (tinyllm/io/safetensors.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/L0.6), and the chapter section it comes from.

The worked example of the chapter (section 3) is formats/safetensors.md's
tensor w = [[1, 2], [3, 4]] with metadata {"format": "tinyllm"}, written as
BF16 instead of F32: a 93-byte header padded to 96, then 8 data bytes,
80 3f 00 40 40 40 80 40, so the file is 8 + 96 + 8 = 112 bytes.

The golden files in course/fixtures/L0.6/safetensors/ come from the pinned
`safetensors` library (course/oracle/L0.6/safetensors_golden.py).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.pcg32 import PCG32
from tinyllm.io.safetensors import load_safetensors, read_header, save_safetensors

FIX = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "L0.6" / "safetensors"
INT = {"I8": np.int8, "U8": np.uint8, "I32": np.int32}


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def cases() -> dict:
    return json.loads((FIX / "cases.json").read_text())["cases"]


def inputs(case: dict) -> tuple[dict, dict]:
    """What a caller passes for one golden case: float32 values (or ints) and
    the dtype names numpy cannot express."""
    tensors, dtypes = {}, {}
    for t in case["tensors"]:
        tensors[t["name"]] = np.array(
            t["values"], dtype=INT.get(t["dtype"], np.float32)
        ).reshape(t["shape"])
        if t["dtype"] in ("F16", "BF16", "F8_E4M3"):
            dtypes[t["name"]] = t["dtype"]
    return tensors, dtypes


def written(
    tmp_path: Path, tensors: dict, meta: dict, dtypes: dict | None = None
) -> bytes:
    p = tmp_path / "out.safetensors"
    if dtypes is None:
        save_safetensors(str(p), tensors, meta)
    else:
        save_safetensors(str(p), tensors, meta, dtypes)
    return p.read_bytes()


def header_of(blob: bytes) -> dict:
    n = int.from_bytes(blob[:8], "little")
    return json.loads(blob[8 : 8 + n])


def file_with_header(tmp_path: Path, header: dict, data: bytes) -> str:
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    p = tmp_path / "hand.safetensors"
    p.write_bytes(len(text).to_bytes(8, "little") + text + data)
    return str(p)


# --- writing ----------------------------------------------------------------------


def test_worked_example_bf16_bytes(tmp_path):
    # WHY: the chapter's worked example, byte for byte: BF16 keeps the top 16
    #      bits of each float32 (1.0 = 0x3F800000 -> 0x3F80, stored 80 3f),
    #      the header is the format page's with "BF16" and [0,8], padded with
    #      three spaces to 96. You and the test agree on the layout first.
    # KIND: unit
    # CATCHES: s03
    # CHAPTER: L0.6 section 3, Worked example by hand
    text = b'{"__metadata__":{"format":"tinyllm"},"w":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]}}'
    assert len(text) == 93
    want = (
        (96).to_bytes(8, "little") + text + b"   " + bytes.fromhex("803f004040408040")
    )
    got = written(
        tmp_path,
        {"w": np.array([[1, 2], [3, 4]], dtype=np.float32)},
        {"format": "tinyllm"},
        {"w": "BF16"},
    )
    assert got == want, got
    assert got == (FIX / "worked_bf16.safetensors").read_bytes()


@pytest.mark.parametrize(
    "case", ["worked_bf16", "mixed", "bf16_rounding", "f8_rounding", "no_metadata"]
)
def test_matches_library_bytes(tmp_path, case):
    # WHY: byte-identical to the pinned `safetensors` library for every
    #      dtype, so any reader (yours in Rust, HF's) loads your checkpoints
    #      and a sha256 of the file means the same thing everywhere. The
    #      mixed case puts the dtypes in a file whose names run against the
    #      dtype order: name order alone lays the data out wrong.
    # KIND: golden
    # CATCHES: s01, s02, s03, s04, s05
    # CHAPTER: L0.6 section 2, Principles (canonical order for many dtypes)
    c = cases()[case]
    tensors, dtypes = inputs(c)
    got = written(tmp_path, tensors, c["meta"], dtypes)
    assert got == (FIX / c["file"]).read_bytes(), header_of(got)


def test_raw_bits_are_written_as_is(tmp_path):
    # WHY: a BF16 or F8_E4M3 tensor you already hold as bits (read from an HF
    #      file, or produced by your M09.4 quantizer later) must go to disk
    #      untouched: uint16 bits for BF16, uint8 codes for F8_E4M3. Rounding
    #      them as if they were values corrupts every weight.
    # KIND: unit
    # CATCHES: s08
    # CHAPTER: L0.6 section 4, The interface
    bits = np.array([[0x3F80, 0x4000], [0x4040, 0x4080]], dtype=np.uint16)
    got = written(tmp_path, {"w": bits}, {"format": "tinyllm"}, {"w": "BF16"})
    assert got == (FIX / "worked_bf16.safetensors").read_bytes()
    codes = np.array([0x00, 0x01, 0x38, 0x7E, 0xFE], dtype=np.uint8)
    blob = written(tmp_path, {"q": codes}, {}, {"q": "F8_E4M3"})
    assert blob[-5:] == bytes([0x00, 0x01, 0x38, 0x7E, 0xFE])
    assert header_of(blob)["q"]["dtype"] == "F8_E4M3"


def test_bf16_rounds_to_nearest_even(tmp_path):
    # WHY: float32 to BF16 drops 16 bits. Truncating them biases every weight
    #      toward zero; the rule is round to nearest, ties to even:
    #      1 + 2^-8 is halfway between 1 and 1 + 2^-7 and goes to 1 (0x3F80),
    #      1 + 3 * 2^-8 is halfway between 1 + 2^-7 and 1 + 2^-6 and goes up
    #      to the even one (0x3F82).
    # KIND: boundary
    # CATCHES: s02
    # CHAPTER: L0.6 section 5, Pitfalls, item 2
    x = np.array([1.00390625, 1.01171875, 1.0078125 + 2.0**-9], dtype=np.float32)
    blob = written(tmp_path, {"x": x}, {}, {"x": "BF16"})
    bits = np.frombuffer(blob[-6:], dtype="<u2").tolist()
    assert bits == [0x3F80, 0x3F82, 0x3F81], [hex(b) for b in bits]


def test_f8_e4m3_codes(tmp_path):
    # WHY: F8_E4M3 ("fn") has 4 exponent bits with bias 7, 3 mantissa bits,
    #      subnormals at exponent 0, no infinity, and S.1111.111 for NaN.
    #      These codes pin every part of that table: zero, the smallest
    #      subnormal 2^-9, the smallest normal 2^-6, 1.0, the largest 448,
    #      NaN, negative zero, and -448.
    # KIND: unit
    # CATCHES: s05, s06
    # CHAPTER: L0.6 section 2, Principles (F8_E4M3)
    codes = np.array([0x00, 0x01, 0x08, 0x38, 0x7E, 0x7F, 0x80, 0xFE], dtype=np.uint8)
    p = tmp_path / "q.safetensors"
    save_safetensors(str(p), {"q": codes}, {}, {"q": "F8_E4M3"})
    q = load_safetensors(str(p))[0]["q"]
    assert q.dtype == np.float32
    assert q[[0, 1, 2, 3, 4, 7]].tolist() == [0.0, 2.0**-9, 2.0**-6, 1.0, 448.0, -448.0]
    assert np.isnan(q[5]) and q[6] == 0.0 and np.signbit(q[6])


def test_f8_rounds_to_nearest_even(tmp_path):
    # WHY: between two F8_E4M3 values the nearer wins and a tie goes to the
    #      even code: 1.0625 sits halfway between 1.0 (0x38) and 1.125 (0x39)
    #      and becomes 0x38; 1.1875 between 0x39 and 0x3A becomes 0x3A; 2^-10,
    #      half the smallest subnormal, becomes 0; 250 rounds up to 256 (0x78).
    # KIND: boundary
    # CATCHES: s04
    # CHAPTER: L0.6 section 3, Worked example by hand (F8_E4M3)
    x = np.array(
        [1.0625, 1.1875, 2.0**-10, 1.5 * 2.0**-9, 250.0, -0.0], dtype=np.float32
    )
    blob = written(tmp_path, {"q": x}, {}, {"q": "F8_E4M3"})
    assert list(blob[-6:]) == [0x38, 0x3A, 0x00, 0x02, 0x78, 0x80], [
        hex(b) for b in blob[-6:]
    ]


def test_f8_rejects_out_of_range(tmp_path):
    # WHY: F8_E4M3 has no infinity, so a value beyond 448 has no code to
    #      round to: silently saturating or wrapping hides a missing scale
    #      (M09.4). NaN is representable and is written as 0x7F.
    # KIND: boundary
    # CATCHES: s07
    # CHAPTER: L0.6 section 5, Pitfalls, item 3
    for bad in (449.0, -1000.0, np.inf):
        with pytest.raises(ValueError):
            written(
                tmp_path,
                {"q": np.array([1.0, bad], dtype=np.float32)},
                {},
                {"q": "F8_E4M3"},
            )
    blob = written(
        tmp_path,
        {"q": np.array([np.nan, 448.0], dtype=np.float32)},
        {},
        {"q": "F8_E4M3"},
    )
    assert list(blob[-2:]) == [0x7F, 0x7E]


def test_dtype_order_then_name(tmp_path):
    # WHY: writer rule 1 sorts by dtype first (F32, I32, BF16, F16, F8_E4M3,
    #      I8, U8: larger alignment first), then by name. "a" as U8 and "b"
    #      as F32 are laid out b first, so b's data starts at offset 0.
    # KIND: unit
    # CATCHES: s01
    # CHAPTER: L0.6 section 2, Principles (canonical order for many dtypes)
    h = header_of(
        written(
            tmp_path,
            {
                "a": np.zeros(3, np.uint8),
                "b": np.ones(2, np.float32),
                "c": np.zeros(1, np.int32),
            },
            {"format": "tinyllm"},
        )
    )
    assert list(h) == ["__metadata__", "b", "c", "a"]
    assert (
        h["b"]["data_offsets"] == [0, 8]
        and h["c"]["data_offsets"] == [8, 12]
        and h["a"]["data_offsets"] == [12, 15]
    )
    assert [h[k]["dtype"] for k in ("b", "c", "a")] == ["F32", "I32", "U8"]


def test_save_rejects_bad_dtypes(tmp_path):
    # WHY: float64 is never converted silently (a checkpoint of float64
    #      parameters would load as different numbers in an F32 engine), an
    #      unknown dtype name or a dtypes key with no tensor is a typo, and
    #      BF16 from float16 values would round twice.
    # KIND: boundary
    # CATCHES: s09, m01
    # CHAPTER: L0.6 section 4, The interface
    f64, f16 = np.zeros(2, np.float64), np.zeros(2, np.float16)
    bad = [
        ({"x": f64}, None),
        ({"x": f64}, {"x": "F32"}),
        ({"x": np.zeros(2, np.int64)}, None),
        ({"x": np.zeros(2, np.float32)}, {"x": "F64"}),
        ({"x": np.zeros(2, np.float32)}, {"y": "BF16"}),
        ({"x": f16}, {"x": "BF16"}),
        ({"x": np.zeros(2, np.float32)}, {"x": "I32"}),
    ]
    for tensors, dtypes in bad:
        with pytest.raises(ValueError):
            written(tmp_path, tensors, {}, dtypes or {})


# --- reading ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "case", ["worked_bf16", "mixed", "bf16_rounding", "f8_rounding", "no_metadata"]
)
def test_load_decodes_every_dtype(case):
    # WHY: every library file loads to the values torch decodes, in the
    #      numpy type the contract names: float16 for F16, int arrays for
    #      I8/U8/I32, and float32 for BF16 and F8_E4M3 (exact: every value of
    #      both fits in a float32). Arrays come back writable and C-ordered,
    #      a scalar keeps shape ().
    # KIND: golden
    # CATCHES: s10, m02
    # CHAPTER: L0.6 section 4, The interface
    c = cases()[case]
    got, meta = load_safetensors(str(FIX / c["file"]))
    assert meta == c["meta"]
    assert set(got) == {t["name"] for t in c["tensors"]}
    want_type = {
        "F32": np.float32,
        "F16": np.float16,
        "BF16": np.float32,
        "F8_E4M3": np.float32,
    }
    for t in c["tensors"]:
        a = got[t["name"]]
        assert a.dtype == want_type.get(t["dtype"], INT.get(t["dtype"])), (
            t["name"],
            a.dtype,
        )
        assert (
            a.shape == tuple(t["shape"]) and a.flags.c_contiguous and a.flags.writeable
        )
        want = np.array(t["decoded"], dtype=np.float64)
        assert np.array_equal(a.astype(np.float64).reshape(-1), want, equal_nan=True), (
            t["name"],
            a,
            want,
        )
        assert np.array_equal(
            np.signbit(a.astype(np.float64).reshape(-1)), np.signbit(want)
        )


def test_read_header_checks_without_reading(tmp_path):
    # WHY: read_header gives each tensor's dtype and shape (which load
    #      cannot: BF16 and F32 both come back as float32) after checking all
    #      five reader rules, from the header and the file size alone. L10.1
    #      memory-maps the data; checking the layout first is what makes that
    #      safe.
    # KIND: unit
    # CATCHES: s12
    # CHAPTER: L0.6 section 4, The interface
    heads, meta = read_header(str(FIX / "mixed.safetensors"))
    assert meta == {"format": "tinyllm"}
    assert heads["e_bf16"] == ("BF16", (3,)) and heads["c_f8"] == ("F8_E4M3", (4,))
    assert heads["h_scalar"] == ("F32", ()) and heads["i_empty"] == ("F16", (0,))
    p = tmp_path / "trail.safetensors"
    p.write_bytes((FIX / "mixed.safetensors").read_bytes() + b"\0")
    with pytest.raises(ValueError):
        read_header(str(p))
    p.write_bytes((2**62).to_bytes(8, "little") + b"{}      ")
    with pytest.raises(ValueError):
        read_header(str(p))


def test_load_rejects_unknown_dtypes(tmp_path):
    # WHY: reader rule 4: every dtype is known. A course reader knows the
    #      seven of the format page; F64, BOOL, or a misspelling is an error,
    #      and so is a byte count that disagrees with the dtype's width.
    # KIND: boundary
    # CATCHES: s11
    # CHAPTER: L0.6 section 5, Pitfalls, item 5
    for dt, nbytes in (("F64", 16), ("BOOL", 2), ("bf16", 4), ("F16", 8)):
        p = file_with_header(
            tmp_path,
            {"x": {"dtype": dt, "shape": [2], "data_offsets": [0, nbytes]}},
            b"\0" * nbytes,
        )
        with pytest.raises(ValueError):
            load_safetensors(p)
        with pytest.raises(ValueError):
            read_header(p)


def test_raw_codes_roundtrip(tmp_path):
    # WHY: for any BF16 bits and any F8_E4M3 code (NaN excepted), writing
    #      the bits, loading the float32 values, and writing those values
    #      again gives the same file: decode is exact and rounding an exact
    #      value is the identity. A table or a rounding rule that is off for
    #      one code breaks it.
    # KIND: property
    # CATCHES: s10
    # CHAPTER: L0.6 section 2, Principles (BF16 and F8_E4M3)
    rng = PCG32(seed())
    codes = np.array([c for c in range(256) if c & 0x7F != 0x7F], dtype=np.uint8)
    bits = np.array([rng.below(1 << 16) for _ in range(300)], dtype=np.uint16)
    bits = bits[(bits & 0x7F80) != 0x7F80]  # drop BF16 infinities and NaNs
    raw = written(tmp_path, {"b": bits, "q": codes}, {}, {"b": "BF16", "q": "F8_E4M3"})
    p = tmp_path / "out.safetensors"
    vals = load_safetensors(str(p))[0]
    again = written(
        tmp_path, {"b": vals["b"], "q": vals["q"]}, {}, {"b": "BF16", "q": "F8_E4M3"}
    )
    assert again == raw


def test_v0_calls_still_work(tmp_path):
    # WHY: L0.6 takes over L0.0's unit behind the same signatures: the Pass 1
    #      CLI still calls save_safetensors(path, tensors, meta) and unpacks
    #      (tensors, meta) from load_safetensors, and the tracer engine still
    #      reads the F32 file it wrote.
    # KIND: regression
    # CHAPTER: L0.6 section 6, Where it's used next
    w = np.arange(6, dtype=np.float32).reshape(2, 3)
    p = tmp_path / "model.safetensors"
    save_safetensors(str(p), {"bigram.weight": w}, {"format": "tinyllm"})
    tensors, meta = load_safetensors(str(p))
    assert (
        meta == {"format": "tinyllm"} and tensors["bigram.weight"].dtype == np.float32
    )
    assert (tensors["bigram.weight"] == w).all()
