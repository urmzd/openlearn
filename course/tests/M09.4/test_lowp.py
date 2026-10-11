"""Course tests for M09.4 (Python half): FP8 E4M3/E5M2, FP4 E2M1, and MX
block scaling with E8M0 scales (tinyllm/num/lowp.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M09.4), and the chapter section it comes from.

The golden fixture course/fixtures/M09.4/lowp_golden.npz comes from
ml_dtypes 0.6.0 (course/oracle/M09.4/lowp_golden.py): the value of every
code, and codes for float32 inputs at every code, every midpoint between
neighbouring codes (the ties), their float32 neighbours, and random bit
patterns. ml_dtypes does not saturate, so saturation is checked by hand
here. The C half is test_lowp.c; test_lowp_c_vs_python.py compares the two.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from _lib.pcg32 import PCG32
from tinyllm.num.lowp import (
    dequantize_fp8,
    e2m1_bits_to_f32,
    e8m0_scale_code,
    e8m0_to_f32,
    f32_to_e2m1_bits,
    f32_to_fp8_bits,
    fp8_bits_to_f32,
    fp8_max,
    fp8_scale,
    mx_dequantize,
    mx_quantize,
    quantize_fp8,
)

FIX = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "M09.4" / "lowp_golden.npz"
FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def golden() -> dict:
    with np.load(FIX, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def bits32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def code(x: float, fmt: str) -> int:
    return int(f32_to_fp8_bits(np.float32(x), fmt))


# --- the worked example -------------------------------------------------------


def test_hand_example():
    # WHY: section 3 by hand: 0.3 lies in the binade [0.25, 0.5), where e4m3
    #      steps by 2^-5: 0.3 * 32 = 9.6 rounds to 10, so 0.3125 with code
    #      (4 << 3) + 10 = 0x2A; e5m2 steps by 2^-4: 4.8 rounds to 5, code
    #      (12 << 2) + 5 = 0x35. 1.0625 is a tie in e4m3 (8.5 steps of 1/8)
    #      and goes to the even 8 (1.0); 1.1875 (9.5) goes to 10 (1.25).
    # KIND: unit, smoke
    # CATCHES: s01, s03, s04
    # CHAPTER: M09.4 section 3
    assert code(0.3, "e4m3") == 0x2A
    assert float(fp8_bits_to_f32(0x2A, "e4m3")) == 0.3125
    assert code(0.3, "e5m2") == 0x35
    assert float(fp8_bits_to_f32(0x35, "e5m2")) == 0.3125
    assert code(1.0625, "e4m3") == 0x38 and float(fp8_bits_to_f32(0x38, "e4m3")) == 1.0
    assert code(1.1875, "e4m3") == 0x3A and float(fp8_bits_to_f32(0x3A, "e4m3")) == 1.25
    assert code(-0.3, "e4m3") == 0xAA


def test_hand_example_mx():
    # WHY: section 3's MX block [0.25, -1.4, 3.1, 11.0] in fp4 with block 4:
    #      amax 11, floor(log2 11) = 3, fp4's emax is 2, so X = 2^1 (code
    #      128); x / X = [0.125, -0.7, 1.55, 5.5] rounds to [0, -0.5, 1.5, 6]
    #      (codes 0, 9, 3, 7) and dequantizes to [0, -1, 3, 12].
    # KIND: unit, smoke
    # CATCHES: s02, s03, s04, s05, s11, s12, s14
    # CHAPTER: M09.4 section 3
    codes, scales = mx_quantize(np.array([0.25, -1.4, 3.1, 11.0]), block=4)
    assert codes.dtype == np.uint8 and scales.dtype == np.uint8
    assert codes.tolist() == [0, 9, 3, 7]
    assert scales.tolist() == [128]
    assert mx_dequantize(codes, scales, 4, "fp4_e2m1").tolist() == [
        0.0,
        -1.0,
        3.0,
        12.0,
    ]


# --- golden: every code, and rounding against ml_dtypes ---------------------------------


def test_decode_every_code_golden():
    # WHY: decoding is exact, so all 256 fp8 codes, 16 fp4 codes, and 256
    #      e8m0 codes must equal ml_dtypes bit for bit (NaN codes as NaN):
    #      this is the table L8.5 and the KV v2 format (craft.13) read.
    # KIND: golden
    # CATCHES: s04, s05, s14
    # CHAPTER: M09.4 section 2.2
    g = golden()
    every = np.arange(256)
    for fmt in ("e4m3", "e5m2"):
        got = fp8_bits_to_f32(every, fmt)
        assert got.dtype == np.float32
        want = g[f"dec_{fmt}"]
        assert np.array_equal(np.isnan(got), np.isnan(want)), fmt
        ok = ~np.isnan(want)
        assert np.array_equal(bits32(got[ok]), bits32(want[ok])), fmt
    assert np.array_equal(bits32(e2m1_bits_to_f32(np.arange(16))), bits32(g["dec_fp4"]))
    e = e8m0_to_f32(every)
    assert np.isnan(e[255]) and np.array_equal(
        bits32(e[:255]), bits32(g["dec_e8m0"][:255])
    )


def test_encode_golden():
    # WHY: rounding to nearest with ties to even at every code, every tie
    #      between neighbouring codes, their float32 neighbours (one step on
    #      either side of a tie must round away from it), subnormals, and
    #      random bit patterns, against ml_dtypes.
    # KIND: golden
    # CATCHES: s01, s02, s03, s06, m01
    # CHAPTER: M09.4 section 2.3
    g = golden()
    for fmt in ("e4m3", "e5m2"):
        x, want = g[f"x_{fmt}"], g[f"c_{fmt}"]
        got = f32_to_fp8_bits(x, fmt)
        assert got.dtype == np.uint8
        bad = np.flatnonzero(got != want)
        assert bad.size == 0, (
            f"{fmt}: {bad.size} differ; first x={x[bad[0]]!r} got {got[bad[0]]:#04x} want {want[bad[0]]:#04x}"
        )


# --- edges -----------------------------------------------------------------------------------


def test_saturation_and_specials():
    # WHY: the course saturates instead of overflowing (an outlier activation
    #      must not become NaN in a quantized model): e4m3 has no infinity, so
    #      +-inf and everything past 448 go to +-448; e5m2 keeps +-inf but a
    #      finite value that rounds past 57344 (61440 is a tie that rounds up)
    #      saturates; NaN gives 0x7F whatever its sign.
    # KIND: boundary
    # CATCHES: s07, s08, s09
    # CHAPTER: M09.4 section 2.4
    assert fp8_max("e4m3") == 448.0 and fp8_max("e5m2") == 57344.0
    for x, want in [
        (448.0, 0x7E),
        (456.0, 0x7E),
        (464.0, 0x7E),
        (1e6, 0x7E),
        (np.inf, 0x7E),
        (-np.inf, 0xFE),
        (-500.0, 0xFE),
        (np.nan, 0x7F),
        (-np.nan, 0x7F),
    ]:
        assert code(x, "e4m3") == want, x
    for x, want in [
        (57344.0, 0x7B),
        (61439.0, 0x7B),
        (61440.0, 0x7B),
        (3e38, 0x7B),
        (-1e9, 0xFB),
        (np.inf, 0x7C),
        (-np.inf, 0xFC),
        (np.nan, 0x7F),
    ]:
        assert code(x, "e5m2") == want, x
    with pytest.raises(ValueError):
        fp8_max("e3m4")
    with pytest.raises(ValueError):
        f32_to_fp8_bits(1.0, "fp8")


def test_tiny_values_and_signed_zero():
    # WHY: the smallest e4m3 subnormal is 2^-9; exactly half of it (2^-10) is
    #      a tie that goes to the even code 0, keeping its sign, and anything
    #      above it rounds up to 2^-9. e5m2's smallest subnormal is 2^-16.
    # KIND: boundary
    # CATCHES: s01, s02, s03, s04, s05, s06, m01
    # CHAPTER: M09.4 section 2.2
    assert code(2.0**-10, "e4m3") == 0x00
    assert code(-(2.0**-10), "e4m3") == 0x80
    assert code(-0.0, "e4m3") == 0x80 and code(0.0, "e4m3") == 0x00
    assert code(2.0**-10 * (1 + 2.0**-20), "e4m3") == 0x01
    assert float(fp8_bits_to_f32(0x01, "e4m3")) == 2.0**-9
    assert float(fp8_bits_to_f32(0x08, "e4m3")) == 2.0**-6  # smallest normal
    assert code(2.0**-16, "e5m2") == 0x01 and code(2.0**-17, "e5m2") == 0x00
    assert code(1.5 * 2.0**-9, "e4m3") == 0x02  # tie between 1 and 2 steps: even


def test_encode_inverts_decode():
    # WHY: every finite code is a fixed point: decode, re-encode, same code.
    #      A wrong bias or an exponent field off by one breaks this for whole
    #      binades even when most values look plausible.
    # KIND: property
    # CATCHES: s02, s03, s04, s05, s06, s11, s14, m01
    # CHAPTER: M09.4 section 2.2
    every = np.arange(256)
    for fmt in ("e4m3", "e5m2"):
        v = fp8_bits_to_f32(every, fmt)
        ok = np.isfinite(v)
        assert np.array_equal(
            f32_to_fp8_bits(v[ok], fmt), every[ok].astype(np.uint8)
        ), fmt
    assert np.array_equal(
        f32_to_e2m1_bits(e2m1_bits_to_f32(np.arange(16))), np.arange(16)
    )
    # -0 decodes to -0.0 and re-encodes to 0x80
    assert np.signbit(fp8_bits_to_f32(0x80, "e4m3"))


def test_rounding_is_nearest_and_monotone():
    # WHY: two laws define the rounding: the result is the closest code
    #      value (never off by one code), and a larger input never gets a
    #      smaller value. Checked on 20000 random float32 values in range.
    # KIND: property
    # CATCHES: s02, s03, s04, s05
    # CHAPTER: M09.4 section 2.3
    rng = PCG32(seed(), 41)
    for fmt, lim in [("e4m3", 448.0), ("e5m2", 57344.0)]:
        table = fp8_bits_to_f32(np.arange(0x7C if fmt == "e5m2" else 0x7F), fmt).astype(
            np.float64
        )
        x = np.sort(
            (rng.uniform_array((20000,), -1.0, 1.0) ** 3 * lim).astype(np.float32)
        )
        v = fp8_bits_to_f32(f32_to_fp8_bits(x, fmt), fmt).astype(np.float64)
        assert (np.diff(v) >= 0).all()
        nearest = np.min(
            np.abs(np.abs(x.astype(np.float64))[:, None] - table[None, :]), axis=1
        )
        assert np.array_equal(np.abs(np.abs(v) - np.abs(x)), nearest)


def test_ties_go_to_even():
    # WHY: at an exact midpoint between two neighbouring codes, IEEE rounding
    #      picks the code whose last fraction bit is 0; "round half up" drifts
    #      every tie upward and breaks bit parity with the C kernels.
    # KIND: boundary
    # CATCHES: s01, s02, s03, s04, s05
    # CHAPTER: M09.4 section 2.3
    for fmt, top in [("e4m3", 0x7E), ("e5m2", 0x7B)]:
        v = fp8_bits_to_f32(np.arange(top + 1), fmt).astype(np.float64)
        mids = ((v[:-1] + v[1:]) / 2).astype(np.float32)
        c = f32_to_fp8_bits(mids, fmt)
        assert (c % 2 == 0).all(), fmt


# --- scaling ------------------------------------------------------------------------------------


def test_scaled_quantization_roundtrip():
    # WHY: per-tensor scaling maps the largest magnitude onto 448 (L8.5's
    #      fp8 weights): after dequantization every normal value is within a
    #      relative 2^-4 of the original (half a step of 3 fraction bits), and
    #      the maximum comes back exactly.
    # KIND: property
    # CATCHES: s03, s04, s15, s16, m02
    # CHAPTER: M09.4 section 2.5
    rng = PCG32(seed(), 42)
    w = rng.normal_array((4096,)) * 0.02
    amax = float(np.abs(w).max())
    s = fp8_scale(amax, "e4m3")
    assert s == amax / 448.0
    q = quantize_fp8(w, "e4m3", s)
    back = dequantize_fp8(q, "e4m3", s).astype(np.float64)
    assert back.dtype == np.float64 and dequantize_fp8(q, "e4m3", s).dtype == np.float32
    i = int(np.argmax(np.abs(w)))
    assert abs(back[i]) == pytest.approx(amax, rel=1e-6)
    normal = np.abs(w) / s >= 2.0**-6
    assert (np.abs(back - w)[normal] <= 2.0**-4 * np.abs(w)[normal] * (1 + 1e-6)).all()
    assert fp8_scale(0.0, "e4m3") == 1.0
    for bad in [-1.0, np.inf, np.nan]:
        with pytest.raises(ValueError):
            fp8_scale(bad, "e4m3")
    for bad in [0.0, -2.0, np.inf]:
        with pytest.raises(ValueError):
            quantize_fp8(w, "e4m3", bad)
        with pytest.raises(ValueError):
            dequantize_fp8(q, "e4m3", bad)


def test_quantize_rounds_once():
    # WHY: x / scale must round once, from the exact quotient: 1.0625 + 2^-30
    #      is just above a tie and belongs to 1.125, but rounding it to
    #      float32 first lands exactly on the tie, which then goes to 1.0
    #      (double rounding).
    # KIND: boundary
    # CATCHES: s03, s17
    # CHAPTER: M09.4 section 2.5
    x = 1.0625 + 2.0**-30
    assert int(quantize_fp8(np.array([x]), "e4m3", 1.0)[0]) == 0x39
    assert int(quantize_fp8(np.array([2 * x]), "e4m3", 2.0)[0]) == 0x39


# --- fp4, e8m0, and MX blocks -------------------------------------------------------------------


def test_fp4_values_and_ties():
    # WHY: fp4 E2M1 has eight magnitudes, 0, 0.5, 1, 1.5, 2, 3, 4, 6; ties go
    #      to the even code (0.75 to 1, 2.5 to 2, 3.5 to 4, 5 to 4), and
    #      anything past 6 saturates. It has no NaN or infinity to round to.
    # KIND: unit
    # CATCHES: s01, s02, s03, s04, s05, s10, s11, s14
    # CHAPTER: M09.4 section 2.6
    assert e2m1_bits_to_f32(np.arange(8)).tolist() == FP4_VALUES
    assert e2m1_bits_to_f32(np.arange(8, 16)).tolist() == [-v for v in FP4_VALUES]
    x = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.9, 100.0, -5.0, 0.26]
    want = [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, -4.0, 0.5]
    assert e2m1_bits_to_f32(f32_to_e2m1_bits(x)).tolist() == want
    for bad in [np.nan, np.inf]:
        with pytest.raises(ValueError):
            f32_to_e2m1_bits([1.0, bad])
    with pytest.raises(ValueError):
        e2m1_bits_to_f32([16])


def test_e8m0_scale_code():
    # WHY: the shared scale is 2^(floor(log2 amax) - emax), so the block's
    #      largest value lands in the element format's top binade: for fp4
    #      (emax 2) amax 6 gives code 127 (X = 1), amax 6.5 too, amax 8 gives
    #      128; for e4m3 (emax 8) amax 448 gives 127. Codes clamp to 0..254
    #      (255 is NaN) and an all-zero block uses amax = 1.
    # KIND: unit
    # CATCHES: s12, s13, m03
    # CHAPTER: M09.4 section 2.6
    assert e8m0_scale_code(6.0, "fp4_e2m1") == 127
    assert e8m0_scale_code(6.5, "fp4_e2m1") == 127
    assert e8m0_scale_code(7.999, "fp4_e2m1") == 127
    assert e8m0_scale_code(8.0, "fp4_e2m1") == 128
    assert e8m0_scale_code(448.0, "fp8_e4m3") == 127
    assert e8m0_scale_code(0.0, "fp4_e2m1") == 125
    assert e8m0_scale_code(2.0**-300, "fp4_e2m1") == 0
    assert e8m0_scale_code(2.0**300, "fp4_e2m1") == 254
    assert float(e8m0_to_f32(0)) == 2.0**-127 and float(e8m0_to_f32(254)) == 2.0**127
    for bad in [(-1.0, "fp4_e2m1"), (np.inf, "fp4_e2m1"), (1.0, "fp6")]:
        with pytest.raises(ValueError):
            e8m0_scale_code(*bad)


def test_mx_golden():
    # WHY: 32-element MX blocks of magnitudes from 2^-20 to 2^25, including an
    #      all-zero block, against ml_dtypes' fp4 and e4m3 rounding with the
    #      spec's scale rule, for both element types.
    # KIND: golden
    # CATCHES: s02, s03, s08, s10, s11, s12, s18, m01
    # CHAPTER: M09.4 section 2.6
    g = golden()
    for elem, key in [("fp4_e2m1", "fp4"), ("fp8_e4m3", "e4m3")]:
        codes, scales = mx_quantize(g["mx_x"], block=32, elem=elem)
        assert scales.shape == (16, 2)
        assert np.array_equal(scales, g[f"mx_{key}_scales"]), elem
        assert np.array_equal(codes, g[f"mx_{key}_codes"]), elem


def test_mx_block_law():
    # WHY: within a block every element is rounded on the same grid, so the
    #      dequantized value is the nearest element value times X (an
    #      independent nearest-value search here), and values past the top
    #      saturate; blocks never share or leak scales.
    # KIND: property
    # CATCHES: s02, s03, s04, s05, s10, s11, s14
    # CHAPTER: M09.4 section 2.6
    rng = PCG32(seed(), 43)
    grid_vals = np.array(FP4_VALUES)
    for _ in range(50):
        x = rng.normal_array((3, 64)) * np.ldexp(1.0, rng.below(30) - 15)
        codes, scales = mx_quantize(x, block=16)
        assert scales.shape == (3, 4)
        back = mx_dequantize(codes, scales, 16, "fp4_e2m1").astype(np.float64)
        X = np.repeat(np.ldexp(1.0, scales.astype(np.int64) - 127), 16, axis=-1)
        y = np.abs(x) / X
        best = np.min(np.abs(y[..., None] - grid_vals), axis=-1)
        assert np.array_equal(np.abs(np.abs(back) / X - y), best)


def test_mx_shapes_and_errors():
    # WHY: blocks run along the last axis and must tile it exactly; an
    #      all-zero block is legal (scale code 125 for fp4, codes 0); NaN or
    #      infinity has no MX encoding here; mismatched scales are an error.
    # KIND: boundary
    # CATCHES: m04
    # CHAPTER: M09.4 section 4
    codes, scales = mx_quantize(np.zeros((2, 8)), block=4)
    assert codes.shape == (2, 8) and scales.shape == (2, 2)
    assert scales.tolist() == [[125, 125], [125, 125]] and codes.max() == 0
    assert mx_dequantize(codes, scales, 4, "fp4_e2m1").tolist() == [[0.0] * 8] * 2
    for bad in [
        lambda: mx_quantize(np.ones(10), block=4),
        lambda: mx_quantize(np.ones(8), block=0),
        lambda: mx_quantize(np.array([1.0, np.inf, 0, 0]), block=4),
        lambda: mx_quantize(np.float64(1.0), block=1),
        lambda: mx_quantize(np.ones(4), block=4, elem="fp6"),
        lambda: mx_dequantize(
            np.zeros(8, np.uint8), np.zeros(3, np.uint8), 4, "fp4_e2m1"
        ),
    ]:
        with pytest.raises(ValueError):
            bad()
