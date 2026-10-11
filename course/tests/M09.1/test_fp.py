"""Course tests for M09.1: float32 anatomy, ulp, and rounding to bfloat16 and
binary16 (tinyllm/num/fp.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M09.1), and the chapter section it comes from.

The chapter's worked examples (section 3): -6.25 = -1.5625 * 2^2 has fields
(1, 129, 0x480000); float32(0.1) = 0x3DCCCCCD rounds up to bf16 0x3DCD; and
1 + 2^-8 is a tie that rounds to the even neighbour 1.0.
The golden file course/fixtures/M09.1/round_f32.npz holds 10 156 float32
bit patterns with their bf16 and f16 codes, rounded with exact rational
arithmetic (course/oracle/M09.1/lowp_golden.py).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from _lib.pcg32 import PCG32
from tinyllm.num.fp import (
    bf16_bits_to_f32,
    compose_f32,
    decompose_f32,
    f32_to_bf16_bits,
    round_to_bf16,
    round_to_fp16,
    ulp,
)

GOLDEN = np.load(
    Path(os.environ.get("TINYLLM_FIXTURES", "")) / "M09.1" / "round_f32.npz",
    allow_pickle=False,
)
F32_BITS = GOLDEN["f32_bits"]
F32 = F32_BITS.view(np.float32)


def bits(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def same_f32(a, b) -> np.ndarray:
    """Bitwise equality, with any NaN equal to any NaN."""
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return (a.view(np.uint32) == b.view(np.uint32)) | (np.isnan(a) & np.isnan(b))


def test_decompose_hand_example():
    # WHY: the chapter's worked example: -6.25 = -(1 + 0.5625) * 2^2, so
    #      sign 1, biased exponent 2 + 127 = 129, mantissa 0.5625 * 2^23 =
    #      0x480000; together 0xC0C80000.
    # KIND: unit, smoke
    # CATCHES: s10
    # CHAPTER: M09.1 section 3
    assert decompose_f32(-6.25) == (1, 129, 0x480000)
    assert decompose_f32(1.0) == (0, 127, 0)
    assert decompose_f32(-2.5) == (1, 128, 0x200000)
    assert bits(-6.25) == 0xC0C80000


def test_decompose_every_class():
    # WHY: zero, the smallest subnormal (mantissa 1, exponent 0), the largest
    #      finite value, infinity, and NaN each live in a different corner of
    #      the encoding; the reader of a safetensors header (L7.9) meets all.
    # KIND: boundary
    # CATCHES: s10
    # CHAPTER: M09.1 section 2.1
    assert decompose_f32(0.0) == (0, 0, 0)
    assert decompose_f32(-0.0) == (1, 0, 0)
    assert decompose_f32(2.0**-149) == (0, 0, 1)
    assert decompose_f32(2.0**-126) == (0, 1, 0)
    assert decompose_f32(float(np.finfo(np.float32).max)) == (0, 254, 0x7FFFFF)
    assert decompose_f32(float("inf")) == (0, 255, 0)
    s, e, m = decompose_f32(float("nan"))
    assert e == 255 and m != 0


def test_compose_roundtrip():
    # WHY: compose is the inverse of decompose on every non-NaN float32, so
    #      the fields fully describe the number.
    # KIND: property
    # CATCHES: s10, m01
    # CHAPTER: M09.1 section 2.1
    for x in F32[~np.isnan(F32)][:3000]:
        s, e, m = decompose_f32(float(x))
        assert same_f32(compose_f32(s, e, m), x)
    assert np.isnan(compose_f32(0, 255, 1))
    for bad in ((2, 0, 0), (0, 256, 0), (0, 0, 1 << 23), (0, -1, 0)):
        with pytest.raises(ValueError):
            compose_f32(*bad)


def test_ulp_hand_values():
    # WHY: ulp(1) = 2^-23 in float32, 2^-10 in binary16, 2^-7 in bfloat16:
    #      the relative precision of each format. ulp(0) is the smallest
    #      subnormal of each, and inf or NaN have no ulp.
    # KIND: unit, smoke
    # CATCHES: m03
    # CHAPTER: M09.1 section 2.2
    assert ulp(1.0, "f32") == 2.0**-23
    assert ulp(1.0, "f16") == 2.0**-10
    assert ulp(1.0, "bf16") == 2.0**-7
    assert ulp([0.0], "f32").tolist() == [2.0**-149]
    assert ulp(0.0, "f16") == 2.0**-24 and ulp(0.0, "bf16") == 2.0**-133
    assert np.isnan(ulp([np.inf, -np.inf, np.nan], "bf16")).all()
    with pytest.raises(ValueError):
        ulp(1.0, "f8")


def test_ulp_matches_numpy_spacing():
    # WHY: numpy's spacing(x) is the gap from x to the next float away from
    #      zero: the same quantity for float32 and float16, including
    #      subnormals, where the spacing stops shrinking. (At the largest
    #      finite value numpy says inf, the next float being infinity; the
    #      contract keeps the binade's spacing there, so that one is left out.)
    # KIND: differential
    # CATCHES: s05
    # CHAPTER: M09.1 section 2.2
    x32 = F32[np.abs(F32) < np.finfo(np.float32).max]
    assert (ulp(x32, "f32") == np.abs(np.spacing(x32)).astype(np.float64)).all()
    g = PCG32(int(os.environ.get("OL_SEED", "0")), seq=111)
    x16 = np.array([g.below(0x7BFF) for _ in range(2000)], dtype=np.uint16).view(
        np.float16
    )
    assert (ulp(x16, "f16") == np.spacing(x16).astype(np.float64)).all()


def test_ulp_just_below_a_power_of_two():
    # WHY: floor(log2 x) computed with np.log2 rounds up to k for x a hair
    #      below 2^k (log2 of 1024 * (1 - 2^-53) is 10 - 1.6e-16, which rounds
    #      to 10.0); the exponent field, or frexp, is exact.
    # KIND: boundary
    # CATCHES: s04
    # CHAPTER: M09.1 section 5, Pitfalls
    x = np.nextafter(1024.0, 0.0)
    assert ulp(x, "f32") == 2.0 ** (9 - 23)
    assert ulp(np.nextafter(2.0**100, 0.0), "bf16") == 2.0 ** (99 - 7)


def test_bf16_hand_examples():
    # WHY: the chapter's worked examples. float32(0.1) = 0x3DCCCCCD: the
    #      dropped half 0xCCCD is above 0x8000, so it rounds up to 0x3DCD
    #      (0.10009765625). 1 + 2^-8 is exactly halfway between 1 and
    #      1 + 2^-7 and goes to the even code 0x3F80; 1 + 3 * 2^-8 is halfway
    #      between odd 0x3F81 and even 0x3F82 and goes up.
    # KIND: unit, smoke
    # CATCHES: s01, s02
    # CHAPTER: M09.1 section 3
    assert int(f32_to_bf16_bits(0.1)) == 0x3DCD
    assert float(round_to_bf16(0.1)) == 0.10009765625
    assert f32_to_bf16_bits([1.0 + 2.0**-8, 1.0 + 3 * 2.0**-8]).tolist() == [
        0x3F80,
        0x3F82,
    ]


def test_bf16_matches_exact_rational_golden():
    # WHY: every float32 class and every tie, against rounding done with
    #      exact fractions: ties to even, overflow past 0x7F7F to inf, the
    #      sign kept on zeros and infinities.
    # KIND: golden
    # CATCHES: s01, s02, s03
    # CHAPTER: M09.1 section 2.3
    got = f32_to_bf16_bits(F32)
    assert got.dtype == np.uint16
    bad = np.flatnonzero(got != GOLDEN["bf16_bits"])
    assert bad.size == 0, (
        f"{bad.size} codes differ; first f32 0x{int(F32_BITS[bad[0]]):08X}"
    )


def test_bf16_nan_stays_nan():
    # WHY: 0x7F800001 is a NaN, but adding the rounding bias to its bits
    #      gives 0x7F808000, whose top half 0x7F80 is +inf. A NaN loss that
    #      turns into inf no longer trips isnan checks (L11.1 skips steps on
    #      them). Every NaN maps to the one quiet code 0x7FC0.
    # KIND: boundary
    # CATCHES: s03
    # CHAPTER: M09.1 section 5, Pitfalls
    nans = np.array(
        [0x7F800001, 0xFF800001, 0x7FC00000, 0xFFFFFFFF], dtype=np.uint32
    ).view(np.float32)
    assert f32_to_bf16_bits(nans).tolist() == [0x7FC0] * 4
    assert np.isnan(round_to_bf16(nans)).all()


def test_bf16_all_codes_roundtrip():
    # WHY: all 65536 bf16 codes: decoding is exact and encoding the result
    #      gives the code back (NaN codes give 0x7FC0). This is how L7.9 loads
    #      bf16 checkpoints into float32.
    # KIND: property
    # CATCHES: s09
    # CHAPTER: M09.1 section 2.3
    codes = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    f = bf16_bits_to_f32(codes)
    assert f.dtype == np.float32
    assert (f.view(np.uint32) == codes.astype(np.uint32) << 16).all()
    back = f32_to_bf16_bits(f)
    nan = np.isnan(f)
    assert (back[~nan] == codes[~nan]).all() and (back[nan] == 0x7FC0).all()
    assert float(bf16_bits_to_f32(0x3F80)) == 1.0


def test_rounding_is_idempotent_and_monotone():
    # WHY: rounding twice changes nothing (a bf16 value is its own nearest
    #      bf16), and rounding never reorders: x <= y implies
    #      round(x) <= round(y). Mixed-precision training (L11.1) relies on
    #      both when it compares and clips rounded values.
    # KIND: property
    # CATCHES: s07
    # CHAPTER: M09.1 section 2.3
    x = np.sort(F32[np.isfinite(F32)])
    for rnd in (round_to_bf16, round_to_fp16):
        r = rnd(x)
        assert same_f32(rnd(r), r).all()
        assert (r[1:] >= r[:-1]).all()


def test_fp16_matches_numpy_bitwise():
    # WHY: numpy's float32 to float16 cast is IEEE round to nearest even with
    #      subnormals; on the golden inputs (ties, overflow, subnormals) and
    #      on 10^5 random bit patterns your bit-level version must agree
    #      exactly, and the golden f16 column holds the same codes.
    # KIND: differential, golden
    # CATCHES: s06, s07, s08, m02
    # CHAPTER: M09.1 section 2.4
    g = PCG32(int(os.environ.get("OL_SEED", "0")), seq=112)
    rnd = np.array([g.next_u32() for _ in range(100_000)], dtype=np.uint32).view(
        np.float32
    )
    for x in (F32, rnd):
        with np.errstate(all="ignore"):
            want = x.astype(np.float16).astype(np.float32)
        got = round_to_fp16(x)
        assert got.dtype == np.float32
        bad = np.flatnonzero(~same_f32(got, want))
        assert bad.size == 0, (
            f"{bad.size} differ; first f32 0x{int(x.view(np.uint32)[bad[0]]):08X}"
        )
    gold = GOLDEN["f16_bits"].view(np.float16).astype(np.float32)
    assert same_f32(round_to_fp16(F32), gold).all()


def test_fp16_edges():
    # WHY: binary16 tops out at 65504; 65519.99 still rounds down to it and
    #      65520 (the tie with the would-be 65536) rounds to inf. Below
    #      2^-14 the values are subnormal, spaced 2^-24 apart, and 2^-25 is a
    #      tie that goes to 0. Activations hit both ends in fp16 training,
    #      which is why bf16 exists.
    # KIND: boundary
    # CATCHES: s06, s07, s08
    # CHAPTER: M09.1 section 2.4
    got = round_to_fp16(
        [65504.0, 65519.0, 65520.0, -1e6, 2.0**-20, 2.0**-25, 3 * 2.0**-25, -(2.0**-24)]
    )
    want = [65504.0, 65504.0, np.inf, -np.inf, 2.0**-20, 0.0, 2.0**-23, -(2.0**-24)]
    assert got.tolist() == want


def test_shapes_and_scalars():
    # WHY: callers pass scalars, vectors, and weight matrices; the result
    #      keeps the input's shape (a 0-d array for a scalar).
    # KIND: boundary
    # CATCHES: m02
    # CHAPTER: M09.1 section 4
    W = np.arange(-6, 6, dtype=np.float32).reshape(3, 4) / 3
    assert round_to_bf16(W).shape == (3, 4)
    assert f32_to_bf16_bits(W).shape == (3, 4)
    assert round_to_fp16(W).shape == (3, 4)
    assert np.asarray(round_to_fp16(1.5)).shape == ()
    assert float(round_to_fp16(-1.5)) == -1.5
