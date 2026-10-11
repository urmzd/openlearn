"""Course tests for M00.1: units of information (tinyllm/num/units.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M00.1), and the chapter section it comes from.

The chapter's worked example (section 3) prices the text "abbacab" under the
L0.0 bigram model: six predictions costing ln 2 three times, ln 2.5 twice,
and ln 3 once, 5.0106 nats in total, which is 1.2048 bits per byte.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.num.units import bits_per_byte, bits_to_nats, log_base, nats_to_bits

GOLDEN = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "M00.1" / "units.json"

HAND_NLL_SUM = 3 * math.log(2) + 2 * math.log(2.5) + math.log(3)  # 5.0106352940...
HAND_BPB = 1.2048031150826468  # HAND_NLL_SUM / (6 ln 2), worked out in section 3


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def golden() -> dict:
    return json.loads(GOLDEN.read_text())


# --- the worked example ----------------------------------------------------------


def test_hand_example_bits_per_byte():
    # WHY: the chapter's worked example, number for number. The bigram model
    #      of L0.0 predicts 6 bytes of "abbacab" at a total cost of
    #      3 ln 2 + 2 ln 2.5 + ln 3 nats; shared by 6 bytes and divided by
    #      ln 2 per bit, that is 1.2048 bits per byte.
    # KIND: unit
    # CATCHES: s01, s04, s12
    # CHAPTER: M00.1 section 3, Worked example by hand
    assert_close(bits_per_byte(HAND_NLL_SUM, 6), HAND_BPB, dtype="float64")
    # The same text in nats per byte, for comparison: 0.8351 nats per byte.
    assert_close(nats_to_bits(HAND_NLL_SUM / 6), HAND_BPB, dtype="float64")


def test_uniform_byte_model_is_eight_bits():
    # WHY: a model that gives every one of the 256 bytes probability 1/256
    #      pays ln 256 nats per byte, which is exactly log2(256) = 8 bits.
    #      Any real model must beat 8 bits per byte or it has learned nothing.
    # KIND: unit
    # CATCHES: s04, s12
    # CHAPTER: M00.1 section 2.4, Units of information
    for n in (1, 7, 1000):
        assert_close(bits_per_byte(n * math.log(256), n), 8.0, dtype="float64")


# --- nats and bits -----------------------------------------------------------------


def test_one_bit_is_ln2_nats():
    # WHY: the conversion factor itself. 1 bit = ln 2 = 0.693 nats, and
    #      1 nat = 1 / ln 2 = 1.4427 bits. Multiplying where you should divide
    #      is the most common slip, and it is off by a factor of 2.08.
    # KIND: unit
    # CATCHES: s01, s08
    # CHAPTER: M00.1 section 2.4, Units of information
    assert_close(nats_to_bits(math.log(2)), 1.0, dtype="float64")
    assert_close(nats_to_bits(1.0), 1.4426950408889634, dtype="float64")
    assert_close(bits_to_nats(1.0), math.log(2), dtype="float64")
    assert_close(bits_to_nats(8.0), math.log(256), dtype="float64")


def test_nats_to_bits_golden():
    # WHY: values computed by mpmath at 60 digits (course/oracle/M00.1), so
    #      the factor is checked to full float64 precision, not just roughly.
    # KIND: golden
    # CATCHES: s01
    # CHAPTER: M00.1 section 2.4, Units of information
    for case in golden()["nats_to_bits"]:
        assert_close(nats_to_bits(case["nats"]), case["bits"], dtype="float64")


def test_nats_bits_roundtrip():
    # WHY: the two conversions are inverses: converting to bits and back
    #      returns the number you started with, for any magnitude.
    # KIND: property
    # CATCHES: s01, s08
    # CHAPTER: M00.1 section 2.4, Units of information
    rng = PCG32(seed=seed())
    for _ in range(200):
        x = math.exp(rng.uniform() * 80 - 40)  # 4e-18 .. 2e17
        assert_close(bits_to_nats(nats_to_bits(x)), x, dtype="float64")
        assert_close(nats_to_bits(bits_to_nats(x)), x, dtype="float64")


# --- change of base ------------------------------------------------------------------


def test_change_of_base_hand_values():
    # WHY: log_8(32) = ln 32 / ln 8 = 5 ln 2 / 3 ln 2 = 5/3, the section 2.3
    #      example; log_10(1000) = 3; log_{1/2}(8) = -3 (a base below 1 flips
    #      the sign).
    # KIND: unit
    # CATCHES: s02, s03, s09, s10, m02
    # CHAPTER: M00.1 section 2.3, Change of base
    assert_close(log_base(32.0, 8.0), 5.0 / 3.0, dtype="float64")
    assert_close(log_base(1000.0, 10.0), 3.0, dtype="float64")
    assert_close(log_base(8.0, 0.5), -3.0, dtype="float64")
    assert_close(log_base(math.e, math.e), 1.0, dtype="float64")


def test_log_base_golden():
    # WHY: 66 (x, b) pairs from 1e-300 to 1e300 and bases 0.5 to 256,
    #      against mpmath. Computing in float32 anywhere loses 9 digits.
    # KIND: golden
    # CATCHES: s02, s03, s09, s10, m02
    # CHAPTER: M00.1 section 2.3, Change of base
    for case in golden()["log_base"]:
        got = log_base(case["x"], case["b"])
        assert_close(
            got,
            case["log"],
            dtype="float64",
            msg=f"log_{case['base_name']}({case['x']})",
        )


def test_log_laws():
    # WHY: the laws of section 2.2 hold for your function on random inputs:
    #      log_b(xy) = log_b x + log_b y, log_b(x^k) = k log_b x, and
    #      log_b(x) = -log_{1/b}(x).
    # KIND: property
    # CATCHES: s02, s09, s10, m02
    # CHAPTER: M00.1 section 2.2, Laws of exponents and logarithms
    rng = PCG32(seed=seed())
    for _ in range(100):
        x = math.exp(rng.uniform() * 40 - 20)
        y = math.exp(rng.uniform() * 40 - 20)
        b = 1.5 + rng.uniform() * 30
        k = rng.below(7) - 3
        lx, ly = float(log_base(x, b)), float(log_base(y, b))
        assert_close(log_base(x * y, b), lx + ly, rtol=1e-12, atol=1e-12)
        assert_close(log_base(x**k, b), k * lx, rtol=1e-12, atol=1e-12)
        assert_close(log_base(x, 1.0 / b), -lx, dtype="float64")


def test_log_base_keeps_shape_and_dtype():
    # WHY: L1.6 and M11.2 pass whole arrays of probabilities; the result has
    #      the input's shape in float64, and a Python scalar gives a float64 scalar.
    # KIND: boundary
    # CATCHES: s02, s03, s09, s10, m02
    # CHAPTER: M00.1 section 4, The interface
    x = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    got = log_base(x, 2.0)
    assert got.shape == (3, 4) and got.dtype == np.float64
    assert_close(got[1, 3], 3.0, dtype="float64")  # log2(8)
    s = log_base(4, 2)
    assert np.shape(s) == () and np.asarray(s).dtype == np.float64
    assert_close(s, 2.0, dtype="float64")


def test_log_of_zero_is_infinite():
    # WHY: probability 0 is infinitely surprising: log_b(0) = -inf for b > 1
    #      (+inf for b < 1). That is a value, not an error, and it must not
    #      warn (numpy's divide-by-zero warning would fail a strict test run).
    # KIND: boundary
    # CATCHES: s02, s11, m02
    # CHAPTER: M00.1 section 5, Pitfalls, item 4
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = log_base([0.0, 1.0], 2.0)
        assert got[0] == -math.inf and got[1] == 0.0
        assert log_base(0.0, 0.5) == math.inf


@pytest.mark.parametrize("b", [1.0, 0.0, -2.0, math.inf, math.nan])
def test_bad_base_rejected(b):
    # WHY: ln 1 = 0, so base 1 divides by zero and returns inf or nan
    #      silently; bases <= 0 have no real logarithm. All are ValueError.
    # KIND: boundary
    # CATCHES: s05, m01, m02
    # CHAPTER: M00.1 section 5, Pitfalls, item 3
    with pytest.raises(ValueError):
        log_base(2.0, b)


def test_negative_or_nan_x_rejected():
    # WHY: log of a negative number is not real; numpy would return nan with a
    #      warning, and a nan surprisal poisons every later sum.
    # KIND: boundary
    # CATCHES: s06
    # CHAPTER: M00.1 section 4, The interface
    with pytest.raises(ValueError):
        log_base([1.0, -0.5], 2.0)
    with pytest.raises(ValueError):
        log_base(math.nan, 2.0)


# --- bits per byte -----------------------------------------------------------------------


def test_bits_per_byte_golden():
    # WHY: four (nll, bytes) pairs against mpmath, including the worked
    #      example and a uniform byte model.
    # KIND: golden
    # CATCHES: s04, s12
    # CHAPTER: M00.1 section 2.4, Units of information
    for case in golden()["bits_per_byte"]:
        got = bits_per_byte(case["nll_nats_sum"], case["n_bytes"])
        assert_close(got, case["bpb"], dtype="float64", msg=case["expr"])


def test_bits_per_byte_is_per_byte():
    # WHY: doubling the text with the same per-byte cost leaves bpb unchanged;
    #      doubling only the cost doubles it. A function that ignores n_bytes,
    #      or divides by it twice, breaks one of the two.
    # KIND: property
    # CATCHES: s12
    # CHAPTER: M00.1 section 2.4, Units of information
    rng = PCG32(seed=seed())
    for _ in range(50):
        n = 1 + rng.below(5000)
        nll = rng.uniform() * 8 * n
        one = bits_per_byte(nll, n)
        assert_close(bits_per_byte(2 * nll, 2 * n), one, dtype="float64")
        assert_close(bits_per_byte(2 * nll, n), 2 * one, dtype="float64")


@pytest.mark.parametrize(
    "nll,n", [(1.0, 0), (1.0, -3), (1.0, 2.5), (-0.1, 10), (math.nan, 10)]
)
def test_bits_per_byte_rejects_bad_input(nll, n):
    # WHY: an empty text has no bits per byte, a fractional byte count is a
    #      bug upstream, and a negative or NaN summed NLL cannot come from
    #      probabilities. ValueError, not ZeroDivisionError or a silent nan.
    # KIND: boundary
    # CATCHES: s07, m03
    # CHAPTER: M00.1 section 4, The interface
    with pytest.raises(ValueError):
        bits_per_byte(nll, n)
