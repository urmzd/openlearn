"""Course tests for M00.3: sequences, geometric series, frequency ladders
(tinyllm/num/series.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M00.3), and the chapter section it comes from.

The chapter's worked example (section 3): 1 + 1/2 + 1/4 + 1/8 = 15/8; the
RoPE ladder for d_rot = 8 and base 10000 is [1, 0.1, 0.01, 0.001]; ALiBi with
6 heads is [1/4, 1/16, 1/64, 1/256, 1/2, 1/8].
"""

from __future__ import annotations

import json
import math
import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.num.series import alibi_slopes, geometric, geometric_sum, rope_inv_freq

GOLDEN = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "M00.3" / "ladders.json"


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def exact_sum(a: float, r: float, n: int) -> float:
    """sum_{j<n} a r^j in exact rational arithmetic on the float64 inputs,
    rounded once at the end: the oracle for geometric_sum."""
    fa, fr = Fraction(a), Fraction(r)
    total, term = Fraction(0), fa
    for _ in range(n):
        total += term
        term *= fr
    return float(total)


# --- the worked example ----------------------------------------------------------


def test_hand_example_geometric_sum():
    # WHY: the chapter's worked example: 1 + 1/2 + 1/4 + 1/8 = 15/8, which the
    #      closed form gives as (1 - (1/2)^4) / (1 - 1/2) = (15/16) / (1/2).
    # KIND: unit
    # CATCHES: s01, s02
    # CHAPTER: M00.3 section 3, Worked example by hand
    assert_close(geometric(1.0, 0.5, 4), [1.0, 0.5, 0.25, 0.125], dtype="float64")
    assert_close(geometric_sum(1.0, 0.5, 4), 15.0 / 8.0, dtype="float64")
    assert_close(geometric_sum(3.0, 0.5, 4), 45.0 / 8.0, dtype="float64")


def test_hand_example_rope_ladder():
    # WHY: d_rot = 8, base 10000: exponents 0, -2/8, -4/8, -6/8 give
    #      10000^0, 10000^-0.25, 10000^-0.5, 10000^-0.75 = 1, 0.1, 0.01, 0.001,
    #      a geometric ladder with ratio 0.1.
    # KIND: unit
    # CATCHES: s04, s05, s06, s07
    # CHAPTER: M00.3 section 3, Worked example by hand
    got = rope_inv_freq(8, 10000.0)
    assert got.shape == (4,) and got.dtype == np.float64
    assert_close(got, [1.0, 0.1, 0.01, 0.001], dtype="float64")


def test_hand_example_alibi_six_heads():
    # WHY: 6 is not a power of two. p = 4 gives 2^-2, 2^-4, 2^-6, 2^-8; the
    #      two extra heads take slopes 0 and 2 of the 8-head ladder, 2^-1 and
    #      2^-3, which fall between the first ones.
    # KIND: unit
    # CATCHES: s01, s08, s09, s10
    # CHAPTER: M00.3 section 3, Worked example by hand
    got = alibi_slopes(6)
    assert_close(got, [1 / 4, 1 / 16, 1 / 64, 1 / 256, 1 / 2, 1 / 8], dtype="float64")


# --- sequences and sums ----------------------------------------------------------------


def test_geometric_terms():
    # WHY: term j is a r^j, starting at j = 0, for any sign of r; n = 0 is
    #      the empty sequence (no terms), not an error.
    # KIND: unit
    # CATCHES: s01
    # CHAPTER: M00.3 section 2.1, Sequences
    assert_close(geometric(2.0, -3.0, 4), [2.0, -6.0, 18.0, -54.0], dtype="float64")
    assert_close(geometric(5.0, 1.0, 3), [5.0, 5.0, 5.0], dtype="float64")
    e = geometric(1.0, 0.5, 0)
    assert e.shape == (0,) and e.dtype == np.float64


def test_sum_matches_exact_rationals():
    # WHY: against exact rational arithmetic on the same float64 inputs
    #      (fractions.Fraction), for ratios below, above, and on both sides
    #      of 1, including negative ratios and a = 0.
    # KIND: differential
    # CATCHES: s02
    # CHAPTER: M00.3 section 2.2, The geometric sum
    rng = PCG32(seed=seed())
    cases = [
        (1.0, 1.0, 7),
        (2.5, 1.0, 10),
        (0.0, 0.3, 5),
        (1.0, -1.0, 7),
        (1.0, -1.0, 8),
        (1.0, 0.0, 4),
    ]
    for _ in range(60):
        r = rng.uniform() * 4.0 - 2.0
        cases.append((rng.uniform() * 10 - 5, r, rng.below(40)))
    for a, r, n in cases:
        assert_close(
            geometric_sum(a, r, n),
            exact_sum(a, r, n),
            rtol=1e-11,
            atol=1e-12,
            msg=f"a={a}, r={r}, n={n}",
        )


def test_sum_near_one_keeps_its_digits():
    # WHY: with r = 1 - 2^-40 the textbook (1 - r^n) / (1 - r) subtracts two
    #      numbers that agree in their first 12 digits and keeps only about 7
    #      correct ones (pitfall 1). Adam's bias correction 1 - beta^t
    #      (M02.2) lives exactly here.
    # KIND: boundary
    # CATCHES: s02, s03
    # CHAPTER: M00.3 section 5, Pitfalls, item 1
    for r in (1.0 - 2.0**-40, 1.0 + 2.0**-40, 1.0 - 1e-9, 0.999999):
        for n in (2, 10, 1000):
            assert_close(
                geometric_sum(1.0, r, n),
                exact_sum(1.0, r, n),
                rtol=1e-12,
                atol=0.0,
                msg=f"r={r!r}, n={n}",
            )


def test_ema_bias_correction_identity():
    # WHY: an EMA with decay beta weights the last t gradients by
    #      (1 - beta) beta^j, and their total is 1 - beta^t: the number Adam
    #      divides by (M02.2, M10.3).
    # KIND: property
    # CATCHES: s02
    # CHAPTER: M00.3 section 2.3, The EMA as a geometric series
    for beta in (0.9, 0.99, 0.999):
        for t in (1, 2, 10, 100, 5000):
            want = -math.expm1(t * math.log(beta))  # 1 - beta^t without cancellation
            assert_close(geometric_sum(1 - beta, beta, t), want, rtol=1e-12, atol=0.0)


def test_sum_overflows_to_infinity():
    # WHY: 1 + 1e10 + ... + 1e10^99 does not fit in a float64. The answer is
    #      +inf (or -inf by the sign of the last term), not an OverflowError.
    # KIND: boundary
    # CATCHES: s12
    # CHAPTER: M00.3 section 4, The interface
    assert geometric_sum(1.0, 1e10, 100) == math.inf
    assert geometric_sum(1.0, -1e10, 100) == -math.inf
    assert geometric_sum(1.0, -1e10, 101) == math.inf


@pytest.mark.parametrize("n", [-1, 2.5, True])
def test_bad_n_rejected(n):
    # WHY: a count of terms is a non-negative integer; True would quietly be 1.
    # KIND: boundary
    # CATCHES: s11
    # CHAPTER: M00.3 section 4, The interface
    with pytest.raises(ValueError):
        geometric(1.0, 0.5, n)
    with pytest.raises(ValueError):
        geometric_sum(1.0, 0.5, n)


# --- RoPE ladder ----------------------------------------------------------------------------


def test_rope_inv_freq_golden_hf():
    # WHY: Hugging Face's default RoPE initialisation (torch, float32) for
    #      SmolLM2-135M, Llama 2, Llama 3, and a partial rotary dimension:
    #      exactly what L7.9 loads and L7.3 must reproduce.
    # KIND: golden
    # CATCHES: s04, s05, s06, s07
    # CHAPTER: M00.3 section 2.4, Frequency ladders: RoPE
    for case in json.loads(GOLDEN.read_text())["rope_inv_freq"]:
        got = rope_inv_freq(case["d_rot"], case["base"])
        assert_close(got, np.array(case["inv_freq"]), dtype="float32", msg=case["name"])


def test_rope_ladder_is_geometric():
    # WHY: consecutive frequencies have one ratio, base^(-2/d): the first is
    #      1 (one radian per position) and the last is just above 1/base.
    # KIND: property
    # CATCHES: s04, s05, s06, s07
    # CHAPTER: M00.3 section 2.4, Frequency ladders: RoPE
    for d, base in [(64, 1e5), (128, 5e5), (6, 7.5)]:
        f = rope_inv_freq(d, base)
        assert f.shape == (d // 2,)
        assert_close(f[0], 1.0, dtype="float64")
        assert_close(
            f[1:] / f[:-1], np.full(d // 2 - 1, base ** (-2 / d)), dtype="float64"
        )
        assert 1.0 / base < f[-1] < 1.0


@pytest.mark.parametrize(
    "d,base", [(0, 1e4), (7, 1e4), (8, 1.0), (8, 0.5), (8, math.inf)]
)
def test_rope_bad_arguments(d, base):
    # WHY: an odd d_rot has a dimension with no partner; a base <= 1 makes the
    #      ladder flat or climb instead of fall.
    # KIND: boundary
    # CATCHES: s13
    # CHAPTER: M00.3 section 4, The interface
    with pytest.raises(ValueError):
        rope_inv_freq(d, base)


# --- ALiBi ladder ----------------------------------------------------------------------------


def test_alibi_powers_of_two_exact():
    # WHY: for 8 heads the ladder is 1/2, 1/4, ..., 1/256 exactly (start and
    #      ratio 2^(-8/8)); for 16 it is 2^-0.5, 2^-1, ..., 2^-8. The steepest
    #      head looks only at the last few tokens, the flattest at hundreds.
    # KIND: unit
    # CATCHES: s01
    # CHAPTER: M00.3 section 2.5, Frequency ladders: ALiBi
    assert_close(alibi_slopes(8), [2.0**-j for j in range(1, 9)], dtype="float64")
    assert_close(
        alibi_slopes(16), [2.0 ** (-j / 2) for j in range(1, 17)], dtype="float64"
    )
    assert_close(alibi_slopes(1), [2.0**-8], dtype="float64")


def test_alibi_golden_hf():
    # WHY: transformers' BLOOM build_alibi_tensor (float32) for 20 head
    #      counts from 1 to 128, including 3, 5, 6, 7, 9, 12, 20, 24, 40, 48,
    #      80, and 96, which are not powers of two.
    # KIND: golden
    # CATCHES: s01, s08, s09, s10
    # CHAPTER: M00.3 section 2.5, Frequency ladders: ALiBi
    for case in json.loads(GOLDEN.read_text())["alibi_slopes"]:
        got = alibi_slopes(case["n_heads"])
        assert got.dtype == np.float64
        assert_close(
            got,
            np.array(case["slopes"]),
            dtype="float32",
            msg=f"n_heads={case['n_heads']}",
        )


@pytest.mark.parametrize("n", [0, -4, 2.5])
def test_alibi_bad_head_count(n):
    # WHY: a model has at least one head, and a whole number of them.
    # KIND: boundary
    # CATCHES: s14
    # CHAPTER: M00.3 section 4, The interface
    with pytest.raises(ValueError):
        alibi_slopes(n)
