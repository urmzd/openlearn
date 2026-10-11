"""Course tests for M11.4: mutual_information, pmi_matrix, and ppmi
(tinyllm/info/pmi.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M11.4), and the chapter section it comes from.

The chapter's worked example (section 3) is the table
    cooc = [[2, 0, 2],      cat:  pet, bark, meow
            [2, 4, 0]]      dog
with D = 10: PMI(cat, meow) = ln 2.5, PMI(dog, pet) = ln(5/6) < 0 (PPMI 0),
unseen pairs -inf, and I(X; Y) = 0.3957527947852783 nats. The golden
values in $TINYLLM_FIXTURES/M11.4/pmi_golden.json are computed from the
definitions with mpmath at 50 digits by course/oracle/M11.4/pmi_golden.py.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.info.pmi import mutual_information, pmi_matrix, ppmi

HAND = np.array([[2.0, 0.0, 2.0], [2.0, 4.0, 0.0]])
GOLDEN = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "M11.4" / "pmi_golden.json"


def rng(seq: int) -> PCG32:
    return PCG32(int(os.environ.get("OL_SEED", "0")), seq=seq)


def golden() -> dict:
    return json.loads(GOLDEN.read_text())["cases"]


def as_array(rows) -> np.ndarray:
    return np.array([[float(x) for x in r] for r in rows])


def counts(seq: int, shape, top: int) -> np.ndarray:
    g = rng(seq)
    return np.array(
        [[g.below(top) for _ in range(shape[1])] for _ in range(shape[0])],
        dtype=np.float64,
    )


def test_hand_example_pmi():
    # WHY: the chapter's worked example with alpha = 1 (plain PMI).
    #      P(cat, meow) = 0.2, P(cat) = 0.4, P(meow) = 0.2: ln(0.2 / 0.08)
    #      = ln 2.5. dog and pet co-occur less than chance: ln(5/6) < 0,
    #      which PPMI turns into 0. Unseen pairs are -inf, and 0 in PPMI.
    # KIND: unit, smoke
    # CATCHES: s01, m03
    # CHAPTER: M11.4 section 3
    P = pmi_matrix(HAND, cds_alpha=1.0)
    want = np.array(
        [
            [math.log(1.25), -math.inf, math.log(2.5)],
            [math.log(5 / 6), math.log(5 / 3), -math.inf],
        ]
    )
    assert_close(P, want, rtol=1e-14, atol=1e-15)
    assert_close(
        ppmi(HAND, cds_alpha=1.0), np.maximum(want, 0.0), rtol=1e-14, atol=1e-15
    )


def test_hand_example_mutual_information():
    # WHY: I(X; Y) is the expected PMI under the joint distribution:
    #      0.2 ln 1.25 + 0.2 ln 2.5 + 0.2 ln(5/6) + 0.4 ln(5/3)
    #      = 0.3957527947852783 nats. Counts are normalized first, so the
    #      table and the table / 10 give the same answer.
    # KIND: unit, smoke
    # CATCHES: s04, s05
    # CHAPTER: M11.4 section 3
    assert_close(mutual_information(HAND), 0.3957527947852783, rtol=1e-14, atol=0.0)
    assert_close(
        mutual_information(HAND / 10.0), 0.3957527947852783, rtol=1e-14, atol=0.0
    )


@pytest.mark.parametrize("alpha", ["1", "0.75", "0.5"])
def test_golden_tables(alpha):
    # WHY: random, fractional, independent, and one-row tables against
    #      mpmath values from the definitions. alpha = 0.75 is word2vec's
    #      smoothing; the smoothed context probabilities must be
    #      renormalized over contexts, or every PMI shifts by a constant.
    # KIND: golden
    # CATCHES: s01, s02, s05, m03
    # CHAPTER: M11.4 section 2.3
    for name, case in golden().items():
        table = np.array(case["table"])
        want = as_array(case[f"pmi_alpha_{alpha}"])
        got = pmi_matrix(table, cds_alpha=float(alpha))
        assert_close(got, want, rtol=1e-12, atol=1e-13, msg=f"{name} alpha {alpha}")
        assert_close(
            ppmi(table, cds_alpha=float(alpha)),
            np.maximum(want, 0.0),
            rtol=1e-12,
            atol=1e-13,
            msg=name,
        )
        assert_close(
            mutual_information(table),
            float(case["mutual_information"]),
            rtol=1e-12,
            atol=1e-14,
            msg=name,
        )


def test_mutual_information_properties():
    # WHY: an outer product (independent X and Y) has I = 0; transposing
    #      the table swaps X and Y and leaves I unchanged; and
    #      0 <= I <= min(H(X), H(Y)), because knowing X can save at most
    #      all of Y's uncertainty.
    # KIND: property
    # CATCHES: s05
    # CHAPTER: M11.4 section 2.1
    g = rng(71)
    a, b = g.uniform_array((5,), 0.1, 1.0), g.uniform_array((7,), 0.1, 1.0)
    assert_close(mutual_information(np.outer(a, b)), 0.0, rtol=0.0, atol=1e-15)
    for seq in range(72, 77):
        T = counts(seq, (6, 9), 8)
        P = T / T.sum()
        px, py = P.sum(1), P.sum(0)
        hx = -float(sum(p * math.log(p) for p in px if p > 0))
        hy = -float(sum(p * math.log(p) for p in py if p > 0))
        mi = mutual_information(T)
        assert -1e-15 <= mi <= min(hx, hy) + 1e-12
        assert_close(mutual_information(T.T), mi, rtol=1e-13, atol=1e-15)


def test_mutual_information_is_expected_pmi():
    # WHY: I(X; Y) = sum over pairs of P(w, c) PMI(w, c) with plain PMI
    #      (alpha = 1); unseen pairs have weight 0 and contribute nothing.
    #      L2.3's PPMI matrix is the pointwise version of this number.
    # KIND: property
    # CATCHES: s01, s04
    # CHAPTER: M11.4 section 2.2
    T = counts(78, (8, 6), 5)
    P = T / T.sum()
    pm = pmi_matrix(T, cds_alpha=1.0)
    expected = float(np.sum(P[T > 0] * pm[T > 0]))
    assert_close(mutual_information(T), expected, rtol=1e-12, atol=1e-15)


def test_cds_lowers_rare_context_pmi():
    # WHY: raising context counts to alpha < 1 moves probability toward rare
    #      contexts, so a pair with a rare context loses PMI and a pair with
    #      the most frequent context gains it. In the hand table meow is
    #      rarest: ln 2.5 = 0.916 at alpha = 1 drops to 0.780 at 0.75.
    # KIND: property
    # CATCHES: s01, s02
    # CHAPTER: M11.4 section 2.3
    p1 = pmi_matrix(HAND, cds_alpha=1.0)
    p75 = pmi_matrix(HAND, cds_alpha=0.75)
    assert p75[0, 2] < p1[0, 2]
    assert p75[0, 0] > p1[0, 0]
    assert_close(p75[0, 2], 0.7801469381313082, rtol=1e-13, atol=0.0)


def test_ppmi_is_finite_nonnegative_and_quiet():
    # WHY: PPMI is the matrix L2.3 hands to an SVD, which cannot take -inf
    #      or NaN. A table with an all-zero row and an all-zero column
    #      (a word never seen, a context never used) must give exact zeros
    #      there, with no divide-by-zero warnings leaking out.
    # KIND: boundary
    # CATCHES: s03, s06
    # CHAPTER: M11.4 section 2.4
    T = counts(79, (7, 5), 4)
    T[2, :] = 0.0
    T[:, 3] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        M = ppmi(T)
        P = pmi_matrix(T)
    assert np.isfinite(M).all() and (M >= 0.0).all()
    assert (M[2, :] == 0.0).all() and (M[:, 3] == 0.0).all()
    assert np.isneginf(P[T == 0]).all()
    assert np.isfinite(P[T > 0]).all()


def test_rejects_bad_tables_and_alpha():
    # WHY: a negative count, a 1-D vector, or an empty (all-zero) table has
    #      no probabilities; alpha <= 0 would give every context, even
    #      unused ones (0^0 = 1), the same weight. All are caller bugs.
    # KIND: boundary
    # CATCHES: s07, m01, m02
    # CHAPTER: M11.4 section 4
    for bad in (
        np.array([[1.0, -0.5], [2.0, 1.0]]),
        np.ones(3),
        np.zeros((2, 2)),
        np.array([[1.0, math.nan]]),
    ):
        with pytest.raises(ValueError):
            pmi_matrix(bad)
        with pytest.raises(ValueError):
            mutual_information(bad)
    for alpha in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            pmi_matrix(HAND, cds_alpha=alpha)
    with pytest.raises(ValueError):
        ppmi(HAND, cds_alpha=0.0)


def test_integer_counts_and_inputs_untouched():
    # WHY: co-occurrence counts arrive as int64 arrays from a counter;
    #      results are float64 and the caller's table is unchanged (L2.3
    #      reuses it for the SVD baseline and the word2vec comparison).
    # KIND: boundary
    # CATCHES: s08
    # CHAPTER: M11.4 section 4
    F = counts(80, (5, 5), 6) + 1.0
    before = F.copy()
    mutual_information(F)
    ppmi(F)
    assert (F == before).all()
    T = F.astype(np.int64)
    out = pmi_matrix(T)
    assert out.dtype == np.float64
    assert_close(out, pmi_matrix(F), rtol=0.0, atol=0.0)
    assert_close(mutual_information(T), mutual_information(F), rtol=0.0, atol=0.0)
