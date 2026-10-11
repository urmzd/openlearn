"""Course tests for M03.6: normalize, cosine_sim, project, and topk_cosine
(tinyllm/linalg/inner.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M03.6), and the chapter section it comes from.

The chapter's worked example (section 3) is a = (3, 4), b = (4, 3):
a . b = 24, cos = 24/25 = 0.96, the projection of a onto b is
(3.84, 2.88), and the residual (-0.84, 1.12) is orthogonal to b. Against
the rows (4, 3), (6, 8), (-3, -4), (10, 0), (0, 2), the top 3 by cosine
are rows 1, 0, 4 with scores 1, 0.96, 0.8; by raw dot product they would
be rows 1, 3, 0. numpy formulas written out in the test are the oracle for
random inputs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.linalg.inner import cosine_sim, normalize, project, topk_cosine

ROWS = np.array([[4.0, 3.0], [6.0, 8.0], [-3.0, -4.0], [10.0, 0.0], [0.0, 2.0]])


def rng(seq: int) -> PCG32:
    return PCG32(int(os.environ.get("OL_SEED", "0")), seq=seq)


def test_hand_example_cosine_and_projection():
    # WHY: the chapter's worked example. cos = 24 / (5 * 5) = 0.96; the
    #      projection coefficient is (a . b) / (b . b) = 24/25, so
    #      proj = 0.96 * (4, 3) = (3.84, 2.88), and the residual
    #      (-0.84, 1.12) has zero dot product with b.
    # KIND: unit, smoke
    # CATCHES: s04
    # CHAPTER: M03.6 section 3
    a, b = np.array([3.0, 4.0]), np.array([4.0, 3.0])
    assert_close(cosine_sim(a, b), 0.96, rtol=0.0, atol=1e-15)
    p = project(a, b)
    assert_close(p, [3.84, 2.88], rtol=0.0, atol=1e-14)
    assert_close((a - p) @ b, 0.0, rtol=0.0, atol=1e-14)
    assert_close(normalize(a), [0.6, 0.8], rtol=0.0, atol=1e-15)


def test_hand_example_topk():
    # WHY: the chapter's retrieval example. Cosine ranks rows 1, 0, 4
    #      (scores 1, 0.96, 0.8); the long row (10, 0) has the second
    #      largest dot product (30) but cosine 0.6. Ranking by dot product
    #      lets long vectors win, which is the bug this test exists for.
    # KIND: unit, smoke
    # CATCHES: s01, s07
    # CHAPTER: M03.6 section 3
    idx, scores = topk_cosine(np.array([3.0, 4.0]), ROWS, 3)
    assert idx.dtype == np.int64
    assert idx.tolist() == [1, 0, 4]
    assert_close(scores, [1.0, 0.96, 0.8], rtol=0.0, atol=1e-15)


def test_cosine_matches_formula_and_broadcasts():
    # WHY: a batch of 7 vectors against one vector (broadcast), and the same
    #      along axis 0, against (a . b) / (||a|| ||b||) written out with
    #      numpy. L6.7 scores whole vocabularies at once this way.
    # KIND: differential
    # CATCHES: s08, m01
    # CHAPTER: M03.6 section 2.2
    g = rng(61)
    A = g.uniform_array((7, 5), -1.0, 1.0)
    b = g.uniform_array((5,), -1.0, 1.0)
    want = (A @ b) / (np.sqrt((A * A).sum(1)) * np.sqrt(b @ b))
    got = cosine_sim(A, b)
    assert got.shape == (7,)
    assert_close(got, want, rtol=1e-14, atol=1e-15)
    got0 = cosine_sim(A.T, b[:, None], axis=0)
    assert got0.shape == (7,)
    assert_close(got0, want, rtol=1e-14, atol=1e-15)


def test_cosine_properties():
    # WHY: cosine depends only on direction: scaling either vector by a
    #      positive number leaves it unchanged, a negative one flips its
    #      sign, it is symmetric, and it lies in [-1, 1] (Cauchy-Schwarz).
    # KIND: property
    # CATCHES: m01
    # CHAPTER: M03.6 section 2.2
    g = rng(62)
    A = g.uniform_array((50, 8), -1.0, 1.0)
    B = g.uniform_array((50, 8), -1.0, 1.0)
    c = cosine_sim(A, B)
    assert_close(cosine_sim(B, A), c, rtol=0.0, atol=1e-15)
    assert_close(cosine_sim(3.5 * A, 0.25 * B), c, rtol=1e-14, atol=1e-15)
    assert_close(cosine_sim(-A, B), -c, rtol=0.0, atol=1e-15)
    assert (np.abs(c) <= 1.0 + 1e-15).all()
    assert_close(cosine_sim(A, A), np.ones(50), rtol=0.0, atol=1e-15)


def test_zero_and_tiny_vectors():
    # WHY: a zero vector has no direction: cosine is 0 (each norm is
    #      clamped to eps), never NaN, and normalize leaves it at zero. Two
    #      tiny vectors (norm 1e-5 each) still have a well-defined angle:
    #      clamping each norm separately gives cos = 1 for parallel ones,
    #      while clamping the product of the norms (1e-10 < 1e-8) gives 0.01.
    # KIND: boundary
    # CATCHES: s03, s05
    # CHAPTER: M03.6 section 2.3
    z = np.zeros(4)
    v = np.array([1.0, 2.0, 3.0, 4.0])
    assert cosine_sim(z, v) == 0.0
    assert (normalize(z) == 0.0).all()
    t = 1e-5 * np.array([0.6, 0.8])
    assert_close(cosine_sim(t, t), 1.0, rtol=0.0, atol=1e-12)


def test_normalize_gives_unit_vectors():
    # WHY: unit rows turn cosine similarity into a plain dot product, which
    #      is how a vector index stores embeddings (ag.07): normalize once,
    #      then search with matrix products.
    # KIND: property
    # CATCHES: m02
    # CHAPTER: M03.6 section 2.2
    X = rng(63).uniform_array((20, 6), -3.0, 3.0)
    N = normalize(X)
    assert_close((N * N).sum(1), np.ones(20), rtol=0.0, atol=1e-14)
    assert_close(N * np.sqrt((X * X).sum(1, keepdims=True)), X, rtol=0.0, atol=1e-14)
    assert_close(normalize(X.T, axis=0), N.T, rtol=0.0, atol=1e-15)


def test_projection_properties():
    # WHY: projecting twice changes nothing (P^2 = P), the residual is
    #      orthogonal to v, and the projection does not depend on v's
    #      length. Dividing by ||v|| instead of v . v only works for unit v.
    # KIND: property
    # CATCHES: s04
    # CHAPTER: M03.6 section 2.4
    g = rng(64)
    X = g.uniform_array((10, 5), -1.0, 1.0)
    v = g.uniform_array((5,), -1.0, 1.0)
    P = project(X, v)
    assert_close(project(P, v), P, rtol=1e-14, atol=1e-15)
    assert_close((X - P) @ v, np.zeros(10), rtol=0.0, atol=1e-14)
    assert_close(project(X, 7.0 * v), P, rtol=1e-14, atol=1e-15)


def test_projection_rejects_zero_direction():
    # WHY: the line spanned by the zero vector is a point; (x . 0) / (0 . 0)
    #      is 0/0. Raise instead of returning NaN; mismatched lengths raise
    #      too instead of broadcasting into nonsense.
    # KIND: boundary
    # CATCHES: s06
    # CHAPTER: M03.6 section 4
    with pytest.raises(ValueError):
        project(np.ones(3), np.zeros(3))
    with pytest.raises(ValueError):
        project(np.ones(3), np.ones(4))
    with pytest.raises(ValueError):
        cosine_sim(np.ones(3), np.ones(4))


def test_topk_matches_brute_force():
    # WHY: a batch of 6 queries against 200 random rows: the indices and
    #      scores equal a full sort of the brute-force cosine matrix, which
    #      is what retrieval@k in L6.7 and ag.07 assumes ("exact" top-k).
    # KIND: differential
    # CATCHES: s01, s07, m03
    # CHAPTER: M03.6 section 2.5
    g = rng(65)
    M = g.uniform_array((200, 16), -1.0, 1.0) * g.uniform_array((200, 1), 0.1, 10.0)
    Q = g.uniform_array((6, 16), -1.0, 1.0)
    full = (Q @ M.T) / (
        np.sqrt((Q * Q).sum(1))[:, None] * np.sqrt((M * M).sum(1))[None, :]
    )
    idx, scores = topk_cosine(Q, M, 10)
    assert idx.shape == (6, 10) and scores.shape == (6, 10)
    want = np.argsort(-full, axis=1, kind="stable")[:, :10]
    assert (idx == want).all()
    assert_close(scores, np.take_along_axis(full, want, axis=1), rtol=1e-13, atol=1e-15)


def test_topk_ties_go_to_lowest_index():
    # WHY: duplicated or rescaled rows score exactly the same. Ties go to
    #      the lowest row index (D11), so two runs, and the Go port in
    #      ag.07, return the same list. Sorting ascending and reversing
    #      puts the highest index first instead.
    # KIND: boundary
    # CATCHES: s02
    # CHAPTER: M03.6 section 2.5
    M = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 0.0], [0.0, 3.0], [1.0, 0.0]])
    idx, scores = topk_cosine(np.array([1.0, 0.0]), M, 4)
    assert idx.tolist() == [1, 2, 4, 0]
    assert_close(scores, [1.0, 1.0, 1.0, 0.0], rtol=0.0, atol=0.0)


def test_topk_rejects_bad_k():
    # WHY: k = 0 returns nothing useful and k > n cannot be filled; both are
    #      caller bugs. The query length must match the rows.
    # KIND: boundary
    # CATCHES: m04
    # CHAPTER: M03.6 section 4
    for bad in (0, 6):
        with pytest.raises(ValueError):
            topk_cosine(np.ones(2), ROWS, bad)
    with pytest.raises(ValueError):
        topk_cosine(np.ones(3), ROWS, 2)
    idx, _ = topk_cosine(np.ones(2), ROWS, 5)
    assert sorted(idx.tolist()) == [0, 1, 2, 3, 4]
