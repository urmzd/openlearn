"""Course tests for M09.3: error analysis, condition numbers, tolerance budgets
(tinyllm/num/tolerance.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M09.3), and the chapter section it comes from.

"Exact" values are computed with float64 products of float32 inputs (exact:
24 + 24 significand bits fit in 53) summed by math.fsum (correctly rounded),
so the observed error of a float32 computation is measured to about 1e-16
relative. Random inputs come from the frozen PCG32 at OL_SEED.

These are the only course tests that import your tolerance helpers (D35:
they decide this module's verdict and nothing else). They import the module
as `from tinyllm.num import tolerance`; every other course test asserts
through the frozen tests/_lib/close.py.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.num.diff import central_diff
from tinyllm.num import tolerance as tol

assert_close_bounded = tol.assert_close_bounded
bound_ratio = tol.bound_ratio
cond = tol.cond
dot_error_bound = tol.dot_error_bound
fd_error_model = tol.fd_error_model
gamma = tol.gamma
matmul_error_bound = tol.matmul_error_bound
optimal_fd_step = tol.optimal_fd_step
relative_condition = tol.relative_condition
sum_error_bound = tol.sum_error_bound
unit_roundoff = tol.unit_roundoff

U32 = 2.0**-24
U64 = 2.0**-53


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def f32_dot(x: np.ndarray, y: np.ndarray) -> float:
    """Recursive float32 dot product: each product and each running sum
    rounded to float32, left to right."""
    p = (x.astype(np.float32) * y.astype(np.float32)).astype(np.float32)
    return float(np.cumsum(p, dtype=np.float32)[-1])


def exact_dot(x: np.ndarray, y: np.ndarray) -> float:
    return math.fsum((x.astype(np.float64) * y.astype(np.float64)).tolist())


# --- the worked example -------------------------------------------------------


def test_hand_example():
    # WHY: section 3 by hand: x = [1, 1e-8, 1e-8] dotted with ones in float32
    #      returns 1 (each 1e-8 is below half a gap at 1 and is rounded
    #      away), the exact answer is 1.00000002, and the error 2e-8 sits
    #      inside gamma_3 |x|.|y| = 3u/(1 - 3u) * 1.00000002 = 1.788e-7.
    # KIND: unit, smoke
    # CATCHES: s01, s02, s04
    # CHAPTER: M09.3 section 3
    x = np.array([1.0, 1e-8, 1e-8], dtype=np.float32)
    y = np.ones(3, dtype=np.float32)
    assert f32_dot(x, y) == 1.0
    exact = exact_dot(x, y)
    assert_close(exact, 1.00000002, rtol=1e-12, atol=0)
    assert unit_roundoff("f32") == U32
    assert gamma(3, "f32") == 3 * U32 / (1 - 3 * U32)
    bound = dot_error_bound(3, "f32", exact)
    assert_close(bound, 1.7881394e-7, rtol=1e-6, atol=0)
    assert abs(f32_dot(x, y) - exact) <= bound
    assert_close_bounded(np.float32(1.0), exact, 3, "f32", exact, slack=1.0)


def test_hand_example_condition():
    # WHY: the section 3 system A = [[1, 1], [1, 1.0001]]: cond(A) is about
    #      40002, and moving b = [2, 2.0001] by 1e-4 in its second entry
    #      moves the solution from [1, 1] to [0, 2], an amplification of
    #      about 28000, below the condition number as the theory says.
    # KIND: unit, smoke
    # CHAPTER: M09.3 section 3
    A = np.array([[1.0, 1.0], [1.0, 1.0001]])
    k = cond(A)
    assert_close(k, np.linalg.cond(A, 2), rtol=1e-9, atol=0)
    assert 40001.9 < k < 40002.1
    b, db = np.array([2.0, 2.0001]), np.array([0.0, 1e-4])
    x, x2 = np.linalg.solve(A, b), np.linalg.solve(A, b + db)
    assert_close(x, [1.0, 1.0], rtol=1e-9, atol=1e-9)
    assert_close(x2, [0.0, 2.0], rtol=1e-6, atol=1e-6)
    amp = (np.linalg.norm(x2 - x) / np.linalg.norm(x)) / (
        np.linalg.norm(db) / np.linalg.norm(b)
    )
    assert 27000 < amp <= k


# --- unit roundoff and gamma -------------------------------------------------------------


def test_unit_roundoff_table():
    # WHY: u = 2^-p counts the implicit bit: bf16 stores 7 fraction bits but
    #      rounds to 8 significant bits, so u = 2^-8, not 2^-7; every bound
    #      in the course scales with it. numpy names are accepted too.
    # KIND: unit
    # CATCHES: s01, s03
    # CHAPTER: M09.3 section 2.1
    want = {
        "f64": 2.0**-53,
        "f32": 2.0**-24,
        "f16": 2.0**-11,
        "bf16": 2.0**-8,
        "e4m3": 2.0**-4,
        "e5m2": 2.0**-3,
    }
    for name, u in want.items():
        assert unit_roundoff(name) == u, name
    for np_name, short in [
        ("float64", "f64"),
        ("float32", "f32"),
        ("float16", "f16"),
        ("bfloat16", "bf16"),
    ]:
        assert unit_roundoff(np_name) == want[short]
    assert unit_roundoff("f32") == np.finfo(np.float32).eps / 2
    with pytest.raises(ValueError):
        unit_roundoff("int8")


def test_gamma_edges():
    # WHY: gamma_k = k u / (1 - k u) is the price of k roundings; gamma_0 is
    #      0, and once k u >= 1 the bound is void (bf16 at k = 256), which a
    #      test must hear about instead of receiving a negative tolerance.
    # KIND: boundary
    # CATCHES: s01, s02, s03, m04
    # CHAPTER: M09.3 section 2.2
    assert gamma(0, "f32") == 0.0
    assert gamma(1, "f64") == U64 / (1 - U64)
    assert gamma(200, "bf16") == (200 / 256) / (1 - 200 / 256)
    assert gamma(255, "bf16") > 200
    for k, dt in [(256, "bf16"), (16, "e4m3"), (8, "e5m2"), (-1, "f32")]:
        with pytest.raises(ValueError):
            gamma(k, dt)
    ks = [1, 2, 10, 100, 1000]
    g = [gamma(k, "f32") for k in ks]
    assert g == sorted(g) and all(gi >= k * U32 for gi, k in zip(g, ks))


# --- sum and dot bounds ----------------------------------------------------------------------


def test_dot_bound_holds_and_is_not_loose():
    # WHY: the bound must hold on every input (1000 random float32 dot
    #      products of length 1 to 512, with mixed signs and magnitudes), or
    #      a differential test built on it fails a correct kernel; and it must
    #      not be loose by orders of magnitude, or it hides bugs: on the
    #      same-sign half (no cancellation, so |x|.|y| = |x.y|) the median of
    #      bound / observed error stays under 100.
    # KIND: property
    # CATCHES: s01, s04
    # CHAPTER: M09.3 section 2.2
    rng = PCG32(seed(), 31)
    ratios = []
    for case in range(1000):
        k = 1 + rng.below(512)
        x = (rng.normal_array((k,)) * 10.0 ** (rng.below(5) - 2)).astype(np.float32)
        y = rng.normal_array((k,)).astype(np.float32)
        if case % 2:
            x, y = np.abs(x), np.abs(y)  # same signs: errors cannot cancel
        got, exact = f32_dot(x, y), exact_dot(x, y)
        absdot = exact_dot(np.abs(x), np.abs(y))
        bound = float(dot_error_bound(k, "f32", absdot))
        err = abs(got - exact)
        assert err <= bound, f"case {case}: k={k}, error {err:.3e} > bound {bound:.3e}"
        if err > 0 and case % 2:
            ratios.append(bound / err)
    assert np.median(ratios) < 100, f"median bound/error {np.median(ratios):.1f}"


def test_sum_bound_holds():
    # WHY: recursive summation of k terms makes k - 1 roundings, so its
    #      bound is gamma_(k-1) sum|x_i| (one term alone is exact, bound 0);
    #      checked on float16 running sums, where errors are large.
    # KIND: property
    # CATCHES: s05, m01
    # CHAPTER: M09.3 section 2.2
    rng = PCG32(seed(), 32)
    for _ in range(300):
        k = 1 + rng.below(1000)
        x = rng.uniform_array((k,), 0.0, 4.0).astype(np.float16)
        got = float(np.cumsum(x, dtype=np.float16)[-1])
        exact = math.fsum(x.astype(np.float64).tolist())
        assert abs(got - exact) <= float(sum_error_bound(k, "f16", exact))
    assert float(sum_error_bound(1, "f32", 7.0)) == 0.0
    assert float(sum_error_bound(2, "f32", 7.0)) == 7.0 * gamma(1, "f32")
    with pytest.raises(ValueError):
        sum_error_bound(0, "f32", 1.0)
    with pytest.raises(ValueError):
        dot_error_bound(0, "f32", 1.0)


def test_bounds_are_elementwise_arrays():
    # WHY: a kernel test passes a whole matrix of |A| @ |B| at once; the
    #      bound must come back with the same shape, in float64.
    # KIND: unit
    # CATCHES: s04
    # CHAPTER: M09.3 section 4
    absdot = np.array([[1.0, 2.0], [0.0, 4.0]], dtype=np.float32)
    b = dot_error_bound(5, "f32", absdot)
    assert b.shape == (2, 2) and b.dtype == np.float64
    assert b.tolist() == (gamma(5, "f32") * absdot.astype(np.float64)).tolist()


def test_matmul_bound_covers_any_order():
    # WHY: BLAS sums in blocks, in an order you do not control; the
    #      componentwise bound gamma_k |A| |B| holds for every order, so it is
    #      the right tolerance for numpy's float32 matmul and for your C
    #      optional kernels (L9.1), including k = 300 with A not square.
    # KIND: property
    # CATCHES: s08, s20
    # CHAPTER: M09.3 section 2.3
    rng = PCG32(seed(), 33)
    for m, k, n in [(2, 300, 3), (7, 64, 5), (1, 1, 1), (16, 513, 2)]:
        A = rng.normal_array((m, k)).astype(np.float32)
        B = rng.normal_array((k, n)).astype(np.float32)
        got = (A @ B).astype(np.float64)
        exact = np.array(
            [[exact_dot(A[i], B[:, j]) for j in range(n)] for i in range(m)]
        )
        bound = matmul_error_bound(A, B, "f32")
        assert bound.shape == (m, n)
        assert bound_ratio(got, exact, bound) <= 1.0
    # a sequential sum of 2000 copies of float32(0.1): every addition rounds
    # the same way, so the error piles up to about 250 u, still covered by
    # gamma_2000 (k is A's column count, not its row count)
    a = np.full((1, 2000), 0.1, dtype=np.float32)
    b = np.ones((2000, 1), dtype=np.float32)
    got = f32_dot(a[0], b[:, 0])
    exact = exact_dot(a[0], b[:, 0])
    assert bound_ratio(got, exact, matmul_error_bound(a, b, "f32")[0, 0]) <= 1.0
    assert abs(got - exact) > 100 * U32 * exact
    # cancellation: the bound scales with |A| @ |B|, not with |A @ B|
    a = np.array([[1.0, 1e-8, -1.0]], dtype=np.float32)
    b = np.ones((3, 1), dtype=np.float32)
    got, exact = f32_dot(a[0], b[:, 0]), exact_dot(a[0], b[:, 0])
    assert got == 0.0 and exact > 0
    assert bound_ratio(got, exact, matmul_error_bound(a, b, "f32")[0, 0]) <= 1.0


def test_matmul_bound_budgets_a_perturbation():
    # WHY: a tolerance budget adds the error you introduce on purpose
    #      (quantizing weights to a grid of step s moves each by at most
    #      s / 2, L8.5) to the rounding error; without the dA @ |B| term the
    #      quantized product falls outside the budget.
    # KIND: property
    # CATCHES: s06, s07, s08, s20, m03
    # CHAPTER: M09.3 section 2.4
    rng = PCG32(seed(), 34)
    A = rng.normal_array((6, 128))
    B = rng.normal_array((128, 4))
    s = 1.0 / 16
    Aq = np.round(A / s) * s
    exact = A @ B
    got = (Aq.astype(np.float32) @ B.astype(np.float32)).astype(np.float64)
    budget = matmul_error_bound(A, B, "f32", dA=s / 2)
    assert bound_ratio(got, exact, budget) <= 1.0
    assert bound_ratio(got, exact, matmul_error_bound(A, B, "f32")) > 1.0
    want = np.full_like(A, s / 2) @ np.abs(B) + gamma(128, "f32") * (
        (np.abs(A) + s / 2) @ np.abs(B)
    )
    assert_close(budget, want, rtol=1e-12, atol=0)
    with pytest.raises(ValueError):
        matmul_error_bound(A, B, "f32", dA=-1.0)
    with pytest.raises(ValueError):
        matmul_error_bound(A, B.T, "f32")


# --- assertions built on the bound -------------------------------------------------------------


def test_assert_close_bounded_passes_and_fails():
    # WHY: the helper is a test's verdict: it passes a correct float32 dot
    #      product and rejects one that dropped a single term, and its message
    #      names the index, the error, the bound, and their ratio.
    # KIND: unit
    # CATCHES: s09, m02
    # CHAPTER: M09.3 section 4
    rng = PCG32(seed(), 35)
    X = rng.normal_array((8, 200)).astype(np.float32)
    y = rng.normal_array((200,)).astype(np.float32)
    exact = np.array([exact_dot(r, y) for r in X])
    absdot = np.array([exact_dot(np.abs(r), np.abs(y)) for r in X])
    good = np.array([f32_dot(r, y) for r in X])
    assert_close_bounded(good, exact, 200, "f32", absdot)
    bad = good.copy()
    bad[3] = f32_dot(X[3][:-1], y[:-1])  # dropped the last term
    with pytest.raises(AssertionError, match=r"\(3,\).*ratio"):
        assert_close_bounded(bad, exact, 200, "f32", absdot)
    # slack scales the allowance: an error of 3x the bound
    b = float(dot_error_bound(10, "f32", 1.0))
    assert_close_bounded(1.0 + 3 * b, 1.0, 10, "f32", 1.0, slack=4.0)
    with pytest.raises(AssertionError):
        assert_close_bounded(1.0 + 3 * b, 1.0, 10, "f32", 1.0, slack=2.0)
    with pytest.raises(ValueError):
        assert_close_bounded(1.0, 1.0, 10, "f32", 1.0, slack=0.0)


def test_assert_close_bounded_special_values():
    # WHY: NaN equals NaN and an infinity equals itself for this purpose
    #      (both sides overflowed the same way), a zero bound accepts only an
    #      exact match, and mismatched shapes are a failure, not a broadcast.
    # KIND: boundary
    # CATCHES: s10
    # CHAPTER: M09.3 section 4
    assert_close_bounded(
        [np.nan, np.inf, -np.inf, 2.0],
        [np.nan, np.inf, -np.inf, 2.0],
        4,
        "f32",
        [1.0, 1.0, 1.0, 0.0],
    )
    with pytest.raises(AssertionError):
        assert_close_bounded([np.inf], [-np.inf], 4, "f32", [1.0])
    with pytest.raises(AssertionError):
        assert_close_bounded([np.nan], [1.0], 4, "f32", [1.0])
    with pytest.raises(AssertionError):
        assert_close_bounded([2.0 + 1e-15], [2.0], 4, "f32", [0.0])
    with pytest.raises(AssertionError):
        assert_close_bounded([1.0, 2.0], [[1.0, 2.0]], 4, "f32", [1.0, 1.0])


def test_bound_ratio():
    # WHY: optional C parity reports the worst error over its budget
    #      (the maximum, not an average that hides one bad element). A zero
    #      error over a zero budget is fine; a nonzero one is infinitely over.
    # KIND: unit
    # CATCHES: s18, s19
    # CHAPTER: M09.3 section 4
    assert bound_ratio([1.0, 2.0, 3.0], [1.0, 2.5, 3.0], [1.0, 1.0, 1.0]) == 0.5
    assert bound_ratio([1.0, 2.0, 3.5], [1.0, 2.0, 3.0], [1.0, 0.0, 0.25]) == 2.0
    assert bound_ratio([1.0, 2.0], [1.0, 2.0], [0.0, 0.0]) == 0.0
    assert bound_ratio([1.0], [2.0], [0.0]) == math.inf
    assert bound_ratio([], [], []) == 0.0


# --- condition numbers --------------------------------------------------------------------------


def test_cond_matches_numpy():
    # WHY: an independent implementation (LAPACK through numpy) agrees on
    #      square, tall, and wide matrices, including the 6x6 Hilbert matrix
    #      (cond about 1.5e7), the classic ill-conditioned example.
    # KIND: golden
    # CATCHES: s12, s13
    # CHAPTER: M09.3 section 2.5
    rng = PCG32(seed(), 36)
    for shape in [(3, 3), (5, 2), (2, 5), (8, 8), (1, 4)]:
        A = rng.normal_array(shape)
        assert_close(cond(A), np.linalg.cond(A, 2), rtol=1e-8, atol=0)
    H = 1.0 / (np.arange(6)[:, None] + np.arange(6)[None, :] + 1.0)
    assert_close(cond(H), np.linalg.cond(H, 2), rtol=1e-6, atol=0)
    assert 1.4e7 < cond(H) < 1.6e7
    N = np.array([[1.0, 1e3], [0.0, 1.0]])  # non-normal: eigenvalues are both 1
    assert_close(cond(N), np.linalg.cond(N, 2), rtol=1e-9, atol=0)


def test_cond_edges():
    # WHY: cond(I) = 1, cond is scale invariant, an orthogonal matrix is
    #      perfectly conditioned, and a singular matrix (to working
    #      precision) is infinitely ill-conditioned, not "1e17".
    # KIND: boundary
    # CATCHES: s11, s13
    # CHAPTER: M09.3 section 2.5
    assert cond(np.eye(4)) == pytest.approx(1.0, abs=1e-12)
    rng = PCG32(seed(), 37)
    A = rng.normal_array((4, 4))
    assert_close(cond(1e-30 * A), cond(A), rtol=1e-9, atol=0)
    Q = np.linalg.qr(rng.normal_array((5, 5)))[0]
    assert cond(Q) == pytest.approx(1.0, abs=1e-9)
    S = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 0.0, 1.0]])
    assert cond(S) == math.inf
    assert cond(np.zeros((2, 3))) == math.inf
    for bad in [np.zeros((0, 3)), np.ones(3), np.array([[1.0, np.nan], [0.0, 1.0]])]:
        with pytest.raises(ValueError):
            cond(bad)


def test_cond_bounds_the_amplification():
    # WHY: the meaning of cond(A): for Ax = b, a relative change in b moves
    #      x by at most cond(A) times as much, and the bound is reached when b
    #      lies along the largest singular direction and the change along the
    #      smallest. This is why no algorithm can solve an ill-conditioned
    #      system accurately.
    # KIND: property
    # CATCHES: s12, s13
    # CHAPTER: M09.3 section 2.5
    rng = PCG32(seed(), 38)
    for _ in range(50):
        n = 2 + rng.below(5)
        A = rng.normal_array((n, n))
        k = cond(A)
        b = rng.normal_array((n,))
        db = 1e-7 * rng.normal_array((n,))
        x, x2 = np.linalg.solve(A, b), np.linalg.solve(A, b + db)
        amp = (np.linalg.norm(x2 - x) / np.linalg.norm(x)) / (
            np.linalg.norm(db) / np.linalg.norm(b)
        )
        assert amp <= k * (1 + 1e-6)
        U, _, _ = np.linalg.svd(A)
        b, db = U[:, 0], 1e-7 * U[:, -1]
        x, x2 = np.linalg.solve(A, b), np.linalg.solve(A, b + db)
        amp = (np.linalg.norm(x2 - x) / np.linalg.norm(x)) / (
            np.linalg.norm(db) / np.linalg.norm(b)
        )
        assert_close(amp, k, rtol=1e-5, atol=0)


def test_relative_condition():
    # WHY: kappa_f(x) = |x f'(x) / f(x)| separates problems from algorithms:
    #      sqrt halves relative errors (kappa 1/2), log near 1 and x - 1 near
    #      1 amplify them enormously, which is cancellation seen as
    #      conditioning.
    # KIND: unit
    # CATCHES: s14, s15
    # CHAPTER: M09.3 section 2.5
    assert relative_condition(math.sqrt, lambda x: 0.5 / math.sqrt(x), 9.0) == 0.5
    assert relative_condition(math.log, lambda x: 1.0 / x, math.e) == pytest.approx(
        1.0, rel=1e-15
    )
    assert relative_condition(math.log, lambda x: 1.0 / x, 1.001) == pytest.approx(
        1.0 / math.log(1.001), rel=1e-12
    )
    k = relative_condition(lambda x: x - 1.0, lambda x: 1.0, 1.0 + 2.0**-20)
    assert k == (1.0 + 2.0**-20) / 2.0**-20
    assert relative_condition(lambda x: x - 1.0, lambda x: 1.0, 1.0) == math.inf
    assert relative_condition(math.cos, lambda x: -math.sin(x), 0.0) == 0.0
    assert relative_condition(lambda x: 3.0 * x, lambda x: 3.0, -2.0) == 1.0


# --- finite-difference steps ---------------------------------------------------------------------


def test_fd_model_and_optimal_step():
    # WHY: total error = truncation (falls with h) + rounding (grows as
    #      1/h); setting the derivative to zero gives h* = 2 sqrt(u F / D)
    #      for forward and (3 u F / D)^(1/3) for central differences, the
    #      steps M01.1 and gradcheck use.
    # KIND: unit
    # CATCHES: s01, s16, s17, m05
    # CHAPTER: M09.3 section 2.6
    assert fd_error_model(1e-3, 1, "f64") == 1e-3 / 2 + 2 * U64 / 1e-3
    assert fd_error_model(1e-3, 2, "f64") == 1e-6 / 6 + U64 / 1e-3
    h1, h2 = optimal_fd_step(1, "f64"), optimal_fd_step(2, "f64")
    assert h1 == 2 * math.sqrt(U64)
    assert h2 == (3 * U64) ** (1 / 3)
    assert optimal_fd_step(2, "f32", f_scale=8.0, deriv_scale=1.0) == pytest.approx(
        2 * (3 * U32) ** (1 / 3), rel=1e-15
    )
    for order, h in [(1, h1), (2, h2)]:
        e = fd_error_model(h, order, "f64")
        for f in (0.5, 0.9, 1.1, 2.0):
            assert fd_error_model(h * f, order, "f64") > e
    for bad in [
        lambda: fd_error_model(0.0, 1, "f64"),
        lambda: fd_error_model(1e-3, 3, "f64"),
        lambda: optimal_fd_step(3, "f64"),
        lambda: optimal_fd_step(1, "f64", deriv_scale=0.0),
    ]:
        with pytest.raises(ValueError):
            bad()


def test_optimal_step_on_real_differences():
    # WHY: the model predicts real behavior: M01.1's central difference of
    #      sin at x = 1 is most accurate near h*, a thousand times smaller step
    #      drowns in rounding and a thousand times larger one in truncation,
    #      and at h* the observed error is within the model's estimate.
    # KIND: property
    # CATCHES: s16
    # CHAPTER: M09.3 section 2.6
    x, true = 1.0, math.cos(1.0)
    h = optimal_fd_step(
        2, "f64", f_scale=abs(math.sin(x)), deriv_scale=abs(math.cos(x))
    )
    err = lambda hh: abs(central_diff(math.sin, x, hh) - true)
    e_star = err(h)
    assert e_star <= 2 * fd_error_model(h, 2, "f64", abs(math.sin(x)), abs(math.cos(x)))
    assert err(h / 1000) > 10 * e_star
    assert err(h * 1000) > 10 * e_star
