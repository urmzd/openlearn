"""Course tests for M01.2: Newton's method (tinyllm/num/newton.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M01.2), and the chapter section it comes from.

The worked example of the chapter (section 3) is sqrt(2) as the root of
f(x) = x**2 - 2 from x0 = 1: iterates 1, 3/2, 17/12, 577/408,
665857/470832, then two more updates to converge (6 in all).
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.num.newton import newton, rsqrt_newton

SEED = int(os.environ.get("OL_SEED", "0"))
SQRT2 = math.sqrt(2.0)


class Recorder:
    """f(x) = x**2 - 2 that remembers every point it is evaluated at: the
    Newton iterates x0, x1, ... (the final iterate is returned, not evaluated)."""

    def __init__(self, target: float = 2.0) -> None:
        self.xs: list[float] = []
        self.target = target

    def __call__(self, x: float) -> float:
        self.xs.append(x)
        return x * x - self.target


def magic_rsqrt_guess(x: np.ndarray) -> np.ndarray:
    """The float32 bit trick 0x5f3759df - (bits >> 1): a first guess for
    1/sqrt(x) within 3.5 percent, the starting point M09.5 uses in C."""
    i = x.astype(np.float32).view(np.int32)
    return (np.int32(0x5F3759DF) - (i >> 1)).view(np.float32)


# --- the worked example -------------------------------------------------------


def test_hand_example():
    # WHY: the chapter's worked example step for step: the iterates of
    #      x - (x^2 - 2) / (2x) from 1 are the fractions 3/2, 17/12, 577/408,
    #      665857/470832, and the method reports 6 updates at tol 1e-12.
    # KIND: unit
    # CATCHES: s01, s02, m01, m02
    # CHAPTER: M01.2 section 3, Worked example by hand
    f = Recorder()
    root, n = newton(f, lambda x: 2 * x, 1.0)
    assert_close(root, SQRT2, rtol=0, atol=4.5e-16)
    assert n == 6
    want = [1.0, 3 / 2, 17 / 12, 577 / 408, 665857 / 470832]
    assert_close(f.xs[:5], want, rtol=1e-15, atol=0)
    assert len(f.xs) == 6  # x0..x5 evaluated; x6 is the answer


def test_digits_double_per_step():
    # WHY: quadratic convergence: e[n+1] ~ C e[n]**2 with C = |f''| / (2|f'|)
    #      = 1 / (2 sqrt 2) at the root. The ratio e[n+1] / e[n]**2 settles
    #      at 0.354 within a few percent: the number of correct digits doubles.
    # KIND: property
    # CATCHES: s01, s03
    # CHAPTER: M01.2 section 2.3, Quadratic convergence
    f = Recorder()
    newton(f, lambda x: 2 * x, 1.0)
    e = [abs(x - SQRT2) for x in f.xs[:5]]
    ratios = [e[k + 1] / e[k] ** 2 for k in range(1, 4)]
    for r in ratios[1:]:
        assert 0.33 < r < 0.37, ratios


def test_returns_on_exact_root():
    # WHY: if f(x0) is exactly 0, x0 is the root and no update is needed:
    #      (x0, 0). An implementation that always steps once divides 0 by
    #      f'(x0) needlessly and counts an update that never happened.
    # KIND: boundary
    # CATCHES: s04
    # CHAPTER: M01.2 section 4, The interface
    assert newton(lambda x: x - 3.0, lambda x: 1.0, 3.0) == (3.0, 0)
    root, n = newton(lambda x: x * x - 4.0, lambda x: 2 * x, 2.0)
    assert (root, n) == (2.0, 0)


def test_relative_tolerance_for_large_roots():
    # WHY: floats near 12649 are 1.8e-12 apart, so near sqrt(1.6e8) the last
    #      Newton steps hop between two neighbouring floats and an absolute
    #      |x[n+1] - x[n]| <= 1e-12 never holds: the method would report no
    #      convergence at the right answer. The test is relative,
    #      tol * max(1, |x|). Same for the cube root of 3e20 (6.7e6).
    # KIND: boundary
    # CATCHES: s05
    # CHAPTER: M01.2 section 5, Pitfalls, item 3
    root, n = newton(lambda x: x * x - 1.6e8, lambda x: 2 * x, 3e4)
    assert_close(root, math.sqrt(1.6e8), rtol=4e-16, atol=0)
    assert n <= 10
    root, n = newton(lambda x: x**3 - 3e20, lambda x: 3 * x * x, 1e7)
    assert_close(root, 3e20 ** (1 / 3), rtol=4e-15, atol=0)
    assert n <= 10


def test_converges_on_many_roots():
    # WHY: Newton from a reasonable start converges to the nearest simple
    #      root for any smooth f; the oracle is the closed form. Starts are
    #      seeded draws on the side where the tangent is well behaved.
    # KIND: golden
    # CATCHES: s01, s02
    # CHAPTER: M01.2 section 2.2, The tangent-line step
    rng = PCG32(seed=SEED)
    for _ in range(30):
        a = 0.5 + 99.5 * rng.uniform()  # find sqrt(a)
        x0 = math.sqrt(a) * (1 + rng.uniform())
        root, _ = newton(lambda x: x * x - a, lambda x: 2 * x, x0)
        assert_close(root, math.sqrt(a), rtol=4e-16, atol=0)
        c = -5 + 10 * rng.uniform()  # cube root of c by x^3 - c from 1 + |c|
        root, _ = newton(
            lambda x: x**3 - c, lambda x: 3 * x * x, math.copysign(1 + abs(c), c)
        )
        assert_close(root, math.copysign(abs(c) ** (1 / 3), c), rtol=1e-14, atol=0)


def test_numeric_derivative_when_df_is_none():
    # WHY: df = None differentiates with your central_diff (M01.1). Newton
    #      only needs a good enough slope: with a relative slope error near
    #      1e-10 it still lands on the root to full precision.
    # KIND: unit
    # CATCHES: s06
    # CHAPTER: M01.2 section 4, The interface
    root, n = newton(lambda x: math.exp(x) - 3.0, None, 0.0)
    assert_close(root, math.log(3.0), rtol=1e-14, atol=0)
    assert n < 10


# --- failures are reported, not hidden ----------------------------------------


def test_zero_derivative_raises():
    # WHY: at a stationary point the tangent is flat and never crosses zero;
    #      x - f / 0 is not an iterate. The contract wants RuntimeError
    #      naming the derivative, not ZeroDivisionError or inf.
    # KIND: boundary
    # CATCHES: s07
    # CHAPTER: M01.2 section 5, Pitfalls, item 1
    with pytest.raises(RuntimeError, match="derivative"):
        newton(lambda x: x * x - 2.0, lambda x: 2 * x, 0.0)
    with pytest.raises(RuntimeError, match="derivative"):
        newton(lambda x: x * x - 2.0, lambda x: float("nan"), 1.0)


def test_cycle_raises_after_max_iter():
    # WHY: f(x) = x^3 - 2x + 2 from 0 cycles 0, 1, 0, 1, ... forever. Newton
    #      is only locally convergent; after max_iter updates the method must
    #      say so instead of returning the last iterate as a root.
    # KIND: boundary
    # CATCHES: s08
    # CHAPTER: M01.2 section 5, Pitfalls, item 2
    with pytest.raises(RuntimeError, match="convergence"):
        newton(lambda x: x**3 - 2 * x + 2, lambda x: 3 * x * x - 2, 0.0, max_iter=25)


def test_max_iter_counts_updates():
    # WHY: max_iter bounds the number of updates exactly: sqrt(2) from 1
    #      needs 6, so max_iter = 6 succeeds and max_iter = 5 fails. An off-by-
    #      one here makes a budget that reads right but runs one step short.
    # KIND: boundary
    # CATCHES: m03
    # CHAPTER: M01.2 section 4, The interface
    assert newton(lambda x: x * x - 2, lambda x: 2 * x, 1.0, max_iter=6)[1] == 6
    with pytest.raises(RuntimeError):
        newton(lambda x: x * x - 2, lambda x: 2 * x, 1.0, max_iter=5)


def test_divergence_raises():
    # WHY: for f(x) = cbrt(x) the tangent at x crosses zero at -2x: every
    #      step doubles the distance from the root, 1, -2, 4, -8, ... until
    #      after about 1024 steps the step itself overflows to inf. A
    #      non-finite iterate must stop the method with a RuntimeError that
    #      says it diverged, not return inf as a root or run to max_iter.
    # KIND: boundary
    # CATCHES: s09
    # CHAPTER: M01.2 section 5, Pitfalls, item 2
    xs: list[float] = []

    def f(x: float) -> float:
        xs.append(x)
        return math.cbrt(x)

    with pytest.raises(RuntimeError, match="diverged"):
        newton(f, lambda x: 1.0 / (3.0 * math.cbrt(x) ** 2), 1.0, max_iter=2000)
    assert_close(xs[:4], [1.0, -2.0, 4.0, -8.0], rtol=1e-14, atol=0)


@pytest.mark.parametrize(
    "kw",
    [
        {"x0": float("nan")},
        {"x0": float("inf")},
        {"tol": 0.0},
        {"tol": -1.0},
        {"max_iter": 0},
    ],
)
def test_rejects_bad_arguments(kw):
    # WHY: a nan start, a tolerance that can never be met, or no updates at
    #      all are caller bugs; ValueError says so before any iteration.
    # KIND: boundary
    # CATCHES: m04, m05
    # CHAPTER: M01.2 section 4, The interface
    args = {"x0": 1.0, "tol": 1e-12, "max_iter": 50} | kw
    with pytest.raises(ValueError):
        newton(
            lambda x: x * x - 2,
            lambda x: 2 * x,
            args["x0"],
            args["tol"],
            args["max_iter"],
        )


# --- 1/sqrt(x) without division -----------------------------------------------


def test_rsqrt_hand_example():
    # WHY: one step of y <- y (3/2 - x y^2 / 2) for x = 4 from y0 = 0.4:
    #      0.4 * (1.5 - 0.5 * 4 * 0.16) = 0.4 * 1.18 = 0.472, then 0.4982...;
    #      the chapter's second worked example.
    # KIND: unit
    # CATCHES: s10, m06
    # CHAPTER: M01.2 section 3, Worked example by hand
    y1 = rsqrt_newton(4.0, 0.4, 1)
    assert_close(y1, 0.472, rtol=1e-15, atol=0)
    y2 = rsqrt_newton(4.0, 0.4, 2)
    assert_close(y2, 0.472 * (1.5 - 2 * 0.472**2), rtol=1e-15, atol=0)


def test_rsqrt_error_squares_each_step():
    # WHY: with relative error e, one step leaves about 1.5 e^2 (section 2.4).
    #      From the bit-trick guess (e <= 3.5e-2) two float32 steps reach
    #      5e-6: the accuracy M09.5's C rsqrt must match (DESIGN M09.5 row).
    # KIND: golden
    # CATCHES: s10, s11, m06
    # CHAPTER: M01.2 section 2.4, Newton for 1/sqrt(x)
    x = np.linspace(0.01, 100.0, 4001, dtype=np.float32)
    true = 1.0 / np.sqrt(x.astype(np.float64))
    y0 = magic_rsqrt_guess(x)
    errs = []
    for k in range(3):
        y = rsqrt_newton(x, y0, k)
        assert y.dtype == np.float32
        errs.append(float(np.max(np.abs(y.astype(np.float64) - true) / true)))
    assert errs[0] < 3.5e-2
    assert errs[1] < 1.8e-3
    assert errs[2] < 5e-6


def test_rsqrt_float64_reaches_full_precision():
    # WHY: in float64 the same iteration keeps squaring the error: four steps
    #      from 3.5e-2 reach the rounding floor (a few ulps).
    # KIND: golden
    # CATCHES: s11
    # CHAPTER: M01.2 section 2.4, Newton for 1/sqrt(x)
    rng = PCG32(seed=SEED)
    x = rng.uniform_array((200,), 1e-3, 1e3)
    true = 1.0 / np.sqrt(x)
    y = rsqrt_newton(x, true * (1 + 0.035), 4)
    assert y.dtype == np.float64
    assert_close(y, true, rtol=1e-15, atol=0)


def test_rsqrt_keeps_dtype_and_broadcasts():
    # WHY: float32 in, float32 out, computed in float32 step for step (the
    #      C port's arithmetic); int inputs compute in float64; a scalar y0
    #      broadcasts over an array x; iters = 0 returns y0 itself.
    # KIND: unit
    # CATCHES: s12, m07
    # CHAPTER: M01.2 section 4, The interface
    x32 = np.array([2.0, 9.0, 0.25], dtype=np.float32)
    y = rsqrt_newton(x32, np.float32(0.5), 3)
    assert y.dtype == np.float32 and y.shape == (3,)
    want = np.full(3, 0.5, dtype=np.float32)
    for _ in range(3):
        want = want * (np.float32(1.5) - np.float32(0.5) * x32 * want * want)
    assert y.tobytes() == want.tobytes()
    yi = rsqrt_newton(np.array([4, 16]), np.array([1, 1]), 0)
    assert yi.dtype == np.float64 and yi.tolist() == [1.0, 1.0]
    with pytest.raises(ValueError):
        rsqrt_newton(x32, 0.5, -1)
