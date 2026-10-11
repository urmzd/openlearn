"""Course tests for M08.4: Hessian-vector products and checkpoint schedules
(tinyllm/autograd/hvp.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M08.4), and the chapter section it comes from.

The worked examples of the chapter (section 3): the quartic f(x) = x^4 / 24
at x = 1 with v = 1 and eps = 0.1 (exact H v = 1/2, central difference
0.5016667, error eps^2 / 6), and the schedules of a 6-layer stack.

Two golden tests compare with torch's double backward on the mean softmax
cross-entropy of a linear model. Its gradient, X^T (softmax(X W) - onehot) / n,
is M08.3's cross-entropy VJP written out here, so these tests call no other
module.
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.autograd.hvp import (
    checkpoint_cost,
    checkpoint_schedule,
    hessian_fd,
    hvp_fd,
    min_checkpoint_memory,
)

GOLDEN = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "M08.4" / "torch_hvp.npz"


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def normals(rng: PCG32, shape) -> np.ndarray:
    """Standard normals by Box-Muller from the frozen PCG32 (deterministic)."""
    n = int(np.prod(shape))
    u1 = np.array([1.0 - rng.uniform() for _ in range(n)])
    u2 = np.array([rng.uniform() for _ in range(n)])
    return (np.sqrt(-2.0 * np.log(u1)) * np.cos(2 * np.pi * u2)).reshape(shape)


def ce_grad_logits(z, t):
    """d mean CE / d logits = (softmax(z) - onehot(t)) / n (M08.3, section 2)."""
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    p = e / e.sum(axis=-1, keepdims=True)
    p[np.arange(len(t)), t] -= 1.0
    return p / len(t)


def quartic_grad(x):
    """Gradient of f(x) = sum(x^4) / 24: x^3 / 6. Its Hessian is diag(x^2 / 2)."""
    return np.asarray(x) ** 3 / 6.0


def rosenbrock_grad(p):
    x, y = p
    return np.array([-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)])


def rosenbrock_hessian(p):
    x, y = p
    return np.array([[1200 * x * x - 400 * y + 2, -400 * x], [-400 * x, 200.0]])


def brute_force(n: int, budget: int):
    """Fewest recomputed layers over all 2^(n-1) schedules with peak <= budget."""
    best = None
    for cut in itertools.product((0, 1), repeat=n - 1):
        starts = [0] + [i + 1 for i, c in enumerate(cut) if c]
        sizes = [b - a for a, b in zip(starts, starts[1:] + [n])]
        peak = max(i + s for i, s in enumerate(sizes))
        if peak <= budget:
            r = n - sizes[-1]
            best = r if best is None else min(best, r)
    return best


# --- the worked examples ---------------------------------------------------------


def test_hand_example_quartic():
    # WHY: section 3, number for number. f(x) = x^4 / 24 has g = x^3 / 6 and
    #      H = x^2 / 2 = 1/2 at x = 1. The central difference with eps = 0.1
    #      is (1.1^3 - 0.9^3) / (6 * 0.2) = 0.602 / 1.2 = 0.5016667, off by
    #      exactly eps^2 v^3 / 6 = 1/600. A one-sided difference gives
    #      0.5516667, thirty times further off; dropping the 2 gives 1.003.
    # KIND: unit
    # CATCHES: s01, s02
    # CHAPTER: M08.4 section 3, Worked example by hand
    got = hvp_fd(quartic_grad, np.array([1.0]), np.array([1.0]), eps=0.1)
    assert got.shape == (1,) and got.dtype == np.float64
    assert_close(got, [0.602 / 1.2], rtol=1e-12, atol=1e-14)
    assert_close(got - 0.5, [0.1**2 / 6], rtol=1e-9, atol=1e-14)


def test_hand_example_schedule():
    # WHY: section 3's six-layer stack. Keeping everything holds 6 units;
    #      the least peak is 3 (3 * 4 / 2 = 6 >= 6), reached by segments of
    #      3, 2, 1 layers (starts [0, 3, 5]) at 5 recomputed layers. With a
    #      budget of 4, two segments [0, 3] recompute only 3 layers. Uniform
    #      segments of 2 (the sqrt(n) rule) peak at 4 and recompute 4.
    # KIND: unit
    # CATCHES: s05, s06, s07, s09, s10, m02
    # CHAPTER: M08.4 section 3, Worked example by hand
    assert min_checkpoint_memory(6) == 3
    assert checkpoint_schedule(6, 3) == [0, 3, 5]
    assert checkpoint_cost(6, [0, 3, 5]) == (3, 5)
    assert checkpoint_schedule(6, 4) == [0, 3]
    assert checkpoint_cost(6, [0, 3]) == (4, 3)
    assert checkpoint_schedule(6, 6) == [0]
    assert checkpoint_cost(6, [0]) == (6, 0)
    assert checkpoint_cost(6, [0, 2, 4]) == (4, 4)


# --- Hessian-vector products ------------------------------------------------------


def test_quadratic_hvp_is_exact():
    # WHY: for f(x) = x^T A x / 2 + b^T x the gradient (A + A^T) x / 2 + b is
    #      affine, so the central difference has no truncation error at any
    #      eps: H v = (A + A^T) v / 2 up to rounding. A non-symmetric A and a
    #      v of norm about 3 check that the step is eps * v, not normalized.
    #      (A one-sided difference is exact here too: only the next test
    #      tells the two apart.)
    # KIND: property
    # CATCHES: s02, s12
    # CHAPTER: M08.4 section 2, Principles
    rng = PCG32(seed(), 84)
    for n, eps in ((1, 1e-1), (4, 1e-3), (9, 1e-2)):
        A, b = normals(rng, (n, n)), normals(rng, (n,))
        x, v = normals(rng, (n,)), 3.0 * normals(rng, (n,))
        got = hvp_fd(lambda z: 0.5 * (A + A.T) @ z + b, x, v, eps=eps)
        assert_close(
            got, 0.5 * (A + A.T) @ v, rtol=1e-8, atol=1e-9, msg=f"n={n} eps={eps}"
        )


def test_error_shrinks_as_eps_squared():
    # WHY: the central difference cancels the eps^1 term, so halving eps
    #      quarters the error: the ratio of errors is 4 (a one-sided
    #      difference gives 2). That is why eps = 1e-4 is accurate to about
    #      1e-9 in float64 while staying far above rounding noise.
    # KIND: property
    # CATCHES: s01
    # CHAPTER: M08.4 section 2, Principles
    rng = PCG32(seed(), 85)
    x, v = 1.0 + normals(rng, (5,)) * 0.3, normals(rng, (5,))
    exact = x * x / 2 * v
    e1 = np.abs(hvp_fd(quartic_grad, x, v, eps=0.02) - exact).max()
    e2 = np.abs(hvp_fd(quartic_grad, x, v, eps=0.01) - exact).max()
    assert 3.9 < e1 / e2 < 4.1, (
        f"halving eps divided the error by {e1 / e2:.2f}, expected 4"
    )


def test_matches_torch_hvp():
    # WHY: torch's exact double-backward H v of the mean softmax
    #      cross-entropy of a linear model, recorded by
    #      course/oracle/M08.4/torch_hvp_golden.py. The gradient is M08.3's
    #      rule: dL/dW = X^T (softmax(X W) - onehot(t)) / n. With eps = 1e-4
    #      the central difference agrees to about 1e-9.
    # KIND: golden
    # CATCHES: s01, s02, m01
    # CHAPTER: M08.4 section 2, Principles
    d = np.load(GOLDEN, allow_pickle=False)
    X, t, W = d["ce/X"], d["ce/t"], d["ce/W"]

    def grad(w):
        return X.T @ ce_grad_logits(X @ w, t)

    for v, hv in zip(d["ce/V"], d["ce/HV"]):
        assert_close(hvp_fd(grad, W, v), hv, rtol=1e-6, atol=1e-9)


def test_hessian_fd_matches_torch():
    # WHY: the dense Hessian built column by column from H e_j equals
    #      torch.autograd.functional.hessian on a 6-parameter model (W [3, 2]
    #      flattened row-major), and it is exactly symmetric.
    # KIND: golden
    # CATCHES: s01, s02, s04
    # CHAPTER: M08.4 section 4, The interface
    d = np.load(GOLDEN, allow_pickle=False)
    X, t, W = d["hess/X"], d["hess/t"], d["hess/W"]

    def grad(w):
        return (X.T @ ce_grad_logits(X @ w.reshape(3, 2), t)).reshape(-1)

    H = hessian_fd(grad, W.reshape(-1))
    assert H.shape == (6, 6)
    assert_close(H, d["hess/H"], rtol=1e-6, atol=1e-9)
    assert np.array_equal(H, H.T)


def test_hessian_is_exactly_symmetric():
    # WHY: away from the minimum of the Rosenbrock function the columns
    #      H e_j carry O(eps^2) errors that differ between H[i, j] and
    #      H[j, i]; Newton and eigenvalue code (M10.5) assume symmetry, so
    #      the result is (H + H^T) / 2. It still matches the analytic
    #      Hessian to about 1e-6 relative.
    # KIND: unit
    # CATCHES: s04
    # CHAPTER: M08.4 section 5, Pitfalls
    p = np.array([-1.2, 1.0])
    H = hessian_fd(rosenbrock_grad, p, eps=1e-3)
    assert np.array_equal(H, H.T), "hessian_fd must return an exactly symmetric matrix"
    assert_close(H, rosenbrock_hessian(p), rtol=1e-6, atol=1e-6)
    for bad in (np.zeros((2, 2)), np.zeros(0)):
        with pytest.raises(ValueError):
            hessian_fd(rosenbrock_grad, bad)


def test_grad_fn_may_return_its_argument():
    # WHY: f(x) = |x|^2 / 2 has the gradient x itself, and a cheap grad_fn
    #      returns its argument (or a view of it). If hvp_fd reused one
    #      buffer for x + eps v and x - eps v, the first gradient would change
    #      under it and H v would come out 0. Each call gets a new array, the
    #      caller's x is never written, and grad_fn runs exactly twice.
    # KIND: unit
    # CATCHES: s03
    # CHAPTER: M08.4 section 5, Pitfalls
    x = np.array([1.0, -2.0, 0.5])
    keep = x.copy()
    v = np.array([0.25, 1.0, -3.0])
    seen = []

    def grad(z):
        seen.append(z)
        return z

    got = hvp_fd(grad, x, v, eps=1e-3)
    assert_close(got, v, rtol=1e-9, atol=1e-12)
    assert np.array_equal(x, keep), "hvp_fd wrote into the caller's x"
    assert len(seen) == 2, f"grad_fn was called {len(seen)} times, expected 2"
    assert all(z is not x for z in seen)


def test_hvp_rejects_bad_inputs():
    # WHY: a direction of the wrong shape, a zero or non-finite step, or a
    #      gradient of the wrong shape is a caller bug; returning a number
    #      for it hides the bug in a curvature estimate.
    # KIND: boundary
    # CATCHES: m01
    # CHAPTER: M08.4 section 4, The interface
    x = np.ones(3)
    with pytest.raises(ValueError):
        hvp_fd(quartic_grad, x, np.ones(2))
    for eps in (0.0, -1e-3, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            hvp_fd(quartic_grad, x, np.ones(3), eps=eps)
    with pytest.raises(ValueError):
        hvp_fd(lambda z: z[:2], x, np.ones(3))
    # the default step is 1e-4: accurate to 1e-8 on the quartic
    assert_close(
        hvp_fd(quartic_grad, x, np.ones(3)), np.full(3, 0.5), rtol=1e-8, atol=1e-9
    )


# --- checkpoint schedules ---------------------------------------------------------------


def test_checkpoint_cost_extremes():
    # WHY: the two ends of the trade-off cost the same memory: no
    #      checkpointing keeps all n activations, and a checkpoint at every
    #      layer keeps n inputs and recomputes n - 1 layers for nothing.
    #      The peak is max(i + s_i), not (number of segments) + (largest
    #      segment), which overcounts by one when the largest is first.
    # KIND: unit
    # CATCHES: s05, s06, s11
    # CHAPTER: M08.4 section 2, Principles
    for n in (1, 2, 7, 12):
        assert checkpoint_cost(n, [0]) == (n, 0)
        assert checkpoint_cost(n, list(range(n))) == (n, n - 1)
    assert checkpoint_cost(10, [0, 4, 7, 9]) == (4, 9)
    assert checkpoint_cost(10, [0, 5, 7]) == (5, 7)
    for bad in ([], [1, 3], [0, 0, 2], [0, 3, 2], [0, 10]):
        with pytest.raises(ValueError):
            checkpoint_cost(10, bad)
    with pytest.raises(ValueError):
        checkpoint_cost(0, [0])


def test_min_memory_is_triangular():
    # WHY: the least peak P is the first triangular number at or above n:
    #      P (P + 1) / 2 >= n > (P - 1) P / 2. It grows like sqrt(2 n),
    #      below the 2 sqrt(n) - 1 of uniform segments. Computed with
    #      integers, so n = 10^12 is exact (a float sqrt is not).
    # KIND: unit
    # CATCHES: s07, m02
    # CHAPTER: M08.4 section 2, Principles
    assert [min_checkpoint_memory(n) for n in range(1, 12)] == [
        1,
        2,
        2,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        5,
    ]
    for n in (100, 10**6, 10**12, 10**12 + 1, 2**61 - 1):
        p = min_checkpoint_memory(n)
        assert p * (p + 1) // 2 >= n > (p - 1) * p // 2, n
    assert min_checkpoint_memory(16) == 6
    assert checkpoint_cost(16, [0, 4, 8, 12]) == (7, 12)  # uniform sqrt(n) segments
    with pytest.raises(ValueError):
        min_checkpoint_memory(0)


def test_schedule_is_optimal():
    # WHY: for every stack of up to 10 layers and every feasible budget, the
    #      schedule stays within the budget and recomputes exactly as few
    #      layers as the best of all 2^(n-1) schedules (brute force).
    # KIND: property
    # CATCHES: s05, s09, s10
    # CHAPTER: M08.4 section 2, Principles
    for n in range(1, 11):
        for budget in range(min_checkpoint_memory(n), n + 2):
            s = checkpoint_schedule(n, budget)
            peak, rec = checkpoint_cost(n, s)
            assert peak <= budget, f"n={n} budget={budget}: {s} peaks at {peak}"
            assert rec == brute_force(n, budget), (
                f"n={n} budget={budget}: {s} recomputes {rec}"
            )


def test_schedule_ties_go_to_the_largest_first_segment():
    # WHY: several schedules can tie on recomputation; the contract fixes one
    #      (largest segments first), so your C1 trainer and the reference
    #      checkpoint the same layers and their peak memory matches.
    # KIND: unit
    # CATCHES: s08
    # CHAPTER: M08.4 section 4, The interface
    assert checkpoint_schedule(10, 4) == [0, 4, 7, 9]
    assert checkpoint_schedule(10, 5) == [0, 5, 7]
    assert checkpoint_schedule(16, 6) == [0, 6, 11, 13]
    assert checkpoint_schedule(24, 8) == [0, 8, 15, 19]


def test_schedule_rejects_impossible_budget():
    # WHY: no schedule of 10 layers peaks below 4; returning one anyway would
    #      let a trainer promise memory it then overruns.
    # KIND: boundary
    # CATCHES: s10
    # CHAPTER: M08.4 section 5, Pitfalls
    with pytest.raises(ValueError):
        checkpoint_schedule(10, 3)
    with pytest.raises(ValueError):
        checkpoint_schedule(0, 5)
    assert checkpoint_schedule(1, 1) == [0]
