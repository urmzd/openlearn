"""Course tests for M10.1: gradient descent and Armijo line search (tinyllm/optim/gd.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M10.1), and the chapter section it comes from.

The worked example of the chapter (section 3) is
f(x) = (x1^2 + 10 x2^2) / 2 (L = 10, mu = 1, kappa = 10) from x0 = (10, 1)
with lr = 1/L: x1 = (9, 0), x2 = (8.1, 0), f = 55, 40.5, 32.805; and one
Armijo search on f(x) = x^2 at x = 1 along d = -2, which halves alpha
once and returns 0.5.
"""

from __future__ import annotations

import math
import os
from fractions import Fraction

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.optim.gd import armijo_step, gradient_descent


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def quadratic(H: np.ndarray):
    """f(x) = x^T H x / 2 and its gradient H x; the minimum is f(0) = 0."""
    return (lambda x: 0.5 * float(x @ H @ x)), (lambda x: H @ x)


def spd(rng: PCG32, n: int, mu: float, L: float) -> np.ndarray:
    """A random symmetric matrix with eigenvalues spread over [mu, L],
    mu and L included: Q diag(lambda) Q^T with Q orthogonal."""
    Q, _ = np.linalg.qr(rng.normal_array((n, n)))
    lam = np.concatenate([[mu, L], mu + (L - mu) * rng.uniform_array(n - 2)])
    return (Q * lam) @ Q.T


def rosenbrock(x):
    return float((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2)


def rosenbrock_grad(x):
    return np.array(
        [-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)]
    )


# --- gradient descent ----------------------------------------------------------


def test_hand_example():
    # WHY: the chapter's worked example. With lr = 1/L = 0.1 the gradient
    #      (x1, 10 x2) at (10, 1) is (10, 10), so x1 = (9, 0): the stiff
    #      coordinate is solved in one step, the flat one shrinks by
    #      1 - mu/L = 0.9 per step. Then the Armijo search: alpha = 1 lands
    #      on x = -1 (f = 1, not below 1 - 4e-4), alpha = 0.5 lands on 0.
    # KIND: unit
    # CATCHES: s01, s02, s06
    # CHAPTER: M10.1 section 3, Worked example by hand
    f, g = quadratic(np.diag([1.0, 10.0]))
    xs = gradient_descent(f, g, np.array([10.0, 1.0]), lr=0.1, steps=2)
    assert len(xs) == 3
    assert_close(np.array(xs), [[10.0, 1.0], [9.0, 0.0], [8.1, 0.0]])
    assert_close([f(x) for x in xs], [55.0, 40.5, 32.805])
    alpha = armijo_step(
        lambda x: float(x[0] ** 2), lambda x: 2 * x, np.array([1.0]), np.array([-2.0])
    )
    assert alpha == 0.5


def test_returns_every_iterate_as_copies():
    # WHY: the trajectory is data (L0.5 plots it, tests compare it), so the
    #      steps + 1 iterates must be independent arrays: updating x in place
    #      makes every entry of the list the same final array, and the
    #      caller's x0 must survive untouched. steps = 0 is just [x0].
    # KIND: unit
    # CATCHES: s02, s03, s04
    # CHAPTER: M10.1 section 4, The interface
    f, g = quadratic(np.diag([1.0, 4.0]))
    x0 = np.array([1.0, 1.0])
    xs = gradient_descent(f, g, x0, lr=0.2, steps=3)
    assert len(xs) == 4 and x0.tolist() == [1.0, 1.0]
    assert len({id(x) for x in xs}) == 4 and all(x is not x0 for x in xs)
    assert_close(np.array(xs)[:, 0], [1.0, 0.8, 0.64, 0.512])
    xs[0][0] = 99.0
    assert x0[0] == 1.0
    one = gradient_descent(f, g, [3, 4], lr=0.1, steps=0)
    assert (
        len(one) == 1 and one[0].dtype == np.float64 and one[0].tolist() == [3.0, 4.0]
    )


def test_rate_on_quadratics():
    # WHY: the convergence theorem: on an L-smooth, mu-strongly convex
    #      function, gradient descent with lr = 1/L satisfies
    #      f(x_t) - f* <= (1 - 1/kappa)^t (f(x_0) - f*), kappa = L / mu.
    #      Checked at every step on random quadratics with kappa from 2 to 100.
    # KIND: property
    # CATCHES: s01
    # CHAPTER: M10.1 section 2, Principles (the convergence rate)
    rng = PCG32(seed=seed())
    for kappa in (2.0, 10.0, 100.0):
        H = spd(rng, 6, 1.0, kappa)
        f, g = quadratic(H)
        x0 = rng.normal_array(6)
        xs = gradient_descent(f, g, x0, lr=1.0 / kappa, steps=60)
        gap0 = f(x0)
        for t, x in enumerate(xs):
            assert f(x) <= (1 - 1 / kappa) ** t * gap0 * (1 + 1e-9) + 1e-15, (
                f"kappa {kappa}, step {t}"
            )


def test_exact_rational_trajectory():
    # WHY: on a quadratic with rational H, x0, and lr, every iterate is a
    #      rational number. The test recomputes the trajectory exactly with
    #      fractions.Fraction (independently of floats) and your float64
    #      trajectory must match it to rounding.
    # KIND: differential
    # CATCHES: s01, s03
    # CHAPTER: M10.1 section 2, Principles (gradient descent)
    Hq = [[Fraction(3), Fraction(1)], [Fraction(1), Fraction(2)]]
    lr = Fraction(1, 4)
    x = [Fraction(5), Fraction(-3)]
    exact = [list(x)]
    for _ in range(12):
        gq = [Hq[0][0] * x[0] + Hq[0][1] * x[1], Hq[1][0] * x[0] + Hq[1][1] * x[1]]
        x = [x[0] - lr * gq[0], x[1] - lr * gq[1]]
        exact.append(list(x))
    f, g = quadratic(np.array([[3.0, 1.0], [1.0, 2.0]]))
    xs = gradient_descent(f, g, np.array([5.0, -3.0]), lr=0.25, steps=12)
    assert_close(np.array(xs), np.array([[float(a) for a in row] for row in exact]))


def test_divergence_raises():
    # WHY: a step larger than 2/L makes the stiff coordinate grow by
    #      |1 - lr L| > 1 per step until it overflows. The trajectory must
    #      fail loudly, naming the step, instead of returning a list of
    #      inf and nan that later code averages into nonsense.
    # KIND: boundary
    # CATCHES: s05
    # CHAPTER: M10.1 section 5, Pitfalls, item 1
    f, g = quadratic(np.diag([1.0, 10.0]))
    with pytest.raises(FloatingPointError, match="step"):
        with np.errstate(over="ignore", invalid="ignore"):
            gradient_descent(f, g, np.array([1.0, 1.0]), lr=3.0, steps=1000)
    xs = gradient_descent(f, g, np.array([1.0, 1.0]), lr=0.19, steps=200)
    assert f(xs[-1]) < 1e-10


def test_rejects_bad_arguments():
    # WHY: lr <= 0 never descends and steps < 0 is meaningless; both are
    #      caller bugs.
    # KIND: boundary
    # CATCHES: m01
    # CHAPTER: M10.1 section 4, The interface
    f, g = quadratic(np.eye(2))
    for lr, steps in ((0.0, 5), (-0.1, 5), (0.1, -1)):
        with pytest.raises(ValueError):
            gradient_descent(f, g, np.ones(2), lr=lr, steps=steps)


# --- Armijo line search --------------------------------------------------------


def test_armijo_returns_the_first_acceptable_step():
    # WHY: backtracking tries alpha0, rho alpha0, rho^2 alpha0, ... and
    #      returns the FIRST that satisfies f(x + a d) <= f(x) + c a g.d:
    #      the largest acceptable step on that grid. Checked on 50 random
    #      points of the Rosenbrock function along -grad, with c = 0.3 so the
    #      bound is far from f(x).
    # KIND: property
    # CATCHES: s06, s09
    # CHAPTER: M10.1 section 2, Principles (the Armijo condition)
    rng = PCG32(seed=seed())
    c, rho = 0.3, 0.5
    for _ in range(50):
        x = rng.uniform_array(2, -1.5, 1.5)
        d = -rosenbrock_grad(x)
        a = armijo_step(rosenbrock, rosenbrock_grad, x, d, alpha0=1.0, c=c, rho=rho)
        slope = float(rosenbrock_grad(x) @ d)
        assert rosenbrock(x + a * d) <= rosenbrock(x) + c * a * slope
        k = round(math.log(a, rho))
        assert a == rho**k
        if k > 0:
            bigger = a / rho
            assert not rosenbrock(x + bigger * d) <= rosenbrock(x) + c * bigger * slope


def test_armijo_accepts_alpha0():
    # WHY: when the full step already decreases f enough, the search returns
    #      alpha0 unchanged; it never shrinks a step that works.
    # KIND: unit
    # CATCHES: s09, m04
    # CHAPTER: M10.1 section 2, Principles (the Armijo condition)
    a = armijo_step(
        lambda x: float(x[0] ** 2), lambda x: 2 * x, np.array([1.0]), np.array([-0.5])
    )
    assert a == 1.0
    a = armijo_step(
        lambda x: float(x[0] ** 2),
        lambda x: 2 * x,
        np.array([1.0]),
        np.array([-2.0]),
        alpha0=0.75,
    )
    assert a == 0.75


def test_armijo_keeps_halving():
    # WHY: each trial shrinks the PREVIOUS alpha by rho. On f(x) = x^2 at
    #      x = 1 along d = -4 (slope -8), alpha = 1 lands on -3 (f = 9) and
    #      alpha = 0.5 on -1 (f = 1, not below 1 - 4e-4); only the second
    #      halving, alpha = 0.25, lands on the minimum. A loop that restarts
    #      from alpha0 * rho every time never gets past 0.5.
    # KIND: unit
    # CATCHES: s12
    # CHAPTER: M10.1 section 5, Pitfalls, item 10
    a = armijo_step(
        lambda x: float(x[0] ** 2), lambda x: 2 * x, np.array([1.0]), np.array([-4.0])
    )
    assert a == 0.25


def test_armijo_rejects_nan_steps():
    # WHY: a step that leaves the function's domain returns nan (a log of a
    #      negative number, an overflow). Written as `while f(new) > bound`,
    #      the loop stops at nan, because every comparison with nan is
    #      False, and accepts the bad step. Written as `if f(new) <= bound:
    #      return`, nan fails and alpha keeps shrinking: here to 0.125.
    # KIND: boundary
    # CATCHES: s07
    # CHAPTER: M10.1 section 5, Pitfalls, item 2
    def f(x):
        return float(x[0] ** 2) if abs(x[0]) <= 1.2 else float("nan")

    a = armijo_step(f, lambda x: 2 * x, np.array([1.0]), np.array([-10.0]))
    assert a == 0.125


def test_armijo_needs_a_descent_direction():
    # WHY: if grad(x) . d >= 0, no small step decreases f, so backtracking
    #      would shrink alpha to nothing. The search refuses up front, and
    #      refuses parameters that make the condition meaningless.
    # KIND: boundary
    # CATCHES: s08, m02
    # CHAPTER: M10.1 section 5, Pitfalls, item 3
    f, g = quadratic(np.eye(2))
    x = np.array([1.0, 0.0])
    for d in (np.array([1.0, 0.0]), np.array([0.0, 1.0])):
        with pytest.raises(ValueError):
            armijo_step(f, g, x, d)
    for kw in (
        dict(c=0.0),
        dict(c=1.0),
        dict(rho=1.0),
        dict(rho=0.0),
        dict(alpha0=0.0),
    ):
        with pytest.raises(ValueError):
            armijo_step(f, g, x, -x, **kw)


def test_armijo_gives_up():
    # WHY: with a gradient that lies (it claims d descends while f actually
    #      rises), no alpha ever qualifies. After 60 halvings the search
    #      raises instead of looping forever or returning a step of 1e-18.
    #      The 60th halving itself is still tried: a function that accepts
    #      only steps up to 2^-60 gets exactly 2^-60 back.
    # KIND: boundary
    # CATCHES: s10, s12, m03
    # CHAPTER: M10.1 section 4, The interface
    with pytest.raises(RuntimeError):
        armijo_step(
            lambda x: float(x[0]),
            lambda x: np.array([-1.0]),
            np.array([0.0]),
            np.array([1.0]),
        )

    def tiny_only(x):
        return float(-x[0]) if x[0] <= 2.0**-60 else float(x[0])

    a = armijo_step(
        tiny_only, lambda x: np.array([-1.0]), np.array([0.0]), np.array([1.0])
    )
    assert a == 2.0**-60


def test_line_search_descends_rosenbrock():
    # WHY: steepest descent with an Armijo step needs no knowledge of L and
    #      decreases f at every step, even on the badly conditioned
    #      Rosenbrock valley where any fixed lr either crawls or diverges.
    # KIND: property
    # CATCHES: s06
    # CHAPTER: M10.1 section 2, Principles (line search)
    x = np.array([-1.2, 1.0])
    fs = [rosenbrock(x)]
    for _ in range(300):
        d = -rosenbrock_grad(x)
        x = x + armijo_step(rosenbrock, rosenbrock_grad, x, d, c=0.3) * d
        fs.append(rosenbrock(x))
    assert all(b < a for a, b in zip(fs, fs[1:]))
    assert fs[-1] < 0.1 * fs[0]
