"""Course tests for M04.1: partial derivatives, gradients, and gradcheck
(tinyllm/num/gradcheck.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M04.1), and the chapter section it comes from.

This module IS the learner's gradcheck, so these tests are the one place a
course test imports it (D35: the module's own verdict). They import it as
`from tinyllm.num import gradcheck`; every other course test uses the frozen
course/tests/_lib/gradcheck.py. Expected gradients here come from closed
forms, never from the code under test.

The worked example of the chapter (section 3) is f(x) = x0^2 x1 + 3 x1 at
x = (1, 2): gradient (2 x0 x1, x0^2 + 3) = (4, 4), and central differences
with eps = 0.1 give exactly (4, 4) because f is quadratic in x0 and linear
in x1.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32

from tinyllm.num import gradcheck as gc

SEED = int(os.environ.get("OL_SEED", "0"))


def hand_f(x: np.ndarray) -> float:
    return float(x[0] ** 2 * x[1] + 3 * x[1])


def tanh_matmul(W: np.ndarray, X: np.ndarray) -> float:
    """sum(tanh(W @ X)): a two-input scalar loss with non-square W."""
    return float(np.sum(np.tanh(W @ X)))


def tanh_matmul_grads(W: np.ndarray, X: np.ndarray) -> list[np.ndarray]:
    G = 1 - np.tanh(W @ X) ** 2  # d sum(tanh(Z)) / dZ
    return [G @ X.T, W.T @ G]


# --- the worked example -------------------------------------------------------


def test_hand_example():
    # WHY: the chapter's worked example: partial derivatives one coordinate
    #      at a time. With eps = 0.1, (f(1.1, 2) - f(0.9, 2)) / 0.2 = 4 and
    #      (f(1, 2.1) - f(1, 1.9)) / 0.2 = 4: exact, because central
    #      differences are exact on quadratics.
    # KIND: unit
    # CATCHES: s01, s02, s03, m01
    # CHAPTER: M04.1 section 3, Worked example by hand
    x = np.array([1.0, 2.0])
    (g,) = gc.numerical_grad(hand_f, [x], eps=0.1)
    assert g.dtype == np.float64 and g.shape == (2,)
    assert_close(g, [4.0, 4.0], rtol=0, atol=1e-13)
    rep = gc.gradcheck(hand_f, [x], [np.array([4.0, 4.0])])
    assert rep.ok is True
    assert rep.max_abs_err < 1e-8


def test_exact_on_quadratics():
    # WHY: for f(x) = x^T A x / 2 + b^T x the central difference has no
    #      truncation error at all (the third derivative is 0), so the
    #      numerical gradient equals A x + b up to rounding: the design's
    #      property test. A one-sided difference is off by eps * A_ii / 2.
    # KIND: property
    # CATCHES: s01, s02
    # CHAPTER: M04.1 section 2.3, Central differences per coordinate
    rng = PCG32(seed=SEED)
    for n in (1, 3, 8):
        M = rng.normal_array((n, n))
        A = M + M.T
        b = rng.normal_array((n,))
        x = rng.normal_array((n,))
        (g,) = gc.numerical_grad(lambda v: float(0.5 * v @ A @ v + b @ v), [x])
        assert_close(g, A @ x + b, rtol=0, atol=1e-8)


def test_matrix_inputs_and_several_inputs():
    # WHY: a loss of two matrices, sum(tanh(W X)), with W 2 x 3 and X 3 x 2:
    #      one gradient per input, each in its input's shape, matching the
    #      closed forms G X^T and W^T G. Layers have weights and inputs, and
    #      each gets its own gradient.
    # KIND: golden
    # CATCHES: s03, s04, m02
    # CHAPTER: M04.1 section 2.2, The gradient
    rng = PCG32(seed=SEED)
    W = rng.normal_array((2, 3))
    X = rng.normal_array((3, 2))
    gW, gX = gc.numerical_grad(tanh_matmul, [W, X])
    want = tanh_matmul_grads(W, X)
    assert gW.shape == (2, 3) and gX.shape == (3, 2)
    assert_close(gW, want[0], rtol=1e-8, atol=1e-9)
    assert_close(gX, want[1], rtol=1e-8, atol=1e-9)
    rep = gc.gradcheck(tanh_matmul, [W, X], want)
    assert rep.ok and rep.max_abs_err < 1e-8


# --- planted wrong gradients (DESIGN M04.1 `U`) -------------------------------


def test_rejects_transposed_gradient():
    # WHY: the classic backward bug: the gradient of a weight returned
    #      transposed. W is square here so the shapes agree and only the
    #      numbers can tell; gradcheck must say not ok and point at the
    #      first input.
    # KIND: unit
    # CATCHES: s05, m02
    # CHAPTER: M04.1 section 5, Pitfalls, item 1
    rng = PCG32(seed=SEED)
    W = rng.normal_array((3, 3))
    X = rng.normal_array((3, 2))
    gW, gX = tanh_matmul_grads(W, X)
    rep = gc.gradcheck(tanh_matmul, [W, X], [gW.T, gX])
    assert rep.ok is False
    assert rep.worst_input == 0
    assert rep.max_abs_err > 1e-3


def test_rejects_gradient_off_by_two():
    # WHY: a dropped or doubled factor of 2 (d x^2 = x instead of 2x) keeps
    #      every sign and direction right; only the size is wrong.
    # KIND: unit
    # CATCHES: s02, m08
    # CHAPTER: M04.1 section 5, Pitfalls, item 1
    x = np.array([0.3, -1.2, 2.0])
    f = lambda v: float(np.sum(v**2))  # noqa: E731
    assert gc.gradcheck(f, [x], [2 * x]).ok is True
    rep = gc.gradcheck(f, [x], [x])
    assert rep.ok is False
    assert rep.worst_index == (2,)  # |2x - x| / (atol + rtol |2x|) is largest at x = 2
    assert_close(rep.max_abs_err, 2.0, rtol=1e-6, atol=0)
    assert_close(rep.max_rel_err, 0.5, rtol=1e-6, atol=0)


def test_rejects_one_zeroed_coordinate_and_locates_it():
    # WHY: a backward that forgets one element (a broadcast that drops a
    #      row, a mask off by one) is right everywhere else: the check must
    #      fail on that one element of the second input and report its
    #      multi-index, so you know where to look.
    # KIND: unit
    # CATCHES: s05, s06, m03
    # CHAPTER: M04.1 section 5, Pitfalls, item 1
    rng = PCG32(seed=SEED)
    W = rng.normal_array((2, 3))
    X = rng.normal_array((3, 2))
    gW, gX = tanh_matmul_grads(W, X)
    bad = gX.copy()
    bad[2, 1] = 0.0
    rep = gc.gradcheck(tanh_matmul, [W, X], [gW, bad])
    assert rep.ok is False
    assert rep.worst_input == 1
    assert rep.worst_index == (2, 1)
    assert_close(rep.max_abs_err, abs(gX[2, 1]), rtol=1e-6, atol=0)


def test_tolerance_is_atol_plus_rtol():
    # WHY: an element passes when |a - n| <= atol + rtol |n|. Near a
    #      gradient of 1e-3 the allowance is atol-dominated (1.1e-7): an
    #      error of 9e-8 passes and 2e-7 fails. At a gradient of 100 it grows
    #      to 1e-3: 9e-4 passes, 2e-3 fails, and passes again with rtol 1e-4.
    # KIND: boundary
    # CATCHES: m04
    # CHAPTER: M04.1 section 2.4, When is a gradient right?
    x0 = np.array([0.0])
    small = lambda v: float(1e-3 * v[0])  # noqa: E731
    assert gc.gradcheck(small, [x0], [np.array([1e-3 + 9e-8])]).ok is True
    assert gc.gradcheck(small, [x0], [np.array([1e-3 + 2e-7])]).ok is False
    x1 = np.array([50.0])
    big = lambda v: float(v[0] ** 2)  # gradient 100  # noqa: E731
    assert gc.gradcheck(big, [x1], [np.array([100 + 9e-4])]).ok is True
    assert gc.gradcheck(big, [x1], [np.array([100 + 2e-3])]).ok is False
    assert gc.gradcheck(big, [x1], [np.array([100 + 2e-3])], rtol=1e-4).ok is True


def test_worst_element_is_the_most_out_of_tolerance():
    # WHY: the report points at the element furthest past its own
    #      allowance, not the largest absolute error: an error of 5e-3 on a
    #      gradient of 1000 is fine (allowance 1e-2), an error of 2e-4 on a
    #      gradient of 1e-3 is a bug. Pointing at the first would send you to
    #      the wrong place.
    # KIND: unit
    # CATCHES: s07
    # CHAPTER: M04.1 section 2.4, When is a gradient right?
    x = np.array([1.0, 1.0])
    f = lambda v: float(1000 * v[0] + 1e-3 * v[1])  # noqa: E731
    rep = gc.gradcheck(f, [x], [np.array([1000.005, 1.2e-3])])
    assert rep.ok is False
    assert rep.worst_input == 0 and rep.worst_index == (1,)
    assert_close(rep.max_abs_err, 5e-3, rtol=1e-4, atol=0)


def test_nan_gradient_fails():
    # WHY: a nan in a backward pass is the loudest bug there is, and a
    #      comparison with nan is always False: `diff > tol` would let it
    #      through. The check must count nan as a failure.
    # KIND: boundary
    # CATCHES: s08
    # CHAPTER: M04.1 section 5, Pitfalls, item 4
    x = np.array([1.0, 2.0])
    rep = gc.gradcheck(hand_f, [x], [np.array([4.0, np.nan])])
    assert rep.ok is False


# --- the inputs are not yours to change ---------------------------------------


def test_inputs_are_left_unchanged():
    # WHY: numerical_grad perturbs one element at a time; the caller's arrays
    #      (model parameters, in L0.2) must come back bit for bit, and f must
    #      see every other element at its original value while one moves.
    # KIND: unit
    # CATCHES: s09, m02
    # CHAPTER: M04.1 section 5, Pitfalls, item 2
    rng = PCG32(seed=SEED)
    W = rng.normal_array((2, 3))
    X = rng.normal_array((3, 2))
    W0, X0 = W.copy(), X.copy()
    gc.numerical_grad(tanh_matmul, [W, X])
    assert W.tobytes() == W0.tobytes() and X.tobytes() == X0.tobytes()
    gc.gradcheck(tanh_matmul, [W, X], tanh_matmul_grads(W, X))
    assert W.tobytes() == W0.tobytes() and X.tobytes() == X0.tobytes()


def test_integer_and_float32_inputs_are_promoted():
    # WHY: x + 1e-6 on an int array rounds back to x (gradient 0), and on
    #      float32 it moves x by a multiple of 6e-8 |x| (gradient off in the
    #      second digit). Perturbing float64 copies gives the true gradient
    #      whatever dtype the caller holds.
    # KIND: boundary
    # CATCHES: s10
    # CHAPTER: M04.1 section 5, Pitfalls, item 3
    xi = np.array([3, 4])
    (gi,) = gc.numerical_grad(lambda v: float(v @ v), [xi])
    assert gi.dtype == np.float64
    assert_close(gi, [6.0, 8.0], rtol=0, atol=1e-8)
    x32 = np.array([0.7, -1.3, 2.9], dtype=np.float32)
    seen = []

    def f(v):
        seen.append(v.dtype)
        return float(np.sum(np.sin(v)))

    (g32,) = gc.numerical_grad(f, [x32])
    assert set(seen) == {np.dtype(np.float64)}
    assert_close(g32, np.cos(x32.astype(np.float64)), rtol=0, atol=1e-9)


def test_default_eps_and_tolerances():
    # WHY: the defaults (eps 1e-6, rtol 1e-5, atol 1e-7) are the ones every
    #      later backward is held to, and the ones L0.2's gradcheck_all
    #      uses. A different default eps changes every numerical gradient.
    # KIND: unit
    # CATCHES: m05
    # CHAPTER: M04.1 section 4, The interface
    x = np.array([0.5])
    calls = []

    def f(v):
        calls.append(float(v[0]))
        return float(v[0] ** 3)

    gc.numerical_grad(f, [x])
    assert_close(sorted(calls), [0.5 - 1e-6, 0.5 + 1e-6], rtol=0, atol=1e-15)
    rep = gc.gradcheck(lambda v: float(v[0] ** 3), [x], [np.array([0.75 + 5e-6])])
    assert rep.ok is True  # 5e-6 <= 1e-7 + 1e-5 * 0.75


# --- argument checks ----------------------------------------------------------


def test_rejects_mismatched_arguments():
    # WHY: a gradient list shorter than the inputs, or a gradient in the
    #      wrong shape, is a bug in the caller's backward, not a numeric
    #      disagreement: ValueError, before any evaluation of f.
    # KIND: boundary
    # CATCHES: m06
    # CHAPTER: M04.1 section 4, The interface
    x = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        gc.gradcheck(hand_f, [x], [])
    with pytest.raises(ValueError):
        gc.gradcheck(hand_f, [x], [np.zeros(3)])
    with pytest.raises(ValueError):
        gc.gradcheck(hand_f, [x], [np.zeros((2, 1))])
    with pytest.raises(ValueError):
        gc.numerical_grad(hand_f, [])
    with pytest.raises(ValueError):
        gc.numerical_grad(hand_f, [x], eps=0.0)


def test_rejects_non_scalar_f():
    # WHY: a gradient exists for a scalar function; f returning a vector is a
    #      Jacobian question (M04.2). Summing it silently would check the
    #      wrong thing.
    # KIND: boundary
    # CATCHES: m07
    # CHAPTER: M04.1 section 2.2, The gradient
    with pytest.raises(ValueError):
        gc.numerical_grad(lambda v: v * 2.0, [np.array([1.0, 2.0])])
    (g,) = gc.numerical_grad(lambda v: np.array([[v[0] * 3.0]]), [np.array([1.0])])
    assert_close(g, [3.0], rtol=0, atol=1e-9)
