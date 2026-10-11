"""Course tests for M04.2: Jacobians, the multivariable chain rule, and
numeric JVP and VJP (tinyllm/num/jacobian.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M04.2), and the chapter section it comes from.

The worked example of the chapter (section 3) is f(x, y) = (x^2 y, 5x + sin y)
at (1, 2): J = [[2xy, x^2], [5, cos y]] = [[4, 1], [5, -0.4161468]],
J v for v = (1, 0) is the first column (4, 5), and u^T J for u = (1, 0) is
the first row (4, 1).
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.num.jacobian import jacobian, jvp_numeric, vjp_numeric

SEED = int(os.environ.get("OL_SEED", "0"))
TOL = {"rtol": 1e-7, "atol": 1e-8}  # central differences at eps 1e-6, float64


def hand_f(v: np.ndarray) -> np.ndarray:
    x, y = v
    return np.array([x * x * y, 5 * x + math.sin(y)])


HAND_X = np.array([1.0, 2.0])
HAND_J = np.array([[4.0, 1.0], [5.0, math.cos(2.0)]])


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


# --- the worked example -------------------------------------------------------


def test_hand_example():
    # WHY: the chapter's worked example: row i of J is the gradient of output
    #      i, column j is how every output moves when input j moves. Rows
    #      and columns are not interchangeable: J is not symmetric here.
    # KIND: unit
    # CATCHES: s01, s02, s09, m01
    # CHAPTER: M04.2 section 3, Worked example by hand
    J = jacobian(hand_f, HAND_X)
    assert J.shape == (2, 2) and J.dtype == np.float64
    assert_close(J, HAND_J, **TOL)


def test_hand_example_products():
    # WHY: J v with v = e_1 is the first column (4, 5); u^T J with u = e_1 is
    #      the first row (4, 1). Mixing the two up is the most common
    #      autodiff bug; on this example the answers differ.
    # KIND: unit
    # CATCHES: s03, s04
    # CHAPTER: M04.2 section 3, Worked example by hand
    e1 = np.array([1.0, 0.0])
    assert_close(jvp_numeric(hand_f, HAND_X, e1), [4.0, 5.0], **TOL)
    assert_close(vjp_numeric(hand_f, HAND_X, e1), [4.0, 1.0], **TOL)


# --- shapes -------------------------------------------------------------------


def test_shapes_flatten_row_major():
    # WHY: x of shape (2, 3) and f(x) of shape (4,) give J of shape (4, 6)
    #      with columns in row-major order of x: J[:, 1 * 3 + 2] is the
    #      derivative with respect to x[1, 2]. vjp has x's shape, jvp has
    #      f(x)'s shape, so they plug straight into backward passes.
    # KIND: unit
    # CATCHES: s05, m02
    # CHAPTER: M04.2 section 2.1, The Jacobian
    rng = PCG32(seed=SEED)
    A = rng.normal_array((4, 6))
    f = lambda X: A @ X.reshape(-1)  # noqa: E731
    X = rng.normal_array((2, 3))
    J = jacobian(f, X)
    assert J.shape == (4, 6)
    assert_close(J, A, **TOL)
    assert vjp_numeric(f, X, np.ones(4)).shape == (2, 3)
    assert jvp_numeric(f, X, np.ones((2, 3))).shape == (4,)
    with pytest.raises(ValueError):
        jvp_numeric(f, X, np.ones(6))
    with pytest.raises(ValueError):
        vjp_numeric(f, X, np.ones(3))


# --- oracles ------------------------------------------------------------------


def test_linear_map_jacobian_is_the_matrix():
    # WHY: for f(x) = A x the Jacobian is A itself, everywhere; central
    #      differences are exact on linear maps up to rounding. The matmul of
    #      M03.1 is the linear map every layer starts with.
    # KIND: golden
    # CATCHES: s01, s06
    # CHAPTER: M04.2 section 2.1, The Jacobian
    rng = PCG32(seed=SEED)
    A = rng.normal_array((3, 5))
    x = rng.normal_array((5,))
    assert_close(jacobian(lambda v: A @ v, x), A, rtol=1e-8, atol=1e-9)


def test_softmax_jacobian_closed_form():
    # WHY: softmax has J = diag(p) - p p^T (section 2.4): symmetric, rows sum
    #      to 0 (the probabilities always sum to 1). The closed form is the
    #      oracle; M08.3 derives the same VJP by hand.
    # KIND: golden
    # CATCHES: s06
    # CHAPTER: M04.2 section 2.4, A Jacobian you will meet again
    rng = PCG32(seed=SEED)
    z = rng.normal_array((5,))
    p = softmax(z)
    J = jacobian(softmax, z)
    assert_close(J, np.diag(p) - np.outer(p, p), **TOL)
    assert_close(J.sum(axis=0), np.zeros(5), rtol=0, atol=1e-9)


def test_elementwise_jacobian_is_diagonal():
    # WHY: tanh applied elementwise only couples x_i with y_i: J is
    #      diag(1 - tanh(x)^2). Backward passes of activations (M01.3) never
    #      build this n x n matrix; they multiply elementwise, which is u^T J.
    # KIND: golden
    # CATCHES: s02, s06
    # CHAPTER: M04.2 section 2.3, Products without the matrix
    rng = PCG32(seed=SEED)
    x = rng.uniform_array((6,), -2.0, 2.0)
    J = jacobian(np.tanh, x)
    assert_close(J, np.diag(1 - np.tanh(x) ** 2), **TOL)
    u = rng.normal_array((6,))
    assert_close(vjp_numeric(np.tanh, x, u), u * (1 - np.tanh(x) ** 2), **TOL)


# --- the chain rule and the two products --------------------------------------


def test_chain_rule_is_a_matrix_product():
    # WHY: the multivariable chain rule: J_{g o f}(x) = J_g(f(x)) J_f(x),
    #      a matrix product in that order (outer function on the left). The
    #      product in the other order does not even have the right shape here.
    # KIND: differential
    # CATCHES: s01, s02, s09
    # CHAPTER: M04.2 section 2.2, The chain rule
    rng = PCG32(seed=SEED)
    A = rng.normal_array((4, 3))
    B = rng.normal_array((2, 4))
    f = lambda x: np.tanh(A @ x)  # noqa: E731  R^3 -> R^4
    g = lambda y: B @ (y * y)  # noqa: E731  R^4 -> R^2
    x = rng.normal_array((3,))
    Jgf = jacobian(lambda v: g(f(v)), x)
    assert Jgf.shape == (2, 3)
    assert_close(Jgf, jacobian(g, f(x)) @ jacobian(f, x), rtol=1e-6, atol=1e-7)


def test_jvp_and_vjp_agree():
    # WHY: the design's property: u^T (J v) = (u^T J) v for every u and v,
    #      one number computed two ways (forward mode and reverse mode). Both
    #      also agree with the explicit J. Seeded random u, v, x.
    # KIND: property
    # CATCHES: s03, s04, s07, m03
    # CHAPTER: M04.2 section 2.3, Products without the matrix
    rng = PCG32(seed=SEED)
    A = rng.normal_array((3, 4))
    f = lambda x: np.sin(A @ x) * x[0]  # noqa: E731  R^4 -> R^3
    for _ in range(5):
        x = rng.normal_array((4,))
        u = rng.normal_array((3,))
        v = rng.normal_array((4,))
        J = jacobian(f, x)
        jv = jvp_numeric(f, x, v)
        uj = vjp_numeric(f, x, u)
        assert_close(jv, J @ v, rtol=1e-6, atol=1e-7)
        assert_close(uj, u @ J, rtol=1e-6, atol=1e-7)
        assert_close(u @ jv, uj @ v, rtol=1e-6, atol=1e-7)


def test_jvp_scales_its_step_to_v():
    # WHY: J (c v) = c J v exactly. With v of size 1e6, a fixed step of 1e-6
    #      moves x by 1 and the difference quotient measures the secant over
    #      a whole unit, not the tangent; the step must shrink with max|v|.
    #      v = 0 gives 0 without evaluating a 0 / 0.
    # KIND: boundary
    # CATCHES: m03, m04
    # CHAPTER: M04.2 section 5, Pitfalls, item 3
    x = np.array([0.3, -0.7])
    v = np.array([1.0, 2.0])
    f = lambda z: np.array([np.sin(z[0] * z[1]), np.exp(z[0]) * z[1] ** 2])  # noqa: E731
    base = jvp_numeric(f, x, v)
    assert_close(jvp_numeric(f, x, 1e6 * v), 1e6 * base, rtol=1e-6, atol=0)
    assert_close(jvp_numeric(f, x, 1e-6 * v), 1e-6 * base, rtol=1e-6, atol=0)
    assert jvp_numeric(f, x, np.zeros(2)).tolist() == [0.0, 0.0]


# --- hygiene ------------------------------------------------------------------


def test_inputs_are_left_unchanged_and_float64():
    # WHY: x is perturbed one coordinate at a time on a float64 copy: the
    #      caller's array (an int array here) comes back untouched, and f
    #      always sees float64, or an int x would round every step to 0.
    # KIND: unit
    # CATCHES: s08, m05
    # CHAPTER: M04.2 section 5, Pitfalls, item 2
    x = np.array([1, 2])
    x0 = x.copy()
    seen = []

    def f(v):
        seen.append(v.dtype)
        return hand_f(v)

    J = jacobian(f, x)
    assert_close(J, HAND_J, **TOL)
    assert x.tobytes() == x0.tobytes()
    assert set(seen) == {np.dtype(np.float64)}
    xf = HAND_X.copy()
    jacobian(hand_f, xf)
    jvp_numeric(hand_f, xf, np.ones(2))
    vjp_numeric(hand_f, xf, np.ones(2))
    assert xf.tobytes() == HAND_X.tobytes()


def test_rejects_bad_eps_and_changing_shapes():
    # WHY: eps <= 0 is no step at all, and an f whose output shape changes
    #      with x has no Jacobian: ValueError instead of a ragged matrix.
    # KIND: boundary
    # CATCHES: m06
    # CHAPTER: M04.2 section 4, The interface
    with pytest.raises(ValueError):
        jacobian(hand_f, HAND_X, eps=0.0)
    with pytest.raises(ValueError):
        jacobian(lambda v: np.ones(2) if v[0] > 1.0 else np.ones(3), HAND_X)
