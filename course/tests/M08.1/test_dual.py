"""Course tests for M08.1: dual numbers and forward mode (tinyllm/autograd/dual.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M08.1), and the chapter section it comes from.

The worked example of the chapter (section 3) is f(x) = x e^x + 3 at x = 1:
(1 + eps)(e + e eps) + 3 = (e + 3) + 2e eps, so f(1) = 5.71828 and
f'(1) = 2e = 5.43656.

Two tests call earlier modules through the overlay: M01.3's closed-form
activation derivatives and M04.2's numeric JVP (central differences).
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.autograd.dual import (
    Dual,
    derivative,
    dual_erf,
    dual_exp,
    dual_log,
    dual_tanh,
    jvp,
)

E = math.e
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
# A derivative computed two ways in float64 agrees to roundoff: relative
# 1e-12, and absolute 1e-14 where the value is near 0 (M01.3's erf, a
# 160-term series from M02.1, is within 3e-15 of the true erf, so 1 + erf(x)
# for x near -6 is a few 1e-15 off either way).
EXACT = dict(rtol=1e-12, atol=1e-14)


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


# The activations of L0.2, written only with Dual operations, so one
# forward pass gives the value and the derivative.
def d_sigmoid(x):
    return 1.0 / (1.0 + dual_exp(-x))


def d_silu(x):
    return x * d_sigmoid(x)


def d_softplus(x):
    return dual_log(1.0 + dual_exp(x))


def d_gelu_tanh(x):
    return 0.5 * x * (1.0 + dual_tanh(SQRT_2_OVER_PI * (x + 0.044715 * x**3)))


def d_gelu_erf(x):
    return 0.5 * x * (1.0 + dual_erf(x / math.sqrt(2.0)))


# --- the worked example --------------------------------------------------------


def test_hand_example():
    # WHY: the chapter's worked example: with x = 1 + eps and eps^2 = 0,
    #      x e^x + 3 = (1 + eps)(e + e eps) + 3 = (e + 3) + 2e eps. One
    #      evaluation carries the value 5.71828 and the derivative 5.43656.
    # KIND: unit
    # CATCHES: s01, s14
    # CHAPTER: M08.1 section 3, Worked example by hand
    y = Dual(1.0, 1.0) * dual_exp(Dual(1.0, 1.0)) + 3
    assert isinstance(y, Dual)
    assert_close(y.val, E + 3)
    assert_close(y.eps, 2 * E)
    assert_close(derivative(lambda x: x * dual_exp(x) + 3, 1.0), 2 * E)


@pytest.mark.parametrize(
    "name, f, x, value, slope",
    [
        ("add", lambda x: x + 2.5, 1.5, 4.0, 1.0),
        ("radd", lambda x: 2.5 + x, 1.5, 4.0, 1.0),
        ("sub", lambda x: x - 4.0, 1.5, -2.5, 1.0),
        ("rsub", lambda x: 4.0 - x, 1.5, 2.5, -1.0),
        ("mul", lambda x: x * x * x, 2.0, 8.0, 12.0),
        ("rmul", lambda x: 3.0 * x, 2.0, 6.0, 3.0),
        ("div", lambda x: (x + 1.0) / (x - 1.0), 3.0, 2.0, -0.5),
        ("rdiv", lambda x: 6.0 / x, 2.0, 3.0, -1.5),
        ("neg", lambda x: -x, 2.0, -2.0, -1.0),
        ("log", lambda x: dual_log(x), 4.0, math.log(4.0), 0.25),
        ("exp", lambda x: dual_exp(2.0 * x), 0.5, E, 2 * E),
        ("tanh", lambda x: dual_tanh(x), 0.5, math.tanh(0.5), 1 - math.tanh(0.5) ** 2),
        ("erf", lambda x: dual_erf(x), 1.0, math.erf(1.0), 2 / math.sqrt(math.pi) / E),
    ],
)
def test_arithmetic_rules(name, f, x, value, slope):
    # WHY: one rule per primitive, each checked against the derivative you
    #      know from calculus (M01): sum, product, quotient, the reflected
    #      forms where the constant is on the left (4 - x has slope -1), and
    #      the four elementary functions.
    # KIND: unit
    # CATCHES: s01, s02, s03, s04, s05, s06, s07, s14, m01
    # CHAPTER: M08.1 section 2, Principles (one rule per primitive)
    y = f(Dual(x, 1.0))
    assert isinstance(y, Dual), name
    assert_close(y.val, value, msg=name)
    assert_close(y.eps, slope, msg=name)


def test_pow_rules():
    # WHY: three different rules hide behind `**`: x^k for a constant k
    #      (k x^(k-1)), c^x for a constant base (c^x ln c), and x^x, where
    #      both depend on x: d(x^x) = x^x (ln x + 1), 4 (ln 2 + 1) at x = 2.
    # KIND: unit
    # CATCHES: s10, s11
    # CHAPTER: M08.1 section 2, Principles (powers)
    cases = [
        (lambda x: x**3, 2.0, 8.0, 12.0),
        (lambda x: x**0.5, 4.0, 2.0, 0.25),
        (lambda x: x**-1, 2.0, 0.5, -0.25),
        (lambda x: 2.0**x, 3.0, 8.0, 8.0 * math.log(2)),
        (lambda x: x**x, 2.0, 4.0, 4.0 * (math.log(2) + 1)),
    ]
    for f, x, value, slope in cases:
        y = f(Dual(x, 1.0))
        assert_close(y.val, value)
        assert_close(y.eps, slope)


def test_matches_activation_derivatives():
    # WHY: the activations of L0.2 written with Dual operations give the
    #      same derivatives as M01.3's closed forms, at 1001 points in
    #      [-40, 40], to roundoff. This is the derivative oracle L0.2 uses
    #      for its elementwise ops: exact, with no step size to tune.
    # KIND: differential
    # CATCHES: s01, s02, s05, s06, s14
    # CHAPTER: M08.1 section 6, Where it's used next
    from tinyllm.num import activations as act

    x = np.linspace(-40.0, 40.0, 1001)
    pairs = [
        (d_sigmoid, act.dsigmoid),
        (dual_tanh, act.dtanh),
        (d_silu, act.dsilu),
        (d_gelu_tanh, act.dgelu_tanh),
        (d_gelu_erf, act.dgelu_erf),
        (d_softplus, act.sigmoid),  # softplus' = sigmoid
    ]
    for f, df in pairs:
        y = f(Dual(x, np.ones_like(x)))
        assert_close(y.eps, df(x), msg=getattr(df, "__name__", "df"), **EXACT)
    assert_close(d_gelu_erf(Dual(x, 1.0)).val, act.gelu_erf(x), **EXACT)


def test_jvp_matches_numeric_jvp():
    # WHY: for f from R^4 to R^3 the dual pass gives J v exactly; M04.2's
    #      central-difference jvp_numeric approximates it to about 1e-9. The
    #      two agree, which is the differential test that used to live in
    #      M04.1, now with an exact side.
    # KIND: differential
    # CATCHES: s12, s13, m02
    # CHAPTER: M08.1 section 2, Principles (Jacobian-vector products)
    from tinyllm.num.jacobian import jvp_numeric

    rng = PCG32(seed=seed())
    W = rng.normal_array((3, 4), scale=0.7)

    def f(x):
        h = W @ x
        return dual_tanh(h) * x[1:] + dual_exp(0.1 * h) - x[:3] / (2.0 + x[3])

    def f_np(x):
        h = W @ x
        return np.tanh(h) * x[1:] + np.exp(0.1 * h) - x[:3] / (2.0 + x[3])

    for _ in range(5):
        x = rng.normal_array(4)
        v = rng.normal_array(4)
        y, jv = jvp(f, x, v)
        assert y.shape == (3,) and jv.shape == (3,)
        assert_close(y, f_np(x))
        assert_close(jv, jvp_numeric(f_np, x, v), rtol=1e-6, atol=1e-8)


def test_numpy_operand_on_the_left():
    # WHY: np.float64(2) * d and W @ d start in numpy. Without
    #      __array_ufunc__ = None, numpy builds an object array or a plain
    #      float and the tangent is lost silently; with it, numpy steps aside
    #      and Dual's reflected method runs.
    # KIND: boundary
    # CATCHES: s08
    # CHAPTER: M08.1 section 5, Pitfalls, item 3
    d = Dual(3.0, 1.0)
    for y in (np.float64(2.0) * d, d * np.float64(2.0)):
        assert isinstance(y, Dual)
        assert_close(y.val, 6.0)
        assert_close(y.eps, 2.0)
    y = np.float64(1.0) - d
    assert isinstance(y, Dual) and y.eps == -1.0
    v = Dual(np.array([1.0, 2.0]), np.array([1.0, 0.0]))
    z = np.array([10.0, 20.0]) + v
    assert isinstance(z, Dual)
    assert z.val.tolist() == [11.0, 22.0] and z.eps.tolist() == [1.0, 0.0]


def test_constant_function_has_zero_derivative():
    # WHY: a function that ignores its input returns a float, not a Dual;
    #      its derivative is 0, not an AttributeError. jvp gives zeros of
    #      the output's shape.
    # KIND: boundary
    # CATCHES: s09
    # CHAPTER: M08.1 section 4, The interface
    assert derivative(lambda x: 7.0, 2.0) == 0.0
    assert derivative(lambda x: x, 2.0) == 1.0
    y, jv = jvp(lambda x: np.ones(2), np.zeros(3), np.ones(3))
    assert y.tolist() == [1.0, 1.0] and jv.tolist() == [0.0, 0.0]
    with pytest.raises(TypeError):
        Dual(Dual(1.0, 1.0), 0.0)


def test_chain_rule_composition():
    # WHY: forward mode applies the chain rule one step at a time, so a
    #      composition of rules is the derivative of the composition:
    #      d/dx tanh(x e^x) = (1 - tanh(x e^x)^2)(1 + x) e^x, checked at
    #      40 random points.
    # KIND: property
    # CATCHES: s01, s05, s14
    # CHAPTER: M08.1 section 2, Principles (the chain rule)
    rng = PCG32(seed=seed())
    for x in rng.uniform_array(40, -2.0, 2.0):
        u = x * math.exp(x)
        expect = (1 - math.tanh(u) ** 2) * (1 + x) * math.exp(x)
        assert_close(derivative(lambda t: dual_tanh(t * dual_exp(t)), float(x)), expect)


def test_jvp_is_linear_in_v():
    # WHY: J v is linear in v: jvp(a v1 + b v2) = a jvp(v1) + b jvp(v2), and
    #      the basis vectors e_i give the columns of the Jacobian. Forward
    #      mode builds a full Jacobian with one pass per input.
    # KIND: property
    # CATCHES: s01, s12, m02
    # CHAPTER: M08.1 section 2, Principles (cost of forward mode)
    rng = PCG32(seed=seed())
    A = rng.normal_array((2, 3))

    def f(x):
        return dual_exp(A @ x) * (A @ x)

    x = rng.normal_array(3)
    v1, v2 = rng.normal_array(3), rng.normal_array(3)
    _, j1 = jvp(f, x, v1)
    _, j2 = jvp(f, x, v2)
    _, j12 = jvp(f, x, 2.0 * v1 - 3.0 * v2)
    assert_close(j12, 2.0 * j1 - 3.0 * j2)
    h = A @ x
    J = (np.exp(h) * (1 + h))[:, None] * A
    cols = np.stack([jvp(f, x, np.eye(3)[i])[1] for i in range(3)], axis=1)
    assert_close(cols, J)


def test_vector_duals_index_and_matmul():
    # WHY: a vector of dual numbers carries one tangent per entry. Indexing
    #      and slicing must move val and eps together, and W @ x maps the
    #      tangent by the same W (a linear map is its own derivative).
    # KIND: unit
    # CATCHES: s12, s13
    # CHAPTER: M08.1 section 4, The interface
    x = Dual(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 0.0]))
    assert x[1].val == 2.0 and x[1].eps == 1.0
    tail = x[1:]
    assert tail.val.tolist() == [2.0, 3.0] and tail.eps.tolist() == [1.0, 0.0]
    W = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 4.0], [5.0, 0.0, 1.0]])
    y = W @ x
    assert isinstance(y, Dual)
    assert y.val.tolist() == [14.0, 10.0, 8.0] and y.eps.tolist() == [2.0, -1.0, 0.0]
    z = x @ W.T
    assert z.val.tolist() == [14.0, 10.0, 8.0] and z.eps.tolist() == [2.0, -1.0, 0.0]
    s = Dual(np.array([1.0, 2.0]))
    assert s.eps.tolist() == [0.0, 0.0]
