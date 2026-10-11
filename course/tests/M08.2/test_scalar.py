"""Course tests for M08.2: scalar reverse mode (tinyllm/autograd/scalar.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M08.2), and the chapter section it comes from.

The worked example of the chapter (section 3) is L = (a b + c) a at
a = 2, b = -3, c = 10: L = 8, dL/da = 2ab + c = -2, dL/db = a^2 = 4,
dL/dc = a = 2. The node a is used twice, so its gradient is a sum.

backward() walks the graph in the order of M06.1's toposort, and two tests
compare against M08.1's dual numbers and M04.2's numeric VJP, all through
the overlay.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.gradcheck import gradcheck
from _lib.pcg32 import PCG32
from tinyllm.autograd.dual import Dual, dual_exp, dual_log, dual_tanh
from tinyllm.autograd.scalar import Value


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def grads(f, *xs: float) -> tuple[float, list[float]]:
    """Run f on fresh leaves, backward, and return (value, [dL/dx_i])."""
    leaves = [Value(x) for x in xs]
    out = f(*leaves)
    out.backward()
    return out.data, [v.grad for v in leaves]


# Elementary functions that work on Value and on Dual, so one expression
# runs through both modes.
def exp(x):
    return x.exp() if isinstance(x, Value) else dual_exp(x)


def log(x):
    return x.log() if isinstance(x, Value) else dual_log(x)


def tanh(x):
    return x.tanh() if isinstance(x, Value) else dual_tanh(x)


def expression(x, y):
    """Every Value operation once: a small model-shaped expression."""
    h = tanh(x * y + 0.5) - (x / (1.0 + y * y)) ** 2
    return log(1.0 + exp(-h)) + 3.0 / (2.0 + x * x) - 0.25 * y


# --- the worked example --------------------------------------------------------


def test_hand_example():
    # WHY: the chapter's worked example, node by node: forward d = ab = -6,
    #      e = d + c = 4, L = e a = 8; backward from L.grad = 1 gives
    #      e.grad = a = 2, a.grad = e (from L) + b e.grad (from d) = 4 - 6 = -2,
    #      b.grad = a e.grad = 4, c.grad = e.grad = 2. The two paths into a
    #      add up.
    # KIND: unit
    # CATCHES: s01, s02, s03, m01
    # CHAPTER: M08.2 section 3, Worked example by hand
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    d = a * b
    e = d + c
    L = e * a
    L.backward()
    assert L.data == 8.0 and L.grad == 1.0
    assert e.grad == 2.0 and d.grad == 2.0
    assert (a.grad, b.grad, c.grad) == (-2.0, 4.0, 2.0)


def test_gradcheck_each_op():
    # WHY: one backward rule per operation, each checked against central
    #      differences (the frozen float64 gradcheck, never your M04.1) at
    #      three points: + - * / and their reflected forms, ** with integer,
    #      fractional, and negative exponents, exp, log, tanh, relu away from 0.
    # KIND: gradcheck
    # CATCHES: s02, s06, s08, s09, s10, m01
    # CHAPTER: M08.2 section 2, Principles (one rule per operation)
    ops = {
        "add": lambda x, y: x + y,
        "sub": lambda x, y: x - y,
        "rsub": lambda x, y: 2.0 - x * y,
        "mul": lambda x, y: x * y,
        "div": lambda x, y: x / y,
        "rdiv": lambda x, y: 3.0 / (x + y),
        "pow": lambda x, y: x**3 + y**0.5 + x**-2,
        "exp": lambda x, y: (x * y).exp(),
        "log": lambda x, y: (x * y).log(),
        "tanh": lambda x, y: (x - y).tanh(),
        "relu": lambda x, y: (x - y).relu() + (y - x).relu() * 2.0,
    }
    xs = np.array([0.7, 1.3, 2.1])
    ys = np.array([1.9, 0.4, 0.8])
    for name, op in ops.items():

        def f(x, y, op=op):
            return sum(op(Value(a), Value(b)).data for a, b in zip(x, y))

        gx, gy = np.zeros(3), np.zeros(3)
        for i in range(3):
            _, (gx[i], gy[i]) = grads(op, xs[i], ys[i])
        gradcheck(f, [xs, ys], [gx, gy], names=[f"{name} dx", f"{name} dy"])


def test_matches_dual():
    # WHY: forward mode (M08.1) and reverse mode compute the same derivative
    #      by opposite walks of the same chain rule. One expression using
    #      every operation, at 25 random points, agrees to roundoff in both
    #      partial derivatives.
    # KIND: differential
    # CATCHES: s02, s06, s08, s09, s10, s12, m02
    # CHAPTER: M08.2 section 2, Principles (forward vs reverse)
    rng = PCG32(seed=seed())
    for _ in range(25):
        x, y = rng.uniform() * 2 + 0.2, rng.uniform() * 2 - 1
        value, (gx, gy) = grads(expression, x, y)
        dx = expression(Dual(x, 1.0), Dual(y, 0.0))
        dy = expression(Dual(x, 0.0), Dual(y, 1.0))
        assert_close(value, dx.val)
        assert_close(gx, dx.eps, rtol=1e-12, atol=1e-15)
        assert_close(gy, dy.eps, rtol=1e-12, atol=1e-15)


def test_vjp_matches_numeric_vjp():
    # WHY: for a function with several outputs y = f(x), reverse mode gives
    #      u^T J in one backward pass from L = sum_i u_i y_i. M04.2's numeric
    #      VJP approximates the same row vector by central differences.
    # KIND: differential
    # CATCHES: s01, s03, m01
    # CHAPTER: M08.2 section 2, Principles (vector-Jacobian products)
    from tinyllm.num.jacobian import vjp_numeric

    rng = PCG32(seed=seed())

    def f_values(x):
        return [
            x[0] * x[1] + x[2].tanh(),
            (x[1] * x[2]).exp() - x[0],
            x[0] * x[0] * x[2] + 1.0,
        ]

    def f_np(x):
        return np.array(
            [
                x[0] * x[1] + np.tanh(x[2]),
                np.exp(x[1] * x[2]) - x[0],
                x[0] * x[0] * x[2] + 1.0,
            ]
        )

    for _ in range(5):
        x = rng.normal_array(3, scale=0.8)
        u = rng.normal_array(3)
        leaves = [Value(v) for v in x]
        ys = f_values(leaves)
        L = sum((float(ui) * yi for ui, yi in zip(u, ys)), Value(0.0))
        L.backward()
        got = np.array([v.grad for v in leaves])
        assert_close(got, vjp_numeric(f_np, x, u), rtol=1e-6, atol=1e-8)


# --- graph walking -------------------------------------------------------------


def test_reused_node_accumulates():
    # WHY: a node used k times receives k contributions. x + x has gradient
    #      2, x * x has 2x, and x * x + x has 2x + 1. Assigning (=) instead
    #      of accumulating (+=) keeps only the last contribution.
    # KIND: unit
    # CATCHES: s01, m01
    # CHAPTER: M08.2 section 5, Pitfalls, item 1
    assert grads(lambda x: x + x, 3.0)[1] == [2.0]
    assert grads(lambda x: x * x, 3.0)[1] == [6.0]
    assert grads(lambda x: x * x + x, 3.0)[1] == [7.0]


def test_diamond_visits_each_node_once():
    # WHY: a = e^x feeds b = 3a and c = 4a, and L = b + c. The walk must run
    #      a's backward once, after both b and c have added to a.grad
    #      (7 in total): run too early it pushes a partial gradient, run
    #      twice it doubles. dL/dx = 7 e^x.
    # KIND: unit
    # CATCHES: s03, s13
    # CHAPTER: M08.2 section 5, Pitfalls, item 2
    x = Value(0.5)
    a = x.exp()
    L = a * 3.0 + a * 4.0
    L.backward()
    assert_close(a.grad, 7.0)
    assert_close(x.grad, 7.0 * math.exp(0.5))


def test_deep_chain_no_recursion_error():
    # WHY: an unrolled RNN (L3.1) or a long training graph is a deep chain.
    #      A recursive topological sort hits Python's recursion limit (1000
    #      frames by default) long before 50000 nodes; backward must use the
    #      iterative toposort of M06.1.
    # KIND: boundary
    # CATCHES: s05
    # CHAPTER: M08.2 section 5, Pitfalls, item 3
    x = Value(1.0)
    y = x
    for _ in range(50_000):
        y = y * 1.0 + 0.0
    y.backward()
    assert x.grad == 1.0


def test_relu_gradient_at_zero():
    # WHY: relu is not differentiable at 0; the convention (PyTorch's) is a
    #      gradient of 0 there. Engines must agree, or a gradcheck at a kink
    #      differs from the reference by a whole unit.
    # KIND: boundary
    # CATCHES: s07
    # CHAPTER: M08.2 section 4, The interface
    assert grads(lambda x: x.relu(), 0.0) == (0.0, [0.0])
    assert grads(lambda x: x.relu(), -2.0) == (0.0, [0.0])
    assert grads(lambda x: x.relu(), 2.0) == (2.0, [1.0])


def test_constants_on_either_side():
    # WHY: 2 * x, x * 2, 1 - x, 6 / x, and 2 + x start from a Python number
    #      on one side; Python calls the reflected method, which must wrap the
    #      number as a constant leaf and keep the order of subtraction and
    #      division.
    # KIND: unit
    # CATCHES: s11, s12, m02
    # CHAPTER: M08.2 section 4, The interface
    cases = [
        (lambda x: 2.0 * x, 3.0, 6.0, 2.0),
        (lambda x: x * 2.0, 3.0, 6.0, 2.0),
        (lambda x: 2.0 + x, 3.0, 5.0, 1.0),
        (lambda x: 1.0 - x, 3.0, -2.0, -1.0),
        (lambda x: x - 1.0, 3.0, 2.0, 1.0),
        (lambda x: 6.0 / x, 3.0, 2.0, -6.0 / 9.0),
        (lambda x: x / 2.0, 3.0, 1.5, 0.5),
        (lambda x: -x, 3.0, -3.0, -1.0),
    ]
    for f, x, value, slope in cases:
        v, (g,) = grads(f, x)
        assert_close(v, value)
        assert_close(g, slope)


def test_pow_needs_a_constant_exponent():
    # WHY: Value ** k supports a constant k only; x ** y with y a Value needs
    #      the rule for exp(y log x), which this module does not implement.
    #      Raising beats silently treating y as a constant.
    # KIND: boundary
    # CATCHES: s10
    # CHAPTER: M08.2 section 4, The interface
    with pytest.raises(TypeError):
        Value(2.0) ** Value(3.0)
    assert grads(lambda x: x**2, 3.0) == (9.0, [6.0])


def test_backward_twice_accumulates():
    # WHY: backward adds into .grad and never resets the graph, exactly like
    #      PyTorch: a second call doubles every leaf gradient. Training loops
    #      (L0.5) zero gradients between steps for this reason.
    # KIND: unit
    # CATCHES: s04
    # CHAPTER: M08.2 section 4, The interface
    x = Value(3.0)
    y = x * x
    y.backward()
    assert x.grad == 6.0
    y.backward()
    assert x.grad == 12.0 and y.grad == 1.0
