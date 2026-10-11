"""Course tests for L0.1: the autograd Tensor and grad mode
(tinyllm/autograd/tensor.py, tinyllm/autograd/mode.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/L0.1), and the chapter section it comes from.

The worked example of the chapter (section 3): x = [[1, 2, 3], [4, 5, 6]],
w = [10, 20, 30], y = x * w + w, upstream gradient all ones.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.gradcheck import gradcheck
from _lib.pcg32 import PCG32
from tinyllm.autograd.mode import is_grad_enabled, no_grad
from tinyllm.autograd.scalar import Value
from tinyllm.autograd.tensor import Tensor, from_op

F64 = np.float64


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def check_op(fn, *shapes, lo=-2.0, hi=2.0, salt=0):
    """Gradcheck fn (a function of Tensors) against the frozen central
    differences: random float64 inputs, a random upstream weight w, and the
    scalar sum(fn(...) * w), whose gradient is backward(w)."""
    rng = PCG32(seed=seed() * 1000 + salt)
    xs = [rng.uniform_array(s, lo, hi) for s in shapes]
    out = fn(*[Tensor(x, dtype=F64) for x in xs])
    w = rng.uniform_array(out.shape, -1.0, 1.0)

    def f(*arrays):
        return float((fn(*[Tensor(a, dtype=F64) for a in arrays]).data * w).sum())

    ts = [Tensor(x, requires_grad=True, dtype=F64) for x in xs]
    fn(*ts).backward(w)
    gradcheck(f, xs, [t.grad for t in ts])


# --- the worked example ---------------------------------------------------------


def test_hand_example_broadcast_backward():
    # WHY: the chapter's worked example, number for number. w is broadcast
    #      down two rows AND used twice (in x * w and in + w), so its gradient
    #      is the column sum of x plus 2 per column: [7, 9, 11]. x's gradient
    #      is w copied to every row.
    # KIND: unit
    # CATCHES: s01, s02, s17
    # CHAPTER: L0.1 section 3, Worked example by hand
    x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    w = Tensor([10, 20, 30], requires_grad=True)
    y = x * w + w
    assert y.shape == (2, 3)
    assert_close(y.data, [[20, 60, 120], [50, 120, 210]], dtype="float32")
    y.backward(np.ones((2, 3)))
    assert w.grad.shape == (3,) and x.grad.shape == (2, 3)
    assert_close(w.grad, [7, 9, 11], dtype="float32")
    assert_close(x.grad, [[10, 20, 30], [10, 20, 30]], dtype="float32")


def test_matches_scalarized_value():
    # WHY: the same expression built element by element from M08.2's scalar
    #      Value must give the same gradients: a Tensor op is a batch of
    #      scalar ops, and broadcasting reuses one scalar in several places.
    # KIND: differential
    # CATCHES: s01
    # CHAPTER: L0.1 section 2, Principles (broadcasting is reuse)
    rng = PCG32(seed=seed())
    xa = rng.uniform_array((2, 3), -1, 1)
    ba = rng.uniform_array((3,), -1, 1)
    ca = rng.uniform_array((2, 3), -1, 1)  # upstream weights
    xv = [[Value(float(xa[i, j])) for j in range(3)] for i in range(2)]
    bv = [Value(float(ba[j])) for j in range(3)]
    loss = Value(0.0)
    for i in range(2):
        for j in range(3):
            loss = loss + (xv[i][j] * bv[j] + bv[j]) * float(ca[i, j])
    loss.backward()
    x = Tensor(xa, requires_grad=True, dtype=F64)
    b = Tensor(ba, requires_grad=True, dtype=F64)
    (x * b + b).backward(ca)
    assert_close(x.grad, [[xv[i][j].grad for j in range(3)] for i in range(2)])
    assert_close(b.grad, [bv[j].grad for j in range(3)])


# --- each op against central differences -----------------------------------------


@pytest.mark.parametrize(
    "shapes",
    [((3,), (3,)), ((2, 3), (3,)), ((2, 1), (1, 4)), ((), (2, 2)), ((4, 1, 3), (2, 1))],
    ids=["same", "row", "outer", "scalar", "rank3"],
)
def test_add_sub_mul_grads(shapes):
    # WHY: every binary op under every broadcasting pattern: equal shapes, a
    #      row added to a matrix, (2,1) with (1,4), a 0-d scalar, rank 3 with
    #      a missing leading axis. Each needs the gradient summed back.
    # KIND: gradcheck
    # CATCHES: s01, s20, m05
    # CHAPTER: L0.1 section 2, Principles (unbroadcast)
    check_op(lambda a, b: a + b, *shapes, salt=1)
    check_op(lambda a, b: a - b, *shapes, salt=2)
    check_op(lambda a, b: a * b, *shapes, salt=3)
    check_op(lambda a, b: -(a * b) - a, *shapes, salt=4)


def test_div_grads():
    # WHY: d(a/b)/db = -a/b^2: the sign and the square are the classic slips.
    #      The denominator stays in [0.5, 2] so the quotient is smooth.
    # KIND: gradcheck
    # CATCHES: s08
    # CHAPTER: L0.1 section 5, Pitfalls, item 8
    rng = PCG32(seed=seed())
    a = rng.uniform_array((2, 3), -2, 2)
    b = rng.uniform_array((3,), 0.5, 2)
    w = rng.uniform_array((2, 3), -1, 1)
    ta = Tensor(a, requires_grad=True, dtype=F64)
    tb = Tensor(b, requires_grad=True, dtype=F64)
    (ta / tb).backward(w)

    def f(x, y):
        return float((x / y * w).sum())

    gradcheck(f, [a, b], [ta.grad, tb.grad])


@pytest.mark.parametrize("p", [2.0, 3.0, 0.5, -1.0])
def test_pow_grads(p):
    # WHY: d(x^p)/dx = p x^(p-1) for integer, fractional, and negative p
    #      (inputs in [0.5, 2] so x^0.5 and x^-1 are smooth).
    # KIND: gradcheck
    # CATCHES: s09
    # CHAPTER: L0.1 section 2, Principles (one vjp per op)
    check_op(lambda a: a**p, (2, 3), lo=0.5, hi=2.0, salt=5)


@pytest.mark.parametrize(
    "shapes",
    [
        ((3, 4), (4, 2)),
        ((3, 3), (3, 3)),
        ((2, 3, 4), (4, 5)),
        ((4,), (4, 2)),
        ((3, 4), (4,)),
        ((4,), (4,)),
        ((2, 1, 3, 4), (5, 4, 2)),
    ],
    ids=["mm", "square", "batched", "vec-mat", "mat-vec", "dot", "broadcast-batch"],
)
def test_matmul_grads(shapes):
    # WHY: C = A B gives dA = G B^T and dB = A^T G. Square matrices make a
    #      missing transpose shape-correct but wrong; 1-D operands and batch
    #      broadcasting follow numpy's rules and must be undone in backward.
    # KIND: gradcheck
    # CATCHES: s07, m04
    # CHAPTER: L0.1 section 5, Pitfalls, item 7
    check_op(lambda a, b: a @ b, *shapes, salt=6)


@pytest.mark.parametrize(
    "index",
    [(slice(1, 3),), (1,), (slice(None), 2), (Ellipsis, -1), (None, 0)],
    ids=["slice", "int", "column", "ellipsis", "newaxis"],
)
def test_getitem_basic_grads(index):
    # WHY: basic indexing selects a view; its gradient is zero everywhere
    #      except the selected positions, which get the upstream gradient.
    # KIND: gradcheck
    # CATCHES: m10
    # CHAPTER: L0.1 section 2, Principles (indexing)
    check_op(lambda a: a[index], (4, 3), salt=7)


def test_getitem_repeated_indices_accumulate():
    # WHY: x[[0, 2, 0]] uses row 0 twice, so row 0's gradient is the sum of
    #      both upstream rows. `out[idx] = g` keeps only the last write; this
    #      is exactly how an embedding lookup with a repeated token behaves.
    # KIND: boundary
    # CATCHES: s04
    # CHAPTER: L0.1 section 5, Pitfalls, item 4
    x = Tensor(np.zeros((3, 2)), requires_grad=True)
    y = x[np.array([0, 2, 0])]
    y.backward(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    assert_close(x.grad, [[6.0, 8.0], [0.0, 0.0], [3.0, 4.0]], dtype="float32")
    check_op(lambda a: a[np.array([1, 1, 0, 3])], (4, 2), salt=8)
    m = np.array([True, False, True])
    check_op(lambda a: a[m], (3, 2), salt=9)


# --- graph rules -------------------------------------------------------------------


def test_shared_input_accumulates():
    # WHY: a appears three times in a * a + a, so da = 2a + 1 = 7 at a = 3.
    #      A backward that overwrites instead of adding keeps one path only.
    # KIND: unit
    # CATCHES: s02, s17
    # CHAPTER: L0.1 section 5, Pitfalls, item 2
    a = Tensor(3.0, requires_grad=True)
    (a * a + a).backward()
    assert_close(a.grad, 7.0, dtype="float32")


def test_diamond_graph():
    # WHY: b = 2a and c = a^2 both feed d = b * c; backward must finish d's
    #      gradient before splitting it, then sum both routes into a:
    #      d = 2 a^3, dd/da = 6 a^2 = 24 at a = 2.
    # KIND: unit
    # CATCHES: s02, s17
    # CHAPTER: L0.1 section 2, Principles (reverse topological order)
    a = Tensor(2.0, requires_grad=True, dtype=F64)
    b = a * 2.0
    c = a**2
    (b * c).backward()
    assert_close(a.grad, 24.0)


def test_grad_accumulates_across_backward_calls():
    # WHY: like torch, a leaf's .grad adds up over backward() calls until you
    #      clear it (an optimizer's zero_grad). Gradient accumulation over
    #      micro-batches (L11.1) relies on exactly this.
    # KIND: property
    # CATCHES: s03
    # CHAPTER: L0.1 section 5, Pitfalls, item 3
    a = Tensor([1.0, 2.0], requires_grad=True)
    (a * 3.0).backward(np.ones(2))
    (a * 3.0).backward(np.ones(2))
    assert_close(a.grad, [6.0, 6.0], dtype="float32")
    a.grad = None
    (a * 5.0).backward(np.ones(2))
    assert_close(a.grad, [5.0, 5.0], dtype="float32")


def test_backward_on_constant_raises():
    # WHY: nothing to differentiate: a tensor that does not require grad has
    #      no graph, and silently doing nothing hides a forgotten flag.
    # KIND: boundary
    # CATCHES: m01
    # CHAPTER: L0.1 section 4, The interface
    with pytest.raises(RuntimeError):
        Tensor([1.0, 2.0]).backward(np.ones(2))


def test_backward_needs_grad_for_non_scalar():
    # WHY: backward() with no argument means d(self)/d(self) = 1, which only
    #      makes sense for one number. A loss that was never reduced to a
    #      scalar must fail loudly, not backpropagate ones.
    # KIND: boundary
    # CATCHES: s13
    # CHAPTER: L0.1 section 5, Pitfalls, item 9
    a = Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(RuntimeError):
        (a * 2.0).backward()
    with pytest.raises(ValueError):
        (a * 2.0).backward(np.ones(3))


def test_vjp_shape_mismatch_raises():
    # WHY: an op whose vjp returns the wrong shape (a missing unbroadcast)
    #      must fail at backward with the op's name, not corrupt .grad.
    # KIND: boundary
    # CATCHES: m02
    # CHAPTER: L0.1 section 4, The interface (from_op)
    a = Tensor([1.0, 2.0], requires_grad=True)
    bad = from_op(a.data.sum(), (a,), lambda g: (np.ones(3),), "bad")
    with pytest.raises(RuntimeError, match="bad"):
        bad.backward()
    b = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    with pytest.raises(RuntimeError):
        from_op(b.data[:2], (b,), lambda g: (g,), "slice2").backward(np.ones(2))


def test_from_op_custom_op():
    # WHY: from_op is the seam the op library (L0.2) builds every op on:
    #      the output keeps its parents and vjp, and backward calls the vjp
    #      with the upstream gradient. Here c * x^2 with a constant c: the
    #      output requires grad because ONE parent does, and the vjp's None
    #      for the constant is skipped.
    # KIND: unit
    # CATCHES: s12
    # CHAPTER: L0.1 section 4, The interface (from_op)
    x = Tensor([1.0, -2.0, 3.0], requires_grad=True, dtype=F64)
    c = Tensor([1.0, 1.0, 2.0], dtype=F64)
    y = from_op(
        c.data * x.data**2, (x, c), lambda g: (2 * c.data * x.data * g, None), "csquare"
    )
    assert y.requires_grad
    y.backward(np.array([1.0, 1.0, 0.5]))
    assert_close(x.grad, [2.0, -4.0, 6.0])
    assert c.grad is None


def test_grad_dtype_matches_data():
    # WHY: a leaf's .grad has its data's dtype and shape even when an op's vjp
    #      computes in float64: optimizers update data in place with it.
    # KIND: boundary
    # CATCHES: m03
    # CHAPTER: L0.1 section 4, The interface
    x = Tensor([1.0, 2.0], requires_grad=True)  # float32
    y = from_op(
        x.data * 2, (x,), lambda g: (np.asarray(g, dtype=np.float64) * 2.0,), "double"
    )
    y.backward(np.ones(2, dtype=np.float32))
    assert x.grad.dtype == np.float32 and x.grad.shape == (2,)


# --- constants and dtypes ------------------------------------------------------------


def test_constants_mix_in():
    # WHY: an operand that does not require grad (a number, an ndarray, a
    #      Tensor without the flag) is a constant: the output still requires
    #      grad through the other operand, and the constant gets no .grad.
    # KIND: unit
    # CATCHES: s12
    # CHAPTER: L0.1 section 2, Principles (constants)
    x = Tensor([1.0, 2.0], requires_grad=True)
    c = Tensor([3.0, 4.0])
    y = x * c + np.array([1.0, 1.0]) - 2.0
    assert y.requires_grad
    y.backward(np.ones(2))
    assert_close(x.grad, [3.0, 4.0], dtype="float32")
    assert c.grad is None


def test_float32_stays_float32():
    # WHY: Python floats are float64. A constant converted without the
    #      tensor's dtype silently promotes every model in the course to
    #      float64 (twice the memory, and a different result than the C and
    #      Rust ports, which run float32).
    # KIND: boundary
    # CATCHES: s10
    # CHAPTER: L0.1 section 5, Pitfalls, item 10
    x = Tensor([1.0, 2.0], requires_grad=True)
    for y in (x * 0.5, x + 1.0, 2.0 - x, x / 3.0, x**2.0, x * np.array([1.0, 2.0])):
        assert y.dtype == np.float32
    (x * 0.5).backward(np.ones(2))
    assert x.grad.dtype == np.float32
    d = Tensor([1.0], dtype=F64)
    assert (d * 2.0).dtype == np.float64


def test_reflected_operators():
    # WHY: 2 - x, 3 / x, 1 + x, and A @ x put the number or array on the
    #      left; the reflected methods must keep the operand order (2 - x is
    #      not x - 2).
    # KIND: unit
    # CATCHES: s15, m08
    # CHAPTER: L0.1 section 4, The interface
    x = Tensor([1.0, 2.0], requires_grad=True, dtype=F64)
    assert_close((2.0 - x).data, [1.0, 0.0])
    assert_close((3.0 / x).data, [3.0, 1.5])
    assert_close((1.0 + x).data, [2.0, 3.0])
    assert_close((np.array([[1.0, 2.0], [3.0, 4.0]]) @ x).data, [5.0, 11.0])
    (2.0 - x).backward(np.ones(2))
    assert_close(x.grad, [-1.0, -1.0])


def test_ndarray_on_the_left():
    # WHY: numpy's own operator wins by default and broadcasts the Tensor
    #      object into an array of objects. `__array_ufunc__ = None` makes
    #      numpy defer, so ndarray + Tensor is a Tensor with a graph.
    # KIND: boundary
    # CATCHES: s14
    # CHAPTER: L0.1 section 5, Pitfalls, item 6
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = np.array([10.0, 20.0]) * x
    assert isinstance(y, Tensor) and y.requires_grad
    y.backward(np.ones(2))
    assert_close(x.grad, [10.0, 20.0], dtype="float32")


def test_constructor_copies_and_checks_dtype():
    # WHY: a leaf owns its data: changing the caller's array afterwards must
    #      not change the parameter, and integer dtypes cannot carry gradients.
    # KIND: boundary
    # CATCHES: s18
    # CHAPTER: L0.1 section 4, The interface
    a = np.array([1.0, 2.0], dtype=np.float32)
    t = Tensor(a)
    a[0] = 99.0
    assert t.data[0] == 1.0 and t.dtype == np.float32
    with pytest.raises(ValueError):
        Tensor([1, 2], dtype=np.int64)


def test_detach_stops_gradient():
    # WHY: detach() returns a constant that shares the data: a target or a
    #      frozen value takes part in the forward pass without receiving or
    #      passing gradient.
    # KIND: unit
    # CATCHES: s16
    # CHAPTER: L0.1 section 4, The interface
    x = Tensor([1.0, 2.0], requires_grad=True)
    d = (x * 2.0).detach()
    assert not d.requires_grad
    y = x * d
    y.backward(np.ones(2))
    assert_close(
        x.grad, [2.0, 4.0], dtype="float32"
    )  # d is a constant: d(x * d)/dx = d
    assert d.numpy() is d.data


# --- grad mode -------------------------------------------------------------------------


def test_no_grad_builds_no_graph():
    # WHY: inside no_grad, ops return constants and keep no parents, so
    #      evaluation and sampling hold no activations alive; backward on the
    #      result is an error.
    # KIND: property
    # CATCHES: s11
    # CHAPTER: L0.1 section 2, Principles (grad mode)
    x = Tensor([1.0, 2.0], requires_grad=True)
    with no_grad():
        assert not is_grad_enabled()
        y = x * 2.0 + x
        assert not y.requires_grad
        with pytest.raises(RuntimeError):
            y.backward(np.ones(2))
    assert is_grad_enabled()
    assert (x * 2.0).requires_grad


def test_no_grad_restores_after_exception():
    # WHY: an exception inside the block (a failed eval batch) must not leave
    #      grad mode off: every later training step would silently learn
    #      nothing.
    # KIND: boundary
    # CATCHES: s05
    # CHAPTER: L0.1 section 5, Pitfalls, item 5
    with pytest.raises(KeyError):
        with no_grad():
            raise KeyError("boom")
    assert is_grad_enabled()


def test_no_grad_nests():
    # WHY: no_grad restores the PREVIOUS mode, not True: an inner block that
    #      exits must leave the outer block still without grad.
    # KIND: boundary
    # CATCHES: s06
    # CHAPTER: L0.1 section 5, Pitfalls, item 5
    with no_grad():
        with no_grad():
            pass
        assert not is_grad_enabled()
        assert not (Tensor([1.0], requires_grad=True) * 2.0).requires_grad
    assert is_grad_enabled()


def test_deep_chain_no_recursion():
    # WHY: a long chain of ops (an unrolled RNN, many layers) is a deep graph;
    #      a recursive backward hits Python's recursion limit near 1000.
    # KIND: boundary
    # CATCHES: s17
    # CHAPTER: L0.1 section 2, Principles (reverse topological order)
    x = Tensor(1.0, requires_grad=True, dtype=F64)
    y = x
    for _ in range(5000):
        y = y + 1.0
    y.backward()
    assert_close(x.grad, 1.0)
    assert_close(y.data, 5001.0)


def test_shape_and_len():
    # WHY: shape, dtype, ndim, and len read through to the data array:
    #      model code reads them on every forward pass.
    # KIND: unit
    # CATCHES: m09
    # CHAPTER: L0.1 section 4, The interface
    t = Tensor(np.zeros((4, 3)))
    assert t.shape == (4, 3) and t.ndim == 2 and len(t) == 4 and t.dtype == np.float32
