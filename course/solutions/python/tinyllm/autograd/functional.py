"""The op library: every differentiable op a model uses (L0.2).

    from tinyllm.autograd import functional as F

Each op is a numpy forward plus a vector-Jacobian product (vjp), joined by
tensor.from_op (L0.1). The math comes from earlier modules: activation
derivatives from M01.3, stable softmax and logsumexp from M09.2, the softmax
vjps from M08.3. gradcheck_all() proves every vjp against central
differences (M04.1) and, for elementwise ops, forward-mode duals (M08.1).

Contract: contracts/py/tinyllm/autograd/functional.pyi.
"""

from __future__ import annotations

import builtins
import dataclasses
import math
from typing import Any, Callable, Literal, Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd import dual
from tinyllm.autograd.tensor import Tensor, from_op
from tinyllm.autograd.vjp import log_softmax_vjp, softmax_vjp, unbroadcast
from tinyllm.num import activations as act
from tinyllm.num import stable
from tinyllm.num.gradcheck import GradcheckReport, gradcheck
from tinyllm.num.rng import PCG32

Axis = Union[int, tuple[int, ...], None]


def _t(x: Any) -> Tensor:
    """x as a Tensor: Tensors pass through, anything else is a constant."""
    if isinstance(x, Tensor):
        return x
    a = np.asarray(x)
    return Tensor(a, dtype=a.dtype if a.dtype.kind == "f" else np.float32)


def _elementwise(x: Any, f: Callable, df: Callable, name: str) -> Tensor:
    """y = f(x) with vjp g * df(x), both in x's dtype."""
    x = _t(x)
    a = x.data
    y = np.asarray(f(a), dtype=a.dtype)
    return from_op(y, (x,), lambda g: (g * np.asarray(df(a), dtype=a.dtype),), name)


# -- elementwise ------------------------------------------------------------------


def exp(x: Tensor) -> Tensor:
    x = _t(x)
    y = np.exp(x.data)
    return from_op(y, (x,), lambda g: (g * y,), "exp")  # d e^x = e^x: reuse the output


def log(x: Tensor) -> Tensor:
    x = _t(x)
    a = x.data
    return from_op(np.log(a), (x,), lambda g: (g / a,), "log")


def tanh(x: Tensor) -> Tensor:
    return _elementwise(x, act.tanh, act.dtanh, "tanh")


def sigmoid(x: Tensor) -> Tensor:
    return _elementwise(x, act.sigmoid, act.dsigmoid, "sigmoid")


def relu(x: Tensor) -> Tensor:
    return _elementwise(x, act.relu, act.drelu, "relu")


def silu(x: Tensor) -> Tensor:
    return _elementwise(x, act.silu, act.dsilu, "silu")


def gelu(x: Tensor, approximate: Literal["none", "tanh"] = "none") -> Tensor:
    if approximate == "none":
        return _elementwise(x, act.gelu_erf, act.dgelu_erf, "gelu")
    if approximate == "tanh":
        return _elementwise(x, act.gelu_tanh, act.dgelu_tanh, "gelu_tanh")
    raise ValueError(f"approximate must be 'none' or 'tanh', got {approximate!r}")


# -- reductions ---------------------------------------------------------------------


def _axes(axis: Axis, ndim: int) -> tuple[int, ...]:
    """axis as a sorted tuple of non-negative axes (None: every axis)."""
    if axis is None:
        return tuple(range(ndim))
    ax = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
    out = []
    for a in ax:
        a = int(a)
        if not -ndim <= a < builtins.max(ndim, 1):
            raise ValueError(f"axis {a} is out of range for {ndim} dimensions")
        out.append(a % ndim if ndim else 0)
    return tuple(sorted(set(out)))


def _expand(g: NDArray, axes: tuple[int, ...], keepdims: bool) -> NDArray:
    """An upstream gradient of a reduction, with the reduced axes put back as size 1."""
    return g if keepdims else np.expand_dims(g, axes)


def sum(x: Tensor, axis: Axis = None, keepdims: bool = False) -> Tensor:
    x = _t(x)
    axes = _axes(axis, x.ndim)
    shape = x.shape
    y = np.sum(x.data, axis=axes, keepdims=keepdims)
    # Every input element contributed with weight 1: copy g back over the axes.
    return from_op(
        y, (x,), lambda g: (np.broadcast_to(_expand(g, axes, keepdims), shape).copy(),), "sum"
    )


def mean(x: Tensor, axis: Axis = None, keepdims: bool = False) -> Tensor:
    x = _t(x)
    axes = _axes(axis, x.ndim)
    n = math.prod(x.shape[a] for a in axes)
    if n == 0:
        raise ValueError("mean over an empty axis")
    return sum(x, axis, keepdims) * (1.0 / n)


def max(x: Tensor, axis: Axis = None, keepdims: bool = False) -> Tensor:
    x = _t(x)
    axes = _axes(axis, x.ndim)
    a = x.data
    y = np.max(a, axis=axes, keepdims=keepdims)

    def vjp(g: NDArray) -> tuple[NDArray]:
        ym = np.max(a, axis=axes, keepdims=True)
        hit = (a == ym).astype(a.dtype)
        # Ties share the gradient equally (torch.amax), so it still sums to g.
        share = hit / np.sum(hit, axis=axes, keepdims=True)
        return (share * _expand(g, axes, keepdims),)

    return from_op(y, (x,), vjp, "max")


def var(x: Tensor, axis: Axis = None, keepdims: bool = False, correction: int = 0) -> Tensor:
    x = _t(x)
    axes = _axes(axis, x.ndim)
    n = math.prod(x.shape[a] for a in axes)
    if n - correction <= 0:
        raise ValueError(f"var over {n} elements with correction {correction}")
    a = x.data
    d = a - np.mean(a, axis=axes, keepdims=True)
    y = np.sum(d * d, axis=axes, keepdims=keepdims) / (n - correction)
    # d/dx_i of sum_j (x_j - mu)^2 is 2 (x_i - mu): the terms through mu sum
    # to zero because sum_j (x_j - mu) = 0.
    return from_op(
        y.astype(a.dtype, copy=False),
        (x,),
        lambda g: (_expand(g, axes, keepdims) * (2.0 / (n - correction)) * d,),
        "var",
    )


# -- shape ------------------------------------------------------------------------------


def reshape(x: Tensor, shape: Sequence[int]) -> Tensor:
    x = _t(x)
    s = x.shape
    return from_op(x.data.reshape(tuple(shape)), (x,), lambda g: (g.reshape(s),), "reshape")


def transpose(x: Tensor, a: int, b: int) -> Tensor:
    x = _t(x)
    return from_op(np.swapaxes(x.data, a, b), (x,), lambda g: (np.swapaxes(g, a, b),), "transpose")


def permute(x: Tensor, dims: Sequence[int]) -> Tensor:
    x = _t(x)
    dims = tuple(int(d) % builtins.max(x.ndim, 1) for d in dims)
    if sorted(dims) != list(range(x.ndim)):
        raise ValueError(f"dims {dims} is not a permutation of {x.ndim} axes")
    inv = tuple(int(i) for i in np.argsort(dims))  # undo the permutation in backward
    return from_op(np.transpose(x.data, dims), (x,), lambda g: (np.transpose(g, inv),), "permute")


def concat(xs: Sequence[Tensor], axis: int = 0) -> Tensor:
    ts = [_t(x) for x in xs]
    if not ts:
        raise ValueError("concat of nothing")
    sizes = [t.shape[axis] for t in ts]
    cuts = np.cumsum(sizes)[:-1]
    return from_op(
        np.concatenate([t.data for t in ts], axis=axis),
        ts,
        lambda g: tuple(np.split(g, cuts, axis=axis)),
        "concat",
    )


def stack(xs: Sequence[Tensor], axis: int = 0) -> Tensor:
    ts = [_t(x) for x in xs]
    if not ts:
        raise ValueError("stack of nothing")
    y = np.stack([t.data for t in ts], axis=axis)
    ax = axis % y.ndim
    return from_op(
        y, ts, lambda g: tuple(np.take(g, i, axis=ax) for i in range(len(ts))), "stack"
    )


# -- selection ----------------------------------------------------------------------------


def _int_index(idx: ArrayLike, n: int, what: str) -> NDArray:
    """An integer index array checked against [0, n)."""
    a = np.asarray(idx)
    if a.size and not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"{what} must be integers, got dtype {a.dtype}")
    a = a.astype(np.int64)
    if a.size and (a.min() < 0 or a.max() >= n):
        raise ValueError(f"{what} must lie in [0, {n}), got min {a.min()} and max {a.max()}")
    return a


def where(cond: ArrayLike, a: Any, b: Any) -> Tensor:
    c = np.asarray(cond, dtype=bool)
    ta = a if isinstance(a, Tensor) else None
    tb = b if isinstance(b, Tensor) else None
    like = ta if ta is not None else tb
    dt = like.dtype if like is not None else np.float32
    ta = ta if ta is not None else Tensor(a, dtype=dt)
    tb = tb if tb is not None else Tensor(b, dtype=dt)
    sa, sb = ta.shape, tb.shape
    y = np.where(c, ta.data, tb.data)

    def vjp(g: NDArray) -> tuple[NDArray, NDArray]:
        zero = np.zeros((), dtype=g.dtype)
        return unbroadcast(np.where(c, g, zero), sa), unbroadcast(np.where(c, zero, g), sb)

    return from_op(y, (ta, tb), vjp, "where")


def gather(x: Tensor, idx: ArrayLike, axis: int) -> Tensor:
    x = _t(x)
    ax = axis % x.ndim
    i = _int_index(idx, x.shape[ax], "gather index")
    if i.ndim != x.ndim:
        raise ValueError(f"gather index has {i.ndim} dimensions, x has {x.ndim}")
    a = x.data

    def vjp(g: NDArray) -> tuple[NDArray]:
        out = np.zeros_like(a)
        # Full coordinates of every output element, with axis replaced by the
        # index: add.at so a position picked twice gets both gradients.
        coords = list(np.indices(i.shape, sparse=True))
        coords[ax] = i
        np.add.at(out, tuple(coords), g)
        return (out,)

    return from_op(np.take_along_axis(a, i, axis=ax), (x,), vjp, "gather")


def embedding(weight: Tensor, ids: ArrayLike) -> Tensor:
    w = _t(weight)
    if w.ndim != 2:
        raise ValueError(f"embedding weight must be [n, d], got shape {w.shape}")
    i = _int_index(ids, w.shape[0], "ids")
    W = w.data

    def vjp(g: NDArray) -> tuple[NDArray]:
        out = np.zeros_like(W)
        np.add.at(out, i, g)  # a token that appears twice gets both rows of gradient
        return (out,)

    return from_op(W[i], (w,), vjp, "embedding")


def masked_fill(x: Tensor, mask: ArrayLike, value: float) -> Tensor:
    x = _t(x)
    m = np.broadcast_to(np.asarray(mask, dtype=bool), x.shape)
    y = np.where(m, np.asarray(value, dtype=x.dtype), x.data)
    return from_op(y, (x,), lambda g: (np.where(m, np.zeros((), dtype=g.dtype), g),), "masked_fill")


# -- softmax family ------------------------------------------------------------------------


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    x = _t(x)
    y = np.asarray(stable.softmax(x.data, axis=axis), dtype=x.dtype)
    return from_op(y, (x,), lambda g: (softmax_vjp(g, y, axis),), "softmax")


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    x = _t(x)
    y = np.asarray(stable.log_softmax(x.data, axis=axis), dtype=x.dtype)
    return from_op(y, (x,), lambda g: (log_softmax_vjp(g, y, axis),), "log_softmax")


def logsumexp(x: Tensor, axis: int = -1, keepdims: bool = False) -> Tensor:
    x = _t(x)
    a = x.data
    y = np.asarray(stable.logsumexp(a, axis=axis, keepdims=keepdims), dtype=x.dtype)
    ax = axis % builtins.max(x.ndim, 1)
    # d logsumexp(x) / dx_i = softmax(x)_i
    return from_op(
        y,
        (x,),
        lambda g: (_expand(g, (ax,), keepdims) * np.asarray(stable.softmax(a, axis=ax), dtype=a.dtype),),
        "logsumexp",
    )


# -- the rest -------------------------------------------------------------------------------


def dropout(x: Tensor, p: float, training: bool, rng: Any) -> Tensor:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"dropout p must be in [0, 1], got {p}")
    x = _t(x)
    if not training or p == 0.0:
        return x
    u = np.asarray(rng.uniforms(x.data.size), dtype=np.float64).reshape(x.shape)
    keep = u >= p
    # Inverted dropout: scale at training time so evaluation is the identity.
    scale = 0.0 if p == 1.0 else 1.0 / (1.0 - p)
    m = (keep * scale).astype(x.dtype)
    return from_op(x.data * m, (x,), lambda g: (g * m,), "dropout")


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return _t(a) @ b


# -- gradcheck_all ----------------------------------------------------------------------------


class _Rng:
    """A fixed generator for dropout inside gradcheck: every call replays the
    same mask, so the op is a fixed linear map that central differences can probe."""

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def uniforms(self, n: int) -> NDArray:
        return PCG32(self.seed).uniforms(n)


def _cases() -> list[tuple[str, Callable, list[tuple[int, ...]], tuple[float, float]]]:
    """(name, fn of Tensors, input shapes, input range) for every op, looked up
    in this module at call time so a replaced op is the one checked."""
    F = globals()
    idx3 = np.array([[2, 0], [1, 1], [0, 2]])
    ids = np.array([[1, 0, 1], [2, 2, 0]])
    cond = np.array([[True, False, True], [False, True, True]])
    mask = np.array([[False, True, False], [True, False, False]])
    pos = (0.5, 2.0)
    sym = (-2.0, 2.0)
    return [
        ("exp", lambda a: F["exp"](a), [(2, 3)], sym),
        ("log", lambda a: F["log"](a), [(2, 3)], pos),
        ("tanh", lambda a: F["tanh"](a), [(2, 3)], sym),
        ("sigmoid", lambda a: F["sigmoid"](a), [(2, 3)], sym),
        ("relu", lambda a: F["relu"](a), [(2, 3)], sym),
        ("silu", lambda a: F["silu"](a), [(2, 3)], sym),
        ("gelu", lambda a: F["gelu"](a), [(2, 3)], sym),
        ("gelu_tanh", lambda a: F["gelu"](a, approximate="tanh"), [(2, 3)], sym),
        ("sum", lambda a: F["sum"](a, axis=1), [(2, 3)], sym),
        ("mean", lambda a: F["mean"](a, axis=(0, 2), keepdims=True), [(2, 3, 2)], sym),
        ("max", lambda a: F["max"](a, axis=-1), [(2, 3)], sym),
        ("var", lambda a: F["var"](a, axis=-1, correction=1), [(2, 3)], sym),
        ("reshape", lambda a: F["reshape"](a, (3, 2)), [(2, 3)], sym),
        ("transpose", lambda a: F["transpose"](a, 0, 2), [(2, 3, 2)], sym),
        ("permute", lambda a: F["permute"](a, (2, 0, 1)), [(2, 3, 2)], sym),
        ("concat", lambda a, b: F["concat"]([a, b], axis=1), [(2, 3), (2, 2)], sym),
        ("stack", lambda a, b: F["stack"]([a, b], axis=1), [(2, 3), (2, 3)], sym),
        ("where", lambda a, b: F["where"](cond, a, b), [(2, 3), (3,)], sym),
        ("gather", lambda a: F["gather"](a, idx3, axis=1), [(3, 3)], sym),
        ("embedding", lambda w: F["embedding"](w, ids), [(3, 2)], sym),
        ("masked_fill", lambda a: F["masked_fill"](a, mask, -1.0), [(2, 3)], sym),
        ("softmax", lambda a: F["softmax"](a, axis=-1), [(2, 3)], sym),
        ("log_softmax", lambda a: F["log_softmax"](a, axis=0), [(2, 3)], sym),
        ("logsumexp", lambda a: F["logsumexp"](a, axis=-1), [(2, 3)], sym),
        ("dropout", lambda a: F["dropout"](a, 0.4, True, _Rng(7)), [(2, 3)], sym),
        ("matmul", lambda a, b: F["matmul"](a, b), [(2, 3), (3, 2)], sym),
        ("add", lambda a, b: a + b, [(2, 3), (3,)], sym),
        ("sub", lambda a, b: a - b, [(2, 1), (1, 3)], sym),
        ("mul", lambda a, b: a * b, [(2, 3), (2, 1)], sym),
        ("div", lambda a, b: a / b, [(2, 3), (3,)], pos),
        ("pow", lambda a: a**3.0, [(2, 3)], sym),
        ("neg", lambda a: -a, [(2, 3)], sym),
        ("getitem", lambda a: a[np.array([0, 2, 0])], [(3, 2)], sym),
    ]


# Forward-mode versions of the elementwise ops, written with the Dual on the
# left of every operator (Dual op Dual, Dual op float).
_SQRT1_2 = 1.0 / math.sqrt(2.0)
_SQRT2_PI = math.sqrt(2.0 / math.pi)
_DUAL = {
    "exp": lambda d: dual.dual_exp(d),
    "log": lambda d: dual.dual_log(d),
    "tanh": lambda d: dual.dual_tanh(d),
    "sigmoid": lambda d: dual.dual_exp(d) / (dual.dual_exp(d) + 1.0),
    "silu": lambda d: d * dual.dual_exp(d) / (dual.dual_exp(d) + 1.0),
    "gelu": lambda d: d * 0.5 * (dual.dual_erf(d * _SQRT1_2) + 1.0),
    "gelu_tanh": lambda d: d * 0.5 * (dual.dual_tanh((d + d * d * d * 0.044715) * _SQRT2_PI) + 1.0),
}


def _dual_error(name: str, fn: Callable) -> float:
    """Largest relative gap between fn's backward and the forward-mode
    derivative at a few points (relu has no dual form: it is piecewise)."""
    worst = 0.0
    for x0 in (-1.7, -0.3, 0.4, 1.3) if name != "log" else (0.3, 0.9, 1.7, 2.5):
        t = Tensor(np.array([x0]), requires_grad=True, dtype=np.float64)
        fn(t).backward(np.ones(1))
        want = dual.derivative(_DUAL[name], x0)
        got = float(t.grad[0])
        worst = builtins.max(worst, abs(got - want) / builtins.max(1.0, abs(want)))
    return worst


def gradcheck_all(rtol: float = 1e-5) -> dict[str, GradcheckReport]:
    reports: dict[str, GradcheckReport] = {}
    rng = PCG32(0)
    for name, fn, shapes, (lo, hi) in _cases():
        xs = [lo + (hi - lo) * rng.uniforms(math.prod(s)).reshape(s) for s in shapes]
        # Keep kinks out of reach: relu at 0 and max ties have no derivative.
        xs = [np.where(np.abs(x) < 0.05, x + 0.1, x) for x in xs]
        out_shape = fn(*[Tensor(x, dtype=np.float64) for x in xs]).shape
        w = rng.uniforms(math.prod(out_shape)).reshape(out_shape) - 0.5

        def f(*arrays: NDArray, fn: Callable = fn, w: NDArray = w) -> float:
            return float(np.sum(fn(*[Tensor(a, dtype=np.float64) for a in arrays]).data * w))

        ts = [Tensor(x, requires_grad=True, dtype=np.float64) for x in xs]
        fn(*ts).backward(w)
        analytic = [t.grad if t.grad is not None else np.zeros_like(t.data) for t in ts]
        rep = gradcheck(f, xs, analytic, rtol=rtol)
        if name in _DUAL and _dual_error(name, fn) > 1e-9:
            rep = dataclasses.replace(rep, ok=False)
        reports[name] = rep
    return reports
