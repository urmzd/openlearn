"""Tensor: a numpy array that remembers how it was computed (L0.1).

Every op on tensors that require grad returns a tensor holding its inputs
(`_parents`) and a vector-Jacobian product (`_vjp`). `backward()` orders the
graph with toposort (M06.1), root first, and hands each node's finished
gradient to its vjp. Broadcasting is undone with unbroadcast (M08.3): an
input that numpy stretched along an axis gets its gradient summed over that
axis.

Contract: contracts/py/tinyllm/autograd/tensor.pyi.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from tinyllm.autograd.graph import toposort
from tinyllm.autograd.mode import is_grad_enabled
from tinyllm.autograd.vjp import unbroadcast


class Tensor:
    # numpy defers to us: `ndarray + Tensor` calls Tensor.__radd__ instead of
    # broadcasting the Tensor object into an array of objects.
    __array_ufunc__ = None

    def __init__(
        self, data: ArrayLike, requires_grad: bool = False, dtype: DTypeLike = np.float32
    ) -> None:
        dt = np.dtype(dtype)
        if not np.issubdtype(dt, np.floating):
            raise ValueError(f"Tensor dtype must be floating point, got {dt}")
        self.data = np.array(data, dtype=dt)  # a copy: a leaf never aliases the caller's array
        self.grad: Optional[NDArray] = None
        self.requires_grad = bool(requires_grad)
        self._parents: tuple[Tensor, ...] = ()
        self._vjp: Optional[Callable] = None
        self._op = ""

    # -- views of the data ------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> Any:
        return self.data.dtype

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        rg = ", requires_grad=True" if self.requires_grad else ""
        op = f", op={self._op}" if self._op else ""
        return f"Tensor(shape={self.shape}, dtype={self.dtype}{rg}{op})"

    # -- backward ---------------------------------------------------------------

    def backward(self, grad: Optional[ArrayLike] = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("backward() on a tensor that does not require grad")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError(
                    f"backward() without a gradient needs a one-element tensor, got shape {self.shape}"
                )
            g0 = np.ones_like(self.data)
        else:
            g0 = np.asarray(grad, dtype=self.data.dtype)
            if g0.shape != self.shape:
                raise ValueError(f"grad has shape {g0.shape}, the tensor has {self.shape}")
        # Root first, every node before its parents: when a node comes up, every
        # consumer has already added its share, so its gradient is complete.
        order = toposort(self, lambda t: [p for p in t._parents if p.requires_grad])
        grads: dict[int, NDArray] = {id(self): g0}
        for node in order:
            g = grads.pop(id(node), None)
            if g is None:
                continue
            if node._vjp is None:
                # A leaf: accumulate across backward() calls, as torch does.
                node.grad = g.copy() if node.grad is None else node.grad + g
                continue
            pgrads = node._vjp(g)
            if len(pgrads) != len(node._parents):
                raise RuntimeError(
                    f"op {node._op or '?'}: vjp returned {len(pgrads)} gradients for {len(node._parents)} inputs"
                )
            for p, pg in zip(node._parents, pgrads):
                if pg is None or not p.requires_grad:
                    continue
                pg = np.asarray(pg, dtype=p.data.dtype)
                if pg.shape != p.shape:
                    raise RuntimeError(
                        f"op {node._op or '?'}: gradient of shape {pg.shape} for an input of shape {p.shape}"
                    )
                # A tensor used by several ops (or twice by one) sums every path.
                prev = grads.get(id(p))
                grads[id(p)] = pg if prev is None else prev + pg

    def detach(self) -> "Tensor":
        t = Tensor.__new__(Tensor)
        t.data = self.data
        t.grad = None
        t.requires_grad = False
        t._parents, t._vjp, t._op = (), None, ""
        return t

    def numpy(self) -> NDArray:
        return self.data

    # -- arithmetic -------------------------------------------------------------

    def __add__(self, other: Any) -> "Tensor":
        a, b = self, _lift(other, self)
        sa, sb = a.shape, b.shape
        return from_op(
            a.data + b.data, (a, b), lambda g: (unbroadcast(g, sa), unbroadcast(g, sb)), "add"
        )

    def __radd__(self, other: Any) -> "Tensor":
        return _lift(other, self) + self

    def __sub__(self, other: Any) -> "Tensor":
        a, b = self, _lift(other, self)
        sa, sb = a.shape, b.shape
        return from_op(
            a.data - b.data, (a, b), lambda g: (unbroadcast(g, sa), unbroadcast(-g, sb)), "sub"
        )

    def __rsub__(self, other: Any) -> "Tensor":
        return _lift(other, self) - self

    def __mul__(self, other: Any) -> "Tensor":
        a, b = self, _lift(other, self)
        x, y = a.data, b.data
        return from_op(
            x * y,
            (a, b),
            lambda g: (unbroadcast(g * y, x.shape), unbroadcast(g * x, y.shape)),
            "mul",
        )

    def __rmul__(self, other: Any) -> "Tensor":
        return _lift(other, self) * self

    def __truediv__(self, other: Any) -> "Tensor":
        a, b = self, _lift(other, self)
        x, y = a.data, b.data
        # d(x/y)/dx = 1/y, d(x/y)/dy = -x/y^2
        return from_op(
            x / y,
            (a, b),
            lambda g: (unbroadcast(g / y, x.shape), unbroadcast(-g * x / (y * y), y.shape)),
            "div",
        )

    def __rtruediv__(self, other: Any) -> "Tensor":
        return _lift(other, self) / self

    def __matmul__(self, other: Any) -> "Tensor":
        a, b = self, _lift(other, self)
        A, B = a.data, b.data
        if A.ndim == 0 or B.ndim == 0:
            raise ValueError("matmul needs operands with at least one dimension")
        out = A @ B

        def vjp(g: NDArray) -> tuple[NDArray, NDArray]:
            # Promote 1-D operands to matrices the way numpy does: a vector on
            # the left is a row (1, k), on the right a column (k, 1).
            A2 = A[None, :] if A.ndim == 1 else A
            B2 = B[:, None] if B.ndim == 1 else B
            g2 = g
            if B.ndim == 1:  # restore the column axis first: a dot product's g is 0-d
                g2 = np.expand_dims(g2, -1)
            if A.ndim == 1:
                g2 = np.expand_dims(g2, -2)
            # C = A B  =>  dA = G B^T, dB = A^T G (per batch), then sum the
            # batch axes an operand was broadcast along.
            dA = unbroadcast(g2 @ np.swapaxes(B2, -1, -2), A2.shape)
            dB = unbroadcast(np.swapaxes(A2, -1, -2) @ g2, B2.shape)
            return dA.reshape(A.shape), dB.reshape(B.shape)

        return from_op(out, (a, b), vjp, "matmul")

    def __rmatmul__(self, other: Any) -> "Tensor":
        return _lift(other, self) @ self

    def __pow__(self, exponent: float) -> "Tensor":
        if isinstance(exponent, Tensor) or not np.isscalar(exponent):
            raise TypeError("Tensor ** x needs a Python number x")
        x = self.data
        p = float(exponent)
        return from_op(x**p, (self,), lambda g: (g * p * x ** (p - 1),), "pow")

    def __neg__(self) -> "Tensor":
        return from_op(-self.data, (self,), lambda g: (-g,), "neg")

    def __getitem__(self, index: Any) -> "Tensor":
        idx = index
        x = self.data

        def vjp(g: NDArray) -> tuple[NDArray]:
            out = np.zeros_like(x)
            # add.at, not out[idx] = g: an id that appears twice in an integer
            # index gets both gradients.
            np.add.at(out, idx, g)
            return (out,)

        return from_op(x[idx], (self,), vjp, "getitem")


def _lift(other: Any, like: Tensor) -> Tensor:
    """other as a Tensor; numbers and arrays become constants of like's dtype."""
    if isinstance(other, Tensor):
        return other
    return Tensor(np.asarray(other), requires_grad=False, dtype=like.data.dtype)


def from_op(
    data: ArrayLike,
    parents: Sequence[Tensor],
    vjp: Callable[[NDArray], Sequence[Optional[NDArray]]],
    op: str = "",
) -> Tensor:
    parents = tuple(parents)
    t = Tensor.__new__(Tensor)
    t.data = np.asarray(data)
    t.grad = None
    t._op = op
    track = is_grad_enabled() and any(p.requires_grad for p in parents)
    t.requires_grad = track
    # Without tracking, keep nothing: no_grad must not hold the inputs alive.
    t._parents = parents if track else ()
    t._vjp = vjp if track else None
    return t
