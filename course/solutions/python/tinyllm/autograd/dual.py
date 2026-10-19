"""Dual numbers: forward-mode automatic differentiation (M08.1).

A dual number a + b eps with eps^2 = 0 makes Taylor's theorem exact to
first order: f(a + b eps) = f(a) + f'(a) b eps. Every operation below is
that rule for one primitive, so composing them differentiates the whole
program in the same pass that evaluates it.

Contract: contracts/py/tinyllm/autograd/dual.pyi.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)


class Dual:
    """value + tangent * eps, with eps^2 = 0."""

    # numpy then returns NotImplemented for `ndarray op Dual`, and Python
    # calls Dual's reflected method instead of building an object array.
    __array_ufunc__ = None

    def __init__(self, val: Any, eps: Any = 0.0) -> None:
        if isinstance(val, Dual) or isinstance(eps, Dual):
            raise TypeError("Dual(val, eps) takes numbers or arrays, not Dual")
        if np.ndim(val) == 0 and np.ndim(eps) == 0:
            self.val, self.eps = float(val), float(eps)
        else:
            v = np.asarray(val, dtype=np.float64)
            e = np.asarray(eps, dtype=np.float64)
            shape = np.broadcast_shapes(v.shape, e.shape)
            self.val = np.broadcast_to(v, shape).copy()
            self.eps = np.broadcast_to(e, shape).copy()

    def __repr__(self) -> str:
        return f"Dual({self.val!r}, {self.eps!r})"

    def __add__(self, other: Any) -> Dual:
        o = _lift(other)
        return Dual(self.val + o.val, self.eps + o.eps)

    def __radd__(self, other: Any) -> Dual:
        return self.__add__(other)

    def __sub__(self, other: Any) -> Dual:
        o = _lift(other)
        return Dual(self.val - o.val, self.eps - o.eps)

    def __rsub__(self, other: Any) -> Dual:
        o = _lift(other)
        return Dual(o.val - self.val, o.eps - self.eps)

    def __mul__(self, other: Any) -> Dual:
        o = _lift(other)
        return Dual(self.val * o.val, self.val * o.eps + self.eps * o.val)

    def __rmul__(self, other: Any) -> Dual:
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> Dual:
        o = _lift(other)
        return Dual(self.val / o.val, (self.eps * o.val - self.val * o.eps) / (o.val * o.val))

    def __rtruediv__(self, other: Any) -> Dual:
        return _lift(other).__truediv__(self)

    def __neg__(self) -> Dual:
        return Dual(-self.val, -self.eps)

    def __pow__(self, other: Any) -> Dual:
        if isinstance(other, Dual):
            # x^y = exp(y ln x): d = x^y (y' ln x + y x' / x)
            v = self.val**other.val
            return Dual(v, v * (other.eps * np.log(self.val) + other.val * self.eps / self.val))
        k = other
        return Dual(self.val**k, k * self.val ** (k - 1) * self.eps)

    def __rpow__(self, other: Any) -> Dual:
        v = other**self.val
        return Dual(v, v * np.log(other) * self.eps)

    def __matmul__(self, other: Any) -> Dual:
        o = _lift(other)
        return Dual(self.val @ o.val, self.val @ o.eps + self.eps @ o.val)

    def __rmatmul__(self, other: Any) -> Dual:
        w = np.asarray(other, dtype=np.float64)
        return Dual(w @ self.val, w @ self.eps)

    def __getitem__(self, idx: Any) -> Dual:
        return Dual(self.val[idx], self.eps[idx])


def _lift(x: Any) -> Dual:
    """x as a Dual: a constant gets tangent 0."""
    return x if isinstance(x, Dual) else Dual(x, 0.0)


def dual_exp(x: Any) -> Dual:
    d = _lift(x)
    e = np.exp(d.val)
    return Dual(e, e * d.eps)


def dual_log(x: Any) -> Dual:
    d = _lift(x)
    return Dual(np.log(d.val), d.eps / d.val)


def dual_tanh(x: Any) -> Dual:
    d = _lift(x)
    t = np.tanh(d.val)
    return Dual(t, (1.0 - t * t) * d.eps)


def _erf(a: Any) -> Any:
    """math.erf, elementwise for arrays."""
    if np.ndim(a) == 0:
        return math.erf(float(a))
    return np.vectorize(math.erf, otypes=[np.float64])(a)


def dual_erf(x: Any) -> Dual:
    d = _lift(x)
    return Dual(_erf(d.val), TWO_OVER_SQRT_PI * np.exp(-d.val * d.val) * d.eps)


def derivative(f: Callable[[Dual], Any], x: float) -> float:
    out = f(Dual(float(x), 1.0))
    return float(out.eps) if isinstance(out, Dual) else 0.0


def jvp(f: Callable[[Dual], Any], x: ArrayLike, v: ArrayLike) -> tuple[NDArray, NDArray]:
    xa = np.asarray(x, dtype=np.float64)
    va = np.asarray(v, dtype=np.float64)
    if xa.shape != va.shape:
        raise ValueError(f"x has shape {xa.shape} but v has shape {va.shape}")
    out = f(Dual(xa, va))
    if not isinstance(out, Dual):
        y = np.asarray(out, dtype=np.float64)
        return y, np.zeros_like(y)
    y = np.asarray(out.val, dtype=np.float64)
    return y, np.broadcast_to(np.asarray(out.eps, dtype=np.float64), y.shape).copy()
