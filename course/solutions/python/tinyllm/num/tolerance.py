"""Error analysis, condition numbers, tolerance budgets (M09.3).

Every floating-point operation rounds: fl(a op b) = (a op b)(1 + delta) with
|delta| <= u. Chaining that model through a sum or a dot product of k terms
gives a bound, gamma_k |x| . |y|, that holds for every input: a test that
allows exactly that much never fails a correct implementation and still
fails one that drops or doubles a term. The condition number says how much
of the error comes from the problem itself rather than from the algorithm.

Contract: contracts/py/tinyllm/num/tolerance.pyi.
"""

from __future__ import annotations

import math
from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.linalg.svd import svd

# significand bits including the implicit leading 1
_PRECISION = {
    "f64": 53,
    "float64": 53,
    "f32": 24,
    "float32": 24,
    "f16": 11,
    "float16": 11,
    "bf16": 8,
    "bfloat16": 8,
    "e4m3": 4,
    "e5m2": 3,
}


def unit_roundoff(dtype: str) -> float:
    if dtype not in _PRECISION:
        raise ValueError(f"unknown dtype {dtype!r}; one of {sorted(_PRECISION)}")
    return 2.0 ** -_PRECISION[dtype]


def gamma(k: int, dtype: str) -> float:
    u = unit_roundoff(dtype)
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    ku = k * u
    if ku >= 1.0:
        raise ValueError(f"k u = {ku} >= 1: the bound is void for k = {k} in {dtype}")
    return ku / (1.0 - ku)


def sum_error_bound(k: int, dtype: str, abs_sum: ArrayLike) -> NDArray:
    if k < 1:
        raise ValueError(f"a sum has k >= 1 terms, got {k}")
    return gamma(k - 1, dtype) * np.asarray(abs_sum, dtype=np.float64)


def dot_error_bound(k: int, dtype: str, abs_dot: ArrayLike) -> NDArray:
    if k < 1:
        raise ValueError(f"a dot product has k >= 1 terms, got {k}")
    return gamma(k, dtype) * np.asarray(abs_dot, dtype=np.float64)


def matmul_error_bound(A: ArrayLike, B: ArrayLike, dtype: str, dA: ArrayLike = 0.0) -> NDArray:
    a = np.abs(np.asarray(A, dtype=np.float64))
    b = np.abs(np.asarray(B, dtype=np.float64))
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0] or a.shape[1] < 1:
        raise ValueError(f"need A [m, k] and B [k, n] with k >= 1, got {a.shape} and {b.shape}")
    d = np.broadcast_to(np.asarray(dA, dtype=np.float64), a.shape)
    if (d < 0).any():
        raise ValueError("dA must be >= 0")
    k = a.shape[1]
    # The perturbation moves the exact product by at most dA @ |B|; rounding
    # the perturbed product adds gamma_k |A'| @ |B| <= gamma_k (|A| + dA) @ |B|.
    return d @ b + gamma(k, dtype) * ((a + d) @ b)


def assert_close_bounded(
    actual: ArrayLike,
    expected: ArrayLike,
    k: int,
    dtype: str,
    abs_dot: ArrayLike,
    slack: float = 4.0,
) -> None:
    if not slack > 0:
        raise ValueError(f"slack must be > 0, got {slack}")
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    if a.shape != e.shape:
        raise AssertionError(f"shape {a.shape} != expected {e.shape}")
    allowed = slack * np.broadcast_to(dot_error_bound(k, dtype, abs_dot), a.shape)
    with np.errstate(invalid="ignore"):
        err = np.abs(a - e)
        ok = (err <= allowed) | (np.isnan(a) & np.isnan(e)) | (a == e)
    if ok.all():
        return
    i = tuple(int(j) for j in np.argwhere(~ok)[0])
    ratio = err[i] / allowed[i] if allowed[i] > 0 else math.inf
    raise AssertionError(
        f"{int((~ok).sum())} of {a.size} elements outside the bound; first at {i}: "
        f"actual {a[i]!r}, expected {e[i]!r}, |error| {err[i]:.3e} > "
        f"{slack:g} * gamma_{k} * |x|.|y| = {allowed[i]:.3e} (ratio {ratio:.3g}, {dtype})"
    )


def bound_ratio(actual: ArrayLike, expected: ArrayLike, bound: ArrayLike) -> float:
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    b = np.broadcast_to(np.asarray(bound, dtype=np.float64), a.shape)
    if a.size == 0:
        return 0.0
    err = np.abs(a - e)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(err == 0, 0.0, err / b)
    return float(np.max(r))


def cond(A: ArrayLike) -> float:
    a = np.asarray(A, dtype=np.float64)
    if a.ndim != 2 or a.size == 0 or not np.isfinite(a).all():
        raise ValueError(f"cond needs a finite, non-empty 2-D array, got shape {a.shape}")
    s = svd(a)[1]
    smax, smin = float(s[0]), float(s[-1])
    if smax == 0.0 or smin <= max(a.shape) * 2.0**-52 * smax:
        return math.inf
    return smax / smin


def relative_condition(f: Callable[[float], float], df: Callable[[float], float], x: float) -> float:
    num = abs(x * df(x))
    if num == 0.0:
        return 0.0
    fx = abs(f(x))
    return math.inf if fx == 0.0 else num / fx


def fd_error_model(
    h: float,
    order: Literal[1, 2],
    dtype: str,
    f_scale: float = 1.0,
    deriv_scale: float = 1.0,
) -> float:
    if not h > 0:
        raise ValueError(f"h must be > 0, got {h}")
    u = unit_roundoff(dtype)
    if order == 1:
        return deriv_scale * h / 2.0 + 2.0 * u * f_scale / h
    if order == 2:
        return deriv_scale * h * h / 6.0 + u * f_scale / h
    raise ValueError(f"order must be 1 (forward) or 2 (central), got {order!r}")


def optimal_fd_step(
    order: Literal[1, 2], dtype: str, f_scale: float = 1.0, deriv_scale: float = 1.0
) -> float:
    if not (f_scale > 0 and deriv_scale > 0):
        raise ValueError("f_scale and deriv_scale must be > 0")
    u = unit_roundoff(dtype)
    if order == 1:
        # d/dh (D h / 2 + 2 u F / h) = D / 2 - 2 u F / h^2 = 0
        return 2.0 * math.sqrt(u * f_scale / deriv_scale)
    if order == 2:
        # d/dh (D h^2 / 6 + u F / h) = D h / 3 - u F / h^2 = 0
        return (3.0 * u * f_scale / deriv_scale) ** (1.0 / 3.0)
    raise ValueError(f"order must be 1 (forward) or 2 (central), got {order!r}")
