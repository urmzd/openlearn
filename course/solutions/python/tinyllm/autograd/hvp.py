"""Hessian-vector products and checkpoint schedules (M08.4).

hvp_fd differentiates the gradient along v with a central difference; the
schedule functions implement the recompute-versus-memory cost model of the
chapter (segments kept as inputs only, recomputed during backward, the last
segment kept whole, as torch.utils.checkpoint.checkpoint_sequential).

Contract: contracts/py/tinyllm/autograd/hvp.pyi.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _grad_at(grad_fn: Callable, x: NDArray) -> NDArray:
    """grad_fn(x) as a new float64 array of x's shape (a copy, never a view)."""
    g = np.array(grad_fn(x), dtype=np.float64)
    if g.shape != x.shape:
        raise ValueError(f"grad_fn returned shape {g.shape} for an input of shape {x.shape}")
    return g


def hvp_fd(
    grad_fn: Callable[[NDArray], ArrayLike], x: ArrayLike, v: ArrayLike, eps: float = 1e-4
) -> NDArray:
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if x.shape != v.shape:
        raise ValueError(f"x has shape {x.shape} and v {v.shape}: they must match")
    if not (math.isfinite(eps) and eps > 0):
        raise ValueError(f"eps must be a positive finite number, got {eps}")
    gp = _grad_at(grad_fn, x + eps * v)
    gm = _grad_at(grad_fn, x - eps * v)
    return (gp - gm) / (2.0 * eps)


def hessian_fd(grad_fn: Callable[[NDArray], ArrayLike], x: ArrayLike, eps: float = 1e-4) -> NDArray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"hessian_fd needs a non-empty 1-D x, got shape {x.shape}")
    n = x.size
    H = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        H[:, j] = hvp_fd(grad_fn, x, e, eps)
    return (H + H.T) / 2.0


def _sizes(n_layers: int, starts: Sequence[int]) -> list[int]:
    """Segment sizes of a valid schedule; ValueError otherwise."""
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    s = [int(b) for b in starts]
    if not s or s[0] != 0:
        raise ValueError(f"a schedule starts at layer 0, got {list(starts)}")
    if any(b <= a for a, b in zip(s, s[1:])) or s[-1] >= n_layers:
        raise ValueError(f"starts must increase strictly and stay below {n_layers}, got {s}")
    return [b - a for a, b in zip(s, s[1:] + [n_layers])]


def checkpoint_cost(n_layers: int, starts: Sequence[int]) -> tuple[int, int]:
    sizes = _sizes(n_layers, starts)
    peak = max(i + s for i, s in enumerate(sizes))
    return peak, n_layers - sizes[-1]


def min_checkpoint_memory(n_layers: int) -> int:
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    p = math.isqrt(2 * n_layers)
    while p * (p + 1) // 2 < n_layers:
        p += 1
    while p > 1 and (p - 1) * p // 2 >= n_layers:
        p -= 1
    return p


def _cover(k: int, budget: int) -> int:
    """Most layers k segments can hold when segment i has at most budget - i."""
    return k * budget - k * (k - 1) // 2


def checkpoint_schedule(n_layers: int, mem_budget_layers: int) -> list[int]:
    n, b = n_layers, mem_budget_layers
    if b < min_checkpoint_memory(n):
        raise ValueError(
            f"no schedule of {n} layers peaks at {b} units; the least is {min_checkpoint_memory(n)}"
        )
    for k in range(1, n + 1):
        last = min(b - k + 1, n - k + 1)
        if last >= 1 and n - last <= _cover(k - 1, b):
            break
    rest = n - last
    sizes = []
    for i in range(k - 1):
        s = min(b - i, rest - (k - 2 - i))
        sizes.append(s)
        rest -= s
    sizes.append(last)
    starts, at = [], 0
    for s in sizes:
        starts.append(at)
        at += s
    return starts
