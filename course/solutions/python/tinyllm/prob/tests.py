"""Hypothesis tests: paired and two-sample permutation, McNemar, Holm (M07.5).

A p-value answers one question: if nothing changed (the null hypothesis), how
often would chance alone produce a difference at least this large? The
permutation tests answer it by brute force, relabelling the data the way the
null says is harmless (flip which model a paired score belongs to, or
reshuffle two pools of scores) and counting. McNemar is the exact answer for
paired right/wrong outcomes, and Holm keeps a whole table of comparisons
honest.

Contract: contracts/py/tinyllm/prob/tests.pyi. L6.7's zoo compares models
with these; L8.5 and L8.6 assert "no regression" with them; load.02 ports the
two-sample test to Go and must match it draw for draw.
"""

from __future__ import annotations

import math
from typing import Callable, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

TIE_RTOL = 1e-9


class UniformSource(Protocol):
    def uniform(self) -> float: ...


def _vector(x: ArrayLike, name: str) -> NDArray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    return a


def _n_perm(n_perm: int) -> int:
    if int(n_perm) != n_perm or n_perm < 1:
        raise ValueError(f"n_perm must be an integer >= 1, got {n_perm!r}")
    return int(n_perm)


def mean_difference(a: NDArray, b: NDArray) -> float:
    return float(np.mean(a) - np.mean(b))


def paired_permutation_test(
    a: ArrayLike, b: ArrayLike, n_perm: int, rng: UniformSource
) -> float:
    x, y = _vector(a, "a"), _vector(b, "b")
    if x.shape != y.shape:
        raise ValueError(f"a and b must pair up, got lengths {x.size} and {y.size}")
    n_perm = _n_perm(n_perm)
    d = x - y
    t_obs = abs(float(np.sum(d)))
    bar = t_obs * (1.0 - TIE_RTOL)
    count = 0
    for _ in range(n_perm):
        # Under H0 the two scores of a pair could have come from either model:
        # each difference keeps or flips its sign with probability 1/2.
        s = np.array([-1.0 if rng.uniform() < 0.5 else 1.0 for _ in range(d.size)])
        if abs(float(np.sum(s * d))) >= bar:
            count += 1
    return (1 + count) / (1 + n_perm)


def permutation_test(
    a: ArrayLike,
    b: ArrayLike,
    stat: Callable[[NDArray, NDArray], float],
    n_perm: int,
    rng: UniformSource,
) -> float:
    x, y = _vector(a, "a"), _vector(b, "b")
    n_perm = _n_perm(n_perm)
    z = np.concatenate([x, y])
    na, total = x.size, z.size
    t_obs = abs(float(stat(x, y)))
    bar = t_obs * (1.0 - TIE_RTOL)
    count = 0
    for _ in range(n_perm):
        # Under H0 the group labels are arbitrary: deal the pooled values into
        # two groups of the original sizes, uniformly over all orders.
        idx = list(range(total))
        for i in range(total - 1, 0, -1):
            j = min(int(rng.uniform() * (i + 1)), i)
            idx[i], idx[j] = idx[j], idx[i]
        p = z[idx]
        if abs(float(stat(p[:na], p[na:]))) >= bar:
            count += 1
    return (1 + count) / (1 + n_perm)


def mcnemar(b01: int, b10: int, exact: bool = True) -> float:
    for v, name in ((b01, "b01"), (b10, "b10")):
        if int(v) != v or v < 0:
            raise ValueError(f"{name} must be an integer >= 0, got {v!r}")
    b01, b10 = int(b01), int(b10)
    n = b01 + b10
    if n == 0:
        return 1.0
    if exact:
        # Under H0 each discordant pair is a fair coin: b01 ~ Binomial(n, 1/2).
        # Two-sided: double the smaller tail. Integers until the one division.
        k = min(b01, b10)
        tail = sum(math.comb(n, i) for i in range(k + 1))
        return min(1.0, 2 * tail / 2**n)
    chi2 = max(abs(b01 - b10) - 1, 0) ** 2 / n
    return math.erfc(math.sqrt(chi2 / 2.0))


def _pvalues(pvalues: ArrayLike) -> NDArray:
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"pvalues must be 1-D, got shape {p.shape}")
    if not np.all((p >= 0.0) & (p <= 1.0)):
        raise ValueError("every p-value must be in [0, 1]")
    return p


def holm(pvalues: ArrayLike, alpha: float = 0.05) -> NDArray:
    p = _pvalues(pvalues)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    m = p.size
    order = np.argsort(p, kind="stable")
    reject = np.zeros(m, dtype=bool)
    for k, i in enumerate(order):
        # The smallest p-value faces the strictest bar alpha / m; each
        # rejection removes one hypothesis from the family.
        if p[i] > alpha / (m - k):
            break
        reject[i] = True
    return reject


def holm_adjust(pvalues: ArrayLike) -> NDArray:
    p = _pvalues(pvalues)
    m = p.size
    order = np.argsort(p, kind="stable")
    adj = np.empty(m, dtype=np.float64)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[i]))
        adj[i] = running
    return adj
