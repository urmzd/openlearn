"""Entropy, cross-entropy, KL and JS divergences, the k3 estimator (M11.1).

All logs are natural, so every quantity is in nats. The zero conventions
(0 log 0 = 0; p log(p / 0) = +inf for p > 0) are applied with np.where on
the terms, never by adding a small epsilon, which would bias every answer.

Contract: contracts/py/tinyllm/info/entropy.pyi.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.num.stable import log_softmax


def _probs(p: ArrayLike, name: str) -> NDArray:
    """p as a float64 array, ValueError on a negative entry."""
    a = np.asarray(p, dtype=np.float64)
    if (a < 0).any():
        raise ValueError(f"{name} has a negative probability (min {a.min()!r})")
    return a


def _xlogy(x: NDArray, y: NDArray) -> NDArray:
    """x * log(y) elementwise with 0 * log(anything) = 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = x * np.log(y)
    return np.where(x == 0, 0.0, t)


def entropy(p: ArrayLike, axis: int = -1) -> NDArray:
    a = _probs(p, "p")
    return -np.sum(_xlogy(a, a), axis=axis)


def cross_entropy(p: ArrayLike, q: ArrayLike, axis: int = -1) -> NDArray:
    a, b = _probs(p, "p"), _probs(q, "q")
    return -np.sum(_xlogy(a, b), axis=axis)


def kl(p: ArrayLike, q: ArrayLike, axis: int = -1) -> NDArray:
    a, b = _probs(p, "p"), _probs(q, "q")
    with np.errstate(divide="ignore", invalid="ignore"):
        t = a * (np.log(a) - np.log(b))
    # p = 0 costs nothing; p > 0 where q = 0 is log(p / 0) = +inf.
    t = np.where(a == 0, 0.0, np.where(b == 0, np.inf, t))
    return np.sum(t, axis=axis)


def kl_from_logprobs(logp: ArrayLike, logq: ArrayLike, axis: int = -1) -> NDArray:
    lp = np.asarray(logp, dtype=np.float64)
    lq = np.asarray(logq, dtype=np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        t = np.exp(lp) * (lp - lq)
    # logp = -inf is p = 0: no cost. logq = -inf under p > 0 is +inf, even when
    # exp(logp) underflows to 0 (0 * inf would be nan).
    t = np.where(np.isneginf(lp), 0.0, np.where(np.isneginf(lq), np.inf, t))
    return np.sum(t, axis=axis)


def js(p: ArrayLike, q: ArrayLike) -> NDArray:
    a, b = _probs(p, "p"), _probs(q, "q")
    m = 0.5 * (a + b)
    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def kl_k3(logp_ref: ArrayLike, logp: ArrayLike) -> NDArray:
    r = np.asarray(logp_ref, dtype=np.float64) - np.asarray(logp, dtype=np.float64)
    # exp(r) - 1 - r loses everything to rounding when |r| is small: exp(r)
    # is 1 + r + r^2/2 and the 1 swallows the r^2/2. expm1 keeps it.
    return np.expm1(r) - r


def entropy_from_logits(z: ArrayLike, axis: int = -1) -> NDArray:
    lp = log_softmax(np.asarray(z, dtype=np.float64), axis=axis)
    with np.errstate(invalid="ignore"):
        t = np.exp(lp) * lp
    return -np.sum(np.where(np.isneginf(lp), 0.0, t), axis=axis)
