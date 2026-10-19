"""Losses with a fused backward (L0.3).

Each loss is one graph node. Its forward uses stable numerics (M09.2), and
its vjp is the closed-form gradient: for softmax cross-entropy that is
softmax(z) - q, never the product of a log, a softmax, and a gather
Jacobian. One node keeps one [N, V] array alive instead of three.

Contract: contracts/py/tinyllm/autograd/losses.pyi.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd.tensor import Tensor, from_op
from tinyllm.num import stable

_REDUCTIONS = ("mean", "sum", "none")


def _targets(targets: ArrayLike, shape: tuple[int, ...], v: int, ignore_index: int) -> tuple[NDArray, NDArray]:
    """(targets as int64 with ignored rows set to 0, the keep mask), validated."""
    t = np.asarray(targets)
    if t.size and not np.issubdtype(t.dtype, np.integer):
        raise ValueError(f"targets must be integers, got dtype {t.dtype}")
    t = t.astype(np.int64)
    if t.shape != shape:
        raise ValueError(f"targets have shape {t.shape}, logits need {shape}")
    keep = t != ignore_index
    kept = t[keep]
    if kept.size and (kept.min() < 0 or kept.max() >= v):
        raise ValueError(f"targets must lie in [0, {v}) or equal ignore_index {ignore_index}")
    return np.where(keep, t, 0), keep


def cross_entropy(
    logits: Tensor,
    targets: ArrayLike,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    if reduction not in _REDUCTIONS:
        raise ValueError(f"reduction must be one of {_REDUCTIONS}, got {reduction!r}")
    eps = float(label_smoothing)
    if not 0.0 <= eps <= 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1], got {eps}")
    z = logits if isinstance(logits, Tensor) else Tensor(logits)
    a = z.data
    if a.ndim < 1:
        raise ValueError("logits need a class axis")
    v = a.shape[-1]
    t, keep = _targets(targets, a.shape[:-1], v, ignore_index)
    lp = np.asarray(stable.log_softmax(a, axis=-1), dtype=a.dtype)
    nll = -np.take_along_axis(lp, t[..., None], axis=-1)[..., 0]
    rows = nll if eps == 0.0 else (1.0 - eps) * nll - eps * lp.mean(axis=-1)
    rows = np.where(keep, rows, 0.0).astype(a.dtype)
    n = int(keep.sum())
    if reduction == "mean":
        # Divide by the rows that count: padding must not dilute the loss.
        out = rows.sum() / n if n else np.zeros((), dtype=a.dtype)
    elif reduction == "sum":
        out = rows.sum()
    else:
        out = rows
    out = np.asarray(out, dtype=a.dtype)

    def vjp(g: NDArray) -> tuple[NDArray]:
        # d loss_i / d z_i = softmax(z_i) - q_i, q_i the (smoothed) target.
        d = np.exp(lp)
        if eps:
            d = d - eps / v
        np.put_along_axis(d, t[..., None], np.take_along_axis(d, t[..., None], axis=-1) - (1.0 - eps), axis=-1)
        d = d * keep[..., None]
        if reduction == "mean":
            d = d * (g / n if n else 0.0)
        elif reduction == "sum":
            d = d * g
        else:
            d = d * g[..., None]
        return (d.astype(a.dtype, copy=False),)

    return from_op(out, (z,), vjp, "cross_entropy")


def mse(pred: Tensor, target: ArrayLike) -> Tensor:
    p = pred if isinstance(pred, Tensor) else Tensor(pred)
    y = np.asarray(target, dtype=p.dtype)
    if y.shape != p.shape:
        raise ValueError(f"target has shape {y.shape}, pred has {p.shape}")
    d = p.data - y
    n = d.size
    out = np.asarray((d * d).sum() / n, dtype=p.dtype)
    return from_op(out, (p,), lambda g: (g * (2.0 / n) * d,), "mse")


def bce_with_logits(
    logits: Tensor, targets: ArrayLike, pos_weight: Optional[ArrayLike] = None
) -> Tensor:
    z = logits if isinstance(logits, Tensor) else Tensor(logits)
    x = z.data
    y = np.asarray(targets, dtype=x.dtype)
    if y.shape != x.shape:
        raise ValueError(f"targets have shape {y.shape}, logits have {x.shape}")
    w = np.ones((), dtype=x.dtype) if pos_weight is None else np.asarray(pos_weight, dtype=x.dtype)
    c = 1.0 + (w - 1.0) * y  # the weight on softplus(-x) = -log sigmoid(x)
    # softplus(-x) = max(-x, 0) + log1p(exp(-|x|)): the exponent is never positive.
    sp = np.maximum(-x, 0.0) + np.log1p(np.exp(-np.abs(x)))
    n = x.size
    out = np.asarray(((1.0 - y) * x + c * sp).sum() / n, dtype=x.dtype)
    sig_neg = np.exp(-np.logaddexp(0.0, x))  # sigmoid(-x) = 1 / (1 + e^x), stable for either sign

    def vjp(g: NDArray) -> tuple[NDArray]:
        return ((g / n * ((1.0 - y) - c * sig_neg)).astype(x.dtype, copy=False),)

    return from_op(out, (z,), vjp, "bce_with_logits")
