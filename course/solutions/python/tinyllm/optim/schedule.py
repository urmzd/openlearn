"""Learning-rate schedules and global gradient-norm clipping (M10.4).

A schedule is a pure function of `step`, the number of optimizer updates
already taken (0 for the first update): the train loop sets
opt.lr = schedule(step, ...) before every opt.step(). Every warmup ramps
linearly from 0.

Contract: contracts/py/tinyllm/optim/schedule.pyi.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

CLIP_EPS = 1e-6  # torch's guard against dividing by a zero norm


def _check_lrs(lr_max: float, lr_min: float) -> None:
    if not 0.0 <= lr_min <= lr_max:
        raise ValueError(f"need 0 <= lr_min <= lr_max, got lr_min={lr_min}, lr_max={lr_max}")


def cosine_with_warmup(step: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    if step < 0 or warmup < 0 or total < 1 or warmup > total:
        raise ValueError(f"need 0 <= step, 0 <= warmup <= total, total >= 1; got {step}, {warmup}, {total}")
    _check_lrs(lr_max, lr_min)
    if step < warmup:
        return lr_max * step / warmup
    if step >= total:
        return lr_min
    progress = (step - warmup) / (total - warmup)
    return lr_min + (lr_max - lr_min) * 0.5 * (1.0 + math.cos(math.pi * progress))


def wsd(step: int, warmup: int, stable: int, decay: int, lr_max: float, lr_min: float) -> float:
    if step < 0 or warmup < 0 or stable < 0 or decay < 1:
        raise ValueError(f"need step, warmup, stable >= 0 and decay >= 1; got {step}, {warmup}, {stable}, {decay}")
    _check_lrs(lr_max, lr_min)
    if step < warmup:
        return lr_max * step / warmup
    if step < warmup + stable:
        return lr_max
    if step < warmup + stable + decay:
        progress = (step - warmup - stable) / decay
        return lr_min + (lr_max - lr_min) * (1.0 - progress)
    return lr_min


def noam(step: int, d_model: int, warmup: int) -> float:
    if step < 0 or d_model < 1 or warmup < 1:
        raise ValueError(f"need step >= 0, d_model >= 1, warmup >= 1; got {step}, {d_model}, {warmup}")
    if step == 0:
        return 0.0  # min(inf, 0); Python raises on 0 ** -0.5
    return d_model**-0.5 * min(step**-0.5, step * warmup**-1.5)


def clip_grad_norm_(params: Iterable[Any], max_norm: float) -> float:
    if not max_norm > 0.0:
        raise ValueError(f"max_norm must be > 0, got {max_norm}")
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    total = math.sqrt(sum(float(np.sum(np.square(g, dtype=np.float64))) for g in grads))
    if not math.isfinite(total):
        return total
    coef = max_norm / (total + CLIP_EPS)
    if coef < 1.0:
        for g in grads:
            g *= coef
    return total
