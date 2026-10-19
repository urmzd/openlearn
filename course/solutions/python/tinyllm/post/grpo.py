"""Group relative policy objectives for L12.3."""
from __future__ import annotations
import numpy as np


def group_advantages(rewards, eps=1e-8):
    r = np.asarray(rewards, dtype=np.float64)
    if r.ndim != 1 or r.size == 0:
        raise ValueError("rewards must be a non-empty vector")
    return (r - r.mean()) / (r.std() + eps)


def grpo_loss(logp_new, logp_old, advantages, kl, beta=0.0, clip=0.2):
    new, old, adv, k = (np.asarray(x, dtype=np.float64) for x in (logp_new, logp_old, advantages, kl))
    if not (new.shape == old.shape == adv.shape == k.shape):
        raise ValueError("GRPO arrays must have equal shapes")
    if beta < 0 or not 0 < clip < 1:
        raise ValueError("beta must be nonnegative and clip in (0, 1)")
    ratio = np.exp(np.clip(new - old, -60.0, 60.0))
    objective = np.minimum(ratio * adv, np.clip(ratio, 1.0 - clip, 1.0 + clip) * adv)
    return float(-objective.mean() + beta * np.maximum(k, 0.0).mean())
