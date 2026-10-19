"""Preference objectives for L12.2."""
from __future__ import annotations
import numpy as np


def _margin(pc, pr, rc, rr):
    pc, pr, rc, rr = (np.asarray(x, dtype=np.float64) for x in (pc, pr, rc, rr))
    if not (pc.shape == pr.shape == rc.shape == rr.shape):
        raise ValueError("preference log-probability shapes differ")
    return (pc - pr) - (rc - rr)


def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.1):
    if beta <= 0:
        raise ValueError("beta must be positive")
    z = beta * _margin(policy_chosen, policy_rejected, reference_chosen, reference_rejected)
    return float(np.mean(np.logaddexp(0.0, -z)))


def ipo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.1):
    if beta <= 0:
        raise ValueError("beta must be positive")
    z = _margin(policy_chosen, policy_rejected, reference_chosen, reference_rejected)
    return float(np.mean((z - 1.0 / (2.0 * beta)) ** 2))
