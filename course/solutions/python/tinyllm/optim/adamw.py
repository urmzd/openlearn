"""Adam and AdamW: bias-corrected moment estimates and decoupled weight decay (M10.3).

A parameter is any object with `data` (a float ndarray, updated in place) and
`grad` (an ndarray of the same shape, or None). The update is torch's
single-tensor Adam/AdamW (torch 2.14) with one global step counter t:

    t += 1, for every p whose grad g is not None:
      Adam : g = g + weight_decay * p            (L2 penalty coupled into g)
      AdamW: p = p * (1 - lr * weight_decay)      (decoupled, before the update)
      m = beta1 * m + (1 - beta1) * g             (first moment, an EMA of g)
      v = beta2 * v + (1 - beta2) * g * g         (second moment, an EMA of g^2)
      p = p - (lr / (1 - beta1**t)) * m / (sqrt(v) / sqrt(1 - beta2**t) + eps)

Contract: contracts/py/tinyllm/optim/adamw.pyi.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


def _check_hyper(lr: float, betas: tuple[float, float], eps: float, weight_decay: float) -> None:
    """ValueError for a hyperparameter outside its domain (torch's checks)."""
    if not lr >= 0.0:
        raise ValueError(f"lr must be >= 0, got {lr}")
    if not eps >= 0.0:
        raise ValueError(f"eps must be >= 0, got {eps}")
    if not weight_decay >= 0.0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
    if len(betas) != 2:
        raise ValueError(f"betas must be a pair, got {betas!r}")
    for i, b in enumerate(betas):
        if not 0.0 <= b < 1.0:
            raise ValueError(f"betas[{i}] must be in [0, 1), got {b}")


class Adam:
    """Adam (Kingma and Ba, 2015) with an L2 penalty folded into the gradient."""

    _decoupled = False  # AdamW decays the weights directly instead

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        _check_hyper(lr, betas, eps, weight_decay)
        self.params = list(params)
        if not self.params:
            raise ValueError("params is empty: nothing to optimize")
        for i, p in enumerate(self.params):
            d = getattr(p, "data", None)
            if not isinstance(d, np.ndarray) or not np.issubdtype(d.dtype, np.floating):
                raise ValueError(f"parameter {i}: data must be a floating-point ndarray")
        self.lr = float(lr)
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0
        # Moments in the parameter's own dtype, allocated up front so every
        # parameter has an entry in the checkpoint even before its first grad.
        self.exp_avg = [np.zeros_like(p.data) for p in self.params]
        self.exp_avg_sq = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        lr, (beta1, beta2), eps, wd = self.lr, self.betas, self.eps, self.weight_decay
        bias_correction1 = 1.0 - beta1**self.t
        bias_correction2_sqrt = math.sqrt(1.0 - beta2**self.t)
        step_size = lr / bias_correction1
        for p, m, v in zip(self.params, self.exp_avg, self.exp_avg_sq):
            g = p.grad
            if g is None:
                continue
            g = np.asarray(g, dtype=p.data.dtype)
            if g.shape != p.data.shape:
                raise ValueError(f"grad shape {g.shape} != parameter shape {p.data.shape}")
            if self._decoupled:
                p.data *= 1.0 - lr * wd
            elif wd != 0.0:
                g = g + wd * p.data
            m *= beta1
            m += (1.0 - beta1) * g
            v *= beta2
            v += (1.0 - beta2) * (g * g)
            denom = np.sqrt(v) / bias_correction2_sqrt + eps
            p.data -= step_size * (m / denom)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def state_dict(self) -> dict:
        return {
            "step": self.t,
            "lr": self.lr,
            "betas": [self.betas[0], self.betas[1]],
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "exp_avg": [m.copy() for m in self.exp_avg],
            "exp_avg_sq": [v.copy() for v in self.exp_avg_sq],
        }

    def load_state_dict(self, sd: dict) -> None:
        ms, vs = list(sd["exp_avg"]), list(sd["exp_avg_sq"])
        if len(ms) != len(self.params) or len(vs) != len(self.params):
            raise ValueError(
                f"state has {len(ms)} and {len(vs)} moments for {len(self.params)} parameters"
            )
        for i, (p, m, v) in enumerate(zip(self.params, ms, vs)):
            if np.shape(m) != p.data.shape or np.shape(v) != p.data.shape:
                raise ValueError(f"parameter {i}: moment shape does not match {p.data.shape}")
        betas = (float(sd["betas"][0]), float(sd["betas"][1]))
        _check_hyper(float(sd["lr"]), betas, float(sd["eps"]), float(sd["weight_decay"]))
        self.t = int(sd["step"])
        self.lr = float(sd["lr"])
        self.betas = betas
        self.eps = float(sd["eps"])
        self.weight_decay = float(sd["weight_decay"])
        self.exp_avg = [np.array(m, dtype=p.data.dtype, copy=True) for p, m in zip(self.params, ms)]
        self.exp_avg_sq = [np.array(v, dtype=p.data.dtype, copy=True) for p, v in zip(self.params, vs)]


class AdamW(Adam):
    """Adam with decoupled weight decay (Loshchilov and Hutter, 2019)."""

    _decoupled = True

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
