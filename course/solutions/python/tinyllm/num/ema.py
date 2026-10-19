"""Exponential moving average with bias correction (M02.2).

Unrolling m_t = beta m_{t-1} + (1 - beta) x_t from m_0 = 0 gives
m_t = sum_{i=1}^{t} (1 - beta) beta^(t-i) x_i: a geometric series of weights
whose total is 1 - beta^t. Early on that total is far below 1 (0.1 after one
step with beta = 0.9), so m_t is biased toward 0. Dividing by the total gives
a weighted average again. Adam (M10.3) applies exactly this to its moments;
the training loop (L0.5) uses it to smooth the loss curve. The geometric
sums come from M00.3's series.py.

Contract: contracts/py/tinyllm/num/ema.pyi.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.num.series import geometric, geometric_sum


def _check_beta(beta: float) -> float:
    b = float(beta)
    if not (math.isfinite(b) and 0.0 <= b < 1.0):
        raise ValueError(f"beta must satisfy 0 <= beta < 1, got {beta}")
    return b


def ema_weights(beta: float, t: int) -> NDArray:
    b = _check_beta(beta)
    if t < 0:
        raise ValueError(f"t must be >= 0, got {t}")
    # (1 - b), (1 - b) b, (1 - b) b^2, ... is newest first; x_1 is the oldest.
    return geometric(1.0 - b, b, int(t))[::-1].copy()


class EMA:
    def __init__(self, beta: float) -> None:
        self.beta = _check_beta(beta)
        self._t = 0
        self._m: float | NDArray = 0.0
        self._shape: tuple | None = None

    @property
    def t(self) -> int:
        return self._t

    @property
    def value(self) -> float | NDArray:
        m = self._m
        return m.copy() if isinstance(m, np.ndarray) else float(m)

    def update(self, x: ArrayLike) -> float | NDArray:
        a = np.asarray(x, dtype=np.float64)
        if not np.isfinite(a).all():
            raise ValueError(
                "x must be finite: one nan or inf would poison every later average"
            )
        if self._shape is not None and a.shape != self._shape:
            raise ValueError(
                f"x has shape {a.shape}, earlier updates had {self._shape}"
            )
        # Start from m_0 = 0, never from x_1: the bias correction below
        # assumes the zero start, and the two together would double-correct.
        m = self.beta * self._m + (1.0 - self.beta) * (a if a.ndim else float(a))
        self._shape = a.shape
        self._m = m
        self._t += 1
        return self.value

    def value_debiased(self) -> float | NDArray:
        if self._t == 0:
            raise RuntimeError(
                "value_debiased before any update: the average of nothing is 0 / 0"
            )
        # The weights of m_t add up to (1 - b)(1 + b + ... + b^(t-1)) = 1 - b^t.
        # geometric_sum (M00.3) gets that total right even for b near 1, where
        # 1 - b**t subtracts two nearly equal numbers.
        out = self._m / geometric_sum(1.0 - self.beta, self.beta, self._t)
        return out.copy() if isinstance(out, np.ndarray) else float(out)
