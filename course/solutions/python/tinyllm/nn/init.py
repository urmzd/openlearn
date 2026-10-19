"""Variance-preserving weight initialization (M07.3).

If z = sum_{j=1}^{n} w_j x_j with independent zero-mean weights of variance
s^2, independent of the inputs, then Var(z) = n s^2 E[x^2]. Choosing s^2 so
that n s^2 E[x^2] equals the previous layer's variance keeps the signal from
exploding or vanishing through depth: Xavier (Glorot and Bengio, 2010) for
linear and tanh-like layers, Kaiming (He et al., 2015) for ReLU.

Every initializer returns float32, filled in C order from `rng` (PCG32,
spec/pcg32.md): uniforms are rng.uniform(), normals tinyllm.prob.rv.normal.

Contract: contracts/py/tinyllm/nn/init.pyi.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from tinyllm.prob.rv import normal

_GAIN_ONE = (
    "linear",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "sigmoid",
)


def _shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """shape as a tuple of non-negative ints."""
    s = tuple(int(d) for d in shape)
    if any(d < 0 for d in s):
        raise ValueError(f"shape {shape!r} has a negative dimension")
    return s


def fans(shape: tuple[int, ...]) -> tuple[int, int]:
    s = _shape(shape)
    if len(s) < 2:
        raise ValueError(f"fan_in and fan_out need at least 2 dimensions, got shape {s}")
    receptive = math.prod(s[2:])
    fan_in, fan_out = s[1] * receptive, s[0] * receptive
    if fan_in == 0 or fan_out == 0:
        raise ValueError(f"shape {s} has a fan of 0")
    return fan_in, fan_out


def calculate_gain(nonlinearity: str, param: float | None = None) -> float:
    if nonlinearity in _GAIN_ONE:
        return 1.0
    if nonlinearity == "tanh":
        return 5.0 / 3.0
    if nonlinearity == "relu":
        return math.sqrt(2.0)
    if nonlinearity == "leaky_relu":
        if param is None:
            slope = 0.01
        elif isinstance(param, Real) and not isinstance(param, bool):
            slope = float(param)
        else:
            raise ValueError(f"leaky_relu slope must be a number, got {param!r}")
        return math.sqrt(2.0 / (1.0 + slope**2))
    if nonlinearity == "selu":
        return 3.0 / 4.0
    raise ValueError(f"unsupported nonlinearity {nonlinearity!r}")


def _normal(shape: tuple[int, ...], std: float, rng: Any) -> NDArray:
    """std * standard normals in C order, as float32."""
    n = math.prod(shape)
    return (std * normal(rng, n)).reshape(shape).astype(np.float32)


def xavier_uniform(shape: tuple[int, ...], gain: float, rng: Any) -> NDArray:
    if not gain >= 0.0:
        raise ValueError(f"gain must be >= 0, got {gain}")
    s = _shape(shape)
    fan_in, fan_out = fans(s)
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    u = np.array([rng.uniform() for _ in range(math.prod(s))], dtype=np.float64)
    return (-a + 2.0 * a * u).reshape(s).astype(np.float32)


def xavier_normal(shape: tuple[int, ...], gain: float, rng: Any) -> NDArray:
    if not gain >= 0.0:
        raise ValueError(f"gain must be >= 0, got {gain}")
    s = _shape(shape)
    fan_in, fan_out = fans(s)
    return _normal(s, gain * math.sqrt(2.0 / (fan_in + fan_out)), rng)


def kaiming_normal(
    shape: tuple[int, ...],
    fan_mode: Literal["fan_in", "fan_out"],
    nonlinearity: str,
    rng: Any,
) -> NDArray:
    if fan_mode not in ("fan_in", "fan_out"):
        raise ValueError(f"fan_mode must be 'fan_in' or 'fan_out', got {fan_mode!r}")
    s = _shape(shape)
    fan_in, fan_out = fans(s)
    fan = fan_in if fan_mode == "fan_in" else fan_out
    return _normal(s, calculate_gain(nonlinearity) / math.sqrt(fan), rng)


def normal_init(shape: tuple[int, ...], std: float, rng: Any) -> NDArray:
    if not std >= 0.0:
        raise ValueError(f"std must be >= 0, got {std}")
    return _normal(_shape(shape), std, rng)


def scaled_residual_std(base_std: float, n_layers: int) -> float:
    if not base_std >= 0.0:
        raise ValueError(f"base_std must be >= 0, got {base_std}")
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    return base_std / math.sqrt(2.0 * n_layers)
