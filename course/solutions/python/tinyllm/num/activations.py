"""Activation functions and their derivatives (M01.3).

A neural network layer is a matrix product followed by one of these
elementwise nonlinearities; without them a stack of layers collapses to a
single matrix. Training needs each one's derivative (the chain rule
multiplies it in during backpropagation), so every function comes with its
d-twin, and both must stay finite and warning-free for any finite input:
a logit of -1000 after a bad step must not turn into nan.

Contract: contracts/py/tinyllm/num/activations.pyi. L0.2's op library,
the LSTM gates (L3.2), and SwiGLU (L7.2) call these.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.num.taylor import erf_series

ERF_TERMS = 160  # enough for erf_series to be within 3e-15 everywhere (M02.1)
GELU_C = math.sqrt(2.0 / math.pi)
GELU_A = 0.044715
GELU_CLAMP = 10.0  # |x| beyond which tanh(u(x)) is exactly +-1 in float64
PDF_CLAMP = 40.0  # exp(-40**2 / 2) underflows to 0: the normal density is 0 beyond


def _arr(x: ArrayLike) -> NDArray:
    """x as a float array: float32 and float64 keep their dtype."""
    a = np.asarray(x)
    if a.dtype in (np.float32, np.float64):
        return a
    return a.astype(np.float64)


def sigmoid(x: ArrayLike) -> NDArray:
    x = _arr(x)
    # exp(-|x|) is in (0, 1]: it can never overflow. For x >= 0 that is
    # 1 / (1 + e^-x); for x < 0 multiply top and bottom by e^x.
    z = np.exp(-np.abs(x))
    one = x.dtype.type(1)
    return np.where(x >= 0, one / (one + z), z / (one + z))


def dsigmoid(x: ArrayLike) -> NDArray:
    x = _arr(x)
    # s(x)(1 - s(x)) = s(x) s(-x). The second form never subtracts from 1,
    # so dsigmoid(30) keeps its relative accuracy instead of 1e-3.
    return sigmoid(x) * sigmoid(-x)


def tanh(x: ArrayLike) -> NDArray:
    return np.tanh(_arr(x))


def dtanh(x: ArrayLike) -> NDArray:
    t = np.tanh(_arr(x))
    return t.dtype.type(1) - t * t


def relu(x: ArrayLike) -> NDArray:
    x = _arr(x)
    return np.maximum(x, x.dtype.type(0))


def drelu(x: ArrayLike) -> NDArray:
    x = _arr(x)
    return (x > 0).astype(x.dtype)  # strict: the derivative at the kink is 0


def softplus(x: ArrayLike) -> NDArray:
    x = _arr(x)
    return np.maximum(x, x.dtype.type(0)) + np.log1p(np.exp(-np.abs(x)))


def dsoftplus(x: ArrayLike) -> NDArray:
    return sigmoid(x)  # d/dx log(1 + e^x) = e^x / (1 + e^x)


def _gelu_u(x: NDArray) -> NDArray:
    """sqrt(2/pi) (x + 0.044715 x^3) with x clamped to [-10, 10]: tanh is
    already +-1 there (u(10) = 43.6), and x^3 would overflow for |x| > 1e102
    (|x| > 7e12 in float32)."""
    t = x.dtype.type
    xc = np.clip(x, t(-GELU_CLAMP), t(GELU_CLAMP))
    return t(GELU_C) * (xc + t(GELU_A) * xc * xc * xc)


def gelu_tanh(x: ArrayLike) -> NDArray:
    x = _arr(x)
    t = x.dtype.type
    return t(0.5) * x * (t(1) + np.tanh(_gelu_u(x)))


def dgelu_tanh(x: ArrayLike) -> NDArray:
    x = _arr(x)
    t = x.dtype.type
    xc = np.clip(x, t(-GELU_CLAMP), t(GELU_CLAMP))
    th = np.tanh(_gelu_u(x))
    du = t(GELU_C) * (t(1) + t(3 * GELU_A) * xc * xc)
    # product rule on 0.5 x (1 + tanh u), with tanh' = 1 - tanh^2
    return t(0.5) * (t(1) + th) + t(0.5) * x * (t(1) - th * th) * du


def _phi_cdf(x: NDArray) -> NDArray:
    """Phi(x) = (1 + erf(x / sqrt 2)) / 2 in float64."""
    return 0.5 * (1.0 + erf_series(x.astype(np.float64) / math.sqrt(2.0), ERF_TERMS))


def gelu_erf(x: ArrayLike) -> NDArray:
    x = _arr(x)
    return (x.astype(np.float64) * _phi_cdf(x)).astype(x.dtype)


def dgelu_erf(x: ArrayLike) -> NDArray:
    x = _arr(x)
    xd = x.astype(np.float64)
    xc = np.clip(xd, -PDF_CLAMP, PDF_CLAMP)  # xd * xd overflows for |x| > 1e154
    pdf = np.exp(-0.5 * xc * xc) / math.sqrt(2.0 * math.pi)
    return (_phi_cdf(x) + xd * pdf).astype(x.dtype)


def silu(x: ArrayLike) -> NDArray:
    x = _arr(x)
    return x * sigmoid(x)


def dsilu(x: ArrayLike) -> NDArray:
    x = _arr(x)
    # d/dx [x s(x)] = s(x) + x s(x) (1 - s(x)) = s(x) (1 + x s(-x))
    return sigmoid(x) * (x.dtype.type(1) + x * sigmoid(-x))
