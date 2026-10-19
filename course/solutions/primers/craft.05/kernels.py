"""craft.05 kata: four forward passes and their hand-written backward passes.

These are the ops of Parts 5 to 7 (a Linear layer, RMSNorm, the SwiGLU gate,
scaled dot-product attention) written the way a C kernel (L9) or a fused
CUDA kernel has to write them: no autograd, the vector-Jacobian product
derived by hand. Every array is numpy float64. Your graded tests check them
against oracles (chapter section 4); the planted faults are the chapter's
pitfalls, among them the transposed weight gradient.

The rules, exactly:

linear(x, W, b) = x @ W.T + b          x [N, i], W [o, i], b [o] -> y [N, o]
linear_backward(x, W, gy) -> (gx, gW, gb) for an upstream gy [N, o]:
    gx = gy @ W          [N, i]
    gW = gy.T @ x        [o, i]   (the shape of W)
    gb = gy.sum(axis=0)  [o]

rmsnorm(x, w, eps) = x / r * w,  r = sqrt(mean(x * x, axis=-1) + eps)   (per row)
rmsnorm_backward(x, w, gy, eps) -> (gx, gw):
    n = x / r;  gw = (gy * n).sum over every axis but the last
    u = gy * w;  gx = (u - n * mean(u * n, axis=-1)) / r

swiglu(a, b) = silu(a) * b,  silu(a) = a * sigmoid(a)
swiglu_backward(a, b, gh) -> (ga, gb):
    ga = gh * b * (s + a * s * (1 - s)),  s = sigmoid(a)
    gb = gh * silu(a)

attention(q, k, v, causal=False) = P @ v,  P = softmax(S), S = q @ k^T / sqrt(d)
    q [..., Tq, d], k [..., Tk, d], v [..., Tk, dv]; causal hides key j > query i
    (rows aligned at the start: query i sits at position i). The softmax
    subtracts the row maximum first.
attention_backward(q, k, v, go, causal=False) -> (gq, gk, gv):
    gv = P^T @ go
    gP = go @ v^T
    gS = P * (gP - sum(P * gP, axis=-1))      (the softmax VJP; hidden keys get 0)
    gq = gS @ k / sqrt(d);  gk = gS^T @ q / sqrt(d)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def linear(x: NDArray, W: NDArray, b: NDArray) -> NDArray:
    return x @ W.T + b


def linear_backward(x: NDArray, W: NDArray, gy: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    gx = gy @ W
    gW = gy.T @ x
    gb = gy.sum(axis=0)
    return gx, gW, gb


def _rms(x: NDArray, eps: float) -> NDArray:
    return np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def rmsnorm(x: NDArray, w: NDArray, eps: float = 1e-6) -> NDArray:
    return x / _rms(x, eps) * w


def rmsnorm_backward(x: NDArray, w: NDArray, gy: NDArray, eps: float = 1e-6) -> tuple[NDArray, NDArray]:
    r = _rms(x, eps)
    n = x / r
    gw = (gy * n).reshape(-1, x.shape[-1]).sum(axis=0)
    u = gy * w
    gx = (u - n * np.mean(u * n, axis=-1, keepdims=True)) / r
    return gx, gw


def _sigmoid(a: NDArray) -> NDArray:
    return 0.5 * (1.0 + np.tanh(0.5 * a))  # no overflow for large |a|


def swiglu(a: NDArray, b: NDArray) -> NDArray:
    return a * _sigmoid(a) * b


def swiglu_backward(a: NDArray, b: NDArray, gh: NDArray) -> tuple[NDArray, NDArray]:
    s = _sigmoid(a)
    ga = gh * b * (s + a * s * (1.0 - s))
    gb = gh * a * s
    return ga, gb


def _probs(q: NDArray, k: NDArray, causal: bool) -> NDArray:
    S = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
    if causal:
        Tq, Tk = S.shape[-2], S.shape[-1]
        S = np.where(np.arange(Tk)[None, :] <= np.arange(Tq)[:, None], S, -np.inf)
    S = S - S.max(axis=-1, keepdims=True)
    P = np.exp(S)
    return P / P.sum(axis=-1, keepdims=True)


def attention(q: NDArray, k: NDArray, v: NDArray, causal: bool = False) -> NDArray:
    return _probs(q, k, causal) @ v


def attention_backward(
    q: NDArray, k: NDArray, v: NDArray, go: NDArray, causal: bool = False
) -> tuple[NDArray, NDArray, NDArray]:
    P = _probs(q, k, causal)
    gv = np.swapaxes(P, -1, -2) @ go
    gP = go @ np.swapaxes(v, -1, -2)
    gS = P * (gP - np.sum(P * gP, axis=-1, keepdims=True))
    scale = 1.0 / np.sqrt(q.shape[-1])
    gq = gS @ k * scale
    gk = np.swapaxes(gS, -1, -2) @ q * scale
    return gq, gk, gv
