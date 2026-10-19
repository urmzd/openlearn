"""A vanilla RNN with backpropagation through time written by hand (L3.1).

    h_t = tanh(x_t @ Wxh + h_{t-1} @ Whh + bh)

The forward keeps every h_t. The backward walks time in reverse: the gradient
reaching h_t is what the loss sends it directly (dh_all[t]) plus what step
t + 1 sends back through Whh. Each step is two matmul VJPs (M08.3) and the
tanh derivative 1 - h_t^2, read from the saved output. Truncated BPTT limits
how far back that walk goes; gradient_flow shows why it explodes or vanishes,
with the spectral radius of Whh (M03.4) as the predictor.

Contract: contracts/py/tinyllm/rnn/manual.pyi.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd.vjp import matmul_vjp
from tinyllm.linalg.eig import spectral_radius


class RNNCache(NamedTuple):
    x: NDArray
    h0: NDArray
    h: NDArray
    Wxh: NDArray
    Whh: NDArray


def _f64(a: ArrayLike) -> NDArray:
    return np.asarray(a, dtype=np.float64)


def _check(x: NDArray, h0: NDArray, Wxh: NDArray, Whh: NDArray, bh: NDArray) -> None:
    if x.ndim != 3:
        raise ValueError(f"x must be [T, B, D], got shape {x.shape}")
    T, B, D = x.shape
    if Whh.ndim != 2 or Whh.shape[0] != Whh.shape[1]:
        raise ValueError(f"Whh must be [H, H], got shape {Whh.shape}")
    H = Whh.shape[0]
    if Wxh.shape != (D, H):
        raise ValueError(f"Wxh must be [D, H] = {(D, H)}, got {Wxh.shape}")
    if h0.shape != (B, H):
        raise ValueError(f"h0 must be [B, H] = {(B, H)}, got {h0.shape}")
    if bh.shape != (H,):
        raise ValueError(f"bh must be [H] = {(H,)}, got {bh.shape}")


def rnn_forward(
    x: ArrayLike, h0: ArrayLike, Wxh: ArrayLike, Whh: ArrayLike, bh: ArrayLike
) -> tuple[NDArray, RNNCache]:
    x, h0, Wxh, Whh, bh = (_f64(a) for a in (x, h0, Wxh, Whh, bh))
    _check(x, h0, Wxh, Whh, bh)
    T, B, _ = x.shape
    h = np.empty((T, B, Whh.shape[0]), dtype=np.float64)
    prev = h0
    for t in range(T):
        prev = np.tanh(x[t] @ Wxh + prev @ Whh + bh)
        h[t] = prev
    return h, RNNCache(x, h0, h, Wxh, Whh)


def rnn_backward(
    dh_all: ArrayLike, cache: RNNCache, dh_next: Optional[ArrayLike] = None
) -> dict[str, NDArray]:
    x, h0, h, Wxh, Whh = cache
    g = _f64(dh_all)
    if g.shape != h.shape:
        raise ValueError(f"dh_all must have the shape of h, {h.shape}, got {g.shape}")
    carry = np.zeros_like(h0) if dh_next is None else _f64(dh_next)
    if carry.shape != h0.shape:
        raise ValueError(f"dh_next must be [B, H] = {h0.shape}, got {carry.shape}")
    dx = np.zeros_like(x)
    dWxh = np.zeros_like(Wxh)
    dWhh = np.zeros_like(Whh)
    dbh = np.zeros(Whh.shape[0], dtype=np.float64)
    for t in range(h.shape[0] - 1, -1, -1):
        # Both paths into h_t: the loss at step t, and step t + 1 through Whh.
        dh = g[t] + carry
        da = dh * (1.0 - h[t] * h[t])  # tanh'(a_t) = 1 - tanh(a_t)^2 = 1 - h_t^2
        h_prev = h[t - 1] if t > 0 else h0
        dx[t], dW = matmul_vjp(da, x[t], Wxh)
        dWxh += dW
        carry, dW = matmul_vjp(da, h_prev, Whh)
        dWhh += dW
        dbh += da.sum(axis=0)
    return {"x": dx, "h0": carry, "Wxh": dWxh, "Whh": dWhh, "bh": dbh}


def tbptt_windows(T: int, k1: int, k2: int) -> list[tuple[int, int]]:
    if T < 0 or k1 < 1 or k2 < k1:
        raise ValueError(f"need T >= 0 and 1 <= k1 <= k2, got T={T}, k1={k1}, k2={k2}")
    ends = list(range(k1, T + 1, k1))
    if T > 0 and (not ends or ends[-1] != T):
        ends.append(T)
    return [(max(0, e - k2), e) for e in ends]


def tbptt_grads(
    x: ArrayLike,
    h0: ArrayLike,
    Wxh: ArrayLike,
    Whh: ArrayLike,
    bh: ArrayLike,
    dh_all: ArrayLike,
    k1: int,
    k2: int,
) -> dict[str, NDArray]:
    h, cache = rnn_forward(x, h0, Wxh, Whh, bh)
    g = _f64(dh_all)
    if g.shape != h.shape:
        raise ValueError(f"dh_all must have the shape of h, {h.shape}, got {g.shape}")
    out = {
        "x": np.zeros_like(cache.x),
        "h0": np.zeros_like(cache.h0),
        "Wxh": np.zeros_like(cache.Wxh),
        "Whh": np.zeros_like(cache.Whh),
        "bh": np.zeros(cache.Whh.shape[0], dtype=np.float64),
    }
    prev_end = 0
    for start, end in tbptt_windows(h.shape[0], k1, k2):
        # The state entering the window is a constant: the cut.
        h_in = cache.h0 if start == 0 else h[start - 1]
        wcache = RNNCache(cache.x[start:end], h_in, h[start:end], cache.Wxh, cache.Whh)
        gw = np.zeros_like(h[start:end])
        # Only the losses since the last cut; earlier ones were counted then.
        gw[prev_end - start :] = g[prev_end:end]
        grads = rnn_backward(gw, wcache)
        out["x"][start:end] += grads["x"]
        if start == 0:
            out["h0"] += grads["h0"]
        for k in ("Wxh", "Whh", "bh"):
            out[k] += grads[k]
        prev_end = end
    return out


def gradient_flow(cache: RNNCache, dh_last: ArrayLike) -> tuple[NDArray, float]:
    h, Whh = cache.h, cache.Whh
    T = h.shape[0]
    if T == 0:
        raise ValueError("gradient_flow needs at least one step")
    g = _f64(dh_last)
    if g.shape != cache.h0.shape:
        raise ValueError(f"dh_last must be [B, H] = {cache.h0.shape}, got {g.shape}")
    norms = np.empty(T, dtype=np.float64)
    norms[T - 1] = np.linalg.norm(g)
    for t in range(T - 1, 0, -1):
        g = (g * (1.0 - h[t] * h[t])) @ Whh.T
        norms[t - 1] = np.linalg.norm(g)
    return norms, float(spectral_radius(Whh))
