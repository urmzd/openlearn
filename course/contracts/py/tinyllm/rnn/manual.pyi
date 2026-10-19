# contracts/py/tinyllm/rnn/manual.pyi (L3.1): a vanilla RNN with hand-written BPTT
# chapter: content/courses/build-an-llm/04-recurrent/01-vanilla-rnn-manual-bptt.md
#
# Plain numpy, no autograd: the forward keeps every hidden state, and the
# backward walks time in reverse, which is backpropagation through time.
# Row-vector convention (one row per batch element):
#
#     h_t = tanh(x_t @ Wxh + h_{t-1} @ Whh + bh),   t = 0 .. T-1, h_{-1} = h0
#
#   x    [T, B, D]  inputs, time first
#   h0   [B, H]     initial state
#   Wxh  [D, H]     input-to-hidden weights   (torch weight_ih_l0 is Wxh.T)
#   Whh  [H, H]     hidden-to-hidden weights  (torch weight_hh_l0 is Whh.T)
#   bh   [H]        bias                      (torch bias_ih_l0 + bias_hh_l0)
#   h    [T, B, H]  every hidden state, h[t] = h_t
#
# Arithmetic is float64 (inputs are converted). The matmul VJPs come from
# tinyllm.autograd.vjp.matmul_vjp (M08.3) and the spectral radius from
# tinyllm.linalg.eig.spectral_radius (M03.4). L3.6 (`cell='rnn'`) wraps
# rnn_forward and rnn_backward as one fused autograd op.
from typing import NamedTuple, Optional

from numpy.typing import ArrayLike, NDArray

class RNNCache(NamedTuple):
    """What the backward needs from the forward, nothing recomputed."""

    x: NDArray  # [T, B, D]
    h0: NDArray  # [B, H]
    h: NDArray  # [T, B, H], the hidden states the forward returned
    Wxh: NDArray  # [D, H]
    Whh: NDArray  # [H, H]

def rnn_forward(
    x: ArrayLike, h0: ArrayLike, Wxh: ArrayLike, Whh: ArrayLike, bh: ArrayLike
) -> tuple[NDArray, RNNCache]:
    """(h, cache) for the recurrence above. T = 0 gives h of shape (0, B, H).
    ValueError when the shapes do not fit together (x not 3-D, h0 not
    [B, H], Wxh not [D, H], Whh not [H, H], bh not [H])."""

def rnn_backward(
    dh_all: ArrayLike, cache: RNNCache, dh_next: Optional[ArrayLike] = None
) -> dict[str, NDArray]:
    """Gradients of a scalar loss L given dh_all[t] = the partial derivative
    of L with respect to h_t through every path EXCEPT the recurrence (the
    output layer's gradient at step t), [T, B, H]; dh_next [B, H] is the
    gradient arriving at h_{T-1} from after the sequence (None means 0).
    Walks t = T-1 .. 0 carrying dh_t = dh_all[t] + (gradient from step t+1):
        da_t   = dh_t * (1 - h_t^2)              tanh' from the saved output
        dx_t, dWxh += matmul_vjp(da_t, x_t, Wxh)
        dh_{t-1}, dWhh += matmul_vjp(da_t, h_{t-1}, Whh)
        dbh   += sum of da_t over the batch
    Returns {"x": [T,B,D], "h0": [B,H], "Wxh": [D,H], "Whh": [H,H], "bh": [H]},
    where "h0" is the gradient carried past t = 0. ValueError when dh_all is
    not the shape of cache.h or dh_next not [B, H]."""

def tbptt_windows(T: int, k1: int, k2: int) -> list[tuple[int, int]]:
    """Truncated BPTT(k1, k2) (Williams and Peng 1990): the forward runs
    straight through, and every k1 steps (and at T, if T is not a multiple
    of k1) the gradient of the losses of the steps since the last cut is
    propagated back k2 steps. Returns one (start, end) per cut, half-open:
    end = k1, 2 k1, ..., then T; start = max(0, end - k2). [] for T = 0.
    Example: T = 10, k1 = 4, k2 = 6 gives [(0, 4), (2, 8), (4, 10)].
    ValueError unless T >= 0 and 1 <= k1 <= k2."""

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
    """The truncated gradient: one forward over all T steps, then for each
    (start, end) of tbptt_windows(T, k1, k2), in order, rnn_backward over the
    window's slice of the cache with its own start state (h0 at start = 0,
    else h[start - 1], treated as a constant) and upstream dh_all restricted
    to the window's loss steps [previous end, end) (zero at the earlier steps
    of the window, whose losses an earlier cut already counted). Sums the
    window gradients; "x" gets each window's slice, "h0" only from windows
    that start at 0. Keys and shapes as rnn_backward. With k2 >= T it equals
    rnn_backward(dh_all, cache) exactly. ValueError as tbptt_windows and
    rnn_forward."""

def gradient_flow(cache: RNNCache, dh_last: ArrayLike) -> tuple[NDArray, float]:
    """The exploding and vanishing gradient diagnostic. With the loss's only
    gradient dh_last [B, H] at the last state h_{T-1}, returns
    (norms, rho): norms[t] = Frobenius norm of dL/dh_t for t = 0 .. T-1
    (norms[T-1] = ||dh_last||, then each step back multiplies by
    diag(1 - h_t^2) and Whh^T), and rho = spectral_radius(Whh) (M03.4). Near
    the linear regime (small h) the norms grow about like rho^(T-1-t) when
    rho > 1 and shrink when rho < 1. ValueError for T = 0 or a dh_last not
    [B, H]."""
