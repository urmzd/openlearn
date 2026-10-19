"""Bidirectional RNNs over padded batches (L3.4).

The backward direction reads every sequence from its own last real token to
its first. With padding, that is not x[::-1]: sequence b is reversed inside
its first lengths[b] steps and its padding stays at the end, so the backward
RNN starts on a real token and never reads padding before real input. One
gather does it, with the index lengths[b] - 1 - t, and the same gather puts
the backward outputs back in time order.

Contract: contracts/py/tinyllm/rnn/bi.pyi.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.module import Module


def reversal_index(lengths: ArrayLike, T: int) -> NDArray:
    n = np.asarray(lengths)
    if n.ndim != 1 or not np.issubdtype(n.dtype, np.integer):
        raise ValueError(f"lengths must be a 1-D integer array, got {n.dtype} {n.shape}")
    if (n < 1).any() or (n > T).any():
        raise ValueError(f"lengths must lie in 1..{T}, got {n.tolist()}")
    t = np.arange(T, dtype=np.int64)[:, None]
    n = n.astype(np.int64)[None, :]
    # Inside the length: mirror around the sequence's own middle. Padding: stay.
    return np.where(t < n, n - 1 - t, t)


def reverse_padded(x: Any, lengths: Optional[ArrayLike]) -> Any:
    if x.ndim < 2:
        raise ValueError(f"x must be [T, B, ...], got shape {x.shape}")
    T, B = x.shape[0], x.shape[1]
    if lengths is None:
        lengths = np.full(B, T, dtype=np.int64)
    if np.asarray(lengths).shape != (B,):
        raise ValueError(f"lengths must have {B} entries, got shape {np.asarray(lengths).shape}")
    idx = reversal_index(lengths, T)
    # x[idx[t, b], b]: a gather with two integer index arrays; for a Tensor
    # the getitem backward scatters the gradient back to the same places.
    return x[idx, np.arange(B)[None, :]]


def bidirectional(
    fwd: Module, bwd: Module, x: Tensor, lengths: Optional[ArrayLike] = None
) -> Tensor:
    if not isinstance(x, Tensor):
        x = Tensor(np.asarray(x))
    out_f, _ = fwd(x, lengths=lengths)
    out_r, _ = bwd(reverse_padded(x, lengths), lengths=lengths)
    return F.concat([out_f, reverse_padded(out_r, lengths)], axis=-1)
