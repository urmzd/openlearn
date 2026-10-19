"""Bahdanau additive attention (L4.2).

A decoder state asks "which source positions matter now?" by scoring every
encoder output with a one-hidden-layer network, turning the scores into
weights with a softmax, and reading the weighted average. Padding positions
are scored -inf before the softmax, so their weight is exactly 0.

Contract: contracts/py/tinyllm/seq2seq/additive.pyi.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import Linear
from tinyllm.nn.module import Module
from tinyllm.num.rng import PCG32


def length_mask(lengths: ArrayLike, S: int) -> NDArray:
    n = np.asarray(lengths)
    if n.ndim != 1 or (n.size and n.dtype.kind not in "iu"):
        raise ValueError(
            f"lengths must be a 1-D integer array, got {n.dtype} {n.shape}"
        )
    if n.size and (n.min() < 0 or n.max() > S):
        raise ValueError(f"lengths must lie in [0, {S}], got {n.tolist()}")
    return np.arange(S)[None, :] < n[:, None]


def _check_mask(mask: ArrayLike, B: int, S: int) -> NDArray:
    m = np.asarray(mask)
    if m.shape != (B, S) or m.dtype.kind not in "biu":
        raise ValueError(
            f"mask must be bool [B, S] = {(B, S)}, got {m.dtype} {m.shape}"
        )
    return m.astype(bool)


class AdditiveAttention(Module):
    query_from: Literal["previous"] = "previous"

    def __init__(self, d_query: int, d_key: int, d_attn: int, rng: Any = None) -> None:
        super().__init__()
        if min(d_query, d_key, d_attn) < 1:
            raise ValueError(f"sizes must be positive, got {(d_query, d_key, d_attn)}")
        r = rng if rng is not None else PCG32(0).substream("init")
        self.d_query, self.d_key, self.d_attn = d_query, d_key, d_attn
        self.query = Linear(d_query, d_attn, bias=False, rng=r)
        self.key = Linear(d_key, d_attn, rng=r)
        self.v = Linear(d_attn, 1, bias=False, rng=r)

    def project_keys(self, keys: Tensor) -> Tensor:
        if keys.ndim != 3 or keys.shape[2] != self.d_key:
            raise ValueError(f"keys must be [B, S, {self.d_key}], got {keys.shape}")
        return self.key(keys)

    def scores(
        self, query: Tensor, keys: Tensor, proj: Optional[Tensor] = None
    ) -> Tensor:
        if query.ndim != 2 or query.shape[1] != self.d_query:
            raise ValueError(f"query must be [B, {self.d_query}], got {query.shape}")
        B, S = keys.shape[0], keys.shape[1]
        if proj is None:
            proj = self.project_keys(keys)
        elif proj.shape != (B, S, self.d_attn):
            raise ValueError(
                f"proj must be [B, S, A] = {(B, S, self.d_attn)}, got {proj.shape}"
            )
        # W q is one vector per batch row, added to every source position.
        wq = F.reshape(self.query(query), (B, 1, self.d_attn))
        e = self.v(F.tanh(proj + wq))  # [B, S, 1]
        return F.reshape(e, (B, S))

    def forward(
        self,
        query: Tensor,
        keys: Tensor,
        mask: ArrayLike,
        proj: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        if keys.ndim != 3 or keys.shape[2] != self.d_key:
            raise ValueError(f"keys must be [B, S, {self.d_key}], got {keys.shape}")
        B, S = keys.shape[0], keys.shape[1]
        if query.ndim != 2 or query.shape[0] != B:
            raise ValueError(f"query must be [{B}, {self.d_query}], got {query.shape}")
        m = _check_mask(mask, B, S)
        e = F.masked_fill(self.scores(query, keys, proj), ~m, -np.inf)
        # Mask BEFORE the softmax: a masked score of -inf has weight exactly 0
        # and the rest still sum to 1.
        a = F.softmax(e, axis=-1)
        ctx = F.matmul(F.reshape(a, (B, 1, S)), keys)  # [B, 1, Dk]
        return F.reshape(ctx, (B, self.d_key)), a
