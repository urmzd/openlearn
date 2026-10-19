"""Luong attention: dot, general, and concat scores, and input feeding (L4.3).

The read is Bahdanau's (L4.2): scores, a softmax with padding masked to -inf,
a weighted average of the keys. What changes is the query (the state after
this step's recurrence) and the score function. The attentional state
h~ = tanh(W_c [c ; h]) is what the decoder's output layer reads and what
input feeding passes to the next step.

Contract: contracts/py/tinyllm/seq2seq/luong.pyi.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
from numpy.typing import ArrayLike

from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import Linear
from tinyllm.nn.module import Module
from tinyllm.num.rng import PCG32

SCORES = ("dot", "general", "concat")


class LuongAttention(Module):
    query_from: Literal["current"] = "current"

    def __init__(
        self, d: int, score: Literal["dot", "general", "concat"], rng: Any = None
    ) -> None:
        super().__init__()
        if d < 1:
            raise ValueError(f"d must be positive, got {d}")
        if score not in SCORES:
            raise ValueError(f"score must be one of {SCORES}, got {score!r}")
        r = rng if rng is not None else PCG32(0).substream("init")
        self.d, self.score = d, score
        if score == "general":
            self.score_proj = Linear(d, d, bias=False, rng=r)
        elif score == "concat":
            self.score_proj = Linear(2 * d, d, bias=False, rng=r)
            self.v = Linear(d, 1, bias=False, rng=r)
        self.combine = Linear(2 * d, d, bias=False, rng=r)

    def _check_keys(self, keys: Tensor) -> None:
        if keys.ndim != 3 or keys.shape[2] != self.d:
            raise ValueError(f"keys must be [B, S, {self.d}], got {keys.shape}")

    def project_keys(self, keys: Tensor) -> Tensor:
        self._check_keys(keys)
        if self.score == "dot":
            return keys
        if self.score == "general":
            return self.score_proj(keys)
        # concat: W_a [h ; k] = W_h h + W_k k, and W_k k does not depend on h.
        w_k = self.score_proj.weight[:, self.d :]
        return F.matmul(keys, F.transpose(w_k, 0, 1))

    def scores(
        self, query: Tensor, keys: Tensor, proj: Optional[Tensor] = None
    ) -> Tensor:
        self._check_keys(keys)
        B, S, d = keys.shape
        if query.ndim != 2 or query.shape != (B, d):
            raise ValueError(f"query must be [{B}, {d}], got {query.shape}")
        if proj is None:
            proj = self.project_keys(keys)
        elif proj.shape != (B, S, d):
            raise ValueError(f"proj must be [B, S, d] = {(B, S, d)}, got {proj.shape}")
        if self.score in ("dot", "general"):
            # One dot product per position: [B, S, d] @ [B, d, 1].
            e = F.matmul(proj, F.reshape(query, (B, d, 1)))
            return F.reshape(e, (B, S))
        w_h = self.score_proj.weight[:, :d]
        hq = F.reshape(F.matmul(query, F.transpose(w_h, 0, 1)), (B, 1, d))
        e = self.v(F.tanh(proj + hq))
        return F.reshape(e, (B, S))

    def forward(
        self,
        query: Tensor,
        keys: Tensor,
        mask: ArrayLike,
        proj: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        self._check_keys(keys)
        B, S, d = keys.shape
        m = np.asarray(mask)
        if m.shape != (B, S) or m.dtype.kind not in "biu":
            raise ValueError(
                f"mask must be bool [B, S] = {(B, S)}, got {m.dtype} {m.shape}"
            )
        e = F.masked_fill(self.scores(query, keys, proj), ~m.astype(bool), -np.inf)
        a = F.softmax(e, axis=-1)
        ctx = F.matmul(F.reshape(a, (B, 1, S)), keys)
        return F.reshape(ctx, (B, d)), a

    def attentional(self, query: Tensor, context: Tensor) -> Tensor:
        if query.ndim != 2 or query.shape[1] != self.d or context.shape != query.shape:
            raise ValueError(
                f"query and context must both be [B, {self.d}], got {query.shape} and {context.shape}"
            )
        # Luong's order: the context first, then the decoder state.
        return F.tanh(self.combine(F.concat([context, query], axis=-1)))
