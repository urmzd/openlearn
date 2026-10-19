"""The GRU in torch's gate order r, z, n (L3.3).

One state h and two gates: z interpolates between the old state and a
candidate n (h' = (1 - z) * n + z * h), and r decides how much of the old
state the candidate sees. With z near 1 the state, and its gradient, pass
through a step almost unchanged, which is the LSTM's trick with one state
instead of two and three gate blocks instead of four.

Everything is built from op-library calls (L0.2), so autograd does the
backward. Parameter names and shapes are torch's.

Contract: contracts/py/tinyllm/rnn/gru.pyi.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.linalg.qr import orthogonal_init
from tinyllm.nn.init import xavier_uniform
from tinyllm.nn.layers import Dropout
from tinyllm.nn.module import Module
from tinyllm.num.rng import PCG32

GATES = 3  # r, z, n


def _param(a: ArrayLike) -> Tensor:
    return Tensor(np.asarray(a, dtype=np.float32), requires_grad=True)


def _as_tensor(a: Any, dtype: Any) -> Tensor:
    """A Tensor stays itself; anything else becomes a constant."""
    return a if isinstance(a, Tensor) else Tensor(np.asarray(a), dtype=dtype)


def _init(d_in: int, d_h: int, rng: Any) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """(w_ih, w_hh, b_ih, b_hh) for one layer, drawn in the contract's order."""
    r = rng if rng is not None else PCG32(0).substream("init")
    w_ih = np.concatenate([xavier_uniform((d_h, d_in), 1.0, r) for _ in range(GATES)])
    w_hh = np.concatenate([orthogonal_init((d_h, d_h), 1.0, r) for _ in range(GATES)])
    return w_ih, w_hh, np.zeros(GATES * d_h), np.zeros(GATES * d_h)


def gru_cell(
    x: Tensor,
    h: Tensor,
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
) -> Tensor:
    a = F.matmul(x, F.transpose(w_ih, 0, 1)) + b_ih
    b = F.matmul(h, F.transpose(w_hh, 0, 1)) + b_hh
    H = a.shape[-1] // GATES
    r = F.sigmoid(a[..., 0:H] + b[..., 0:H])
    z = F.sigmoid(a[..., H : 2 * H] + b[..., H : 2 * H])
    # torch's (and cuDNN's) form: r scales h @ w_hn^T + b_hn, bias included.
    n = F.tanh(a[..., 2 * H : 3 * H] + r * b[..., 2 * H : 3 * H])
    return (1.0 - z) * n + z * h


def _mask(lengths: Optional[ArrayLike], T: int, B: int) -> Optional[NDArray]:
    """bool [T, B, 1]: step t of sequence b is real (t < lengths[b])."""
    if lengths is None:
        return None
    n = np.asarray(lengths)
    if n.shape != (B,) or not np.issubdtype(n.dtype, np.integer):
        raise ValueError(f"lengths must be {B} integers, got {n.dtype} {n.shape}")
    if (n < 1).any() or (n > T).any():
        raise ValueError(f"lengths must lie in 1..{T}, got {n.tolist()}")
    return (np.arange(T)[:, None] < n[None, :])[:, :, None]


class GRUCell(Module):
    def __init__(self, d_in: int, d_h: int, rng: Any = None) -> None:
        super().__init__()
        if d_in < 1 or d_h < 1:
            raise ValueError(f"GRUCell({d_in}, {d_h}): sizes must be positive")
        w_ih, w_hh, b_ih, b_hh = _init(d_in, d_h, rng)
        self.weight_ih = _param(w_ih)
        self.weight_hh = _param(w_hh)
        self.bias_ih = _param(b_ih)
        self.bias_hh = _param(b_hh)

    def forward(self, x: Tensor, h: Optional[Any] = None) -> Tensor:
        x = _as_tensor(x, np.float32)
        if h is None:
            h = np.zeros(x.shape[:-1] + (self.weight_hh.shape[1],), dtype=x.dtype)
        return gru_cell(x, h, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)


class GRU(Module):
    def __init__(
        self,
        d_in: int,
        d_h: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        rng: Any = None,
    ) -> None:
        super().__init__()
        if d_in < 1 or d_h < 1 or num_layers < 1:
            raise ValueError(f"GRU({d_in}, {d_h}, num_layers={num_layers}): sizes must be positive")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must lie in [0, 1], got {dropout}")
        self.input_size = d_in
        self.hidden_size = d_h
        self.num_layers = num_layers
        r = rng if rng is not None else PCG32(0).substream("init")
        for k in range(num_layers):
            w_ih, w_hh, b_ih, b_hh = _init(d_in if k == 0 else d_h, d_h, r)
            setattr(self, f"weight_ih_l{k}", _param(w_ih))
            setattr(self, f"weight_hh_l{k}", _param(w_hh))
            setattr(self, f"bias_ih_l{k}", _param(b_ih))
            setattr(self, f"bias_hh_l{k}", _param(b_hh))
        self.drop = Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        state: Optional[Any] = None,
        lengths: Optional[ArrayLike] = None,
    ) -> tuple[Tensor, Tensor]:
        x = _as_tensor(x, np.float32)
        if x.ndim != 3 or x.shape[2] != self.input_size or x.shape[0] < 1:
            raise ValueError(f"x must be [T >= 1, B, {self.input_size}], got {x.shape}")
        T, B, _ = x.shape
        L, H = self.num_layers, self.hidden_size
        h0 = _as_tensor(np.zeros((L, B, H), dtype=x.dtype) if state is None else state, x.dtype)
        if h0.shape != (L, B, H):
            raise ValueError(f"state must be [{L}, {B}, {H}], got {h0.shape}")
        mask = _mask(lengths, T, B)
        seq = x
        h_n = []
        for k in range(L):
            w = [getattr(self, f"{n}_l{k}") for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")]
            h = h0[k]
            outs = []
            for t in range(T):
                h_new = gru_cell(seq[t], h, *w)
                if mask is None:
                    h = h_new
                    outs.append(h_new)
                else:
                    m = mask[t]
                    h = F.where(m, h_new, h)
                    outs.append(F.where(m, h_new, 0.0))
            seq = F.stack(outs, axis=0)
            if k < L - 1:
                seq = self.drop(seq)
            h_n.append(h)
        return seq, F.stack(h_n, axis=0)
