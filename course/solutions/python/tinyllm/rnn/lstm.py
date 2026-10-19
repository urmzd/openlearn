"""The LSTM in torch's gate order i, f, g, o (L3.2).

A cell keeps two states: c, the memory, updated additively
(c' = f * c + i * g), and h = o * tanh(c), what the rest of the network sees.
Because c' depends on c through a multiplication by f and not through a
squashing matmul, the gradient along the cell state is scaled by f at each
step, and a forget gate near 1 carries it across hundreds of steps.

Everything is built from op-library calls (L0.2), so autograd does the
backward. Parameter names and shapes are torch's, so a torch state_dict
loads by name.

Contract: contracts/py/tinyllm/rnn/lstm.pyi.
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

GATES = 4  # i, f, g, o


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
    b_ih = np.zeros(GATES * d_h)
    b_ih[d_h : 2 * d_h] = 1.0  # forget gate open at the start: remember by default
    b_hh = np.zeros(GATES * d_h)
    return w_ih, w_hh, b_ih, b_hh


def lstm_cell(
    x: Tensor,
    h: Tensor,
    c: Tensor,
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
) -> tuple[Tensor, Tensor]:
    z = (
        F.matmul(x, F.transpose(w_ih, 0, 1))
        + b_ih
        + F.matmul(h, F.transpose(w_hh, 0, 1))
        + b_hh
    )
    H = z.shape[-1] // GATES
    i = F.sigmoid(z[..., 0:H])
    f = F.sigmoid(z[..., H : 2 * H])
    g = F.tanh(z[..., 2 * H : 3 * H])
    o = F.sigmoid(z[..., 3 * H : 4 * H])
    c_new = f * c + i * g
    return o * F.tanh(c_new), c_new


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


class LSTMCell(Module):
    def __init__(self, d_in: int, d_h: int, rng: Any = None) -> None:
        super().__init__()
        if d_in < 1 or d_h < 1:
            raise ValueError(f"LSTMCell({d_in}, {d_h}): sizes must be positive")
        w_ih, w_hh, b_ih, b_hh = _init(d_in, d_h, rng)
        self.weight_ih = _param(w_ih)
        self.weight_hh = _param(w_hh)
        self.bias_ih = _param(b_ih)
        self.bias_hh = _param(b_hh)

    def forward(
        self, x: Tensor, state: Optional[tuple[Any, Any]] = None
    ) -> tuple[Tensor, Tensor]:
        x = _as_tensor(x, np.float32)
        if state is None:
            H = self.weight_hh.shape[1]
            z = np.zeros(x.shape[:-1] + (H,), dtype=x.dtype)
            state = (z, z)
        h, c = state
        return lstm_cell(
            x, h, c, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh
        )


class LSTM(Module):
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
            raise ValueError(f"LSTM({d_in}, {d_h}, num_layers={num_layers}): sizes must be positive")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must lie in [0, 1], got {dropout}")
        self.input_size = d_in
        self.hidden_size = d_h
        self.num_layers = num_layers
        r = rng if rng is not None else PCG32(0).substream("init")
        for k in range(num_layers):
            w_ih, w_hh, b_ih, b_hh = _init(d_in if k == 0 else d_h, d_h, r)
            # torch's names, in torch's order: the safetensors key contract.
            setattr(self, f"weight_ih_l{k}", _param(w_ih))
            setattr(self, f"weight_hh_l{k}", _param(w_hh))
            setattr(self, f"bias_ih_l{k}", _param(b_ih))
            setattr(self, f"bias_hh_l{k}", _param(b_hh))
        self.drop = Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        state: Optional[tuple[Any, Any]] = None,
        lengths: Optional[ArrayLike] = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        x = _as_tensor(x, np.float32)
        if x.ndim != 3 or x.shape[2] != self.input_size or x.shape[0] < 1:
            raise ValueError(f"x must be [T >= 1, B, {self.input_size}], got {x.shape}")
        T, B, _ = x.shape
        L, H = self.num_layers, self.hidden_size
        if state is None:
            z = np.zeros((L, B, H), dtype=x.dtype)
            state = (z, z)
        h0, c0 = (_as_tensor(s, x.dtype) for s in state)
        if h0.shape != (L, B, H) or c0.shape != (L, B, H):
            raise ValueError(f"state must be two [{L}, {B}, {H}] arrays, got {h0.shape}, {c0.shape}")
        mask = _mask(lengths, T, B)
        seq = x
        h_n, c_n = [], []
        for k in range(L):
            w = [getattr(self, f"{n}_l{k}") for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")]
            h, c = h0[k], c0[k]
            outs = []
            for t in range(T):
                h_new, c_new = lstm_cell(seq[t], h, c, *w)
                if mask is None:
                    h, c = h_new, c_new
                    outs.append(h_new)
                else:
                    # Past its length a sequence keeps its state and outputs 0.
                    m = mask[t]
                    h = F.where(m, h_new, h)
                    c = F.where(m, c_new, c)
                    outs.append(F.where(m, h_new, 0.0))
            seq = F.stack(outs, axis=0)
            if k < L - 1:
                seq = self.drop(seq)
            h_n.append(h)
            c_n.append(c)
        return seq, (F.stack(h_n, axis=0), F.stack(c_n, axis=0))
