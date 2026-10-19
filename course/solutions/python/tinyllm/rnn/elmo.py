"""ELMo: a bidirectional language model, ScalarMix, and linear probes (L3.5).

Two LSTM language models read the same sentence in opposite directions and
share the embedding and the softmax. Their hidden states, layer by layer,
are contextual word vectors; ScalarMix learns how much of each layer a task
wants, and a linear probe measures what a frozen layer already encodes.

Contract: contracts/py/tinyllm/rnn/elmo.pyi.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd import functional as F
from tinyllm.autograd.losses import cross_entropy
from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import Embedding, Linear, ModuleList
from tinyllm.nn.module import Module
from tinyllm.num.rng import PCG32
from tinyllm.rnn.bi import reverse_padded
from tinyllm.rnn.lstm import LSTM


class ScalarMix(Module):
    def __init__(self, n_layers: int) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.n_layers = n_layers
        self.scalar_parameters = Tensor(
            np.zeros(n_layers, dtype=np.float32), requires_grad=True
        )
        self.gamma = Tensor(np.ones(1, dtype=np.float32), requires_grad=True)

    def weights(self) -> NDArray:
        s = self.scalar_parameters.data.astype(np.float64)
        w = np.exp(s - s.max())
        return w / w.sum()

    def forward(self, layers: Sequence[Tensor]) -> Tensor:
        if len(layers) != self.n_layers:
            raise ValueError(
                f"ScalarMix expects {self.n_layers} layers, got {len(layers)}"
            )
        shape = layers[0].shape
        if any(x.shape != shape for x in layers):
            raise ValueError(
                f"every layer must have shape {shape}, got {[x.shape for x in layers]}"
            )
        # Softmax of the scalars: the mix is a convex combination, rescaled by gamma.
        w = F.softmax(self.scalar_parameters, axis=0)
        mixed = layers[0] * w[0]
        for k in range(1, self.n_layers):
            mixed = mixed + layers[k] * w[k]
        return mixed * self.gamma


def _ids(
    ids: ArrayLike, vocab: int, lengths: Optional[ArrayLike]
) -> tuple[NDArray, NDArray]:
    a = np.asarray(ids)
    if a.ndim != 2 or a.shape[1] < 1 or (a.size and a.dtype.kind not in "iu"):
        raise ValueError(f"ids must be integers [B, T >= 1], got {a.dtype} {a.shape}")
    if a.min() < 0 or a.max() >= vocab:
        raise ValueError(f"ids must lie in [0, {vocab})")
    B, T = a.shape
    n = np.full(B, T, dtype=np.int64) if lengths is None else np.asarray(lengths)
    if n.shape != (B,) or n.dtype.kind not in "iu" or n.min() < 1 or n.max() > T:
        raise ValueError(
            f"lengths must be {B} integers in 1..{T}, got {np.asarray(lengths).tolist()}"
        )
    return a.astype(np.int64), n.astype(np.int64)


class BiLM(Module):
    def __init__(self, vocab: int, d: int, n_layers: int = 2, rng: Any = None) -> None:
        super().__init__()
        if min(vocab, d, n_layers) < 1:
            raise ValueError(f"sizes must be positive, got {(vocab, d, n_layers)}")
        r = rng if rng is not None else PCG32(0).substream("init")
        self.vocab, self.d, self.n_layers = vocab, d, n_layers
        self.emb = Embedding(vocab, d, rng=r)
        self.fwd = ModuleList([LSTM(d, d, rng=r) for _ in range(n_layers)])
        self.bwd = ModuleList([LSTM(d, d, rng=r) for _ in range(n_layers)])
        self.out = Linear(d, vocab, rng=r)

    def _stacks(
        self, ids: ArrayLike, lengths: Optional[ArrayLike]
    ) -> tuple[Tensor, list[Tensor], list[Tensor], NDArray]:
        """(e, forward layers, backward layers in source order), all time first."""
        a, n = _ids(ids, self.vocab, lengths)
        e = self.emb(a.T)  # [T, B, d]
        f_layers, b_layers = [], []
        f = e
        # The backward LM reads each sentence from its last REAL token, so the
        # reversal stays inside each length (L3.4), not a flip of the padding.
        b = reverse_padded(e, n)
        for k in range(self.n_layers):
            f, _ = self.fwd[k](f, lengths=n)
            b, _ = self.bwd[k](b, lengths=n)
            f_layers.append(f)
            b_layers.append(reverse_padded(b, n))  # back to source order
        return e, f_layers, b_layers, n

    def layers(
        self, ids: ArrayLike, lengths: Optional[ArrayLike] = None
    ) -> list[Tensor]:
        e, fs, bs, _ = self._stacks(ids, lengths)
        reps = [F.concat([e, e], axis=-1)] + [
            F.concat([f, b], axis=-1) for f, b in zip(fs, bs)
        ]
        return [F.transpose(r, 0, 1) for r in reps]

    def forward(
        self, ids: ArrayLike, lengths: Optional[ArrayLike] = None
    ) -> tuple[Tensor, Tensor]:
        _, fs, bs, _ = self._stacks(ids, lengths)
        return self.out(F.transpose(fs[-1], 0, 1)), self.out(F.transpose(bs[-1], 0, 1))


def bilm_loss(
    model: BiLM, ids: ArrayLike, lengths: Optional[ArrayLike] = None
) -> Tensor:
    a, n = _ids(ids, model.vocab, lengths)
    B, T = a.shape
    fl, bl = model(a, n)
    t = np.arange(T)[None, :]
    # Forward: position t predicts t + 1 (inside the length); backward: t - 1.
    fwd_tgt = np.full((B, T), -100, dtype=np.int64)
    fwd_tgt[:, :-1] = np.where(t[:, :-1] + 1 < n[:, None], a[:, 1:], -100)
    bwd_tgt = np.full((B, T), -100, dtype=np.int64)
    bwd_tgt[:, 1:] = np.where(t[:, 1:] < n[:, None], a[:, :-1], -100)
    if (fwd_tgt >= 0).sum() == 0:
        raise ValueError("bilm_loss: no row has a token to predict (every length is 1)")
    V = model.vocab
    lf = cross_entropy(F.reshape(fl, (-1, V)), fwd_tgt.reshape(-1))
    lb = cross_entropy(F.reshape(bl, (-1, V)), bwd_tgt.reshape(-1))
    return (lf + lb) * 0.5


def fit_linear_probe(
    features: ArrayLike,
    labels: ArrayLike,
    n_classes: int,
    l2: float = 1e-3,
    steps: int = 200,
    lr: float = 0.5,
) -> tuple[NDArray, NDArray]:
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    if X.ndim != 2:
        raise ValueError(f"features must be [N, D], got {X.shape}")
    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2, got {n_classes}")
    if (
        y.shape != (X.shape[0],)
        or y.dtype.kind not in "iu"
        or y.min() < 0
        or y.max() >= n_classes
    ):
        raise ValueError(f"labels must be {X.shape[0]} integers in [0, {n_classes})")
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")
    N, D = X.shape
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd
    Y = np.zeros((N, n_classes))
    Y[np.arange(N), y] = 1.0
    W = np.zeros((D, n_classes))
    b = np.zeros(n_classes)
    for _ in range(steps):
        s = Z @ W + b
        s = s - s.max(axis=1, keepdims=True)
        P = np.exp(s)
        P /= P.sum(axis=1, keepdims=True)
        G = P - Y
        W -= lr * (Z.T @ G / N + l2 * W)
        b -= lr * G.sum(axis=0) / N
    # Fold the standardization in: Z W + b = X (W / sd) + (b - (mu / sd) W).
    W_raw = W / sd[:, None]
    return W_raw, b - mu @ W_raw


def probe_accuracy(
    W: ArrayLike, b: ArrayLike, features: ArrayLike, labels: ArrayLike
) -> float:
    s = np.asarray(features, dtype=np.float64) @ np.asarray(
        W, dtype=np.float64
    ) + np.asarray(b)
    return float(np.mean(np.argmax(s, axis=1) == np.asarray(labels)))
