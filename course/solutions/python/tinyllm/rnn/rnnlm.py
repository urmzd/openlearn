"""An RNN language model trained with stateful truncated BPTT (L3.6).

The model is an embedding, a recurrent layer (Elman RNN, LSTM, or GRU), and
a linear read-out. Training cuts the token stream into contiguous lanes and
walks them window by window: the hidden state flows from one window into the
next, but the gradient stops at the window boundary (the state is detached).
That is what lets a recurrent model see context longer than the window it
backpropagates through.

Contract: contracts/py/tinyllm/rnn/rnnlm.pyi.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd import functional as F
from tinyllm.autograd.losses import cross_entropy
from tinyllm.autograd.mode import no_grad
from tinyllm.autograd.tensor import Tensor, from_op
from tinyllm.io.safetensors import load_safetensors, save_safetensors
from tinyllm.linalg.qr import orthogonal_init
from tinyllm.nn.init import xavier_uniform
from tinyllm.nn.layers import Embedding, Linear
from tinyllm.nn.module import Module
from tinyllm.num.rng import PCG32
from tinyllm.rnn.gru import GRU
from tinyllm.rnn.lstm import LSTM
from tinyllm.rnn.manual import rnn_backward, rnn_forward
from tinyllm.train.loop import train_step

CELLS = ("rnn", "lstm", "gru")
_TOKENIZERS = ("bytes", "file")


def _param(a: ArrayLike) -> Tensor:
    return Tensor(np.asarray(a, dtype=np.float32), requires_grad=True)


def _tensor(a: Any, dtype: Any) -> Tensor:
    return a if isinstance(a, Tensor) else Tensor(np.asarray(a), dtype=dtype)


def _rnn_layer(
    x: Tensor, h0: Tensor, w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor
) -> Tensor:
    """One Elman layer over the whole sequence as a single autograd op."""
    h, cache = rnn_forward(
        x.data, h0.data, w_ih.data.T, w_hh.data.T, b_ih.data + b_hh.data
    )

    def vjp(g: NDArray):
        d = rnn_backward(g, cache)
        db = d["bh"].astype(b_ih.dtype)
        # torch's weights are the transposes of L3.1's Wxh and Whh; the two
        # biases are summed in the forward, so each gets the same gradient.
        return (
            d["x"].astype(x.dtype),
            d["h0"].astype(h0.dtype),
            d["Wxh"].T.astype(w_ih.dtype),
            d["Whh"].T.astype(w_hh.dtype),
            db,
            db.copy(),
        )

    return from_op(h.astype(x.dtype), (x, h0, w_ih, w_hh, b_ih, b_hh), vjp, "elman_rnn")


class ElmanRNN(Module):
    def __init__(
        self, d_in: int, d_h: int, num_layers: int = 1, rng: Any = None
    ) -> None:
        super().__init__()
        if d_in < 1 or d_h < 1 or num_layers < 1:
            raise ValueError(
                f"ElmanRNN({d_in}, {d_h}, num_layers={num_layers}): sizes must be positive"
            )
        self.input_size, self.hidden_size, self.num_layers = d_in, d_h, num_layers
        r = rng if rng is not None else PCG32(0).substream("init")
        for k in range(num_layers):
            setattr(
                self,
                f"weight_ih_l{k}",
                _param(xavier_uniform((d_h, d_in if k == 0 else d_h), 1.0, r)),
            )
            setattr(
                self, f"weight_hh_l{k}", _param(orthogonal_init((d_h, d_h), 1.0, r))
            )
            setattr(self, f"bias_ih_l{k}", _param(np.zeros(d_h)))
            setattr(self, f"bias_hh_l{k}", _param(np.zeros(d_h)))

    def forward(
        self,
        x: Tensor,
        state: Optional[Any] = None,
        lengths: Optional[ArrayLike] = None,
    ) -> tuple[Tensor, Tensor]:
        if lengths is not None:
            raise ValueError(
                "ElmanRNN takes no lengths: a language-model stream has no padding"
            )
        x = _tensor(x, np.float32)
        if x.ndim != 3 or x.shape[2] != self.input_size or x.shape[0] < 1:
            raise ValueError(f"x must be [T >= 1, B, {self.input_size}], got {x.shape}")
        L, B, H = self.num_layers, x.shape[1], self.hidden_size
        h0 = _tensor(
            np.zeros((L, B, H), dtype=x.dtype) if state is None else state, x.dtype
        )
        if h0.shape != (L, B, H):
            raise ValueError(f"state must be [{L}, {B}, {H}], got {h0.shape}")
        seq, h_n = x, []
        for k in range(L):
            w = [
                getattr(self, f"{n}_l{k}")
                for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")
            ]
            seq = _rnn_layer(seq, h0[k], *w)
            h_n.append(seq[seq.shape[0] - 1])
        return seq, F.stack(h_n, axis=0)


class RNNLM(Module):
    def __init__(
        self,
        vocab: int,
        d_emb: int,
        d_h: int,
        cell: Literal["rnn", "lstm", "gru"],
        n_layers: int = 1,
        rng: Any = None,
    ) -> None:
        super().__init__()
        if min(vocab, d_emb, d_h, n_layers) < 1:
            raise ValueError(
                f"sizes must be positive, got {(vocab, d_emb, d_h, n_layers)}"
            )
        if cell not in CELLS:
            raise ValueError(f"cell must be one of {CELLS}, got {cell!r}")
        r = rng if rng is not None else PCG32(0).substream("init")
        self.vocab, self.d_emb, self.d_h, self.cell, self.n_layers = (
            vocab,
            d_emb,
            d_h,
            cell,
            n_layers,
        )
        self.emb = Embedding(vocab, d_emb, rng=r)
        kind = {"rnn": ElmanRNN, "lstm": LSTM, "gru": GRU}[cell]
        self.rnn = kind(d_emb, d_h, n_layers, rng=r)
        self.out = Linear(d_h, vocab, rng=r)

    def init_state(self, batch: int) -> Any:
        z = np.zeros((self.n_layers, batch, self.d_h), dtype=np.float32)
        return (z, z.copy()) if self.cell == "lstm" else z

    def _ids(self, ids: ArrayLike) -> NDArray:
        a = np.asarray(ids)
        if a.ndim != 2 or a.shape[1] < 1 or (a.size and a.dtype.kind not in "iu"):
            raise ValueError(
                f"ids must be integers [B, T >= 1], got {a.dtype} {a.shape}"
            )
        if a.size and (a.min() < 0 or a.max() >= self.vocab):
            raise ValueError(
                f"ids must lie in [0, {self.vocab}), got [{a.min()}, {a.max()}]"
            )
        return a.astype(np.int64)

    def forward(
        self, ids: ArrayLike, state: Optional[Any] = None
    ) -> tuple[Tensor, Any]:
        a = self._ids(ids)
        B, T = a.shape
        if state is None:
            state = self.init_state(B)
        x = self.emb(a.T)  # time first for the recurrent layer: [T, B, d_emb]
        out, new_state = self.rnn(x, state)
        logits = self.out(F.transpose(out, 0, 1))  # back to batch first: [B, T, V]
        return logits, new_state

    def detach_state(self, state: Any) -> Any:
        if isinstance(state, tuple):
            return tuple(self.detach_state(s) for s in state)
        return state.detach() if isinstance(state, Tensor) else state

    def nll(self, ids: ArrayLike, chunk: int = 256) -> NDArray:
        a = np.asarray(ids)
        if a.ndim != 1 or len(a) < 2:
            raise ValueError(
                f"nll needs a 1-D stream of at least 2 ids, got shape {a.shape}"
            )
        if chunk < 1:
            raise ValueError(f"chunk must be >= 1, got {chunk}")
        out = np.empty(len(a) - 1, dtype=np.float64)
        state = None
        with no_grad():
            for s in range(0, len(a) - 1, chunk):
                inp = a[s : min(s + chunk, len(a) - 1)]
                logits, state = self.forward(inp[None, :], state)
                z = logits.data[0].astype(np.float64)
                m = z.max(axis=-1, keepdims=True)
                ls = z - m - np.log(np.exp(z - m).sum(axis=-1, keepdims=True))
                tgt = a[s + 1 : s + 1 + len(inp)]
                out[s : s + len(inp)] = -ls[np.arange(len(inp)), tgt]
        return out

    def generate(
        self, prefix: Sequence[int], n: int, temperature: float, seed: int
    ) -> list[int]:
        hist = [int(t) for t in prefix]
        if not hist:
            raise ValueError("RNNLM.generate needs a non-empty prefix")
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        if not temperature >= 0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        rng = PCG32(seed)
        out: list[int] = []
        with no_grad():
            logits, state = self.forward(np.asarray([hist], dtype=np.int64))
            for _ in range(n):
                row = logits.data[0, -1].astype(np.float64)
                nxt = (
                    int(np.argmax(row))
                    if temperature == 0
                    else _draw(row, float(temperature), rng.uniform())
                )
                out.append(nxt)
                logits, state = self.forward(np.asarray([[nxt]], dtype=np.int64), state)
        return out


def _draw(row: NDArray, temperature: float, u: float) -> int:
    """L0.5's draw: weights exp(z - max z) of z = row / temperature, summed in
    id order; the first id whose running sum exceeds u * total."""
    z = row / temperature
    w = np.exp(z - z.max()).tolist()
    total = 0.0
    for wi in w:
        total += wi
    target = u * total
    cum, last = 0.0, 0
    for i, wi in enumerate(w):
        if wi > 0.0:
            cum += wi
            last = i
            if cum > target:
                return i
    return last


def tbptt_batches(
    stream: ArrayLike, k: int, batch: int
) -> list[tuple[NDArray, NDArray]]:
    s = np.asarray(stream)
    if s.ndim != 1 or (s.size and s.dtype.kind not in "iu"):
        raise ValueError(f"stream must be a 1-D integer array, got {s.dtype} {s.shape}")
    if k < 1 or batch < 1:
        raise ValueError(f"k and batch must be >= 1, got k={k}, batch={batch}")
    L = len(s) // batch
    if L < k + 1:
        raise ValueError(
            f"each of {batch} lanes has {L} tokens; a window needs k + 1 = {k + 1}"
        )
    # Lane b is one contiguous stretch of the stream, so the state that lane b
    # carries from window j into window j + 1 belongs to the same text.
    lanes = s[: L * batch].astype(np.int64).reshape(batch, L)
    return [
        (lanes[:, p : p + k], lanes[:, p + 1 : p + k + 1]) for p in range(0, L - k, k)
    ]


def train_tbptt(
    model: RNNLM,
    stream: ArrayLike,
    k: int,
    batch: int,
    opt: Any,
    clip: Optional[float],
    steps: int,
) -> list[float]:
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")
    windows = tbptt_batches(stream, k, batch)
    carry: dict[str, Any] = {"state": None}

    def loss_fn(m: RNNLM, b: dict) -> Tensor:
        logits, state = m(b["x"], carry["state"])
        # Keep the values, cut the graph: the next window starts here.
        carry["state"] = m.detach_state(state)
        V = logits.shape[-1]
        return cross_entropy(F.reshape(logits, (-1, V)), b["y"].reshape(-1))

    losses: list[float] = []
    model.train()
    j = 0
    while len(losses) < steps:
        if j == 0:
            carry["state"] = None  # a new epoch starts from zeros
        x, y = windows[j]
        losses.append(
            train_step(model, {"x": x, "y": y}, loss_fn, opt, clip=clip)["loss"]
        )
        j = (j + 1) % len(windows)
    return losses


def save_rnnlm(model: RNNLM, dir: str, tokenizer: str = "bytes") -> None:
    if tokenizer not in _TOKENIZERS:
        raise ValueError(
            f"save_rnnlm: tokenizer must be one of {_TOKENIZERS}, got {tokenizer!r}"
        )
    d = Path(dir)
    d.mkdir(parents=True, exist_ok=True)
    config = {
        "tl_arch": "rnnlm",
        "tl_tokenizer": tokenizer,
        "vocab_size": model.vocab,
        "tl_format": 1,
        "tl_cell": model.cell,
        "tl_d_emb": model.d_emb,
        "hidden_size": model.d_h,
        "num_hidden_layers": model.n_layers,
    }
    (d / "config.json").write_text(json.dumps(config) + "\n")
    tensors = {
        k: np.asarray(v, dtype=np.float32) for k, v in model.state_dict().items()
    }
    save_safetensors(
        str(d / "model.safetensors"), tensors, {"format": "tinyllm", "tl_arch": "rnnlm"}
    )


def load_rnnlm(dir: str) -> RNNLM:
    d = Path(dir)
    cfg = json.loads((d / "config.json").read_text())
    if cfg.get("tl_arch") != "rnnlm":
        raise ValueError(
            f"{d}/config.json: tl_arch is {cfg.get('tl_arch')!r}, not 'rnnlm'"
        )
    try:
        model = RNNLM(
            int(cfg["vocab_size"]),
            int(cfg["tl_d_emb"]),
            int(cfg["hidden_size"]),
            cfg["tl_cell"],
            int(cfg["num_hidden_layers"]),
        )
    except KeyError as e:
        raise ValueError(f"{d}/config.json: missing key {e.args[0]!r}") from None
    tensors, _ = load_safetensors(str(d / "model.safetensors"))
    model.load_state_dict(tensors, strict=True)
    return model
