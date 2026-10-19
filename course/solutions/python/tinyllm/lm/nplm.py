"""tinyllm.lm.nplm (L2.2): Bengio's neural probabilistic language model.

An n-gram model treats "the cat sat" and "the dog sat" as unrelated
strings: what it learns about one tells it nothing about the other. Bengio
et al. (2003) replaced the count table with two learned pieces. Every token
gets a vector (its embedding, one row of a shared table C), and a small
neural network maps the concatenated vectors of the last n - 1 tokens to a
distribution over the next one:

    y = b + W x + U tanh(d + H x),      x = [C[w_{t-n+1}], ..., C[w_{t-1}]]

Tokens that occur in similar windows end up with similar rows of C, so a
window never seen in training still gets a sensible prediction from the
windows it resembles. The direct path W x is the paper's optional linear
shortcut from the embeddings to the output.

This is the first model of the course trained with the course's own autograd
(L0.1 to L0.4) and training loop (L0.5) on real text, and its checkpoint is
the NPLM row of the model zoo (L6.7).

Contract: contracts/py/tinyllm/lm/nplm.pyi.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd import functional as F
from tinyllm.autograd.losses import cross_entropy
from tinyllm.autograd.mode import no_grad
from tinyllm.autograd.tensor import Tensor
from tinyllm.info.ppl import NLLAccumulator
from tinyllm.io.safetensors import load_safetensors, save_safetensors
from tinyllm.nn.layers import Embedding, Linear
from tinyllm.nn.module import Module
from tinyllm.num.rng import PCG32
from tinyllm.optim.sgd import SGD
from tinyllm.train.loop import DataLoader, train_step

_TOKENIZERS = ("bytes", "file")


def windows(ids: ArrayLike, context: int) -> tuple[NDArray, NDArray]:
    a = np.asarray(ids)
    if a.ndim != 1 or (a.size and not np.issubdtype(a.dtype, np.integer)):
        raise ValueError(
            f"windows: ids must be a 1-D integer array, got {a.dtype} {a.shape}"
        )
    if context < 1:
        raise ValueError(f"windows: context must be >= 1, got {context}")
    if a.size <= context:
        raise ValueError(f"windows: {a.size} ids hold no full window of {context} + 1")
    a = a.astype(np.int64)
    n = a.size - context
    # Row i is ids[i : i + context]: a strided view, copied so callers own it.
    ctx = np.lib.stride_tricks.sliding_window_view(a, context)[:n].copy()
    return ctx, a[context:].copy()


class NPLM(Module):
    def __init__(
        self,
        vocab: int,
        context: int,
        d_emb: int,
        d_hidden: int,
        direct: bool = True,
        rng: Any = None,
    ) -> None:
        super().__init__()
        for name, v in (
            ("vocab", vocab),
            ("context", context),
            ("d_emb", d_emb),
            ("d_hidden", d_hidden),
        ):
            if int(v) < 1:
                raise ValueError(f"NPLM: {name} must be >= 1, got {v}")
        self.vocab, self.context = int(vocab), int(context)
        self.d_emb, self.d_hidden = int(d_emb), int(d_hidden)
        self.use_direct = bool(direct)
        # ONE generator for every layer, in registration order: passing None to
        # each layer would restart PCG32(0).substream("init") per layer and give
        # correlated weights.
        r = rng if rng is not None else PCG32(0).substream("init")
        width = self.context * self.d_emb
        self.emb = Embedding(self.vocab, self.d_emb, rng=r)
        self.hidden = Linear(width, self.d_hidden, rng=r)
        self.out = Linear(self.d_hidden, self.vocab, rng=r)
        self.direct = (
            Linear(width, self.vocab, bias=False, rng=r) if self.use_direct else None
        )

    def forward(self, ctx_ids: ArrayLike) -> Tensor:
        ids = np.asarray(ctx_ids)
        if ids.ndim != 2 or ids.shape[1] != self.context:
            raise ValueError(
                f"NPLM.forward: want windows [B, {self.context}], got shape {ids.shape}"
            )
        e = self.emb(ids)  # [B, context, d_emb]: the op, so C gets gradients
        # Row-major reshape concatenates the rows oldest first: x = [C[w1], C[w2], ...].
        x = F.reshape(e, (ids.shape[0], self.context * self.d_emb))
        y = self.out(F.tanh(self.hidden(x)))
        if self.direct is not None:
            y = y + self.direct(x)
        return y

    def nll(self, ids: ArrayLike, batch_size: int = 1024) -> NDArray:
        ctx, tgt = windows(ids, self.context)
        if batch_size < 1:
            raise ValueError(f"NPLM.nll: batch_size must be >= 1, got {batch_size}")
        out = np.empty(tgt.size, dtype=np.float64)
        with no_grad():
            for i in range(0, tgt.size, batch_size):
                z = self.forward(ctx[i : i + batch_size]).data.astype(np.float64)
                m = z.max(axis=1, keepdims=True)
                lse = m[:, 0] + np.log(np.exp(z - m).sum(axis=1))
                rows = np.arange(z.shape[0])
                out[i : i + batch_size] = lse - z[rows, tgt[i : i + batch_size]]
        return out

    def perplexity(self, ids: ArrayLike) -> float:
        acc = NLLAccumulator()
        acc.add(self.nll(ids))
        return float(acc.result()["ppl"])

    def generate(
        self, prefix: Sequence[int], n: int, temperature: float, seed: int
    ) -> list[int]:
        hist = [int(t) for t in prefix]
        if len(hist) < self.context:
            raise ValueError(
                f"NPLM.generate: the prefix needs at least {self.context} ids, got {len(hist)}"
            )
        if n < 0:
            raise ValueError(f"NPLM.generate: n must be >= 0, got {n}")
        if not temperature >= 0:
            raise ValueError(
                f"NPLM.generate: temperature must be >= 0, got {temperature}"
            )
        rng = PCG32(seed)  # one generator per call, stream 54 (L0.5's sampler)
        out: list[int] = []
        with no_grad():
            for _ in range(n):
                window = np.asarray([hist[-self.context :]], dtype=np.int64)
                row = self.forward(window).data[0].astype(np.float64)
                if temperature == 0:
                    nxt = int(np.argmax(row))  # first maximum: ties to the lowest id
                else:
                    nxt = _draw(row, float(temperature), rng.uniform())
                out.append(nxt)
                hist.append(nxt)
        return out


def _draw(row: NDArray, temperature: float, u: float) -> int:
    """weights exp(z - max z) of z = row / temperature in float64, summed in
    id order; the first id whose running sum exceeds u * total."""
    z = row / temperature
    w = np.exp(z - z.max()).tolist()
    total = 0.0
    for wi in w:  # in id order, the order the Rust engine sums in
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


def _window_loss(model: NPLM, batch: dict) -> Tensor:
    """Mean cross-entropy of a batch of windows ("x") against their next tokens ("y")."""
    return cross_entropy(model(batch["x"]), batch["y"])


# fmt: off
def train_nplm(
    model: NPLM, ids: ArrayLike, steps: int, batch_size: int, lr: float, rng: Any,
    momentum: float = 0.0, weight_decay: float = 0.0, clip: Optional[float] = None,
) -> list[float]:
    if steps < 0:
        raise ValueError(f"train_nplm: steps must be >= 0, got {steps}")
    ctx, tgt = windows(ids, model.context)
    if tgt.size < batch_size:
        raise ValueError(
            f"train_nplm: {tgt.size} windows, fewer than one batch of {batch_size}"
        )
    loader = DataLoader(
        {"x": ctx, "y": tgt}, batch_size, shuffle=True, rng=rng, drop_last=True
    )
    opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    losses: list[float] = []
    model.train()
    while len(losses) < steps:
        for batch in loader:  # each pass is a new permutation (an epoch)
            losses.append(
                train_step(model, batch, _window_loss, opt, clip=clip)["loss"]
            )
            if len(losses) == steps:
                break
    return losses
# fmt: on


def save_nplm(model: NPLM, dir: str, tokenizer: str = "bytes") -> None:
    if tokenizer not in _TOKENIZERS:
        raise ValueError(
            f"save_nplm: tokenizer must be one of {_TOKENIZERS}, got {tokenizer!r}"
        )
    d = Path(dir)
    d.mkdir(parents=True, exist_ok=True)
    config = {
        "tl_arch": "nplm",
        "tl_tokenizer": tokenizer,
        "vocab_size": model.vocab,
        "tl_format": 1,
        "tl_context": model.context,
        "tl_d_emb": model.d_emb,
        "hidden_size": model.d_hidden,
        "tl_direct": model.use_direct,
    }
    (d / "config.json").write_text(json.dumps(config) + "\n")
    tensors = {
        k: np.asarray(v, dtype=np.float32) for k, v in model.state_dict().items()
    }
    save_safetensors(
        str(d / "model.safetensors"), tensors, {"format": "tinyllm", "tl_arch": "nplm"}
    )


def load_nplm(dir: str) -> NPLM:
    d = Path(dir)
    cfg = json.loads((d / "config.json").read_text())
    if cfg.get("tl_arch") != "nplm":
        raise ValueError(
            f"{d}/config.json: tl_arch is {cfg.get('tl_arch')!r}, not 'nplm'"
        )
    try:
        model = NPLM(
            int(cfg["vocab_size"]),
            int(cfg["tl_context"]),
            int(cfg["tl_d_emb"]),
            int(cfg["hidden_size"]),
            direct=bool(cfg["tl_direct"]),
        )
    except KeyError as e:
        raise ValueError(f"{d}/config.json: missing key {e.args[0]!r}") from None
    tensors, _ = load_safetensors(str(d / "model.safetensors"))
    model.load_state_dict(tensors, strict=True)
    return model
