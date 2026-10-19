"""The encoder-decoder Transformer of Vaswani et al. (2017) (L5.5).

An encoder stack of self-attention and feed-forward sublayers reads the
source once; a decoder stack of masked self-attention, cross-attention over
the encoder's output, and feed-forward sublayers writes the target. Every
sublayer is wrapped in a residual connection and a LayerNorm, placed either
after the residual sum (post-LN, the 2017 paper) or before the sublayer
(pre-LN, `norm="pre"`, which trains without warmup). Training uses the
paper's recipe: Adam (0.9, 0.98, 1e-9), the Noam learning-rate schedule
(M10.4), and label smoothing (L0.3's cross_entropy).

Contract: contracts/py/tinyllm/xfmr/transformer.pyi.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, NamedTuple, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd import functional as F
from tinyllm.autograd.losses import cross_entropy
from tinyllm.autograd.mode import no_grad
from tinyllm.autograd.tensor import Tensor
from tinyllm.infer.beam import beam_search
from tinyllm.io.safetensors import load_safetensors, save_safetensors
from tinyllm.nn.init import normal_init
from tinyllm.nn.layers import Dropout, Embedding, LayerNorm, Linear, ModuleList
from tinyllm.nn.module import Module
from tinyllm.num.rng import PCG32
from tinyllm.optim.adamw import Adam
from tinyllm.optim.schedule import noam
from tinyllm.train.loop import DataLoader, train_step
from tinyllm.xfmr.masks import causal_mask, combine
from tinyllm.xfmr.mha import MultiHeadAttention
from tinyllm.xfmr.pos import SinusoidalPE


@dataclass
class TransformerConfig:
    src_vocab: int
    tgt_vocab: int
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_enc: int = 6
    n_dec: int = 6
    dropout: float = 0.1
    norm: Literal["post", "pre"] = "post"
    tie_embeddings: bool = True
    max_len: int = 256
    ln_eps: float = 1e-5


class DecodeState(NamedTuple):
    prefix: NDArray  # int64 [B, t]: the target tokens fed so far (bos first)
    src_mask: NDArray  # bool [B, S]


def _key_mask(pad: NDArray) -> NDArray:
    # [B, S] key padding -> [B, 1, 1, S]: every head, every query row.
    m = np.asarray(pad)
    if m.dtype != np.bool_ or m.ndim != 2:
        raise ValueError(
            f"padding masks are bool [B, T] (True = a real token), got {m.dtype} {m.shape}"
        )
    return m[:, None, None, :]


class EncoderLayer(Module):
    def __init__(self, cfg: TransformerConfig, rng: Any) -> None:
        super().__init__()
        d = cfg.d_model
        self.pre = cfg.norm == "pre"
        self.self_attn = MultiHeadAttention(d, cfg.n_heads, cfg.dropout, rng=rng)
        self.linear1 = Linear(d, cfg.d_ff, rng=rng)
        self.linear2 = Linear(cfg.d_ff, d, rng=rng)
        self.norm1 = LayerNorm(d, cfg.ln_eps)
        self.norm2 = LayerNorm(d, cfg.ln_eps)
        self.drop = Dropout(cfg.dropout)

    def ff(self, x: Tensor) -> Tensor:
        return self.linear2(self.drop(F.relu(self.linear1(x))))

    def forward(self, x: Tensor, mask: NDArray) -> Tensor:
        if self.pre:
            h = self.norm1(x)
            x = x + self.drop(self.self_attn(h, h, mask))
            return x + self.drop(self.ff(self.norm2(x)))
        x = self.norm1(x + self.drop(self.self_attn(x, x, mask)))
        return self.norm2(x + self.drop(self.ff(x)))


class DecoderLayer(Module):
    def __init__(self, cfg: TransformerConfig, rng: Any) -> None:
        super().__init__()
        d = cfg.d_model
        self.pre = cfg.norm == "pre"
        self.self_attn = MultiHeadAttention(d, cfg.n_heads, cfg.dropout, rng=rng)
        self.cross_attn = MultiHeadAttention(d, cfg.n_heads, cfg.dropout, rng=rng)
        self.linear1 = Linear(d, cfg.d_ff, rng=rng)
        self.linear2 = Linear(cfg.d_ff, d, rng=rng)
        self.norm1 = LayerNorm(d, cfg.ln_eps)
        self.norm2 = LayerNorm(d, cfg.ln_eps)
        self.norm3 = LayerNorm(d, cfg.ln_eps)
        self.drop = Dropout(cfg.dropout)

    def ff(self, x: Tensor) -> Tensor:
        return self.linear2(self.drop(F.relu(self.linear1(x))))

    def forward(
        self, y: Tensor, memory: Tensor, self_mask: NDArray, cross_mask: NDArray
    ) -> Tensor:
        if self.pre:
            h = self.norm1(y)
            y = y + self.drop(self.self_attn(h, h, self_mask))
            y = y + self.drop(self.cross_attn(self.norm2(y), memory, cross_mask))
            return y + self.drop(self.ff(self.norm3(y)))
        y = self.norm1(y + self.drop(self.self_attn(y, y, self_mask)))
        # Queries from the decoder, keys and values from the encoder.
        y = self.norm2(y + self.drop(self.cross_attn(y, memory, cross_mask)))
        return self.norm3(y + self.drop(self.ff(y)))


class Transformer(Module):
    def __init__(self, cfg: TransformerConfig, rng: Any = None) -> None:
        super().__init__()
        if cfg.norm not in ("post", "pre"):
            raise ValueError(f"norm must be 'post' or 'pre', got {cfg.norm!r}")
        if (
            min(
                cfg.src_vocab,
                cfg.tgt_vocab,
                cfg.n_enc,
                cfg.n_dec,
                cfg.d_ff,
                cfg.max_len,
            )
            < 1
        ):
            raise ValueError(f"sizes must be positive: {cfg}")
        r = rng if rng is not None else PCG32(0).substream("init")
        self.cfg = cfg
        d = cfg.d_model
        # Embeddings start at std d^-1/2 so that emb * sqrt(d) has unit scale.
        self.src_emb = Embedding(cfg.src_vocab, d, rng=r)
        self.src_emb.weight.data = normal_init((cfg.src_vocab, d), d**-0.5, r)
        if cfg.tie_embeddings and cfg.src_vocab == cfg.tgt_vocab:
            self.tgt_emb = Embedding(cfg.tgt_vocab, d, rng=r)
            self.tgt_emb.weight = self.src_emb.weight  # one table, listed once
        else:
            self.tgt_emb = Embedding(cfg.tgt_vocab, d, rng=r)
            self.tgt_emb.weight.data = normal_init((cfg.tgt_vocab, d), d**-0.5, r)
        self.pos = SinusoidalPE(cfg.max_len, d)
        self.encoder = ModuleList([EncoderLayer(cfg, r) for _ in range(cfg.n_enc)])
        self.decoder = ModuleList([DecoderLayer(cfg, r) for _ in range(cfg.n_dec)])
        if cfg.norm == "pre":
            # Pre-LN leaves the residual stream unnormalized: one last LN per stack.
            self.enc_norm = LayerNorm(d, cfg.ln_eps)
            self.dec_norm = LayerNorm(d, cfg.ln_eps)
        self.out = Linear(d, cfg.tgt_vocab, bias=False, rng=r)
        if cfg.tie_embeddings:
            self.out.weight = self.tgt_emb.weight
        self.drop = Dropout(cfg.dropout)

    def embed(self, ids: ArrayLike, table: Embedding, offset: int = 0) -> Tensor:
        x = table(ids) * math.sqrt(self.cfg.d_model)
        return self.drop(self.pos(x, offset))

    def encode(self, src: ArrayLike, src_mask: ArrayLike) -> Tensor:
        src = np.asarray(src)
        if src.ndim != 2 or src.dtype.kind not in "iu":
            raise ValueError(f"src must be int [B, S], got {src.dtype} {src.shape}")
        m = _key_mask(src_mask)
        if m.shape[0] != src.shape[0] or m.shape[3] != src.shape[1]:
            raise ValueError(
                f"src_mask must be [B, S] = {src.shape}, got {np.shape(src_mask)}"
            )
        x = self.embed(src, self.src_emb)
        for layer in self.encoder:
            x = layer(x, m)
        return self.enc_norm(x) if self.cfg.norm == "pre" else x

    def decode(
        self,
        tgt_in: ArrayLike,
        memory: Tensor,
        src_mask: ArrayLike,
        tgt_mask: ArrayLike,
    ) -> Tensor:
        tgt_in = np.asarray(tgt_in)
        if tgt_in.ndim != 2 or tgt_in.dtype.kind not in "iu":
            raise ValueError(
                f"tgt_in must be int [B, T], got {tgt_in.dtype} {tgt_in.shape}"
            )
        T = tgt_in.shape[1]
        tm = _key_mask(tgt_mask)
        if tm.shape[0] != tgt_in.shape[0] or tm.shape[3] != T:
            raise ValueError(
                f"tgt_mask must be [B, T] = {tgt_in.shape}, got {np.shape(tgt_mask)}"
            )
        # Position t may read targets 0..t only: causal AND not padding.
        self_mask = combine(causal_mask(T), tm)
        cross_mask = _key_mask(src_mask)
        y = self.embed(tgt_in, self.tgt_emb)
        for layer in self.decoder:
            y = layer(y, memory, self_mask, cross_mask)
        if self.cfg.norm == "pre":
            y = self.dec_norm(y)
        return self.out(y)

    def forward(
        self,
        src: ArrayLike,
        tgt_in: ArrayLike,
        src_mask: ArrayLike,
        tgt_mask: ArrayLike,
    ) -> Tensor:
        return self.decode(tgt_in, self.encode(src, src_mask), src_mask, tgt_mask)

    def init_state(self, src_mask: ArrayLike) -> DecodeState:
        m = np.asarray(src_mask)
        return DecodeState(np.zeros((m.shape[0], 0), dtype=np.int64), m)

    def decode_step(
        self, y_prev: ArrayLike, memory: Tensor, state: DecodeState
    ) -> tuple[Tensor, DecodeState]:
        y = np.asarray(y_prev, dtype=np.int64).reshape(-1, 1)
        prefix = np.concatenate([state.prefix, y], axis=1)
        if prefix.shape[0] != memory.shape[0]:
            raise ValueError(
                f"{prefix.shape[0]} tokens for a memory of batch {memory.shape[0]}"
            )
        # No cache yet (L8.2 adds one): recompute the whole prefix, keep the last row.
        logits = self.decode(
            prefix, memory, state.src_mask, np.ones(prefix.shape, dtype=bool)
        )
        return logits[:, -1, :], DecodeState(prefix, state.src_mask)


def label_smoothed_loss(
    logits: Tensor, targets: ArrayLike, eps: float, pad_id: int
) -> Tensor:
    t = np.asarray(targets).astype(np.int64)
    # Padding positions are not predictions: no loss, no gradient, not counted.
    t = np.where(t == pad_id, -100, t)
    return cross_entropy(logits, t, ignore_index=-100, label_smoothing=eps)


def noam_rate(step: int, d_model: int, warmup: int, factor: float = 1.0) -> float:
    # Update number `step` (0 first) is the paper's step_num = step + 1.
    return factor * noam(step + 1, d_model, warmup)


def make_optimizer(model: Module) -> Adam:
    return Adam(list(model.parameters()), lr=0.0, betas=(0.9, 0.98), eps=1e-9)


# fmt: off
def fit(model: Transformer, src: ArrayLike, tgt: ArrayLike, steps: int, batch: int,
        pad_id: int, warmup: int = 4000, factor: float = 1.0, smoothing: float = 0.1,
        lr: Optional[float] = None, clip: Optional[float] = None, rng: Any = None,
        on_step: Optional[Callable[[int, dict], None]] = None) -> list[float]:
    src, tgt = np.asarray(src, dtype=np.int64), np.asarray(tgt, dtype=np.int64)
    if (
        src.ndim != 2
        or tgt.ndim != 2
        or src.shape[0] != tgt.shape[0]
        or tgt.shape[1] < 2
    ):
        raise ValueError(
            f"src [N, S] and tgt [N, T+1] with T >= 1, got {src.shape} and {tgt.shape}"
        )
    r = rng if rng is not None else PCG32(0).substream("shuffle")
    loader = DataLoader({"src": src, "tgt": tgt}, batch, shuffle=True, rng=r)
    opt = make_optimizer(model)
    model.train()

    def loss_fn(m: Transformer, b: dict) -> Tensor:
        s, t = b["src"], b["tgt"]
        # Teacher forcing: read bos y_1 .. y_{T-1}, predict y_1 .. y_T.
        logits = m(s, t[:, :-1], s != pad_id, t[:, :-1] != pad_id)
        return label_smoothed_loss(logits, t[:, 1:], smoothing, pad_id)

    losses: list[float] = []
    step = 0
    while step < steps:
        for b in loader:
            if step >= steps:
                break
            opt.lr = (
                lr
                if lr is not None
                else noam_rate(step, model.cfg.d_model, warmup, factor)
            )
            stats = train_step(model, b, loss_fn, opt, clip)
            losses.append(stats["loss"])
            if on_step is not None:
                on_step(step, stats)
            step += 1
    return losses


# fmt: on


def translate(
    model: Transformer,
    src: ArrayLike,
    bos: int,
    eos: int,
    pad_id: int,
    max_len: int,
    beam_size: int = 1,
    length_penalty: float = 1.0,
) -> list[list[int]]:
    src = np.asarray(src, dtype=np.int64)
    if src.ndim != 2:
        raise ValueError(f"src must be int [B, S], got {src.shape}")
    was = model.training
    model.eval()
    out: list[list[int]] = []
    try:
        with no_grad():
            for row in src:
                s = row[None, :]
                mask = s != pad_id
                memory = model.encode(s, mask)

                def step(st: tuple, y: NDArray) -> tuple:
                    mem, ds = st
                    logits, ds2 = model.decode_step(y, mem, ds)
                    return logits.data, (mem, ds2)

                hyps = beam_search(
                    step,
                    (memory, model.init_state(mask)),
                    bos,
                    eos,
                    beam_size,
                    max_len,
                    length_penalty,
                )
                toks = hyps[0].tokens
                out.append(toks[:-1] if hyps[0].finished else toks)
    finally:
        model.train(was)
    return out


def save_transformer(model: Transformer, dir: str) -> None:
    d = Path(dir)
    d.mkdir(parents=True, exist_ok=True)
    c = model.cfg
    cfg = {
        "tl_arch": "transformer",
        "tl_tokenizer": "file",
        "vocab_size": c.tgt_vocab,
        "tl_src_vocab": c.src_vocab,
        "tl_format": 1,
        "d_model": c.d_model,
        "n_heads": c.n_heads,
        "d_ff": c.d_ff,
        "n_enc": c.n_enc,
        "n_dec": c.n_dec,
        "tl_norm": c.norm,
        "tie_word_embeddings": c.tie_embeddings,
        "max_position_embeddings": c.max_len,
        "layer_norm_eps": c.ln_eps,
    }
    (d / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    tensors = {
        k: np.ascontiguousarray(v, dtype=np.float32)
        for k, v in model.state_dict().items()
    }
    save_safetensors(
        str(d / "model.safetensors"),
        tensors,
        {"format": "tinyllm", "tl_arch": "transformer"},
    )


def load_transformer(dir: str) -> Transformer:
    d = Path(dir)
    c = json.loads((d / "config.json").read_text())
    if c.get("tl_arch") != "transformer":
        raise ValueError(
            f"{d}/config.json: tl_arch is {c.get('tl_arch')!r}, not 'transformer'"
        )
    cfg = TransformerConfig(
        src_vocab=c["tl_src_vocab"],
        tgt_vocab=c["vocab_size"],
        d_model=c["d_model"],
        n_heads=c["n_heads"],
        d_ff=c["d_ff"],
        n_enc=c["n_enc"],
        n_dec=c["n_dec"],
        dropout=0.0,
        norm=c["tl_norm"],
        tie_embeddings=c["tie_word_embeddings"],
        max_len=c["max_position_embeddings"],
        ln_eps=c["layer_norm_eps"],
    )
    model = Transformer(cfg)
    tensors, _ = load_safetensors(str(d / "model.safetensors"))
    model.load_state_dict(tensors)
    model.eval()
    return model
