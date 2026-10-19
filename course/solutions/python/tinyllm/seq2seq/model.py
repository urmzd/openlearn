"""An encoder-decoder with teacher forcing (L4.1).

The encoder is a bidirectional GRU over the source (L3.3, L3.4); a bridge
turns its two final states into the decoder's first state. The decoder is a
GRU or LSTM cell that writes one target token per step. During training it
is fed the true previous token (teacher forcing), so every step's loss is
computed in one pass. Attention (L4.2 Bahdanau, L4.3 Luong) plugs in as a
module that reads the encoder outputs at every step.

Contract: contracts/py/tinyllm/seq2seq/model.pyi.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, NamedTuple, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd import functional as F
from tinyllm.autograd.mode import no_grad
from tinyllm.autograd.tensor import Tensor
from tinyllm.io.safetensors import load_safetensors, save_safetensors
from tinyllm.nn.layers import Embedding, Linear
from tinyllm.nn.module import Module
from tinyllm.num.rng import PCG32
from tinyllm.rnn.bi import bidirectional
from tinyllm.rnn.gru import GRU, GRUCell
from tinyllm.rnn.lstm import LSTMCell
from tinyllm.seq2seq.additive import AdditiveAttention, length_mask
from tinyllm.seq2seq.luong import LuongAttention


class EncoderState(NamedTuple):
    keys: Tensor
    mask: NDArray
    proj: Optional[Tensor]
    init: Tensor


class DecoderState(NamedTuple):
    h: Tensor
    c: Optional[Tensor]
    feed: Optional[Tensor]
    keys: Tensor
    mask: NDArray
    proj: Optional[Tensor]


def _ints(a: ArrayLike, name: str, ndim: int) -> NDArray:
    x = np.asarray(a)
    if x.ndim != ndim or (x.size and x.dtype.kind not in "iu"):
        raise ValueError(
            f"{name} must be a {ndim}-D integer array, got {x.dtype} {x.shape}"
        )
    return x.astype(np.int64)


class Seq2Seq(Module):
    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_emb: int,
        d_h: int,
        cell: Literal["gru", "lstm"] = "gru",
        attention: Optional[Module] = None,
        rng: Any = None,
    ) -> None:
        super().__init__()
        if min(src_vocab, tgt_vocab, d_emb, d_h) < 1:
            raise ValueError(
                f"sizes must be positive, got {(src_vocab, tgt_vocab, d_emb, d_h)}"
            )
        if d_h % 2:
            raise ValueError(f"d_h must be even (two directions of d_h / 2), got {d_h}")
        if cell not in ("gru", "lstm"):
            raise ValueError(f"cell must be 'gru' or 'lstm', got {cell!r}")
        when = getattr(attention, "query_from", None) if attention is not None else None
        if attention is not None and when not in ("previous", "current"):
            raise ValueError(
                f"attention.query_from must be 'previous' or 'current', got {when!r}"
            )
        r = rng if rng is not None else PCG32(0).substream("init")
        self.src_vocab, self.tgt_vocab, self.d_emb, self.d_h = (
            src_vocab,
            tgt_vocab,
            d_emb,
            d_h,
        )
        self.cell_type = cell
        self.src_emb = Embedding(src_vocab, d_emb, rng=r)
        self.enc_fwd = GRU(d_emb, d_h // 2, rng=r)
        self.enc_bwd = GRU(d_emb, d_h // 2, rng=r)
        self.bridge = Linear(d_h, d_h, rng=r)
        self.tgt_emb = Embedding(tgt_vocab, d_emb, rng=r)
        d_in = d_emb + (d_h if attention is not None else 0)
        self.cell = (
            GRUCell(d_in, d_h, rng=r) if cell == "gru" else LSTMCell(d_in, d_h, rng=r)
        )
        self.attention = attention  # None registers nothing
        self.out = Linear(2 * d_h if when == "previous" else d_h, tgt_vocab, rng=r)

    def encode(self, src: ArrayLike, src_lens: ArrayLike) -> EncoderState:
        s = _ints(src, "src", 2)
        n = _ints(src_lens, "src_lens", 1)
        B, S = s.shape
        if n.shape != (B,) or S < 1 or n.min() < 1 or n.max() > S:
            raise ValueError(
                f"src_lens must be {B} lengths in 1..{S}, got {n.tolist()}"
            )
        mask = length_mask(n, S)
        # Padding may hold any id; read it as 0 so the embedding lookup is valid.
        s = np.where(mask, s, 0)
        if s.min() < 0 or s.max() >= self.src_vocab:
            raise ValueError(f"src ids must lie in [0, {self.src_vocab})")
        out = bidirectional(
            self.enc_fwd, self.enc_bwd, self.src_emb(s.T), n
        )  # [S, B, d_h]
        H2 = self.d_h // 2
        rows = np.arange(B)
        # Forward state after each sequence's own last token; backward state
        # after reading it back to the first token (position 0).
        last_f = out[n - 1, rows][:, :H2]
        first_b = out[0][:, H2:]
        init = F.tanh(self.bridge(F.concat([last_f, first_b], axis=-1)))
        keys = F.transpose(out, 0, 1)
        proj = self.attention.project_keys(keys) if self.attention is not None else None
        return EncoderState(keys, mask, proj, init)

    def init_state(self, enc: EncoderState) -> DecoderState:
        B = enc.init.shape[0]
        zeros = np.zeros((B, self.d_h), dtype=np.float32)
        c = Tensor(zeros) if self.cell_type == "lstm" else None
        luong = self.attention is not None and self.attention.query_from == "current"
        feed = Tensor(zeros) if luong else None
        return DecoderState(enc.init, c, feed, enc.keys, enc.mask, enc.proj)

    def _cell(self, x: Tensor, st: DecoderState) -> tuple[Tensor, Optional[Tensor]]:
        if self.cell_type == "lstm":
            return self.cell(x, (st.h, st.c))
        return self.cell(x, st.h), None

    def decode_step(
        self, y_prev: ArrayLike, state: DecoderState
    ) -> tuple[Tensor, DecoderState, Optional[Tensor]]:
        y = _ints(y_prev, "y_prev", 1)
        if y.size and (y.min() < 0 or y.max() >= self.tgt_vocab):
            raise ValueError(f"y_prev ids must lie in [0, {self.tgt_vocab})")
        e = self.tgt_emb(y)
        att = self.attention
        if att is None:
            h, c = self._cell(e, state)
            return self.out(h), state._replace(h=h, c=c), None
        if att.query_from == "previous":
            # Bahdanau: read the source with the state from BEFORE this step.
            ctx, a = att(state.h, state.keys, state.mask, state.proj)
            h, c = self._cell(F.concat([e, ctx], axis=-1), state)
            logits = self.out(F.concat([h, ctx], axis=-1))
            return logits, state._replace(h=h, c=c), a
        # Luong: step first (input feeding), then read with the NEW state.
        h, c = self._cell(F.concat([e, state.feed], axis=-1), state)
        ctx, a = att(h, state.keys, state.mask, state.proj)
        feed = att.attentional(h, ctx)
        return self.out(feed), state._replace(h=h, c=c, feed=feed), a

    def forward(
        self,
        src: ArrayLike,
        src_lens: ArrayLike,
        tgt_in: ArrayLike,
        teacher_forcing: float = 1.0,
        rng: Any = None,
    ) -> Tensor:
        tgt = _ints(tgt_in, "tgt_in", 2)
        src_a = np.asarray(src)
        if tgt.shape[1] < 1 or (src_a.ndim >= 1 and tgt.shape[0] != src_a.shape[0]):
            raise ValueError(
                f"tgt_in must be [B, T >= 1] with the batch of src, got {tgt.shape}"
            )
        if tgt.min() < 0 or tgt.max() >= self.tgt_vocab:
            raise ValueError(f"tgt_in ids must lie in [0, {self.tgt_vocab})")
        if not 0.0 <= teacher_forcing <= 1.0:
            raise ValueError(
                f"teacher_forcing must lie in [0, 1], got {teacher_forcing}"
            )
        if teacher_forcing < 1.0 and rng is None:
            raise ValueError("teacher_forcing < 1 draws from an rng; pass one")
        state = self.init_state(self.encode(src, src_lens))
        steps = []
        y = tgt[:, 0]
        for t in range(tgt.shape[1]):
            if t > 0:
                if teacher_forcing >= 1.0 or rng.uniform() < teacher_forcing:
                    y = tgt[:, t]
                else:
                    # The model's own guess: a choice, so no gradient through it.
                    y = np.argmax(steps[-1].data, axis=-1)
            logits, state, _ = self.decode_step(y, state)
            steps.append(logits)
        return F.stack(steps, axis=1)

    def greedy(
        self, src: ArrayLike, src_lens: ArrayLike, bos: int, eos: int, max_len: int
    ) -> list[list[int]]:
        if max_len < 1:
            raise ValueError(f"max_len must be >= 1, got {max_len}")
        with no_grad():
            state = self.init_state(self.encode(src, src_lens))
            B = state.h.shape[0]
            y = np.full(B, bos, dtype=np.int64)
            outs: list[list[int]] = [[] for _ in range(B)]
            done = np.zeros(B, dtype=bool)
            for _ in range(max_len):
                logits, state, _ = self.decode_step(y, state)
                y = np.argmax(logits.data, axis=-1)
                for b in range(B):
                    if not done[b]:
                        outs[b].append(int(y[b]))
                        done[b] = y[b] == eos
                if done.all():
                    break
        return outs


def _attention_name(model: Seq2Seq) -> str:
    att = model.attention
    if att is None:
        return "none"
    if att.query_from == "previous":
        return "bahdanau"
    return f"luong-{att.score}"


def save_seq2seq(model: Seq2Seq, dir: str) -> None:
    d = Path(dir)
    d.mkdir(parents=True, exist_ok=True)
    config = {
        "tl_arch": "seq2seq",
        "tl_tokenizer": "file",
        "vocab_size": model.tgt_vocab,
        "tl_src_vocab": model.src_vocab,
        "tl_format": 1,
        "tl_cell": model.cell_type,
        "tl_d_emb": model.d_emb,
        "hidden_size": model.d_h,
        "tl_seq2seq_attention": _attention_name(model),
    }
    if config["tl_seq2seq_attention"] == "bahdanau":
        config["tl_d_attn"] = model.attention.d_attn
    (d / "config.json").write_text(json.dumps(config) + "\n")
    tensors = {
        k: np.asarray(v, dtype=np.float32) for k, v in model.state_dict().items()
    }
    save_safetensors(
        str(d / "model.safetensors"),
        tensors,
        {"format": "tinyllm", "tl_arch": "seq2seq"},
    )


def load_seq2seq(dir: str) -> Seq2Seq:
    d = Path(dir)
    cfg = json.loads((d / "config.json").read_text())
    if cfg.get("tl_arch") != "seq2seq":
        raise ValueError(
            f"{d}/config.json: tl_arch is {cfg.get('tl_arch')!r}, not 'seq2seq'"
        )
    try:
        d_h = int(cfg["hidden_size"])
        kind = cfg["tl_seq2seq_attention"]
        if kind == "none":
            att = None
        elif kind == "bahdanau":
            att = AdditiveAttention(d_h, d_h, int(cfg["tl_d_attn"]))
        elif kind.startswith("luong-"):
            att = LuongAttention(d_h, kind[len("luong-") :])
        else:
            raise ValueError(f"{d}/config.json: unknown tl_seq2seq_attention {kind!r}")
        model = Seq2Seq(
            int(cfg["tl_src_vocab"]),
            int(cfg["vocab_size"]),
            int(cfg["tl_d_emb"]),
            d_h,
            cfg["tl_cell"],
            att,
        )
    except KeyError as e:
        raise ValueError(f"{d}/config.json: missing key {e.args[0]!r}") from None
    tensors, _ = load_safetensors(str(d / "model.safetensors"))
    model.load_state_dict(tensors, strict=True)
    return model
