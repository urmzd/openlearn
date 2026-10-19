"""BERT: a bidirectional encoder and its masked-LM objective (L6.2).

Keep only L5.5's encoder, read the whole sentence in both directions, and
train it to fill in blanks: pick 15% of the positions, show [MASK] at 80% of
them, a random token at 10%, the token itself at 10%, and predict the
original token at exactly the picked positions.

Contract: contracts/py/tinyllm/obj/bert.pyi.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd import functional as F
from tinyllm.autograd.losses import cross_entropy
from tinyllm.autograd.tensor import Tensor
from tinyllm.io.safetensors import load_safetensors, save_safetensors
from tinyllm.nn.init import normal_init
from tinyllm.nn.layers import Dropout, Embedding, LayerNorm, Linear, ModuleList
from tinyllm.nn.module import Module
from tinyllm.num.rng import PCG32
from tinyllm.prob.sampling import sample_categorical
from tinyllm.tok.wordpiece import WordPieceTokenizer
from tinyllm.xfmr.mha import MultiHeadAttention
from tinyllm.xfmr.pos import LearnedPE

INIT_STD = 0.02  # BERT's initializer_range
MASK, RANDOM, KEEP = 0, 1, 2
SPLIT = np.array(
    [0.8, 0.1, 0.1]
)  # what a picked position shows: [MASK], random, itself
SPECIALS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")


@dataclass
class BertConfig:
    vocab: int
    max_len: int = 512
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    type_vocab: int = 2
    dropout: float = 0.1
    ln_eps: float = 1e-12


def mlm_mask(
    ids: ArrayLike,
    special_mask: ArrayLike,
    mask_id: int,
    vocab: int,
    p: float,
    rng: Any,
) -> tuple[NDArray, NDArray]:
    x = np.asarray(ids)
    sp = np.asarray(special_mask, dtype=bool)
    if x.dtype.kind not in "iu" or sp.shape != x.shape:
        raise ValueError(
            f"ids int and special_mask bool of one shape, got {x.dtype} {x.shape}, {sp.shape}"
        )
    if not 0.0 <= p <= 1.0 or not 0 <= mask_id < vocab:
        raise ValueError(
            f"need 0 <= p <= 1 and 0 <= mask_id < vocab, got p={p}, mask_id={mask_id}"
        )
    inputs = x.astype(np.int64).copy()
    labels = np.full(x.shape, -100, dtype=np.int64)
    flat_in, flat_lab, flat_sp = inputs.reshape(-1), labels.reshape(-1), sp.reshape(-1)
    for i in range(flat_in.size):  # C order; a special position draws nothing
        if flat_sp[i]:
            continue
        if rng.uniform() >= p:
            continue
        flat_lab[i] = flat_in[i]  # the loss asks for the ORIGINAL token here
        action = sample_categorical(SPLIT, rng.uniform())  # M07.1, inverse CDF
        if action == MASK:
            flat_in[i] = mask_id
        elif action == RANDOM:
            flat_in[i] = rng.below(vocab)
        # KEEP: the input shows the token itself, but it is still predicted.
    return inputs, labels


def special_ids(tok: WordPieceTokenizer) -> list[int]:
    out = [tok.token_to_id(s) for s in SPECIALS]
    return sorted(i for i in out if i is not None)


def mlm_batch(
    tok: WordPieceTokenizer, texts: Sequence[str], max_len: int, p: float, rng: Any
) -> dict[str, NDArray]:
    pad, mask_id = tok.token_to_id("[PAD]"), tok.token_to_id("[MASK]")
    if pad is None or mask_id is None:
        raise ValueError("the tokenizer needs [PAD] and [MASK] tokens")
    if max_len < 2:
        raise ValueError(f"max_len must leave room for [CLS] and [SEP], got {max_len}")
    rows = []
    for t in texts:
        ids = tok.encode(t, add_special=True)
        if len(ids) > max_len:
            ids = ids[: max_len - 1] + ids[-1:]  # keep [SEP] last
        rows.append(ids)
    T = max(len(r) for r in rows)
    ids = np.full((len(rows), T), pad, dtype=np.int64)
    attn = np.zeros((len(rows), T), dtype=bool)
    for b, r in enumerate(rows):
        ids[b, : len(r)] = r
        attn[b, : len(r)] = True
    special = np.isin(ids, special_ids(tok)) | ~attn
    vocab = max(tok.vocab.values()) + 1
    inputs, labels = mlm_mask(ids, special, mask_id, vocab, p, rng)
    return {
        "input_ids": inputs,
        "token_type_ids": np.zeros_like(inputs),
        "attention_mask": attn,
        "labels": labels,
    }


class BertLayer(Module):
    def __init__(self, cfg: BertConfig, rng: Any) -> None:
        super().__init__()
        d = cfg.d_model
        self.attn = MultiHeadAttention(d, cfg.n_heads, cfg.dropout, rng=rng)
        self.attn_norm = LayerNorm(d, cfg.ln_eps)
        self.ff1 = Linear(d, cfg.d_ff, rng=rng)
        self.ff2 = Linear(cfg.d_ff, d, rng=rng)
        self.ff_norm = LayerNorm(d, cfg.ln_eps)
        self.drop = Dropout(cfg.dropout)
        for lin in (
            self.attn.q_proj,
            self.attn.k_proj,
            self.attn.v_proj,
            self.attn.out_proj,
            self.ff1,
            self.ff2,
        ):
            lin.weight.data = normal_init(lin.weight.shape, INIT_STD, rng)

    def forward(self, x: Tensor, mask: NDArray) -> Tensor:
        # Post-LN, as the 2017 transformer (L5.5's norm="post"), exact GELU.
        x = self.attn_norm(x + self.drop(self.attn(x, x, mask)))
        return self.ff_norm(x + self.drop(self.ff2(F.gelu(self.ff1(x)))))


class BertEncoder(Module):
    def __init__(self, cfg: BertConfig, rng: Any = None) -> None:
        super().__init__()
        if (
            min(
                cfg.vocab,
                cfg.max_len,
                cfg.d_model,
                cfg.n_layers,
                cfg.d_ff,
                cfg.type_vocab,
            )
            < 1
        ):
            raise ValueError(f"sizes must be positive: {cfg}")
        r = rng if rng is not None else PCG32(0).substream("init")
        self.cfg = cfg
        self.word_emb = Embedding(cfg.vocab, cfg.d_model, rng=r)
        self.word_emb.weight.data = normal_init((cfg.vocab, cfg.d_model), INIT_STD, r)
        self.pos_emb = LearnedPE(cfg.max_len, cfg.d_model, std=INIT_STD, rng=r)
        self.type_emb = Embedding(cfg.type_vocab, cfg.d_model, rng=r)
        self.type_emb.weight.data = normal_init(
            (cfg.type_vocab, cfg.d_model), INIT_STD, r
        )
        self.emb_norm = LayerNorm(cfg.d_model, cfg.ln_eps)
        self.drop = Dropout(cfg.dropout)
        self.layers = ModuleList([BertLayer(cfg, r) for _ in range(cfg.n_layers)])

    def forward(
        self,
        ids: ArrayLike,
        token_type_ids: Optional[ArrayLike] = None,
        attn_mask: Optional[ArrayLike] = None,
    ) -> Tensor:
        ids = np.asarray(ids)
        if ids.ndim != 2 or ids.dtype.kind not in "iu":
            raise ValueError(f"ids must be int [B, T], got {ids.dtype} {ids.shape}")
        tt = (
            np.zeros_like(ids) if token_type_ids is None else np.asarray(token_type_ids)
        )
        am = (
            np.ones(ids.shape, dtype=bool)
            if attn_mask is None
            else np.asarray(attn_mask).astype(bool)
        )
        if tt.shape != ids.shape or am.shape != ids.shape:
            raise ValueError(
                f"token_type_ids and attn_mask must be [B, T] = {ids.shape}"
            )
        x = self.pos_emb(self.word_emb(ids)) + self.type_emb(tt)
        x = self.drop(self.emb_norm(x))
        # Every query may read every real key, left and right: no causal mask.
        mask = am[:, None, None, :]
        for layer in self.layers:
            x = layer(x, mask)
        return x


class BertForMLM(Module):
    def __init__(self, cfg: BertConfig, rng: Any = None) -> None:
        super().__init__()
        r = rng if rng is not None else PCG32(0).substream("init")
        self.bert = BertEncoder(cfg, r)
        self.transform = Linear(cfg.d_model, cfg.d_model, rng=r)
        self.transform.weight.data = normal_init(
            self.transform.weight.shape, INIT_STD, r
        )
        self.transform_norm = LayerNorm(cfg.d_model, cfg.ln_eps)
        # The decoder's weight IS word_emb.weight (tied); only its bias is new.
        self.decoder_bias = Tensor(
            np.zeros(cfg.vocab, dtype=np.float32), requires_grad=True
        )

    def forward(
        self,
        ids: ArrayLike,
        token_type_ids: Optional[ArrayLike] = None,
        attn_mask: Optional[ArrayLike] = None,
        labels: Optional[ArrayLike] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        h = self.bert(ids, token_type_ids, attn_mask)
        h = self.transform_norm(F.gelu(self.transform(h)))
        logits = (
            F.matmul(h, F.transpose(self.bert.word_emb.weight, 0, 1))
            + self.decoder_bias
        )
        if labels is None:
            return logits, None
        # Only the picked positions carry a label; the rest are -100.
        return logits, cross_entropy(
            logits, np.asarray(labels).astype(np.int64), ignore_index=-100
        )


def _hf(sd: Mapping[str, Any], name: str) -> NDArray:
    for k in ("bert." + name, name):
        if k in sd:
            return np.asarray(sd[k], dtype=np.float32)
    raise KeyError(f"BERT state dict has no {name!r} (with or without 'bert.')")


def hf_bert_encoder_sd(sd: Mapping[str, Any], n_layers: int) -> dict[str, NDArray]:
    e = "embeddings."
    out = {
        "word_emb.weight": _hf(sd, e + "word_embeddings.weight"),
        "pos_emb.weight": _hf(sd, e + "position_embeddings.weight"),
        "type_emb.weight": _hf(sd, e + "token_type_embeddings.weight"),
        "emb_norm.weight": _hf(sd, e + "LayerNorm.weight"),
        "emb_norm.bias": _hf(sd, e + "LayerNorm.bias"),
    }
    for i in range(n_layers):
        h, m = f"encoder.layer.{i}.", f"layers.{i}."
        # HF's BERT uses nn.Linear ([out, in]): no transpose, unlike GPT-2's Conv1D.
        for ours, theirs in (
            ("attn.q_proj", "attention.self.query"),
            ("attn.k_proj", "attention.self.key"),
            ("attn.v_proj", "attention.self.value"),
            ("attn.out_proj", "attention.output.dense"),
            ("attn_norm", "attention.output.LayerNorm"),
            ("ff1", "intermediate.dense"),
            ("ff2", "output.dense"),
            ("ff_norm", "output.LayerNorm"),
        ):
            out[f"{m}{ours}.weight"] = _hf(sd, f"{h}{theirs}.weight")
            out[f"{m}{ours}.bias"] = _hf(sd, f"{h}{theirs}.bias")
    return out


def load_hf_bert(model: Module, sd: Mapping[str, Any]) -> None:
    if isinstance(model, BertForMLM):
        n = model.bert.cfg.n_layers
        out = {"bert." + k: v for k, v in hf_bert_encoder_sd(sd, n).items()}
        p = "cls.predictions."
        out["transform.weight"] = np.asarray(
            sd[p + "transform.dense.weight"], dtype=np.float32
        )
        out["transform.bias"] = np.asarray(
            sd[p + "transform.dense.bias"], dtype=np.float32
        )
        out["transform_norm.weight"] = np.asarray(
            sd[p + "transform.LayerNorm.weight"], dtype=np.float32
        )
        out["transform_norm.bias"] = np.asarray(
            sd[p + "transform.LayerNorm.bias"], dtype=np.float32
        )
        out["decoder_bias"] = np.asarray(sd[p + "bias"], dtype=np.float32)
        model.load_state_dict(out)
    elif isinstance(model, BertEncoder):
        model.load_state_dict(hf_bert_encoder_sd(sd, model.cfg.n_layers))
    else:
        raise TypeError(
            f"load_hf_bert loads a BertEncoder or a BertForMLM, got {type(model).__name__}"
        )


def save_bert(model: BertForMLM, dir: str, tokenizer: str = "file") -> None:
    d = Path(dir)
    d.mkdir(parents=True, exist_ok=True)
    c = model.bert.cfg
    cfg = {
        "tl_arch": "bert",
        "tl_tokenizer": tokenizer,
        "tl_format": 1,
        "vocab_size": c.vocab,
        "max_position_embeddings": c.max_len,
        "hidden_size": c.d_model,
        "num_attention_heads": c.n_heads,
        "num_hidden_layers": c.n_layers,
        "intermediate_size": c.d_ff,
        "type_vocab_size": c.type_vocab,
        "layer_norm_eps": c.ln_eps,
    }
    (d / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    tensors = {
        k: np.ascontiguousarray(v, dtype=np.float32)
        for k, v in model.state_dict().items()
    }
    save_safetensors(
        str(d / "model.safetensors"), tensors, {"format": "tinyllm", "tl_arch": "bert"}
    )


def load_bert(dir: str) -> BertForMLM:
    d = Path(dir)
    c = json.loads((d / "config.json").read_text())
    if c.get("tl_arch") != "bert":
        raise ValueError(
            f"{d}/config.json: tl_arch is {c.get('tl_arch')!r}, not 'bert'"
        )
    cfg = BertConfig(
        vocab=c["vocab_size"],
        max_len=c["max_position_embeddings"],
        d_model=c["hidden_size"],
        n_heads=c["num_attention_heads"],
        n_layers=c["num_hidden_layers"],
        d_ff=c["intermediate_size"],
        type_vocab=c["type_vocab_size"],
        dropout=0.0,
        ln_eps=c["layer_norm_eps"],
    )
    model = BertForMLM(cfg)
    tensors, _ = load_safetensors(str(d / "model.safetensors"))
    model.load_state_dict(tensors)
    model.eval()
    return model
