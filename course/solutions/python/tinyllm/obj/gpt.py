"""GPT: a decoder-only transformer, the causal LM loss, GPT-2 weights (L6.1).

Drop the encoder and the cross-attention from L5.5 and keep one stack of
masked self-attention blocks: every position predicts the next token from
the tokens before it. GPT-2's block is pre-LN (L5.5's `norm="pre"`) with
learned positions (L5.4), a tanh-approximated GELU, a final LayerNorm, and
an output layer tied to the token embedding.

Contract: contracts/py/tinyllm/obj/gpt.pyi.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd import functional as F
from tinyllm.autograd.losses import cross_entropy
from tinyllm.autograd.tensor import Tensor
from tinyllm.io.safetensors import load_safetensors, save_safetensors
from tinyllm.nn.init import normal_init, scaled_residual_std
from tinyllm.nn.layers import Dropout, Embedding, LayerNorm, Linear, ModuleList
from tinyllm.nn.module import Module
from tinyllm.num.rng import PCG32
from tinyllm.optim.adamw import AdamW
from tinyllm.optim.schedule import cosine_with_warmup
from tinyllm.train.loop import train_step
from tinyllm.xfmr.masks import causal_mask
from tinyllm.xfmr.mha import MultiHeadAttention
from tinyllm.xfmr.pos import LearnedPE

INIT_STD = 0.02  # GPT-2's initializer_range


@dataclass
class GPTConfig:
    vocab: int
    n_ctx: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    dropout: float = 0.0
    ln_eps: float = 1e-5
    tie: bool = True


def _init_linear(lin: Linear, std: float, rng: Any) -> None:
    lin.weight.data = normal_init(lin.weight.shape, std, rng)


class Block(Module):
    def __init__(self, cfg: GPTConfig, rng: Any) -> None:
        super().__init__()
        d = cfg.d_model
        self.ln_1 = LayerNorm(d, cfg.ln_eps)
        self.attn = MultiHeadAttention(d, cfg.n_heads, cfg.dropout, rng=rng)
        self.ln_2 = LayerNorm(d, cfg.ln_eps)
        self.mlp_fc = Linear(d, cfg.d_ff, rng=rng)
        self.mlp_proj = Linear(cfg.d_ff, d, rng=rng)
        self.drop = Dropout(cfg.dropout)
        resid = scaled_residual_std(INIT_STD, cfg.n_layers)
        for lin in (self.attn.q_proj, self.attn.k_proj, self.attn.v_proj, self.mlp_fc):
            _init_linear(lin, INIT_STD, rng)
        # The two projections that write into the residual stream.
        for lin in (self.attn.out_proj, self.mlp_proj):
            _init_linear(lin, resid, rng)

    def forward(self, x: Tensor, mask: NDArray) -> Tensor:
        h = self.ln_1(x)
        x = x + self.drop(self.attn(h, h, mask))
        m = self.mlp_proj(F.gelu(self.mlp_fc(self.ln_2(x)), approximate="tanh"))
        return x + self.drop(m)


class GPT(Module):
    def __init__(self, cfg: GPTConfig, rng: Any = None) -> None:
        super().__init__()
        if (
            min(cfg.vocab, cfg.n_ctx, cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.d_ff)
            < 1
        ):
            raise ValueError(f"sizes must be positive: {cfg}")
        r = rng if rng is not None else PCG32(0).substream("init")
        self.cfg = cfg
        self.wte = Embedding(cfg.vocab, cfg.d_model, rng=r)
        self.wte.weight.data = normal_init((cfg.vocab, cfg.d_model), INIT_STD, r)
        self.wpe = LearnedPE(cfg.n_ctx, cfg.d_model, std=INIT_STD, rng=r)
        self.drop = Dropout(cfg.dropout)
        self.h = ModuleList([Block(cfg, r) for _ in range(cfg.n_layers)])
        self.ln_f = LayerNorm(cfg.d_model, cfg.ln_eps)
        if not cfg.tie:
            self.lm_head = Linear(cfg.d_model, cfg.vocab, bias=False, rng=r)
            _init_linear(self.lm_head, INIT_STD, r)

    def hidden(self, ids: ArrayLike) -> Tensor:
        ids = np.asarray(ids)
        if ids.ndim != 2 or ids.dtype.kind not in "iu":
            raise ValueError(f"ids must be int [B, T], got {ids.dtype} {ids.shape}")
        T = ids.shape[1]
        if T > self.cfg.n_ctx:
            raise ValueError(f"T = {T} exceeds the context n_ctx = {self.cfg.n_ctx}")
        x = self.drop(self.wpe(self.wte(ids)))
        mask = causal_mask(T)  # position t reads tokens 0..t
        for block in self.h:
            x = block(x, mask)
        return self.ln_f(x)

    def forward(
        self, ids: ArrayLike, targets: Optional[ArrayLike] = None
    ) -> tuple[Tensor, Optional[Tensor]]:
        x = self.hidden(ids)
        if self.cfg.tie:
            # The output layer IS the embedding table: logits = x E^T.
            logits = F.matmul(x, F.transpose(self.wte.weight, 0, 1))
        else:
            logits = self.lm_head(x)
        loss = (
            None
            if targets is None
            else cross_entropy(logits, np.asarray(targets).astype(np.int64))
        )
        return logits, loss


def clm_loss(
    logits: Tensor, ids: ArrayLike, mask: Optional[ArrayLike] = None
) -> Tensor:
    ids = np.asarray(ids).astype(np.int64)
    if logits.ndim != 3 or logits.shape[:2] != ids.shape:
        raise ValueError(
            f"logits [B, T, V] must match ids [B, T]: {logits.shape} vs {ids.shape}"
        )
    if ids.shape[1] < 2:
        raise ValueError(
            "the causal LM loss needs T >= 2 (position t predicts token t + 1)"
        )
    # Position t predicts token t + 1: drop the last logit row and the first id.
    targets = ids[:, 1:].copy()
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != ids.shape:
            raise ValueError(f"mask must be [B, T] = {ids.shape}, got {m.shape}")
        targets[~m[:, 1:]] = -100  # a padded target is not a prediction
    return cross_entropy(logits[:, :-1, :], targets, ignore_index=-100)


def _hf_key(sd: Mapping[str, Any], name: str) -> NDArray:
    for k in ("transformer." + name, name):
        if k in sd:
            return np.asarray(sd[k], dtype=np.float32)
    raise KeyError(f"GPT-2 state dict has no {name!r} (with or without 'transformer.')")


def load_hf_gpt2(model: GPT, sd: Mapping[str, Any]) -> None:
    c = model.cfg
    d = c.d_model
    out: dict[str, NDArray] = {
        "wte.weight": _hf_key(sd, "wte.weight"),
        "wpe.weight": _hf_key(sd, "wpe.weight"),
        "ln_f.weight": _hf_key(sd, "ln_f.weight"),
        "ln_f.bias": _hf_key(sd, "ln_f.bias"),
    }
    for i in range(c.n_layers):
        p = f"h.{i}."
        # HF's Conv1D stores weight as [in, out] (x @ W): transpose to Linear's [out, in].
        w_attn = _hf_key(sd, p + "attn.c_attn.weight")
        b_attn = _hf_key(sd, p + "attn.c_attn.bias")
        if w_attn.shape != (d, 3 * d):
            raise ValueError(
                f"{p}attn.c_attn.weight must be [d, 3 d] = {(d, 3 * d)}, got {w_attn.shape}"
            )
        # The 3 d output columns are q | k | v, each d wide.
        for j, n in enumerate(("q_proj", "k_proj", "v_proj")):
            out[f"{p}attn.{n}.weight"] = w_attn[:, j * d : (j + 1) * d].T
            out[f"{p}attn.{n}.bias"] = b_attn[j * d : (j + 1) * d]
        out[p + "attn.out_proj.weight"] = _hf_key(sd, p + "attn.c_proj.weight").T
        out[p + "attn.out_proj.bias"] = _hf_key(sd, p + "attn.c_proj.bias")
        out[p + "mlp_fc.weight"] = _hf_key(sd, p + "mlp.c_fc.weight").T
        out[p + "mlp_fc.bias"] = _hf_key(sd, p + "mlp.c_fc.bias")
        out[p + "mlp_proj.weight"] = _hf_key(sd, p + "mlp.c_proj.weight").T
        out[p + "mlp_proj.bias"] = _hf_key(sd, p + "mlp.c_proj.bias")
        for ln in ("ln_1", "ln_2"):
            out[f"{p}{ln}.weight"] = _hf_key(sd, f"{p}{ln}.weight")
            out[f"{p}{ln}.bias"] = _hf_key(sd, f"{p}{ln}.bias")
    if not c.tie:
        out["lm_head.weight"] = np.asarray(sd["lm_head.weight"], dtype=np.float32)
    model.load_state_dict(out)


def fit_gpt(
    model: GPT,
    stream: Any,
    steps: int,
    lr: float,
    warmup: int = 0,
    weight_decay: float = 0.1,
    clip: Optional[float] = 1.0,
) -> list[float]:
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    opt = AdamW(
        list(model.parameters()), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay
    )
    model.train()

    def loss_fn(m: GPT, b: dict) -> Tensor:
        return m(b["x"], b["y"])[1]

    losses = []
    for step in range(steps):
        x, y = stream.next_batch()
        opt.lr = cosine_with_warmup(step, warmup, steps, lr, 0.1 * lr)
        losses.append(train_step(model, {"x": x, "y": y}, loss_fn, opt, clip)["loss"])
    return losses


def save_gpt(model: GPT, dir: str, tokenizer: str = "bytes") -> None:
    d = Path(dir)
    d.mkdir(parents=True, exist_ok=True)
    c = model.cfg
    cfg = {
        "tl_arch": "gpt",
        "tl_tokenizer": tokenizer,
        "tl_format": 1,
        "vocab_size": c.vocab,
        "n_positions": c.n_ctx,
        "n_embd": c.d_model,
        "n_head": c.n_heads,
        "n_layer": c.n_layers,
        "n_inner": c.d_ff,
        "layer_norm_epsilon": c.ln_eps,
        "tie_word_embeddings": c.tie,
    }
    (d / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    tensors = {
        k: np.ascontiguousarray(v, dtype=np.float32)
        for k, v in model.state_dict().items()
    }
    save_safetensors(
        str(d / "model.safetensors"), tensors, {"format": "tinyllm", "tl_arch": "gpt"}
    )


def load_gpt(dir: str) -> GPT:
    d = Path(dir)
    c = json.loads((d / "config.json").read_text())
    if c.get("tl_arch") != "gpt":
        raise ValueError(f"{d}/config.json: tl_arch is {c.get('tl_arch')!r}, not 'gpt'")
    cfg = GPTConfig(
        vocab=c["vocab_size"],
        n_ctx=c["n_positions"],
        d_model=c["n_embd"],
        n_heads=c["n_head"],
        n_layers=c["n_layer"],
        d_ff=c["n_inner"],
        ln_eps=c["layer_norm_epsilon"],
        tie=c["tie_word_embeddings"],
    )
    model = GPT(cfg)
    tensors, _ = load_safetensors(str(d / "model.safetensors"))
    model.load_state_dict(tensors)
    model.eval()
    return model
