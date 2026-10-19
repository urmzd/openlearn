"""ELECTRA: replaced-token detection (L6.3).

BERT learns from the 15% of positions it masks. ELECTRA (Clark et al. 2020)
learns from every position: a small generator (a masked LM, L6.2) fills the
masked positions with samples, and the discriminator, the model we keep,
labels every token of the result as original or replaced. A sampled token
that happens to equal the original counts as original. The generator trains
on its MLM loss only; nothing flows back through the sampling.

Contract: contracts/py/tinyllm/obj/electra.pyi.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd import functional as F
from tinyllm.autograd.losses import bce_with_logits
from tinyllm.autograd.mode import no_grad
from tinyllm.autograd.tensor import Tensor
from tinyllm.io.safetensors import load_safetensors, save_safetensors
from tinyllm.nn.init import normal_init
from tinyllm.nn.layers import Linear
from tinyllm.nn.module import Module
from tinyllm.num.rng import PCG32
from tinyllm.obj.bert import BertConfig, BertEncoder, BertForMLM, mlm_mask
from tinyllm.prob.sampling import sample_categorical

INIT_STD = 0.02  # HF ElectraConfig.initializer_range, as BERT's
# BertConfig field -> the Hugging Face config.json key (formats/config.schema.json).
HF_KEYS = {
    "vocab": "vocab_size",
    "max_len": "max_position_embeddings",
    "d_model": "hidden_size",
    "n_heads": "num_attention_heads",
    "n_layers": "num_hidden_layers",
    "d_ff": "intermediate_size",
    "type_vocab": "type_vocab_size",
    "ln_eps": "layer_norm_eps",
}


def _init_rng(rng: Any) -> Any:
    return rng if rng is not None else PCG32(0).substream("init")


class ElectraDiscriminatorHead(Module):
    def __init__(self, d_model: int, rng: Any = None) -> None:
        super().__init__()
        r = _init_rng(rng)
        self.dense = Linear(d_model, d_model, rng=r)
        self.dense.weight.data = normal_init((d_model, d_model), INIT_STD, r).astype(np.float32)
        self.dense_prediction = Linear(d_model, 1, rng=r)
        self.dense_prediction.weight.data = normal_init((1, d_model), INIT_STD, r).astype(np.float32)

    def forward(self, h: Tensor) -> Tensor:
        z = self.dense_prediction(F.gelu(self.dense(h)))  # [..., 1]
        return F.reshape(z, z.shape[:-1])


class ElectraDiscriminator(Module):
    def __init__(self, cfg: BertConfig, rng: Any = None) -> None:
        super().__init__()
        r = _init_rng(rng)
        self.cfg = cfg
        self.electra = BertEncoder(cfg, r)
        self.discriminator_predictions = ElectraDiscriminatorHead(cfg.d_model, r)

    def forward(
        self, ids: ArrayLike, token_type_ids: Optional[ArrayLike] = None, attn_mask: Optional[ArrayLike] = None
    ) -> Tensor:
        return self.discriminator_predictions(self.electra(ids, token_type_ids, attn_mask))


class ELECTRA(Module):
    def __init__(
        self, gen_cfg: BertConfig, disc_cfg: BertConfig, rng: Any = None, tie_embeddings: bool = True
    ) -> None:
        super().__init__()
        r = _init_rng(rng)
        if gen_cfg.vocab != disc_cfg.vocab:
            raise ValueError(f"generator vocab {gen_cfg.vocab} != discriminator vocab {disc_cfg.vocab}")
        same = (gen_cfg.d_model, gen_cfg.max_len, gen_cfg.type_vocab) == (
            disc_cfg.d_model,
            disc_cfg.max_len,
            disc_cfg.type_vocab,
        )
        if tie_embeddings and not same:
            raise ValueError("tied embeddings need equal d_model, max_len, and type_vocab")
        self.tie_embeddings = bool(tie_embeddings)
        self.discriminator = ElectraDiscriminator(disc_cfg, r)
        self.generator = BertForMLM(gen_cfg, r)
        if tie_embeddings:
            # One Tensor under two names: listed once, under the discriminator's.
            g, d = self.generator.bert, self.discriminator.electra
            g.word_emb.weight = d.word_emb.weight
            g.pos_emb.weight = d.pos_emb.weight
            g.type_emb.weight = d.type_emb.weight

    def forward(self, ids: ArrayLike, rng: Any, **kw: Any) -> dict[str, Any]:
        return electra_step(self.generator, self.discriminator, ids, rng, **kw)


def replace_tokens(
    ids: ArrayLike, labels: ArrayLike, gen_logits: ArrayLike, rng: Any, ignore_index: int = -100
) -> tuple[NDArray, NDArray]:
    x = np.asarray(ids)
    lab = np.asarray(labels)
    z = np.asarray(gen_logits, dtype=np.float64)
    if x.dtype.kind not in "iu" or lab.shape != x.shape or z.shape[:-1] != x.shape:
        raise ValueError(f"ids, labels [B, T] and logits [B, T, V], got {x.shape}, {lab.shape}, {z.shape}")
    corrupt = x.astype(np.int64).copy()
    flat_c, flat_l = corrupt.reshape(-1), lab.reshape(-1)
    rows = z.reshape(-1, z.shape[-1])
    for i in range(flat_c.size):  # C order, one uniform per masked position
        if flat_l[i] == ignore_index:
            continue
        w = np.exp(rows[i] - rows[i].max())
        flat_c[i] = sample_categorical(w / w.sum(), rng.uniform())  # M07.1
    # Replaced means DIFFERENT from the original: a lucky sample is original.
    return corrupt, (corrupt != x).astype(np.float32)


def rtd_loss(disc_logits: Tensor, is_replaced: ArrayLike, attn_mask: Optional[ArrayLike] = None) -> Tensor:
    y = np.asarray(is_replaced, dtype=np.float32)
    if y.shape != disc_logits.shape:
        raise ValueError(f"labels {y.shape} != logits {disc_logits.shape}")
    real = np.ones(y.shape, dtype=bool) if attn_mask is None else np.asarray(attn_mask).astype(bool)
    if real.shape != y.shape:
        raise ValueError(f"attn_mask {real.shape} != logits {y.shape}")
    # Mean over the real tokens only, as HF's ElectraForPreTraining.
    return bce_with_logits(disc_logits[real], y[real])


def electra_step(
    gen: Module, disc: Module, ids: ArrayLike, rng: Any, lam: float = 50.0, *, mask_id: int,
    vocab: Optional[int] = None, special_mask: Optional[ArrayLike] = None,
    token_type_ids: Optional[ArrayLike] = None, attn_mask: Optional[ArrayLike] = None, p: float = 0.15,
) -> dict[str, Any]:
    x = np.asarray(ids)
    real = np.ones(x.shape, dtype=bool) if attn_mask is None else np.asarray(attn_mask).astype(bool)
    special = np.zeros(x.shape, dtype=bool) if special_mask is None else np.asarray(special_mask, dtype=bool)
    V = vocab if vocab is not None else gen.bert.cfg.vocab
    # 1. mask like BERT (padding counts as special: never picked).
    inputs, labels = mlm_mask(x, special | ~real, mask_id, V, p, rng)
    # 2. the generator's MLM loss on the picked positions.
    gen_logits, gen_loss = gen(inputs, token_type_ids, attn_mask, labels)
    # 3. sample replacements: data only, no gradient through the choice.
    corrupt, is_replaced = replace_tokens(x, labels, gen_logits.data, rng)
    # 4. the discriminator labels every real token.
    disc_logits = disc(corrupt, token_type_ids, attn_mask)
    disc_loss = rtd_loss(disc_logits, is_replaced, real)
    return {
        "loss": gen_loss + disc_loss * lam,
        "gen_loss": gen_loss,
        "disc_loss": disc_loss,
        "disc_logits": disc_logits,
        "inputs": inputs,
        "labels": labels,
        "corrupt": corrupt,
        "is_replaced": is_replaced,
    }


def rtd_accuracy(
    model: ELECTRA, ids: ArrayLike, rng: Any, *, mask_id: int, special_mask: Optional[ArrayLike] = None,
    token_type_ids: Optional[ArrayLike] = None, attn_mask: Optional[ArrayLike] = None, p: float = 0.15,
) -> NDArray:
    with no_grad():
        out = electra_step(
            model.generator,
            model.discriminator,
            ids,
            rng,
            mask_id=mask_id,
            special_mask=special_mask,
            token_type_ids=token_type_ids,
            attn_mask=attn_mask,
            p=p,
        )
    x = np.asarray(ids)
    real = np.ones(x.shape, dtype=bool) if attn_mask is None else np.asarray(attn_mask).astype(bool)
    said = out["disc_logits"].data > 0  # logit > 0: "replaced"
    right = said == (out["is_replaced"] > 0.5)
    return right[real].astype(np.float64)


def _cfg_json(cfg: BertConfig) -> dict[str, Any]:
    d = asdict(cfg)
    return {HF_KEYS[k]: d[k] for k in HF_KEYS}


def _cfg_from(d: dict[str, Any]) -> BertConfig:
    return BertConfig(**{k: d[v] for k, v in HF_KEYS.items()}, dropout=0.0)


def save_electra(
    model: ELECTRA, dir: str, mask_id: int, special_ids: Sequence[int] = (), tokenizer: str = "file"
) -> None:
    out = Path(dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {
        "tl_arch": "electra",
        "tl_tokenizer": tokenizer,
        "tl_format": 1,
        **_cfg_json(model.discriminator.cfg),
        "tl_generator": _cfg_json(model.generator.bert.cfg),
        "tl_tie_embeddings": model.tie_embeddings,
        "tl_mask_id": int(mask_id),
        "tl_special_ids": sorted(int(i) for i in special_ids),
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    save_safetensors(str(out / "model.safetensors"), model.state_dict(), {"format": "tinyllm", "tl_arch": "electra"})


def load_electra(dir: str) -> tuple[ELECTRA, dict[str, Any]]:
    d = Path(dir)
    cfg = json.loads((d / "config.json").read_text())
    if cfg.get("tl_arch") != "electra":
        raise ValueError(f"{d}/config.json: tl_arch is {cfg.get('tl_arch')!r}, not 'electra'")
    model = ELECTRA(_cfg_from(cfg["tl_generator"]), _cfg_from(cfg), tie_embeddings=bool(cfg["tl_tie_embeddings"]))
    tensors, _ = load_safetensors(str(d / "model.safetensors"))
    model.load_state_dict(tensors)
    model.eval()
    return model, cfg
