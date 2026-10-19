"""Counting a decoder: parameters, FLOPs, KV bytes, training memory (M05.1).

Every number here is a product of a few config integers, summed over the
parts of a block. Getting them exactly right is what lets `{tinyllm} info`
report the same parameter count as Hugging Face, the engine admit requests
by KV bytes, and the capstone pick a model that fits in memory.

Contract: contracts/py/tinyllm/accounting.pyi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelConfig:
    vocab: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_head: int
    d_ff: int
    tie_embeddings: bool
    attn: Literal["mha", "gqa", "mla"] = "gqa"
    kv_lora_rank: int = 0
    qk_rope_dim: int = 0
    n_experts: int = 0
    top_k: int = 0
    n_shared: int = 0
    q_lora_rank: int = 0
    v_head_dim: int = 0
    d_ff_expert: int = 0
    n_dense_layers: int = 0
    qkv_bias: bool = False


def _check(cfg: ModelConfig) -> None:
    for name in ("vocab", "d_model", "n_layers", "n_heads", "n_kv_heads", "d_head", "d_ff"):
        v = getattr(cfg, name)
        if isinstance(v, bool) or int(v) != v or v < 1:
            raise ValueError(f"{name} must be a positive integer, got {v!r}")
    for name in ("kv_lora_rank", "qk_rope_dim", "n_experts", "top_k", "n_shared",
                 "q_lora_rank", "v_head_dim", "d_ff_expert", "n_dense_layers"):
        v = getattr(cfg, name)
        if isinstance(v, bool) or int(v) != v or v < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {v!r}")
    if cfg.attn == "mha":
        if cfg.n_kv_heads != cfg.n_heads:
            raise ValueError(f"mha needs n_kv_heads == n_heads, got {cfg.n_kv_heads} and {cfg.n_heads}")
    elif cfg.attn == "gqa":
        if cfg.n_heads % cfg.n_kv_heads:
            raise ValueError(f"gqa needs n_heads % n_kv_heads == 0, got {cfg.n_heads} and {cfg.n_kv_heads}")
    elif cfg.attn == "mla":
        if cfg.kv_lora_rank < 1:
            raise ValueError("mla needs kv_lora_rank >= 1")
    else:
        raise ValueError(f"attn must be 'mha', 'gqa', or 'mla', got {cfg.attn!r}")
    if cfg.n_experts > 0:
        if not 1 <= cfg.top_k <= cfg.n_experts:
            raise ValueError(f"top_k must lie in [1, n_experts = {cfg.n_experts}], got {cfg.top_k}")
    elif cfg.top_k or cfg.n_shared:
        raise ValueError("top_k and n_shared need n_experts > 0")
    if cfg.n_dense_layers > cfg.n_layers:
        raise ValueError(f"n_dense_layers must lie in [0, n_layers = {cfg.n_layers}], got {cfg.n_dense_layers}")


def _widths(cfg: ModelConfig) -> tuple[int, int, int]:
    """(q width, k width, v width) per token: the per-layer attention activations."""
    H, dh = cfg.n_heads, cfg.d_head
    if cfg.attn == "mla":
        dv = cfg.v_head_dim or dh
        return H * (dh + cfg.qk_rope_dim), H * (dh + cfg.qk_rope_dim), H * dv
    return H * dh, cfg.n_kv_heads * dh, cfg.n_kv_heads * dh


def _attn_params(cfg: ModelConfig) -> int:
    """Attention parameters of one layer, its norm weights excluded."""
    d, H, dh = cfg.d_model, cfg.n_heads, cfg.d_head
    if cfg.attn == "mla":
        dr, r, rq = cfg.qk_rope_dim, cfg.kv_lora_rank, cfg.q_lora_rank
        dv = cfg.v_head_dim or dh
        q = d * H * (dh + dr) if rq == 0 else d * rq + rq * H * (dh + dr)
        kv = d * (r + dr) + r * H * (dh + dv)
        return q + kv + H * dv * d
    q, kv = H * dh, cfg.n_kv_heads * dh
    bias = (q + 2 * kv) if cfg.qkv_bias else 0
    return d * q + 2 * d * kv + q * d + bias


def _moe_layers(cfg: ModelConfig) -> int:
    return cfg.n_layers - cfg.n_dense_layers if cfg.n_experts > 0 else 0


def param_count(cfg: ModelConfig) -> dict[str, int]:
    _check(cfg)
    d, V, L = cfg.d_model, cfg.vocab, cfg.n_layers
    fe = cfg.d_ff_expert or cfg.d_ff
    n_moe = _moe_layers(cfg)
    dense_mlp = 3 * d * cfg.d_ff
    moe_mlp = cfg.n_experts * d + (cfg.n_experts + cfg.n_shared) * 3 * d * fe
    mlp = (L - n_moe) * dense_mlp + n_moe * moe_mlp
    norm_per_layer = 2 * d
    if cfg.attn == "mla":
        norm_per_layer += cfg.kv_lora_rank + cfg.q_lora_rank
    out = {
        "embed": V * d,
        "attn": L * _attn_params(cfg),
        "mlp": mlp,
        "norm": L * norm_per_layer + d,
        "lm_head": 0 if cfg.tie_embeddings else V * d,
    }
    out["total"] = sum(out.values())
    out["active"] = out["total"] - n_moe * (cfg.n_experts - cfg.top_k) * 3 * d * fe
    return out


def flops_per_token(cfg: ModelConfig, seq_len: int, training: bool) -> int:
    _check(cfg)
    if isinstance(seq_len, bool) or int(seq_len) != seq_len or seq_len < 1:
        raise ValueError(f"seq_len must be a positive integer, got {seq_len!r}")
    p = param_count(cfg)
    n_moe = _moe_layers(cfg)
    fe = cfg.d_ff_expert or cfg.d_ff
    active_mlp = p["mlp"] - n_moe * (cfg.n_experts - cfg.top_k) * 3 * cfg.d_model * fe
    n = p["attn"] + active_mlp + cfg.vocab * cfg.d_model
    if cfg.attn == "mla":
        d_qk, d_v = cfg.d_head + cfg.qk_rope_dim, cfg.v_head_dim or cfg.d_head
    else:
        d_qk = d_v = cfg.d_head
    fwd = 2 * n + 2 * cfg.n_layers * cfg.n_heads * int(seq_len) * (d_qk + d_v)
    return 3 * fwd if training else fwd


def kv_bytes_per_token(cfg: ModelConfig, dtype_bytes: int) -> int:
    _check(cfg)
    if isinstance(dtype_bytes, bool) or int(dtype_bytes) != dtype_bytes or dtype_bytes < 1:
        raise ValueError(f"dtype_bytes must be a positive integer, got {dtype_bytes!r}")
    if cfg.attn == "mla":
        per_layer = cfg.kv_lora_rank + cfg.qk_rope_dim
    else:
        per_layer = 2 * cfg.n_kv_heads * cfg.d_head
    return cfg.n_layers * per_layer * int(dtype_bytes)


def _saved_per_token(cfg: ModelConfig, seq: int, moe: bool) -> int:
    """A_l: values one layer keeps per token for its backward."""
    d = cfg.d_model
    q, k, v = _widths(cfg)
    d_v = (cfg.v_head_dim or cfg.d_head) if cfg.attn == "mla" else cfg.d_head
    attn = 2 * d + q + k + v + cfg.n_heads * seq + cfg.n_heads * d_v
    if cfg.attn == "mla":
        attn += cfg.kv_lora_rank + cfg.qk_rope_dim + cfg.q_lora_rank
    if moe:
        f_tok = (cfg.top_k + cfg.n_shared) * (cfg.d_ff_expert or cfg.d_ff) + cfg.n_experts
    else:
        f_tok = cfg.d_ff
    return attn + 2 * d + 4 * f_tok


def memory_plan(
    cfg: ModelConfig, batch: int, seq: int, dtype_bytes: int, optimizer: Literal["sgd", "adamw"]
) -> dict[str, int]:
    _check(cfg)
    for name, v in (("batch", batch), ("seq", seq)):
        if isinstance(v, bool) or int(v) != v or v < 1:
            raise ValueError(f"{name} must be a positive integer, got {v!r}")
    if dtype_bytes not in (1, 2, 4, 8) or isinstance(dtype_bytes, bool):
        raise ValueError(f"dtype_bytes must be 1, 2, 4, or 8, got {dtype_bytes!r}")
    states = {"sgd": 1, "adamw": 2}.get(optimizer)
    if states is None:
        raise ValueError(f"optimizer must be 'sgd' or 'adamw', got {optimizer!r}")
    P = param_count(cfg)["total"]
    n_moe = _moe_layers(cfg)
    n_dense = cfg.n_layers - n_moe
    per_token = (
        n_dense * _saved_per_token(cfg, seq, False) + n_moe * _saved_per_token(cfg, seq, True)
    ) * dtype_bytes + 4 * cfg.vocab
    out = {
        "weights": P * dtype_bytes,
        "grads": P * dtype_bytes,
        "master": 4 * P if dtype_bytes < 4 else 0,
        "optimizer": 4 * P * states,
        "activations": batch * seq * per_token,
    }
    out["total"] = sum(out.values())
    return out
