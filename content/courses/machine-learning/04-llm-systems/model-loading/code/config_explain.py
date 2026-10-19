#!/usr/bin/env python3
"""Explain a Hugging Face config.json: params, KV cache, attention, MoE, RoPE.

Stdlib only. Reads the same fields every loader reads, then does the
arithmetic an FDE does on a whiteboard before a customer deploy:

    params        embed + L * (attn + mlp + norms) + lm_head (unless tied)
    attn (GQA)    d*H*hd (q) + 2*d*KV*hd (k, v) + H*hd*d (o)
    mlp (SwiGLU)  3 * d * I                          (gate, up, down)
    moe layer     router d*E + E * 3*d*I_moe (+ shared experts)
    active        same, with E replaced by top_k
    KV bytes/tok  2 * L * KV * hd * bytes           (MHA/GQA/MQA)
                  L * (kv_lora_rank + rope_dim) * bytes   (MLA latent cache)
    load time     weight_bytes / bandwidth

Usage:
    python config_explain.py samples/llama-3.1-8b-instruct.config.json
    python config_explain.py samples/*.json --kv-dtype fp8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DTYPE_BYTES = {
    "float32": 4,
    "fp32": 4,
    "bfloat16": 2,
    "bf16": 2,
    "float16": 2,
    "fp16": 2,
    "fp8": 1,
    "float8": 1,
    "int8": 1,
    "int4": 0.5,
    "fp4": 0.5,
}

# Effective sequential bandwidths (GB/s) for the cold-start estimate. Order of
# magnitude only; measure your own path with fio / dd / iperf.
BANDWIDTHS = [
    ("single HTTP stream", 0.2),
    ("25 GbE object store, parallel", 3.0),
    ("local NVMe Gen4", 6.5),
    ("PCIe Gen5 x16 host->GPU", 50.0),
]


def load_config(path: str) -> dict:
    cfg = json.loads(Path(path).read_text())
    # Multimodal configs nest the LM under text_config; flatten it.
    if "text_config" in cfg and "hidden_size" not in cfg:
        cfg = {**cfg, **cfg["text_config"]}
    return cfg


def attention_variant(cfg: dict) -> str:
    h = cfg["num_attention_heads"]
    kv = cfg.get("num_key_value_heads") or h
    if cfg.get("kv_lora_rank"):
        return f"MLA (latent KV, kv_lora_rank={cfg['kv_lora_rank']}, rope_dim={cfg.get('qk_rope_head_dim')})"
    if kv == 1:
        return f"MQA (1 KV head shared by {h} query heads)"
    if kv < h:
        return f"GQA ({h} query heads / {kv} KV heads = group of {h // kv})"
    return f"MHA ({h} heads)"


def attn_params(cfg: dict) -> int:
    d, h = cfg["hidden_size"], cfg["num_attention_heads"]
    if cfg.get("kv_lora_rank"):  # DeepSeek MLA
        nope, rope = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"]
        v, r_kv, r_q = cfg["v_head_dim"], cfg["kv_lora_rank"], cfg.get("q_lora_rank")
        q = (d * r_q + r_q + r_q * h * (nope + rope)) if r_q else d * h * (nope + rope)
        kv_a = d * (r_kv + rope) + r_kv  # down-projection + its RMSNorm
        kv_b = r_kv * h * (nope + v)  # up-projection to per-head K_nope, V
        o = h * v * d
        return q + kv_a + kv_b + o
    kv = cfg.get("num_key_value_heads") or h
    hd = cfg.get("head_dim") or d // h
    p = d * h * hd + 2 * d * kv * hd + h * hd * d
    if cfg.get("attention_bias"):
        p += h * hd + 2 * kv * hd
    if cfg.get("model_type", "").startswith("qwen3"):
        p += 2 * hd  # q_norm, k_norm
    return p


def is_moe_layer(cfg: dict, i: int) -> bool:
    if cfg.get("n_routed_experts"):  # DeepSeek V2/V3
        return (
            i >= cfg.get("first_k_dense_replace", 0)
            and i % cfg.get("moe_layer_freq", 1) == 0
        )
    if cfg.get("num_experts") or cfg.get("num_local_experts"):  # Qwen MoE, Mixtral
        if i in (cfg.get("mlp_only_layers") or []):
            return False
        step = cfg.get("decoder_sparse_step", 1) or 1
        return (i + 1) % step == 0
    return False


def mlp_params(cfg: dict, moe: bool) -> tuple[int, int]:
    """(total, active) params for one MLP block."""
    d = cfg["hidden_size"]
    if not moe:
        p = 3 * d * cfg["intermediate_size"]
        return p, p
    e = (
        cfg.get("n_routed_experts")
        or cfg.get("num_experts")
        or cfg.get("num_local_experts")
    )
    k = cfg["num_experts_per_tok"]
    i_moe = cfg.get("moe_intermediate_size") or cfg["intermediate_size"]
    expert = 3 * d * i_moe
    router = d * e
    shared = 0
    if cfg.get("n_shared_experts"):
        shared = cfg["n_shared_experts"] * expert
    elif cfg.get("shared_expert_intermediate_size"):
        shared = 3 * d * cfg["shared_expert_intermediate_size"] + d  # + shared gate
    return router + e * expert + shared, router + k * expert + shared


def count_params(cfg: dict) -> tuple[int, int]:
    d, L, V = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["vocab_size"]
    embed = V * d
    head = 0 if cfg.get("tie_word_embeddings") else V * d
    total = active = embed + head + d  # + final norm
    a = attn_params(cfg)
    for i in range(L):
        t, act = mlp_params(cfg, is_moe_layer(cfg, i))
        total += a + t + 2 * d
        active += a + act + 2 * d
    return total, active


def kv_bytes_per_token(cfg: dict, nbytes: float) -> float:
    L = cfg["num_hidden_layers"]
    if cfg.get("kv_lora_rank"):
        return L * (cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]) * nbytes
    h = cfg["num_attention_heads"]
    kv = cfg.get("num_key_value_heads") or h
    hd = cfg.get("head_dim") or cfg["hidden_size"] // h
    return 2 * L * kv * hd * nbytes


def rope_summary(cfg: dict) -> str:
    theta = cfg.get("rope_theta")
    rs = cfg.get("rope_scaling") or cfg.get("rope_parameters")
    if not rs:
        return f"theta={theta}, no scaling"
    kind = rs.get("rope_type") or rs.get("type")
    factor = rs.get("factor")
    orig = rs.get("original_max_position_embeddings")
    extra = f", trained at {orig} then x{factor}" if orig and factor else ""
    return f"theta={theta or rs.get('rope_theta')}, scaling={kind}{extra}"


def weight_dtype_bytes(cfg: dict) -> tuple[str, float]:
    q = cfg.get("quantization_config") or {}
    method = q.get("quant_method")
    if method == "fp8":
        return "fp8 (pre-quantized, block scales)", 1
    if method in ("awq", "gptq"):
        return f"{method} int{q.get('bits', 4)}", q.get("bits", 4) / 8
    if method == "mxfp4":
        return "mxfp4 (MoE experts only; ~4.25 bits)", 0.53
    if method:
        return str(method), 1
    dt = cfg.get("dtype") or cfg.get("torch_dtype") or "float32"
    return dt, DTYPE_BYTES.get(dt, 2)


def secs(t: float) -> str:
    return f"{t:.1f}s" if t < 10 else f"{t:,.0f}s"


def gb(x: float) -> str:
    return f"{x / 1e9:,.2f} GB"


def explain(path: str, kv_dtype: str) -> None:
    cfg = load_config(path)
    total, active = count_params(cfg)
    wname, wbytes = weight_dtype_bytes(cfg)
    kvb = DTYPE_BYTES[kv_dtype]
    per_tok = kv_bytes_per_token(cfg, kvb)
    ctx = cfg.get("max_position_embeddings")
    weights = total * wbytes

    print(f"== {path}")
    print(
        f"  architecture    {cfg.get('architectures', ['?'])[0]}  (model_type={cfg.get('model_type')})"
    )
    if cfg.get("auto_map"):
        print(
            "  custom code     auto_map present: stock transformers may need trust_remote_code"
        )
    print(
        f"  shape           L={cfg['num_hidden_layers']} d={cfg['hidden_size']} vocab={cfg['vocab_size']}"
        f" tied_embeddings={bool(cfg.get('tie_word_embeddings'))}"
    )
    print(f"  attention       {attention_variant(cfg)}")
    moe_layers = sum(is_moe_layer(cfg, i) for i in range(cfg["num_hidden_layers"]))
    if moe_layers:
        e = (
            cfg.get("n_routed_experts")
            or cfg.get("num_experts")
            or cfg.get("num_local_experts")
        )
        print(
            f"  MoE             {moe_layers} MoE layers, {e} experts, top-{cfg['num_experts_per_tok']}"
            f", shared={cfg.get('n_shared_experts', 0)}"
        )
    print(
        f"  params          total={total / 1e9:.2f}B  active/token={active / 1e9:.2f}B"
    )
    if cfg.get("num_nextn_predict_layers"):
        print(
            f"                  (+{cfg['num_nextn_predict_layers']} MTP layer(s) in checkpoint, not counted)"
        )
    print(f"  weights         {wname} -> ~{gb(weights)}")
    print(f"  context         max_position_embeddings={ctx}; rope: {rope_summary(cfg)}")
    print(
        f"  KV cache        {per_tok / 1024:,.1f} KiB/token ({kv_dtype})"
        + (f"; one {ctx:,}-token sequence = {gb(per_tok * ctx)}" if ctx else "")
    )
    eos = cfg.get("eos_token_id")
    print(
        f"  eos_token_id    {eos}  (generation_config.json may override; mismatch => runaway output)"
    )
    print(
        "  cold load       "
        + "; ".join(f"{n}: {secs(weights / (bw * 1e9))}" for n, bw in BANDWIDTHS)
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("configs", nargs="+")
    p.add_argument("--kv-dtype", default="bf16", choices=sorted(DTYPE_BYTES))
    a = p.parse_args()
    for path in a.configs:
        explain(path, a.kv_dtype)


if __name__ == "__main__":
    main()
