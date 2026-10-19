#!/usr/bin/env python3
"""LLM serving capacity calculator (stdlib only).

Answers the first questions an FDE gets on a sizing call:

  1. Do the weights fit?           weights = params x bytes/param / TP
  2. How much KV cache is left?    budget = util x gpu_mem - weights - activation reserve
  3. How many tokens / sequences?  kv_bytes_per_token = 2 x layers x kv_heads x head_dim x bytes
  4. How fast can decode go?       tok/s <= HBM_BW / bytes read per step (roofline, memory side)
  5. What does a token cost?       $/1M = GPU $/hr x n_gpus / (tok/s x 3600) x 1e6

Usage:
  python capacity.py --preset llama-3.1-8b
  python capacity.py --preset llama-3.1-70b --tp 4 --weight-bytes 1 --kv-bytes 1
  python capacity.py --hf-config path/to/config.json --tp 2 --context 32768
  python capacity.py --list-presets

All numbers are first-order: they ignore TP all-reduce time, kernel launch
overhead, and attention FLOPs. Real engines reach roughly 60-85% of the
bandwidth bound (model bandwidth utilization, MBU); use --mbu to apply that.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace

GIB = 1024**3


@dataclass(frozen=True)
class Model:
    name: str
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    params: float  # total parameters (all experts)
    active_params: float | None = None  # per-token params for MoE; None = dense
    # MLA (DeepSeek-V2/V3): cache one compressed latent + a RoPE key per layer
    mla_latent_dim: int = 0  # kv_lora_rank + qk_rope_head_dim; 0 = standard K/V

    @property
    def active(self) -> float:
        return self.active_params if self.active_params else self.params


@dataclass(frozen=True)
class Gpu:
    name: str
    mem_gib: float
    hbm_tbps: float  # TB/s (1e12 bytes/s)
    bf16_tflops: float  # dense
    fp8_tflops: float  # dense


PRESETS: dict[str, Model] = {
    # https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json
    "llama-3.1-8b": Model("llama-3.1-8b", 32, 4096, 32, 8, 128, 8.03e9),
    # https://huggingface.co/meta-llama/Llama-3.1-70B/blob/main/config.json
    "llama-3.1-70b": Model("llama-3.1-70b", 80, 8192, 64, 8, 128, 70.6e9),
    # MHA baseline for comparison: what Llama-3.1-8B would cost without GQA
    "llama-3.1-8b-mha": Model("llama-3.1-8b-mha", 32, 4096, 32, 32, 128, 8.03e9),
    # https://huggingface.co/mistralai/Mixtral-8x7B-v0.1 (MoE, 2 of 8 experts)
    "mixtral-8x7b": Model("mixtral-8x7b", 32, 4096, 32, 8, 128, 46.7e9, 12.9e9),
    # https://huggingface.co/deepseek-ai/DeepSeek-V3 (MoE + MLA: 512 latent + 64 rope)
    "deepseek-v3": Model("deepseek-v3", 61, 7168, 128, 128, 128, 671e9, 37e9, 576),
}

GPUS: dict[str, Gpu] = {
    # Datasheet dense numbers (sparsity figures halved). SXM parts.
    "h100": Gpu("H100 SXM 80GB", 80, 3.35, 989, 1979),
    "h200": Gpu("H200 SXM 141GB", 141, 4.8, 989, 1979),
    "a100": Gpu("A100 SXM 80GB", 80, 2.0, 312, 0),
    "l40s": Gpu("L40S 48GB", 48, 0.864, 362, 733),
    "b200": Gpu("B200 180GB", 180, 8.0, 2250, 4500),
}


def kv_bytes_per_token(m: Model, kv_bytes: float) -> float:
    """Bytes of KV cache one token occupies across all layers (whole model, before TP)."""
    if m.mla_latent_dim:
        # MLA caches a single latent vector per layer, shared by all heads: no factor 2.
        return m.layers * m.mla_latent_dim * kv_bytes
    return 2 * m.layers * m.kv_heads * m.head_dim * kv_bytes


def model_from_hf_config(path: str, params: float | None) -> Model:
    with open(path) as f:
        c = json.load(f)
    c = c.get("text_config", c)  # multimodal wrappers nest the LM config
    layers = c["num_hidden_layers"]
    hidden = c["hidden_size"]
    heads = c["num_attention_heads"]
    kv_heads = c.get("num_key_value_heads", heads)
    head_dim = c.get("head_dim") or hidden // heads
    inter = c.get("intermediate_size", 4 * hidden)
    vocab = c.get("vocab_size", 32000)
    tied = c.get("tie_word_embeddings", False)
    n_exp = c.get("num_local_experts") or c.get("num_experts") or 1
    k_exp = c.get("num_experts_per_tok") or 1
    moe_inter = c.get("moe_intermediate_size", inter)
    mla = 0
    if "kv_lora_rank" in c:
        mla = c["kv_lora_rank"] + c.get("qk_rope_head_dim", 64)

    attn = (
        hidden * heads * head_dim
        + 2 * hidden * kv_heads * head_dim
        + heads * head_dim * hidden
    )
    mlp_one = (
        3 * hidden * (moe_inter if n_exp > 1 else inter)
    )  # gate, up, down (SwiGLU)
    embed = vocab * hidden * (1 if tied else 2)
    total = embed + layers * (attn + n_exp * mlp_one)
    active = embed + layers * (attn + k_exp * mlp_one)
    if mla and params is None:
        sys.exit(
            "MLA config detected: the estimator does not model MLA/shared experts, pass --params"
        )
    return Model(
        name=path,
        layers=layers,
        hidden=hidden,
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        params=params or total,
        active_params=(active if n_exp > 1 and params is None else None),
        mla_latent_dim=mla,
    )


def fmt_gib(b: float) -> str:
    return f"{b / GIB:8.2f} GiB"


def report(a: argparse.Namespace, m: Model, g: Gpu) -> int:
    tp = a.tp
    n_gpus = tp * a.pp
    w_total = m.params * a.weight_bytes
    w_per_gpu = w_total / n_gpus
    budget_per_gpu = a.util * g.mem_gib * GIB
    act_per_gpu = a.activation_gib * GIB
    kv_per_gpu = budget_per_gpu - w_per_gpu - act_per_gpu
    kv_tok = kv_bytes_per_token(m, a.kv_bytes)
    # KV heads are sharded by TP (replicated once TP > kv_heads); layers sharded by PP.
    shard = min(tp, m.kv_heads) if not m.mla_latent_dim else 1
    kv_tok_per_gpu = kv_tok / shard / a.pp
    print(
        f"model  {m.name}: {m.params / 1e9:.1f}B params"
        + (f" ({m.active / 1e9:.1f}B active)" if m.active_params else "")
        + f", {m.layers} layers, {m.heads} q-heads, {m.kv_heads} kv-heads, head_dim {m.head_dim}"
        + (f", MLA latent {m.mla_latent_dim}" if m.mla_latent_dim else "")
    )
    print(
        f"gpu    {n_gpus} x {g.name} (TP={tp}, PP={a.pp}), util={a.util}, "
        f"weights {a.weight_bytes} B/param, KV {a.kv_bytes} B/elem"
    )
    print()
    print(f"weights total            {fmt_gib(w_total)}")
    print(f"weights per GPU          {fmt_gib(w_per_gpu)}")
    print(
        f"budget per GPU           {fmt_gib(budget_per_gpu)}  (util x {g.mem_gib:.0f} GiB)"
    )
    print(f"activation reserve/GPU   {fmt_gib(act_per_gpu)}")
    print(f"KV pool per GPU          {fmt_gib(kv_per_gpu)}")
    print(f"KV per token (model)     {kv_tok / 1024:8.1f} KiB")
    print(f"KV per token (per GPU)   {kv_tok_per_gpu / 1024:8.1f} KiB")
    if kv_per_gpu <= 0:
        print(
            "\nDOES NOT FIT: weights + reserve exceed the budget. Raise TP, quantize, or use a bigger GPU."
        )
        return 1
    max_tokens = kv_per_gpu / kv_tok_per_gpu
    max_seqs = max_tokens / a.context
    print(f"max tokens in KV         {max_tokens:10,.0f}")
    print(f"max concurrent seqs      {max_seqs:10,.1f}  at {a.context:,} tokens each")
    if max_seqs < 1:
        print(
            "  warning: a single max-length sequence does not fit; lower --max-model-len"
        )

    # Decode roofline. One decode step reads every active weight once plus each
    # sequence's KV history. Per GPU: weights/TP and KV/TP, in parallel across GPUs.
    bw = g.hbm_tbps * 1e12 * a.mbu
    batch = a.batch if a.batch else max(1, int(min(max_seqs, 256)))
    avg_ctx = a.avg_ctx if a.avg_ctx else a.context // 2
    w_read = m.active * a.weight_bytes / n_gpus
    if m.active_params and batch > 1:
        # Large batches touch most experts: every expert's weights get read once per step.
        w_read = min(m.params, m.active * batch) * a.weight_bytes / n_gpus
    step_bytes_1 = w_read + 1 * avg_ctx * kv_tok_per_gpu
    step_bytes_b = w_read + batch * avg_ctx * kv_tok_per_gpu
    t_mem_1 = step_bytes_1 / bw
    t_mem_b = step_bytes_b / bw
    peak = (
        g.fp8_tflops if a.weight_bytes <= 1 and g.fp8_tflops else g.bf16_tflops
    ) * 1e12
    t_cmp_b = 2 * m.active * batch / (peak * tp)  # linear-layer FLOPs only
    t_b = max(t_mem_b, t_cmp_b)
    single = 1 / t_mem_1
    agg = batch / t_b
    ridge = peak / (g.hbm_tbps * 1e12)
    intensity = 2 * batch / a.weight_bytes  # FLOPs per weight byte, weights-only view
    print()
    print(f"decode bound (MBU={a.mbu}, avg ctx {avg_ctx:,})")
    print(
        f"  single stream          {single:10,.1f} tok/s   (ITL {t_mem_1 * 1e3:.2f} ms)"
    )
    print(
        f"  batch {batch:<4d}             {agg:10,.1f} tok/s   (ITL {t_b * 1e3:.2f} ms, "
        f"{'compute' if t_cmp_b > t_mem_b else 'memory'}-bound)"
    )
    print(
        f"  arithmetic intensity   {intensity:10,.0f} FLOP/B vs ridge {ridge:,.0f} FLOP/B"
    )

    if a.gpu_hourly:
        dollars = a.gpu_hourly * n_gpus
        cost = dollars / (agg * 3600) * 1e6
        print()
        print(
            f"cost at ${a.gpu_hourly:.2f}/GPU-hr x {n_gpus}: ${cost:,.3f} per 1M output tokens "
            f"(at batch {batch}, 100% busy)"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--preset", choices=sorted(PRESETS), default="llama-3.1-8b")
    src.add_argument("--hf-config", help="path to a Hugging Face config.json")
    p.add_argument("--list-presets", action="store_true")
    p.add_argument("--params", type=float, help="override total params (e.g. 8.03e9)")
    # manual model description (overrides preset fields when given)
    for f in ("layers", "hidden", "heads", "kv-heads", "head-dim"):
        p.add_argument(f"--{f}", type=int)
    p.add_argument("--gpu", choices=sorted(GPUS), default="h100")
    p.add_argument("--gpu-mem-gib", type=float, help="override GPU memory")
    p.add_argument("--hbm-tbps", type=float, help="override HBM bandwidth (TB/s)")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument(
        "--weight-bytes", type=float, default=2.0, help="2=BF16, 1=FP8/INT8, 0.5=INT4"
    )
    p.add_argument("--kv-bytes", type=float, default=2.0, help="2=BF16, 1=FP8 KV cache")
    p.add_argument(
        "--util", type=float, default=0.90, help="like vLLM --gpu-memory-utilization"
    )
    p.add_argument(
        "--activation-gib",
        type=float,
        default=2.0,
        help="per-GPU reserve: activations, CUDA graphs, NCCL",
    )
    p.add_argument(
        "--context", type=int, default=8192, help="tokens per sequence (prompt+output)"
    )
    p.add_argument(
        "--avg-ctx",
        type=int,
        help="average live context for the decode bound (default context/2)",
    )
    p.add_argument(
        "--batch",
        type=int,
        help="decode batch for the aggregate bound (default min(max seqs, 256))",
    )
    p.add_argument(
        "--mbu",
        type=float,
        default=1.0,
        help="model bandwidth utilization, 0.6-0.85 is realistic",
    )
    p.add_argument(
        "--gpu-hourly", type=float, default=2.50, help="$/GPU-hour; 0 to skip cost"
    )
    a = p.parse_args(argv)

    if a.list_presets:
        for k, m in PRESETS.items():
            print(
                f"{k:18s} {m.params / 1e9:7.1f}B  layers={m.layers} kv_heads={m.kv_heads} head_dim={m.head_dim}"
            )
        return 0

    m = (
        model_from_hf_config(a.hf_config, a.params)
        if a.hf_config
        else PRESETS[a.preset]
    )
    overrides = {
        k: v
        for k, v in {
            "layers": a.layers,
            "hidden": a.hidden,
            "heads": a.heads,
            "kv_heads": a.kv_heads,
            "head_dim": a.head_dim,
            "params": a.params,
        }.items()
        if v is not None
    }
    m = replace(m, **overrides)
    g = GPUS[a.gpu]
    if a.gpu_mem_gib or a.hbm_tbps:
        g = replace(
            g, mem_gib=a.gpu_mem_gib or g.mem_gib, hbm_tbps=a.hbm_tbps or g.hbm_tbps
        )
    return report(a, m, g)


if __name__ == "__main__":
    raise SystemExit(main())
