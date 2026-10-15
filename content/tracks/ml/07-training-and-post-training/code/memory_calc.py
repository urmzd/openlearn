"""memory_calc.py -- training memory, FLOPs, and GPU-hours, stdlib only (README §2, §3, §5, §6).

Math -> code:
    params (Llama-style)   N = V*d*(2 if untied) + L*(attn + 3*d*d_ff + 2d) + d
    full FT, mixed AdamW   16 B/param = 2 (bf16 W) + 2 (bf16 grad) + 4 (fp32 master) + 8 (Adam m, v)
                           18 B/param if gradients are accumulated in fp32
    LoRA                   2 B/param frozen bf16 base + 16 B per adapter param
    QLoRA                  ~0.516 B/param NF4 base (4 bits + double-quantized absmax) + 16 B per adapter param
    activations            ~34*s*b*d bytes/layer with FlashAttention (Korthikanti et al. 2022);
                           full checkpointing keeps 2*s*b*d per layer + one layer recomputed
    training FLOPs         full: 6*N*D     LoRA/QLoRA: ~4*N*D (no weight-grad matmuls for frozen W)
    GPU-hours              FLOPs / (peak_flops * MFU) / 3600

Estimates, not guarantees: ignores fragmentation, comms buffers, CUDA context
(~1-3 GB/GPU), and kernel workspaces. Use them to scope, then measure.

Run:
    python memory_calc.py                          # both presets, all methods + pretraining example
    python memory_calc.py --preset 70b --tokens 50e6 --seq 4096 --batch 1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

GB = 1e9


@dataclass(frozen=True)
class Arch:
    name: str
    layers: int
    d_model: int
    d_ff: int
    n_heads: int
    n_kv_heads: int
    vocab: int
    tied: bool = False

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def params(self) -> int:
        d, kv = self.d_model, self.n_kv_heads * self.head_dim
        attn = d * d + 2 * d * kv + d * d  # q, k, v, o
        mlp = 3 * d * self.d_ff  # gate, up, down (SwiGLU)
        per_layer = attn + mlp + 2 * d  # two RMSNorm gains
        emb = self.vocab * d * (1 if self.tied else 2)
        return emb + self.layers * per_layer + d

    def lora_params(self, rank: int) -> int:
        """LoRA on all linear projections: q, k, v, o, gate, up, down. Each adds r*(in+out)."""
        d, kv, f = self.d_model, self.n_kv_heads * self.head_dim, self.d_ff
        per_layer = rank * ((d + d) + 2 * (d + kv) + (d + d) + 2 * (d + f) + (f + d))
        return self.layers * per_layer


PRESETS = {
    # Llama 3.x 8B / 70B shapes (public configs)
    "8b": Arch("Llama-3-8B", 32, 4096, 14336, 32, 8, 128256),
    "70b": Arch("Llama-3-70B", 80, 8192, 28672, 64, 8, 128256),
}

GPUS = {  # dense BF16 peak, no sparsity; memory in GB
    "h100": (989e12, 80),
    "h200": (989e12, 141),
    "b200": (2250e12, 180),
    "a100": (312e12, 80),
}


def activation_bytes(a: Arch, seq: int, batch: int, checkpoint: bool) -> float:
    per_layer_full = 34 * seq * batch * a.d_model  # bf16, FlashAttention (no s^2 term)
    logits = seq * batch * a.vocab * 4  # fp32 logits for the loss, often the surprise
    if checkpoint:
        return a.layers * 2 * seq * batch * a.d_model + per_layer_full + logits
    return a.layers * per_layer_full + logits


def state_bytes(a: Arch, method: str, rank: int) -> tuple[float, int]:
    n = a.params()
    if method == "full":
        return 16 * n, n
    lora = a.lora_params(rank)
    base = (
        2 * n if method == "lora" else n * (4 + 0.127) / 8
    )  # NF4 + double-quant constants
    return base + 16 * lora, lora


def flops(n_params: int, tokens: float, method: str) -> float:
    return (6 if method == "full" else 4) * n_params * tokens


def gpu_hours(total_flops: float, gpu: str, mfu: float) -> float:
    return total_flops / (GPUS[gpu][0] * mfu) / 3600


def report(
    a: Arch, tokens: float, seq: int, batch: int, rank: int, gpu: str, mfu: float
) -> None:
    n = a.params()
    mem_gb = GPUS[gpu][1]
    print(
        f"\n== {a.name}: {n / 1e9:.2f}B params, fine-tune on {tokens:.3g} tokens, "
        f"seq {seq} x micro-batch {batch}, {gpu.upper()} @ {mfu:.0%} MFU"
    )
    print(
        f"{'method':<7}{'trainable':>14}{'states GB':>11}{'act GB(ckpt)':>14}"
        f"{'total GB':>10}{'min GPUs':>10}{'FLOPs':>11}{'GPU-h':>9}"
    )
    for method in ("full", "lora", "qlora"):
        states, trainable = state_bytes(a, method, rank)
        act = activation_bytes(a, seq, batch, checkpoint=True)
        total = states + act
        # FSDP / ZeRO-3 shards states (and a frozen base) across GPUs; activations stay per GPU
        min_gpus = 1
        while states / min_gpus + act > 0.9 * mem_gb * GB:
            min_gpus += 1
        f = flops(n, tokens, method)
        print(
            f"{method:<7}{trainable / 1e6:>13.1f}M{states / GB:>11.1f}{act / GB:>14.1f}"
            f"{total / GB:>10.1f}{min_gpus:>10}{f:>11.2e}{gpu_hours(f, gpu, mfu):>9.1f}"
        )
    no_ckpt = activation_bytes(a, seq, batch, checkpoint=False)
    print(
        f"activations without checkpointing would be {no_ckpt / GB:.1f} GB per micro-batch"
    )


def pretraining_example() -> None:
    a, tokens, mfu = PRESETS["8b"], 15e12, 0.40
    n = a.params()
    f = 6 * n * tokens
    h = gpu_hours(f, "h100", mfu)
    chinchilla = 20 * n
    print(f"\n== Pretraining {a.name} on {tokens:.3g} tokens")
    print(f"FLOPs = 6*N*D = 6 * {n:.3e} * {tokens:.3g} = {f:.2e}")
    print(
        f"H100 BF16 @ {mfu:.0%} MFU: {h:,.0f} GPU-hours = {h / 24 / 1024:,.1f} days on 1,024 GPUs"
    )
    print(
        f"Chinchilla-optimal D ~ 20N = {chinchilla:.2e} tokens; this run is "
        f"{tokens / chinchilla:.0f}x past it (inference-optimal overtraining)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", choices=[*PRESETS, "all"], default="all")
    ap.add_argument(
        "--tokens",
        type=float,
        default=30e6,
        help="fine-tune tokens (examples x len x epochs)",
    )
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=1, help="micro-batch per GPU")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--gpu", choices=list(GPUS), default="h100")
    ap.add_argument("--mfu", type=float, default=0.35)
    args = ap.parse_args()

    names = list(PRESETS) if args.preset == "all" else [args.preset]
    for name in names:
        report(
            PRESETS[name],
            args.tokens,
            args.seq,
            args.batch,
            args.rank,
            args.gpu,
            args.mfu,
        )

    # sanity checks on the formulas
    a8 = PRESETS["8b"]
    assert 7.9e9 < a8.params() < 8.1e9, a8.params()
    assert 69e9 < PRESETS["70b"].params() < 71.5e9, PRESETS["70b"].params()
    assert (
        state_bytes(a8, "qlora", 16)[0]
        < state_bytes(a8, "lora", 16)[0]
        < state_bytes(a8, "full", 16)[0]
    )
    if args.preset == "all":
        pretraining_example()


if __name__ == "__main__":
    main()
