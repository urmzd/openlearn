"""The Pass 9 training verb (course/milestones/MS-L11.toml fixes its flags
and final line).

    train llama --cfg <config.json> --data <text file> [--steps S] [--seq-len T]
                [--micro-batch B] [--accum K] [--bf16] [--checkpoint-activations]
                [--lr LR] [--seed S] [--baseline <run.json>] [--out <run.json>]

One run of a byte-level Llama (config.json with tl_tokenizer "bytes") on a
text file: K micro-batches of B windows of T + 1 bytes per optimizer step,
the windows drawn with PCG32(seed).substream("shuffle") (spec/pcg32.md), AdamW with gradient
clipping at 1.0, under emulated bf16 autocast with --bf16, with the decoder
layers checkpointed with --checkpoint-activations (a budget of all but one layer's saved inputs, M08.4's schedule). Python's tracemalloc measures the peak memory of
the training loop. Final line:

    {"loss": fp32 loss on 16 held-out windows (the file's last tenth),
     "train_loss": mean of the last 10 steps, "first_loss", "steps",
     "tokens", "precision", "micro_batch", "accum", "recompute",
     "peak_mem": bytes, "params",
     and with --baseline: "loss_rel_diff": |loss - base| / base,
     "peak_mem_ratio": peak_mem / base peak_mem}

--out also writes that line to a file (the next run's --baseline).

Entry-point territory (D16): this file is yours. It is glue over L7.9
(LlamaConfig, LlamaForCausalLM), L11.1 (train_step_mixed,
checkpoint_sequential), M10.3 (AdamW), L0.3 (cross_entropy), L0.4 (Module),
and M06.3 (PCG32).
"""

from __future__ import annotations

import argparse
import json
import sys
import tracemalloc
from pathlib import Path

import numpy as np


class UsageError(Exception):
    exit_code = 2


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def parser() -> argparse.ArgumentParser:
    ap = _Parser(prog="tinyllm train llama")
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--micro-batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--checkpoint-activations", action="store_true")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline")
    ap.add_argument("--out")
    return ap


def _run(a: argparse.Namespace) -> dict:
    from tinyllm.autograd.losses import cross_entropy
    from tinyllm.autograd.mode import no_grad
    from tinyllm.modern.llama import LlamaConfig, LlamaForCausalLM
    from tinyllm.nn.module import Module
    from tinyllm.num.rng import PCG32
    from tinyllm.optim.adamw import AdamW
    from tinyllm.train.precision import train_step_mixed
    from tinyllm.train.recompute import checkpoint_sequential

    for k in ("steps", "seq_len", "micro_batch", "accum"):
        if getattr(a, k) < 1:
            raise UsageError(f"--{k.replace('_', '-')} must be at least 1")
    cfg_path = Path(a.cfg)
    if json.loads(cfg_path.read_text()).get("tl_tokenizer") != "bytes":
        raise UsageError("train llama reads text as bytes: the config needs tl_tokenizer \"bytes\"")
    cfg = LlamaConfig.from_hf(str(cfg_path))
    text = np.frombuffer(Path(a.data).read_bytes(), dtype=np.uint8).astype(np.int64)
    T = a.seq_len
    cut = len(text) * 9 // 10  # the last tenth is held out
    data, held = text[:cut], text[cut:]
    if len(held) <= T + 1:
        raise UsageError(f"--data holds {len(text)} bytes; its last tenth must hold a window of {T + 1}")
    model = LlamaForCausalLM(cfg, rng=PCG32(a.seed).substream("init"))
    if cfg.tie_word_embeddings:
        # L7.9 draws embeddings with std 1 (HF's default for an untied
        # table); tied to the output, that makes the first logits about
        # sqrt(d) wide and the first loss near 100. GPT-2's 0.02 instead.
        model.model.embed_tokens.weight.data *= 0.02
    params = list(model.parameters())
    opt = AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    rng = PCG32(a.seed).substream("shuffle")

    class AtPositions(Module):
        """One decoder layer as a one-argument module, for checkpoint_sequential."""

        def __init__(self, layer: Module, i: int) -> None:
            super().__init__()
            self.layer, self.i = layer, i

        def forward(self, h):
            return self.layer(h, np.arange(T), None, self.i)

    wrapped = [AtPositions(layer, i) for i, layer in enumerate(model.model.layers)]

    def loss_fn(m, mb):
        if not a.checkpoint_activations:
            logits = m(mb["x"])
        else:
            from tinyllm.autograd import functional as F

            h = checkpoint_sequential(wrapped, m.model.embed_tokens(mb["x"]), max(1, len(wrapped) - 1))
            h = m.model.norm(h)
            logits = F.matmul(h, F.transpose(m.model.embed_tokens.weight, 0, 1)) if m.lm_head is None else m.lm_head(h)
        return cross_entropy(logits, mb["y"])

    precision = "bf16" if a.bf16 else "fp32"
    losses = []
    tracemalloc.start()
    for _ in range(a.steps):
        mbs = []
        for _ in range(a.accum):
            starts = [rng.below(len(data) - T - 1) for _ in range(a.micro_batch)]
            w = np.stack([data[s : s + T + 1] for s in starts])
            mbs.append({"x": w[:, :-1], "y": w[:, 1:]})
        out = train_step_mixed(model, mbs, loss_fn, opt, precision=precision, clip=1.0)
        losses.append(float(out["loss"]))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    erng = PCG32(a.seed).substream("sample")
    ev = np.stack([held[s : s + T + 1] for s in (erng.below(len(held) - T - 1) for _ in range(16))])
    with no_grad():
        eval_loss = float(cross_entropy(model(ev[:, :-1]), ev[:, 1:]).data)
    res = {
        "loss": eval_loss,
        "train_loss": float(np.mean(losses[-10:])),
        "first_loss": losses[0],
        "steps": a.steps,
        "tokens": a.steps * a.accum * a.micro_batch * T,
        "precision": precision,
        "micro_batch": a.micro_batch,
        "accum": a.accum,
        "recompute": bool(a.checkpoint_activations),
        "peak_mem": int(peak),
        "params": int(sum(p.data.size for p in params)),
    }
    if a.baseline:
        base = json.loads(Path(a.baseline).read_text())
        res["loss_rel_diff"] = abs(res["loss"] - base["loss"]) / base["loss"]
        res["peak_mem_ratio"] = res["peak_mem"] / base["peak_mem"]
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res) + "\n")
    return res


def intercept(argv: list[str]) -> int | None:
    if len(argv) < 2 or argv[0] != "train" or argv[1] != "llama":
        return None
    try:
        result = _run(parser().parse_args(argv[2:]))
    except Exception as e:  # noqa: BLE001 - every failure is reported, not traced
        code = getattr(e, "exit_code", 1)
        why = str(e) if hasattr(e, "exit_code") else f"{type(e).__name__}: {e}"
        print(f"tinyllm train llama: {why}", file=sys.stderr)
        return code
    print(json.dumps(result), flush=True)
    return 0
