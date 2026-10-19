"""Capstone activity glue over the course's Llama, trainer, and checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np


def print_summary(argv: list[str]) -> None:
    """Report the same completed result on fresh execution and idempotent retry."""
    progress = None
    for index, arg in enumerate(argv):
        if arg == "--progress" and index + 1 < len(argv):
            progress = argv[index + 1]
        elif arg.startswith("--progress="):
            progress = arg.split("=", 1)[1]
    if progress is None:
        return
    outputs = json.loads((Path(progress).parent / "DONE.json").read_text())["outputs"]
    artifacts = Path(os.environ.get("TL_ARTIFACTS", "/artifacts"))
    for output in outputs:
        checkpoint = artifacts / output
        config = checkpoint / "config.json"
        if (
            not config.is_file()
            or json.loads(config.read_text()).get("tl_arch") != "llama"
        ):
            continue
        final = json.loads((checkpoint / "final.json").read_text())
        print(
            json.dumps(
                {"steps": final["step"], "final_loss": final["loss"], "ckpt": output}
            ),
            flush=True,
        )


def check_spec(spec: dict) -> list[str]:
    errors = []
    model = spec["model"]
    if model.get("tl_tokenizer") not in ("bytes", "file"):
        errors.append("llama requires tl_tokenizer bytes or file")
    if model.get("tl_tokenizer") == "file" and not spec.get("tokenizer", {}).get(
        "path"
    ):
        errors.append("file tokenization requires tokenizer.path")
    if spec.get("precision", {}).get("dtype", "fp32") not in ("fp32", "bf16"):
        errors.append("capstone precision must be fp32 or bf16")
    schedule = spec["schedule"]
    if schedule.get("name") not in ("constant", "cosine", "wsd"):
        errors.append("schedule must be constant, cosine, or wsd")
    warmup = schedule.get("warmup_steps", 0)
    if not isinstance(warmup, int) or not 0 <= warmup < spec["steps"]:
        errors.append("warmup_steps must be an integer in [0, steps)")
    fraction = schedule.get("decay_frac", 0.2)
    if not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
        errors.append("decay_frac must be in (0, 1]")
    ratio = schedule.get("min_lr_ratio", 0.0)
    if not isinstance(ratio, (int, float)) or not 0 <= ratio <= 1:
        errors.append("min_lr_ratio must be in [0, 1]")
    for key in ("keep_ckpts", "ckpt_every"):
        if not isinstance(spec.get(key, 1), int) or spec.get(key, 1) < 1:
            errors.append(f"{key} must be a positive integer")
    return errors


def train_shards(paths: list[Path], vocab: int) -> list[str]:
    """Resolve token manifests and verify each selected shard before training."""
    from tinyllm.io.activity import SpecError
    from tinyllm.io.tokens import read_tokens_header

    result = []
    for path in paths:
        if path.suffix != ".json":
            result.append(str(path))
            continue
        manifest = json.loads(path.read_text())
        if manifest.get("vocab_size") != vocab:
            raise SpecError(f"{path}: manifest vocabulary differs from the model")
        for entry in manifest["files"]:
            if entry["split"] != "train":
                continue
            shard = (path.parent / entry["name"]).resolve()
            if not shard.is_relative_to(path.parent.resolve()):
                raise SpecError(f"{path}: shard escapes the manifest directory")
            with shard.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != entry["sha256"]:
                raise SpecError(f"{shard}: token shard checksum mismatch")
            if read_tokens_header(str(shard))["n_tokens"] != entry["n_tokens"]:
                raise SpecError(f"{shard}: token count differs from manifest")
            result.append(str(shard))
    if not result:
        raise SpecError("data.train contains no training shards")
    return result


def publish_checkpoint(
    staged: Path, root: Path, tokenizer: Path | None, final: dict, keep: int
) -> Path:
    """Publish weights, tokenizer, and loss as one complete checkpoint."""
    if tokenizer is not None:
        shutil.copyfile(tokenizer, staged / "tokenizer.json")
    (staged / "final.json").write_text(json.dumps(final) + "\n")
    files = []
    for path in sorted(staged.iterdir()):
        if path.name == "MANIFEST.json":
            continue
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            os.fsync(stream.fileno())
        files.append(
            {"name": path.name, "sha256": digest, "bytes": path.stat().st_size}
        )
    with (staged / "MANIFEST.json").open("w") as stream:
        json.dump({"kind": "checkpoint", "files": files}, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(staged, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    root.mkdir(parents=True, exist_ok=True)
    target = root / staged.name
    if target.exists():
        if (target / "MANIFEST.json").read_bytes() != (
            staged / "MANIFEST.json"
        ).read_bytes():
            raise ValueError(f"refusing to replace a different checkpoint at {target}")
        shutil.rmtree(staged)
    else:
        os.rename(staged, target)
    with (root / "LATEST.tmp").open("w") as stream:
        stream.write(target.name + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(root / "LATEST.tmp", root / "LATEST")
    fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    checkpoints = sorted(
        p for p in root.glob("step-*") if p.is_dir() and p.name[5:].isdigit()
    )
    for old in checkpoints[:-keep]:
        if old != target:
            shutil.rmtree(old)
    return target


def train(act, spec: dict) -> list[str]:
    from tinyllm.autograd import functional as F
    from tinyllm.autograd.losses import cross_entropy
    from tinyllm.cli_activity import under
    from tinyllm.cli_train import Failpoints, config_sha256, git_sha, rng_tuple
    from tinyllm.io.activity import Cancelled, SpecError
    from tinyllm.io.checkpoint import load_checkpoint, save_checkpoint
    from tinyllm.io.telemetry import Tracer, should_sample_step
    from tinyllm.io.tokens import TokenStream
    from tinyllm.modern.llama import LlamaConfig, LlamaForCausalLM
    from tinyllm.nn.module import Module
    from tinyllm.num.rng import PCG32
    from tinyllm.optim.adamw import AdamW
    from tinyllm.optim.schedule import cosine_with_warmup, wsd
    from tinyllm.train.precision import train_step_mixed
    from tinyllm.train.recompute import checkpoint_sequential

    identity = act.key or spec["name"]
    run_rel = spec.get("out") or f"runs/{identity}"
    run_dir = under(act, run_rel)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = spec["model"]
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True) + "\n")
    cfg = LlamaConfig.from_hf(str(config_path))
    seq_len = int(spec["batch"]["seq_len"])
    if seq_len > cfg.max_position_embeddings:
        raise SpecError("batch.seq_len exceeds max_position_embeddings")
    tokenizer = None
    if cfg.tokenizer == "file":
        from tinyllm.tok.bpe import BPETokenizer

        tokenizer = under(act, spec["tokenizer"]["path"])
        if BPETokenizer.from_hf_json(str(tokenizer)).vocab_size != cfg.vocab_size:
            raise SpecError("tokenizer vocabulary differs from the model")
    shards = train_shards(
        [under(act, p) for p in spec["data"]["train"]], cfg.vocab_size
    )
    rng = PCG32(int(spec["seed"])).substream("shuffle")
    stream = TokenStream(
        shards, seq_len, spec["batch"]["micro_batch"], rng, cfg.vocab_size
    )
    model = LlamaForCausalLM(cfg, rng=PCG32(spec["seed"]).substream("init"))
    if cfg.tie_word_embeddings:
        model.model.embed_tokens.weight.data *= 0.02
    optimizer = spec["optimizer"]
    lr_max = float(optimizer["lr"])
    opt = AdamW(
        list(model.parameters()),
        lr=lr_max,
        betas=tuple(optimizer.get("betas", [0.9, 0.95])),
        weight_decay=float(optimizer.get("weight_decay", 0.1)),
    )
    steps = int(spec["steps"])
    every = int(spec.get("ckpt_every", 500)) or steps
    signature = config_sha256(spec)
    ckpt_root = run_dir / "ckpt"
    step, tokens_seen, loss = 0, 0, None
    resume = None
    if act.resume:
        path = under(act, act.resume)
        if not path.name.startswith("step-") or not path.name[5:].isdigit():
            raise SpecError("--resume must name a checkpoint step directory")
        resume = load_checkpoint(str(path.parent), int(path.name[5:]))
    elif (ckpt_root / "LATEST").is_file():
        resume = load_checkpoint(str(ckpt_root))
    if resume is not None:
        if resume.extra["config_sha256"] != signature:
            raise SpecError("checkpoint belongs to a different training spec")
        model.load_state_dict(resume.model)
        opt.load_state_dict(resume.opt)
        step, tokens_seen = resume.step, resume.extra["tokens_seen"]
        stream.restore(
            {**resume.extra["data_cursor"], "rng": rng_tuple(resume.rng_state)}
        )
        previous = json.loads((Path(resume.path) / "final.json").read_text())
        if previous["step"] != step:
            raise SpecError("final.json does not match the resumed checkpoint")
        loss = float(previous["loss"])
    (run_dir / "spec.json").write_text(json.dumps(spec, sort_keys=True) + "\n")

    class AtPositions(Module):
        def __init__(self, layer, index):
            super().__init__()
            self.layer, self.index = layer, index

        def forward(self, h):
            return self.layer(h, np.arange(seq_len), None, self.index)

    layers = [AtPositions(layer, i) for i, layer in enumerate(model.model.layers)]

    def loss_fn(m, batch):
        if spec.get("precision", {}).get("recompute", False):
            h = checkpoint_sequential(
                layers, m.model.embed_tokens(batch["x"]), max(1, len(layers) - 1)
            )
            h = m.model.norm(h)
            logits = (
                F.matmul(h, F.transpose(m.model.embed_tokens.weight, 0, 1))
                if m.lm_head is None
                else m.lm_head(h)
            )
        else:
            logits = m(batch["x"])
        return cross_entropy(logits, batch["y"])

    schedule = spec["schedule"]
    warmup = int(schedule.get("warmup_steps", 0))
    decay = min(steps - warmup, max(1, int(steps * schedule.get("decay_frac", 0.2))))
    lr_min = lr_max * schedule.get("min_lr_ratio", 0.0)
    tracer = Tracer.from_env()
    failpoints = Failpoints(os.environ.get("TL_FAILPOINTS", ""))

    def checkpoint():
        final = {
            "step": step,
            "tokens_seen": tokens_seen,
            "loss": loss,
            "loss_hex": float(loss).hex(),
        }
        temporary = run_dir / "final.json.tmp"
        temporary.write_text(json.dumps(final) + "\n")
        os.replace(temporary, run_dir / "final.json")
        cursor = stream.cursor()
        path = save_checkpoint(
            str(run_dir / ".checkpoint-staging"),
            model,
            opt,
            step,
            rng_tuple(cursor["rng"]),
            {
                "tokens_seen": tokens_seen,
                "data_cursor": {"shard": cursor["shard"], "offset": cursor["offset"]},
                "lr": opt.lr,
                "config_sha256": signature,
                "git_sha": git_sha(),
                "config": config,
            },
            keep=1,
        )
        path = str(
            publish_checkpoint(
                Path(path), ckpt_root, tokenizer, final, int(spec.get("keep_ckpts", 3))
            )
        )
        with tracer.span(
            "train.checkpoint", {"tl.train.step": step, "tl.ckpt.path": path}
        ):
            act.checkpoint(step, path)
        return path

    try:
        with tracer.span("train.run", {"tl.run.id": identity, "tl.train.steps": steps}):
            while step < steps:
                if schedule["name"] == "wsd":
                    opt.lr = wsd(
                        step, warmup, steps - warmup - decay, decay, lr_max, lr_min
                    )
                elif schedule["name"] == "cosine":
                    opt.lr = cosine_with_warmup(step, warmup, steps, lr_max, lr_min)
                else:
                    opt.lr = lr_max
                batches = []
                for _ in range(spec["batch"].get("accum", 1)):
                    x, y = stream.next_batch()
                    batches.append({"x": x, "y": y})
                stats = train_step_mixed(
                    model,
                    batches,
                    loss_fn,
                    opt,
                    precision=spec.get("precision", {}).get("dtype", "fp32"),
                    clip=1.0,
                )
                step += 1
                tokens_seen += sum(batch["x"].size for batch in batches)
                loss = float(stats["loss"])
                if not math.isfinite(loss):
                    raise FloatingPointError(f"loss is {loss} at step {step}")
                act.step(step, loss, opt.lr, tokens_seen)
                if should_sample_step(step):
                    with tracer.span(
                        "train.step",
                        {
                            "tl.train.step": step,
                            "tl.train.loss": loss,
                            "tl.train.tokens": tokens_seen,
                        },
                    ):
                        pass
                if step % every == 0 or step == steps or act.cancelled:
                    checkpoint()
                failpoints.inject("train/after-step")
                if act.cancelled:
                    raise Cancelled(f"cancelled at step {step}")
    finally:
        tracer.shutdown()
    from tinyllm.io.checkpoint import step_name

    path = f"{run_rel}/ckpt/{step_name(step)}"
    return [path]
