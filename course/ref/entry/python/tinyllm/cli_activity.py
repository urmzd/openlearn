"""The `tinyllm` CLI, Pass 8 verbs: the subprocess activities of TrainRun
and EvalSuite (dur.09, dur.11; contracts/spec/subprocess-activity.md).

    train --spec <dir>/spec.json --progress <dir>/progress.jsonl [--resume <ckpt dir>]
    eval  --spec <dir>/spec.json --progress <dir>/progress.jsonl

Both run under tinyllm.io.activity.run: the exit code is the verdict (65 a
spec that does not validate, 75 retryable, 130 cancelled), progress events
go to --progress, and DONE.json is the result.

train (formats/train-spec.schema.json). This reference trains the families
the course has a streaming trainer for by Pass 8: tl_arch "bigram" with the
byte tokenizer, on .bin token streams (data.07), with AdamW, through L0.6's
TokenStream and atomic checkpoints. The run directory is spec.out (default
runs/<name>; ctl's train sets runs/<workflow id>) under TL_ARTIFACTS; it
resumes from --resume (the checkpoint the runner heartbeated) or else from
<run>/ckpt/LATEST, so a later segment of a TrainRun continues where the
previous one stopped. <run>/final.json, the loss of the newest step, is
written before each checkpoint, so an attempt that resumes from the final
checkpoint reports the same loss. Two runs of one spec end with the same
loss, bit for bit, however often they were interrupted.
C1 adds tl_arch llama.

eval (formats/eval-spec.schema.json). Built-in suite "ppl": each subject is a
checkpoint step directory or a model directory of a bigram; the cases are
512-token chunks of the val streams of the run that wrote the checkpoint
(<run>/spec.json, data.val). Writes evals/ppl/<run_id>/results.jsonl and
summary.json (formats/eval-result.schema.json); run_id is the spec's name
plus the activity's key.

Entry-point territory (D16): this file is yours. It is glue over L0.1 to L0.6,
M06.3 (PCG32), M10.3 (AdamW), and dur.09 (activity, telemetry).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np

VOCAB = 256
CONFIG = {"tl_arch": "bigram", "tl_tokenizer": "bytes", "vocab_size": VOCAB, "tl_format": 1}
CHUNK = 512


def claims(argv: list[str]) -> bool:
    return bool(argv) and argv[0] in ("train", "eval") and any(
        a == "--spec" or a.startswith("--spec=") for a in argv
    )


def intercept(argv: list[str]) -> int | None:
    if not claims(argv):
        return None
    from tinyllm.io.activity import run

    return run(train_main if argv[0] == "train" else eval_main, argv[1:])


# -- spec checks (the subset of the schemas this reference relies on) -----------


def check_train_spec(spec: dict) -> list[str]:
    errs = []
    for k in ("name", "model", "data", "optimizer", "schedule", "batch", "steps", "seed"):
        if k not in spec:
            errs.append(f"missing {k}")
    if errs:
        return errs
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", str(spec["name"])):
        errs.append(f"name {spec['name']!r} does not match ^[a-z0-9][a-z0-9._-]*$")
    if not isinstance(spec["steps"], int) or spec["steps"] < 1:
        errs.append("steps must be an integer >= 1")
    if not isinstance(spec["seed"], int) or spec["seed"] < 0:
        errs.append("seed must be an integer >= 0")
    model = spec["model"]
    if model.get("tl_arch") != "bigram" or model.get("tl_tokenizer", "bytes") != "bytes":
        errs.append("this trainer runs tl_arch bigram with tl_tokenizer bytes (C1 adds llama)")
    if not spec["data"].get("train"):
        errs.append("data.train needs at least one .bin file")
    b = spec["batch"]
    if not all(isinstance(b.get(k, 1), int) and b.get(k, 1) >= 1 for k in ("micro_batch", "accum", "seq_len")) or "micro_batch" not in b or "seq_len" not in b:
        errs.append("batch needs micro_batch >= 1, seq_len >= 1, and accum >= 1 when given")
    if spec["optimizer"].get("name") != "adamw" or not spec["optimizer"].get("lr", 0) > 0:
        errs.append("optimizer: this trainer runs adamw with lr > 0")
    return errs


def check_eval_spec(spec: dict) -> list[str]:
    errs = [f"missing {k}" for k in ("name", "suites", "subjects", "seed") if k not in spec]
    if errs:
        return errs
    for s in spec["suites"]:
        if s != "ppl":
            errs.append(f"suite {s!r}: this reference runs the built-in ppl suite")
    for subj in spec["subjects"]:
        if not subj.get("id") or not subj.get("model"):
            errs.append(f"subject {subj}: needs id and model")
    return errs


def under(act, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else Path(act.artifacts) / p


# -- train ------------------------------------------------------------------------------


def train_main(act) -> list[str]:
    from tinyllm.autograd import functional as F
    from tinyllm.autograd.losses import cross_entropy
    from tinyllm.cli_train import Failpoints, config_sha256, git_sha, rng_tuple
    from tinyllm.io.activity import Cancelled, SpecError
    from tinyllm.io.checkpoint import load_checkpoint, save_checkpoint, step_name
    from tinyllm.io.telemetry import Tracer, should_sample_step
    from tinyllm.io.tokens import TokenStream
    from tinyllm.lm.bigram import BigramLogits
    from tinyllm.num.rng import PCG32
    from tinyllm.optim.adamw import AdamW

    spec = act.load_spec(check_train_spec)
    run_rel = spec.get("out") or f"runs/{spec['name']}"
    run_dir = under(act, run_rel)
    ckpt = run_dir / "ckpt"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.json").write_text(json.dumps(spec, sort_keys=True) + "\n")
    shards = [str(under(act, p)) for p in spec["data"]["train"]]
    for s in shards:
        if not os.path.isfile(s):
            raise SpecError(f"data.train: {s} does not exist")
    steps = int(spec["steps"])
    every = int(spec.get("ckpt_every", 500)) or steps
    lr = float(spec["optimizer"]["lr"])
    fp = Failpoints(os.environ.get("TL_FAILPOINTS", ""))

    model = BigramLogits(VOCAB)
    opt = AdamW(list(model.parameters()), lr=lr, weight_decay=float(spec["optimizer"].get("weight_decay", 0.0)))
    rng = PCG32(int(spec["seed"]))
    # One optimizer step sees micro_batch * accum windows; this trainer takes
    # them as one batch (the bigram needs no accumulation to fit in memory).
    rows = int(spec["batch"]["micro_batch"]) * int(spec["batch"].get("accum", 1))
    stream = TokenStream(shards, int(spec["batch"]["seq_len"]), rows, rng, vocab_size=VOCAB)
    step, tokens_seen = 0, 0
    resume = None
    if act.resume:
        m = re.search(r"step-(\d+)$", act.resume.rstrip("/"))
        if not m:
            raise SpecError(f"--resume {act.resume} is not a checkpoint step directory")
        resume = load_checkpoint(str(under(act, act.resume).parent), int(m.group(1)))
    elif (ckpt / "LATEST").is_file():
        resume = load_checkpoint(str(ckpt))
    if resume is not None:
        model.load_state_dict(resume.model)
        opt.load_state_dict(resume.opt)
        step, tokens_seen = int(resume.step), int(resume.extra["tokens_seen"])
        cur = resume.extra["data_cursor"]
        stream.restore({"shard": cur["shard"], "offset": cur["offset"], "rng": rng_tuple(resume.rng_state)})

    cfg_sha = config_sha256({**CONFIG, **spec["model"], "seed": spec["seed"], "lr": lr})
    tracer = Tracer.from_env()
    loss_v = float("nan")

    def record_loss() -> None:
        # final.json holds the loss of the newest step trained. It is
        # published (tmp, rename) before that step's checkpoint, so a retry
        # that resumes from the final checkpoint and trains no step still
        # reports the loss the uninterrupted run reported.
        final = {"step": step, "tokens_seen": tokens_seen, "loss": loss_v, "loss_hex": float(loss_v).hex()}
        tmp = run_dir / "final.json.tmp"
        tmp.write_text(json.dumps(final) + "\n")
        os.replace(tmp, run_dir / "final.json")

    def checkpoint() -> str:
        record_loss()
        c = stream.cursor()
        path = save_checkpoint(
            str(ckpt), model, opt, step, rng_tuple(c["rng"]),
            {"tokens_seen": tokens_seen, "data_cursor": {"shard": int(c["shard"]), "offset": int(c["offset"])},
             "lr": lr, "config_sha256": cfg_sha, "git_sha": git_sha(), "config": CONFIG},
            keep=int(spec.get("keep_ckpts", 3)),
        )
        with tracer.span("train.checkpoint", {"tl.train.step": step, "tl.ckpt.path": f"{run_rel}/ckpt/{step_name(step)}"}):
            act.checkpoint(step, path)
        return path

    with tracer.span("train.run", {"tl.run.id": spec["name"], "tl.train.steps": steps}):
        while step < steps:
            xb, yb = stream.next_batch()
            opt.zero_grad()
            n = xb.size
            loss = cross_entropy(F.reshape(model(np.asarray(xb).reshape(n)), (n, VOCAB)), np.asarray(yb).reshape(n))
            loss.backward()
            opt.step()
            step += 1
            tokens_seen += n
            loss_v = float(loss.data)
            if not math.isfinite(loss_v):
                raise FloatingPointError(f"loss is {loss_v} at step {step}")
            act.step(step, loss_v, lr, tokens_seen)
            if should_sample_step(step):
                with tracer.span("train.step", {"tl.train.step": step, "tl.train.loss": loss_v, "tl.train.tokens": tokens_seen}):
                    pass
                tracer.gauge("tl.train.loss", loss_v, {"tl.workflow.id": act.key.split("/")[0] if act.key else spec["name"]})
            if step % every == 0 or step == steps:
                checkpoint()
            fp.inject("train/after-step")
            if act.cancelled:
                if step % every != 0 and step != steps:
                    checkpoint()
                tracer.shutdown()
                raise Cancelled(f"cancelled at step {step}")
    tracer.shutdown()
    if math.isfinite(loss_v):
        record_loss()
    else:
        # Resumed at or past `steps`: no step ran in this attempt, so the
        # loss is the one recorded with that checkpoint.
        try:
            prev = json.loads((run_dir / "final.json").read_text())
        except (OSError, ValueError):
            prev = {}
        loss = prev.get("loss")
        if prev.get("step") != step or not isinstance(loss, (int, float)) or not math.isfinite(loss):
            raise SpecError(
                f"resumed at step {step} of {steps} with no step to train, and {run_rel}/final.json "
                f"records no loss for step {step} (another run's checkpoints? give the spec its own out)"
            )
    return [f"{run_rel}/ckpt/{step_name(step)}"]


# -- eval -------------------------------------------------------------------------------


def bigram_weight_of(path: Path) -> np.ndarray:
    from tinyllm.io.safetensors import load_safetensors

    tensors, _ = load_safetensors(str(path / "model.safetensors"))
    for name in ("bigram.weight", "weight"):
        if name in tensors:
            return np.asarray(tensors[name], dtype=np.float64)
    raise ValueError(f"{path}: no bigram weight")


def eval_main(act) -> list[str]:
    from tinyllm.io.activity import SpecError
    from tinyllm.io.tokens import open_tokens

    spec = act.load_spec(check_eval_spec)
    run_id = spec["name"] + ("-" + act.key.replace("/", "-") if act.key else "")
    out_dir = Path(act.artifacts) / "evals" / "ppl" / run_id
    rows, metrics, subjects = [], {}, []
    for subj in spec["subjects"]:
        mdir = under(act, subj["model"])
        run_spec = mdir.parent.parent / "spec.json"
        if not run_spec.is_file():
            raise SpecError(f"{subj['model']}: no {run_spec} names the val data of this checkpoint")
        val = json.loads(run_spec.read_text())["data"].get("val") or []
        if not val:
            raise SpecError(f"{run_spec}: data.val is empty")
        w = bigram_weight_of(mdir)
        logp = w - np.log(np.exp(w - w.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True)) - w.max(axis=1, keepdims=True)
        nlls = []
        for vf in val:
            ids = np.asarray(open_tokens(str(under(act, vf))), dtype=np.int64)
            for i in range(0, len(ids) - 1, CHUNK):
                x, y = ids[i : i + CHUNK], ids[i + 1 : i + CHUNK + 1]
                x = x[: len(y)]
                if len(y) == 0:
                    continue
                nll = float(-logp[x, y].mean())
                case = f"{Path(vf).name}:{i}"
                nlls.append(nll)
                rows.append({
                    "suite": "ppl", "case_id": case, "subject": subj["id"],
                    "input_sha": hashlib.sha256(ids[i : i + CHUNK + 1].tobytes()).hexdigest(),
                    "output": None, "scores": {"nll": nll}, "latency_ms": 0.0, "ttft_ms": None,
                    "tokens": int(len(y)), "trace_id": None,
                })
            act.check_cancel()
        arr = np.asarray(nlls)
        g = np.random.default_rng(int(spec["seed"]))
        boots = arr[g.integers(0, len(arr), size=(1000, len(arr)))].mean(axis=1)
        metrics[subj["id"]] = {"nll": {"mean": float(arr.mean()), "ci_low": float(np.quantile(boots, 0.025)),
                                       "ci_high": float(np.quantile(boots, 0.975)), "n": int(len(arr)), "errored": 0}}
        subjects.append(subj["id"])
        act.metric(f"ppl.{subj['id']}.nll", float(arr.mean()))
    summary = {"suite": "ppl", "run_id": run_id, "subjects": subjects, "metrics": metrics, "n_boot": 1000, "seed": int(spec["seed"])}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("results.jsonl", "".join(json.dumps(r) + "\n" for r in rows)), ("summary.json", json.dumps(summary) + "\n")):
        tmp = out_dir / (name + ".tmp")
        tmp.write_text(data)
        os.replace(tmp, out_dir / name)
    rel = f"evals/ppl/{run_id}"
    return [f"{rel}/summary.json", f"{rel}/results.jsonl"]
