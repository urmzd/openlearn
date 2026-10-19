"""The Pass 5 objective verbs (course/milestones/MS-L6.toml fixes their flags
and final lines).

    train gpt      --cfg <config.json> --data <train.bin> --val <val.bin> --out <dir> [--steps S] [--seed S]
    train bert     --cfg <config.json> --data <train.bin> --out <dir> [--steps S] [--seed S]
    train electra  --cfg <config.json> --data <train.bin> --out <dir> [--steps S] [--seed S]
    finetune classify --base <dir> --data <split.tsv> --lora r=8[,alpha=16] --out <dir> [--steps S] [--seed S]
    eval ppl       --model <gpt dir> --data <tokens.bin> [--ctx C] [--stride S]
    zoo add        --manifest <zoo.json> --id <id> --dir <dir> --task <name> --data <file> [--set k=v ...]
    eval --suite zoo --manifest <zoo.json> [--out <report.json>] [--seed S]

`intercept(argv)` runs these before every other glue file and returns an
exit code when argv is one of them, None otherwise.

The byte tokenizer (D32) with four control bytes as specials: 0 [PAD],
1 [CLS], 2 [SEP], 3 [MASK]. Every corpus here is printable ASCII, so they
never collide with text.

Entry-point territory (D16): this file is yours. It is glue over L6.1 (GPT,
fit_gpt, save_gpt, load_gpt), L6.2 (BertConfig, BertForMLM, mlm_mask,
save_bert, load_bert), L6.3 (ELECTRA, electra_step, save_electra,
load_electra), L6.5 (SequenceClassifier, lora_classifier, train_classifier,
predict, save_classifier), L6.6 (merge_lora, trainable_fraction), L6.7
(eval_ppl, run_zoo, write_report, encode_batch), L0.6 (TokenStream,
open_tokens), M10.3 (AdamW), M07.4 (wilson_interval), and M06.3 (PCG32).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PAD, CLS, SEP, MASK = 0, 1, 2, 3
SPECIALS = [PAD, CLS, SEP, MASK]


class UsageError(Exception):
    exit_code = 2


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def _cfg(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _rng(seed: int, purpose: str):
    from tinyllm.num.rng import PCG32

    return PCG32(seed).substream(purpose)


def _bert_cfg(c: dict, layers: int | None = None, d_ff: int | None = None):
    from tinyllm.obj.bert import BertConfig

    return BertConfig(
        vocab=256,
        max_len=int(c["max_len"]),
        d_model=int(c["d_model"]),
        n_heads=int(c["n_heads"]),
        n_layers=int(layers if layers is not None else c["n_layers"]),
        d_ff=int(d_ff if d_ff is not None else c["d_ff"]),
        dropout=float(c.get("dropout", 0.0)),
    )


def _windows(ids: np.ndarray, batch: int, width: int, rng) -> np.ndarray:
    """batch rows of [CLS] + width bytes at random offsets + [SEP]."""
    out = np.empty((batch, width + 2), dtype=np.int64)
    for b in range(batch):
        o = rng.below(len(ids) - width)
        out[b] = [CLS] + ids[o : o + width].tolist() + [SEP]
    return out


def _tail_mean(xs: list[float], k: int = 20) -> float:
    return float(np.mean(xs[-k:]))


def _params(model) -> int:
    return int(sum(p.data.size for p in model.parameters()))


# -- train ---------------------------------------------------------------------------------


def cmd_train_gpt(a) -> dict:
    from tinyllm.eval.lm import eval_ppl
    from tinyllm.io.tokens import TokenStream, open_tokens
    from tinyllm.obj.gpt import GPT, GPTConfig, fit_gpt, save_gpt

    c = _cfg(a.cfg)
    steps = a.steps or int(c["steps"])
    cfg = GPTConfig(
        vocab=256,
        n_ctx=int(c["n_ctx"]),
        d_model=int(c["d_model"]),
        n_heads=int(c["n_heads"]),
        n_layers=int(c["n_layers"]),
        d_ff=int(c["d_ff"]),
        dropout=float(c.get("dropout", 0.0)),
    )
    model = GPT(cfg, _rng(a.seed, "init"))
    stream = TokenStream(
        [a.data], cfg.n_ctx, int(c["batch"]), _rng(a.seed, "shuffle"), vocab_size=256
    )
    losses = fit_gpt(
        model, stream, steps, float(c["lr"]), warmup=int(c.get("warmup", 0))
    )
    model.eval()
    save_gpt(model, a.out, tokenizer="bytes")
    val = np.asarray(open_tokens(a.val), dtype=np.int64)
    r = eval_ppl(model, val, cfg.n_ctx, cfg.n_ctx // 2, n_bytes=val.size - 1)
    return {
        "out": a.out,
        "arch": "gpt",
        "steps": steps,
        "train_loss": _tail_mean(losses),
        "val_loss": r["nll_mean"],
        "val_bpb": r["bpb"],
        "params": _params(model),
    }


def _pretrain(a, model, step_loss) -> list[float]:
    from tinyllm.io.tokens import open_tokens
    from tinyllm.optim.adamw import AdamW

    c = _cfg(a.cfg)
    steps = a.steps or int(c["steps"])
    ids = np.asarray(open_tokens(a.data), dtype=np.int64)
    batch, width = int(c["batch"]), int(c["window"])
    draw, sample = _rng(a.seed, "shuffle"), _rng(a.seed, "sample")
    opt = AdamW(
        list(model.parameters()),
        lr=float(c["lr"]),
        weight_decay=float(c.get("weight_decay", 0.01)),
    )
    model.train()
    losses = []
    for _ in range(steps):
        x = _windows(ids, batch, width, draw)
        opt.zero_grad()
        loss = step_loss(x, sample)
        loss.backward()
        opt.step()
        losses.append(float(loss.data))
    model.eval()
    return losses


def cmd_train_bert(a) -> dict:
    from tinyllm.obj.bert import BertForMLM, mlm_mask, save_bert

    c = _cfg(a.cfg)
    model = BertForMLM(_bert_cfg(c), _rng(a.seed, "init"))
    p = float(c.get("p", 0.15))

    def step_loss(x, rng):
        special = np.isin(x, SPECIALS)
        inputs, labels = mlm_mask(x, special, MASK, 256, p, rng)
        return model(inputs, None, None, labels)[1]

    losses = _pretrain(a, model, step_loss)
    save_bert(model, a.out, tokenizer="bytes")
    return {
        "out": a.out,
        "arch": "bert",
        "steps": len(losses),
        "mlm_loss": _tail_mean(losses),
        "params": _params(model),
    }


def cmd_train_electra(a) -> dict:
    from tinyllm.obj.electra import ELECTRA, electra_step, save_electra

    c = _cfg(a.cfg)
    gen = _bert_cfg(c, layers=int(c["gen_layers"]), d_ff=int(c["gen_d_ff"]))
    model = ELECTRA(gen, _bert_cfg(c), _rng(a.seed, "init"))
    p, lam = float(c.get("p", 0.15)), float(c.get("lambda", 50.0))
    last = {}

    def step_loss(x, rng):
        out = electra_step(
            model.generator,
            model.discriminator,
            x,
            rng,
            lam,
            mask_id=MASK,
            special_mask=np.isin(x, SPECIALS),
            p=p,
        )
        last["disc"] = float(out["disc_loss"].data)
        return out["loss"]

    losses = _pretrain(a, model, step_loss)
    save_electra(model, a.out, mask_id=MASK, special_ids=SPECIALS, tokenizer="bytes")
    return {
        "out": a.out,
        "arch": "electra",
        "steps": len(losses),
        "loss": _tail_mean(losses),
        "disc_loss": last.get("disc", float("nan")),
        "params": _params(model),
    }


# -- finetune -------------------------------------------------------------------------------


def _lora_spec(s: str) -> dict:
    out = {"r": 8, "alpha": 16.0}
    for part in s.split(","):
        k, _, v = part.partition("=")
        if k not in out or not v:
            raise UsageError(f"--lora wants r=N[,alpha=X], got {s!r}")
        out[k] = int(v) if k == "r" else float(v)
    return out


def _sentences(path: str, split: str) -> tuple[list[str], np.ndarray]:
    rows = [
        line.split("\t")
        for line in Path(path).read_text().splitlines()[1:]
        if line.strip()
    ]
    keep = [r for r in rows if r[0] == split]
    if not keep:
        raise ValueError(f"{path}: no {split!r} rows")
    return [r[1] for r in keep], np.array([int(r[2]) for r in keep], dtype=np.int64)


def cmd_finetune(a) -> dict:
    from tinyllm.eval.zoo import encode_batch, read_config
    from tinyllm.obj.bert import load_bert
    from tinyllm.obj.electra import load_electra
    from tinyllm.obj.heads import (
        SequenceClassifier,
        lora_classifier,
        predict,
        save_classifier,
        train_classifier,
    )
    from tinyllm.obj.lora import merge_lora, trainable_fraction
    from tinyllm.prob.stats import wilson_interval

    lora = _lora_spec(a.lora)
    arch = read_config(a.base).get("tl_arch")
    if arch == "bert":
        backbone = load_bert(a.base).bert
    elif arch == "electra":
        backbone = load_electra(a.base)[0].discriminator.electra
    else:
        raise ValueError(
            f"{a.base}: finetune classify takes a bert or electra directory, got {arch!r}"
        )
    d = backbone.cfg.d_model
    clf = SequenceClassifier(backbone, d, 2, pool=a.pool, rng=_rng(a.seed, "init"))
    lora_classifier(clf, lora["r"], lora["alpha"], rng=_rng(a.seed + 1, "init"))
    frac = trainable_fraction(clf)
    texts, labels = _sentences(a.data, "train")
    ids, real = encode_batch(texts, backbone.cfg.max_len, CLS, SEP, PAD)
    losses = train_classifier(
        clf,
        ids,
        labels,
        a.steps,
        a.batch,
        a.lr,
        _rng(a.seed, "shuffle"),
        attn_mask=real,
    )
    merge_lora(clf)
    save_classifier(
        clf, a.out, arch, tokenizer="bytes", labels=["negative", "positive"]
    )
    vt, vl = _sentences(a.data, "val")
    vids, vreal = encode_batch(vt, backbone.cfg.max_len, CLS, SEP, PAD)
    k = int(np.sum(predict(clf, vids, vreal) == vl))
    lo, hi = wilson_interval(k, len(vl))
    return {
        "out": a.out,
        "arch": arch,
        "steps": a.steps,
        "train_loss": _tail_mean(losses),
        "acc": k / len(vl),
        "acc_lo": lo,
        "acc_hi": hi,
        "n": len(vl),
        "trainable_frac": frac,
    }


# -- eval and the zoo ------------------------------------------------------------------------


def cmd_eval_ppl(a) -> dict:
    from tinyllm.eval.lm import eval_ppl
    from tinyllm.io.tokens import open_tokens
    from tinyllm.obj.gpt import load_gpt

    model = load_gpt(a.model)
    ids = np.asarray(open_tokens(a.data), dtype=np.int64)
    ctx = a.ctx or model.cfg.n_ctx
    r = eval_ppl(model, ids, ctx, a.stride or ctx // 2, n_bytes=ids.size - 1)
    return {
        "model": a.model,
        "arch": "gpt",
        "ctx": ctx,
        "stride": a.stride or ctx // 2,
        "tokens": int(r["n_tokens"]),
        "ppl": r["ppl"],
        "ppl_lo": r["ppl_lo"],
        "ppl_hi": r["ppl_hi"],
        "bpb": r["bpb"],
        "bpb_lo": r["bpb_lo"],
        "bpb_hi": r["bpb_hi"],
        "nll_mean": r["nll_mean"],
    }


def cmd_zoo_add(a) -> dict:
    m = Path(a.manifest)
    doc = json.loads(m.read_text()) if m.is_file() else {"suite": "zoo", "models": []}
    entry = {
        "id": a.id,
        "dir": str(Path(a.dir).resolve()),
        "task": a.task,
        "data": str(Path(a.data).resolve()),
    }
    for kv in a.set:
        k, _, v = kv.partition("=")
        if not v:
            raise UsageError(f"--set wants key=value, got {kv!r}")
        try:
            entry[k] = json.loads(v)
        except json.JSONDecodeError:
            entry[k] = v
    doc["models"] = [e for e in doc["models"] if e["id"] != a.id] + [entry]
    m.parent.mkdir(parents=True, exist_ok=True)
    m.write_text(json.dumps(doc, indent=1) + "\n")
    return {"manifest": str(m), "models": len(doc["models"]), "added": a.id}


def cmd_eval_zoo(a) -> dict:
    from tinyllm.eval.zoo import run_zoo, write_report
    from tinyllm.num.rng import PCG32

    report = run_zoo(a.manifest, PCG32(a.seed).substream("sample"), seed=a.seed)
    out = a.out or str(Path(a.manifest).with_name("zoo-report.json"))
    write_report(report, out)
    for r in report["rows"]:
        v = r["value"]
        shown = (
            f"{v:.4f} [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]"
            if r["status"] == "ok"
            else r.get("reason", "")
        )
        print(
            f"{r['model']:<14} {r.get('tl_arch', '-'):<9} {r['task']:<10} {r['metric']:<9} {r['status']:<7} {shown}",
            flush=True,
        )
    st = [r["status"] for r in report["rows"]]
    return {
        "suite": report["suite"],
        "report": out,
        "rows": len(st),
        "ok": st.count("ok"),
        "errors": st.count("error"),
        "skipped": st.count("skipped"),
    }


def parser() -> argparse.ArgumentParser:
    ap = _Parser(prog="tinyllm")
    sub = ap.add_subparsers(dest="verb", required=True, parser_class=_Parser)
    for what, fn in (
        ("gpt", cmd_train_gpt),
        ("bert", cmd_train_bert),
        ("electra", cmd_train_electra),
    ):
        t = sub.add_parser(f"train-{what}")
        t.add_argument("--cfg", required=True)
        t.add_argument("--data", required=True)
        t.add_argument("--out", required=True)
        t.add_argument("--steps", type=int)
        t.add_argument("--seed", type=int, default=0)
        if what == "gpt":
            t.add_argument("--val", required=True)
        t.set_defaults(fn=fn)
    f = sub.add_parser("finetune-classify")
    f.add_argument("--base", required=True)
    f.add_argument("--data", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--lora", default="r=8,alpha=16")
    f.add_argument("--pool", choices=["cls", "mean", "last"], default="mean")
    f.add_argument("--steps", type=int, default=300)
    f.add_argument("--batch", type=int, default=16)
    f.add_argument("--lr", type=float, default=3e-3)
    f.add_argument("--seed", type=int, default=0)
    f.set_defaults(fn=cmd_finetune)
    e = sub.add_parser("eval-ppl")
    e.add_argument("--model", required=True)
    e.add_argument("--data", required=True)
    e.add_argument("--ctx", type=int)
    e.add_argument("--stride", type=int)
    e.set_defaults(fn=cmd_eval_ppl)
    z = sub.add_parser("zoo-add")
    z.add_argument("--manifest", required=True)
    z.add_argument("--id", required=True)
    z.add_argument("--dir", required=True)
    z.add_argument("--task", required=True)
    z.add_argument("--data", required=True)
    z.add_argument("--set", action="append", default=[])
    z.set_defaults(fn=cmd_zoo_add)
    s = sub.add_parser("eval-zoo")
    s.add_argument("--suite", required=True, choices=["zoo"])
    s.add_argument("--manifest", required=True)
    s.add_argument("--out")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_eval_zoo)
    return ap


def _mine(argv: list[str]) -> list[str] | None:
    if len(argv) >= 2 and argv[0] == "train" and argv[1] in ("gpt", "bert", "electra"):
        return [f"train-{argv[1]}"] + argv[2:]
    if len(argv) >= 2 and argv[0] == "finetune" and argv[1] == "classify":
        return ["finetune-classify"] + argv[2:]
    if len(argv) >= 2 and argv[0] == "eval" and argv[1] == "ppl":
        return ["eval-ppl"] + argv[2:]
    if len(argv) >= 2 and argv[0] == "zoo" and argv[1] == "add":
        return ["zoo-add"] + argv[2:]
    if argv and argv[0] == "eval" and "--suite" in argv:
        return ["eval-zoo"] + argv[1:]
    return None


def intercept(argv: list[str]) -> int | None:
    mine = _mine(argv)
    if mine is None:
        return None
    try:
        a = parser().parse_args(mine)
        result = a.fn(a)
    except Exception as e:  # noqa: BLE001 - every failure is reported, not traced
        code = getattr(e, "exit_code", 1)
        why = str(e) if hasattr(e, "exit_code") else f"{type(e).__name__}: {e}"
        print(f"tinyllm {argv[0]}: {why}", file=sys.stderr)
        return code
    print(json.dumps(result), flush=True)
    return 0
