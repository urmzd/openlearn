"""The Pass 3 language-model verbs (course/milestones/MS-L2.toml fixes their
flags and final lines).

    lm train ngram --n N --train <tokens.bin> --out <dir> [--discount modified|D]
    lm train nplm  --train <tokens.bin> --out <dir> --context C --d-emb M --hidden H
                   [--no-direct] --steps S --batch B --lr X [--momentum MU]
                   [--weight-decay WD] [--clip G] [--seed S]
    eval --model <dir> --data <tokens.bin>
    generate --model <dir> ... [--out <file>]     (an nplm directory; bigram stays in __main__)

Entry-point territory (D16): this file is yours. It is glue over L0.6
(open_tokens, the formats/tokens-bin.md reader), L2.1 (NGramLM), L2.2
(NPLM, train_nplm, save_nplm, load_nplm), M11.2 (NLLAccumulator), and M06.3
(PCG32). Token files hold byte ids (the byte tokenizer, D32), so one token
is one byte and bits per byte is the mean NLL in bits.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

VOCAB = 256  # the byte tokenizer (formats/tokenizer.md)


class UsageError(Exception):
    exit_code = 2  # __main__ maps it to exit 2 (spec/cli-roles.md)


def read_ids(path: str) -> np.ndarray:
    from tinyllm.io.tokens import open_tokens

    return np.asarray(open_tokens(path), dtype=np.int64)


def arch_of(model_dir: str) -> str:
    cfg = Path(model_dir) / "config.json"
    if not cfg.is_file():
        raise ValueError(f"{model_dir}: no config.json (not a model directory)")
    return str(json.loads(cfg.read_text()).get("tl_arch", "llama"))


def write_out(a, result: dict) -> dict:
    """`generate --out <file>`: the final line, also written to a file."""
    if getattr(a, "out", None):
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result) + "\n")
    return result


# -- lm train -------------------------------------------------------------------


def cmd_train_ngram(a) -> dict:
    from tinyllm.lm.ngram import NGramLM

    if a.n is None:
        raise UsageError("lm train ngram needs --n")
    disc = a.discount if a.discount == "modified" else float(a.discount)
    ids = read_ids(a.train)
    lm = NGramLM(a.n, disc, VOCAB)
    lm.fit([ids.tolist()])  # one stream: <s> only before its first byte
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lm.save(str(out / "model.safetensors"))
    cfg = {
        "tl_arch": "ngram",
        "tl_tokenizer": "bytes",
        "vocab_size": VOCAB,
        "tl_format": 1,
        "tl_order": a.n,
    }
    (out / "config.json").write_text(json.dumps(cfg) + "\n")
    return {"out": str(out), "arch": "ngram", "n": a.n, "tokens": int(ids.size)}


def cmd_train_nplm(a) -> dict:
    from tinyllm.lm.nplm import NPLM, save_nplm, train_nplm
    from tinyllm.num.rng import PCG32

    need = {
        "--context": a.context,
        "--d-emb": a.d_emb,
        "--hidden": a.hidden,
        "--steps": a.steps,
        "--batch": a.batch,
        "--lr": a.lr,
    }
    missing = [k for k, v in need.items() if v is None]
    if missing:
        raise UsageError(f"lm train nplm needs {' '.join(missing)}")
    ids = read_ids(a.train)
    root = PCG32(a.seed)
    model = NPLM(
        VOCAB,
        a.context,
        a.d_emb,
        a.hidden,
        direct=not a.no_direct,
        rng=root.substream("init"),
    )
    losses = train_nplm(
        model,
        ids,
        a.steps,
        a.batch,
        a.lr,
        root.substream("shuffle"),
        momentum=a.momentum,
        weight_decay=a.weight_decay,
        clip=a.clip,
    )
    save_nplm(model, a.out, tokenizer="bytes")
    params = sum(int(p.data.size) for p in model.parameters())
    tail = losses[-50:] if losses else [float("nan")]
    return {
        "out": a.out,
        "arch": "nplm",
        "steps": len(losses),
        "loss": float(np.mean(tail)),
        "params": params,
    }


def run_lm(a) -> dict:
    if a.action != "train" or a.what not in ("ngram", "nplm"):
        raise UsageError("lm has one form: lm train <ngram|nplm>")
    if a.train is None or a.out is None:
        raise UsageError(f"lm train {a.what} needs --train and --out")
    return cmd_train_ngram(a) if a.what == "ngram" else cmd_train_nplm(a)


# -- eval ----------------------------------------------------------------------------


def cmd_eval(a) -> dict:
    """Perplexity of a model directory on a token file: every token the
    model can predict is scored (an n-gram from <s>, an NPLM from its first
    full window), summed with M11.2's NLLAccumulator."""
    from tinyllm.info.ppl import NLLAccumulator

    ids = read_ids(a.data)
    arch = arch_of(a.model)
    if arch == "ngram":
        from tinyllm.lm.ngram import NGramLM

        nll = NGramLM.load(str(Path(a.model) / "model.safetensors")).nll(ids.tolist())
    elif arch == "nplm":
        from tinyllm.lm.nplm import load_nplm

        nll = load_nplm(a.model).nll(ids)
    else:
        raise ValueError(f"{a.model}: eval scores tl_arch ngram or nplm, got {arch!r}")
    acc = NLLAccumulator()
    acc.add(nll, n_bytes=int(nll.size))  # byte tokens: each scored token is one byte
    r = acc.result()
    return {
        "model": arch,
        "ppl": r["ppl"],
        "nll_mean": r["nll_mean"],
        "bpb": r["bpb"],
        "tokens": int(r["n_tokens"]),
        "bytes": int(r["n_bytes"]),
    }


# -- generate (nplm) ------------------------------------------------------------------


def cmd_generate_lm(a) -> dict:
    from tinyllm.lm.nplm import load_nplm

    arch = arch_of(a.model)
    if arch != "nplm":
        raise ValueError(
            f"{a.model}: generate serves tl_arch bigram or nplm, got {arch!r}"
        )
    prompt = list(a.prompt.encode("utf-8"))
    temperature = 0.0 if a.greedy else a.temperature
    ids = load_nplm(a.model).generate(prompt, a.max_tokens, temperature, a.seed)
    text = bytes(ids).decode("utf-8", errors="replace")
    print(a.prompt + text, flush=True)
    return {"ids": ids, "text": text}


def add_parser(sub) -> None:
    """Called by __main__.parser(): the `lm` and `eval` verbs."""
    lm = sub.add_parser("lm")
    lm.add_argument("action", choices=["train"])
    lm.add_argument("what")
    lm.add_argument("--train")
    lm.add_argument("--out")
    lm.add_argument("--n", type=int)
    lm.add_argument("--discount", default="modified")
    lm.add_argument("--context", type=int)
    lm.add_argument("--d-emb", type=int)
    lm.add_argument("--hidden", type=int)
    lm.add_argument("--no-direct", action="store_true")
    lm.add_argument("--steps", type=int)
    lm.add_argument("--batch", type=int)
    lm.add_argument("--lr", type=float)
    lm.add_argument("--momentum", type=float, default=0.0)
    lm.add_argument("--weight-decay", type=float, default=0.0)
    lm.add_argument("--clip", type=float)
    lm.add_argument("--seed", type=int, default=0)
    lm.set_defaults(fn=run_lm)
    ev = sub.add_parser("eval")
    ev.add_argument("--model", required=True)
    ev.add_argument("--data", required=True)
    ev.set_defaults(fn=cmd_eval)
