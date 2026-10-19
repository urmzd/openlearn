"""The Pass 5 transformer verbs (course/milestones/MS-L5.toml fixes their
flags and final lines).

    train transformer --task <pairs.tsv> --cfg <config.json> --out <dir>
                      [--steps S] [--seed S] [--norm post|pre] [--warmup W]
                      [--lr X] [--factor F] [--smoothing E] [--batch B]
    translate         --model <transformer dir> --in <pairs.tsv> [--beam 4] [--max-len 8]

`intercept(argv)` runs these before the Pass 4 verbs (cli_seq.py), so
`translate` on a transformer directory comes here and every other
`translate` goes on to the seq2seq glue. It returns an exit code when argv
is one of them and None otherwise.

Entry-point territory (D16): this file is yours. It is glue over L5.5
(TransformerConfig, Transformer, fit, translate, save_transformer,
load_transformer), L1.1 (CharTokenizer), L4.5 (exact_match, metric_ci),
and M06.3 (PCG32).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

SPECIALS = ["<pad>", "<bos>", "<eos>"]
CFG_KEYS = (
    "d_model",
    "n_heads",
    "d_ff",
    "n_enc",
    "n_dec",
    "dropout",
    "norm",
    "tie_embeddings",
    "max_len",
)


class UsageError(Exception):
    exit_code = 2


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def read_pairs(path: str) -> list[tuple[str, str]]:
    pairs = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(f"{path}:{i}: want `source<TAB>target`")
        pairs.append((parts[0], parts[1]))
    if not pairs:
        raise ValueError(f"{path}: no pairs")
    return pairs


def _pad(rows: list[list[int]], width: int, pad: int) -> np.ndarray:
    out = np.full((len(rows), width), pad, dtype=np.int64)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


def _arch(model_dir: str) -> str | None:
    cfg = Path(model_dir) / "config.json"
    if not cfg.is_file():
        return None
    return str(json.loads(cfg.read_text()).get("tl_arch", ""))


def cmd_train(a) -> dict:
    from tinyllm.num.rng import PCG32
    from tinyllm.tok.char import CharTokenizer
    from tinyllm.xfmr.transformer import (
        Transformer,
        TransformerConfig,
        fit,
        save_transformer,
    )

    c = json.loads(Path(a.cfg).read_text())
    for k in ("norm", "warmup", "lr", "factor", "smoothing", "batch", "steps"):
        if getattr(a, k) is not None:
            c[k] = getattr(a, k)
    warmup = int(c.get("warmup", 4000))
    lr = c.get("lr")
    if warmup == 0 and lr is None:
        raise UsageError("--warmup 0 trains at a constant rate: give it with --lr")
    pairs = read_pairs(a.task)
    tok = CharTokenizer.train(
        [s for s, _ in pairs] + [t for _, t in pairs], specials=SPECIALS
    )
    pad, bos, eos = (tok.token_to_id(s) for s in SPECIALS)
    V = len(tok.vocab)
    src = [tok.encode(s) for s, _ in pairs]
    tgt = [[bos] + tok.encode(t) + [eos] for _, t in pairs]
    S, T = max(map(len, src)), max(map(len, tgt))
    cfg = TransformerConfig(V, V, **{k: c[k] for k in CFG_KEYS if k in c})
    root = PCG32(a.seed)
    model = Transformer(cfg, rng=root.substream("init"))
    first_bad = None
    losses: list[float] = []
    try:
        losses = fit(
            model,
            _pad(src, S, pad),
            _pad(tgt, T, pad),
            int(c.get("steps", 2500)),
            int(c.get("batch", 64)),
            pad,
            warmup=max(warmup, 1),
            factor=float(c.get("factor", 1.0)),
            smoothing=float(c.get("smoothing", 0.1)),
            lr=None if warmup else float(lr),
            rng=root.substream("shuffle"),
            on_step=lambda i, s: losses.append(s["loss"]),
        )
    except FloatingPointError as e:  # the loss went to inf or nan: that is divergence
        first_bad = str(e)
    save_transformer(model, a.out)
    tok.save(a.out)
    first = float(np.mean(losses[:10])) if losses else float("nan")
    final = (
        float(np.mean(losses[-20:])) if losses and first_bad is None else float("inf")
    )
    ratio = final / first if math.isfinite(final) and first > 0 else float("inf")
    # Diverged: the loss blew up, or training kept more than 80% of its first loss.
    diverged = int(first_bad is not None or not ratio <= 0.8)
    out = {
        "out": a.out,
        "arch": "transformer",
        "norm": cfg.norm,
        "warmup": warmup,
        "steps": len(losses),
        "first_loss": first,
        "final_loss": final if math.isfinite(final) else 1e9,
        "loss_ratio": ratio if math.isfinite(ratio) else 1e9,
        "diverged": diverged,
        "params": int(sum(p.data.size for p in model.parameters())),
    }
    if cfg.norm == "post" and warmup == 0:
        out["post_ln_no_warmup_diverged"] = bool(diverged)
    return out


def cmd_translate(a) -> dict:
    from tinyllm.eval.seqmetrics import exact_match, metric_ci
    from tinyllm.num.rng import PCG32
    from tinyllm.tok.char import CharTokenizer
    from tinyllm.xfmr.transformer import load_transformer, translate

    model = load_transformer(a.model)
    tok = CharTokenizer.load(a.model)
    pad, bos, eos = (tok.token_to_id(s) for s in SPECIALS)
    pairs = read_pairs(a.input)
    rows = [tok.encode(s) for s, _ in pairs]
    src = _pad(rows, max(map(len, rows)), pad)
    refs = [t for _, t in pairs]

    def run(beam: int) -> list[str]:
        return [
            tok.decode(ids, skip_special=True)
            for ids in translate(model, src, bos, eos, pad, a.max_len, beam)
        ]

    beam = run(a.beam)
    greedy = beam if a.beam == 1 else run(1)
    em, lo, hi = metric_ci("exact_match", beam, refs, 1000, 0.05, PCG32(0))
    for (s, _), h in zip(pairs[:3], beam[:3]):
        print(f"{s} -> {h}", flush=True)
    return {
        "model": a.model,
        "arch": "transformer",
        "beam": a.beam,
        "n": len(pairs),
        "em_transformer": em,
        "em": em,
        "em_lo": lo,
        "em_hi": hi,
        "em_greedy": exact_match(greedy, refs),
    }


def parser() -> argparse.ArgumentParser:
    ap = _Parser(prog="tinyllm")
    sub = ap.add_subparsers(dest="verb", required=True, parser_class=_Parser)
    t = sub.add_parser("train-transformer")
    t.add_argument("--task", required=True)
    t.add_argument("--cfg", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--steps", type=int)
    t.add_argument("--norm", choices=["post", "pre"])
    t.add_argument("--warmup", type=int)
    t.add_argument("--lr", type=float)
    t.add_argument("--factor", type=float)
    t.add_argument("--smoothing", type=float)
    t.add_argument("--batch", type=int)
    t.set_defaults(fn=cmd_train)
    r = sub.add_parser("translate")
    r.add_argument("--model", required=True)
    r.add_argument("--in", dest="input", required=True)
    r.add_argument("--beam", type=int, default=4)
    r.add_argument("--max-len", type=int, default=8)
    r.set_defaults(fn=cmd_translate)
    return ap


def _mine(argv: list[str]) -> list[str] | None:
    if len(argv) >= 2 and argv[0] == "train" and argv[1] == "transformer":
        return ["train-transformer"] + argv[2:]
    if argv and argv[0] == "translate":
        model = next(
            (argv[i + 1] for i, x in enumerate(argv[:-1]) if x == "--model"), None
        )
        if model is not None and _arch(model) == "transformer":
            return argv
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
