"""The Pass 4 sequence-model verbs (course/milestones/MS-L3.toml and
MS-L4.toml fix their flags and final lines).

    train rnnlm   --cell rnn|lstm|gru --data <file> --out <dir> --steps S [--seed S]
                  [--d-emb 32] [--hidden 128] [--layers 1] [--bptt 32] [--batch 32]
                  [--lr 0.01] [--clip 1.0] [--val-frac 0.1]
    train seq2seq --task <pairs.tsv> --attn none|bahdanau|luong --out <dir> --steps S [--seed S]
                  [--cell gru] [--d-emb 32] [--hidden 64] [--d-attn 64] [--score general]
                  [--batch 64] [--lr 0.005] [--clip 1.0]
    translate     --model <seq2seq dir> --in <pairs.txt> [--beam 5] [--max-len 16] [--long-chars 20]
    eval          --model <rnnlm dir> --data <tokens.bin | text file> [--tail-frac F]
    generate      --model <rnnlm dir> --prompt <text> [--max-tokens N] [--greedy | --temperature T] [--seed S] [--out <file>]

`intercept(argv)` runs these before the Pass 1 to 3 parser, which keeps its
verbs unchanged: it returns an exit code when argv is one of them and None
otherwise (`eval` and `generate` only for an rnnlm directory).

Entry-point territory (D16): this file is yours. It is glue over L3.6
(RNNLM, train_tbptt, save_rnnlm, load_rnnlm), L4.1 (Seq2Seq, save_seq2seq,
load_seq2seq), L4.2 and L4.3 (the attention modules), L4.4 (beam_search),
L4.5 (exact_match, chrf, metric_ci), L1.1 (CharTokenizer), L0.6
(open_tokens), M11.2 (NLLAccumulator), M10.3 (AdamW), L0.3
(cross_entropy), L0.5 (train_step), and M06.3 (PCG32). A text file is read
as UTF-8 bytes (the byte tokenizer, D32); a .bin file is a
formats/tokens-bin.md shard of byte ids.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

VOCAB = 256  # the byte tokenizer (formats/tokenizer.md)
SPECIALS = ["<pad>", "<bos>", "<eos>"]


class UsageError(Exception):
    exit_code = 2


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # usage errors exit 2, as argparse does
        raise UsageError(message)


def _arch(model_dir: str) -> str | None:
    cfg = Path(model_dir) / "config.json"
    if not cfg.is_file():
        return None
    return str(json.loads(cfg.read_text()).get("tl_arch", ""))


def _flag(argv: list[str], name: str) -> str | None:
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def read_stream(path: str) -> np.ndarray:
    """Byte ids of a text file, or the ids of a tokens-bin shard."""
    if path.endswith(".bin"):
        from tinyllm.io.tokens import open_tokens

        return np.asarray(open_tokens(path), dtype=np.int64)
    return np.frombuffer(Path(path).read_bytes(), dtype=np.uint8).astype(np.int64)


# -- train rnnlm -------------------------------------------------------------------------


def cmd_train_rnnlm(a) -> dict:
    from tinyllm.num.rng import PCG32
    from tinyllm.optim.adamw import AdamW
    from tinyllm.rnn.rnnlm import RNNLM, save_rnnlm, train_tbptt

    ids = read_stream(a.data)
    n_val = int(len(ids) * a.val_frac)
    if n_val < 2 or len(ids) - n_val < a.batch * (a.bptt + 1):
        raise UsageError(f"{a.data}: {len(ids)} tokens is too few for --batch {a.batch} --bptt {a.bptt} and a validation tail")
    train, val = ids[: len(ids) - n_val], ids[len(ids) - n_val :]
    root = PCG32(a.seed)
    model = RNNLM(VOCAB, a.d_emb, a.hidden, a.cell, n_layers=a.layers, rng=root.substream("init"))
    opt = AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    losses = train_tbptt(model, train, a.bptt, a.batch, opt, a.clip, a.steps)
    nll = model.nll(val)
    save_rnnlm(model, a.out, tokenizer="bytes")
    params = sum(int(p.data.size) for p in model.parameters())
    tail = losses[-50:] if losses else [float("nan")]
    # bits per character: one byte per token, and the text is ASCII here.
    return {
        "out": a.out,
        "arch": "rnnlm",
        "cell": a.cell,
        "steps": len(losses),
        "loss": float(np.mean(tail)),
        "val_bpc": float(np.mean(nll) / math.log(2)),
        "val_tokens": int(nll.size),
        "params": params,
    }


# -- train seq2seq ------------------------------------------------------------------------


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


def build_attention(kind: str, d_h: int, d_attn: int, score: str, rng):
    if kind == "none":
        return None
    if kind == "bahdanau":
        from tinyllm.seq2seq.additive import AdditiveAttention

        return AdditiveAttention(d_h, d_h, d_attn, rng=rng)
    if kind == "luong":
        from tinyllm.seq2seq.luong import LuongAttention

        return LuongAttention(d_h, score, rng=rng)
    raise UsageError(f"--attn must be none, bahdanau, or luong, got {kind!r}")


def cmd_train_seq2seq(a) -> dict:
    from tinyllm.autograd import functional as F
    from tinyllm.autograd.losses import cross_entropy
    from tinyllm.num.rng import PCG32
    from tinyllm.optim.adamw import AdamW
    from tinyllm.seq2seq.model import Seq2Seq, save_seq2seq
    from tinyllm.tok.char import CharTokenizer
    from tinyllm.train.loop import train_step

    pairs = read_pairs(a.task)
    tok = CharTokenizer.train([s for s, _ in pairs] + [t for _, t in pairs], specials=SPECIALS)
    pad, bos, eos = (tok.token_to_id(s) for s in SPECIALS)
    V = len(tok.vocab)
    src = [tok.encode(s) for s, _ in pairs]
    tgt = [tok.encode(t) for _, t in pairs]
    root = PCG32(a.seed)
    init = root.substream("init")  # the attention module first, then the model, from one stream
    att = build_attention(a.attn, a.hidden, a.d_attn, a.score, init)
    model = Seq2Seq(V, V, a.d_emb, a.hidden, cell=a.cell, attention=att, rng=init)
    opt = AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    order_rng = root.substream("shuffle")
    order: list[int] = []

    def loss_fn(m, b):
        logits = m(b["src"], b["lens"], b["tgt_in"])
        return cross_entropy(F.reshape(logits, (-1, V)), b["tgt_out"].reshape(-1))

    losses = []
    for _ in range(a.steps):
        if len(order) < a.batch:
            fresh = list(range(len(pairs)))
            order_rng.shuffle(fresh)
            order += fresh
        idx, order = order[: a.batch], order[a.batch :]
        s_rows = [src[i] for i in idx]
        t_in = [[bos] + tgt[i] for i in idx]
        t_out = [tgt[i] + [eos] for i in idx]
        T = max(len(r) for r in t_in)
        tgt_out = _pad(t_out, T, -100)  # padding positions predict nothing
        batch = {
            "src": _pad(s_rows, max(len(r) for r in s_rows), pad),
            "lens": np.array([len(r) for r in s_rows], dtype=np.int64),
            "tgt_in": _pad(t_in, T, pad),
            "tgt_out": tgt_out,
        }
        losses.append(train_step(model, batch, loss_fn, opt, clip=a.clip)["loss"])
    save_seq2seq(model, a.out)
    tok.save(a.out)
    params = sum(int(p.data.size) for p in model.parameters())
    tail = losses[-50:] if losses else [float("nan")]
    return {"out": a.out, "arch": "seq2seq", "attn": a.attn, "steps": len(losses),
            "loss": float(np.mean(tail)), "params": params}


# -- translate ----------------------------------------------------------------------------


def _decode(model, tok, text: str, beam: int, max_len: int) -> str:
    from tinyllm.autograd.mode import no_grad
    from tinyllm.infer.beam import beam_search

    pad, bos, eos = (tok.token_to_id(s) for s in SPECIALS)
    ids = tok.encode(text)
    with no_grad():
        enc = model.encode(np.array([ids], dtype=np.int64), np.array([len(ids)], dtype=np.int64))

        def step(state, y_prev):
            logits, new_state, _ = model.decode_step(np.asarray(y_prev, dtype=np.int64), state)
            return logits.data, new_state

        hyps = beam_search(step, model.init_state(enc), bos, eos, beam, max_len)
    out = [t for t in hyps[0].tokens if t != eos]
    return tok.decode(out, skip_special=True)


def cmd_translate(a) -> dict:
    from tinyllm.eval.seqmetrics import chrf, exact_match, metric_ci
    from tinyllm.num.rng import PCG32
    from tinyllm.seq2seq.model import load_seq2seq
    from tinyllm.tok.char import CharTokenizer

    model = load_seq2seq(a.model)
    model.eval()
    tok = CharTokenizer.load(a.model)
    pairs = read_pairs(a.input)
    srcs, refs = [s for s, _ in pairs], [t for _, t in pairs]
    beam = [_decode(model, tok, s, a.beam, a.max_len) for s in srcs]
    greedy = beam if a.beam == 1 else [_decode(model, tok, s, 1, a.max_len) for s in srcs]
    long = [i for i, s in enumerate(srcs) if len(s) >= a.long_chars]
    em, lo, hi = metric_ci("exact_match", beam, refs, 1000, 0.05, PCG32(0))
    em_greedy = exact_match(greedy, refs)
    short = [i for i in range(len(srcs)) if len(srcs[i]) < a.long_chars]
    for s, h in zip(srcs[:3], beam[:3]):
        print(f"{s} -> {h}", flush=True)
    return {
        "model": a.model,
        "beam": a.beam,
        "n": len(pairs),
        "em": em,
        "em_lo": lo,
        "em_hi": hi,
        "em_greedy": em_greedy,
        "em_gain": em - em_greedy,  # what beam search adds over greedy decoding
        "em_long": exact_match([beam[i] for i in long], [refs[i] for i in long]) if long else 0.0,
        "em_short": exact_match([beam[i] for i in short], [refs[i] for i in short]) if short else 0.0,
        "n_long": len(long),
        "chrf": chrf(beam, [[r] for r in refs]),
    }


# -- eval and generate on an rnnlm directory ----------------------------------------------------


def cmd_eval_rnnlm(a) -> dict:
    from tinyllm.info.ppl import NLLAccumulator
    from tinyllm.rnn.rnnlm import load_rnnlm

    ids = read_stream(a.data)
    if a.tail_frac is not None:  # the validation tail `train rnnlm --val-frac` held out
        if not 0 < a.tail_frac < 1:
            raise UsageError("--tail-frac must lie in (0, 1)")
        ids = ids[len(ids) - int(len(ids) * a.tail_frac) :]
    nll = load_rnnlm(a.model).nll(ids)
    acc = NLLAccumulator()
    acc.add(nll, n_bytes=int(nll.size))  # byte tokens: each scored token is one byte
    r = acc.result()
    return {"model": "rnnlm", "ppl": r["ppl"], "nll_mean": r["nll_mean"], "bpb": r["bpb"],
            "tokens": int(r["n_tokens"]), "bytes": int(r["n_bytes"])}


def cmd_generate_rnnlm(a) -> dict:
    from tinyllm.rnn.rnnlm import load_rnnlm

    prompt = list(a.prompt.encode("utf-8"))
    if not prompt:
        raise UsageError("generate: --prompt must not be empty")
    temperature = 0.0 if a.greedy else a.temperature
    ids = load_rnnlm(a.model).generate(prompt, a.max_tokens, temperature, a.seed)
    text = bytes(ids).decode("utf-8", errors="replace")
    print(a.prompt + text, flush=True)
    result = {"ids": ids, "text": text}
    if a.out:  # the final line, also written to a file (as MS-L2's generate --out)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result) + "\n")
    return result


# -- dispatch ---------------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    ap = _Parser(prog="tinyllm")
    sub = ap.add_subparsers(dest="verb", required=True, parser_class=_Parser)
    r = sub.add_parser("train-rnnlm")
    r.add_argument("--cell", choices=["rnn", "lstm", "gru"], required=True)
    r.add_argument("--data", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--steps", type=int, required=True)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--d-emb", type=int, default=32)
    r.add_argument("--hidden", type=int, default=128)
    r.add_argument("--layers", type=int, default=1)
    r.add_argument("--bptt", type=int, default=32)
    r.add_argument("--batch", type=int, default=32)
    r.add_argument("--lr", type=float, default=0.01)
    r.add_argument("--clip", type=float, default=1.0)
    r.add_argument("--val-frac", type=float, default=0.1)
    r.set_defaults(fn=cmd_train_rnnlm)
    s = sub.add_parser("train-seq2seq")
    s.add_argument("--task", required=True)
    s.add_argument("--attn", choices=["none", "bahdanau", "luong"], required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--steps", type=int, required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--cell", choices=["gru", "lstm"], default="gru")
    s.add_argument("--d-emb", type=int, default=32)
    s.add_argument("--hidden", type=int, default=64)
    s.add_argument("--d-attn", type=int, default=64)
    s.add_argument("--score", choices=["dot", "general", "concat"], default="general")
    s.add_argument("--batch", type=int, default=64)
    s.add_argument("--lr", type=float, default=0.005)
    s.add_argument("--clip", type=float, default=1.0)
    s.set_defaults(fn=cmd_train_seq2seq)
    t = sub.add_parser("translate")
    t.add_argument("--model", required=True)
    t.add_argument("--in", dest="input", required=True)
    t.add_argument("--beam", type=int, default=5)
    t.add_argument("--max-len", type=int, default=16)
    t.add_argument("--long-chars", type=int, default=20)
    t.set_defaults(fn=cmd_translate)
    e = sub.add_parser("eval")
    e.add_argument("--model", required=True)
    e.add_argument("--data", required=True)
    e.add_argument("--tail-frac", type=float)
    e.set_defaults(fn=cmd_eval_rnnlm)
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--prompt", required=True)
    g.add_argument("--max-tokens", type=int, default=16)
    g.add_argument("--greedy", action="store_true")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out")
    g.set_defaults(fn=cmd_generate_rnnlm)
    return ap


def _mine(argv: list[str]) -> list[str] | None:
    """argv rewritten for parser() when it is a Pass 4 verb, else None."""
    if len(argv) >= 2 and argv[0] == "train" and argv[1] in ("rnnlm", "seq2seq"):
        return [f"train-{argv[1]}"] + argv[2:]
    if argv and argv[0] == "translate":
        return argv
    if argv and argv[0] in ("eval", "generate"):
        model = _flag(argv, "--model")
        if model is not None and _arch(model) == "rnnlm":
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
