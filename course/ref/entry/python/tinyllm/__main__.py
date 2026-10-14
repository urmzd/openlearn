"""The `tinyllm` CLI (contracts/spec/cli-roles.md).

Pass 1 verbs:

    train bigram --data <file> --out <dir> [--alpha A]
    generate --model <dir> --prompt <text> [--max-tokens N] [--greedy | --temperature T] [--seed S]
    logits --model <dir> --prompt <text> [--prefix-ids a,b,...]

Pass 2 verbs and flags (course/milestones/MS-L0.toml, glue in cli_train.py):

    gradcheck --suite all
    train bigram --method autograd ...      train mlp ...

Pass 3 verbs (course/milestones/MS-L1.toml, glue in cli_tok.py):

    tok train --algo bpe ...    tok encode ...    tok bench --impl rust ...

Pass 3 verbs (course/milestones/MS-L2.toml, glue in cli_lm.py):

    lm train ngram ...    lm train nplm ...    eval --model <dir> --data <tokens.bin>
    generate --model <nplm dir> ... [--out <file>]

Pass 4 verbs (course/milestones/MS-L3.toml and MS-L4.toml, glue in cli_seq.py):

    train rnnlm ...    train seq2seq ...    translate ...
    eval and generate on an rnnlm directory

Pass 5 verbs (course/milestones/MS-L7.toml, glue in cli_modern.py):

    pull <owner/name> ...    info, logits, generate on a Llama-family directory

Pass 5 verbs (course/milestones/MS-L5.toml, glue in cli_xfmr.py):

    train transformer ...    translate on a transformer directory

Pass 5 verbs (course/milestones/MS-L6.toml, glue in cli_obj.py):

    train gpt|bert|electra ...    finetune classify ...    eval ppl ...
    zoo add ...    eval --suite zoo ...

Pass 6 verbs (course/milestones/MS-L8.toml, glue in cli_infer.py):

    generate --cache none|contiguous|paged --kv-dtype f32|f16    generate --spec ngram|prompt-lookup --k K
    generate --json-schema <file>    eval ppl --quant <scheme>    bench decode

Pass 9 verbs (course/milestones/MS-L11.toml, glue in cli_capstone.py):

    train llama --cfg <config.json> --data <text> [--micro-batch B] [--accum K] [--bf16] [--checkpoint-activations]

Entry-point territory (D16): this file is yours. It is glue over L0.0
(BigramLM, safetensors); the reference is used
by course CI only. Runs as `python -m tinyllm` with python/ on the path, or
as `python python/tinyllm/__main__.py` from the repo root.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Run as a script: make the package importable from python/.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tinyllm.io.safetensors import load_safetensors, save_safetensors
from tinyllm.lm.bigram import BigramLM

VOCAB = 256  # the byte tokenizer (formats/tokenizer.md)
TENSOR = "bigram.weight"
CONFIG = {
    "tl_arch": "bigram",
    "tl_tokenizer": "bytes",
    "vocab_size": VOCAB,
    "tl_format": 1,
}


class UsageError(Exception):
    exit_code = 2


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(ids: list[int]) -> str:
    return bytes(ids).decode("utf-8", errors="replace")


def load_model(model_dir: str) -> BigramLM:
    d = Path(model_dir)
    cfg = json.loads((d / "config.json").read_text())
    if cfg.get("tl_arch") != "bigram" or cfg.get("tl_tokenizer") != "bytes":
        raise ValueError(
            f"{d}/config.json: this CLI serves tl_arch=bigram with tl_tokenizer=bytes, got {cfg}"
        )
    tensors, _ = load_safetensors(str(d / "model.safetensors"))
    if TENSOR not in tensors:
        raise ValueError(f"{d}/model.safetensors has no {TENSOR} tensor")
    return BigramLM(weight=tensors[TENSOR])


def cmd_train(a: argparse.Namespace) -> dict:
    if a.what != "bigram" or a.method != "counts":
        from tinyllm.cli_train import cmd_train_p2

        return cmd_train_p2(a)
    if a.out is None:
        raise UsageError("train bigram needs --out <dir>")
    ids = np.frombuffer(Path(a.data).read_bytes(), dtype=np.uint8).astype(np.int64)
    if ids.size < 2:
        raise ValueError(f"{a.data}: need at least two bytes to count a bigram")
    model = BigramLM()
    model.fit_counts(ids, VOCAB, alpha=a.alpha)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_safetensors(
        str(out / "model.safetensors"),
        {TENSOR: np.asarray(model.weight, dtype=np.float32)},
        {"format": "tinyllm"},
    )
    (out / "config.json").write_text(json.dumps(CONFIG) + "\n")
    print(f"trained bigram on {ids.size} bytes into {out}", flush=True)
    return {"out": str(out), "tokens": int(ids.size), "nll": model.nll(ids)}


def cmd_generate(a: argparse.Namespace) -> dict:
    # Pass 3 (MS-L2): an nplm directory, and --out <file>, glue in cli_lm.py.
    from tinyllm.cli_lm import arch_of, cmd_generate_lm, write_out

    if arch_of(a.model) != "bigram":
        return write_out(a, cmd_generate_lm(a))
    prompt = encode(a.prompt)
    if not prompt:
        raise UsageError(
            "generate: --prompt must not be empty (the byte tokenizer has no start token)"
        )
    temperature = 0.0 if a.greedy else a.temperature
    ids = load_model(a.model).sample(prompt, a.max_tokens, temperature, a.seed)
    text = decode(ids)
    print(a.prompt + text, flush=True)
    return write_out(a, {"ids": ids, "text": text})


def cmd_logits(a: argparse.Namespace) -> dict:
    prefix = (
        [int(x) for x in a.prefix_ids.split(",") if x.strip()] if a.prefix_ids else []
    )
    ids = encode(a.prompt) + prefix
    if not ids:
        raise UsageError("logits: needs a non-empty --prompt or --prefix-ids")
    row = load_model(a.model).logits(ids[-1:])[0]
    return {"logits": [float(x) for x in row]}


def cmd_gradcheck(a: argparse.Namespace) -> dict:
    from tinyllm.cli_train import cmd_gradcheck as run

    return run(a)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tinyllm")
    sub = ap.add_subparsers(dest="verb", required=True)
    t = sub.add_parser("train")
    t.add_argument("what")
    t.add_argument("--data", required=True)
    t.add_argument("--out")
    t.add_argument("--alpha", type=float, default=1.0)
    # Pass 2 (MS-L0): the autograd bigram, the token-stream run, the MLP.
    t.add_argument("--method", choices=["counts", "autograd"], default="counts")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--lr", type=float)
    t.add_argument("--steps", type=int, default=300)
    t.add_argument("--max-steps", type=int)
    t.add_argument("--batch", type=int)
    t.add_argument("--seq-len", type=int)
    t.add_argument("--ckpt-every", type=int)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--ckpt")
    t.set_defaults(fn=cmd_train)
    gc = sub.add_parser("gradcheck")
    gc.add_argument("--suite", required=True)
    gc.set_defaults(fn=cmd_gradcheck)
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--prompt", required=True)
    g.add_argument("--max-tokens", type=int, default=16)
    g.add_argument("--greedy", action="store_true")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out")  # MS-L2: also write the final line to this file
    g.set_defaults(fn=cmd_generate)
    lg = sub.add_parser("logits")
    lg.add_argument("--model", required=True)
    lg.add_argument("--prompt", default="")
    lg.add_argument("--prefix-ids", default="")
    lg.set_defaults(fn=cmd_logits)
    # Pass 3 (MS-L1): tok train, tok encode, tok bench, glue in cli_tok.py.
    from tinyllm.cli_tok import add_parser as add_tok

    add_tok(sub)
    # Pass 3 (MS-L2): lm train, eval, glue in cli_lm.py.
    from tinyllm.cli_lm import add_parser as add_lm

    add_lm(sub)
    return ap


def main(argv: list[str] | None = None) -> int:
    from tinyllm.cli_speech import intercept as intercept_speech

    code = intercept_speech(sys.argv[1:] if argv is None else list(argv))
    if code is not None:
        return code
    from tinyllm.cli_c1_data import intercept as intercept_c1_data

    code = intercept_c1_data(sys.argv[1:] if argv is None else list(argv))
    if code is not None:
        return code
    # Pass 9 (MS-L11): train llama with --bf16, --accum, and
    # --checkpoint-activations, glue in cli_capstone.py (it claims only that form).
    from tinyllm.cli_capstone import intercept as intercept_capstone

    code = intercept_capstone(sys.argv[1:] if argv is None else list(argv))
    if code is not None:
        return code
    # Pass 8 (MS-durable): train --spec and eval --spec, the subprocess
    # activities of TrainRun and EvalSuite, glue in cli_activity.py (first:
    # it claims only those two verbs with --spec).
    from tinyllm.cli_activity import intercept as intercept_activity

    code = intercept_activity(sys.argv[1:] if argv is None else list(argv))
    if code is not None:
        return code
    from tinyllm.cli_c1_eval import intercept as intercept_c1_eval

    code = intercept_c1_eval(sys.argv[1:] if argv is None else list(argv))
    if code is not None:
        return code
    # Pass 6 (MS-L8): generate with --cache, --kv-dtype, --spec, or
    # --json-schema, eval ppl --quant, and bench decode (no --backend) on a
    # Llama-family directory, glue in cli_infer.py (it claims only those).
    from tinyllm.cli_infer import intercept as intercept_infer

    code = intercept_infer(sys.argv[1:] if argv is None else list(argv))
    if code is not None:
        return code
    # Pass 5 (MS-L7): pull, and info, logits, generate on a Llama-family
    # directory, glue in cli_modern.py (every other form falls through).
    from tinyllm.cli_modern import intercept as intercept_modern

    code = intercept_modern(sys.argv[1:] if argv is None else list(argv))
    if code is not None:
        return code
    # Pass 5 (MS-L6): train gpt|bert|electra, finetune classify, eval ppl,
    # zoo add, eval --suite zoo, glue in cli_obj.py (first: it claims only
    # those verbs).
    from tinyllm.cli_obj import intercept as intercept_obj

    code = intercept_obj(sys.argv[1:] if argv is None else list(argv))
    if code is not None:
        return code
    # Pass 5 (MS-L5): train transformer, and translate on a transformer
    # directory, glue in cli_xfmr.py (before cli_seq, which takes every
    # other translate).
    from tinyllm.cli_xfmr import intercept as intercept_xfmr

    code = intercept_xfmr(sys.argv[1:] if argv is None else list(argv))
    if code is not None:
        return code
    # Pass 4 (MS-L3, MS-L4): train rnnlm|seq2seq, translate, and eval or
    # generate on an rnnlm directory, glue in cli_seq.py.
    from tinyllm.cli_seq import intercept

    code = intercept(sys.argv[1:] if argv is None else list(argv))
    if code is not None:
        return code
    a = parser().parse_args(argv)  # argparse exits 2 on a usage error
    try:
        result = a.fn(a)
    except Exception as e:  # noqa: BLE001 - every failure is reported, not traced
        # A usage error exits 2; a check that ran and failed prints its final
        # line and exits 1 (cli_train.CheckFailed); anything else exits 1.
        code = getattr(e, "exit_code", 1)
        why = str(e) if hasattr(e, "exit_code") else f"{type(e).__name__}: {e}"
        print(f"tinyllm{'' if code == 2 else ' ' + a.verb}: {why}", file=sys.stderr)
        if getattr(e, "result", None) is not None:
            print(json.dumps(e.result), flush=True)
        return code
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
