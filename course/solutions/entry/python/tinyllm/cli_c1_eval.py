"""Evaluate the capstone's own checkpoints and seeded quality samples."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def eval_ppl(
    model_path: str, data: str, ctx: int | None = None, stride: int | None = None
) -> dict:
    from tinyllm.cli_modern import load
    from tinyllm.eval.lm import eval_ppl as evaluate
    from tinyllm.io.tokens import open_tokens

    model, text = load(model_path)
    ids = np.asarray(open_tokens(data), dtype=np.int64)
    context = ctx or model.config.max_position_embeddings
    if context > model.config.max_position_embeddings:
        raise ValueError("evaluation context exceeds the model context")
    if text.bytes:
        n_bytes = len(ids) - 1
    else:
        added = {value: key for key, value in text.tok.added.items()}
        n_bytes = sum(
            len(added[int(i)].encode("utf-8"))
            if int(i) in added
            else len(text.tok.id_to_token(int(i)))
            for i in ids[1:]
        )
    result = evaluate(
        model, ids, context, stride or max(1, context // 2), n_bytes=n_bytes
    )
    return {
        "model": model_path,
        "arch": "llama",
        "tokens": int(result["n_tokens"]),
        "ppl": float(result["ppl"]),
        "bpb": float(result["bpb"]),
    }


def quality(model_path: str, seed: int, samples: int) -> dict:
    from tinyllm.cli_modern import load
    from tinyllm.infer.generate import generate
    from tinyllm.infer.sample import SamplingParams

    if samples < 1 or seed < 0:
        raise ValueError("samples must be positive and seed nonnegative")
    path = Path("primers/craft.22/evals.py")
    spec = importlib.util.spec_from_file_location("capstone_quality", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load the learner's quality suite at {path}")
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    model, text = load(model_path)
    repeats, distinct = [], []
    for sample_seed in range(seed, seed + samples):
        generation = generate(
            model,
            text,
            "Once upon a time",
            SamplingParams(temperature=0.8, seed=sample_seed, max_tokens=64),
        )
        words = suite.words(generation.text)
        repeats.append(suite.repeat_ngram_rate(words, 8))
        distinct.append(len(set(words)) / len(words) if words else 0.0)
    return {
        "samples": samples,
        "repeat8": float(np.mean(repeats)),
        "distinct1": float(np.mean(distinct)),
    }


def intercept(argv: list[str]) -> int | None:
    from tinyllm.cli_modern import UsageError, _Parser, is_llama_dir

    if not argv or argv[0] != "eval":
        return None
    values = {}
    for i, arg in enumerate(argv):
        if "=" in arg:
            key, value = arg.split("=", 1)
            values[key] = value
        elif arg.startswith("--") and i + 1 < len(argv):
            values[arg] = argv[i + 1]
    is_quality = values.get("--suite") == "quality"
    is_ppl = argv[:2] == ["eval", "ppl"] and not {"--quant", "--tokens"}.intersection(
        values
    )
    if not is_quality and not is_ppl:
        return None
    try:
        if not is_llama_dir(values.get("--model")):
            return None
        parser = _Parser(prog="tinyllm eval")
        parser.add_argument("--model", required=True)
        if is_quality:
            parser.add_argument("--suite", choices=["quality"], required=True)
            parser.add_argument("--seed", type=int, default=0)
            parser.add_argument("--samples", type=int, default=20)
            args = parser.parse_args(argv[1:])
            result = quality(args.model, args.seed, args.samples)
        else:
            parser.add_argument("--data", required=True)
            parser.add_argument("--ctx", type=int)
            parser.add_argument("--stride", type=int)
            args = parser.parse_args(argv[2:])
            result = eval_ppl(args.model, args.data, args.ctx, args.stride)
    except (UsageError, ValueError, OSError, KeyError) as error:
        print(f"tinyllm eval: {error}", file=sys.stderr)
        return getattr(error, "exit_code", 1)
    print(json.dumps(result), flush=True)
    return 0
