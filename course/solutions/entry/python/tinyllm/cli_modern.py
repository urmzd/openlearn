"""The Pass 5 Llama-family verbs (course/milestones/MS-L7.toml fixes their
flags and final lines).

    pull <owner/name> [--revision R] [--cache-dir DIR] [--files a,b,...]
    info --model <llama dir>
    logits --model <llama dir> (--prompt TEXT [--prefix-ids a,b,...] | --prompts FILE.jsonl --out FILE.npz)
    generate --model <llama dir> (--prompt TEXT | --prompt-file FILE) [--max-tokens N]
             [--greedy | --temperature T] [--seed S] [--out FILE]

A "llama dir" (given as the directory or as its config.json) is a model
directory whose config.json is a Llama-family config (tl_arch "llama", or model_type llama, mistral, or qwen2 without
another tl_arch). `intercept(argv)` handles `pull` and these three verbs on
such a directory, and returns None for everything else, so the earlier
passes' verbs keep their meaning.

Entry-point territory (D16): this file is yours. It is glue over L7.9
(LlamaForCausalLM, hf_download), L7.5 (ConcatKVCache for decoding), L1.2
(BPETokenizer.from_hf_json for tokenizer.json), M05.1 (kv_bytes_per_token),
and M06.3 (PCG32 for sampling).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

DEFAULT_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


class UsageError(Exception):
    exit_code = 2


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def model_dir(path: str) -> str:
    """--model names a model directory, or the config.json inside it."""
    p = Path(path)
    p = p.parent if p.name == "config.json" else p
    if (p / "LATEST").is_file():
        name = (p / "LATEST").read_text().strip()
        if not name.startswith("step-") or not name[5:].isdigit():
            raise UsageError(f"{p}/LATEST does not name a checkpoint step")
        p = p / name
    return str(p)


def is_llama_dir(path: str | None) -> bool:
    if not path:
        return False
    p = Path(model_dir(path)) / "config.json"
    if not p.is_file():
        return False
    c = json.loads(p.read_text())
    arch = c.get("tl_arch")
    return arch == "llama" or (
        arch is None and c.get("model_type") in ("llama", "mistral", "qwen2")
    )


class Text:
    """The model directory's tokenizer: bytes (D32) or tokenizer.json (L1.2)."""

    def __init__(self, model_dir: str, cfg) -> None:
        self.bytes = cfg.tokenizer == "bytes"
        if not self.bytes:
            from tinyllm.tok.bpe import BPETokenizer

            self.tok = BPETokenizer.from_hf_json(
                str(Path(model_dir) / "tokenizer.json")
            )

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8")) if self.bytes else self.tok.encode(text)

    def decode(self, ids: list[int]) -> str:
        return (
            bytes(ids).decode("utf-8", errors="replace")
            if self.bytes
            else self.tok.decode(ids)
        )


def load(path: str):
    from tinyllm.modern.llama import LlamaForCausalLM

    d = model_dir(path)
    m = LlamaForCausalLM.from_pretrained(d)
    return m, Text(d, m.config)


def cmd_pull(a) -> dict:
    from tinyllm.io.hf import hf_download

    cache = a.cache_dir or str(
        Path(os.environ.get("TINYLLM_CACHE", Path.home() / ".cache" / "openlearn"))
        / "models"
    )
    files = [f for f in a.files.split(",") if f] if a.files else DEFAULT_FILES
    d = hf_download(a.repo, files, cache, revision=a.revision)
    return {"dir": d, "files": files}


def cmd_info(a) -> dict:
    from tinyllm.accounting import kv_bytes_per_token

    m, _ = load(a.model)
    c = m.config
    return {
        "params": m.param_count(),
        "arch": "llama",
        "layers": c.num_hidden_layers,
        "attention": c.attention,
        "vocab": c.vocab_size,
        "kv_bytes_per_token_bf16": kv_bytes_per_token(c.model_config(), 2),
    }


def next_logits(m, ids: list[int]) -> np.ndarray:
    if not ids:
        raise UsageError("logits: the prompt has no tokens")
    return np.asarray(m(np.array([ids])).data[0, -1], dtype=np.float64)


def cmd_logits(a) -> dict:
    m, text = load(a.model)
    if a.prompts:
        if not a.out:
            raise UsageError("logits --prompts needs --out <file.npz>")
        rows = [
            json.loads(line)
            for line in Path(a.prompts).read_text().splitlines()
            if line.strip()
        ]
        out = {}
        for i, r in enumerate(rows):
            ids = r["ids"] if "ids" in r else text.encode(r["prompt"])
            out[f"logits{i}"] = next_logits(m, list(ids)).astype(np.float32)
        np.savez(a.out, **out)
        return {"out": a.out, "prompts": len(rows)}
    prefix = (
        [int(x) for x in a.prefix_ids.split(",") if x.strip()] if a.prefix_ids else []
    )
    return {
        "logits": [float(x) for x in next_logits(m, text.encode(a.prompt) + prefix)]
    }


def generate_ids(
    m, prompt: list[int], n: int, temperature: float, seed: int
) -> list[int]:
    from tinyllm.modern.gqa import ConcatKVCache
    from tinyllm.modern.mla import ConcatLatentCache
    from tinyllm.num.rng import PCG32

    if not prompt:
        raise UsageError("generate: the prompt has no tokens")
    cache = ConcatLatentCache() if m.config.attention == "mla" else ConcatKVCache()
    rng = PCG32(seed).substream("sample")
    row = np.asarray(m(np.array([prompt]), cache=cache).data[0, -1], dtype=np.float64)
    out: list[int] = []
    for _ in range(n):
        if temperature == 0:
            nxt = int(np.argmax(row))  # ties to the lowest id
        else:
            z = row / temperature
            p = np.exp(z - z.max())
            p /= p.sum()
            nxt = min(
                int(np.searchsorted(np.cumsum(p), rng.uniform(), side="right")),
                len(p) - 1,
            )
        out.append(nxt)
        if len(out) < n:
            row = np.asarray(
                m(np.array([[nxt]]), cache=cache).data[0, -1], dtype=np.float64
            )
    return out


def cmd_generate(a) -> dict:
    if bool(a.prompt) == bool(a.prompt_file):
        raise UsageError("generate needs exactly one of --prompt and --prompt-file")
    m, text = load(a.model)
    t = 0.0 if a.greedy else a.temperature
    prompts = (
        [a.prompt]
        if a.prompt
        else [line for line in Path(a.prompt_file).read_text().splitlines() if line]
    )
    per = [generate_ids(m, text.encode(p), a.max_tokens, t, a.seed) for p in prompts]
    ids = [i for row in per for i in row]
    result = {"ids": ids, "text": "\n".join(text.decode(r) for r in per)}
    if a.prompt_file:
        result["per_prompt"] = per
    if a.out:
        Path(a.out).write_text(json.dumps(result) + "\n")
    return result


def parser() -> argparse.ArgumentParser:
    ap = _Parser(prog="tinyllm")
    sub = ap.add_subparsers(dest="verb", required=True)
    p = sub.add_parser("pull")
    p.add_argument("repo")
    p.add_argument("--revision", default="main")
    p.add_argument("--cache-dir")
    p.add_argument("--files")
    p.set_defaults(fn=cmd_pull)
    i = sub.add_parser("info")
    i.add_argument("--model", required=True)
    i.set_defaults(fn=cmd_info)
    lg = sub.add_parser("logits")
    lg.add_argument("--model", required=True)
    lg.add_argument("--prompt", default="")
    lg.add_argument("--prefix-ids", default="")
    lg.add_argument("--prompts")
    lg.add_argument("--out")
    lg.set_defaults(fn=cmd_logits)
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--prompt")
    g.add_argument("--prompt-file")
    g.add_argument("--max-tokens", type=int, default=16)
    g.add_argument("--greedy", action="store_true")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out")
    g.set_defaults(fn=cmd_generate)
    return ap


def _model_arg(argv: list[str]) -> str | None:
    for k, v in enumerate(argv):
        if v == "--model" and k + 1 < len(argv):
            return argv[k + 1]
        if v.startswith("--model="):
            return v.split("=", 1)[1]
    return None


def intercept(argv: list[str]) -> int | None:
    if not argv:
        return None
    if argv[0] != "pull" and not (
        argv[0] in ("info", "logits", "generate") and is_llama_dir(_model_arg(argv))
    ):
        return None
    try:
        a = parser().parse_args(argv)
        result = a.fn(a)
    except Exception as e:  # noqa: BLE001 - every failure is reported, not traced
        code = getattr(e, "exit_code", 1)
        why = str(e) if hasattr(e, "exit_code") else f"{type(e).__name__}: {e}"
        print(f"tinyllm {argv[0]}: {why}", file=sys.stderr)
        return code
    print(json.dumps(result), flush=True)
    return 0
