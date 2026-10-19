"""Python tokenizer verbs for the ``tinyllm`` CLI.

The Python implementation owns training and encoding. Rust parity is checked
through fixture files, so this entry point does not load a native extension.
"""

from __future__ import annotations

import json
from pathlib import Path


class UsageError(Exception):
    exit_code = 2


def read_texts(path: str) -> list[str]:
    if not path.endswith(".jsonl"):
        return Path(path).read_text(encoding="utf-8").splitlines()
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                obj = json.loads(line)
                if not isinstance(obj, dict) or not isinstance(obj.get("text"), str):
                    raise UsageError(f'{path}:{n}: want a JSON object with a string "text"')
                out.append(obj["text"])
    return out


def bytes_per_token(texts: list[str], n_tokens: int) -> float:
    total = sum(len(t.encode("utf-8")) for t in texts)
    return total / n_tokens if n_tokens else 0.0


def cmd_train(a) -> dict:
    from tinyllm.tok.bpe import BPETokenizer

    if a.algo != "bpe":
        raise UsageError(f"tok train: --algo {a.algo} is not built yet (Pass 3 has bpe)")
    if a.vocab is None or a.input is None or a.out is None:
        raise UsageError("tok train needs --vocab, --in, and --out")
    lines = Path(a.input).read_text(encoding="utf-8").splitlines()
    tok = BPETokenizer.train(lines, vocab_size=a.vocab, specials=a.special or (), min_freq=a.min_freq)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tok.save(a.out)
    n = sum(len(tok.encode(t)) for t in lines)
    print(f"trained {tok.vocab_size} ids ({len(tok.merges)} merges) into {a.out}/tokenizer.json", flush=True)
    return {"out": a.out, "algo": "bpe", "vocab_size": tok.vocab_size,
            "merges": len(tok.merges), "bytes_per_token": bytes_per_token(lines, n)}


def cmd_encode(a) -> dict:
    from tinyllm.tok.bpe import BPETokenizer

    texts = read_texts(a.input)
    tok = BPETokenizer.from_hf_json(a.tokenizer)
    ids = [i for row in tok.encode_batch(texts) for i in row]
    return {"ids": ids, "texts": len(texts), "tokens": len(ids),
            "bytes_per_token": bytes_per_token(texts, len(ids))}


def add_parser(sub) -> None:
    t = sub.add_parser("tok")
    t.add_argument("action", choices=["train", "encode"])
    t.add_argument("--algo", default="bpe")
    t.add_argument("--vocab", type=int)
    t.add_argument("--in", dest="input")
    t.add_argument("--out")
    t.add_argument("--min-freq", type=int, default=2)
    t.add_argument("--special", action="append")
    t.add_argument("--tokenizer")
    t.set_defaults(fn=run)


def run(a) -> dict:
    if a.action == "train":
        return cmd_train(a)
    if a.tokenizer is None or a.input is None:
        raise UsageError("tok encode needs --tokenizer and --in")
    return cmd_encode(a)
