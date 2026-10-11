"""Bench for L1.2 (DESIGN 4.3, the "1 MB train budget"): train a 4096-token
byte-level BPE on 1 MB of text and print {"seconds", "mb_per_s", "merges"}
as the last line. `ol bench L1.2` runs it against your units and compares
`seconds` with the [bench] budget in course/modules/L1.2.toml; it is never
part of `ol check`.

The text is synthetic and deterministic (random.Random(20261009)): words
built from English-like syllables with a Zipf-like frequency, so the trainer
sees tens of thousands of distinct pre-tokens, as on real prose.
"""

from __future__ import annotations

import json
import random
import time

from tinyllm.tok.bpe import BPETokenizer

ONSETS = [
    "",
    "b",
    "c",
    "d",
    "f",
    "g",
    "h",
    "l",
    "m",
    "n",
    "p",
    "r",
    "s",
    "t",
    "w",
    "st",
    "th",
    "ch",
    "sh",
    "br",
    "tr",
]
NUCLEI = ["a", "e", "i", "o", "u", "ea", "oo", "ai", "ou", "y"]
CODAS = ["", "", "n", "t", "s", "r", "l", "d", "ng", "ck", "st"]
PUNCT = [".", ",", "!", "?", ";", ":"]


def corpus(n_bytes: int = 1 << 20) -> list[str]:
    rng = random.Random(20261009)
    vocab = []
    for _ in range(20000):
        k = 1 + int(rng.random() * 3)
        vocab.append(
            "".join(
                rng.choice(ONSETS) + rng.choice(NUCLEI) + rng.choice(CODAS)
                for _ in range(k)
            )
        )
    weights = [1.0 / (r + 1) for r in range(len(vocab))]
    lines, size = [], 0
    while size < n_bytes:
        words = rng.choices(vocab, weights, k=8 + int(rng.random() * 12))
        if rng.random() < 0.3:
            words[0] = words[0].capitalize()
        line = " ".join(words) + rng.choice(PUNCT) + "\n"
        lines.append(line)
        size += len(line.encode("utf-8"))
    return lines


def main() -> None:
    texts = corpus()
    t0 = time.perf_counter()
    tok = BPETokenizer.train(texts, vocab_size=4096)
    dt = time.perf_counter() - t0
    mb = sum(len(t.encode("utf-8")) for t in texts) / (1 << 20)
    print(
        json.dumps(
            {
                "seconds": round(dt, 3),
                "mb_per_s": round(mb / dt, 4),
                "merges": len(tok.merges),
            }
        )
    )


if __name__ == "__main__":
    main()
