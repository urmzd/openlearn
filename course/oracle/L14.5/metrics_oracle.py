# /// script
# requires-python = ">=3.12"
# dependencies = ["jiwer==4.0.0", "sacrebleu==2.6.0", "transformers==5.19.0", "numpy==2.5.3"]
# ///
"""Maintainer generator for the L14.5 golden values (multimodal metrics).

metrics_oracle.json holds:

  normalize   strings with bracketed tags, parentheses, punctuation,
              accents, ligatures, full-width forms, CJK, emoji, and odd
              white space, with transformers' BasicTextNormalizer output
              (the port of openai-whisper's basic normalizer)
  asr         a corpus of (reference, hypothesis) pairs, random word edits
              over a small vocabulary plus hand-written cases, with jiwer's
              corpus WER and CER and, per item, jiwer's S, D, I, and N for
              words (jiwer breaks alignment ties by rapidfuzz's backtrace,
              which prefers a deletion; the course rule prefers a
              substitution, so only S + D + I and N are compared)
  captions    predictions with 2 to 3 references each, sacreBLEU 2.6
              corpus BLEU and chrF with the defaults

    uv run --python 3.12 --script course/oracle/L14.5/metrics_oracle.py

Run from the repo root and paste the printed row into course/fixtures/MANIFEST.tsv.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import jiwer
import sacrebleu
import transformers
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

OUT = Path("course/fixtures/L14.5/metrics_oracle.json")
GEN = "course/oracle/L14.5/metrics_oracle.py"
VERS = f"jiwer==4.0.0,sacrebleu=={sacrebleu.__version__},transformers=={transformers.__version__}"

NORMALIZE = [
    "Hello, World!",
    '  He said: "don\'t" -- twice...  ',
    "[NOISE] the cat <unk> sat (inaudible) down",
    "Mr. O'Neill paid $3.50 for 2½ apples; 100% sure?",
    "Café crème, naïve façade",
    "ﬁne ﬂour from ＦＵＬＬ　ＷＩＤＴＨ",
    "東京は 晴れ、です。",
    "emoji 🙂 and ♪ notes ♫",
    "tabs\tand\nnewlines\r\nhere",
    "unclosed (paren and [bracket",
    "nested <a [b] c> end",
    "ÀÉÎÕÜ Straße ΣΊΣΥΦΟΣ",
    "a-b_c/d\\e|f~g",
    "",
]

VOCAB = "the a cat dog sat on mat ran far away quick brown fox jumps over lazy".split()


def edits(words: list[str], rng: random.Random) -> list[str]:
    out = list(words)
    for _ in range(rng.randint(0, 4)):
        op = rng.choice("sdi")
        if op == "s" and out:
            out[rng.randrange(len(out))] = rng.choice(VOCAB)
        elif op == "d" and out:
            del out[rng.randrange(len(out))]
        else:
            out.insert(rng.randint(0, len(out)), rng.choice(VOCAB))
    return out


def main() -> None:
    norm = BasicTextNormalizer()
    rng = random.Random(20261012)
    refs, hyps = [], []
    for _ in range(60):
        r = [rng.choice(VOCAB) for _ in range(rng.randint(1, 10))]
        refs.append(" ".join(r))
        hyps.append(" ".join(edits(r, rng)))
    refs += ["the cat sat on the mat", "a quick brown fox", "far away", "dog"]
    hyps += ["the cat sat on the mat", "", "far far away away", "the dog ran"]
    items = []
    for r, h in zip(refs, hyps):
        o = jiwer.process_words(r, h)
        items.append([o.substitutions, o.deletions, o.insertions, len(r.split())])
    cap_preds = [
        "a red circle on the left",
        "two blue squares",
        "a green triangle above a red circle",
        "nothing here",
        "a small yellow star",
    ]
    cap_refs = [
        ["a red circle on the left side", "red circle at left"],
        [
            "two blue squares side by side",
            "a pair of blue squares",
            "two squares, both blue",
        ],
        ["a green triangle is above a red circle", "green triangle over red circle"],
        ["a purple hexagon", "one purple hexagon"],
        ["a small yellow star in the corner", "yellow star, small"],
    ]
    streams = [[r[i] if i < len(r) else r[-1] for r in cap_refs] for i in range(3)]
    data = {
        "generator": GEN,
        "versions": VERS,
        "normalize": [[s, norm(s)] for s in NORMALIZE],
        "asr": {
            "refs": refs,
            "hyps": hyps,
            "wer": jiwer.wer(refs, hyps),
            "cer": jiwer.cer(refs, hyps),
            "items_sdin": items,
        },
        "captions": {
            "preds": cap_preds,
            "refs": cap_refs,
            # sacreBLEU takes reference streams; items with two references
            # repeat their last one in the third stream, which changes
            # neither BLEU (clipped counts, closest length) nor chrF (best ref).
            "bleu4": sacrebleu.corpus_bleu(cap_preds, streams).score,
            "chrf": sacrebleu.corpus_chrf(cap_preds, streams).score,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(data, indent=1, ensure_ascii=False) + "\n").encode()
    OUT.write_bytes(raw)
    print(
        f"{OUT}\t{hashlib.sha256(raw).hexdigest()}\t{len(raw)}\t{GEN}\t{VERS}\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
