"""Independent HF tokenization of a pinned official Whisper tokenizer file."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import tokenizers
from tokenizers import Tokenizer

REVISION = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
SHA256 = "27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566"
TEXTS = [
    "",
    "hello world",
    " Hello, world!",
    "I'm WE'RE we've they'll can't",
    "café naïve Straße",
    "日本語 中文 한국어",
    "مرحبا بالعالم",
    "नमस्ते दुनिया",
    "1234567890 1.25 -0.5",
    "a\n\nb\t c  ",
    "x\u00a0y\u2028z",
    "👩🏽‍💻 🎵",
    "(laughter) [speaker] -- yes?",
    "'s 'T 'RE 've",
    "\x00\x01\x1f\x7f",
    "e\u0301 é",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if tokenizers.__version__ != "0.23.3":
        raise ValueError("oracle requires tokenizers==0.23.3")
    root = Path("course/fixtures/L14.4")
    model = root / "whisper-tokenizer"
    path = model / "tokenizer.json"
    if path.exists():
        raw = path.read_bytes()
    else:
        with urlopen(
            f"https://huggingface.co/openai/whisper-tiny/resolve/{REVISION}/tokenizer.json",
            timeout=30,
        ) as response:
            raw = response.read()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("checkpoint tokenizer fingerprint changed")
    tok = Tokenizer.from_str(raw.decode())
    cases = []
    for text in TEXTS:
        ids = tok.encode(text, add_special_tokens=False).ids
        cases.append(
            {
                "text": text,
                "ids": ids,
                "decoded": tok.decode(ids, skip_special_tokens=False),
            }
        )
    result = {
        "revision": REVISION,
        "tokenizer_sha256": SHA256,
        "tokenizers": tokenizers.__version__,
        "cases": cases,
    }
    target = root / "tokenizer_hf.json"
    if args.check:
        if not path.exists() or json.loads(target.read_text()) != result:
            raise ValueError("Whisper tokenizer oracle drift")
    else:
        model.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        target.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        )
    for file in (path, target):
        data = file.read_bytes()
        print(f"{file}\t{len(data)}\t{hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
