"""Independent real-audio WER oracle using pinned OpenAI Whisper, not learner code."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import soundfile as sf
import torch
import whisper
from whisper.normalizers import BasicTextNormalizer

REVISION = "86098128c0b4f24f0e2aa2994de830614b474227"
NORMALIZE = BasicTextNormalizer()


def edits(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(hypothesis, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right))
            )
        previous = current
    return previous[-1]


def score(cases):
    words = sum(len(NORMALIZE(c["reference"]).split()) for c in cases)
    errors = sum(
        edits(NORMALIZE(c["reference"]).split(), NORMALIZE(c["text"]).split())
        for c in cases
    )
    if not words:
        raise ValueError("empty normalized references")
    return {"wer": errors / words, "errors": errors, "reference_words": words}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-cache", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = whisper.load_model("tiny", device="cpu", download_root=args.model_cache)
    rows = [
        json.loads(line)
        for line in (args.data / "manifest.jsonl").read_text().splitlines()
    ]
    rows.append(
        {
            "audio": "longform-90s.wav",
            "text": (args.data / "longform-90s.txt").read_text(),
        }
    )
    cases = []
    for row in rows:
        path = args.data / row["audio"]
        samples, rate = sf.read(path, dtype="float32")
        assert rate == 16000 and samples.ndim == 1
        torch.manual_seed(0)
        started = time.monotonic()
        result = model.transcribe(
            samples,
            language="en",
            fp16=False,
            verbose=None,
            beam_size=5 if row["audio"] == "longform-90s.wav" else None,
            best_of=5,
            sample_len=224,
        )
        case = {
            "audio": row["audio"],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "reference": row["text"],
            "text": result["text"],
            "seconds": time.monotonic() - started,
            "audio_seconds": len(samples) / rate,
            "segments": result["segments"],
        }
        cases.append(case)
        print(
            json.dumps(
                {"audio": case["audio"], "seconds": case["seconds"], **score([case])}
            ),
            flush=True,
        )
    output = {
        "revision": REVISION,
        "model_sha256": whisper._MODELS["tiny"].split("/")[-2],
        "normalizer": "whisper-basic",
        "corpus": score(cases[:-1]),
        "longform": score(cases[-1:]),
        "cases": cases,
    }
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
