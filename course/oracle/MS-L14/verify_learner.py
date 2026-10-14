"""Compare the real learner WAV path with independently generated HF tokens."""

import argparse
import json
from pathlib import Path
import time

from tinyllm.io.wav import read_wav
from tinyllm.speech.decode import DecodeOptions, transcribe
from tinyllm.speech.decoder import Whisper
from tinyllm.speech.frontend import FeatureConfig, to_mono_16k
from tinyllm.speech.tokenizer import WhisperTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--oracle", required=True)
    args = parser.parse_args()
    expected = json.loads(Path(args.oracle).read_text())
    model = Whisper.from_pretrained(args.model_dir)
    tok = WhisperTokenizer.from_pretrained(args.model_dir, language="en")
    cfg = FeatureConfig(feature_size=model.config.n_mels)
    options = DecodeOptions(language="en", temperatures=(0.0,), max_tokens=128)
    failures = []
    for case in expected["cases"]:
        if case["minimum_margin"] is None or case["minimum_margin"] < 0.001:
            raise ValueError(f"oracle clip lacks required margin: {case['clip']}")
        audio = to_mono_16k(
            read_wav((Path("course/fixtures/L14.1/clips") / case["clip"]).read_bytes()),
            cfg,
        )
        started = time.monotonic()
        segments = transcribe(model, tok, audio, options)
        ids = [token for segment in segments for token in segment.tokens]
        matches = ids == case["ids"]
        print(
            json.dumps(
                {
                    "clip": case["clip"],
                    "matches": matches,
                    "seconds": time.monotonic() - started,
                    "actual": ids,
                    "expected": case["ids"],
                }
            ),
            flush=True,
        )
        if not matches:
            failures.append(case["clip"])
    if failures:
        raise AssertionError(f"learner timestamp tokens differ: {failures}")


if __name__ == "__main__":
    main()
