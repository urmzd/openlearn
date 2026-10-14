"""Exercise public learner CLI accuracy gates against the independent oracle."""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads(args.oracle.read_text())
    for case in expected["cases"]:
        assert (
            hashlib.sha256((args.data / case["audio"]).read_bytes()).hexdigest()
            == case["sha256"]
        )
    spec = importlib.util.spec_from_file_location(
        "speech_cli", "course/ref/entry/python/tinyllm/cli_speech.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    def invoke(argv):
        stdout = io.StringIO()
        started = time.monotonic()
        with contextlib.redirect_stdout(stdout):
            code = cli.intercept(argv)
        assert code == 0, (argv, code, stdout.getvalue())
        return json.loads(stdout.getvalue()), time.monotonic() - started

    corpus, corpus_seconds = invoke(
        [
            "eval",
            "asr",
            "--model",
            args.model_dir,
            "--data",
            str(args.data / "manifest.jsonl"),
            "--language",
            "en",
            "--seed",
            "0",
        ]
    )
    prediction, longform_seconds = invoke(
        [
            "transcribe",
            "--model",
            args.model_dir,
            "--audio",
            str(args.data / "longform-90s.wav"),
            "--language",
            "en",
            "--beam",
            "5",
        ]
    )
    prediction_path = args.out.with_suffix(".segments.json")
    prediction_path.write_text(json.dumps(prediction))
    longform, _ = invoke(
        [
            "eval",
            "asr",
            "--pred",
            str(prediction_path),
            "--ref",
            str(args.data / "longform-90s.txt"),
        ]
    )
    report = {
        "corpus": corpus,
        "longform": longform,
        "corpus_seconds": corpus_seconds,
        "longform_seconds": longform_seconds,
        "longform_rtf": longform_seconds / 90,
        "oracle_corpus": expected["corpus"],
        "oracle_longform": expected["longform"],
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    assert corpus["items"] == 24
    assert len(corpus["wer_ci"]) == 2 and all(
        math.isfinite(x) for x in corpus["wer_ci"]
    )
    assert 0 <= corpus["wer_ci"][0] <= corpus["wer_ci"][1]
    for label, actual in (("corpus", corpus), ("longform", longform)):
        assert actual["reference_words"] == expected[label]["reference_words"], label
        assert (
            math.isfinite(actual["wer"])
            and actual["wer"] <= expected[label]["wer"] + 0.01
        ), label
    assert prediction["segments"], "empty long-form output"
    for segment in prediction["segments"]:
        assert 0 <= segment["start"] <= segment["end"] <= 90
        assert isinstance(segment["text"], str)


if __name__ == "__main__":
    main()
