"""Check the public learner language-detection CLI on attributed MLS excerpts."""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "speech_cli", "course/ref/entry/python/tinyllm/cli_speech.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    root = Path("course/fixtures/whisper-parity")
    cases = json.loads((root / "languages.json").read_text())
    assert {c["language"] for c in cases} == {"de", "es", "fr"}
    for case in cases:
        path = root / case["audio"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == case["sha256"]
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = cli.intercept(
                [
                    "transcribe",
                    "--model",
                    args.model_dir,
                    "--audio",
                    str(path),
                    "--detect-language",
                ]
            )
        assert code == 0
        result = json.loads(stdout.getvalue())
        print(json.dumps({"expected": case["language"], **result}), flush=True)
        assert result["language"] == case["language"]


if __name__ == "__main__":
    main()
