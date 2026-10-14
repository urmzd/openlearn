"""Validate the milestone CLI on independent recorded search traces."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path


def main():
    spec = importlib.util.spec_from_file_location(
        "speech_cli", "course/ref/entry/python/tinyllm/cli_speech.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    root = Path("course/fixtures/MS-L14")
    for name in (
        "greedy",
        "beam5",
        "longform-conditioned",
        "longform-partial-tail",
        "longform-unconditioned",
        "longform-fallback-reset",
    ):
        expected = json.loads((root / f"{name}-expected.json").read_text())
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = cli.intercept(
                [
                    "transcribe",
                    "--logits-trace",
                    str(root / f"{name}.npz"),
                    "--opts",
                    str(root / f"{name}.json"),
                ]
            )
        assert code == 0, name
        actual = json.loads(stdout.getvalue())
        assert actual["ids"] == expected["ids"], (name, actual["ids"], expected["ids"])
        assert actual["text"] == expected["text"], name
        if "temperatures" in expected:
            assert [
                segment["temperature"] for segment in actual["segments"]
            ] == expected["temperatures"], name
        print(json.dumps({"trace": name, "ids": actual["ids"], "matches": True}))


if __name__ == "__main__":
    main()
