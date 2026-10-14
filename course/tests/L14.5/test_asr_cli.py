"""ASR milestone scoring through the learner entry point."""

import importlib.util
import json
from pathlib import Path

import pytest


def test_cli_asr_scores_generated_segments(tmp_path, capsys):
    # WHY: Segment files must be scored as one normalized transcript, not averaged per segment.
    # KIND: golden
    path = (
        Path(__file__).resolve().parents[2] / "ref/entry/python/tinyllm/cli_speech.py"
    )
    spec = importlib.util.spec_from_file_location("speech_eval_cli_test", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    prediction, reference = tmp_path / "segments.json", tmp_path / "reference.txt"
    reference.write_text("The cat sat on the mat.")
    prediction.write_text(json.dumps([{"text": "THE dog"}, {"text": "sat on mat!"}]))
    argv = ["eval", "asr", "--pred", str(prediction), "--ref", str(reference)]
    assert cli.intercept(argv) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["wer"] == pytest.approx(2 / 6)
    assert result["substitutions"] == 1
    assert result["deletions"] == 1
    assert result["reference_words"] == 6
    prediction.write_text('{"segments": [{"text": 4}]}')
    assert cli.intercept(argv) == 2
    assert "each segment must contain text" in capsys.readouterr().err
