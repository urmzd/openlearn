"""Recorded logits must represent an exact, fail-closed decoding history."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from tinyllm.speech.trace import RecordedLogits


def sparse(path, **updates):
    data = dict(
        prefix_offsets=np.array([0, 1, 3]),
        prefix_ids=np.array([1, 1, 2]),
        token_ids=np.array([[2, 3], [4, 5]]),
        values=np.array([[0.0, -1.0], [2.0, 3.0]]),
        vocab_size=np.array(8),
        seeks=np.array([0, 10]),
    )
    data.update(updates)
    np.savez(path, **data)


def test_sparse_seek_and_prefix_identity(tmp_path):
    # WHY: Same-prefix logits from a different audio window are not reusable.
    # KIND: boundary
    path = tmp_path / "trace.npz"
    sparse(path)
    trace = RecordedLogits(str(path))
    row = trace.at_seek(0).step([1])
    assert row[2] == 0 and row[3] == -1 and np.isneginf(row[0])
    row[2] = 100
    assert trace.at_seek(0).step([1])[2] == 0
    assert trace.at_seek(10).step([1, 2])[5] == 3
    with pytest.raises(ValueError, match="no logits"):
        trace.at_seek(0).step([1, 2])


@pytest.mark.parametrize(
    "updates",
    [
        {"prefix_offsets": np.array([0, 0, 3])},
        {"prefix_ids": np.array([1.0, 1.0, 2.0])},
        {"token_ids": np.array([[2, 2], [4, 5]])},
        {"values": np.array([[np.nan, 0], [1, 2]])},
        {"values": np.array([[-np.inf, -np.inf], [1, 2]])},
        {"seeks": np.array([-1, 10])},
    ],
)
def test_bad_sparse_recordings_rejected(tmp_path, updates):
    # WHY: Broken traces must not be repaired into plausible model outputs.
    # KIND: boundary
    path = tmp_path / "trace.npz"
    sparse(path, **updates)
    with pytest.raises(ValueError):
        RecordedLogits(str(path))


def test_dense_rows_and_duplicate_prefix_validation(tmp_path):
    # WHY: Dense and sparse storage share the same unambiguous prefix contract.
    # KIND: property
    path = tmp_path / "trace.npz"
    np.savez(
        path,
        prefix_offsets=np.array([0, 1]),
        prefix_ids=np.array([1]),
        logits=np.array([[1.0, 2.0]]),
    )
    window = RecordedLogits(str(path)).at_seek(0)
    row = window.step([1])
    assert row.tolist() == [1.0, 2.0]
    row[:] = 0
    assert window.step([1]).tolist() == [1.0, 2.0]
    np.savez(
        path,
        prefix_offsets=np.array([0, 1, 2]),
        prefix_ids=np.array([1, 1]),
        logits=np.ones((2, 2)),
    )
    with pytest.raises(ValueError, match="duplicate"):
        RecordedLogits(str(path))


def test_cli_replays_sparse_logits_without_model_weights(tmp_path, capsys):
    # WHY: The parity CLI must invoke the real decoding policy without a model.
    # KIND: golden
    helper_path = Path(__file__).with_name("test_speech_tokenizer.py")
    spec = importlib.util.spec_from_file_location(
        "trace_tokenizer_fixture", helper_path
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.write_tokenizer(tmp_path)
    # SOT silence row, initial text row, and EOT after the emitted ASCII A.
    prefixes = [[257], [257, 258, 261, 265], [257, 258, 261, 265, 65]]
    np.savez(
        tmp_path / "trace.npz",
        prefix_offsets=np.array([0, 1, 5, 10]),
        prefix_ids=np.array(sum(prefixes, [])),
        vocab_size=np.array(1767),
        token_ids=np.array([[264, 65], [65, 256], [256, 65]]),
        values=np.array([[-20.0, 0.0], [0.0, -20.0], [0.0, -20.0]]),
    )
    opts = {
        "tokenizer": ".",
        "duration": 0.1,
        "window_frames": 10,
        "decode": {"language": "en", "timestamps": False, "temperatures": [0.0]},
    }
    (tmp_path / "opts.json").write_text(json.dumps(opts))
    cli_path = (
        Path(__file__).resolve().parents[2] / "ref/entry/python/tinyllm/cli_speech.py"
    )
    spec = importlib.util.spec_from_file_location("trace_cli", cli_path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert (
        cli.intercept(
            [
                "transcribe",
                "--logits-trace",
                str(tmp_path / "trace.npz"),
                "--opts",
                str(tmp_path / "opts.json"),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["text"] == "A"
    assert result["segments"][0]["tokens"] == [65]
    assert result["segments"][0]["end"] == 0.1
