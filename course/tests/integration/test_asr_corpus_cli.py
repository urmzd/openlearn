"""Cross-module ASR corpus command: inference orchestration and pooled intervals."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_corpus_infers_once_per_item_and_pools_words(tmp_path, capsys, monkeypatch):
    from tinyllm.speech import decode, decoder, tokenizer
    from tinyllm.io import wav
    from tinyllm.speech import frontend

    path = (
        Path(__file__).resolve().parents[2] / "ref/entry/python/tinyllm/cli_speech.py"
    )
    spec = importlib.util.spec_from_file_location("corpus_cli_test", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    calls = []
    model = SimpleNamespace(config=SimpleNamespace(vocab=10, n_mels=80))

    def load(*args):
        calls.append("load")
        return model

    monkeypatch.setattr(decoder.Whisper, "from_pretrained", load)
    monkeypatch.setattr(
        tokenizer.WhisperTokenizer,
        "from_pretrained",
        lambda *a, **kw: SimpleNamespace(bpe=SimpleNamespace(vocab_size=10)),
    )
    monkeypatch.setattr(wav, "read_wav", lambda data: data)
    monkeypatch.setattr(frontend, "to_mono_16k", lambda data, cfg: data)

    def transcribe(m, tok, audio, opts):
        assert m is model
        calls.append(audio)
        return [SimpleNamespace(text=audio.decode())]

    monkeypatch.setattr(decode, "transcribe", transcribe)
    (tmp_path / "a.wav").write_bytes(b"wrong")
    (tmp_path / "b.wav").write_bytes(b"two three four")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"audio": "a.wav", "text": "one"},
                {"audio": "b.wav", "text": "two three four"},
            ]
        )
    )
    argv = [
        "eval",
        "asr",
        "--model",
        "fixture",
        "--data",
        str(manifest),
        "--bootstrap",
        "100",
        "--seed",
        "7",
    ]
    assert cli.intercept(argv) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["wer"] == pytest.approx(0.25)
    assert result["wer_ci"] == [0.0, 1.0]
    assert result["items"] == 2
    assert calls == ["load", b"wrong", b"two three four"]
    assert cli.intercept(argv) == 0
    assert json.loads(capsys.readouterr().out) == result
    calls.clear()
    manifest.write_text('{"audio": "missing.wav", "text": "one"}')
    assert cli.intercept(argv) == 2
    assert calls == []
    assert "corpus audio missing" in capsys.readouterr().err
