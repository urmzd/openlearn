"""Whisper tokenizer format boundaries and an actual WAV-to-CLI path."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct

import numpy as np
import pytest
from tinyllm.speech.tokenizer import WhisperTokenizer
from tinyllm.tok.bytes_unicode import bytes_to_unicode


def write_tokenizer(root):
    vocab = {symbol: byte for byte, symbol in bytes_to_unicode().items()}
    labels = [
        "endoftext",
        "startoftranscript",
        "en",
        "fr",
        "translate",
        "transcribe",
        "startoflm",
        "startofprev",
        "nocaptions",
        "notimestamps",
    ]
    labels += [f"{i / 50:.2f}" for i in range(1501)]
    added = [
        {"content": f"<|{label}|>", "id": 256 + i, "special": True}
        for i, label in enumerate(labels)
    ]
    doc = {
        "version": "1.0",
        "normalizer": None,
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False},
        "decoder": {"type": "ByteLevel"},
        "post_processor": {"type": "TemplateProcessing"},
        "added_tokens": added,
    }
    (root / "tokenizer.json").write_text(json.dumps(doc))
    (root / "generation_config.json").write_text(
        json.dumps({"is_multilingual": True, "suppress_tokens": []})
    )
    return doc


def test_whisper_loader_preserves_bytes_and_controls(tmp_path):
    # WHY: Template controls must not be appended to ordinary transcript tokens.
    # KIND: property
    write_tokenizer(tmp_path)
    tok = WhisperTokenizer.from_pretrained(str(tmp_path), language="fr")
    text = "Hello, café 日本語!"
    assert tok.encode(text) == list(text.encode("utf-8"))
    assert tok.decode(tok.encode(text)) == text
    assert tok.all_language_codes == ("en", "fr")
    assert tok.no_speech == 264
    assert tok.timestamp_begin == 266
    assert tok.decode([tok.timestamp_begin, 65]) == "A"
    with pytest.raises(ValueError, match="special token"):
        tok.encode("hello <|startoftranscript|>")


def test_missing_timestamp_or_byte_is_rejected(tmp_path):
    # WHY: Timestamp arithmetic and byte preservation cannot silently degrade.
    # KIND: boundary
    doc = write_tokenizer(tmp_path)
    doc["added_tokens"][-1]["content"] = "<|wrong|>"
    (tmp_path / "tokenizer.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="timestamps"):
        WhisperTokenizer.from_pretrained(str(tmp_path))
    doc = write_tokenizer(tmp_path)
    symbol = bytes_to_unicode()[42]
    doc["model"]["vocab"]["not-a-byte"] = doc["model"]["vocab"].pop(symbol)
    (tmp_path / "tokenizer.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="every byte"):
        WhisperTokenizer.from_pretrained(str(tmp_path))


def test_english_only_and_suppression_defaults(tmp_path):
    # WHY: English checkpoints still contain language tokens but must not use them.
    # KIND: boundary
    write_tokenizer(tmp_path)
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"is_multilingual": False})
    )
    tok = WhisperTokenizer.from_pretrained(str(tmp_path), language="en")
    assert tok.all_language_codes == ()
    assert ord("#") in tok.non_speech_tokens
    with pytest.raises(ValueError, match="unsupported language"):
        WhisperTokenizer.from_pretrained(str(tmp_path), language="fr")


def test_cli_transcribes_real_wav_with_real_model(tmp_path, capsys):
    # WHY: Tokenizer, WAV frontend, model loading and search must work together.
    # KIND: property
    from tinyllm.io.safetensors import load_safetensors, save_safetensors

    fixture = Path(os.environ["TINYLLM_FIXTURES"]) / "L14.2/tiny-whisper-128"
    model_dir = tmp_path / "model"
    shutil.copytree(fixture, model_dir)
    write_tokenizer(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    config["vocab_size"] = 1767
    (model_dir / "config.json").write_text(json.dumps(config))
    weights, metadata = load_safetensors(str(model_dir / "model.safetensors"))
    for key in list(weights):
        if key.endswith("decoder.embed_tokens.weight") or key == "proj_out.weight":
            weights[key] = np.zeros((1767, config["d_model"]), dtype=np.float32)
    save_safetensors(str(model_dir / "model.safetensors"), weights, metadata)
    pcm = np.zeros(320, dtype="<i2").tobytes()
    fmt = struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16)
    body = (
        b"WAVEfmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(pcm))
        + pcm
    )
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    path = (
        Path(__file__).resolve().parents[2] / "ref/entry/python/tinyllm/cli_speech.py"
    )
    spec = importlib.util.spec_from_file_location("speech_cli_test", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    output = tmp_path / "transcript.json"
    assert (
        cli.intercept(
            [
                "transcribe",
                "--model",
                str(model_dir),
                "--audio",
                str(audio),
                "--language",
                "en",
                "--greedy",
                "--no-timestamps",
                "--max-tokens",
                "1",
                "--out",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads(output.read_text())
    assert result["segments"][0]["tokens"] == [0]
    assert result["segments"][0]["end"] == 0.02
    assert cli.intercept(["generate"]) is None


def test_pinned_real_whisper_tokenizer_matches_hf():
    # WHY: A synthetic alphabet cannot expose real checkpoint merge-rank mistakes.
    # KIND: golden
    root = Path(os.environ["TINYLLM_FIXTURES"]) / "L14.4"
    fixture = json.loads((root / "tokenizer_hf.json").read_text())
    tok = WhisperTokenizer.from_pretrained(str(root / "whisper-tokenizer"))
    for case in fixture["cases"]:
        assert tok.encode(case["text"]) == case["ids"]
        assert tok.decode(case["ids"]) == case["decoded"]
    assert tok.sot == 50258
    assert tok.no_speech == 50362
    assert tok.timestamp_begin == 50364
