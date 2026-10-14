"""L14.3: full-window and incremental Whisper must agree with independent HF logits."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from tinyllm.io.hf_whisper import read_whisper, whisper_state_from_hf
from tinyllm.io.safetensors import load_safetensors, save_safetensors
from tinyllm.speech.decoder import CrossKV, Whisper

FIX = Path(os.environ["TINYLLM_FIXTURES"])


def model():
    return Whisper.from_pretrained(str(FIX / "L14.2/tiny-whisper-128"))


def features():
    return np.load(FIX / "L14.2/encoder_hf.npz")["small.mel"]


def test_hf_logits_and_parameter_count():
    # WHY: Shared mistakes in cached and uncached paths must not agree unnoticed.
    # KIND: golden
    m = model()
    f = np.load(FIX / "L14.3/decoder_hf.npz")
    assert_close(m(features(), f["ids"]).data, f["logits"], atol=1e-5, rtol=0)
    assert m.param_count() == int(f["parameter_count"])


def test_cached_chunks_equal_full_and_reuse_cross_kv():
    # WHY: Positions must advance once per token, not once per decoder layer.
    # KIND: property
    m = model()
    kv = m.encode(features())
    ids = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    full = m.decoder(ids, kv).data
    cache = m.new_cache(2)
    first = m.decoder(ids[:, :2], kv, cache).data
    last = m.decoder(ids[:, 2:], kv, cache).data
    assert cache.seq_len() == 4
    assert_close(np.concatenate([first, last], axis=1), full, atol=1e-5, rtol=0)
    saved = [x.copy() for x in kv.k + kv.v]
    m.decoder(ids, kv)
    for a, b in zip(kv.k + kv.v, saved):
        assert np.array_equal(a, b)


def test_causality_and_one_dimensional_ids():
    # WHY: Future decoder tokens must never influence a prefix logit.
    # KIND: property
    m = model()
    kv = m.encode(features()[:1])
    a = m.decoder(np.array([1, 2, 3]), kv).data
    b = m.decoder(np.array([1, 9, 8]), kv).data
    assert a.shape == (3, m.config.vocab)
    assert_close(a[:1], b[:1], atol=1e-5, rtol=0)
    assert_close(a, m.decoder(np.array([[1, 2, 3]]), kv).data[0], atol=0, rtol=0)


@pytest.mark.parametrize(
    "ids",
    [
        np.array([1.5]),
        np.array([-1]),
        np.array([50]),
        np.array([]),
        np.zeros((1, 1, 1), dtype=int),
        np.ones(17, dtype=int),
    ],
)
def test_bad_tokens_and_positions_are_rejected(ids):
    # WHY: Invalid requests must fail before changing a decoding cache.
    # KIND: boundary
    m = model()
    cache = m.new_cache()
    with pytest.raises(ValueError):
        m.decoder(ids, m.encode(features()[:1]), cache)
    assert cache.seq_len() == 0


def test_cross_kv_and_cache_shapes_are_validated():
    # WHY: Reusing another request's cache must not silently mix audio or batch rows.
    # KIND: boundary
    m = model()
    kv = m.encode(features())
    with pytest.raises(ValueError):
        m.decoder([1], kv)
    with pytest.raises(ValueError):
        m.decoder([[1], [2]], CrossKV([], []))
    with pytest.raises(ValueError):
        m.decoder([[1], [2]], kv, m.new_cache(1))
    with pytest.raises(ValueError):
        m.decoder.cross_kv(np.zeros((1, 8, 15), dtype=np.float32))


def test_loader_rejects_untied_weights_and_wrong_sinusoids():
    # WHY: Silent HF name or position mismatches can produce plausible wrong logits.
    # KIND: boundary
    m = model()
    tensors, _ = load_safetensors(str(FIX / "L14.2/tiny-whisper-128/model.safetensors"))
    bad = dict(tensors, **{"proj_out.weight": np.zeros((50, 16), dtype=np.float32)})
    with pytest.raises(ValueError, match="tied"):
        whisper_state_from_hf(bad, m.config)
    bad = dict(tensors)
    bad["model.encoder.embed_positions.weight"] = np.zeros((8, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="sinusoids"):
        whisper_state_from_hf(bad, m.config)
    with pytest.raises(ValueError, match="name"):
        whisper_state_from_hf({"unexpected": np.zeros(1)}, m.config)


def test_sharded_loader_and_index_integrity(tmp_path):
    # WHY: Shards must follow the index exactly and stay inside the checkpoint root.
    # KIND: conformance
    root = FIX / "L14.2/tiny-whisper-128"
    (tmp_path / "config.json").write_bytes((root / "config.json").read_bytes())
    tensors, _ = load_safetensors(str(root / "model.safetensors"))
    keys = sorted(tensors)
    groups = [keys[::2], keys[1::2]]
    mapping = {}
    for i, names in enumerate(groups):
        name = f"part-{i}.safetensors"
        save_safetensors(str(tmp_path / name), {k: tensors[k] for k in names}, {})
        mapping.update({k: name for k in names})
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": mapping}))
    loaded = Whisper.from_pretrained(str(tmp_path))
    assert loaded.param_count() == model().param_count()
    for k, v in loaded.state_dict().items():
        assert np.array_equal(v, model().state_dict()[k])
    mapping[keys[0]] = "../escape.safetensors"
    index.write_text(json.dumps({"weight_map": mapping}))
    with pytest.raises(ValueError, match="escapes"):
        read_whisper(str(tmp_path))


def test_loader_is_strict_about_missing_and_wrong_shapes(tmp_path):
    # WHY: Partial checkpoint loads must not leave random parameters in a served model.
    # KIND: boundary
    root = FIX / "L14.2/tiny-whisper-128"
    (tmp_path / "config.json").write_bytes((root / "config.json").read_bytes())
    tensors, _ = load_safetensors(str(root / "model.safetensors"))
    missing = dict(tensors)
    missing.pop("model.decoder.layers.0.fc1.weight")
    save_safetensors(str(tmp_path / "model.safetensors"), missing, {})
    with pytest.raises(KeyError, match="missing"):
        Whisper.from_pretrained(str(tmp_path))
    wrong = dict(tensors)
    wrong["model.decoder.layers.0.fc1.weight"] = np.zeros((1, 1), dtype=np.float32)
    save_safetensors(str(tmp_path / "model.safetensors"), wrong, {})
    with pytest.raises(ValueError, match="shape"):
        Whisper.from_pretrained(str(tmp_path))


def test_wav_entry_point_matches_explicit_frontend():
    # WHY: Raw stereo audio must traverse the frontend before either attention tower.
    # KIND: property
    import io
    import wave

    from tinyllm.speech.frontend import FeatureConfig, pad_or_trim, whisper_log_mel

    m = model()
    pcm = np.stack([np.arange(400), -np.arange(400) // 2], axis=1).astype("<i2")
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())
    cfg = FeatureConfig(feature_size=m.config.n_mels)
    mono = pcm.astype(np.float64).mean(axis=1) / 32768
    mel = whisper_log_mel(pad_or_trim(mono, 2 * m.config.n_audio_ctx * 160), cfg)
    expected = m.encode(mel[None].astype(np.float32))
    actual = m.encode_wav(out.getvalue())
    for a, b in zip(actual.k + actual.v, expected.k + expected.v):
        assert_close(a, b, atol=1e-6, rtol=0)
    assert_close(
        m.decoder([1, 2], actual).data,
        m.decoder([1, 2], expected).data,
        atol=1e-6,
        rtol=0,
    )
    with pytest.raises(ValueError, match="RIFF"):
        m.encode_wav(b"invalid")


def test_loader_preserves_half_precision_position_table():
    # WHY: Published Whisper weights round fixed sinusoids to float16 before storage.
    # KIND: boundary
    from tinyllm.speech.encoder import WhisperConfig, whisper_sinusoids

    cfg = WhisperConfig(d=384, n_audio_ctx=1500)
    table = (
        whisper_sinusoids(cfg.n_audio_ctx, cfg.d).astype(np.float16).astype(np.float32)
    )
    assert np.max(np.abs(table - whisper_sinusoids(cfg.n_audio_ctx, cfg.d))) > 1e-4
    _, positions = whisper_state_from_hf(
        {"model.encoder.embed_positions.weight": table}, cfg
    )
    np.testing.assert_array_equal(positions, table)
    positions[0, 0] = 1
    assert table[0, 0] == 0
    table[0, 0] = 0.01
    with pytest.raises(ValueError, match="sinusoids"):
        whisper_state_from_hf({"model.encoder.embed_positions.weight": table}, cfg)
