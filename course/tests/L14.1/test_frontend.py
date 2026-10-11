"""Course tests for L14.1: read_wav and write_wav (tinyllm/io/wav.py), and
FeatureConfig, to_mono_16k, pad_or_trim, whisper_log_mel
(tinyllm/speech/frontend.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/L14.1), and the chapter section it comes from.

The chapter's worked example (section 3) is a 52-byte stereo WAV written
out by hand: two frames of s16, left 16384 and -32768, right -1 and 32767,
which read back as [[0.5, -1.0], [-1/32768, 32767/32768]]; and a 30 s
window of silence, whose 3000 log-mel frames are all -1.5.

Fixtures are in $TINYLLM_FIXTURES/L14.1, written by
course/oracle/L14.1/frontend_hf.py: hand-built WAV files read by
scipy.io.wavfile (wav_scipy.npz), six LibriSpeech test-clean clips (CC BY
4.0, attribution in clips/clips.json), and HF WhisperFeatureExtractor
features of each clip with 80 and 128 mels (hf_features.npz).
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.io.wav import Wav, read_wav, write_wav
from tinyllm.sig.resample import resample_audio
from tinyllm.speech.frontend import (
    FeatureConfig,
    pad_or_trim,
    to_mono_16k,
    whisper_log_mel,
)

FIX = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "L14.1"
GOOD = ["s16-mono", "s24-mono", "f32-stereo", "s16-list-odd", "ext-s24-stereo"]


def rng(seq: int) -> PCG32:
    return PCG32(int(os.environ.get("OL_SEED", "0")), seq=seq)


def chunk(cid: bytes, payload: bytes) -> bytes:
    return (
        cid + struct.pack("<I", len(payload)) + payload + b"\x00" * (len(payload) % 2)
    )


def fmt(tag=1, ch=1, sr=16000, bits=16, block=None, rate=None) -> bytes:
    block = ch * bits // 8 if block is None else block
    rate = sr * block if rate is None else rate
    return chunk(b"fmt ", struct.pack("<HHIIHH", tag, ch, sr, rate, block, bits))


def riff(*chunks: bytes) -> bytes:
    body = b"WAVE" + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def test_hand_example_wav_header():
    # WHY: section 3 builds this file byte by byte: RIFF header, a 16-byte
    #      fmt chunk, a data chunk of two interleaved stereo frames. Reading
    #      it pins the little-endian fields, the s16 scale 1/32768, and the
    #      interleaving L R L R into [channels, frames].
    # KIND: unit, smoke
    # CATCHES: s04, s05
    # CHAPTER: L14.1 section 3
    data = b"".join(
        [
            b"RIFF",
            struct.pack("<I", 44),
            b"WAVE",
            b"fmt ",
            struct.pack("<IHHIIHH", 16, 1, 2, 16000, 64000, 4, 16),
            b"data",
            struct.pack("<I", 8),
            struct.pack("<hhhh", 16384, -1, -32768, 32767),
        ]
    )
    assert len(data) == 52
    w = read_wav(data)
    assert (w.sr, w.channels, w.fmt) == (16000, 2, "s16")
    assert w.samples.dtype == np.float32 and w.samples.shape == (2, 2)
    assert_close(
        w.samples,
        np.array([[0.5, -1.0], [-1 / 32768, 32767 / 32768]], dtype=np.float32),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("name", GOOD)
def test_reads_fixture_wavs_like_scipy(name):
    # WHY: s24 needs sign extension of three bytes, f32 an 18-byte fmt
    #      chunk and a fact chunk to skip, the LIST file odd-sized chunks
    #      with their pad bytes, and the extensible file its subformat GUID
    #      and trailing bytes after the RIFF size; scipy.io.wavfile decodes
    #      every one of them, so the samples must agree exactly.
    # KIND: golden
    # CATCHES: s01, s02, s03, s04, s05
    # CHAPTER: L14.1 section 2.1
    z = np.load(FIX / "wav_scipy.npz")
    w = read_wav((FIX / "wav" / f"{name}.wav").read_bytes())
    want = z[f"{name}.samples"]
    assert w.sr == int(z[f"{name}.sr"])
    assert w.channels == want.shape[0]
    assert w.samples.dtype == np.float32
    assert_close(w.samples, want, rtol=0.0, atol=0.0)
    kinds = {"s16-mono": "s16", "s24-mono": "s24", "f32-stereo": "f32"}
    kinds |= {"s16-list-odd": "s16", "ext-s24-stereo": "s24"}
    assert w.fmt == kinds[name]


def test_rejects_broken_files():
    # WHY: size fields are where a truncated or hostile file lies. A data
    #      chunk longer than the file, a RIFF size of about 4 GiB, a missing
    #      or late fmt chunk, an encoding we do not support, three channels,
    #      a zero sample rate, an inconsistent block_align, and a partial
    #      frame must each fail loudly instead of returning garbage.
    # KIND: fault
    # CATCHES: s12, m01, m02, m03
    # CHAPTER: L14.1 section 2.1
    for name in ("trunc-data", "riff-too-big"):
        with pytest.raises(ValueError):
            read_wav((FIX / "wav" / f"{name}.wav").read_bytes())
    pcm = chunk(b"data", b"\x00\x01" * 8)
    bad = {
        "not riff": b"RIFX" + riff(fmt(), pcm)[4:],
        "no fmt": riff(pcm),
        "data first": riff(pcm, fmt()),
        "no data": riff(fmt()),
        "8-bit": riff(fmt(bits=8), chunk(b"data", b"\x80" * 8)),
        "float 64": riff(fmt(tag=3, bits=64), chunk(b"data", b"\x00" * 16)),
        "3 channels": riff(fmt(ch=3), chunk(b"data", b"\x00" * 12)),
        "rate 0": riff(fmt(sr=0, rate=0), pcm),
        "rate too high": riff(fmt(sr=384001), pcm),
        "block align": riff(fmt(block=3), pcm),
        "byte rate": riff(fmt(rate=12345), pcm),
        "partial frame": riff(fmt(ch=2), chunk(b"data", b"\x00" * 6)),
        "short fmt": riff(chunk(b"fmt ", b"\x01\x00\x01\x00"), pcm),
        "short header": b"RIFF\x04\x00",
    }
    ok = riff(fmt(), pcm)
    bad["RIFF size 2 past the end"] = ok[:4] + struct.pack("<I", len(ok) - 6) + ok[8:]
    for why, data in bad.items():
        with pytest.raises(ValueError):
            read_wav(data)
            pytest.fail(f"accepted a file with {why}")
    # The edges of the accepted range still parse.
    assert read_wav(riff(fmt(sr=384000), pcm)).sr == 384000
    assert read_wav(riff(fmt(sr=1), pcm)).sr == 1
    assert read_wav(riff(fmt(ch=2), chunk(b"data", b"\x00" * 8))).channels == 2


def test_write_wav_round_trip():
    # WHY: write_wav is how every later step (and the engine's fixtures)
    #      hands audio to a WAV reader. Its output must be a plain 44-byte
    #      header s16 mono file, quantized by round half to even of x 2^15
    #      and clipped, so a read gives back the quantized samples exactly.
    # KIND: property
    # CATCHES: s13, m04
    # CHAPTER: L14.1 section 2.1
    r = rng(1)
    x = (r.uniform_array(997, -1.2, 1.2)).astype(np.float64)  # some samples clip
    data = write_wav(x, 22050)
    assert data[:4] == b"RIFF" and data[8:16] == b"WAVEfmt "
    assert len(data) == 44 + 2 * x.size
    w = read_wav(data)
    assert (w.sr, w.channels, w.fmt) == (22050, 1, "s16")
    q = np.clip(np.round(x * 32768.0), -32768, 32767) / 32768.0
    assert_close(w.samples[0], q.astype(np.float32), rtol=0.0, atol=0.0)
    # Already-quantized samples survive a second trip unchanged.
    assert (
        read_wav(write_wav(w.samples[0], 22050)).samples.tobytes()
        == w.samples.tobytes()
    )
    # Halves round to even: 0.5 / 32768 -> 0, 1.5 / 32768 -> 2.
    w2 = read_wav(write_wav(np.array([0.5, 1.5, -0.5]) / 32768.0, 16000))
    assert_close(
        w2.samples[0], np.array([0.0, 2.0, 0.0], np.float32) / 32768, rtol=0.0, atol=0.0
    )
    for bad in (np.zeros((2, 3)), np.array([0.0, np.nan])):
        with pytest.raises(ValueError):
            write_wav(bad, 16000)
    with pytest.raises(ValueError):
        write_wav(np.zeros(3), 0)


@pytest.mark.parametrize("n_mels", [80, 128])
@pytest.mark.parametrize("i", range(6))
def test_log_mel_matches_hf_feature_extractor(i, n_mels):
    # WHY: the encoder was trained on HF's features; parity within 1e-4 on
    #      real speech, through the whole path (WAV bytes, mono, pad to 30 s,
    #      STFT, drop the last frame, mel, log10, clip, rescale), is the
    #      contract L14.2 and the engine (L15.1) build on.
    # KIND: golden
    # CATCHES: s06, s07, s08, m07
    # CHAPTER: L14.1 section 2.3
    cfg = FeatureConfig(feature_size=n_mels)
    w = read_wav((FIX / "clips" / f"en-{i}.wav").read_bytes())
    feats = whisper_log_mel(pad_or_trim(to_mono_16k(w, cfg), cfg.n_samples), cfg)
    want = np.load(FIX / "hf_features.npz")[f"en-{i}.f{n_mels}"]
    assert feats.shape == (n_mels, 3000) and feats.dtype == np.float32
    assert_close(feats, want, rtol=0.0, atol=1e-4)


def test_silence_gives_the_floor_everywhere():
    # WHY: section 3: zero power floors at log10(1e-10) = -10 in every band
    #      and frame, the clip at max - 8 changes nothing, and (-10 + 4) / 4
    #      = -1.5. A 30 s window is 480000 samples, 3001 STFT frames, 3000
    #      kept.
    # KIND: property
    # CATCHES: s06, s07
    # CHAPTER: L14.1 section 3
    cfg = FeatureConfig()
    assert cfg.n_samples == 480000
    feats = whisper_log_mel(pad_or_trim(np.zeros(0)), cfg)
    assert feats.shape == (80, 3000)
    assert np.all(feats == np.float32(-1.5))


def test_clip_uses_the_kept_frames_only():
    # WHY: Whisper drops the last STFT frame BEFORE taking the maximum the
    #      8-unit clip is measured from. A loud burst at the very end of the
    #      signal is loudest in the dropped frame, so taking the maximum
    #      first moves every clipped value. Shapes: n samples give n // hop
    #      frames.
    # KIND: unit
    # CATCHES: s06, s07, s09
    # CHAPTER: L14.1 section 2.3
    cfg = FeatureConfig()
    n = 1600
    x = np.zeros(n)
    t = np.arange(100)
    x[-100:] = 0.8 * np.sin(2 * np.pi * 1000.0 * t / 16000.0)
    feats = whisper_log_mel(x, cfg)
    assert feats.shape == (80, 10)
    # Recompute by hand with the reference recipe from M12.5 and M12.6.
    from tinyllm.sig.mel import mel_filterbank
    from tinyllm.sig.stft import hann, power, stft

    p = power(stft(x, 400, 160, hann(400)))
    mel = mel_filterbank(16000, 400, 80) @ p[:, :-1]
    L = np.log10(np.maximum(mel, 1e-10))
    want = (np.maximum(L, L.max() - 8.0) + 4.0) / 4.0
    assert_close(feats, want.astype(np.float32), rtol=0.0, atol=1e-5)
    # The dropped frame really is the loudest, so the order matters here.
    Lall = np.log10(np.maximum(mel_filterbank(16000, 400, 80) @ p, 1e-10))
    assert Lall.max() > L.max() + 0.1


def test_to_mono_averages_channels_then_resamples():
    # WHY: Whisper hears one channel at 16 kHz. Stereo is averaged (not the
    #      left channel alone), and other rates go through M12.3's
    #      band-limited resampler; at 16 kHz the samples pass through as a
    #      copy.
    # KIND: property
    # CATCHES: s10
    # CHAPTER: L14.1 section 2.2
    cfg = FeatureConfig()
    w = read_wav((FIX / "wav" / "f32-stereo.wav").read_bytes())
    got = to_mono_16k(w, cfg)
    mean = w.samples.astype(np.float64).mean(axis=0)
    assert_close(got, resample_audio(mean, 22050, 16000), rtol=0.0, atol=1e-12)
    assert got.size == int(np.ceil(w.samples.shape[1] * 16000 / 22050))
    r = rng(2)
    a = r.uniform_array(64).astype(np.float32)
    b = r.uniform_array(64).astype(np.float32)
    w2 = Wav(samples=np.stack([a, b]), sr=16000, channels=2, fmt="f32")
    before = w2.samples.copy()
    m = to_mono_16k(w2, cfg)
    assert_close(m, (a.astype(np.float64) + b) / 2, rtol=0.0, atol=1e-12)
    m[:] = 7.0
    assert np.array_equal(w2.samples, before)


def test_pad_or_trim():
    # WHY: every window is exactly n_samples long: short audio is padded
    #      with zeros at the END (the encoder sees speech from frame 0),
    #      long audio is cut. The input is never modified.
    # KIND: unit
    # CATCHES: s11, m05
    # CHAPTER: L14.1 section 2.2
    x = np.arange(1.0, 6.0)
    assert_close(pad_or_trim(x, 8), [1, 2, 3, 4, 5, 0, 0, 0], rtol=0.0, atol=0.0)
    assert_close(pad_or_trim(x, 3), [1, 2, 3], rtol=0.0, atol=0.0)
    assert_close(pad_or_trim(x, 5), x, rtol=0.0, atol=0.0)
    y = pad_or_trim(x, 5)
    y[0] = 9.0
    assert x[0] == 1.0
    assert pad_or_trim(np.zeros(3)).shape == (480000,)
    with pytest.raises(ValueError):
        pad_or_trim(np.zeros((2, 3)), 4)
    with pytest.raises(ValueError):
        pad_or_trim(x, 0)
    assert pad_or_trim(x, 1).shape == (1,)


def test_feature_config_from_hf(tmp_path):
    # WHY: the frontend's numbers come from the checkpoint's
    #      preprocessor_config.json (large-v3 uses 128 mels); unknown keys,
    #      including the 80 x 201 mel_filters table HF ships, are ignored,
    #      and a config for another processor is refused.
    # KIND: unit
    # CATCHES: m06
    # CHAPTER: L14.1 section 4
    raw = {
        "feature_extractor_type": "WhisperFeatureExtractor",
        "feature_size": 128,
        "sampling_rate": 16000,
        "hop_length": 160,
        "n_fft": 400,
        "chunk_length": 30,
        "padding_value": 0.0,
        "mel_filters": [[0.0] * 201] * 2,
        "processor_class": "WhisperProcessor",
    }
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(raw))
    for src in (tmp_path, tmp_path / "preprocessor_config.json"):
        cfg = FeatureConfig.from_hf(str(src))
        assert cfg == FeatureConfig(feature_size=128)
    (tmp_path / "p.json").write_text(json.dumps({"chunk_length": 10}))
    short = FeatureConfig.from_hf(str(tmp_path / "p.json"))
    assert short.n_samples == 160000 and short.feature_size == 80
    (tmp_path / "bad.json").write_text(
        json.dumps({"feature_extractor_type": "CLIPImageProcessor"})
    )
    with pytest.raises(ValueError):
        FeatureConfig.from_hf(str(tmp_path / "bad.json"))


def test_rejects_bad_frontend_arguments():
    # WHY: a 2-D array is a caller who forgot to_mono_16k; fewer than
    #      n_fft // 2 + 1 samples cannot be reflect-padded. Both fail before
    #      any arithmetic.
    # KIND: boundary
    # CATCHES: m07
    # CHAPTER: L14.1 section 4
    cfg = FeatureConfig()
    with pytest.raises(ValueError):
        whisper_log_mel(np.zeros((2, 4000)), cfg)
    with pytest.raises(ValueError):
        whisper_log_mel(np.zeros(200), cfg)
    one = whisper_log_mel(np.zeros(201), cfg)
    assert one.shape == (80, 1) and one.dtype == np.float32
