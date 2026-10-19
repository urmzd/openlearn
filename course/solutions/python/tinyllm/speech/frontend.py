"""tinyllm.speech.frontend (L14.1): Whisper's audio frontend.

Whisper never sees samples. It sees a picture: 80 (or 128) mel bands by
3000 frames, one frame every 10 ms of a 30 s window. Getting from a WAV file
to that picture is a fixed recipe, and the encoder was trained on exactly
that recipe, so every step must match Hugging Face's WhisperFeatureExtractor
to 1e-4: average the channels, resample to 16 kHz (M12.3), pad with zeros or
cut to 480000 samples, take the power STFT with a periodic Hann window of
400 samples at hop 160 with reflect padding (M12.5), drop the last of the
3001 frames, pool the 201 bins into mel bands with librosa's Slaney filters
(M12.6), and apply Whisper's log10, 8-unit clip, and (L + 4) / 4.

Contract: contracts/py/tinyllm/speech/frontend.pyi.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tinyllm.io.wav import Wav
from tinyllm.sig.mel import log_mel_whisper, mel_filterbank
from tinyllm.sig.resample import resample_audio
from tinyllm.sig.stft import hann, power, stft


@dataclass
class FeatureConfig:
    sampling_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    feature_size: int = 80
    chunk_length: int = 30
    padding_value: float = 0.0

    @classmethod
    def from_hf(cls, preprocessor_config_json: str) -> "FeatureConfig":
        p = Path(preprocessor_config_json)
        if p.is_dir():
            p = p / "preprocessor_config.json"
        raw = json.loads(p.read_text())
        kind = raw.get("feature_extractor_type")
        if kind is not None and kind != "WhisperFeatureExtractor":
            raise ValueError(
                f"FeatureConfig.from_hf: feature_extractor_type {kind!r}, need WhisperFeatureExtractor"
            )
        known = {f.name: f.type for f in fields(cls)}
        kw = {}
        for k in known:
            if k in raw:
                kw[k] = float(raw[k]) if k == "padding_value" else int(raw[k])
        return cls(**kw)

    @property
    def n_samples(self) -> int:
        return self.chunk_length * self.sampling_rate


def to_mono_16k(w: Wav, cfg: FeatureConfig) -> NDArray:
    x = np.asarray(w.samples, dtype=np.float64)
    mono = x.mean(axis=0) if x.ndim == 2 else x.copy()
    if w.sr == cfg.sampling_rate:
        return mono
    return resample_audio(mono, w.sr, cfg.sampling_rate)


def pad_or_trim(x: NDArray, n_samples: int = 480000) -> NDArray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"pad_or_trim: need a 1-D signal, got shape {a.shape}")
    if n_samples < 1:
        raise ValueError(f"pad_or_trim: n_samples must be >= 1, got {n_samples}")
    if a.size >= n_samples:
        return a[:n_samples].copy()
    return np.concatenate([a, np.zeros(n_samples - a.size)])


def whisper_log_mel(x: NDArray, cfg: FeatureConfig) -> NDArray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"whisper_log_mel: need a 1-D signal, got shape {a.shape}")
    if a.size < cfg.n_fft // 2 + 1:
        raise ValueError(
            f"whisper_log_mel: {a.size} samples; reflect padding needs at least {cfg.n_fft // 2 + 1}"
        )
    spec = stft(a, cfg.n_fft, cfg.hop_length, hann(cfg.n_fft, periodic=True))
    # 1 + n // hop frames; Whisper keeps n // hop of them (3001 -> 3000).
    p = power(spec)[:, :-1]
    fb = mel_filterbank(cfg.sampling_rate, cfg.n_fft, cfg.feature_size)
    # The clip at max - 8 sees only the kept frames.
    return log_mel_whisper(p, fb).astype(np.float32)
