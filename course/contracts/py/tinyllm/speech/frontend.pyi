# contracts/py/tinyllm/speech/frontend.pyi (L14.1): Whisper's audio frontend
# chapter: content/courses/build-an-llm/15-speech/01-audio-frontend.md
#
# Hugging Face WhisperFeatureExtractor semantics (formats/mel-spec.md):
# 16 kHz mono, pad or trim to chunk_length * sampling_rate samples (480000,
# 30 s), periodic Hann(n_fft = 400), hop 160, center=True with reflect
# padding (M12.5's stft), power spectrum, drop the LAST frame (3001 -> 3000),
# M12.6's Slaney mel filterbank (fmax = sampling_rate / 2) and
# log_mel_whisper (log10 with a 1e-10 floor, clip 8 below the maximum over
# the window, (L + 4) / 4). Results are float32; inputs are never modified.
from dataclasses import dataclass

from numpy.typing import NDArray

from tinyllm.io.wav import Wav

@dataclass
class FeatureConfig:
    sampling_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    feature_size: int = 80  # n_mels: 80, or 128 for large-v3
    chunk_length: int = 30  # seconds per window
    padding_value: float = 0.0

    @classmethod
    def from_hf(cls, preprocessor_config_json: str) -> "FeatureConfig":
        """The six fields from a preprocessor_config.json (a file path, or a
        directory holding one); missing keys keep their defaults, other keys
        (mel_filters, processor_class, ...) are ignored. ValueError when
        feature_extractor_type is present and is not
        "WhisperFeatureExtractor"."""

    @property
    def n_samples(self) -> int:
        """chunk_length * sampling_rate (480000)."""

def to_mono_16k(w: Wav, cfg: FeatureConfig) -> NDArray:
    """float64 [n]: the mean over channels, then M12.3's resample_audio from
    w.sr to cfg.sampling_rate (a copy when they are equal)."""

def pad_or_trim(x: NDArray, n_samples: int = 480000) -> NDArray:
    """float64 [n_samples]: the first n_samples of 1-D x, or x followed by
    zeros. ValueError for x not 1-D or n_samples < 1."""

def whisper_log_mel(x: NDArray, cfg: FeatureConfig) -> NDArray:
    """float32 [cfg.feature_size, len(x) // hop_length]: the log-mel of 1-D x
    exactly as above, without padding or trimming x (call pad_or_trim first
    for a 30 s window: 480000 samples give 3000 frames). The clip uses the
    maximum over all kept frames, and silence gives -1.5 everywhere.
    ValueError for x not 1-D or shorter than n_fft // 2 + 1 samples (reflect
    padding needs them)."""
