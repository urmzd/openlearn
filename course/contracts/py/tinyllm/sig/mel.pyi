# contracts/py/tinyllm/sig/mel.pyi (M12.6): mel scale, filterbanks, decibels
# chapter: math/12-signals-and-images/06-mel-scale-and-filterbanks.md
#
# librosa semantics (librosa.hz_to_mel, mel_to_hz, filters.mel,
# power_to_db) and Whisper's log-mel scaling (openai/whisper audio.py).
# Slaney mel: m = f / (200/3) below 1000 Hz (15 mel), and
# m = 15 + ln(f / 1000) / (ln 6.4 / 27) above. HTK mel: m = 2595 log10(1 + f / 700).
# Logarithms in base 10 go through M00.1's log_base. float64 results;
# inputs are never modified.
from typing import Literal

from numpy.typing import NDArray

def hz_to_mel(f: NDArray, scale: Literal["slaney", "htk"] = "slaney") -> NDArray:
    """Frequencies in Hz to mels, elementwise. ValueError for another scale."""

def mel_to_hz(m: NDArray, scale: Literal["slaney", "htk"] = "slaney") -> NDArray:
    """The inverse of hz_to_mel. ValueError for another scale."""

def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None,
                   scale: Literal["slaney", "htk"] = "slaney", norm: Literal["slaney", None] = "slaney") -> NDArray:
    """librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin,
    fmax=fmax, htk=(scale == 'htk'), norm=norm, dtype=float64):
    [n_mels, n_fft // 2 + 1]. Edges e_0 .. e_{n_mels+1} are n_mels + 2 points
    equally spaced in mel from fmin to fmax (default sr / 2), mapped back to Hz;
    filter i is the triangle max(0, min((f - e_i)/(e_{i+1} - e_i),
    (e_{i+2} - f)/(e_{i+2} - e_{i+1}))) sampled at the rfft bin centers
    f = k sr / n_fft. norm 'slaney' multiplies filter i by 2 / (e_{i+2} - e_i),
    giving each triangle unit area in Hz. ValueError unless sr, n_fft,
    n_mels >= 1 and 0 <= fmin < fmax, or for another norm or scale."""

def power_to_db(p: NDArray, ref: float = 1.0, amin: float = 1e-10, top_db: float | None = 80.0) -> NDArray:
    """librosa.power_to_db: 10 log10(max(amin, p)) - 10 log10(max(amin, |ref|)),
    then, if top_db is not None, raised to at least (its maximum - top_db).
    ValueError for complex p (take power() first), amin <= 0, or top_db < 0."""

def log_mel_whisper(power_spec: NDArray, filters: NDArray) -> NDArray:
    """Whisper's features from a power spectrogram [F, T] and filters [M, F]:
    mel = filters @ power_spec; L = log10(max(mel, 1e-10)); L = max(L, L.max() - 8);
    return (L + 4) / 4, [M, T]. (Whisper drops the last STFT frame before
    this; that is L14.1's job.) ValueError when the shapes do not chain."""
