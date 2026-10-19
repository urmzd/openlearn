"""tinyllm.sig.mel (M12.6): the mel scale, mel filterbanks, and decibels.

Hearing resolves pitch finely at low frequencies and coarsely at high ones.
The mel scale models that: roughly linear below 1 kHz and logarithmic
above. A mel filterbank places n_mels triangles at equal mel spacing, so they
are narrow in Hz at the bottom and wide at the top, and a matrix product
pools a power spectrum's 201 FFT bins into 80 or 128 mel bands. Loudness is
heard on a log scale too, so the bands go through a logarithm: decibels in
librosa, log10 with a floor and an 8-unit dynamic range in Whisper. The
numbers here must match librosa's filters (which Whisper ships) to 1e-7,
because every Whisper feature is a dot product with them.

The logarithms are M00.1's `log_base(x, 10)`: decibels are a change of base.

Contract: contracts/py/tinyllm/sig/mel.pyi.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from tinyllm.num.units import log_base

_F_SP = 200.0 / 3.0  # Slaney: 200/3 Hz per mel below the break
_MIN_LOG_HZ = 1000.0  # the break between the linear and the log part
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP  # = 15 mel
_LOGSTEP = math.log(6.4) / 27.0  # 27 mel per factor 6.4 above the break


def hz_to_mel(f: NDArray, scale: Literal["slaney", "htk"] = "slaney") -> NDArray:
    f = np.asarray(f, dtype=np.float64)
    if scale == "htk":
        return 2595.0 * np.log10(1.0 + f / 700.0)
    if scale != "slaney":
        raise ValueError(f"hz_to_mel: scale is 'slaney' or 'htk', got {scale!r}")
    lin = f / _F_SP
    with np.errstate(divide="ignore", invalid="ignore"):
        log = _MIN_LOG_MEL + np.log(f / _MIN_LOG_HZ) / _LOGSTEP
    return np.where(f >= _MIN_LOG_HZ, log, lin)


def mel_to_hz(m: NDArray, scale: Literal["slaney", "htk"] = "slaney") -> NDArray:
    m = np.asarray(m, dtype=np.float64)
    if scale == "htk":
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    if scale != "slaney":
        raise ValueError(f"mel_to_hz: scale is 'slaney' or 'htk', got {scale!r}")
    lin = _F_SP * m
    with np.errstate(over="ignore"):
        log = _MIN_LOG_HZ * np.exp(_LOGSTEP * (m - _MIN_LOG_MEL))
    return np.where(m >= _MIN_LOG_MEL, log, lin)


def mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: float | None = None,
    scale: Literal["slaney", "htk"] = "slaney",
    norm: Literal["slaney", None] = "slaney",
) -> NDArray:
    if sr < 1 or n_fft < 1 or n_mels < 1:
        raise ValueError(
            f"mel_filterbank: need positive sr, n_fft, n_mels, got {sr}, {n_fft}, {n_mels}"
        )
    fmax = sr / 2.0 if fmax is None else float(fmax)
    if not 0.0 <= fmin < fmax:
        raise ValueError(f"mel_filterbank: need 0 <= fmin < fmax, got {fmin}, {fmax}")
    if norm not in ("slaney", None):
        raise ValueError(f"mel_filterbank: norm is 'slaney' or None, got {norm!r}")
    # Bin k sits at k sr / n_fft Hz (numpy's rfftfreq, written its way).
    fft_hz = np.arange(n_fft // 2 + 1) * (1.0 / (n_fft * (1.0 / sr)))
    # n_mels + 2 edges equally spaced in mel: band i rises from edge i to its
    # peak at edge i + 1 and falls to zero at edge i + 2.
    edges = mel_to_hz(
        np.linspace(hz_to_mel(fmin, scale), hz_to_mel(fmax, scale), n_mels + 2), scale
    )
    fdiff = np.diff(edges)
    ramps = edges[:, None] - fft_hz[None, :]
    w = np.empty((n_mels, fft_hz.size))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        w[i] = np.maximum(0.0, np.minimum(lower, upper))
    if norm == "slaney":
        # Divide each triangle by half its width in Hz: unit area, so a wide
        # high band does not collect more energy just for being wide.
        w *= (2.0 / (edges[2 : n_mels + 2] - edges[:n_mels]))[:, None]
    return w


def power_to_db(
    p: NDArray, ref: float = 1.0, amin: float = 1e-10, top_db: float | None = 80.0
) -> NDArray:
    s = np.asarray(p)
    if np.iscomplexobj(s):
        raise ValueError("power_to_db: take power(S) of a complex spectrum first")
    if amin <= 0:
        raise ValueError(f"power_to_db: amin must be positive, got {amin}")
    if top_db is not None and top_db < 0:
        raise ValueError(f"power_to_db: top_db must be non-negative, got {top_db}")
    s = s.astype(np.float64)
    # Floor at amin before the log: 10 log10(0) would be -inf.
    db = 10.0 * log_base(np.maximum(amin, s), 10.0)
    db = db - 10.0 * log_base(np.maximum(amin, abs(ref)), 10.0)
    if top_db is not None and db.size:
        db = np.maximum(db, db.max() - top_db)
    return db


def log_mel_whisper(power_spec: NDArray, filters: NDArray) -> NDArray:
    spec = np.asarray(power_spec, dtype=np.float64)
    fb = np.asarray(filters, dtype=np.float64)
    if spec.ndim != 2 or fb.ndim != 2 or fb.shape[1] != spec.shape[0]:
        raise ValueError(
            f"log_mel_whisper: need filters [M, F] and power [F, T], got {fb.shape}, {spec.shape}"
        )
    mel = fb @ spec
    log_spec = log_base(np.maximum(mel, 1e-10), 10.0)
    # Keep 8 units (80 dB) below the loudest value, then map to about [-1, 1].
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0
