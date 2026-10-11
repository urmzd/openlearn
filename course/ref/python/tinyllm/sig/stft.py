"""tinyllm.sig.stft (M12.5): windows, frames, and the short-time Fourier transform.

Speech changes every few tens of milliseconds, so one Fourier transform of a
whole clip says which frequencies occur but not when. The STFT cuts the
signal into overlapping frames (Whisper: 400 samples = 25 ms, a new frame
every 160 samples = 10 ms), tapers each with a window so its edges do not
leak energy into every bin, and takes the real FFT of each frame. With
center=True the signal is first padded by n_fft // 2 reflected samples on
each side, so frame t is centered on sample t * hop and there are
1 + n // hop frames. A periodic Hann window at hop n_fft / 4 has a constant
overlap-add of its square, so the inverse (weighted overlap-add) gives back
the signal exactly. L14.1's Whisper frontend is this power spectrum
followed by M12.6's mel filterbank.

Contract: contracts/py/tinyllm/sig/stft.pyi.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from tinyllm.sig.fft import irfft, rfft


def hann(n: int, periodic: bool = True) -> NDArray:
    # SOLUTION-BEGIN M12.5
    if n < 0:
        raise ValueError(f"hann: n must be >= 0, got {n}")
    if n <= 1:
        return np.ones(n)
    # Periodic: one period of a length-n cosine (the window an FFT frame
    # repeats); symmetric: the length-n window whose last sample is 0 again.
    period = n if periodic else n - 1
    k = np.arange(n)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * k / period)
    # SOLUTION-END


def frame_count(n_samples: int, n_fft: int, hop: int, center: bool = True) -> int:
    # SOLUTION-BEGIN M12.5
    if n_samples < 1 or n_fft < 1 or hop < 1:
        raise ValueError(
            f"frame_count: need positive sizes, got {n_samples}, {n_fft}, {hop}"
        )
    if center:
        # Padded length n + 2 (n_fft // 2); frames start every hop samples.
        # For even n_fft this is 1 + n // hop: the pad exactly pays for one frame.
        return 1 + (n_samples + 2 * (n_fft // 2) - n_fft) // hop
    if n_samples < n_fft:
        raise ValueError(
            f"frame_count: {n_samples} samples cannot fill one frame of {n_fft}"
        )
    return 1 + (n_samples - n_fft) // hop
    # SOLUTION-END


def stft(
    x: NDArray,
    n_fft: int,
    hop: int,
    window: NDArray,
    center: bool = True,
    pad_mode: Literal["reflect", "constant"] = "reflect",
) -> NDArray:
    # SOLUTION-BEGIN M12.5
    a = np.asarray(x, dtype=np.float64)
    w = np.asarray(window, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"stft: need a 1-D signal, got shape {a.shape}")
    if w.shape != (n_fft,):
        raise ValueError(
            f"stft: the window must have n_fft = {n_fft} samples, got {w.shape}"
        )
    if hop < 1:
        raise ValueError(f"stft: hop must be >= 1, got {hop}")
    if center:
        half = n_fft // 2
        if pad_mode == "reflect":
            if a.size <= half:
                raise ValueError(
                    f"stft: reflect padding by {half} needs more than {half} samples"
                )
            # Mirror without repeating the edge sample: [3, 2, | 1, 2, 3].
            a = np.pad(a, half, mode="reflect")
        elif pad_mode == "constant":
            a = np.pad(a, half)
        else:
            raise ValueError(
                f"stft: pad_mode is 'reflect' or 'constant', got {pad_mode!r}"
            )
    if a.size < n_fft:
        raise ValueError(f"stft: {a.size} samples cannot fill one frame of {n_fft}")
    t = 1 + (a.size - n_fft) // hop
    starts = np.arange(t)[:, None] * hop
    frames = a[starts + np.arange(n_fft)[None, :]] * w  # [frames, n_fft]
    return np.ascontiguousarray(rfft(frames).T)  # [n_fft // 2 + 1, frames]
    # SOLUTION-END


def istft(S: NDArray, n_fft: int, hop: int, window: NDArray, length: int) -> NDArray:
    # SOLUTION-BEGIN M12.5
    spec = np.asarray(S, dtype=np.complex128)
    w = np.asarray(window, dtype=np.float64)
    if spec.ndim != 2 or spec.shape[0] != n_fft // 2 + 1:
        raise ValueError(f"istft: need [{n_fft // 2 + 1}, frames], got {spec.shape}")
    if w.shape != (n_fft,):
        raise ValueError(
            f"istft: the window must have n_fft = {n_fft} samples, got {w.shape}"
        )
    t = spec.shape[1]
    frames = irfft(spec.T, n_fft) * w  # windowed again (synthesis window)
    total = n_fft + hop * (t - 1)
    y = np.zeros(total)
    wsum = np.zeros(total)
    for i in range(t):
        y[i * hop : i * hop + n_fft] += frames[i]
        wsum[i * hop : i * hop + n_fft] += w * w
    # Weighted overlap-add: divide by the summed squared window, which is
    # constant in the interior when (window^2, hop) is COLA.
    half = n_fft // 2
    sl = slice(half, half + length)
    den = wsum[sl]
    if den.size < length or (np.abs(den) < 1e-11).any():
        raise ValueError(
            "istft: the window overlap-add vanishes; the frames cannot be inverted"
        )
    return y[sl] / den
    # SOLUTION-END


def power(S: NDArray) -> NDArray:
    # SOLUTION-BEGIN M12.5
    s = np.asarray(S)
    # |z|^2 = re^2 + im^2, with no square root to undo.
    return s.real * s.real + s.imag * s.imag
    # SOLUTION-END
