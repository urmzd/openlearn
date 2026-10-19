# contracts/py/tinyllm/sig/stft.pyi (M12.5): windows, frames, and the STFT
# chapter: content/courses/mathematics-for-computing/12-signals-and-images/05-sampling-windows-and-stft.md
#
# Semantics of torch.stft(x, n_fft, hop_length, window=window, center,
# pad_mode, return_complex=True) for a 1-D float signal and a window of
# exactly n_fft samples (n_fft even, since frames go through M12.4's rfft).
# Frame t covers padded samples [t hop, t hop + n_fft). float64 / complex128
# results; inputs are never modified.
from typing import Literal

from numpy.typing import NDArray

def hann(n: int, periodic: bool = True) -> NDArray:
    """torch.hann_window(n, periodic) in float64: 0.5 - 0.5 cos(2 pi k / N)
    for k = 0 .. n-1 with N = n (periodic, the DFT-even window used for
    spectra) or N = n - 1 (symmetric). n = 0 gives [], n = 1 gives [1].
    ValueError for n < 0."""

def frame_count(n_samples: int, n_fft: int, hop: int, center: bool = True) -> int:
    """Frames stft produces: center=True pads n_fft // 2 on each side, giving
    1 + (n_samples + 2 (n_fft // 2) - n_fft) // hop, which is 1 + n_samples // hop
    for even n_fft (Whisper: 480000 samples, hop 160 -> 3001; it then drops
    the last frame); center=False gives 1 + (n_samples - n_fft) // hop.
    ValueError for a size < 1, or center=False with n_samples < n_fft."""

def stft(x: NDArray, n_fft: int, hop: int, window: NDArray, center: bool = True,
         pad_mode: Literal["reflect", "constant"] = "reflect") -> NDArray:
    """complex128 [n_fft // 2 + 1, frames]: S[f, t] = sum_j w[j] xp[t hop + j]
    exp(-2 pi i f j / n_fft), where xp is x padded by n_fft // 2 on each side
    when center (reflect: mirrored without repeating the edge sample, as
    numpy's 'reflect'; constant: zeros), else x itself. Frequency bin f is
    f sr / n_fft Hz. ValueError for a non-1-D x, a window that is not
    [n_fft], hop < 1, reflect padding of a signal no longer than n_fft // 2,
    an unknown pad_mode, or a signal shorter than one frame."""

def istft(S: NDArray, n_fft: int, hop: int, window: NDArray, length: int) -> NDArray:
    """The inverse of stft(center=True) by weighted overlap-add: irfft each
    frame, multiply by the window again, overlap-add, divide by the
    overlap-added squared window, drop the n_fft // 2 padding, and return
    `length` samples. Exact (to rounding) whenever the squared window
    overlap-adds to a nonzero value everywhere (NOLA), e.g. a periodic Hann
    at hop n_fft / 4. ValueError when S is not [n_fft // 2 + 1, frames], the
    window is not [n_fft], or the overlap-add vanishes inside the output."""

def power(S: NDArray) -> NDArray:
    """|S|^2 = re^2 + im^2 elementwise, float64, same shape."""
