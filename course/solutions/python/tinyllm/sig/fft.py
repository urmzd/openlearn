"""tinyllm.sig.fft (M12.4): the DFT and a mixed-radix FFT.

The discrete Fourier transform writes n samples as a sum of n complex
sinusoids: X = W x with W[k, j] = exp(-2 pi i k j / n). W / sqrt(n) is
unitary, so the transform keeps energy (Parseval) and is undone by its
conjugate transpose. Multiplying by W costs n^2; the fast Fourier transform
gets the same numbers in O(n log n) by splitting x into p interleaved
subsequences, transforming each, and recombining them with twiddle factors
and one small DFT of size p. Whisper's 400-point frames factor as
4 * 4 * 5 * 5, so a radix-2-only FFT is not enough: this one handles every n
whose prime factors are 2, 3, and 5, which covers every size the course uses.

A real signal's spectrum is conjugate symmetric, so rfft computes only the
n//2 + 1 non-redundant bins, from one complex FFT of half the size. The
DCT-II that pHash uses (data.10) is a reordered FFT too.

Contract: contracts/py/tinyllm/sig/fft.pyi.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_RADICES = (4, 2, 3, 5)


def dft_matrix(n: int, unitary: bool = False) -> NDArray:
    if n < 1:
        raise ValueError(f"dft_matrix: n must be >= 1, got {n}")
    j = np.arange(n)
    # Reduce k * j mod n before scaling: the phase stays in [0, 2 pi) and
    # exp is evaluated where it is accurate.
    w = np.exp(-2j * np.pi * (np.outer(j, j) % n) / n)
    return w / np.sqrt(n) if unitary else w


def _radix(n: int) -> int:
    """The first radix the FFT splits n by (4 before 2); ValueError if none."""
    for p in _RADICES:
        if n % p == 0:
            return p
    raise ValueError(f"fft: n = {n} has a prime factor other than 2, 3, 5")


def _fft(x: NDArray) -> NDArray:
    """Decimation in time over the last axis of a complex array."""
    n = x.shape[-1]
    if n == 1:
        return x.copy()
    p = _radix(n)
    m = n // p
    lead = x.shape[:-1]
    # Subsequence r holds x[r], x[r + p], x[r + 2p], ...: shape [..., p, m].
    sub = _fft(np.swapaxes(x.reshape(*lead, m, p), -1, -2))
    r = np.arange(p)[:, None]
    k = np.arange(m)[None, :]
    tw = np.exp(-2j * np.pi * ((r * k) % n) / n)  # W_n^{r k}, [p, m]
    # X[q m + k] = sum_r W_p^{r q} (W_n^{r k} Y_r[k]): a size-p DFT per k.
    out = np.einsum("qr,...rk->...qk", dft_matrix(p), sub * tw)
    return out.reshape(*lead, n)


def fft(x: NDArray) -> NDArray:
    a = np.asarray(x, dtype=np.complex128)
    if a.ndim == 0 or a.shape[-1] < 1:
        raise ValueError("fft: need at least one sample on the last axis")
    _check_size(a.shape[-1])
    return _fft(a)


def _check_size(n: int) -> None:
    m = n
    for p in (2, 3, 5):
        while m % p == 0:
            m //= p
    if m != 1:
        raise ValueError(f"fft: n = {n} has a prime factor other than 2, 3, 5")


def ifft(X: NDArray) -> NDArray:
    a = np.asarray(X, dtype=np.complex128)
    # The inverse is the conjugate transform, divided by n.
    return np.conj(fft(np.conj(a))) / a.shape[-1]


def rfft(x: NDArray) -> NDArray:
    a = np.asarray(x)
    if np.iscomplexobj(a):
        raise ValueError("rfft: the input must be real")
    a = a.astype(np.float64)
    n = a.shape[-1] if a.ndim else 0
    if n < 2 or n % 2:
        raise ValueError(f"rfft: need an even length >= 2, got {n}")
    h = n // 2
    # Pack: even samples as the real part, odd samples as the imaginary part.
    z = fft(a[..., 0::2] + 1j * a[..., 1::2])  # [..., h]
    zr = np.conj(np.roll(z[..., ::-1], 1, axis=-1))  # conj(Z[(h - k) mod h])
    even = 0.5 * (z + zr)  # DFT of the even samples
    odd = -0.5j * (z - zr)  # DFT of the odd samples
    k = np.arange(h)
    lower = even + np.exp(-2j * np.pi * k / n) * odd  # bins 0 .. h-1
    nyq = even[..., :1] - odd[..., :1]  # bin h: W_n^h = -1
    return np.concatenate([lower, nyq], axis=-1)


def irfft(X: NDArray, n: int) -> NDArray:
    a = np.asarray(X, dtype=np.complex128)
    if n < 2 or n % 2:
        raise ValueError(f"irfft: need an even length >= 2, got {n}")
    h = n // 2
    if a.shape[-1] != h + 1:
        raise ValueError(f"irfft: need {h + 1} bins for n = {n}, got {a.shape[-1]}")
    a = a.copy()
    # The DC and Nyquist bins of a real signal are real: drop any imaginary
    # part, as numpy does, instead of letting it leak into the samples.
    a[..., 0] = a[..., 0].real
    a[..., h] = a[..., h].real
    k = np.arange(h)
    xk = a[..., :h]
    xr = np.conj(a[..., h - k])  # conj(X[h - k]), k = 0 .. h-1
    even = 0.5 * (xk + xr)
    odd = 0.5 * (xk - xr) * np.exp(2j * np.pi * k / n)
    z = ifft(even + 1j * odd)
    out = np.empty(a.shape[:-1] + (n,))
    out[..., 0::2] = z.real
    out[..., 1::2] = z.imag
    return out


def _fast_size(n: int) -> int:
    """The smallest m >= n whose prime factors are 2, 3, and 5."""
    m = max(n, 1)
    while True:
        r = m
        for p in (2, 3, 5):
            while r % p == 0:
                r //= p
        if r == 1:
            return m
        m += 1


def fft_convolve(a: NDArray, b: NDArray) -> NDArray:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim != 1 or b.ndim != 1 or a.size == 0 or b.size == 0:
        raise ValueError(
            f"fft_convolve: need two non-empty 1-D arrays, got {a.shape}, {b.shape}"
        )
    full = a.size + b.size - 1
    # Zero-pad past the full length so the circular convolution the DFT
    # computes never wraps around: then it equals the linear one.
    m = _fast_size(full)
    fa = fft(np.pad(a.astype(np.complex128), (0, m - a.size)))
    fb = fft(np.pad(b.astype(np.complex128), (0, m - b.size)))
    y = ifft(fa * fb)[:full]
    if not (np.iscomplexobj(a) or np.iscomplexobj(b)):
        return y.real.copy()
    return y


def dct2_ortho(x: NDArray, axis: int = -1) -> NDArray:
    a = np.moveaxis(np.asarray(x, dtype=np.float64), axis, -1)
    n = a.shape[-1]
    if n < 1:
        raise ValueError("dct2_ortho: need at least one sample")
    # Makhoul: even samples forward, odd samples backward, then one FFT.
    v = np.concatenate([a[..., 0::2], a[..., 1::2][..., ::-1]], axis=-1)
    vf = fft(v)
    k = np.arange(n)
    # scipy's unnormalized DCT-II: y[k] = 2 Re(exp(-i pi k / 2n) V[k]).
    y = 2.0 * (vf * np.exp(-1j * np.pi * k / (2 * n))).real
    scale = np.full(n, np.sqrt(1.0 / (2 * n)))
    scale[0] = np.sqrt(1.0 / (4 * n))  # norm='ortho' makes the DCT matrix orthogonal
    return np.moveaxis(y * scale, -1, axis)
