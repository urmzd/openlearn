# contracts/py/tinyllm/sig/fft.pyi (M12.4): the DFT and a mixed-radix FFT
# chapter: content/courses/mathematics-for-computing/12-signals-and-images/04-dft-and-fft.md
#
# Sign and scale follow numpy.fft: X[k] = sum_j x[j] exp(-2 pi i j k / n),
# the inverse divides by n. Transforms act on the LAST axis and broadcast
# over the leading ones. Sizes: every n >= 1 whose prime factors are 2, 3,
# and 5 (400 = 4 * 4 * 5 * 5); anything else is a ValueError, never a slow
# fallback. Accuracy: within 8 log2(max(n, 2)) eps64 ||x||_2 of the exact
# transform in every entry. Inputs are never modified.
from numpy.typing import NDArray

def dft_matrix(n: int, unitary: bool = False) -> NDArray:
    """complex128 [n, n], W[k, j] = exp(-2 pi i (k j mod n) / n), divided by
    sqrt(n) when unitary (then W^H W = I). ValueError for n < 1."""

def fft(x: NDArray) -> NDArray:
    """complex128 DFT along the last axis by mixed-radix decimation in time
    over radices 4, 2, 3, 5 (in that order of preference). Equals
    dft_matrix(n) @ x for a vector. ValueError for an empty last axis or n
    with another prime factor."""

def ifft(X: NDArray) -> NDArray:
    """The inverse: conj(fft(conj(X))) / n, so ifft(fft(x)) == x."""

def rfft(x: NDArray) -> NDArray:
    """The n // 2 + 1 non-negative-frequency bins of the DFT of real x
    (numpy.fft.rfft), computed from ONE complex FFT of size n/2 of
    z[m] = x[2m] + i x[2m + 1]. ValueError unless x is real with an even
    last-axis length n >= 2 whose half n/2 the FFT supports."""

def irfft(X: NDArray, n: int) -> NDArray:
    """The real signal of even length n with half-spectrum X [..., n//2 + 1]
    (numpy.fft.irfft(X, n)): the imaginary parts of X[0] and X[n/2] are
    ignored, and the work is one complex inverse FFT of size n/2. ValueError
    for odd n, n < 2, or a wrong number of bins."""

def fft_convolve(a: NDArray, b: NDArray) -> NDArray:
    """The full linear convolution of 1-D a and b (numpy.convolve(a, b)),
    length len(a) + len(b) - 1, by zero-padding both to the smallest
    m >= that length whose prime factors are 2, 3, 5 and multiplying their
    FFTs. Real inputs give a real float64 result. ValueError for an empty or
    non-1-D input."""

def dct2_ortho(x: NDArray, axis: int = -1) -> NDArray:
    """scipy.fft.dct(x, type=2, norm='ortho', axis=axis): y[k] = s_k * 2 *
    sum_j x[j] cos(pi k (2j + 1) / (2n)), s_0 = sqrt(1/(4n)), s_k = sqrt(1/(2n)),
    computed with one FFT of size n (even samples forward, odd samples
    reversed). The DCT matrix is orthogonal. float64; same shape as x."""
