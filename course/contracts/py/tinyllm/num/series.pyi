# contracts/py/tinyllm/num/series.pyi (M00.3)
# chapter: content/tracks/math/00-precalculus/03-sequences-and-frequency-ladders.md
#
# Geometric sequences and their sums, and the two frequency ladders a
# transformer uses for positions: RoPE inverse frequencies (L7.3) and ALiBi
# head slopes (L7.4). Every result is float64.
from numpy.typing import NDArray

def geometric(a: float, r: float, n: int) -> NDArray:
    """The first n terms [a, a r, a r^2, ..., a r^(n-1)] as float64 [n]; each
    term is computed as a * r**j (not a running product). n = 0 gives an
    empty array. ValueError when n is not a non-negative integer."""

def geometric_sum(a: float, r: float, n: int) -> float:
    """a + a r + ... + a r^(n-1) = a (1 - r^n) / (1 - r), and a n when r == 1.
    For r > 0, including r within 1e-15 of 1, the relative error is below
    1e-12 (the textbook closed form loses digits there). A sum too large
    for float64 is +inf or -inf. n = 0 gives 0.0. ValueError when n is not a
    non-negative integer."""

def rope_inv_freq(d_rot: int, base: float) -> NDArray:
    """RoPE inverse frequencies, float64 [d_rot / 2]:
        inv_freq[i] = base ** (-2 i / d_rot),   i = 0, 1, ..., d_rot/2 - 1,
    a geometric sequence from 1 down with ratio base ** (-2 / d_rot).
    ValueError when d_rot is not a positive even integer, or base is not
    finite and > 1."""

def alibi_slopes(n_heads: int) -> NDArray:
    """ALiBi head slopes (Press et al. 2022), float64 [n_heads]. For n a power
    of two: the geometric sequence with first term and ratio 2 ** (-8 / n).
    Otherwise, with p the largest power of two below n: the p slopes for p
    heads, followed by the first n - p entries of every other slope (indices
    0, 2, 4, ...) of the 2p-head sequence. ValueError when n_heads is not a
    positive integer."""
