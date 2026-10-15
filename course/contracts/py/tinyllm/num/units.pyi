# contracts/py/tinyllm/num/units.pyi (M00.1)
# chapter: content/tracks/math/00-precalculus/01-exponents-logs-and-units-of-information.md
#
# Units of information. A probability p carries -log_b(p) units of surprise;
# b = e gives nats, b = 2 gives bits. Losses in the course are in nats (the
# natural log); comparisons across tokenizers are in bits per byte.
from numpy.typing import ArrayLike, NDArray

LN2: float  # math.log(2.0): nats in one bit

def nats_to_bits(x: float) -> float:
    """x nats expressed in bits: x / ln 2. Any float, including inf."""

def bits_to_nats(x: float) -> float:
    """x bits expressed in nats: x * ln 2. The inverse of nats_to_bits."""

def log_base(x: ArrayLike, b: float) -> NDArray:
    """log_b(x) = ln(x) / ln(b), elementwise, as float64 with x's shape
    (a scalar x gives a float64 scalar). log_base(0, b) is -inf for b > 1 and +inf
    for 0 < b < 1, with no warning. ValueError when b <= 0, b == 1, b is not
    finite, or any x is negative or NaN."""

def bits_per_byte(nll_nats_sum: float, n_bytes: int) -> float:
    """Bits per byte of a text: nll_nats_sum / (n_bytes * ln 2), where
    nll_nats_sum is the total negative log-likelihood in nats a model assigns
    to the text and n_bytes is the text's length in UTF-8 bytes (not tokens).
    ValueError when n_bytes is not a positive integer, or nll_nats_sum is
    negative or NaN."""
