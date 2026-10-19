# contracts/py/tinyllm/xfmr/pos.pyi (L5.4): positional encodings
# chapter: content/courses/build-an-llm/06-transformer-2017/04-positional-encodings.md
#
# Sinusoidal (Vaswani et al. 2017, section 3.5), with d even and
# omega_i = base^(-2i/d), i = 0 .. d/2 - 1 (a geometric ladder, M00.3):
#
#     PE(p, 2i)     = sin(p * omega_i)
#     PE(p, 2i + 1) = cos(p * omega_i)
#
# so pair i of PE(p) is the point (sin, cos) of angle p * omega_i, and
# moving k positions turns every pair by the fixed angle k * omega_i:
# PE(p + k) = rotate_pairs(PE(p), -k * omega) with M00.2's interleaved pairs.
# Angles are computed in float64 and the table returned as float32.
#
# Learned (GPT-2's wpe, BERT's position_embeddings): a [max_len, d] float32
# parameter, initialized N(0, std^2) from the rng (PCG32; None means
# PCG32(0).substream("init")), whose row p is added at position p.
#
# Both modules add positions offset .. offset + T - 1 to an input [..., T, d],
# where T = x.shape[-2]; offset is the number of tokens before x (incremental
# decoding, L8.2).
from typing import Any

from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.module import Module

def sinusoidal_freqs(d: int, base: float = 10000.0) -> NDArray:
    """omega, float64 [d / 2]. ValueError unless d is a positive even integer
    and base is finite and > 1."""

def sinusoidal_pe(T: int, d: int, base: float = 10000.0, offset: int = 0) -> NDArray:
    """float32 [T, d]: rows PE(offset) .. PE(offset + T - 1). ValueError for
    T or offset negative, and as sinusoidal_freqs."""

def shift_pe(pe: ArrayLike, k: float, d: int, base: float = 10000.0) -> NDArray:
    """PE(p + k) from PE(p) without knowing p: every pair turned by
    k * omega_i (rotate_pairs with angle -k * omega_i). pe is [..., d]; the
    result has its shape and is float32 for float32 input. ValueError for a
    last axis other than d."""

class SinusoidalPE(Module):
    # L5.5's transformer adds it to both embeddings. It has no parameters.
    max_len: int
    d: int
    table: NDArray  # sinusoidal_pe(max_len, d, base): a constant, not in the state_dict

    def __init__(self, max_len: int, d: int, base: float = 10000.0) -> None:
        """ValueError for max_len < 1, and as sinusoidal_freqs."""

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        """x + table[offset : offset + T] (no gradient to the table).
        ValueError for a width other than d, or positions past max_len."""

class LearnedPE(Module):
    # L6.1's GPT (wpe) and L6.2's BERT (position_embeddings).
    max_len: int
    d: int
    weight: Tensor  # [max_len, d]

    def __init__(self, max_len: int, d: int, std: float = 0.02, rng: Any = None) -> None:
        """ValueError for max_len or d below 1."""

    def positions(self, T: int, offset: int = 0) -> Tensor:
        """Rows offset .. offset + T - 1 of weight, [T, d], with their
        gradient (an embedding lookup). ValueError for positions outside
        [0, max_len)."""

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        """x + positions(T, offset). ValueError for a width other than d, and
        as positions."""
