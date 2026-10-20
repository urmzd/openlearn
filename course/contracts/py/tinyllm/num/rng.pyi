# contracts/py/tinyllm/num/rng.pyi (M06.3; the C half is tinyllm/numerics.h)
# chapter: content/courses/mathematics/12-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md
#
# The one random generator of the system, exactly as spec/pcg32.md: PCG-XSH-RR
# with a 64-bit state and 32-bit outputs, seeded like O'Neill's
# pcg32_srandom_r, sub-streams derived with SplitMix64, plus the FNV-1a 64
# hash that rt.04 chains over KV blocks. All arithmetic is on Python ints
# reduced modulo 2^64 (state) or 2^32 (outputs).
from typing import MutableSequence, TypeVar

from numpy.typing import NDArray

T = TypeVar("T")

MULT: int  # 6364136223846793005, the PCG multiplier (an LCG constant)
GOLDEN: int  # 0x9E3779B97F4A7C15, the SplitMix64 increment
FNV_OFFSET: int  # 0xCBF29CE484222325, FNV-1a 64 offset basis
FNV_PRIME: int  # 0x100000001B3, FNV-1a 64 prime
PURPOSES: dict[str, int]  # {"init": 1, "dropout": 2, "shuffle": 3, "sample": 4, "mutation": 5}

class PCG32:
    seed: int  # the seed passed in (mod 2^64), kept for substream()

    def __init__(self, seed: int, seq: int = 54) -> None:
        """pcg32_srandom_r(seed, seq): inc = (seq << 1) | 1 (mod 2^64),
        state = 0, one step, state += seed, one step. PCG32(42) yields
        0xa15c02b7 0x7b47f409 ... (spec/pcg32.md)."""

    def next_u32(self) -> int:
        """One step: old = state; state = old * MULT + inc (mod 2^64); return
        rotr32(u32(((old >> 18) ^ old) >> 27), old >> 59)."""

    def uniform(self) -> float:
        """a, b = next_u32(), next_u32(); ((a >> 5) * 2^26 + (b >> 6)) * 2^-53:
        a float in [0, 1) with 53 random bits, identical in C (tl_pcg32_uniform)."""

    def uniforms(self, n: int) -> NDArray:
        """float64 [n]: n calls of uniform() in order. ValueError for n < 0."""

    def below(self, n: int) -> int:
        """Unbiased integer in [0, n) by rejection: t = (2^32 - n) mod n, draw
        r = next_u32() until r >= t, return r mod n. ValueError unless
        1 <= n <= 2^32."""

    def shuffle(self, xs: MutableSequence[T]) -> None:
        """Fisher-Yates in place, from the end: for i = len - 1 down to 1,
        j = below(i + 1), swap xs[i] and xs[j]."""

    def substream(self, purpose: str) -> "PCG32":
        """stream(seed, purpose) = PCG32(child_seed(seed, p), seq=p) with p =
        PURPOSES[purpose]; independent of this generator's current state.
        KeyError for an unknown purpose."""

    def state(self) -> tuple[int, int]:
        """(state, inc): everything needed to resume the stream."""

    def set_state(self, s: tuple[int, int]) -> None:
        """Resume from state(). ValueError when inc is even (not a PCG stream)."""

def splitmix64(x: int) -> int:
    """SplitMix64's output function (mix64 in spec/pcg32.md) on x mod 2^64:
    z = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9; z = (z ^ (z >> 27)) *
    0x94D049BB133111EB; return z ^ (z >> 31), all mod 2^64. The k-th output
    (k >= 1) of a SplitMix64 generator seeded with s is
    splitmix64(s + k * GOLDEN)."""

def child_seed(seed: int, purpose_id: int) -> int:
    """splitmix64(seed + purpose_id * GOLDEN): the seed of a sub-stream."""

def fnv1a64(data: bytes, h: int = 0xCBF29CE484222325) -> int:
    """FNV-1a 64 continuing from h: for each byte b, h = (h ^ b) * FNV_PRIME
    mod 2^64. fnv1a64(b"a") == 0xAF63DC4C8601EC8C; fnv1a64(b"") == FNV_OFFSET."""

def universal_hash(x: int, a: int, b: int, p: int, m: int) -> int:
    """Carter-Wegman: ((a * x + b) mod p) mod m, for a prime p > x, 1 <= a < p,
    0 <= b < p, m >= 1. ValueError when a, b, or m is out of range."""
