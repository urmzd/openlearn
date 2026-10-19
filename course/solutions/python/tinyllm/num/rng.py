"""tinyllm.num.rng (M06.3): PCG32, SplitMix64, FNV-1a, and universal hashing.

Everything here is modular arithmetic on Python ints: the generator state
lives in Z / 2^64, outputs in Z / 2^32, and `& MASK64` is the reduction mod
2^64 that C gets for free from uint64_t overflow. The exact algorithms are in
contracts/spec/pcg32.md; Python, Rust, and Go implementations share its
published vectors through process and file-based conformance checks.

    g = PCG32(0)                  # pcg32_srandom_r(0, 54)
    g.next_u32()                  # 0x47C28B93 = 1203932051
    g.uniform()                   # 53-bit float in [0, 1)
    g.substream("dropout")        # independent stream for one purpose

Contract: contracts/py/tinyllm/num/rng.pyi.
"""

from __future__ import annotations

from typing import MutableSequence, TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T")

MASK64 = (1 << 64) - 1
MASK32 = (1 << 32) - 1
MULT = 6364136223846793005
GOLDEN = 0x9E3779B97F4A7C15
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
PURPOSES = {"init": 1, "dropout": 2, "shuffle": 3, "sample": 4, "mutation": 5}


class PCG32:
    """PCG-XSH-RR 64/32 (O'Neill 2014): a 64-bit LCG whose state is hidden
    behind a xorshift and a data-dependent rotation."""

    def __init__(self, seed: int, seq: int = 54) -> None:
        self.seed = seed & MASK64
        self._state = 0
        # inc must be odd: an LCG mod 2^64 has full period only with an odd
        # increment. seq picks one of 2^63 distinct streams.
        self._inc = ((seq << 1) | 1) & MASK64
        self.next_u32()
        self._state = (self._state + self.seed) & MASK64
        self.next_u32()

    def next_u32(self) -> int:
        old = self._state
        self._state = (old * MULT + self._inc) & MASK64
        # Output from the OLD state (so the multiply overlaps the permutation
        # in C): xorshift high bits down, keep 32, rotate by the top 5 bits.
        xs = (((old >> 18) ^ old) >> 27) & MASK32
        rot = old >> 59
        return ((xs >> rot) | (xs << ((-rot) & 31))) & MASK32

    def uniform(self) -> float:
        a = self.next_u32() >> 5  # 27 bits
        b = self.next_u32() >> 6  # 26 bits
        # a * 2^26 + b is an integer below 2^53, exact in a double, and the
        # product with 2^-53 is exact too: no rounding anywhere.
        return (a * 67108864 + b) * (1.0 / 9007199254740992.0)

    def uniforms(self, n: int) -> NDArray:
        if n < 0:
            raise ValueError(f"uniforms: n must be >= 0, got {n}")
        return np.array([self.uniform() for _ in range(n)], dtype=np.float64)

    def below(self, n: int) -> int:
        if not 1 <= n <= 1 << 32:
            raise ValueError(f"below: n must be in [1, 2^32], got {n}")
        # 2^32 mod n raw values would map to the low residues one extra
        # time; rejecting r < t leaves a multiple of n values, each residue
        # equally often.
        t = ((1 << 32) - n) % n
        while True:
            r = self.next_u32()
            if r >= t:
                return r % n

    def shuffle(self, xs: MutableSequence[T]) -> None:
        for i in range(len(xs) - 1, 0, -1):
            j = self.below(i + 1)  # 0 <= j <= i: xs[i] may stay put
            xs[i], xs[j] = xs[j], xs[i]

    def substream(self, purpose: str) -> "PCG32":
        p = PURPOSES[purpose]
        return PCG32(child_seed(self.seed, p), seq=p)

    def state(self) -> tuple[int, int]:
        return (self._state, self._inc)

    def set_state(self, s: tuple[int, int]) -> None:
        state, inc = int(s[0]), int(s[1])
        if inc % 2 == 0:
            raise ValueError(f"set_state: inc must be odd, got {inc}")
        self._state = state & MASK64
        self._inc = inc & MASK64


def splitmix64(x: int) -> int:
    z = x & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def child_seed(seed: int, purpose_id: int) -> int:
    return splitmix64((seed + purpose_id * GOLDEN) & MASK64)


def fnv1a64(data: bytes, h: int = FNV_OFFSET) -> int:
    h &= MASK64
    for byte in bytes(data):
        h ^= byte  # xor first, then multiply: that order is the "1a"
        h = (h * FNV_PRIME) & MASK64
    return h


def universal_hash(x: int, a: int, b: int, p: int, m: int) -> int:
    if not 1 <= a < p:
        raise ValueError(f"universal_hash: need 1 <= a < p, got a={a}, p={p}")
    if not 0 <= b < p:
        raise ValueError(f"universal_hash: need 0 <= b < p, got b={b}, p={p}")
    if m < 1:
        raise ValueError(f"universal_hash: need m >= 1, got m={m}")
    return ((a * x + b) % p) % m
