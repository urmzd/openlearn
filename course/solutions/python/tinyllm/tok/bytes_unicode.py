"""Finite functions and the GPT-2 byte map (M05.2).

A finite function is a Mapping: its keys are the domain and its values the
outputs. is_injective, is_surjective, and inverse are the three definitions
of the chapter as code. bytes_to_unicode is the bijection byte-level BPE
(L1.2) uses to write any byte string as printable, non-whitespace text, so a
vocabulary fits in JSON and a merges file splits on spaces.

Contract: contracts/py/tinyllm/tok/bytes_unicode.pyi.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping


def is_injective(f: Mapping[Hashable, Hashable]) -> bool:
    seen: set = set()
    for v in f.values():
        if v in seen:
            return False
        seen.add(v)
    return True


def is_surjective(f: Mapping[Hashable, Hashable], codomain: Iterable[Hashable]) -> bool:
    target = set(codomain)
    image = set(f.values())
    outside = image - target
    if outside:
        raise ValueError(
            f"f is not a function into the codomain: value {next(iter(outside))!r} is outside it"
        )
    return image == target


def inverse(f: Mapping[Hashable, Hashable]) -> dict:
    inv: dict = {}
    for k, v in f.items():
        if v in inv:
            raise ValueError(
                f"f is not injective: keys {inv[v]!r} and {k!r} both map to {v!r}"
            )
        inv[v] = k
    return inv


def bytes_to_unicode() -> dict[int, str]:
    # The printable Latin-1 ranges keep their own code point. 0xAD (soft
    # hyphen) sits between the last two and is invisible, so it is remapped.
    ranges = ((0x21, 0x7E), (0xA1, 0xAC), (0xAE, 0xFF))
    keep = {b for lo, hi in ranges for b in range(lo, hi + 1)}
    out: dict[int, str] = {}
    n = 0
    for b in range(256):
        if b in keep:
            out[b] = chr(b)
        else:
            out[b] = chr(256 + n)
            n += 1
    return out


def unicode_to_bytes() -> dict[str, int]:
    return inverse(bytes_to_unicode())


def encode_bytes(data: bytes) -> str:
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"encode_bytes takes bytes, got {type(data).__name__}")
    table = bytes_to_unicode()
    return "".join(table[b] for b in data)


def decode_chars(text: str) -> bytes:
    table = unicode_to_bytes()
    out = bytearray()
    for i, ch in enumerate(text):
        b = table.get(ch)
        if b is None:
            raise ValueError(
                f"character {ch!r} (U+{ord(ch):04X}) at index {i} is not in the byte map's image"
            )
        out.append(b)
    return bytes(out)
