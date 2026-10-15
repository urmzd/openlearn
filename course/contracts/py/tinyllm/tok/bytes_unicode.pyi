# contracts/py/tinyllm/tok/bytes_unicode.pyi (M05.2)
# chapter: content/tracks/math/05-discrete-math-1/02-injective-surjective-bijective-gpt2-byte-map.md
#
# A finite function f: A -> B is a Mapping whose keys are its domain A and
# whose values are its outputs. Injective: no two keys share a value.
# Surjective onto B: every element of B is some value. Bijective: both, and
# then the inverse map exists. The GPT-2 byte map is a bijection from the 256
# byte values onto 256 printable, non-whitespace characters; byte-level BPE
# (L1.2) stores its vocabulary and merges as strings over that alphabet.
from collections.abc import Hashable, Iterable, Mapping
from typing import TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V", bound=Hashable)

def is_injective(f: Mapping[K, V]) -> bool:
    """True when no two keys of f map to the same value. An empty map is injective."""

def is_surjective(f: Mapping[K, V], codomain: Iterable[V]) -> bool:
    """True when every element of codomain is a value of f.
    ValueError when some value of f lies outside codomain (then f is not a
    function into codomain at all, and the question has no answer)."""

def inverse(f: Mapping[K, V]) -> dict[V, K]:
    """The inverse map {f(a): a}. ValueError when f is not injective; the
    message names the shared value and two keys that collide on it."""

def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 byte map, byte value 0..255 -> one-character string.
    The 188 printable Latin-1 bytes, 0x21..0x7E, 0xA1..0xAC and 0xAE..0xFF,
    map to themselves; every other byte b, in increasing order of b, maps to
    chr(256 + n) where n counts the remapped bytes before it (0x00 -> U+0100,
    0x20 -> U+0120 'Ġ', 0x0A -> U+010A 'Ċ'). Returns a new dict on each call,
    with keys in increasing byte order."""

def unicode_to_bytes() -> dict[str, int]:
    """The inverse of bytes_to_unicode(): one-character string -> byte value."""

def encode_bytes(data: bytes) -> str:
    """Each byte of data replaced by its bytes_to_unicode() character, so
    len(result) == len(data). TypeError unless data is bytes or bytearray."""

def decode_chars(text: str) -> bytes:
    """The inverse of encode_bytes. ValueError naming the first character of
    text (and its index) that is not in the image of the byte map."""
