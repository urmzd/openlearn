"""Course tests for M05.2: finite functions and the GPT-2 byte map
(tinyllm/tok/bytes_unicode.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M05.2), and the chapter section it comes from.

The chapter's worked example (section 3) encodes the bytes of "Hi there\\n"
as "HiĠthereĊ" and checks three small maps by hand.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from _lib.pcg32 import PCG32
from tinyllm.tok.bytes_unicode import (
    bytes_to_unicode,
    decode_chars,
    encode_bytes,
    inverse,
    is_injective,
    is_surjective,
    unicode_to_bytes,
)

GOLDEN = (
    Path(os.environ.get("TINYLLM_FIXTURES", ""))
    / "M05.2"
    / "gpt2_bytes_to_unicode.json"
)

# The 188 printable Latin-1 bytes that keep their own code point (chapter section 2).
PRINTABLE = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


# --- the worked example -------------------------------------------------------


def test_hand_example_encode():
    # WHY: the chapter's worked example, character for character. "H", "i",
    #      and the letters of "there" are printable and stay; the space 0x20
    #      is the 33rd remapped byte (n = 32), so it becomes chr(256 + 32) =
    #      U+0120 "Ġ"; the newline 0x0A is n = 10, U+010A "Ċ". This is why
    #      GPT-2 vocabularies are full of "Ġthe".
    # KIND: unit
    # CATCHES: s01, s03, m02
    # CHAPTER: M05.2 section 3, Worked example by hand
    assert encode_bytes(b"Hi there\n") == "HiĠthereĊ"
    assert encode_bytes(b"\x00") == "Ā"
    assert encode_bytes(b"") == ""
    assert decode_chars("HiĠthereĊ") == b"Hi there\n"


def test_hand_example_small_maps():
    # WHY: the three maps of the worked example. f sends 1 and 3 to "a", so it
    #      is not injective and has no inverse; g is injective, surjective onto
    #      {a, b}, and not onto {a, b, c}, where "c" has no preimage.
    # KIND: unit
    # CATCHES: s04, s07, s08
    # CHAPTER: M05.2 section 3, Worked example by hand
    f = {1: "a", 2: "b", 3: "a"}
    g = {1: "a", 2: "b"}
    assert is_injective(f) is False
    assert is_injective(g) is True
    assert is_surjective(g, {"a", "b"}) is True
    assert is_surjective(g, {"a", "b", "c"}) is False
    assert inverse(g) == {"a": 1, "b": 2}
    with pytest.raises(ValueError):
        inverse(f)


# --- the byte map -------------------------------------------------------------


def test_matches_gpt2_table():
    # WHY: L1.2 loads GPT-2 and SmolLM2 vocabularies whose tokens are written
    #      in this exact alphabet. One byte mapped differently and every token
    #      containing it fails to load or decodes to the wrong byte. The golden
    #      file comes from Hugging Face transformers, cross-checked against
    #      the tokenizers ByteLevel alphabet (course/oracle/M05.2).
    # KIND: golden
    # CATCHES: s01, s02, s03, m01, m02, m05
    # CHAPTER: M05.2 section 2, Principles (the GPT-2 byte map)
    want = json.loads(GOLDEN.read_text())["code_points"]
    table = bytes_to_unicode()
    assert sorted(table) == list(range(256))
    got = [ord(table[b]) for b in range(256)]
    first = next((b for b in range(256) if got[b] != want[b]), None)
    assert first is None, (
        f"byte 0x{first:02X} maps to U+{got[first]:04X}, GPT-2 maps it to U+{want[first]:04X}"
    )


def test_is_a_bijection_onto_256_characters():
    # WHY: a tokenizer must turn bytes into text and back without loss. That
    #      needs every byte to have exactly one character (a function on all
    #      of 0..255) and no two bytes to share one (injective). 256 distinct
    #      outputs from 256 inputs is then onto the 256-character image.
    # KIND: property
    # CATCHES: s01, m02, m05
    # CHAPTER: M05.2 section 2, Principles (bijection)
    table = bytes_to_unicode()
    assert len(table) == 256
    assert all(isinstance(c, str) and len(c) == 1 for c in table.values())
    assert is_injective(table)
    assert len(set(table.values())) == 256


def test_every_character_is_printable_and_not_whitespace():
    # WHY: the whole point of the map. A merges file is one merge per line,
    #      two tokens separated by a space ("Ġ t"); a space or newline inside
    #      a token would split the line in the wrong place, and a control
    #      character would vanish in a JSON viewer.
    # KIND: property
    # CATCHES: s01, s02
    # CHAPTER: M05.2 section 5, Pitfalls, item 1
    for b, c in bytes_to_unicode().items():
        assert c.isprintable() and not c.isspace(), (
            f"byte 0x{b:02X} maps to {c!r}, which is not a visible character"
        )


def test_printable_bytes_map_to_themselves():
    # WHY: the 188 printable Latin-1 bytes keep their code point, so ASCII text
    #      reads as itself inside a vocabulary ("hello" is the token "hello").
    #      0xAD, the soft hyphen, is the trap: it lies between the ranges and
    #      is invisible, so it is remapped.
    # KIND: unit
    # CATCHES: s02, m01
    # CHAPTER: M05.2 section 5, Pitfalls, item 2
    table = bytes_to_unicode()
    assert len(PRINTABLE) == 188
    for b in PRINTABLE:
        assert table[b] == chr(b)
    assert table[0xAD] != chr(0xAD)


def test_remapped_bytes_are_consecutive_from_256():
    # WHY: the other 68 bytes, in increasing order, take U+0100 to U+0143 with
    #      no gaps. Counting with the byte value (256 + b) instead of the
    #      running count n gives a different, still injective map that no
    #      GPT-2 vocabulary uses.
    # KIND: unit
    # CATCHES: s03, m02
    # CHAPTER: M05.2 section 5, Pitfalls, item 3
    table = bytes_to_unicode()
    others = [b for b in range(256) if b not in set(PRINTABLE)]
    assert len(others) == 68
    assert [ord(table[b]) for b in others] == list(range(256, 256 + 68))


def test_keys_in_byte_order_and_a_fresh_dict_each_call():
    # WHY: callers build their own tables from this one (L1.2 adds the inverse
    #      to its vocabulary loader). A shared module-level dict would let one
    #      caller's edit corrupt every later tokenizer in the process.
    # KIND: boundary
    # CATCHES: s10
    # CHAPTER: M05.2 section 4, The interface
    a = bytes_to_unicode()
    assert list(a) == list(range(256))
    a[32] = " "
    b = bytes_to_unicode()
    assert b[32] == "Ġ"
    assert b is not a


def test_unicode_to_bytes_is_the_inverse():
    # WHY: decoding a generated token back to bytes uses the inverse map;
    #      composing the two in either order must be the identity.
    # KIND: property
    # CATCHES: m04
    # CHAPTER: M05.2 section 2, Principles (inverse)
    fwd, back = bytes_to_unicode(), unicode_to_bytes()
    assert len(back) == 256
    assert all(back[fwd[b]] == b for b in range(256))
    assert all(fwd[back[c]] == c for c in back)


def test_roundtrip_random_bytes():
    # WHY: any byte string, including invalid UTF-8 (a token can end halfway
    #      through a character), survives encode then decode. Random bytes
    #      cover every value many times over.
    # KIND: property
    # CATCHES: s05, m04
    # CHAPTER: M05.2 section 2, Principles (inverse)
    rng = PCG32(seed=seed())
    for n in (1, 7, 256, 1000):
        data = bytes(rng.below(256) for _ in range(n))
        text = encode_bytes(data)
        assert len(text) == n
        assert decode_chars(text) == data
    assert decode_chars(encode_bytes(bytes(range(256)))) == bytes(range(256))


def test_roundtrip_utf8_text():
    # WHY: the map works on bytes, not characters. "é" is two UTF-8 bytes and
    #      becomes two characters; the emoji is four. L1.2 pre-tokenizes text,
    #      encodes it to UTF-8, then maps each byte.
    # KIND: unit
    # CATCHES: s05
    # CHAPTER: M05.2 section 3, Worked example by hand
    s = "héllo wörld \U0001f642\t\r\n"
    data = s.encode("utf-8")
    text = encode_bytes(data)
    assert len(text) == len(data) == len(s) + 1 + 1 + 3
    assert encode_bytes("é".encode()) == "Ã©"
    assert decode_chars(text).decode("utf-8") == s


def test_decode_rejects_characters_outside_the_image():
    # WHY: a real space, a real newline, or a character past U+0143 is not
    #      the image of any byte. Masking with & 0xFF or subtracting 256 would
    #      invent a byte and decode garbage silently.
    # KIND: boundary
    # CATCHES: s05, m03
    # CHAPTER: M05.2 section 5, Pitfalls, item 4
    for bad in (" ", "\n", "ń", "€", "ab c"):
        with pytest.raises(ValueError):
            decode_chars(bad)
    assert decode_chars("Ń") == bytes([0xAD])


def test_encode_rejects_str():
    # WHY: encode_bytes maps bytes. Passing a str is a caller bug (they forgot
    #      .encode("utf-8")); quietly mapping code points would accept "é" as
    #      one byte, which is wrong for anything outside Latin-1.
    # KIND: boundary
    # CATCHES: s09
    # CHAPTER: M05.2 section 4, The interface
    with pytest.raises(TypeError):
        encode_bytes("abc")
    assert encode_bytes(bytearray(b"ab")) == "ab"


# --- finite functions ---------------------------------------------------------


def test_inverse_rejects_a_collision():
    # WHY: building an inverse with a dict comprehension keeps the last key
    #      and drops the other without a word. A tokenizer loader that does
    #      this maps two ids to one string and decodes the wrong token.
    # KIND: boundary
    # CATCHES: s04, m04
    # CHAPTER: M05.2 section 5, Pitfalls, item 5
    with pytest.raises(ValueError, match="'x'"):
        inverse({0: "x", 1: "y", 2: "x"})
    assert inverse({}) == {}
    assert inverse({0: "x", 1: "y"}) == {"x": 0, "y": 1}


def test_is_injective_cases():
    # WHY: injective means distinct inputs give distinct outputs, so the
    #      question is about the values. Checking that the keys are distinct
    #      always says yes, because a dict's keys always are.
    # KIND: unit
    # CATCHES: s08
    # CHAPTER: M05.2 section 2, Principles (injective)
    assert is_injective({}) is True
    assert is_injective({0: 0}) is True
    assert is_injective({0: 1, 1: 0, 2: 2}) is True
    assert is_injective({0: 5, 1: 5}) is False
    assert is_injective({b: b % 128 for b in range(256)}) is False


def test_is_surjective_cases():
    # WHY: surjective is a statement about the codomain, so it needs one.
    #      Every target needs a preimage (a missing one makes it False), and a
    #      value outside the codomain means f is not a map into it at all.
    # KIND: boundary
    # CATCHES: s06, s07
    # CHAPTER: M05.2 section 2, Principles (surjective)
    assert is_surjective({}, []) is True
    assert is_surjective({}, [1]) is False
    assert is_surjective({0: 1, 1: 1}, [1]) is True
    assert is_surjective({0: 1, 1: 2}, [1, 2, 3]) is False
    with pytest.raises(ValueError):
        is_surjective({0: 1, 1: 9}, [1, 2])
    assert is_surjective(bytes_to_unicode(), unicode_to_bytes().keys()) is True
