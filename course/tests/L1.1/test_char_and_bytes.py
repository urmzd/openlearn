"""Course tests for L1.1: the Tokenizer protocol, the byte tokenizer
(tinyllm/tok/base.py), and the char tokenizer (tinyllm/tok/char.py).

Rung R0 for these course tests (your own tests for this module are rung R2:
section 4 of the chapter lists their names). Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/L1.1), and the chapter section it comes from.

The chapter's worked example (section 3) trains a char tokenizer on the two
texts "hello" and "hi" with the specials <bos> and <eos>:

    id    0      1      2      3  4  5  6  7
    text  <unk>  <bos>  <eos>  e  h  i  l  o
"""

from __future__ import annotations

import json
import os
import unicodedata

import pytest
from _lib.pcg32 import PCG32
from tinyllm.tok.base import ByteTokenizer, Tokenizer, check_ids
from tinyllm.tok.char import CharTokenizer

HAND_VOCAB = ["<unk>", "<bos>", "<eos>", "e", "h", "i", "l", "o"]

# Code points from many scripts and planes: ASCII, Latin-1, a combining
# accent, Greek, CJK, Hangul, an astral emoji, a ZWJ, white space variants.
ALPHABET = list("abcXYZ019 .,!?'\t\n") + [
    "é",
    "́",
    "ß",
    "Σ",
    "σ",
    "日",
    "本",
    "한",
    "\U0001f642",
    "\U0001f468",
    "‍",
    " ",
    "　",
    "\U0001d54f",
    "�",
]


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def random_text(rng: PCG32, n: int) -> str:
    return "".join(ALPHABET[rng.below(len(ALPHABET))] for _ in range(n))


def hand() -> CharTokenizer:
    return CharTokenizer.train(["hello", "hi"], specials=("<bos>", "<eos>"))


# --- the char tokenizer: the worked example -----------------------------------


def test_hand_example_vocab():
    # WHY: the chapter's table, id for id: <unk> is always 0, the specials
    #      follow in the order given, then the code points sorted by code
    #      point (not by first appearance, which would make ids depend on the
    #      order of the training texts).
    # KIND: unit
    # CATCHES: s01, s02
    # CHAPTER: L1.1 section 3, Worked example by hand
    tok = hand()
    assert tok.vocab_size == 8
    assert [tok.id_to_token(i) for i in range(8)] == HAND_VOCAB
    assert tok.special_ids == {"<unk>": 0, "<bos>": 1, "<eos>": 2}
    assert tok.unk_id == 0
    assert tok.token_to_id("l") == 6 and tok.token_to_id("z") is None
    assert (
        CharTokenizer.train(["hi", "hello"], specials=("<bos>", "<eos>")).vocab
        == HAND_VOCAB
    )


def test_hand_example_encode_decode():
    # WHY: "hello" is [4, 3, 6, 6, 7]; the unseen "!" becomes <unk> (0), it
    #      does not raise; decode writes the special's text unless asked to
    #      skip specials.
    # KIND: unit
    # CATCHES: s03, s04
    # CHAPTER: L1.1 section 3, Worked example by hand
    tok = hand()
    assert tok.encode("hello") == [4, 3, 6, 6, 7]
    assert tok.encode("hi!") == [4, 5, 0]
    assert tok.decode([4, 5, 0]) == "hi<unk>"
    assert tok.decode([4, 5, 0], skip_special=True) == "hi"
    assert tok.decode([]) == "" and tok.encode("") == []


def test_add_special_wraps_bos_eos():
    # WHY: add_special puts <bos> first and <eos> last, once each, and only
    #      when those roles exist; it never changes the ids of the text.
    # KIND: unit
    # CATCHES: s05
    # CHAPTER: L1.1 section 4, The interface
    assert hand().encode("hi", add_special=True) == [1, 4, 5, 2]
    plain = CharTokenizer.train(["hi"])
    assert plain.encode("hi", add_special=True) == plain.encode("hi") == [1, 2]
    only_bos = CharTokenizer.train(["hi"], specials=("<bos>",))
    assert only_bos.encode("hi", add_special=True) == [1, 2, 3]


def test_no_normalization():
    # WHY: U+00E9 and "e" + U+0301 render the same but are different texts; a
    #      tokenizer that normalizes (NFC) cannot give the decomposed one back.
    # KIND: boundary
    # CATCHES: s06
    # CHAPTER: L1.1 section 5, Pitfalls, item 1
    composed, decomposed = "café", "café"
    assert unicodedata.normalize("NFC", decomposed) == composed
    tok = CharTokenizer.train([composed, decomposed])
    assert tok.encode(composed) != tok.encode(decomposed)
    assert len(tok.encode(decomposed)) == 5
    assert tok.decode(tok.encode(decomposed)) == decomposed


def test_roundtrip_unicode_property():
    # WHY: decode(encode(x)) == x for every text whose code points were seen
    #      in training, across planes (astral emoji are one code point, one id)
    #      and with combining marks and white space kept as they are.
    # KIND: property
    # CATCHES: s06, s07
    # CHAPTER: L1.1 section 2, Principles
    rng = PCG32(seed(), 11)
    texts = [random_text(rng, 1 + rng.below(40)) for _ in range(200)]
    tok = CharTokenizer.train(texts)
    for t in texts:
        ids = tok.encode(t)
        assert len(ids) == len(t)
        assert tok.decode(ids) == t
    assert tok.encode("\U0001f642") == [tok.token_to_id("\U0001f642")]


def test_special_text_is_not_matched():
    # WHY: special texts in the input are ordinary characters. If a user's
    #      prompt could type "<eos>" and get the end-of-sequence id, any text
    #      could stop generation (formats/tokenizer.md, the char tokenizer).
    # KIND: boundary
    # CATCHES: s08
    # CHAPTER: L1.1 section 5, Pitfalls, item 5
    tok = CharTokenizer.train(["<eos>"], specials=("<eos>",))
    ids = tok.encode("<eos>")
    assert len(ids) == 5 and tok.special_ids["<eos>"] not in ids
    assert tok.decode(ids) == "<eos>"


def test_decode_rejects_bad_ids():
    # WHY: -1 must be an error, not the last vocab entry (Python indexing
    #      wraps negative indexes); out-of-range ids, floats, and bools too.
    # KIND: boundary
    # CATCHES: s09, m01, m02
    # CHAPTER: L1.1 section 5, Pitfalls, item 4
    tok = hand()
    for bad in ([-1], [8], [1.0], [True], ["1"]):
        with pytest.raises(ValueError):
            tok.decode(bad)
    with pytest.raises(ValueError):
        tok.id_to_token(-1)
    assert check_ids([0, 7], 8) == [0, 7]
    with pytest.raises(ValueError):
        check_ids([8], 8)


def test_save_writes_the_char_format():
    # WHY: the file is formats/tokenizer.md's tinyllm_char.json: `specials`
    #      maps the roles unk, bos, eos to ids (not texts to ids), so any
    #      reader of the format, in any language, finds <unk> the same way.
    # KIND: unit
    # CATCHES: s10
    # CHAPTER: L1.1 section 4, The interface
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        hand().save(d)
        with open(os.path.join(d, "tinyllm_char.json"), encoding="utf-8") as f:
            doc = json.load(f)
        assert doc == {
            "type": "char",
            "vocab": HAND_VOCAB,
            "specials": {"unk": 0, "bos": 1, "eos": 2},
        }
        back = CharTokenizer.load(d)
        assert back.encode("hello!") == hand().encode("hello!")
        assert back.special_ids == hand().special_ids


def test_load_rejects_invalid_files():
    # WHY: a reader never trusts the file: no `unk`, a duplicate entry, an
    #      entry of two code points, a role outside unk/bos/eos/pad, or a
    #      special id past the vocabulary is a ValueError, not a tokenizer
    #      that silently encodes wrong.
    # KIND: boundary
    # CATCHES: s11, s12
    # CHAPTER: L1.1 section 5, Pitfalls, item 6
    import tempfile

    bad_docs = [
        {"type": "char", "vocab": ["<unk>", "a"], "specials": {}},
        {"type": "char", "vocab": ["<unk>", "a", "a"], "specials": {"unk": 0}},
        {"type": "char", "vocab": ["<unk>", "ab"], "specials": {"unk": 0}},
        {"type": "char", "vocab": ["<unk>", "a"], "specials": {"unk": 0, "mask": 1}},
        {"type": "char", "vocab": ["<unk>", "a"], "specials": {"unk": 2}},
        {"type": "bpe", "vocab": ["<unk>"], "specials": {"unk": 0}},
    ]
    for doc in bad_docs:
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "tinyllm_char.json"), "w", encoding="utf-8") as f:
                json.dump(doc, f)
            with pytest.raises(ValueError):
                CharTokenizer.load(d)


def test_train_rejects_unknown_special():
    # WHY: the format has four roles; a "<mask>" special has no place in it.
    # KIND: boundary
    # CATCHES: s13
    # CHAPTER: L1.1 section 4, The interface
    with pytest.raises(ValueError):
        CharTokenizer.train(["a"], specials=("<mask>",))
    tok = CharTokenizer.train(["a"], specials=("<pad>", "<pad>", "<unk>"))
    assert tok.vocab == ["<unk>", "<pad>", "a"]


# --- the protocol and the byte tokenizer ---------------------------------------


def test_both_satisfy_the_protocol():
    # WHY: every consumer (the metrics of L1.6, the corpus pipeline, the
    #      engine's tokenizer factory) is written against Tokenizer only, so
    #      both classes must carry every member, attributes included.
    # KIND: unit
    # CATCHES: s14
    # CHAPTER: L1.1 section 2, Principles
    for tok in (hand(), ByteTokenizer()):
        assert isinstance(tok, Tokenizer)
    b = ByteTokenizer()
    assert b.vocab_size == 256 and b.special_ids == {} and b.unk_id is None


def test_hand_example_bytes():
    # WHY: the byte tokenizer of the tracer, by hand: "hé" is [104, 195, 169]
    #      because é is two UTF-8 bytes; one byte of it alone decodes to U+FFFD.
    # KIND: unit
    # CATCHES: s15, s17
    # CHAPTER: L1.1 section 3, Worked example by hand
    b = ByteTokenizer()
    assert b.encode("hé") == [104, 195, 169]
    assert b.decode([104, 195, 169]) == "hé"
    assert b.decode([195]) == "�"
    assert b.decode([226, 130]) == "�"  # one maximal ill-formed subpart
    with pytest.raises(ValueError):
        b.decode([256])


def test_byte_tokens_are_the_gpt2_byte_map():
    # WHY: a byte's token string is its M05.2 byte-map character ("Ġ" for the
    #      space), the alphabet byte-level BPE starts from, so a byte token
    #      and a BPE token with the same text are the same symbol.
    # KIND: unit
    # CATCHES: s16
    # CHAPTER: L1.1 section 2, Principles
    b = ByteTokenizer()
    assert b.id_to_token(32) == "Ġ" and b.id_to_token(10) == "Ċ"
    assert b.id_to_token(65) == "A" and b.token_to_id("Ġ") == 32
    assert b.token_to_id("<0x41>") is None
    assert sorted(b.token_to_id(b.id_to_token(i)) for i in range(256)) == list(
        range(256)
    )


def test_byte_roundtrip_property():
    # WHY: every text survives encode then decode, whatever its script.
    # KIND: property
    # CATCHES: s17
    # CHAPTER: L1.1 section 2, Principles
    rng = PCG32(seed(), 12)
    b = ByteTokenizer()
    for _ in range(200):
        t = random_text(rng, rng.below(30))
        assert b.decode(b.encode(t)) == t
        assert all(0 <= i < 256 for i in b.encode(t))
