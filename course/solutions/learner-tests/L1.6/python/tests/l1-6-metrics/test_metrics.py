"""Model test suite for L1.6: the bodies of the test names and docstrings
listed in the chapter, section 4. They import only names in
contracts/py/tinyllm/tok/{base,char,metrics}.pyi, and each one fails on a
solution with the bug its name describes."""

import pytest
from tinyllm.tok.base import ByteTokenizer
from tinyllm.tok.char import CharTokenizer
from tinyllm.tok.metrics import (
    byte_fallback_rate,
    bytes_per_token,
    fertility,
    is_fallback,
)


def test_bytes_per_token_counts_utf8_bytes():
    """ByteTokenizer on "naïve café" gives 1.0: 12 bytes, 12 tokens, not 10 code points."""
    assert bytes_per_token(ByteTokenizer(), ["naïve café"]) == 1.0
    assert bytes_per_token(CharTokenizer.train(["naïve"]), ["naïve café"]) == 1.2


def test_bytes_per_token_is_a_ratio_of_totals():
    """Over ["ab", "é"] with a char tokenizer trained on "ab": 4 bytes / 3 tokens."""
    assert bytes_per_token(CharTokenizer.train(["ab"]), ["ab", "é"]) == 4 / 3


def test_fertility_encodes_each_word_alone():
    """ByteTokenizer: fertility(["naïve", "café"]) is (6 + 5) / 2 = 5.5, no space tokens."""
    assert fertility(ByteTokenizer(), ["naïve", "café"]) == 5.5


def test_fertility_counts_repeated_words():
    """["a", "a", "bb"] in bytes is 4 / 3."""
    assert fertility(ByteTokenizer(), ["a", "a", "bb"]) == 4 / 3


def test_specials_are_not_added():
    """A char tokenizer with <bos> and <eos>: fertility(["hi"]) is 2, not 4."""
    tok = CharTokenizer.train(["hi"], specials=("<bos>", "<eos>"))
    assert fertility(tok, ["hi"]) == 2.0
    assert bytes_per_token(tok, ["hi"]) == 1.0


def test_unk_and_partial_bytes_are_fallbacks():
    """The unknown id and a lone UTF-8 lead byte are fallbacks; "a" and <bos> are not."""
    tok = CharTokenizer.train(["a"], specials=("<bos>",))
    assert is_fallback(tok, tok.unk_id)
    assert not is_fallback(tok, tok.special_ids["<bos>"])
    assert not is_fallback(tok, tok.token_to_id("a"))
    assert is_fallback(ByteTokenizer(), 0xC3)


def test_fallback_rate_counts_occurrences():
    """ "éé" in bytes is 4 fallbacks of 4 tokens: rate 1.0; "naïve café" in chars is 0.4."""
    assert byte_fallback_rate(ByteTokenizer(), ["éé"]) == 1.0
    assert byte_fallback_rate(CharTokenizer.train(["naïve"]), ["naïve café"]) == 0.4


def test_empty_inputs_raise():
    """No words, or texts with no tokens, are ValueError."""
    b = ByteTokenizer()
    with pytest.raises(ValueError):
        fertility(b, [])
    with pytest.raises(ValueError):
        bytes_per_token(b, [""])
    with pytest.raises(ValueError):
        byte_fallback_rate(b, [])
