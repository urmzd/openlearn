"""Reference learner tests for L1.3 (rung R2): bodies for the names and
docstrings in the chapter, section 4. Only contract names are imported."""

import pytest
from tinyllm.tok.wordpiece import WordPieceTokenizer

VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "the",
    "un",
    "##aff",
    "##able",
    "##a",
    "!",
    "u",
    "##n",
    "cafe",
    "##s",
]


def tok(**kw):
    return WordPieceTokenizer({t: i for i, t in enumerate(VOCAB)}, **kw)


def test_unaffable_is_three_pieces():
    """ "The UNAFFABLE!" encodes to the, un, ##aff, ##able, !."""
    assert tok().encode("The UNAFFABLE!") == [5, 6, 7, 8, 10]


def test_unsegmentable_word_is_one_unk():
    """ "unaffablex" is a single [UNK], not three pieces and an [UNK]."""
    assert tok().encode("unaffablex") == [1]


def test_continuation_pieces_only_inside_words():
    """ "unn" is un ##n; "aff" alone is [UNK]."""
    assert tok().wordpiece("unn") == [6, 12] and tok().wordpiece("aff") == [1]


def test_word_length_limit_is_inclusive():
    """With a limit of 5, "unaff" segments and "unaffa" is [UNK]."""
    t = tok(max_input_chars_per_word=5)
    assert t.wordpiece("unaff") == [6, 7] and t.wordpiece("unaffa") == [1]


def test_cleaning_drops_controls_and_spaces_white_space():
    """U+000B and U+200B vanish; tab and U+00A0 become spaces."""
    assert tok().normalize("a\x0bb​c\td e") == "abc d e"


def test_accent_rules():
    """Uncased strips accents; cased keeps them; strip_accents=False keeps them."""
    assert tok().normalize("Café") == "cafe"
    assert tok(lowercase=False).normalize("Café") == "Café"
    assert tok(strip_accents=False).normalize("Café") == "café"


def test_spacing_marks_survive():
    """The Devanagari vowel sign U+093E (category Mc) is not stripped."""
    assert tok().normalize("ना") == "ना"


def test_ascii_symbols_and_cjk_split():
    """$ + ^ are punctuation; each CJK ideograph is its own word."""
    assert tok().basic_tokenize("$5+x^2") == ["$", "5", "+", "x", "^", "2"]
    assert tok().basic_tokenize("日本") == ["日", "本"]


def test_sigma_lowercases_per_code_point():
    """ "ΣΑΣ" lowercases to "σασ" (no final sigma)."""
    assert tok().normalize("ΣΑΣ") == "σασ"


def test_mask_in_text_and_template():
    """ "[MASK]" is one id, "[mask]" is not; add_special wraps [CLS] ... [SEP]."""
    assert tok().encode("the [MASK]") == [5, 4]
    assert 4 not in tok().encode("the [mask]")
    assert tok().encode("the", add_special=True) == [2, 5, 3]


def test_decode_glues_and_cleans():
    """[the, un, ##aff, ##able, !] decodes to "the unaffable!"."""
    assert tok().decode([5, 6, 7, 8, 10]) == "the unaffable!"
    assert tok().decode([2, 5, 3], skip_special=True) == "the"


def test_vocab_file_and_save_load(tmp_path):
    """vocab.txt line n is id n; save then load keeps ids and casing."""
    (tmp_path / "vocab.txt").write_text(
        "".join(t + "\n" for t in VOCAB), encoding="utf-8"
    )
    t = WordPieceTokenizer.from_vocab(str(tmp_path / "vocab.txt"), lowercase=False)
    assert t.encode("the cafe") == [5, 13]
    t.save(str(tmp_path / "out"))
    back = WordPieceTokenizer.load(str(tmp_path / "out"))
    assert back.encode("The") == t.encode("The") == [1]


def test_vocab_without_unk_is_rejected():
    """A vocab with no [UNK], or with an id gap, is ValueError."""
    with pytest.raises(ValueError):
        WordPieceTokenizer({"a": 0})
    with pytest.raises(ValueError):
        WordPieceTokenizer({"[UNK]": 0, "a": 2})
