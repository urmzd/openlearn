"""Reference learner tests for L1.2 (rung R2): bodies for the names and
docstrings in the chapter, section 4. Only contract names are imported."""

import pytest
from tinyllm.tok.bpe import BPETokenizer
from tinyllm.tok.pretok import pretokenize_gpt2, split_digits

HUG = ["hug", "hug", "pug", "hugs"]


def byte_level(extra: dict[str, int], merges):
    v = dict(BPETokenizer.train(["x"], vocab_size=256).vocab)
    v.update(extra)
    return BPETokenizer(v, merges)


def test_hand_example_merges_and_tie():
    """hug hug pug hugs, min_freq 1, 259 ids: merges (u,g), (h,ug), (p,ug)."""
    tok = BPETokenizer.train(HUG, vocab_size=259, min_freq=1)
    assert tok.merges == [("u", "g"), ("h", "ug"), ("p", "ug")]
    assert tok.encode("hugs pug") == [257, 82, 220, 258]


def test_count_equal_to_min_freq_still_merges():
    """min_freq 3 keeps the (h, ug) merge, whose count is exactly 3."""
    assert len(BPETokenizer.train(HUG, vocab_size=999, min_freq=3).merges) == 2


def test_lowest_rank_merges_first():
    """With (b,c) ranked before (a,b), "abc" encodes as [a, bc]."""
    t = byte_level({"bc": 256, "ab": 257}, [("b", "c"), ("a", "b")])
    assert [t.id_to_token(i) for i in t.encode("abc")] == ["a", "bc"]


def test_equal_ranks_merge_leftmost():
    """With the one merge (a,a), "aaa" encodes as [aa, a]."""
    t = byte_level({"aa": 256}, [("a", "a")])
    assert [t.id_to_token(i) for i in t.encode("aaa")] == ["aa", "a"]


def test_pretokenizer_examples():
    """Contractions, white-space runs, U+001C, U+00A0, a leading tab, and ½."""
    assert pretokenize_gpt2("I'm   here!!\n") == ["I", "'m", "  ", " here", "!!", "\n"]
    assert pretokenize_gpt2("a\x1c\x1cb") == ["a", "\x1c\x1c", "b"]
    assert pretokenize_gpt2("a b") == ["a", " ", "b"]
    assert pretokenize_gpt2("IT'S") == ["IT", "'", "S"]
    assert pretokenize_gpt2("\tx") == ["\t", "x"]
    assert pretokenize_gpt2(" ½") == [" ½"]


def test_split_digits_both_modes():
    """individual: one piece per digit; otherwise one piece per run."""
    assert split_digits("a12b") == ["a", "1", "2", "b"]
    assert split_digits("a12b", individual=False) == ["a", "12", "b"]


def test_character_split_across_tokens_decodes_together():
    """The three bytes of a CJK character decode to it only together."""
    tok = BPETokenizer.train(["x"], vocab_size=256)
    ids = tok.encode("日")
    assert tok.decode(ids) == "日" and tok.decode(ids[:1]) == "�"


def test_special_token_is_one_id_and_never_trained_on():
    """A special is id 0, matched in text, skipped on request, never merged."""
    tok = BPETokenizer.train(["ab<|e|>ab<|e|>"] * 3, vocab_size=300, specials=["<|e|>"])
    assert tok.merges == [("a", "b")]
    ids = tok.encode("ab<|e|>")
    assert ids[-1] == 0
    assert tok.decode(ids, skip_special=True) == "ab"


def test_save_load_keeps_specials(tmp_path):
    """load(save(t)) has the same merges, specials, and ids."""
    tok = BPETokenizer.train(
        ["hello hello world"] * 3, vocab_size=270, specials=["<s>"]
    )
    tok.save(str(tmp_path))
    back = BPETokenizer.load(str(tmp_path))
    assert back.special_ids == {"<s>": 0} and back.encode("<s>hello") == tok.encode(
        "<s>hello"
    )


def test_bad_arguments_raise():
    """vocab_size below 256 + specials, or min_freq 0, is ValueError."""
    with pytest.raises(ValueError):
        BPETokenizer.train(["a"], vocab_size=256, specials=["<s>"])
    with pytest.raises(ValueError):
        BPETokenizer.train(["a"], vocab_size=300, min_freq=0)


def test_round_trip_any_text():
    """decode(encode(x)) == x for text with emoji, accents, tabs, and runs of spaces."""
    tok = BPETokenizer.train(["the cat sat on the mat"] * 4, vocab_size=280)
    for x in ["", "the  cat\t\U0001f642", "café é 日本", "  \n "]:
        assert tok.decode(tok.encode(x)) == x
    assert tok.encode_batch(["", "cat"]) == [tok.encode(""), tok.encode("cat")]
