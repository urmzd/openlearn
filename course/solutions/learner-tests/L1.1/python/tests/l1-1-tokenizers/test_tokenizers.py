"""Model test suite for L1.1: the bodies of the test names and docstrings
listed in the chapter, section 4. They import only names in
contracts/py/tinyllm/tok/{base,char}.pyi, and each one fails on a solution
with the bug its name describes."""

import json

import pytest
from tinyllm.tok.base import ByteTokenizer, Tokenizer
from tinyllm.tok.char import CharTokenizer


def test_vocab_is_unk_specials_then_sorted_code_points():
    """train(["hello", "hi"], ("<bos>", "<eos>")) has vocab <unk> <bos> <eos> e h i l o."""
    tok = CharTokenizer.train(["hi", "hello"], specials=("<bos>", "<eos>"))
    assert [tok.id_to_token(i) for i in range(tok.vocab_size)] == [
        "<unk>",
        "<bos>",
        "<eos>",
        "e",
        "h",
        "i",
        "l",
        "o",
    ]


def test_unseen_character_is_unk():
    """An unseen character encodes to unk_id, it does not raise."""
    tok = CharTokenizer.train(["hi"])
    assert tok.encode("h?") == [tok.token_to_id("h"), tok.unk_id]


def test_skip_special_drops_unk_bos_eos():
    """decode(..., skip_special=True) drops every special id, <unk> included."""
    tok = CharTokenizer.train(["hi"], specials=("<bos>", "<eos>"))
    ids = tok.encode("hi?", add_special=True)
    assert ids[0] == tok.special_ids["<bos>"] and ids[-1] == tok.special_ids["<eos>"]
    assert tok.decode(ids, skip_special=True) == "hi"
    assert tok.decode(ids) == "<bos>hi<unk><eos>"


def test_decomposed_accent_round_trips():
    """ "e" + U+0301 survives encode and decode unchanged (no normalization)."""
    text = " café\t\n"
    tok = CharTokenizer.train([text])
    assert tok.decode(tok.encode(text)) == text
    assert len(tok.encode(text)) == len(text)


def test_special_text_in_input_is_plain_text():
    """Typing "<eos>" gives five character ids, never the eos id."""
    tok = CharTokenizer.train(["<eos>"], specials=("<eos>",))
    assert tok.special_ids["<eos>"] not in tok.encode("<eos>")


def test_negative_and_out_of_range_ids_raise():
    """decode([-1]) and decode([vocab_size]) are ValueError."""
    tok = CharTokenizer.train(["ab"])
    for bad in ([-1], [tok.vocab_size], [True]):
        with pytest.raises(ValueError):
            tok.decode(bad)


def test_save_load_round_trip(tmp_path):
    """load(save(t)) encodes like t, and the file's specials map roles to ids."""
    tok = CharTokenizer.train(["hello"], specials=("<pad>",))
    tok.save(str(tmp_path))
    doc = json.loads((tmp_path / "tinyllm_char.json").read_text(encoding="utf-8"))
    assert doc["specials"] == {"unk": 0, "pad": 1}
    assert CharTokenizer.load(str(tmp_path)).encode("hello!") == tok.encode("hello!")


def test_load_rejects_two_character_entry(tmp_path):
    """A vocab entry of two code points, or a role outside unk/bos/eos/pad, is ValueError."""
    for doc in (
        {"type": "char", "vocab": ["<unk>", "ab"], "specials": {"unk": 0}},
        {"type": "char", "vocab": ["<unk>", "a"], "specials": {"unk": 0, "sep": 1}},
    ):
        (tmp_path / "tinyllm_char.json").write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(ValueError):
            CharTokenizer.load(str(tmp_path))


def test_byte_tokenizer_utf8_and_replacement():
    """ByteTokenizer encodes UTF-8 bytes and decodes a lone continuation byte to U+FFFD."""
    b = ByteTokenizer()
    assert isinstance(b, Tokenizer)
    assert b.encode("é") == [0xC3, 0xA9]
    assert b.decode([0xC3]) == "�"
    assert b.id_to_token(32) == "Ġ"
