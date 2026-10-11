"""Course tests for L1.2: byte-level BPE (tinyllm/tok/bpe.py) and the GPT-2
pre-tokenizer (tinyllm/tok/pretok.py).

Rung R0 for these course tests (your own tests for this module are rung R2:
section 4 of the chapter lists their names). Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/L1.2), and the chapter section it comes from.

Oracles (course/oracle/tok/golden.py): the ids of 300 strings under GPT-2's
and SmolLM2's tokenizer.json, computed by Hugging Face tokenizers (GPT-2's
also by tiktoken), and the merges the Hugging Face BpeTrainer learns on
course/fixtures/L1.2/train.txt.

The chapter's worked example (section 3) trains on the four texts "hug",
"hug", "pug", "hugs" with min_freq 1 up to 259 ids: the merges are
(u, g), (h, ug), (p, ug), and "hugs pug" encodes to [257, 82, 220, 258].
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from _lib.pcg32 import PCG32
from tinyllm.tok.bpe import BPETokenizer
from tinyllm.tok.pretok import pretokenize_gpt2, split_digits

FX = Path(os.environ.get("TINYLLM_FIXTURES", ""))
HAND_TEXTS = ["hug", "hug", "pug", "hugs"]
ALPHABET = list("ab hgpsu'.,!?0129\t\n") + [
    "  ",
    "é",
    "́",
    "日",
    "\U0001f642",
    " ",
    "　",
    "\x1c",
    "½",
    "'s",
    "'T",
]

_cache: dict[str, BPETokenizer] = {}


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def cases(name: str) -> list[dict]:
    with open(FX / name / "cases.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def gpt2() -> BPETokenizer:
    if "gpt2" not in _cache:
        _cache["gpt2"] = BPETokenizer.from_hf_json(
            str(FX / "tok-gpt2" / "tokenizer.json")
        )
    return _cache["gpt2"]


def smollm2() -> BPETokenizer:
    if "smollm2" not in _cache:
        _cache["smollm2"] = BPETokenizer.from_hf_json(
            str(FX / "tok-smollm2" / "tokenizer.json")
        )
    return _cache["smollm2"]


def random_text(rng: PCG32, n: int) -> str:
    return "".join(ALPHABET[rng.below(len(ALPHABET))] for _ in range(n))


# --- training -----------------------------------------------------------------


def test_hand_example_training():
    # WHY: section 3 by hand. Counts are pair counts summed over words times
    #      each word's frequency: (u, g) = 4 beats (h, u) = 3. After two merges
    #      (p, ug) and (hug, s) tie at 1, and the tie goes to the pair with
    #      the smaller ids: p is a byte symbol (79), hug a merged token (257).
    # KIND: unit
    # CATCHES: s01
    # CHAPTER: L1.2 section 3, Worked example by hand
    tok = BPETokenizer.train(HAND_TEXTS, vocab_size=259, min_freq=1)
    assert tok.merges == [("u", "g"), ("h", "ug"), ("p", "ug")]
    assert tok.vocab_size == 259
    assert (
        tok.token_to_id("ug") == 256
        and tok.token_to_id("hug") == 257
        and tok.token_to_id("pug") == 258
    )
    assert (
        tok.token_to_id("!") == 0
        and tok.token_to_id("g") == 70
        and tok.token_to_id("Ġ") == 220
    )


def test_hand_example_encode():
    # WHY: "hugs pug" pre-tokenizes to "hugs" and " pug" (the space joins the
    #      word after it, as "Ġ"); merges replay by rank: [hug, s, Ġ, pug].
    # KIND: unit
    # CATCHES: s12
    # CHAPTER: L1.2 section 3, Worked example by hand
    tok = BPETokenizer.train(HAND_TEXTS, vocab_size=259, min_freq=1)
    assert tok.encode("hugs pug") == [257, 82, 220, 258]
    assert tok.decode([257, 82, 220, 258]) == "hugs pug"
    assert [tok.id_to_token(i) for i in [257, 82, 220, 258]] == ["hug", "s", "Ġ", "pug"]


def test_min_freq_stops_training():
    # WHY: with the default min_freq = 2 the two pairs seen once are never
    #      merged: training stops after (u, g) and (h, ug).
    # KIND: boundary
    # CATCHES: m01
    # CHAPTER: L1.2 section 2, Principles
    tok = BPETokenizer.train(HAND_TEXTS, vocab_size=1000)
    assert tok.merges == [("u", "g"), ("h", "ug")]
    assert tok.vocab_size == 258
    # a count equal to min_freq still merges: (h, ug) has count 3
    assert len(BPETokenizer.train(HAND_TEXTS, vocab_size=1000, min_freq=3).merges) == 2
    assert len(BPETokenizer.train(HAND_TEXTS, vocab_size=1000, min_freq=4).merges) == 1


def test_trainer_matches_hf_merges():
    # WHY: the same merges, in the same order, as the Hugging Face BpeTrainer
    #      on a 3 KB story, for five (vocab_size, min_freq, specials) configs:
    #      the documented tie rule and the per-word counts are what HF does.
    # KIND: golden
    # CATCHES: s01, s02, s05
    # CHAPTER: L1.2 section 2, Principles
    lines = (
        (FX / "L1.2" / "train.txt")
        .read_text(encoding="utf-8")
        .splitlines(keepends=True)
    )
    golden = json.loads((FX / "L1.2" / "train_merges.json").read_text(encoding="utf-8"))
    for g in golden:
        tok = BPETokenizer.train(lines, g["vocab_size"], g["specials"], g["min_freq"])
        assert [list(m) for m in tok.merges] == g["merges"], (
            g["vocab_size"],
            g["min_freq"],
        )
        assert (
            len(tok.vocab) + len(tok.added) - len(set(tok.vocab) & set(tok.added))
            == g["n_vocab"]
        )


def test_specials_are_not_trained_on():
    # WHY: special tokens are ids 0.. and are cut out of the training text,
    #      so "<|endoftext|>" between documents never becomes merges like "<|".
    # KIND: boundary
    # CATCHES: s06
    # CHAPTER: L1.2 section 5, Pitfalls, item 5
    texts = ["ab<|endoftext|>ab<|endoftext|>ab<|endoftext|>"] * 3
    tok = BPETokenizer.train(texts, vocab_size=300, specials=["<|endoftext|>"])
    assert tok.special_ids == {"<|endoftext|>": 0}
    assert tok.merges == [("a", "b")]
    assert tok.encode("ab<|endoftext|>") == [tok.token_to_id("ab"), 0]


def test_train_rejects_bad_args():
    # WHY: the 256 byte symbols and the specials must fit; min_freq 0 would
    #      merge pairs that do not occur.
    # KIND: boundary
    # CATCHES: m02
    # CHAPTER: L1.2 section 4, The interface
    with pytest.raises(ValueError):
        BPETokenizer.train(["ab"], vocab_size=256, specials=["<s>"])
    with pytest.raises(ValueError):
        BPETokenizer.train(["ab"], vocab_size=300, min_freq=0)
    assert BPETokenizer.train(["ab"], vocab_size=256).merges == []


# --- the pre-tokenizer -----------------------------------------------------------


def test_hand_example_pretokenize():
    # WHY: section 3: a contraction is its own piece, a run of three spaces
    #      gives two to the white-space piece and one to " here", and the
    #      final newline is white space at the end.
    # KIND: unit
    # CATCHES: s08
    # CHAPTER: L1.2 section 3, Worked example by hand
    assert pretokenize_gpt2("I'm   here!!\n") == ["I", "'m", "  ", " here", "!!", "\n"]
    assert pretokenize_gpt2("don't 2024 café") == ["don", "'t", " 2024", " café"]


def test_pretokenize_white_space_rules():
    # WHY: \s is Unicode White_Space (U+00A0 and U+3000 are, U+001C is not,
    #      although str.isspace says it is); contractions are lowercase only;
    #      only U+0020 may lead a word, never a tab.
    # KIND: boundary
    # CATCHES: s07, s09, s10
    # CHAPTER: L1.2 section 5, Pitfalls, item 2
    assert pretokenize_gpt2("a b") == ["a", " ", "b"]
    assert pretokenize_gpt2("a\x1c\x1cb") == ["a", "\x1c\x1c", "b"]
    assert pretokenize_gpt2("IT'S") == ["IT", "'", "S"]
    assert pretokenize_gpt2("\tx") == ["\t", "x"]
    assert pretokenize_gpt2("x  ") == ["x", "  "]
    assert pretokenize_gpt2(" ½²") == [" ½²"]
    assert pretokenize_gpt2("") == []


def test_pretokenize_property():
    # WHY: pre-tokens partition the text: they join back to it and none is
    #      empty, for any text (otherwise decode(encode(x)) loses characters).
    # KIND: property
    # CATCHES: s08
    # CHAPTER: L1.2 section 2, Principles
    rng = PCG32(seed(), 21)
    for _ in range(300):
        t = random_text(rng, rng.below(25))
        pieces = pretokenize_gpt2(t)
        assert "".join(pieces) == t and all(pieces)
        # (?!\S): a white-space piece of two or more characters never sits
        # right before a piece that starts with a non-space character
        for p, q in zip(pieces, pieces[1:]):
            if len(p) >= 2 and p.isspace() and p.strip(" ") == "":
                assert q[0].isspace(), (t, pieces)


def test_split_digits():
    # WHY: SmolLM2 runs Digits(individual_digits) before the regex, so every
    #      number is spelled one digit per token; a run stays whole otherwise.
    # KIND: unit
    # CATCHES: s22
    # CHAPTER: L1.2 section 2, Principles
    assert split_digits("ab 123x½") == ["ab ", "1", "2", "3", "x", "½"]
    assert split_digits("ab 123x", individual=False) == ["ab ", "123", "x"]
    assert split_digits("") == []


# --- GPT-2 and SmolLM2 compatibility ------------------------------------------------


def test_gpt2_ids_match_oracle():
    # WHY: GPT-2's own tokenizer.json, 300 strings (emoji, CJK, white-space
    #      runs, contractions, control characters, <|endoftext|>): ids equal
    #      Hugging Face's and tiktoken's exactly. Every merge rule, the byte
    #      map, and the regex are exercised against a real vocabulary.
    # KIND: golden
    # CATCHES: s03, s04, s08, s09, s10, s12, m03
    # CHAPTER: L1.2 section 4, The interface
    tok = gpt2()
    assert tok.vocab_size == 50257
    for c in cases("tok-gpt2"):
        assert tok.encode(c["text"]) == c["ids"], repr(c["text"])


def test_gpt2_files_load_like_tokenizer_json(tmp_path):
    # WHY: vocab.json + merges.txt (with its "#version" header line) is the
    #      original GPT-2 release; from_gpt2 must give the same ids, including
    #      <|endoftext|> as one special id.
    # KIND: golden
    # CATCHES: s13
    # CHAPTER: L1.2 section 4, The interface
    doc = json.loads((FX / "tok-gpt2" / "tokenizer.json").read_text(encoding="utf-8"))
    (tmp_path / "vocab.json").write_text(
        json.dumps(doc["model"]["vocab"]), encoding="utf-8"
    )
    lines = ["#version: 0.2"] + [
        m if isinstance(m, str) else " ".join(m) for m in doc["model"]["merges"]
    ]
    (tmp_path / "merges.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tok = BPETokenizer.from_gpt2(
        str(tmp_path / "vocab.json"), str(tmp_path / "merges.txt")
    )
    assert tok.special_ids == {"<|endoftext|>": 50256}
    for c in cases("tok-gpt2")[:120]:
        assert tok.encode(c["text"]) == c["ids"], repr(c["text"])


def test_smollm2_ids_match_oracle():
    # WHY: SmolLM2 declares Sequence[Digits(individual), ByteLevel] and 17
    #      added tokens; from_hf_json must honor the declared sequence.
    # KIND: golden
    # CATCHES: s11, s12, s14
    # CHAPTER: L1.2 section 4, The interface
    tok = smollm2()
    assert tok.vocab_size == 49152 and len(tok.special_ids) == 17
    for c in cases("tok-smollm2"):
        assert tok.encode(c["text"]) == c["ids"], repr(c["text"])


def test_smollm2_drops_unrepresentable_bytes():
    # WHY: SmolLM2's vocab has no symbol for six control bytes (0x04 among
    #      them) and no unk token; Hugging Face drops such a byte silently,
    #      and matching ids means doing the same. Decoding cannot restore it.
    # KIND: boundary
    # CATCHES: s15
    # CHAPTER: L1.2 section 5, Pitfalls, item 6
    tok = smollm2()
    ids = tok.encode("a\x04b")
    assert ids == [tok.token_to_id("a"), tok.token_to_id("b")]
    assert tok.decode(ids) == "ab"


def test_decode_matches_oracle():
    # WHY: decode concatenates token BYTES and decodes UTF-8 once, with
    #      replacement; skip_special drops special added tokens.
    # KIND: golden
    # CATCHES: s16, s17
    # CHAPTER: L1.2 section 4, The interface
    for name, tok in (("tok-gpt2", gpt2()), ("tok-smollm2", smollm2())):
        for c in cases(name):
            assert tok.decode(c["ids"]) == c["decoded"], repr(c["text"])
    s = smollm2()
    assert s.decode(s.encode("<|im_start|>hi<|im_end|>"), skip_special=True) == "hi"


def test_decode_joins_bytes_before_utf8():
    # WHY: "日" is three bytes; when its bytes land in two tokens, decoding
    #      each token alone gives U+FFFD, decoding them together gives "日".
    # KIND: boundary
    # CATCHES: s16
    # CHAPTER: L1.2 section 5, Pitfalls, item 3
    tok = BPETokenizer.train(["x"], vocab_size=256)
    ids = tok.encode("日")
    assert len(ids) == 3
    assert tok.decode(ids) == "日"
    assert tok.decode(ids[:1]) == "�"


def test_added_tokens_split_first_leftmost_longest():
    # WHY: added tokens are cut out before pre-tokenizing (the regex would
    #      split "<|im_start|>" into "<|", "im", "_", ...); a prefix of one
    #      ("<|im") is plain text.
    # KIND: unit
    # CATCHES: s14
    # CHAPTER: L1.2 section 2, Principles
    s = smollm2()
    ids = s.encode("<|im_start|>user")
    assert ids[0] == 1 and s.decode(ids[1:]) == "user"
    assert 1 not in s.encode("<|im")
    assert gpt2().encode("a<|endoftext|>") == [64, 50256]


def test_merges_apply_by_rank():
    # WHY: encoding replays merges by rank, lowest first, not left to right:
    #      with (b, c) ranked before (a, b), "abc" is [a, bc]; swap the ranks
    #      and it is [ab, c].
    # KIND: unit
    # CATCHES: s03
    # CHAPTER: L1.2 section 5, Pitfalls, item 1
    base = BPETokenizer.train(["x"], vocab_size=256)
    v = dict(base.vocab)
    v["bc"], v["ab"] = 256, 257
    t1 = BPETokenizer(v, [("b", "c"), ("a", "b")])
    t2 = BPETokenizer(v, [("a", "b"), ("b", "c")])
    assert [t1.id_to_token(i) for i in t1.encode("abc")] == ["a", "bc"]
    assert [t2.id_to_token(i) for i in t2.encode("abc")] == ["ab", "c"]


def test_rank_ties_go_leftmost():
    # WHY: in "aaa" the pair (a, a) occurs twice; the leftmost one merges
    #      first, so the result is [aa, a], not [a, aa].
    # KIND: boundary
    # CATCHES: s04
    # CHAPTER: L1.2 section 5, Pitfalls, item 1
    base = BPETokenizer.train(["x"], vocab_size=256)
    v = dict(base.vocab)
    v["aa"] = 256
    t = BPETokenizer(v, [("a", "a")])
    assert [t.id_to_token(i) for i in t.encode("aaa")] == ["aa", "a"]
    assert [t.id_to_token(i) for i in t.encode("aaaa")] == ["aa", "aa"]


def test_roundtrip_property():
    # WHY: a trained byte-level BPE encodes every text, seen or not, and
    #      decode(encode(x)) == x: the 256 byte symbols are a fallback.
    # KIND: property
    # CATCHES: s16
    # CHAPTER: L1.2 section 2, Principles
    lines = (
        (FX / "L1.2" / "train.txt")
        .read_text(encoding="utf-8")
        .splitlines(keepends=True)
    )
    tok = BPETokenizer.train(lines, vocab_size=400)
    rng = PCG32(seed(), 22)
    for _ in range(200):
        t = random_text(rng, rng.below(30))
        assert tok.decode(tok.encode(t)) == t


def test_save_load_roundtrip(tmp_path):
    # WHY: save writes the tokenizer.json subset and load reads it back to a
    #      tokenizer with the same ids and specials (the engine loads it).
    # KIND: property
    # CATCHES: s18
    # CHAPTER: L1.2 section 4, The interface
    lines = (
        (FX / "L1.2" / "train.txt")
        .read_text(encoding="utf-8")
        .splitlines(keepends=True)
    )
    tok = BPETokenizer.train(lines, vocab_size=400, specials=["<|endoftext|>"])
    tok.save(str(tmp_path))
    doc = json.loads((tmp_path / "tokenizer.json").read_text(encoding="utf-8"))
    assert doc["model"]["type"] == "BPE" and doc["pre_tokenizer"]["type"] == "ByteLevel"
    assert (
        doc["added_tokens"][0]["content"] == "<|endoftext|>"
        and doc["added_tokens"][0]["special"] is True
    )
    back = BPETokenizer.load(str(tmp_path))
    assert back.merges == tok.merges and back.special_ids == tok.special_ids
    for line in lines[:20] + ["unseen \U0001f642<|endoftext|>"]:
        assert back.encode(line) == tok.encode(line)


def test_from_hf_json_rejects_outside_subset(tmp_path):
    # WHY: a loader that ignores a field it does not implement (a normalizer,
    #      dropout, an unk token, byte fallback, a template) encodes wrong ids
    #      without a word; it must refuse and name the field.
    # KIND: boundary
    # CATCHES: s19
    # CHAPTER: L1.2 section 5, Pitfalls, item 4
    good = json.loads((FX / "tok-gpt2" / "tokenizer.json").read_text(encoding="utf-8"))
    edits = [
        ("normalizer", {"type": "NFC"}),
        ("model.dropout", 0.1),
        ("model.unk_token", "<unk>"),
        ("model.byte_fallback", True),
        (
            "post_processor",
            {
                "type": "TemplateProcessing",
                "single": [],
                "pair": [],
                "special_tokens": {},
            },
        ),
        ("pre_tokenizer", {"type": "Metaspace", "replacement": "▁"}),
        ("version", "2.0"),
    ]
    for key, value in edits:
        doc = json.loads(json.dumps(good))
        node, last = doc, key.split(".")
        for k in last[:-1]:
            node = node[k]
        node[last[-1]] = value
        p = tmp_path / "t.json"
        p.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(ValueError):
            BPETokenizer.from_hf_json(str(p))


def test_encode_batch_equals_encode():
    # WHY: encode_batch is the API the corpus pipeline calls (and L1.5's Rust
    #      port parallelizes); it is defined as encode on each text.
    # KIND: property
    # CATCHES: s20
    # CHAPTER: L1.2 section 4, The interface
    texts = [c["text"] for c in cases("tok-gpt2")[:60]]
    assert gpt2().encode_batch(texts) == [gpt2().encode(t) for t in texts]
    assert gpt2().encode_batch([]) == []


def test_ids_must_be_contiguous():
    # WHY: vocab_size is max id + 1 and every id below it must decode; a
    #      gap means a broken file, and decode would raise on a valid id.
    # KIND: boundary
    # CATCHES: s21
    # CHAPTER: L1.2 section 4, The interface
    base = BPETokenizer.train(["x"], vocab_size=256)
    v = dict(base.vocab)
    v["xx"] = 300
    with pytest.raises(ValueError):
        BPETokenizer(v, [("x", "x")])
