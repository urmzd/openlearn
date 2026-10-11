"""Course tests for L1.4: the Unigram LM tokenizer (tinyllm/tok/unigram.py).

Rung R0 for these course tests (your own tests for this module are rung R2:
section 4 of the chapter lists their names). Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/L1.4), and the chapter section it comes from.

Oracles (course/oracle/tok/golden.py): a Hugging Face UnigramTrainer
tokenizer.json and its ids for 300 strings, and a sentencepiece unigram
model (pieces and scores as tokenizer.json) with sentencepiece's ids for the
236 of those strings on which sentencepiece and Hugging Face agree.

The chapter's worked example (section 3) is the vocab HAND below: eight
pieces whose probabilities sum to 1, and the word "abc" (Metaspace: "▁abc")
with its six segmentations, total probability Z = 0.0451.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from collections import Counter
from pathlib import Path

import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.tok.unigram import UnigramTokenizer, metaspace

FX = Path(os.environ.get("TINYLLM_FIXTURES", ""))
S = "▁"
HAND_P = {
    S: 0.1,
    "a": 0.1,
    "b": 0.1,
    "c": 0.1,
    S + "a": 0.2,
    "ab": 0.15,
    "bc": 0.15,
    S + "ab": 0.1,
}
# the six segmentations of "▁abc" and their probabilities (section 3)
SEGS = {
    (S, "a", "b", "c"): 1e-4,
    (S + "a", "b", "c"): 2e-3,
    (S, "ab", "c"): 1.5e-3,
    (S, "a", "bc"): 1.5e-3,
    (S + "a", "bc"): 3e-2,
    (S + "ab", "c"): 1e-2,
}
Z = 0.0451
CHI2_DF5_P001 = 20.515  # chi-square, 5 degrees of freedom, p = 1e-3


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def hand() -> UnigramTokenizer:
    return UnigramTokenizer(
        [("<unk>", 0.0)] + [(p, math.log(q)) for p, q in HAND_P.items()]
    )


def toks(tok: UnigramTokenizer, ids: list[int]) -> tuple[str, ...]:
    return tuple(tok.id_to_token(i) for i in ids)


def cases(name: str) -> list[dict]:
    with open(FX / "L1.4" / name / "cases.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def chi_square(counts: Counter, probs: dict) -> float:
    n = sum(counts.values())
    return sum((counts[k] - n * p) ** 2 / (n * p) for k, p in probs.items())


def brute_force_em(tok: UnigramTokenizer, words: Counter) -> dict[str, float]:
    """One EM step by enumerating every segmentation of every word."""
    score = {p: s for p, s in tok.pieces[1:]}
    counts = Counter()
    for w, c in words.items():
        segs = []
        for cuts in itertools.product([0, 1], repeat=len(w) - 1):
            pieces, start = [], 0
            for k, cut in enumerate(cuts, 1):
                if cut:
                    pieces.append(w[start:k])
                    start = k
            pieces.append(w[start:])
            if all(p in score for p in pieces):
                segs.append((pieces, math.exp(sum(score[p] for p in pieces))))
        z = sum(pr for _, pr in segs)
        for pieces, pr in segs:
            for p in pieces:
                counts[p] += c * pr / z
    total = sum(counts.values())
    return {p: c / total for p, c in counts.items()}


# --- the worked example -----------------------------------------------------------


def test_hand_example_viterbi():
    # WHY: section 3: of the six segmentations of "▁abc" the most probable is
    #      ▁a + bc (0.2 x 0.15 = 0.03), ahead of ▁ab + c (0.01).
    # KIND: unit
    # CATCHES: s09
    # CHAPTER: L1.4 section 3, Worked example by hand
    tok = hand()
    assert toks(tok, tok.encode("abc")) == (S + "a", "bc")
    assert tok.encode("abc") == [5, 7]
    assert tok.decode([5, 7]) == "abc"


def test_hand_example_likelihood():
    # WHY: log_likelihood sums over ALL segmentations: ln Z = ln 0.0451,
    #      not the Viterbi path's ln 0.03 (that is hard EM's objective).
    # KIND: unit
    # CATCHES: s19
    # CHAPTER: L1.4 section 3, Worked example by hand
    assert_close(hand().log_likelihood(["abc"]), math.log(Z), rtol=1e-12, atol=1e-12)
    assert_close(
        hand().log_likelihood(["abc", "abc"]), 2 * math.log(Z), rtol=1e-12, atol=1e-12
    )


def test_hand_example_em_step():
    # WHY: section 3's E-step table: expected counts are posterior-weighted
    #      piece counts (▁a: (0.002 + 0.03) / Z), and the M-step divides by
    #      their total 0.0954 / Z, so p(▁a) becomes 0.032 / 0.0954.
    # KIND: unit
    # CATCHES: s08, s14
    # CHAPTER: L1.4 section 3, Worked example by hand
    new = hand().em_step(["abc"])
    want = {
        S: 3.1,
        "a": 1.6,
        "b": 2.1,
        "c": 13.6,
        S + "a": 32.0,
        "ab": 1.5,
        "bc": 31.5,
        S + "ab": 10.0,
    }
    got = {p: math.exp(s) for p, s in new.pieces if p != "<unk>"}
    assert set(got) == set(want)
    for p, v in want.items():
        assert_close(got[p], v / 95.4, rtol=1e-12, atol=1e-15, msg=p)
    assert new.pieces[new.unk_id][0] == "<unk>"


def test_em_step_matches_enumeration():
    # WHY: forward-backward must equal brute-force enumeration of every
    #      segmentation, with each word weighted by how often it occurs.
    # KIND: differential
    # CATCHES: s07, s08, s14
    # CHAPTER: L1.4 section 2, Principles
    tok = hand()
    texts = ["abc abc ab", "bc a", "abc"]
    words = Counter(w for t in texts for w in metaspace(t))
    want = brute_force_em(tok, words)
    got = {p: math.exp(s) for p, s in tok.em_step(texts).pieces if p != "<unk>"}
    assert set(got) == {p for p, v in want.items() if v > 0}
    for p in got:
        assert_close(got[p], want[p], rtol=1e-12, atol=1e-15, msg=p)


def test_em_step_never_decreases_likelihood():
    # WHY: EM's guarantee (Dempster, Laird, Rubin 1977): each step's
    #      log-likelihood is at least the last one's, up to rounding.
    # KIND: property
    # CATCHES: s19, s21
    # CHAPTER: L1.4 section 2, Principles
    lines = (FX / "L1.2" / "train.txt").read_text(encoding="utf-8").splitlines()[:12]
    words = sorted({w for line in lines for w in metaspace(line)})
    pieces = sorted(
        {
            w[i:j]
            for w in words
            for i in range(len(w))
            for j in range(i + 1, min(len(w), i + 4) + 1)
        }
    )
    tok = UnigramTokenizer(
        [("<unk>", 0.0)] + [(p, -math.log(len(pieces))) for p in pieces]
    )
    ll = tok.log_likelihood(lines)
    for _ in range(4):
        tok = tok.em_step(lines)
        nxt = tok.log_likelihood(lines)
        assert nxt >= ll - 1e-9 * abs(ll), (nxt, ll)
        assert nxt <= 0.0
        ll = nxt


# --- Viterbi details -----------------------------------------------------------------


def test_viterbi_tie_rule():
    # WHY: "▁aaa" with ▁ = a = -1 and aa = -2 has three paths of score -4;
    #      the rule (sentencepiece's and Hugging Face's backpointer) keeps the
    #      first candidate that reaches the best score, so the last piece is
    #      the one that starts earliest: [▁, a, aa].
    # KIND: boundary
    # CATCHES: s01
    # CHAPTER: L1.4 section 5, Pitfalls, item 2
    tok = UnigramTokenizer([("<unk>", 0.0), (S, -1.0), ("a", -1.0), ("aa", -2.0)])
    assert toks(tok, tok.encode("aaa")) == (S, "a", "aa")


def test_unknown_characters_fuse():
    # WHY: a character no piece covers is <unk>, and a run of them is ONE
    #      <unk>: "▁a日本b" is ▁a <unk> b.
    # KIND: boundary
    # CATCHES: s02
    # CHAPTER: L1.4 section 2, Principles
    tok = hand()
    assert tok.encode("a日本b") == [5, 0, 3]
    assert tok.decode(tok.encode("a日本b")) == "a<unk>b"


def test_unknown_character_penalty():
    # WHY: an unknown character scores min_score - 10. With no penalty
    #      ▁ <unk> ab (-14) would beat ▁ za b (-18); with it, -24 loses.
    # KIND: boundary
    # CATCHES: s03
    # CHAPTER: L1.4 section 2, Principles
    tok = UnigramTokenizer(
        [
            ("<unk>", 0.0),
            (S, -1.0),
            ("a", -3.0),
            ("b", -5.0),
            ("ab", -1.0),
            ("za", -12.0),
        ]
    )
    assert tok.min_score == -12.0
    assert toks(tok, tok.encode("zab")) == (S, "za", "b")


def test_hand_example_metaspace():
    # WHY: spaces become ▁, one ▁ is prepended unless the text starts with
    #      one, and each word starts at a ▁: a double space leaves a lone ▁.
    # KIND: unit
    # CATCHES: s09, s10
    # CHAPTER: L1.4 section 3, Worked example by hand
    assert metaspace("a  b") == [S + "a", S, S + "b"]
    assert metaspace(" x") == [S + "x"]
    assert metaspace(S + "x") == [S + "x"]
    assert metaspace("") == []
    assert metaspace("ab\ncd") == [S + "ab\ncd"]


# --- subword sampling -------------------------------------------------------------------


def sample_counts(alpha: float, n: int, stream: int) -> Counter:
    tok = hand()
    rng = PCG32(seed(), stream)
    return Counter(toks(tok, tok.sample_encode("abc", alpha, rng)) for _ in range(n))


def test_sample_matches_posterior():
    # WHY: sample_encode draws a segmentation with probability P(s)^alpha / Z;
    #      6000 draws at alpha = 1 and alpha = 0.5 fit the exact enumeration
    #      of section 3 (chi-square, p > 1e-3).
    # KIND: statistical
    # CATCHES: s04, s05
    # CHAPTER: L1.4 section 2, Principles
    for alpha, stream in ((1.0, 41), (0.5, 42)):
        w = {s: p**alpha for s, p in SEGS.items()}
        z = sum(w.values())
        counts = sample_counts(alpha, 6000, stream)
        assert set(counts) <= set(SEGS)
        assert chi_square(counts, {s: v / z for s, v in w.items()}) < CHI2_DF5_P001, (
            alpha,
            counts,
        )


def test_sample_alpha_zero_is_uniform():
    # WHY: alpha = 0 flattens P(s)^alpha to 1: every segmentation, even the
    #      least likely ▁ a b c, is equally likely.
    # KIND: statistical
    # CATCHES: s04
    # CHAPTER: L1.4 section 2, Principles
    counts = sample_counts(0.0, 6000, 43)
    assert chi_square(counts, {s: 1 / 6 for s in SEGS}) < CHI2_DF5_P001, counts


def test_sample_draws_one_uniform_per_piece():
    # WHY: the draw order is part of the contract (Python and Rust replay
    #      the same stream): one rng.uniform() per chosen piece, so after a
    #      sample the generator has advanced exactly len(ids) uniforms.
    # KIND: unit
    # CATCHES: s21
    # CHAPTER: L1.4 section 4, The interface
    tok = hand()
    rng, ref = PCG32(seed(), 44), PCG32(seed(), 44)
    for _ in range(50):
        ids = tok.sample_encode("abc", 1.0, rng)
        for _ in ids:
            ref.uniform()
        assert (rng.state, rng.inc) == (ref.state, ref.inc)
    a, b = PCG32(seed(), 45), PCG32(seed(), 45)
    assert [tok.sample_encode("abc ab", 1.0, a) for _ in range(20)] == [
        tok.sample_encode("abc ab", 1.0, b) for _ in range(20)
    ]


def test_sample_rejects_negative_alpha():
    # WHY: P(s)^alpha with alpha < 0 favors the LEAST likely segmentations.
    # KIND: boundary
    # CATCHES: s20
    # CHAPTER: L1.4 section 4, The interface
    with pytest.raises(ValueError):
        hand().sample_encode("abc", -0.5, PCG32(seed(), 46))


# --- compatibility --------------------------------------------------------------------------


def test_hf_unigram_ids_match_oracle():
    # WHY: a tokenizer.json written by the Hugging Face UnigramTrainer loads
    #      unchanged and encodes 300 strings (unknown characters, runs of
    #      spaces, "<unk>" typed in the text) to exactly its ids.
    # KIND: golden
    # CATCHES: s03, s09, s17
    # CHAPTER: L1.4 section 4, The interface
    tok = UnigramTokenizer.from_hf_json(
        str(FX / "L1.4" / "hf-unigram" / "tokenizer.json")
    )
    for c in cases("hf-unigram"):
        assert tok.encode(c["text"]) == c["ids"], repr(c["text"])


def test_spm_unigram_ids_match_sentencepiece():
    # WHY: sentencepiece itself is the second oracle: the same Viterbi on a
    #      model it trained, 236 strings.
    # KIND: golden
    # CATCHES: s03
    # CHAPTER: L1.4 section 4, The interface
    tok = UnigramTokenizer.from_hf_json(
        str(FX / "L1.4" / "spm-unigram" / "tokenizer.json")
    )
    rows = cases("spm-unigram")
    assert len(rows) >= 200
    for c in rows:
        assert tok.encode(c["text"]) == c["ids"], repr(c["text"])


def test_hf_unigram_decode_matches_oracle():
    # WHY: decode joins pieces, turns ▁ back into spaces, and drops the
    #      prepended space of the first piece only.
    # KIND: golden
    # CATCHES: s16
    # CHAPTER: L1.4 section 4, The interface
    tok = UnigramTokenizer.from_hf_json(
        str(FX / "L1.4" / "hf-unigram" / "tokenizer.json")
    )
    for c in cases("hf-unigram"):
        assert tok.decode(c["ids"]) == c["decoded"], repr(c["text"])
        assert tok.decode(c["ids"], skip_special=True) == c["decoded_skip"], repr(
            c["text"]
        )


# --- training and files ---------------------------------------------------------------------


def test_train_properties():
    # WHY: train gives exactly vocab_size ids on a corpus that supports them:
    #      <unk> first, every training character kept (so training text never
    #      hits <unk>), pieces ordered by score; and it is deterministic.
    # KIND: property
    # CATCHES: s11, s12
    # CHAPTER: L1.4 section 2, Principles
    lines = (FX / "L1.2" / "train.txt").read_text(encoding="utf-8").splitlines()
    tok = UnigramTokenizer.train(lines, vocab_size=200)
    assert tok.vocab_size == 200
    assert tok.pieces[0] == ("<unk>", 0.0) and tok.unk_id == 0
    scores = [s for _, s in tok.pieces[1:]]
    assert scores == sorted(scores, reverse=True)
    chars = {ch for line in lines for w in metaspace(line) for ch in w}
    assert chars <= {p for p, _ in tok.pieces}
    for line in lines:
        ids = tok.encode(line)
        assert 0 not in ids
        if line and not line.startswith(" "):
            assert tok.decode(ids) == line
    assert UnigramTokenizer.train(lines, vocab_size=200).pieces == tok.pieces


def test_save_load_roundtrip(tmp_path):
    # WHY: save writes a Unigram tokenizer.json (Metaspace, prepend always)
    #      that load reads back to the same pieces, scores, and ids.
    # KIND: property
    # CATCHES: s23
    # CHAPTER: L1.4 section 4, The interface
    tok = hand()
    tok.save(str(tmp_path))
    doc = json.loads((tmp_path / "tokenizer.json").read_text(encoding="utf-8"))
    assert doc["model"]["type"] == "Unigram" and doc["model"]["unk_id"] == 0
    back = UnigramTokenizer.load(str(tmp_path))
    assert back.pieces == tok.pieces
    assert back.encode("abc ab日") == tok.encode("abc ab日")


def test_from_hf_json_rejects_outside_subset(tmp_path):
    # WHY: byte fallback, a normalizer, or another Metaspace scheme would
    #      change the ids; the loader refuses them instead of ignoring them.
    # KIND: boundary
    # CATCHES: s18
    # CHAPTER: L1.4 section 4, The interface
    good = json.loads(
        (FX / "L1.4" / "hf-unigram" / "tokenizer.json").read_text(encoding="utf-8")
    )
    for key, value in (
        ("model.byte_fallback", True),
        ("normalizer", {"type": "NFKC"}),
        ("pre_tokenizer.prepend_scheme", "first"),
        ("model.type", "BPE"),
    ):
        doc = json.loads(json.dumps(good))
        node, path = doc, key.split(".")
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value
        (tmp_path / "t.json").write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(ValueError):
            UnigramTokenizer.from_hf_json(str(tmp_path / "t.json"))


def test_constructor_rejects_bad_vocab():
    # WHY: duplicate pieces make ids ambiguous; unk_id must be a piece.
    # KIND: boundary
    # CATCHES: m01, m02
    # CHAPTER: L1.4 section 4, The interface
    with pytest.raises(ValueError):
        UnigramTokenizer([("<unk>", 0.0), ("a", -1.0), ("a", -2.0)])
    with pytest.raises(ValueError):
        UnigramTokenizer([("<unk>", 0.0), ("a", -1.0)], unk_id=2)
    with pytest.raises(ValueError):
        UnigramTokenizer([("<unk>", 0.0), ("a", -1.0)], specials=["<s>"])
