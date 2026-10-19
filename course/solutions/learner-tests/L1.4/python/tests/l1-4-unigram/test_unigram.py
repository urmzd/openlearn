"""Model test suite for L1.4: the bodies of the test names and docstrings
listed in the chapter, section 4. They import only names in
contracts/py/tinyllm/tok/unigram.pyi, and each one fails on a solution with
the bug its name describes."""

import itertools
import math
from collections import Counter

import pytest
from tinyllm.tok.unigram import UnigramTokenizer, metaspace

S = "▁"
HAND = {
    S: 0.1,
    "a": 0.1,
    "b": 0.1,
    "c": 0.1,
    S + "a": 0.2,
    "ab": 0.15,
    "bc": 0.15,
    S + "ab": 0.1,
}
Z = 0.0451  # sum over the six segmentations of "▁abc"


class Fixed:
    """A stand-in generator: uniform() replays a fixed list, cycling."""

    def __init__(self, us):
        self.us, self.k = list(us), 0

    def uniform(self):
        u = self.us[self.k % len(self.us)]
        self.k += 1
        return u


def hand():
    return UnigramTokenizer(
        [("<unk>", 0.0)] + [(p, math.log(q)) for p, q in HAND.items()]
    )


def pieces(tok, ids):
    return [tok.id_to_token(i) for i in ids]


def test_metaspace_marks_word_starts():
    """ "a  b" is ["▁a", "▁", "▁b"]; a leading ▁ is not doubled; "" is []."""
    assert metaspace("a  b") == [S + "a", S, S + "b"]
    assert metaspace(S + "x") == [S + "x"]
    assert metaspace("") == []


def test_viterbi_picks_the_most_probable_segmentation():
    """With the section 3 vocab, "abc" is ▁a + bc (0.03), not ▁ab + c (0.01)."""
    tok = hand()
    assert pieces(tok, tok.encode("abc")) == [S + "a", "bc"]


def test_viterbi_tie_goes_to_earliest_last_piece():
    """▁ = a = -1, aa = -2: "aaa" is [▁, a, aa]."""
    tok = UnigramTokenizer([("<unk>", 0.0), (S, -1.0), ("a", -1.0), ("aa", -2.0)])
    assert pieces(tok, tok.encode("aaa")) == [S, "a", "aa"]


def test_unknown_run_is_one_unk_with_penalty():
    """ "a日本b" is ▁a <unk> b; an unknown scores min_score - 10."""
    tok = hand()
    assert pieces(tok, tok.encode("a日本b")) == [S + "a", "<unk>", "b"]
    pen = UnigramTokenizer(
        [
            ("<unk>", 0.0),
            (S, -1.0),
            ("a", -3.0),
            ("b", -5.0),
            ("ab", -1.0),
            ("za", -12.0),
        ]
    )
    assert pieces(pen, pen.encode("zab")) == [S, "za", "b"]


def test_log_likelihood_sums_all_segmentations():
    """log_likelihood(["abc"]) is ln 0.0451, not the Viterbi path's ln 0.03."""
    assert abs(hand().log_likelihood(["abc"]) - math.log(Z)) < 1e-12


def test_em_step_expected_counts():
    """One EM step on "abc": p(▁a) becomes 32.0 / 95.4 and p(c) 13.6 / 95.4."""
    new = hand().em_step(["abc"])
    p = {t: math.exp(s) for t, s in new.pieces}
    assert abs(p[S + "a"] - 32.0 / 95.4) < 1e-12
    assert abs(p["c"] - 13.6 / 95.4) < 1e-12


def test_em_step_weights_words_by_count():
    """em_step(["abc abc ab"]) equals the enumeration with "▁abc" counted twice."""
    tok = hand()
    words = Counter(metaspace("abc abc ab"))
    score = {t: s for t, s in tok.pieces[1:]}
    counts = Counter()
    for w, c in words.items():
        segs = []
        for cuts in itertools.product([0, 1], repeat=len(w) - 1):
            ps, start = [], 0
            for k, cut in enumerate(cuts, 1):
                if cut:
                    ps.append(w[start:k])
                    start = k
            ps.append(w[start:])
            if all(x in score for x in ps):
                segs.append((ps, math.exp(sum(score[x] for x in ps))))
        z = sum(pr for _, pr in segs)
        for ps, pr in segs:
            for x in ps:
                counts[x] += c * pr / z
    total = sum(counts.values())
    got = {
        t: math.exp(s) for t, s in tok.em_step(["abc abc ab"]).pieces if t != "<unk>"
    }
    for t, v in got.items():
        assert abs(v - counts[t] / total) < 1e-12


def test_sample_with_fixed_uniforms():
    """Drawing backward, u = 0 picks the candidate that starts earliest (▁a bc); u near 1 the latest (▁ a b c)."""
    tok = hand()
    # Pieces ending at "c", by start ascending: bc (start 2), c (start 3).
    # u = 0 picks bc, then, of ▁a and a ending at "a", ▁a.
    assert pieces(tok, tok.sample_encode("abc", 1.0, Fixed([0.0]))) == [S + "a", "bc"]
    # u just below 1 picks the last candidate each time: c, b, a, ▁.
    assert pieces(tok, tok.sample_encode("abc", 1.0, Fixed([0.999999]))) == [
        S,
        "a",
        "b",
        "c",
    ]


def test_sample_uses_one_uniform_per_piece():
    """After a sample of k pieces the generator has been asked exactly k times."""
    tok = hand()
    rng = Fixed([0.3, 0.7, 0.1, 0.9])
    ids = tok.sample_encode("abc", 1.0, rng)
    assert rng.k == len(ids)


def test_negative_alpha_raises():
    """sample_encode with alpha < 0 is ValueError."""
    with pytest.raises(ValueError):
        hand().sample_encode("abc", -1.0, Fixed([0.5]))


def test_decode_drops_only_the_first_space():
    """decode(encode("ab c")) is "ab c": the first piece's prepended ▁ goes, the others become spaces."""
    tok = hand()
    assert tok.decode(tok.encode("ab c")) == "ab c"


def test_train_keeps_characters_and_size():
    """train on a few lines: vocab_size ids, <unk> first, every character a piece."""
    lines = ["the cat sat on the mat", "the dog sat on the log", "a cat and a dog"] * 3
    tok = UnigramTokenizer.train(lines, vocab_size=40)
    assert tok.vocab_size <= 40 and tok.id_to_token(0) == "<unk>"
    chars = {ch for line in lines for w in metaspace(line) for ch in w}
    assert all(tok.token_to_id(ch) is not None for ch in chars)
    scores = [s for _, s in tok.pieces[1:]]
    assert scores == sorted(scores, reverse=True)
