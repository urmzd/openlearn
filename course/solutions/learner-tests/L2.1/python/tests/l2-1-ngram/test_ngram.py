"""Model test suite for L2.1: what each test protects. Written test-first;
the chapter's section 3 numbers are the first test.
They import only the contract."""

import math

import numpy as np
import pytest
from tinyllm.lm.ngram import FALLBACK_DISCOUNTS, NGramLM

A, B, C = 0, 1, 2
HAND = [[A, B, C], [A, B, A, B], [C, A, B, C]]
TEXT = (
    b"the cat sat on the mat. the dog sat on the log. a cat and a dog met on the mat.\n"
    b"then the dog ran to the log and the cat ran to the mat. the end of the day came.\n"
    b"on the next day the cat sat by the dog and the dog sat by the cat on the log.\n"
    b"a bird sang on the log; the cat and the dog and the bird sat in the sun all day.\n"
) * 3


def lines():
    return [list(x) for x in TEXT.split(b"\n") if x]


def model(seqs, n, d="modified", V=None):
    lm = NGramLM(n, d, V)
    lm.fit(seqs)
    return lm


def test_hand_example_bigram():
    """Continuation counts a 3, b 1, c 2: p1 = (1/2, 1/6, 1/3); after b: (1/3, 1/18, 11/18)."""
    lm = model(HAND, 2, 0.5, 3)
    np.testing.assert_allclose(
        [lm.prob([B], w) for w in range(3)], [1 / 3, 1 / 18, 11 / 18], atol=1e-15
    )
    np.testing.assert_allclose(
        [lm.prob([C], w) for w in range(3)], [3 / 4, 1 / 12, 1 / 6], atol=1e-15
    )
    np.testing.assert_allclose(
        [lm.prob([], w) for w in range(3)], [2 / 3, 1 / 18, 5 / 18], atol=1e-15
    )


def test_unigram_discounts_from_counts_of_counts():
    """Counts 1,1,1,1 / 2,2,2 / 3,3 / 4: Y = 0.4, D = (0.4, 1.2, 2.2)."""
    seq = [0, 1, 2, 3] + [4, 5, 6] * 2 + [7, 8] * 3 + [9] * 4
    lm = model([seq], 1)
    np.testing.assert_allclose(lm.discounts(1), [0.4, 1.2, 2.2], atol=1e-15)


def test_fallback_on_tiny_corpus():
    lm = model(HAND, 3, "modified", 3)
    for k in (1, 2, 3):
        assert lm.discounts(k) == FALLBACK_DISCOUNTS
    bad = [0] + [1] * 2 + [2] * 3 + [t for t in range(3, 13) for _ in range(4)]
    assert model([bad], 1).discounts(1) == FALLBACK_DISCOUNTS


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_sums_to_one_with_modified_discounts(n):
    lm = model(lines(), n, "modified", 256)
    for ctx in ([], list(b"th"), list(b"the c"), list(b"zzz"), list(b"on the l")):
        assert abs(math.fsum(lm.prob(ctx, w) for w in range(256)) - 1.0) < 1e-12


def test_start_of_sequence_differs_from_middle():
    lm = model(lines(), 3, "modified", 256)
    assert lm.prob(list(b"t"), ord("h")) != lm.prob(list(b" t"), ord("h"))
    # longer contexts keep their last n - 1 tokens
    for w in (ord("h"), ord("e"), ord(" ")):
        assert lm.prob(list(b"the cat sat t"), w) == lm.prob(list(b" t"), w)


def test_unseen_history_backs_off_to_lower_order():
    lm = model(HAND, 2, 0.5, 4)
    np.testing.assert_allclose(
        [lm.prob([3], w) for w in range(4)],
        [23 / 48, 7 / 48, 15 / 48, 3 / 48],
        atol=1e-15,
    )


def test_logprobs_match_prob():
    lm = model(lines(), 3, "modified", 256)
    for ctx in ([], list(b"the "), list(b"q")):
        lp = lm.logprobs(ctx)
        np.testing.assert_allclose(
            np.exp(lp), [lm.prob(ctx, w) for w in range(256)], rtol=1e-12
        )


def test_nll_scores_every_token_from_start():
    lm = model(HAND, 2, 0.5, 3)
    nll = lm.nll([A, B])
    np.testing.assert_allclose(
        nll, [-math.log(2 / 3), -math.log(lm.prob([A], B))], atol=1e-15
    )
    assert lm.perplexity([A, B]) == pytest.approx(math.exp(nll.mean()), rel=1e-15)


def test_save_load_roundtrip(tmp_path):
    lm = model(lines(), 3, "modified", 256)
    p = tmp_path / "m.safetensors"
    lm.save(str(p))
    back = NGramLM.load(str(p))
    for ctx in ([], list(b"th"), list(b"the c")):
        np.testing.assert_array_equal(back.logprobs(ctx), lm.logprobs(ctx))
    q = tmp_path / "again.safetensors"
    back.save(str(q))
    assert q.read_bytes() == p.read_bytes()


def test_refit_and_vocab_inference():
    lm = NGramLM(2, 0.5)
    lm.fit([[5, 5, 5]])
    lm.fit(HAND)
    assert lm.vocab_size == 3
    np.testing.assert_array_equal(lm.logprobs([A]), model(HAND, 2, 0.5).logprobs([A]))
    with pytest.raises(ValueError):
        lm.prob([A], 3)
