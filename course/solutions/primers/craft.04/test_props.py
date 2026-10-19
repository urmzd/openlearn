"""Reference property tests for the craft.04 kata (textops.py).

What a learner's primers/craft.04/test_props.py looks like when it passes
the check: every law of the chapter's section 4 as a Hypothesis property,
plus a model (a slow, obviously correct BPE) that the fast one must agree
with. The check runs this file against your kata, the course's kata, and
ten planted faults.
"""

from __future__ import annotations

import re
import unicodedata

from hypothesis import given
from hypothesis import strategies as st

from textops import MiniBPE, normalize

# Text that reaches every rule: line endings, every kind of white space,
# combining marks, and ordinary letters.
messy = st.text(
    alphabet=st.sampled_from(list("ab \t\n\r 　 ́ée가")),
    max_size=30,
)
any_text = st.one_of(messy, st.text(max_size=30))

MERGES = [(97, 98), (256, 97), (98, 98), (97, 97), (257, 258), (0xC3, 0xA9)]
bpe = MiniBPE(MERGES)
bpe_text = st.text(alphabet=st.sampled_from(list("abéx")), max_size=24) | st.text(
    max_size=12
)


def model_encode(text: str) -> list[int]:
    """The definition, slowly: find the lowest rank, take its leftmost
    occurrence, merge, repeat."""
    ids = list(text.encode("utf-8"))
    rank = {pair: r for r, pair in reversed(list(enumerate(MERGES)))}
    while True:
        cands = [
            (rank[(a, b)], i)
            for i, (a, b) in enumerate(zip(ids, ids[1:]))
            if (a, b) in rank
        ]
        if not cands:
            return ids
        r, i = min(cands)
        ids[i : i + 2] = [256 + r]


# --- normalize ---------------------------------------------------------------


@given(any_text)
def test_normalize_is_idempotent(x):
    once = normalize(x)
    assert normalize(once) == once


@given(any_text)
def test_normalize_output_shape(x):
    out = normalize(x)
    assert out == out.strip()
    assert "\r" not in out
    assert "\n\n\n" not in out
    assert not re.search(r"[^\S\n]{2,}|[^\S\n ]", out), repr(out)
    assert all(line == line.strip(" ") for line in out.split("\n"))
    assert unicodedata.is_normalized("NFC", out)


@given(any_text)
def test_carriage_returns_are_line_ends(x):
    # "\r\n" and a lone "\r" both end a line, exactly like "\n"
    assert normalize(x.replace("\r\n", "\n").replace("\r", "\n")) == normalize(x)


@given(
    st.text(
        alphabet=st.characters(blacklist_categories=("Zs", "Zl", "Zp", "Cc")),
        max_size=20,
    )
)
def test_text_without_space_is_only_nfc(x):
    assert normalize(x) == unicodedata.normalize("NFC", x)


# --- MiniBPE -------------------------------------------------------------------


@given(st.text())
def test_roundtrip(x):
    assert bpe.decode(bpe.encode(x)) == x
    assert bpe.decode_bytes(bpe.encode(x)) == x.encode("utf-8")


@given(bpe_text)
def test_encode_agrees_with_the_model(x):
    assert bpe.encode(x) == model_encode(x)


@given(bpe_text)
def test_encode_reaches_a_fixpoint(x):
    ids = bpe.encode(x)
    assert all((a, b) not in set(MERGES) for a, b in zip(ids, ids[1:]))


@given(st.lists(st.integers(min_value=0, max_value=255 + len(MERGES)), max_size=20))
def test_decode_is_bytes_with_replacement(ids):
    assert bpe.decode(ids) == bpe.decode_bytes(ids).decode("utf-8", errors="replace")


def test_hand_example():
    assert bpe.encode("abab") == [256, 256]
    assert bpe.encode("aba") == [257]
    assert bpe.encode("aaa") == [259, 97]
    assert bpe.decode([257, 258]) == "ababb"
