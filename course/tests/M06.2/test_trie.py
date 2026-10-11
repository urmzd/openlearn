"""Course tests for M06.2: the character trie (tinyllm/tok/trie.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M06.2), and the chapter section it comes from.

The chapter's worked example (section 3) is the trie of the five keys
un=1, unbe=2, unbeliev=3, able=4, a=5, matched against "unbelievable".
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from _lib.pcg32 import PCG32
from tinyllm.tok.trie import Trie

GOLDEN = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "M06.2" / "segmentation.json"
HAND = {"un": 1, "unbe": 2, "unbeliev": 3, "able": 4, "a": 5}


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def build(keys: dict[str, int]) -> Trie:
    t = Trie()
    for k, v in keys.items():
        t.insert(k, v)
    return t


def brute_prefixes(keys: dict[str, int], s: str, start: int) -> list[tuple[int, int]]:
    """Every key that is a prefix of s[start:], shortest first, by scanning all keys."""
    return sorted((len(k), v) for k, v in keys.items() if s.startswith(k, start))


def random_word(rng: PCG32, alphabet: str, max_len: int) -> str:
    return "".join(
        alphabet[rng.below(len(alphabet))] for _ in range(1 + rng.below(max_len))
    )


def greedy_segment(t: Trie, s: str) -> list[list[int]]:
    """WordPiece-style segmentation by repeated longest_prefix (L1.3).

    The match length must fit in the rest of the text. Checking it here keeps
    a wrong length (negative, or counted from 0 instead of from i) from
    moving i backwards and looping forever."""
    out, i = [], 0
    while i < len(s):
        n, v = t.longest_prefix(s, i)
        assert 0 <= n <= len(s) - i, f"longest_prefix(s, {i}) gave length {n}"
        if n == 0:
            out.append([1, -1])
            i += 1
        else:
            out.append([n, v])
            i += n
    return out


# --- the worked example -------------------------------------------------------


def test_hand_example_longest_prefix():
    # WHY: the chapter's worked example. From position 0 the longest key is
    #      "unbeliev" (8 characters, value 3); from position 8 it is "able".
    #      That is the greedy segmentation WordPiece makes: unbeliev + able.
    #      13 nodes: the root plus the 12 distinct non-empty prefixes.
    # KIND: unit
    # CATCHES: s01, s02, m04
    # CHAPTER: M06.2 section 3, Worked example by hand
    t = build(HAND)
    assert len(t) == 5
    assert t.node_count() == 13
    assert t.longest_prefix("unbelievable") == (8, 3)
    assert t.longest_prefix("unbelievable", 8) == (4, 4)
    assert greedy_segment(t, "unbelievable") == [[8, 3], [4, 4]]


def test_hand_example_prefixes():
    # WHY: the Unigram lattice (L1.4) needs every key that starts at a
    #      position, not only the longest: "un", "unbe", "unbeliev", shortest
    #      first, which is the order the walk meets them.
    # KIND: unit
    # CATCHES: s09, s01
    # CHAPTER: M06.2 section 3, Worked example by hand
    t = build(HAND)
    assert list(t.prefixes("unbelievable")) == [(2, 1), (4, 2), (8, 3)]
    assert list(t.prefixes("unbelievable", 8)) == [(1, 5), (4, 4)]
    assert list(t.prefixes("xyz")) == []


# --- longest_prefix -------------------------------------------------------------


def test_deepest_key_not_deepest_node():
    # WHY: "unbelie" walks 7 edges down the tree, but the node it stops at is
    #      not the end of any key (only "unbeliev" goes through it). The
    #      answer is the deepest node passed that ends a key: "unbe".
    # KIND: boundary
    # CATCHES: s01
    # CHAPTER: M06.2 section 5, Pitfalls, item 1
    t = build(HAND)
    assert t.longest_prefix("unbelie") == (4, 2)
    assert t.longest_prefix("unb") == (2, 1)
    assert t.longest_prefix("ab") == (1, 5)


def test_longest_not_first_match():
    # WHY: stopping at the first key you pass returns the shortest match.
    #      WordPiece would then split "unbelievable" into "un" + ... and the
    #      ids would differ from BERT's.
    # KIND: boundary
    # CATCHES: s02
    # CHAPTER: M06.2 section 5, Pitfalls, item 2
    t = build({"a": 1, "ab": 2, "abc": 3})
    assert t.longest_prefix("abcd") == (3, 3)
    assert t.longest_prefix("abx") == (2, 2)


def test_no_match_is_zero_none():
    # WHY: (0, None) is the only "nothing matched" answer, so callers test
    #      the length. An empty key is not allowed, so length 0 never means
    #      "the empty key matched".
    # KIND: boundary
    # CATCHES: s10
    # CHAPTER: M06.2 section 4, The interface
    t = build(HAND)
    assert t.longest_prefix("xyz") == (0, None)
    assert t.longest_prefix("") == (0, None)
    assert Trie().longest_prefix("abc") == (0, None)


def test_start_offset_and_bounds():
    # WHY: tokenizers match from every position of one long string, so the
    #      length returned is counted from start, start == len(s) is a valid
    #      empty suffix, and anything outside [0, len(s)] is a caller bug.
    # KIND: boundary
    # CATCHES: s03, s11, m01, m04
    # CHAPTER: M06.2 section 5, Pitfalls, item 3
    t = build(HAND)
    s = "xxunbe"
    assert t.longest_prefix(s, 2) == (4, 2)
    assert list(t.prefixes(s, 2)) == [(2, 1), (4, 2)]
    assert t.longest_prefix(s, len(s)) == (0, None)
    assert list(t.prefixes(s, len(s))) == []
    assert t.longest_prefix("unbe", 0) == (4, 2)
    for bad in (-1, len(s) + 1):
        with pytest.raises(ValueError):
            t.longest_prefix(s, bad)
        with pytest.raises(ValueError):
            list(t.prefixes(s, bad))


def test_value_zero_is_a_value():
    # WHY: token id 0 is a real token. Testing `if value:` instead of
    #      `if value is not None:` makes id 0 invisible: get, `in`,
    #      longest_prefix, and items all lose it.
    # KIND: boundary
    # CATCHES: s06
    # CHAPTER: M06.2 section 5, Pitfalls, item 4
    t = build({"a": 0, "ab": 7})
    assert t.get("a") == 0
    assert "a" in t
    assert t.longest_prefix("ac") == (1, 0)
    assert list(t.prefixes("abc")) == [(1, 0), (2, 7)]
    assert list(t.items()) == [("a", 0), ("ab", 7)]


def test_code_points_are_not_normalized():
    # WHY: "é" (U+00E9) and "e" + U+0301 render the same and are different
    #      strings. The trie compares code points; normalization is the
    #      tokenizer's job (L1.1, NFC/NFKC), done before matching.
    # KIND: unit
    # CATCHES: s01
    # CHAPTER: M06.2 section 2, Principles (edges are characters)
    t = build({"café": 1, "café": 2, "\U0001f642": 3})
    assert t.longest_prefix("café!") == (4, 1)
    assert t.longest_prefix("café!") == (5, 2)
    assert t.longest_prefix("\U0001f642\U0001f642") == (1, 3)


def test_matches_brute_force():
    # WHY: for random keys and texts, longest_prefix and prefixes agree with
    #      scanning every key with str.startswith, at every start position.
    #      A tiny alphabet makes keys share long prefixes, which is where
    #      trie bugs live.
    # KIND: differential
    # CATCHES: s01, s02, s03, s09, m04
    # CHAPTER: M06.2 section 2, Principles (longest-prefix match)
    rng = PCG32(seed=seed())
    for _ in range(20):
        keys = {random_word(rng, "ab", 6): i for i in range(15)}
        t = build(keys)
        for _ in range(5):
            s = random_word(rng, "abc", 10)
            for start in range(len(s) + 1):
                want = brute_prefixes(keys, s, start)
                assert list(t.prefixes(s, start)) == want
                assert t.longest_prefix(s, start) == (want[-1] if want else (0, None))


# --- insert, get, len, node_count -----------------------------------------------


def test_insert_a_prefix_of_an_existing_key():
    # WHY: when "abc" is inserted first, the node for "ab" already exists as
    #      a pass-through. Inserting "ab" must mark it as a key, not skip it
    #      because no node had to be created.
    # KIND: boundary
    # CATCHES: s04
    # CHAPTER: M06.2 section 5, Pitfalls, item 5
    t = Trie()
    t.insert("abc", 1)
    assert t.get("ab") is None and "ab" not in t
    t.insert("ab", 2)
    assert t.get("ab") == 2 and "ab" in t
    assert len(t) == 2
    assert t.node_count() == 4
    assert t.longest_prefix("abd") == (2, 2)


def test_reinsert_replaces_the_value():
    # WHY: a vocabulary loaded twice, or an added token that overrides a
    #      merge, must update the value without counting a new key or adding
    #      nodes.
    # KIND: unit
    # CATCHES: s05, m02
    # CHAPTER: M06.2 section 4, The interface
    t = build({"ab": 1, "abc": 2})
    nodes = t.node_count()
    t.insert("ab", 9)
    assert len(t) == 2
    assert t.node_count() == nodes
    assert t.get("ab") == 9


def test_get_and_contains_need_a_key_not_a_prefix():
    # WHY: "unbel" is a path in the tree (on the way to "unbeliev") but not a
    #      key. Treating "a node exists" as "a key exists" makes WordPiece
    #      emit pieces that are not in the vocabulary.
    # KIND: boundary
    # CATCHES: s12, s06
    # CHAPTER: M06.2 section 5, Pitfalls, item 6
    t = build(HAND)
    assert t.get("unbel") is None and "unbel" not in t
    assert t.get("zzz") is None and "zzz" not in t
    assert t.get("unbe") == 2 and "unbe" in t
    assert "" not in t


def test_empty_key_rejected():
    # WHY: the root spells the empty string. Storing a value there would make
    #      every text match a zero-length key, and a greedy segmenter that
    #      advances by the match length would loop forever.
    # KIND: boundary
    # CATCHES: s10
    # CHAPTER: M06.2 section 4, The interface
    t = Trie()
    with pytest.raises(ValueError):
        t.insert("", 1)
    assert len(t) == 0 and t.node_count() == 1


def test_node_count_is_distinct_prefixes():
    # WHY: a trie stores each shared prefix once. Its size is 1 + the number
    #      of distinct non-empty prefixes of the keys, which is why a 50 000
    #      token vocabulary needs far fewer nodes than characters.
    # KIND: property
    # CATCHES: m02, m03
    # CHAPTER: M06.2 section 2, Principles (counting nodes)
    rng = PCG32(seed=seed())
    for _ in range(10):
        keys = {random_word(rng, "abcd", 8): i for i in range(30)}
        t = build(keys)
        prefixes = {k[:j] for k in keys for j in range(1, len(k) + 1)}
        assert t.node_count() == 1 + len(prefixes)
        assert len(t) == len(keys)


# --- items --------------------------------------------------------------------------


def test_items_in_code_point_order():
    # WHY: a preorder walk that visits children in increasing character order
    #      lists keys exactly as sorted() does, because a key sorts before
    #      its extensions. L1.2 relies on it to write a vocabulary
    #      deterministically.
    # KIND: property
    # CATCHES: s08
    # CHAPTER: M06.2 section 2, Principles (preorder is sorted order)
    rng = PCG32(seed=seed())
    alphabet = "baézĀ "
    for _ in range(10):
        keys = {random_word(rng, alphabet, 6): i for i in range(25)}
        t = build(keys)
        assert list(t.items()) == sorted(keys.items())


def test_items_survives_a_very_deep_key():
    # WHY: a recursive walk uses one Python stack frame per character and
    #      dies at about 1000. A tokenizer that indexes whole documents or a
    #      long run of one byte must not, so every walk is a loop (M06.1
    #      made the same choice for toposort).
    # KIND: boundary
    # CATCHES: s07
    # CHAPTER: M06.2 section 5, Pitfalls, item 7
    deep = "a" * 100_000
    t = build({deep: 1, "a": 2})
    assert list(t.items()) == [("a", 2), (deep, 1)]
    assert t.longest_prefix(deep + "b") == (100_000, 1)
    assert t.node_count() == 100_001


# --- golden -------------------------------------------------------------------------


def test_golden_greedy_segmentation():
    # WHY: WordPiece (L1.3) is repeated longest_prefix from the left. The
    #      fixture's segmentations come from a brute-force oracle over the
    #      whole vocabulary (course/oracle/M06.2), including keys that are
    #      prefixes of other keys, combining accents, and emoji.
    # KIND: golden
    # CATCHES: s01, s02, s03
    # CHAPTER: M06.2 section 4, What the tests check
    doc = json.loads(GOLDEN.read_text())
    t = build({w: i for i, w in enumerate(doc["vocab"])})
    for case in doc["cases"]:
        assert greedy_segment(t, case["s"]) == case["greedy"], case["s"]


def test_golden_lattice():
    # WHY: the Unigram lattice (L1.4) has one edge per vocabulary match at
    #      every position; a missing or extra edge changes the Viterbi path.
    # KIND: golden
    # CATCHES: s01, s09
    # CHAPTER: M06.2 section 4, What the tests check
    doc = json.loads(GOLDEN.read_text())
    t = build({w: i for i, w in enumerate(doc["vocab"])})
    for case in doc["cases"]:
        s = case["s"]
        got = [[list(e) for e in t.prefixes(s, i)] for i in range(len(s) + 1)]
        assert got == case["lattice"], s
