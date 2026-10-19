"""Model test suite for data.04 (properties): the properties the chapter
lists in section 4, written with Hypothesis where an input space is worth
searching. They import only names in contracts/py/corpus/{minhash,stage}.pyi
and contracts/py/tinyllm/num/rng.pyi."""

import itertools
import json

import numpy as np
import pytest
from corpus.minhash import (
    LSH,
    MERSENNE31,
    UnionFind,
    clusters,
    contaminated,
    decontaminate,
    estimate,
    jaccard,
    minhash,
    near_dedup,
    perm_params,
    protected_ngrams,
    shingle_hashes,
    shingles,
    signatures,
)
from corpus.stage import Doc
from hypothesis import given
from hypothesis import strategies as st
from tinyllm.num.rng import PCG32, fnv1a64

P = MERSENNE31
word = st.text(alphabet="abcdefghij", min_size=1, max_size=6)


def doc(i, text, **meta):
    return Doc(id=i, source_id="s", text=text, meta=meta)


def line(lo, n):
    return " ".join(f"v{i}" for i in range(lo, lo + n))


def test_hand_example():
    """A and B share 4 of 6 3-shingles; LSH(16, 8) has threshold 0.7071 and
    P(candidate | 0.8) = 0.9470."""
    a = shingles("the cat sat on the mat today", 3)
    b = shingles("the cat sat on the mat again", 3)
    assert jaccard(a, b) == pytest.approx(4 / 6)
    lsh = LSH(16, 8)
    assert lsh.threshold == pytest.approx(0.70710678, abs=1e-8)
    assert lsh.probability(0.8) == pytest.approx(0.94704880, abs=1e-8)


def test_signature_follows_the_formula():
    """min over x of (a_i x + b_i) mod P, x = fnv1a64 mod P, (a, b) interleaved from PCG32(seed)."""
    rng = PCG32(3)
    a, b = [], []
    for _ in range(6):
        a.append(1 + rng.below(P - 1))
        b.append(rng.below(P))
    sh = ["x y", "y z", "café ok"]
    xs = [fnv1a64(s.encode()) % P for s in sh]
    assert [int(v) for v in shingle_hashes(sh)] == xs
    got = minhash(sh, 6, 3)
    assert [int(v) for v in got] == [
        min((a[i] * x + b[i]) % P for x in xs) for i in range(6)
    ]
    ga, gb = perm_params(6, 3)
    assert [int(v) for v in ga] == a and [int(v) for v in gb] == b


@given(st.lists(word, min_size=1, max_size=4), st.integers(5, 9))
def test_short_texts_are_one_shingle(ws, k):
    """Fewer than k words: exactly one shingle, all the words."""
    assert shingles(" ".join(ws), k) == {" ".join(ws)}


def test_wordless_texts_never_cluster():
    """ "", "!!!", "..." have no shingles and are not each other's duplicates."""
    assert shingles("!!!") == set()
    assert clusters({"a": "", "b": "!!!", "c": "..."}, num_perm=16, bands=4) == []


def test_estimate_is_unbiased_over_fresh_pairs():
    """Mean estimate over 30 pairs at J = 1/3 is within 3.3 standard errors."""
    texts = []
    n = 0
    for _ in range(30):
        shared = line(n, 100)
        texts += [shared + " " + line(n + 100, 100), shared + " " + line(n + 200, 100)]
        n += 300
    sig = signatures(texts, k=1)
    est = np.mean([estimate(sig[2 * i], sig[2 * i + 1]) for i in range(30)])
    assert abs(est - 1 / 3) < 3.3 * np.sqrt((1 / 3) * (2 / 3) / (128 * 30))


def test_bands_are_kept_apart():
    """Equal values in different bands, or in r - 1 rows of one band, make no candidate."""
    lsh = LSH(3, 2)
    lsh.insert("x", np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64))
    lsh.insert("y", np.array([9, 9, 1, 2, 7, 7], dtype=np.uint64))
    lsh.insert("z", np.array([1, 8, 6, 3, 5, 9], dtype=np.uint64))
    assert lsh.candidates() == []
    lsh.insert("w", np.array([0, 0, 3, 4, 0, 0], dtype=np.uint64))
    assert lsh.candidates() == [("w", "x")]


def test_candidates_sorted_and_unique():
    lsh = LSH(2, 2)
    for k in "cab":
        lsh.insert(k, np.array([1, 2, 3, 4], dtype=np.uint64))
    assert lsh.candidates() == [("a", "b"), ("a", "c"), ("b", "c")]


@given(
    st.lists(
        st.tuples(st.sampled_from("abcdefgh"), st.sampled_from("abcdefgh")), max_size=12
    )
)
def test_union_find_roots_are_minimal_and_groups_partition(pairs):
    uf = UnionFind()
    for x, y in pairs:
        uf.union(x, y)
    groups = uf.groups()
    seen = set()
    for g in groups:
        assert len(g) >= 2 and not (g & seen)
        seen |= g
        assert all(uf.find(x) == min(g) for x in g)
    assert groups == sorted(groups, key=min)


def test_a_chain_is_one_cluster_and_keeps_its_smallest_id():
    """x ~ y ~ z with x !~ z: one cluster; near_dedup keeps x."""
    t = {"x": line(0, 200), "y": line(40, 200), "z": line(80, 200)}
    opts = dict(k=1, num_perm=512, bands=128, threshold=0.55)
    assert clusters(t, **opts) == [{"x", "y", "z"}]
    assert [d.id for d in near_dedup([doc(k, v) for k, v in t.items()], **opts)] == [
        "x"
    ]


def test_candidates_are_confirmed():
    """At J = 0.6 with 32 bands of 4 rows every pair is a candidate; none
    reaches the 0.8 estimate, so no cluster."""
    t = {"a": line(0, 120), "b": line(30, 120)}  # J = 90 / 150 = 0.6
    assert clusters(t, k=1, bands=32) == []


def test_workers_do_not_change_the_result():
    texts = [line(i * 7, 60) for i in range(9)]
    assert np.array_equal(signatures(texts, workers=1), signatures(texts, workers=3))


def test_near_dedup_tags_and_keeps_order():
    t = line(0, 60)
    docs = [doc("b", t, url="u"), doc("c", line(500, 60)), doc("a", t + " extra")]
    tagged = list(near_dedup(docs, drop=False))
    assert [d.id for d in tagged] == ["b", "c", "a"]
    assert [d.meta["minhash_cluster"] for d in tagged] == ["a", "c", "a"]
    assert tagged[0].meta["url"] == "u"
    assert [d.id for d in near_dedup(docs)] == ["c", "a"]
    with pytest.raises(ValueError):
        list(near_dedup([doc("a", "x"), doc("a", "y")]))


@given(st.integers(0, 20), st.integers(0, 20))
def test_a_protected_13_gram_anywhere_contaminates(before, after):
    gram = " ".join(f"p{i}" for i in range(13))
    text = " ".join(
        [line(100, before), gram.upper().replace(" ", ", "), line(200, after)]
    )
    assert contaminated(text, {gram})
    assert not contaminated(" ".join(f"p{i}" for i in range(12)), {gram})


def test_protected_files_skip_ids_tags_and_scorer_args(tmp_path):
    w13 = " ".join(f"m{i}" for i in range(13))
    p = tmp_path / "s.jsonl"
    p.write_text(
        json.dumps(
            {"case_id": w13, "tags": [w13], "scorer_args": {"r": w13}, "input": "short"}
        )
        + "\n\n"
    )
    assert protected_ngrams([p]) == set()
    q = tmp_path / "t.txt"
    q.write_text(w13)
    assert protected_ngrams([q]) == {w13}


def test_decontaminate_pulls_one_document_at_a_time(tmp_path):
    q = tmp_path / "t.txt"
    q.write_text(" ".join(f"m{i}" for i in range(13)))

    def stream():
        for i in itertools.count():
            assert i < 20
            yield doc(f"d{i}", "plain")

    assert len(list(itertools.islice(decontaminate(stream(), [q]), 3))) == 3
