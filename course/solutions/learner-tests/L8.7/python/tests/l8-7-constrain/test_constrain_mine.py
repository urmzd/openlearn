"""My tests for L8.7 (rung R5). Oracles that are not my code: Python's re
(fullmatch on bytes) for the language, a brute-force walk of every token
through the DFA for the masks, json.loads plus a small validator for the
schema subset. They import only the contract."""

from __future__ import annotations

import json
import math
import os
import re

import numpy as np
import pytest

from tinyllm.infer.constrain import (
    DEAD,
    Constraint,
    TokenIndex,
    apply_mask,
    constrained_sample,
    json_schema_to_regex,
    regex_to_dfa,
    vocab_bytes,
)
from tinyllm.infer.sample import SamplingParams, request_rng
from tinyllm.num.rng import PCG32
from tinyllm.tok.bytes_unicode import bytes_to_unicode

SEED = int(os.environ.get("OL_SEED", "0"))

HAND = r"(cat|car|dog)s?"
HAND_VOCAB = [
    b"c",
    b"a",
    b"t",
    b"r",
    b"d",
    b"o",
    b"g",
    b"s",
    b"ca",
    b"cat",
    b"dog",
    b"ts",
    b"x",
    None,
    b"",
]
HAND_EOS = 13
BYTE_VOCAB = [bytes([i]) for i in range(256)] + [None]  # id 256: EOS
EOS = 256

PROBES = [
    b"",
    b"\n",
    b"a\nb",
    b"a_b",
    b"a\xffb",
    b"\xff",
    b"ab\n",
    b"cat\n",
    b"[_ ]",
    b"(1.5)",
    b"x\x80",
]
PATTERNS = [
    HAND,
    r"[a-c]{2,3}(?:x|\d)*[^a-z]",
    r"\(\d+\.\d\)|\[\w\s?\]",
    r"a.b",
    r"(?:ab|a)(?:bc|c)?",
    r"[^\x00-\x2f]+|\x41{0}z",
    r"x{2,}|y{0,3}",
    r"(a|b)*abb",
    r'"(?:[^"\\]|\\.)*"',
    r"\D\W\S",
]


def walk(dfa, rng: PCG32, max_len: int = 12) -> bytes:
    """A random string: at each state take a uniformly chosen live byte,
    stopping at an accepting state with probability 1/3."""
    s, out = 0, bytearray()
    for _ in range(max_len):
        if dfa.accept[s] and rng.uniform() < 1 / 3:
            break
        live = np.flatnonzero(dfa.trans[s] != DEAD)
        if live.size == 0:
            break
        b = int(live[rng.below(live.size)])
        out.append(b)
        s = int(dfa.trans[s, b])
    return bytes(out)


def near_misses(data: bytes, rng: PCG32) -> list[bytes]:
    """data with one byte changed, dropped, or added, and every prefix."""
    out = [data[:i] for i in range(len(data) + 1)]
    pool = b'abcdgorstxyz0129.()[]_ \n"\\!AB\x80\xc3\xff'
    for _ in range(6):
        i = rng.below(len(data) + 1)
        c = pool[rng.below(len(pool))]
        out += [
            data[:i] + bytes([c]) + data[i:],
            data[:i] + bytes([c]) + data[i + 1 :],
            data[:i] + data[i + 1 :],
        ]
    return out


# --- the worked example ---------------------------------------------------------------


def test_hand_example():
    d = regex_to_dfa(HAND)
    assert d.n_states == 7
    assert d.accept.tolist() == [False, False, False, False, False, True, True]
    edges = {
        (s, chr(b)): int(d.trans[s, b])
        for s in range(7)
        for b in range(256)
        if d.trans[s, b] != DEAD
    }
    assert edges == {
        (0, "c"): 1, (0, "d"): 2, (1, "a"): 3, (2, "o"): 4,
        (3, "r"): 5, (3, "t"): 5, (4, "g"): 5, (5, "s"): 6,
    }  # fmt: skip
    assert d.trans.dtype == np.int32 and d.trans.shape == (7, 256)
    assert (
        d.step(0, b"cats") == 6
        and d.step(0, b"cab") == DEAD
        and d.step(DEAD, b"") == DEAD
    )
    assert d.matches(b"dogs") and not d.matches(b"do") and not d.matches(b"catss")


def test_hand_example_masks():
    idx = TokenIndex(regex_to_dfa(HAND), HAND_VOCAB, eos_id=HAND_EOS)
    allowed = [np.flatnonzero(idx.mask(s)).tolist() for s in range(7)]
    assert allowed == [[0, 4, 8, 9, 10], [1], [5], [2, 3, 11], [6], [7, 13], [13]]
    assert idx.next_state(0, 9) == 5 and idx.next_state(3, 11) == 6
    assert idx.next_state(5, HAND_EOS) == 5 and idx.next_state(3, HAND_EOS) == DEAD
    assert idx.next_state(0, 12) == DEAD and idx.next_state(0, 14) == DEAD


# --- the automaton against Python's re --------------------------------------------------


@pytest.mark.parametrize("pattern", PATTERNS)
def test_matches_python_re(pattern):
    d = regex_to_dfa(pattern)
    oracle = re.compile(pattern.encode("ascii"))
    for s in PROBES:  # fixed edge strings first: newline, high bytes, empty
        assert d.matches(s) == bool(oracle.fullmatch(s)), f"{pattern!r} on {s!r}"
    rng = PCG32(SEED + 7 + PATTERNS.index(pattern))
    for _ in range(25):
        for s in near_misses(walk(d, rng), rng):
            assert d.matches(s) == bool(oracle.fullmatch(s)), f"{pattern!r} on {s!r}"


def test_every_state_is_live():
    for pattern in PATTERNS:
        d = regex_to_dfa(pattern)
        live = set(np.flatnonzero(d.accept).tolist())
        grew = True
        while grew:
            grew = False
            for s in range(d.n_states):
                if s not in live and any(
                    int(t) in live for t in d.trans[s] if t != DEAD
                ):
                    live.add(s)
                    grew = True
        assert live == set(range(d.n_states)), (
            f"{pattern!r}: dead states {set(range(d.n_states)) - live}"
        )


def test_canonical_form():
    pairs = [
        (r"a|b", r"[ab]"),
        (r"(?:ab)*a", r"a(?:ba)*"),
        (r"x{2,}", r"xxx*"),
        (r"(cat|car|dog)s?", r"(?:ca[rt]|dog)(?:s|)"),
        (r"[0-9]{1,3}", r"\d(?:\d\d?)?"),
    ]
    for p, q in pairs:
        a, b = regex_to_dfa(p), regex_to_dfa(q)
        assert a.n_states == b.n_states, (p, q)
        assert np.array_equal(a.trans, b.trans) and np.array_equal(
            a.accept, b.accept
        ), (p, q)


def test_unsupported_syntax_raises():
    for bad in [
        r"^a",
        r"a$",
        r"(a)\1",
        r"(?=a)a",
        r"a*?",
        "\u00e9",
        r"a{3,1}",
        r"a{300}",
        r"[z-a]",
        r"(a",
        r"a)",
        r"[^\x00-\xff]",
        r"[",
        r"\q",
    ]:
        with pytest.raises(ValueError):
            regex_to_dfa(bad)


# --- masks ---------------------------------------------------------------------------------


def test_masks_match_brute_force():
    vocab = BYTE_VOCAB[:256] + [
        b"ca",
        b"cat",
        b"cats",
        b"do",
        b"dog",
        b"(1",
        b"(1.",
        b".5)",
        b"[a",
        b"[a ]",
        b"ab",
        b"abb",
        None,
        b"",
    ]
    for pattern in (HAND, r"\(\d+\.\d\)|\[\w\s?\]", r"(a|b)*abb"):
        d = regex_to_dfa(pattern)
        idx = TokenIndex(d, vocab)
        for s in range(d.n_states):
            want = np.array(
                [tb is not None and tb != b"" and d.step(s, tb) != DEAD for tb in vocab]
            )
            assert np.array_equal(idx.mask(s), want), (pattern, s)
            for t in np.flatnonzero(want):
                assert idx.next_state(s, int(t)) == d.step(s, vocab[t])


def test_reachable_states_never_have_an_empty_mask():
    for pattern in PATTERNS + [
        json_schema_to_regex(
            {"type": "array", "items": {"type": "integer"}, "maxItems": 2}
        )
    ]:
        idx = TokenIndex(regex_to_dfa(pattern), BYTE_VOCAB, eos_id=EOS)
        seen, todo = {0}, [0]
        while todo:
            s = todo.pop()
            m = idx.mask(s)
            assert m.any(), f"{pattern!r}: state {s} has an empty mask"
            for t in np.flatnonzero(m):
                n = idx.next_state(s, int(t))
                if n not in seen:
                    seen.add(n)
                    todo.append(n)


def test_constraint_lifecycle():
    idx = TokenIndex(regex_to_dfa(HAND), HAND_VOCAB, eos_id=HAND_EOS)
    c = Constraint(idx)
    assert c.state == 0 and not c.done and not c.is_complete()
    with pytest.raises(ValueError):
        c.advance(12)  # x
    with pytest.raises(ValueError):
        c.advance(HAND_EOS)  # not complete yet
    c.advance(8)  # ca
    c.advance(2)  # t
    assert c.is_complete() and c.mask().tolist() == [i in (7, 13) for i in range(15)]
    c.advance(HAND_EOS)
    assert c.done and not c.mask().any()
    with pytest.raises(ValueError):
        c.advance(7)


# --- sampling under a mask --------------------------------------------------------------------


def test_apply_mask():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = apply_mask(x, [True, False, True])
    assert y.dtype == np.float64 and y.tolist() == [1.0, -math.inf, 3.0] and x[1] == 2.0
    with pytest.raises(ValueError):
        apply_mask(x, [False, False, False])
    with pytest.raises(ValueError):
        apply_mask(x, [True, False])


def test_greedy_takes_the_best_allowed_token():
    idx = TokenIndex(regex_to_dfa(HAND), HAND_VOCAB, eos_id=HAND_EOS)
    logits = np.zeros(15)
    logits[12] = 9.0  # x: not allowed
    logits[[4, 9]] = 2.0  # d and cat tie
    c = Constraint(idx)
    tok, lp = constrained_sample(
        logits, c, SamplingParams(temperature=0.0), [], request_rng(0)
    )
    assert tok == 4 and c.state == 2
    z = math.log(2 * math.exp(2.0) + 3 * math.exp(0.0))  # allowed: c, d, ca, cat, dog
    assert abs(lp - (2.0 - z)) < 1e-12


def schema_ok(v, s) -> bool:
    """A small validator for the subset (the test's own, not yours)."""
    if "enum" in s:
        return any(v == e and type(v) is type(e) for e in s["enum"])
    if "anyOf" in s:
        return any(schema_ok(v, x) for x in s["anyOf"])
    t = s["type"]
    if t == "object":
        return (
            isinstance(v, dict)
            and set(s.get("required", [])) <= set(v)
            and set(v) <= set(s["properties"])
            and list(v) == [k for k in s["properties"] if k in v]
            and all(schema_ok(v[k], s["properties"][k]) for k in v)
        )
    if t == "array":
        return (
            isinstance(v, list)
            and s.get("minItems", 0) <= len(v) <= s.get("maxItems", 1 << 30)
            and all(schema_ok(x, s["items"]) for x in v)
        )
    return {
        "string": lambda: isinstance(v, str),
        "integer": lambda: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda: isinstance(v, bool),
        "null": lambda: v is None,
    }[t]()


TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "days": {"type": "integer"},
        "units": {"enum": ["c", "f"]},
        "hourly": {"type": "boolean"},
        "at": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 1,
            "maxItems": 3,
        },
    },
    "required": ["city", "days"],
}


def test_constrained_generation_parses():
    idx = TokenIndex(
        regex_to_dfa(json_schema_to_regex(TOOL_SCHEMA)), BYTE_VOCAB, eos_id=EOS
    )
    noise = np.random.default_rng(
        SEED + 87
    )  # a seeded generator object, never the global one
    p = SamplingParams(temperature=1.0)
    closers = [ord(ch) for ch in '"},]']
    finished = 0
    for req in range(40):
        c, out, rng = Constraint(idx), [], request_rng(SEED + req)
        while not c.done and len(out) < 400:
            logits = 2.0 * noise.standard_normal(257)
            logits[closers] += 5.0  # keep strings, numbers, and arrays short
            logits[EOS] += 4.0 if c.is_complete() else 0.0
            tok, _ = constrained_sample(logits, c, p, out, rng)
            out.append(tok)
        if not c.done:
            continue  # ran out of budget mid-document: allowed, just not checked
        finished += 1
        text = bytes(t for t in out if t != EOS)
        doc = json.loads(text)  # bytes: json decodes UTF-8 and rejects anything else
        assert schema_ok(doc, TOOL_SCHEMA), text
    assert finished >= 30, f"only {finished} of 40 requests reached EOS"


# --- JSON schema -------------------------------------------------------------------------------


def test_json_schema_documents():
    d = regex_to_dfa(json_schema_to_regex(TOOL_SCHEMA))

    def enc(v) -> bytes:
        return json.dumps(v, separators=(",", ":"), ensure_ascii=False).encode()

    good = [
        {"city": "Oslo", "days": 3},
        {
            "city": 'a"b\\cé\n',
            "days": -12,
            "units": "f",
            "hourly": False,
            "at": [1, -2.5, 3e10],
        },
        {"city": "", "days": 0, "at": [0.0]},
        {"city": "x", "days": 7, "hourly": True},
    ]
    for v in good:
        assert d.matches(enc(v)), enc(v)
    bad = [
        enc({"city": "Oslo"}),
        enc({"city": 3, "days": 3}),
        enc({"days": 3, "city": "Oslo"}),
        enc({"city": "x", "days": 3, "wind": 1}),
        enc({"city": "x", "days": 3.5}),
        enc({"city": "x", "days": 3, "units": "k"}),
        enc({"city": "x", "days": 3, "at": []}),
        enc({"city": "x", "days": 3, "at": [1, 2, 3, 4]}),
        b'{"city":"x","days":3,}',
        b'{"city": "x","days":3}',
        b'{"city":"x","days":03}',
        b'{"city":"\x01","days":3}',
        b'{"city":"\xff","days":3}',
        b'{"city":"\xc3","days":3}',
        b'{"city":"x","days":3,"at":[1e]}',
    ]
    for s in bad:
        assert not d.matches(s), s


def test_json_schema_rejects_keywords_outside_the_subset():
    for s in [
        {"type": "string", "pattern": "a+"},
        {"type": "string", "maxLength": 3},
        {"$ref": "#/defs/x"},
        {"type": "object", "properties": {}, "additionalProperties": True},
        {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["b"]},
        {"type": "tuple"},
        "string",
    ]:
        with pytest.raises(ValueError):
            json_schema_to_regex(s)
    assert regex_to_dfa(
        json_schema_to_regex({"type": "string", "description": "ignored", "title": "t"})
    ).matches(b'"ok"')


def test_vocab_bytes():
    class Bytes:  # the tracer's byte tokenizer, as L1.1 writes its tokens
        vocab_size = 256
        special_ids: dict = {}

        def id_to_token(self, i: int) -> str:
            return bytes_to_unicode()[i]

    assert vocab_bytes(Bytes()) == [bytes([i]) for i in range(256)]

    class Tiny:
        vocab_size = 4
        special_ids = {"<eos>": 3}

        def id_to_token(self, i: int) -> str:
            return ["\u0120cat", "\u00e9t\u00e9", "\u65e5\u672c", "<eos>"][i]

    # The second string is made of byte-map characters (U+00E9 stands for
    # byte 0xE9), so it is three bytes; CJK characters are not in the map.
    assert vocab_bytes(Tiny()) == [
        b" cat",
        b"\xe9t\xe9",
        "\u65e5\u672c".encode("utf-8"),
        None,
    ]
