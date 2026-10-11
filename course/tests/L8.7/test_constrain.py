"""Course tests for L8.7: constrained decoding (tinyllm/infer/constrain.py).

Rung R0 for the course suite: read these before you write code. Each test
names why it exists (WHY), what kind of check it is (KIND), the planted bugs
it kills (CATCHES, mutants in course/mutants/L8.7), and the chapter section
it comes from.

Two oracles, neither of them your code: Python's own `re` (fullmatch on
bytes) decides which strings a pattern matches, and
course/fixtures/L8.7/masks_golden.json holds token masks computed by an
independent stdlib program (course/oracle/L8.7/masks_golden.py: Brzozowski
derivatives over trees from Python's regex parser, no automaton at all).
The same golden file holds the Rust port (L10.9) to your masks. Random
inputs come from the frozen PCG32.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pytest
from _lib.pcg32 import PCG32

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
from tinyllm.tok.bytes_unicode import bytes_to_unicode

SEED = int(os.environ.get("OL_SEED", "0"))
FIX = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "L8.7" / "masks_golden.json"

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
    # WHY: the chapter's worked example, state by state. (cat|car|dog)s? has
    #      a minimal DFA of 7 states, numbered breadth first with bytes in
    #      ascending order: 0 -c-> 1, 0 -d-> 2, 1 -a-> 3, 2 -o-> 4, 3 -r,t-> 5,
    #      4 -g-> 5, 5 -s-> 6; 5 and 6 accept. "cat", "car", and "dog" all
    #      reach 5, which is why they share it.
    # KIND: unit, smoke
    # CATCHES: s03, s04, s09
    # CHAPTER: L8.7 section 3
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
    # WHY: the chapter's token table. In state 0 the tokens c, d, ca, cat,
    #      dog are allowed; in state 3 ("ca" read) t, r, and ts (t then s);
    #      EOS only in the accepting states 5 and 6; the special (None) and
    #      the empty token never. next_state follows the token's bytes.
    # KIND: unit, smoke
    # CATCHES: s05, s06, m002
    # CHAPTER: L8.7 section 3
    idx = TokenIndex(regex_to_dfa(HAND), HAND_VOCAB, eos_id=HAND_EOS)
    allowed = [np.flatnonzero(idx.mask(s)).tolist() for s in range(7)]
    assert allowed == [[0, 4, 8, 9, 10], [1], [5], [2, 3, 11], [6], [7, 13], [13]]
    assert idx.next_state(0, 9) == 5 and idx.next_state(3, 11) == 6
    assert idx.next_state(5, HAND_EOS) == 5 and idx.next_state(3, HAND_EOS) == DEAD
    assert idx.next_state(0, 12) == DEAD and idx.next_state(0, 14) == DEAD


# --- the automaton against Python's re --------------------------------------------------


@pytest.mark.parametrize("pattern", PATTERNS)
def test_matches_python_re(pattern):
    # WHY: Python's re (fullmatch on bytes) is the definition of the subset.
    #      Strings drawn from your DFA, every prefix of them, and one-byte
    #      edits of them must get the same yes or no from both, so a wrong
    #      class, a wrong repeat count, or an accepting state lost in
    #      minimization shows up as a disagreement.
    # KIND: golden
    # CATCHES: s01, s02, s04, s08, s09, s10, s11, m001, m003
    # CHAPTER: L8.7 section 2.1
    d = regex_to_dfa(pattern)
    oracle = re.compile(pattern.encode("ascii"))
    for s in PROBES:  # fixed edge strings first: newline, high bytes, empty
        assert d.matches(s) == bool(oracle.fullmatch(s)), f"{pattern!r} on {s!r}"
    rng = PCG32(SEED + 7 + PATTERNS.index(pattern))
    for _ in range(25):
        for s in near_misses(walk(d, rng), rng):
            assert d.matches(s) == bool(oracle.fullmatch(s)), f"{pattern!r} on {s!r}"


def test_every_state_is_live():
    # WHY: dead-state removal is what makes masks safe: from every state
    #      some accepting state must still be reachable, or the sampler can
    #      walk into a state where no token is allowed and generation is
    #      stuck. Checked by a backward search on each pattern's DFA.
    # KIND: property
    # CATCHES: s12, m001
    # CHAPTER: L8.7 section 2.2
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
    # WHY: the DFA is minimal and numbered canonically, so two patterns with
    #      the same language give the same arrays: the Rust port (L10.9) and
    #      your Python can be compared table to table.
    # KIND: property
    # CATCHES: s03, s11
    # CHAPTER: L8.7 section 2.2
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
    # WHY: a pattern outside the subset must fail loudly at compile time,
    #      not silently mean something else (an anchor read as a literal ^,
    #      a lazy *? read as an optional star).
    # KIND: boundary
    # CATCHES: s13, m004
    # CHAPTER: L8.7 section 4
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
    # WHY: the trie walk is a speedup of one rule: token t is allowed in s
    #      exactly when stepping its bytes from s stays live. Checked against
    #      that rule, token by token, for every state of three patterns and a
    #      vocabulary with multi-byte tokens sharing prefixes.
    # KIND: property
    # CATCHES: s06, s14, m002
    # CHAPTER: L8.7 section 2.3
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


def test_masks_match_the_golden_file():
    # WHY: masks along recorded token paths, from an oracle that shares no
    #      algorithm with yours (derivatives, not automata), for three
    #      regexes and two JSON schemas: the allowed ids at every step, and
    #      whether the text so far is complete. This file is also the Rust
    #      port's (L10.9) contract.
    # KIND: golden
    # CATCHES: s05, s06, s08, s10, s16, s17, s23, s24
    # CHAPTER: L8.7 section 4
    doc = json.loads(FIX.read_text())
    for case in doc["cases"]:
        pattern = (
            case["pattern"]
            if "pattern" in case
            else json_schema_to_regex(case["schema"])
        )
        vocab = [None if v is None else bytes.fromhex(v) for v in case["vocab"]]
        idx = TokenIndex(regex_to_dfa(pattern), vocab, eos_id=case["eos_id"])
        for path in case["paths"]:
            c = Constraint(idx)
            for i, step in enumerate(path):
                assert np.flatnonzero(c.mask()).tolist() == step["mask"], (
                    f"{case['name']} step {i}"
                )
                assert c.is_complete() == step["complete"], f"{case['name']} step {i}"
                if step["token"] is not None:
                    c.advance(step["token"])


def test_reachable_states_never_have_an_empty_mask():
    # WHY: the catalog's invariant. With every byte in the vocabulary and an
    #      EOS id, every state reachable through allowed tokens allows at
    #      least one token, so constrained generation can always continue or
    #      stop. Explored breadth first over token transitions.
    # KIND: property
    # CATCHES: s12, s14, m002
    # CHAPTER: L8.7 section 2.3
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
    # WHY: one request's walk: advance moves the state, a token outside the
    #      mask is refused (so an engine bug cannot silently leave the
    #      language), EOS is allowed only when complete and then nothing is.
    # KIND: unit
    # CATCHES: s05, s07, s19, s20
    # CHAPTER: L8.7 section 4
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
    # WHY: masking is "-inf where not allowed", in float64, on a copy; the
    #      sampler (L8.1) then treats -inf as a token it can never draw. A
    #      mask that allows nothing is an error, not an all -inf row.
    # KIND: boundary
    # CATCHES: s21, m005
    # CHAPTER: L8.7 section 4
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = apply_mask(x, [True, False, True])
    assert y.dtype == np.float64 and y.tolist() == [1.0, -math.inf, 3.0] and x[1] == 2.0
    with pytest.raises(ValueError):
        apply_mask(x, [False, False, False])
    with pytest.raises(ValueError):
        apply_mask(x, [True, False])


def test_greedy_takes_the_best_allowed_token():
    # WHY: greedy under a constraint is the largest ALLOWED logit, ties to
    #      the lowest id, even when a disallowed token has the largest logit
    #      overall; the logprob is renormalized over the allowed tokens.
    # KIND: unit
    # CATCHES: s05, s06, s22
    # CHAPTER: L8.7 section 2.4
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
    # WHY: the property the engine relies on for tool calls (L10.9): with
    #      random logits, sampling under the schema's mask until EOS always
    #      produces text that json.loads accepts and that satisfies the
    #      schema, including well-formed UTF-8 inside strings. 40 seeded
    #      requests at temperature 1 over the byte vocabulary.
    # KIND: property
    # CATCHES: s15, s18, s20, s22, s23, s24
    # CHAPTER: L8.7 section 2.5
    idx = TokenIndex(
        regex_to_dfa(json_schema_to_regex(TOOL_SCHEMA)), BYTE_VOCAB, eos_id=EOS
    )
    noise = PCG32(SEED + 87)
    p = SamplingParams(temperature=1.0)
    closers = [ord(ch) for ch in '"},]']
    finished = 0
    for req in range(40):
        c, out, rng = Constraint(idx), [], request_rng(SEED + req)
        while not c.done and len(out) < 400:
            logits = noise.normal_array((257,), scale=2.0)
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
    # WHY: the schema subset decided on documents written by json.dumps
    #      (compact): valid ones match; a missing required key, a wrong
    #      type, a key out of order, an unknown key, a trailing comma, a
    #      space, and too many items do not.
    # KIND: unit
    # CATCHES: s15, s16, s17, s18
    # CHAPTER: L8.7 section 2.5
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
    ]
    for s in bad:
        assert not d.matches(s), s


def test_json_schema_rejects_keywords_outside_the_subset():
    # WHY: a schema keyword the translation does not implement would be
    #      silently ignored, and the model would then be allowed output the
    #      schema forbids. Every such keyword is a ValueError that names it.
    # KIND: boundary
    # CATCHES: s25
    # CHAPTER: L8.7 section 5, Pitfalls
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
    # WHY: masks need each token's BYTES. Byte-level tokens are written in
    #      the GPT-2 byte map (id 32 is "G with a dot", not a space), so the
    #      map must be undone; a token string with a character outside the
    #      map is UTF-8; special tokens have no bytes and are never allowed.
    # KIND: unit
    # CATCHES: s26, s27
    # CHAPTER: L8.7 section 2.3
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
