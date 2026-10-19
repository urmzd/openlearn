"""Constrained decoding: regex -> NFA -> DFA -> token masks (L8.7).

Contract: contracts/py/tinyllm/infer/constrain.pyi. The pipeline:

1. parse   a recursive-descent parser turns the pattern into a tree of
           byte sets, concatenations, alternations, and repetitions;
2. NFA     Thompson's construction: one small fragment per tree node,
           glued with epsilon edges;
3. DFA     the subset construction over byte classes (bytes that every
           NFA edge treats alike), then dead states removed, Moore
           minimization, and canonical breadth-first numbering;
4. masks   a token is allowed in state s when walking its bytes from s
           stays live; a byte trie over the vocabulary shares the walk
           between tokens with a common prefix.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.infer.sample import SamplingParams, sample
from tinyllm.prob.sampling import UniformSource
from tinyllm.tok.bytes_unicode import unicode_to_bytes

DEAD = -1
MAX_REPEAT = 256
ALL = (1 << 256) - 1  # a byte set is an int with bit b set for byte b


def _bits(bs) -> int:
    v = 0
    for b in bs:
        v |= 1 << b
    return v


def _range(lo: int, hi: int) -> int:
    return ((1 << (hi + 1)) - 1) ^ ((1 << lo) - 1)


DIGIT = _range(0x30, 0x39)
WORD = DIGIT | _range(0x41, 0x5A) | _range(0x61, 0x7A) | _bits([0x5F])
SPACE = _bits(b" \t\n\r\f\v")
DOT = ALL & ~_bits(b"\n")
CLASS_ESC = {
    "d": DIGIT,
    "w": WORD,
    "s": SPACE,
    "D": ALL & ~DIGIT,
    "W": ALL & ~WORD,
    "S": ALL & ~SPACE,
}
CHAR_ESC = {"n": 0x0A, "t": 0x09, "r": 0x0D, "f": 0x0C, "v": 0x0B}
LITERAL_ESC = set("\\.*+?()[]{}|^$/\"-,:#&~ '")


# -- 1. parse -------------------------------------------------------------------
# Nodes: ("set", bits) | ("cat", [nodes]) | ("alt", [nodes]) | ("rep", node, m, n)
# with n = None for unbounded. ("cat", []) is the empty string.


class _Parser:
    def __init__(self, pattern: str) -> None:
        for ch in pattern:
            if ord(ch) > 0x7F:
                raise ValueError(
                    f"non-ASCII character {ch!r} in pattern: write it as \\xHH bytes"
                )
        self.s = pattern
        self.i = 0

    def error(self, msg: str) -> ValueError:
        return ValueError(f"regex: {msg} at position {self.i} in {self.s!r}")

    def peek(self) -> str:
        return self.s[self.i] if self.i < len(self.s) else ""

    def parse(self):
        node = self.alt()
        if self.i != len(self.s):
            raise self.error(f"unexpected {self.peek()!r}")
        return node

    def alt(self):
        branches = [self.cat()]
        while self.peek() == "|":
            self.i += 1
            branches.append(self.cat())
        return branches[0] if len(branches) == 1 else ("alt", branches)

    def cat(self):
        items = []
        while self.peek() not in ("", "|", ")"):
            items.append(self.repeat())
        return items[0] if len(items) == 1 else ("cat", items)

    def repeat(self):
        node = self.atom()
        while self.peek() in ("*", "+", "?", "{"):
            c = self.peek()
            if c == "{":
                m, n = self.braces()
            else:
                self.i += 1
                m, n = {"*": (0, None), "+": (1, None), "?": (0, 1)}[c]
            if self.peek() == "?":
                raise self.error("lazy quantifiers are not in the subset")
            node = ("rep", node, m, n)
        return node

    def braces(self) -> tuple[int, Optional[int]]:
        j = self.s.find("}", self.i)
        if j < 0:
            raise self.error("unclosed {")
        body = self.s[self.i + 1 : j]
        parts = body.split(",")
        try:
            if len(parts) == 1:
                m = n = int(parts[0])
            elif len(parts) == 2:
                m = int(parts[0])
                n = int(parts[1]) if parts[1] else None
            else:
                raise ValueError
        except ValueError:
            raise self.error(f"bad repetition {{{body}}}") from None
        if m < 0 or (n is not None and n < m) or max(m, n or 0) > MAX_REPEAT:
            raise self.error(
                f"repetition {{{body}}} out of range (0 <= m <= n <= {MAX_REPEAT})"
            )
        self.i = j + 1
        return m, n

    def escape(self) -> tuple[int, bool]:
        """After a backslash: (byte set, is_class)."""
        c = self.peek()
        if c == "":
            raise self.error("trailing backslash")
        self.i += 1
        if c in CLASS_ESC:
            return CLASS_ESC[c], True
        if c in CHAR_ESC:
            return 1 << CHAR_ESC[c], False
        if c == "x":
            h = self.s[self.i : self.i + 2]
            if len(h) != 2 or any(ch not in "0123456789abcdefABCDEF" for ch in h):
                raise self.error("\\x needs two hex digits")
            self.i += 2
            return 1 << int(h, 16), False
        if c in LITERAL_ESC:
            return 1 << ord(c), False
        raise self.error(f"unsupported escape \\{c}")

    def atom(self):
        c = self.peek()
        if c == "(":
            self.i += 1
            if self.s.startswith("?:", self.i):
                self.i += 2
            elif self.peek() == "?":
                raise self.error("only (?:...) groups are in the subset")
            node = self.alt()
            if self.peek() != ")":
                raise self.error("unclosed (")
            self.i += 1
            return node
        if c == "[":
            return ("set", self.bracket())
        if c == ".":
            self.i += 1
            return ("set", DOT)
        if c == "\\":
            self.i += 1
            return ("set", self.escape()[0])
        if c in ("*", "+", "?", "{", ")", "^", "$", "]", "}"):
            raise self.error(f"unexpected {c!r}")
        self.i += 1
        return ("set", 1 << ord(c))

    def bracket(self) -> int:
        self.i += 1  # [
        neg = self.peek() == "^"
        if neg:
            self.i += 1
        bits, first = 0, True
        while True:
            c = self.peek()
            if c == "":
                raise self.error("unclosed [")
            if c == "]" and not first:
                self.i += 1
                break
            first = False
            if c == "\\":
                self.i += 1
                lo_set, is_class = self.escape()
            else:
                self.i += 1
                lo_set, is_class = 1 << ord(c), False
            if (
                not is_class
                and self.peek() == "-"
                and self.s[self.i + 1 : self.i + 2] not in ("]", "")
            ):
                self.i += 1
                d = self.peek()
                if d == "\\":
                    self.i += 1
                    hi_set, hi_class = self.escape()
                    if hi_class:
                        raise self.error("a class cannot end a range")
                else:
                    self.i += 1
                    hi_set = 1 << ord(d)
                lo, hi = lo_set.bit_length() - 1, hi_set.bit_length() - 1
                if hi < lo:
                    raise self.error("range out of order")
                bits |= _range(lo, hi)
            else:
                bits |= lo_set
        return (ALL & ~bits) if neg else bits


# -- 2. NFA (Thompson) ------------------------------------------------------------


class _NFA:
    def __init__(self) -> None:
        self.eps: list[list[int]] = []
        self.edges: list[list[tuple[int, int]]] = []  # (byte set, target)

    def state(self) -> int:
        self.eps.append([])
        self.edges.append([])
        return len(self.eps) - 1

    def build(self, node) -> tuple[int, int]:
        """A fragment (start, end) for node: end has no outgoing edges yet."""
        kind = node[0]
        if kind == "set":
            s, e = self.state(), self.state()
            self.edges[s].append((node[1], e))
            return s, e
        if kind == "cat":
            s = e = self.state()
            for child in node[1]:
                cs, ce = self.build(child)
                self.eps[e].append(cs)
                e = ce
            return s, e
        if kind == "alt":
            s, e = self.state(), self.state()
            for child in node[1]:
                cs, ce = self.build(child)
                self.eps[s].append(cs)
                self.eps[ce].append(e)
            return s, e
        _, child, m, n = node
        s = e = self.state()
        for _ in range(m):  # m mandatory copies
            cs, ce = self.build(child)
            self.eps[e].append(cs)
            e = ce
        if n is None:  # then a loop: zero or more
            cs, ce = self.build(child)
            loop = self.state()
            self.eps[e].append(loop)
            self.eps[loop].append(cs)
            self.eps[ce].append(loop)
            return s, loop
        end = self.state()
        self.eps[e].append(end)
        for _ in range(n - m):  # n - m optional copies, each may stop early
            cs, ce = self.build(child)
            self.eps[e].append(cs)
            self.eps[ce].append(end)
            e = ce
        return s, end

    def closure(self, states) -> frozenset[int]:
        seen = set(states)
        stack = list(states)
        while stack:
            for t in self.eps[stack.pop()]:
                if t not in seen:
                    seen.add(t)
                    stack.append(t)
        return frozenset(seen)


# -- 3. DFA ---------------------------------------------------------------------


class DFA:
    def __init__(self, trans: NDArray, accept: NDArray) -> None:
        self.trans = np.ascontiguousarray(trans, dtype=np.int32)
        self.accept = np.ascontiguousarray(accept, dtype=bool)
        self.n_states = int(self.trans.shape[0])

    def step(self, state: int, data: bytes) -> int:
        if state != DEAD and not 0 <= state < self.n_states:
            raise ValueError(f"state {state} outside [0, {self.n_states})")
        t = self.trans
        for b in data:
            if state == DEAD:
                return DEAD
            state = int(t[state, b])
        return state

    def matches(self, data: bytes) -> bool:
        s = self.step(0, data)
        return s != DEAD and bool(self.accept[s])


def _byte_classes(nfa: _NFA) -> list[int]:
    """Partition the 256 bytes so every NFA edge set is a union of classes."""
    classes = [ALL]
    for sets in nfa.edges:
        for bits, _ in sets:
            nxt = []
            for c in classes:
                a, b = c & bits, c & ~bits
                nxt += [x for x in (a, b) if x]
            classes = nxt
    return classes


def regex_to_dfa(pattern: str) -> DFA:
    tree = _Parser(pattern).parse()
    nfa = _NFA()
    start, final = nfa.build(tree)
    classes = _byte_classes(nfa)
    # Each edge's byte set is a union of whole classes: list them once.
    covers = [
        [([ci for ci, c in enumerate(classes) if c & bits], t) for bits, t in out]
        for out in nfa.edges
    ]
    closures: dict[frozenset[int], frozenset[int]] = {}

    # Subset construction: a DFA state is the closed set of NFA states.
    s0 = nfa.closure([start])
    ids = {s0: 0}
    sets = [s0]
    moves: list[list[int]] = []
    q = deque([s0])
    while q:
        cur = q.popleft()
        targets: list[list[int]] = [[] for _ in classes]
        for s in cur:
            for cis, t in covers[s]:
                for ci in cis:
                    targets[ci].append(t)
        row = []
        for tgt in targets:
            if not tgt:
                row.append(DEAD)
                continue
            key = frozenset(tgt)
            if key not in closures:
                closures[key] = nfa.closure(key)
            nxt = closures[key]
            if nxt not in ids:
                ids[nxt] = len(sets)
                sets.append(nxt)
                q.append(nxt)
            row.append(ids[nxt])
        moves.append(row)
    acc = [final in s for s in sets]
    n = len(sets)

    # Live states: those that can still reach an accepting state.
    rev: list[list[int]] = [[] for _ in range(n)]
    for s, row in enumerate(moves):
        for t in row:
            if t != DEAD:
                rev[t].append(s)
    live = [False] * n
    stack = [s for s in range(n) if acc[s]]
    for s in stack:
        live[s] = True
    while stack:
        for p in rev[stack.pop()]:
            if not live[p]:
                live[p] = True
                stack.append(p)
    if not live[0]:
        raise ValueError(f"regex: {pattern!r} matches nothing")
    moves = [[t if t != DEAD and live[t] else DEAD for t in row] for row in moves]

    # Moore minimization over the live states: split blocks until every
    # state of a block goes to the same block on every class.
    alive = [s for s in range(n) if live[s]]
    block = {s: int(acc[s]) for s in alive}
    while True:
        sig = {
            s: (block[s],) + tuple(block[t] if t != DEAD else -1 for t in moves[s])
            for s in alive
        }
        keys = {k: i for i, k in enumerate(sorted(set(sig.values())))}
        nb = {s: keys[sig[s]] for s in alive}
        if len(keys) == len(set(block.values())):
            block = nb
            break
        block = nb

    # Canonical numbering: breadth-first from the start block, bytes ascending.
    byte_class = [0] * 256
    for ci, c in enumerate(classes):
        for b in range(256):
            if c >> b & 1:
                byte_class[b] = ci
    rep_of = {}
    for s in alive:
        rep_of.setdefault(block[s], s)
    order = {block[0]: 0}
    q = deque([block[0]])
    rows = []
    while q:
        b0 = q.popleft()
        s = rep_of[b0]
        row = []
        for byte in range(256):
            t = moves[s][byte_class[byte]]
            if t == DEAD:
                row.append(DEAD)
                continue
            bt = block[t]
            if bt not in order:
                order[bt] = len(order)
                q.append(bt)
            row.append(order[bt])
        rows.append(row)
    accept = [False] * len(order)
    for b, i in order.items():
        accept[i] = acc[rep_of[b]]
    return DFA(np.array(rows, dtype=np.int32), np.array(accept, dtype=bool))


# -- JSON schema subset -------------------------------------------------------------

# One character inside a JSON string: printable ASCII except '"' and '\\', or
# a well-formed UTF-8 sequence of 2 to 4 bytes (RFC 3629, table 3-7 of the
# Unicode standard: no overlong forms, no surrogates, nothing above U+10FFFF).
UTF8_CHAR = (
    r"[\x20\x21\x23-\x5b\x5d-\x7f]|[\xc2-\xdf][\x80-\xbf]|\xe0[\xa0-\xbf][\x80-\xbf]"
    r"|[\xe1-\xec\xee\xef][\x80-\xbf]{2}|\xed[\x80-\x9f][\x80-\xbf]|\xf0[\x90-\xbf][\x80-\xbf]{2}"
    r"|[\xf1-\xf3][\x80-\xbf]{3}|\xf4[\x80-\x8f][\x80-\xbf]{2}"
)
JSON_STRING = r'"(?:' + UTF8_CHAR + r'|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*"'
JSON_INTEGER = r"-?(?:0|[1-9][0-9]*)"
JSON_NUMBER = r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
_IGNORED = {"title", "description", "default"}


def _escape(text: str) -> str:
    """A regex matching exactly the UTF-8 bytes of text."""
    out = []
    for b in text.encode("utf-8"):
        c = chr(b)
        if c.isalnum() and b < 0x80:
            out.append(c)
        elif b < 0x80 and c in LITERAL_ESC:
            out.append("\\" + c)
        else:
            out.append(f"\\x{b:02x}")
    return "".join(out)


def json_schema_to_regex(schema: dict[str, Any]) -> str:
    if not isinstance(schema, dict):
        raise ValueError(f"schema must be a dict, got {type(schema).__name__}")
    keys = set(schema) - _IGNORED

    def only(*allowed: str) -> None:
        extra = sorted(keys - set(allowed))
        if extra:
            raise ValueError(f"JSON schema keyword {extra[0]!r} is not in the subset")

    if "enum" in schema or "const" in schema:
        only("enum", "const", "type")
        values = schema["enum"] if "enum" in schema else [schema["const"]]
        if not values:
            raise ValueError("enum must not be empty")
        alts = [
            _escape(json.dumps(v, separators=(",", ":"), ensure_ascii=False))
            for v in values
        ]
        return "(?:" + "|".join(alts) + ")"
    if "anyOf" in schema:
        only("anyOf")
        subs = schema["anyOf"]
        if not subs:
            raise ValueError("anyOf must not be empty")
        return "(?:" + "|".join(json_schema_to_regex(s) for s in subs) + ")"
    t = schema.get("type")
    if t == "string":
        only("type")
        return JSON_STRING
    if t == "integer":
        only("type")
        return JSON_INTEGER
    if t == "number":
        only("type")
        return JSON_NUMBER
    if t == "boolean":
        only("type")
        return "(?:true|false)"
    if t == "null":
        only("type")
        return "null"
    if t == "array":
        only("type", "items", "minItems", "maxItems")
        item = json_schema_to_regex(schema.get("items", {}))
        lo = int(schema.get("minItems", 0))
        hi = schema.get("maxItems")
        hi = None if hi is None else int(hi)
        if lo < 0 or (hi is not None and hi < lo):
            raise ValueError(f"bad array bounds minItems={lo} maxItems={hi}")
        if hi == 0:
            return r"\[\]"
        tail_hi = "" if hi is None else str(hi - 1)
        tail_lo = max(lo - 1, 0)
        body = f"{item}(?:,{item}){{{tail_lo},{tail_hi}}}"
        return rf"\[{body}\]" if lo >= 1 else rf"\[(?:{body})?\]"
    if t == "object":
        only("type", "properties", "required", "additionalProperties")
        if schema.get("additionalProperties", False) is not False:
            raise ValueError(
                "additionalProperties must be false (or absent) in the subset"
            )
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        unknown = required - set(props)
        if unknown:
            raise ValueError(f"required names unknown properties {sorted(unknown)}")
        members = [
            (
                _escape(json.dumps(k, ensure_ascii=False))
                + ":"
                + json_schema_to_regex(v),
                k in required,
            )
            for k, v in props.items()
        ]
        # Built from the end. rest_after[i]: members i.. once something has
        # been written (each needs a leading comma); rest_first[i]: members
        # i.. when nothing has been written yet.
        after, first = "", ""
        for m, req in reversed(members):
            first = f"{m}{after}" if req else f"(?:{m}{after}|{first})"
            after = f",{m}{after}" if req else f"(?:,{m})?{after}"
        return r"\{" + first + r"\}"
    raise ValueError(f"JSON schema type {t!r} is not in the subset")


# -- 4. token masks -------------------------------------------------------------------


class TokenIndex:
    def __init__(
        self, dfa: DFA, vocab: Sequence[Optional[bytes]], eos_id: Optional[int] = None
    ) -> None:
        self.dfa = dfa
        self.vocab = list(vocab)
        self.vocab_size = len(self.vocab)
        if eos_id is not None and not 0 <= eos_id < self.vocab_size:
            raise ValueError(f"eos_id {eos_id} outside [0, {self.vocab_size})")
        self.eos_id = eos_id
        # A byte trie: node = (children: dict byte -> node, token ids ending here).
        self._root: tuple[dict, list[int]] = ({}, [])
        for i, tb in enumerate(self.vocab):
            if not tb or i == eos_id:
                continue  # specials (None), empty tokens, and EOS never walk
            node = self._root
            for b in tb:
                node = node[0].setdefault(b, ({}, []))
            node[1].append(i)
        self._masks: dict[int, NDArray] = {}
        self._next: dict[int, NDArray] = {}

    def _build(self, state: int) -> None:
        mask = np.zeros(self.vocab_size, dtype=bool)
        nxt = np.full(self.vocab_size, DEAD, dtype=np.int64)
        trans = self.dfa.trans
        stack = [(self._root, state)]
        while stack:
            (children, _), s = stack.pop()
            for b, child in children.items():
                t = int(trans[s, b])
                if t == DEAD:
                    continue  # every token below this byte is dead too
                for tid in child[1]:
                    mask[tid] = True
                    nxt[tid] = t
                stack.append((child, t))
        if self.eos_id is not None and self.dfa.accept[state]:
            mask[self.eos_id] = True
            nxt[self.eos_id] = state
        self._masks[state] = mask
        self._next[state] = nxt

    def mask(self, state: int) -> NDArray:
        if not 0 <= state < self.dfa.n_states:
            raise ValueError(f"state {state} outside [0, {self.dfa.n_states})")
        if state not in self._masks:
            self._build(state)
        return self._masks[state].copy()

    def next_state(self, state: int, token_id: int) -> int:
        if not 0 <= state < self.dfa.n_states:
            raise ValueError(f"state {state} outside [0, {self.dfa.n_states})")
        if not 0 <= token_id < self.vocab_size:
            raise ValueError(f"token {token_id} outside [0, {self.vocab_size})")
        if state not in self._next:
            self._build(state)
        return int(self._next[state][token_id])


class Constraint:
    def __init__(self, index: TokenIndex) -> None:
        self.index = index
        self.state = 0
        self.done = False

    def mask(self) -> NDArray:
        if self.done:
            return np.zeros(self.index.vocab_size, dtype=bool)
        return self.index.mask(self.state)

    def advance(self, token_id: int) -> None:
        if self.done:
            raise ValueError("the constraint is done (EOS was emitted)")
        nxt = self.index.next_state(self.state, int(token_id))
        if nxt == DEAD:
            raise ValueError(f"token {token_id} is not allowed in state {self.state}")
        if token_id == self.index.eos_id:
            self.done = True
        self.state = nxt

    def is_complete(self) -> bool:
        return bool(self.index.dfa.accept[self.state])


def apply_mask(logits: ArrayLike, mask: ArrayLike) -> NDArray:
    out = np.array(logits, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if out.ndim != 1 or m.shape != out.shape:
        raise ValueError(
            f"logits {out.shape} and mask {m.shape} must be the same 1-D shape"
        )
    if not m.any():
        raise ValueError("the mask allows no token")
    out[~m] = -np.inf
    return out


def vocab_bytes(tok: Any) -> list[Optional[bytes]]:
    byte_of = unicode_to_bytes()
    specials = set(getattr(tok, "special_ids", {}).values())
    out: list[Optional[bytes]] = []
    for i in range(tok.vocab_size):
        if i in specials:
            out.append(None)
            continue
        s = tok.id_to_token(i)
        if s and all(c in byte_of for c in s):
            out.append(bytes(byte_of[c] for c in s))
        else:
            out.append(s.encode("utf-8"))
    return out


def constrained_sample(
    logits: ArrayLike,
    c: Constraint,
    p: SamplingParams,
    history: Sequence[int],
    rng: UniformSource,
    prompt: Sequence[int] = (),
) -> tuple[int, float]:
    masked = apply_mask(logits, c.mask())
    tok, lp = sample(masked, p, history, rng, prompt)
    c.advance(tok)
    return tok, lp
