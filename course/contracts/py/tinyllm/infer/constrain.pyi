# contracts/py/tinyllm/infer/constrain.pyi (L8.7)
# chapter: content/courses/ai-engineering/21-inference/07-constrained-decoding.md
#
# Constrained decoding: the model may only emit tokens that keep the output a
# prefix of some string in a regular language. A regex (or a JSON schema,
# translated to a regex) becomes a deterministic finite automaton (DFA) over
# BYTES; every token is a byte string; a token is allowed in DFA state s
# exactly when walking its bytes from s never leaves the live states (states
# from which an accepting state is still reachable). The sampler (L8.1) then
# draws from the logits with every disallowed token set to -inf.
#
# Regex subset (a whole-string match, like re.fullmatch on bytes): literal
# characters (ASCII); escapes \\ \. \* \+ \? \( \) \[ \] \{ \} \| \^ \$ \/ \"
# \- \, \: and \n \t \r \f \v \xHH; classes \d \w \s (as in Python's bytes
# regexes: [0-9], [A-Za-z0-9_], [ \t\n\r\f\v]) and their negations \D \W \S;
# `.` (any byte except \n); bracket classes [a-z_] and [^...] (complement
# over the 256 byte values) with ranges and the escapes above; groups (...)
# and (?:...); alternation |; quantifiers * + ? {m} {m,} {m,n} (m <= n <=
# 256). Non-ASCII characters in a pattern are a ValueError (write \xHH), as
# are anchors, backreferences, lookarounds, lazy quantifiers, and any other
# syntax.
#
# The DFA is canonical: dead states removed, minimized (Moore), and numbered
# in breadth-first order from the start state (0) taking bytes 0..255 in
# ascending order. Two patterns with the same language have the same DFA, so
# the Rust port (L10.9) can compare transition tables and masks directly.
from typing import Any, Optional, Sequence

from numpy.typing import ArrayLike, NDArray

from tinyllm.infer.sample import SamplingParams
from tinyllm.prob.sampling import UniformSource

DEAD: int  # -1: the "state" after a byte with no live continuation

class DFA:
    n_states: int
    accept: NDArray  # bool [n_states]
    trans: NDArray  # int32 [n_states, 256]; DEAD where no live continuation

    def step(self, state: int, data: bytes) -> int:
        """The state after reading data from state; DEAD once a byte has no
        live continuation (and DEAD stays DEAD). ValueError for a state
        outside [0, n_states) other than DEAD."""

    def matches(self, data: bytes) -> bool:
        """True iff data is in the language (step from 0 ends accepting)."""

def regex_to_dfa(pattern: str) -> DFA:
    """Parse the subset above, build a Thompson NFA, determinize it over
    byte classes (subset construction), drop dead states, minimize, number
    canonically. ValueError for syntax outside the subset or a pattern whose
    language is empty."""

def json_schema_to_regex(schema: dict[str, Any]) -> str:
    """A regex (in the subset above) for the compact JSON texts (no
    whitespace, as json.dumps(v, separators=(",", ":")) writes them) that
    are valid under a JSON-schema subset:

      {"type": "string"}     a JSON string: characters as well-formed UTF-8
                             (RFC 3629) except '"', '\\' and control bytes
                             < 0x20, or an escape \\" \\\\ \\/ \\b \\f \\n
                             \\r \\t \\uXXXX
      {"type": "integer"}    -?(0|[1-9][0-9]*)
      {"type": "number"}     an integer, then optional .digits and exponent
      {"type": "boolean"}    true|false;  {"type": "null"}  null
      {"enum": [v, ...]}     exactly one of the values (any JSON values),
                             each as compact JSON; {"const": v} = {"enum": [v]}
      {"type": "array", "items": S, "minItems": a, "maxItems": b}
                             [] or [S,S,...] with a..b elements (a defaults
                             to 0, b to unbounded)
      {"type": "object", "properties": {name: S, ...}, "required": [...]}
                             the properties in the order given, each once;
                             the required ones always, the others optional;
                             no other keys ("additionalProperties" may only
                             be false)
      {"anyOf": [S, ...]}    any one of the subschemas

    "title", "description", and "default" are ignored. Any other keyword
    ("pattern", "format", "$ref", "minLength", ...) is a ValueError naming
    it, as is a schema that is not a dict."""

class TokenIndex:
    """Which tokens each DFA state allows, and where each one leads.
    vocab[i] is token i's bytes; None marks a special token, which is never
    allowed (nor is an empty token). eos_id, when given, is allowed exactly
    in accepting states. Masks are computed on first use per state and
    cached."""

    dfa: DFA
    vocab_size: int
    eos_id: Optional[int]

    def __init__(
        self, dfa: DFA, vocab: Sequence[Optional[bytes]], eos_id: Optional[int] = None
    ) -> None:
        """ValueError for an eos_id outside [0, len(vocab))."""

    def mask(self, state: int) -> NDArray:
        """bool [vocab_size]: token t is allowed iff dfa.step(state,
        vocab[t]) != DEAD (or t == eos_id and state is accepting).
        ValueError for a state outside [0, n_states)."""

    def next_state(self, state: int, token_id: int) -> int:
        """dfa.step(state, vocab[token_id]); for eos_id, the state itself
        when accepting, else DEAD; DEAD for a special token."""

class Constraint:
    """One request's place in the language. Starts in state 0."""

    index: TokenIndex
    state: int
    done: bool  # eos_id was emitted

    def __init__(self, index: TokenIndex) -> None: ...
    def mask(self) -> NDArray:
        """index.mask(state); all False once done."""
    def advance(self, token_id: int) -> None:
        """Move past an allowed token (eos_id sets done). ValueError for a
        token the current mask does not allow, or after done."""
    def is_complete(self) -> bool:
        """The output so far is a whole string of the language."""

def apply_mask(logits: ArrayLike, mask: ArrayLike) -> NDArray:
    """float64 copy of 1-D logits with -inf wherever mask is False.
    ValueError for mismatched shapes or a mask that allows nothing."""

def vocab_bytes(tok: Any) -> list[Optional[bytes]]:
    """Each id's bytes for a tokenizer with the L1.1 protocol: ids in
    tok.special_ids give None; a token string made only of the GPT-2 byte
    map's characters (M05.2, byte-level BPE and the byte tokenizer) gives
    the bytes it stands for; any other token string gives its UTF-8
    encoding."""

def constrained_sample(
    logits: ArrayLike,
    c: Constraint,
    p: SamplingParams,
    history: Sequence[int],
    rng: UniformSource,
    prompt: Sequence[int] = (),
) -> tuple[int, float]:
    """L8.1's sample on apply_mask(logits, c.mask()), then c.advance(id).
    The logprob is the masked distribution's (renormalized over the allowed
    tokens, before temperature, as L8.1 defines it). Greedy picks the
    largest allowed logit, ties to the lowest id."""
