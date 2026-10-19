# Constrained decoding: from regex to token masks

Agents call tools. The model must emit `{"city": "Oslo", "days": 3}` for a function whose arguments are described by a JSON schema, and the gateway hands those arguments to code that expects exactly that shape. A 135M-parameter model left alone produces almost-JSON: a missing quote, a trailing comma, `"days": "three"`. Retrying until it parses wastes passes and still fails sometimes.

Constrained decoding makes invalid output impossible instead. Before each sampling step, every token that would take the text outside the allowed language gets logit $-\infty$, and [the sampler](01-sampling-and-logit-processors.md) never draws it. The serving engine will do this in Rust, per request, at serving speed ([tool calls and constrained decoding](../11-serving/10-tool-calls-and-constrained-decoding.md)); this section builds the semantics in Python, from regex to masks, and fixes the masks the Rust port must match. On a real model, every generated tool-call document then parses and validates.

You will need `sample` on masked logits from [sampling](01-sampling-and-logit-processors.md), and `unicode_to_bytes` from [the GPT-2 byte map](../../mathematics-for-computing/06-discrete-math-1/02-injective-surjective-bijective-gpt2-byte-map.md) in *Mathematics for Computing*; [graphs and topological sort](../../mathematics-for-computing/07-discrete-math-2/01-graphs-dags-and-topological-sort.md) is useful background for reachability and breadth-first search. Two oracles are worth keeping in mind as you read: Python's `re` is an exact oracle for matching, and walking each token's bytes through the DFA is the definition of a mask. The solution is [`constrain.py`](../../../../course/solutions/python/tinyllm/infer/constrain.py); the contract is [`constrain.pyi`](../../../../course/contracts/py/tinyllm/infer/constrain.pyi).

> **Learning objectives**
> - Build an NFA from a regex with Thompson's construction and turn it into a DFA with the subset construction.
> - Make the DFA canonical with liveness, Moore minimization, and breadth-first numbering.
> - Compute a state's token mask with a byte trie, and explain why no reachable state has an empty mask.
> - Compile a JSON schema subset into a regex, refusing any keyword outside it.

## Notation

| Symbol | Meaning | Type / shape |
|---|---|---|
| $\Sigma$ | the alphabet: the 256 byte values | |
| $L$ | the language: a set of byte strings | |
| $N$ | an NFA: states, $\varepsilon$-edges, and edges labeled by byte sets | |
| $D = (S, s_0, \delta, F)$ | a DFA: states $S$, start $s_0 = 0$, transition $\delta(s, b)$, accepting states $F$ | `trans: int32 [S, 256]`, `accept: bool [S]` |
| $\varnothing$ | the dead state: $\delta(s, b) = $ `DEAD` ($-1$) when no string of $L$ continues | |
| $t$, $\beta(t)$ | a token id and its bytes | int, `bytes` |
| $\delta^*(s, w)$ | the state after reading the string $w$ from $s$ | |
| $m_s$ | the mask in state $s$: $m_s[t] = (\delta^*(s, \beta(t)) \ne \varnothing)$ | `bool [V]` |

## From regex to NFA

A regular expression is a finite automaton in disguise. A regex is parsed into a tree: byte sets (a literal, a class `[a-z]`, `.`, `\d`), concatenation, alternation `|`, and repetition (`*`, `+`, `?`, `{m,n}`). **Thompson's construction** turns each node into an NFA fragment with one entry and one exit, joined by $\varepsilon$-edges (edges that consume nothing):

- a byte set is two states and one labeled edge;
- a concatenation chains fragments;
- an alternation adds a new entry with $\varepsilon$-edges into each branch and a new exit they all reach;
- `x*` adds a loop state that can enter $x$ again or leave;
- `x{m,n}` is $m$ copies of $x$, then $n - m$ optional copies that may each skip to the end.

The **$\varepsilon$-closure** of a set of states is everything reachable from it through $\varepsilon$-edges alone, a graph search.

The regex subset is Python's own (for bytes patterns): `\d` is `[0-9]`, `\w` is `[A-Za-z0-9_]`, `\s` is `[ \t\n\r\f\v]`, `.` is any byte except `\n`, and `[^...]` is the complement over all 256 bytes. Python's `re.fullmatch` on bytes is therefore an exact oracle.

## From NFA to a canonical DFA

The NFA can be in several states at once; the **subset construction** makes that set the DFA state. Start from the closure of the NFA's entry. For each DFA state $T$ and byte $b$, the next state is the closure of every NFA state reachable from $T$ by one edge whose set contains $b$. An empty set means no string continues: that is $\varnothing$, stored as `DEAD`. Looping over 256 bytes per state is wasteful, so first partition the bytes into **classes** that every edge treats alike (each edge set is a union of classes) and loop over classes. A DFA state is **accepting** when its set holds the NFA's exit.

Three steps then make the result canonical:

1. **Liveness.** A state from which no accepting state is reachable is dead; transitions into it become `DEAD` (a backward graph search from the accepting states). A Thompson NFA never builds such a state except the empty set, but an implementation that keeps the empty set as a "sink" state must remove it, or masks will lead into it.
2. **Minimization (Moore).** Start with two blocks, accepting and not; repeatedly split a block whose states go to different blocks on some class; stop when nothing splits. States in one block accept the same futures and merge.
3. **Numbering.** Breadth first from the start block, bytes in ascending order: the first block reached gets the next number.

The minimal DFA of a regular language is unique up to renaming, and the numbering removes the renaming. So `trans` and `accept` are a function of the language alone: `a|b` and `[ab]` give the same arrays. That is what lets the Rust port compare tables.

## From DFA to token masks

The DFA reads bytes, but the model emits tokens. A tokenizer's tokens are byte strings, and `vocab_bytes` gets them back from token strings. Byte-level BPE writes bytes in the GPT-2 byte map (byte 32 is the character `Ġ`), so a token string made only of map characters is decoded through `unicode_to_bytes`; any other token string (a WordPiece or char tokenizer's) is its UTF-8 encoding; special tokens get `None` and are never allowed, nor is an empty token. Then

$$m_s[t] = \big(\delta^*(s, \beta(t)) \ne \varnothing\big), \qquad m_s[\text{eos}] = (s \in F).$$

A token is allowed if it keeps the text a **prefix** of some string of $L$; it need not complete one. Computing $m_s$ token by token costs the total length of the vocabulary per state. A **trie** of the token bytes shares the work: walk it depth first from $s$, and when a byte is dead, skip the whole subtree under it, since every token there starts with that dead prefix. Masks are computed for a state the first time it is needed and cached.

Masks never strand the sampler. After the liveness pass, every state either accepts or has a byte that leads to a live state. With every single byte in the vocabulary (byte-level BPE has all 256) and an EOS id, every reachable state allows at least one token, so constrained generation can always continue or stop, and every finished output is in the language.

## Sampling under a mask

`apply_mask` sets disallowed logits to $-\infty$ (in float64), and `sample` already treats $-\infty$ as a token it can never draw. Greedy picks the largest **allowed** logit (ties to the lowest id); the logprob is the sampler's: the log-softmax over the allowed tokens. `constrained_sample` samples and then advances the constraint, which refuses any token outside the mask: if the engine ever samples one, that is a bug, and it must surface.

## A JSON schema subset as a regex

The last piece connects schemas to the pipeline. Compact JSON (no whitespace, as `json.dumps(v, separators=(",", ":"))` writes it) for a schema subset is a regular language:

- **string**: a quote, then characters, then a quote. A character is printable ASCII except `"` and `\`, or a **well-formed UTF-8 sequence** of 2 to 4 bytes (lead byte `C2` to `DF` plus one continuation byte `80` to `BF`; `E0` then `A0` to `BF`; ... per RFC 3629, which excludes overlong forms and surrogates), or an escape `\" \\ \/ \b \f \n \r \t \uXXXX`. Control bytes below `0x20` are not allowed raw. Requiring well-formed UTF-8 matters because the model emits bytes: a byte-level vocabulary has tokens that are half a character.
- **integer** `-?(0|[1-9][0-9]*)` (no leading zeros); **number** adds `(\.[0-9]+)?([eE][+-]?[0-9]+)?`; **boolean** `true|false`; **null**.
- **enum / const**: an alternation of the values' compact JSON.
- **array** with `items`, `minItems` $a$, `maxItems` $b$: `\[` item `(,` item`){a-1,b-1}` `\]`, all of it optional when $a = 0$.
- **object** with ordered `properties` and `required`: each property at most once, in the order given, the required ones always. It is built from the end so the commas come out right: "the rest, after something was written" is `(,m_i)?` (optional) or `,m_i` (required) followed by the rest; "the rest, when nothing was written yet" is $m_i$ followed by the first form of the rest, or (when optional) the second form skipping $m_i$. The size grows with the square of the property count, not exponentially.
- **anyOf**: an alternation.

Anything else (`pattern`, `minLength`, `$ref`, `additionalProperties: true`, ...) raises `ValueError`. Ignoring an unknown keyword would let the model emit output the schema forbids, and the caller would never know.

## Working it by hand

Take the pattern `(cat|car|dog)s?`. After the subset construction and minimization, the DFA, numbered breadth first with bytes ascending, is:

| State | Meaning (text so far) | Accepting | Transitions |
|---|---|---|---|
| 0 | "" | no | `c` to 1, `d` to 2 |
| 1 | "c" | no | `a` to 3 |
| 2 | "d" | no | `o` to 4 |
| 3 | "ca" | no | `r` to 5, `t` to 5 |
| 4 | "do" | no | `g` to 5 |
| 5 | "cat", "car", "dog" | yes | `s` to 6 |
| 6 | "cats", "cars", "dogs" | yes | none |

"cat", "car", and "dog" reach the same state because they have the same futures ("" or "s"): that is what minimization merged. From state 0 the first byte `c` is seen before `d`, so "c" gets 1.

Now the masks, for the vocabulary `c a t r d o g s ca cat dog ts x <eos> ""` (ids 0 to 14; EOS is 13; the last token is empty):

| State | Allowed ids | Why |
|---|---|---|
| 0 | 0 `c`, 4 `d`, 8 `ca`, 9 `cat`, 10 `dog` | every token that starts a word |
| 3 | 2 `t`, 3 `r`, 11 `ts` | `ts` reads t (to 5) then s (to 6): still alive |
| 5 | 7 `s`, 13 EOS | accepting, so EOS too |
| 6 | 13 EOS | nothing can follow |

`x` is never allowed, nor the empty token.

## Designing the API

```python
DEAD = -1
class DFA:  n_states; accept: bool[S]; trans: int32[S, 256]; def step(state, data) -> int; def matches(data) -> bool
def regex_to_dfa(pattern: str) -> DFA
def json_schema_to_regex(schema: dict) -> str
class TokenIndex:  def __init__(dfa, vocab: Sequence[bytes | None], eos_id=None); def mask(state) -> bool[V]; def next_state(state, token_id) -> int
class Constraint:  state; done; def mask() -> bool[V]; def advance(token_id) -> None; def is_complete() -> bool
def apply_mask(logits, mask) -> float64[V]
def vocab_bytes(tok) -> list[bytes | None]
def constrained_sample(logits, c: Constraint, p: SamplingParams, history, rng, prompt=()) -> tuple[int, float]
```

## Building it step by step

### The reasoning

1. **The guarantee needed is "every finished output is in $L$, and the sampler is never stuck".** So at every step we need the set of tokens that keep the text a prefix of some string in $L$, plus EOS exactly when the text is in $L$.
2. **"Is this a prefix of something in $L$?" is a question about a state, not about the whole text.** For a regular language a DFA answers it in $O(1)$ per byte: the state after the text is all that matters, and a state is live iff an accepting state is reachable from it. So the constraint carries one integer.
3. **The DFA is built the classical way**: parse to a tree, Thompson NFA (each operator a fragment with one entry and one exit, easy to get right), then the subset construction. Working over byte classes instead of 256 bytes keeps the construction fast without changing the result.
4. **Dead states must be removed before masks are computed**, otherwise a token could lead into a state that can never accept, and the sampler would be stranded there.
5. **The Rust port must produce the same tables**, so the DFA must be canonical: minimise (unique up to renaming) and then number breadth first with bytes ascending (removes the renaming).
6. **Masks are a per-state function of the vocabulary.** Computing them by walking every token is correct but repeats work; a byte trie shares prefixes and prunes at the first dead byte. Cache per state, because generation revisits few states many times.
7. **JSON schemas in the subset compile to a regex**, so the same pipeline handles them; anything outside the subset must be refused, not ignored.

### Byte sets and parsing

Byte sets are Python integers with bit $b$ set for byte $b$, so union, intersection, and complement over all 256 bytes are `|`, `&`, and `ALL & ~x`. The classes match Python's bytes semantics:

```python
ALL = (1 << 256) - 1                       # a byte set is an int with bit b set for byte b
DIGIT = _range(0x30, 0x39)
WORD = DIGIT | _range(0x41, 0x5A) | _range(0x61, 0x7A) | _bits([0x5F])   # \w includes _
DOT = ALL & ~_bits(b"\n")                  # . is any byte except newline
# and in bracket(): return (ALL & ~bits) if neg else bits   # [^...] over all 256 bytes
```

The parser is recursive descent (`alt` > `cat` > `repeat` > `atom`) and raises on anything outside the subset: `^` and `$` are `unexpected`; `(?=...)` and other `(?` groups except `(?:` are refused; a `?` after a quantifier is a lazy quantifier and refused; non-ASCII pattern characters must be written as `\xHH`.

### Thompson's construction

The repetition case is the subtle one:

```python
def build(self, node) -> tuple[int, int]:
    """A fragment (start, end) for node: end has no outgoing edges yet."""
    kind = node[0]
    if kind == "set":                       # two states, one labelled edge
        s, e = self.state(), self.state()
        self.edges[s].append((node[1], e))
        return s, e
    if kind == "cat":                       # chain fragments with epsilon edges
        s = e = self.state()
        for child in node[1]:
            cs, ce = self.build(child)
            self.eps[e].append(cs)
            e = ce
        return s, e
    if kind == "alt":                       # fan out from s, fan in to e
        s, e = self.state(), self.state()
        for child in node[1]:
            cs, ce = self.build(child)
            self.eps[s].append(cs)
            self.eps[ce].append(e)
        return s, e
    _, child, m, n = node                   # ("rep", child, m, n)
    s = e = self.state()
    for _ in range(m):                      # m mandatory copies
        cs, ce = self.build(child)
        self.eps[e].append(cs)
        e = ce
    if n is None:                           # then a loop: zero or more
        cs, ce = self.build(child)
        loop = self.state()
        self.eps[e].append(loop)
        self.eps[loop].append(cs)
        self.eps[ce].append(loop)
        return s, loop                      # x+ is one copy then the loop: never empty
    end = self.state()
    self.eps[e].append(end)                 # may stop after the m copies
    for _ in range(n - m):                  # n - m optional copies, each may stop early
        cs, ce = self.build(child)
        self.eps[e].append(cs)
        self.eps[ce].append(end)
        e = ce
    return s, end
```

### The subset construction

Byte classes refine the partition once per edge set, so every edge's set is a union of classes; the subset construction then loops over classes:

```python
s0 = nfa.closure([start])
ids = {s0: 0}; sets = [s0]; moves = []
q = deque([s0])
while q:
    cur = q.popleft()
    targets = [[] for _ in classes]
    for s in cur:                          # one edge step from any NFA state in cur
        for cis, t in covers[s]:
            for ci in cis:
                targets[ci].append(t)
    row = []
    for tgt in targets:
        if not tgt:
            row.append(DEAD)               # no NFA state continues: the dead state
            continue
        key = frozenset(tgt)
        if key not in closures:            # closures are memoised per target set
            closures[key] = nfa.closure(key)
        nxt = closures[key]
        if nxt not in ids:                 # a new DFA state
            ids[nxt] = len(sets); sets.append(nxt); q.append(nxt)
        row.append(ids[nxt])
    moves.append(row)
acc = [final in s for s in sets]           # accepting iff the set holds the NFA's exit
```

### Liveness, minimization, and numbering

Then comes liveness (a backward search from the accepting states over reversed edges; a pattern whose start is not live "matches nothing" and raises), transitions into non-live states rewritten to `DEAD`, and Moore minimisation:

```python
alive = [s for s in range(n) if live[s]]
block = {s: int(acc[s]) for s in alive}    # two blocks: accepting or not
while True:
    # A state's signature: its block and the blocks it moves to on each class.
    sig = {s: (block[s],) + tuple(block[t] if t != DEAD else -1 for t in moves[s])
           for s in alive}
    keys = {k: i for i, k in enumerate(sorted(set(sig.values())))}
    nb = {s: keys[sig[s]] for s in alive}
    if len(keys) == len(set(block.values())):   # nothing split: a fixed point
        block = nb
        break
    block = nb
```

Canonical numbering walks blocks breadth first from the start block, bytes `0..255` in order, assigning the next number to each new block reached; each row is filled per byte through `byte_class[byte]`, so the stored table is the full `[S, 256]` array. In the hand example, state 0 sees `c` (0x63) before `d` (0x64), so "c" becomes 1 and "d" 2.

### Masks

A trie of the vocabulary's bytes is walked depth first from the state:

```python
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
            for tid in child[1]:           # tokens ending at this trie node
                mask[tid] = True
                nxt[tid] = t               # the state after the whole token
            stack.append((child, t))       # keep walking longer tokens
    if self.eos_id is not None and self.dfa.accept[state]:
        mask[self.eos_id] = True           # EOS only where the text is complete
        nxt[self.eos_id] = state
    self._masks[state] = mask
    self._next[state] = nxt
```

The trie is built in `__init__` skipping `None` (specials), empty tokens, and EOS, so those never appear in a mask through the walk. `mask` and `next_state` build a state's arrays on first use and cache them; `mask` returns a copy so a caller cannot corrupt the cache.

### The constraint

The constraint is a state plus a done flag, and it refuses what the mask would not have allowed:

```python
def advance(self, token_id: int) -> None:
    if self.done:
        raise ValueError("the constraint is done (EOS was emitted)")
    nxt = self.index.next_state(self.state, int(token_id))
    if nxt == DEAD:                        # an engine bug: surface it
        raise ValueError(f"token {token_id} is not allowed in state {self.state}")
    if token_id == self.index.eos_id:
        self.done = True
    self.state = nxt
```

`constrained_sample` is three lines: `apply_mask` (a float64 copy with $-\infty$ outside the mask, refusing an all-false mask), `sample` on it, then `advance`.

### JSON schemas

The JSON object rule is built from the end:

```python
after, first = "", ""
for m, req in reversed(members):   # m is '"key":' + the value's regex
    # first: members i.. when nothing has been written yet
    first = f"{m}{after}" if req else f"(?:{m}{after}|{first})"
    # after: members i.. once something was written (each needs a comma)
    after = f",{m}{after}" if req else f"(?:,{m})?{after}"
return r"\{" + first + r"\}"
```

Each step copies `after` into `first` once, so the pattern grows with the square of the number of properties. Arrays are `item(?:,item){lo-1,hi-1}`, wrapped in `(?:...)?` when `minItems` is 0, and `\[\]` when `maxItems` is 0. Every schema branch calls `only(...)` with its allowed keywords and raises on any other (only `title`, `description`, and `default` are ignored, since they do not constrain values).

`vocab_bytes` decodes a token through the GPT-2 byte map when every character is in the map, otherwise encodes it as UTF-8, and gives specials `None`.

The full solution is [`constrain.py`](../../../../course/solutions/python/tinyllm/infer/constrain.py).

### Why this design

The subset construction can in principle produce $2^{|N|}$ states; for the patterns here (and JSON schemas) it stays small. Per DFA state the mask costs at most the trie's size, and usually much less thanks to pruning; per generation step it is a dictionary lookup once the state is cached. `advance` is $O(1)$.

The alternatives each lose something. Checking candidate tokens after sampling (sample, check, resample if invalid) changes the distribution in an uncontrolled way and can loop; masking before sampling is exactly "the model's distribution restricted to valid tokens". An NFA simulation at run time would avoid the subset construction but cost a set of states per byte per token; a DFA is one table lookup per byte and has a canonical form to compare across languages. A full JSON grammar (a pushdown automaton) handles nested and recursive schemas, which need a CFG; the regular subset covers tool-call arguments with bounded nesting, and a DFA keeps the state a single integer the Rust engine can store per request.

One tradeoff is accepted: compact JSON only (no whitespace), properties in schema order, and `{m,n}` bounded at 256. These rule out outputs the model might have liked, in exchange for a small DFA and a mask that is exact.

## What can go wrong

A correct solution must handle these cases.

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| The hand-example DFA | unit, smoke | 7 states, the accepting ones, every edge; `step` and `matches` | the canonical form is pinned by a table you can check by hand |
| The hand-example masks | unit, smoke | the masks; `next_state` for tokens, EOS, dead tokens | the mask semantics |
| Agreement with Python's `re` | golden | 10 patterns: random strings from the DFA, their prefixes, and one-byte edits get the same answer from Python's `re` | the regex subset means what Python means |
| Every state is live | property | from every state an accepting state is reachable | masks never lead into a dead end |
| Canonical form | property | five pairs of equivalent patterns give identical `trans` and `accept` | the Rust engine compares tables with these |
| Unsupported syntax raises | boundary | anchors, backreferences, lookarounds, lazy quantifiers, non-ASCII, bad repeats, empty language | no silent misreading |
| Trie masks equal brute force | property | every state and token: the trie mask equals walking the token's bytes | the trie is only a speedup |
| Masks along recorded paths | golden | masks and completeness along recorded paths, from an independent oracle using regex derivatives, not automata | the parity data the Rust engine is held to |
| No reachable state has an empty mask | property | with all bytes and EOS, every state reachable by tokens allows something | generation never gets stuck |
| The constraint's lifecycle | unit | advancing, refusing a disallowed token, EOS only when complete, nothing after EOS | engine bugs surface |
| `apply_mask` | boundary | float64 copy with $-\infty$; an all-false mask and a wrong shape are errors | the sampler never sees an all $-\infty$ row |
| Greedy takes the best allowed token | unit | the largest allowed logit wins (ties lowest); the logprob is renormalized | greedy tool calls |
| Constrained generation parses | property | 40 seeded requests under a tool schema: at least 30 finish, every finished one parses and validates | the engine's tool-call guarantee |
| JSON schema documents | unit | valid documents match; missing keys, wrong types, order, extra keys, trailing commas, spaces, bounds, raw control bytes do not | the schema subset's meaning |
| Keywords outside the subset are refused | boundary | `pattern`, `maxLength`, `$ref`, `additionalProperties: true`, unknown required names, unknown types | no silently ignored constraints |
| `vocab_bytes` | unit | the byte map undone; non-map strings as UTF-8; specials as `None` | masks over the real tokenizer |

Most failures come from small disagreements with Python's semantics or off-by-one repetition. This table lists them.

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Classes that differ from Python's: `.` matching `\n`, `[^...]` complemented over ASCII only, `\w` without `_`, ranges missing their upper end | strings `re` accepts are refused, or the reverse | `DOT = ALL & ~\n`, complement against `ALL` (256 bits), `WORD` includes `0x5F`, `_range(lo, hi)` is inclusive |
| No minimization | equivalent patterns give different tables; the hand example has more than 7 states | Moore refinement until no block splits |
| Numbering in another order | tables that do not match the Rust port's | breadth first from the start block, bytes ascending |
| EOS allowed everywhere, or a dead sink kept as a state | outputs stop mid-word; masks lead into a state with no way out | EOS only in accepting states; the liveness pass rewrites non-live targets to `DEAD` |
| The trie walk stopping at the first dead byte, or recording the wrong next state | allowed tokens missing; the constraint drifts from the text | skips only that byte's subtree (`continue`), records `t` after the token's last byte |
| Repetition built wrong: one optional copy too few, `+` as `*` | `{2,3}` accepts only 2; `x+` accepts "" | $m$ mandatory copies, then exactly $n - m$ optional copies or one loop |
| Anchors read as literal characters | a pattern means something else than in Python | `^` and `$` are parse errors |
| JSON strings allowing raw control bytes or any high byte; integers with leading zeros; an exponent without digits | output that `json.loads` rejects | `UTF8_CHAR` starts at `\x20` and lists only well-formed sequences; `0\|[1-9][0-9]*`; `[eE][+-]?[0-9]+` |
| Optional properties made required after the first member | valid tool calls without optional arguments refused | the `after` form wraps each optional member in `(?:,m)?` |
| `maxItems` off by one | one item too many accepted | the tail repeats `{lo-1, hi-1}` after the first item |
| Unknown schema keywords ignored | output violates the schema silently | `only(...)` raises on any keyword outside the branch's list |
| Token strings encoded as UTF-8 without undoing the byte map; specials given bytes | masks over the wrong bytes: `Ġ` is two bytes, not a space | `vocab_bytes` maps through `unicode_to_bytes` and returns `None` for specials |

## Where this leads

In the serving engine, [tool calls and constrained decoding](../11-serving/10-tool-calls-and-constrained-decoding.md) turns a tool's JSON schema into a DFA, and the per-request masks must equal this solution's on recorded paths. (Graph reachability and breadth-first search come from *Mathematics for Computing*, though topological sort does not apply, because automata have cycles.)

Production libraries go further on every axis:

- Outlines (`outlines-core`, `src/index.rs`) builds the regex-to-mask index once per (regex, tokenizer): for each state, the token to next-state map, in Rust.
- XGrammar and llguidance (XGrammar paper, Dong et al., 2024; `guidance-ai/llguidance`) go beyond the JSON-schema subset to context-free grammars (nested objects, `$ref`, recursion) with a pushdown automaton, computing masks in microseconds per token with adaptive caches.
- vLLM's structured outputs (`v1/structured_output/`) extend the per-state cache with grammar compilation off the critical path and masks applied as a bitmask on the GPU.
- RE2 and Rust's `regex-automata` (its docs on the "hybrid" DFA) refine the subset construction with lazy DFA construction, memory bounds, and Unicode classes.

### Further reading

- Sipser, *Introduction to the Theory of Computation*, ch. 1 (finite automata, regular expressions).
- Thompson, *Regular Expression Search Algorithm* (CACM 1968).
- Willard and Louf, *Efficient Guided Generation for Large Language Models* (2023).

## Key terms

- **NFA**: a finite automaton that may be in several states at once, with $\varepsilon$-edges that consume nothing.
- **Thompson's construction**: building an NFA fragment with one entry and one exit per regex operator.
- **Subset construction**: turning an NFA into a DFA whose states are sets of NFA states.
- **Live state**: a DFA state from which some accepting state is reachable.
- **Moore minimization**: refining a partition of states until no block splits, merging states with the same futures.
- **Token mask**: the set of tokens whose bytes keep the DFA alive from the current state, plus EOS in accepting states.
- **Byte trie**: a prefix tree over the vocabulary's bytes, letting one walk serve every token with a shared prefix.

## Summary

- **A regular expression is a finite automaton in disguise**: Thompson's construction builds an NFA fragment per operator, and the subset construction turns it into a DFA whose state is "the set of NFA states we could be in".
- **Minimized and numbered breadth first, the DFA is canonical**: two patterns with the same language give identical tables, which is what lets the Rust port compare tables.
- **A token is allowed when walking its bytes keeps the DFA alive**: a trie over the vocabulary shares the walk between tokens with a common prefix, and EOS is allowed exactly in accepting states.
- **Every reachable state has a non-empty mask**, so constrained generation can always continue or stop, and every finished output is in the language.
- **A JSON schema subset is a regular language**: objects with ordered properties, arrays with bounds, enums, and strings of well-formed UTF-8 become one regex, and any keyword outside the subset is refused rather than ignored.

## Review questions

1. Why does `x+` not accept the empty string in this construction, while `x*` does?
   <details><summary>Answer</summary>`x+` is `("rep", x, 1, None)`: one mandatory copy, then the loop. The fragment's start reaches the loop only through that copy. `x*` has $m = 0$, so the start connects to the loop state directly, which is also the end.</details>
2. Why is the liveness pass needed if the subset construction already writes `DEAD` for empty sets?
   <details><summary>Answer</summary>A non-empty set of NFA states can still have no path to the exit. Without the pass, a mask would allow a token into such a state, and from there no token could ever complete the text. After the pass every state either accepts or has a live successor.</details>
3. Why do `a|b` and `[ab]` give byte-identical tables?
   <details><summary>Answer</summary>They denote the same language. The minimal DFA of a language is unique up to renaming states, and breadth-first numbering with bytes ascending fixes the names. So the tables depend on the language only.</details>
4. In state 3 of the worked example, why is `ts` allowed though "cats" is not a word plus `t`?
   <details><summary>Answer</summary>From "ca", `t` goes to state 5 ("cat") and `s` to state 6 ("cats"), both live. A token is allowed when its whole byte string keeps the text a prefix of some string in $L$; it may cross the boundary of what a human would call a word.</details>
5. Why must an unknown schema keyword raise instead of being ignored?
   <details><summary>Answer</summary>Ignoring `maxLength` or `pattern` would let the model emit values the schema forbids while every check here passes, and the tool receiving the arguments would get invalid input with no warning.</details>
