<!-- ol:module L8.7 -->
# Constrained decoding: regex to DFA to token masks, JSON-schema subset

## Overview

| | |
|---|---|
| **Module** | `L8.7` · build · Python · Pass 6 · 4 to 5 h |
| **You build** | `python/tinyllm/infer/constrain.py`: `regex_to_dfa` (parser, Thompson NFA, subset construction, minimization), `DFA`, `json_schema_to_regex`, `TokenIndex`, `Constraint`, `apply_mask`, `vocab_bytes`, `constrained_sample` |
| **Contract** | [`course/contracts/py/tinyllm/infer/constrain.pyi`](../../../course/contracts/py/tinyllm/infer/constrain.pyi) |
| **Tests** | `course/tests/L8.7/test_constrain.py`, with the golden masks `course/fixtures/L8.7/masks_golden.json` (what they check: section 4) |
| **Needs** | [`L8.1` sampling](01-sampling-and-logit-processors.md) (`sample` on masked logits), [`M05.2` the GPT-2 byte map](../../../math/05-discrete-math-1/02-injective-surjective-bijective-gpt2-byte-map.md) (`unicode_to_bytes`) (or `--ref-deps`). Reading: [`M06.1` graphs](../../../math/06-discrete-math-2/01-graphs-dags-and-topological-sort.md) |
| **Used by** | `L10.9` tool calls with constrained JSON in the Rust engine, held to the golden masks (joins `used_by` when registered) |
| **Milestone** | `MS-L8` (step `constrained-json`: every generated document parses and validates) |
| **Optional depth** | Sipser, *Introduction to the Theory of Computation*, ch. 1 (finite automata, regular expressions); Thompson, *Regular Expression Search Algorithm* (CACM 1968); Willard and Louf, *Efficient Guided Generation for Large Language Models* (2023) |

## Key Takeaways

- **A regular expression is a finite automaton in disguise**: Thompson's construction builds an NFA fragment per operator, and the subset construction turns it into a DFA whose state is "the set of NFA states we could be in" (`test_matches_python_re`).
- **Minimized and numbered breadth first, the DFA is canonical**: two patterns with the same language give identical tables, which is what lets the Rust port compare against yours (`test_canonical_form`).
- **A token is allowed when walking its bytes keeps the DFA alive**: a trie over the vocabulary shares the walk between tokens with a common prefix, and EOS is allowed exactly in accepting states (`test_masks_match_brute_force`, `test_hand_example_masks`).
- **Every reachable state has a non-empty mask**, so constrained generation can always continue or stop, and every finished output is in the language (`test_reachable_states_never_have_an_empty_mask`, `test_constrained_generation_parses`).
- **A JSON schema subset is a regular language**: objects with ordered properties, arrays with bounds, enums, and strings of well-formed UTF-8 become one regex, and any keyword outside the subset is refused rather than ignored (`test_json_schema_documents`, `test_json_schema_rejects_keywords_outside_the_subset`).

## How to work this chapter

```bash
ol start L8.7              # stubs constrain.py into your repo, contract alongside
ol tests L8.7              # read the test catalog first
ol check L8.7              # exit code is the verdict
ol check L8.7 --ref-deps   # only if you skipped L8.1 or M05.2
ol diff  L8.7              # after passing: your code against the reference
```

You also write graded tests (rung R5) in `python/tests/l8-7-constrain/`. Python's `re` is a fine oracle for matching; for masks, a brute-force walk of each token through your DFA is the definition.

---

## 1. Why now

Part 10's agents call tools: the model must emit `{"city": "Oslo", "days": 3}` for a function whose arguments are described by a JSON schema, and your gateway hands those arguments to code that expects exactly that shape. A 135M-parameter model left alone produces almost-JSON: a missing quote, a trailing comma, `"days": "three"`. Retrying until it parses wastes passes and still fails sometimes. Constrained decoding makes invalid output impossible instead: before each sampling step, every token that would take the text outside the allowed language gets logit $-\infty$, and L8.1's sampler never draws it. The engine (L10.9) will do this in Rust, per request, at serving speed; this module builds the semantics in Python, from regex to masks, and a golden file that the Rust port must match.

## 2. Principles

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

### 2.1 From regex to NFA

A regex is parsed into a tree: byte sets (a literal, a class `[a-z]`, `.`, `\d`), concatenation, alternation `|`, and repetition (`*`, `+`, `?`, `{m,n}`). **Thompson's construction** turns each node into an NFA fragment with one entry and one exit, joined by $\varepsilon$-edges (edges that consume nothing): a byte set is two states and one labeled edge; a concatenation chains fragments; an alternation adds a new entry with $\varepsilon$-edges into each branch and a new exit they all reach; `x*` adds a loop state that can enter $x$ again or leave; `x{m,n}` is $m$ copies of $x$, then $n - m$ optional copies that may each skip to the end. The **$\varepsilon$-closure** of a set of states is everything reachable from it through $\varepsilon$-edges alone, a graph search (M06.1).

The subset is Python's own (for bytes patterns): `\d` is `[0-9]`, `\w` is `[A-Za-z0-9_]`, `\s` is `[ \t\n\r\f\v]`, `.` is any byte except `\n`, and `[^...]` is the complement over all 256 bytes. Python's `re.fullmatch` on bytes is therefore an exact oracle, and the tests use it.

### 2.2 From NFA to a canonical DFA

The NFA can be in several states at once; the **subset construction** makes that set the DFA state. Start from the closure of the NFA's entry; for each DFA state $T$ and byte $b$, the next state is the closure of every NFA state reachable from $T$ by one edge whose set contains $b$. An empty set means no string continues: that is $\varnothing$, stored as `DEAD`. Looping over 256 bytes per state is wasteful, so first partition the bytes into **classes** that every edge treats alike (each edge set is a union of classes) and loop over classes. A DFA state is **accepting** when its set holds the NFA's exit.

Three steps make the result canonical:

1. **Liveness.** A state from which no accepting state is reachable is dead; transitions into it become `DEAD` (a backward graph search from the accepting states). A Thompson NFA never builds such a state except the empty set, but an implementation that keeps the empty set as a "sink" state must remove it, or masks will lead into it.
2. **Minimization (Moore).** Start with two blocks, accepting and not; repeatedly split a block whose states go to different blocks on some class; stop when nothing splits. States in one block accept the same futures and merge.
3. **Numbering.** Breadth first from the start block, bytes in ascending order: the first block reached gets the next number.

The minimal DFA of a regular language is unique up to renaming, and the numbering removes the renaming. So `trans` and `accept` are a function of the language alone: `a|b` and `[ab]` give the same arrays.

### 2.3 From DFA to token masks

A tokenizer's tokens are byte strings: `vocab_bytes` gets them back from token strings. Byte-level BPE writes bytes in M05.2's GPT-2 byte map (byte 32 is the character `Ġ`), so a token string made only of map characters is decoded through `unicode_to_bytes`; any other token string (a WordPiece or char tokenizer's) is its UTF-8 encoding; special tokens get `None` and are never allowed, nor is an empty token. Then

$$m_s[t] = \big(\delta^*(s, \beta(t)) \ne \varnothing\big), \qquad m_s[\text{eos}] = (s \in F).$$

A token is allowed if it keeps the text a **prefix** of some string of $L$; it need not complete one. Computing $m_s$ token by token costs the total length of the vocabulary per state. A **trie** of the token bytes shares the work: walk it depth first from $s$, and when a byte is dead, skip the whole subtree under it, since every token there starts with that dead prefix. Masks are computed for a state the first time it is needed and cached.

Why masks never strand the sampler: after the liveness pass, every state either accepts or has a byte that leads to a live state. With every single byte in the vocabulary (byte-level BPE has all 256) and an EOS id, every reachable state allows at least one token.

### 2.4 Sampling under a mask

`apply_mask` sets disallowed logits to $-\infty$ (in float64), and L8.1's `sample` already treats $-\infty$ as a token it can never draw. Greedy picks the largest **allowed** logit (ties to the lowest id); the logprob is L8.1's: the log-softmax over the allowed tokens. `constrained_sample` samples and then advances the constraint, which refuses any token outside the mask: if the engine ever samples one, that is a bug, and it must surface.

### 2.5 A JSON schema subset as a regex

Compact JSON (no whitespace, as `json.dumps(v, separators=(",", ":"))` writes it) for a schema subset is a regular language:

- **string**: a quote, then characters, then a quote. A character is printable ASCII except `"` and `\`, or a **well-formed UTF-8 sequence** of 2 to 4 bytes (lead byte `C2` to `DF` plus one continuation byte `80` to `BF`; `E0` then `A0` to `BF`; ... per RFC 3629, which excludes overlong forms and surrogates), or an escape `\" \\ \/ \b \f \n \r \t \uXXXX`. Control bytes below `0x20` are not allowed raw. Requiring well-formed UTF-8 matters because the model emits bytes: a byte-level vocabulary has tokens that are half a character.
- **integer** `-?(0|[1-9][0-9]*)` (no leading zeros), **number** adds `(\.[0-9]+)?([eE][+-]?[0-9]+)?`, **boolean** `true|false`, **null**.
- **enum / const**: an alternation of the values' compact JSON.
- **array** with `items`, `minItems` $a$, `maxItems` $b$: `\[` item `(,` item`){a-1,b-1}` `\]`, all of it optional when $a = 0$.
- **object** with ordered `properties` and `required`: each property at most once, in the order given, the required ones always. Built from the end so the commas come out right: "the rest, after something was written" is `(,m_i)?` (optional) or `,m_i` (required) followed by the rest; "the rest, when nothing was written yet" is `m_i` + the first form of the rest, or (when optional) the second form skipping $m_i$. The size grows with the square of the property count, not exponentially.
- **anyOf**: an alternation.

Anything else (`pattern`, `minLength`, `$ref`, `additionalProperties: true`, ...) raises `ValueError`. Ignoring an unknown keyword would let the model emit output the schema forbids, and the caller would never know.

## 3. Worked example by hand

The pattern `(cat|car|dog)s?`. After the subset construction and minimization, the DFA, numbered breadth first with bytes ascending:

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

Masks for the vocabulary `c a t r d o g s ca cat dog ts x <eos> ""` (ids 0 to 14, EOS is 13, the last token is empty):

| State | Allowed ids | Why |
|---|---|---|
| 0 | 0 `c`, 4 `d`, 8 `ca`, 9 `cat`, 10 `dog` | every token that starts a word |
| 3 | 2 `t`, 3 `r`, 11 `ts` | `ts` reads t (to 5) then s (to 6): still alive |
| 5 | 7 `s`, 13 EOS | accepting, so EOS too |
| 6 | 13 EOS | nothing can follow |

`x` is never allowed, nor the empty token. These two tables are `test_hand_example` and `test_hand_example_masks`.

## 4. The interface

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit, smoke | section 3's DFA: 7 states, the accepting ones, every edge; `step` and `matches` | you and the test agree on the canonical form |
| `test_hand_example_masks` | unit, smoke | section 3's masks; `next_state` for tokens, EOS, dead tokens | the mask semantics |
| `test_matches_python_re` | golden | 10 patterns: random strings from your DFA, their prefixes, and one-byte edits get the same answer from Python's `re` | the regex subset means what Python means |
| `test_every_state_is_live` | property | from every state an accepting state is reachable | masks never lead into a dead end |
| `test_canonical_form` | property | five pairs of equivalent patterns give identical `trans` and `accept` | L10.9 compares tables with yours |
| `test_unsupported_syntax_raises` | boundary | anchors, backreferences, lookarounds, lazy quantifiers, non-ASCII, bad repeats, empty language | no silent misreading |
| `test_masks_match_brute_force` | property | every state and token: the trie mask equals walking the token's bytes | the trie is only a speedup |
| `test_masks_match_the_golden_file` | golden | masks and completeness along recorded paths, from an oracle using derivatives, not automata | the L10.9 parity file |
| `test_reachable_states_never_have_an_empty_mask` | property | with all bytes and EOS, every state reachable by tokens allows something | generation never gets stuck |
| `test_constraint_lifecycle` | unit | advancing, refusing a disallowed token, EOS only when complete, nothing after EOS | engine bugs surface |
| `test_apply_mask` | boundary | float64 copy with $-\infty$; an all-false mask and a wrong shape are errors | L8.1 never sees an all $-\infty$ row |
| `test_greedy_takes_the_best_allowed_token` | unit | the largest allowed logit wins (ties lowest); the logprob is renormalized | greedy tool calls |
| `test_constrained_generation_parses` | property | 40 seeded requests under a tool schema: at least 30 finish, every finished one parses and validates | the L10.9 guarantee |
| `test_json_schema_documents` | unit | valid documents match; missing keys, wrong types, order, extra keys, trailing commas, spaces, bounds, raw control bytes do not | the schema subset's meaning |
| `test_json_schema_rejects_keywords_outside_the_subset` | boundary | `pattern`, `maxLength`, `$ref`, `additionalProperties: true`, unknown required names, unknown types | no silently ignored constraints |
| `test_vocab_bytes` | unit | the byte map undone; non-map strings as UTF-8; specials as `None` | masks over the real tokenizer |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. classes that differ from Python's: `.` matching `\n`, `[^...]` complemented over ASCII only, `\w` without `_`, ranges missing their upper end | strings `re` accepts are refused, or the reverse | `test_matches_python_re` (mutants `s01`, `s02`, `s08`, `s11`) |
| 2. no minimization | equivalent patterns give different tables; the hand example has more than 7 states | `test_canonical_form` (mutant `s03`) |
| 3. numbering in another order | tables that do not match the Rust port's | `test_hand_example` (mutant `s04`) |
| 4. EOS allowed everywhere, or a dead sink kept as a state | outputs stop mid-word; masks lead into a state with no way out | `test_hand_example_masks` (mutant `s05`), `test_every_state_is_live` (mutant `s12`) |
| 5. the trie walk stopping at the first dead byte, or recording the wrong next state | allowed tokens missing; the constraint drifts from the text | `test_masks_match_brute_force` (mutants `s06`, `s14`) |
| 6. repetition built wrong: one optional copy too few, `+` as `*` | `{2,3}` accepts only 2; `x+` accepts "" | `test_matches_python_re` (mutants `s09`, `s10`) |
| 7. anchors read as literal characters | a pattern means something else than in Python | `test_unsupported_syntax_raises` (mutant `s13`) |
| 8. JSON strings allowing raw control bytes or any high byte; integers with leading zeros; an exponent without digits | output that `json.loads` rejects | `test_constrained_generation_parses`, `test_json_schema_documents` (mutants `s15`, `s16`, `s23`, `s24`) |
| 9. optional properties made required after the first member | valid tool calls without optional arguments refused | `test_json_schema_documents` (mutant `s17`) |
| 10. `maxItems` off by one | one item too many accepted | `test_json_schema_documents` (mutant `s18`) |
| 11. unknown schema keywords ignored | output violates the schema silently | `test_json_schema_rejects_keywords_outside_the_subset` (mutant `s25`) |
| 12. token strings encoded as UTF-8 without undoing the byte map; specials given bytes | masks over the wrong bytes: `Ġ` is two bytes, not a space | `test_vocab_bytes` (mutants `s26`, `s27`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L8.1` | `constrained_sample` is `sample` on masked logits; $-\infty$ is a token that cannot be drawn |
| Back | `M05.2` | `vocab_bytes` undoes the GPT-2 byte map with `unicode_to_bytes` |
| Back | `M06.1` | graphs, reachability, breadth-first search (reading; toposort does not apply, because automata have cycles) |
| Forward | `L10.9` | the Rust engine's tool calls: a tool's JSON schema becomes a DFA, and the per-request masks must equal `masks_golden.json` and yours |

If you skip this module, `ol check L10.9` stops with `BLOCKED ... needs L8.7`: build it, or pass `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| regex to DFA to masks | Outlines (`outlines-core`) | the index built once per (regex, tokenizer): for each state, the token to next-state map, in Rust | `outlines-core` `src/index.rs` |
| JSON-schema subset | XGrammar, llguidance | context-free grammars (nested objects, `$ref`, recursion) with a pushdown automaton; masks computed in microseconds per token with adaptive caches | XGrammar paper (Dong et al., 2024); `guidance-ai/llguidance` |
| per-state cache | vLLM structured outputs | grammar compilation off the critical path, masks applied as a bitmask on the GPU | vLLM `v1/structured_output/` |
| subset construction | RE2, Rust `regex-automata` | lazy DFA construction, memory bounds, Unicode classes | `regex-automata` docs, "hybrid" DFA |
