# Property-based tests with Hypothesis, proptest, rapid, and ol_prop.h

So far the tests in this chapter have been examples: this input, that output. The tokenizer chapter of *AI Engineering* is full of code whose correctness is a **law**, not a table. A tokenizer must give back every string it encodes ([Byte-level BPE](../../ai-engineering/07-tokenizers/02-byte-level-bpe.md), [Rust fast BPE](../../ai-engineering/07-tokenizers/05-rust-fast-bpe.md)), and the Rust port must agree with the Python one on every string, not on the 300 in a fixture. The corpus pipeline in *Data Engineering*, from [Async fetch with resume](../../data-engineering/05-corpus-pipeline/01-async-fetch-resume-checksums.md) to [the licensing ledger and datasheet](../../data-engineering/05-corpus-pipeline/08-ledger-and-datasheet.md), reruns its stages after a crash, so each stage must be idempotent: running it twice is running it once.

Examples cannot cover "every string." A property test states the law once and lets a framework search for a counterexample. That is how the property suites for the Rust BPE and the [A Bloom filter](../../programming-and-algorithms/15-systems-data-structures/08-bloom-filter.md) are written, and how the golden tests of the Rust BPE found that one Egyptian hieroglyph splits differently under two Unicode versions.

This section builds on [TDD, unit tests, and mutation testing](01-tdd-unit-tests-and-mutation-grading.md): a property suite is judged exactly like an example suite, by the planted faults it catches. It also assumes you have seen BPE in Python ([Byte-level BPE](../../ai-engineering/07-tokenizers/02-byte-level-bpe.md)) and helps if you have read the Rust BPE and the [Robin Hood hash map](../../programming-and-algorithms/15-systems-data-structures/05-robin-hood-hash-map.md) in *Programming and Algorithms*, whose property tests use proptest. The kata is [`textops.py`](../../../../course/solutions/primers/craft.04/textops.py) (`normalize` and `MiniBPE`, the shapes of the corpus normalizer and of BPE) and its property suite is [`test_props.py`](../../../../course/solutions/primers/craft.04/test_props.py), which catches all ten planted faults. The suite runs with pytest and Hypothesis:

```bash
uv run --no-project --with pytest --with hypothesis python -m pytest -q primers/craft.04
```

As you read, try to name, for each rule of the kata, the law that would break if the rule were wrong.

> **Learning objectives**
> - State roundtrip, idempotence, invariant, model, fixpoint, and metamorphic properties for a piece of code.
> - Design generators that reach every rule of a spec instead of relying on uniform random input.
> - Configure Hypothesis, proptest, rapid, and `ol_prop.h` to run deterministically.
> - Explain why a roundtrip property alone misses faults that a model or fixpoint property catches.

## Properties, generators, and shrinking

A small vocabulary covers the whole technique:

| Term | Meaning |
|---|---|
| property | a function of generated inputs that asserts a law; it passes when no generated input breaks the law |
| generator (strategy) | describes a set of inputs and how to draw from it: `st.text()`, `prop::collection::vec(any::<u8>(), 0..64)`, `rapid.String()` |
| example | one generated input; a run tries `max_examples` of them (100 in the deterministic profile below) |
| shrinking | after a failure, the framework simplifies the input (shorter strings, smaller numbers) while it still fails, and reports the smallest one |
| model | a second, slow, obviously correct implementation the real one must agree with |

Most code is covered by six kinds of law:

1. **Roundtrip** (inverse functions): `decode(encode(x)) == x`, `from_bytes(to_bytes(b)) == b`.
2. **Idempotence**: `f(f(x)) == f(x)` for normalizers, dedup, formatting, `sort`.
3. **Invariants of the output**: no two consecutive spaces, every pre-token is non-empty, the heap property after every push.
4. **Model agreement** (differential): the fast implementation equals the slow one, or `std::HashMap`, on every input.
5. **Fixpoint**: an encoder that stops when no pair merges leaves no mergeable pair.
6. **Metamorphic relations**: changing the input in a known way changes the output in a known way (`normalize(x.replace("\r", "\n")) == normalize(x)`; `encode_batch` on any thread count equals serial `encode`).

## Generators decide what you test

A property is only as good as its inputs. `st.text()` draws from all of Unicode, so the chance that a 20-character string contains `\r\n` is tiny, and a normalizer bug on carriage returns survives a thousand examples. Draw instead from a **small alphabet that hits every rule** (`st.sampled_from(list("ab \t\n\r 　́"))`) and combine it with the broad one (`st.one_of(messy, st.text())`). For a BPE, draw from the letters its merges use, or almost no merge ever fires.

## Reproducibility

Random tests that fail once in a hundred runs are noise. Every framework has a deterministic mode, and the solution uses it in each language:

| Language | Framework | Deterministic setting |
|---|---|---|
| Python | Hypothesis (`@given`) | a settings profile: `derandomize=True`, `database=None`, `deadline=None`, `max_examples=100` |
| Rust | proptest (`proptest!`) | `Config { rng_seed: RngSeed::Fixed(n), failure_persistence: None, cases: 128, .. }`: a fixed seed and no `proptest-regressions/` files written into the tree |
| Go | `pgregory.net/rapid` (`rapid.Check`) | `-rapid.seed=N` (a flag of `go test`), checks 100 cases by default |
| C | [`ol_prop.h`](../../../../course/contracts/c/include/ol_prop.h) (`OL_CHECK_PROP`) | cases seeded from `OL_SEED` with SplitMix64; shrinks by halving the size and prints the seed |

Failures still shrink and print the counterexample, so you can turn it into an ordinary example test that stays forever as a regression test.

## How a property suite is judged

As in [TDD, unit tests, and mutation testing](01-tdd-unit-tests-and-mutation-grading.md), a suite is run against the correct kata and then against one planted fault at a time. Here there are ten faults, each one the bug of one pitfall (listed in [What can go wrong](#what-can-go-wrong)), and a good suite catches all ten. Two of them break no roundtrip and no idempotence: a BPE whose ties merge the rightmost pair still round-trips, and so does a BPE that stops after one merge. Only a model property or a fixpoint property catches them. That is the lesson of this section: write more than one kind of law. The next example shows why by hand.

## Working it by hand: why roundtrip is not enough

Take the kata's `MiniBPE` with merges `[(97, 98), (256, 97), (98, 98), (97, 97), ...]`: rank 0 makes 256 = `ab`, rank 1 makes 257 = `aba`, rank 2 makes 258 = `bb`, rank 3 makes 259 = `aa`. Start with the obvious property:

```python
@given(st.text())
def test_roundtrip(x):
    assert bpe.decode(bpe.encode(x)) == x
```

Run it against the fault "ties merge the rightmost pair." In `encode("aaa")`, the bytes `97 97 97` have the pair `(97, 97)` with rank 3 at positions 0 and 1. The correct encoder merges position 0 and gives `[259, 97]`; the faulty one merges position 1 and gives `[97, 259]`. Both decode to `aaa`, so **the roundtrip passes**, and it does on every input.

Now add a model:

```python
def model_encode(text):
    ids = list(text.encode("utf-8"))
    rank = {pair: r for r, pair in reversed(list(enumerate(MERGES)))}   # first rank wins
    while True:
        cands = [(rank[(a, b)], i) for i, (a, b) in enumerate(zip(ids, ids[1:])) if (a, b) in rank]
        if not cands:
            return ids
        r, i = min(cands)                    # lowest rank, then the smallest position
        ids[i : i + 2] = [256 + r]

@given(st.text(alphabet=st.sampled_from(list("abé")), max_size=24))
def test_encode_agrees_with_the_model(x):
    assert bpe.encode(x) == model_encode(x)
```

Hypothesis generates strings over `a`, `b`, `é`. Within a few examples it hits `aaa`, the faulty encoder returns `[97, 259]`, the model `[259, 97]`, the test fails, and Hypothesis shrinks the counterexample to the shortest string that still fails: `aaa`. The same model catches "stops after one merge."

The pattern is the same in every language. In Rust the model property is the one the Rust BPE uses:

```rust
proptest! {
    #![proptest_config(Config { cases: 128, rng_seed: RngSeed::Fixed(5), failure_persistence: None, ..Config::default() })]
    #[test]
    fn heap_equals_naive(s in "[abert é0]{0,40}") {
        for piece in pretokenize_gpt2(&s) {
            let mut fast = Vec::new();
            gpt2().encode_piece(piece.as_bytes(), &mut fast);
            prop_assert_eq!(fast, naive(gpt2(), piece.as_bytes()));
        }
    }
}
```

In Go (used later for the gateway and the ring) a roundtrip with rapid looks like this:

```go
func TestKeyRoundtrip(t *testing.T) {
    rapid.Check(t, func(t *rapid.T) {
        id := rapid.StringMatching(`[a-z0-9]{1,12}`).Draw(t, "id")
        if got := parseKey(formatKey(id)); got != id { t.Fatalf("got %q", got) }
    })
}
```

In C with `ol_prop.h`, a property is a function of a generator and a size that returns nonzero when it holds, and `OL_CHECK_PROP(prop, 200, 64)` runs it for 200 seeds of sizes up to 64.

## The interface and the laws to state

The kata, `primers/craft.04/textops.py`, has its spec in its docstring:

```python
def normalize(text: str) -> str: ...          # NFC; \r\n and \r to \n; collapse horizontal space; at most one blank line; strip
class MiniBPE:
    def __init__(self, merges: Sequence[tuple[int, int]]) -> None: ...   # merge r makes id 256 + r
    vocab_size: int
    def encode(self, text: str) -> list[int]: ...      # lowest rank first, leftmost on ties, until no pair merges
    def decode_bytes(self, ids: Sequence[int]) -> bytes: ...
    def decode(self, ids: Sequence[int]) -> str: ...   # UTF-8 with errors="replace"
```

The property suite, `primers/craft.04/test_props.py`, is made of `@given` properties that import only `normalize` and `MiniBPE` from `textops` and draw no randomness of their own. These are the laws it states; each one catches at least one planted fault:

| Law | For |
|---|---|
| idempotence: `normalize(normalize(x)) == normalize(x)` | `normalize` |
| output shape: stripped, no `\r`, no three newlines, no run of horizontal space and none but `" "`, lines stripped, NFC | `normalize` |
| line endings: `\r\n` and a lone `\r` behave like `\n` (a metamorphic relation) | `normalize` |
| text with no white space only gets NFC | `normalize` |
| roundtrip: `decode(encode(x)) == x` and `decode_bytes(encode(x)) == x.encode()` | `MiniBPE` |
| model: `encode(x) == model_encode(x)` over an alphabet the merges use | `MiniBPE` |
| fixpoint: no adjacent pair of `encode(x)` has a merge | `MiniBPE` |
| replacement: `decode(ids) == decode_bytes(ids).decode("utf-8", "replace")` for any ids | `MiniBPE` |

A correct solution is therefore a suite of `@given` properties with no `random` of their own, every property holds for the kata, the properties never reject correct code, and against each of the 10 planted faults at least one property fails.

## Building the kata step by step

Seven observations shape both the kata and its suite.

1. **Each rule of the spec is one step of code.** `normalize` applies rules a to e in order: NFC, line endings, horizontal space per line, blank lines, ends. Doing them in this order is what makes the function idempotent: after the last step none of the earlier ones has anything left to change.
2. **Strip lines before collapsing blank lines.** A line holding only spaces becomes empty after stripping, so `\n \n \n` turns into `\n\n\n` and is then collapsed; the reverse order leaves work for a second pass.
3. **"White space" means Unicode white space.** The class `[^\S\n]` is every character `str.isspace()` accepts except `\n`, so U+3000 and U+00A0 collapse like a space.
4. **BPE encodes by repeated lowest-rank, leftmost merge.** Scan all adjacent pairs, keep the best rank with a strict `<` (so the first, leftmost, of equal ranks wins), merge it, repeat until no pair has a rank.
5. **Decoding expands left before right.** An id above 255 is a pair `(a, b)`; pushing `b` then `a` on a stack pops `a` first, so bytes come out in order without recursion.
6. **Each law in the suite targets faults the others miss.** Idempotence and output shape cover `normalize`; roundtrip covers lost or reordered bytes; only a model or a fixpoint notices a wrong merge order or an early stop, because those still round-trip.
7. **Generators reach every rule.** Use a small alphabet with `\r`, tabs, U+3000, U+00A0, a combining accent, and letters the merges use, combined with the broad `st.text()`.

The normalizer is five lines, one per rule:

```python
_HSPACE = re.compile(r"[^\S\n]+")  # white space other than "\n"
_BLANKS = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)                        # rule a
    text = text.replace("\r\n", "\n").replace("\r", "\n")           # rule b: \r\n first, then lone \r
    lines = [_HSPACE.sub(" ", line).strip(" ") for line in text.split("\n")]   # rule c, per line
    text = "\n".join(lines)
    text = _BLANKS.sub("\n\n", text)                                  # rule d, after lines are stripped
    return text.strip()                                               # rule e: all white space
```

The order of the two `replace` calls matters: replacing lone `\r` first would turn `\r\n` into two newlines.

The BPE is a merge table, a repeated scan, and a stack-based decoder:

```python
class MiniBPE:
    def __init__(self, merges: Sequence[tuple[int, int]]) -> None:
        self.merges = [(int(a), int(b)) for a, b in merges]
        self.rank: dict[tuple[int, int], int] = {}
        self.parts: list[tuple[int, int]] = []
        for r, (a, b) in enumerate(self.merges):
            if not (0 <= a < 256 + r and 0 <= b < 256 + r):
                raise ValueError(...)                 # a merge may only use ids that already exist
            self.rank.setdefault((a, b), r)           # a repeated pair keeps its first (lowest) rank
            self.parts.append((a, b))

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) > 1:                           # until no pair merges, not just once
            best = None
            for i in range(len(ids) - 1):             # every adjacent pair, including the last
                r = self.rank.get((ids[i], ids[i + 1]))
                if r is not None and (best is None or r < best[0]):   # strict <: leftmost on ties
                    best = (r, i)
            if best is None:
                break                                 # fixpoint: nothing left to merge
            r, i = best
            ids[i : i + 2] = [256 + r]
        return ids

    def decode_bytes(self, ids: Sequence[int]) -> bytes:
        out = bytearray()
        for i in ids:
            if not 0 <= i < self.vocab_size:
                raise ValueError(...)
            stack = [i]
            while stack:
                x = stack.pop()
                if x < 256:
                    out.append(x)
                else:
                    a, b = self.parts[x - 256]
                    stack.append(b)                   # pushed first, popped last
                    stack.append(a)                   # left part expands first
        return bytes(out)

    def decode(self, ids: Sequence[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")   # invalid bytes become U+FFFD
```

`encode` is $O(n^2)$ in the worst case (one scan per merge). That is fine for a kata, and it is exactly why the Rust BPE replaces it with a heap; the slow version then becomes the model the fast one must agree with. The full kata is [`textops.py`](../../../../course/solutions/primers/craft.04/textops.py).

## Building the property suite

The suite starts with its generators, because they decide what the laws are ever tested on:

```python
# Text that reaches every rule: line endings, every kind of white space,
# combining marks, and ordinary letters.
messy = st.text(alphabet=st.sampled_from(list("ab \t\n\r 　 ́ée가")), max_size=30)
any_text = st.one_of(messy, st.text(max_size=30))

MERGES = [(97, 98), (256, 97), (98, 98), (97, 97), (257, 258), (0xC3, 0xA9)]
bpe = MiniBPE(MERGES)
bpe_text = st.text(alphabet=st.sampled_from(list("abéx")), max_size=24) | st.text(max_size=12)
```

`messy` contains a tab, `\n`, `\r`, U+3000, U+00A0, a combining acute accent, a precomposed `é` and a Hangul syllable, so every normalize rule fires within a few examples. `bpe_text` uses the letters the merges are built from (including the two UTF-8 bytes of `é`, merge 5), so merges actually happen.

The `normalize` laws come next:

```python
@given(any_text)
def test_normalize_is_idempotent(x):                 # catches: blank lines collapsed before stripping
    once = normalize(x)
    assert normalize(once) == once


@given(any_text)
def test_normalize_output_shape(x):                  # catches: ASCII-only space, ends keep newlines
    out = normalize(x)
    assert out == out.strip()
    assert "\r" not in out
    assert "\n\n\n" not in out
    assert not re.search(r"[^\S\n]{2,}|[^\S\n ]", out), repr(out)   # no runs, and only " " remains
    assert all(line == line.strip(" ") for line in out.split("\n"))
    assert unicodedata.is_normalized("NFC", out)     # catches: NFD


@given(any_text)
def test_carriage_returns_are_line_ends(x):          # catches: lone \r treated as a space
    assert normalize(x.replace("\r\n", "\n").replace("\r", "\n")) == normalize(x)


@given(st.text(alphabet=st.characters(blacklist_categories=("Zs", "Zl", "Zp", "Cc")), max_size=20))
def test_text_without_space_is_only_nfc(x):
    assert normalize(x) == unicodedata.normalize("NFC", x)
```

The metamorphic property states the line-ending rule without restating the implementation: whatever `normalize` does, `\r` must behave like `\n`.

Then the `MiniBPE` laws, with the model:

```python
def model_encode(text: str) -> list[int]:
    """The definition, slowly: find the lowest rank, take its leftmost
    occurrence, merge, repeat."""
    ids = list(text.encode("utf-8"))
    rank = {pair: r for r, pair in reversed(list(enumerate(MERGES)))}   # first rank wins
    while True:
        cands = [(rank[(a, b)], i) for i, (a, b) in enumerate(zip(ids, ids[1:])) if (a, b) in rank]
        if not cands:
            return ids
        r, i = min(cands)                            # tuple order: rank, then position
        ids[i : i + 2] = [256 + r]


@given(st.text())
def test_roundtrip(x):                               # catches: right part first, a lost byte
    assert bpe.decode(bpe.encode(x)) == x
    assert bpe.decode_bytes(bpe.encode(x)) == x.encode("utf-8")


@given(bpe_text)
def test_encode_agrees_with_the_model(x):            # catches: rightmost on ties, one merge only
    assert bpe.encode(x) == model_encode(x)


@given(bpe_text)
def test_encode_reaches_a_fixpoint(x):               # catches: one merge only
    ids = bpe.encode(x)
    assert all((a, b) not in set(MERGES) for a, b in zip(ids, ids[1:]))


@given(st.lists(st.integers(min_value=0, max_value=255 + len(MERGES)), max_size=20))
def test_decode_is_bytes_with_replacement(ids):      # catches: errors="ignore"
    assert bpe.decode(ids) == bpe.decode_bytes(ids).decode("utf-8", errors="replace")


def test_hand_example():
    assert bpe.encode("abab") == [256, 256]
    assert bpe.encode("aba") == [257]
    assert bpe.encode("aaa") == [259, 97]
    assert bpe.decode([257, 258]) == "ababb"
```

The model is obviously correct because it is the definition written with `min` over `(rank, position)` tuples; it is slow, and that does not matter. The replacement property draws arbitrary ids, not encoded text, because `encode` never produces invalid UTF-8, so a decode fault on invalid bytes would be invisible to every property that starts from text. The hand example pins the numbers from the worked example as a regression test. The full suite is [`test_props.py`](../../../../course/solutions/primers/craft.04/test_props.py).

## Why this design

Several kinds of law beat many examples of one kind. Two of the ten faults round-trip perfectly, and only the model and the fixpoint see them. Each law type is cheap to state and covers faults the others cannot.

Targeted alphabets beat pure random text. `st.text()` draws from all of Unicode, so `\r\n` or U+3000 appear rarely; the targeted alphabet finds those faults in a handful of examples, and the broad strategy is kept alongside for coverage.

Deterministic runs beat truly random ones for a gate. Derandomized Hypothesis gives the same verdict every run. The cost is that a deterministic suite explores the same inputs each time, which longer offline runs or fuzzing (see [Where this leads](#where-this-leads)) compensate for.

One tradeoff is accepted: the model is $O(n^2)$ like the kata, so the suite keeps strings short (`max_size=24`). That is enough, because every fault shows on short inputs once shrinking runs.

## What can go wrong

Each row below is one pitfall: ten of them are the planted faults, and the last three are mistakes in writing the suite itself.

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Collapsing blank lines before stripping the lines | `normalize` is not idempotent: `a\n \n \nb` needs a second pass | lines are stripped first, then `\n{3,}` collapses; caught by `test_normalize_is_idempotent` |
| Collapsing only ASCII spaces and tabs | U+3000 and U+00A0 runs survive normalization | `[^\S\n]+` matches every white space but `\n`; caught by `test_normalize_output_shape` over `messy` |
| NFD instead of NFC | `é` comes out as `e` plus a combining accent | `unicodedata.normalize("NFC", ...)`; caught by the NFC assertion and `test_text_without_space_is_only_nfc` |
| Treating a lone `\r` as a space | old Mac line endings merge two lines into one | `\r\n`, then lone `\r`, become `\n`; caught by `test_carriage_returns_are_line_ends` |
| Expanding a merged token right part first | `ab` decodes as `ba` | the stack pushes `b` then `a`, so `a` pops first; caught by `test_roundtrip` |
| Merging the rightmost of equal-rank pairs | ids differ from every other implementation; the roundtrip still passes | strict `r < best[0]` keeps the leftmost; caught by `test_encode_agrees_with_the_model` |
| Decoding with `errors="ignore"` | invalid bytes vanish instead of becoming U+FFFD | `errors="replace"`; caught by `test_decode_is_bytes_with_replacement` |
| Dropping the last byte of an odd-length input | text loses its last character | the pair loop covers every byte; caught by `test_roundtrip` |
| Stripping only spaces at the ends | a trailing newline survives | the final `strip()` removes all white space; caught by `out == out.strip()` |
| Stopping after one merge | longer token sequences; the roundtrip still passes | `while len(ids) > 1` until no pair merges; caught by the fixpoint and model properties |
| Generating only from `st.text()` | every normalize fault on `\r` or U+3000 survives | `messy` draws from an alphabet that hits every rule, combined with `st.text()` |
| A property that the correct kata fails (asserting a bug as the law) | every fault "dies" for the wrong reason | every property is derived from the docstring's rules and holds for the kata |
| Unseeded randomness (`random`, proptest without a fixed seed) | the verdict changes from run to run | no randomness outside Hypothesis, which runs derandomized |

> **Warning:** A suite of roundtrip properties alone can report green while the encoder produces ids no other implementation agrees with. Always pair a roundtrip with a model or a fixpoint.

## Where this leads

This section reused mutation testing, the correct-code baseline, and faults from pitfalls from [TDD, unit tests, and mutation testing](01-tdd-unit-tests-and-mutation-grading.md), and the roundtrip law that the Python BPE in [Byte-level BPE](../../ai-engineering/07-tokenizers/02-byte-level-bpe.md) already obeys. Going forward, the same approach appears in several places:

- [Rust fast BPE](../../ai-engineering/07-tokenizers/05-rust-fast-bpe.md) has a proptest suite: roundtrip, heap equals naive, stream equals decode, batch equals serial.
- The [A Bloom filter](../../programming-and-algorithms/15-systems-data-structures/08-bloom-filter.md) in *Programming and Algorithms* has a proptest suite: no false negatives, byte roundtrip, union equals one filter.
- [Extract, normalize, quality filters](../../data-engineering/05-corpus-pipeline/02-extract-normalize-and-quality-filters.md) in *Data Engineering* contains the real `normalize_unicode` stage and the pipeline idempotence it must keep.
- [Oracles, golden and differential tests, gradcheck as a test](03-oracles-golden-and-gradcheck.md) builds oracles and differential tests on the model property.
- Properties for the recurrent models, the durable engine, and the data structures are written the same way.

Each piece of the solution has a production equivalent that goes further:

- Beyond `@given` properties, Hypothesis stateful testing (`hypothesis.stateful`) offers `RuleBasedStateMachine`: random sequences of operations against a model, which is how the LRU of [Intrusive list + LRU](../../programming-and-algorithms/15-systems-data-structures/03-intrusive-list-and-lru.md) would be tested.
- Beyond proptest, [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz) and libFuzzer do coverage-guided search that finds inputs no generator describes; see the `tokenizers` and `regex` crates' fuzz targets.
- Beyond model properties, [Jepsen](https://jepsen.io/) checks properties over histories of a distributed system (linearizability), as [Raft HA for the durable log](../../systems/07-durable-orchestration/10-raft-ha.md) does; see the Elle checker.
- Beyond basic shrinking, Hypothesis's internal test-case reduction shrinks any generator, including composed ones, without hand-written shrinkers (MacIver and Donaldson, *Test-Case Reduction via Test-Case Generation*, 2020).

### Further reading

- Claessen and Hughes, *QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs* (2000).
- MacIver et al., *Hypothesis: A new approach to property-based testing* (JOSS, 2019).
- The [Hypothesis](https://hypothesis.readthedocs.io/), [proptest](https://proptest-rs.github.io/proptest/), and [rapid](https://pkg.go.dev/pgregory.net/rapid) documentation.

## Key terms

- **Property**: a function of generated inputs that asserts a law that must hold for every input of a kind.
- **Generator (strategy)**: a description of a set of inputs and how to draw from it.
- **Shrinking**: simplifying a failing input while it still fails, to report a minimal counterexample.
- **Model**: a slow, obviously correct implementation the real one must agree with.
- **Fixpoint property**: a law that the output admits no further step of the algorithm.
- **Metamorphic relation**: a law relating outputs for inputs changed in a known way.

## Summary

- A property is a law that must hold for every input of a kind (`decode(encode(x)) == x`, `normalize(normalize(x)) == normalize(x)`); the framework generates the inputs and, on failure, shrinks them to a minimal counterexample.
- Roundtrip alone is weak: a BPE that merges nothing round-trips. Add a fixpoint property, a model to agree with, and the output's shape.
- Generators decide what you test: random `str` almost never contains `\r\n`, U+3000, or a combining accent, so draw from an alphabet that hits every rule.
- Reproducible or worthless: Hypothesis runs derandomized with no example database and proptest with a fixed seed and no failure files, so a planted fault is caught or survives the same way every run.
- A property suite is judged like any test suite: it must accept the correct kata and catch the planted faults, here all ten.

## Review questions

1. Why does a BPE that merges the rightmost of two equal-rank pairs still pass the roundtrip?

   <details><summary>Answer</summary>

   Either merge produces a token that expands back to the same bytes: `aaa` gives `[259, 97]` or `[97, 259]`, and both decode to `aaa`. The roundtrip only checks that information is preserved, not that the ids are the canonical ones; a model property compares the ids themselves.

   </details>

2. Why does `test_decode_is_bytes_with_replacement` generate lists of ids instead of encoding generated text?

   <details><summary>Answer</summary>

   Encoded text always decodes to valid UTF-8, so `errors="ignore"` and `errors="replace"` agree on it and the fault is invisible. Arbitrary id lists include stray bytes such as a lone `0xC3`, where "replace" gives U+FFFD and "ignore" drops it.

   </details>

3. Why must `normalize` strip each line before collapsing runs of newlines?

   <details><summary>Answer</summary>

   `a\n \n \nb` has lines holding only spaces. Collapsing first finds no `\n\n\n` (the spaces separate the newlines); stripping afterwards produces `a\n\n\nb`, which a second call would collapse. The function would not be idempotent. Stripping first exposes the run, and the collapse then finishes the job in one pass.

   </details>

4. What does the metamorphic property `normalize(x with \r replaced) == normalize(x)` gain over asserting `"\r" not in normalize(x)`?

   <details><summary>Answer</summary>

   The shape assertion is also satisfied by an implementation that turns a lone `\r` into a space: the `\r` is gone. The metamorphic property says how `\r` must behave, as a line end, so that implementation fails it: two lines merged into one differ from the same text with `\n`.

   </details>

5. Why does the strict `<` in `encode` implement "leftmost on ties"?

   <details><summary>Answer</summary>

   The scan goes left to right and replaces `best` only when a strictly lower rank appears. A later pair of the same rank does not replace it, so the first, leftmost occurrence stays. With `<=`, each later equal-rank pair would replace it and the rightmost would win.

   </details>
