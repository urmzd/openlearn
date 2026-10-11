<!-- ol:module craft.04 -->
# Property-based tests (R4) with Hypothesis, proptest, rapid, and ol_prop.h

## Overview

| | |
|---|---|
| **Module** | `craft.04` · practice · Python · Pass 3 · 3 to 4 h |
| **You build** | `primers/craft.04/textops.py` (a kata: `normalize` and `MiniBPE`, the shapes of the corpus normalizer and of BPE) and `primers/craft.04/test_props.py`, your property tests for it |
| **Contract** | the rules in the kata's docstring (`ol check craft.04` writes the kata with stubs on its first run) |
| **Tests** | `course/tests/craft.04/`: the grade of your property tests by planted faults (section 4) |
| **Needs** | reading: [`craft.03`](01-tdd-unit-tests-and-mutation-grading.md) how tests are graded · [`L1.2`](../../ml/08-tinyllm/p01-tokenizers/02-byte-level-bpe.md) BPE in Python · [`L1.5`](../../ml/08-tinyllm/p01-tokenizers/05-rust-fast-bpe.md) and [`ds.05`](../../algorithms/16-systems-data-structures/05-robin-hood-hash-map.md), whose R4 tests you write with proptest |
| **Used by** | no call site (a practice): rung R4 grades your proptest suites in `L1.5` and `ds.08`, and later your properties for `L3`, the durable engine, and the data structures |
| **Milestone** | `MS-L1` (the Part 1 milestone requires a passing `craft.04`) |
| **Optional depth** | Claessen and Hughes, *QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs* (2000); MacIver et al., *Hypothesis: A new approach to property-based testing* (JOSS, 2019); the [Hypothesis](https://hypothesis.readthedocs.io/), [proptest](https://proptest-rs.github.io/proptest/), and [rapid](https://pkg.go.dev/pgregory.net/rapid) documentation |

## Key Takeaways

- A property is a law that must hold for every input of a kind (`decode(encode(x)) == x`, `normalize(normalize(x)) == normalize(x)`); the framework generates the inputs, and on failure shrinks them to a minimal counterexample.
- Roundtrip alone is weak: a BPE that merges nothing round-trips. Add a fixpoint property, a model (a slow, obviously correct implementation) to agree with, and the output's shape (`test_planted_faults_are_caught`).
- Generators decide what you test: random `str` almost never contains `\r\n`, U+3000, or a combining accent, so draw from an alphabet that hits every rule.
- Reproducible or worthless: the course runs Hypothesis with the `ol` profile (derandomized, no example database) and proptest with a fixed seed and no failure files, so a planted fault is caught or survives the same way every run.
- A property test is graded like any test: it must accept the correct kata (`test_your_tests_accept_the_reference`) and catch at least 80% of ten planted faults, every one of them required.

## How to work this chapter

```bash
ol check craft.04           # first run: writes primers/craft.04/textops.py (stubs) and fails
# write primers/craft.04/test_props.py (section 4), then the kata; run your tests yourself:
uv run --no-project --with pytest --with hypothesis python -m pytest -q primers/craft.04
ol check craft.04           # grades your properties against the planted faults
```

Write the properties first, against the stub: they must fail. Then make the kata pass them. `ol check` never shows you a planted fault's code; a survivor prints only its one-line description.

---

## 1. Why now

So far your tests have been examples: this input, that output. Pass 3 is full of code whose correctness is a **law**, not a table. A tokenizer must give back every string it encodes (`L1.2`, `L1.5`), and the Rust port must agree with the Python one on every string, not on the 300 in a fixture. The corpus pipeline you build next (`data.01` to `data.08`) reruns its stages after a crash, so each stage must be idempotent: running it twice is running it once. Examples cannot cover "every string". A property test states the law once and lets a framework search for a counterexample, which is how the `L1.5` and `ds.08` tests you write are graded (rung R4), and how the golden tests of `L1.5` found that one Egyptian hieroglyph splits differently under two Unicode versions.

## 2. Principles

### 2.1 Properties, generators, shrinking

| Term | Meaning |
|---|---|
| property | a function of generated inputs that asserts a law; it passes when no generated input breaks the law |
| generator (strategy) | describes a set of inputs and how to draw from it: `st.text()`, `prop::collection::vec(any::<u8>(), 0..64)`, `rapid.String()` |
| example | one generated input; a run tries `max_examples` of them (100 in the `ol` profile) |
| shrinking | after a failure, the framework simplifies the input (shorter strings, smaller numbers) while it still fails, and reports the smallest one |
| model | a second, slow, obviously correct implementation the real one must agree with |

The law types that cover most code:

1. **Roundtrip** (inverse functions): `decode(encode(x)) == x`, `from_bytes(to_bytes(b)) == b`.
2. **Idempotence**: `f(f(x)) == f(x)` for normalizers, dedup, formatting, `sort`.
3. **Invariants of the output**: no two consecutive spaces, every pre-token is non-empty, the heap property after every push.
4. **Model agreement** (differential): the fast implementation equals the slow one, or `std::HashMap`, on every input.
5. **Fixpoint**: an encoder that stops when no pair merges leaves no mergeable pair.
6. **Metamorphic relations**: changing the input in a known way changes the output in a known way (`normalize(x.replace("\r", "\n")) == normalize(x)`; `encode_batch` on any thread count equals serial `encode`).

### 2.2 Generators decide what you test

A property is only as good as its inputs. `st.text()` draws from all of Unicode: the chance that a 20-character string contains `\r\n` is tiny, and a normalizer bug on carriage returns survives a thousand examples. Draw from a **small alphabet that hits every rule** (`st.sampled_from(list("ab \t\n\r 　́"))`) and combine it with the broad one (`st.one_of(messy, st.text())`). For a BPE, draw from the letters its merges use, or almost no merge ever fires.

### 2.3 Reproducibility

Random tests that fail once in a hundred runs are noise. Every framework has a deterministic mode, and the course uses it:

| Language | Framework | Deterministic setting the course uses |
|---|---|---|
| Python | Hypothesis (`@given`) | profile `ol`: `derandomize=True`, `database=None`, `deadline=None`, `max_examples=100` (loaded by `ol` for your graded tests) |
| Rust | proptest (`proptest!`) | `Config { rng_seed: RngSeed::Fixed(n), failure_persistence: None, cases: 128, .. }`: a fixed seed and no `proptest-regressions/` files written into the tree |
| Go | `pgregory.net/rapid` (`rapid.Check`) | `-rapid.seed=N` (a flag of `go test`), checks 100 cases by default |
| C | `ol_prop.h` (`OL_CHECK_PROP`) | cases seeded from `OL_SEED` with SplitMix64; shrinks by halving the size and prints the seed |

Failures still shrink and print the counterexample, so you turn it into an ordinary example test that stays forever (a regression test).

### 2.4 How a property suite is graded

As in `craft.03`, your tests run against the course's correct kata with one planted fault at a time. Rung R4 asks for 0.80 of the faults and every required one, and in this module all ten are required, each one the bug of one pitfall (section 5). Two of them break no roundtrip and no idempotence: a BPE whose ties merge the rightmost pair still round-trips, and so does a BPE that stops after one merge. Only a model property or a fixpoint property catches them. That is the lesson: write more than one kind of law.

## 3. Worked example by hand

The kata's `MiniBPE` with merges `[(97, 98), (256, 97), (98, 98), (97, 97), ...]` (rank 0 makes 256 = `ab`, rank 1 makes 257 = `aba`, rank 2 makes 258 = `bb`, rank 3 makes 259 = `aa`). A first property, by hand:

```python
@given(st.text())
def test_roundtrip(x):
    assert bpe.decode(bpe.encode(x)) == x
```

Run it against the fault "ties merge the rightmost pair" (`s06`). `encode("aaa")`: the bytes `97 97 97` have the pair `(97, 97)` with rank 3 at positions 0 and 1; the correct encoder merges position 0 and gives `[259, 97]`, the faulty one merges position 1 and gives `[97, 259]`. Both decode to `aaa`: **the roundtrip passes**, and so it does on every input. Now add a model:

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

Hypothesis generates strings over `a`, `b`, `é`; within a few examples it hits `aaa`, the faulty encoder returns `[97, 259]`, the model `[259, 97]`, the test fails, and Hypothesis shrinks the counterexample to the shortest string that still fails: `aaa`. The same model catches "stops after one merge" (`s10`). In Rust the model property is the one you write for `L1.5`:

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

and in Go (for the gateway and the ring later) a roundtrip with rapid:

```go
func TestKeyRoundtrip(t *testing.T) {
    rapid.Check(t, func(t *rapid.T) {
        id := rapid.StringMatching(`[a-z0-9]{1,12}`).Draw(t, "id")
        if got := parseKey(formatKey(id)); got != id { t.Fatalf("got %q", got) }
    })
}
```

and in C with `ol_prop.h`: a property function of a generator and a size returns nonzero when it holds, and `OL_CHECK_PROP(prop, 200, 64)` runs it for 200 seeds of sizes up to 64.

## 4. The artifact and its check

**The kata**, `primers/craft.04/textops.py` (its docstring is the spec):

```python
def normalize(text: str) -> str: ...          # NFC; \r\n and \r to \n; collapse horizontal space; at most one blank line; strip
class MiniBPE:
    def __init__(self, merges: Sequence[tuple[int, int]]) -> None: ...   # merge r makes id 256 + r
    vocab_size: int
    def encode(self, text: str) -> list[int]: ...      # lowest rank first, leftmost on ties, until no pair merges
    def decode_bytes(self, ids: Sequence[int]) -> bytes: ...
    def decode(self, ids: Sequence[int]) -> str: ...   # UTF-8 with errors="replace"
```

**Your tests**, `primers/craft.04/test_props.py`: at least five test functions, at least four of them `@given` properties, importing `normalize` and `MiniBPE` from `textops` and drawing no randomness of their own. The laws to state (each one catches at least one planted fault):

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

**The check** (`ol check craft.04`, `course/tests/craft.04/check`) runs four tests:

| Test | KIND | Checks |
|---|---|---|
| `test_your_tests_are_properties` | unit | the file exists, at least 5 tests, at least 4 `@given`, no `random` |
| `test_your_kata_passes_your_tests` | unit | your properties hold for your kata |
| `test_your_tests_accept_the_reference` | conformance | your properties hold for the course's kata: they never reject a correct implementation |
| `test_planted_faults_are_caught` | fault | against each of the 10 planted faults your tests fail; score at least 0.80 and every required fault caught |

Each run copies your test file next to one version of the kata in a scratch directory and runs pytest with the `ol` Hypothesis profile, so the verdict is the same on every run and your tree is never touched.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Collapsing blank lines before stripping the lines | `normalize` is not idempotent: `a\n \n \nb` needs a second pass | an idempotence property (mutant `s01`) |
| Collapsing only ASCII spaces and tabs | U+3000 and U+00A0 runs survive normalization | an output-shape property over a white-space alphabet (mutant `s02`) |
| NFD instead of NFC | `é` comes out as `e` plus a combining accent | the NFC property (mutant `s03`) |
| Treating a lone `\r` as a space | old Mac line endings merge two lines into one | the line-ending metamorphic property (mutant `s04`) |
| Expanding a merged token right part first | `ab` decodes as `ba` | the roundtrip (mutant `s05`) |
| Merging the rightmost of equal-rank pairs | ids differ from every other implementation; the roundtrip still passes | a model property (mutant `s06`) |
| Decoding with `errors="ignore"` | invalid bytes vanish instead of becoming U+FFFD | the replacement property over arbitrary ids (mutant `s07`) |
| Dropping the last byte of an odd-length input | text loses its last character | the roundtrip (mutant `s08`) |
| Stripping only spaces at the ends | a trailing newline survives | an output-shape property (mutant `s09`) |
| Stopping after one merge | longer token sequences; the roundtrip still passes | a fixpoint or model property (mutant `s10`) |
| Generating only from `st.text()` | every normalize fault on `\r` or U+3000 survives | `test_planted_faults_are_caught` reports the survivors |
| A property that the correct kata fails (asserting your own bug as the law) | every fault "dies" for the wrong reason | `test_your_tests_accept_the_reference` |
| Unseeded randomness (`random`, proptest without a fixed seed) | the grade changes from run to run | `test_your_tests_are_properties` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `craft.03` | mutation grading, baselines A and B, required faults |
| Back | `L1.2` | BPE encode and the roundtrip law your Python already obeys |
| Forward | `L1.5` | your rung R4 suite in proptest: roundtrip, heap equals naive, stream equals decode, batch equals serial |
| Forward | `ds.08` | your rung R4 suite in proptest: no false negatives, byte roundtrip, union equals one filter |
| Forward | `data.02` | the real `normalize_unicode` stage, and the pipeline idempotence it must keep |
| Forward | `craft.05` | oracles and differential tests (R5) build on the model property |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `@given` properties | Hypothesis stateful testing | `RuleBasedStateMachine`: random sequences of operations against a model (how you would test the LRU of `ds.03`) | `hypothesis.stateful` |
| proptest | [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz) and libFuzzer | coverage-guided search that finds inputs no generator describes | the `tokenizers` and `regex` crates' fuzz targets |
| model properties | [Jepsen](https://jepsen.io/) | properties over histories of a distributed system (linearizability), as `dur.10` checks | the Elle checker |
| shrinking | Hypothesis's internal test-case reduction | shrinks any generator, including composed ones, without hand-written shrinkers | MacIver and Donaldson, *Test-Case Reduction via Test-Case Generation* (2020) |
