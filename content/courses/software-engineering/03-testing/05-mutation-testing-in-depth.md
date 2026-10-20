# Mutation testing in depth: equivalent mutants, semantic mutants from pitfalls, reading survivors

The model suite of [Sequence metrics: exact match, BLEU, chrF](../../ai-engineering/10-attention/05-sequence-metrics.md) in *AI Engineering* kills at least 80% of its planted faults, every semantic one included. That number hides two questions. Which 20% survived, and why? And could any test ever have killed them? The sequence metrics are the first place where the suites guard real numerical code: they are the metrics the [LM evaluation harness](../../ai-engineering/12-objectives-and-adaptation/07-lm-evaluation-harness-and-the-model-zoo.md) will print for every model trained. Later in this chapter the mutant classes cannot be read as patches at all: a 2x slowdown the benchmark gate must notice ([Benchmarks and perf gates](04-benchmarks-and-perf-gates.md)), a zeroed layer the model eval must flag ([Model evals as tests](08-model-evals-as-tests.md)). Reading survivors is the skill underneath all of them.

This section takes the sequence-metrics unit and ten mutants that a typical 80%-kill suite lets live. It sorts the real gaps from the impossible ones, closes the gaps, and adds mutants of its own from the unit's pitfalls. It assumes you know how a suite is judged by planted faults from [TDD, unit tests, and mutation testing](01-tdd-unit-tests-and-mutation-grading.md) and laws as tests from [Property-based tests](02-property-based-tests.md). Read [Sequence metrics](../../ai-engineering/10-attention/05-sequence-metrics.md) first, since the survivors are small diffs against its `seqmetrics.py`; the contract the suite tests is [`seqmetrics.pyi`](../../../../course/contracts/py/tinyllm/eval/seqmetrics.pyi).

The solution has three artifacts in [`primers/craft.07/`](../../../../course/solutions/primers/craft.07/): [`test_seqmetrics.py`](../../../../course/solutions/primers/craft.07/test_seqmetrics.py), the sequence-metrics suite deepened until it kills every mutant; [`triage.toml`](../../../../course/solutions/primers/craft.07/triage.toml), the verdict on ten survivors; and [`mutants/p01.patch`](../../../../course/solutions/primers/craft.07/mutants/p01.patch) and [`mutants/p02.patch`](../../../../course/solutions/primers/craft.07/mutants/p02.patch), two semantic mutants written from the unit's pitfalls. Before reading each verdict, decide it yourself from the diff alone.

> **Learning objectives**
> - Distinguish killed, survived, equivalent, and killable mutants, and compute the honest mutation score.
> - Argue a survivor's verdict with either a concrete input or one of four invariant patterns.
> - Write a test that kills a killable survivor alone and still passes on every correct implementation.
> - Write semantic mutants from a pitfalls list that are real, new, small, and named.

## Killed, survived, equivalent

The mutation score `killed / total` has a hidden assumption: every mutant in the denominator is killable. Making that assumption explicit needs a slightly larger vocabulary than the one in [TDD, unit tests, and mutation testing](01-tdd-unit-tests-and-mutation-grading.md):

| Term | Meaning |
|---|---|
| mutant | the program with one small change (an operator flipped, a bound moved, a statement deleted, a pitfall planted) |
| killed | at least one test fails (or times out) on the mutant |
| survived | every test passes on the mutant |
| equivalent | for every input in the contract's domain, the mutant's observable behavior equals the original's; no test can kill it |
| killable | some input makes an observable output differ; a survivor that is killable is a missing test |
| score | `killed / total`; the honest version is `killed / (total - equivalent)` |

An equivalent mutant can never be killed, so it caps the score below 1 and wastes time until you recognize it. Deciding equivalence for arbitrary programs is undecidable: it contains program equivalence, and so the halting problem. In practice you argue each one. A survivor is **killable** when you can write down an input and the output that differs. It is **equivalent** when an invariant of the code makes the change unobservable, and those invariants come in four patterns:

- the changed branch only runs when both sides give the same value (`x < y` to `x <= y` where `x == y` gives the same result);
- an earlier check rules out the inputs that would differ (a length checked equal before use);
- the changed value is overwritten or never read;
- the domain excludes the difference (a sum of non-negative terms is never below zero).

Write the argument down. "Equivalent because..." is a claim a reviewer can check, and the same reasoning tells you whether a later refactor turns it killable.

## Reading a survivor

Given a survivor, ask three questions in order.

1. **What changed, exactly?** Read the diff, then the lines around it. Which values can reach the changed line?
2. **Is there an input that reaches it and changes an output?** Work backwards from the change to the function's inputs. Small, adversarial inputs win: ties, empty strings, a line ending in `-\n`, a reference shorter than the n-gram order.
3. **Is that input inside the contract?** A difference only on inputs the contract rejects (a `ValueError` path both versions take) is not a difference.

If question 2 finds an input, write the test with exactly that input and assert the contract's answer, computed by hand, not printed by the code under test. Then run it against the mutant: it must fail, alone.

## Semantic mutants from pitfalls

Generated mutants (operator flips, boundary shifts) are cheap and numerous, and most are killed by any reasonable suite. The mutants that teach are **semantic**: the bug a person actually writes, taken from a unit's pitfalls list. A good semantic mutant has four properties:

| Property | Check |
|---|---|
| real | an oracle suite (the reference tests) kills it, so it is not equivalent |
| new | no other mutant produces the same file; two mutants killed by exactly the same tests are redundant |
| small | one idea, a few lines, so a survivor points at one missing test |
| named | the pitfall it plants, in one line |

This is how the book's own mutants are made: one per pitfall, kept only if the reference tests kill it.

## Working two mutants by hand

Before the ten survivors, work two mutants of `seqmetrics.py` that are **not** among them, plus one new semantic mutant.

**Mutant A** changes `exact_match`'s last line from `return hits / len(hyps)` to `return hits / len(refs)`. Question 2 asks for an input where `len(refs) != len(hyps)`. But `_check_pairs` runs first and raises `ValueError` unless the lengths are equal, so on every input that reaches the division the two lengths are the same number. Question 3 confirms it: the only inputs where they differ are outside the contract. Mutant A is **equivalent**, because of an earlier check.

**Mutant B** deletes the padding line `line = f" {line} "` in `tokenize_13a`. It looks equivalent, because the last line, `" ".join(line.split())`, removes any leading and trailing spaces anyway. Work backwards anyway: the padding matters only where a regex needs a character before or after the match. The rule "split a period unless preceded by a digit" is `([^0-9])([\.,])`, and the rule "unless followed by a digit" is `([\.,])([^0-9])`. For the input `5.`, the period is preceded by a digit and followed by nothing, so without padding neither rule fires and `5.` stays one token. With padding, the trailing space is a non-digit after the period, so it becomes `5 .`. Mutant B is **killable**, and the test is one line:

```python
def test_13a_splits_a_final_period_after_a_digit():
    assert tokenize_13a("It costs 5.") == "It costs 5 ."
```

"Looks equivalent" is not "is equivalent": deleting padding before a tokenizer that collapses white space is harmless on almost every input and wrong on one.

**A semantic mutant.** The pitfall "arithmetic mean instead of geometric mean" is already planted among the unit's own mutants (`s07` there), so write a new one from the rule that [BLEU](../../ai-engineering/10-attention/05-sequence-metrics.md#bleu) counts orders 1 to 4, as a third patch, `p03.patch`:

```diff
# pitfall: BLEU uses n-gram orders 1 to 4 (L4.5 section 2.2); this mutant stops at 3
--- a/python/tinyllm/eval/seqmetrics.py
+++ b/python/tinyllm/eval/seqmetrics.py
@@ -91,7 +91,7 @@
 def _word_ngrams(tokens: list[str]) -> Counter:
     """Every n-gram of orders 1..4, as tuples, with counts."""
     grams: Counter = Counter()
-    for n in range(1, MAX_ORDER + 1):
+    for n in range(1, MAX_ORDER):
         for i in range(len(tokens) - n + 1):
             grams[tuple(tokens[i : i + n])] += 1
     return grams
```

It applies to the reference, it is new, the reference golden tests kill it (every 4-gram count becomes 0), and the unit's hand example `[6, 6, 5, 3, 1, 0, 6, 5, 4, 3]` kills it too: the mutant gives `[6, 6, 5, 3, 1, 0, 6, 5, 4, 0]`.

## The artifacts

The solution delivers three artifacts with fixed formats.

**The suite**, `primers/craft.07/test_seqmetrics.py`, is one file of at least 12 tests, importing only `tinyllm.eval.seqmetrics` (plus `pytest`, `math`, `json`, `itertools`, `collections`, `numpy`, `os`, `pathlib`). It starts as the sequence-metrics suite. The bar is higher than 80%: it must kill **every** one of the unit's mutants (they are all killable, since the reference tests kill each one) and each survivor called killable.

**The triage**, `primers/craft.07/triage.toml`, has one table per survivor:

```toml
[s01]
verdict = "equivalent"            # or "killable"
reason  = "why no input can tell it apart, or the input that does"
test    = ""                      # killable: the test in test_seqmetrics.py that kills it alone
```

**The mutants**, `primers/craft.07/mutants/*.patch`, are at least two unified diffs with paths `a/python/tinyllm/eval/seqmetrics.py` and `b/python/tinyllm/eval/seqmetrics.py`, against the reference, each starting with a `# pitfall: ...` line.

A correct solution satisfies seven checks, each for a reason that matters downstream:

| Case | Kind of check | Why it matters downstream |
|---|---|---|
| the three artifacts are well formed: 12 tests, contract-only imports, ten verdicts with reasons, two `# pitfall:` patches | unit | a suite that imports internals breaks on the first refactor |
| the suite passes on the reference (baseline A) and on any other correct implementation (baseline B) | conformance | a test that rejects a correct implementation kills mutants for the wrong reason |
| all 25 of the unit's mutants fail the suite | fault | the metric the evaluation harness reports has no known untested bug |
| each verdict is right (five of each kind) | unit | a wrong "equivalent" leaves a real bug in place |
| each killable survivor fails the named test run alone, and that test passes on the reference | fault | the kill is a checkable claim, not a coincidence of the whole suite |
| each patch applies, changes the file, duplicates no committed mutant, and fails the reference tests | fault | an equivalent or duplicate mutant teaches nothing |
| each hand-written patch fails the suite | fault | a pitfall you know about is tested |

## Triaging the survivors step by step

Five principles drive the triage.

1. **A survivor is a claim about inputs.** Either some input in the contract's domain makes an observable output differ (killable), or none does (equivalent). So every verdict needs either a concrete input or an invariant, never "my tests pass."
2. **Equivalence comes from the four patterns.** Both sides agree on the changed branch, an earlier check excludes the differing inputs, the value is dead, or the domain excludes the difference. For each diff, check those four first; if none applies, look for the input.
3. **Killable inputs are small and adversarial.** Ties, a final `-\n`, an HTML entity, a reference shorter than the n-gram order. The diff tells you which: a changed comparison points at a tie, a deleted normalization at the input it normalizes.
4. **The killing test must assert the contract's answer, computed by hand,** and it must fail on the mutant alone. Otherwise it may kill the mutant by asserting the reference's accident, and it will reject a correct baseline B.
5. **Hand-written mutants come from pitfalls,** because the generated operators (flips, off-by-ones) are what a suite already kills. One pitfall, one small patch, one named test.

### The ten verdicts

Each row is the diff (one line of `seqmetrics.py`), the verdict, and the argument recorded in [`triage.toml`](../../../../course/solutions/primers/craft.07/triage.toml):

| Id | Change | Verdict | Argument |
|---|---|---|---|
| s01 | brevity penalty: `sys_len < ref_len` became `<=` | equivalent | at `sys_len == ref_len` the penalty is $\exp(1 - 1) = 1$ either way, and `sys_len = 0` never gets there: no match returns 0 first (both sides agree on the changed branch) |
| s02 | closest-reference tie: `r < best_len` became `r > best_len` | killable | equidistant references now resolve to the longer one; a 7-token hypothesis against references of 9 and 5 tokens shows it |
| s03 | empty order: `break` became `continue` | equivalent | `total_n = 0` means the hypothesis is shorter than $n$ tokens, so every higher order is 0 too and every remaining log stays $\log 0$ (the domain excludes the difference) |
| s04 | 13a entity block deleted (`&quot; &amp; &lt; &gt;`) | killable | without unescaping, `&amp;` tokenizes as `& amp ;` and no longer matches a reference `&` |
| s05 | reference maximum: `c > max_ref[g]` became `>=` | equivalent | when `c` equals the stored maximum, the assignment writes the value already there (a dead write) |
| s06 | best chrF reference: `f > best_f` became `>=` | killable | two references can tie on sentence chrF with different statistics; keeping the last changes the pooled corpus score |
| s07 | `line = line.rstrip()` deleted before tokenizing | killable | 13a deletes a dash-newline pair, so a line ending in `-\n` tokenizes differently without the `rstrip` |
| s08 | bootstrap: `len(picked)` became `len(rows)` | equivalent | every bootstrap resample has exactly $n$ rows, so the two lengths are the same number on every call (an invariant of the caller) |
| s09 | chrF hypothesis count: `sum(h.values()) if g else 0` became `sum(h.values())` | killable | a reference shorter than the order adds hypothesis n-grams to the pooled precision denominator |
| s10 | chrF zero check: `prec + rec == 0` became `<= 0` | equivalent | precision and recall are averages of non-negative ratios, so their sum is never below zero (the domain) |

Five of each. Notice the shape: all five equivalent ones change a comparison at a point where an invariant makes both sides agree, and four of the five killable ones are a tie rule or a deleted normalization, which a typical suite never exercises because its inputs have no ties and no entities.

### The tests that kill the killable survivors

Each test sits under a heading in the suite and names its survivor in the docstring, so a reviewer can run it against that one diff. Survivor s02 needs a tie between references:

```python
def test_closest_reference_tie_goes_to_the_shorter():
    """Survivor s02: 7 tokens, references of 9 and 5 tokens, both 2 away: 5."""
    # bleu_stats(...)[1] is the reference length the brevity penalty uses.
    # Both orders of the references: the rule is "shorter wins", not "first wins".
    assert bleu_stats("a b c d e f g", ["a b c d e f g h i", "a b c d e"])[1] == 5
    assert bleu_stats("a b c d e f g", ["a b c d e", "a b c d e f g h i"])[1] == 5
```

Listing the references both ways matters: with one order only, a mutant that picks "the first of the tied" would also pass.

Survivor s04 needs an entity:

```python
def test_13a_unescapes_entities():
    """Survivor s04: &amp; is one & token, &quot; a quote."""
    assert tokenize_13a("a&amp;b &quot;x&quot;") == 'a & b " x "'
    # and through BLEU: all 5 unigrams match only if &amp; became &
    assert bleu_stats("a&amp;b c d", ["a & b c d"])[2] == 5
```

Survivor s06 needs a chrF tie whose statistics differ:

```python
def test_chrf_best_reference_tie_keeps_the_first():
    """Survivor s06: "aa" and "acab" give "abca" the same sentence chrF with
    different statistics; the first one listed wins, and pooling shows it."""
    first = chrf_stats("abca", ["aa"])
    assert chrf_stats("abca", ["aa", "acab"]) == first
    assert chrf_stats("abca", ["acab", "aa"]) == chrf_stats("abca", ["acab"])
    assert chrf_stats("abca", ["aa"]) != chrf_stats("abca", ["acab"])  # the tie is real: stats differ
```

The last line is what makes this a kill and not a tautology: if the two references had the same statistics, keeping the last would be equivalent.

Survivor s07 needs a line ending in `-\n`:

```python
def test_13a_sees_the_line_after_rstrip():
    """Survivor s07: rstrip first, so a final "-\\n" keeps its dash."""
    assert bleu_stats("e-mail-\n", ["e-mail-"])[2:3] == [1]   # the one unigram matches
```

Without the `rstrip`, 13a's "delete `-\n`" rule fires and `e-mail-\n` becomes `e-mail`, which does not match `e-mail-`.

Survivor s09 is the subtle one, and needs a corpus:

```python
def test_chrf_hypothesis_count_needs_reference_ngrams():
    """Survivor s09: an order the reference has no n-gram of counts 0
    hypothesis n-grams (sacreBLEU's rule), which shows when pooled."""
    stats = chrf_stats("abcdefgh", ["abc"])
    assert stats[9:12] == [0, 0, 0]  # order 4: "abc" has no 4-gram
    assert chrf(["abcdefgh", "abc"], [["abc"], ["abcdefgh"]]) == pytest.approx(
        chrf_from_stats_by_hand(), rel=1e-12
    )


def chrf_from_stats_by_hand() -> float:
    # sentence 1: orders 1..3 [8, 3, 3], [7, 2, 2], [6, 1, 1]; 4..6 zero (no ref n-grams)
    # sentence 2: orders 1..3 [3, 8, 3], [2, 7, 2], [1, 6, 1]; 4..6 [0, 5, 0], [0, 4, 0], [0, 3, 0]
    # pooled orders 1..3: hyp 11, 9, 7; ref 11, 9, 7; match 6, 4, 2; orders 4..6 have hyp 0
    p = (6 / 11 + 4 / 9 + 2 / 7) / 3
    return 100 * 5 * p * p / (4 * p + p)   # beta = 2: F = (1 + 4) P R / (4 P + R), with P = R here
```

A single sentence does not show s09: with one sentence, the order whose reference is empty contributes 0 matches either way. Pooling two sentences, one in each direction, makes the extra hypothesis count land in the corpus denominator. The expected value is derived by hand from the counts in the comments.

### The two hand-written mutants

**p01** comes from the [tokenization rule](../../ai-engineering/10-attention/05-sequence-metrics.md#tokens) that 13a splits a dash only after a digit:

```diff
# pitfall: 13a splits a dash only after a digit (L4.5 section 2.1); this mutant forgets the rule, so 10-12 stays one token
--- a/python/tinyllm/eval/seqmetrics.py
+++ b/python/tinyllm/eval/seqmetrics.py
@@ -36,8 +36,6 @@
     (re.compile(r"([^0-9])([\.,])"), r"\1 \2 "),
     # period and comma unless followed by a digit
     (re.compile(r"([\.,])([^0-9])"), r" \1 \2"),
-    # dash when preceded by a digit
-    (re.compile(r"([0-9])(-)"), r"\1 \2 "),
 ]
```

```python
def test_13a_splits_a_dash_after_a_digit():
    """My mutant p01: the fourth 13a rule deleted."""
    assert tokenize_13a("pages 10-12") == "pages 10 - 12"
```

**p02** comes from the rule that [chrF](../../ai-engineering/10-attention/05-sequence-metrics.md#chrf)'s beta is a parameter:

```diff
# pitfall: chrF's beta is a parameter (L4.5 section 2.3); this mutant always weights recall by 2^2
@@ -179,7 +179,7 @@
-    factor = beta**2
+    factor = 4.0
```

```python
def test_chrf_beta_is_a_parameter():
    """My mutant p02: beta fixed at 2. With beta = 1, F is the harmonic mean."""
    # "cat" vs "cats": precision 1 at orders 1..3; recall 3/4, 2/3, 1/2
    r = (3 / 4 + 2 / 3 + 1 / 2) / 3
    assert chrf(["cat"], [["cats"]], beta=1.0) == pytest.approx(100 * 2 * r / (1 + r), rel=1e-12)
```

Both are real (the reference tests kill them), new (no committed mutant deletes that regex or fixes beta), and small (one idea each). Every default-argument test passes on p02, because the default is $\beta = 2$ and $2^2 = 4$; only a non-default $\beta$ kills it, which is why a pitfall-driven mutant finds a test nobody had written. The full suite is [`test_seqmetrics.py`](../../../../course/solutions/primers/craft.07/test_seqmetrics.py).

## Why this design

The machine cost is small. Running a suite once per mutant is $O(\text{mutants} \times \text{suite time})$; with 25 mutants and a sub-second suite, that is seconds. The cost that matters is human: each survivor takes minutes to argue, which is why the equivalent ones must be recognized fast, by the four patterns, rather than "fixed" with ever more tests.

Arguing equivalence beats deleting the mutant. Deleting an equivalent survivor without a written reason loses the argument; a later refactor (say, removing the early `return 0` that makes s01 equivalent) can make it killable, and the written reason says exactly what to recheck.

A test per survivor complements a property. A property from [Property-based tests](02-property-based-tests.md) can kill a family at once, but a named test per survivor makes each kill checkable alone, and it documents the exact input. The suite keeps both.

Hand-written mutants beat more generated ones. Generated operators mostly duplicate each other; one pitfall-driven mutant per pitfall targets the bugs people actually write.

## What can go wrong

Most mistakes in this work are verdicts or kills made for the wrong reason:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Calling a survivor equivalent because no existing test kills it | a real bug stays in the metric the evaluation harness reports | every equivalent verdict cites one of the four invariant patterns |
| Calling a survivor killable without an input that shows it | a named test that also passes on the survivor | every killable verdict names the input, and the test uses exactly that input |
| "Killing" a survivor with a test that rejects the reference (asserting an implementation detail, or a bug) | every mutant dies for the wrong reason | expected values derived by hand (`chrf_from_stats_by_hand`), contract-only imports |
| Stopping at the 80% threshold | the 20% that survived stays untested | the suite kills all 25 mutants, not 80% of them |
| A hand-written mutant that is equivalent, or that copies a committed one | it teaches nothing and inflates the count | p01 and p02 each plant a distinct pitfall that the reference tests kill |
| Writing a mutant and never checking the suite against it | a pitfall you know is still untested | each patch has its own test (`test_13a_splits_a_dash_after_a_digit`, `test_chrf_beta_is_a_parameter`) |
| Expected values printed by the code under test | the test agrees with the bug, and the reference rejects it | every expected value is worked by hand in the test or its comments |

> **Warning:** "No test kills it" is never an argument for equivalence. It is the definition of a survivor, and half of these survivors are real bugs.

## Where this leads

This section extended killed and survived, the two baselines, and required mutants from [TDD, unit tests, and mutation testing](01-tdd-unit-tests-and-mutation-grading.md), used laws as tests from [Property-based tests](02-property-based-tests.md) (a property often kills a whole family of survivors at once), and took its unit and pitfalls from [Sequence metrics](../../ai-engineering/10-attention/05-sequence-metrics.md). Every later testing section reads survivors this way, and several use mutant classes of their own:

- [Oracles, golden and differential tests, gradcheck as a test](03-oracles-golden-and-gradcheck.md) judges oracle suites for the transformer kernels with the same survivor reading.
- [Benchmarks and perf gates](04-benchmarks-and-perf-gates.md) uses perf mutants: a survivor is a slowdown the benchmark gate did not trip.
- [Resilience tests](07-resilience-tests.md) uses resilience mutants: a survivor is a fault the kill loop did not expose.
- [Model evals as tests](08-model-evals-as-tests.md) and [Agent evals as tests](09-agent-evals-as-tests.md) use model and agent mutants.

Each piece has a production equivalent:

- Reading survivors is what Google's mutation testing in code review automates: mutants surfaced as review comments on changed lines only, with heuristics that suppress likely-equivalent ("arid") nodes (Petrovic et al., "Practical Mutation Testing at Scale," 2021).
- The triage has a mechanical helper in trivial compiler equivalence (TCE), which compiles both versions and calls a mutant equivalent when the machine code is identical, removing a share of equivalent mutants for free (Papadakis et al., ICSE 2015).
- The book's mutants correspond to [mutmut](https://github.com/boxed/mutmut), [cosmic-ray](https://github.com/sixty-north/cosmic-ray), [cargo-mutants](https://mutants.rs/), and [Stryker](https://stryker-mutator.io/), which generate mutants with incremental runs and caches.
- Semantic mutants generalize to higher-order and subsuming mutants: combinations harder to kill than any single one, and the minimal subsuming set (Jia and Harman, "Higher Order Mutation Testing," 2009).

### Further reading

- DeMillo, Lipton, and Sayward, "Hints on Test Data Selection" (1978).
- Offutt and Pan, "Automatically Detecting Equivalent Mutants and Infeasible Paths" (1997).
- Papadakis et al., "Mutation Testing Advances: An Analysis and Survey" (2019).
- Petrovic and Ivankovic, "State of Mutation Testing at Google" (ICSE SEIP 2018).

## Key terms

- **Equivalent mutant**: a mutant whose observable behavior equals the original's on every input in the contract's domain.
- **Killable survivor**: a surviving mutant for which some input makes an observable output differ; a missing test.
- **Honest mutation score**: `killed / (total - equivalent)`.
- **Semantic mutant**: a hand-written mutant that plants one known pitfall.
- **Triage**: the recorded verdict and argument for each survivor.
- **Baseline B**: any correct implementation other than the reference, which every test must also accept.

## Summary

- The mutation score assumes every mutant is killable; an equivalent mutant caps the score below 1 and wastes time until you recognize it.
- Equivalence is undecidable in general, so you decide it per mutant with an argument: an input that makes an observable output differ, or an invariant that proves none exists.
- "Looks equivalent" is not "is equivalent": deleting padding before a tokenizer that collapses white space is harmless on almost every input and wrong on one.
- The most valuable mutants are semantic, one per pitfall, written by hand: real, new, and small.
- A deep suite kills every killable mutant, accepts the correct implementation, and names the test that kills each survivor, so each kill is a claim you can check.

## Review questions

1. s05 changes `>` to `>=` and is equivalent; s06 changes `>` to `>=` and is killable. What is the difference?

   <details><summary>Answer</summary>In s05 the tied branch assigns the value that is already stored, so the state after the tie is identical. In s06 the tied branch replaces the chosen reference with a different one whose statistics differ, and those statistics are pooled into the corpus score. Same operator, different consequence of the tie.</details>

2. Why does `test_closest_reference_tie_goes_to_the_shorter` list the references in both orders?

   <details><summary>Answer</summary>With only one order, an implementation that keeps the first of two tied references would pass by accident when the shorter one happens to be first. Both orders pin the rule "shorter wins" independent of position.</details>

3. What would make s01 killable after a refactor?

   <details><summary>Answer</summary>Its equivalence rests on two facts: the penalty is 1 when the lengths are equal, and an empty hypothesis never reaches the comparison because no match returns 0 first. Remove that early return and an empty hypothesis against an empty reference (`sys_len = ref_len = 0`) reaches the comparison: the original's `<` is false and keeps the penalty at 1, while the mutant's `<=` is true and takes the `sys_len > 0` guard's other branch, a penalty of 0.</details>

4. Why does a single-sentence test not kill s09?

   <details><summary>Answer</summary>`chrf_from_stats` skips an order unless both its hypothesis and reference counts are positive. For one sentence whose reference has no n-grams of an order, the reference count is 0, so the order is skipped whatever the hypothesis count is. Pooled over a corpus, another sentence supplies reference n-grams of that order; now the mutant's hypothesis count makes the pooled order count, with 0 extra matches, and lowers the averaged precision. The reference keeps the pooled hypothesis count at 0, so the order is still skipped.</details>

5. Why must the killing test pass on the reference and on any correct baseline, not just fail on the mutant?

   <details><summary>Answer</summary>A test that fails on the mutant but also fails on a correct implementation is asserting an accident (an implementation detail, or a value printed from buggy code). It "kills" mutants for the wrong reason and blocks correct refactors.</details>
