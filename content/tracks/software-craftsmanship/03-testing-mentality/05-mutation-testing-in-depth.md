# Mutation testing in depth: equivalent mutants, semantic mutants from pitfalls, reading survivors

## Overview

| | |
|---|---|
| **Module** | `craft.07` · practice · Python · Pass 4 · 2 h reading |
| **Solution** | three artifacts in [`primers/craft.07/`](../../../../course/solutions/primers/craft.07/): [`test_seqmetrics.py`](../../../../course/solutions/primers/craft.07/test_seqmetrics.py) (the L4.5 suite, deepened until it kills every L4.5 mutant), [`triage.toml`](../../../../course/solutions/primers/craft.07/triage.toml) (the verdict on ten survivors), and [`mutants/p01.patch`](../../../../course/solutions/primers/craft.07/mutants/p01.patch), [`mutants/p02.patch`](../../../../course/solutions/primers/craft.07/mutants/p02.patch) (two semantic mutants written from L4.5's pitfalls) |
| **Contract** | the L4.5 contract the suite tests, [`course/contracts/py/tinyllm/eval/seqmetrics.pyi`](../../../../course/contracts/py/tinyllm/eval/seqmetrics.pyi); the artifact formats in section 4 |
| **Needs** | reading: [`craft.03`](01-tdd-unit-tests-and-mutation-grading.md) how a test suite is judged by planted faults · [`craft.04`](02-property-based-tests.md) laws as tests · [`L4.5`](../../../courses/build-an-llm/p04-attention-origins/05-sequence-metrics.md) the unit under test, and its rung R5 suite |
| **Used by** | no call site (a practice): every later rung reads survivors this way, and rungs R7 to R10 use the special mutant classes (perf, model, agent, resilience). Part 4 ends with the metrics `L6.7` reports guarded by a suite that kills every killable mutant. |
| **Optional depth** | DeMillo, Lipton, and Sayward, "Hints on Test Data Selection" (1978); Offutt and Pan, "Automatically Detecting Equivalent Mutants and Infeasible Paths" (1997); Papadakis et al., "Mutation Testing Advances: An Analysis and Survey" (2019); Petrovic and Ivankovic, "State of Mutation Testing at Google" (ICSE SEIP 2018) |

## Key Takeaways

- The mutation score `killed / total` has a hidden assumption: every mutant in the denominator is killable. An **equivalent** mutant (no input tells it from the original) can never be killed, so it caps the score below 1 and wastes time until you recognize it.
- Equivalence is undecidable in general; you decide it per mutant with an argument: find an input that makes an observable output differ, or prove that none exists from the code's invariants.
- "Looks equivalent" is not "is equivalent": deleting a `rstrip()` before a tokenizer that collapses white space is harmless on almost every input and wrong on one.
- The most valuable mutants are **semantic**: one per pitfall, written by hand. A good one is real (an oracle suite kills it), new (not a duplicate of another), and small.
- A deep suite kills every killable mutant, accepts the correct implementation, and names the test that kills each survivor, so the kill is a claim you can check.

## How to read this chapter

Sections 1 to 3 build the vocabulary (killed, survived, equivalent) and work two mutants by hand. Section 4 fixes the three artifacts and their formats. Section 5 walks through the solution: all ten survivors with their verdict and argument, the tests that kill the killable ones, and the two hand-written mutants. Before reading each verdict, decide it yourself from the diff alone. The unit under test is L4.5's `seqmetrics.py`; read that chapter first, since the survivors are small diffs against it.

---

## 1. Why now

L4.5's suite met rung R5: it kills at least 80% of the planted faults, every semantic one included. That number hides two questions. Which 20% survived, and why? And could any test ever have killed them? Pass 4 is the first pass where the suites guard real numerical code, the metrics `L6.7` will print for every model trained, and from Pass 6 on the mutant classes cannot be read as patches at all: a 2x slowdown the benchmark gate must notice (R7), a zeroed layer the model eval must flag (R8). Reading survivors is the skill underneath all of them. This module takes the L4.5 unit, ten mutants that a typical R5 suite lets live, sorts the real gaps from the impossible ones, closes the gaps, and adds mutants of its own from the chapter's pitfalls.

## 2. Principles

### 2.1 Killed, survived, equivalent

| Term | Meaning |
|---|---|
| mutant | the program with one small change (an operator flipped, a bound moved, a statement deleted, a pitfall planted) |
| killed | at least one test fails (or times out) on the mutant |
| survived | every test passes on the mutant |
| equivalent | for every input in the contract's domain, the mutant's observable behavior equals the original's; no test can kill it |
| killable | some input makes an observable output differ; a survivor that is killable is a missing test |
| score | `killed / total`; the honest version is `killed / (total - equivalent)` |

Deciding equivalence for arbitrary programs is undecidable (it contains program equivalence, and so the halting problem). In practice you argue each one. A survivor is **killable** when you can write down an input and the output that differs. It is **equivalent** when an invariant of the code makes the change unobservable:

- the changed branch only runs when both sides give the same value (`x < y` to `x <= y` where `x == y` gives the same result);
- an earlier check rules out the inputs that would differ (a length checked equal before use);
- the changed value is overwritten or never read;
- the domain excludes the difference (a sum of non-negative terms is never below zero).

Write the argument down: "equivalent because..." is a claim a reviewer can check, and the same reasoning tells you whether a later refactor turns it killable.

### 2.2 Reading a survivor

Three questions, in order:

1. **What changed, exactly?** Read the diff, then the lines around it. Which values can reach the changed line?
2. **Is there an input that reaches it and changes an output?** Work backwards from the change to the function's inputs. Small, adversarial inputs win: ties, empty strings, a line ending in `-\n`, a reference shorter than the n-gram order.
3. **Is that input inside the contract?** A difference only on inputs the contract rejects (a `ValueError` path both versions take) is not a difference.

If question 2 finds an input, write the test with exactly that input and assert the contract's answer (computed by hand, not printed by the code under test). Then run it against the mutant: it must fail, alone.

### 2.3 Semantic mutants from pitfalls

Generated mutants (operator flips, boundary shifts) are cheap and numerous; most are killed by any reasonable suite. The mutants that teach are **semantic**: the bug a person actually writes, taken from a chapter's pitfalls table. A good semantic mutant is:

| Property | Check |
|---|---|
| real | an oracle suite (the course's tests) kills it, so it is not equivalent |
| new | no other mutant produces the same file; two mutants killed by exactly the same tests are redundant |
| small | one idea, a few lines, so a survivor points at one missing test |
| named | the pitfall it plants, in one line |

This is how the course's own mutants are made: one per pitfall, kept only if the reference tests kill it.

## 3. Worked example by hand

Two mutants of L4.5's `seqmetrics.py` that are **not** among the ten survivors of section 5.

**Mutant A** changes `exact_match`'s last line from `return hits / len(hyps)` to `return hits / len(refs)`. Question 2: an input where `len(refs) != len(hyps)`? `_check_pairs` runs first and raises `ValueError` unless the lengths are equal, so on every input that reaches the division the two lengths are the same number. Question 3 confirms it: the only inputs where they differ are outside the contract. **Equivalent**, because of an earlier check.

**Mutant B** deletes the padding line `line = f" {line} "` in `tokenize_13a`. It looks equivalent: the last line, `" ".join(line.split())`, removes any leading and trailing spaces anyway. Question 2, working backwards: the padding matters only where a regex needs a character before or after the match. The rule "split a period unless preceded by a digit" is `([^0-9])([\.,])`, and the rule "unless followed by a digit" is `([\.,])([^0-9])`. For the input `5.`, the period is preceded by a digit and followed by nothing, so without padding neither rule fires and `5.` stays one token; with padding the trailing space is a non-digit after the period, so it becomes `5 .`. **Killable**, and the test is one line:

```python
def test_13a_splits_a_final_period_after_a_digit():
    assert tokenize_13a("It costs 5.") == "It costs 5 ."
```

**A semantic mutant** from L4.5's pitfall "arithmetic mean instead of geometric mean" is already planted (`s07` of L4.5), so write a new one from section 2.2 of L4.5, "BLEU counts orders 1 to 4", as a third patch, `p03.patch`:

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

It applies to the reference, it is new, the course's golden tests kill it (every 4-gram count becomes 0), and L4.5's hand example `[6, 6, 5, 3, 1, 0, 6, 5, 4, 3]` kills it too: the mutant gives `[6, 6, 5, 3, 1, 0, 6, 5, 4, 0]`.

## 4. The artifacts

**The suite**, `primers/craft.07/test_seqmetrics.py`: one file, at least 12 tests, importing only `tinyllm.eval.seqmetrics` (plus `pytest`, `math`, `json`, `itertools`, `collections`, `numpy`, `os`, `pathlib`). It starts as the L4.5 suite. The bar is higher than R5: it must kill **every** L4.5 mutant (they are all killable: the reference tests kill each one) and each survivor called killable.

**The triage**, `primers/craft.07/triage.toml`, one table per survivor:

```toml
[s01]
verdict = "equivalent"            # or "killable"
reason  = "why no input can tell it apart, or the input that does"
test    = ""                      # killable: the test in test_seqmetrics.py that kills it alone
```

**The mutants**, `primers/craft.07/mutants/*.patch`, at least two: unified diffs with paths `a/python/tinyllm/eval/seqmetrics.py` and `b/python/tinyllm/eval/seqmetrics.py`, against the L4.5 reference, each starting with a `# pitfall: ...` line.

### What a correct solution must handle

| Case | Kind of check | Why it matters downstream |
|---|---|---|
| the three artifacts are well formed: 12 tests, contract-only imports, ten verdicts with reasons, two `# pitfall:` patches | unit | a suite that imports internals breaks on the first refactor |
| the suite passes on the reference L4.5 (baseline A) and on any other correct L4.5 (baseline B) | conformance | a test that rejects a correct implementation kills mutants for the wrong reason |
| all 25 L4.5 mutants fail the suite | fault | the metric `L6.7` reports has no known untested bug |
| each verdict is right (five of each kind) | unit | a wrong "equivalent" leaves a real bug in place |
| each killable survivor fails the named test run alone, and that test passes on the reference | fault | the kill is a checkable claim, not a coincidence of the whole suite |
| each patch applies, changes the file, duplicates no committed mutant, and fails the reference tests | fault | an equivalent or duplicate mutant teaches nothing |
| each hand-written patch fails the suite | fault | a pitfall you know about is tested |

## 5. The solution, step by step

### 5.1 From first principles

1. **A survivor is a claim about inputs.** Either some input in the contract's domain makes an observable output differ (killable), or none does (equivalent). So every verdict needs either a concrete input or an invariant, never "my tests pass".
2. **Equivalence comes from four places** (section 2.1): both sides agree on the changed branch, an earlier check excludes the differing inputs, the value is dead, or the domain excludes the difference. So for each diff, check those four first; if none applies, look for the input.
3. **Killable inputs are small and adversarial.** Ties, a final `-\n`, an HTML entity, a reference shorter than the n-gram order. The diff tells you which: a changed comparison points at a tie, a deleted normalization at the input it normalizes.
4. **The killing test must assert the contract's answer, computed by hand,** and it must fail on the mutant alone. Otherwise it may kill the mutant by asserting the reference's accident, and it will reject a correct baseline B.
5. **Hand-written mutants come from pitfalls,** because the generated operators (flips, off-by-ones) are what a suite already kills. One pitfall, one small patch, one named test.

### 5.2 The ten survivors, triaged

Each row is the diff (one line of `seqmetrics.py`), the verdict, and the argument from [`triage.toml`](../../../../course/solutions/primers/craft.07/triage.toml).

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

Five of each. Note the shape: all five equivalent ones change a comparison at a point where an invariant makes both sides agree; four of the five killable ones are a tie rule or a deleted normalization, which a typical suite never exercises because its inputs have no ties and no entities.

### 5.3 The tests that kill the killable survivors

Each test sits under a heading in the suite and names its survivor in the docstring, so a reviewer can run it against that one diff.

```python
def test_closest_reference_tie_goes_to_the_shorter():
    """Survivor s02: 7 tokens, references of 9 and 5 tokens, both 2 away: 5."""
    # bleu_stats(...)[1] is the reference length the brevity penalty uses.
    # Both orders of the references: the rule is "shorter wins", not "first wins".
    assert bleu_stats("a b c d e f g", ["a b c d e f g h i", "a b c d e"])[1] == 5
    assert bleu_stats("a b c d e f g", ["a b c d e", "a b c d e f g h i"])[1] == 5
```

Listing the references both ways matters: with one order only, a mutant that picks "the first of the tied" would also pass.

```python
def test_13a_unescapes_entities():
    """Survivor s04: &amp; is one & token, &quot; a quote."""
    assert tokenize_13a("a&amp;b &quot;x&quot;") == 'a & b " x "'
    # and through BLEU: all 5 unigrams match only if &amp; became &
    assert bleu_stats("a&amp;b c d", ["a & b c d"])[2] == 5
```

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

```python
def test_13a_sees_the_line_after_rstrip():
    """Survivor s07: rstrip first, so a final "-\\n" keeps its dash."""
    assert bleu_stats("e-mail-\n", ["e-mail-"])[2:3] == [1]   # the one unigram matches
```

Without the `rstrip`, 13a's "delete `-\n`" rule fires and `e-mail-\n` becomes `e-mail`, which does not match `e-mail-`.

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

### 5.4 The two hand-written mutants

**p01**, from L4.5 section 2.1, "13a splits a dash only after a digit":

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

**p02**, from L4.5 section 2.3, "chrF's beta is a parameter":

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

Both are real (the reference tests kill them), new (no committed mutant deletes that regex or fixes beta), and small (one idea each). Every default-argument test passes on p02, because the default is $\beta = 2$ and $2^2 = 4$; only a non-default $\beta$ kills it, which is why a pitfall-driven mutant finds a test nobody had written.

Full suite: [`test_seqmetrics.py`](../../../../course/solutions/primers/craft.07/test_seqmetrics.py).

### 5.5 Why this solution

- **Cost.** Running a suite once per mutant is $O(\text{mutants} \times \text{suite time})$; with 25 mutants and a sub-second suite, seconds. The cost that matters is human: each survivor takes minutes to argue, which is why the equivalent ones must be recognized fast (the four patterns of section 2.1) rather than "fixed" with ever more tests.
- **Arguing equivalence versus deleting the mutant.** Deleting an equivalent survivor without a written reason loses the argument; a later refactor (say, removing the early `return 0` that makes s01 equivalent) can make it killable, and the written reason says exactly what to recheck.
- **A test per survivor versus one big property.** A property (craft.04) can kill a family at once, but a named test per survivor makes each kill checkable alone, and it documents the exact input. The suite keeps both.
- **Hand-written mutants versus more generated ones.** Generated operators mostly duplicate each other; one pitfall-driven mutant per chapter pitfall targets the bugs people actually write.

### 5.6 Explain it back

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

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Calling a survivor equivalent because no existing test kills it | a real bug stays in the metric `L6.7` reports | every equivalent verdict cites one of the four invariant patterns (5.2) |
| Calling a survivor killable without an input that shows it | a named test that also passes on the survivor | every killable verdict names the input, and the test uses exactly that input (5.3) |
| "Killing" a survivor with a test that rejects the reference (asserting an implementation detail, or a bug) | every mutant dies for the wrong reason | expected values derived by hand (`chrf_from_stats_by_hand`), contract-only imports |
| Stopping at the rung's threshold | the 20% that survived R5 stays untested | the suite kills all 25 L4.5 mutants, not 80% of them |
| A hand-written mutant that is equivalent, or that copies a committed one | it teaches nothing and inflates the count | p01 and p02 each plant a distinct pitfall that the reference tests kill |
| Writing a mutant and never checking the suite against it | a pitfall you know is still untested | each patch has its own test (`test_13a_splits_a_dash_after_a_digit`, `test_chrf_beta_is_a_parameter`) |
| Expected values printed by the code under test | the test agrees with the bug, and the reference rejects it | every expected value is worked by hand in the test or its comments |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `craft.03` | killed and survived, baselines A and B, required mutants |
| Back | `craft.04` | laws as tests: a property often kills a whole family of survivors at once |
| Back | `L4.5` | the unit under test and the pitfalls the mutants come from |
| Forward | `craft.05` | oracles and differential tests (R5) for L5 to L7, judged with the same survivor reading |
| Forward | `craft.06` | perf mutants (R7): a survivor is a slowdown the benchmark gate did not trip |
| Forward | `craft.21` | resilience mutants (R10): a survivor is a fault the kill loop did not expose |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| reading survivors | Google's mutation testing in code review | mutants surfaced as review comments on changed lines only, with heuristics that suppress likely-equivalent ("arid") nodes | Petrovic et al., "Practical Mutation Testing at Scale" (2021) |
| the triage | trivial compiler equivalence (TCE) | compiles both versions and calls a mutant equivalent when the machine code is identical, which removes a share of equivalent mutants for free | Papadakis et al., ICSE 2015 |
| the course's mutants | [mutmut](https://github.com/boxed/mutmut), [cosmic-ray](https://github.com/sixty-north/cosmic-ray), [cargo-mutants](https://mutants.rs/), [Stryker](https://stryker-mutator.io/) | generated mutants with incremental runs and caches | each tool's documentation |
| semantic mutants | higher-order and subsuming mutants | combinations of mutants that are harder to kill than any single one; the minimal subsuming set | Jia and Harman, "Higher Order Mutation Testing" (2009) |
