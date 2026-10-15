<!-- ol:module craft.07 -->
# Mutation testing in depth: equivalent mutants, semantic mutants from pitfalls, reading survivors

## Overview

| | |
|---|---|
| **Module** | `craft.07` · practice · Python · Pass 4 · 3 to 4 h |
| **You build** | three artifacts in `primers/craft.07/`: `test_seqmetrics.py` (your L4.5 suite, deepened until it kills every L4.5 mutant), `triage.toml` (your verdict on ten survivors), and `mutants/*.patch` (at least two semantic mutants you write from L4.5's pitfalls) |
| **Contract** | the L4.5 contract your suite tests, [`course/contracts/py/tinyllm/eval/seqmetrics.pyi`](../../../../course/contracts/py/tinyllm/eval/seqmetrics.pyi); the artifact formats in section 4 |
| **Tests** | `course/tests/craft.07/`: the grade of your suite, your triage, and your mutants against the course's L4.5 reference (section 4) |
| **Needs** | reading: [`craft.03`](01-tdd-unit-tests-and-mutation-grading.md) how a test is graded · [`craft.04`](02-property-based-tests.md) laws as tests · [`L4.5`](../../../courses/build-an-llm/p04-attention-origins/05-sequence-metrics.md) the unit under test, and your rung R5 suite for it |
| **Used by** | no call site (a practice): every later rung reads survivors this way, and rungs R7 to R10 grade with the special mutant classes (perf, model, agent, resilience) |
| **Milestone** | `MS-L4` (the Part 4 milestone requires a passing `craft.07`) |
| **Optional depth** | DeMillo, Lipton, and Sayward, "Hints on Test Data Selection" (1978); Offutt and Pan, "Automatically Detecting Equivalent Mutants and Infeasible Paths" (1997); Papadakis et al., "Mutation Testing Advances: An Analysis and Survey" (2019); Petrovic and Ivankovic, "State of Mutation Testing at Google" (ICSE SEIP 2018) |

## Key Takeaways

- The mutation score `killed / total` has a hidden assumption: every mutant in the denominator is killable. An **equivalent** mutant (no input tells it from the original) can never be killed, so it caps the score below 1 and wastes your time until you recognize it (`test_triage_matches_the_survivors`).
- Equivalence is undecidable in general; you decide it per mutant with an argument: find an input that makes an observable output differ, or prove that none exists from the code's invariants.
- "Looks equivalent" is not "is equivalent": deleting a `rstrip()` before a tokenizer that collapses white space is harmless on almost every input and wrong on one (`test_killable_survivors_die_by_the_named_test`).
- The most valuable mutants are **semantic**: one per pitfall, written by hand. A good one is real (an oracle suite kills it), new (not a duplicate of another), and small (`test_your_mutants_are_real_and_new`).
- A deep suite kills every killable mutant, accepts the correct implementation, and names the test that kills each survivor, so the kill is a claim you can check (`test_suite_kills_every_l45_mutant`, `test_suite_accepts_the_reference`).

## How to work this chapter

```bash
ol check L4.5               # first: L4.5 passes, and your rung R5 suite is graded
ol mutate L4.5              # the full grade: which L4.5 mutants your suite misses
ol check craft.07           # first run: writes primers/craft.07/survivors/ and an empty triage.toml, and fails
cp python/tests/l4-5-seqmetrics/test_*.py primers/craft.07/test_seqmetrics.py   # one file: start from your R5 suite
# read each survivors/sNN.diff, decide, write triage.toml; add a test per killable survivor;
# write two patches in primers/craft.07/mutants/ and the tests that kill them
ol check craft.07           # grades the suite, the triage, and your mutants
```

The survivors are diffs against the course's L4.5 reference, so they show a few of its lines: finish L4.5 first.

---

## 1. Why now

`ol check L4.5` graded your tests at rung R5: at least 80% of the planted faults, every semantic one included. That number hides two questions. Which 20% survived, and why? And could any test ever have killed them? Pass 4 is the first pass where your graded suites guard real numerical code, the metrics `L6.7` will print for every model you train, and from Pass 6 on the grade uses mutant classes you cannot read as patches at all: a 2x slowdown your benchmark gate must notice (R7), a zeroed layer your model eval must flag (R8). Reading survivors is the skill underneath all of them. This module takes the L4.5 unit you just built, hands you ten mutants that a typical R5 suite lets live, and asks you to sort the real gaps from the impossible ones, close the gaps, and write mutants of your own from the chapter's pitfalls.

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

If question 2 finds an input, write the test with exactly that input and assert the contract's answer (computed by hand, not printed by your code). Then run it against the mutant: it must fail, alone.

### 2.3 Semantic mutants from pitfalls

Generated mutants (operator flips, boundary shifts) are cheap and numerous; most are killed by any reasonable suite. The mutants that teach are **semantic**: the bug a person actually writes, taken from a chapter's pitfalls table. A good semantic mutant is:

| Property | Check |
|---|---|
| real | an oracle suite (the course's tests) kills it, so it is not equivalent |
| new | no other mutant produces the same file; two mutants killed by exactly the same tests are redundant |
| small | one idea, a few lines, so a survivor points at one missing test |
| named | the pitfall it plants, in one line |

This is how the course's own mutants are made (DESIGN 5.6): one per pitfall, kept only if the course tests kill it.

## 3. Worked example by hand

Two mutants of L4.5's `seqmetrics.py` that are **not** among your ten survivors.

**Mutant A** changes `exact_match`'s last line from `return hits / len(hyps)` to `return hits / len(refs)`. Question 2: an input where `len(refs) != len(hyps)`? `_check_pairs` runs first and raises `ValueError` unless the lengths are equal, so on every input that reaches the division the two lengths are the same number. Question 3 confirms it: the only inputs where they differ are outside the contract. **Equivalent**, because of an earlier check.

**Mutant B** deletes the padding line `line = f" {line} "` in `tokenize_13a`. It looks equivalent: the last line, `" ".join(line.split())`, removes any leading and trailing spaces anyway. Question 2, working backwards: the padding matters only where a regex needs a character before or after the match. The rule "split a period unless preceded by a digit" is `([^0-9])([\.,])`, and the rule "unless followed by a digit" is `([\.,])([^0-9])`. For the input `5.`, the period is preceded by a digit and followed by nothing, so without padding neither rule fires and `5.` stays one token; with padding the trailing space is a non-digit after the period, so it becomes `5 .`. **Killable**, and the test is one line:

```python
def test_13a_splits_a_final_period_after_a_digit():
    assert tokenize_13a("It costs 5.") == "It costs 5 ."
```

**A semantic mutant** from L4.5's pitfall "arithmetic mean instead of geometric mean" is already planted (`s07` of L4.5), so write a new one from section 2.2 of L4.5, "BLEU counts orders 1 to 4", as `primers/craft.07/mutants/p03.patch`:

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

It applies to the reference, it is new, the course's golden tests kill it (every 4-gram count becomes 0), and your hand example `[6, 6, 5, 3, 1, 0, 6, 5, 4, 3]` kills it too: the mutant gives `[6, 6, 5, 3, 1, 0, 6, 5, 4, 0]`.

## 4. The artifact and its check

**Your suite**, `primers/craft.07/test_seqmetrics.py`: one file, at least 12 tests, importing only `tinyllm.eval.seqmetrics` (plus `pytest`, `math`, `json`, `itertools`, `collections`, `numpy`, `os`, `pathlib`). It starts as your L4.5 suite. The bar is higher than R5: it must kill **every** L4.5 mutant (they are all killable: the course tests kill each one) and each survivor you call killable.

**Your triage**, `primers/craft.07/triage.toml`, one table per survivor (the first run of the check writes the ten ids and their one-line descriptions):

```toml
[s01]
verdict = "equivalent"            # or "killable"
reason  = "why no input can tell it apart, or the input that does"
test    = ""                      # killable: the test in test_seqmetrics.py that kills it alone
```

**Your mutants**, `primers/craft.07/mutants/*.patch`, at least two: unified diffs with paths `a/python/tinyllm/eval/seqmetrics.py` and `b/python/tinyllm/eval/seqmetrics.py`, against the reference shown in the survivors' context lines, each starting with a `# pitfall: ...` line.

**The check** (`ol check craft.07`, `course/tests/craft.07/check`) runs eight tests:

| Test | KIND | Checks |
|---|---|---|
| `test_artifacts_are_complete` | unit | the three artifacts exist and are well formed: 12 tests, contract-only imports, ten verdicts with reasons, two `# pitfall:` patches |
| `test_suite_accepts_the_reference` | conformance | your suite passes on the course's L4.5 (baseline A) |
| `test_suite_accepts_your_l45` | unit | your suite passes on your own L4.5 (baseline B) |
| `test_suite_kills_every_l45_mutant` | fault | all 25 L4.5 mutants fail your suite |
| `test_triage_matches_the_survivors` | unit | each verdict is right (five of each kind) |
| `test_killable_survivors_die_by_the_named_test` | fault | each killable survivor fails the test you named, run alone, which passes on the reference |
| `test_your_mutants_are_real_and_new` | fault | each patch applies, changes the file, matches no committed mutant, and fails the course's L4.5 tests |
| `test_your_suite_kills_your_mutants` | fault | each of your patches fails your suite |

Every run puts one version of `seqmetrics.py` (and the reference `M07.4` `stats.py` it imports) in a scratch tree and runs pytest there in its own process group, so your repo is never touched and a hung test cannot hang the check.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Calling a survivor equivalent because no test you have kills it | a real bug stays in the metric `L6.7` reports | `test_triage_matches_the_survivors` |
| Calling a survivor killable without an input that shows it | a named test that also passes on the survivor | `test_killable_survivors_die_by_the_named_test` |
| "Killing" a survivor with a test that rejects the reference (asserting an implementation detail, or your own bug) | every mutant dies for the wrong reason | `test_suite_accepts_the_reference` |
| Stopping at the rung's threshold | the 20% that survived R5 stays untested | `test_suite_kills_every_l45_mutant` |
| A hand-written mutant that is equivalent, or that copies a committed one | it teaches nothing and inflates the count | `test_your_mutants_are_real_and_new` |
| Writing a mutant and never checking your suite against it | a pitfall you know is still untested | `test_your_suite_kills_your_mutants` |
| Expected values printed by your own code | the test agrees with your bug, and the reference rejects it | `test_suite_accepts_the_reference` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `craft.03` | killed and survived, baselines A and B, required mutants |
| Back | `craft.04` | laws as tests: a property often kills a whole family of survivors at once |
| Back | `L4.5` | the unit under test and the pitfalls your mutants come from |
| Forward | `craft.05` | oracles and differential tests (R5) for L5 to L7, graded with the same survivor reading |
| Forward | `craft.06` | perf mutants (R7): a survivor is a slowdown your benchmark gate did not trip |
| Forward | `craft.21` | resilience mutants (R10): a survivor is a fault your kill loop did not expose |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| reading survivors | Google's mutation testing in code review | mutants surfaced as review comments on changed lines only, with heuristics that suppress likely-equivalent ("arid") nodes | Petrovic et al., "Practical Mutation Testing at Scale" (2021) |
| your triage | trivial compiler equivalence (TCE) | compiles both versions and calls a mutant equivalent when the machine code is identical, which removes a share of equivalent mutants for free | Papadakis et al., ICSE 2015 |
| the course's mutants | [mutmut](https://github.com/boxed/mutmut), [cosmic-ray](https://github.com/sixty-north/cosmic-ray), [cargo-mutants](https://mutants.rs/), [Stryker](https://stryker-mutator.io/) | generated mutants with incremental runs and caches | each tool's documentation |
| semantic mutants | higher-order and subsuming mutants | combinations of mutants that are harder to kill than any single one; the minimal subsuming set | Jia and Harman, "Higher Order Mutation Testing" (2009) |
