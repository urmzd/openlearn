<!-- ol:module craft.03 -->
# TDD, unit tests, and how your tests are graded

## Overview

| | |
|---|---|
| **Module** | `craft.03` · practice · Python · Pass 2 · 2 to 3 h |
| **You build** | `primers/craft.03/failpoints.py` (a kata: the failpoint spec your CLI evaluates in `MS-L0`) and `primers/craft.03/test_failpoints.py`, your tests for it, written first |
| **Contract** | the spec in the kata's docstring (`ol start craft.03` writes it with the stubs) |
| **Tests** | `course/tests/craft.03/`: the course's tests of your kata, and the grade of your tests by mutation (section 4) |
| **Needs** | reading: [`craft.01`](../08-code-review-and-ci/01-your-repo-and-ci-gate.md) your repo and CI gate · [`lang.01`](../12-language-and-tool-primers/01-python-and-numpy.md) Python and uv |
| **Used by** | no call site (a practice): the rungs it teaches grade your tests from [`L0.3`](../../ml/08-tinyllm/p00-foundations/03-losses-with-fused-backward.md) (R2) and [`L0.4`](../../ml/08-tinyllm/p00-foundations/04-module-system-and-layers.md) (R3) on |
| **Milestone** | `MS-P2` (the Pass 2 gate requires a passing `craft.03`) |
| **Optional depth** | Kent Beck, *Test-Driven Development: By Example*; Petrović et al., "Practical Mutation Testing at Scale" (Google, 2021); the PIT and `mutmut` documentation |

## Key Takeaways

- A test is graded by what it catches: it runs against a correct implementation with one planted fault, and it must fail (`test_your_tests_kill_the_planted_faults`).
- A test that rejects a correct implementation is worse than none: every fault would then "die" for the wrong reason, so that baseline comes first (`test_your_tests_accept_the_course_kata`).
- Graded tests are black-box: they use only the names the spec defines, which is what lets them run against anyone's correct implementation (`test_your_tests_import_only_the_spec`).
- Red then green: a test written before the code fails first; seeing it fail is the only proof it can fail. `ol tdd` records that for rung R3 (`L0.4`).
- The score is killed over total, with a threshold per rung (0.60 at R2, 0.70 at R3) and required faults that must die whatever the score.

## How to work this chapter

```bash
ol start craft.03          # writes primers/craft.03/failpoints.py (stubs and the spec)
# write ONE test in primers/craft.03/test_failpoints.py; see it fail:
uv run --no-project --with pytest python -m pytest -q primers/craft.03
# write the least code that makes it pass; repeat, test first, until the spec is covered
ol tests craft.03          # the course's tests of the kata, and how your tests are graded
ol check craft.03          # your code against the course tests, then your tests against planted faults
```

---

## 1. Why now

Until now every test in your repo came from the course (rung R0: you read them). From `L0.3` on you write tests that the course grades, and the grade is not "do they pass": a test that passes on everything proves nothing. Before the first graded module you need to know three things: how to write a test before the code it tests, what "your test killed a mutant" means, and why your tests may touch only the contract. This module teaches them on a kata small enough to finish in an evening and useful enough to keep: the failpoint spec that your `tinyllm` CLI evaluates after every training step, so `MS-L0` can kill a run at step 60 and check that it resumes.

## 2. Principles

**The rungs (DESIGN 5.12).** The course climbs a ladder of what it gives you and what you write:

| Rung | You get | You write | Graded by | First at |
|---|---|---|---|---|
| R0 Read | the full annotated suite | nothing | the course tests pass | tracer, `L0.1`, `L0.2` |
| R1 Fill the oracle | tests with the expected values left out | the expected values, by hand | your values against hidden ones | the habit of `L0.3`'s hand values |
| R2 Name-given cases | test names and what each must show | the bodies | mutation, 0.60 | `L0.3` |
| R3 Red then green | the interface and one test | the tests first, then the code | mutation, 0.70 and one required fault, plus the journal | `L0.4`, this kata |

**A test is a falsifiable claim.** `assert parse("x=2*crash") == {"x": Rule("crash", "", 2)}` claims one fact about the spec. Its value is the set of wrong implementations it rejects. A test that asserts what the code currently returns (run it, paste the output) rejects nothing the code already does wrong; the number in an assertion should come from the spec and your own arithmetic.

**Mutation testing.** A **mutant** is the correct implementation with one small, plausible fault planted: a `<` where the spec needs `!=`, a deleted check, a constant off by one. Your tests run against it. If any test fails, the mutant is **killed**; if all pass, it **survived**, and some behavior of the spec is pinned down by none of your tests. The **score** is killed divided by total. Two kinds of mutant: **auto** ones from mechanical operators (flip a comparison, delete a statement, replace a constant, swap arguments) and **semantic** ones written from a chapter's pitfalls, one per pitfall: the bugs people actually make. A semantic mutant can be **required**: it must be killed whatever the score. A surviving semantic mutant is shown only as "a planted bug from Pitfall N survived" until you pass, so you look for the behavior, not the patch.

**Two baselines.** Before any mutant runs: (A) your tests must pass against the correct implementation, the course's reference, or they reject correct code and the score means nothing; (B) your tests must pass against your own implementation. A test that fails A is a wrong test; one that fails B is unfinished code.

**Black-box.** Graded tests run against the reference with a fault planted, never against your code, so they may use only what the contract (here, the spec) promises: its names and their documented behavior. A test that reads a private attribute (`fp.counts`) or imports a helper of yours breaks on any other correct implementation.

**Red then green.** Write one test for the next behavior, run it, and watch it fail **for the reason you expect** (red). Write the least code that makes it pass (green). Then the next test. A test you never saw fail may pass for the wrong reason, a wrong import or an assertion that is always true, and would kill no mutant. `ol tdd red <ID>` records a failing run of your current test files and `ol tdd green <ID>` a passing one with the same files; from R3 `ol check` requires a red record for each test file before its green.

**Cost.** `ol check` runs the required mutants plus a seeded sample of 8 and reports an estimate; `ol mutate` computes the full grade once and caches it by the hash of your test files, the reference unit, and the patch, so the full cost is paid only when your tests change.

**The kata's spec.** `TL_FAILPOINTS="train/after-step=60*crash; data/fetch=error(timeout)"` names places where a test wants a fault. An entry is `name=[N*]action`; actions are `crash`, `panic`, `off`, `error[(msg)]`, and `sleep(duration)` with a duration such as `50ms` or `2s`. `N*` fires on the Nth evaluation of that name only; without it a rule fires every time; `off` never fires and is never counted. Whitespace around names and actions is ignored and empty entries (a trailing `;`) are skipped; everything malformed is a `ValueError`. Your kata parses the spec and decides, evaluation by evaluation, whether a failpoint fires; what to do then (exit 137, raise, sleep) is the caller's job.

## 3. Worked example by hand

One red-green cycle, then a mutant.

1. **Red.** The first behavior: `N*` fires on the Nth evaluation only.

    ```python
    def test_nth_fires_once_on_the_nth_evaluation():
        fp = Failpoints("x=2*crash")
        assert [fp.evaluate("x") is not None for _ in range(4)] == [False, True, False, False]
    ```

    Against the stub it fails with `NotImplementedError`: red, for the expected reason.

2. **Green.** The least code: count evaluations per name and fire when the count equals `nth` (or always, when `nth` is 0). The test passes.

3. **A planted fault.** The mutant `s02` changes `n != rule.nth` to `n < rule.nth` in the "do not fire yet" check, so it fires on the Nth evaluation **and every one after**. Against it the four evaluations give `[False, True, True, True]`: the test above fails, so `s02` is killed. A weaker test that checks only the first two evaluations (`None`, then a rule) passes on the mutant: `s02` survives it.

4. **The score.** The kata has 13 planted faults. Suppose your tests kill 10: the score is $10/13 = 0.77 \ge 0.70$. If the required one, `s01` (an `off` rule counted and returned as fired), is among the 3 survivors, the grade still fails: required faults must die whatever the score.

The course's own version of this example is `test_hand_example_evaluations` in `course/tests/craft.03/`.

## 4. The artifact and its check

Two files in your repo, both yours:

- `primers/craft.03/failpoints.py`: `ACTIONS`, `Rule(action, arg, nth)`, `sleep_seconds(duration)`, `parse(spec)`, and `Failpoints(spec)` with `evaluate(name)` and `count(name)`. `ol start craft.03` writes the stubs and the spec; the standard library is enough.
- `primers/craft.03/test_failpoints.py`: your tests, at least one per rule of the spec, importing only those five names from `failpoints` (plus pytest).

`ol check craft.03` runs `course/tests/craft.03/check` in your repo, which runs two annotated files:

1. `test_craft03_kata.py`: the course's tests of **your** code (rung R0 for the kata: read them as exemplars).
2. `test_craft03_grading.py`: the grade of **your tests**: (a) they import only the spec's names; (b) they pass on your kata; (c) they pass on the course's kata (baseline A); (d) for each planted fault in `course/mutants/craft.03/`, one at a time, they fail. Pass at a score of 0.70 with the required fault killed.

| Test | KIND | Checks |
|---|---|---|
| `test_hand_example_spec` | unit | section 2's spec parses to three rules; whitespace and a trailing `;` are fine |
| `test_hand_example_evaluations` | unit | `3*crash` fires on the third of five evaluations only; `off` and unknown names are never counted |
| `test_counts_are_per_name` | unit | each name counts its own evaluations |
| `test_sleep_durations` | unit | `50ms` is 0.05 s; malformed durations are errors |
| `test_parse_rejects` | boundary | nine malformed specs, including `a=b=crash` and `a=0*crash` |
| `test_every_action_parses` | unit | the five actions and error's trimmed message |
| `test_your_tests_import_only_the_spec` | unit | your tests use only the spec's names, and there are at least three |
| `test_your_tests_pass_on_your_kata` | unit | green on your code |
| `test_your_tests_accept_the_course_kata` | unit | baseline A: green on a correct implementation |
| `test_your_tests_kill_the_planted_faults` | unit | score at least 0.70, the required fault killed; survivors listed |

The same machinery grades your tests from here on: `ol mutate L0.3` (rung R2: the names come from the chapter), then `ol tdd red L0.4` / `ol tdd green L0.4` and `ol check L0.4` (rung R3: the journal is enforced).

## 5. Pitfalls

The bugs your tests must catch, one planted fault each (the semantic mutants of the kata):

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. an `off` rule counted, or returned as fired | `x=off` still crashes the run | `test_hand_example_evaluations` (mutant `s01`) |
| 2. `N*` read as "from the Nth on", or `0*` accepted | a run killed at step 60 is killed again at every later step; `0*crash` never or always fires | `test_hand_example_evaluations` (mutant `s02`), `test_parse_rejects` (mutant `s07`) |
| 3. splitting at the last `=`, or accepting an argument on `crash`, `panic`, `off` | `a=b=crash` becomes a failpoint named `a=b` | `test_parse_rejects` (mutants `s03`, `s09`) |
| 4. one counter shared by every name | "fire on the 2nd train step" depends on unrelated failpoints | `test_counts_are_per_name` (mutant `s08`) |
| 5. durations: `ms` not converted, or `sleep` without one accepted | a 50 ms sleep lasts 50 s and times the suite out | `test_sleep_durations` (mutant `s05`), `test_parse_rejects` (mutant `s11`) |
| 6. whitespace kept, or a trailing `;` rejected | specs typed by hand into an env var fail or name the wrong point | `test_hand_example_spec` (mutants `s04`, `s10`) |
| 7. a name given twice: the later entry wins | a typo silently disables a failpoint | `test_parse_rejects` (mutant `s06`) |

And the testing mistakes the grade punishes:

| Mistake | What the grade shows | Fix |
|---|---|---|
| expected values pasted from the code's output | the fault was in the code, so the test agrees with it: mutants survive | derive each expected value from the spec, by hand |
| tests that reach into private state (`fp.counts`) | baseline A fails: a correct implementation keeps its counters elsewhere | assert only through the spec's names |
| a test never seen red | it may assert nothing (a typo in a name, a loop that never runs) | run it before the code exists |
| one giant test | the first failure hides the rest; survivors are hard to place | one behavior per test, named for it |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `craft.01` | your repo and its CI gate run `ol check --all --ci`, which includes this check |
| Back | `lang.01` | `uv run` and pytest |
| Forward | `L0.3` | rung R2: your tests of the losses, graded by mutation at 0.60 |
| Forward | `L0.4` | rung R3: red-then-green journal (`ol tdd`), 0.70, one required fault |
| Forward | `MS-L0` | your CLI evaluates `train/after-step` with this spec, so the milestone can kill and resume a run |

Later rungs build on it: properties (R4, `craft.04`), mutation testing in depth (`craft.07`), oracles and gradchecks (R5, `craft.05`).

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| planted faults and a score | PIT (Java), `mutmut` and `cosmic-ray` (Python), `cargo-mutants` (Rust) | operators generated over a whole codebase, incremental runs on changed lines | `pitest.org`, `github.com/sourcefrog/cargo-mutants` |
| semantic mutants from pitfalls | Google's mutation testing in code review | mutants surfaced as review comments on the changed lines only, with suppression of unproductive ones | Petrović et al., ICSE-SEIP 2021 |
| the failpoint spec | FreeBSD/TiKV `fail` and `failpoint` crates, etcd `gofail` | failpoints compiled out in release builds, actions such as `return(value)` and `pause` | `github.com/tikv/fail-rs` |
| red then green | Beck's TDD cycle, mutation-score gates in CI | a minimum score per module enforced on pull requests | Stryker dashboard |
