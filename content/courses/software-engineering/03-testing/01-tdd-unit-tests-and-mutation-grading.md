# TDD, unit tests, and mutation testing

Every component you build in *AI Engineering*, starting with [Losses with a fused backward](../../ai-engineering/06-numerics-and-autograd/04-losses-with-fused-backward.md), arrives with tests of its own. "They pass" is not what makes those tests good: a test that passes on everything proves nothing. Three ideas decide whether a test suite is worth reading. The first is how a test is written before the code it tests. The second is what it means for a test to "kill a mutant." The third is why tests should touch only the contract and nothing behind it.

This section shows all three on a kata small enough to read in one sitting and useful enough to keep: the failpoint spec that the `tinyllm` CLI evaluates after every training step. With it, the end-to-end run can kill a training run at step 60 and check that it resumes. You will read the kata line by line, then read a model test suite for it, and see which planted fault each test exists to catch. As you go, try to predict, for each line of the kata, the test that would fail if the line were wrong.

The section assumes you have the repository and its checks from [The repository and its CI gate](../06-code-review-and-ci/01-your-repo-and-ci-gate.md), and that you are comfortable with Python and `uv` from [Python and numpy](../../programming-and-algorithms/01-language-primers/01-python-and-numpy.md) in *Programming and Algorithms*. The kata and its tests run with nothing but pytest:

```bash
uv run --no-project --with pytest python -m pytest -q primers/craft.03
```

> **Learning objectives**
> - Write a test first, watch it fail for the expected reason, and make it pass with the least code.
> - Define a mutant, explain killed versus survived, and compute a mutation score.
> - Explain why a suite must pass on a correct implementation before its mutation score means anything.
> - Write black-box tests that use only the names a spec defines.
> - Name the planted faults in the failpoint kata and the test that kills each one.

## A test is a falsifiable claim

Consider `assert parse("x=2*crash") == {"x": Rule("crash", "", 2)}`. It claims one fact about the spec, and its value is the set of wrong implementations it rejects. That framing immediately rules out a common habit. A test that asserts what the code currently returns (run it, paste the output) rejects nothing the code already does wrong, because if the code is wrong the test now agrees with it. The number in an assertion should come from the spec and hand arithmetic, never from the implementation.

Once you think of a test as a claim that rejects wrong implementations, the natural way to measure a suite is to hand it wrong implementations and see which it rejects. That is mutation testing.

## Mutation testing

A **mutant** is the correct implementation with one small, plausible fault planted: a `<` where the spec needs `!=`, a deleted check, a constant off by one. The tests run against it. If any test fails, the mutant is **killed**; if all pass, it **survived**, and some behavior of the spec is pinned down by none of the tests. The **score** is killed divided by total.

Mutants come in two kinds. **Auto** mutants come from mechanical operators: flip a comparison, delete a statement, replace a constant, swap arguments. **Semantic** mutants are written by hand from a component's known pitfalls, one per pitfall: the bugs people actually make. The distinction matters because the score is an average and averages hide things. A suite that lets a semantic mutant survive misses a known bug, whatever its overall score.

Running every mutant means running the suite once per mutant, so the cost is real. Practical tools sample mutants while you develop and run the full set when the tests change, caching the result by the hash of the tests and the code.

### Two baselines come first

Before any mutant runs, the tests must pass against a correct implementation. If they do not, they reject correct code, and the score means nothing: a test that fails on correct code also fails on every mutant, so every mutant counts as killed for the wrong reason. A test that fails on correct code is a wrong test, not a strict one.

### Black-box tests

A test suite is only reusable if it touches only what the contract (here, the spec) promises: its names and their documented behavior. A test that reads a private attribute such as `fp.counts`, or imports an internal helper, breaks on any other correct implementation. Black-box tests are what let one suite judge anyone's implementation, including the mutants.

### Red then green

Test-driven development adds a discipline about order. Write one test for the next behavior, run it, and watch it fail **for the reason you expect**: that is red. Write the least code that makes it pass: that is green. Then write the next test. Seeing a test fail is the only proof that it can fail. A test never seen failing may pass for the wrong reason, a wrong import or an assertion that is always true, and would kill no mutant.

## The failpoint spec

With those ideas in hand, here is the kata they will be applied to. A string such as

```
TL_FAILPOINTS="train/after-step=60*crash; data/fetch=error(timeout)"
```

names places where a test wants a fault. An entry is `name=[N*]action`. The actions are `crash`, `panic`, `off`, `error[(msg)]`, and `sleep(duration)` with a duration such as `50ms` or `2s`. A prefix `N*` makes the rule fire on the Nth evaluation of that name only; without it a rule fires every time; `off` never fires and is never counted. Whitespace around names and actions is ignored and empty entries (a trailing `;`) are skipped; everything malformed is a `ValueError`.

The kata parses the spec and decides, evaluation by evaluation, whether a failpoint fires. What to do then (exit 137, raise, sleep) is the caller's job.

## Working one cycle by hand

Before reading the full kata, work one red-green cycle and one planted fault by hand.

Start with red. The first behavior to pin down is that `N*` fires on the Nth evaluation only:

```python
def test_nth_fires_once_on_the_nth_evaluation():
    fp = Failpoints("x=2*crash")
    assert [fp.evaluate("x") is not None for _ in range(4)] == [False, True, False, False]
```

Against a stub it fails with `NotImplementedError`: red, for the expected reason.

Now green. The least code that passes is to count evaluations per name and fire when the count equals `nth` (or always, when `nth` is 0). The test passes.

Next, plant a fault. Change `n != rule.nth` to `n < rule.nth` in the "do not fire yet" check, so the rule fires on the Nth evaluation **and every one after**. Against this mutant the four evaluations give `[False, True, True, True]`, the test above fails, and the mutant is killed. A weaker test that checks only the first two evaluations (`None`, then a rule) passes on the mutant, so the mutant survives that test.

Finally, the score. The kata has 13 planted faults, listed in [What can go wrong](#what-can-go-wrong). A suite that kills 10 scores $10/13 = 0.77$. If one of the 3 survivors is an `off` rule counted and returned as fired, the suite still misses the most dangerous bug in the spec: a disabled failpoint that crashes the run. That is exactly why semantic mutants are reported on their own and not folded into the average.

## The interface

The kata exposes five names, and the standard library is enough to implement them:

```python
ACTIONS = ("crash", "panic", "error", "sleep", "off")

@dataclass(frozen=True)
class Rule:
    action: str  # one of ACTIONS
    arg: str     # error's message or sleep's duration as written; "" otherwise
    nth: int     # 0: fire on every evaluation; N >= 1: on the Nth only

def sleep_seconds(duration: str) -> float: ...       # "50ms" -> 0.05
def parse(spec: str) -> dict[str, Rule]: ...          # ValueError on any malformed entry

class Failpoints:
    def __init__(self, spec: str) -> None: ...
    def evaluate(self, name: str) -> Optional[Rule]: ...   # the Rule when it fires now, else None
    def count(self, name: str) -> int: ...
```

A correct solution has to handle six groups of cases, summarized here:

| Case | Kind of check | Checks |
|---|---|---|
| the hand-worked spec | unit | the spec parses to three rules; whitespace and a trailing `;` are fine |
| the hand-worked evaluations | unit | `3*crash` fires on the third of five evaluations only; `off` and unknown names are never counted |
| counts per name | unit | each name counts its own evaluations |
| sleep durations | unit | `50ms` is 0.05 s; malformed durations are errors |
| malformed specs | boundary | nine malformed specs, including `a=b=crash` and `a=0*crash` |
| every action | unit | the five actions and error's trimmed message |

A test suite for the kata has its own requirements: it imports only the spec's names, passes on a correct implementation, and fails on each planted fault.

## Building the kata step by step

The design follows from seven decisions, each forced by the spec or by a failure you want to avoid.

1. **Separate parsing from evaluating.** A spec is parsed once into a table; evaluation is a lookup plus a counter. Errors in the spec then surface at startup, not at step 60.
2. **Split each entry at the first `=`.** A name never contains `=`, so the first one separates name from action. Splitting at the last would turn `a=b=crash` into a failpoint called `a=b` instead of an error.
3. **Recognize the action with one anchored pattern.** Match `[N*]word[(arg)]`, anchored at both ends, then check the word against `ACTIONS`. Everything that does not match is an error naming the entry.
4. **Validate everything the grammar allows but the meaning forbids.** `N` counts from 1; `crash`, `panic`, and `off` take no argument; `sleep` needs a duration that parses. Check these at parse time so the caller never sees a half-valid rule.
5. **Treat duplicates as errors.** A name given twice is almost always a typo; silently letting the later entry win would disable a failpoint.
6. **Count per name, and only what can fire.** `off` and unknown names return `None` before the counter moves, so `count` reports evaluations that could have fired.
7. **Fire on exactly the Nth.** With `nth` set, fire when the count equals it, never before or after.

Durations come first, because `parse` uses them for validation:

```python
_ENTRY = re.compile(r"^(?:(\d+)\s*\*\s*)?([a-z]+)\s*(?:\((.*)\))?$")   # [N*] word [(arg)]
_DURATION = re.compile(r"^(\d+(?:\.\d+)?)(ms|s)$")


def sleep_seconds(duration: str) -> float:
    """"50ms" -> 0.05, "2s" -> 2.0, "1.5s" -> 1.5. ValueError otherwise."""
    m = _DURATION.match(duration.strip())
    if not m:
        raise ValueError(f"duration {duration!r}: want digits then ms or s, like 50ms or 2s")
    value = float(m.group(1))
    return value / 1000.0 if m.group(2) == "ms" else value   # the unit is mandatory
```

Requiring the unit means `sleep(50)` is an error rather than 50 seconds or 50 milliseconds by guess.

The parser applies decisions 2 through 5 in order:

```python
def parse(spec: str) -> dict[str, Rule]:
    table: dict[str, Rule] = {}
    for raw in spec.split(";"):
        entry = raw.strip()
        if not entry:
            continue                                   # a trailing ";" is fine
        name, eq, act = entry.partition("=")  # the FIRST "=": a name never holds one
        name = name.strip()                            # " a " and "a" are the same failpoint
        if not eq or not name:
            raise ValueError(f"failpoint {entry!r}: want name=[N*]action")
        if name in table:
            raise ValueError(f"failpoint {name!r} is given twice")   # step 5
        m = _ENTRY.match(act.strip())
        if not m or m.group(2) not in ACTIONS:
            raise ValueError(f"failpoint {entry!r}: unknown action (one of {', '.join(ACTIONS)})")
        nth = 0
        if m.group(1) is not None:
            nth = int(m.group(1))
            if nth < 1:
                raise ValueError(f"failpoint {entry!r}: N in N* counts from 1")   # 0* is meaningless
        action, arg = m.group(2), m.group(3)
        if arg is not None and action in ("crash", "panic", "off"):
            raise ValueError(f"failpoint {entry!r}: {action} takes no argument")
        if action == "sleep":
            if arg is None:
                raise ValueError(f"failpoint {entry!r}: sleep needs a duration, like sleep(50ms)")
            sleep_seconds(arg)                         # validate now, convert later
        table[name] = Rule(action, (arg or "").strip(), nth)   # error( boom ) -> "boom"
    return table
```

With `a=b=crash`, `partition` gives the name `a` and the action `b=crash`, which `_ENTRY` does not match: an error, as the spec requires. The `Rule` keeps the duration as written; `sleep_seconds` converts it when the caller sleeps.

Evaluation is then a lookup and a counter, applying decisions 6 and 7:

```python
class Failpoints:
    def __init__(self, spec: str) -> None:
        self.table = parse(spec)
        self.counts: dict[str, int] = {}               # one counter per name (step 6)

    def evaluate(self, name: str) -> Optional[Rule]:
        rule = self.table.get(name)
        if rule is None or rule.action == "off":
            return None                                # never fires, never counted
        n = self.counts.get(name, 0) + 1
        self.counts[name] = n
        if rule.nth and n != rule.nth:
            return None                                # exactly the Nth (step 7)
        return rule

    def count(self, name: str) -> int:
        return self.counts.get(name, 0)                # a name never evaluated has count 0
```

The full solution is [`failpoints.py`](../../../../course/solutions/primers/craft.03/failpoints.py).

## A model test suite

The kata is half the lesson; the suite that judges it is the other half. Each test below derives its expected values from the spec, uses only the five public names, and is aimed at specific faults. Run against the kata it passes; run against each of the 13 planted faults it fails at least once.

```python
import pytest
from failpoints import ACTIONS, Failpoints, Rule, parse, sleep_seconds


def fired(fp, name, times):
    return [fp.evaluate(name) is not None for _ in range(times)]


def test_parse_reads_every_entry():
    t = parse("train/after-step=60*crash; data/fetch=error(timeout); x=sleep(50ms)")
    assert t == {
        "train/after-step": Rule("crash", "", 60),
        "data/fetch": Rule("error", "timeout", 0),
        "x": Rule("sleep", "50ms", 0),
    }


def test_whitespace_and_empty_entries_are_ignored():     # kills: names keep spaces; ";;" rejected
    assert parse("  a = 2 * crash ;; b=off; ") == {"a": Rule("crash", "", 2), "b": Rule("off", "", 0)}


def test_every_action_parses_and_error_message_is_trimmed():   # kills: strip() dropped
    t = parse("a=crash; b=panic; c=off; d=error; e=error( boom ); f=sleep(2s)")
    assert {r.action for r in t.values()} == set(ACTIONS)
    assert t["d"].arg == "" and t["e"].arg == "boom"


def test_nth_fires_once_on_the_nth_evaluation():         # kills: "from the Nth on"
    fp = Failpoints("x=2*crash")
    assert fired(fp, "x", 4) == [False, True, False, False]


def test_without_n_fires_every_time():
    fp = Failpoints("x=panic")
    assert fired(fp, "x", 3) == [True, True, True]
    assert fp.count("x") == 3


def test_off_never_fires_and_is_never_counted():         # kills: off counted or fired
    fp = Failpoints("x=off")
    assert fired(fp, "x", 3) == [False, False, False]
    assert fp.count("x") == 0


def test_unknown_names_are_not_counted():                # kills: count() defaults to 1
    fp = Failpoints("x=crash")
    assert fp.evaluate("y") is None
    assert fp.count("y") == 0 and fp.count("x") == 0


def test_counts_are_per_name():                          # kills: one shared counter
    fp = Failpoints("a=2*crash; b=2*crash")
    assert fp.evaluate("a") is None
    assert fp.evaluate("b") is None  # b's first evaluation, not the second overall
    assert fp.evaluate("a") is not None
    assert (fp.count("a"), fp.count("b")) == (2, 1)


def test_sleep_durations():                              # kills: ms not converted
    assert sleep_seconds("50ms") == pytest.approx(0.05)
    assert sleep_seconds("2s") == 2.0
    assert sleep_seconds("1.5s") == 1.5
    for bad in ["50", "ms", "5 m", "-1s", "1.s"]:
        with pytest.raises(ValueError):
            sleep_seconds(bad)


@pytest.mark.parametrize(
    "spec",
    [
        "a",                # no "="
        "=crash",           # empty name
        "a=b=crash",        # "=" inside what would be the name: kills split-at-last
        "a=crash; a=off",   # a name given twice: kills later-wins
        "a=explode",        # unknown action
        "a=0*crash",        # N counts from 1: kills 0* accepted
        "a=crash(now)",     # crash takes no argument
        "a=off(1)",         # nor does off
        "a=sleep",          # sleep needs a duration: kills the parse-time check removed
        "a=sleep(5m)",      # and a valid one
    ],
)
def test_parse_rejects(spec):
    with pytest.raises(ValueError):
        parse(spec)
```

Notice what the suite does not do. It never reads `fp.counts` or `fp.table`, so it passes on any correct implementation. Each expected list such as `[False, True, False, False]` was written from the spec before running anything. `test_counts_are_per_name` uses the same `N` on two names, because only then does a shared counter change the outcome. And `test_without_n_fires_every_time` kills no planted fault on its own, but it pins down the default behavior so a later change cannot break it unnoticed.

Model test suites written the same way accompany the next components in *AI Engineering*: one for [the losses](../../../../course/solutions/learner-tests/L0.3/python/tests/l0-3-losses/) and one for [the module system](../../../../course/solutions/learner-tests/L0.4/python/tests/l0-4-module/).

## Why this design

The kata is cheap where it needs to be. `parse` is linear in the spec's length, and `evaluate` is one dictionary lookup and one increment, cheap enough to call after every training step.

It parses strictly rather than leniently. A lenient parser that ignores bad entries makes a typo in an environment variable silently disable a fault injection, and the test that relied on it passes for the wrong reason. Strict errors at startup cost nothing in a correct spec.

It is judged by mutation testing rather than coverage. Line coverage says a line ran; it does not say any assertion would notice if the line were wrong. The planted faults measure exactly that: every fault below sits on a line a weak suite would cover and still miss.

One tradeoff is accepted deliberately. The kata leaves out the probability form (`10%crash`) that real failpoint libraries have, because it would make evaluation non-deterministic and the tests harder to write by hand.

## What can go wrong

These are the bugs a test suite of the kata must catch, one planted fault each, with the line of the solution that avoids it and the test that kills it:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. an `off` rule counted, or returned as fired | `x=off` still crashes the run | `evaluate` returns `None` for `off` before counting; caught by `test_off_never_fires_and_is_never_counted` |
| 2. `N*` read as "from the Nth on", or `0*` accepted | a run killed at step 60 is killed again at every later step; `0*crash` never or always fires | `n != rule.nth` and `nth < 1` raises; caught by the Nth test and `a=0*crash` |
| 3. splitting at the last `=`, or accepting an argument on `crash`, `panic`, `off` | `a=b=crash` becomes a failpoint named `a=b` | `partition` at the first `=`, and an explicit argument check; caught by `a=b=crash`, `a=crash(now)`, `a=off(1)` |
| 4. one counter shared by every name | "fire on the 2nd train step" depends on unrelated failpoints | `self.counts` keyed by name; caught by `test_counts_are_per_name` |
| 5. durations: `ms` not converted, or `sleep` without one accepted | a 50 ms sleep lasts 50 s and times the suite out | `value / 1000.0` for `ms`, and `sleep` requires an argument; caught by `test_sleep_durations` and `a=sleep` |
| 6. whitespace kept, or a trailing `;` rejected | specs typed by hand into an env var fail or name the wrong point | `strip()` on entry and name, empty entries skipped; caught by `test_whitespace_and_empty_entries_are_ignored` |
| 7. a name given twice: the later entry wins | a typo silently disables a failpoint | `name in table` raises; caught by `a=crash; a=off` |
| error's message keeps its surrounding spaces | `error( boom )` reports `" boom "` | `(arg or "").strip()`; caught by the trimmed-message assertion |
| `count` of a name never evaluated is 1 | callers see evaluations that never happened | default 0 in `count`; caught by `test_unknown_names_are_not_counted` |

The suite itself can be wrong in ways that let faults survive. Four testing mistakes account for most of them:

| Mistake | What mutation testing shows | Fix |
|---|---|---|
| expected values pasted from the code's output | the fault was in the code, so the test agrees with it: mutants survive | derive each expected value from the spec, by hand |
| tests that reach into private state (`fp.counts`) | the suite fails on a correct implementation that keeps its counters elsewhere | assert only through the spec's names |
| a test never seen red | it may assert nothing (a typo in a name, a loop that never runs) | run it before the code exists |
| one giant test | the first failure hides the rest; survivors are hard to place | one behavior per test, named for it |

> **Warning:** A high mutation score with a surviving semantic mutant is still a failing suite. The `off` pitfall in particular turns a disabled failpoint into a crash in the run you meant to leave alone.

## Where this leads

The repository's CI gate from [The repository and its CI gate](../06-code-review-and-ci/01-your-repo-and-ci-gate.md) runs every test suite on each change, and `uv run` with pytest is the runner throughout. Going forward, the model test suites for [Losses with a fused backward](../../ai-engineering/06-numerics-and-autograd/04-losses-with-fused-backward.md) and [Module system and basic layers](../../ai-engineering/06-numerics-and-autograd/05-module-system-and-layers.md) are written the way this section explains, the second one test first. The failpoint spec itself is used by the first end-to-end run: the CLI evaluates `train/after-step` with it, so a run can be killed at step 60 and resumed.

Within this chapter, the next sections build directly on these ideas: [Property-based tests](02-property-based-tests.md), [Mutation testing in depth](05-mutation-testing-in-depth.md), and [Oracles, golden and differential tests, gradcheck as a test](03-oracles-golden-and-gradcheck.md).

Each piece of the kata has a production equivalent worth knowing:

- Planted faults and a score are what PIT (Java), `mutmut` and `cosmic-ray` (Python), and `cargo-mutants` (Rust) generate over a whole codebase, with incremental runs on changed lines (`pitest.org`, `github.com/sourcefrog/cargo-mutants`).
- Semantic mutants from pitfalls correspond to Google's mutation testing in code review, which surfaces mutants as review comments on changed lines only and suppresses unproductive ones (Petrović et al., ICSE-SEIP 2021).
- The failpoint spec mirrors the TiKV `fail` and `failpoint` crates and etcd's `gofail`, which compile failpoints out of release builds and add actions such as `return(value)` and `pause` (`github.com/tikv/fail-rs`).
- Red then green, enforced, becomes a minimum mutation score per module gated on pull requests, as in the Stryker dashboard.

### Further reading

- Kent Beck, *Test-Driven Development: By Example*.
- Petrović et al., "Practical Mutation Testing at Scale" (Google, 2021).
- The PIT and `mutmut` documentation.

## Key terms

- **Mutant**: the correct implementation with one small, plausible fault planted.
- **Killed / survived**: a mutant is killed when at least one test fails against it, and survives when every test passes.
- **Mutation score**: the number of killed mutants divided by the total.
- **Semantic mutant**: a hand-written mutant that encodes one known pitfall.
- **Black-box test**: a test that uses only the names and documented behavior of the contract.
- **Red then green**: write a failing test, see it fail for the expected reason, then write the least code that passes it.
- **Failpoint**: a named place in the code where a test can inject a crash, error, or delay.

## Summary

- A test is worth what it catches: run it against a correct implementation with one planted fault, and it should fail.
- A test that rejects a correct implementation is worse than none, because every fault would then "die" for the wrong reason; passing on correct code comes first.
- Good tests are black-box: they use only the names the spec defines, which is what lets them run against anyone's correct implementation.
- A test written before the code fails first, and seeing it fail is the only proof it can fail.
- The mutation score is killed over total, but a suite that misses a semantic mutant is inadequate whatever its score.

## Review questions

1. Why does `evaluate` return early for `off` before incrementing the counter, rather than after?

   <details><summary>Answer</summary>

   The spec says `off` is never counted. Returning after incrementing would make `count("x")` grow for a disabled failpoint; returning `rule` instead of `None` would fire it. A test that only checks `evaluate` returns `None` would not catch the counting fault; `test_off_never_fires_and_is_never_counted` checks both.

   </details>

2. A suite passes on the correct kata and on every mutant. What does that say?

   <details><summary>Answer</summary>

   It constrains nothing the mutants change: every fault leaves its assertions true. Typical causes are expected values pasted from the code's output, tests that only check a value is not `None`, or tests that were never seen failing.

   </details>

3. Why must a test suite pass on a correct implementation before mutants mean anything?

   <details><summary>Answer</summary>

   A test that fails on correct code fails on every mutant too, so every mutant counts as "killed" and the score is 100% for the wrong reason. Passing on correct code first shows that failures on a mutant are caused by the fault.

   </details>

4. Why does `test_counts_are_per_name` evaluate `b` between two evaluations of `a`?

   <details><summary>Answer</summary>

   With one shared counter, `b`'s first evaluation would be the second overall and fire. The order a, b, a separates per-name counting from shared counting, which a test that evaluates one name at a time cannot.

   </details>

5. Why split at the first `=` and not the last?

   <details><summary>Answer</summary>

   Names never contain `=`, so the first `=` is always the separator, and `a=b=crash` must be an error. Splitting at the last `=` would accept it as a failpoint named `a=b`, and the caller evaluating `a` would never see it fire.

   </details>
