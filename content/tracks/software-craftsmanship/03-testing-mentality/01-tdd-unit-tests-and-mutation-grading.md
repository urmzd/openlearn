# TDD, unit tests, and mutation testing

## Overview

| | |
|---|---|
| **Module** | `craft.03` · practice · Python · Pass 2 · 1 to 1.5 h reading |
| **Solution** | the kata [`course/solutions/primers/craft.03/failpoints.py`](../../../../course/solutions/primers/craft.03/failpoints.py) (`ACTIONS`, `Rule`, `sleep_seconds`, `parse`, `Failpoints` with `evaluate` and `count`: the failpoint spec the `tinyllm` CLI evaluates after every training step) and a model test suite for it in section 5, which kills every planted fault listed in section 6 |
| **Contract** | the spec in the kata's docstring (quoted in section 2) |
| **Needs** | reading: [`craft.01`](../08-code-review-and-ci/01-your-repo-and-ci-gate.md) the repo and CI gate · [`lang.01`](../12-language-and-tool-primers/01-python-and-numpy.md) Python and uv |
| **Used by** | no call site (a practice): the model test suites of [`L0.3`](../../../courses/build-an-llm/p00-foundations/03-losses-with-fused-backward.md) and [`L0.4`](../../../courses/build-an-llm/p00-foundations/04-module-system-and-layers.md) are written the way this chapter explains; the failpoint spec lets the Pass 0 end-to-end run kill training at step 60 and check that it resumes |
| **Optional depth** | Kent Beck, *Test-Driven Development: By Example*; Petrović et al., "Practical Mutation Testing at Scale" (Google, 2021); the PIT and `mutmut` documentation |

## Key Takeaways

- A test is worth what it catches: run it against a correct implementation with one planted fault, and it should fail.
- A test that rejects a correct implementation is worse than none: every fault would then "die" for the wrong reason, so passing on correct code comes first.
- Good tests are black-box: they use only the names the spec defines, which is what lets them run against anyone's correct implementation.
- Red then green: a test written before the code fails first; seeing it fail is the only proof it can fail.
- The mutation score is killed over total; some faults (the ones a chapter's pitfalls describe) matter enough that a suite that misses them is inadequate whatever its score.

## How to read this chapter

Sections 1 and 2 explain test-first development, mutation testing, and the failpoint spec. Section 3 works one red-green cycle and one planted fault by hand. Section 4 fixes the interface. Section 5 walks through the kata line by line, then through a model test suite, and shows which fault each test exists to catch. Try to predict, for each line of the kata, the test that would fail if the line were wrong.

The kata and its tests run with nothing but pytest:

```bash
uv run --no-project --with pytest python -m pytest -q primers/craft.03
```

---

## 1. Why now

Every module from `L0.3` on comes with tests of its own, and "they pass" is not what makes them good: a test that passes on everything proves nothing. Three ideas decide whether a test suite is worth reading: how a test is written before the code it tests, what it means for a test to "kill a mutant", and why tests touch only the contract. This module shows them on a kata small enough to read in one sitting and useful enough to keep: the failpoint spec that the `tinyllm` CLI evaluates after every training step, so the end-to-end run can kill a training run at step 60 and check that it resumes.

## 2. Principles

**A test is a falsifiable claim.** `assert parse("x=2*crash") == {"x": Rule("crash", "", 2)}` claims one fact about the spec. Its value is the set of wrong implementations it rejects. A test that asserts what the code currently returns (run it, paste the output) rejects nothing the code already does wrong; the number in an assertion should come from the spec and hand arithmetic.

**Mutation testing.** A **mutant** is the correct implementation with one small, plausible fault planted: a `<` where the spec needs `!=`, a deleted check, a constant off by one. The tests run against it. If any test fails, the mutant is **killed**; if all pass, it **survived**, and some behavior of the spec is pinned down by none of the tests. The **score** is killed divided by total. Two kinds of mutant: **auto** ones from mechanical operators (flip a comparison, delete a statement, replace a constant, swap arguments) and **semantic** ones written from a chapter's pitfalls, one per pitfall: the bugs people actually make. A suite that lets a semantic mutant survive misses a known bug, whatever its overall score.

**Two baselines.** Before any mutant runs, the tests must pass against a correct implementation; otherwise they reject correct code and the score means nothing. A test that fails on correct code is a wrong test, not a strict one.

**Black-box.** A test suite is only reusable if it touches only what the contract (here, the spec) promises: its names and their documented behavior. A test that reads a private attribute (`fp.counts`) or imports a helper breaks on any other correct implementation.

**Red then green.** Write one test for the next behavior, run it, and watch it fail **for the reason you expect** (red). Write the least code that makes it pass (green). Then the next test. A test never seen failing may pass for the wrong reason, a wrong import or an assertion that is always true, and would kill no mutant.

**Cost.** Running every mutant means running the suite once per mutant. Practical tools sample mutants while developing and run the full set when the tests change, caching the result by the hash of the tests and the code.

**The kata's spec.** `TL_FAILPOINTS="train/after-step=60*crash; data/fetch=error(timeout)"` names places where a test wants a fault. An entry is `name=[N*]action`; actions are `crash`, `panic`, `off`, `error[(msg)]`, and `sleep(duration)` with a duration such as `50ms` or `2s`. `N*` fires on the Nth evaluation of that name only; without it a rule fires every time; `off` never fires and is never counted. Whitespace around names and actions is ignored and empty entries (a trailing `;`) are skipped; everything malformed is a `ValueError`. The kata parses the spec and decides, evaluation by evaluation, whether a failpoint fires; what to do then (exit 137, raise, sleep) is the caller's job.

## 3. Worked example by hand

One red-green cycle, then a mutant.

1. **Red.** The first behavior: `N*` fires on the Nth evaluation only.

    ```python
    def test_nth_fires_once_on_the_nth_evaluation():
        fp = Failpoints("x=2*crash")
        assert [fp.evaluate("x") is not None for _ in range(4)] == [False, True, False, False]
    ```

    Against a stub it fails with `NotImplementedError`: red, for the expected reason.

2. **Green.** The least code: count evaluations per name and fire when the count equals `nth` (or always, when `nth` is 0). The test passes.

3. **A planted fault.** Change `n != rule.nth` to `n < rule.nth` in the "do not fire yet" check, so it fires on the Nth evaluation **and every one after**. Against it the four evaluations give `[False, True, True, True]`: the test above fails, so the mutant is killed. A weaker test that checks only the first two evaluations (`None`, then a rule) passes on the mutant: the mutant survives it.

4. **The score.** The kata has 13 planted faults (section 6). A suite that kills 10 scores $10/13 = 0.77$. If one of the 3 survivors is an `off` rule counted and returned as fired, the suite still misses the most dangerous bug in the spec: a disabled failpoint that crashes the run.

## 4. The interface

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

The standard library is enough.

### What a correct solution must handle

| Case | Kind of check | Checks |
|---|---|---|
| the hand-worked spec | unit | section 2's spec parses to three rules; whitespace and a trailing `;` are fine |
| the hand-worked evaluations | unit | `3*crash` fires on the third of five evaluations only; `off` and unknown names are never counted |
| counts per name | unit | each name counts its own evaluations |
| sleep durations | unit | `50ms` is 0.05 s; malformed durations are errors |
| malformed specs | boundary | nine malformed specs, including `a=b=crash` and `a=0*crash` |
| every action | unit | the five actions and error's trimmed message |

And for a test suite of the kata: it imports only the spec's names, passes on a correct implementation, and fails on each planted fault of section 6.

## 5. The solution, step by step

### From first principles

1. **Separate parsing from evaluating.** A spec is parsed once into a table; evaluation is a lookup plus a counter. Errors in the spec then surface at startup, not at step 60.
2. **Split each entry at the first `=`.** A name never contains `=`, so the first one separates name from action; splitting at the last would turn `a=b=crash` into a failpoint called `a=b` instead of an error.
3. **Recognize the action with one anchored pattern.** `[N*]word[(arg)]`, anchored at both ends, then check the word against `ACTIONS`. Everything that does not match is an error naming the entry.
4. **Validate everything the grammar allows but the meaning forbids.** `N` counts from 1; `crash`, `panic`, `off` take no argument; `sleep` needs a duration that parses. Check these at parse time so the caller never sees a half-valid rule.
5. **Duplicates are errors.** A name given twice is almost always a typo; silently letting the later entry win would disable a failpoint.
6. **Count per name, and only what can fire.** `off` and unknown names return `None` before the counter moves, so `count` reports evaluations that could have fired.
7. **Fire on exactly the Nth.** With `nth` set, fire when the count equals it, never before or after.

### The kata, annotated

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

Full solution: [`failpoints.py`](../../../../course/solutions/primers/craft.03/failpoints.py).

### A model test suite, annotated

Each test derives its expected values from the spec, uses only the five public names, and is aimed at specific faults. Run against the kata it passes; run against each of the 13 planted faults of section 6 it fails at least once.

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

Notice what the suite does not do: it never reads `fp.counts` or `fp.table`, so it passes on any correct implementation; and each expected list such as `[False, True, False, False]` was written from the spec before running anything. `test_counts_are_per_name` uses the same `N` on two names, because only then does a shared counter change the outcome. `test_without_n_fires_every_time` kills no planted fault on its own, but it pins down the default behavior so a later change cannot break it unnoticed.

Model test suites written the same way for the next modules: [`L0.3`](../../../../course/solutions/learner-tests/L0.3/python/tests/l0-3-losses/) and [`L0.4`](../../../../course/solutions/learner-tests/L0.4/python/tests/l0-4-module/).

### Why this solution

- **Cost.** `parse` is linear in the spec's length; `evaluate` is one dictionary lookup and one increment, cheap enough to call after every training step.
- **Strict parsing over lenient parsing.** A lenient parser that ignores bad entries makes a typo in an environment variable silently disable a fault injection, and the test that relied on it passes for the wrong reason. Strict errors at startup cost nothing in a correct spec.
- **Mutation testing over coverage.** Line coverage says a line ran; it does not say any assertion would notice if the line were wrong. The planted faults measure exactly that: every fault in section 6 is on a line a weak suite would cover and still miss.
- **Tradeoffs accepted.** The kata leaves out a probability form (`10%crash`) that real failpoint libraries have, because it would make evaluation non-deterministic and the tests harder to write by hand.

### Explain it back

**1. Why does `evaluate` return early for `off` before incrementing the counter, rather than after?**

<details><summary>Answer</summary>

The spec says `off` is never counted. Returning after incrementing would make `count("x")` grow for a disabled failpoint; returning `rule` instead of `None` would fire it. A test that only checks `evaluate` returns `None` would not catch the counting fault; `test_off_never_fires_and_is_never_counted` checks both.

</details>

**2. A suite passes on the correct kata and on every mutant. What does that say?**

<details><summary>Answer</summary>

It constrains nothing the mutants change: every fault leaves its assertions true. Typical causes are expected values pasted from the code's output, tests that only check a value is not `None`, or tests that were never seen failing.

</details>

**3. Why must a test suite pass on a correct implementation before mutants mean anything?**

<details><summary>Answer</summary>

A test that fails on correct code fails on every mutant too, so every mutant counts as "killed" and the score is 100% for the wrong reason. Passing on correct code first shows that failures on a mutant are caused by the fault.

</details>

**4. Why does `test_counts_are_per_name` evaluate `b` between two evaluations of `a`?**

<details><summary>Answer</summary>

With one shared counter, `b`'s first evaluation would be the second overall and fire. The order a, b, a separates per-name counting from shared counting, which a test that evaluates one name at a time cannot.

</details>

**5. Why split at the first `=` and not the last?**

<details><summary>Answer</summary>

Names never contain `=`, so the first `=` is always the separator, and `a=b=crash` must be an error. Splitting at the last `=` would accept it as a failpoint named `a=b`, and the caller evaluating `a` would never see it fire.

</details>

## 6. Pitfalls

The bugs a test suite of the kata must catch, one planted fault each:

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

And the testing mistakes that let faults survive:

| Mistake | What mutation testing shows | Fix |
|---|---|---|
| expected values pasted from the code's output | the fault was in the code, so the test agrees with it: mutants survive | derive each expected value from the spec, by hand |
| tests that reach into private state (`fp.counts`) | the suite fails on a correct implementation that keeps its counters elsewhere | assert only through the spec's names |
| a test never seen red | it may assert nothing (a typo in a name, a loop that never runs) | run it before the code exists |
| one giant test | the first failure hides the rest; survivors are hard to place | one behavior per test, named for it |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `craft.01` | the repo and its CI gate run every test suite on each change |
| Back | `lang.01` | `uv run` and pytest |
| Forward | `L0.3` | a model test suite for the losses, written to kill the chapter's pitfalls |
| Forward | `L0.4` | a model test suite for the module system, written test first |
| Forward | Pass 0 end to end | the CLI evaluates `train/after-step` with this spec, so a run can be killed and resumed |

Later chapters build on it: properties (`craft.04`), mutation testing in depth (`craft.07`), oracles and gradchecks (`craft.05`).

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| planted faults and a score | PIT (Java), `mutmut` and `cosmic-ray` (Python), `cargo-mutants` (Rust) | operators generated over a whole codebase, incremental runs on changed lines | `pitest.org`, `github.com/sourcefrog/cargo-mutants` |
| semantic mutants from pitfalls | Google's mutation testing in code review | mutants surfaced as review comments on the changed lines only, with suppression of unproductive ones | Petrović et al., ICSE-SEIP 2021 |
| the failpoint spec | FreeBSD/TiKV `fail` and `failpoint` crates, etcd `gofail` | failpoints compiled out in release builds, actions such as `return(value)` and `pause` | `github.com/tikv/fail-rs` |
| red then green | Beck's TDD cycle, mutation-score gates in CI | a minimum score per module enforced on pull requests | Stryker dashboard |
