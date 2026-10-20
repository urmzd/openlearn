# A performance regression bisect

A performance regression can consume headroom gradually and evade functional tests: every test still passes, but TTFT p95 is 15% worse than last month, and nobody can say which of two hundred commits did it. This drill practices a controlled bisect that narrows the change while preserving a stable workload. It is the operational side of [Performance regression bisect](../../software-engineering/08-maintenance/04-performance-regression-bisect.md) in *Software Engineering*.

A performance regression is a measurement problem first. Unless the machine class, model, request mix, warm-up, and sample count are the same, noise looks like a regression. The drill follows [A breaking dependency upgrade](07-dependency-upgrade.md), which tried to stop regressions at the gate; this one finds the ones that got through.

The section states the principles and the drill scenario, lists what a correct runbook must cover, and derives the procedure, ending with the model runbook `docs/runbooks/PerformanceRegression.md`. Try to write the procedure yourself before you read the solution.

> **Learning objectives**
> - Decide whether a latency shift is real, using repeated runs and a threshold fixed in advance.
> - Control the variables that fake or hide a regression.
> - Drive `git bisect run` with a measuring script, and estimate its cost as about $\log_2 n$ measurements.
> - Confirm the verdict at the boundary commit before reverting or fixing.

## Principles

Use the same machine class, model, request mix, warmup, and sample count. Set a threshold and rerun the candidate commit before declaring it bad. Set the threshold before measuring, then bisect with a script that measures and decides, so `git bisect run` finds the first bad revision. Re-measure base, head, and the candidate commit before declaring it bad: one noisy run must not convict a commit.

## The drill scenario

Inject the seeded regression, run the saved load profile, identify the first bad revision with git bisect, and repeat base/head/candidate measurements.

## What a correct runbook must cover

A runbook for this investigation is complete when it answers five questions:

| Case | What the runbook must contain | Why it matters |
|---|---|---|
| Owner and baseline | the metric (for example TTFT p95 or tokens per second), its baseline, and the threshold | a decision rule fixed in advance |
| Controlled workload | the saved load profile: rate, mix, seed, duration, warm-up, machine class | removes noise sources |
| Bisect | the good and bad revisions, the measuring script, `git bisect run` | finds the first bad commit mechanically |
| Confirmation | repeated base, head, and candidate measurements | rules out a noisy convicting run |
| Fix and verification | revert or fix, then the same profile showing the metric back within threshold | done means measured |

## Deriving the procedure

### Reasoning from first principles

Is it real? Latency varies between runs. Repeating the same profile several times gives a spread; a regression is a shift larger than that spread, decided by a threshold set before looking.

Remove variables. Use the same machine class, model, request mix, warm-up, and number of samples: each uncontrolled one adds variance that can hide or fake the shift.

Search, do not guess. Between a good and a bad revision, `git bisect` halves the range each step, so $n$ commits need about $\log_2 n$ measurements. A script that runs the profile and exits 0 (good) or 1 (bad) lets `git bisect run` do it unattended.

Confirm the verdict. Re-run base, head, and the first bad commit (and its parent) several times; the shift must reproduce at that boundary.

Fix and prove it. Revert or fix the commit and measure again with the same profile.

### The model runbook

The procedure written out as `docs/runbooks/PerformanceRegression.md`:

````markdown
# Runbook: performance regression

Owner: the owner of the regressed service.

## Preconditions

- The metric and threshold, for example "TTFT p95 more than 10% above the baseline".
- The saved load profile (loadgen: rate, mix, seed, duration) and machine class.
- A known good and a known bad revision.

## Measure

Run the profile 3 times on base and on head after warm-up; record the
loadgen report's `ttft_ms`, `tpot_ms`, `tokens_per_s`, `error_rate`. A
regression is a shift larger than the run-to-run spread and the threshold.

## Bisect

```bash
git bisect start <bad> <good>
git bisect run ./scripts/perf-check.sh   # builds, runs the profile, exits 1 when over threshold
```

## Confirm and fix

Re-measure the first bad commit and its parent 3 times each. Revert or fix,
then run the profile on the fix: the metric must be back within threshold.
Record the commit, the numbers, and the cause.
````

### Why this solution

Setting the threshold first prevents moving the goalposts after seeing the data. `git bisect run` turns a search over hundreds of commits into about $\log_2 n$ automated measurements. You might be tempted to profile head directly instead, but a profiler shows where time goes now, not what changed. Bisect finds the change; a profile of that one diff then explains it.

## What can go wrong

The three classic mistakes all let noise make the decision:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Declaring a regression from one run | chasing noise; reverting a good commit | repeated runs and a threshold set in advance |
| Comparing runs on different machines or mixes | a shift caused by the environment | the saved profile and machine class are preconditions |
| Bisecting by hand with ad hoc judgment | inconsistent verdicts; a wrong first bad commit | a script with a fixed threshold drives `git bisect run` |

## Where this leads

The technique this runbook relies on is [Performance regression bisect](../../software-engineering/08-maintenance/04-performance-regression-bisect.md), and the saved load profile comes from the load generator of [An open-loop load generator](../../ai-engineering/23-model-serving/11-load-generator.md) in *AI Engineering*. The next drill, [A data incident](09-data-incident.md), leaves code changes behind and traces bad data instead. To compare this inject with the adjacent drills and reuse the same containment and verification fields, see the [chapter opening](README.md).

## Key terms

- **Run-to-run spread**: the variation in a metric across repeated runs of the same profile; a regression must exceed it.
- **Saved load profile**: a fixed rate, mix, seed, duration, and warm-up that makes runs comparable.
- **`git bisect run`**: git's binary search over history, driven by a script whose exit code marks each revision good or bad.
- **First bad commit**: the earliest revision on which the regression reproduces, confirmed against its parent.

## Summary

- A performance regression is a measurement problem first: control machine class, model, request mix, warm-up, and sample count.
- Fix the metric and threshold before measuring, and call a regression only when the shift exceeds the run-to-run spread.
- A measuring script that exits 0 or 1 lets `git bisect run` find the first bad revision in about $\log_2 n$ measurements.
- Re-measure the candidate and its parent before convicting it, then prove the fix with the same profile.

## Review questions

1. How many measurements does bisect need for 200 commits?
   <details><summary>Answer</summary>About $\log_2 200 \approx 8$, each a full profile run, since each step halves the range.</details>
2. Why re-measure the candidate and its parent?
   <details><summary>Answer</summary>One noisy run can misclassify a step of the bisect. Reproducing the shift exactly at that boundary confirms the commit is the cause.</details>
3. Why fix the request mix and warm-up?
   <details><summary>Answer</summary>Both change latency on their own; a different mix or a cold cache can produce a shift with no code change at all.</details>
