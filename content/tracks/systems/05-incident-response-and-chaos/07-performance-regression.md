# Drill: performance regression bisect

## Overview

| | |
|---|---|
| **Module** | `ops.07` · drill · ops, docs · Pass 11 · 1 to 2 h |
| **Solution** | the model runbook `docs/runbooks/PerformanceRegression.md`, written out with its reasoning in section 5 |
| **Needs** | none |
| **Used by** | no code call site: a drill, one of the Pass 11 operations drills |

## Key Takeaways

- A performance regression is a **measurement** problem first: same machine class, model, request mix, warm-up, and sample count, or noise looks like a regression.
- Set the threshold before measuring, then bisect with a script that measures and decides, so `git bisect run` finds the first bad revision.
- Re-measure base, head, and the candidate commit before declaring it bad: one noisy run must not convict a commit.

## How to read this chapter

This chapter is a worked drill. Sections 1 to 3 state the risk, the principles, and the drill scenario. Section 4 lists what a correct runbook must cover. Section 5 derives the procedure from first principles, gives the model runbook, and explains why each step is the right one. Try to write the procedure yourself from sections 2 and 3 before reading section 5.

---

## 1. Why now

A performance regression can consume headroom gradually and evade functional tests. A controlled bisect narrows the change while preserving a stable workload.

## 2. Principles

Use the same machine class, model, request mix, warmup, and sample count. Set a threshold and rerun the candidate commit before declaring it bad.

## 3. Worked example

Inject the seeded regression, run the saved load profile, identify the first bad revision with git bisect, and repeat base/head/candidate measurements.

## 4. What a correct runbook must cover

| Case | What the runbook must contain | Why it matters |
|---|---|---|
| Owner and baseline | the metric (for example TTFT p95 or tokens per second), its baseline, and the threshold | a decision rule fixed in advance |
| Controlled workload | the saved load profile: rate, mix, seed, duration, warm-up, machine class | removes noise sources |
| Bisect | the good and bad revisions, the measuring script, `git bisect run` | finds the first bad commit mechanically |
| Confirmation | repeated base, head, and candidate measurements | rules out a noisy convicting run |
| Fix and verification | revert or fix, then the same profile showing the metric back within threshold | done means measured |

## 5. The solution, step by step

### 5.1 From first principles

1. **Is it real?** Latency varies between runs. Repeating the same profile several times gives a spread; a regression is a shift larger than that spread, decided by a threshold set before looking.
2. **Remove variables.** Same machine class, model, request mix, warm-up, and number of samples: each uncontrolled one adds variance that can hide or fake the shift.
3. **Search, do not guess.** Between a good and a bad revision, `git bisect` halves the range each step, so $n$ commits need about $\log_2 n$ measurements. A script that runs the profile and exits 0 (good) or 1 (bad) lets `git bisect run` do it unattended.
4. **Confirm the verdict.** Re-run base, head, and the first bad commit (and its parent) several times; the shift must reproduce at that boundary.
5. **Fix and prove it.** Revert or fix the commit and measure again with the same profile.

### 5.2 The model runbook

`docs/runbooks/PerformanceRegression.md`:

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

### 5.3 Why this solution

- **Threshold first** prevents moving the goalposts after seeing the data.
- **`git bisect run`** turns a search over hundreds of commits into about $\log_2 n$ automated measurements.
- **Why not profile head directly?** A profiler shows where time goes now, not what changed; bisect finds the change, then a profile of that one diff explains it.

### 5.4 Explain it back

1. How many measurements does bisect need for 200 commits?
   <details><summary>Answer</summary>About $\log_2 200 \approx 8$, each a full profile run, since each step halves the range.</details>
2. Why re-measure the candidate and its parent?
   <details><summary>Answer</summary>One noisy run can misclassify a step of the bisect. Reproducing the shift exactly at that boundary confirms the commit is the cause.</details>
3. Why fix the request mix and warm-up?
   <details><summary>Answer</summary>Both change latency on their own; a different mix or a cold cache can produce a shift with no code change at all.</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | Declaring a regression from one run | chasing noise; reverting a good commit | repeated runs and a threshold set in advance |
| 2 | Comparing runs on different machines or mixes | a shift caused by the environment | the saved profile and machine class are preconditions |
| 3 | Bisecting by hand with ad hoc judgment | inconsistent verdicts; a wrong first bad commit | a script with a fixed threshold drives `git bisect run` |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `craft.16` | A prerequisite: the contract or evidence this runbook relies on. |
| Back | `ops.06` | A prerequisite: the contract or evidence this runbook relies on. |

## Going further

| Resource | What to inspect |
|---|---|
| [Incident drill catalog](README.md) | Compare this inject with the adjacent failure drills and use the same containment and verification fields. |
