# Perf regression bisect

## Overview

| | |
|---|---|
| **Module** | `craft.16` · practice · docs · Pass 11 · 1 h reading |
| **Solution** | Model record: [`perf-bisect.md`](../../../../course/solutions/docs/maintenance/perf-bisect.md), a TTFT regression bisect with its reproduction envelope, decision rule, and procedure |
| **Needs** | [`craft.06`](../03-testing-mentality/04-benchmarks-and-perf-gates.md) (benchmarks and performance gates) |
| **Used by** | no code call site; this is an independent artifact. The [`ops.07`](../../systems/05-incident-response-and-chaos/07-performance-regression.md) drill runs the same procedure on an injected regression. |

## Key Takeaways

- A bisect is only as good as its oracle: fix the workload, host, configuration, and sampling, and write the regression rule before looking at commits.
- Verify that the base is good and the head is bad under that rule before bisecting; then confirm the first bad commit with repeated runs.
- The record names the owner, the rollback, and the approval threshold, and keeps raw results.

## How to read this chapter

Sections 1 to 3 explain why a bisect turns "it got slower" into one commit and sketch the procedure. Section 4 lists what a complete bisect record must contain. Section 5 walks through the model record, works its numbers, and explains the exit-code protocol `git bisect run` depends on. Try to write the regression rule yourself before reading the model's.

---

## 1. Why now

A latency regression costs users and capacity. A bisect turns "it got slower" into a reproducible change range.

## 2. Principles

Use a stable workload, fixed machine and configuration, warm-up, repeated samples, and a threshold chosen before inspecting commits. Validate both endpoints before trusting the midpoint result.

## 3. Worked example

Reproduce a 20% TTFT regression with the saved request fixture. Run the same benchmark at the base and head, then use `git bisect run` with a script that exits nonzero only when the median exceeds the agreed bound. Re-run the identified commit.

With the model record's numbers: base median 41 ms, head median 50 ms. The increase is $50 - 41 = 9$ ms and $9/41 = 22.0\%$. The rule is "at least 15% slower **and** more than 3 ms", so head is bad on both counts. If 128 commits separate base and head, bisection needs $\lceil \log_2 128 \rceil = 7$ tested revisions; at 7 runs each plus warm-up, that is the cost to budget, and the reason every run must be trustworthy.

## 4. The artifact

The artifact is `docs/maintenance/perf-bisect.md`.

### What a correct record must contain

| Element | What it states | Why it matters downstream |
|---|---|---|
| Reproduction envelope | Metric, workload fixture, host class, configuration, sampling. | Every revision is measured under identical conditions. |
| Regression rule | A relative and an absolute bound, fixed before bisecting. | The oracle cannot drift toward a preferred answer. |
| Endpoint results | Base and head medians and p95 with raw output attached. | Proves the regression reproduces before searching for it. |
| Procedure | Endpoint check, `git bisect run` with a gate script, confirmation runs. | The search is mechanical and repeatable. |
| Rollback, owner, approval | The recovery, who owns it, and what closes the issue. | The finding leads to an action, not just a commit hash. |

## 5. The solution, step by step

### 5.1 From first principles

1. Bisection finds the first bad commit in $O(\log n)$ tests, but only if the test is a **consistent oracle**: every commit before the culprit says "good", every commit from it on says "bad". A noisy benchmark breaks that assumption and bisection then converges confidently on a random commit.
2. Noise comes from the environment (host, neighbours, thermal state), the input (workload), the state (cold caches, JIT, connection setup), and sampling. Each source gets a control: pinned host class with no concurrent jobs, a fixed fixture, warm-up rounds, and repeated runs summarised by the median.
3. A relative bound alone flags tiny absolute changes on fast paths (10% of 2 ms is noise); an absolute bound alone ignores scale. Requiring both means the change is large in proportion and in milliseconds.
4. The bound is written down before any commit is measured, so the oracle cannot be tuned to blame a suspect.
5. If the endpoints do not reproduce (base not good or head not bad under the rule), there is nothing to bisect. Checking them first costs two measurements and can save the whole search.
6. Some revisions cannot be measured (they do not build, or the runner fails). Calling them good or bad would corrupt the search; they must be skipped or the bisect aborted.
7. The final answer is confirmed by measuring the first bad commit and its neighbours again, since a single noisy midpoint can still mislead.

### 5.2 The model record, annotated

**Reproduction envelope.**

| Field | Fixed value for this example |
|---|---|
| Metric | time to first token (TTFT), milliseconds |
| Workload | `fixtures/perf/ttft-64-prompts.jsonl`, 64 prompt cases, 128-token output cap |
| Host | one pinned CI runner class; no concurrent benchmark jobs |
| Configuration | same model digest, tokenizer digest, batch limit, and runtime config at every revision |
| Sampling | 2 warm-up rounds, then 7 runs; compare medians and p95, retain raw JSON |
| Regression rule | candidate median is at least 15% slower and the increase exceeds 3 ms |

Each row is a control from step 2. The output cap matters for TTFT less than for total latency, but it keeps runs the same length so the host's state is comparable between them. Model and tokenizer are pinned by digest because a different model would change TTFT for reasons that have nothing to do with the code. Seven runs give a median that one outlier cannot move. The rule is step 3: relative **and** absolute.

**Endpoint results.**

| Revision | Median TTFT | p95 TTFT | Raw result |
|---|---:|---:|---|
| `base` | 41 ms | 55 ms | attach `base.json` |
| `head` | 50 ms | 67 ms | attach `head.json` |

> The median increase is 22.0%, or 9 ms, so the candidate crosses both predeclared regression limits. The p95 moves in the same direction. The check is not accepted if the baseline itself fails to reproduce within 5% over three repetitions.

The p95 rises by $12/55 \approx 22\%$ too, so the regression is a shift of the whole distribution, not a few slow outliers. The last sentence is step 5 applied to the baseline: if base itself varies by more than 5%, the 15% rule is too close to the noise to be trusted. The record says the measurements are hypothetical and must be replaced with raw output from the same host, which keeps a template from being mistaken for a finding.

**Procedure.**

> 1. Pin the workload, model, host class, and runner configuration before inspecting commits.
> 2. Verify `base` is good and `head` is bad using the regression rule above.
> 3. Run `git bisect run ./scripts/bench-ttft-gate.sh`; the script returns 0 for good, 1 for bad, and a distinct infrastructure error for an invalid run.
> 4. Repeat the first bad revision and both neighbors with seven runs. A single noisy midpoint is not a finding.
> 5. Attach raw results, the bisect log, and the identified commit to the issue. The owner proposes a fix or documents why the change is intentional.

Step 3's "distinct infrastructure error" is where `git bisect run`'s exit-code protocol matters (step 6). Git reads the script's exit status as: `0` good; `125` skip this revision (it cannot be tested, for example it does not build); any other code from 1 to 127 bad; anything else (128 and above, or a signal) aborts the bisect. So a gate script should exit `125` for "this revision cannot be measured" and a code of 128 or more for "the runner itself is broken, stop". The shape of such a gate, with the benchmark command left to the project:

```bash
#!/usr/bin/env bash
# Exit codes follow `git bisect run`: 0 good, 1 bad, 125 skip, >=128 abort.
set -u
BASE_MEDIAN_MS=41                        # measured at base, recorded in the issue

build || exit 125                        # cannot build this revision: skip it, not "bad"
median_ms=$(measure) || exit 128         # runner failure: abort the whole bisect

# the predeclared rule: >= 15% slower AND more than 3 ms slower
awk -v m="$median_ms" -v b="$BASE_MEDIAN_MS" \
    'BEGIN { exit !(m >= 1.15 * b && m - b > 3) }' && exit 1
exit 0
```

`build` and `measure` stand for the project's build step and for "two warm-up rounds, seven runs, print the median". Under `set -e` a failing `measure` would exit 1 and be counted as **bad**, which is why the script handles each failure explicitly instead.

Step 4 of the procedure is step 7 above: the first bad revision and its two neighbours are measured again with seven runs. If the "first bad" commit and its parent are within noise of each other, the search went wrong somewhere and the log shows where.

**Rollback, owner, approval.**

> **Rollback:** revert the identified change or pin the prior image digest if the regression is already deployed. Keep the workload and raw results with the incident. **Owner:** serving performance on-call. **Approval threshold:** reviewer confirms endpoint reproducibility and the final rerun is below both limits.

Two recoveries for two situations: a revert when the regression is caught before release, a pinned digest when it is already serving. Closing requires a rerun below **both** limits, the same rule used to open the issue.

Full model answer: [`perf-bisect.md`](../../../../course/solutions/docs/maintenance/perf-bisect.md).

### 5.3 Why this procedure

- **Cost** is $\lceil \log_2 n \rceil$ measurements of 9 rounds each, plus 2 endpoint checks and 3 confirmation measurements. Linear search would cost $n$; for a 128-commit range that is 128 against about 12.
- **Medians over means:** one slow run from a noisy neighbour moves a mean of seven noticeably and a median not at all.
- **Automated `git bisect run`** rather than marking by hand: a human marking revisions after glancing at numbers applies the rule inconsistently and may skip the confirmation step.
- **Tradeoff accepted:** the strict envelope means a regression that only appears under a different workload or host will not be found by this bisect. The record names its workload so that limitation is visible.

### 5.4 Explain it back

**Why must base be verified good and head verified bad before bisecting?**

<details><summary>Answer</summary>

Bisection assumes the property flips from good to bad somewhere in the range. If base is already bad under the rule (or head is not bad), the search still returns a commit, but it is meaningless. Two endpoint measurements check the assumption.

</details>

**What happens if the gate script exits 1 when a revision fails to build?**

<details><summary>Answer</summary>

Git counts the revision as bad. If the unbuildable commit lies before the real culprit, bisection blames it (or something near it) instead. Exiting 125 tells git to skip that revision and choose another.

</details>

**Why require both "at least 15%" and "more than 3 ms"?**

<details><summary>Answer</summary>

A relative bound alone would flag a 0.5 ms change on a 3 ms path, which is within run-to-run noise. An absolute bound alone would treat 4 ms as significant on a 2 s path. Both together flag changes that are large in proportion and in real time.

</details>

**The bisect names commit X. X and its parent measure 49 ms and 47 ms. Is X the culprit?**

<details><summary>Answer</summary>

Not on this evidence. A 2 ms step is below the 3 ms bound, so the oracle should not have separated them. Either the regression accumulated over several commits, or a noisy measurement misled the search. The confirmation step exists to catch this; widen the confirmation to more neighbours and rerun.

</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | The record names a commit without the evidence that supports it | Reviewers cannot verify the claim against the repository or system. | Raw JSON for both endpoints, the bisect log, and the confirmation runs are attached to the issue. |
| 2 | The failure or rollback case is omitted | The happy path passes while an operator has no safe response under failure. | The record gives a revert for undeployed changes and a pinned prior digest for deployed ones. |
| 3 | The owner, threshold, or scope is implicit | Two reviewers can reach different release or risk decisions from the same evidence. | The regression rule is in the envelope table before any commit is measured; owner and approval threshold are named. |
| 4 | Treating an untestable revision as bad | The search converges on a build break, not the regression. | The gate distinguishes skip (125) and abort (128 or more) from bad. |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | [`craft.06`](../03-testing-mentality/04-benchmarks-and-perf-gates.md) | Benchmarks with budgets: the measurement discipline the gate script reuses. |
| Forward | [`ops.07`](../../systems/05-incident-response-and-chaos/07-performance-regression.md) | The performance regression drill, where this procedure finds an injected slowdown. |

## Going further

| Resource | What to inspect |
|---|---|
| `git help bisect`, section "Bisect run" | The exit-code protocol: 0, 125, 1 to 127, and the abort range. |
| [Benchmarks and perf gates](../03-testing-mentality/04-benchmarks-and-perf-gates.md) | How a budget file and a gate turn a benchmark into a pass/fail decision. |
