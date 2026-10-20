# Performance regression bisect

A latency regression costs users and capacity, and it rarely announces which change caused it. Somewhere between last week's release and today's, time to first token got 20% worse, and there are a hundred commits in between. Staring at diffs will not find it. Regressions are found by search: a benchmark with a threshold, plus `git bisect run`, finds the first bad commit in $\log_2 n$ steps and turns "it got slower" into one reproducible change.

The catch is that a bisect is only as good as its oracle. If the benchmark is noisy, bisection converges confidently on a random commit. So most of this section is about making the measurement trustworthy: fixing the workload, host, configuration, and sampling, and writing the regression rule down before looking at any commit. The model record is [`perf-bisect.md`](../../../../course/solutions/docs/maintenance/perf-bisect.md), a TTFT regression bisect with its reproduction envelope, decision rule, and procedure.

This section builds on [Benchmarks and perf gates](../03-testing/04-benchmarks-and-perf-gates.md), whose measurement discipline the gate script reuses. The same procedure is run on an injected regression in [A performance regression bisect](../../systems/12-incident-response-and-chaos/08-performance-regression.md) in *Systems Engineering*. Try writing the regression rule yourself before you read the model's.

> **Learning objectives**
> - Define a reproduction envelope that makes a benchmark a consistent oracle for bisection.
> - Write a regression rule with a relative and an absolute bound, fixed before measuring any commit.
> - Verify the endpoints, run `git bisect run` with a gate script that uses the exit-code protocol correctly, and confirm the result.
> - Estimate the cost of a bisect and record the rollback, owner, and approval threshold.

## What makes a bisect trustworthy

The principles are short. Use a stable workload, a fixed machine and configuration, a warm-up, repeated samples, and a threshold chosen before inspecting commits. Validate both endpoints before trusting any midpoint result.

Here is the procedure in outline. Reproduce a 20% TTFT regression with the saved request fixture. Run the same benchmark at the base and at the head, then use `git bisect run` with a script that exits nonzero only when the median exceeds the agreed bound. Finally, re-run the identified commit.

## Working the numbers

With the model record's numbers, the base median is 41 ms and the head median is 50 ms. The increase is $50 - 41 = 9$ ms, and $9/41 = 22.0\%$. The rule is "at least 15% slower **and** more than 3 ms", so the head is bad on both counts.

Now the cost. If 128 commits separate base and head, bisection needs $\lceil \log_2 128 \rceil = 7$ tested revisions. At 7 runs each plus warm-up, that is the cost to budget, and it is the reason every run must be trustworthy: a wrong verdict at any one of the seven steps sends the rest of the search into the wrong half.

## What a complete record contains

The artifact is `docs/maintenance/perf-bisect.md`. A complete bisect record states each of these:

| Element | What it states | Why it matters downstream |
|---|---|---|
| Reproduction envelope | Metric, workload fixture, host class, configuration, sampling. | Every revision is measured under identical conditions. |
| Regression rule | A relative and an absolute bound, fixed before bisecting. | The oracle cannot drift toward a preferred answer. |
| Endpoint results | Base and head medians and p95 with raw output attached. | Proves the regression reproduces before searching for it. |
| Procedure | Endpoint check, `git bisect run` with a gate script, confirmation runs. | The search is mechanical and repeatable. |
| Rollback, owner, approval | The recovery, who owns it, and what closes the issue. | The finding leads to an action, not just a commit hash. |

## Building the bisect step by step

Seven observations explain the record.

1. Bisection finds the first bad commit in $O(\log n)$ tests, but only if the test is a **consistent oracle**: every commit before the culprit says "good", and every commit from it on says "bad". A noisy benchmark breaks that assumption, and bisection then converges confidently on a random commit.
2. Noise comes from the environment (host, neighbours, thermal state), the input (workload), the state (cold caches, JIT, connection setup), and sampling. Each source gets a control: a pinned host class with no concurrent jobs, a fixed fixture, warm-up rounds, and repeated runs summarised by the median.
3. A relative bound alone flags tiny absolute changes on fast paths (10% of 2 ms is noise); an absolute bound alone ignores scale. Requiring both means the change is large in proportion and in milliseconds.
4. The bound is written down before any commit is measured, so the oracle cannot be tuned to blame a suspect.
5. If the endpoints do not reproduce (base not good, or head not bad, under the rule), there is nothing to bisect. Checking them first costs two measurements and can save the whole search.
6. Some revisions cannot be measured: they do not build, or the runner fails. Calling them good or bad would corrupt the search, so they must be skipped or the bisect aborted.
7. The final answer is confirmed by measuring the first bad commit and its neighbours again, since a single noisy midpoint can still mislead.

### The reproduction envelope

| Field | Fixed value for this example |
|---|---|
| Metric | time to first token (TTFT), milliseconds |
| Workload | `fixtures/perf/ttft-64-prompts.jsonl`, 64 prompt cases, 128-token output cap |
| Host | one pinned CI runner class; no concurrent benchmark jobs |
| Configuration | same model digest, tokenizer digest, batch limit, and runtime config at every revision |
| Sampling | 2 warm-up rounds, then 7 runs; compare medians and p95, retain raw JSON |
| Regression rule | candidate median is at least 15% slower and the increase exceeds 3 ms |

Each row is a control from the second observation. The output cap matters less for TTFT than for total latency, but it keeps runs the same length, so the host's state is comparable between them. The model and tokenizer are pinned by digest because a different model would change TTFT for reasons that have nothing to do with the code. Seven runs give a median that one outlier cannot move. And the rule is the third observation: relative **and** absolute.

### Endpoint results

| Revision | Median TTFT | p95 TTFT | Raw result |
|---|---:|---:|---|
| `base` | 41 ms | 55 ms | attach `base.json` |
| `head` | 50 ms | 67 ms | attach `head.json` |

> The median increase is 22.0%, or 9 ms, so the candidate crosses both predeclared regression limits. The p95 moves in the same direction. The check is not accepted if the baseline itself fails to reproduce within 5% over three repetitions.

The p95 rises by $12/55 \approx 22\%$ too, so the regression is a shift of the whole distribution, not a few slow outliers. The last sentence applies the fifth observation to the baseline: if the base itself varies by more than 5%, the 15% rule is too close to the noise to be trusted. The record also says its measurements are hypothetical and must be replaced with raw output from the same host, which keeps a template from being mistaken for a finding.

### The procedure

> 1. Pin the workload, model, host class, and runner configuration before inspecting commits.
> 2. Verify `base` is good and `head` is bad using the regression rule above.
> 3. Run `git bisect run ./scripts/bench-ttft-gate.sh`; the script returns 0 for good, 1 for bad, and a distinct infrastructure error for an invalid run.
> 4. Repeat the first bad revision and both neighbors with seven runs. A single noisy midpoint is not a finding.
> 5. Attach raw results, the bisect log, and the identified commit to the issue. The owner proposes a fix or documents why the change is intentional.

The "distinct infrastructure error" in step 3 is where the exit-code protocol of `git bisect run` matters (the sixth observation). Git reads the script's exit status as follows: `0` is good; `125` means skip this revision because it cannot be tested, for example because it does not build; any other code from 1 to 127 is bad; and anything else (128 and above, or a signal) aborts the bisect. So a gate script should exit `125` for "this revision cannot be measured" and a code of 128 or more for "the runner itself is broken, stop". Here is the shape of such a gate, with the benchmark command left to the project:

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

`build` and `measure` stand for the project's build step and for "two warm-up rounds, seven runs, print the median". Under `set -e`, a failing `measure` would exit 1 and be counted as **bad**, which is why the script handles each failure explicitly instead.

Step 4 of the procedure is the seventh observation: the first bad revision and its two neighbours are measured again with seven runs. If the "first bad" commit and its parent are within noise of each other, the search went wrong somewhere, and the log shows where.

### Rollback, owner, and approval

> **Rollback:** revert the identified change or pin the prior image digest if the regression is already deployed. Keep the workload and raw results with the incident. **Owner:** serving performance on-call. **Approval threshold:** reviewer confirms endpoint reproducibility and the final rerun is below both limits.

There are two recoveries for two situations: a revert when the regression is caught before release, and a pinned digest when it is already serving. Closing the issue requires a rerun below **both** limits, the same rule that opened it. The full model answer is [`perf-bisect.md`](../../../../course/solutions/docs/maintenance/perf-bisect.md).

### Why this design

The cost is $\lceil \log_2 n \rceil$ measurements of 9 rounds each, plus 2 endpoint checks and 3 confirmation measurements. Linear search would cost $n$; for a 128-commit range that is 128 measurements against about 12.

The record uses **medians rather than means**, because one slow run from a noisy neighbour moves a mean of seven noticeably and a median not at all. It **automates the search with `git bisect run`** rather than marking revisions by hand, because a human marking revisions after glancing at numbers applies the rule inconsistently and may skip the confirmation step.

One tradeoff is accepted. The strict envelope means a regression that appears only under a different workload or host will not be found by this bisect. The record names its workload so that the limitation is visible.

## What can go wrong

Four pitfalls recur. A record that names a commit without the evidence that supports it cannot be verified against the repository or the system; the model attaches raw JSON for both endpoints, the bisect log, and the confirmation runs to the issue. A record that omits the failure or rollback case leaves an operator with no safe response; the model gives a revert for undeployed changes and a pinned prior digest for deployed ones. If the owner, threshold, or scope is implicit, two reviewers can reach different decisions from the same evidence; the model puts the regression rule in the envelope table before any commit is measured, and names the owner and the approval threshold.

> **Warning:** Treating an untestable revision as bad makes the search converge on a build break, not the regression. The gate distinguishes skip (125) and abort (128 or more) from bad.

## Where this leads

The measurement discipline here comes from [Benchmarks and perf gates](../03-testing/04-benchmarks-and-perf-gates.md), which shows how a budget file and a gate turn a benchmark into a pass/fail decision. Next, [A performance regression bisect](../../systems/12-incident-response-and-chaos/08-performance-regression.md) runs this procedure to find an injected slowdown.

### Further reading

- `git help bisect`, section "Bisect run": the exit-code protocol of 0, 125, 1 to 127, and the abort range.
- The [`git bisect`](https://git-scm.com/docs/git-bisect) documentation online (free).

## Key terms

- **Bisection**: binary search over a commit range for the first commit at which a property changes, taking $\lceil \log_2 n \rceil$ tests.
- **Consistent oracle**: a test that says "good" for every commit before the culprit and "bad" for every commit from it on.
- **Reproduction envelope**: the fixed metric, workload, host, configuration, and sampling under which every revision is measured.
- **Regression rule**: a predeclared threshold, here a relative and an absolute bound, that decides good or bad.
- **`git bisect run` exit codes**: 0 good, 125 skip, 1 to 127 (other than 125) bad, 128 or more abort.

## Summary

- Regressions are found by search, not by staring: a gated benchmark plus `git bisect run` finds the commit in $\log_2 n$ steps.
- A bisect is only as good as its oracle: fix the workload, host, configuration, and sampling, and write the rule before looking at commits.
- Require both a relative and an absolute bound, so the change is large in proportion and in milliseconds.
- Verify that the base is good and the head is bad before bisecting, then confirm the first bad commit and its neighbours with repeated runs.
- Skip untestable revisions with 125 and abort on runner failure; never count them as bad.
- The record names the owner, the rollback, and the approval threshold, and keeps the raw results.

## Review questions

1. Why must base be verified good and head verified bad before bisecting?
   <details><summary>Answer</summary>Bisection assumes the property flips from good to bad somewhere in the range. If base is already bad under the rule (or head is not bad), the search still returns a commit, but it is meaningless. Two endpoint measurements check the assumption.</details>
2. What happens if the gate script exits 1 when a revision fails to build?
   <details><summary>Answer</summary>Git counts the revision as bad. If the unbuildable commit lies before the real culprit, bisection blames it (or something near it) instead. Exiting 125 tells git to skip that revision and choose another.</details>
3. Why require both "at least 15%" and "more than 3 ms"?
   <details><summary>Answer</summary>A relative bound alone would flag a 0.5 ms change on a 3 ms path, which is within run-to-run noise. An absolute bound alone would treat 4 ms as significant on a 2 s path. Both together flag changes that are large in proportion and in real time.</details>
4. The bisect names commit X. X and its parent measure 49 ms and 47 ms. Is X the culprit?
   <details><summary>Answer</summary>Not on this evidence. A 2 ms step is below the 3 ms bound, so the oracle should not have separated them. Either the regression accumulated over several commits, or a noisy measurement misled the search. The confirmation step exists to catch this; widen the confirmation to more neighbours and rerun.</details>
5. How many measurements does the bisect of a 128-commit range cost, end to end?
   <details><summary>Answer</summary>Seven bisection measurements ($\lceil \log_2 128 \rceil = 7$), plus 2 endpoint checks and 3 confirmation measurements: about 12, each of 9 rounds (2 warm-up and 7 runs). A linear search would need 128.</details>
