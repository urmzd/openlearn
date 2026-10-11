<!-- ol:module ag.12 -->
# Paired A/B experiments

## Overview

| | |
|---|---|
| **Module** | `ag.12` · build · Go · Pass 10 · 3 to 4 h |
| **You build** | paired subject runs, paired bootstrap confidence intervals, and an experiment report |
| **Tests** | `course/tests/go/ag_12/`: `TestPairedDeltaHandExample`, `TestFailedPairsExcluded`, `TestBootstrapPairs`, `TestKnownEffect`, `TestNullCoverage`, `TestEvalSuiteSpecCarriesAB` |
| **Needs** | `ag.09` runner, `ag.10` scorers, `ag.11` judge, `load.01`, `dur.11`; reading: `M07.4`, `M07.5` |
| **Used by** | `craft.23`; optional MS-C2 |
| **Milestone** | MS-agent |

## 1. Why now

Two agent versions should answer the same cases so the comparison is not dominated by an easier or harder sample. The experiment runner pairs by case id, computes each case's score difference, and estimates uncertainty over those differences.

## 2. Principles

For case `i`, define `d_i = score_exp_i - score_base_i`. The reported effect is `mean(d)`. A paired bootstrap samples case indices with replacement and keeps each base/experiment pair together. Its percentile interval estimates uncertainty in the effect. The same seed and inputs must reproduce the same result.

Cases missing a score on either side are excluded from the paired metric and counted. They must never be converted to zero. Reject duplicate ids and mismatched scorer names before computing a comparison.

## 3. Worked example

Let base scores be `[0, 1, 0]` and experiment scores be `[1, 1, 0]`. The paired deltas are `[1, 0, 0]`, so the effect is `1/3`. An unpaired comparison would throw away the fact that the second and third cases were shared, widening or shifting the result unnecessarily.

## 4. The interface

Implement `RunExperiment` in `go/agent/eval/experiment`. Return each metric's mean delta, confidence bounds, p-value, and pair count. Upgrade the durable `EvalSuite` workflow so the same run can produce ordinary eval results and an optional A/B block. `TestPairedDeltaHandExample` checks the hand calculation; `TestFailedPairsExcluded` ensures failed pairs are counted but omitted; `TestBootstrapPairs` checks deterministic resampling; `TestKnownEffect` and `TestNullCoverage` check interval behavior; `TestEvalSuiteSpecCarriesAB` checks workflow serialization.

| Test | What it checks |
|---|---|
| `TestPairedDeltaHandExample` | worked paired-delta calculation |
| `TestFailedPairsExcluded` | excludes incomplete pairs |
| `TestBootstrapPairs` | resamples paired rows deterministically |
| `TestKnownEffect` | detects a synthetic positive effect |
| `TestNullCoverage` | interval coverage under a null effect |
| `TestEvalSuiteSpecCarriesAB` | forwards experiment configuration |

```bash
ol start ag.12
ol tests ag.12
ol check ag.12
```

## 5. Pitfalls

- Resample pairs, never the two arms independently.
- Preserve deterministic case ordering before sampling.
- Use the same scorer definition and seed on both arms.
- A confidence interval containing zero is uncertainty, not proof of equivalence.

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `ag.09` | runs each subject and summarizes scores |
| Back | `ag.10` | supplies deterministic scorers |
| Back | `ag.11` | supplies optional judge scores |
| Back | `load.01` | provides deterministic observation loading |
| Back | `dur.11` | provides the durable `EvalSuite` workflow being upgraded |
| Forward | `craft.23` | uses paired evaluations as regression tests |
| Forward | `C2` | optional post-training comparison |

MS-agent reports a known synthetic effect, while optional MS-C2 compares the post-trained model against its base.

## Going further

| Your piece | Production equivalent | What it adds |
|---|---|---|
| paired bootstrap | randomized controlled experiment | uncertainty while controlling case difficulty |
| EvalSuite workflow | release evaluation pipeline | durable, resumable experiment runs |
| delta and interval | sequential monitoring | evidence gathered over repeated releases |
