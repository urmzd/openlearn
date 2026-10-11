<!-- ol:module load.02 -->
# Run comparison and regression gate

## Overview

| | |
|---|---|
| **Module** | `load.02` · side · Go · Pass 7 · 3 to 5 h (optional) |
| **You build** | `go/loadgen/compare/compare.go`: metric and budget parsing, histogram reports expanded into samples, nearest-rank statistics by selection, a seeded one-sided two-sample permutation test, and the gate that fails a head run only when it is worse beyond the budget AND significantly so |
| **Contract** | the input: [`formats/loadgen-report.schema.json`](../../../course/contracts/formats/loadgen-report.schema.json) · the role `{loadgen} compare`: [`spec/cli-roles.md`](../../../course/contracts/spec/cli-roles.md) |
| **Tests** | `course/tests/go/load_02/`, 10 tests (what they check: section 4) |
| **Needs** | `load.01` your report, histograms, and `rng.Stream` ([chapter](10-load-generator.md)) · reading: `M07.5` permutation tests ([chapter](../../../math/07-probability-statistics/05-hypothesis-tests.md)), re-implemented here in Go |
| **Used by** | no module: your `{loadgen} compare` entry point runs it, in `dep.05`'s optional perf-gate job and in drill `ops.07` (`git bisect run`) |
| **Milestone** | none (optional; not part of `MS-L10`) |
| **Optional depth** | Good, *Permutation, Parametric, and Bootstrap Tests of Hypotheses*, ch. 3; [Kalibera and Jones, Rigorous Benchmarking in Reasonable Time](https://kar.kent.ac.uk/33611/) (free); [Mytkowicz et al., Producing Wrong Data Without Doing Anything Obviously Wrong](https://dl.acm.org/doi/10.1145/1508244.1508275) |

## Key Takeaways

- A regression is a head run worse than the base by more than the budget AND unlikely under noise; either condition alone cries wolf (`TestGateNeedsBothBudgetAndSignificance`).
- The permutation test asks how often a random relabeling of the pooled samples is at least as bad as what was observed; it needs no distributional assumption about latencies (`TestCompareHandExample`).
- One-sided: a faster head is never a regression, so its p-value is near 1 (`TestPermutationTestIsSeededAndOneSided`).
- $p = (1 + \text{count})/(1 + N)$ is never 0, and a seeded shuffle gives the same verdict twice on the same files (`TestPValueIsNeverZero`).
- Over many simulated pairs, identical runs pass at least 95% of the time and a +10% shift is caught at least 90% of the time (`TestIdenticalRunsPassAndShiftsAreCaught`).

## How to work this chapter

```bash
ol start load.02             # stubs go/loadgen/compare/compare.go
ol tests load.02
ol check load.02             # exit code is the verdict
ol check load.02 --ref-deps  # only if load.01 is not passing yet
ol diff  load.02
```

Your `{loadgen} compare <base.json> <head.json> --metric ttft_p95 --max-regress 5% [--alpha 0.05] [--permutations 1000] [--seed 0]` reads the two reports, calls `compare.Compare`, prints the verdict as its final JSON line, and exits 1 when `Regressed`.

---

## 1. Why now

`load.01` turns a run into numbers; nothing yet turns two runs into a decision. CI (`dep.05`) must fail a pull request that makes TTFT worse, and drill `ops.07` hands you twenty commits and asks which one did it, which `git bisect run` can answer only if one command exits 0 for "fine" and 1 for "regressed". Latency is noisy: two runs of the same build differ by several percent, so "head p95 > base p95" fails half of all pull requests, and "head p95 > 1.05 base p95" still fails a few percent of identical builds and misses real regressions hidden in noise. This module writes the gate: an effect-size budget plus a significance test. It is optional: only your `{loadgen} compare` entry point calls it, so no milestone requires it, and `dep.05` checks its perf-gate job only when your workflow has one.

## 2. Principles

| Symbol | Meaning | Type |
|---|---|---|
| $a_1..a_{n_a}$, $b_1..b_{n_b}$ | base and head samples of one metric (ms, or 0/1 errors) | `[]float64` |
| $S(\cdot)$ | the statistic: nearest-rank percentile or mean | function |
| $\Delta = (S(b) - S(a))/S(a)$ | relative change, worse when positive | `float64` |
| $\beta$ | the budget (`--max-regress 5%` is 0.05) | `float64` |
| $N$ | random relabelings; $\alpha$ the significance level (0.05) | |

### 2.1 Samples from a report

A report keeps each latency as histogram buckets `[upper_ms, count]` (`load.01`): a bucket becomes `count` copies of its upper bound (within 1/128 of the true values). `error_rate` becomes one 1 per failed request and one 0 per other request, so the same test covers it.

### 2.2 The permutation test

If base and head come from the same distribution, the labels are arbitrary: every way of choosing which $n_a$ of the $n_a + n_b$ pooled samples are "base" is equally likely. Observe $d = S(b) - S(a)$. Draw $N$ random relabelings (Fisher-Yates with `rng.Stream(seed, shuffle)`, `load.01`), compute $d^*$ for each, and count those with $d^* \ge d$. The one-sided p-value is $p = (1 + \#\{d^* \ge d\})/(1 + N)$: the observed labeling is one of the possible ones, so $p \ge 1/(1 + N)$. Only "worse" counts: a head that got faster has $d < 0$ and $p$ near 1.

### 2.3 The gate

Regressed $\iff \Delta > \beta$ and $p < \alpha$. The budget keeps tiny real changes (a 2% slowdown on a huge sample is significant but allowed); the test keeps big apparent changes on tiny samples (one request each, ten times slower, $p \approx 0.5$) from failing CI. With a base statistic of 0 (no errors), any rise is $\Delta = +\infty$.

### 2.4 Statistics by selection

The nearest-rank percentile is the $\lceil qn - 10^{-9} \rceil$-th smallest value, the same rule as `load.01`'s histogram. Each relabeling needs it on a fresh split, so the gate uses quickselect (expected $O(n)$) on a copy, never sorting or reordering the caller's slice.

## 3. Worked example by hand

Base e2e samples $\{10, 20, 30\}$ ms, head $\{40, 50, 60\}$ ms, metric `e2e_mean`, budget 5% (test `TestCompareHandExample`).

1. $S(a) = 20$, $S(b) = 50$, $\Delta = (50 - 20)/20 = 1.5 > 0.05$.
2. $d = 30$. There are $\binom{6}{3} = 20$ ways to pick three of the six values as "head"; only $\{40, 50, 60\}$ gives a difference of 30 or more (every other choice moves a smaller value into the head mean). So the exact $p = 1/20 = 0.05$, and 20000 random relabelings estimate it within 0.005.
3. $p = 0.05$ is not below $\alpha = 0.05$: three samples cannot establish a regression, however large the change. Not regressed.

With 400 samples per run instead, a 10% shift gives $p$ far below 0.05 and $\Delta \approx 0.10 > 0.05$: regressed.

## 4. The interface

```go
// go/loadgen/compare/compare.go
type Stat struct { Name string; Q float64; Mean bool }
func (s Stat) Value(xs []float64) float64                       // nearest rank or mean; xs unchanged; NaN when empty
type Options struct { Metric string; MaxRegress, Alpha float64; Permutations int; Seed uint64 }   // Alpha 0.05, Permutations 1000 by default
type Verdict struct { Metric string; Base, Head, Delta, PValue float64; Regressed bool }
func ParseRegress(s string) (float64, error)                    // "5%" or "0.05"
func ParseMetric(m string) (string, Stat, error)                // "ttft_p95" -> ("ttft_ms", p95); "error_rate"
func Samples(r *loadgen.Report, metric string) ([]float64, error)
func PermutationTest(base, head []float64, s Stat, n int, r *rng.PCG32) float64
func Gate(base, head []float64, s Stat, o Options) (Verdict, error)
func Compare(base, head *loadgen.Report, o Options) (Verdict, error)
```

### What the tests check

Samples come from `math/rand/v2` with fixed seeds (lognormal latencies); the statistical test bounds its counts binomially at about $p = 10^{-3}$.

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestCompareHandExample` | unit | section 3: statistics, $\Delta = 1.5$, $p \approx 0.05$, not regressed | the worked example |
| `TestParseMetricAndBudget` | unit | `5%` is 0.05; metric names map to report keys; bad input refused | the CLI's flags mean what they say |
| `TestValueIsNearestRankAndLeavesInputAlone` | property | 300 random slices with ties: equal to the sorted rank; input unchanged | the statistic and the split stay correct |
| `TestSamplesExpandTheHistogram` | unit | buckets to samples; error_rate to 0/1; a missing metric is an error | reports become test data |
| `TestPermutationTestIsSeededAndOneSided` | unit | same seed, same p; faster head $p > 0.9$; 30% slower $p < 0.01$ | CI verdicts are reproducible |
| `TestPValueIsNeverZero` | boundary | completely separated samples: $p = 1/1000$ | an honest bound, not certainty |
| `TestGateNeedsBothBudgetAndSignificance` | unit | 1 sample each: noise; 2% on 4000 samples: within budget; 10%: regressed | neither condition alone decides |
| `TestZeroBaseline` | boundary | 0 to 0 errors is no change; 0 to 30 of 170 is $+\infty$ and regressed | error-rate gates |
| `TestCompareReportsEndToEnd` | unit | two `load.01` reports, default alpha and permutations: 30% slower fails, self passes | the CI path |
| `TestIdenticalRunsPassAndShiftsAreCaught` | statistical | 100 pairs each: at most 12 false alarms, at least 82 catches | the gate's two promises |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| A two-sided test | a faster build fails CI | `TestPermutationTestIsSeededAndOneSided` (mutant `s01`) |
| $p = \text{count}/N$ | p = 0 claims certainty from a finite sample | `TestPValueIsNeverZero` (mutant `s02`) |
| The budget alone decides | identical builds fail CI several percent of the time | `TestGateNeedsBothBudgetAndSignificance` (mutant `s03`) |
| Significance alone decides | a real but allowed 2% change fails CI | `TestGateNeedsBothBudgetAndSignificance` (mutant `s04`) |
| `5%` read as 5 | a head six times slower passes | `TestParseMetricAndBudget` (mutant `s05`) |
| One sample per bucket | counts ignored: the test sees a handful of values | `TestSamplesExpandTheHistogram` (mutant `s06`) |
| Never shuffling | every relabeling equals the observed one: p = 1, nothing is ever caught | `TestPermutationTestIsSeededAndOneSided` (mutant `s07`) |
| Rank by floor | the statistic differs from the report's percentile | `TestValueIsNearestRankAndLeavesInputAlone` (mutant `s08`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `load.01` | `Report`, its histograms, and `rng.Stream` for the shuffles |

`dep.05`'s optional perf-gate job runs `{loadgen} compare` on main against the last good run; drill `ops.07` bisects a perf regression with it.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| a permutation test per metric | Bencher, Conbench | continuous benchmarking with history and change-point detection | [Bencher](https://bencher.dev/), [Conbench](https://github.com/conbench/conbench) |
| one comparison | `benchstat` | Mann-Whitney U across repeated runs, with confidence intervals | [benchstat](https://pkg.go.dev/golang.org/x/perf/cmd/benchstat) |
| a fixed budget | change-point detection (E-divisive) | finds when a series shifted, not just whether two runs differ | [MongoDB's change point detection](https://arxiv.org/abs/2003.00584) |
