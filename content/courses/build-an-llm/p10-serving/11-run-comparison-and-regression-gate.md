# Run comparison and regression gate

## Overview

| | |
|---|---|
| **Module** | `load.02` · side · Go · Pass 7 · 2 to 3 h reading (optional) |
| **Solution** | [`go/loadgen/compare/compare.go`](../../../../course/solutions/go/loadgen/compare/compare.go): `ParseRegress` and `ParseMetric`, `Samples` (histogram reports expanded into samples), `Stat.Value` (nearest-rank statistics by selection), `PermutationTest` (a seeded one-sided two-sample permutation test), and `Gate` / `Compare`, which fail a head run only when it is worse beyond the budget AND significantly so · the `loadgen compare` verb of [`go/cmd/loadgen`](../../../../course/solutions/entry/go/cmd/loadgen/main.go) |
| **Contract** | the input: [`formats/loadgen-report.schema.json`](../../../../course/contracts/formats/loadgen-report.schema.json) · the role `{loadgen} compare`: [`spec/cli-roles.md`](../../../../course/contracts/spec/cli-roles.md) |
| **Needs** | `load.01` the report, histograms, and `rng.Stream` ([chapter](10-load-generator.md)) · reading: `M07.5` permutation tests ([chapter](../../mathematics-for-computing/07-probability-statistics/05-hypothesis-tests.md)), re-implemented here in Go |
| **Used by** | no module: the `{loadgen} compare` entry point runs it, in `dep.05`'s optional perf-gate job and in the `ops.07` incident walkthrough (`git bisect run`) |
| **Optional depth** | Good, *Permutation, Parametric, and Bootstrap Tests of Hypotheses*, ch. 3; [Kalibera and Jones, Rigorous Benchmarking in Reasonable Time](https://kar.kent.ac.uk/33611/) (free); [Mytkowicz et al., Producing Wrong Data Without Doing Anything Obviously Wrong](https://dl.acm.org/doi/10.1145/1508244.1508275) |

## Key Takeaways

- A regression is a head run worse than the base by more than the budget AND unlikely under noise; either condition alone cries wolf.
- The permutation test asks how often a random relabeling of the pooled samples is at least as bad as what was observed; it needs no distributional assumption about latencies.
- One-sided: a faster head is never a regression, so its p-value is near 1.
- $p = (1 + \text{count})/(1 + N)$ is never 0, and a seeded shuffle gives the same verdict twice on the same files.
- Over many simulated pairs, identical runs pass at least 95% of the time and a +10% shift is caught at least 90% of the time.

## How to read this chapter

Sections 1 to 3 explain why a gate needs both an effect-size budget and a significance test, how a permutation test works, and work one comparison by hand. Section 4 fixes the interface. Section 5 walks through the solution: parsing, samples from histograms, the statistic, the test, and the gate. Try to predict each step before reading it.

The entry point runs it as `loadgen compare <base.json> <head.json> --metric ttft_p95 --max-regress 5% [--alpha 0.05] [--permutations 1000] [--seed 0]`: it reads the two reports, calls `compare.Compare`, prints the verdict as its final JSON line, and exits 1 when `Regressed`.

---

## 1. Why now

`load.01` turns a run into numbers; nothing yet turns two runs into a decision. CI (`dep.05`) must fail a pull request that makes TTFT worse, and the `ops.07` incident has twenty commits and asks which one did it, which `git bisect run` can answer only if one command exits 0 for "fine" and 1 for "regressed". Latency is noisy: two runs of the same build differ by several percent, so "head p95 > base p95" fails half of all pull requests, and "head p95 > 1.05 base p95" still fails a few percent of identical builds and misses real regressions hidden in noise. This module is the gate: an effect-size budget plus a significance test. It is optional: only the `{loadgen} compare` entry point calls it, and `dep.05` runs its perf-gate job only when the workflow has one.

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

Base e2e samples $\{10, 20, 30\}$ ms, head $\{40, 50, 60\}$ ms, metric `e2e_mean`, budget 5%.

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

### What a correct solution must handle

Samples come from `math/rand/v2` with fixed seeds (lognormal latencies); the statistical case bounds its counts binomially at about $p = 10^{-3}$.

| Case | Checks | Why it matters downstream |
|---|---|---|
| The hand comparison | section 3: statistics, $\Delta = 1.5$, $p \approx 0.05$, not regressed | the worked example |
| Metric and budget parsing | `5%` is 0.05; metric names map to report keys; bad input refused | the CLI's flags mean what they say |
| Nearest rank, input untouched | 300 random slices with ties: equal to the sorted rank; input unchanged | the statistic and the split stay correct |
| Samples from the histogram | buckets to samples; error_rate to 0/1; a missing metric is an error | reports become test data |
| Seeded and one-sided | same seed, same p; faster head $p > 0.9$; 30% slower $p < 0.01$ | CI verdicts are reproducible |
| p is never zero | completely separated samples: $p = 1/1000$ | an honest bound, not certainty |
| Both budget and significance | 1 sample each: noise; 2% on 4000 samples: within budget; 10%: regressed | neither condition alone decides |
| A zero baseline | 0 to 0 errors is no change; 0 to 30 of 170 is $+\infty$ and regressed | error-rate gates |
| Reports end to end | two `load.01` reports, default alpha and permutations: 30% slower fails, self passes | the CI path |
| Identical runs pass, shifts are caught | 100 pairs each: at most 12 false alarms, at least 82 catches | the gate's two promises |

## 5. The solution, step by step

### 5.1 From first principles

1. **A gate must make two separate claims.** "Worse by more than we tolerate" is about effect size; "not explained by noise" is about evidence. Either alone fails in a predictable way (section 2.3), so the verdict is the conjunction $\Delta > \beta \wedge p < \alpha$.
2. **Latencies are not normal.** They are skewed, multi-modal, and heavy-tailed, so a t-test's assumptions do not hold, and a percentile has no simple standard error. A permutation test needs only exchangeability under the null: if nothing changed, the labels "base" and "head" are arbitrary (section 2.2).
3. **Only "worse" is a regression.** The test statistic is $S(\text{head}) - S(\text{base})$, and only relabelings at least that large count, so a faster head gets $p$ near 1.
4. **A Monte Carlo p-value must not claim certainty.** The observed labeling is one of the possible labelings, so it is counted: $p = (1 + \text{count})/(1 + N) \ge 1/(1 + N)$.
5. **The verdict must repeat.** CI and `git bisect run` need the same answer on the same files, so the shuffles draw from a seeded `rng.Stream(seed, PurposeShuffle)` (`load.01`).
6. **The report holds histograms, not raw samples.** Each bucket `[upper_ms, count]` becomes `count` copies of `upper_ms`; that keeps the sample size (which drives the test's power) and is within $1/128$ of the true values. Errors become 0/1 samples, so the same machinery gates `error_rate`.
7. **The statistic is computed thousands of times.** $N = 1000$ relabelings each need a percentile of a fresh split, so selection in expected $O(n)$, on a copy, beats sorting.

### 5.2 Parsing the gate's inputs

[`compare.go`](../../../../course/solutions/go/loadgen/compare/compare.go):

```go
// ParseRegress reads a budget: "5%" or "0.05" both give 0.05.
func ParseRegress(s string) (float64, error) {
	s = strings.TrimSpace(s)
	pct := strings.HasSuffix(s, "%")
	v, err := strconv.ParseFloat(strings.TrimSuffix(s, "%"), 64)
	if err != nil || v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, fmt.Errorf("compare: bad regression budget %q", s)
	}
	if pct {
		v /= 100 // "5%" must not become a budget of 5 (500%)
	}
	return v, nil
}
```

`ParseMetric` splits `ttft_p95` at the last underscore into the report key `ttft_ms` and a `Stat{Name: "p95", Q: 0.95}`; `error_rate` maps to the mean of 0/1 samples.

### 5.3 Samples from a report

```go
func Samples(r *loadgen.Report, metric string) ([]float64, error) {
	if metric == "error_rate" {
		// ... errors must be within [0, requests]
		out := make([]float64, r.Requests)
		for i := 0; i < r.Errors; i++ {
			out[i] = 1 // the mean of these is the error rate
		}
		return out, nil
	}
	rows, ok := r.Histogram[metric]
	// ... a missing metric is an error
	var out []float64
	for _, b := range rows {
		n := int(b[1])
		// ... the count must be a non-negative integer
		for i := 0; i < n; i++ {
			out = append(out, b[0]) // count copies of the bucket's upper bound
		}
	}
	return out, nil
}
```

### 5.4 The statistic by selection

```go
func (s Stat) Value(xs []float64) float64 {
	if len(xs) == 0 {
		return math.NaN()
	}
	if s.Mean {
		// ... the plain mean
	}
	// the same rank rule as loadgen.Histogram.Quantile, so the gate and the report agree
	rank := int(math.Ceil(s.Q*float64(len(xs)) - 1e-9))
	// ... clamp to [1, n]
	tmp := append([]float64(nil), xs...) // a copy: the caller's slice (the shuffled pool) is untouched
	return nth(tmp, rank-1)
}
```

`nth` is Hoare-partition quickselect with a median-of-three pivot, so already sorted input (common: histogram expansion produces sorted runs) stays linear in expectation.

### 5.5 The permutation test

```go
func PermutationTest(base, head []float64, s Stat, n int, r *rng.PCG32) float64 {
	obs := s.Value(head) - s.Value(base) // positive: head is worse
	pool := make([]float64, 0, len(base)+len(head))
	pool = append(append(pool, base...), head...)
	nb := len(base)
	count := 0
	for i := 0; i < n; i++ {
		// a uniformly random relabeling: Fisher-Yates from load.01's PCG32
		r.Shuffle(len(pool), func(a, b int) { pool[a], pool[b] = pool[b], pool[a] })
		// one-sided: only relabelings at least as bad as observed count
		if s.Value(pool[nb:])-s.Value(pool[:nb]) >= obs {
			count++
		}
	}
	return float64(1+count) / float64(1+n) // the observed labeling is one of them: p > 0
}
```

The pool is shuffled in place each time, which is fine: every shuffle of a uniformly random permutation is again uniformly random, so the starting order does not matter.

### 5.6 The gate

```go
func Gate(base, head []float64, s Stat, o Options) (Verdict, error) {
	// ... both runs need samples; Alpha 0.05 and Permutations 1000 by default
	v := Verdict{Metric: o.Metric, Base: s.Value(base), Head: s.Value(head)}
	switch {
	case v.Base > 0:
		v.Delta = (v.Head - v.Base) / v.Base
	case v.Head > v.Base:
		v.Delta = math.Inf(1) // from 0 errors to some errors: an unbounded relative rise
	}                         // 0 to 0: Delta stays 0, no change
	v.PValue = PermutationTest(base, head, s, n, rng.Stream(o.Seed, rng.PurposeShuffle))
	v.Regressed = v.Delta > o.MaxRegress && v.PValue < alpha // both conditions, always
	return v, nil
}
```

Section 3 by this code: `Base = 20`, `Head = 50`, `Delta = 1.5`; $p \approx 1/20 = 0.05$ (the only relabeling as bad as observed is the observed one, out of 20 equally likely), which is not `< 0.05`, so `Regressed` is false.

The entry point's `loadgen compare <base.json> <head.json> --metric ttft_p95 --max-regress 5% [--alpha 0.05] [--permutations 1000] [--seed 0]` in [`go/cmd/loadgen`](../../../../course/solutions/entry/go/cmd/loadgen/main.go) reads the two reports, calls `compare.Compare`, prints the verdict as its final JSON line, and exits 1 when `Regressed`, which is what `git bisect run` needs.

Full solution: [`compare.go`](../../../../course/solutions/go/loadgen/compare/compare.go).

### 5.7 Why this solution

- **Cost.** $O(N (n_a + n_b))$ expected: each of $N$ relabelings shuffles the pool and runs two selections. With 1000 relabelings and a few thousand samples that is well under a second, small next to the load runs it compares.
- **Why not a t-test or Mann-Whitney?** A t-test compares means under approximate normality, and the gate usually needs a percentile. Mann-Whitney tests a shift in the whole distribution, not the statistic the SLO names. The permutation test uses exactly the statistic in the budget, with no distributional assumption.
- **Why the upper bound of each bucket rather than the midpoint?** It is what the report's percentiles already use, so `S(head)` equals the report's own p95; and the shift from the true value is the same for base and head, so it cancels in the comparison.
- **Tradeoffs accepted.** Monte Carlo $p$ has resolution $1/(1 + N)$; with $N = 1000$ and $\alpha = 0.05$ that is ample. Expanded histograms mean samples within a bucket tie; ties make the test slightly conservative, never anti-conservative.

### 5.8 Explain it back

**Two builds are identical. How often does a budget-only gate (`Delta > 5%`) fail them, and how does the permutation test prevent it?**

<details><summary>Answer</summary>

Two runs of the same build differ by several percent in p95, so a fixed 5% threshold fails some fraction of identical pairs, every time noise happens to exceed it. Requiring $p < \alpha$ as well means the difference must be unusual among random relabelings of the pooled samples; for identical builds it usually is not, so the false-alarm rate drops to at most about $\alpha$.

</details>

**Why is $p$ computed as $(1 + \text{count})/(1 + N)$ and not $\text{count}/N$?**

<details><summary>Answer</summary>

The observed labeling is itself one of the labelings under the null, so it belongs in both numerator and denominator. $\text{count}/N$ can be 0 for completely separated samples, which claims certainty from a finite random sample; the corrected form is at least $1/(1 + N)$, an honest bound, and is a valid p-value.

</details>

**A head run is 30% faster. What does the test report, and why is that the right behaviour?**

<details><summary>Answer</summary>

`obs` is negative, and almost every relabeling gives a difference above it, so `count` is close to $N$ and $p$ is near 1: not regressed. The question CI asks is "did this get worse?"; a two-sided test would flag the speedup as a significant change and fail a good pull request.

</details>

**Why does `Value` copy `xs` before selecting?**

<details><summary>Answer</summary>

`nth` permutes its input. In the permutation test, `Value` is called on slices of the pool; reordering them would change which values are in the "base" and "head" halves mid-comparison and corrupt later relabelings, and callers of `Value` would see their data reordered. The copy keeps every call pure.

</details>

**Base has 0 errors, head has 30 of 170. What are $\Delta$ and the verdict?**

<details><summary>Answer</summary>

`Base = 0`, `Head > 0`, so `Delta = +Inf`, which exceeds any budget. With 30 ones among 170 head samples and none in the base, few random relabelings put that many errors in the head, so $p$ is far below 0.05: regressed. If both had 0 errors, `Delta` would stay 0 and nothing would be flagged.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| A two-sided test | a faster build fails CI | `PermutationTest` counts only relabelings with `d* >= obs`; see 5.5 |
| $p = \text{count}/N$ | p = 0 claims certainty from a finite sample | $p = (1 + \text{count})/(1 + N)$; see 5.5 |
| The budget alone decides | identical builds fail CI several percent of the time | `Regressed` also requires `PValue < alpha`; see 5.6 |
| Significance alone decides | a real but allowed 2% change fails CI | `Regressed` also requires `Delta > MaxRegress`; see 5.6 |
| `5%` read as 5 | a head six times slower passes | `ParseRegress` divides by 100 after a `%`; see 5.2 |
| One sample per bucket | counts ignored: the test sees a handful of values | `Samples` appends `count` copies per bucket; see 5.3 |
| Never shuffling | every relabeling equals the observed one: p = 1, nothing is ever caught | the pool is reshuffled with the seeded stream before every relabeling; see 5.5 |
| Rank by floor | the statistic differs from the report's percentile | `Value` uses `ceil(q * n - 1e-9)`, `load.01`'s rule; see 5.4 |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `load.01` | `Report`, its histograms, and `rng.Stream` for the shuffles |

`dep.05`'s optional perf-gate job runs `{loadgen} compare` on main against the last good run; the `ops.07` incident walkthrough bisects a perf regression with it.

## Going further

| This solution | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| a permutation test per metric | Bencher, Conbench | continuous benchmarking with history and change-point detection | [Bencher](https://bencher.dev/), [Conbench](https://github.com/conbench/conbench) |
| one comparison | `benchstat` | Mann-Whitney U across repeated runs, with confidence intervals | [benchstat](https://pkg.go.dev/golang.org/x/perf/cmd/benchstat) |
| a fixed budget | change-point detection (E-divisive) | finds when a series shifted, not just whether two runs differ | [MongoDB's change point detection](https://arxiv.org/abs/2003.00584) |
