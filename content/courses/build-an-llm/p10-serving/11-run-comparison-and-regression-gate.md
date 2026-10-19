# Run comparison and the regression gate

[The load generator](10-load-generator.md) turns a run into numbers. Nothing yet turns two runs into a decision. CI needs to fail a pull request that makes TTFT worse, and when a performance incident lands on twenty commits, you want to ask which one did it. `git bisect run` can answer that only if one command exits 0 for "fine" and 1 for "regressed".

The difficulty is that latency is noisy. Two runs of the same build differ by several percent, so "head p95 > base p95" fails half of all pull requests, and "head p95 > 1.05 base p95" still fails a few percent of identical builds while missing real regressions hidden in noise. This section builds the gate that does better: an effect-size budget combined with a significance test.

This section is optional. Only the `{loadgen} compare` entry point calls it, and the [CI pipeline](../../infrastructure/01-containers-kubernetes/05-ci-for-your-repo.md) in *Infrastructure* runs its perf-gate job only when the workflow has one. It builds on the report, histograms, and `rng.Stream` of the load generator, and re-implements in Go the permutation test from [hypothesis tests](../../mathematics-for-computing/07-probability-statistics/05-hypothesis-tests.md) in *Mathematics for Computing*.

The input format is [`formats/loadgen-report.schema.json`](../../../../course/contracts/formats/loadgen-report.schema.json), and the `{loadgen} compare` role is defined in [`spec/cli-roles.md`](../../../../course/contracts/spec/cli-roles.md). The solution is [`go/loadgen/compare/compare.go`](../../../../course/solutions/go/loadgen/compare/compare.go): `ParseRegress` and `ParseMetric`, `Samples` (histogram reports expanded into samples), `Stat.Value` (nearest-rank statistics by selection), `PermutationTest` (a seeded one-sided two-sample permutation test), and `Gate` / `Compare`, which fail a head run only when it is worse beyond the budget *and* significantly so. The `loadgen compare` verb lives in [`go/cmd/loadgen`](../../../../course/solutions/entry/go/cmd/loadgen/main.go) and runs as:

```bash
loadgen compare <base.json> <head.json> --metric ttft_p95 --max-regress 5% [--alpha 0.05] [--permutations 1000] [--seed 0]
```

It reads the two reports, calls `compare.Compare`, prints the verdict as its final JSON line, and exits 1 when `Regressed`.

> **Learning objectives**
> - Explain why a regression gate needs both an effect-size budget and a significance test.
> - Run a one-sided permutation test on a percentile or mean, and compute its p-value so that it is never zero.
> - Expand histogram reports into samples, including 0/1 samples for error rate.
> - Compute nearest-rank statistics by selection without disturbing the caller's data.

## Two claims, not one

Here is the notation.

| Symbol | Meaning | Type |
|---|---|---|
| $a_1..a_{n_a}$, $b_1..b_{n_b}$ | base and head samples of one metric (ms, or 0/1 errors) | `[]float64` |
| $S(\cdot)$ | the statistic: nearest-rank percentile or mean | function |
| $\Delta = (S(b) - S(a))/S(a)$ | relative change, worse when positive | `float64` |
| $\beta$ | the budget (`--max-regress 5%` is 0.05) | `float64` |
| $N$ | random relabelings; $\alpha$ the significance level (0.05) | |

A gate must make two separate claims. "Worse by more than we tolerate" is about effect size. "Not explained by noise" is about evidence. Either alone fails in a predictable way, so the verdict is the conjunction:

$$\text{Regressed} \iff \Delta > \beta \ \text{and}\ p < \alpha.$$

The budget keeps tiny real changes from failing CI: a 2% slowdown on a huge sample is significant but allowed. The test keeps big apparent changes on tiny samples from failing CI: one request each, ten times slower, gives $p \approx 0.5$. With a base statistic of 0 (no errors), any rise is $\Delta = +\infty$.

## Samples from a report

A report does not hold raw samples. It keeps each latency as histogram buckets `[upper_ms, count]`, so a bucket becomes `count` copies of its upper bound, which is within 1/128 of the true values. That keeps the sample size, which drives the test's power. `error_rate` becomes one 1 per failed request and one 0 per other request, so the same test covers it.

## The permutation test

Latencies are not normal. They are skewed, multi-modal, and heavy-tailed, so a t-test's assumptions do not hold, and a percentile has no simple standard error. A permutation test needs only exchangeability under the null.

If base and head come from the same distribution, the labels are arbitrary: every way of choosing which $n_a$ of the $n_a + n_b$ pooled samples are "base" is equally likely. So observe $d = S(b) - S(a)$. Draw $N$ random relabelings (Fisher-Yates with `rng.Stream(seed, shuffle)` from the load generator), compute $d^*$ for each, and count those with $d^* \ge d$. The one-sided p-value is

$$p = \frac{1 + \#\{d^* \ge d\}}{1 + N}.$$

The observed labeling is one of the possible ones, so it is counted, and $p \ge 1/(1 + N)$: a Monte Carlo p-value must not claim certainty. Only "worse" counts. A head that got faster has $d < 0$ and $p$ near 1, because a faster head is never a regression.

The verdict must also repeat: CI and `git bisect run` need the same answer on the same files, which is why the shuffles draw from a seeded stream.

## Statistics by selection

The nearest-rank percentile is the $\lceil qn - 10^{-9} \rceil$-th smallest value, the same rule as the load generator's histogram, so the gate and the report agree. The statistic is computed thousands of times, since each of $N = 1000$ relabelings needs it on a fresh split. So the gate uses quickselect (expected $O(n)$) on a copy, never sorting or reordering the caller's slice.

## Working it by hand

Take base e2e samples $\{10, 20, 30\}$ ms, head $\{40, 50, 60\}$ ms, metric `e2e_mean`, and a budget of 5%.

1. $S(a) = 20$, $S(b) = 50$, so $\Delta = (50 - 20)/20 = 1.5 > 0.05$.
2. $d = 30$. There are $\binom{6}{3} = 20$ ways to pick three of the six values as "head", and only $\{40, 50, 60\}$ gives a difference of 30 or more (every other choice moves a smaller value into the head mean). So the exact $p = 1/20 = 0.05$, and 20000 random relabelings estimate it within 0.005.
3. $p = 0.05$ is not below $\alpha = 0.05$. Three samples cannot establish a regression, however large the change. Not regressed.

With 400 samples per run instead, a 10% shift gives $p$ far below 0.05 and $\Delta \approx 0.10 > 0.05$: regressed.

## Designing the interface

The whole gate is one small package:

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

`Gate` works on samples and is what the tests exercise directly; `Compare` is `Gate` applied to two reports through `ParseMetric` and `Samples`.

## Building it step by step

### Reasoning from first principles

The design follows from seven observations, each already motivated above.

1. **A gate must make two separate claims**, so the verdict is $\Delta > \beta \wedge p < \alpha$.
2. **Latencies are not normal**, so the test is a permutation test, which needs only exchangeability under the null.
3. **Only "worse" is a regression.** The test statistic is $S(\text{head}) - S(\text{base})$, and only relabelings at least that large count.
4. **A Monte Carlo p-value must not claim certainty**, so $p = (1 + \text{count})/(1 + N) \ge 1/(1 + N)$.
5. **The verdict must repeat**, so the shuffles draw from a seeded `rng.Stream(seed, PurposeShuffle)`.
6. **The report holds histograms, not raw samples.** Each bucket becomes `count` copies of `upper_ms`, and errors become 0/1 samples.
7. **The statistic is computed thousands of times**, so selection in expected $O(n)$, on a copy, beats sorting.

### Parsing the gate's inputs

Parsing is where a silent unit error would do the most damage. In [`compare.go`](../../../../course/solutions/go/loadgen/compare/compare.go):

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

### Samples from a report

`Samples` expands either the error count or a histogram:

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

### The statistic by selection

The statistic uses the load generator's rank rule and selects on a copy:

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

`nth` is Hoare-partition quickselect with a median-of-three pivot, so already sorted input (common, since histogram expansion produces sorted runs) stays linear in expectation.

### The permutation test

With a statistic in hand, the test itself is a loop of shuffles:

```go
func PermutationTest(base, head []float64, s Stat, n int, r *rng.PCG32) float64 {
	obs := s.Value(head) - s.Value(base) // positive: head is worse
	pool := make([]float64, 0, len(base)+len(head))
	pool = append(append(pool, base...), head...)
	nb := len(base)
	count := 0
	for i := 0; i < n; i++ {
		// a uniformly random relabeling: Fisher-Yates from the load generator's PCG32
		r.Shuffle(len(pool), func(a, b int) { pool[a], pool[b] = pool[b], pool[a] })
		// one-sided: only relabelings at least as bad as observed count
		if s.Value(pool[nb:])-s.Value(pool[:nb]) >= obs {
			count++
		}
	}
	return float64(1+count) / float64(1+n) // the observed labeling is one of them: p > 0
}
```

The pool is shuffled in place each time, which is fine: a uniformly random shuffle of any permutation is again uniformly random, so the starting order does not matter.

### The gate

Finally, `Gate` combines effect size and evidence:

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

Run the hand-worked example through this code: `Base = 20`, `Head = 50`, `Delta = 1.5`; $p \approx 1/20 = 0.05$, because the only relabeling as bad as observed is the observed one, out of 20 equally likely. That is not `< 0.05`, so `Regressed` is false.

The `loadgen compare` verb in [`go/cmd/loadgen`](../../../../course/solutions/entry/go/cmd/loadgen/main.go) wraps this: it reads the two reports, calls `compare.Compare`, prints the verdict as its final JSON line, and exits 1 when `Regressed`, which is exactly what `git bisect run` needs.

The full solution is [`compare.go`](../../../../course/solutions/go/loadgen/compare/compare.go).

### Why this design

The cost is $O(N (n_a + n_b))$ expected: each of $N$ relabelings shuffles the pool and runs two selections. With 1000 relabelings and a few thousand samples that is well under a second, small next to the load runs it compares.

Why not a t-test or Mann-Whitney? A t-test compares means under approximate normality, and the gate usually needs a percentile. Mann-Whitney tests a shift in the whole distribution, not the statistic the SLO names. The permutation test uses exactly the statistic in the budget, with no distributional assumption.

Why the upper bound of each bucket rather than the midpoint? It is what the report's percentiles already use, so `S(head)` equals the report's own p95. And the shift from the true value is the same for base and head, so it cancels in the comparison.

Two tradeoffs are accepted. A Monte Carlo $p$ has resolution $1/(1 + N)$; with $N = 1000$ and $\alpha = 0.05$ that is ample. And expanded histograms mean that samples within a bucket tie; ties make the test slightly conservative, never anti-conservative.

## What can go wrong

The tests draw samples from `math/rand/v2` with fixed seeds (lognormal latencies), and the statistical case bounds its counts binomially at about $p = 10^{-3}$. A correct solution handles each of these:

| Case | Checks | Why it matters downstream |
|---|---|---|
| The hand comparison | the worked statistics, $\Delta = 1.5$, $p \approx 0.05$, not regressed | the worked example |
| Metric and budget parsing | `5%` is 0.05; metric names map to report keys; bad input refused | the CLI's flags mean what they say |
| Nearest rank, input untouched | 300 random slices with ties: equal to the sorted rank; input unchanged | the statistic and the split stay correct |
| Samples from the histogram | buckets to samples; error_rate to 0/1; a missing metric is an error | reports become test data |
| Seeded and one-sided | same seed, same p; faster head $p > 0.9$; 30% slower $p < 0.01$ | CI verdicts are reproducible |
| p is never zero | completely separated samples: $p = 1/1000$ | an honest bound, not certainty |
| Both budget and significance | 1 sample each: noise; 2% on 4000 samples: within budget; 10%: regressed | neither condition alone decides |
| A zero baseline | 0 to 0 errors is no change; 0 to 30 of 170 is $+\infty$ and regressed | error-rate gates |
| Reports end to end | two load generator reports, default alpha and permutations: 30% slower fails, self passes | the CI path |
| Identical runs pass, shifts are caught | 100 pairs each: at most 12 false alarms, at least 82 catches | the gate's two promises |

Most pitfalls come from dropping one half of the design. If the budget alone decides, identical builds fail CI several percent of the time; if significance alone decides, a real but allowed 2% change fails CI. `Regressed` requires both `Delta > MaxRegress` and `PValue < alpha`. A two-sided test makes a faster build fail CI; `PermutationTest` counts only relabelings with `d* >= obs`. And $p = \text{count}/N$ can be 0, claiming certainty from a finite sample; the solution uses $p = (1 + \text{count})/(1 + N)$.

> **Warning:** Reading `5%` as 5 sets a budget of 500%, and a head six times slower passes. The gate never complains; it just stops gating. `ParseRegress` divides by 100 after a `%`.

The remaining pitfalls are in the data handling. Taking one sample per bucket ignores the counts, so the test sees only a handful of values and has no power; `Samples` appends `count` copies per bucket. Never shuffling makes every relabeling equal the observed one, so $p = 1$ and nothing is ever caught; the pool is reshuffled with the seeded stream before every relabeling. And ranking by floor makes the statistic differ from the report's percentile; `Value` uses `ceil(q * n - 1e-9)`, the load generator's rule.

## Where this leads

This section reads the load generator's `Report`, its histograms, and `rng.Stream` for the shuffles. No later section builds on it as code; instead, the `{loadgen} compare` entry point is used directly. The optional perf-gate job of [CI for the platform repo](../../infrastructure/01-containers-kubernetes/05-ci-for-your-repo.md) runs it on main against the last good run, and the [performance regression bisect drill](../../systems/05-incident-response-and-chaos/07-performance-regression.md) in *Systems* uses it with `git bisect run` to find the commit that caused a slowdown.

Production systems take the idea further. Bencher and Conbench provide continuous benchmarking with history and change-point detection, instead of one permutation test per metric. Go's `benchstat` applies Mann-Whitney U across repeated runs, with confidence intervals. And change-point detection (E-divisive, as at MongoDB) replaces a fixed budget between two runs with finding when a series shifted.

### Further reading

- Good, *Permutation, Parametric, and Bootstrap Tests of Hypotheses*, ch. 3.
- Kalibera and Jones, [Rigorous Benchmarking in Reasonable Time](https://kar.kent.ac.uk/33611/) (free).
- Mytkowicz et al., [Producing Wrong Data Without Doing Anything Obviously Wrong](https://dl.acm.org/doi/10.1145/1508244.1508275).
- [Bencher](https://bencher.dev/), [Conbench](https://github.com/conbench/conbench), and [benchstat](https://pkg.go.dev/golang.org/x/perf/cmd/benchstat).
- [MongoDB's change point detection](https://arxiv.org/abs/2003.00584).

## Key terms

- **Regression gate**: a check that fails a head run only when it is worse than the base by more than a budget and the difference is unlikely under noise.
- **Effect-size budget**: the tolerated relative change $\beta$, so that $\Delta > \beta$ is "worse than we accept".
- **Permutation test**: a test that compares the observed statistic difference with its distribution over random relabelings of the pooled samples.
- **One-sided test**: a test that counts only differences in the "worse" direction, so a faster head has $p$ near 1.
- **Monte Carlo p-value**: $(1 + \text{count})/(1 + N)$, an estimate from $N$ random relabelings that counts the observed one and is never zero.
- **Quickselect**: a selection algorithm that finds the $k$-th smallest value in expected $O(n)$ without fully sorting.

## Summary

- A regression is a head run worse than the base by more than the budget *and* unlikely under noise; either condition alone cries wolf.
- The permutation test asks how often a random relabeling of the pooled samples is at least as bad as what was observed; it needs no distributional assumption about latencies.
- One-sided: a faster head is never a regression, so its p-value is near 1.
- $p = (1 + \text{count})/(1 + N)$ is never 0, and a seeded shuffle gives the same verdict twice on the same files.
- Over many simulated pairs, identical runs pass at least 95% of the time and a +10% shift is caught at least 90% of the time.

## Review questions

1. Two builds are identical. How often does a budget-only gate (`Delta > 5%`) fail them, and how does the permutation test prevent it?
   <details><summary>Answer</summary>

   Two runs of the same build differ by several percent in p95, so a fixed 5% threshold fails some fraction of identical pairs, every time noise happens to exceed it. Requiring $p < \alpha$ as well means the difference must be unusual among random relabelings of the pooled samples; for identical builds it usually is not, so the false-alarm rate drops to at most about $\alpha$.

   </details>

2. Why is $p$ computed as $(1 + \text{count})/(1 + N)$ and not $\text{count}/N$?
   <details><summary>Answer</summary>

   The observed labeling is itself one of the labelings under the null, so it belongs in both numerator and denominator. $\text{count}/N$ can be 0 for completely separated samples, which claims certainty from a finite random sample; the corrected form is at least $1/(1 + N)$, an honest bound, and is a valid p-value.

   </details>

3. A head run is 30% faster. What does the test report, and why is that the right behaviour?
   <details><summary>Answer</summary>

   `obs` is negative, and almost every relabeling gives a difference above it, so `count` is close to $N$ and $p$ is near 1: not regressed. The question CI asks is "did this get worse?"; a two-sided test would flag the speedup as a significant change and fail a good pull request.

   </details>

4. Why does `Value` copy `xs` before selecting?
   <details><summary>Answer</summary>

   `nth` permutes its input. In the permutation test, `Value` is called on slices of the pool; reordering them would change which values are in the "base" and "head" halves mid-comparison and corrupt later relabelings, and callers of `Value` would see their data reordered. The copy keeps every call pure.

   </details>

5. Base has 0 errors, head has 30 of 170. What are $\Delta$ and the verdict?
   <details><summary>Answer</summary>

   `Base = 0`, `Head > 0`, so `Delta = +Inf`, which exceeds any budget. With 30 ones among 170 head samples and none in the base, few random relabelings put that many errors in the head, so $p$ is far below 0.05: regressed. If both had 0 errors, `Delta` would stay 0 and nothing would be flagged.

   </details>

6. In the hand-worked example, why is the exact p-value $1/20$, and why does that mean "not regressed" even though the head mean is 2.5 times the base?
   <details><summary>Answer</summary>

   With six pooled values there are $\binom{6}{3} = 20$ equally likely ways to label three of them "head", and only the observed labeling $\{40, 50, 60\}$ gives a mean difference of at least 30. So $p = 1/20 = 0.05$, which is not strictly below $\alpha = 0.05$. Three samples per run are simply too few to rule out noise, however large the change looks.

   </details>
