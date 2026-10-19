# Paired A/B experiments

## Overview

| | |
|---|---|
| **Module** | `ag.12` · build · Go · Pass 10 · 3 to 4 h |
| **Solution** | [`go/agent/eval/experiment/experiment.go`](../../../../course/solutions/go/agent/eval/experiment/experiment.go): `RunExperiment`, `Stat`, `Result`, `WithBootstrap`, `WithSeed`, `WithConcurrency` · [`go/workflows/eval_suite.go`](../../../../course/solutions/go/workflows/eval_suite.go): `EvalSuite` with the `Judge` and `AB` spec options |
| **Needs** | `ag.09` runner, `ag.10` scorers, `ag.11` judge, `load.01`, `dur.11`; reading: `M07.4`, `M07.5` |
| **Used by** | `craft.23`; optionally the C2 capstone's post-training comparison; the `ctl eval experiment` verb runs it |

## How to read this chapter

Sections 1 to 3 explain why pairing matters and compute a paired delta by hand. Section 4 fixes the interface. Section 5 walks through the solution and explains why each step is written the way it is; try to predict each step before reading it. In the finished system, `ctl eval experiment --base <model> --exp <model> --suite <file|agent-smoke>` ([`entry/go/cmd/ctl/eval.go`](../../../../course/solutions/entry/go/cmd/ctl/eval.go)) runs the docs agent on both models over the same cases and prints the paired delta, its interval, and its p-value.

## 1. Why now

Two agent versions should answer the same cases so the comparison is not dominated by an easier or harder sample. The experiment runner pairs by case id, computes each case's score difference, and estimates uncertainty over those differences.

## 2. Principles

For case `i`, define `d_i = score_exp_i - score_base_i`. The reported effect is `mean(d)`. A paired bootstrap samples case indices with replacement and keeps each base/experiment pair together. Its percentile interval estimates uncertainty in the effect. The same seed and inputs must reproduce the same result.

Cases missing a score on either side are excluded from the paired metric and counted. They must never be converted to zero. Reject duplicate ids and mismatched scorer names before computing a comparison.

The p-value answers a different question from the interval: if the two versions were equally good, how often would a mean delta at least this large appear by chance? Under that null hypothesis each $d_i$ is as likely to be $+d_i$ as $-d_i$ (which arm got the higher score is a coin flip), so flipping signs at random generates the null distribution. This is the **sign permutation test**.

## 3. Worked example

Let base scores be `[0, 1, 0]` and experiment scores be `[1, 1, 0]`. The paired deltas are `[1, 0, 0]`, so the effect is `1/3`. An unpaired comparison would throw away the fact that the second and third cases were shared, widening or shifting the result unnecessarily.

**The interval.** A resample of three deltas contains $K$ copies of the 1, binomial with $p = 1/3$: $P(K = 0) = 8/27 = 0.296$, so the 2.5th percentile is 0; $P(K = 3) = 1/27 = 0.037 > 0.025$, so the 97.5th percentile is 1. The 95% interval is $[0, 1]$: three cases cannot rule out "no effect".

**The p-value.** Flipping signs of `[1, 0, 0]` can only change the 1 into $-1$; the zeros stay zero. Every permutation has $|\bar d| = 1/3 \ge 1/3$, so every one is "at least as extreme" and $p = (B + 1)/(B + 1) = 1$. One improved case out of three is no evidence at all.

## 4. The interface

```go
package experiment // import "tinyllm/agent/eval/experiment"

type Stat struct { Delta, CILow, CIHigh, PValue float64; NPairs int }
type Result struct { Base, Exp string; Metrics map[string]Stat; NBoot int; Seed uint64 }
func RunExperiment(ctx context.Context, obs []eval.Observation, base, exp eval.Subject,
	scorers []eval.Scorer, opts ...Option) (*Result, error)
func WithBootstrap(n int, alpha float64) Option // default 2000, 0.05
func WithSeed(seed uint64) Option
func WithConcurrency(n int) Option              // default 4
```

`RunExperiment` returns each metric's mean delta, confidence bounds, p-value, and pair count. The durable `EvalSuite` workflow (`dur.11`) is upgraded so the same run can produce ordinary eval results and an optional A/B block: its input carries `scorers`, `judge`, `ab` (`base`, `exp`, `n_boot`, `alpha`), and `thresholds`, and every per-suite eval spec it hands the eval activity carries them too.

### What a correct solution must handle

| Case | What it checks |
|---|---|
| the paired-delta hand example | worked paired-delta calculation |
| failed pairs excluded | excludes incomplete pairs |
| bootstrap over pairs | resamples paired rows deterministically |
| a known effect | detects a synthetic positive effect |
| null coverage | interval coverage under a null effect |
| the eval-suite spec carries A/B | forwards experiment configuration |

## 5. The solution, step by step

### 5.1 From first principles

1. **Case difficulty is the biggest source of variance, so cancel it.** Scores differ far more between an easy and a hard case than between two versions on one case. Subtracting within a case removes the case's difficulty: $d_i$ measures only the version. So both arms must run on exactly the same cases, and the unit of analysis is the pair.
2. **Reuse the runner for each arm.** `ag.09` already guarantees case-then-sample row order, exclusion of failed scores, and determinism. Running it twice with the same observations and seed gives two row arrays that line up index by index.
3. **Pair, then check the pairing.** Line-up by index is only safe if each index holds the same case and sample in both arms; check `ID` and `Sample` per row and refuse to continue otherwise.
4. **A pair with a missing side is missing.** If either score errored, the delta does not exist; it is dropped, and the pair count `NPairs` shows how many remain.
5. **Resample pairs, as single values.** Each delta becomes a one-element group for `ag.09`'s `BootstrapCI`, so a resample draws whole pairs: the arms are never resampled independently.
6. **Test against the null directly.** Sign flips of the observed deltas give the distribution of $|\bar d|$ when the versions are equal; the p-value is the fraction at least as extreme as observed, with the $+1$ correction so it is never exactly 0.

### 5.2 The code, annotated

**Configuration and the two arms.**

```go
c := config{nBoot: 2000, alpha: .05, concurrency: 4}
for _, o := range opts {
	o(&c)
}
if c.nBoot < 1 || c.alpha <= 0 || c.alpha >= 1 || base == nil || exp == nil {
	return nil, fmt.Errorf("experiment: invalid configuration")
}
// Same observations, same scorers, same seed: the only difference is the subject.
br, err := eval.Run(ctx, "base", obs, scorers, eval.WithSubject("base", base), eval.WithConcurrency(c.concurrency), eval.WithSeed(c.seed))
// ... return err
er, err := eval.Run(ctx, "experiment", obs, scorers, eval.WithSubject("experiment", exp), eval.WithConcurrency(c.concurrency), eval.WithSeed(c.seed))
// ... return err
if len(br.Rows) != len(er.Rows) {
	return nil, fmt.Errorf("experiment: arms produced different rows")
}
```

Duplicate case ids and repeated scorer names are rejected by `eval.Run` itself (`ErrSuite`) before either arm runs any case, and the same scorer list is used for both arms, so scorer names match by construction.

**Paired deltas per metric.**

```go
for si, name := range br.Scorers {
	var diffs []float64
	for i := range br.Rows {
		b, e := br.Rows[i], er.Rows[i]
		if b.ID != e.ID || b.Sample != e.Sample { // ag.09's ordering makes this impossible; check anyway
			return nil, fmt.Errorf("experiment: unpaired row %d", i)
		}
		if b.Errored(si) || e.Errored(si) { // a missing side: no delta, never a 0
			continue
		}
		diffs = append(diffs, e.Scores[si].Value-b.Scores[si].Value)
	}
	if len(diffs) == 0 {
		continue // no complete pair: the metric is absent, not 0
	}
	// ...
}
```

**The interval.** Wrapping each delta as its own group reuses the cluster bootstrap of `ag.09` unchanged; the draws come from the same PCG32 `sample` stream, so a seed reproduces the interval.

```go
groups := make([][]float64, len(diffs))
for i, d := range diffs {
	groups[i] = []float64{d} // one pair = one resampling unit
}
lo, hi := eval.BootstrapCI(groups, c.nBoot, c.alpha, rng.Stream(c.seed, rng.PurposeSample))
```

**The p-value.**

```go
r := rng.Stream(c.seed, rng.PurposeSample) // a fresh stream: same seed, same p-value
observed := math.Abs(eval.Mean(diffs))
extreme := 0
for n := 0; n < c.nBoot; n++ {
	var s float64
	for _, d := range diffs {
		if r.Below(2) == 0 { // under the null, which arm won this case is a coin flip
			d = -d
		}
		s += d
	}
	if math.Abs(s/float64(len(diffs))) >= observed { // two-sided
		extreme++
	}
}
result.Metrics[name] = Stat{Delta: eval.Mean(diffs), CILow: lo, CIHigh: hi,
	PValue: float64(extreme+1) / float64(c.nBoot+1), // counts the observed arrangement itself
	NPairs: len(diffs)}
```

The $+1$ in numerator and denominator treats the observed signs as one of the permutations, so the p-value is at least $1/(B+1)$: with 2000 resamples, no result can claim more certainty than about 0.0005.

**The durable workflow.** `EvalSuite` runs one eval activity per suite, in order, so a crash in a later suite never reruns a finished one. The upgrade threads the new options through to every per-suite spec:

```go
func applyAdvancedSpecOptions(spec *evalSpec, in EvalSuiteInput) {
	// One place, used by both EvalSuite and MarshalSpec, so the workflow and the
	// CLI's dry run can never disagree on what the activity receives.
	spec.Scorers, spec.Judge, spec.AB, spec.Thresholds = in.Scorers, in.Judge, in.AB, in.Thresholds
}
```

Full solution: [`experiment.go`](../../../../course/solutions/go/agent/eval/experiment/experiment.go), [`eval_suite.go`](../../../../course/solutions/go/workflows/eval_suite.go).

### 5.3 Why this solution

- **Cost.** Two full eval runs (the model calls dominate), then per metric O(B · n) for the bootstrap and the same again for the permutation test.
- **Why not compare the two arms' intervals?** Overlapping per-arm intervals are the usual mistake: each arm's interval includes case-difficulty variance that pairing cancels, so two overlapping intervals can hide a clear paired effect.
- **Why a permutation test as well as the interval?** The interval says how large the effect plausibly is; the p-value says how surprising it is under "no difference". With few pairs or many ties, as in section 3, the permutation test is exact in spirit where a normal approximation would not be.
- **Tradeoffs accepted:** the bootstrap and the permutation test draw from two fresh streams with the same seed, so their draws are correlated; each is still valid on its own. Pairing is by row index with samples fixed at 1; multi-sample experiments would need the pair groups to be per case, as in `ag.09`.

### 5.4 Explain it back

1. Why is the experiment's unit of resampling the pair and not each arm separately?

<details><summary>Answer</summary>

Resampling each arm independently mixes different cases into the "base" and "experiment" draws, so a resample's difference includes the difficulty gap between the cases drawn, which is exactly the variance pairing removed. Resampling pairs keeps each $d_i$ intact and estimates the uncertainty of $\bar d$ alone.

</details>

2. In the worked example the effect is $1/3$ and the p-value is 1. Explain the p-value.

<details><summary>Answer</summary>

Only one delta is non-zero. Flipping its sign gives $\pm 1/3$; the zeros cannot change. So every sign arrangement has $|\bar d| = 1/3$, equal to the observed value, and all count as at least as extreme: $p = 1$. With one changed case, the data are equally consistent with the experiment being better or worse.

</details>

3. The base arm's judge is down for 10 cases. What happens to those cases in the comparison, and why not score them 0?

<details><summary>Answer</summary>

They are dropped from that metric's deltas and `NPairs` drops by 10. Scoring the base's missing judgments as 0 would add a delta of up to $+1$ per case, inventing an improvement caused by an outage.

</details>

4. The interval is $[-0.02, 0.08]$. A colleague says "so the versions are equivalent". What is wrong?

<details><summary>Answer</summary>

An interval containing 0 means the data cannot rule out "no difference"; it does not show the difference is 0. Effects up to $+0.08$ are equally consistent with the data. Equivalence needs a pre-stated margin and an interval inside it.

</details>

## 6. Pitfalls

| Pitfall | How the solution avoids it |
|---|---|
| Resampling the two arms independently instead of the pairs. | Each delta is one group for `BootstrapCI`, so every draw takes a whole pair. |
| Losing deterministic case ordering before sampling. | Both arms go through `eval.Run`, which fixes case-then-sample order; rows are checked pair by pair. |
| Using different scorer definitions or seeds on the two arms. | One scorer list and one seed are passed to both `eval.Run` calls. |
| Reading a confidence interval that contains zero as proof of equivalence. | The report gives the interval and the p-value; section 5.4 question 4. |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `ag.09` | runs each subject and summarizes scores |
| Back | `ag.10` | supplies deterministic scorers |
| Back | `ag.11` | supplies optional judge scores |
| Back | `load.01` | provides the PCG32 stream behind the bootstrap and the permutation test |
| Back | `dur.11` | provides the durable `EvalSuite` workflow being upgraded |
| Forward | `craft.23` | uses paired evaluations as regression tests |
| Forward | `C2` | optional post-training comparison |

The agent pass's eval run shows a known synthetic effect detected with its interval, and the optional C2 capstone compares the post-trained model against its base the same way.

## Going further

| Solution piece | Production equivalent | What it adds |
|---|---|---|
| paired bootstrap | randomized controlled experiment | uncertainty while controlling case difficulty |
| EvalSuite workflow | release evaluation pipeline | durable, resumable experiment runs |
| delta and interval | sequential monitoring | evidence gathered over repeated releases |
