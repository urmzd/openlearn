# Paired A/B experiments

You changed the agent's system prompt. Is the new version better? The obvious approach is to run both versions on some cases and compare their means, but that comparison is usually dominated by which cases each version happened to see: an easy sample makes any version look good. The fix is to have both versions answer exactly the same cases and to analyse the difference case by case. This section builds an experiment runner that pairs the two arms by case id, computes each case's score difference, and estimates the uncertainty of the mean difference with a paired bootstrap interval and a sign permutation p-value.

The runner is built entirely from pieces you already have: the [eval runner](01-eval-runner.md) runs each arm, the [scorers](02-scorers.md) and the [LLM judge](03-llm-judge.md) supply the measurements, the Go PCG32 from [An open-loop load generator](../../build-an-llm/11-serving/11-load-generator.md) in *Build Your Own LLM System* drives the resampling, and the durable `EvalSuite` workflow from [Platform workflows: TrainRun and EvalSuite](../05-durable-orchestration-and-workers/11-trainrun-and-evalsuite.md) is upgraded to carry the experiment. The statistics come from [LLN, CLT, confidence intervals, bootstrap](../../mathematics-for-computing/08-probability-statistics/05-lln-clt-confidence-intervals-bootstrap.md) and [Hypothesis tests: paired and two-sample permutation, McNemar, Holm](../../mathematics-for-computing/08-probability-statistics/06-hypothesis-tests.md) in *Mathematics for Computing*.

The solution is [`go/agent/eval/experiment/experiment.go`](../../../../course/solutions/go/agent/eval/experiment/experiment.go), with `RunExperiment`, `Stat`, `Result`, `WithBootstrap`, `WithSeed`, and `WithConcurrency`, plus [`go/workflows/eval_suite.go`](../../../../course/solutions/go/workflows/eval_suite.go), where `EvalSuite` gains the `Judge` and `AB` spec options. In the finished system, `ctl eval experiment --base <model> --exp <model> --suite <file|agent-smoke>` ([`entry/go/cmd/ctl/eval.go`](../../../../course/solutions/entry/go/cmd/ctl/eval.go)) runs the docs agent on both models over the same cases and prints the paired delta, its interval, and its p-value.

> **Learning objectives**
> - Explain why pairing by case cancels case difficulty and why the pair is the unit of resampling.
> - Compute a paired delta, its bootstrap interval, and its sign permutation p-value by hand.
> - Exclude and count incomplete pairs instead of scoring the missing side as 0.
> - Interpret an interval that contains 0 correctly, without reading it as proof of equivalence.

## Pairing, the interval, and the p-value

For case $i$, define $d_i = \text{score}^{\text{exp}}_i - \text{score}^{\text{base}}_i$. The reported effect is $\bar d$, the mean of the deltas. A paired bootstrap samples case indices with replacement and keeps each base/experiment pair together, and its percentile interval estimates the uncertainty of the effect. The same seed and inputs must reproduce the same result.

Cases missing a score on either side are excluded from the paired metric and counted; they must never be converted to zero. Duplicate ids and mismatched scorer names must be rejected before any comparison is computed.

The p-value answers a different question from the interval: if the two versions were equally good, how often would a mean delta at least this large appear by chance? Under that null hypothesis each $d_i$ is as likely to be $+d_i$ as $-d_i$, because which arm got the higher score is a coin flip. Flipping signs at random therefore generates the null distribution. This is the **sign permutation test**.

## Working it by hand

Let the base scores be `[0, 1, 0]` and the experiment scores be `[1, 1, 0]`. The paired deltas are `[1, 0, 0]`, so the effect is $1/3$. An unpaired comparison would throw away the fact that the second and third cases were shared, widening or shifting the result unnecessarily.

For the interval, notice that a resample of three deltas contains $K$ copies of the 1, where $K$ is binomial with $p = 1/3$. $P(K = 0) = 8/27 = 0.296$, so the 2.5th percentile is 0; $P(K = 3) = 1/27 = 0.037 > 0.025$, so the 97.5th percentile is 1. The 95% interval is $[0, 1]$: three cases cannot rule out "no effect".

For the p-value, flipping signs of `[1, 0, 0]` can only change the 1 into $-1$; the zeros stay zero. Every permutation has $|\bar d| = 1/3 \ge 1/3$, so every one is "at least as extreme", and $p = (B + 1)/(B + 1) = 1$. One improved case out of three is no evidence at all.

## Designing the API

The package exposes one function and its options:

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

`RunExperiment` returns each metric's mean delta, confidence bounds, p-value, and pair count. The durable `EvalSuite` workflow is upgraded so the same run can produce ordinary eval results and an optional A/B block: its input carries `scorers`, `judge`, `ab` (`base`, `exp`, `n_boot`, `alpha`), and `thresholds`, and every per-suite eval spec it hands the eval activity carries them too.

## Building it step by step

Six principles lead to the implementation.

1. **Case difficulty is the biggest source of variance, so cancel it.** Scores differ far more between an easy and a hard case than between two versions on one case. Subtracting within a case removes the case's difficulty: $d_i$ measures only the version. So both arms must run on exactly the same cases, and the unit of analysis is the pair.
2. **Reuse the runner for each arm.** The eval runner already guarantees case-then-sample row order, exclusion of failed scores, and determinism. Running it twice with the same observations and seed gives two row arrays that line up index by index.
3. **Pair, then check the pairing.** Lining up by index is only safe if each index holds the same case and sample in both arms, so check `ID` and `Sample` per row and refuse to continue otherwise.
4. **A pair with a missing side is missing.** If either score errored, the delta does not exist; it is dropped, and the pair count `NPairs` shows how many remain.
5. **Resample pairs, as single values.** Each delta becomes a one-element group for the runner's `BootstrapCI`, so a resample draws whole pairs, and the arms are never resampled independently.
6. **Test against the null directly.** Sign flips of the observed deltas give the distribution of $|\bar d|$ when the versions are equal. The p-value is the fraction at least as extreme as observed, with a $+1$ correction so it is never exactly 0.

### Configuration and the two arms

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

### Paired deltas per metric

```go
for si, name := range br.Scorers {
	var diffs []float64
	for i := range br.Rows {
		b, e := br.Rows[i], er.Rows[i]
		if b.ID != e.ID || b.Sample != e.Sample { // the runner's ordering makes this impossible; check anyway
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

### The interval

Wrapping each delta as its own group reuses the runner's cluster bootstrap unchanged. The draws come from the same PCG32 `sample` stream, so a seed reproduces the interval.

```go
groups := make([][]float64, len(diffs))
for i, d := range diffs {
	groups[i] = []float64{d} // one pair = one resampling unit
}
lo, hi := eval.BootstrapCI(groups, c.nBoot, c.alpha, rng.Stream(c.seed, rng.PurposeSample))
```

### The p-value

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

### The durable workflow

`EvalSuite` runs one eval activity per suite, in order, so a crash in a later suite never reruns a finished one. The upgrade threads the new options through to every per-suite spec:

```go
func applyAdvancedSpecOptions(spec *evalSpec, in EvalSuiteInput) {
	// One place, used by both EvalSuite and MarshalSpec, so the workflow and the
	// CLI's dry run can never disagree on what the activity receives.
	spec.Scorers, spec.Judge, spec.AB, spec.Thresholds = in.Scorers, in.Judge, in.AB, in.Thresholds
}
```

The full solution is in [`experiment.go`](../../../../course/solutions/go/agent/eval/experiment/experiment.go) and [`eval_suite.go`](../../../../course/solutions/go/workflows/eval_suite.go).

### Why this design

The cost is two full eval runs, where the model calls dominate, then $O(B \cdot n)$ per metric for the bootstrap and the same again for the permutation test.

Comparing the two arms' separate intervals is the usual mistake. Each arm's interval includes case-difficulty variance that pairing cancels, so two overlapping per-arm intervals can hide a clear paired effect.

The permutation test complements the interval rather than repeating it. The interval says how large the effect plausibly is; the p-value says how surprising it is under "no difference". With few pairs or many ties, as in the hand example, the permutation test is exact in spirit where a normal approximation would not be.

Two tradeoffs are accepted. The bootstrap and the permutation test draw from two fresh streams with the same seed, so their draws are correlated, though each is still valid on its own. And pairing is by row index with samples fixed at 1; multi-sample experiments would need the pair groups to be per case, as in the eval runner.

## What can go wrong

A correct implementation passes six checks: the paired-delta hand example; exclusion of incomplete pairs; deterministic resampling of paired rows; detection of a synthetic positive effect; correct interval coverage under a null effect; and an eval-suite spec that forwards the experiment configuration to the activity.

> **Warning:** Resampling the two arms independently instead of the pairs reintroduces exactly the case-difficulty variance that pairing removed. Each delta is one group for `BootstrapCI`, so every draw takes a whole pair.

The other pitfalls are about keeping the two arms identical except for the subject. Losing deterministic case ordering before sampling would pair the wrong rows; both arms go through `eval.Run`, which fixes case-then-sample order, and rows are checked pair by pair. Using different scorer definitions or seeds on the two arms would measure the scorers, not the versions; one scorer list and one seed are passed to both `eval.Run` calls.

> **Warning:** A confidence interval that contains zero is not proof of equivalence. The report gives both the interval and the p-value; review question 4 below works through why.

## Where this leads

[Agent evals as tests](../../software-craftsmanship/03-testing-mentality/09-agent-evals-as-tests.md) in *Software Craftsmanship* uses paired evaluations as regression tests. The eval run over the finished agent shows a known synthetic effect detected with its interval, and the optional [post-trained TinyStories capstone](../../build-an-llm/capstones/02-post-trained.md) in *Build Your Own LLM System* compares the post-trained model against its base the same way.

In production, the paired bootstrap is a small instance of a randomized controlled experiment, which controls case difficulty while measuring uncertainty. The `EvalSuite` workflow grows into a release evaluation pipeline with durable, resumable experiment runs. And a single delta with its interval grows into sequential monitoring, where evidence is gathered over repeated releases.

## Key terms

- **paired delta**: the per-case difference $d_i$ between the experiment's and the base's score on the same case.
- **paired bootstrap**: a bootstrap that resamples whole pairs, so the two arms are never drawn independently.
- **sign permutation test**: a test that builds the null distribution of $|\bar d|$ by randomly flipping the sign of each delta.
- **p-value**: the fraction of null arrangements at least as extreme as the observed one, with a $+1$ correction.
- **NPairs**: the number of pairs where both sides were scored, on which a metric's delta rests.

## Summary

- Pairing by case cancels case difficulty, the largest source of variance, so the unit of analysis and resampling is the pair.
- Both arms run through the same eval runner with the same observations, scorers, and seed; rows are checked to pair correctly.
- A pair with an errored side is dropped and counted, never scored as 0.
- The interval says how large the effect plausibly is; the sign permutation p-value says how surprising it is under no difference.
- Three cases, as in the hand example, give an interval of $[0, 1]$ and $p = 1$: no evidence either way.

## Review questions

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
