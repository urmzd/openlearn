# Eval runner

## Overview

| | |
|---|---|
| **Module** | `ag.09` · build · Go · Pass 10 · 4 to 5 h |
| **Solution** | `go/agent/eval/`: [`eval.go`](../../../../course/solutions/go/agent/eval/eval.go) (`Run`, observations, scores, aggregation), [`stats.go`](../../../../course/solutions/go/agent/eval/stats.go) (`Mean`, `PercentileBounds`, `BootstrapCI`), [`report.go`](../../../../course/solutions/go/agent/eval/report.go) (`LoadSuite`, `WriteResults`), [`subject.go`](../../../../course/solutions/go/agent/eval/subject.go) (`ProviderSubject`, `AgentSubject`, `DurableSubject`, `RunID`) |
| **Contract** | suites in [`course/contracts/formats/eval-case.schema.json`](../../../../course/contracts/formats/eval-case.schema.json), reports in [`course/contracts/formats/eval-result.schema.json`](../../../../course/contracts/formats/eval-result.schema.json); the Go API in section 4 |
| **Needs** | `ag.01` provider and deltas, `ag.02` tools (the agent under test calls them), `ag.03` the agent loop, `ag.05` durable agent runs, `load.01` the Go PCG32; reading: `M07.4` (confidence intervals and the bootstrap, re-implemented here in Go), [case study 05](../../case-studies/05-agent-eval-harness/) |
| **Used by** | `ag.10` scorers · `ag.11` LLM judge · `ag.12` A/B experiments · `craft.23` eval tests; the `ctl eval` verb drives it, and an eval run over the agent is the last step of the agent pass |
| **Optional depth** | Efron and Tibshirani, *An Introduction to the Bootstrap*, ch. 6 and 13; Miller, [*Adding Error Bars to Evals*](https://arxiv.org/abs/2411.00640) (free) |

## Key Takeaways

- A scorer that **fails** is a missing measurement: it is **excluded** from the mean and **counted**, never averaged in as 0.
- **Outcome, evidence, and grade** are separate: a subject that crashed has no output to grade, and the report says so instead of scoring an empty string.
- Every number in a report carries a **confidence interval**; with samples, the bootstrap resamples **cases**, not rows.
- The result is the same whatever order **concurrent** work finishes in, and the same **seed** gives the same interval in any language.
- Eval reports are a **file format** (`eval-result.schema.json`) that the release gate and the A/B runner read.

## How to read this chapter

Sections 1 to 3 build the runner's rules from first principles (what a failure is, why order must not depend on timing, how wide an interval three cases give) and work the numbers by hand. Section 4 fixes the interface. Section 5 walks through the solution function by function and explains why each choice is the right one; try to predict each step before reading it. In the finished system, the umbrella CLI's `eval run` verb ([`entry/go/cmd/ctl/eval.go`](../../../../course/solutions/entry/go/cmd/ctl/eval.go)) loads a suite (a file, or `agent-smoke`, the `ag.09` fixture suite), runs every case through `eval.AgentSubject` around the docs agent with the `pass` and `tool_success` scorers, and writes `evals/<suite>/<run_id>/results.jsonl` and `summary.json`.

---

## 1. Why now

The agent answers questions over the course docs, calls `query_usage`, and runs durably. Is it any good? Change the system prompt, and is it better or worse? Today you would ask it three questions, read the answers, and decide by feel. That fails twice: three answers say almost nothing (section 3 shows how little), and "by feel" cannot be rerun after the next change. `dur.11`'s `EvalSuite` already runs the model zoo's suites for the language models of Part 6; this module is the same discipline for agents: a runner that sends every case of a suite through a subject, scores every output with code, and reports means with intervals in a file the release gate can read. The scorers come in `ag.10`, the LLM judge in `ag.11`, and the A/B comparison in `ag.12`; all of them plug into this runner.

## 2. Principles

| Symbol | Meaning | Type |
|---|---|---|
| $n$ | cases in the suite | integer |
| $S$ | samples per case (`WithSamples`) | integer, default 1 |
| $x_{i,s}$ | a score of case $i$, sample $s$, when the scorer succeeded | float64 |
| $\bar{x}$ | the mean of every $x_{i,s}$ that exists | float64 |
| $B$ | bootstrap resamples (`n_boot`) | integer, default 2000 |
| $\alpha$ | one minus the confidence level | float64, default 0.05 |
| $\bar{x}^{*}_{(b)}$ | the $b$-th smallest resample mean, $b = 0, \ldots, B-1$ | float64 |

### 2.1 Suites, cases, observations

A **suite** is a JSON-lines file of **cases** (`eval-case.schema.json`): `case_id`, `input` (a prompt string or `{"messages": [...]}` in the OpenAI shape), optional `ground_truth`, `tags`, and `scorer_args`. Loading it is the first place an eval goes wrong silently, so `LoadSuite` rejects a missing required key, an unknown key (a typo such as `groundtruth` would otherwise drop every answer key), and a repeated `case_id` (two cases with one id overwrite each other in every report and A/B pairing), naming the line. Each case becomes an **observation**; after the subject runs, the observation holds the output, its timing, and annotations its scorers need.

### 2.2 Three facts, kept apart

Case study 05's lesson, applied to every row:

| Fact | Where it lives | A failure means |
|---|---|---|
| **outcome** | `Row.SubjectError` | the subject never produced an output: the provider was down, the durable run is waiting for approval |
| **measurement** | `Score.Error` | the scorer could not score this output: it returned an error, panicked, produced NaN or infinity, or set `Error` (an unparsable judge reply, `ag.11`) |
| **grade** | `Score.Value` | the output was scored |

When the subject fails, every score of that row is an error that names the subject's failure, and no scorer is shown an empty output (which an exact-match scorer would happily grade 0). When a scorer fails, that value is **excluded** from its mean and **counted** in `errored`. Averaging a failure in as 0 blames the model for a broken scorer; a run where the judge's endpoint was down would report the model as useless. The counts stay in the report beside the mean, because excluding failures also hides them: 3 of 500 errored is noise, 300 of 500 means the number is about the other 200.

### 2.3 Concurrency without nondeterminism

Cases are independent, so `Run` works on up to `WithConcurrency(n)` of them at once. Completion order is then random, and anything that depends on it changes between runs: the row order of the report, and with it any statistic that walks rows in order. The fix is structural: each (case, sample) has a **slot** decided before anything runs, `case index * S + sample`, and its worker writes only there. Each worker also gets its **own copy** of the case's annotation map; subjects write into it (`tool_calls`), and two workers writing one shared map is a data race that `go test -race` reports and production silently corrupts.

### 2.4 The mean and its interval

$\bar{x}$ alone hides how much it could move. With $n = 3$ the mean can only be $0, 1/3, 2/3, 1$ for pass/fail scores, and section 3 shows that its 95% interval covers almost everything. The **percentile bootstrap** (`M07.4`) estimates the spread without a formula: pretend the cases you have are the population, draw $n$ of them with replacement, compute the mean of the draw, repeat $B$ times, and sort the $B$ means. The interval is

$$\left[\bar{x}^{*}_{(\lfloor B\alpha/2 \rfloor)},\ \bar{x}^{*}_{(\lceil B(1-\alpha/2) \rceil - 1)}\right],$$

indices 50 and 1949 for $B = 2000$, $\alpha = 0.05$. Off by one at either end is the classic bug: `ceil(...)` without the `- 1` reads one past the bound.

**Resample cases, not rows.** With $S > 1$, the $S$ samples of a case are correlated: the same question is hard for every sample. Resampling the $n S$ rows as if independent treats them as $nS$ cases and gives an interval that is too narrow. This is the **cluster bootstrap**: draw case indices, and each drawn case brings all its samples.

**Determinism.** The draws come from PCG32 (`load.01`) on the `sample` sub-stream, `rng.Stream(seed, rng.PurposeSample)`; each resample draws $n$ indices with `Below(n)`, in order. With the same seed the interval is identical in any implementation that draws the same indices from the same stream, Go or Python, so intervals can be compared exactly across languages.

### 2.5 The report

`eval-result.schema.json` fixes two files under `evals/<suite>/<run_id>/`:

- `results.jsonl`: one row per (case, sample): `suite`, `case_id`, `subject`, `sample`, `input_sha` (SHA-256 of the input's **bytes**, so a changed case is visible), `output`, `scores` (a number, or `null` for a failed score), `errors` (the messages of failed scores), `latency_ms`, `ttft_ms` (`null` when no token arrived; 0 would claim an instant answer), `tokens`, `trace_id` (`null` when not traced).
- `summary.json`: `suite`, `run_id`, `subjects`, `metrics` (subject, then score name, then `mean`, `ci_low`, `ci_high`, `n`, `errored`), `n_boot`, `seed`, and in `ag.12` the `ab` block.

Both files are written to a temporary name and renamed, so a crash never leaves half a report for the release gate to read.

### 2.6 Subjects

A **subject** fills an observation from its input. Three come with the runner:

| Subject | Runs | Records |
|---|---|---|
| `ProviderSubject(p, now, opts...)` | one streamed chat completion through any `ag.01` provider | text; **TTFT** (first text delta), every text delta's arrival (`TokenTimes`), total latency, completion tokens |
| `AgentSubject(a, now)` | one run of an `ag.03` agent | the same timing over the run's streamed text, completion tokens summed over its model calls, and a `tool_calls` annotation: name, arguments, gate verdict, error flag, result |
| `DurableSubject(cfg, prefix, now)` | one `ag.05` durable run per case and sample, id `RunID(prefix, case, sample)` | the answer; a run that stops awaiting approval or indeterminate fails the case |

The durable subject is what makes a 500-case agent eval affordable: if the process dies at case 300, rerunning the suite replays the 300 finished runs from their journals without calling the model. Its run ids must be valid file names (separators replaced, long ids hashed), stable for a case and sample, and **different per sample**, or sample 1 silently replays sample 0. `now` is the clock (nil means `time.Now`); a fake clock makes timings exact.

## 3. Worked example by hand

**Aggregation.** Four cases scored by an exact-match scorer that cannot score case `c`:

| Case | Score |
|---|---|
| a | 1 |
| b | 0 |
| c | error |
| d | 1 |

$\bar{x} = (1 + 0 + 1)/3 = 2/3$ with $n = 3$, errored $= 1$. Averaging the error in as 0 would give $2/4 = 0.5$. **How wide is three?** Take the three values that exist, $\{1, 0, 1\}$, and enumerate the bootstrap exactly instead of sampling: each of the 3 draws picks a 1 with probability $2/3$, so the number of ones $K$ in a resample is binomial:

| $K$ | resample mean | probability |
|---|---|---|
| 0 | 0 | $(1/3)^3 = 1/27 = 0.037$ |
| 1 | $1/3$ | $3 \cdot (2/3)(1/3)^2 = 6/27 = 0.222$ |
| 2 | $2/3$ | $3 \cdot (2/3)^2 (1/3) = 12/27 = 0.444$ |
| 3 | 1 | $(2/3)^3 = 8/27 = 0.296$ |

The 2.5th percentile falls in the first row (0.037 > 0.025), so the lower bound is 0; the 97.5th falls in the last row, so the upper bound is 1. The 95% interval is $[0, 1]$: three cases cannot tell a perfect agent from a useless one. With the 20 binary cases of the fixture (14 ones), the sampled interval is $[0.5, 0.9]$: still wide, now informative.

## 4. The interface

```go
package eval // import "tinyllm/agent/eval"

type Timing struct { Start time.Time; TTFT, Total time.Duration; TokenTimes []time.Duration }
type Observation struct {
	ID string; Sample int
	Input, Output, GroundTruth json.RawMessage
	Annotations map[string]json.RawMessage
	Timing Timing; Tokens int; TraceID string
}
type Score struct { Name string; Value float64; Reason, Error string }
type Scorer interface { Name() string; Score(ctx context.Context, o Observation) (Score, error) }
type Subject func(ctx context.Context, o *Observation) error
type Row struct { Observation; Subject string; Scores []Score; SubjectError string }
func (r Row) Errored(i int) bool
type Stat struct { Mean, CILow, CIHigh float64; N, Errored int }
type SuiteResult struct {
	Suite, Subject string; Rows []Row; Scorers []string; Metrics map[string]Stat
	SubjectErrors, NBoot int; Alpha float64; Seed uint64
}
func Run(ctx context.Context, name string, obs []Observation, s []Scorer, opts ...Option) (*SuiteResult, error)
func WithSubject(name string, s Subject) Option
func WithConcurrency(n int) Option     // default 4
func WithSamples(n int) Option         // default 1
func WithBootstrap(nBoot int, alpha float64) Option // default 2000, 0.05
func WithSeed(seed uint64) Option
func WithClock(now func() time.Time) Option
var ErrSuite error

func Mean(xs []float64) float64
func BootstrapRNG(seed uint64) *rng.PCG32
func PercentileBounds(nBoot int, alpha float64) (lo, hi int)
func BootstrapCI(groups [][]float64, nBoot int, alpha float64, r *rng.PCG32) (lo, hi float64)

func LoadSuite(r io.Reader) ([]Observation, error)
func InputSHA(input json.RawMessage) string
type ResultRow struct { /* one results.jsonl line */ }
type ABStat struct { Delta, CILow, CIHigh, PValue float64; NPairs int }
type AB struct { Base, Exp string; Metrics map[string]ABStat }
type Summary struct { Suite, RunID string; Subjects []string; Metrics map[string]map[string]Stat; AB *AB; NBoot int; Seed uint64 }
func (r *SuiteResult) ResultRows() []ResultRow
func (r *SuiteResult) Summary(runID string) Summary
func WriteResults(dir string, rows []ResultRow, sum Summary) error

var ErrInput error
func Messages(input json.RawMessage) ([]types.Message, error)
type ToolCallRecord struct { Name string; Args json.RawMessage; Verdict string; IsError bool; Result string }
func ProviderSubject(p types.Provider, now func() time.Time, opts ...types.CallOption) Subject
func AgentSubject(a *loop.Agent, now func() time.Time) Subject
func RunID(prefix, caseID string, sample int) string
func DurableSubject(cfg durableagent.Config, prefix string, now func() time.Time) Subject
```

Without `WithSubject`, `Run` scores the observations as given (their outputs already filled), which is how recorded answers are scored (as in `ag.10`). Cancelling the context ends the run with the context's error and no result. The package uses only the standard library, the `ag.01`, `ag.03`, `ag.05` packages, and `tinyllm/ds/rng`.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand example | unit | section 3: mean 2/3, n 3, errored 1; the CI contains the mean | one exact definition of exclusion |
| scorer failures excluded | fault | an error, a panic, NaN, infinity, an `Error` field: each excluded and counted; other scorers unaffected | the judge (`ag.11`) returns unparsable replies as errors |
| deterministic order under concurrency | property | 40 cases, 2 samples, 8 workers finishing out of order: rows in case then sample order, numbers equal to 1 worker; no shared annotation map (no data race) | reports diff cleanly between runs |
| the bootstrap matches a reference | conformance | three fixtures (binary, grouped, 90%) equal an independent Python implementation's bounds | the interval is reproducible across languages |
| percentile bounds | boundary | indices for $B$ = 2000, 1000, 500, 10, 1 | no off-by-one at either end |
| `Run` resamples cases | conformance | 10 cases x 3 samples through `Run` equal a reference cluster bootstrap | samples never inflate confidence |
| a subject error fails the row | fault | a failed subject's row: every score errored with its message, counted in `SubjectErrors` | outcome is not confused with grade |
| bad suites rejected | boundary | repeated or empty case ids, repeated scorer names: `ErrSuite` | no case or metric overwrites another |
| loading a suite | unit | the fixture suite loads; missing, unknown, repeated, and malformed lines are errors naming the line | hand-written suites fail loudly |
| the report schema | conformance | `results.jsonl` and `summary.json` keys, `input_sha`, `null` scores with errors, `null` TTFT and trace, the summary's statistics; no temp file left | `dur.12` and `ag.12` read these reports |
| provider subject timing | unit | with a 10 ms clock: TTFT 10 ms, token times 10 and 20 ms, latency 30 ms, 2 tokens, `"Hello"`; an error delta fails the case | `ag.10`'s TTFT, TTLT, and ITL scorers |
| the agent subject records tool calls | unit | two calls in order with verdicts, results, and error flags; tokens summed over model calls | `ag.10`'s tool-success scorer |
| the durable subject replays | fault | a second run of the suite makes no model calls; samples are separate runs | a crashed eval resumes for free |
| run ids | boundary | separators, `..`, 300-byte ids: valid, short, stable, distinct per sample | run ids are file names in `ag.05`'s store |
| case inputs to messages | unit | string and chat inputs; numbers, empty chats, tool roles: `ErrInput` | no case runs with an empty prompt |
| cancel stops the run | fault | cancelling early or with every case in flight returns `context.Canceled` and no result | a cut-short eval never looks complete |

## 5. The solution, step by step

### 5.1 From first principles

1. **An eval is a measurement, so failures of the instrument must not look like readings.** A scorer that errors, panics, or returns NaN produced no reading. The only honest treatments are "exclude and count"; anything else (0, skip silently) biases the mean or hides the failure. So every scorer call is funnelled through one function that converts every way of failing into `Score.Error`.
2. **The subject's failure is a different fact from a bad score.** If the subject failed, there is no output, so no scorer should see one. The row records `SubjectError` and every score of it is an error naming that failure.
3. **Concurrency must not change the answer.** The result must be a function of the inputs and the seed. So every (case, sample) gets a slot index fixed before any goroutine starts, `i * S + k`, and each worker writes only its slot; aggregation then walks slots in order. Each worker also gets its own copy of the annotation map, so subjects can write into it without a race.
4. **A mean needs an interval, and the interval must respect the data's structure.** Samples of one case are correlated, so the unit of resampling is the case: group the values by case, resample groups (section 2.4).
5. **The interval must be reproducible.** Draws come from one named PCG32 stream, `Below(n)` in a fixed order, and the bounds are read at exact indices. The same seed then gives the same bounds in any implementation.
6. **The report is consumed by machines.** It is written atomically (temp file, rename), with `null` for unknowns so that no consumer reads a failure as a 0.

### 5.2 The code, annotated

**One gate for every scorer call.** `scoreOne` is where "a failure is not a 0" is enforced. The deferred `recover` turns a panicking scorer into an errored score; the name is always the scorer's own, so a scorer cannot report under another metric's name.

```go
func scoreOne(ctx context.Context, s Scorer, o Observation) (out Score) {
	name := s.Name()
	defer func() {
		if p := recover(); p != nil { // one buggy scorer must not end the eval
			out = Score{Name: name, Error: fmt.Sprintf("scorer panicked: %v", p)}
		}
	}()
	sc, err := s.Score(ctx, o)
	sc.Name = name
	switch {
	case err != nil:
		return Score{Name: name, Error: err.Error()}
	case sc.Error != "": // a scorer that knows it could not score (an unparsable judge reply)
		return Score{Name: name, Error: sc.Error, Reason: sc.Reason}
	case math.IsNaN(sc.Value) || math.IsInf(sc.Value, 0): // NaN would poison every mean it touches
		return Score{Name: name, Error: fmt.Sprintf("scorer returned %v", sc.Value)}
	}
	return sc
}
```

**Validate before running anything.** Repeated or empty scorer names, and repeated or empty case ids, are rejected with `ErrSuite` up front: a metric map keyed by name and an A/B pairing keyed by case id would otherwise silently overwrite entries.

**The run loop: slots, copies, and a semaphore.**

```go
res.Rows = make([]Row, len(obs)*cfg.samples) // every slot exists before any work starts
sem := make(chan struct{}, cfg.concurrency)  // at most `concurrency` workers in flight
var wg sync.WaitGroup
for i := range obs {
	for k := 0; k < cfg.samples; k++ {
		slot := i*cfg.samples + k // case order, then sample order, whatever finishes first
		o := obs[i]               // a copy of the struct ...
		o.Sample = k
		o.Annotations = make(map[string]json.RawMessage, len(obs[i].Annotations))
		for key, v := range obs[i].Annotations { // ... and of the map, which a copy would share
			o.Annotations[key] = v
		}
		select {
		case sem <- struct{}{}:
		case <-ctx.Done(): // cancelled while waiting for a worker slot
			wg.Wait()
			return nil, ctx.Err()
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			row := Row{Subject: cfg.subjectName}
			var subjErr error
			if cfg.subject != nil {
				start := cfg.now()
				subjErr = func() (err error) { // a panicking subject fails its case, not the run
					defer func() {
						if p := recover(); p != nil {
							err = fmt.Errorf("subject panicked: %v", p)
						}
					}()
					return cfg.subject(ctx, &o)
				}()
				// ... fill Timing.Start and Timing.Total if the subject did not
			}
			row.Observation = o
			for _, sc := range s {
				if subjErr != nil { // no output: never show a scorer an empty string
					row.Scores = append(row.Scores, Score{Name: sc.Name(), Error: "subject failed: " + subjErr.Error()})
					continue
				}
				row.Scores = append(row.Scores, scoreOne(ctx, sc, o))
			}
			if subjErr != nil {
				row.SubjectError = subjErr.Error()
			}
			res.Rows[slot] = row // the only write to shared state, to a slot nobody else owns
		}()
	}
}
wg.Wait()
if err := ctx.Err(); err != nil {
	return nil, err // a cut-short run returns no result, not a smaller one
}
```

Writing distinct elements of one slice from different goroutines is race-free in Go: each element is its own memory location. After `wg.Wait()` the main goroutine reads them all, and the `WaitGroup` is the happens-before edge.

**Aggregation: exclude, count, group by case.**

```go
func aggregate(rows []Row, j, samples, nBoot int, alpha float64, seed uint64) Stat {
	var st Stat
	var groups [][]float64 // one group per case: its samples that did not fail
	var all []float64
	for i := 0; i < len(rows); i += samples { // rows of case i are rows[i : i+samples]
		var g []float64
		for _, r := range rows[i : i+samples] {
			if r.Errored(j) {
				st.Errored++ // counted, not averaged
				continue
			}
			g = append(g, r.Scores[j].Value)
		}
		if len(g) > 0 {
			groups = append(groups, g)
			all = append(all, g...)
		}
	}
	st.N = len(all)
	if st.N == 0 {
		return st // nothing measured: no mean to report
	}
	st.Mean = Mean(all)
	st.CILow, st.CIHigh = BootstrapCI(groups, nBoot, alpha, BootstrapRNG(seed))
	return st
}
```

On the hand example, `all = [1, 0, 1]`, `Errored = 1`, `Mean = 2/3`. Every metric gets a fresh `BootstrapRNG(seed)`, so adding a scorer does not change another scorer's interval.

**The bootstrap.** The bounds are indices into the sorted resample means; `PercentileBounds` clamps them into range for tiny $B$.

```go
func PercentileBounds(nBoot int, alpha float64) (lo, hi int) {
	lo = int(math.Floor(float64(nBoot) * alpha / 2))       // 2000 * 0.025 = 50
	hi = int(math.Ceil(float64(nBoot)*(1-alpha/2))) - 1    // ceil(1950) - 1 = 1949
	return min(max(lo, 0), nBoot-1), min(max(hi, 0), nBoot-1)
}

func BootstrapCI(groups [][]float64, nBoot int, alpha float64, r *rng.PCG32) (lo, hi float64) {
	// ... m = mean of every value; one group or nBoot < 1: return m, m
	means := make([]float64, nBoot)
	n := uint64(len(groups))
	for b := range means {
		var s float64
		var c int
		for i := 0; i < len(groups); i++ {
			g := groups[r.Below(n)] // draw a case; it brings all its samples
			for _, x := range g {
				s += x
			}
			c += len(g)
		}
		means[b] = s / float64(c) // mean over values, so cases with more surviving samples weigh more
	}
	sort.Float64s(means)
	il, ih := PercentileBounds(nBoot, alpha)
	return means[il], means[ih]
}
```

**Subjects.** `ProviderSubject` streams one completion and times it from the deltas; TTFT is set on the **first** text delta only.

```go
for d := range ch {
	acc.Add(d) // ag.01's Accumulator builds the final message and remembers an error delta
	switch v := d.(type) {
	case types.TextDelta:
		at := now().Sub(start)
		if len(o.Timing.TokenTimes) == 0 {
			o.Timing.TTFT = at // first piece of text, not the last
		}
		o.Timing.TokenTimes = append(o.Timing.TokenTimes, at) // inter-token latency for ag.10
	case types.UsageDelta:
		o.Tokens = v.Out
	}
}
o.Timing.Total = now().Sub(start)
if err := acc.Err(); err != nil {
	return err // an error delta fails the case
}
o.Output = textOutput(acc.Message().Content)
```

`AgentSubject` does the same over the agent's events and records tool calls, matching each `EventToolResult` to its call by `CallID` (results can arrive in a different order than calls), so the `tool_calls` annotation holds name, arguments, gate verdict, error flag, and result for each call in call order. It writes the annotation even when the run fails, so a failed case still shows what it tried.

`DurableSubject` names each run with `RunID`, which is what makes a rerun replay instead of recompute:

```go
func RunID(prefix, caseID string, sample int) string {
	// "/" or ".." in a case id would escape the run store's directory: replace them.
	id := fmt.Sprintf("%s-%s-s%d", prefix, runIDUnsafe.ReplaceAllString(caseID, "_"), sample)
	if len(id) > 128 { // keep file names short; the hash keeps them stable and distinct
		h := sha256.Sum256([]byte(caseID))
		id = fmt.Sprintf("%s-%s-s%d", prefix, hex.EncodeToString(h[:8]), sample)
	}
	return strings.TrimLeft(id, "_.-")
}
```

The sample number is part of the id, so sample 1 is its own run and never replays sample 0's journal.

**Loading and writing.** `LoadSuite` decodes each line into a `map[string]json.RawMessage` first, so it can reject unknown keys and name missing ones before decoding values; `tags` and `scorer_args` become annotations. `InputSHA` hashes the raw input bytes as they appear in the suite. `WriteResults` writes `results.jsonl` and `summary.json` each through `writeAtomic` (`path.tmp`, then `os.Rename`).

Full solution: [`eval.go`](../../../../course/solutions/go/agent/eval/eval.go), [`stats.go`](../../../../course/solutions/go/agent/eval/stats.go), [`report.go`](../../../../course/solutions/go/agent/eval/report.go), [`subject.go`](../../../../course/solutions/go/agent/eval/subject.go).

### 5.3 Why this solution

- **Cost.** The run is dominated by the subject (model calls); the runner adds O(nS) memory for rows. Each metric's bootstrap is O(B · nS) draws and additions plus an O(B log B) sort: for 500 cases, 2000 resamples, about a million additions, milliseconds.
- **Why a slot array and not a channel of results sorted at the end?** Sorting works too, but needs a key on every row and a sort step that is easy to forget for a new field. Fixed slots make ordering a property of the data structure: there is nothing to forget.
- **Why the percentile bootstrap and not a normal (Wald) interval?** For pass/fail scores near 0 or 1, or with few cases, $\bar{x} \pm 1.96\,\mathrm{SE}$ runs outside $[0, 1]$ and has poor coverage. The bootstrap needs no distributional formula and handles clusters naturally. BCa intervals correct skew better (see Going further) at more complexity; the percentile version is simple enough to reproduce exactly across languages.
- **Why exclude failures rather than retry them?** Retries belong in the subject or scorer (a judge client can retry its HTTP call). The runner's job is to report what happened; counting `errored` beside `n` lets the reader decide whether the number is trustworthy.
- **Tradeoff accepted:** the mean pools values, so a case with three surviving samples weighs three times one with one. With uniform $S$ and few failures this is the natural per-row mean; per-case averaging first would be a different estimand.

### 5.4 Explain it back

1. A judge endpoint is down for 300 of 500 cases. What does the report say, and what would it say if failures were averaged in as 0?

<details><summary>Answer</summary>

`n = 200`, `errored = 300`, and the mean of the 200 scored cases with its interval: the reader sees the number is about 40% of the suite. Averaged in as 0, the mean would drop by a factor of about 2.5 and nothing would show why: a broken scorer would be reported as a bad model.

</details>

2. Why does each worker copy the annotation map, when the observation struct is already copied with `o := obs[i]`?

<details><summary>Answer</summary>

Copying a struct copies the map header, not the map: both copies point at the same hash table. Two workers (two samples of one case) writing `tool_calls` into it would race, and one row's annotation could appear in another. A fresh map per worker gives each slot its own.

</details>

3. With 10 cases and 3 samples each, why is resampling the 30 rows wrong?

<details><summary>Answer</summary>

The 3 samples of a case are correlated: a hard question is hard every time. Treating them as 30 independent cases pretends there is three times as much information as there is, and the interval comes out too narrow, by up to about $\sqrt{3}$ when samples agree. Resampling cases keeps each case's samples together, so the interval reflects 10 independent units.

</details>

4. For $B = 2000$, $\alpha = 0.05$, why is the upper index 1949 and not 1950?

<details><summary>Answer</summary>

Indices run from 0 to 1999. The upper bound is the value with 97.5% of the resamples at or below it: $\lceil 2000 \cdot 0.975 \rceil = 1950$ values, so the 1950th smallest, which is index 1949. Index 1950 would be one past it. The lower index is $\lfloor 2000 \cdot 0.025 \rfloor = 50$.

</details>

5. Why does the run id include the sample number, and why replace `/` in case ids?

<details><summary>Answer</summary>

A durable run with an existing journal replays instead of calling the model. Without the sample in the id, sample 1 would find sample 0's journal and return the same answer, so $S$ samples would measure nothing new. The id is also a file name in `ag.05`'s store; a `/` or `..` would create subdirectories or escape the store.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. averaging a failed score in as 0 | a dead judge endpoint reports the model as useless | `aggregate` skips errored scores and counts them in `Errored` |
| 2. trusting scorers: no `recover`, NaN accepted, the `Error` field ignored | one bad case ends the eval, or NaN poisons the mean | `scoreOne` recovers panics and turns errors, `Error`, NaN, and infinity into `Score.Error` |
| 3. appending rows as workers finish | two runs of one suite produce different reports | each worker writes `res.Rows[i*S + k]`, fixed before it starts |
| 4. sharing the case's annotation map between workers | a data race; annotations of one row appear in another | a fresh map copied per (case, sample) |
| 5. resampling rows when there are samples | intervals too narrow by about $\sqrt{S}$ for correlated samples | `aggregate` groups by case; `BootstrapCI` draws groups |
| 6. a percentile index off by one, or another random stream | intervals that match no other implementation | `PercentileBounds` uses `floor` and `ceil(...) - 1`; draws come from `rng.Stream(seed, PurposeSample)` |
| 7. scoring a failed subject's empty output | crashes counted as wrong answers | a subject error fills every score with `"subject failed: ..."` and sets `SubjectError` |
| 8. a lax suite loader | a typo in a key silently drops every answer key; two cases share an id | `LoadSuite` rejects unknown and missing keys and repeated ids, naming the line |
| 9. 0 for unknown values in the report, or hashing the decoded input | a failed score reads as a wrong answer; `input_sha` hides changed cases | `ResultRows` writes `null` for failed scores, missing TTFT, and missing traces; `InputSHA` hashes the raw bytes |
| 10. TTFT taken at the last token | latency scorers report the whole answer time as time to first token | TTFT is set only when `TokenTimes` is still empty |
| 11. not matching tool results to their calls | the tool-success scorer sees no failures | `AgentSubject` maps `CallID` to the call's index and fills its result |
| 12. one durable run per case instead of per sample, or ids with separators | sample 1 replays sample 0; a case id with `/` cannot be stored | `RunID` includes `-s<n>` and replaces unsafe characters |
| 13. accepting two scorers with one name | one metric silently replaces the other | `Run` rejects empty or repeated scorer names with `ErrSuite` |
| 14. returning partial results after a cancel | an interrupted eval looks like a smaller, complete one | `Run` returns `nil, ctx.Err()` after `wg.Wait()` |
| 15. accepting any input shape | a case with a typo runs with an empty prompt | `Messages` accepts only a string or a non-empty chat with system, user, or assistant roles |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `ag.01` | `Provider`, the deltas, and `Accumulator` inside `ProviderSubject` |
| Back | `ag.02` | the tool registry of the agent under test |
| Back | `ag.03` | `loop.Agent` and its events inside `AgentSubject` |
| Back | `ag.05` | `durableagent.Run` inside `DurableSubject` |
| Back | `load.01` | `rng.Stream` and `Below` behind every bootstrap |
| Forward | `ag.10` | scorers implement `Scorer` and read `Timing`, `GroundTruth`, and the `tool_calls` annotation |
| Forward | `ag.11` | the judge is a `Scorer`; its unparsable replies are `Score.Error`, excluded and counted |
| Forward | `ag.12` | `RunExperiment` runs two subjects through this runner and fills `Summary.AB` |
| Forward | `craft.23` | eval results become regression-test evidence |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Run` | saige `eval/harness`, OpenAI Evals, Inspect AI | task registries, sandboxes, model-graded and human-graded mixes, eval logs viewers | [saige](https://github.com/urmzd/saige) (free), [Inspect](https://inspect.aisi.org.uk/) (free) |
| `BootstrapCI` | `scipy.stats.bootstrap` | BCa intervals that correct bias and skew | [SciPy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) (free) |
| `DurableSubject` | Temporal-backed eval pipelines | per-case retries, rate limits, and resumable runs across machines | `dur.*`, Temporal docs (free) |
