<!-- ol:module ag.09 -->
# Eval runner

## Overview

| | |
|---|---|
| **Module** | `ag.09` · build · Go · Pass 10 · 4 to 5 h |
| **You build** | `go/agent/eval/`: `eval.go` (`Run`, observations, scores, aggregation), `stats.go` (mean, percentile bootstrap), `report.go` (load suites, write `results.jsonl` and `summary.json`), `subject.go` (provider, agent, and durable-run subjects) |
| **Contract** | suites in [`course/contracts/formats/eval-case.schema.json`](../../course/contracts/formats/eval-case.schema.json), reports in [`course/contracts/formats/eval-result.schema.json`](../../course/contracts/formats/eval-result.schema.json); the Go API in section 4 |
| **Tests** | `course/tests/go/ag_09/` (what they check: section 4) |
| **Needs** | `ag.01` provider and deltas, `ag.02` tools (the tests give an agent a tool), `ag.03` the agent loop, `ag.05` durable agent runs, `load.01` the Go PCG32; reading: `M07.4` (confidence intervals and the bootstrap, re-implemented here in Go), [case study 05](../../case-studies/05-agent-eval-harness/) |
| **Used by** | `ag.10` scorers · `ag.11` LLM judge · `ag.12` A/B experiments · `craft.23` eval tests; your `{ctl} eval` verb drives it |
| **Milestone** | MS-agent |
| **Optional depth** | Efron and Tibshirani, *An Introduction to the Bootstrap*, ch. 6 and 13; Miller, [*Adding Error Bars to Evals*](https://arxiv.org/abs/2411.00640) (free) |

## Key Takeaways

- A scorer that **fails** is a missing measurement: it is **excluded** from the mean and **counted**, never averaged in as 0.
- **Outcome, evidence, and grade** are separate: a subject that crashed has no output to grade, and the report says so instead of scoring an empty string.
- Every number in a report carries a **confidence interval**; with samples, the bootstrap resamples **cases**, not rows.
- The result is the same whatever order **concurrent** work finishes in, and the same **seed** gives the same interval in any language.
- Eval reports are a **file format** (`eval-result.schema.json`) that the release gate and the A/B runner read.

## How to work this chapter

```bash
ol start ag.09          # stubs go/agent/eval/*.go into your repo
ol tests ag.09          # read the test catalog first
ol check ag.09          # exit code is the verdict
ol check ag.09 --ref-deps   # only if you skipped ag.01 to ag.05 or load.01
ol diff  ag.09          # after passing: your code against the reference
```

Then add `{ctl} eval run --suite <file> --base-url <url> --model <name>` to your umbrella CLI (learner territory): load the suite, build a `ProviderSubject` (or an `AgentSubject` around your agent), run it with the scorers of `ag.10`, and write `evals/<suite>/<run_id>/`. MS-agent runs it over `{fixture:ag.09/suite.jsonl}` against your gateway.

---

## 1. Why now

Your agent answers questions over the course docs, calls `query_usage`, and runs durably. Is it any good? Change the system prompt, and is it better or worse? Today you would ask it three questions, read the answers, and decide by feel. That fails twice: three answers say almost nothing (section 3 shows how little), and "by feel" cannot be rerun after the next change. `dur.11`'s `EvalSuite` already runs the model zoo's suites for the language models of Part 6; this module is the same discipline for agents: a runner that sends every case of a suite through a subject, scores every output with code, and reports means with intervals in a file the release gate can read. The scorers come in `ag.10`, the LLM judge in `ag.11`, and the A/B comparison in `ag.12`; all of them plug into what you build here.

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

**Determinism.** The draws come from PCG32 (`load.01`) on the `sample` sub-stream, `rng.Stream(seed, rng.PurposeSample)`; each resample draws $n$ indices with `Below(n)`, in order. With the same seed the interval is identical in Go and in the course's Python oracle, which is how the tests can compare intervals exactly.

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
| `AgentSubject(a, now)` | one run of your `ag.03` agent | the same timing over the run's streamed text, completion tokens summed over its model calls, and a `tool_calls` annotation: name, arguments, gate verdict, error flag, result |
| `DurableSubject(cfg, prefix, now)` | one `ag.05` durable run per case and sample, id `RunID(prefix, case, sample)` | the answer; a run that stops awaiting approval or indeterminate fails the case |

The durable subject is what makes a 500-case agent eval affordable: if the process dies at case 300, rerunning the suite replays the 300 finished runs from their journals without calling the model. Its run ids must be valid file names (separators replaced, long ids hashed), stable for a case and sample, and **different per sample**, or sample 1 silently replays sample 0. `now` is the clock (nil means `time.Now`); tests pass a fake one so timings are exact.

## 3. Worked example by hand

**Aggregation.** Four cases scored by an exact-match scorer that cannot score case `c`:

| Case | Score |
|---|---|
| a | 1 |
| b | 0 |
| c | error |
| d | 1 |

$\bar{x} = (1 + 0 + 1)/3 = 2/3$ with $n = 3$, errored $= 1$. Averaging the error in as 0 would give $2/4 = 0.5$. This is `TestHandExample`.

**How wide is three?** Take the three values that exist, $\{1, 0, 1\}$, and enumerate the bootstrap exactly instead of sampling: each of the 3 draws picks a 1 with probability $2/3$, so the number of ones $K$ in a resample is binomial:

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

Without `WithSubject`, `Run` scores the observations as given (their outputs already filled), which is how `ag.10`'s tests score recorded answers. Cancelling the context ends the run with the context's error and no result. Use only the standard library, your `ag.01`, `ag.03`, `ag.05` packages, and `tinyllm/ds/rng`.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestHandExample` | unit | section 3: mean 2/3, n 3, errored 1; the CI contains the mean | you and the tests agree on exclusion |
| `TestScorerFailuresExcluded` | fault | an error, a panic, NaN, infinity, an `Error` field: each excluded and counted; other scorers unaffected | the judge (`ag.11`) returns unparsable replies as errors |
| `TestDeterministicOrderUnderConcurrency` | property | 40 cases, 2 samples, 8 workers finishing out of order: rows in case then sample order, numbers equal to 1 worker; no shared annotation map (`-race`) | reports diff cleanly between runs |
| `TestBootstrapMatchesOracle` | conformance | three fixtures (binary, grouped, 90%) equal the Python oracle's bounds | the interval is reproducible across languages |
| `TestPercentileBounds` | boundary | indices for $B$ = 2000, 1000, 500, 10, 1 | no off-by-one at either end |
| `TestRunResamplesCases` | conformance | 10 cases x 3 samples through `Run` equal the oracle's cluster bootstrap | samples never inflate confidence |
| `TestSubjectErrorFailsRow` | fault | a failed subject's row: every score errored with its message, counted in `SubjectErrors` | outcome is not confused with grade |
| `TestRejectsBadSuites` | boundary | repeated or empty case ids, repeated scorer names: `ErrSuite` | no case or metric overwrites another |
| `TestLoadSuite` | unit | the fixture suite loads; missing, unknown, repeated, and malformed lines are errors naming the line | hand-written suites fail loudly |
| `TestReportSchema` | conformance | `results.jsonl` and `summary.json` keys, `input_sha`, `null` scores with errors, `null` TTFT and trace, the summary's statistics; no temp file left | `dur.12` and `ag.12` read your reports |
| `TestProviderSubjectTiming` | unit | with a 10 ms clock: TTFT 10 ms, token times 10 and 20 ms, latency 30 ms, 2 tokens, `"Hello"`; an error delta fails the case | `ag.10`'s TTFT, TTLT, and ITL scorers |
| `TestAgentSubjectRecordsToolCalls` | unit | two calls in order with verdicts, results, and error flags; tokens summed over model calls | `ag.10`'s tool-success scorer |
| `TestDurableSubjectReplays` | fault | a second run of the suite makes no model calls; samples are separate runs | a crashed eval resumes for free |
| `TestRunID` | boundary | separators, `..`, 300-byte ids: valid, short, stable, distinct per sample | run ids are file names in `ag.05`'s store |
| `TestMessages` | unit | string and chat inputs; numbers, empty chats, tool roles: `ErrInput` | no case runs with an empty prompt |
| `TestCancelStopsRun` | fault | cancelling early or with every case in flight returns `context.Canceled` and no result | a cut-short eval never looks complete |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. averaging a failed score in as 0 | a dead judge endpoint reports the model as useless | `TestHandExample`, `TestScorerFailuresExcluded` (mutant `s01`) |
| 2. trusting scorers: no `recover`, NaN accepted, the `Error` field ignored | one bad case ends the eval, or NaN poisons the mean | `TestScorerFailuresExcluded` (mutants `s02`, `s03`, `s04`) |
| 3. appending rows as workers finish | two runs of one suite produce different reports | `TestDeterministicOrderUnderConcurrency` (mutant `s05`) |
| 4. sharing the case's annotation map between workers | a data race; annotations of one row appear in another | `TestDeterministicOrderUnderConcurrency` (mutant `s06`) |
| 5. resampling rows when there are samples | intervals too narrow by about $\sqrt{S}$ for correlated samples | `TestRunResamplesCases` (mutant `s07`) |
| 6. a percentile index off by one, or another random stream | intervals that match no other implementation | `TestBootstrapMatchesOracle`, `TestPercentileBounds` (mutants `s08`, `s09`) |
| 7. scoring a failed subject's empty output | crashes counted as wrong answers | `TestSubjectErrorFailsRow` (mutant `s10`) |
| 8. a lax suite loader | a typo in a key silently drops every answer key; two cases share an id | `TestLoadSuite` (mutants `s11`, `s12`) |
| 9. 0 for unknown values in the report, or hashing the decoded input | a failed score reads as a wrong answer; `input_sha` hides changed cases | `TestReportSchema` (mutants `s13`, `s14`, `s15`) |
| 10. TTFT taken at the last token | latency scorers report the whole answer time as time to first token | `TestProviderSubjectTiming` (mutant `s16`) |
| 11. not matching tool results to their calls | the tool-success scorer sees no failures | `TestAgentSubjectRecordsToolCalls` (mutant `s17`) |
| 12. one durable run per case instead of per sample, or ids with separators | sample 1 replays sample 0; a case id with `/` cannot be stored | `TestDurableSubjectReplays`, `TestRunID` (mutants `s18`, `s21`) |
| 13. accepting two scorers with one name | one metric silently replaces the other | `TestRejectsBadSuites` (mutant `s19`) |
| 14. returning partial results after a cancel | an interrupted eval looks like a smaller, complete one | `TestCancelStopsRun` (mutant `s20`) |
| 15. accepting any input shape | a case with a typo runs with an empty prompt | `TestMessages` (mutant `s22`) |

## 6. Where it's used next

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

If you skip this module, `ol check ag.10` reports `needs ag.09: build it, or pass --ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Run` | saige `eval/harness`, OpenAI Evals, Inspect AI | task registries, sandboxes, model-graded and human-graded mixes, eval logs viewers | [saige](https://github.com/urmzd/saige) (free), [Inspect](https://inspect.aisi.org.uk/) (free) |
| `BootstrapCI` | `scipy.stats.bootstrap` | BCa intervals that correct bias and skew | [SciPy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) (free) |
| `DurableSubject` | Temporal-backed eval pipelines | per-case retries, rate limits, and resumable runs across machines | `dur.*`, Temporal docs (free) |
