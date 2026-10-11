<!-- ol:module load.01 -->
# Open-loop load generator, log-linear histogram, Go PCG32 port

## Overview

| | |
|---|---|
| **Module** | `load.01` · build · Go · Pass 7 · 6 to 9 h |
| **You build** | `go/ds/rng/pcg32.go`: the course's random generator in Go (draws, Box-Muller, unbiased bounded ints, Fisher-Yates, sub-streams) · `go/loadgen/schedule.go`: Poisson, constant, and burst arrivals · `go/loadgen/histogram.go`: a log-linear histogram with at most 1/128 relative error · `go/loadgen/run.go`: the SSE reader, per-request measurement, the open-loop runner, and the report |
| **Contract** | the report: [`formats/loadgen-report.schema.json`](../../../course/contracts/formats/loadgen-report.schema.json) · the generator: [`spec/pcg32.md`](../../../course/contracts/spec/pcg32.md) (conformance `parity/rng`) · the stream you measure: [`openapi/openai-subset.v1.yaml`](../../../course/contracts/openapi/openai-subset.v1.yaml) · the role `{loadgen}`: [`spec/cli-roles.md`](../../../course/contracts/spec/cli-roles.md) |
| **Tests** | `course/tests/go/load_01/`, 21 tests (what they check: section 4) |
| **Needs** | nothing to call: reading `lang.06` Go ([primer](../../../software-craftsmanship/12-language-and-tool-primers/06-go.md)), `M07.1` inverse-CDF sampling ([chapter](../../../math/07-probability-statistics/01-categorical-sampling.md)), `M06.3` PCG32 ([chapter](../../../math/06-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md)); both are re-implemented here in Go |
| **Used by** | your `{loadgen}` main drives `MS-L10`, `MS-gateway`, `MS-prod`, and the drills · later: `ag.07`, `ag.09`, `ag.12` |
| **Milestone** | `MS-L10` |
| **Optional depth** | Gil Tene, [How NOT to Measure Latency](https://www.youtube.com/watch?v=lJ8ydIuPFeU) (free talk); [HdrHistogram](http://hdrhistogram.org/) (free); [O'Neill, PCG](https://www.pcg-random.org/paper.html) (free); [Schroeder, Wierman, Harchol-Balter, Open Versus Closed](https://www.usenix.org/legacy/event/nsdi06/tech/full_papers/schroeder/schroeder.pdf) (free) |

## Key Takeaways

- Open loop: arrivals follow the schedule whatever the server does, and each request is charged from its INTENDED send time, so a stall shows up in every request it delayed (`TestNoCoordinatedOmission`).
- A log-linear histogram keeps 128 buckets per power of two: every quantile is within 1/128 of the exact sorted value, never below it, in fixed memory, and two histograms merge exactly (`TestHistogramHandExample`, `TestQuantilesWithinOnePercentOfExactSort`, `TestMergeEqualsRecordingEverything`).
- The nearest-rank quantile is the $\lceil qn \rceil$-th smallest value, with an allowance for float rounding: $0.55 \cdot 100$ is $55.00000000000001$ (`TestQuantileRankRounding`).
- The Go PCG32 gives the same stream as Python, C, and Rust for the same seed, bit for bit (`TestPCG32MatchesGoldenVectors`).
- A failed request (an error status, an SSE error event, a stream cut before `[DONE]`) counts in `errors`, never in a latency histogram; goodput is the share of ALL requests that met every SLO bound (`TestRunCountsErrorsAndRecordsOnlySuccesses`, `TestReportGoodputAndShape`).

## How to work this chapter

```bash
ol start load.01             # stubs go/ds/rng/pcg32.go and go/loadgen/{schedule,histogram,run}.go
ol tests load.01
ol check load.01             # exit code is the verdict
ol diff  load.01
ol parity rng                # your Go generator against the golden vectors and the other three ports
```

Your `go/cmd/loadgen` main (yours, D16) parses `--target --model --prompt --max-tokens --rate --duration --mode poisson|constant|burst --burst --seed --out`, builds the schedule (`Poisson(rate, rng.Stream(seed, rng.PurposeSample))` for poisson), calls `loadgen.Run`, writes the report to `--out`, and prints it as the final stdout line.

---

## 1. Why now

Your engine (`L10.5`) and its metrics (`L10.7`) are ready; how much load they take before the SLOs break is not known. The Pass 4 Python load script in `ml/04` waits for each response before sending the next (closed loop): when the server stalls, it stops sending and records one slow request instead of the dozens a real user population would have sent, so its percentiles look fine during exactly the outages that matter (coordinated omission). Every load number the course grades from here on (the 64-request burst of `MS-L10`, the SLO run of `MS-prod`, the drills, `load.02`'s regression gate in CI) comes from this generator, so it has to be right: open loop, latencies from the intended start, honest histograms, seeded arrivals.

## 2. Principles

| Symbol | Meaning | Type |
|---|---|---|
| $\lambda$ | offered rate (requests per second) | `float64` |
| $g_i$ | gap before arrival $i$ | `time.Duration` |
| $s_i = s_0 + \sum_{j \le i} g_j$ | the INTENDED send time of request $i$ | `time.Time` |
| $v$ | a recorded value in nanoseconds, $v \ge 0$ | `int64` |
| $e = \lfloor \log_2 v \rfloor$ | the value's octave | int |
| $n$ | recorded values; $q \in [0, 1]$ a quantile | |

### 2.1 Arrivals

A Poisson process of rate $\lambda$ has independent exponential gaps; by inverse CDF (`M07.1`) one uniform $u$ gives $g = -\ln(1 - u)/\lambda$ ($1 - u$ is never 0). Constant arrivals use $g = 1/\lambda$ every time; a burst of size $b$ every $\tau$ uses gaps $0, 0, \ldots, \tau, 0, \ldots$. The runner sends request $i$ at $s_i$ in its own goroutine, whether or not earlier requests have finished, and stops scheduling at $s_0 + $ duration (an arrival exactly at the end is not sent).

### 2.2 Coordinated omission

Charge a request from $s_i$, not from when the client got around to sending it or from when the server answered the headers. If the server stalls for one second at 10 rps, ten requests each see the stall, and all ten latencies record it.

### 2.3 What one request measures

From the SSE stream (`data: <json>\n\n` events, `[DONE]` last), stamped as each event is read: TTFT is the first content chunk minus $s_i$; ITL the gaps between consecutive content chunks; E2E is `[DONE]` minus $s_i$; tokens are `usage.completion_tokens` when the server sends usage, else the content chunks; TPOT $= (E2E - TTFT)/(\text{tokens} - 1)$ for at least 2 tokens.

### 2.4 The log-linear histogram

Values below $128$ get a bucket each. Above, with $e = \lfloor \log_2 v \rfloor$ and shift $= e - 7$, the top 8 bits $v \gg \text{shift}$ lie in $[128, 256)$ and select bucket $(\text{shift} + 1) \cdot 128 + (v \gg \text{shift}) - 128$, which covers $[\text{sub} \cdot 2^{\text{shift}}, (\text{sub} + 1) 2^{\text{shift}} - 1]$. A bucket is at most $2^{\text{shift}}/(128 \cdot 2^{\text{shift}}) = 1/128$ of its lower bound wide. The q-quantile is the nearest-rank value: rank $r = \lceil qn - 10^{-9} \rceil$ clamped to $[1, n]$, reported as the upper bound of the bucket holding the $r$-th smallest value, clamped to the true maximum. Memory is fixed (7424 counters for all of `int64`), and merging is adding counters.

### 2.5 The generator

The Go port follows `spec/pcg32.md` exactly: state update $s \leftarrow 6364136223846793005\,s + \text{inc}$, output the XSH-RR permutation of the old state; seeding $\text{inc} = 2\,\text{seq} + 1$; `uniform_f64` from two draws (53 bits); Box-Muller with the spare kept; `Below(n)` rejecting draws under $(2^{32} - n) \bmod n$; Fisher-Yates from the end; sub-streams by SplitMix64.

### 2.6 The report

`formats/loadgen-report.schema.json`: run id, target, mode, offered rate, duration, requests, errors, p50/p90/p95/p99/mean per metric in milliseconds, error rate, goodput (requests within every SLO bound over all requests), output tokens per second, and each histogram's non-empty buckets as `[upper_ms, count]`.

## 3. Worked example by hand

**A histogram** (test `TestHistogramHandExample`). Record 1000, 1002, 1005, 2000 ns.

| Value | $e$ | shift | top bits | bucket |
|---|---|---|---|---|
| 1000 | 9 | 2 | 250 | [1000, 1003] |
| 1002 | 9 | 2 | 250 | [1000, 1003] |
| 1005 | 9 | 2 | 251 | [1004, 1007] |
| 2000 | 10 | 3 | 250 | [2000, 2007] |

The median has rank $\lceil 0.5 \cdot 4 \rceil = 2$: the second smallest value is in [1000, 1003], reported as 1003. The 100th percentile is in [2000, 2007] but clamped to the true maximum, 2000. The mean, from the exact sum, is 1251.75.

**One request** (test `TestMeasureHandExample`). Intended at $t = 0$; content at 120, 150, 190 ms; usage 3 tokens; `[DONE]` at 200 ms. TTFT 120 ms, ITL 30 and 40 ms, E2E 200 ms, TPOT $(200 - 120)/(3 - 1) = 40$ ms.

**The generator** (test `TestPCG32HandExample`). Seeding `pcg32(0)`: inc $= 2 \cdot 54 + 1 = 109$; state 0, one step gives 109, plus seed 0, one more step gives `0x9AE4F7499BA72696`. The first output permutes that state: xs $=$ `0x5C9A3E14`, rot $= 19$, output `0x47C28B93`.

## 4. The interface

```go
// go/ds/rng/pcg32.go
func New(seed, seq uint64) *PCG32; func Seeded(seed uint64) *PCG32          // pcg32_srandom_r; seq 54
func (r *PCG32) Uint32() uint32; func (r *PCG32) Float64() float64; func (r *PCG32) Normal() float64
func (r *PCG32) Below(n uint64) uint32; func (r *PCG32) Shuffle(n int, swap func(i, j int))
func (r *PCG32) State() (state, inc uint64)
func Mix64(z uint64) uint64; func ChildSeed(seed, purpose uint64) uint64; func Stream(seed, purpose uint64) *PCG32

// go/loadgen/schedule.go
type Schedule interface { Next() time.Duration; Name() string; Rate() float64 }
func Poisson(rate float64, r *rng.PCG32) Schedule; func Constant(rate float64) Schedule; func Burst(size int, every time.Duration) Schedule

// go/loadgen/histogram.go
func NewHistogram() *Histogram
func (h *Histogram) Record(ns int64); func (h *Histogram) Quantile(q float64) int64; func (h *Histogram) Merge(o *Histogram)
func (h *Histogram) Count() uint64; func (h *Histogram) Mean() float64; func (h *Histogram) Min() int64; func (h *Histogram) Max() int64
func (h *Histogram) Buckets() []Bucket                                        // Bucket{Lower, Upper int64; Count uint64}

// go/loadgen/run.go
type Clock interface { Now() time.Time; After(d time.Duration) <-chan time.Time }   // a testkit *clock.Fake satisfies it
type Config struct { Target, Model string; Prompts []string; MaxTokens int; APIKey string; Schedule Schedule
                     Duration time.Duration; MaxRequests int; SLO map[string]float64; RunID string; Seed uint64; Clock Clock; Client *http.Client }
type Event struct { At time.Time; Kind EventKind; Tokens int }                   // Content | Usage | Done
type Sample struct { Start time.Time; TTFT, E2E time.Duration; ITL []time.Duration; Tokens int; Err error }   // TPOT()
func ReadStream(r io.Reader, now func() time.Time) ([]Event, error)
func Measure(start time.Time, events []Event) Sample
func Run(ctx context.Context, cfg Config) (*Report, error)
func BuildReport(cfg Config, samples []Sample, elapsed time.Duration) *Report  // Report has the schema's json tags
```

### What the tests check

Runs use the course testkit's fake clock: the test moves time and waits for the runner's timers, so every latency below is an exact number. The generator is checked against `course/fixtures/parity/rng.json`.

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestHistogramHandExample` | unit | section 3's buckets, quantiles, extremes, mean | the worked example |
| `TestQuantilesWithinOnePercentOfExactSort` | property | log-uniform values over six decades: every reported percentile in $[x, x(1 + 1/128)]$ | the report's 1% promise |
| `TestQuantileRankRounding` | boundary | p7 and p55 of 1..100 are 7 and 55 | float rounding in the rank |
| `TestHistogramEdgeValues` | boundary | empty, negative, `MaxInt64` | no panic on odd input |
| `TestMergeEqualsRecordingEverything` | property | merged buckets, count, extremes, mean equal one histogram | per-worker histograms merge exactly |
| `TestPCG32HandExample` | unit | section 3's state and output; O'Neill's demo line | the generator is the spec's |
| `TestPCG32MatchesGoldenVectors` | conformance | 1024 u32, 64 uniforms bit for bit, 64 normals within 4 ulp, three seeds | `parity/rng` across four languages |
| `TestBelowRejectsTheBiasedZone` | unit | bounded draws for n = 10, 3·2^30, 2^32, 1 from the golden stream | unbiased permutations in `load.02` |
| `TestShuffleIsFisherYatesFromTheEnd` | unit | the spec's swap order | the same permutation in every language |
| `TestChildSeedAndStream` | unit | SplitMix64 sub-streams | independent purposes |
| `TestPoissonGapsByInverseCDF` | unit | 32 gaps equal $-\ln(1 - u)/\lambda$ for the golden uniforms | seeded runs repeat |
| `TestPoissonRateAndShape` | statistical | 20000 gaps: mean within 2%, 63.2% below the mean | the offered rate is the reported rate |
| `TestConstantAndBurstGaps` | unit | constant and burst gaps, names, rates | the other two modes |
| `TestReadStreamStampsEachEvent` | unit | one `now()` per data event; role chunk, ping, usage, `[DONE]` | TTFT and ITL timestamps |
| `TestReadStreamReportsBrokenStreams` | fault | cut stream, SSE error event, error object | outages show in `error_rate` |
| `TestMeasureHandExample` | unit | section 3's request | the worked example |
| `TestMeasurePrefersUsageAndRejectsEmpty` | boundary | usage over chunks; no content is an error; TPOT needs 2 tokens | multi-token chunks |
| `TestNoCoordinatedOmission` | fault | five requests sent on schedule into a stalled engine, each charged from its intended start | the reason for open loop |
| `TestRunCountsErrorsAndRecordsOnlySuccesses` | unit | 429s counted as errors, never as latencies; tokens per second | fast errors do not flatter p50 |
| `TestDurationEndsArrivals` | boundary | the arrival at exactly the end is not sent | back-to-back runs |
| `TestReportGoodputAndShape` | conformance | goodput over all requests; the schema's keys exactly; histogram rows | `load.02` and the milestones read it |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Reporting a bucket's lower bound | quantiles below the true value: the SLO looks met when it is not | `TestQuantilesWithinOnePercentOfExactSort` (mutant `s01`) |
| A plain $\lceil qn \rceil$ | p55 of 100 values is the 56th | `TestQuantileRankRounding` (mutant `s02`) |
| An even increment | a short-period stream, different from every other port | `TestPCG32HandExample` (mutant `s03`) |
| 27 bits from the second draw | uniforms differ from Python's | `TestPCG32MatchesGoldenVectors` (mutant `s04`) |
| `x mod n` without rejection | small values favored in permutations | `TestBelowRejectsTheBiasedZone` (mutant `s05`) |
| Fisher-Yates from the front | a different permutation than the other ports | `TestShuffleIsFisherYatesFromTheEnd` (mutant `s06`) |
| $\ln u$ instead of $\ln(1 - u)$ | the same distribution, different seeded runs | `TestPoissonGapsByInverseCDF` (mutant `s07`) |
| The first burst waits | a burst run starts late and sends fewer requests | `TestConstantAndBurstGaps` (mutant `s08`) |
| Stamping events late | TTFT and ITL inflated by parsing time | `TestReadStreamStampsEachEvent` (mutant `s09`) |
| A cut stream counted as a success | outages vanish from `error_rate` | `TestReadStreamReportsBrokenStreams` (mutant `s10`) |
| E2E from the last token | E2E is a few milliseconds | `TestMeasureHandExample` (mutant `s11`) |
| Waiting for each request (closed loop) | the send rate collapses during a stall | `TestNoCoordinatedOmission` (mutant `s12`) |
| Charging from the response headers | a stalled server looks fast (coordinated omission) | `TestNoCoordinatedOmission` (mutant `s13`) |
| Failed requests in the histograms | fast 429s pull p50 down during overload | `TestRunCountsErrorsAndRecordsOnlySuccesses` (mutant `s14`) |
| Goodput over successes only | an engine failing half its requests reports 100% goodput | `TestReportGoodputAndShape` (mutant `s15`) |

## 6. Where it's used next
| Forward | `ag.07` | Registered call site uses this module. |
| Forward | `ag.09` | Registered call site uses this module. |
| Forward | `ag.12` | Registered call site uses this module. |

| Direction | Module | How it uses this |
|---|---|---|
| Forward | `load.02` | optional: compares two reports: the histograms become samples for its permutation test, and `rng.Stream` shuffles them |

Your `{loadgen}` main is the load in `MS-L10` (a burst of 64), `MS-gateway`, `MS-prod` (the SLO rate on kind), and drills `ops.01` and `ops.09`; `L10.7`'s server-side histograms are the other side of the same measurement.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| an open-loop runner | wrk2, k6 (constant-arrival-rate) | corrected latency at fixed rates, many workers | [wrk2](https://github.com/giltene/wrk2), [k6 arrival-rate executors](https://grafana.com/docs/k6/latest/using-k6/scenarios/executors/) |
| a log-linear histogram | HdrHistogram | configurable precision, coordinated-omission correction, logs | [HdrHistogram Go](https://github.com/HdrHistogram/hdrhistogram-go) |
| TTFT, TPOT, ITL | vLLM and SGLang benchmark scripts, LLMPerf | dataset-driven prompts, request-rate sweeps, goodput curves | [vLLM benchmarks](https://github.com/vllm-project/vllm/tree/main/benchmarks), [LLMPerf](https://github.com/ray-project/llmperf) |
| one generator | multi-host load | distributed generators merging histograms | the merge in this module is the building block |
