# An open-loop load generator

The engine from [the tokio server](05-openai-server-on-tokio.md) and its metrics from [serving metrics and tracing](07-serving-metrics-and-tracing.md) are ready. How much load they take before the SLOs break is not yet known, and finding out is harder than it sounds.

The earlier Python load script waits for each response before sending the next one. That is a closed loop, and it has a fatal flaw: when the server stalls, the script stops sending and records one slow request instead of the dozens a real user population would have sent. Its percentiles look fine during exactly the outages that matter. This failure has a name, **coordinated omission**. Every load number from here on (the 64-request burst against the engine, the production SLO run, the incident walkthroughs, and [the regression gate](11-run-comparison-and-regression-gate.md) in CI) comes from the generator this section builds, so it has to be right: open loop, latencies from the intended start, honest histograms, seeded arrivals.

The generator is written in Go. It calls nothing else from the book, but it re-implements two ideas you have seen before: inverse-CDF sampling from [categorical sampling](../../mathematics-for-computing/07-probability-statistics/01-categorical-sampling.md) and the PCG32 generator from [modular arithmetic, hashing, and PCG32](../../mathematics-for-computing/06-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md), both in *Mathematics for Computing*. The [Go primer](../../programming-languages/02-language-and-tool-primers/06-go.md) in *Programming Languages* covers the language.

The contracts are the report format [`formats/loadgen-report.schema.json`](../../../../course/contracts/formats/loadgen-report.schema.json), the generator spec [`spec/pcg32.md`](../../../../course/contracts/spec/pcg32.md) (conformance `parity/rng`), the stream being measured in [`openapi/openai-subset.v1.yaml`](../../../../course/contracts/openapi/openai-subset.v1.yaml), and the `{loadgen}` role in [`spec/cli-roles.md`](../../../../course/contracts/spec/cli-roles.md). The solution is [`go/ds/rng/pcg32.go`](../../../../course/solutions/go/ds/rng/pcg32.go) (`PCG32`: `New`, `Uint32`, `Float64`, `Normal`, `Below`, `Shuffle`, `Stream`), [`go/loadgen/schedule.go`](../../../../course/solutions/go/loadgen/schedule.go) (`Poisson`, `Constant`, and `Burst` arrivals), [`go/loadgen/histogram.go`](../../../../course/solutions/go/loadgen/histogram.go) (a log-linear `Histogram` with at most 1/128 relative error), [`go/loadgen/run.go`](../../../../course/solutions/go/loadgen/run.go) (`ReadStream`, `Measure`, the open-loop `Run`, and `BuildReport`), and the entry point [`go/cmd/loadgen`](../../../../course/solutions/entry/go/cmd/loadgen/main.go). A run looks like this:

```bash
loadgen --target http://127.0.0.1:8000 --model smol-135m --mode poisson --rate 4 --duration 60s --seed 7 --out report.json
```

> **Learning objectives**
> - Explain coordinated omission and why an open-loop generator charging each request from its intended send time avoids it.
> - Measure TTFT, ITL, E2E, and TPOT from a streamed SSE response.
> - Build a log-linear histogram whose quantiles are within 1/128 of exact, always on the pessimistic side, in fixed memory.
> - Port PCG32 to Go so it produces the same bits as the other ports for the same seed.
> - Compute error rate and goodput so that failures never flatter the results.

## Arrivals

Here is the notation for the section.

| Symbol | Meaning | Type |
|---|---|---|
| $\lambda$ | offered rate (requests per second) | `float64` |
| $g_i$ | gap before arrival $i$ | `time.Duration` |
| $s_i = s_0 + \sum_{j \le i} g_j$ | the *intended* send time of request $i$ | `time.Time` |
| $v$ | a recorded value in nanoseconds, $v \ge 0$ | `int64` |
| $e = \lfloor \log_2 v \rfloor$ | the value's octave | int |
| $n$ | recorded values; $q \in [0, 1]$ a quantile | |

A load generator models users, not a client, and real users do not wait for each other. So arrival times must come from the schedule alone.

A Poisson process of rate $\lambda$ has independent exponential gaps. By inverse CDF, one uniform $u$ gives $g = -\ln(1 - u)/\lambda$; note that $1 - u$ is never 0. Constant arrivals use $g = 1/\lambda$ every time. A burst of size $b$ every $\tau$ uses gaps $0, 0, \ldots, \tau, 0, \ldots$.

The runner sends request $i$ at $s_i$ in its own goroutine, whether or not earlier requests have finished, and stops scheduling at $s_0 + $ duration. An arrival exactly at the end is not sent.

## Coordinated omission

The second half of being open loop is how latency is charged. Charge a request from $s_i$, not from when the client got around to sending it, and not from when the server answered the headers. If request $i$ is sent late because the client was busy, the user still waited from $s_i$. If the server stalls for one second at 10 rps, ten requests each see the stall, and all ten latencies record it.

## What one request measures

Each request reads an SSE stream: `data: <json>\n\n` events, with `[DONE]` last. Each event is stamped as it is read, because TTFT and ITL are differences of arrival times, and stamping after parsing a batch of lines would inflate them. From those stamps:

- TTFT is the first content chunk minus $s_i$.
- ITL is the gaps between consecutive content chunks.
- E2E is `[DONE]` minus $s_i$.
- Tokens are `usage.completion_tokens` when the server sends usage, else the content chunks.
- TPOT is $(E2E - TTFT)/(\text{tokens} - 1)$ for at least 2 tokens.

A failure is not a fast success. A 429 returns in a millisecond, and recording it as a latency would pull p50 down exactly when the system is overloaded. So an error status, an SSE error event, or a stream cut before `[DONE]` counts in `errors` and is kept out of every histogram.

## The log-linear histogram

Percentiles need bounded relative error in fixed memory. Storing every value grows without bound, and fixed linear buckets lose precision at either the low or the high end. The answer is to split every power of two into 128 equal buckets.

Values below $128$ get a bucket each. Above that, with $e = \lfloor \log_2 v \rfloor$ and shift $= e - 7$, the top 8 bits $v \gg \text{shift}$ lie in $[128, 256)$ and select bucket $(\text{shift} + 1) \cdot 128 + (v \gg \text{shift}) - 128$, which covers $[\text{sub} \cdot 2^{\text{shift}}, (\text{sub} + 1) 2^{\text{shift}} - 1]$. A bucket is at most $2^{\text{shift}}/(128 \cdot 2^{\text{shift}}) = 1/128$ of its lower bound wide.

The q-quantile is the nearest-rank value. The rank is $r = \lceil qn - 10^{-9} \rceil$, clamped to $[1, n]$; the small epsilon forgives float rounding, since $0.55 \cdot 100$ is $55.00000000000001$ in float64. The quantile is reported as the upper bound of the bucket holding the $r$-th smallest value, clamped to the true maximum. Reporting the upper bound means the estimate is never below the exact value: a reported percentile must never flatter the system.

Memory is fixed at 7424 counters for all of `int64`, and merging two histograms is adding counters.

## The generator

Seeded runs must repeat across languages. The Poisson gaps draw from PCG32, so the Go port must produce the same bits as the Python, C, and Rust ports for the same seed.

The port follows `spec/pcg32.md` exactly. The state update is $s \leftarrow 6364136223846793005\,s + \text{inc}$, and the output is the XSH-RR permutation of the old state. Seeding sets $\text{inc} = 2\,\text{seq} + 1$. `uniform_f64` takes two draws (53 bits). Normals use Box-Muller with the spare kept. `Below(n)` rejects draws under $(2^{32} - n) \bmod n$. Shuffles are Fisher-Yates from the end. Sub-streams come from SplitMix64.

## The report

The report follows `formats/loadgen-report.schema.json`: run id, target, mode, offered rate, duration, requests, errors, p50/p90/p95/p99/mean per metric in milliseconds, error rate, goodput, output tokens per second, and each histogram's non-empty buckets as `[upper_ms, count]`. Goodput is the share of *all* requests, failures included, that met every SLO bound.

## Working it by hand

Three examples pin down the histogram, one request, and the generator.

**A histogram.** Record 1000, 1002, 1005, and 2000 ns. Each lands in a bucket as follows:

| Value | $e$ | shift | top bits | bucket |
|---|---|---|---|---|
| 1000 | 9 | 2 | 250 | [1000, 1003] |
| 1002 | 9 | 2 | 250 | [1000, 1003] |
| 1005 | 9 | 2 | 251 | [1004, 1007] |
| 2000 | 10 | 3 | 250 | [2000, 2007] |

The median has rank $\lceil 0.5 \cdot 4 \rceil = 2$: the second smallest value is in [1000, 1003], reported as 1003. The 100th percentile is in [2000, 2007] but clamped to the true maximum, 2000. The mean, from the exact sum, is 1251.75.

**One request.** Intended at $t = 0$; content at 120, 150, and 190 ms; usage 3 tokens; `[DONE]` at 200 ms. TTFT is 120 ms, ITL is 30 and 40 ms, E2E is 200 ms, and TPOT is $(200 - 120)/(3 - 1) = 40$ ms.

**The generator.** Seeding `pcg32(0)`: inc $= 2 \cdot 54 + 1 = 109$. From state 0, one step gives 109; adding seed 0 and taking one more step gives `0x9AE4F7499BA72696`. The first output permutes that state: xs $=$ `0x5C9A3E14`, rot $= 19$, output `0x47C28B93`.

## Designing the interface

The code splits into the generator package and the load generator package.

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
type Clock interface { Now() time.Time; After(d time.Duration) <-chan time.Time }   // a fake clock satisfies it in tests
type Config struct { Target, Model string; Prompts []string; MaxTokens int; APIKey string; Schedule Schedule
                     Duration time.Duration; MaxRequests int; SLO map[string]float64; RunID string; Seed uint64; Clock Clock; Client *http.Client }
type Event struct { At time.Time; Kind EventKind; Tokens int }                   // Content | Usage | Done
type Sample struct { Start time.Time; TTFT, E2E time.Duration; ITL []time.Duration; Tokens int; Err error }   // TPOT()
func ReadStream(r io.Reader, now func() time.Time) ([]Event, error)
func Measure(start time.Time, events []Event) Sample
func Run(ctx context.Context, cfg Config) (*Report, error)
func BuildReport(cfg Config, samples []Sample, elapsed time.Duration) *Report  // Report has the schema's json tags
```

The `Clock` interface is the one design decision worth calling out: it lets the same runner loop run on a fake clock in tests, so every latency the tests check is an exact number.

## Building it step by step

### Reasoning from first principles

Seven observations, each already motivated above, fix the design.

1. **The generator models users, not a client.** Arrival times come from the schedule alone. That forces an open loop: the runner computes $s_i$ from gaps, sleeps until $s_i$, and starts request $i$ in its own goroutine.
2. **Latency is charged from the intended time.** Every `Sample` carries `Start = s_i`, and TTFT and E2E subtract it.
3. **Each event's timestamp must be taken when it is read.** So `ReadStream` calls `now()` once per data line, before parsing it.
4. **A failure is not a fast success.** Failures are counted in `errors` and kept out of every histogram, and goodput divides by all requests.
5. **Percentiles need bounded relative error in fixed memory.** Splitting every power of two into 128 equal buckets bounds the error at $1/128$ of the value, everywhere, in 7424 counters.
6. **A reported percentile must never flatter the system.** Reporting a bucket's upper bound (clamped to the true max) makes the estimate $\ge$ the exact value; the lower bound would make an SLO look met when it is not.
7. **Seeded runs must repeat across languages.** The Go PCG32 must produce the same bits as the Python, C, and Rust ports.

### The generator

Start at the bottom, with [`pcg32.go`](../../../../course/solutions/go/ds/rng/pcg32.go):

```go
// New is pcg32_srandom_r(seed, seq): inc = (seq << 1) | 1, state = 0, one
// step, state += seed, one step.
func New(seed, seq uint64) *PCG32 {
	r := &PCG32{inc: seq<<1 | 1} // odd increment: the LCG has full period 2^64
	r.Uint32()
	r.state += seed
	r.Uint32()
	return r
}

// Uint32 is one step; the output is a permutation of the OLD state.
func (r *PCG32) Uint32() uint32 {
	old := r.state
	r.state = old*mult + r.inc               // uint64 arithmetic wraps mod 2^64, as the spec needs
	xs := uint32(((old >> 18) ^ old) >> 27)  // XSH: xorshift high bits down
	rot := uint32(old >> 59)                 // RR: the top 5 bits pick the rotation
	return xs>>rot | xs<<((-rot)&31)
}

// Float64 is uniform_f64: 27 bits from the first draw, 26 from the second, 53 in all.
func (r *PCG32) Float64() float64 {
	a := uint64(r.Uint32() >> 5)
	b := uint64(r.Uint32() >> 6)
	return float64(a<<26+b) * (1.0 / 9007199254740992.0) // 2^-53: exactly representable, [0, 1)
}
```

The hand-worked example is `New(0, 54)`: `inc = 109`, one step from state 0 gives 109, plus seed 0, one more step gives `0x9AE4F7499BA72696`, and the first output permutes that state.

Bounded draws reject the biased zone so that every value of $[0, n)$ is equally likely:

```go
func (r *PCG32) Below(n uint64) uint32 {
	// ... 1 <= n <= 2^32
	t := ((1 << 32) - n) % n // the 2^32 mod n smallest draws would favor small results
	for {
		x := uint64(r.Uint32())
		if x >= t {
			return uint32(x % n)
		}
	}
}
```

`Shuffle` is Fisher-Yates from the end (`for i := n - 1; i >= 1; i--`, `j := Below(i + 1)`), and `Stream(seed, purpose)` is `New(ChildSeed(seed, purpose), purpose)` with SplitMix64's finalizer, so each purpose gets an independent stream from one seed.

### Arrivals

The schedules in [`schedule.go`](../../../../course/solutions/go/loadgen/schedule.go) are each a few lines:

```go
func (p *poisson) Next() time.Duration {
	u := p.r.Float64()
	sec := -math.Log(1.0-u) / p.rate // inverse CDF of Exp(rate); 1 - u is in (0, 1], so the log is finite
	return time.Duration(sec * float64(time.Second))
}

func (b *burst) Next() time.Duration {
	i := b.i
	b.i++
	if i > 0 && i%b.size == 0 { // the first burst starts at once
		return b.every
	}
	return 0
}
```

$-\ln(1 - u)$ and $-\ln u$ have the same distribution, but only the first matches the other ports draw for draw, which is what makes seeded runs repeat.

### Reading the stream

On the response side, [`run.go`](../../../../course/solutions/go/loadgen/run.go) reads SSE line by line and stamps each data line before parsing it:

```go
func ReadStream(r io.Reader, now func() time.Time) ([]Event, error) {
	br := bufio.NewReader(r)
	var out []Event
	errEvent := false
	for {
		line, err := br.ReadString('\n')
		if len(line) > 0 {
			line = strings.TrimRight(line, "\r\n")
			switch {
			case line == "" || strings.HasPrefix(line, ":"): // event separators and ": ping"
			case strings.HasPrefix(line, "event:"):
				errEvent = strings.TrimSpace(line[len("event:"):]) == "error"
			case strings.HasPrefix(line, "data:"):
				at := now() // stamped before parsing: parse time is not latency
				data := strings.TrimSpace(line[len("data:"):])
				if data == "[DONE]" {
					return append(out, Event{At: at, Kind: Done}), nil
				}
				// ... an error event or an "error" member is an error;
				// parseChunk classifies Content, Usage, or nothing (the role chunk)
			}
		}
		if err != nil {
			if errors.Is(err, io.EOF) {
				return out, errors.New("stream ended before [DONE]") // a cut stream is a failure
			}
			return out, err
		}
	}
}
```

`Measure` turns the events into one sample, relative to the intended start:

```go
func Measure(start time.Time, events []Event) Sample {
	s := Sample{Start: start}
	var prev time.Time
	usage := -1
	for _, ev := range events {
		switch ev.Kind {
		case Content:
			if s.Tokens == 0 {
				s.TTFT = ev.At.Sub(start) // from the intended send time
			} else {
				s.ITL = append(s.ITL, ev.At.Sub(prev))
			}
			prev = ev.At
			s.Tokens++
		case Usage:
			usage = ev.Tokens
		case Done:
			s.E2E = ev.At.Sub(start) // [DONE], not the last token
		}
	}
	if s.Tokens == 0 {
		s.Err = errors.New("no content before [DONE]")
		return s
	}
	if usage >= 0 {
		s.Tokens = usage // a chunk can carry several tokens: the server's count wins
	}
	return s
}
```

On the hand-worked request, content at 120, 150, and 190 ms gives TTFT 120 and ITL `[30, 40]`; `[DONE]` at 200 gives E2E 200; usage 3 gives TPOT $(200 - 120)/2 = 40$ ms (`Sample.TPOT` needs at least 2 tokens).

### The open-loop runner

The runner is the heart of the section, and it is short:

```go
next := start
for i := 0; cfg.MaxRequests <= 0 || i < cfg.MaxRequests; i++ {
	// The intended time comes from the schedule alone, never from when
	// the previous request finished.
	next = next.Add(cfg.Schedule.Next())
	if cfg.Duration > 0 && !next.Before(start.Add(cfg.Duration)) {
		break // an arrival exactly at the end is not sent
	}
	if wait := next.Sub(clk.Now()); wait > 0 {
		select {
		case <-clk.After(wait):
		case <-ctx.Done():
		}
	}
	if ctx.Err() != nil {
		break
	}
	wg.Add(1)
	go func(i int, at time.Time) { // one goroutine per request: a stalled one blocks nothing
		defer wg.Done()
		s := one(ctx, &cfg, client, clk, i, at) // `at` is the intended time, passed through to Measure
		mu.Lock()
		samples = append(samples, s)
		mu.Unlock()
	}(i, next)
}
wg.Wait()
```

`next` accumulates gaps from `start`, not from `clk.Now()`. If the loop wakes late, the next arrival is still at its scheduled time, and `wait` is negative, so it is sent at once. `one` sends the request with `temperature: 0`, `stream: true`, and `include_usage`, and treats a non-200 status as a failed sample.

### The histogram

Samples are recorded into histograms from [`histogram.go`](../../../../course/solutions/go/loadgen/histogram.go):

```go
// bucketOf maps v >= 0 to its bucket index: v itself below 128; else, with
// e = floor(log2 v) and shift = e - 7, the top 8 bits sub = v >> shift (in
// [128, 256)) select index (shift + 1) * 128 + (sub - 128).
func bucketOf(v int64) int {
	if v < subCount {
		return int(v) // exact below 128
	}
	e := bits.Len64(uint64(v)) - 1
	shift := e - SubBits
	sub := v >> uint(shift)
	return (shift+1)*subCount + int(sub-subCount)
}

func (h *Histogram) Quantile(q float64) int64 {
	// ... empty -> 0; q clamped to [0, 1]
	// ceil(q * n), forgiving the rounding of q * n: 0.55 * 100 is
	// 55.00000000000001 in float64, and its rank is 55, not 56.
	rank := uint64(math.Ceil(q*float64(h.n) - 1e-9))
	// ... clamp rank to [1, n]
	var cum uint64
	for i, c := range h.counts {
		cum += c
		if cum >= rank {
			_, hi := bounds(i)
			if hi > h.max {
				hi = h.max // the upper bound, but never past the true maximum
			}
			return hi
		}
	}
	return h.max
}
```

Check it against the hand-worked values. 1000 has `bits.Len64 = 10`, so $e = 9$, shift 2, sub $= 1000 \gg 2 = 250$, bucket $[1000, 1003]$. The median of four values has rank 2, found in that bucket, reported as 1003; p100 is in $[2000, 2007]$ and clamped to the max, 2000. `Mean` uses the exact running sum, not bucket midpoints. `Merge` adds counters and combines count, sum, min, and max, so merged per-worker histograms equal one histogram that saw everything.

### The report

Finally, `BuildReport` folds the samples into the report:

```go
for _, s := range samples {
	if s.Err != nil {
		r.Errors++
		continue // never in a latency histogram
	}
	hs["ttft_ms"].Record(int64(s.TTFT))
	hs["e2e_ms"].Record(int64(s.E2E))
	if t, ok := s.TPOT(); ok {
		hs["tpot_ms"].Record(int64(t))
	}
	for _, d := range s.ITL {
		hs["itl_ms"].Record(int64(d))
	}
	tokens += s.Tokens
	if good(s, cfg.SLO) {
		nGood++
	}
}
if r.Requests > 0 {
	r.ErrorRate = float64(r.Errors) / float64(r.Requests)
	r.Goodput = float64(nGood) / float64(r.Requests) // over ALL requests, failures included
}
```

`good` checks every SLO key it recognizes (`ttft_ms_p95` bounds `ttft_ms`, and so on); for ITL it uses the request's worst gap. Each histogram's non-empty buckets go into the report as `[upper_ms, count]` rows, which the regression gate reads back as samples.

The entry point [`go/cmd/loadgen`](../../../../course/solutions/entry/go/cmd/loadgen/main.go) parses `--target --model --prompt --max-tokens --rate --duration --mode poisson|constant|burst --burst --seed --out` (and the goodput bounds `--slo-ttft-ms`, default 500, and `--slo-tpot-ms`, default 60). It builds the schedule (`Poisson(rate, rng.Stream(seed, rng.PurposeSample))` for poisson), calls `loadgen.Run`, writes the report to `--out`, and prints it as the final stdout line.

The full solution is [`pcg32.go`](../../../../course/solutions/go/ds/rng/pcg32.go), [`schedule.go`](../../../../course/solutions/go/loadgen/schedule.go), [`histogram.go`](../../../../course/solutions/go/loadgen/histogram.go), and [`run.go`](../../../../course/solutions/go/loadgen/run.go).

### Why this design

The cost is small. `Record` is $O(1)$: a bit length, a shift, an increment. `Quantile` scans at most 7424 counters, about 58 KiB of memory per histogram regardless of how many values it holds. One goroutine per request is cheap in Go at the rates a single engine sustains.

Why not store all values and sort? That is exact, but memory grows with the run, and per-worker results cannot be merged without shipping every value. The log-linear histogram is within $1/128$ (0.78%) of exact, always on the pessimistic side, and merges by addition.

Why not a closed loop with N workers? It is simpler and is what many tools do, but its send rate drops when the server stalls, so the stall is recorded once instead of in every request it delayed. The open loop measures what users would see at the offered rate.

One tradeoff is accepted: goroutines are unbounded. If the server stops answering at a high rate, in-flight requests pile up in the client. That is the honest behaviour of an open loop; the run's `ctx` and the duration bound it.

## What can go wrong

The tests run on a fake clock: time is moved by hand and the runner's timers are awaited, so every latency is an exact number. The generator is checked against the shared PCG32 golden vectors of `parity/rng`. A correct solution handles each of these cases:

| Case | Checks | Why it matters downstream |
|---|---|---|
| The hand histogram | the worked buckets, quantiles, extremes, mean | the worked example |
| Quantiles within 1/128 of exact | log-uniform values over six decades: every reported percentile in $[x, x(1 + 1/128)]$ | the report's 1% promise |
| Rank rounding | p7 and p55 of 1..100 are 7 and 55 | float rounding in the rank |
| Edge values | empty, negative, `MaxInt64` | no panic on odd input |
| Merge equals recording everything | merged buckets, count, extremes, mean equal one histogram | per-worker histograms merge exactly |
| The hand PCG32 | the worked state and output; O'Neill's demo line | the generator is the spec's |
| Golden vectors | 1024 u32, 64 uniforms bit for bit, 64 normals within 4 ulp, three seeds | `parity/rng` across four languages |
| `Below` rejects the biased zone | bounded draws for n = 10, 3·2^30, 2^32, 1 from the golden stream | unbiased permutations in the regression gate |
| Fisher-Yates from the end | the spec's swap order | the same permutation in every language |
| Child seeds and streams | SplitMix64 sub-streams | independent purposes |
| Poisson gaps by inverse CDF | 32 gaps equal $-\ln(1 - u)/\lambda$ for the golden uniforms | seeded runs repeat |
| Poisson rate and shape | 20000 gaps: mean within 2%, 63.2% below the mean | the offered rate is the reported rate |
| Constant and burst gaps | constant and burst gaps, names, rates | the other two modes |
| Stamping each event | one `now()` per data event; role chunk, ping, usage, `[DONE]` | TTFT and ITL timestamps |
| Broken streams | cut stream, SSE error event, error object | outages show in `error_rate` |
| The hand request | the worked request | the worked example |
| Usage over chunks, empty rejected | usage over chunks; no content is an error; TPOT needs 2 tokens | multi-token chunks |
| No coordinated omission | five requests sent on schedule into a stalled engine, each charged from its intended start | the reason for open loop |
| Errors counted, only successes recorded | 429s counted as errors, never as latencies; tokens per second | fast errors do not flatter p50 |
| Duration ends arrivals | the arrival at exactly the end is not sent | back-to-back runs |
| Goodput and report shape | goodput over all requests; the schema's keys exactly; histogram rows | the regression gate and every system-level run read it |

> **Warning:** The measurement pitfalls are the ones that matter most, because they make a broken system look healthy. Waiting for each request (a closed loop) makes the send rate collapse during a stall. Charging from the response headers makes a stalled server look fast. Failed requests in the histograms let fast 429s pull p50 down during overload. Goodput over successes only lets an engine that fails half its requests report 100% goodput. The solution guards these with one goroutine per request and arrivals from the schedule, `Measure` subtracting the intended start passed from `Run`, `BuildReport` skipping samples with `Err`, and goodput dividing by `Requests`.

The histogram has two quieter pitfalls. Reporting a bucket's lower bound gives quantiles below the true value, so the SLO looks met when it is not; `Quantile` returns the bucket's upper bound, clamped to the max. A plain $\lceil qn \rceil$ makes p55 of 100 values the 56th; the rank is `ceil(q * n - 1e-9)`.

The generator's pitfalls break cross-language repeatability. An even increment gives a short-period stream, different from every other port; `New` sets `inc = seq<<1 | 1`. Taking 27 bits from the second draw makes uniforms differ from Python's; `Float64` takes `>> 5` (27 bits) then `>> 6` (26 bits). `x mod n` without rejection favors small values in permutations; `Below` redraws below `(2^32 - n) mod n`. Fisher-Yates from the front gives a different permutation than the other ports; `Shuffle` runs `i` from `n - 1` down to 1. Using $\ln u$ instead of $\ln(1 - u)$ gives the same distribution but different seeded runs; the Poisson gap is `-math.Log(1.0-u) / rate`.

The stream reader and schedule have the rest. If the first burst waits, a burst run starts late and sends fewer requests; `burst.Next` returns `every` only for `i > 0`. Stamping events late inflates TTFT and ITL by parsing time; `ReadStream` calls `now()` before parsing each data line. Counting a cut stream as a success makes outages vanish from `error_rate`; EOF before `[DONE]` is an error. And taking E2E from the last token instead of `[DONE]` shaves a few milliseconds off E2E; E2E is taken at the `Done` event.

## Where this leads

The optional next section, [run comparison and the regression gate](11-run-comparison-and-regression-gate.md), compares two reports: the histograms become samples for its permutation test, and `rng.Stream` shuffles them. In *AI Platform Engineering*, the Go PCG32 (`rng.Seeded`, `rng.Stream`, `Below`, `Float64`) is reused for k-means++ seeding in [retrieval](../../ai-platform-engineering/07-retrieval-and-rag/07-retrieval.md), the bootstrap of the [eval runner](../../ai-platform-engineering/09-llm-evaluation/09-eval-runner.md), and the permutation test of [paired A/B experiments](../../ai-platform-engineering/09-llm-evaluation/12-ab-experiments.md).

The `{loadgen}` entry point is the load behind every system-level run: a burst of 64 against the engine, the gateway run, the production SLO run on kind, and the incident drills [a decode worker dies mid-stream](../../systems/05-incident-response-and-chaos/01-kill-decode.md) and [noisy neighbor](../../systems/05-incident-response-and-chaos/09-noisy-neighbor.md) in *Systems*. The server-side histograms from [serving metrics and tracing](07-serving-metrics-and-tracing.md) are the other side of the same measurement.

Production tools extend each piece. wrk2 and k6's constant-arrival-rate executors provide corrected latency at fixed rates with many workers. HdrHistogram adds configurable precision, coordinated-omission correction, and logs. The vLLM and SGLang benchmark scripts and LLMPerf add dataset-driven prompts, request-rate sweeps, and goodput curves on top of TTFT, TPOT, and ITL. And multi-host load runs distributed generators that merge their histograms, for which the exact merge here is the building block.

### Further reading

- Gil Tene, [How NOT to Measure Latency](https://www.youtube.com/watch?v=lJ8ydIuPFeU) (free talk).
- [HdrHistogram](http://hdrhistogram.org/) (free) and [HdrHistogram Go](https://github.com/HdrHistogram/hdrhistogram-go).
- O'Neill, [PCG](https://www.pcg-random.org/paper.html) (free).
- Schroeder, Wierman, Harchol-Balter, [Open Versus Closed](https://www.usenix.org/legacy/event/nsdi06/tech/full_papers/schroeder/schroeder.pdf) (free).
- [wrk2](https://github.com/giltene/wrk2) and [k6 arrival-rate executors](https://grafana.com/docs/k6/latest/using-k6/scenarios/executors/).
- [vLLM benchmarks](https://github.com/vllm-project/vllm/tree/main/benchmarks) and [LLMPerf](https://github.com/ray-project/llmperf).

## Key terms

- **Open loop**: a load model in which arrivals follow a schedule regardless of how the server responds.
- **Coordinated omission**: the measurement error of a closed-loop client that stops sending during a stall and so records it once instead of in every request it delayed.
- **Intended send time**: $s_i = s_0 + \sum_{j \le i} g_j$, the time from which request $i$'s latency is charged.
- **ITL**: inter-token latency, the gap between consecutive content chunks of a stream.
- **Log-linear histogram**: a histogram with 128 equal buckets per power of two, giving at most 1/128 relative error in fixed memory.
- **Nearest-rank quantile**: the $\lceil qn \rceil$-th smallest value, here with a $10^{-9}$ allowance for float rounding.
- **Goodput**: the share of all requests, failures included, that met every SLO bound.

## Summary

- Open loop: arrivals follow the schedule whatever the server does, and each request is charged from its *intended* send time, so a stall shows up in every request it delayed.
- A log-linear histogram keeps 128 buckets per power of two: every quantile is within 1/128 of the exact sorted value, never below it, in fixed memory, and two histograms merge exactly.
- The nearest-rank quantile is the $\lceil qn \rceil$-th smallest value, with an allowance for float rounding: $0.55 \cdot 100$ is $55.00000000000001$.
- The Go PCG32 gives the same stream as Python, C, and Rust for the same seed, bit for bit.
- A failed request (an error status, an SSE error event, a stream cut before `[DONE]`) counts in `errors`, never in a latency histogram; goodput is the share of *all* requests that met every SLO bound.

## Review questions

1. The engine stalls for 1 s at 10 rps. What does a closed-loop client record, and what does this runner record?
   <details><summary>Answer</summary>

   A closed loop with one worker is stuck on one request for the second and records one slow latency; it then resumes, and the requests users would have sent during the stall never existed. This runner keeps sending at $s_i$ regardless, so about ten requests are in flight during the stall, and each is charged from its own intended start: ten latencies show the stall, with the earliest seeing nearly the full second.

   </details>

2. Why does `Quantile` subtract `1e-9` before `math.Ceil`?
   <details><summary>Answer</summary>

   $0.55 \cdot 100$ in float64 is $55.00000000000001$, so a plain ceiling gives rank 56 and p55 of 1..100 would be 56. The nearest-rank definition is $\lceil qn \rceil = 55$. Subtracting a tiny epsilon forgives the rounding error without changing any rank whose product is genuinely above an integer.

   </details>

3. Why is a quantile reported as its bucket's upper bound, and why clamped to the max?
   <details><summary>Answer</summary>

   The upper bound is never below any value in the bucket, so the estimate is never below the exact quantile and at most $1/128$ above it: the error is always pessimistic, so an SLO can only look worse than it is, never better. The clamp keeps p100 (and any quantile in the top bucket) from exceeding the largest value actually seen.

   </details>

4. Why does `next` accumulate from `start` instead of from `clk.Now()` after each send?
   <details><summary>Answer</summary>

   Adding gaps to the current time would push every later arrival back by however long the loop was late, so a slow client would silently lower the offered rate. Accumulating from `start` keeps $s_i = s_0 + \sum g_j$ exactly; if the loop wakes late, `wait` is negative and the request goes at once, still charged from its intended $s_i$.

   </details>

5. An engine fails half of its requests with fast 429s and serves the rest within SLO. What are `error_rate` and `goodput`?
   <details><summary>Answer</summary>

   Both 0.5. The 429s count in `requests` and `errors`, are kept out of every histogram (so they do not pull p50 down), and are in goodput's denominator. Dividing by successes only would report 100% goodput for an engine that turned away half its users.

   </details>

6. Which bucket does the value 1005 ns fall in, and what are its bounds?
   <details><summary>Answer</summary>

   `bits.Len64(1005) = 10`, so $e = 9$ and shift $= 2$. The top bits are $1005 \gg 2 = 251$, which selects the bucket $[251 \cdot 4, 252 \cdot 4 - 1] = [1004, 1007]$. Its width, 4, is $4/1004 < 1/128$ of its lower bound, as the design promises.

   </details>
