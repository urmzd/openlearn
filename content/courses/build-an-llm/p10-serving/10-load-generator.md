# Open-loop load generator, log-linear histogram, Go PCG32 port

## Overview

| | |
|---|---|
| **Module** | `load.01` · build · Go · Pass 7 · 3 to 5 h reading |
| **Solution** | [`go/ds/rng/pcg32.go`](../../../../course/solutions/go/ds/rng/pcg32.go): `PCG32` (`New`, `Uint32`, `Float64`, `Normal`, `Below`, `Shuffle`, `Stream`), the course's random generator in Go · [`go/loadgen/schedule.go`](../../../../course/solutions/go/loadgen/schedule.go): `Poisson`, `Constant`, and `Burst` arrivals · [`go/loadgen/histogram.go`](../../../../course/solutions/go/loadgen/histogram.go): `Histogram`, log-linear with at most 1/128 relative error · [`go/loadgen/run.go`](../../../../course/solutions/go/loadgen/run.go): `ReadStream`, `Measure`, the open-loop `Run`, and `BuildReport` · the entry point [`go/cmd/loadgen`](../../../../course/solutions/entry/go/cmd/loadgen/main.go) |
| **Contract** | the report: [`formats/loadgen-report.schema.json`](../../../../course/contracts/formats/loadgen-report.schema.json) · the generator: [`spec/pcg32.md`](../../../../course/contracts/spec/pcg32.md) (conformance `parity/rng`) · the stream it measures: [`openapi/openai-subset.v1.yaml`](../../../../course/contracts/openapi/openai-subset.v1.yaml) · the role `{loadgen}`: [`spec/cli-roles.md`](../../../../course/contracts/spec/cli-roles.md) |
| **Needs** | nothing to call: reading `lang.06` Go ([primer](../../../tracks/software-craftsmanship/12-language-and-tool-primers/06-go.md)), `M07.1` inverse-CDF sampling ([chapter](../../../tracks/math/07-probability-statistics/01-categorical-sampling.md)), `M06.3` PCG32 ([chapter](../../../tracks/math/06-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md)); both are re-implemented here in Go |
| **Used by** | the `{loadgen}` entry point drives every load measurement of the system: the 64-request burst against the engine, the gateway, the production SLO run, and the incident walkthroughs · later: `ag.07`, `ag.09`, `ag.12` |
| **Optional depth** | Gil Tene, [How NOT to Measure Latency](https://www.youtube.com/watch?v=lJ8ydIuPFeU) (free talk); [HdrHistogram](http://hdrhistogram.org/) (free); [O'Neill, PCG](https://www.pcg-random.org/paper.html) (free); [Schroeder, Wierman, Harchol-Balter, Open Versus Closed](https://www.usenix.org/legacy/event/nsdi06/tech/full_papers/schroeder/schroeder.pdf) (free) |

## Key Takeaways

- Open loop: arrivals follow the schedule whatever the server does, and each request is charged from its INTENDED send time, so a stall shows up in every request it delayed.
- A log-linear histogram keeps 128 buckets per power of two: every quantile is within 1/128 of the exact sorted value, never below it, in fixed memory, and two histograms merge exactly.
- The nearest-rank quantile is the $\lceil qn \rceil$-th smallest value, with an allowance for float rounding: $0.55 \cdot 100$ is $55.00000000000001$.
- The Go PCG32 gives the same stream as Python, C, and Rust for the same seed, bit for bit.
- A failed request (an error status, an SSE error event, a stream cut before `[DONE]`) counts in `errors`, never in a latency histogram; goodput is the share of ALL requests that met every SLO bound.

## How to read this chapter

Sections 1 to 3 explain why the load generator must be open loop, how arrivals, measurements, the histogram, and the generator work, and work a histogram, a request, and the generator by hand. Section 4 fixes the interface. Section 5 walks through the solution: PCG32, the schedules, the stream reader, the runner, the histogram, and the report. Try to predict each step before reading it.

The entry point [`go/cmd/loadgen`](../../../../course/solutions/entry/go/cmd/loadgen/main.go) wires them together; a run looks like

```bash
loadgen --target http://127.0.0.1:8000 --model smol-135m --mode poisson --rate 4 --duration 60s --seed 7 --out report.json
```

---

## 1. Why now

The engine (`L10.5`) and its metrics (`L10.7`) are ready; how much load they take before the SLOs break is not known. The Pass 4 Python load script in `ml/04` waits for each response before sending the next (closed loop): when the server stalls, it stops sending and records one slow request instead of the dozens a real user population would have sent, so its percentiles look fine during exactly the outages that matter (coordinated omission). Every load number from here on (the 64-request burst against the engine, the production SLO run, the incident walkthroughs, `load.02`'s regression gate in CI) comes from this generator, so it has to be right: open loop, latencies from the intended start, honest histograms, seeded arrivals.

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

**A histogram.** Record 1000, 1002, 1005, 2000 ns.

| Value | $e$ | shift | top bits | bucket |
|---|---|---|---|---|
| 1000 | 9 | 2 | 250 | [1000, 1003] |
| 1002 | 9 | 2 | 250 | [1000, 1003] |
| 1005 | 9 | 2 | 251 | [1004, 1007] |
| 2000 | 10 | 3 | 250 | [2000, 2007] |

The median has rank $\lceil 0.5 \cdot 4 \rceil = 2$: the second smallest value is in [1000, 1003], reported as 1003. The 100th percentile is in [2000, 2007] but clamped to the true maximum, 2000. The mean, from the exact sum, is 1251.75.

**One request.** Intended at $t = 0$; content at 120, 150, 190 ms; usage 3 tokens; `[DONE]` at 200 ms. TTFT 120 ms, ITL 30 and 40 ms, E2E 200 ms, TPOT $(200 - 120)/(3 - 1) = 40$ ms.

**The generator.** Seeding `pcg32(0)`: inc $= 2 \cdot 54 + 1 = 109$; state 0, one step gives 109, plus seed 0, one more step gives `0x9AE4F7499BA72696`. The first output permutes that state: xs $=$ `0x5C9A3E14`, rot $= 19$, output `0x47C28B93`.

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

### What a correct solution must handle

Runs use a fake clock: time is moved by hand and the runner's timers are awaited, so every latency below is an exact number. The generator is checked against the shared PCG32 golden vectors of `parity/rng`.

| Case | Checks | Why it matters downstream |
|---|---|---|
| The hand histogram | section 3's buckets, quantiles, extremes, mean | the worked example |
| Quantiles within 1/128 of exact | log-uniform values over six decades: every reported percentile in $[x, x(1 + 1/128)]$ | the report's 1% promise |
| Rank rounding | p7 and p55 of 1..100 are 7 and 55 | float rounding in the rank |
| Edge values | empty, negative, `MaxInt64` | no panic on odd input |
| Merge equals recording everything | merged buckets, count, extremes, mean equal one histogram | per-worker histograms merge exactly |
| The hand PCG32 | section 3's state and output; O'Neill's demo line | the generator is the spec's |
| Golden vectors | 1024 u32, 64 uniforms bit for bit, 64 normals within 4 ulp, three seeds | `parity/rng` across four languages |
| `Below` rejects the biased zone | bounded draws for n = 10, 3·2^30, 2^32, 1 from the golden stream | unbiased permutations in `load.02` |
| Fisher-Yates from the end | the spec's swap order | the same permutation in every language |
| Child seeds and streams | SplitMix64 sub-streams | independent purposes |
| Poisson gaps by inverse CDF | 32 gaps equal $-\ln(1 - u)/\lambda$ for the golden uniforms | seeded runs repeat |
| Poisson rate and shape | 20000 gaps: mean within 2%, 63.2% below the mean | the offered rate is the reported rate |
| Constant and burst gaps | constant and burst gaps, names, rates | the other two modes |
| Stamping each event | one `now()` per data event; role chunk, ping, usage, `[DONE]` | TTFT and ITL timestamps |
| Broken streams | cut stream, SSE error event, error object | outages show in `error_rate` |
| The hand request | section 3's request | the worked example |
| Usage over chunks, empty rejected | usage over chunks; no content is an error; TPOT needs 2 tokens | multi-token chunks |
| No coordinated omission | five requests sent on schedule into a stalled engine, each charged from its intended start | the reason for open loop |
| Errors counted, only successes recorded | 429s counted as errors, never as latencies; tokens per second | fast errors do not flatter p50 |
| Duration ends arrivals | the arrival at exactly the end is not sent | back-to-back runs |
| Goodput and report shape | goodput over all requests; the schema's keys exactly; histogram rows | `load.02` and every system-level run read it |

## 5. The solution, step by step

### 5.1 From first principles

1. **The generator models users, not a client.** Real users do not wait for each other, so arrival times must come from the schedule alone. That forces an open loop: the runner computes $s_i$ from gaps, sleeps until $s_i$, and starts request $i$ in its own goroutine (section 2.1).
2. **Latency is charged from the intended time.** If request $i$ is sent late because the client was busy, the user still waited from $s_i$. So every `Sample` carries `Start = s_i`, and TTFT and E2E subtract it (section 2.2).
3. **Each event's timestamp must be taken when it is read.** TTFT and ITL are differences of arrival times; stamping after parsing a batch of lines would inflate them. So `ReadStream` calls `now()` once per data line, before parsing it.
4. **A failure is not a fast success.** A 429 returns in a millisecond; recording it as a latency would pull p50 down exactly when the system is overloaded. So failures are counted in `errors` and kept out of every histogram, and goodput divides by all requests.
5. **Percentiles need bounded relative error in fixed memory.** Storing every value grows without bound; fixed linear buckets lose precision either at the low or the high end. Splitting every power of two into 128 equal buckets bounds the error at $1/128$ of the value, everywhere, in 7424 counters (section 2.4).
6. **A reported percentile must never flatter the system.** Reporting a bucket's upper bound (clamped to the true max) makes the estimate $\ge$ the exact value; the lower bound would make an SLO look met when it is not.
7. **Seeded runs must repeat across languages.** The Poisson gaps draw from PCG32, so the Go port must produce the same bits as the Python, C, and Rust ports for the same seed (section 2.5).

### 5.2 The generator

[`pcg32.go`](../../../../course/solutions/go/ds/rng/pcg32.go):

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

Section 3's hand example is `New(0, 54)`: `inc = 109`, one step from state 0 gives 109, plus seed 0, one more step gives `0x9AE4F7499BA72696`, and the first output permutes that state.

Bounded draws reject the biased zone so every value of $[0, n)$ is equally likely:

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

### 5.3 Arrivals

[`schedule.go`](../../../../course/solutions/go/loadgen/schedule.go):

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

### 5.4 Reading the stream

[`run.go`](../../../../course/solutions/go/loadgen/run.go):

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

`Measure` turns events into one sample relative to the intended start:

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

Section 3's request: content at 120, 150, 190 ms gives TTFT 120 and ITL `[30, 40]`; `[DONE]` at 200 gives E2E 200; usage 3 gives TPOT $(200 - 120)/2 = 40$ ms (`Sample.TPOT` needs at least 2 tokens).

### 5.5 The open-loop runner

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

`next` accumulates gaps from `start`, not from `clk.Now()`: if the loop wakes late, the next arrival is still at its scheduled time, and `wait` is negative so it is sent at once. The `Clock` interface lets the same loop run on a fake clock, which is how every latency in this chapter's cases is an exact number. `one` sends the request with `temperature: 0`, `stream: true`, and `include_usage`, and treats a non-200 status as a failed sample.

### 5.6 The histogram

[`histogram.go`](../../../../course/solutions/go/loadgen/histogram.go):

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

Section 3's values: 1000 has `bits.Len64 = 10`, so $e = 9$, shift 2, sub $= 1000 \gg 2 = 250$, bucket $[1000, 1003]$. The median of four values has rank 2, found in that bucket, reported as 1003; p100 is in $[2000, 2007]$ and clamped to the max, 2000. `Mean` uses the exact running sum, not bucket midpoints. `Merge` adds counters and combines count, sum, min, and max, so merged per-worker histograms equal one histogram that saw everything.

### 5.7 The report

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

`good` checks every SLO key it recognizes (`ttft_ms_p95` bounds `ttft_ms`, and so on); for ITL it uses the request's worst gap. Each histogram's non-empty buckets go into the report as `[upper_ms, count]` rows, which `load.02` reads back as samples.

The entry point [`go/cmd/loadgen`](../../../../course/solutions/entry/go/cmd/loadgen/main.go) parses `--target --model --prompt --max-tokens --rate --duration --mode poisson|constant|burst --burst --seed --out` (and the goodput bounds `--slo-ttft-ms`, default 500, and `--slo-tpot-ms`, default 60), builds the schedule (`Poisson(rate, rng.Stream(seed, rng.PurposeSample))` for poisson), calls `loadgen.Run`, writes the report to `--out`, and prints it as the final stdout line.

Full solution: [`pcg32.go`](../../../../course/solutions/go/ds/rng/pcg32.go), [`schedule.go`](../../../../course/solutions/go/loadgen/schedule.go), [`histogram.go`](../../../../course/solutions/go/loadgen/histogram.go), [`run.go`](../../../../course/solutions/go/loadgen/run.go).

### 5.8 Why this solution

- **Cost.** `Record` is $O(1)$ (a bit length, a shift, an increment); `Quantile` scans at most 7424 counters, about 58 KiB of memory per histogram regardless of how many values it holds. One goroutine per request is cheap in Go at the rates a single engine sustains.
- **Why not store all values and sort?** Exact, but memory grows with the run and per-worker results cannot be merged without shipping every value. The log-linear histogram is within $1/128$ (0.78%) of exact, always on the pessimistic side, and merges by addition.
- **Why not a closed loop with N workers?** It is simpler and is what many tools do, but its send rate drops when the server stalls, so the stall is recorded once instead of in every request it delayed (coordinated omission). The open loop measures what users would see at the offered rate.
- **Tradeoffs accepted.** Unbounded goroutines: if the server stops answering at high rate, in-flight requests pile up in the client. That is the honest behaviour of an open loop; the run's `ctx` and the duration bound it.

### 5.9 Explain it back

**The engine stalls for 1 s at 10 rps. What does a closed-loop client record, and what does this runner record?**

<details><summary>Answer</summary>

A closed loop with one worker is stuck on one request for the second and records one slow latency; it then resumes, and the requests users would have sent during the stall never existed. This runner keeps sending at $s_i$ regardless, so about ten requests are in flight during the stall, and each is charged from its own intended start: ten latencies show the stall, with the earliest seeing nearly the full second.

</details>

**Why does `Quantile` subtract `1e-9` before `math.Ceil`?**

<details><summary>Answer</summary>

$0.55 \cdot 100$ in float64 is $55.00000000000001$, so a plain ceiling gives rank 56 and p55 of 1..100 would be 56. The nearest-rank definition is $\lceil qn \rceil = 55$. Subtracting a tiny epsilon forgives the rounding error without changing any rank whose product is genuinely above an integer.

</details>

**Why is a quantile reported as its bucket's upper bound, and why clamped to the max?**

<details><summary>Answer</summary>

The upper bound is never below any value in the bucket, so the estimate is never below the exact quantile and at most $1/128$ above it: the error is always pessimistic, so an SLO can only look worse than it is, never better. The clamp keeps p100 (and any quantile in the top bucket) from exceeding the largest value actually seen.

</details>

**Why does `next` accumulate from `start` instead of from `clk.Now()` after each send?**

<details><summary>Answer</summary>

Adding gaps to the current time would push every later arrival back by however long the loop was late, so a slow client would silently lower the offered rate. Accumulating from `start` keeps $s_i = s_0 + \sum g_j$ exactly; if the loop wakes late, `wait` is negative and the request goes at once, still charged from its intended $s_i$.

</details>

**An engine fails half of its requests with fast 429s and serves the rest within SLO. What are `error_rate` and `goodput`?**

<details><summary>Answer</summary>

Both 0.5. The 429s count in `requests` and `errors`, are kept out of every histogram (so they do not pull p50 down), and are in goodput's denominator. Dividing by successes only would report 100% goodput for an engine that turned away half its users.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Reporting a bucket's lower bound | quantiles below the true value: the SLO looks met when it is not | `Quantile` returns the bucket's upper bound, clamped to the max; see 5.6 |
| A plain $\lceil qn \rceil$ | p55 of 100 values is the 56th | the rank is `ceil(q * n - 1e-9)`; see 5.6 |
| An even increment | a short-period stream, different from every other port | `New` sets `inc = seq<<1 \| 1`; see 5.2 |
| 27 bits from the second draw | uniforms differ from Python's | `Float64` takes `>> 5` (27 bits) then `>> 6` (26 bits); see 5.2 |
| `x mod n` without rejection | small values favored in permutations | `Below` redraws below `(2^32 - n) mod n`; see 5.2 |
| Fisher-Yates from the front | a different permutation than the other ports | `Shuffle` runs `i` from `n - 1` down to 1 |
| $\ln u$ instead of $\ln(1 - u)$ | the same distribution, different seeded runs | the Poisson gap is `-math.Log(1.0-u) / rate`; see 5.3 |
| The first burst waits | a burst run starts late and sends fewer requests | `burst.Next` returns `every` only for `i > 0`; see 5.3 |
| Stamping events late | TTFT and ITL inflated by parsing time | `ReadStream` calls `now()` before parsing each data line; see 5.4 |
| A cut stream counted as a success | outages vanish from `error_rate` | EOF before `[DONE]` is an error; see 5.4 |
| E2E from the last token | E2E is a few milliseconds | E2E is taken at the `Done` event; see 5.4 |
| Waiting for each request (closed loop) | the send rate collapses during a stall | every request runs in its own goroutine and arrivals come from the schedule; see 5.5 |
| Charging from the response headers | a stalled server looks fast (coordinated omission) | `Measure` subtracts the intended start passed from `Run`; see 5.4 and 5.5 |
| Failed requests in the histograms | fast 429s pull p50 down during overload | `BuildReport` skips samples with `Err`; see 5.7 |
| Goodput over successes only | an engine failing half its requests reports 100% goodput | goodput divides by `Requests`; see 5.7 |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Forward | `load.02` | optional: compares two reports: the histograms become samples for its permutation test, and `rng.Stream` shuffles them |
| Forward | `ag.07`, `ag.09`, `ag.12` | load runs against the retrieval and evaluation services |

The `{loadgen}` entry point is the load behind every system-level run: a burst of 64 against the engine, the gateway run, the production SLO run on kind, and the `ops.01` and `ops.09` incident walkthroughs; `L10.7`'s server-side histograms are the other side of the same measurement.

## Going further

| This solution | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| an open-loop runner | wrk2, k6 (constant-arrival-rate) | corrected latency at fixed rates, many workers | [wrk2](https://github.com/giltene/wrk2), [k6 arrival-rate executors](https://grafana.com/docs/k6/latest/using-k6/scenarios/executors/) |
| a log-linear histogram | HdrHistogram | configurable precision, coordinated-omission correction, logs | [HdrHistogram Go](https://github.com/HdrHistogram/hdrhistogram-go) |
| TTFT, TPOT, ITL | vLLM and SGLang benchmark scripts, LLMPerf | dataset-driven prompts, request-rate sweeps, goodput curves | [vLLM benchmarks](https://github.com/vllm-project/vllm/tree/main/benchmarks), [LLMPerf](https://github.com/ray-project/llmperf) |
| one generator | multi-host load | distributed generators merging histograms | the merge in this module is the building block |
