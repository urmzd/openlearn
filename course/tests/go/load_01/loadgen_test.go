// Course tests for load.01: the Go PCG32 port (go/ds/rng) and the open-loop
// load generator (go/loadgen): schedules, the log-linear histogram, the SSE
// reader, per-request measurement, the runner, and the report.
//
// Nothing here sleeps. Runs that involve time use the course testkit's fake
// clock: the test moves time with Advance and waits for the runner with
// BlockUntil, so every latency below is an exact number, not a range.
// Random test data comes from the tests' own SplitMix64 generator with fixed
// seeds; the generator under test is checked against the golden vectors of
// parity/rng.
package load_01

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"openlearn.urmzd.com/tl/testkit/clock"
	"tinyllm/ds/rng"
	"tinyllm/loadgen"
)

// testRand is the tests' own generator (SplitMix64), so test data never
// depends on the code under test or on math/rand (DESIGN D35).
type testRand struct{ s uint64 }

func newTestRand(a, b uint64) *testRand { return &testRand{s: a*0x9E3779B97F4A7C15 ^ b} }

func (r *testRand) next() uint64 {
	r.s += 0x9E3779B97F4A7C15
	z := r.s
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

// Float64 is uniform on [0, 1) with 53 bits.
func (r *testRand) Float64() float64 { return float64(r.next()>>11) / (1 << 53) }

// IntN is uniform on [0, n) (the modulo bias is negligible for test sizes).
func (r *testRand) IntN(n int) int { return int(r.next() % uint64(n)) }

// ExpFloat64 is a standard exponential by inverse CDF.
func (r *testRand) ExpFloat64() float64 { return -math.Log(1 - r.Float64()) }

// NormFloat64 is a standard normal by Box-Muller (the cosine half).
func (r *testRand) NormFloat64() float64 {
	u1, u2 := r.Float64(), r.Float64()
	return math.Sqrt(-2*math.Log(1-u1)) * math.Cos(2*math.Pi*u2)
}

const patience = 5 * time.Second

var t0 = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

// ---------------------------------------------------------------------------
// helpers

type rngCase struct {
	Name  string `json:"name"`
	Input struct {
		Seed uint64 `json:"seed"`
		Seq  uint64 `json:"seq"`
	} `json:"input"`
	Output struct {
		U32     []uint32  `json:"u32"`
		Uniform []float64 `json:"uniform"`
		Normal  []float64 `json:"normal"`
	} `json:"output"`
}

// golden loads course/fixtures/parity/rng.json (spec/pcg32.md vectors).
func golden(t *testing.T) []rngCase {
	t.Helper()
	dir := os.Getenv("TINYLLM_FIXTURES")
	if dir == "" {
		t.Fatal("TINYLLM_FIXTURES is not set (ol check sets it)")
	}
	b, err := os.ReadFile(filepath.Join(dir, "parity", "rng.json"))
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Cases []rngCase `json:"cases"`
	}
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	if len(f.Cases) == 0 {
		t.Fatal("no cases in parity/rng.json")
	}
	return f.Cases
}

// exactQuantile is the nearest-rank definition the histogram approximates:
// the ceil(q * n)-th smallest value (rank clamped to [1, n]).
func exactQuantile(sorted []int64, q float64) int64 {
	n := len(sorted)
	rank := int(math.Ceil(q*float64(n) - 1e-9))
	if rank < 1 {
		rank = 1
	}
	if rank > n {
		rank = n
	}
	return sorted[rank-1]
}

// within waits for a value on ch or fails after patience.
func within[T any](t *testing.T, ch <-chan T, what string) T {
	t.Helper()
	select {
	case v := <-ch:
		return v
	case <-time.After(patience):
		t.Fatalf("timed out after %v waiting for %s", patience, what)
		var zero T
		return zero
	}
}

// waitWaiters returns once the fake clock has n pending timers, or fails.
func waitWaiters(t *testing.T, f *clock.Fake, n int) {
	t.Helper()
	done := make(chan struct{})
	go func() { f.BlockUntil(n); close(done) }()
	within(t, done, fmt.Sprintf("%d pending timer(s): the runner should be waiting on Clock.After for its next arrival", n))
}

// sse writes the events of one streamed chat completion with n tokens.
func sse(w http.ResponseWriter, n int, usage bool) {
	w.Header().Set("Content-Type", "text/event-stream")
	fl := http.NewResponseController(w)
	io.WriteString(w, `data: {"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`+"\n\n")
	for i := 0; i < n; i++ {
		fin := "null"
		if i == n-1 {
			fin = `"length"`
		}
		fmt.Fprintf(w, `data: {"choices":[{"index":0,"delta":{"content":"t%d"},"finish_reason":%s}]}`+"\n\n", i, fin)
		fl.Flush()
	}
	if usage {
		fmt.Fprintf(w, `data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":%d,"total_tokens":%d}}`+"\n\n", n, n+3)
	}
	io.WriteString(w, "data: [DONE]\n\n")
	fl.Flush()
}

// ---------------------------------------------------------------------------
// histogram

func TestHistogramHandExample(t *testing.T) {
	// WHY: the chapter's worked example. 1000 has e = 9 (512 <= 1000 < 1024),
	//      shift = 2, top bits 1000 >> 2 = 250, so it shares the bucket
	//      [1000, 1003] with 1002; 1005 lands in [1004, 1007]; 2000 in
	//      [2000, 2007]. The median of {1000, 1002, 1005, 2000} has rank 2,
	//      reported as its bucket's upper bound 1003; p100 is clamped to the
	//      true max 2000, not 2007.
	// KIND: unit
	// CATCHES: m01, m02, m03
	// CHAPTER: load.01 section 3
	h := loadgen.NewHistogram()
	for _, v := range []int64{1000, 1002, 1005, 2000} {
		h.Record(v)
	}
	want := []loadgen.Bucket{{1000, 1003, 2}, {1004, 1007, 1}, {2000, 2007, 1}}
	got := h.Buckets()
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("Buckets() = %v, want %v", got, want)
	}
	for _, c := range []struct {
		q    float64
		want int64
	}{{0.25, 1003}, {0.5, 1003}, {0.75, 1007}, {1.0, 2000}} {
		if got := h.Quantile(c.q); got != c.want {
			t.Errorf("Quantile(%v) = %d, want %d", c.q, got, c.want)
		}
	}
	if h.Count() != 4 || h.Min() != 1000 || h.Max() != 2000 || h.Mean() != 1251.75 {
		t.Fatalf("count/min/max/mean = %d/%d/%d/%v, want 4/1000/2000/1251.75", h.Count(), h.Min(), h.Max(), h.Mean())
	}
}

func TestQuantilesWithinOnePercentOfExactSort(t *testing.T) {
	// WHY: the report promises at most 1% relative error. Latencies span six
	//      decades (1 us to 10 s), so the test draws log-uniform values and
	//      compares every percentile the report prints with the exact
	//      nearest-rank value from a sort: never below it, at most 1/128 of
	//      it above (the bucket width).
	// KIND: property
	// CATCHES: s01, m04
	// CHAPTER: load.01 section 2
	r := newTestRand(1, 2)
	for trial := 0; trial < 20; trial++ {
		h := loadgen.NewHistogram()
		n := 1 + r.IntN(3000)
		vals := make([]int64, n)
		for i := range vals {
			vals[i] = int64(math.Exp(math.Log(1e3) + r.Float64()*(math.Log(1e10)-math.Log(1e3))))
			h.Record(vals[i])
		}
		sort.Slice(vals, func(a, b int) bool { return vals[a] < vals[b] })
		for _, q := range []float64{0, 0.01, 0.5, 0.9, 0.95, 0.99, 0.999, 1} {
			got, exact := h.Quantile(q), exactQuantile(vals, q)
			if got < exact || float64(got-exact) > float64(exact)/128 {
				t.Fatalf("trial %d, n %d: Quantile(%v) = %d, exact %d (allowed [exact, exact*(1+1/128)])", trial, n, q, got, exact)
			}
		}
	}
}

func TestQuantileRankRounding(t *testing.T) {
	// WHY: in float64, 0.55 * 100 = 55.00000000000001 and 0.07 * 100 =
	//      7.000000000000001, so a plain ceil makes the p55 of 1..100 the
	//      56th value. The rank must forgive that rounding. Values below 128
	//      have a bucket each, so the answers are exact.
	// KIND: boundary
	// CATCHES: s02
	// CHAPTER: load.01 section 5, Pitfalls
	h := loadgen.NewHistogram()
	for v := int64(1); v <= 100; v++ {
		h.Record(v)
	}
	for _, c := range []struct {
		q    float64
		want int64
	}{{0, 1}, {0.01, 1}, {0.07, 7}, {0.5, 50}, {0.55, 55}, {0.9, 90}, {0.95, 95}, {0.99, 99}, {1, 100}} {
		if got := h.Quantile(c.q); got != c.want {
			t.Errorf("Quantile(%v) of 1..100 = %d, want %d", c.q, got, c.want)
		}
	}
}

func TestHistogramEdgeValues(t *testing.T) {
	// WHY: an empty histogram reports 0 everywhere (no panic, no NaN mean);
	//      a negative duration (a clock step backwards) is recorded as 0;
	//      the largest int64 still has a bucket.
	// KIND: boundary
	// CATCHES: m05
	// CHAPTER: load.01 section 4
	h := loadgen.NewHistogram()
	if h.Quantile(0.5) != 0 || h.Mean() != 0 || h.Count() != 0 || len(h.Buckets()) != 0 {
		t.Fatalf("empty histogram: Quantile %d, Mean %v, Count %d, %d buckets", h.Quantile(0.5), h.Mean(), h.Count(), len(h.Buckets()))
	}
	h.Record(-5)
	h.Record(math.MaxInt64)
	if h.Min() != 0 || h.Quantile(0) != 0 {
		t.Fatalf("a negative value must be recorded as 0: Min %d, Quantile(0) %d", h.Min(), h.Quantile(0))
	}
	if h.Quantile(1) != math.MaxInt64 {
		t.Fatalf("Quantile(1) = %d, want MaxInt64", h.Quantile(1))
	}
}

func TestMergeEqualsRecordingEverything(t *testing.T) {
	// WHY: the runner keeps one histogram per worker and merges them; a merge
	//      that drops a bucket, the extremes, or the sum changes every number
	//      in the report. Merging must equal recording all values into one.
	// KIND: property
	// CATCHES: m06, m07
	// CHAPTER: load.01 section 4
	r := newTestRand(3, 4)
	a, b, all := loadgen.NewHistogram(), loadgen.NewHistogram(), loadgen.NewHistogram()
	for i := 0; i < 2000; i++ {
		v := int64(r.ExpFloat64() * 5e7)
		all.Record(v)
		if i%3 == 0 {
			a.Record(v)
		} else {
			b.Record(v)
		}
	}
	b.Record(1 << 40) // a's max must come from b
	all.Record(1 << 40)
	a.Merge(b)
	a.Merge(loadgen.NewHistogram()) // merging an empty histogram changes nothing
	if fmt.Sprint(a.Buckets()) != fmt.Sprint(all.Buckets()) {
		t.Fatal("merged buckets differ from recording everything into one histogram")
	}
	if a.Count() != all.Count() || a.Min() != all.Min() || a.Max() != all.Max() || math.Abs(a.Mean()-all.Mean()) > 1e-6*all.Mean() {
		t.Fatalf("merged count/min/max/mean %d/%d/%d/%v, want %d/%d/%d/%v", a.Count(), a.Min(), a.Max(), a.Mean(), all.Count(), all.Min(), all.Max(), all.Mean())
	}
	empty := loadgen.NewHistogram()
	empty.Merge(all)
	if empty.Min() != all.Min() || empty.Quantile(0.5) != all.Quantile(0.5) {
		t.Fatalf("merging into an empty histogram: min %d, p50 %d; want %d, %d", empty.Min(), empty.Quantile(0.5), all.Min(), all.Quantile(0.5))
	}
}

// ---------------------------------------------------------------------------
// PCG32

func TestPCG32HandExample(t *testing.T) {
	// WHY: spec/pcg32.md works the first output of pcg32(0) by hand: after
	//      seeding the state is 0x9AE4F7499BA72696, xs = 0x5C9A3E14, rot =
	//      19, and the output is rotr32(xs, 19) = 0x47C28B93.
	// KIND: unit
	// CATCHES: s03, m08
	// CHAPTER: load.01 section 3
	r := rng.Seeded(0)
	if st, inc := r.State(); st != 0x9AE4F7499BA72696 || inc != 109 {
		t.Fatalf("after seeding: state %#x inc %d, want 0x9AE4F7499BA72696 and 109", st, inc)
	}
	if got := r.Uint32(); got != 0x47C28B93 {
		t.Fatalf("first output %#x, want 0x47C28B93", got)
	}
	r42 := rng.Seeded(42) // O'Neill's pcg32-demo, first line
	for i, want := range []uint32{0xa15c02b7, 0x7b47f409, 0xba1d3330, 0x83d2f293, 0xbfa4784b, 0xcbed606e} {
		if got := r42.Uint32(); got != want {
			t.Fatalf("pcg32(42) output %d = %#x, want %#x", i, got, want)
		}
	}
}

func TestPCG32MatchesGoldenVectors(t *testing.T) {
	// WHY: parity/rng: the same seed must give the same stream in Python, C,
	//      Rust, and Go, bit for bit (spec/pcg32.md). Integers and uniforms
	//      are exact; normals go through the platform's log, sin, and cos and
	//      are compared within 4 ulp.
	// KIND: conformance
	// CATCHES: s04, m09
	// CHAPTER: load.01 section 2
	for _, c := range golden(t) {
		r := rng.New(c.Input.Seed, c.Input.Seq)
		for i, want := range c.Output.U32 {
			if got := r.Uint32(); got != want {
				t.Fatalf("%s: u32[%d] = %d, want %d", c.Name, i, got, want)
			}
		}
		r = rng.New(c.Input.Seed, c.Input.Seq)
		for i, want := range c.Output.Uniform {
			if got := r.Float64(); got != want {
				t.Fatalf("%s: uniform[%d] = %v, want %v (bit for bit)", c.Name, i, got, want)
			}
		}
		r = rng.New(c.Input.Seed, c.Input.Seq)
		for i, want := range c.Output.Normal {
			got := r.Normal()
			ulp := math.Abs(math.Nextafter(want, math.Inf(1)) - want)
			if math.Abs(got-want) > 4*ulp {
				t.Fatalf("%s: normal[%d] = %v, want %v (4 ulp)", c.Name, i, got, want)
			}
		}
	}
}

func TestBelowRejectsTheBiasedZone(t *testing.T) {
	// WHY: x mod n favors small values unless draws below t = (2^32 - n) mod
	//      n are thrown away. With n = 3 * 2^30 a third of all draws fall in
	//      the zone, so a missing rejection shows up within a few draws. The
	//      expected values are computed here from the golden u32 stream and
	//      the spec's rule.
	// KIND: unit
	// CATCHES: s05
	// CHAPTER: load.01 section 5, Pitfalls
	c := golden(t)[0]
	for _, n := range []uint64{10, 3 << 30, 1 << 32, 1} {
		thr := ((1 << 32) - n) % n
		var want []uint32
		for _, x := range c.Output.U32 {
			if uint64(x) >= thr {
				want = append(want, uint32(uint64(x)%n))
			}
			if len(want) == 40 {
				break
			}
		}
		r := rng.New(c.Input.Seed, c.Input.Seq)
		for i, w := range want {
			if got := r.Below(n); got != w {
				t.Fatalf("Below(%d) draw %d = %d, want %d", n, i, got, w)
			}
		}
	}
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("Below(0) must panic: no value lies in [0, 0)")
			}
		}()
		rng.Seeded(1).Below(0)
	}()
}

func TestShuffleIsFisherYatesFromTheEnd(t *testing.T) {
	// WHY: spec/pcg32.md fixes the swap order (i from n-1 down to 1, j =
	//      Below(i+1)); any other order is a fine shuffle but a different
	//      permutation, and load.02's permutation test would stop matching
	//      across languages.
	// KIND: unit
	// CATCHES: s06
	// CHAPTER: load.01 section 2
	c := golden(t)[1]
	want := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	u := c.Output.U32
	k := 0
	for i := 9; i >= 1; i-- {
		n := uint64(i + 1)
		thr := ((1 << 32) - n) % n
		for uint64(u[k]) < thr {
			k++
		}
		j := int(uint64(u[k]) % n)
		k++
		want[i], want[j] = want[j], want[i]
	}
	got := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	rng.New(c.Input.Seed, c.Input.Seq).Shuffle(len(got), func(i, j int) { got[i], got[j] = got[j], got[i] })
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("Shuffle = %v, want %v", got, want)
	}
}

func TestChildSeedAndStream(t *testing.T) {
	// WHY: sub-streams keep purposes independent: stream(seed, p) is
	//      pcg32_srandom_r(mix64(seed + p * 0x9E3779B97F4A7C15), p). The mix
	//      is written out here from the spec; mix64(0) = 0 is its fixed
	//      point.
	// KIND: unit
	// CATCHES: m10
	// CHAPTER: load.01 section 2
	mix := func(z uint64) uint64 {
		z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
		z = (z ^ (z >> 27)) * 0x94D049BB133111EB
		return z ^ (z >> 31)
	}
	if rng.Mix64(0) != 0 {
		t.Fatalf("Mix64(0) = %#x, want 0", rng.Mix64(0))
	}
	for _, seed := range []uint64{0, 1, 1 << 63, 12345} {
		for p := rng.PurposeInit; p <= rng.PurposeMutation; p++ {
			cs := mix(seed + p*0x9E3779B97F4A7C15)
			if got := rng.ChildSeed(seed, p); got != cs {
				t.Fatalf("ChildSeed(%d, %d) = %#x, want %#x", seed, p, got, cs)
			}
			a, b := rng.Stream(seed, p), rng.New(cs, p)
			for i := 0; i < 4; i++ {
				if x, y := a.Uint32(), b.Uint32(); x != y {
					t.Fatalf("Stream(%d, %d) output %d = %d, want %d", seed, p, i, x, y)
				}
			}
		}
	}
}

// ---------------------------------------------------------------------------
// schedules

func TestPoissonGapsByInverseCDF(t *testing.T) {
	// WHY: one uniform per gap, gap = -ln(1 - u) / rate seconds. Drawing u
	//      from the golden uniforms makes every gap an exact number; ln(u)
	//      instead of ln(1 - u) has the same distribution but other gaps, so
	//      seeded runs would stop matching.
	// KIND: unit
	// CATCHES: s07
	// CHAPTER: load.01 section 2
	c := golden(t)[0]
	const rate = 20.0
	s := loadgen.Poisson(rate, rng.New(c.Input.Seed, c.Input.Seq))
	if s.Name() != "poisson" || s.Rate() != rate {
		t.Fatalf("Name/Rate = %q/%v", s.Name(), s.Rate())
	}
	for i, u := range c.Output.Uniform[:32] {
		want := time.Duration(-math.Log(1-u) / rate * float64(time.Second))
		if got := s.Next(); got != want {
			t.Fatalf("gap %d = %v, want %v (u = %v)", i, got, want, u)
		}
	}
}

func TestPoissonRateAndShape(t *testing.T) {
	// WHY: the offered rate is what the report claims. Over 20000 gaps at 50
	//      rps the mean gap is 20 ms within 2%, and the share of gaps below
	//      the mean is 1 - 1/e = 0.632 within 0.01: an exponential, not a
	//      uniform jitter around the mean.
	// KIND: statistical
	// CATCHES: m11
	// CHAPTER: load.01 section 2
	s := loadgen.Poisson(50, rng.Stream(7, rng.PurposeSample))
	const n = 20000
	var sum time.Duration
	below := 0
	for i := 0; i < n; i++ {
		g := s.Next()
		sum += g
		if g < 20*time.Millisecond {
			below++
		}
	}
	mean := sum.Seconds() / n
	if math.Abs(mean-0.02) > 0.02*0.02 {
		t.Fatalf("mean gap %v s, want 0.02 within 2%%", mean)
	}
	if frac := float64(below) / n; math.Abs(frac-(1-1/math.E)) > 0.01 {
		t.Fatalf("share of gaps below the mean %v, want 0.632 within 0.01", frac)
	}
}

func TestConstantAndBurstGaps(t *testing.T) {
	// WHY: constant: every gap 1/rate, the first one too. burst: size
	//      arrivals at once, then a pause of `every`; its rate is size /
	//      every. An off-by-one in the burst counter changes the burst size.
	// KIND: unit
	// CATCHES: s08, m12
	// CHAPTER: load.01 section 4
	c := loadgen.Constant(4)
	for i := 0; i < 3; i++ {
		if g := c.Next(); g != 250*time.Millisecond {
			t.Fatalf("constant gap %d = %v, want 250ms", i, g)
		}
	}
	b := loadgen.Burst(3, time.Second)
	var got []time.Duration
	for i := 0; i < 7; i++ {
		got = append(got, b.Next())
	}
	want := []time.Duration{0, 0, 0, time.Second, 0, 0, time.Second}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("burst gaps %v, want %v", got, want)
	}
	if b.Rate() != 3 || b.Name() != "burst" || c.Name() != "constant" || c.Rate() != 4 {
		t.Fatalf("names/rates: %q %v %q %v", b.Name(), b.Rate(), c.Name(), c.Rate())
	}
}

// ---------------------------------------------------------------------------
// SSE reader and measurement

func TestReadStreamStampsEachEvent(t *testing.T) {
	// WHY: TTFT and ITL are only as good as the moment each event is
	//      stamped: once per data event, as it is read. The role-only first
	//      chunk carries no token, ": ping" comments and the final usage
	//      chunk are not content, and [DONE] ends the stream.
	// KIND: unit
	// CATCHES: s09, m13
	// CHAPTER: load.01 section 4
	body := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`, "",
		`data: {"choices":[{"index":0,"delta":{"content":"Once"},"finish_reason":null}]}`, "",
		": ping", "",
		`data: {"choices":[{"index":0,"delta":{"content":" upon"},"finish_reason":null}]}`, "",
		`data: {"choices":[{"index":0,"delta":{"content":" a"},"finish_reason":"length"}]}`, "",
		`data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":3,"total_tokens":7}}`, "",
		"data: [DONE]", "", "",
	}, "\n")
	calls := 0
	now := func() time.Time { calls++; return t0.Add(time.Duration(calls) * 10 * time.Millisecond) }
	evs, err := loadgen.ReadStream(strings.NewReader(body), now)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 6 {
		t.Fatalf("now() called %d times, want 6 (once per data event)", calls)
	}
	type kv struct {
		k  loadgen.EventKind
		ms int
		n  int
	}
	var got []kv
	for _, e := range evs {
		got = append(got, kv{e.Kind, int(e.At.Sub(t0) / time.Millisecond), e.Tokens})
	}
	want := []kv{{loadgen.Content, 20, 1}, {loadgen.Content, 30, 1}, {loadgen.Content, 40, 1}, {loadgen.Usage, 50, 3}, {loadgen.Done, 60, 0}}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("events %v, want %v", got, want)
	}
}

func TestReadStreamReportsBrokenStreams(t *testing.T) {
	// WHY: a stream cut before [DONE], an SSE error event, or an error
	//      object in a chunk is a FAILED request; counting it as a short
	//      success hides outages from error_rate (DESIGN 2.6: a failure
	//      after the first byte is an SSE error event, never a silent end).
	// KIND: fault
	// CATCHES: s10
	// CHAPTER: load.01 section 5, Pitfalls
	now := func() time.Time { return t0 }
	cut := "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"}}]}\n\n"
	for name, body := range map[string]string{
		"cut before [DONE]": cut,
		"event: error":      cut + "event: error\ndata: {\"message\":\"engine lost\"}\n\n",
		"error member":      cut + "data: {\"error\":{\"message\":\"kv pool exhausted\",\"type\":\"server_error\"}}\n\n",
	} {
		if _, err := loadgen.ReadStream(strings.NewReader(body), now); err == nil {
			t.Errorf("%s: ReadStream returned no error", name)
		}
	}
}

func TestMeasureHandExample(t *testing.T) {
	// WHY: the chapter's worked request, intended at t = 0: content at 120,
	//      150, and 190 ms, usage 3, [DONE] at 200 ms. TTFT 120 ms, ITL 30
	//      and 40 ms, E2E 200 ms, TPOT (200 - 120) / (3 - 1) = 40 ms.
	// KIND: unit
	// CATCHES: s11, m14
	// CHAPTER: load.01 section 3
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }
	s := loadgen.Measure(t0, []loadgen.Event{
		{At: at(120), Kind: loadgen.Content, Tokens: 1},
		{At: at(150), Kind: loadgen.Content, Tokens: 1},
		{At: at(190), Kind: loadgen.Content, Tokens: 1},
		{At: at(195), Kind: loadgen.Usage, Tokens: 3},
		{At: at(200), Kind: loadgen.Done},
	})
	tp, ok := s.TPOT()
	if s.Err != nil || s.TTFT != 120*time.Millisecond || s.E2E != 200*time.Millisecond || s.Tokens != 3 || !ok || tp != 40*time.Millisecond {
		t.Fatalf("sample %+v, TPOT %v %v; want TTFT 120ms, E2E 200ms, 3 tokens, TPOT 40ms", s, tp, ok)
	}
	if fmt.Sprint(s.ITL) != fmt.Sprint([]time.Duration{30 * time.Millisecond, 40 * time.Millisecond}) {
		t.Fatalf("ITL %v, want [30ms 40ms]", s.ITL)
	}
}

func TestMeasurePrefersUsageAndRejectsEmpty(t *testing.T) {
	// WHY: one chunk may carry several tokens, so the usage count wins over
	//      the chunk count when the server sends it; a stream with no content
	//      at all is an error, not a 0-token success with TTFT 0. TPOT needs
	//      two tokens.
	// KIND: boundary
	// CATCHES: m15
	// CHAPTER: load.01 section 4
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }
	s := loadgen.Measure(t0, []loadgen.Event{
		{At: at(10), Kind: loadgen.Content, Tokens: 1},
		{At: at(20), Kind: loadgen.Content, Tokens: 1},
		{At: at(21), Kind: loadgen.Usage, Tokens: 5},
		{At: at(30), Kind: loadgen.Done},
	})
	if s.Tokens != 5 {
		t.Fatalf("tokens %d, want 5 from usage", s.Tokens)
	}
	if tp, _ := s.TPOT(); tp != 5*time.Millisecond {
		t.Fatalf("TPOT %v, want (30 - 10) / 4 = 5ms", tp)
	}
	if e := loadgen.Measure(t0, []loadgen.Event{{At: at(5), Kind: loadgen.Done}}); e.Err == nil {
		t.Fatal("no content before [DONE] must be an error")
	}
	one := loadgen.Measure(t0, []loadgen.Event{{At: at(5), Kind: loadgen.Content, Tokens: 1}, {At: at(6), Kind: loadgen.Done}})
	if _, ok := one.TPOT(); ok {
		t.Fatal("TPOT of a 1-token answer is undefined (ok must be false)")
	}
}

// ---------------------------------------------------------------------------
// runner

func TestNoCoordinatedOmission(t *testing.T) {
	// WHY: the reason the generator is open loop. The fake engine stalls
	//      every request until t = 1 s. At 10 rps all five requests must
	//      still be SENT on schedule (t = 100, 200, ..., 500 ms) while none
	//      has finished, and each must be charged from its INTENDED start:
	//      TTFT = 1000 - 100 (i + 1) ms. A closed-loop or late-stamping
	//      generator reports one slow request and four fast ones.
	// KIND: fault
	// CATCHES: s12, s13, m16
	// CHAPTER: load.01 section 5, Pitfalls
	f := clock.NewFake(t0)
	release := make(chan struct{})
	arrived := make(chan struct{}, 16)
	var done atomic.Int32
	eng := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		arrived <- struct{}{}
		<-release
		sse(w, 1, false)
		done.Add(1)
	}))
	t.Cleanup(eng.Close)
	var once sync.Once
	t.Cleanup(func() { once.Do(func() { close(release) }) })

	type res struct {
		r   *loadgen.Report
		err error
	}
	out := make(chan res, 1)
	go func() {
		r, err := loadgen.Run(context.Background(), loadgen.Config{
			Target: eng.URL, Model: "m", Prompts: []string{"hi"}, MaxTokens: 1,
			Schedule: loadgen.Constant(10), MaxRequests: 5, Clock: f, RunID: "co",
		})
		out <- res{r, err}
	}()
	for i := 0; i < 5; i++ {
		waitWaiters(t, f, 1)
		f.Advance(100 * time.Millisecond)
		within(t, arrived, fmt.Sprintf("request %d to be sent at t = %d ms although the engine is stalled", i, 100*(i+1)))
	}
	if n := done.Load(); n != 0 {
		t.Fatalf("%d requests finished before the stall ended", n)
	}
	f.Advance(500 * time.Millisecond) // t = 1 s
	once.Do(func() { close(release) })
	got := within(t, out, "Run to return")
	if got.err != nil {
		t.Fatal(got.err)
	}
	r := got.r
	if r.Requests != 5 || r.Errors != 0 {
		t.Fatalf("requests %d, errors %d; want 5, 0", r.Requests, r.Errors)
	}
	// TTFTs are 900, 800, 700, 600, 500 ms: mean 700 exactly, p90 and p99
	// the max 900 exactly, p50 700 up to the bucket width (1/128).
	q := r.TTFTms
	if q.Mean != 700 || q.P99 != 900 || q.P90 != 900 || q.P50 < 700 || q.P50 > 700*(1+1.0/128) {
		t.Fatalf("ttft_ms %+v, want mean 700, p90 = p99 = 900, p50 in [700, 705.5] (charged from the intended start)", q)
	}
}

func TestRunCountsErrorsAndRecordsOnlySuccesses(t *testing.T) {
	// WHY: a 429 or 500 is a failed request: it counts in requests, errors,
	//      and error_rate, and never in a latency histogram (a fast error
	//      would make p50 look great). The usage chunk gives the token count.
	// KIND: unit
	// CATCHES: s14, m17
	// CHAPTER: load.01 section 4
	f := clock.NewFake(t0)
	var n atomic.Int32
	eng := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		if n.Add(1)%2 == 0 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusTooManyRequests)
			io.WriteString(w, `{"error":{"message":"queue full","type":"rate_limit_error","param":null,"code":"rate_limit_exceeded"}}`)
			return
		}
		sse(w, 4, true)
	}))
	t.Cleanup(eng.Close)
	out := make(chan *loadgen.Report, 1)
	go func() {
		r, err := loadgen.Run(context.Background(), loadgen.Config{
			Target: eng.URL, Model: "m", Prompts: []string{"a", "b"}, MaxTokens: 4,
			Schedule: loadgen.Constant(5), MaxRequests: 4, Clock: f, RunID: "errs",
		})
		if err != nil {
			t.Error(err)
		}
		out <- r
	}()
	for i := 0; i < 4; i++ {
		waitWaiters(t, f, 1)
		f.Advance(200 * time.Millisecond)
	}
	r := within(t, out, "Run to return")
	if r == nil {
		t.FailNow()
	}
	if r.Requests != 4 || r.Errors != 2 || r.ErrorRate != 0.5 {
		t.Fatalf("requests %d errors %d error_rate %v; want 4, 2, 0.5", r.Requests, r.Errors, r.ErrorRate)
	}
	count := 0.0
	for _, b := range r.Histogram["e2e_ms"] {
		count += b[1]
	}
	if count != 2 {
		t.Fatalf("e2e_ms histogram holds %v requests, want the 2 successes", count)
	}
	if r.Mode != "constant" || r.RateRPS != 5 {
		t.Fatalf("mode %q rate %v, want constant 5", r.Mode, r.RateRPS)
	}
	if r.TokensPerS != 8.0/0.8 {
		t.Fatalf("tokens_per_s %v, want 8 tokens / 0.8 s = 10", r.TokensPerS)
	}
}

func TestDurationEndsArrivals(t *testing.T) {
	// WHY: a 1 s run at 4 rps sends at 250, 500, and 750 ms; the arrival due
	//      at exactly 1 s is outside [start, start + duration) and is never
	//      sent, so back-to-back runs do not double count their boundary.
	// KIND: boundary
	// CATCHES: m18
	// CHAPTER: load.01 section 4
	f := clock.NewFake(t0)
	var n atomic.Int32
	eng := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		n.Add(1)
		sse(w, 2, false)
	}))
	t.Cleanup(eng.Close)
	out := make(chan *loadgen.Report, 1)
	go func() {
		r, _ := loadgen.Run(context.Background(), loadgen.Config{
			Target: eng.URL, Prompts: []string{"x"}, MaxTokens: 2,
			Schedule: loadgen.Constant(4), Duration: time.Second, Clock: f,
		})
		out <- r
	}()
	for i := 0; i < 3; i++ {
		waitWaiters(t, f, 1)
		f.Advance(250 * time.Millisecond)
	}
	r := within(t, out, "Run to return after the third arrival")
	if r == nil || r.Requests != 3 || n.Load() != 3 {
		t.Fatalf("report %+v, engine saw %d; want 3 requests", r, n.Load())
	}
	if r.DurationS != 1 {
		t.Fatalf("duration_s %v, want 1", r.DurationS)
	}
}

func TestReportGoodputAndShape(t *testing.T) {
	// WHY: goodput is the share of ALL requests (failures included) that met
	//      every SLO bound; the JSON must carry exactly the keys of
	//      formats/loadgen-report.schema.json (required ones always), with
	//      histograms as ascending [upper_ms, count] pairs.
	// KIND: conformance
	// CATCHES: s15, m19
	// CHAPTER: load.01 section 4
	ms := func(x int) time.Duration { return time.Duration(x) * time.Millisecond }
	samples := []loadgen.Sample{
		{Start: t0, TTFT: ms(100), E2E: ms(300), ITL: []time.Duration{ms(50)}, Tokens: 3},
		{Start: t0, TTFT: ms(700), E2E: ms(900), ITL: []time.Duration{ms(50)}, Tokens: 3},
		{Start: t0, TTFT: ms(200), E2E: ms(1200), ITL: []time.Duration{ms(500)}, Tokens: 3},
		{Start: t0, Err: fmt.Errorf("HTTP 503")},
	}
	cfg := loadgen.Config{Target: "http://x", Schedule: loadgen.Constant(2), RunID: "r1",
		SLO: map[string]float64{"ttft_ms_p95": 500, "tpot_ms_p95": 300}}
	r := loadgen.BuildReport(cfg, samples, 2*time.Second)
	// Sample 1 is good; 2 breaks TTFT; 3 breaks TPOT ((1200-200)/2 = 500); 4 failed.
	if r.Goodput != 0.25 || r.ErrorRate != 0.25 || r.Requests != 4 || r.Errors != 1 {
		t.Fatalf("goodput %v error_rate %v requests %d errors %d; want 0.25, 0.25, 4, 1", r.Goodput, r.ErrorRate, r.Requests, r.Errors)
	}
	if r.TokensPerS != 4.5 {
		t.Fatalf("tokens_per_s %v, want 9 / 2 s = 4.5", r.TokensPerS)
	}
	b, _ := json.Marshal(r)
	var m map[string]json.RawMessage
	json.Unmarshal(b, &m)
	required := []string{"run_id", "target", "mode", "rate_rps", "duration_s", "requests", "errors", "ttft_ms", "tpot_ms", "itl_ms", "e2e_ms", "error_rate", "goodput", "tokens_per_s", "histogram"}
	allowed := map[string]bool{"model": true, "concurrency": true, "seed": true, "slo": true}
	for _, k := range required {
		if _, ok := m[k]; !ok {
			t.Errorf("report lacks required key %q", k)
		}
		allowed[k] = true
	}
	for k := range m {
		if !allowed[k] {
			t.Errorf("report has key %q, which the schema does not allow", k)
		}
	}
	var q map[string]float64
	json.Unmarshal(m["ttft_ms"], &q)
	for _, k := range []string{"p50", "p90", "p95", "p99", "mean"} {
		if _, ok := q[k]; !ok {
			t.Errorf("ttft_ms lacks %q", k)
		}
	}
	h := r.Histogram["ttft_ms"]
	if len(h) != 3 || !(h[0][0] < h[1][0] && h[1][0] < h[2][0]) || h[0][1]+h[1][1]+h[2][1] != 3 {
		t.Fatalf("ttft_ms histogram %v: want 3 ascending [upper_ms, count] rows holding 3 requests", h)
	}
	if h[0][0] < 100 || h[0][0] > 100*(1+1.0/128) {
		t.Fatalf("first bucket upper bound %v ms, want within 1/128 above 100", h[0][0])
	}
}
