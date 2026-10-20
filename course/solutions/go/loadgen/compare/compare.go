// Package compare is the load generator's regression gate (load.02): it
// compares one metric of two loadgen reports (base and head) and fails the
// head only when it is worse by more than a budget AND the difference is
// unlikely to be noise, judged by a one-sided two-sample permutation test
// (M07.5, re-implemented in Go) over the samples the reports' histograms
// hold.
//
// Contract: course/contracts/formats/loadgen-report.schema.json (the input).
// Chapter: content/courses/ai-engineering/23-model-serving/12-run-comparison-and-regression-gate.md.
// `{loadgen} compare base.json head.json --metric ttft_p95
// --max-regress 5%` exits 1 when Verdict.Regressed; dep.05's CI perf job and
// drill ops.07 (git bisect run) call it.
package compare

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"

	"tinyllm/ds/rng"
	"tinyllm/loadgen"
)

// Stat names the statistic of a metric: a nearest-rank percentile or the
// mean.
type Stat struct {
	Name string  // "p50", "p95", ..., or "mean"
	Q    float64 // the percentile as a fraction; unused for the mean
	Mean bool
}

// Options configures the gate.
type Options struct {
	// Metric is "<metric>_<stat>": metric in ttft, tpot, itl, e2e and stat
	// in p50, p90, p95, p99, mean (ttft_p95), or "error_rate".
	Metric string
	// MaxRegress is the relative budget: 0.05 lets the head be 5% worse.
	MaxRegress float64
	// Alpha is the significance level of the permutation test (default 0.05).
	Alpha float64
	// Permutations is the number of random relabelings (default 1000).
	Permutations int
	// Seed seeds rng.Stream(Seed, rng.PurposeShuffle), so a verdict is
	// reproducible.
	Seed uint64
}

// Verdict is the gate's answer.
type Verdict struct {
	Metric    string
	Base      float64 // the statistic on the base samples
	Head      float64 // ... and on the head samples
	Delta     float64 // relative change, (Head - Base) / Base, in the "worse" direction
	PValue    float64 // one-sided: P(a relabeling is at least this much worse)
	Regressed bool    // Delta > MaxRegress and PValue < Alpha
}

// ParseRegress reads a budget: "5%" or "0.05" both give 0.05.
func ParseRegress(s string) (float64, error) {
	s = strings.TrimSpace(s)
	pct := strings.HasSuffix(s, "%")
	v, err := strconv.ParseFloat(strings.TrimSuffix(s, "%"), 64)
	if err != nil || v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, fmt.Errorf("compare: bad regression budget %q", s)
	}
	if pct {
		v /= 100
	}
	return v, nil
}

// ParseMetric splits "ttft_p95" into the report metric "ttft_ms" and its
// Stat; "error_rate" gives ("error_rate", mean).
func ParseMetric(m string) (string, Stat, error) {
	if m == "error_rate" {
		return "error_rate", Stat{Name: "mean", Mean: true}, nil
	}
	i := strings.LastIndex(m, "_")
	if i < 0 {
		return "", Stat{}, fmt.Errorf("compare: metric %q is not <metric>_<stat>", m)
	}
	base, st := m[:i], m[i+1:]
	switch base {
	case "ttft", "tpot", "itl", "e2e":
	default:
		return "", Stat{}, fmt.Errorf("compare: unknown metric %q (ttft, tpot, itl, e2e, error_rate)", base)
	}
	if st == "mean" {
		return base + "_ms", Stat{Name: st, Mean: true}, nil
	}
	if !strings.HasPrefix(st, "p") {
		return "", Stat{}, fmt.Errorf("compare: unknown statistic %q (p50, p90, p95, p99, mean)", st)
	}
	p, err := strconv.ParseFloat(st[1:], 64)
	if err != nil || p <= 0 || p >= 100 {
		return "", Stat{}, fmt.Errorf("compare: unknown statistic %q", st)
	}
	return base + "_ms", Stat{Name: st, Q: p / 100}, nil
}

// Samples expands a report's metric into one value per recorded sample:
// each histogram bucket [upper_ms, count] gives count copies of upper_ms;
// error_rate gives one 1 per failed request and one 0 per other request.
func Samples(r *loadgen.Report, metric string) ([]float64, error) {
	if metric == "error_rate" {
		if r.Errors > r.Requests || r.Errors < 0 {
			return nil, fmt.Errorf("compare: report %s has %d errors in %d requests", r.RunID, r.Errors, r.Requests)
		}
		out := make([]float64, r.Requests)
		for i := 0; i < r.Errors; i++ {
			out[i] = 1
		}
		return out, nil
	}
	rows, ok := r.Histogram[metric]
	if !ok {
		return nil, fmt.Errorf("compare: report %s has no %s histogram", r.RunID, metric)
	}
	var out []float64
	for _, b := range rows {
		n := int(b[1])
		if n < 0 || float64(n) != b[1] {
			return nil, fmt.Errorf("compare: report %s: bad count %v in %s", r.RunID, b[1], metric)
		}
		for i := 0; i < n; i++ {
			out = append(out, b[0])
		}
	}
	return out, nil
}

// Value is the statistic of xs (not modified). The percentile is the
// nearest-rank one, the ceil(q * n)-th smallest, rank clamped to [1, n]
// with the same rounding allowance as loadgen.Histogram.Quantile. NaN for
// an empty slice.
func (s Stat) Value(xs []float64) float64 {
	if len(xs) == 0 {
		return math.NaN()
	}
	if s.Mean {
		sum := 0.0
		for _, x := range xs {
			sum += x
		}
		return sum / float64(len(xs))
	}
	rank := int(math.Ceil(s.Q*float64(len(xs)) - 1e-9))
	if rank < 1 {
		rank = 1
	}
	if rank > len(xs) {
		rank = len(xs)
	}
	tmp := append([]float64(nil), xs...)
	return nth(tmp, rank-1)
}

// nth is the k-th smallest of xs (0-based), by quickselect; xs is permuted.
func nth(xs []float64, k int) float64 {
	lo, hi := 0, len(xs)-1
	for lo < hi {
		mid := lo + (hi-lo)/2
		// median of three as the pivot keeps sorted input linear
		if xs[mid] < xs[lo] {
			xs[mid], xs[lo] = xs[lo], xs[mid]
		}
		if xs[hi] < xs[lo] {
			xs[hi], xs[lo] = xs[lo], xs[hi]
		}
		if xs[hi] < xs[mid] {
			xs[hi], xs[mid] = xs[mid], xs[hi]
		}
		pivot := xs[mid]
		i, j := lo, hi
		for i <= j {
			for xs[i] < pivot {
				i++
			}
			for xs[j] > pivot {
				j--
			}
			if i <= j {
				xs[i], xs[j] = xs[j], xs[i]
				i++
				j--
			}
		}
		switch {
		case k <= j:
			hi = j
		case k >= i:
			lo = i
		default:
			return xs[k]
		}
	}
	return xs[k]
}

// PermutationTest is the one-sided p-value that head's statistic exceeds
// base's by at least the observed amount when the labels do not matter:
// pool both samples; n times, shuffle the pool with r (Fisher-Yates,
// load.01), call the first len(base) values "base" and the rest "head", and
// count relabelings with stat(head') - stat(base') >= observed. p = (1 +
// count) / (1 + n): the observed labeling counts as one of the
// relabelings, so p is never 0.
func PermutationTest(base, head []float64, s Stat, n int, r *rng.PCG32) float64 {
	obs := s.Value(head) - s.Value(base)
	pool := make([]float64, 0, len(base)+len(head))
	pool = append(append(pool, base...), head...)
	nb := len(base)
	count := 0
	for i := 0; i < n; i++ {
		r.Shuffle(len(pool), func(a, b int) { pool[a], pool[b] = pool[b], pool[a] })
		if s.Value(pool[nb:])-s.Value(pool[:nb]) >= obs {
			count++
		}
	}
	return float64(1+count) / float64(1+n)
}

// Gate compares two sets of samples of one metric (higher is worse).
func Gate(base, head []float64, s Stat, o Options) (Verdict, error) {
	if len(base) == 0 || len(head) == 0 {
		return Verdict{}, errors.New("compare: both runs need samples of the metric")
	}
	alpha, n := o.Alpha, o.Permutations
	if alpha == 0 {
		alpha = 0.05
	}
	if n == 0 {
		n = 1000
	}
	v := Verdict{Metric: o.Metric, Base: s.Value(base), Head: s.Value(head)}
	switch {
	case v.Base > 0:
		v.Delta = (v.Head - v.Base) / v.Base
	case v.Head > v.Base:
		v.Delta = math.Inf(1)
	}
	v.PValue = PermutationTest(base, head, s, n, rng.Stream(o.Seed, rng.PurposeShuffle))
	v.Regressed = v.Delta > o.MaxRegress && v.PValue < alpha
	return v, nil
}

// Compare gates head against base on o.Metric.
func Compare(base, head *loadgen.Report, o Options) (Verdict, error) {
	metric, st, err := ParseMetric(o.Metric)
	if err != nil {
		return Verdict{}, err
	}
	a, err := Samples(base, metric)
	if err != nil {
		return Verdict{}, err
	}
	b, err := Samples(head, metric)
	if err != nil {
		return Verdict{}, err
	}
	return Gate(a, b, st, o)
}
