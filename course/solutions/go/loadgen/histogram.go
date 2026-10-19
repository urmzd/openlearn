package loadgen

import (
	"math"
	"math/bits"
)

// SubBits sets the histogram's resolution: every power-of-two range
// [2^e, 2^(e+1)) with e >= SubBits is cut into 2^SubBits equal buckets, so a
// bucket is at most 1/2^SubBits = 1/128 (0.78%) of its lower bound wide.
// Values below 2^SubBits get one bucket each (exact).
const SubBits = 7

const subCount = 1 << SubBits // 128

// nBuckets covers every non-negative int64: exponents SubBits..62 give
// (62 - SubBits + 1) groups of subCount after the exact range.
const nBuckets = (62 - SubBits + 2) * subCount

// Histogram is a log-linear (HDR-style) histogram of non-negative int64
// values, nanoseconds in the load generator. Record and Quantile are O(1)
// and O(buckets); memory is fixed; Merge adds two histograms exactly. Not
// safe for concurrent use.
type Histogram struct {
	counts   [nBuckets]uint64
	n        uint64
	sum      float64
	min, max int64
}

// Bucket is one non-empty bucket: values in [Lower, Upper], Count of them.
type Bucket struct {
	Lower, Upper int64
	Count        uint64
}

// NewHistogram returns an empty histogram.
func NewHistogram() *Histogram {
	return &Histogram{}
}

// bucketOf maps v >= 0 to its bucket index: v itself below 128; else, with
// e = floor(log2 v) and shift = e - 7, the top 8 bits sub = v >> shift (in
// [128, 256)) select index (shift + 1) * 128 + (sub - 128).
func bucketOf(v int64) int {
	if v < subCount {
		return int(v)
	}
	e := bits.Len64(uint64(v)) - 1
	shift := e - SubBits
	sub := v >> uint(shift)
	return (shift+1)*subCount + int(sub-subCount)
}

// bounds is the inclusive value range of bucket i.
func bounds(i int) (lo, hi int64) {
	if i < subCount {
		return int64(i), int64(i)
	}
	shift := i/subCount - 1
	sub := int64(i%subCount + subCount)
	lo = sub << uint(shift)
	hi = lo + (int64(1) << uint(shift)) - 1
	return lo, hi
}

// Record adds one value; a negative value is recorded as 0.
func (h *Histogram) Record(ns int64) {
	if ns < 0 {
		ns = 0
	}
	if h.n == 0 || ns < h.min {
		h.min = ns
	}
	if h.n == 0 || ns > h.max {
		h.max = ns
	}
	h.counts[bucketOf(ns)]++
	h.n++
	h.sum += float64(ns)
}

// Count is the number of recorded values.
func (h *Histogram) Count() uint64 {
	return h.n
}

// Mean is the exact mean of the recorded values (0 when empty).
func (h *Histogram) Mean() float64 {
	if h.n == 0 {
		return 0
	}
	return h.sum / float64(h.n)
}

// Min and Max are the exact extremes (0 when empty).
func (h *Histogram) Min() int64 {
	return h.min
}

func (h *Histogram) Max() int64 {
	return h.max
}

// Quantile is the nearest-rank q-quantile, 0 <= q <= 1: with rank =
// ceil(q * n) clamped to [1, n], the bucket holding the rank-th smallest
// value, reported as its upper bound clamped to Max. It is never below the
// exact value and at most 1/128 of it above. 0 when empty.
func (h *Histogram) Quantile(q float64) int64 {
	if h.n == 0 {
		return 0
	}
	if q < 0 {
		q = 0
	}
	if q > 1 {
		q = 1
	}
	// ceil(q * n), forgiving the rounding of q * n: 0.55 * 100 is
	// 55.00000000000001 in float64, and its rank is 55, not 56.
	rank := uint64(math.Ceil(q*float64(h.n) - 1e-9))
	if rank < 1 {
		rank = 1
	}
	if rank > h.n {
		rank = h.n
	}
	var cum uint64
	for i, c := range h.counts {
		cum += c
		if cum >= rank {
			_, hi := bounds(i)
			if hi > h.max {
				hi = h.max
			}
			return hi
		}
	}
	return h.max
}

// Merge adds every value of o to h; o is unchanged.
func (h *Histogram) Merge(o *Histogram) {
	if o.n == 0 {
		return
	}
	if h.n == 0 || o.min < h.min {
		h.min = o.min
	}
	if h.n == 0 || o.max > h.max {
		h.max = o.max
	}
	for i, c := range o.counts {
		h.counts[i] += c
	}
	h.n += o.n
	h.sum += o.sum
}

// Buckets lists the non-empty buckets in ascending order.
func (h *Histogram) Buckets() []Bucket {
	var out []Bucket
	for i, c := range h.counts {
		if c == 0 {
			continue
		}
		lo, hi := bounds(i)
		out = append(out, Bucket{Lower: lo, Upper: hi, Count: c})
	}
	return out
}
