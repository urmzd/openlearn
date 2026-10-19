package eval

// stats.go: the statistics every eval report carries (M07.4, re-implemented
// in Go): the mean and a percentile bootstrap confidence interval.

import (
	"math"
	"sort"

	"tinyllm/ds/rng"
)

// Mean is the arithmetic mean, summed in order (NaN for no values).
func Mean(xs []float64) float64 {
	if len(xs) == 0 {
		return math.NaN()
	}
	var s float64
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

// BootstrapRNG is the stream every bootstrap of a run draws from:
// rng.Stream(seed, rng.PurposeSample).
func BootstrapRNG(seed uint64) *rng.PCG32 {
	return rng.Stream(seed, rng.PurposeSample)
}

// PercentileBounds returns the indices of the lower and upper percentile
// bounds among nBoot sorted resample statistics: floor(nBoot * alpha / 2)
// and ceil(nBoot * (1 - alpha / 2)) - 1. For nBoot = 2000 and alpha = 0.05
// that is 50 and 1949.
func PercentileBounds(nBoot int, alpha float64) (lo, hi int) {
	lo = int(math.Floor(float64(nBoot) * alpha / 2))
	hi = int(math.Ceil(float64(nBoot)*(1-alpha/2))) - 1
	return min(max(lo, 0), nBoot-1), min(max(hi, 0), nBoot-1)
}

// BootstrapCI is the percentile bootstrap interval of the mean over
// groups. Each resample draws len(groups) groups with replacement (index
// r.Below(len(groups)), in order) and takes the mean of every value in the
// drawn groups; the nBoot resample means are sorted and the bounds are read
// at PercentileBounds. Resampling whole groups (all samples of one case)
// keeps correlated samples together. One group, or nBoot < 1, gives a
// zero-width interval at the mean.
func BootstrapCI(groups [][]float64, nBoot int, alpha float64, r *rng.PCG32) (lo, hi float64) {
	var all []float64
	for _, g := range groups {
		all = append(all, g...)
	}
	m := Mean(all)
	if len(groups) < 2 || nBoot < 1 {
		return m, m
	}
	means := make([]float64, nBoot)
	n := uint64(len(groups))
	for b := range means {
		var s float64
		var c int
		for i := 0; i < len(groups); i++ {
			g := groups[r.Below(n)]
			for _, x := range g {
				s += x
			}
			c += len(g)
		}
		means[b] = s / float64(c)
	}
	sort.Float64s(means)
	il, ih := PercentileBounds(nBoot, alpha)
	return means[il], means[ih]
}
