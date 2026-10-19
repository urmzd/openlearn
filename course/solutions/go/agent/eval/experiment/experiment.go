// Package experiment runs paired evaluations and estimates score deltas.
package experiment

import (
	"context"
	"fmt"
	"math"

	"tinyllm/agent/eval"
	"tinyllm/ds/rng"
)

type Stat struct {
	Delta, CILow, CIHigh, PValue float64
	NPairs                       int
}
type Result struct {
	Base, Exp string
	Metrics   map[string]Stat
	NBoot     int
	Seed      uint64
}
type Option func(*config)
type config struct {
	nBoot       int
	alpha       float64
	seed        uint64
	concurrency int
}

func WithBootstrap(n int, alpha float64) Option {
	return func(c *config) { c.nBoot = n; c.alpha = alpha }
}
func WithSeed(seed uint64) Option {
	return func(c *config) { c.seed = seed }
}
func WithConcurrency(n int) Option {
	return func(c *config) { c.concurrency = n }
}

// RunExperiment executes both subjects against the same case ids, then
// bootstraps complete case deltas so pairing survives every resample.
func RunExperiment(ctx context.Context, obs []eval.Observation, base, exp eval.Subject, scorers []eval.Scorer, opts ...Option) (*Result, error) {
	c := config{nBoot: 2000, alpha: .05, concurrency: 4}
	for _, o := range opts {
		o(&c)
	}
	if c.nBoot < 1 || c.alpha <= 0 || c.alpha >= 1 || base == nil || exp == nil {
		return nil, fmt.Errorf("experiment: invalid configuration")
	}
	br, err := eval.Run(ctx, "base", obs, scorers, eval.WithSubject("base", base), eval.WithConcurrency(c.concurrency), eval.WithSeed(c.seed))
	if err != nil {
		return nil, err
	}
	er, err := eval.Run(ctx, "experiment", obs, scorers, eval.WithSubject("experiment", exp), eval.WithConcurrency(c.concurrency), eval.WithSeed(c.seed))
	if err != nil {
		return nil, err
	}
	if len(br.Rows) != len(er.Rows) {
		return nil, fmt.Errorf("experiment: arms produced different rows")
	}
	result := &Result{Base: "base", Exp: "experiment", Metrics: map[string]Stat{}, NBoot: c.nBoot, Seed: c.seed}
	for si, name := range br.Scorers {
		var diffs []float64
		for i := range br.Rows {
			b, e := br.Rows[i], er.Rows[i]
			if b.ID != e.ID || b.Sample != e.Sample {
				return nil, fmt.Errorf("experiment: unpaired row %d", i)
			}
			if b.Errored(si) || e.Errored(si) {
				continue
			}
			diffs = append(diffs, e.Scores[si].Value-b.Scores[si].Value)
		}
		if len(diffs) == 0 {
			continue
		}
		groups := make([][]float64, len(diffs))
		for i, d := range diffs {
			groups[i] = []float64{d}
		}
		lo, hi := eval.BootstrapCI(groups, c.nBoot, c.alpha, rng.Stream(c.seed, rng.PurposeSample))
		// Two-sided sign permutation test, with the same deterministic substream.
		r := rng.Stream(c.seed, rng.PurposeSample)
		observed := math.Abs(eval.Mean(diffs))
		extreme := 0
		for n := 0; n < c.nBoot; n++ {
			var s float64
			for _, d := range diffs {
				if r.Below(2) == 0 {
					d = -d
				}
				s += d
			}
			if math.Abs(s/float64(len(diffs))) >= observed {
				extreme++
			}
		}
		result.Metrics[name] = Stat{Delta: eval.Mean(diffs), CILow: lo, CIHigh: hi, PValue: float64(extreme+1) / float64(c.nBoot+1), NPairs: len(diffs)}
	}
	return result, nil
}
