// Package eval is the agent SDK's evaluation runner (ag.09): it runs a
// subject (a model, an agent, a durable agent run) over the cases of a
// suite, scores every output with every scorer, and aggregates each score
// into a mean with a bootstrap confidence interval. Scorers live in
// eval/scorers (ag.10), the LLM judge in eval/judge (ag.11), and paired
// A/B experiments in eval/experiment (ag.12).
//
// Three rules the package exists for:
//
//   - a scorer that fails (an error, a panic, NaN) is excluded from that
//     score's mean and counted, never averaged in as 0
//   - the result is the same whatever order concurrent work finishes in:
//     rows are kept in case order, then sample order
//   - the report is formats/eval-result.schema.json (results.jsonl rows and
//     summary.json), so the release gate (dur.12) and the A/B runner read
//     one format
//
// Chapter: content/courses/ai-platform-engineering/09-llm-evaluation/01-eval-runner.md.
package eval

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sync"
	"time"
)

// Timing is when the subject's output arrived. Durations are offsets from
// Start; zero TTFT means no token arrived.
type Timing struct {
	Start      time.Time
	TTFT       time.Duration   // time to the first token
	Total      time.Duration   // time until the output was complete (end of the stream or run)
	TokenTimes []time.Duration // arrival of every streamed text piece, in order
}

// Observation is one case, and after the subject ran, its output. Sample
// numbers repeated runs of the same case from 0.
type Observation struct {
	ID          string
	Sample      int
	Input       json.RawMessage
	Output      json.RawMessage
	GroundTruth json.RawMessage
	Annotations map[string]json.RawMessage
	Timing      Timing
	Tokens      int    // completion tokens
	TraceID     string // 32 hex, when the subject traced the run
}

// Score is one scorer's verdict on one observation. Error non-empty means
// the scorer could not score it.
type Score struct {
	Name   string
	Value  float64
	Reason string
	Error  string
}

// Scorer scores observations.
type Scorer interface {
	Name() string
	Score(ctx context.Context, o Observation) (Score, error)
}

// Subject runs one case: it reads o.Input (and o.Sample) and fills Output,
// Timing, Tokens, TraceID, and any Annotations its scorers need.
type Subject func(ctx context.Context, o *Observation) error

// Row is one scored observation. Scores are in scorer order; Errored
// reports whether score i failed.
type Row struct {
	Observation
	Subject      string
	Scores       []Score
	SubjectError string
}

// Errored reports whether score i of the row failed.
func (r Row) Errored(i int) bool { return r.Scores[i].Error != "" }

// Stat aggregates one score over a suite: the mean of the values that did
// not fail, its bootstrap CI, how many values went in, how many failed.
type Stat struct {
	Mean    float64 `json:"mean"`
	CILow   float64 `json:"ci_low"`
	CIHigh  float64 `json:"ci_high"`
	N       int     `json:"n"`
	Errored int     `json:"errored"`
}

// SuiteResult is one run of one subject over one suite.
type SuiteResult struct {
	Suite         string
	Subject       string
	Rows          []Row           // case order, then sample order
	Scorers       []string        // scorer names, in the order given
	Metrics       map[string]Stat // by scorer name
	SubjectErrors int
	NBoot         int
	Alpha         float64
	Seed          uint64
}

type config struct {
	concurrency int
	samples     int
	nBoot       int
	alpha       float64
	seed        uint64
	subjectName string
	subject     Subject
	now         func() time.Time
}

// Option configures Run.
type Option func(*config)

// WithSubject runs s on every case (and sample) before scoring; without it,
// the observations are scored as given (their Output already filled).
func WithSubject(name string, s Subject) Option {
	return func(c *config) { c.subjectName, c.subject = name, s }
}

// WithConcurrency runs up to n cases at once (default 4).
func WithConcurrency(n int) Option { return func(c *config) { c.concurrency = n } }

// WithSamples runs every case n times (default 1), as samples 0..n-1.
func WithSamples(n int) Option { return func(c *config) { c.samples = n } }

// WithBootstrap sets the resample count (default 2000) and alpha (default
// 0.05, a 95% interval).
func WithBootstrap(nBoot int, alpha float64) Option {
	return func(c *config) { c.nBoot, c.alpha = nBoot, alpha }
}

// WithSeed seeds the bootstrap (default 0).
func WithSeed(seed uint64) Option { return func(c *config) { c.seed = seed } }

// WithClock replaces time.Now for timing (tests).
func WithClock(now func() time.Time) Option { return func(c *config) { c.now = now } }

// ErrSuite wraps every problem with the suite or the scorers given to Run.
var ErrSuite = errors.New("eval: bad suite")

// scoreOne calls s, turning an error, a panic, an empty or mismatched
// name, or a non-finite value into a Score with Error set.
func scoreOne(ctx context.Context, s Scorer, o Observation) (out Score) {
	name := s.Name()
	defer func() {
		if p := recover(); p != nil {
			out = Score{Name: name, Error: fmt.Sprintf("scorer panicked: %v", p)}
		}
	}()
	sc, err := s.Score(ctx, o)
	sc.Name = name
	switch {
	case err != nil:
		return Score{Name: name, Error: err.Error()}
	case sc.Error != "":
		return Score{Name: name, Error: sc.Error, Reason: sc.Reason}
	case math.IsNaN(sc.Value) || math.IsInf(sc.Value, 0):
		return Score{Name: name, Error: fmt.Sprintf("scorer returned %v", sc.Value)}
	}
	return sc
}

// Run evaluates obs with scorers s. Case ids must be unique and non-empty,
// scorer names unique and non-empty. With a subject, a subject error fails
// every score of that row (counted as errored, with the subject's error)
// and is counted in SubjectErrors. Cancelling ctx stops the run with
// ctx's error.
func Run(ctx context.Context, name string, obs []Observation, s []Scorer, opts ...Option) (*SuiteResult, error) {
	cfg := config{concurrency: 4, samples: 1, nBoot: 2000, alpha: 0.05, now: time.Now}
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.concurrency < 1 {
		cfg.concurrency = 1
	}
	if cfg.samples < 1 {
		cfg.samples = 1
	}
	res := &SuiteResult{Suite: name, Subject: cfg.subjectName, Metrics: map[string]Stat{}, NBoot: cfg.nBoot, Alpha: cfg.alpha, Seed: cfg.seed}
	seen := map[string]bool{}
	for _, sc := range s {
		n := sc.Name()
		if n == "" || seen[n] {
			return nil, fmt.Errorf("%w: scorer name %q is empty or repeated", ErrSuite, n)
		}
		seen[n] = true
		res.Scorers = append(res.Scorers, n)
	}
	ids := map[string]bool{}
	for _, o := range obs {
		if o.ID == "" || ids[o.ID] {
			return nil, fmt.Errorf("%w: case id %q is empty or repeated", ErrSuite, o.ID)
		}
		ids[o.ID] = true
	}

	res.Rows = make([]Row, len(obs)*cfg.samples)
	sem := make(chan struct{}, cfg.concurrency)
	var wg sync.WaitGroup
	for i := range obs {
		for k := 0; k < cfg.samples; k++ {
			slot := i*cfg.samples + k
			o := obs[i]
			o.Sample = k
			o.Annotations = make(map[string]json.RawMessage, len(obs[i].Annotations))
			for key, v := range obs[i].Annotations {
				o.Annotations[key] = v
			}
			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
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
					subjErr = func() (err error) {
						defer func() {
							if p := recover(); p != nil {
								err = fmt.Errorf("subject panicked: %v", p)
							}
						}()
						return cfg.subject(ctx, &o)
					}()
					if o.Timing.Start.IsZero() {
						o.Timing.Start = start
					}
					if o.Timing.Total == 0 {
						o.Timing.Total = cfg.now().Sub(start)
					}
				}
				row.Observation = o
				for _, sc := range s {
					if subjErr != nil {
						row.Scores = append(row.Scores, Score{Name: sc.Name(), Error: "subject failed: " + subjErr.Error()})
						continue
					}
					row.Scores = append(row.Scores, scoreOne(ctx, sc, o))
				}
				if subjErr != nil {
					row.SubjectError = subjErr.Error()
				}
				res.Rows[slot] = row
			}()
		}
	}
	wg.Wait()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	for _, r := range res.Rows {
		if r.SubjectError != "" {
			res.SubjectErrors++
		}
	}
	for j, n := range res.Scorers {
		res.Metrics[n] = aggregate(res.Rows, j, cfg.samples, cfg.nBoot, cfg.alpha, cfg.seed)
	}
	return res, nil
}

// aggregate is score j over rows: the mean of the values that did not fail
// and a cluster-bootstrap CI that resamples cases (all samples of a case
// together), counting the failed values.
func aggregate(rows []Row, j, samples, nBoot int, alpha float64, seed uint64) Stat {
	var st Stat
	var groups [][]float64
	var all []float64
	for i := 0; i < len(rows); i += samples {
		var g []float64
		for _, r := range rows[i : i+samples] {
			if r.Errored(j) {
				st.Errored++
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
		return st
	}
	st.Mean = Mean(all)
	st.CILow, st.CIHigh = BootstrapCI(groups, nBoot, alpha, BootstrapRNG(seed))
	return st
}
