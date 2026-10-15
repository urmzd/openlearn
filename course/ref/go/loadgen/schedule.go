// Package loadgen is the system's open-loop load generator (load.01): it
// sends streamed chat requests on an arrival schedule that does not wait for
// the server, measures TTFT, TPOT, ITL, and end-to-end latency from the
// client's side into log-linear histograms, and writes the report of
// course/contracts/formats/loadgen-report.schema.json.
//
// Chapter: content/courses/build-an-llm/p10-serving/10-load-generator.md. load.02
// (loadgen/compare) compares two reports; the learner's {loadgen} main
// drives MS-L10, MS-gateway, MS-prod, and the drills.
package loadgen

import (
	"math"
	"time"

	"tinyllm/ds/rng"
)

// Schedule yields the gaps between consecutive arrivals. The runner sends
// request i at start + Next() + ... + Next() (i + 1 calls), whatever the
// server is doing: that is what makes the generator open loop.
type Schedule interface {
	// Next is the gap before the next arrival (>= 0).
	Next() time.Duration
	// Name is the report's "mode": poisson, constant, or burst.
	Name() string
	// Rate is the offered rate in requests per second (the report's rate_rps).
	Rate() float64
}

// poisson draws exponential gaps by inverse CDF.
type poisson struct {
	rate float64
	r    *rng.PCG32
}

// Poisson is a Poisson arrival process of `rate` requests per second: each
// gap is -ln(1 - u) / rate seconds with u = r.Float64() (inverse CDF of the
// exponential, M07.1), one uniform per gap. It panics for rate <= 0.
func Poisson(rate float64, r *rng.PCG32) Schedule {
	// SOLUTION-BEGIN load.01
	if !(rate > 0) {
		panic("loadgen: Poisson needs rate > 0")
	}
	return &poisson{rate: rate, r: r}
	// SOLUTION-END
}

func (p *poisson) Next() time.Duration {
	// SOLUTION-BEGIN load.01
	u := p.r.Float64()
	sec := -math.Log(1.0-u) / p.rate
	return time.Duration(sec * float64(time.Second))
	// SOLUTION-END
}

func (p *poisson) Name() string {
	// SOLUTION-BEGIN load.01
	return "poisson"
	// SOLUTION-END
}

func (p *poisson) Rate() float64 {
	// SOLUTION-BEGIN load.01
	return p.rate
	// SOLUTION-END
}

type constant struct{ rate float64 }

// Constant sends one request every 1/rate seconds; the first gap is 1/rate
// too. It panics for rate <= 0.
func Constant(rate float64) Schedule {
	// SOLUTION-BEGIN load.01
	if !(rate > 0) {
		panic("loadgen: Constant needs rate > 0")
	}
	return constant{rate: rate}
	// SOLUTION-END
}

func (c constant) Next() time.Duration {
	// SOLUTION-BEGIN load.01
	return time.Duration(float64(time.Second) / c.rate)
	// SOLUTION-END
}

func (c constant) Name() string {
	// SOLUTION-BEGIN load.01
	return "constant"
	// SOLUTION-END
}

func (c constant) Rate() float64 {
	// SOLUTION-BEGIN load.01
	return c.rate
	// SOLUTION-END
}

type burst struct {
	size  int
	every time.Duration
	i     int
}

// Burst sends `size` requests at once every `every`, starting at once: the
// gaps are 0 (size - 1 times), then every, then 0 (size - 1 times), and so
// on, with a first gap of 0. Its rate is size / every. It panics for size < 1
// or every <= 0.
func Burst(size int, every time.Duration) Schedule {
	// SOLUTION-BEGIN load.01
	if size < 1 || every <= 0 {
		panic("loadgen: Burst needs size >= 1 and every > 0")
	}
	return &burst{size: size, every: every}
	// SOLUTION-END
}

func (b *burst) Next() time.Duration {
	// SOLUTION-BEGIN load.01
	i := b.i
	b.i++
	if i > 0 && i%b.size == 0 {
		return b.every
	}
	return 0
	// SOLUTION-END
}

func (b *burst) Name() string {
	// SOLUTION-BEGIN load.01
	return "burst"
	// SOLUTION-END
}

func (b *burst) Rate() float64 {
	// SOLUTION-BEGIN load.01
	return float64(b.size) / b.every.Seconds()
	// SOLUTION-END
}
