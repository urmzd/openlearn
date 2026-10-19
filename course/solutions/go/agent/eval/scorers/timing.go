package scorers

// timing.go: the latencies a user feels, in milliseconds. A scorer that
// has no tokens to time returns an error: an answer that never streamed is
// not an instant answer.

import (
	"context"
	"errors"
	"time"

	"tinyllm/agent/eval"
)

// ErrNoTokens: the observation has too few streamed tokens to time.
var ErrNoTokens = errors.New("scorers: not enough streamed tokens to time")

func millis(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }

// TTFT is time to first token, ms ("ttft_ms").
func TTFT() eval.Scorer {
	return fn{"ttft_ms", func(_ context.Context, o eval.Observation) (eval.Score, error) {
		if len(o.Timing.TokenTimes) == 0 {
			return eval.Score{}, ErrNoTokens
		}
		return eval.Score{Value: millis(o.Timing.TokenTimes[0])}, nil
	}}
}

// TTLT is time to last token, ms ("ttlt_ms").
func TTLT() eval.Scorer {
	return fn{"ttlt_ms", func(_ context.Context, o eval.Observation) (eval.Score, error) {
		n := len(o.Timing.TokenTimes)
		if n == 0 {
			return eval.Score{}, ErrNoTokens
		}
		return eval.Score{Value: millis(o.Timing.TokenTimes[n-1])}, nil
	}}
}

// ITL is the mean inter-token latency, ms: (last - first) / (n - 1) over
// the n token arrival times, which equals the mean of the n - 1 gaps
// ("itl_ms"). It needs at least two tokens.
func ITL() eval.Scorer {
	return fn{"itl_ms", func(_ context.Context, o eval.Observation) (eval.Score, error) {
		t := o.Timing.TokenTimes
		if len(t) < 2 {
			return eval.Score{}, ErrNoTokens
		}
		return eval.Score{Value: millis(t[len(t)-1]-t[0]) / float64(len(t)-1)}, nil
	}}
}
