// Package activity is what activity code imports (dur.05): its idempotency
// key and attempt, heartbeats with resumable details, typed and
// non-retryable errors, and the retry backoff formula the server uses.
//
// Retries are at-least-once: an activity may run again after a crash at any
// point. Make its effect idempotent with IdempotencyKey, which is the same
// on every attempt.
package activity

import (
	"context"
	"errors"
	"math"
	"time"

	"tinyllm/durable/worker"

	durablev1 "openlearn.urmzd.com/tl/contracts/gen/tl/durable/v1"
)

// Defaults of a RetryPolicy field left at zero (durable.proto).
const (
	DefaultInitial  = time.Second
	DefaultBackoff  = 2.0
	DefaultInterval = time.Minute
)

// IdempotencyKey is "<workflow_id>/<activity_id>": stable across attempts,
// worker restarts, and server restarts. "" outside an activity.
func IdempotencyKey(ctx context.Context) string {
	inf, _ := worker.InfoFrom(ctx)
	return inf.IdempotencyKey
}

// Attempt is the 1-based attempt number; 0 outside an activity.
func Attempt(ctx context.Context) int {
	inf, _ := worker.InfoFrom(ctx)
	return inf.Attempt
}

// HeartbeatDetails are the details an earlier attempt last recorded (nil on
// a first attempt): where to resume from.
func HeartbeatDetails(ctx context.Context) []byte {
	inf, _ := worker.InfoFrom(ctx)
	return inf.HeartbeatDetails
}

// RecordHeartbeat reports progress and extends the attempt's lease. If the
// lease is lost or cancellation is requested, ctx is canceled; check
// ctx.Err() and stop.
func RecordHeartbeat(ctx context.Context, details []byte) {
	_ = worker.Heartbeat(ctx, details)
}

// Error is an activity failure with a type that RetryPolicy.non_retryable
// can name, and an explicit non-retryable flag.
type Error struct {
	Type    string
	Message string
	NoRetry bool
	Cause   error
}

func (e *Error) Error() string       { return e.Type + ": " + e.Message }
func (e *Error) Unwrap() error       { return e.Cause }
func (e *Error) FailureType() string { return e.Type }
func (e *Error) NonRetryable() bool  { return e.NoRetry }

// NewError is a retryable failure of the given type.
func NewError(typ, msg string) error {
	return &Error{Type: typ, Message: msg}
}

// NewNonRetryable fails the activity for good: the workflow sees
// ActivityTaskFailed at once, with no retry and no dead letter.
func NewNonRetryable(typ, msg string) error {
	return &Error{Type: typ, Message: msg, NoRetry: true}
}

// IsNonRetryable reports whether err (or an error it wraps) is non-retryable.
func IsNonRetryable(err error) bool {
	var nr interface{ NonRetryable() bool }
	return errors.As(err, &nr) && nr.NonRetryable()
}

// Backoff is the delay before attempt n+1 after attempt n (n >= 1) failed:
// min(initial * backoff^(n-1), max_interval), with the defaults for zero
// fields (a backoff below 1 counts as the default).
func Backoff(p *durablev1.RetryPolicy, n int) time.Duration {
	if n < 1 {
		n = 1
	}
	initial := time.Duration(p.GetInitialMs()) * time.Millisecond
	if initial <= 0 {
		initial = DefaultInitial
	}
	mult := p.GetBackoff()
	if mult < 1 {
		mult = DefaultBackoff
	}
	max := time.Duration(p.GetMaxIntervalMs()) * time.Millisecond
	if max <= 0 {
		max = DefaultInterval
	}
	d := float64(initial) * math.Pow(mult, float64(n-1))
	if d >= float64(max) {
		return max
	}
	return time.Duration(d)
}

// Jitter is full jitter: the delay is u * d for a uniform u in [0, 1), so
// retries of many failed tasks spread over [0, d) instead of colliding.
func Jitter(d time.Duration, u float64) time.Duration {
	if u < 0 {
		u = 0
	}
	if u >= 1 {
		u = math.Nextafter(1, 0)
	}
	return time.Duration(u * float64(d))
}
