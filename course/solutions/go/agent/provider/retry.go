package provider

import (
	"context"
	"errors"
	"io"
	"math"
	"net"
	"net/http"
	"syscall"
	"time"

	"tinyllm/agent/types"
)

// RetryPolicy is when and how long to wait before trying again. Zero fields
// take the defaults in parentheses.
type RetryPolicy struct {
	MaxAttempts int           // attempts in total, the first included (4)
	Initial     time.Duration // the wait after the first failure (500 ms)
	Max         time.Duration // the cap on a computed wait (30 s)
	Multiplier  float64       // growth per failure (2)
	// Sleep waits d or until ctx ends; nil uses a timer. Tests pass a
	// recorder so no test sleeps.
	Sleep func(ctx context.Context, d time.Duration) error
}

func (p RetryPolicy) withDefaults() RetryPolicy {
	if p.MaxAttempts <= 0 {
		p.MaxAttempts = 4
	}
	if p.Initial <= 0 {
		p.Initial = 500 * time.Millisecond
	}
	if p.Max <= 0 {
		p.Max = 30 * time.Second
	}
	if p.Multiplier <= 0 {
		p.Multiplier = 2
	}
	if p.Sleep == nil {
		p.Sleep = sleepCtx
	}
	return p
}

func sleepCtx(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Delay is the wait after failure number n (n = 1 after the first attempt
// failed): min(Max, Initial * Multiplier^(n-1)). When err carries a
// Retry-After, the server's value is used instead.
func (p RetryPolicy) Delay(n int, err error) time.Duration {
	p = p.withDefaults()
	var ae *APIError
	if errors.As(err, &ae) && ae.RetryAfter > 0 {
		return ae.RetryAfter
	}
	d := float64(p.Initial) * math.Pow(p.Multiplier, float64(n-1))
	if d > float64(p.Max) {
		return p.Max
	}
	return time.Duration(d)
}

// Retryable reports whether trying again can help: overload and server
// errors (408, 429, 500, 502, 503, 504), a broken or refused connection, a
// stream cut before [DONE], and a server_error event. Client errors (400,
// 401, 403, 404, 422), a cancelled or expired ctx, and anything else are
// final.
func Retryable(err error) bool {
	if err == nil || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	var ae *APIError
	if errors.As(err, &ae) {
		switch ae.Status {
		case http.StatusRequestTimeout, http.StatusTooManyRequests, http.StatusInternalServerError,
			http.StatusBadGateway, http.StatusServiceUnavailable, http.StatusGatewayTimeout:
			return true
		}
		return false
	}
	var se *StreamError
	if errors.As(err, &se) {
		return se.Type == "server_error"
	}
	if errors.Is(err, ErrNoDone) || errors.Is(err, io.ErrUnexpectedEOF) ||
		errors.Is(err, syscall.ECONNRESET) || errors.Is(err, syscall.ECONNREFUSED) {
		return true
	}
	var ne net.Error
	return errors.As(err, &ne)
}

// WithRetry wraps p so a failure is retried only while nothing has been
// shown to the caller: a refused request, or a stream that breaks before its
// first content delta (text or a tool call). From the first content delta
// on, the stream is committed: a later failure is passed through as an
// ErrorDelta and never retried, because a second attempt would repeat or
// splice text the caller already has.
func WithRetry(p types.Provider, pol RetryPolicy) types.Provider {
	return &retrying{inner: p, pol: pol.withDefaults()}
}

type retrying struct {
	inner types.Provider
	pol   RetryPolicy
}

func (r *retrying) ChatStream(ctx context.Context, msgs []types.Message, tools []types.ToolDef, opts ...types.CallOption) (<-chan types.Delta, error) {
	for attempt := 1; ; attempt++ {
		ch, err := r.inner.ChatStream(ctx, msgs, tools, opts...)
		if err != nil {
			if !Retryable(err) || attempt >= r.pol.MaxAttempts {
				return nil, err
			}
			if serr := r.pol.Sleep(ctx, r.pol.Delay(attempt, err)); serr != nil {
				return nil, serr
			}
			continue
		}
		// Hold everything until the first content delta or the end.
		var held []types.Delta
		var failed error
		committed := false
		for d := range ch {
			held = append(held, d)
			if types.IsContent(d) {
				committed = true
				break
			}
			if e, ok := d.(types.ErrorDelta); ok {
				failed = e.Err
				break
			}
			if _, ok := d.(types.DoneDelta); ok {
				committed = true
				break
			}
		}
		if !committed && failed == nil {
			failed = ErrNoDone // closed without an end delta
			held = append(held, types.ErrorDelta{Err: failed})
		}
		if !committed && Retryable(failed) && attempt < r.pol.MaxAttempts {
			for range ch { // the inner stream is over; drain what is left
			}
			if serr := r.pol.Sleep(ctx, r.pol.Delay(attempt, failed)); serr != nil {
				return nil, serr
			}
			continue
		}
		out := make(chan types.Delta)
		go func() {
			defer close(out)
			for _, d := range held {
				select {
				case out <- d:
				case <-ctx.Done():
					go func() {
						for range ch {
						}
					}()
					return
				}
			}
			for d := range ch {
				select {
				case out <- d:
				case <-ctx.Done():
					go func() {
						for range ch {
						}
					}()
					return
				}
			}
		}()
		return out, nil
	}
}
