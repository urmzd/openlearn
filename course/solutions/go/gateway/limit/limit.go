// Package limit is the gateway's rate limiter (gw.03): per key, a requests
// per minute (RPM) and a tokens per minute (TPM) token bucket. A request
// reserves its estimated cost before it runs and settles the actual cost
// after, so unused tokens are refunded and overruns are charged.
//
// Contract: openapi/openai-subset.v1.yaml (429 rate_limit_exceeded with
// Retry-After; x-ratelimit-limit-requests, x-ratelimit-remaining-requests,
// x-ratelimit-limit-tokens, x-ratelimit-remaining-tokens,
// x-ratelimit-reset-tokens on every authenticated response). Chapter:
// content/courses/software-engineering/10-authorization/02-rate-limiting.md.
package limit

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"tinyllm/gateway/auth"
	"tinyllm/gateway/server"
)

// Cost is what one request uses: requests (1) and tokens (prompt plus
// completion).
type Cost struct {
	Requests, Tokens int
}

// Limits are one key's budgets per minute; 0 means unlimited.
type Limits struct {
	RPM, TPM int
}

// Status is a key's budget right after a decision, for the x-ratelimit-*
// headers. Remaining values are floored at 0.
type Status struct {
	LimitRequests, RemainingRequests int
	LimitTokens, RemainingTokens     int
	ResetTokens                      time.Duration // until the token bucket is full again
}

// RejectError is the answer when a bucket cannot cover the cost now.
type RejectError struct {
	RetryAfter time.Duration // until both buckets would cover the cost
	Status     Status
}

func (e *RejectError) Error() string {
	return fmt.Sprintf("rate limit exceeded: retry after %v", e.RetryAfter)
}

// Reservation is an admitted request's hold on its budget. Exactly one of
// Settle and Cancel takes effect; later calls do nothing.
type Reservation interface {
	// Settle charges the actual cost: unused reserved tokens go back to the
	// bucket, an overrun is taken from it (the bucket may go negative, which
	// delays the key's next requests).
	Settle(actual Cost)
	// Cancel gives the whole reservation back (the request never ran).
	Cancel()
	// Status is the key's budget right after the reservation.
	Status() Status
}

// bucket is one token bucket: capacity tokens, refilled continuously at
// capacity per minute.
type bucket struct {
	capacity float64
	level    float64
	last     time.Time
}

// refill adds the tokens earned since last, capped at capacity.
func (b *bucket) refill(now time.Time) {
	if dt := now.Sub(b.last); dt > 0 {
		b.level = math.Min(b.capacity, b.level+b.capacity*dt.Minutes())
	}
	b.last = now
}

// wait is how long until the bucket holds n tokens (0 if it does now).
func (b *bucket) wait(n float64) time.Duration {
	if b.level >= n {
		return 0
	}
	return time.Duration(math.Ceil((n - b.level) / b.capacity * float64(time.Minute)))
}

type keyState struct {
	req, tok bucket
	lim      Limits
}

// Limiter holds every key's buckets. Safe for concurrent use.
type Limiter struct {
	clock server.Clock
	mu    sync.Mutex
	keys  map[string]*keyState
}

// New returns a limiter reading time from clock (nil = the wall clock).
func New(clock server.Clock) *Limiter {
	if clock == nil {
		clock = server.WallClock
	}
	return &Limiter{clock: clock, keys: map[string]*keyState{}}
}

// state returns key's buckets, creating them full, and resizing them when
// the key's limits changed (the level keeps its fraction of the capacity).
func (l *Limiter) state(key string, lim Limits, now time.Time) *keyState {
	st, ok := l.keys[key]
	if !ok {
		st = &keyState{
			req: bucket{capacity: float64(lim.RPM), level: float64(lim.RPM), last: now},
			tok: bucket{capacity: float64(lim.TPM), level: float64(lim.TPM), last: now},
			lim: lim,
		}
		l.keys[key] = st
		return st
	}
	if st.lim != lim {
		resize := func(b *bucket, c int) {
			if b.capacity > 0 {
				b.level = b.level / b.capacity * float64(c)
			} else {
				b.level = float64(c)
			}
			b.capacity = float64(c)
		}
		resize(&st.req, lim.RPM)
		resize(&st.tok, lim.TPM)
		st.lim = lim
	}
	st.req.refill(now)
	st.tok.refill(now)
	return st
}

func (st *keyState) status() Status {
	floor := func(x float64) int {
		if x < 0 {
			return 0
		}
		return int(math.Floor(x))
	}
	s := Status{
		LimitRequests: st.lim.RPM, RemainingRequests: floor(st.req.level),
		LimitTokens: st.lim.TPM, RemainingTokens: floor(st.tok.level),
	}
	if st.lim.TPM > 0 {
		s.ResetTokens = st.tok.wait(st.tok.capacity)
	}
	return s
}

// Reserve admits a request of cost c for key under lim, or returns a
// *RejectError with the time until it would fit. A cost larger than a
// bucket's capacity can never fit and is rejected with the time to a full
// bucket. A zero limit is unlimited.
func (l *Limiter) Reserve(ctx context.Context, key string, lim Limits, c Cost) (Reservation, error) {
	now := l.clock.Now()
	l.mu.Lock()
	defer l.mu.Unlock()
	st := l.state(key, lim, now)
	var wait time.Duration
	if lim.RPM > 0 {
		wait = max(wait, st.req.wait(float64(min(c.Requests, lim.RPM))))
	}
	if lim.TPM > 0 {
		wait = max(wait, st.tok.wait(float64(min(c.Tokens, lim.TPM))))
	}
	if wait > 0 || (lim.TPM > 0 && c.Tokens > lim.TPM) || (lim.RPM > 0 && c.Requests > lim.RPM) {
		if wait == 0 {
			wait = st.tok.wait(st.tok.capacity)
		}
		return nil, &RejectError{RetryAfter: wait, Status: st.status()}
	}
	if lim.RPM > 0 {
		st.req.level -= float64(c.Requests)
	}
	if lim.TPM > 0 {
		st.tok.level -= float64(c.Tokens)
	}
	return &reservation{l: l, key: key, cost: c, status: st.status()}, nil
}

type reservation struct {
	l      *Limiter
	key    string
	cost   Cost
	status Status
	once   sync.Once
}

func (r *reservation) Status() Status {
	return r.status
}

// adjust returns dReq requests and dTok tokens to key's buckets (negative
// values take more), capped at capacity.
func (l *Limiter) adjust(key string, dReq, dTok int) {
	now := l.clock.Now()
	l.mu.Lock()
	defer l.mu.Unlock()
	st, ok := l.keys[key]
	if !ok {
		return
	}
	st.req.refill(now)
	st.tok.refill(now)
	if st.lim.RPM > 0 {
		st.req.level = math.Min(st.req.capacity, st.req.level+float64(dReq))
	}
	if st.lim.TPM > 0 {
		st.tok.level = math.Min(st.tok.capacity, st.tok.level+float64(dTok))
	}
}

func (r *reservation) Settle(actual Cost) {
	r.once.Do(func() {
		r.l.adjust(r.key, r.cost.Requests-actual.Requests, r.cost.Tokens-actual.Tokens)
	})
}

func (r *reservation) Cancel() {
	r.once.Do(func() { r.l.adjust(r.key, r.cost.Requests, r.cost.Tokens) })
}

// Counter counts a request's prompt tokens with the model's tokenizer.
type Counter interface {
	CountPrompt(ctx context.Context, model string, body []byte) (int, error)
}

// PromptText is the text whose tokens estimate a request's prompt: the
// completions `prompt`, or the chat messages' contents joined by "\n". The
// chat template adds a few tokens per message that this leaves out; Settle
// charges the exact count afterwards.
func PromptText(body []byte) string {
	var f struct {
		Prompt   any `json:"prompt"`
		Messages []struct {
			Content any `json:"content"`
		} `json:"messages"`
		Input any `json:"input"`
	}
	_ = json.Unmarshal(body, &f)
	flat := func(v any) string {
		switch x := v.(type) {
		case string:
			return x
		case []any:
			parts := make([]string, 0, len(x))
			for _, e := range x {
				if s, ok := e.(string); ok {
					parts = append(parts, s)
				}
			}
			return strings.Join(parts, "\n")
		}
		return ""
	}
	if f.Messages != nil {
		parts := make([]string, 0, len(f.Messages))
		for _, m := range f.Messages {
			parts = append(parts, flat(m.Content))
		}
		return strings.Join(parts, "\n")
	}
	if f.Prompt != nil {
		return flat(f.Prompt)
	}
	return flat(f.Input)
}

// TokenizeCounter counts with an engine's POST /v1/tokenize (engine tier).
type TokenizeCounter struct {
	BaseURL string // the engine, without /v1
	Client  *http.Client
}

func (t TokenizeCounter) CountPrompt(ctx context.Context, model string, body []byte) (int, error) {
	c := t.Client
	if c == nil {
		c = &http.Client{Timeout: 5 * time.Second}
	}
	in, _ := json.Marshal(map[string]any{"model": model, "text": PromptText(body)})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(t.BaseURL, "/")+"/v1/tokenize", bytes.NewReader(in))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("tokenize: status %d", resp.StatusCode)
	}
	var out struct {
		IDs []int `json:"ids"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return 0, fmt.Errorf("tokenize: %v", err)
	}
	return len(out.IDs), nil
}

// DefaultCompletionReserve is the completion tokens reserved for a request
// that sets no max_tokens (Settle charges the real count).
const DefaultCompletionReserve = 256

// EstimateCost is the reservation for a request: 1 request, and the prompt
// tokens (from counter; a counter failure falls back to bytes / 4, rounded
// up) plus max_tokens (or DefaultCompletionReserve).
func EstimateCost(ctx context.Context, counter Counter, req *server.Request) Cost {
	prompt := -1
	if counter != nil {
		if n, err := counter.CountPrompt(ctx, req.Model, req.Body); err == nil {
			prompt = n
		}
	}
	if prompt < 0 {
		prompt = (len(PromptText(req.Body)) + 3) / 4
	}
	completion := req.MaxTokens
	if completion <= 0 {
		completion = DefaultCompletionReserve
	}
	return Cost{Requests: 1, Tokens: prompt + completion}
}

// SetHeaders writes the x-ratelimit-* headers of s.
func SetHeaders(h http.Header, s Status) {
	h.Set("x-ratelimit-limit-requests", strconv.Itoa(s.LimitRequests))
	h.Set("x-ratelimit-remaining-requests", strconv.Itoa(s.RemainingRequests))
	h.Set("x-ratelimit-limit-tokens", strconv.Itoa(s.LimitTokens))
	h.Set("x-ratelimit-remaining-tokens", strconv.Itoa(s.RemainingTokens))
	h.Set("x-ratelimit-reset-tokens", s.ResetTokens.Round(time.Millisecond).String())
}

// RetryAfterSeconds is the Retry-After value for d: whole seconds, rounded
// up, at least 1.
func RetryAfterSeconds(d time.Duration) int {
	s := int(math.Ceil(d.Seconds()))
	if s < 1 {
		s = 1
	}
	return s
}

// Middleware is the ratelimit stage. It keys buckets by the principal's key
// id under the key's RPM and TPM (a request without a principal passes
// untouched), reserves the estimated cost, and answers 429
// rate_limit_exceeded with Retry-After when it does not fit. After the rest
// of the chain returns it settles with the usage the proxy recorded, or
// cancels when the request failed without usage (5xx).
func Middleware(l *Limiter, counter Counter) server.Middleware {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			p, ok := auth.PrincipalFrom(r.Context())
			if !ok {
				next.ServeHTTP(w, r)
				return
			}
			ex := server.ExchangeFrom(r.Context())
			cost := Cost{Requests: 1}
			if r.Method == http.MethodPost {
				req, err := ex.Request(r)
				if err != nil {
					server.WriteError(w, http.StatusBadRequest, "invalid_request_error", "", "", err.Error())
					return
				}
				cost = EstimateCost(r.Context(), counter, req)
			}
			res, err := l.Reserve(r.Context(), p.KeyID, Limits{RPM: p.RPM, TPM: p.TPM}, cost)
			var rej *RejectError
			if errors.As(err, &rej) {
				ex.SetAttr("tl.ratelimit.decision", "reject")
				SetHeaders(w.Header(), rej.Status)
				w.Header().Set("Retry-After", strconv.Itoa(RetryAfterSeconds(rej.RetryAfter)))
				server.WriteError(w, http.StatusTooManyRequests, "rate_limit_error", "rate_limit_exceeded", "",
					fmt.Sprintf("rate limit exceeded for key %s; retry after %ds", p.KeyID, RetryAfterSeconds(rej.RetryAfter)))
				return
			}
			if err != nil {
				server.WriteError(w, http.StatusInternalServerError, "server_error", "internal_error", "", err.Error())
				return
			}
			ex.SetAttr("tl.ratelimit.decision", "allow")
			SetHeaders(w.Header(), res.Status())
			settled := false
			defer func() {
				if !settled {
					res.Cancel() // a panic below: nothing ran to completion
				}
			}()
			next.ServeHTTP(w, r)
			st := ex.Snapshot()
			switch {
			case st.Usage.TotalTokens > 0 || st.Usage.PromptTokens+st.Usage.CompletionTokens > 0:
				tokens := st.Usage.TotalTokens
				if tokens == 0 {
					tokens = st.Usage.PromptTokens + st.Usage.CompletionTokens
				}
				res.Settle(Cost{Requests: 1, Tokens: tokens})
			case st.Status >= 500:
				res.Cancel()
			default:
				res.Settle(Cost{Requests: 1})
			}
			settled = true
		})
	}
}
