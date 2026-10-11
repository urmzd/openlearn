// Course tests for gw.03: the RPM and TPM token buckets, reserve and settle,
// and the ratelimit stage (go/gateway/limit).
//
// Time is the course testkit's fake clock: refill is an exact function of
// Advance, so every remaining count and Retry-After below is a number from
// the chapter, not a range. The concurrency test runs under the race
// detector (ol check runs Go course tests with -race).
package gw_03

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"openlearn.urmzd.com/tl/testkit/clock"
	"tinyllm/gateway/auth"
	"tinyllm/gateway/limit"
	"tinyllm/gateway/server"
)

var t0 = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

func reserve(t *testing.T, l *limit.Limiter, key string, lim limit.Limits, c limit.Cost) limit.Reservation {
	t.Helper()
	r, err := l.Reserve(context.Background(), key, lim, c)
	if err != nil {
		t.Fatalf("Reserve(%s, %+v) rejected: %v", key, c, err)
	}
	return r
}

func rejected(t *testing.T, l *limit.Limiter, key string, lim limit.Limits, c limit.Cost) *limit.RejectError {
	t.Helper()
	_, err := l.Reserve(context.Background(), key, lim, c)
	var rej *limit.RejectError
	if !errors.As(err, &rej) {
		t.Fatalf("Reserve(%s, %+v) = %v, want a *RejectError", key, c, err)
	}
	return rej
}

func TestHandExample(t *testing.T) {
	// WHY: the worked example, step by step: RPM 3, TPM 1000; reserve 300,
	//      settle 120 (180 refunded), then 900 does not fit until 20 tokens
	//      refill at 1000 per minute, which takes 1.2 s: Retry-After 2.
	// KIND: unit
	// CATCHES: s02, s03, s06
	// CHAPTER: gw.03 section 3, worked example
	fc := clock.NewFake(t0)
	l := limit.New(fc)
	lim := limit.Limits{RPM: 3, TPM: 1000}
	r := reserve(t, l, "k", lim, limit.Cost{Requests: 1, Tokens: 300})
	st := r.Status()
	if st.RemainingRequests != 2 || st.RemainingTokens != 700 || st.LimitRequests != 3 || st.LimitTokens != 1000 {
		t.Fatalf("after reserving 300: %+v, want 2 requests and 700 tokens remaining", st)
	}
	if st.ResetTokens != 18*time.Second {
		t.Fatalf("reset-tokens %v, want 18s (300 tokens at 1000 per minute)", st.ResetTokens)
	}
	r.Settle(limit.Cost{Requests: 1, Tokens: 120})
	rej := rejected(t, l, "k", lim, limit.Cost{Requests: 1, Tokens: 900})
	if rej.RetryAfter != 1200*time.Millisecond || limit.RetryAfterSeconds(rej.RetryAfter) != 2 {
		t.Fatalf("RetryAfter %v (%d s), want 1.2s (Retry-After 2): 880 tokens, need 900", rej.RetryAfter, limit.RetryAfterSeconds(rej.RetryAfter))
	}
	if rej.Status.RemainingTokens != 880 {
		t.Fatalf("remaining tokens %d after the settle, want 880", rej.Status.RemainingTokens)
	}
	fc.Advance(1200 * time.Millisecond)
	r = reserve(t, l, "k", lim, limit.Cost{Requests: 1, Tokens: 900})
	if st := r.Status(); st.RemainingTokens != 0 || st.RemainingRequests != 1 {
		t.Fatalf("after 1.2 s: %+v, want 0 tokens and 1 request remaining", st)
	}
}

func TestBurstThenRefill(t *testing.T) {
	// WHY: a full bucket allows a burst of the whole minute's budget, then
	//      exactly one request per 60/RPM seconds; a long idle period never
	//      banks more than one minute's budget.
	// KIND: unit
	// CATCHES: s01
	// CHAPTER: gw.03 section 2.1
	fc := clock.NewFake(t0)
	l := limit.New(fc)
	lim := limit.Limits{RPM: 60}
	for i := 0; i < 60; i++ {
		reserve(t, l, "k", lim, limit.Cost{Requests: 1})
	}
	if rej := rejected(t, l, "k", lim, limit.Cost{Requests: 1}); rej.RetryAfter != time.Second {
		t.Fatalf("61st request: RetryAfter %v, want 1s", rej.RetryAfter)
	}
	fc.Advance(time.Second)
	reserve(t, l, "k", lim, limit.Cost{Requests: 1})
	rejected(t, l, "k", lim, limit.Cost{Requests: 1})
	fc.Advance(10 * time.Minute)
	for i := 0; i < 60; i++ {
		reserve(t, l, "k", lim, limit.Cost{Requests: 1})
	}
	rejected(t, l, "k", lim, limit.Cost{Requests: 1}) // 10 idle minutes still bank only 60
}

func TestSettleChargesOverrun(t *testing.T) {
	// WHY: a request that used more tokens than it reserved (no max_tokens,
	//      a long answer) is charged the difference; the bucket goes negative
	//      and the key waits it off.
	// KIND: unit
	// CATCHES: s04
	// CHAPTER: gw.03 section 2.2
	fc := clock.NewFake(t0)
	l := limit.New(fc)
	lim := limit.Limits{TPM: 600}
	r := reserve(t, l, "k", lim, limit.Cost{Requests: 1, Tokens: 100})
	r.Settle(limit.Cost{Requests: 1, Tokens: 700}) // 600 more than reserved: level -100
	rej := rejected(t, l, "k", lim, limit.Cost{Requests: 1, Tokens: 10})
	if rej.RetryAfter != 11*time.Second {
		t.Fatalf("RetryAfter %v, want 11s: from -100 to 10 tokens at 10 per second", rej.RetryAfter)
	}
}

func TestCancelAndSettleOnce(t *testing.T) {
	// WHY: Cancel returns the whole reservation (the request never ran); only
	//      the first of Settle and Cancel counts, so a deferred Cancel after a
	//      Settle cannot refund twice.
	// KIND: unit
	// CATCHES: s05
	// CHAPTER: gw.03 section 2.2
	fc := clock.NewFake(t0)
	l := limit.New(fc)
	lim := limit.Limits{RPM: 2, TPM: 100}
	r := reserve(t, l, "k", lim, limit.Cost{Requests: 1, Tokens: 60})
	r.Cancel()
	r.Cancel()
	r2 := reserve(t, l, "k", lim, limit.Cost{Requests: 1, Tokens: 100})
	if st := r2.Status(); st.RemainingRequests != 1 || st.RemainingTokens != 0 {
		t.Fatalf("after a cancel: %+v, want the full budget back before this reservation", st)
	}
	r2.Settle(limit.Cost{Requests: 1, Tokens: 100})
	r2.Cancel() // must not refund: the settle already counted
	rejected(t, l, "k", lim, limit.Cost{Requests: 1, Tokens: 1})
}

func TestZeroMeansUnlimited(t *testing.T) {
	// WHY: rpm = 0 and tpm = 0 mean unlimited (admin.v1.yaml), not "nothing".
	// KIND: boundary
	// CATCHES: s07
	// CHAPTER: gw.03 section 4
	l := limit.New(clock.NewFake(t0))
	for i := 0; i < 1000; i++ {
		reserve(t, l, "k", limit.Limits{}, limit.Cost{Requests: 1, Tokens: 1 << 20})
	}
	reserve(t, l, "k2", limit.Limits{RPM: 5}, limit.Cost{Requests: 1, Tokens: 1 << 20})
}

func TestOversizeCostRejected(t *testing.T) {
	// WHY: a request that needs more tokens than the key's TPM can never fit;
	//      admitting it because the bucket is full would let one request take
	//      more than a minute's budget.
	// KIND: boundary
	// CATCHES: s15
	// CHAPTER: gw.03 section 5, Pitfalls, item 5
	l := limit.New(clock.NewFake(t0))
	rejected(t, l, "k", limit.Limits{TPM: 1000}, limit.Cost{Requests: 1, Tokens: 1001})
	reserve(t, l, "k", limit.Limits{TPM: 1000}, limit.Cost{Requests: 1, Tokens: 1000})
}

func TestKeysAreIndependent(t *testing.T) {
	// WHY: one tenant's burst must not spend another tenant's budget (the
	//      noisy-neighbor drill, ops.09, depends on it).
	// KIND: unit
	// CHAPTER: gw.03 section 2.1
	l := limit.New(clock.NewFake(t0))
	lim := limit.Limits{RPM: 2}
	reserve(t, l, "a", lim, limit.Cost{Requests: 2})
	rejected(t, l, "a", lim, limit.Cost{Requests: 1})
	reserve(t, l, "b", lim, limit.Cost{Requests: 2})
}

func TestConcurrentReserve(t *testing.T) {
	// WHY: 1000 goroutines race for a budget of 100 requests: exactly 100 are
	//      admitted, and the race detector sees no unsynchronized access.
	// KIND: property
	// CATCHES: s08
	// CHAPTER: gw.03 section 5, Pitfalls, item 6
	l := limit.New(clock.NewFake(t0))
	var ok atomic.Int64
	var wg sync.WaitGroup
	start := make(chan struct{})
	for i := 0; i < 1000; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			if _, err := l.Reserve(context.Background(), "k", limit.Limits{RPM: 100, TPM: 100000}, limit.Cost{Requests: 1, Tokens: 10}); err == nil {
				ok.Add(1)
			}
		}()
	}
	close(start)
	wg.Wait()
	if ok.Load() != 100 {
		t.Fatalf("%d admitted, want exactly 100", ok.Load())
	}
}

func TestRetryAfterSeconds(t *testing.T) {
	// WHY: Retry-After is whole seconds; rounding down would send clients
	//      back before the budget exists, and 0 would mean "retry now".
	// KIND: boundary
	// CATCHES: s06
	// CHAPTER: gw.03 section 2.3
	for d, want := range map[time.Duration]int{0: 1, time.Millisecond: 1, 1200 * time.Millisecond: 2, 59 * time.Second: 59, 59001 * time.Millisecond: 60} {
		if got := limit.RetryAfterSeconds(d); got != want {
			t.Errorf("RetryAfterSeconds(%v) = %d, want %d", d, got, want)
		}
	}
}

type fakeCounter struct {
	n   int
	err error
}

func (f fakeCounter) CountPrompt(context.Context, string, []byte) (int, error) { return f.n, f.err }

func TestEstimateCost(t *testing.T) {
	// WHY: the reservation is prompt tokens plus the completion limit
	//      (max_completion_tokens over max_tokens, else 256); when the engine's
	//      tokenizer is down the estimate falls back to bytes / 4.
	// KIND: unit
	// CATCHES: s12, s13
	// CHAPTER: gw.03 section 2.2
	req := func(body string) *server.Request {
		r, err := server.ParseRequest([]byte(body))
		if err != nil {
			t.Fatal(err)
		}
		return &r
	}
	ctx := context.Background()
	if c := limit.EstimateCost(ctx, fakeCounter{n: 7}, req(`{"model":"m","prompt":"x","max_tokens":10}`)); c != (limit.Cost{Requests: 1, Tokens: 17}) {
		t.Fatalf("cost %+v, want 7 + 10", c)
	}
	if c := limit.EstimateCost(ctx, fakeCounter{n: 7}, req(`{"model":"m","prompt":"x","max_tokens":10,"max_completion_tokens":4}`)); c.Tokens != 11 {
		t.Fatalf("cost %+v, want 7 + 4 (max_completion_tokens wins)", c)
	}
	if c := limit.EstimateCost(ctx, fakeCounter{n: 7}, req(`{"model":"m","prompt":"x"}`)); c.Tokens != 7+limit.DefaultCompletionReserve {
		t.Fatalf("cost %+v, want 7 + 256 with no max_tokens", c)
	}
	// 9 bytes of prompt text: ceil(9 / 4) = 3.
	if c := limit.EstimateCost(ctx, fakeCounter{err: errors.New("engine down")}, req(`{"model":"m","prompt":"123456789","max_tokens":1}`)); c.Tokens != 4 {
		t.Fatalf("cost %+v, want ceil(9/4) + 1 = 4 when the tokenizer fails", c)
	}
}

func TestPromptText(t *testing.T) {
	// WHY: the text that is tokenized for the estimate: the completions
	//      prompt, the chat contents joined by newlines, or the embeddings input.
	// KIND: unit
	// CHAPTER: gw.03 section 2.2
	for body, want := range map[string]string{
		`{"prompt":"Once upon"}`: "Once upon",
		`{"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Hi"}]}`: "Be brief.\nHi",
		`{"input":["a","b"]}`: "a\nb",
	} {
		if got := limit.PromptText([]byte(body)); got != want {
			t.Errorf("PromptText(%s) = %q, want %q", body, got, want)
		}
	}
}

func TestTokenizeCounter(t *testing.T) {
	// WHY: the production counter asks the engine (POST /v1/tokenize, engine
	//      tier) and counts the ids it returns.
	// KIND: unit
	// CHAPTER: gw.03 section 4
	var got map[string]any
	eng := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/tokenize" {
			http.NotFound(w, r)
			return
		}
		json.NewDecoder(r.Body).Decode(&got)
		io.WriteString(w, `{"ids":[1,2,3,4,5]}`)
	}))
	defer eng.Close()
	n, err := limit.TokenizeCounter{BaseURL: eng.URL + "/"}.CountPrompt(context.Background(), "smol", []byte(`{"prompt":"hello"}`))
	if err != nil || n != 5 || got["model"] != "smol" || got["text"] != "hello" {
		t.Fatalf("CountPrompt = %d, %v; engine saw %v", n, err, got)
	}
}

// --- the stage ------------------------------------------------------------

type gateway struct {
	h      http.Handler
	fc     *clock.Fake
	status int
	usage  server.Usage
}

func newGateway(t *testing.T, p auth.Principal, counter limit.Counter) *gateway {
	g := &gateway{fc: clock.NewFake(t0), status: 200}
	keys := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			next.ServeHTTP(w, r.WithContext(auth.WithPrincipal(r.Context(), p)))
		})
	}
	proxy := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if g.usage != (server.Usage{}) {
			server.ExchangeFrom(r.Context()).SetUsage(g.usage)
		}
		if g.status >= 500 {
			server.WriteError(w, g.status, "server_error", "no_capacity", "", "no worker")
			return
		}
		w.WriteHeader(g.status)
	})
	g.h = server.New(server.Config{}, server.Deps{Clock: g.fc, Keys: keys, Limiter: limit.Middleware(limit.New(g.fc), counter), Proxy: proxy}).Handler()
	return g
}

func (g *gateway) post(body string) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	g.h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))
	return rec
}

func header(t *testing.T, rec *httptest.ResponseRecorder, name string) int {
	t.Helper()
	v := rec.Header().Get(name)
	n, err := strconv.Atoi(v)
	if err != nil {
		t.Fatalf("header %s = %q, want an integer", name, v)
	}
	return n
}

func TestMiddleware429Headers(t *testing.T) {
	// WHY: conformance case ratelimit.429: every authenticated response has
	//      the x-ratelimit-* headers; over the limit the answer is 429
	//      rate_limit_error / rate_limit_exceeded with Retry-After.
	// KIND: conformance
	// CATCHES: s09
	// CHAPTER: gw.03 section 4
	// Each request reserves 10 + 40 = 50 tokens; the fake proxy reports no
	// usage, so each settles at 0 tokens and the next one sees 950 again.
	g := newGateway(t, auth.Principal{KeyID: "k1", RPM: 2, TPM: 1000}, fakeCounter{n: 10})
	body := `{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":40}`
	for i := 0; i < 2; i++ {
		rec := g.post(body)
		if rec.Code != 200 {
			t.Fatalf("request %d: %d", i+1, rec.Code)
		}
		if header(t, rec, "x-ratelimit-limit-requests") != 2 || header(t, rec, "x-ratelimit-remaining-requests") != 1-i ||
			header(t, rec, "x-ratelimit-limit-tokens") != 1000 || header(t, rec, "x-ratelimit-remaining-tokens") != 950 {
			t.Fatalf("request %d headers: %v", i+1, rec.Header())
		}
		if rec.Header().Get("x-ratelimit-reset-tokens") == "" {
			t.Fatal("x-ratelimit-reset-tokens missing")
		}
	}
	rec := g.post(body)
	if rec.Code != 429 || header(t, rec, "Retry-After") != 30 {
		t.Fatalf("third request: %d, Retry-After %q; want 429 and 30 (one request per 30 s at RPM 2)", rec.Code, rec.Header().Get("Retry-After"))
	}
	var e struct {
		Error struct{ Type, Code string } `json:"error"`
	}
	json.Unmarshal(rec.Body.Bytes(), &e)
	if e.Error.Type != "rate_limit_error" || e.Error.Code != "rate_limit_exceeded" {
		t.Fatalf("429 body %s", rec.Body.String())
	}
	if header(t, rec, "x-ratelimit-remaining-requests") != 0 {
		t.Fatal("a 429 carries the headers too")
	}
}

func TestMiddlewareSettlesActualUsage(t *testing.T) {
	// WHY: the proxy records the real usage; settling with it refunds what
	//      the estimate over-reserved, so the next response shows the true
	//      remaining budget.
	// KIND: unit
	// CATCHES: s11
	// CHAPTER: gw.03 section 2.2
	g := newGateway(t, auth.Principal{KeyID: "k1", RPM: 100, TPM: 1000}, fakeCounter{n: 10})
	g.usage = server.Usage{PromptTokens: 12, CompletionTokens: 8, TotalTokens: 20}
	body := `{"model":"m","prompt":"x","max_tokens":290}` // reserves 300
	g.post(body)
	rec := g.post(body)
	if got := header(t, rec, "x-ratelimit-remaining-tokens"); got != 1000-20-300 {
		t.Fatalf("remaining tokens %d, want 680: the first request settled at 20, the second reserved 300", got)
	}
}

func TestMiddlewareCancelsOn5xx(t *testing.T) {
	// WHY: a request that failed before any upstream work (503 no_capacity)
	//      should not cost the client its budget; the reservation is cancelled.
	// KIND: unit
	// CATCHES: s10
	// CHAPTER: gw.03 section 5, Pitfalls, item 4
	g := newGateway(t, auth.Principal{KeyID: "k1", RPM: 1, TPM: 1000}, fakeCounter{n: 10})
	g.status = 503
	if rec := g.post(`{"model":"m","prompt":"x","max_tokens":10}`); rec.Code != 503 {
		t.Fatalf("status %d", rec.Code)
	}
	g.status = 200
	if rec := g.post(`{"model":"m","prompt":"x","max_tokens":10}`); rec.Code != 200 {
		t.Fatalf("after a cancelled 503 the one-request budget is back; got %d", rec.Code)
	}
}

func TestNoPrincipalPassesThrough(t *testing.T) {
	// WHY: without the authn stage there is no key to limit; the stage must
	//      pass the request on rather than invent a shared anonymous bucket.
	// KIND: boundary
	// CHAPTER: gw.03 section 4
	proxy := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(204) })
	h := server.New(server.Config{}, server.Deps{Limiter: limit.Middleware(limit.New(nil), nil), Proxy: proxy}).Handler()
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/completions", strings.NewReader(`{}`)))
	if rec.Code != 204 || rec.Header().Get("x-ratelimit-limit-requests") != "" {
		t.Fatalf("got %d with headers %v", rec.Code, rec.Header())
	}
}

func TestMiddlewareKeysByKeyID(t *testing.T) {
	// WHY: budgets belong to keys: two keys of one tenant (a CI key and a
	//      developer's key) must not drain each other.
	// KIND: unit
	// CATCHES: s14
	// CHAPTER: gw.03 section 2.1
	l := limit.New(clock.NewFake(t0))
	keys := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			p := auth.Principal{KeyID: r.Header.Get("X-Test-Key"), Tenant: "acme", RPM: 1}
			next.ServeHTTP(w, r.WithContext(auth.WithPrincipal(r.Context(), p)))
		})
	}
	ok := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200) })
	h := server.New(server.Config{}, server.Deps{Keys: keys, Limiter: limit.Middleware(l, fakeCounter{n: 1}), Proxy: ok}).Handler()
	call := func(key string) int {
		req := httptest.NewRequest("POST", "/v1/completions", strings.NewReader(`{"model":"m","prompt":"x","max_tokens":1}`))
		req.Header.Set("X-Test-Key", key)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		return rec.Code
	}
	if call("k1") != 200 || call("k1") != 429 {
		t.Fatal("k1: one request per minute, the second is 429")
	}
	if got := call("k2"); got != 200 {
		t.Fatalf("k2 of the same tenant got %d: its budget is its own", got)
	}
}
