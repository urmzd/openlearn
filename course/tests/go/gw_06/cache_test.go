// Course tests for gw.06: the response cache (go/gateway/cache).
//
// The upstream is a counting fake proxy at the end of the chain; a test
// principal stands in for the authn stage. TTLs run on the testkit's fake
// clock. The singleflight test releases 100 goroutines at once and counts
// upstream calls; it runs under the race detector.
package gw_06

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"openlearn.urmzd.com/tl/testkit/clock"
	"tinyllm/gateway/auth"
	"tinyllm/gateway/cache"
	"tinyllm/gateway/server"
)

var t0 = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

const (
	handA = `{"model":"smol","messages":[{"role":"user","content":"hi"}],"temperature":0,"user":"alice"}`
	handB = `{"temperature":0.0,"top_p":1,"messages":[{"role":"user","content":"hi"}],"model":"smol"}`
	// The canonical form both reduce to (chapter section 3).
	handCanon = `{"frequency_penalty":0,"logprobs":null,"messages":[{"content":"hi","role":"user"}],"min_p":0,"model":"smol","n":1,"presence_penalty":0,"repetition_penalty":1,"stop":null,"stream":false,"temperature":0,"top_k":0,"top_p":1}`
)

func canon(t *testing.T, body string) (cache.CanonicalRequest, bool) {
	t.Helper()
	c, ok, err := cache.Canonicalize("/v1/chat/completions", []byte(body))
	if err != nil {
		t.Fatalf("Canonicalize(%s): %v", body, err)
	}
	return c, ok
}

func TestHandExample(t *testing.T) {
	// WHY: the worked example: two bodies that differ in field order, number
	//      spelling, an explicit default, and the `user` field ask for the
	//      same answer, so they reduce to one canonical JSON and one key; the
	//      same body at temperature 0.7 is not cacheable, and with seed 42 it is.
	// KIND: unit
	// CATCHES: s02, s03, s10, s13
	// CHAPTER: gw.06 section 3, worked example
	a, okA := canon(t, handA)
	b, okB := canon(t, handB)
	if string(a.JSON) != handCanon || string(b.JSON) != handCanon || !okA || !okB {
		t.Fatalf("canonical\n A %s (%v)\n B %s (%v)\nwant %s", a.JSON, okA, b.JSON, okB, handCanon)
	}
	p := auth.Principal{Tenant: "acme"}
	if cache.KeyOf(p, "7", a) != cache.KeyOf(p, "7", b) {
		t.Fatal("equal canonical requests must share a key")
	}
	if _, ok := canon(t, `{"model":"smol","messages":[],"temperature":0.7}`); ok {
		t.Fatal("temperature 0.7 without a seed is not deterministic: not cacheable")
	}
	if _, ok := canon(t, `{"model":"smol","messages":[],"temperature":0.7,"seed":42}`); !ok {
		t.Fatal("a fixed seed makes it deterministic: cacheable")
	}
	if _, ok := canon(t, `{"model":"smol","messages":[]}`); ok {
		t.Fatal("no temperature means 1: not cacheable")
	}
}

func TestCanonicalDistinctions(t *testing.T) {
	// WHY: anything that can change the answer must change the key: the
	//      model, max_tokens, the messages, and stream (a streamed answer is
	//      SSE bytes, a plain one JSON).
	// KIND: unit
	// CATCHES: s15
	// CHAPTER: gw.06 section 2.2
	base, _ := canon(t, `{"model":"smol","messages":[{"role":"user","content":"hi"}],"temperature":0,"max_tokens":5}`)
	for _, other := range []string{
		`{"model":"smol2","messages":[{"role":"user","content":"hi"}],"temperature":0,"max_tokens":5}`,
		`{"model":"smol","messages":[{"role":"user","content":"hi"}],"temperature":0,"max_tokens":6}`,
		`{"model":"smol","messages":[{"role":"user","content":"Hi"}],"temperature":0,"max_tokens":5}`,
		`{"model":"smol","messages":[{"role":"user","content":"hi"}],"temperature":0,"max_tokens":5,"stream":true}`,
	} {
		c, _ := canon(t, other)
		if string(c.JSON) == string(base.JSON) {
			t.Fatalf("%s canonicalizes like the base request", other)
		}
	}
	if _, _, err := cache.Canonicalize("/v1/completions", []byte(`[1]`)); err == nil {
		t.Fatal("a non-object body is an error")
	}
}

func TestKeyOfTenantAndRevision(t *testing.T) {
	// WHY: keys are scoped: two tenants never share an entry, two keys of one
	//      tenant do, and a new configuration revision (a model rollout)
	//      makes old answers unreachable.
	// KIND: unit
	// CATCHES: s01, s11
	// CHAPTER: gw.06 section 2.3
	c, _ := canon(t, handA)
	a1 := cache.KeyOf(auth.Principal{Tenant: "acme", KeyID: "k1"}, "1", c)
	a2 := cache.KeyOf(auth.Principal{Tenant: "acme", KeyID: "k2"}, "1", c)
	b1 := cache.KeyOf(auth.Principal{Tenant: "globex", KeyID: "k1"}, "1", c)
	r2 := cache.KeyOf(auth.Principal{Tenant: "acme", KeyID: "k1"}, "2", c)
	if a1 != a2 || a1 == b1 || a1 == r2 {
		t.Fatalf("same tenant %v, other tenant %v, other revision %v", a1 == a2, a1 == b1, a1 == r2)
	}
}

func key(b byte) cache.Key { var k cache.Key; k[0] = b; return k }

func TestLRUEvictionOrder(t *testing.T) {
	// WHY: at capacity the least recently used entry goes, and a Get counts
	//      as a use: put a, b; get a; put c evicts b.
	// KIND: unit
	// CATCHES: s06
	// CHAPTER: gw.06 section 2.1
	c := cache.New(2, clock.NewFake(t0))
	ctx := context.Background()
	c.Put(ctx, key('a'), cache.Entry{Body: []byte("A")}, time.Hour)
	c.Put(ctx, key('b'), cache.Entry{Body: []byte("B")}, time.Hour)
	if _, ok := c.Get(ctx, key('a')); !ok {
		t.Fatal("a missing")
	}
	c.Put(ctx, key('c'), cache.Entry{Body: []byte("C")}, time.Hour)
	if _, ok := c.Get(ctx, key('b')); ok {
		t.Fatal("b should have been evicted (least recently used)")
	}
	for _, k := range []byte{'a', 'c'} {
		if _, ok := c.Get(ctx, key(k)); !ok {
			t.Fatalf("%c evicted, want it kept", k)
		}
	}
	if c.Len() != 2 {
		t.Fatalf("Len = %d", c.Len())
	}
}

func TestTTLExpiry(t *testing.T) {
	// WHY: an entry lives exactly ttl: a hit at put + 9.999 s, a miss at put +
	//      10 s, and the expired entry is dropped.
	// KIND: fault
	// CATCHES: s05
	// CHAPTER: gw.06 section 2.1
	fc := clock.NewFake(t0)
	c := cache.New(4, fc)
	ctx := context.Background()
	c.Put(ctx, key('a'), cache.Entry{Body: []byte("A")}, 10*time.Second)
	fc.Advance(10*time.Second - time.Millisecond)
	if _, ok := c.Get(ctx, key('a')); !ok {
		t.Fatal("a miss before the TTL")
	}
	fc.Advance(time.Millisecond)
	if _, ok := c.Get(ctx, key('a')); ok {
		t.Fatal("a hit at exactly the TTL: an entry lives [put, put + ttl)")
	}
	if c.Len() != 0 {
		t.Fatal("the expired entry is still held")
	}
}

func TestPurge(t *testing.T) {
	// WHY: POST /admin/v1/cache:purge with a model drops only that model's
	//      answers; without one it drops everything.
	// KIND: unit
	// CATCHES: s14
	// CHAPTER: gw.06 section 4
	c := cache.New(8, clock.NewFake(t0))
	ctx := context.Background()
	c.Put(ctx, key('a'), cache.Entry{Model: "smol"}, time.Hour)
	c.Put(ctx, key('b'), cache.Entry{Model: "tiny"}, time.Hour)
	c.Put(ctx, key('c'), cache.Entry{Model: "smol"}, time.Hour)
	if n := c.Purge("smol"); n != 2 || c.Len() != 1 {
		t.Fatalf("Purge(smol) = %d, %d left", n, c.Len())
	}
	if n := c.Purge(""); n != 1 || c.Len() != 0 {
		t.Fatalf("Purge(\"\") = %d", n)
	}
}

// upstream is a counting fake engine at the end of the chain.
type upstream struct {
	calls   atomic.Int64
	gate    chan struct{} // when set, every call waits for it
	status  int
	stream  bool
	cutting bool // stream ends without [DONE]
}

const sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n"

func (u *upstream) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	n := u.calls.Add(1)
	if u.gate != nil {
		<-u.gate
	}
	if u.status != 0 && u.status != 200 {
		server.WriteError(w, u.status, "server_error", "no_capacity", "", "down")
		return
	}
	if u.stream {
		w.Header().Set("Content-Type", "text/event-stream")
		if u.cutting {
			io.WriteString(w, "data: {\"x\":1}\n\n")
			return
		}
		io.WriteString(w, sse)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"id": n, "choices": []any{}})
}

func chain(u *upstream, fc *clock.Fake, tenant func(r *http.Request) string) http.Handler {
	keys := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			next.ServeHTTP(w, r.WithContext(auth.WithPrincipal(r.Context(), auth.Principal{KeyID: "k", Tenant: tenant(r)})))
		})
	}
	c := cache.New(64, fc)
	return server.New(server.Config{}, server.Deps{Clock: fc, Keys: keys, Cache: cache.Middleware(c, cache.Options{TTL: time.Minute}), Proxy: u}).Handler()
}

func acme(*http.Request) string { return "acme" }

func send(h http.Handler, body string, hdr ...string) *httptest.ResponseRecorder {
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	for i := 0; i+1 < len(hdr); i += 2 {
		req.Header.Set(hdr[i], hdr[i+1])
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestHitAfterMiss(t *testing.T) {
	// WHY: the stage end to end: the first identical request is a miss that
	//      reaches the upstream, the second a hit with the same body.
	// KIND: unit
	// CATCHES: s12
	// CHAPTER: gw.06 section 4
	u := &upstream{}
	h := chain(u, clock.NewFake(t0), acme)
	r1, r2 := send(h, handA), send(h, handB)
	if r1.Header().Get("X-TL-Cache") != "miss" || r2.Header().Get("X-TL-Cache") != "hit" || u.calls.Load() != 1 {
		t.Fatalf("X-TL-Cache %q then %q, %d upstream calls", r1.Header().Get("X-TL-Cache"), r2.Header().Get("X-TL-Cache"), u.calls.Load())
	}
	if r1.Body.String() != r2.Body.String() || r2.Header().Get("Content-Type") != "application/json" {
		t.Fatalf("hit body %q content type %q, want the miss's body", r2.Body.String(), r2.Header().Get("Content-Type"))
	}
}

func TestSingleflightOneUpstreamCall(t *testing.T) {
	// WHY: 100 identical requests arriving together must make one upstream
	//      call: the first goes through, the other 99 wait for it and are
	//      answered from its result (a thundering herd would make 100).
	// KIND: fault
	// CATCHES: s07
	// CHAPTER: gw.06 section 2.4
	u := &upstream{gate: make(chan struct{})}
	h := chain(u, clock.NewFake(t0), acme)
	var wg sync.WaitGroup
	bodies := make([]string, 100)
	for i := range bodies {
		wg.Add(1)
		go func() {
			defer wg.Done()
			bodies[i] = send(h, handA).Body.String()
		}()
	}
	for u.calls.Load() == 0 {
		<-time.After(time.Millisecond) // wait for the leader to reach the upstream
	}
	close(u.gate)
	wg.Wait()
	if n := u.calls.Load(); n != 1 {
		t.Fatalf("%d upstream calls for 100 identical concurrent requests, want 1", n)
	}
	for i, b := range bodies {
		if b != bodies[0] || b == "" {
			t.Fatalf("request %d got %q, want the leader's %q", i, b, bodies[0])
		}
	}
}

func TestTenantIsolation(t *testing.T) {
	// WHY: tenant B must never read tenant A's answer, even for a byte-equal
	//      request: B's request is a miss that reaches the upstream.
	// KIND: fault
	// CATCHES: s01
	// CHAPTER: gw.06 section 5, Pitfalls, item 1
	u := &upstream{}
	h := chain(u, clock.NewFake(t0), func(r *http.Request) string { return r.Header.Get("X-Test-Tenant") })
	send(h, handA, "X-Test-Tenant", "acme")
	r := send(h, handA, "X-Test-Tenant", "globex")
	if r.Header().Get("X-TL-Cache") != "miss" || u.calls.Load() != 2 {
		t.Fatalf("globex: X-TL-Cache %q, %d upstream calls; want a miss and 2", r.Header().Get("X-TL-Cache"), u.calls.Load())
	}
}

func TestStreamReplayByteExact(t *testing.T) {
	// WHY: a cached stream is replayed as the same SSE bytes with the same
	//      Content-Type, so a streaming client cannot tell a hit from a miss.
	// KIND: unit
	// CATCHES: s12
	// CHAPTER: gw.06 section 2.5
	u := &upstream{stream: true}
	h := chain(u, clock.NewFake(t0), acme)
	body := `{"model":"smol","messages":[],"temperature":0,"stream":true}`
	send(h, body)
	r := send(h, body)
	if r.Header().Get("X-TL-Cache") != "hit" || r.Body.String() != sse || r.Header().Get("Content-Type") != "text/event-stream" || u.calls.Load() != 1 {
		t.Fatalf("replay: %q %q %q, %d calls", r.Header().Get("X-TL-Cache"), r.Header().Get("Content-Type"), r.Body.String(), u.calls.Load())
	}
}

func TestErrorsAndCutStreamsNotCached(t *testing.T) {
	// WHY: only a whole 200 is an answer worth keeping: a 503 or a stream cut
	//      before [DONE] cached for a minute would serve a failure to everyone.
	// KIND: fault
	// CATCHES: s08, s09
	// CHAPTER: gw.06 section 5, Pitfalls, item 3
	for name, u := range map[string]*upstream{"503": {status: 503}, "cut stream": {stream: true, cutting: true}} {
		h := chain(u, clock.NewFake(t0), acme)
		send(h, handA)
		if r := send(h, handA); r.Header().Get("X-TL-Cache") != "miss" || u.calls.Load() != 2 {
			t.Fatalf("%s: second request %q with %d upstream calls; want another miss", name, r.Header().Get("X-TL-Cache"), u.calls.Load())
		}
	}
}

func TestNotCacheableHasNoHeader(t *testing.T) {
	// WHY: a sampled request (temperature 0.7, no seed) is never cached and
	//      carries no X-TL-Cache header; every one reaches the upstream.
	// KIND: unit
	// CATCHES: s04
	// CHAPTER: gw.06 section 2.2
	u := &upstream{}
	h := chain(u, clock.NewFake(t0), acme)
	body := `{"model":"smol","messages":[],"temperature":0.7}`
	for i := 0; i < 2; i++ {
		if r := send(h, body); r.Header().Get("X-TL-Cache") != "" {
			t.Fatalf("X-TL-Cache %q on a non-cacheable request", r.Header().Get("X-TL-Cache"))
		}
	}
	if u.calls.Load() != 2 {
		t.Fatalf("%d upstream calls, want 2", u.calls.Load())
	}
}

func TestPurgeHandler(t *testing.T) {
	// WHY: POST /admin/v1/cache:purge answers {"purged": n}; an empty body
	//      purges everything.
	// KIND: unit, conformance
	// CATCHES: s14
	// CHAPTER: gw.06 section 4
	c := cache.New(8, clock.NewFake(t0))
	c.Put(context.Background(), key('a'), cache.Entry{Model: "smol"}, time.Hour)
	c.Put(context.Background(), key('b'), cache.Entry{Model: "tiny"}, time.Hour)
	h := cache.PurgeHandler(c)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/admin/v1/cache:purge", strings.NewReader(`{"model":"smol"}`)))
	if rec.Code != 200 || strings.TrimSpace(rec.Body.String()) != `{"purged":1}` {
		t.Fatalf("%d %s", rec.Code, rec.Body.String())
	}
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/admin/v1/cache:purge", nil))
	if strings.TrimSpace(rec.Body.String()) != `{"purged":1}` || c.Len() != 0 {
		t.Fatalf("purge all: %s, %d left", rec.Body.String(), c.Len())
	}
}
