// Course tests for gw.01: the gateway's composition root library
// (go/gateway/server) and the runtime.toml loader (go/config).
//
// Server tests put recording middleware in every Deps slot and a fake proxy
// at the end, then talk to the chain as a client would. Lifecycle tests run
// the real two-listener server on 127.0.0.1:0. Nothing sleeps for a fixed
// time: waits are channels bounded by `patience`, and the one condition that
// must be polled (/readyz turning 503) is polled every 5 ms up to patience.
package gw_01

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"tinyllm/gateway/server"
)

const patience = 3 * time.Second

// trace records the order in which stages see the request and the response.
type trace struct {
	mu  sync.Mutex
	log []string
}

func (t *trace) add(s string) {
	t.mu.Lock()
	t.log = append(t.log, s)
	t.mu.Unlock()
}

func (t *trace) get() []string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return append([]string(nil), t.log...)
}

func recording(tr *trace, name string) server.Middleware {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			tr.add("enter " + name)
			next.ServeHTTP(w, r)
			tr.add("leave " + name)
		})
	}
}

// fakeTracer records spans.
type fakeTracer struct {
	tr    *trace
	mu    sync.Mutex
	spans []*fakeSpan
}

type fakeSpan struct {
	name  string
	mu    sync.Mutex
	attrs map[string]any
	ended bool
}

func (f *fakeTracer) Start(ctx context.Context, name string) (context.Context, server.Span) {
	if f.tr != nil {
		f.tr.add("enter otel")
	}
	s := &fakeSpan{name: name, attrs: map[string]any{}}
	f.mu.Lock()
	f.spans = append(f.spans, s)
	f.mu.Unlock()
	return ctx, s
}

func (s *fakeSpan) SetAttribute(k string, v any) {
	s.mu.Lock()
	s.attrs[k] = v
	s.mu.Unlock()
}

func (s *fakeSpan) End() {
	s.mu.Lock()
	s.ended = true
	s.mu.Unlock()
}

func (f *fakeTracer) only(t *testing.T) *fakeSpan {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.spans) != 1 {
		t.Fatalf("%d spans recorded, want 1", len(f.spans))
	}
	return f.spans[0]
}

func okProxy(tr *trace) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if tr != nil {
			tr.add("proxy")
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"ok":true}`)
	})
}

func do(t *testing.T, h http.Handler, method, path, body string, hdr map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func checkError(t *testing.T, rec *httptest.ResponseRecorder, status int, typ, code string) {
	t.Helper()
	if rec.Code != status {
		t.Fatalf("status = %d, want %d (body %q)", rec.Code, status, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Fatalf("Content-Type = %q, want application/json", ct)
	}
	var e struct {
		Error map[string]any `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &e); err != nil || e.Error == nil {
		t.Fatalf("body is not {\"error\": {...}}: %q", rec.Body.String())
	}
	for _, k := range []string{"message", "type", "param", "code"} {
		if _, ok := e.Error[k]; !ok {
			t.Fatalf("error object lacks %q: %q", k, rec.Body.String())
		}
	}
	if e.Error["type"] != typ || e.Error["code"] != code {
		t.Fatalf("type/code = %v/%v, want %s/%s", e.Error["type"], e.Error["code"], typ, code)
	}
}

func TestMiddlewareOrderHand(t *testing.T) {
	// WHY: the order is a contract: a request is authenticated before policy
	//      and limits look at it, limited before the cache can answer it, and
	//      metered right after the proxy. The chapter's worked example is
	//      exactly this trace.
	// KIND: unit
	// CATCHES: s01, s02
	// CHAPTER: gw.01 section 3, worked example
	tr := &trace{}
	d := server.Deps{
		Tracer:  &fakeTracer{tr: tr},
		Keys:    recording(tr, "authn"),
		Policy:  recording(tr, "policy"),
		Limiter: recording(tr, "ratelimit"),
		Cache:   recording(tr, "cache"),
		Router:  recording(tr, "route"),
		Ledger:  recording(tr, "meter"),
		Proxy:   okProxy(tr),
	}
	h := server.New(server.Config{}, d).Handler()
	rec := do(t, h, "POST", "/v1/chat/completions", `{"model":"m"}`, nil)
	if rec.Code != 200 {
		t.Fatalf("status %d", rec.Code)
	}
	want := []string{
		"enter otel", "enter authn", "enter policy", "enter ratelimit", "enter cache", "enter route",
		"enter meter", "proxy", "leave meter",
		"leave route", "leave cache", "leave ratelimit", "leave policy", "leave authn",
	}
	if got := tr.get(); fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("stage trace\n got %v\nwant %v", got, want)
	}
	if got := fmt.Sprint(server.Order); got != "[requestid otel recover authn policy ratelimit cache route proxy meter]" {
		t.Fatalf("server.Order = %s, the contract order", got)
	}
}

func TestNilStagesAreSkipped(t *testing.T) {
	// WHY: gw.01 lands before the stages exist; with only a proxy the chain
	//      must still serve, so every later module can plug in one at a time.
	// KIND: boundary
	// CHAPTER: gw.01 section 4
	h := server.New(server.Config{}, server.Deps{Proxy: okProxy(nil)}).Handler()
	if rec := do(t, h, "POST", "/v1/completions", `{}`, nil); rec.Code != 200 || rec.Body.String() != `{"ok":true}` {
		t.Fatalf("got %d %q, want the proxy's 200", rec.Code, rec.Body.String())
	}
	h = server.New(server.Config{}, server.Deps{}).Handler()
	checkError(t, do(t, h, "POST", "/v1/completions", `{}`, nil), 503, "server_error", "no_capacity")
}

func TestRequestIDStage(t *testing.T) {
	// WHY: the first stage names the request: a caller's printable id of at
	//      most 128 bytes is kept, anything else replaced; the id is on the
	//      response, on the forwarded request, and in the Exchange.
	// KIND: unit
	// CATCHES: s05
	// CHAPTER: gw.01 section 2.2
	var seenHeader, seenEx string
	proxy := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seenHeader = r.Header.Get("X-Request-Id")
		seenEx = server.ExchangeFrom(r.Context()).RequestID
	})
	h := server.New(server.Config{}, server.Deps{Proxy: proxy}).Handler()
	for _, tc := range []struct {
		in   string
		keep bool
	}{
		{"", false}, {"req-42", true}, {strings.Repeat("x", 128), true},
		{strings.Repeat("x", 129), false}, {"has space", false},
	} {
		hdr := map[string]string{}
		if tc.in != "" {
			hdr["X-Request-Id"] = tc.in
		}
		rec := do(t, h, "POST", "/v1/completions", `{}`, hdr)
		got := rec.Header().Get("X-Request-Id")
		if tc.keep && got != tc.in {
			t.Fatalf("X-Request-Id %q was replaced by %q", tc.in, got)
		}
		if !tc.keep && (got == "" || got == tc.in) {
			t.Fatalf("X-Request-Id %q must be replaced by a generated id, got %q", tc.in, got)
		}
		if seenHeader != got || seenEx != got {
			t.Fatalf("response id %q, forwarded header %q, Exchange %q: all three must agree", got, seenHeader, seenEx)
		}
	}
}

func TestStripsInternalHeaders(t *testing.T) {
	// WHY: X-TL-Priority and X-TL-KV-Handle are set by the gateway toward
	//      engines; a client that sends them could jump the queue or resume
	//      someone else's KV (conformance case priority.internal). Public
	//      API version selection must survive this sanitization.
	// KIND: unit
	// CATCHES: s04
	// CHAPTER: gw.01 section 5, Pitfalls, item 4
	var got http.Header
	proxy := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { got = r.Header.Clone() })
	h := server.New(server.Config{}, server.Deps{Proxy: proxy}).Handler()
	do(t, h, "POST", "/v1/chat/completions", `{}`, map[string]string{
		"X-TL-Priority": "100", "x-tl-kv-handle": "h-1", "X-Tl-Route": "w", "X-Custom": "kept",
		"X-TL-API-Version": "2",
	})
	for k := range got {
		if strings.HasPrefix(strings.ToLower(k), "x-tl-") && !strings.EqualFold(k, "X-TL-API-Version") {
			t.Fatalf("client header %s reached the proxy", k)
		}
	}
	if got.Get("X-Custom") != "kept" {
		t.Fatal("ordinary headers must still reach the proxy")
	}
	if got.Get("X-TL-API-Version") != "2" {
		t.Fatal("public API version selection must reach the version middleware")
	}
}

func TestRecoverTurnsPanicInto500(t *testing.T) {
	// WHY: a bug in one stage must cost one request a 500 in the error shape,
	//      not the process; the server span still records the 500.
	// KIND: fault
	// CATCHES: s03, s06
	// CHAPTER: gw.01 section 2.1
	ft := &fakeTracer{}
	boom := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/boom" {
				panic("route table corrupted")
			}
			next.ServeHTTP(w, r)
		})
	}
	h := server.New(server.Config{}, server.Deps{Router: boom, Proxy: okProxy(nil), Tracer: ft}).Handler()
	rec := do(t, h, "POST", "/boom", `{}`, nil)
	checkError(t, rec, 500, "server_error", "internal_error")
	if rec.Header().Get("X-Request-Id") == "" {
		t.Fatal("the 500 still carries X-Request-Id")
	}
	sp := ft.only(t)
	if !sp.ended || sp.attrs["http.response.status_code"] != 500 {
		t.Fatalf("span ended=%v status=%v, want ended with 500", sp.ended, sp.attrs["http.response.status_code"])
	}
	if rec := do(t, h, "POST", "/v1/completions", `{}`, nil); rec.Code != 200 {
		t.Fatalf("the next request got %d: the server must keep serving", rec.Code)
	}
}

func TestRecoverAfterFirstByte(t *testing.T) {
	// WHY: once a stream has started, the status line is gone; a panic then
	//      must not append a second status or a JSON error to the stream.
	// KIND: fault
	// CATCHES: s19
	// CHAPTER: gw.01 section 5, Pitfalls, item 6
	proxy := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: {\"x\":1}\n\n")
		panic("upstream parser bug")
	})
	h := server.New(server.Config{}, server.Deps{Proxy: proxy}).Handler()
	rec := do(t, h, "POST", "/v1/chat/completions", `{}`, nil)
	if rec.Code != 200 || rec.Body.String() != "data: {\"x\":1}\n\n" {
		t.Fatalf("got %d %q: after the first byte nothing more may be written", rec.Code, rec.Body.String())
	}
}

func TestOTelSpanAttributes(t *testing.T) {
	// WHY: one SERVER span per request named "<METHOD> <path>", carrying the
	//      status and whatever the stages recorded (tl.ratelimit.decision,
	//      tl.cache.hit, ...) on the Exchange: obs.01 exports exactly this.
	// KIND: unit
	// CATCHES: s19
	// CHAPTER: gw.01 section 2.1
	ft := &fakeTracer{}
	lim := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			server.ExchangeFrom(r.Context()).SetAttr("tl.ratelimit.decision", "allow")
			next.ServeHTTP(w, r)
		})
	}
	h := server.New(server.Config{}, server.Deps{Limiter: lim, Proxy: okProxy(nil), Tracer: ft}).Handler()
	do(t, h, "POST", "/v1/chat/completions", `{}`, nil)
	sp := ft.only(t)
	if sp.name != "POST /v1/chat/completions" {
		t.Fatalf("span name %q", sp.name)
	}
	for k, v := range map[string]any{
		"http.request.method": "POST", "http.route": "/v1/chat/completions",
		"http.response.status_code": 200, "tl.ratelimit.decision": "allow",
	} {
		if sp.attrs[k] != v {
			t.Fatalf("span attribute %s = %v, want %v", k, sp.attrs[k], v)
		}
	}
}

func TestWriteErrorShape(t *testing.T) {
	// WHY: every gateway error goes through WriteError, so OpenAI clients
	//      parse every one: all four keys present, empty param and code null.
	// KIND: unit
	// CATCHES: s18
	// CHAPTER: gw.01 section 4
	rec := httptest.NewRecorder()
	server.WriteError(rec, 422, "invalid_request_error", "unsupported_parameter", "n", "n must be 1")
	if rec.Code != 422 || !strings.HasPrefix(rec.Header().Get("Content-Type"), "application/json") {
		t.Fatalf("status %d, content type %q", rec.Code, rec.Header().Get("Content-Type"))
	}
	want := `{"error":{"message":"n must be 1","type":"invalid_request_error","param":"n","code":"unsupported_parameter"}}`
	if strings.TrimSpace(rec.Body.String()) != want {
		t.Fatalf("body %s, want %s", rec.Body.String(), want)
	}
	rec = httptest.NewRecorder()
	server.WriteError(rec, 500, "server_error", "", "", "boom")
	want = `{"error":{"message":"boom","type":"server_error","param":null,"code":null}}`
	if strings.TrimSpace(rec.Body.String()) != want {
		t.Fatalf("body %s, want %s (empty param and code are null)", rec.Body.String(), want)
	}
}

func TestExchangeRequestParsing(t *testing.T) {
	// WHY: auth, limits, cache, and routing all need model and token limits;
	//      the body is read once, parsed once, and put back so the proxy still
	//      forwards it byte for byte.
	// KIND: unit
	// CATCHES: s10, s11
	// CHAPTER: gw.01 section 2.3
	body := `{"model":"smol","max_tokens":9,"max_completion_tokens":5,"temperature":0,"seed":7,"stream":true,"extra":[1]}`
	var parsed *server.Request
	var forwarded string
	stage := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			req, err := server.ExchangeFrom(r.Context()).Request(r) // one read, then the proxy reads
			if err != nil {
				t.Errorf("Request: %v", err)
			}
			parsed = req
			next.ServeHTTP(w, r)
		})
	}
	proxy := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		forwarded = string(b)
	})
	h := server.New(server.Config{}, server.Deps{Keys: stage, Proxy: proxy}).Handler()
	do(t, h, "POST", "/v1/chat/completions", body, nil)
	if forwarded != body {
		t.Fatalf("the proxy read %q, want the original body", forwarded)
	}
	if parsed.Model != "smol" || !parsed.Stream || parsed.MaxTokens != 5 ||
		parsed.Temperature == nil || *parsed.Temperature != 0 || parsed.Seed == nil || *parsed.Seed != 7 {
		t.Fatalf("parsed %+v: max_completion_tokens (5) wins over max_tokens (9)", *parsed)
	}
	ex1 := server.NewExchange("r0", nil, 0)
	hr := httptest.NewRequest("POST", "/v1/completions", strings.NewReader(`{"model":"m"}`))
	first, _ := ex1.Request(hr)
	again, _ := ex1.Request(hr)
	if again != first {
		t.Fatal("a second Request call must return the cached parse")
	}
	if b, _ := io.ReadAll(hr.Body); string(b) != `{"model":"m"}` {
		t.Fatalf("after two Request calls the body reads %q", b)
	}
	r, err := server.ParseRequest([]byte(`{"model":"m","max_tokens":3}`))
	if err != nil || r.MaxTokens != 3 || r.Temperature != nil || r.Seed != nil {
		t.Fatalf("ParseRequest = %+v, %v", r, err)
	}
	for _, bad := range []string{``, `[1,2]`, `"text"`, `{"model":`} {
		if _, err := server.ParseRequest([]byte(bad)); err == nil {
			t.Fatalf("ParseRequest(%q) must fail: not one JSON object", bad)
		}
	}
	ex := server.NewExchange("r1", nil, 16)
	req := httptest.NewRequest("POST", "/v1/completions", strings.NewReader(`{"model":"this-is-longer-than-16"}`))
	if _, err := ex.Request(req); err == nil {
		t.Fatal("a body over the maximum must be an error (400)")
	}
}

func TestReadyz(t *testing.T) {
	// WHY: /healthz says the process is up; /readyz says it can take traffic
	//      (Deps.Ready: at least one routable worker), which is what a
	//      Kubernetes Service and the milestone runner wait for.
	// KIND: unit
	// CHAPTER: gw.01 section 2.4
	ready := errors.New("no routable worker")
	metrics := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, "tl_up 1\n") })
	s := server.New(server.Config{}, server.Deps{Proxy: okProxy(nil), Metrics: metrics,
		Ready: func(context.Context) error { return ready }})
	h := s.HealthHandler()
	if rec := do(t, h, "GET", "/healthz", "", nil); rec.Code != 200 {
		t.Fatalf("/healthz = %d", rec.Code)
	}
	if rec := do(t, h, "GET", "/readyz", "", nil); rec.Code != 503 {
		t.Fatalf("/readyz = %d while Ready fails, want 503", rec.Code)
	}
	ready = nil
	if rec := do(t, h, "GET", "/readyz", "", nil); rec.Code != 200 {
		t.Fatalf("/readyz = %d once Ready passes, want 200", rec.Code)
	}
	if rec := do(t, h, "GET", "/metrics", "", nil); rec.Code != 200 || rec.Body.String() != "tl_up 1\n" {
		t.Fatalf("/metrics = %d %q", rec.Code, rec.Body.String())
	}
	if s.Draining() {
		t.Fatal("a new server is not draining")
	}
}

// --- lifecycle on real listeners ------------------------------------------

type running struct {
	s           *server.Server
	api, health string
	done        chan error
}

func start(t *testing.T, d server.Deps, deadline time.Duration) *running {
	t.Helper()
	al, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	hl, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	rs := &running{s: server.New(server.Config{DrainDeadline: deadline}, d), api: al.Addr().String(), health: hl.Addr().String(), done: make(chan error, 1)}
	go func() { rs.done <- rs.s.Serve(al, hl) }()
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), patience)
		defer cancel()
		rs.s.Shutdown(ctx)
	})
	return rs
}

func status(url string) int {
	c := &http.Client{Timeout: patience}
	resp, err := c.Get(url)
	if err != nil {
		return -1
	}
	resp.Body.Close()
	return resp.StatusCode
}

func eventually(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.After(patience)
	for !cond() {
		select {
		case <-deadline:
			t.Fatalf("%s did not happen within %v", what, patience)
		case <-time.After(5 * time.Millisecond):
		}
	}
}

// streamer is a fake proxy whose stream sends one event, then waits for
// release (or for its context to end).
type streamer struct {
	started   chan struct{}
	release   chan struct{}
	cancelled chan struct{}
}

func newStreamer() *streamer {
	return &streamer{make(chan struct{}, 1), make(chan struct{}), make(chan struct{}, 1)}
}

func (s *streamer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/event-stream")
	io.WriteString(w, "data: {\"n\":1}\n\n")
	http.NewResponseController(w).Flush()
	s.started <- struct{}{}
	select {
	case <-s.release:
		io.WriteString(w, "data: [DONE]\n\n")
	case <-r.Context().Done():
		s.cancelled <- struct{}{}
	}
}

func openStream(t *testing.T, addr string) (*bufio.Reader, func()) {
	t.Helper()
	resp, err := http.Post("http://"+addr+"/v1/chat/completions", "application/json", strings.NewReader(`{"stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	br := bufio.NewReader(resp.Body)
	line, err := br.ReadString('\n')
	if err != nil || line != "data: {\"n\":1}\n" {
		t.Fatalf("first line %q, %v", line, err)
	}
	return br, func() { resp.Body.Close() }
}

func TestShutdownFlipsReadyzThenDrains(t *testing.T) {
	// WHY: SIGTERM must not cut a stream: /readyz turns 503 at once (the load
	//      balancer stops sending), new connections are refused, and the
	//      stream in flight runs to [DONE] before Shutdown returns nil.
	// KIND: fault
	// CATCHES: s07, s08
	// CHAPTER: gw.01 section 3, worked example (drain timeline)
	st := newStreamer()
	rs := start(t, server.Deps{Proxy: st}, patience)
	if got := status("http://" + rs.health + "/readyz"); got != 200 {
		t.Fatalf("/readyz = %d before shutdown", got)
	}
	br, closeBody := openStream(t, rs.api)
	defer closeBody()
	<-st.started
	shut := make(chan error, 1)
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), patience)
		defer cancel()
		shut <- rs.s.Shutdown(ctx)
	}()
	eventually(t, "/readyz turning 503 while the stream is in flight", func() bool {
		return status("http://"+rs.health+"/readyz") == 503
	})
	select {
	case err := <-shut:
		t.Fatalf("Shutdown returned (%v) while a stream was still in flight", err)
	default:
	}
	eventually(t, "the API listener refusing new connections", func() bool {
		c, err := net.DialTimeout("tcp", rs.api, 100*time.Millisecond)
		if err == nil {
			c.Close()
		}
		return err != nil
	})
	close(st.release)
	rest, _ := io.ReadAll(br)
	if !strings.Contains(string(rest), "data: [DONE]") {
		t.Fatalf("the drained stream ended with %q, want it to finish with [DONE]", rest)
	}
	select {
	case err := <-shut:
		if err != nil {
			t.Fatalf("Shutdown = %v, want nil after a clean drain", err)
		}
	case <-time.After(patience):
		t.Fatal("Shutdown did not return after the last stream finished")
	}
}

func TestShutdownCutsStreamsAtDeadline(t *testing.T) {
	// WHY: a drain has a deadline: a stream still running then is cancelled
	//      (its handler context ends, so the engine request is aborted) and
	//      Shutdown reports context.DeadlineExceeded.
	// KIND: fault
	// CATCHES: s09
	// CHAPTER: gw.01 section 2.4
	st := newStreamer()
	rs := start(t, server.Deps{Proxy: st}, patience)
	_, closeBody := openStream(t, rs.api)
	defer closeBody()
	<-st.started
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	err := rs.s.Shutdown(ctx)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Shutdown = %v, want context.DeadlineExceeded", err)
	}
	select {
	case <-st.cancelled:
	case <-time.After(patience):
		t.Fatal("the in-flight handler's context was not cancelled at the deadline")
	}
}

func TestRunShutsDownOnCancel(t *testing.T) {
	// WHY: your main turns SIGTERM into a cancelled context (signal.
	//      NotifyContext); Run must then drain and return nil.
	// KIND: unit
	// CHAPTER: gw.01 section 4
	al, _ := net.Listen("tcp", "127.0.0.1:0")
	hl, _ := net.Listen("tcp", "127.0.0.1:0")
	s := server.New(server.Config{DrainDeadline: patience}, server.Deps{Proxy: okProxy(nil)})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- s.Run(ctx, al, hl) }()
	eventually(t, "the API answering", func() bool {
		resp, err := http.Post("http://"+al.Addr().String()+"/v1/completions", "application/json", strings.NewReader(`{}`))
		if err != nil {
			return false
		}
		resp.Body.Close()
		return resp.StatusCode == 200
	})
	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("Run = %v, want nil", err)
		}
	case <-time.After(patience):
		t.Fatal("Run did not return after its context was cancelled")
	}
	if !s.Draining() {
		t.Fatal("after Run returns the server reports Draining")
	}
}
