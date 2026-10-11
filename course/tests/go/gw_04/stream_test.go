// Course tests for gw.04: the SSE streaming proxy with its observer and its
// commitment boundary (go/gateway/proxy, upgraded from gw.00).
//
// Fake engines are httptest servers that send fixture streams
// (course/fixtures/gw.04/streams.json) in pieces of a chosen size, so event
// boundaries never line up with reads. Faults come from the course testkit:
// chaosproxy resets the engine's connection mid-stream. The gw.00 suite runs
// as this module's smoke regression, so everything it proved still holds.
package gw_04

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"openlearn.urmzd.com/tl/testkit/chaosproxy"
	"tinyllm/gateway/proxy"
	"tinyllm/gateway/server"
)

const patience = 3 * time.Second

type fixture struct {
	Name   string        `json:"name"`
	SSE    string        `json:"sse"`
	Events int           `json:"events"`
	Usage  *server.Usage `json:"usage"`
}

func fixtures(t *testing.T) []fixture {
	t.Helper()
	dir := os.Getenv("TINYLLM_FIXTURES")
	if dir == "" {
		t.Fatal("TINYLLM_FIXTURES is not set (ol check sets it)")
	}
	b, err := os.ReadFile(filepath.Join(dir, "gw.04", "streams.json"))
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Streams []fixture `json:"streams"`
	}
	if err := json.Unmarshal(b, &f); err != nil || len(f.Streams) == 0 {
		t.Fatalf("gw.04/streams.json: %v", err)
	}
	return f.Streams
}

// recObserver records what the proxy reports, and what the client had
// received when the first byte was announced.
type recObserver struct {
	mu        sync.Mutex
	first     int
	chunks    []string
	done      int
	usage     server.Usage
	bodyAtFst int
	rec       *httptest.ResponseRecorder
}

func (o *recObserver) OnFirstByte() {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.first++
	if o.rec != nil {
		o.bodyAtFst = o.rec.Body.Len()
	}
}

func (o *recObserver) OnChunk(d []byte) {
	o.mu.Lock()
	o.chunks = append(o.chunks, string(d))
	o.mu.Unlock()
}

func (o *recObserver) OnDone(u server.Usage) {
	o.mu.Lock()
	o.done++
	o.usage = u
	o.mu.Unlock()
}

// piecewise is an upstream body that returns at most n bytes per Read.
type piecewise struct {
	data []byte
	n    int
}

func (p *piecewise) Read(b []byte) (int, error) {
	if len(p.data) == 0 {
		return 0, io.EOF
	}
	k := min(p.n, len(b), len(p.data))
	copy(b, p.data[:k])
	p.data = p.data[k:]
	return k, nil
}

func sseResponse(body io.Reader) *http.Response {
	h := http.Header{}
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	return &http.Response{StatusCode: 200, Header: h, Body: io.NopCloser(body), Request: req}
}

const handStream = `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}` + "\n\n" +
	`data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}` + "\n\n" +
	`data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}]}` + "\n\n" +
	`data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}` + "\n\n" +
	"data: [DONE]\n\n"

func TestHandExample(t *testing.T) {
	// WHY: the worked example: five events arrive in 40-byte reads; the
	//      client gets the same bytes, the observer hears the first byte once
	//      (before anything was written), five event payloads, and usage 5/2/7
	//      taken from the usage chunk.
	// KIND: unit
	// CATCHES: s02, s03
	// CHAPTER: gw.04 section 3, worked example
	rec := httptest.NewRecorder()
	o := &recObserver{rec: rec}
	err := proxy.StreamProxy(rec, httptest.NewRequest("POST", "/", nil), sseResponse(&piecewise{[]byte(handStream), 40}), o)
	if err != nil {
		t.Fatal(err)
	}
	if rec.Body.String() != handStream {
		t.Fatalf("client body differs from the upstream bytes:\n%q", rec.Body.String())
	}
	if o.first != 1 || o.bodyAtFst != 0 {
		t.Fatalf("OnFirstByte called %d times, with %d bytes already sent; want once, before the first write", o.first, o.bodyAtFst)
	}
	if len(o.chunks) != 5 || o.chunks[4] != "[DONE]" || !strings.Contains(o.chunks[1], `"Hel"`) {
		t.Fatalf("chunks %q", o.chunks)
	}
	if o.done != 1 || o.usage != (server.Usage{PromptTokens: 5, CompletionTokens: 2, TotalTokens: 7}) {
		t.Fatalf("OnDone %d times with %+v", o.done, o.usage)
	}
	if rec.Code != 200 || rec.Header().Get("Content-Type") != "text/event-stream" {
		t.Fatalf("status %d, content type %q", rec.Code, rec.Header().Get("Content-Type"))
	}
}

func TestRecordedStreamsByteExact(t *testing.T) {
	// WHY: whatever the read size (1 byte splits every UTF-8 character and
	//      every event), the client receives the stream byte for byte and the
	//      observer still sees each complete event once, with the right usage.
	// KIND: conformance
	// CATCHES: s04
	// CHAPTER: gw.04 section 2.2
	for _, f := range fixtures(t) {
		for _, n := range []int{1, 7, 64, 1 << 16} {
			rec := httptest.NewRecorder()
			o := &recObserver{}
			if err := proxy.StreamProxy(rec, httptest.NewRequest("POST", "/", nil), sseResponse(&piecewise{[]byte(f.SSE), n}), o); err != nil {
				t.Fatalf("%s/%d: %v", f.Name, n, err)
			}
			if rec.Body.String() != f.SSE {
				t.Fatalf("%s with %d-byte reads: bytes differ", f.Name, n)
			}
			if len(o.chunks) != f.Events {
				t.Fatalf("%s with %d-byte reads: %d events observed, want %d", f.Name, n, len(o.chunks), f.Events)
			}
			want := server.Usage{}
			if f.Usage != nil {
				want = *f.Usage
			}
			if o.usage != want {
				t.Fatalf("%s: usage %+v, want %+v", f.Name, o.usage, want)
			}
		}
	}
}

func TestPingCommentsPassThrough(t *testing.T) {
	// WHY: `: ping` keeps idle connections open; it must reach the client
	//      unchanged but is not an event, so observers never see it.
	// KIND: unit
	// CATCHES: s05
	// CHAPTER: gw.04 section 2.2
	body := ": ping\n\ndata: {\"x\":1}\n\n: ping\n\ndata: [DONE]\n\n"
	rec := httptest.NewRecorder()
	o := &recObserver{}
	proxy.StreamProxy(rec, httptest.NewRequest("POST", "/", nil), sseResponse(strings.NewReader(body)), o)
	if rec.Body.String() != body || len(o.chunks) != 2 || o.chunks[0] != `{"x":1}` {
		t.Fatalf("body %q, chunks %q", rec.Body.String(), o.chunks)
	}
}

func TestNonStreamUsage(t *testing.T) {
	// WHY: a non-streamed answer is one JSON body; its usage (total computed
	//      when the engine omits it) is what the limiter settles and the meter
	//      records. No events are reported.
	// KIND: unit
	// CATCHES: s11, s13
	// CHAPTER: gw.04 section 2.3
	body := `{"id":"c","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1}}`
	h := http.Header{"Content-Type": {"application/json"}}
	resp := &http.Response{StatusCode: 200, Header: h, Body: io.NopCloser(strings.NewReader(body)), Request: httptest.NewRequest("POST", "/", nil)}
	rec := httptest.NewRecorder()
	o := &recObserver{}
	if err := proxy.StreamProxy(rec, httptest.NewRequest("POST", "/", nil), resp, o); err != nil {
		t.Fatal(err)
	}
	if rec.Body.String() != body || len(o.chunks) != 0 || o.usage != (server.Usage{PromptTokens: 3, CompletionTokens: 1, TotalTokens: 4}) {
		t.Fatalf("body %q, chunks %d, usage %+v", rec.Body.String(), len(o.chunks), o.usage)
	}
}

func deadAddr(t *testing.T) string {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := l.Addr().String()
	l.Close()
	return "http://" + addr
}

func outbound(t *testing.T) *proxy.Outbound {
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"m","stream":true}`))
	r.Header.Set("Authorization", "Bearer secret")
	out, _ := proxy.NewOutbound(r, []byte(`{"model":"m","stream":true}`))
	return out
}

func TestForwardUncommittedOnRefused(t *testing.T) {
	// WHY: the commitment boundary routing's failover relies on: when the
	//      engine is unreachable, Forward writes nothing and says so
	//      (Committed = false), so gw.05 can try another worker.
	// KIND: fault
	// CATCHES: s06
	// CHAPTER: gw.04 section 2.4
	rec := httptest.NewRecorder()
	err := proxy.Forward(context.Background(), rec, deadAddr(t), outbound(t), nil, nil)
	var ue *proxy.UpstreamError
	if !errors.As(err, &ue) || ue.Committed {
		t.Fatalf("Forward to a dead engine = %v, want an uncommitted *UpstreamError", err)
	}
	if rec.Body.Len() != 0 || len(rec.Header()) != 0 || rec.Flushed {
		t.Fatalf("Forward wrote %q / %v before failing: nothing may be written", rec.Body.String(), rec.Header())
	}
}

func TestForwardRetry503(t *testing.T) {
	// WHY: an engine that answers 503 (draining, queue full) has done no work;
	//      with Retry503 the answer is handed back uncommitted for another
	//      worker, without it the 503 reaches the client unchanged.
	// KIND: unit
	// CATCHES: s07
	// CHAPTER: gw.04 section 2.4
	eng := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		server.WriteError(w, 503, "server_error", "no_capacity", "", "draining")
	}))
	defer eng.Close()
	out := outbound(t)
	out.Retry503 = true
	rec := httptest.NewRecorder()
	err := proxy.Forward(context.Background(), rec, eng.URL, out, nil, nil)
	var ue *proxy.UpstreamError
	if !errors.As(err, &ue) || ue.Committed || ue.StatusCode != 503 || rec.Body.Len() != 0 {
		t.Fatalf("Retry503: err %v, body %q", err, rec.Body.String())
	}
	out.Retry503 = false
	rec = httptest.NewRecorder()
	if err := proxy.Forward(context.Background(), rec, eng.URL, out, nil, nil); err != nil || rec.Code != 503 {
		t.Fatalf("without Retry503: err %v, status %d; want the 503 passed through", err, rec.Code)
	}
}

func TestNewOutboundDropsKey(t *testing.T) {
	// WHY: whatever builds the upstream request, the gateway's key never
	//      travels to the engine and the request id and trace go with it.
	// KIND: unit
	// CHAPTER: gw.04 section 4
	r := httptest.NewRequest("POST", "/v1/completions?x=1", nil)
	r.Header.Set("Authorization", "Bearer k")
	r.Header.Set("X-Request-Id", "req-9")
	r.Header.Set("Connection", "keep-alive")
	out, sp := proxy.NewOutbound(r, []byte("{}"))
	if out.Header.Get("Authorization") != "" || out.Header.Get("Connection") != "" || out.Header.Get("X-Request-Id") != "req-9" {
		t.Fatalf("headers %v", out.Header)
	}
	if out.Path != "/v1/completions" || out.RawQuery != "x=1" || !strings.Contains(out.Header.Get("Traceparent"), sp.SpanID) {
		t.Fatalf("outbound %+v, span %+v", out, sp)
	}
}

// engine streams events with a flush after each and waits on `step` between
// them, so a test can check that the client sees event k before k+1 exists.
func eventEngine(t *testing.T, events []string, step chan struct{}, cancelled chan time.Time) *httptest.Server {
	stop := make(chan struct{}) // closed at cleanup, before the server closes, so no handler outlives the test
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("X-Request-Id", "engine-made-this-up")
		for i, e := range events {
			if i > 0 {
				select {
				case <-step:
				case <-r.Context().Done():
					if cancelled != nil {
						cancelled <- time.Now()
					}
					return
				case <-stop:
					return
				case <-time.After(2 * patience): // a broken proxy must not hang the suite
					return
				}
			}
			io.WriteString(w, e)
			http.NewResponseController(w).Flush()
		}
	}))
	t.Cleanup(srv.Close)
	t.Cleanup(func() { close(stop) })
	return srv
}

// postStream posts body to url with the whole exchange bounded by patience,
// so a gateway that holds the response back fails the test instead of
// hanging it.
func postStream(t *testing.T, url, body string) *http.Response {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), patience)
	t.Cleanup(cancel)
	req, err := http.NewRequestWithContext(ctx, "POST", url, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST %s: %v (no response headers within %v: the proxy does not flush)", url, err, patience)
	}
	t.Cleanup(func() { resp.Body.Close() })
	return resp
}

func chainGateway(t *testing.T, upstream string, extra proxy.Config) (*httptest.Server, *server.Exchange) {
	var mu sync.Mutex
	var last *server.Exchange
	grab := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			last = server.ExchangeFrom(r.Context())
			mu.Unlock()
			next.ServeHTTP(w, r)
		})
	}
	extra.Upstream = upstream
	g := httptest.NewServer(server.New(server.Config{}, server.Deps{Ledger: grab, Proxy: proxy.Handler(extra)}).Handler())
	t.Cleanup(g.Close)
	return g, last
}

func TestStreamsEventByEvent(t *testing.T) {
	// WHY: through a real socket the client must read event 1 while the
	//      engine has not produced event 2: one flush per upstream read.
	// KIND: unit
	// CATCHES: s01
	// CHAPTER: gw.04 section 2.2
	step := make(chan struct{})
	eng := eventEngine(t, []string{"data: {\"n\":1}\n\n", "data: {\"n\":2}\n\n", "data: [DONE]\n\n"}, step, nil)
	g, _ := chainGateway(t, eng.URL, proxy.Config{})
	resp := postStream(t, g.URL+"/v1/chat/completions", `{"stream":true}`)
	br := bufio.NewReader(resp.Body)
	for i := 1; i <= 2; i++ {
		got := make(chan string, 1)
		go func() { l, _ := br.ReadString('\n'); br.ReadString('\n'); got <- l }()
		select {
		case l := <-got:
			if !strings.Contains(l, "\"n\":") {
				t.Fatalf("event %d: %q", i, l)
			}
		case <-time.After(patience):
			t.Fatalf("event %d did not reach the client while the engine waited: the proxy buffers", i)
		}
		select {
		case step <- struct{}{}:
		case <-time.After(patience):
			t.Fatal("the engine never asked for the next event")
		}
	}
}

func TestHandlerInChain(t *testing.T) {
	// WHY: as the chain's proxy stage, Handler needs no key (authn already
	//      ran), keeps the gateway's X-Request-Id over the engine's, and
	//      records the first byte and the usage on the Exchange for the
	//      limiter and the meter.
	// KIND: unit
	// CATCHES: s10, s12
	// CHAPTER: gw.04 section 4
	eng := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Request-Id") != "req-1" {
			t.Errorf("engine got X-Request-Id %q", r.Header.Get("X-Request-Id"))
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("X-Request-Id", "engine-made-this-up")
		io.WriteString(w, handStream)
	}))
	defer eng.Close()
	var ex *server.Exchange
	grab := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			ex = server.ExchangeFrom(r.Context())
			next.ServeHTTP(w, r)
		})
	}
	h := server.New(server.Config{}, server.Deps{Ledger: grab, Proxy: proxy.Handler(proxy.Config{Upstream: eng.URL})}).Handler()
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"stream":true}`))
	req.Header.Set("X-Request-Id", "req-1")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != 200 || rec.Body.String() != handStream {
		t.Fatalf("no key needed in the chain: got %d %q", rec.Code, rec.Body.String())
	}
	if ids := rec.Header().Values("X-Request-Id"); len(ids) != 1 || ids[0] != "req-1" {
		t.Fatalf("X-Request-Id %q, want exactly the gateway's req-1 (the engine's copy must not be added)", ids)
	}
	st := ex.Snapshot()
	if st.Usage.TotalTokens != 7 || st.FirstByte.IsZero() || st.Status != 200 {
		t.Fatalf("Exchange %+v: usage 7, first byte and status recorded", st)
	}
}

func TestHandlerUpstreamDown503(t *testing.T) {
	// WHY: before the first byte an unreachable engine is a 503 no_capacity
	//      in the error shape, the same answer gw.00 gave.
	// KIND: boundary
	// CHAPTER: gw.04 section 2.4
	h := proxy.Handler(proxy.Config{Upstream: deadAddr(t)})
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{}`)))
	if rec.Code != 503 || !strings.Contains(rec.Body.String(), `"no_capacity"`) {
		t.Fatalf("got %d %q", rec.Code, rec.Body.String())
	}
}

func TestChaosResetMidStream(t *testing.T) {
	// WHY: a connection reset after the first event (chaos proxy) must end
	//      the client's stream with one SSE error event, never a silent
	//      truncation, and the observer must not report a clean finish.
	// KIND: fault
	// CATCHES: s08
	// CHAPTER: gw.04 section 5, Pitfalls, item 5
	step := make(chan struct{}, 4)
	first := "data: {\"n\":1}\n\n"
	eng := eventEngine(t, []string{first, "data: {\"n\":2}\n\n", "data: [DONE]\n\n"}, step, nil)
	host := strings.TrimPrefix(eng.URL, "http://")
	cp, err := chaosproxy.Start(host, chaosproxy.Faults{})
	if err != nil {
		t.Fatal(err)
	}
	defer cp.Close()
	o := &recObserver{}
	h := proxy.Handler(proxy.Config{Upstream: "http://" + cp.Addr(), Observer: func(*http.Request) proxy.StreamObserver { return o },
		Client: &http.Client{Transport: &http.Transport{DisableKeepAlives: true}}})
	g := httptest.NewServer(h)
	defer g.Close()
	resp := postStream(t, g.URL+"/v1/chat/completions", `{"stream":true}`)
	br := bufio.NewReader(resp.Body)
	if l, _ := br.ReadString('\n'); !strings.Contains(l, `"n":1`) {
		t.Fatalf("first line %q", l)
	}
	cp.SetFaults(chaosproxy.Faults{ResetAfterBytes: 1})
	step <- struct{}{}
	step <- struct{}{}
	rest, _ := io.ReadAll(br)
	if !strings.Contains(string(rest), `data: {"error":`) || strings.Contains(string(rest), "[DONE]") {
		t.Fatalf("after the reset the client read %q; want an SSE error event and no [DONE]", rest)
	}
	if o.done != 0 {
		t.Fatal("OnDone was called for a stream that broke")
	}
}

func TestClientDisconnectCancelsUpstream(t *testing.T) {
	// WHY: a client that hangs up must cancel the engine's request (the
	//      contract says within 100 ms; this functional check allows 1 s on a
	//      loaded machine), or the engine keeps generating for nobody.
	// KIND: fault
	// CATCHES: s09
	// CHAPTER: gw.04 section 2.4
	step := make(chan struct{})
	cancelled := make(chan time.Time, 1)
	eng := eventEngine(t, []string{"data: {\"n\":1}\n\n", "data: [DONE]\n\n"}, step, cancelled)
	g, _ := chainGateway(t, eng.URL, proxy.Config{})
	ctx, cancel := context.WithTimeout(context.Background(), patience)
	req, _ := http.NewRequestWithContext(ctx, "POST", g.URL+"/v1/chat/completions", strings.NewReader(`{"stream":true}`))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	bufio.NewReader(resp.Body).ReadString('\n')
	at := time.Now()
	cancel()
	resp.Body.Close()
	select {
	case when := <-cancelled:
		if d := when.Sub(at); d > time.Second {
			t.Fatalf("the engine saw the cancellation after %v", d)
		}
	case <-time.After(patience):
		t.Fatal("the engine's request was never cancelled after the client left")
	}
}

func BenchmarkFirstByteOverhead(b *testing.B) {
	// WHY: the gateway may add at most a few milliseconds to time to first
	//      token (DESIGN gw.04: within 5 ms of upstream); `ol bench gw.04
	//      --assert` checks the reported overhead_ms.
	// KIND: bench
	// CHAPTER: gw.04 section 4
	eng := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: {\"n\":1}\n\n")
		http.NewResponseController(w).Flush()
	}))
	defer eng.Close()
	g := httptest.NewServer(proxy.Handler(proxy.Config{Upstream: eng.URL}))
	defer g.Close()
	ttfb := func(url string) time.Duration {
		t0 := time.Now()
		resp, err := http.Post(url+"/v1/chat/completions", "application/json", strings.NewReader(`{}`))
		if err != nil {
			b.Fatal(err)
		}
		bufio.NewReader(resp.Body).ReadString('\n')
		d := time.Since(t0)
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		return d
	}
	var direct, via time.Duration
	for i := 0; i < b.N; i++ {
		direct += ttfb(eng.URL)
		via += ttfb(g.URL)
	}
	b.ReportMetric(float64(via-direct)/float64(b.N)/1e6, "overhead_ms")
}
