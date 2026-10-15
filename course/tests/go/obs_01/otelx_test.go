// Course tests for obs.01, tracing for the serving path (go/otelx).
//
// Every test builds its own TracerProvider whose spans land in a
// tracetest.SpanRecorder, so nothing depends on the OpenTelemetry globals or
// on another test. Servers are real: net/http/httptest for HTTP, gRPC over an
// in-memory bufconn listener (the standard health service stands in for
// tl.engine.v1.EngineControl), and a fake OTLP/gRPC collector for export.
// Nothing sleeps for a fixed time: waits poll with a deadline (`patience`),
// so a correct otelx passes in milliseconds and a broken one fails with a
// message instead of hanging.
//
// The span names, kinds, attributes, and the propagation rules are
// course/contracts/otel/semconv.md; the chapter is
// content/tracks/systems/04-observability/01-tracing-for-the-serving-path.md.
package obs_01

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
	collectortrace "go.opentelemetry.io/proto/otlp/collector/trace/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/test/bufconn"

	"tinyllm/otelx"
)

const (
	patience = 3 * time.Second
	// The W3C Trace Context example (https://www.w3.org/TR/trace-context/),
	// the worked example of the chapter's section 3.
	w3cHeader    = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
	callerTrace  = "4bf92f3577b34da6a3ce929d0e0e4736"
	callerParent = "00f067aa0ba902b7"
	chatRoute    = "/v1/chat/completions"
)

// -- helpers -------------------------------------------------------------------

// newTP is a provider that records every span (no sampling, no export).
func newTP(t *testing.T) (*sdktrace.TracerProvider, *tracetest.SpanRecorder) {
	t.Helper()
	rec := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.AlwaysSample()), sdktrace.WithSpanProcessor(rec))
	t.Cleanup(func() { _ = tp.Shutdown(context.Background()) })
	return tp, rec
}

// ended waits until the recorder holds n ended spans and returns them.
func ended(t *testing.T, rec *tracetest.SpanRecorder, n int) []sdktrace.ReadOnlySpan {
	t.Helper()
	deadline := time.Now().Add(patience)
	for {
		got := rec.Ended()
		if len(got) >= n || time.Now().After(deadline) {
			if len(got) < n {
				t.Fatalf("want %d ended spans, got %d: %s", n, len(got), describe(got))
			}
			return got
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func describe(spans []sdktrace.ReadOnlySpan) string {
	var b strings.Builder
	for _, s := range spans {
		b.WriteString("\n  " + s.Name() + " kind=" + s.SpanKind().String() +
			" trace=" + s.SpanContext().TraceID().String()[:8] + ".. span=" + s.SpanContext().SpanID().String() +
			" parent=" + s.Parent().SpanID().String())
	}
	if b.Len() == 0 {
		return "(none)"
	}
	return b.String()
}

// byName returns the one span with this name and kind.
func byName(t *testing.T, spans []sdktrace.ReadOnlySpan, name string, kind trace.SpanKind) sdktrace.ReadOnlySpan {
	t.Helper()
	var found []sdktrace.ReadOnlySpan
	for _, s := range spans {
		if s.Name() == name && s.SpanKind() == kind {
			found = append(found, s)
		}
	}
	if len(found) != 1 {
		t.Fatalf("want exactly one %s span named %q, got %d; spans: %s", kind, name, len(found), describe(spans))
	}
	return found[0]
}

func attrs(s sdktrace.ReadOnlySpan) map[attribute.Key]attribute.Value {
	m := map[attribute.Key]attribute.Value{}
	for _, kv := range s.Attributes() {
		m[kv.Key] = kv.Value
	}
	return m
}

// wantAttr checks a span attribute by its printed value (Emit), and its type.
func wantAttr(t *testing.T, s sdktrace.ReadOnlySpan, key string, typ attribute.Type, want string) {
	t.Helper()
	v, ok := attrs(s)[attribute.Key(key)]
	if !ok {
		t.Errorf("%s: no attribute %s (have %v)", s.Name(), key, s.Attributes())
		return
	}
	if v.Type() != typ || v.Emit() != want {
		t.Errorf("%s: %s = %s (%s); want %s (%s)", s.Name(), key, v.Emit(), v.Type(), want, typ)
	}
}

func routeOf(pattern string) func(*http.Request) string {
	return func(*http.Request) string { return pattern }
}

// grpcPair is a health server behind the server interceptors and a client
// connection through the client interceptors, over an in-memory listener.
// extra server interceptors run after otelx's, inside its span.
func grpcPair(t *testing.T, tp trace.TracerProvider, extra ...grpc.UnaryServerInterceptor) (healthpb.HealthClient, *health.Server) {
	t.Helper()
	lis := bufconn.Listen(1 << 20)
	srv := grpc.NewServer(
		grpc.ChainUnaryInterceptor(append([]grpc.UnaryServerInterceptor{otelx.UnaryServerInterceptor(tp)}, extra...)...),
		grpc.StreamInterceptor(otelx.StreamServerInterceptor(tp)),
	)
	hs := health.NewServer() // "" is SERVING; any other service is NotFound
	healthpb.RegisterHealthServer(srv, hs)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)
	cc, err := grpc.NewClient("passthrough:///bufnet",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) { return lis.DialContext(ctx) }),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithUnaryInterceptor(otelx.UnaryClientInterceptor(tp)),
		grpc.WithStreamInterceptor(otelx.StreamClientInterceptor(tp)),
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = cc.Close() })
	return healthpb.NewHealthClient(cc), hs
}

// -- propagation -----------------------------------------------------------------

func TestTraceparentHandExample(t *testing.T) {
	// WHY: the chapter's worked example (section 3): the W3C header parsed
	//      into ids, written back unchanged, and a child span that keeps the
	//      trace id, takes a new span id, and keeps the sampled flag. Every
	//      hop of the serving path does exactly this.
	// KIND: unit
	// CATCHES: s11
	ctx := otelx.ContextWithTraceparent(context.Background(), w3cHeader)
	sc := trace.SpanContextFromContext(ctx)
	if !sc.IsValid() || !sc.IsRemote() || !sc.IsSampled() {
		t.Fatalf("parsed span context valid=%v remote=%v sampled=%v; want all true", sc.IsValid(), sc.IsRemote(), sc.IsSampled())
	}
	if sc.TraceID().String() != callerTrace || sc.SpanID().String() != callerParent {
		t.Fatalf("parsed trace %s span %s; want %s %s", sc.TraceID(), sc.SpanID(), callerTrace, callerParent)
	}
	if got := otelx.Traceparent(ctx); got != w3cHeader {
		t.Fatalf("Traceparent = %q; want %q", got, w3cHeader)
	}
	tid, sid, ok := otelx.IDs(ctx)
	if !ok || tid != callerTrace || sid != callerParent {
		t.Fatalf("IDs = %q %q %v; want %s %s true", tid, sid, ok, callerTrace, callerParent)
	}
	tp, _ := newTP(t)
	child, span := tp.Tracer("test").Start(ctx, "engine.decode")
	defer span.End()
	want := "00-" + callerTrace + "-" + span.SpanContext().SpanID().String() + "-01"
	if got := otelx.Traceparent(child); got != want {
		t.Fatalf("a child's Traceparent = %q; want %q (same trace, its own span id, sampled)", got, want)
	}
	// Not sampled upstream: the flag must travel too, or downstream records
	// what the caller decided to drop.
	unsampled := w3cHeader[:len(w3cHeader)-2] + "00"
	if got := otelx.Traceparent(otelx.ContextWithTraceparent(context.Background(), unsampled)); got != unsampled {
		t.Fatalf("Traceparent of an unsampled context = %q; want %q", got, unsampled)
	}
}

func TestInvalidTraceparentIsIgnored(t *testing.T) {
	// WHY: a malformed header must start a new trace, never a trace with a
	//      garbage or all-zero id that would merge unrelated requests.
	// KIND: boundary
	// CATCHES: s26
	for _, bad := range []string{
		"",
		"00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7",    // no flags
		"ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01", // version ff is forbidden
		"00-00000000000000000000000000000000-00f067aa0ba902b7-01", // all-zero trace id
		"00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01", // all-zero span id
		"00-4BF92F3577B34DA6A3CE929D0E0E4736-00F067AA0BA902B7-01", // upper-case hex
		"00-4bf92f3577b34da6a3ce929d0e0e473-00f067aa0ba902b7-01",  // 31 hex digits
	} {
		ctx := otelx.ContextWithTraceparent(context.Background(), bad)
		if _, _, ok := otelx.IDs(ctx); ok {
			t.Errorf("%q was accepted as a span context", bad)
		}
		if got := otelx.Traceparent(ctx); got != "" {
			t.Errorf("%q: Traceparent = %q; want \"\"", bad, got)
		}
	}
}

// -- HTTP server -------------------------------------------------------------------

func TestMiddlewareContinuesCallerTrace(t *testing.T) {
	// WHY: the server span of a request is the caller's child: same trace id,
	//      the caller's span id as parent. Without Extract the gateway starts
	//      a new trace on every request and a traced client's tree stops at
	//      its own span.
	// KIND: conformance
	// CATCHES: s01
	tp, rec := newTP(t)
	var inHandler [2]string
	h := otelx.Middleware(tp, routeOf(chatRoute), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		inHandler[0], inHandler[1], _ = otelx.IDs(r.Context())
		_, _ = io.WriteString(w, `{"ok":true}`)
	}))
	srv := httptest.NewServer(h)
	defer srv.Close()
	req, _ := http.NewRequest("POST", srv.URL+chatRoute, strings.NewReader(`{}`))
	req.Header.Set("traceparent", w3cHeader)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	s := byName(t, ended(t, rec, 1), "POST "+chatRoute, trace.SpanKindServer)
	if s.SpanContext().TraceID().String() != callerTrace || s.Parent().SpanID().String() != callerParent || !s.Parent().IsRemote() {
		t.Fatalf("server span trace %s parent %s (remote %v); want trace %s, remote parent %s",
			s.SpanContext().TraceID(), s.Parent().SpanID(), s.Parent().IsRemote(), callerTrace, callerParent)
	}
	if inHandler[0] != callerTrace || inHandler[1] != s.SpanContext().SpanID().String() {
		t.Fatalf("the handler's context holds trace %s span %s; want the server span (%s %s)",
			inHandler[0], inHandler[1], callerTrace, s.SpanContext().SpanID())
	}
	wantAttr(t, s, "http.request.method", attribute.STRING, "POST")
	wantAttr(t, s, "http.route", attribute.STRING, chatRoute)
	wantAttr(t, s, "http.response.status_code", attribute.INT64, "200")
}

func TestMiddlewareStartsNewTraceWithoutHeader(t *testing.T) {
	// WHY: a request with no traceparent is the root of a new trace: a valid
	//      random trace id and no parent.
	// KIND: unit
	tp, rec := newTP(t)
	srv := httptest.NewServer(otelx.Middleware(tp, routeOf(chatRoute), http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})))
	defer srv.Close()
	resp, err := http.Post(srv.URL+chatRoute, "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	s := byName(t, ended(t, rec, 1), "POST "+chatRoute, trace.SpanKindServer)
	if !s.SpanContext().IsValid() || s.Parent().IsValid() {
		t.Fatalf("want a root span with a valid trace id; got trace %s parent %s", s.SpanContext().TraceID(), s.Parent().SpanID())
	}
	wantAttr(t, s, "http.response.status_code", attribute.INT64, "200")
}

func TestMiddlewareKeepsSSEStreaming(t *testing.T) {
	// WHY: the engine and the gateway stream tokens as SSE, one Flush per
	//      event. A wrapper ResponseWriter that hides http.Flusher buffers the
	//      whole stream: TTFT becomes the full generation time. The handler
	//      below type-asserts http.Flusher the way your proxy does, and the
	//      client must read the first event while the handler still waits.
	// KIND: unit
	// CATCHES: s03
	tp, _ := newTP(t)
	release := make(chan struct{})
	notFlusher := make(chan struct{}, 1)
	h := otelx.Middleware(tp, routeOf(chatRoute), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"tok\":1}\n\n")
		f, ok := w.(http.Flusher)
		if !ok {
			notFlusher <- struct{}{}
		} else {
			f.Flush()
		}
		select {
		case <-release:
		case <-time.After(patience):
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	srv := httptest.NewServer(h)
	defer srv.Close()
	defer close(release)
	resp, err := http.Post(srv.URL+chatRoute, "application/json", strings.NewReader(`{"stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	first := make(chan string, 1)
	go func() {
		line, _ := bufio.NewReader(resp.Body).ReadString('\n')
		first <- line
	}()
	select {
	case <-notFlusher:
		t.Fatal("the ResponseWriter your middleware hands on is not an http.Flusher: every SSE stream behind it is buffered")
	case line := <-first:
		if !strings.HasPrefix(line, "data: {") {
			t.Fatalf("first line %q; want the first event", line)
		}
	case <-time.After(patience / 2):
		t.Fatal("the first event did not reach the client while the handler was still streaming: the response is buffered")
	}
}

func TestMiddlewareSupportsResponseController(t *testing.T) {
	// WHY: http.NewResponseController reaches the real writer through
	//      Unwrap; without it a handler cannot set a write deadline for a
	//      slow streaming client (http.ErrNotSupported).
	// KIND: unit
	// CATCHES: s14
	tp, _ := newTP(t)
	errc := make(chan error, 1)
	srv := httptest.NewServer(otelx.Middleware(tp, nil, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		errc <- http.NewResponseController(w).SetWriteDeadline(time.Now().Add(time.Minute))
	})))
	defer srv.Close()
	resp, err := http.Get(srv.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if err := <-errc; err != nil {
		t.Fatalf("SetWriteDeadline through your middleware: %v; implement Unwrap() http.ResponseWriter", err)
	}
}

func TestMiddlewareStatusAndErrors(t *testing.T) {
	// WHY: the status code is on every server span, and only server faults
	//      (>= 500) mark the span as an error: a 404 or 401 is the client's
	//      mistake, and counting it as an error would page someone for bad
	//      requests. A handler that writes a body without WriteHeader, or
	//      nothing at all, answered 200.
	// KIND: boundary
	// CATCHES: s05, m01, m02
	cases := []struct {
		name  string
		write func(http.ResponseWriter)
		code  string
		err   bool
	}{
		{"body only", func(w http.ResponseWriter) { _, _ = io.WriteString(w, "ok") }, "200", false},
		{"nothing written", func(http.ResponseWriter) {}, "200", false},
		{"404", func(w http.ResponseWriter) { w.WriteHeader(404) }, "404", false},
		{"499", func(w http.ResponseWriter) { w.WriteHeader(499) }, "499", false},
		{"500", func(w http.ResponseWriter) { w.WriteHeader(500) }, "500", true},
		{"503", func(w http.ResponseWriter) { w.WriteHeader(503) }, "503", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tp, rec := newTP(t)
			srv := httptest.NewServer(otelx.Middleware(tp, routeOf(chatRoute), http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { tc.write(w) })))
			defer srv.Close()
			resp, err := http.Post(srv.URL+chatRoute, "application/json", nil)
			if err != nil {
				t.Fatal(err)
			}
			resp.Body.Close()
			s := byName(t, ended(t, rec, 1), "POST "+chatRoute, trace.SpanKindServer)
			wantAttr(t, s, "http.response.status_code", attribute.INT64, tc.code)
			if tc.err {
				if s.Status().Code != codes.Error {
					t.Errorf("status %s: span status %v; want Error", tc.code, s.Status().Code)
				}
				wantAttr(t, s, "error.type", attribute.STRING, tc.code)
			} else if s.Status().Code == codes.Error {
				t.Errorf("status %s: span status Error; a client error or success leaves it unset", tc.code)
			}
		})
	}
}

func TestMiddlewareNamesSpansByRoute(t *testing.T) {
	// WHY: span names and http.route are low-cardinality: the route pattern,
	//      never the raw path. "/v1/models/smol-135m" in a name makes one span
	//      name per model (per user id, per request id elsewhere), which
	//      breaks grouping in Tempo and every span-metrics pipeline.
	// KIND: unit
	// CATCHES: s04
	tp, rec := newTP(t)
	route := func(r *http.Request) string {
		if strings.HasPrefix(r.URL.Path, "/v1/models/") {
			return "/v1/models/{model}"
		}
		return ""
	}
	srv := httptest.NewServer(otelx.Middleware(tp, route, http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})))
	defer srv.Close()
	for _, p := range []string{"/v1/models/smol-135m", "/v1/models/tinystories-10m", "/unknown/abc123"} {
		resp, err := http.Get(srv.URL + p)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
	}
	spans := ended(t, rec, 3)
	names := map[string]int{}
	for _, s := range spans {
		names[s.Name()]++
		if strings.Contains(s.Name(), "smol") || strings.Contains(s.Name(), "abc123") {
			t.Errorf("span name %q contains the raw path", s.Name())
		}
	}
	if names["GET /v1/models/{model}"] != 2 || names["GET"] != 1 {
		t.Fatalf("span names %v; want 2 x \"GET /v1/models/{model}\" and 1 x \"GET\" (no route known)", names)
	}
	wantAttr(t, byName(t, spans, "GET", trace.SpanKindServer), "http.request.method", attribute.STRING, "GET")
	if _, ok := attrs(byName(t, spans, "GET", trace.SpanKindServer))["http.route"]; ok {
		t.Errorf("a request with no route must not carry http.route")
	}
}

// -- HTTP client -------------------------------------------------------------------

func TestTransportInjectsItsOwnSpan(t *testing.T) {
	// WHY: an outgoing call is a CLIENT span, the child of the caller's span,
	//      and the traceparent it sends names THAT client span as the parent.
	//      Forwarding the caller's own context instead makes the downstream
	//      server a sibling of the client span: the tree loses a level and
	//      the network time between the two is invisible.
	// KIND: conformance
	// CATCHES: s02, s10
	tp, rec := newTP(t)
	got := make(chan string, 1)
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got <- r.Header.Get("traceparent")
		w.WriteHeader(201)
	}))
	defer up.Close()
	ctx, parent := tp.Tracer("test").Start(context.Background(), "gateway.route")
	req, _ := http.NewRequestWithContext(ctx, "POST", up.URL+chatRoute, strings.NewReader(`{}`))
	resp, err := (&http.Client{Transport: otelx.Transport(tp, nil)}).Do(req)
	parent.End()
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if req.Header.Get("traceparent") != "" {
		t.Errorf("Transport added traceparent to the caller's request; a RoundTripper must not modify its input (clone it)")
	}
	spans := ended(t, rec, 2)
	c := byName(t, spans, "POST", trace.SpanKindClient)
	if c.Parent().SpanID() != parent.SpanContext().SpanID() {
		t.Fatalf("client span parent %s; want the caller's span %s", c.Parent().SpanID(), parent.SpanContext().SpanID())
	}
	want := "00-" + c.SpanContext().TraceID().String() + "-" + c.SpanContext().SpanID().String() + "-01"
	if h := <-got; h != want {
		t.Fatalf("upstream saw traceparent %q; want %q (the client span as parent)", h, want)
	}
	wantAttr(t, c, "http.request.method", attribute.STRING, "POST")
	wantAttr(t, c, "server.address", attribute.STRING, "127.0.0.1")
	wantAttr(t, c, "http.response.status_code", attribute.INT64, "201")
	if _, ok := attrs(c)["server.port"]; !ok {
		t.Errorf("client span has no server.port")
	}
}

func TestTransportSpanCoversTheStreamedBody(t *testing.T) {
	// WHY: a streamed completion's call is not over when the headers arrive.
	//      A client span that ends at RoundTrip's return measures 1 ms for a
	//      10 s stream, and the server span below it sticks out of its
	//      parent. It ends when the body reaches EOF or is closed.
	// KIND: unit
	// CATCHES: s19
	tp, rec := newTP(t)
	release := make(chan struct{})
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "data: {\"tok\":1}\n\n")
		w.(http.Flusher).Flush()
		select {
		case <-release:
		case <-time.After(patience):
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer up.Close()
	req, _ := http.NewRequest("POST", up.URL+chatRoute, nil)
	resp, err := (&http.Client{Transport: otelx.Transport(tp, nil)}).Do(req)
	if err != nil {
		t.Fatal(err)
	}
	br := bufio.NewReader(resp.Body)
	if _, err := br.ReadString('\n'); err != nil {
		t.Fatal(err)
	}
	if n := len(rec.Ended()); n != 0 {
		close(release)
		t.Fatalf("the client span ended after the first event (%d ended); it must last until the body is done", n)
	}
	close(release)
	_, _ = io.Copy(io.Discard, br)
	resp.Body.Close()
	c := byName(t, ended(t, rec, 1), "POST", trace.SpanKindClient)
	if d := c.EndTime().Sub(c.StartTime()); d <= 0 {
		t.Fatalf("client span duration %v", d)
	}
}

func TestTransportRecordsConnectionErrors(t *testing.T) {
	// WHY: when no response comes back (the engine is down), the client span
	//      still ends, with status Error and an error.type: that span is the
	//      only evidence in the trace of a failover attempt (gw.05).
	// KIND: fault
	// CATCHES: s25
	tp, rec := newTP(t)
	lis, _ := net.Listen("tcp", "127.0.0.1:0")
	addr := lis.Addr().String()
	lis.Close() // nothing listens there now: connection refused
	req, _ := http.NewRequest("GET", "http://"+addr+"/healthz", nil)
	if _, err := (&http.Client{Transport: otelx.Transport(tp, nil), Timeout: patience}).Do(req); err == nil {
		t.Fatal("want a connection error")
	}
	c := byName(t, ended(t, rec, 1), "GET", trace.SpanKindClient)
	if c.Status().Code != codes.Error {
		t.Errorf("span status %v; want Error", c.Status().Code)
	}
	if v, ok := attrs(c)["error.type"]; !ok || v.AsString() == "" {
		t.Errorf("no error.type on the failed client span")
	}
}

// -- gRPC -------------------------------------------------------------------------

func TestGRPCUnaryPropagation(t *testing.T) {
	// WHY: the gateway reaches prefill engines over gRPC
	//      (tl.engine.v1.EngineControl/Prefill): the CLIENT span's traceparent
	//      travels in metadata and the server's span is its child, with the
	//      rpc.* attributes. Metadata the caller set (request ids) must survive.
	// KIND: conformance
	// CATCHES: s08, s09, s12, s22
	tp, rec := newTP(t)
	seen := make(chan metadata.MD, 1)
	client, _ := grpcPair(t, tp, func(ctx context.Context, req any, _ *grpc.UnaryServerInfo, h grpc.UnaryHandler) (any, error) {
		m, _ := metadata.FromIncomingContext(ctx)
		seen <- m
		return h(ctx, req)
	})
	ctx, root := tp.Tracer("test").Start(context.Background(), "gateway.route")
	ctx = metadata.AppendToOutgoingContext(ctx, "x-request-id", "req-7")
	resp, err := client.Check(ctx, &healthpb.HealthCheckRequest{})
	root.End()
	if err != nil || resp.GetStatus() != healthpb.HealthCheckResponse_SERVING {
		t.Fatalf("Check: %v %v", resp, err)
	}
	spans := ended(t, rec, 3)
	name := "grpc.health.v1.Health/Check"
	c := byName(t, spans, name, trace.SpanKindClient)
	s := byName(t, spans, name, trace.SpanKindServer)
	if c.Parent().SpanID() != root.SpanContext().SpanID() {
		t.Errorf("client span parent %s; want the caller %s", c.Parent().SpanID(), root.SpanContext().SpanID())
	}
	if s.SpanContext().TraceID() != c.SpanContext().TraceID() || s.Parent().SpanID() != c.SpanContext().SpanID() {
		t.Fatalf("server span trace %s parent %s; want trace %s parent %s (the client span)",
			s.SpanContext().TraceID(), s.Parent().SpanID(), c.SpanContext().TraceID(), c.SpanContext().SpanID())
	}
	for _, sp := range []sdktrace.ReadOnlySpan{c, s} {
		wantAttr(t, sp, "rpc.system", attribute.STRING, "grpc")
		wantAttr(t, sp, "rpc.service", attribute.STRING, "grpc.health.v1.Health")
		wantAttr(t, sp, "rpc.method", attribute.STRING, "Check")
		wantAttr(t, sp, "rpc.grpc.status_code", attribute.INT64, "0")
	}
	m := <-seen
	if got := m.Get("x-request-id"); len(got) != 1 || got[0] != "req-7" {
		t.Errorf("server metadata x-request-id = %v; the interceptor must add traceparent to the caller's metadata, not replace it", got)
	}
	if got := m.Get("traceparent"); len(got) != 1 {
		t.Errorf("server metadata traceparent = %v; want exactly one", got)
	}
}

func TestGRPCErrorsMarkSpans(t *testing.T) {
	// WHY: a failed RPC (NotFound, Unavailable during a prefill pool outage)
	//      marks both spans as errors with rpc.grpc.status_code and the code's
	//      name as error.type, so the failing hop is red in the trace.
	// KIND: boundary
	// CATCHES: s20
	tp, rec := newTP(t)
	client, _ := grpcPair(t, tp)
	_, err := client.Check(context.Background(), &healthpb.HealthCheckRequest{Service: "tl.engine.v1.EngineControl"})
	if err == nil {
		t.Fatal("want NotFound for an unregistered service")
	}
	spans := ended(t, rec, 2)
	for _, kind := range []trace.SpanKind{trace.SpanKindClient, trace.SpanKindServer} {
		s := byName(t, spans, "grpc.health.v1.Health/Check", kind)
		if s.Status().Code != codes.Error {
			t.Errorf("%s span status %v; want Error", kind, s.Status().Code)
		}
		wantAttr(t, s, "rpc.grpc.status_code", attribute.INT64, "5")
		wantAttr(t, s, "error.type", attribute.STRING, "NotFound")
	}
}

func TestGRPCStreamPropagation(t *testing.T) {
	// WHY: KV transfer and heartbeats are streams. The stream interceptors
	//      carry the same context: the server span of a stream is the client
	//      span's child, and both end when the stream does.
	// KIND: conformance
	// CATCHES: s13
	tp, rec := newTP(t)
	client, _ := grpcPair(t, tp)
	ctx, cancel := context.WithCancel(context.Background())
	stream, err := client.Watch(ctx, &healthpb.HealthCheckRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if m, err := stream.Recv(); err != nil || m.GetStatus() != healthpb.HealthCheckResponse_SERVING {
		t.Fatalf("first Watch message: %v %v", m, err)
	}
	cancel() // the client ends the stream
	_, _ = stream.Recv()
	spans := ended(t, rec, 2)
	c := byName(t, spans, "grpc.health.v1.Health/Watch", trace.SpanKindClient)
	s := byName(t, spans, "grpc.health.v1.Health/Watch", trace.SpanKindServer)
	if s.Parent().SpanID() != c.SpanContext().SpanID() || s.SpanContext().TraceID() != c.SpanContext().TraceID() {
		t.Fatalf("stream server span parent %s; want the client span %s", s.Parent().SpanID(), c.SpanContext().SpanID())
	}
}

// -- the serving tree ----------------------------------------------------------------

func TestServingTreeAcrossHTTPAndGRPC(t *testing.T) {
	// WHY: the promise of this module (semconv.md, The serving trace): one
	//      request is one tree across HTTP and gRPC. A "gateway" handler calls
	//      a "prefill" engine over gRPC and a "decode" engine over HTTP; every
	//      span shares the trace id and each hop hangs under the right parent.
	// KIND: conformance
	// CATCHES: s01, s02, s08, s09
	tp, rec := newTP(t)
	prefill, _ := grpcPair(t, tp)
	decode := httptest.NewServer(otelx.Middleware(tp, routeOf(chatRoute), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	})))
	defer decode.Close()
	upstream := &http.Client{Transport: otelx.Transport(tp, nil)}
	gateway := httptest.NewServer(otelx.Middleware(tp, routeOf(chatRoute), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, err := prefill.Check(r.Context(), &healthpb.HealthCheckRequest{}); err != nil {
			http.Error(w, err.Error(), 502)
			return
		}
		req, _ := http.NewRequestWithContext(r.Context(), "POST", decode.URL+chatRoute, strings.NewReader(`{}`))
		resp, err := upstream.Do(req)
		if err != nil {
			http.Error(w, err.Error(), 502)
			return
		}
		defer resp.Body.Close()
		_, _ = io.Copy(w, resp.Body)
	})))
	defer gateway.Close()
	resp, err := http.Post(gateway.URL+chatRoute, "application/json", strings.NewReader(`{"stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	spans := ended(t, rec, 5)
	var root sdktrace.ReadOnlySpan
	servers := 0
	for _, s := range spans {
		if s.Name() == "POST "+chatRoute && s.SpanKind() == trace.SpanKindServer {
			servers++
			if !s.Parent().IsValid() {
				root = s
			}
		}
	}
	if root == nil || servers != 2 {
		t.Fatalf("want two server spans %q, one of them the root; spans: %s", "POST "+chatRoute, describe(spans))
	}
	tid := root.SpanContext().TraceID()
	for _, s := range spans {
		if s.SpanContext().TraceID() != tid {
			t.Fatalf("span %s is in trace %s, not the request's trace %s: %s", s.Name(), s.SpanContext().TraceID(), tid, describe(spans))
		}
	}
	gc := byName(t, spans, "grpc.health.v1.Health/Check", trace.SpanKindClient)
	gs := byName(t, spans, "grpc.health.v1.Health/Check", trace.SpanKindServer)
	hc := byName(t, spans, "POST", trace.SpanKindClient)
	var ds sdktrace.ReadOnlySpan
	for _, s := range spans {
		if s.Name() == "POST "+chatRoute && s != root {
			ds = s
		}
	}
	for _, edge := range []struct {
		child, parent sdktrace.ReadOnlySpan
		what          string
	}{
		{gc, root, "the gRPC client span under the gateway's server span"},
		{gs, gc, "the prefill server span under the gRPC client span"},
		{hc, root, "the HTTP client span under the gateway's server span"},
		{ds, hc, "the decode server span under the HTTP client span"},
	} {
		if edge.child.Parent().SpanID() != edge.parent.SpanContext().SpanID() {
			t.Errorf("want %s; tree: %s", edge.what, describe(spans))
		}
	}
}

// -- the gateway chain adapter -------------------------------------------------------

func TestStartServerAdapterKeepsAttributeTypes(t *testing.T) {
	// WHY: the gateway chain (gw.01) records its attributes through
	//      Span.SetAttribute(key, any). Status codes stay integers, cache hits
	//      booleans, finish reasons string arrays: a backend that receives
	//      "200" as a string cannot compare or aggregate it.
	// KIND: unit
	// CATCHES: s15
	tp, rec := newTP(t)
	ctx := otelx.ContextWithTraceparent(context.Background(), w3cHeader)
	ctx, sp := otelx.StartServer(ctx, tp, "POST "+chatRoute)
	sp.SetAttribute("http.response.status_code", 200)
	sp.SetAttribute("tl.cache.hit", false)
	sp.SetAttribute("tl.proxy.first_byte_ms", 12.5)
	sp.SetAttribute("gen_ai.response.finish_reasons", []string{"stop"})
	sp.SetAttribute("tl.route.worker_id", "decode-1")
	sp.SetAttribute("tl.route.cascade_step", int64(2))
	_, sid, _ := otelx.IDs(ctx)
	sp.End()
	s := byName(t, ended(t, rec, 1), "POST "+chatRoute, trace.SpanKindServer)
	if s.Parent().SpanID().String() != callerParent || sid != s.SpanContext().SpanID().String() {
		t.Fatalf("StartServer span parent %s, ctx span %s; want parent %s and ctx holding the new span", s.Parent().SpanID(), sid, callerParent)
	}
	wantAttr(t, s, "http.response.status_code", attribute.INT64, "200")
	wantAttr(t, s, "tl.cache.hit", attribute.BOOL, "false")
	wantAttr(t, s, "tl.proxy.first_byte_ms", attribute.FLOAT64, "12.5")
	wantAttr(t, s, "gen_ai.response.finish_reasons", attribute.STRINGSLICE, `["stop"]`)
	wantAttr(t, s, "tl.route.worker_id", attribute.STRING, "decode-1")
	wantAttr(t, s, "tl.route.cascade_step", attribute.INT64, "2")
}

// -- Setup: resource, sampling, export ------------------------------------------------

func TestSetupResourceAndParentBasedSampling(t *testing.T) {
	// WHY: the resource names the process (service.name, namespace, version,
	//      engine role) on every span. The sampler is ParentBased: a ratio
	//      decides only for new traces, and a caller's decision is always
	//      kept, so a sampled request is never missing its middle hops.
	// KIND: unit
	// CATCHES: s06, s21
	rec := tracetest.NewSpanRecorder()
	tp, err := otelx.Setup(context.Background(), otelx.Config{
		ServiceName: "forge-engine", Namespace: "forge", Version: "0.4.0", SampleRatio: 0,
		Attributes: map[string]string{"tl.engine.role": "decode"},
	}, sdktrace.WithSpanProcessor(rec))
	if err != nil {
		t.Fatal(err)
	}
	defer tp.Shutdown(context.Background())
	tr := tp.Tracer("test")
	_, root := tr.Start(context.Background(), "root")
	root.End()
	if root.SpanContext().IsSampled() {
		t.Errorf("ratio 0: a new trace was sampled")
	}
	sampled := otelx.ContextWithTraceparent(context.Background(), w3cHeader)
	_, child := tr.Start(sampled, "engine.decode")
	child.End()
	if !child.SpanContext().IsSampled() {
		t.Fatalf("ratio 0: the child of a sampled caller was dropped; the sampler must be ParentBased")
	}
	got := rec.Ended()
	if len(got) != 1 || got[0].Name() != "engine.decode" {
		t.Fatalf("recorded %s; want only engine.decode", describe(got))
	}
	res := map[string]string{}
	for _, kv := range got[0].Resource().Attributes() {
		res[string(kv.Key)] = kv.Value.Emit()
	}
	for k, v := range map[string]string{"service.name": "forge-engine", "service.namespace": "forge", "service.version": "0.4.0", "tl.engine.role": "decode"} {
		if res[k] != v {
			t.Errorf("resource %s = %q; want %q (resource: %v)", k, res[k], v, res)
		}
	}
	// Ratio 1 records new traces but still respects a caller that did not sample.
	tp1, err := otelx.Setup(context.Background(), otelx.Config{ServiceName: "forge-gateway", SampleRatio: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer tp1.Shutdown(context.Background())
	_, s1 := tp1.Tracer("test").Start(context.Background(), "root")
	s1.End()
	unsampled := otelx.ContextWithTraceparent(context.Background(), w3cHeader[:len(w3cHeader)-2]+"00")
	_, s2 := tp1.Tracer("test").Start(unsampled, "child")
	s2.End()
	if !s1.SpanContext().IsSampled() || s2.SpanContext().IsSampled() {
		t.Errorf("ratio 1: new trace sampled=%v (want true), child of an unsampled caller sampled=%v (want false)",
			s1.SpanContext().IsSampled(), s2.SpanContext().IsSampled())
	}
}

func TestSetupRejectsBadConfig(t *testing.T) {
	// WHY: a ratio outside [0, 1] or a missing service.name is a config
	//      mistake that would silently drop every trace or file them under
	//      "unknown_service"; refuse it at startup.
	// KIND: boundary
	// CATCHES: s24
	for _, cfg := range []otelx.Config{
		{ServiceName: "", SampleRatio: 1},
		{ServiceName: "forge-gateway", SampleRatio: 1.5},
		{ServiceName: "forge-gateway", SampleRatio: -0.1},
	} {
		if tp, err := otelx.Setup(context.Background(), cfg); err == nil {
			_ = tp.Shutdown(context.Background())
			t.Errorf("Setup(%+v) succeeded; want an error", cfg)
		}
	}
}

// collector is a fake OTLP/gRPC trace receiver.
type collector struct {
	collectortrace.UnimplementedTraceServiceServer
	mu   sync.Mutex
	reqs []*collectortrace.ExportTraceServiceRequest
}

func (c *collector) Export(_ context.Context, r *collectortrace.ExportTraceServiceRequest) (*collectortrace.ExportTraceServiceResponse, error) {
	c.mu.Lock()
	c.reqs = append(c.reqs, r)
	c.mu.Unlock()
	return &collectortrace.ExportTraceServiceResponse{}, nil
}

func TestSetupExportsOverOTLPgRPC(t *testing.T) {
	// WHY: in Kubernetes every service sends its spans to the collector over
	//      OTLP/gRPC at [otel].endpoint (an http:// URL, so no TLS). The fake
	//      collector must receive the span with the process's resource.
	// KIND: conformance
	// CATCHES: s16
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	col := &collector{}
	srv := grpc.NewServer()
	collectortrace.RegisterTraceServiceServer(srv, col)
	go func() { _ = srv.Serve(lis) }()
	defer srv.Stop()
	tp, err := otelx.Setup(context.Background(), otelx.Config{
		Endpoint: "http://" + lis.Addr().String(), ServiceName: "forge-engine", Namespace: "forge", SampleRatio: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	_, span := tp.Tracer("test").Start(context.Background(), "engine.decode")
	span.End()
	ctx, cancel := context.WithTimeout(context.Background(), patience)
	defer cancel()
	if err := tp.ForceFlush(ctx); err != nil {
		t.Fatalf("ForceFlush: %v", err)
	}
	defer tp.Shutdown(context.Background())
	col.mu.Lock()
	defer col.mu.Unlock()
	for _, r := range col.reqs {
		for _, rs := range r.GetResourceSpans() {
			name := ""
			for _, kv := range rs.GetResource().GetAttributes() {
				if kv.GetKey() == "service.name" {
					name = kv.GetValue().GetStringValue()
				}
			}
			for _, ss := range rs.GetScopeSpans() {
				for _, s := range ss.GetSpans() {
					if s.GetName() == "engine.decode" && name == "forge-engine" &&
						bytes.Equal(s.GetTraceId(), func() []byte { id := span.SpanContext().TraceID(); return id[:] }()) {
						return
					}
				}
			}
		}
	}
	t.Fatalf("the collector got %d export request(s) and no engine.decode span from service forge-engine", len(col.reqs))
}

func TestSetupWithoutEndpointShutsDownAtOnce(t *testing.T) {
	// WHY: no [otel].endpoint (a laptop run, a unit test) means no exporter.
	//      An exporter aimed at the default localhost:4317 retries a refused
	//      connection until its timeout, so every shutdown hangs for seconds
	//      and logs errors about a collector nobody asked for.
	// KIND: boundary
	// CATCHES: s17
	rec := tracetest.NewSpanRecorder()
	tp, err := otelx.Setup(context.Background(), otelx.Config{ServiceName: "forge-gateway", SampleRatio: 1}, sdktrace.WithSpanProcessor(rec))
	if err != nil {
		t.Fatal(err)
	}
	_, span := tp.Tracer("test").Start(context.Background(), "gateway.proxy")
	span.End()
	if len(rec.Ended()) != 1 {
		t.Fatalf("spans still go to extra processors without an endpoint; recorded %d", len(rec.Ended()))
	}
	ctx, cancel := context.WithTimeout(context.Background(), patience)
	defer cancel()
	t0 := time.Now()
	err = tp.Shutdown(ctx)
	if took := time.Since(t0); err != nil || took > time.Second {
		t.Fatalf("Shutdown took %v (err %v); with no endpoint there is nothing to flush", took.Round(time.Millisecond), err)
	}
}

func TestEndingSpansNeverWaitsForTheCollector(t *testing.T) {
	// WHY: telemetry must never slow a request. A collector that accepts
	//      connections and never answers (overloaded, half-dead) must not add
	//      latency: spans are queued and exported in the background (a batch
	//      processor), never inline when the span ends.
	// KIND: fault
	// CATCHES: s07
	hole, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer hole.Close()
	var held []net.Conn // held open, never read: the peer waits forever
	var mu sync.Mutex
	go func() {
		for {
			c, err := hole.Accept()
			if err != nil {
				return
			}
			mu.Lock()
			held = append(held, c)
			mu.Unlock()
		}
	}()
	defer func() {
		mu.Lock()
		defer mu.Unlock()
		for _, c := range held {
			c.Close()
		}
	}()
	tp, err := otelx.Setup(context.Background(), otelx.Config{
		Endpoint: "http://" + hole.Addr().String(), ServiceName: "forge-gateway", SampleRatio: 1, ExportTimeout: 2 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()
		_ = tp.Shutdown(ctx)
	}()
	srv := httptest.NewServer(otelx.Middleware(tp, routeOf(chatRoute), http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, "ok")
	})))
	defer srv.Close()
	for i := 0; i < 5; i++ {
		t0 := time.Now()
		resp, err := http.Post(srv.URL+chatRoute, "application/json", nil)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if took := time.Since(t0); took > time.Second {
			t.Fatalf("request %d took %v with a collector that never answers: span export is on the request path", i+1, took.Round(time.Millisecond))
		}
	}
}

// -- logs -------------------------------------------------------------------------

func TestLogHandlerStampsTraceAndSpanIDs(t *testing.T) {
	// WHY: obs.02's log correlation is `kubectl logs | grep <trace id>`: a
	//      line logged inside a request must carry trace_id and span_id, also
	//      through a logger derived with With(...), and a line outside any
	//      span carries neither.
	// KIND: unit
	// CATCHES: s18, s23
	tp, _ := newTP(t)
	var buf bytes.Buffer
	log := slog.New(otelx.LogHandler(slog.NewJSONHandler(&buf, nil))).With("service", "forge-gateway")
	ctx, span := tp.Tracer("test").Start(otelx.ContextWithTraceparent(context.Background(), w3cHeader), "POST "+chatRoute)
	log.InfoContext(ctx, "proxied", "status", 200)
	span.End()
	log.Info("started")
	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("want 2 log lines, got %q", buf.String())
	}
	var in, out map[string]any
	if json.Unmarshal([]byte(lines[0]), &in) != nil || json.Unmarshal([]byte(lines[1]), &out) != nil {
		t.Fatalf("log lines are not JSON: %q", buf.String())
	}
	if in["trace_id"] != callerTrace || in["span_id"] != span.SpanContext().SpanID().String() || in["service"] != "forge-gateway" {
		t.Fatalf("line inside the span: %v; want trace_id %s, span_id %s, service forge-gateway", in, callerTrace, span.SpanContext().SpanID())
	}
	if _, ok := out["trace_id"]; ok {
		t.Fatalf("line outside any span carries trace_id: %v", out)
	}
}
