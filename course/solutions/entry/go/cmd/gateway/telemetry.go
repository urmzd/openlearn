// Telemetry of the gateway entry point (obs.01 and obs.02 reference): the
// OpenTelemetry provider from go/otelx, the gateway's SERVER span, the
// instruments of course/contracts/otel/metrics.yaml served on the health
// port at /metrics, and JSON logs that carry trace_id and span_id.
//
// Wiring, in main (obs.02):
//
//	tel, err := newObsTelemetry()             // reads OTEL_SERVICE_NAME, OTEL_EXPORTER_OTLP_ENDPOINT
//	api.Handler = tel.wrap(api.Handler)       // the SERVER span, metrics, one log line per request
//	health.Handler = tel.withMetrics(health.Handler, upstream)
//	defer tel.shutdown(ctx)
//
// Rules this file follows (content/tracks/systems/04-observability/02-metrics-and-log-correlation.md):
//   - names, bounds, and labels exactly as metrics.yaml; capped labels keep
//     64 values per process, later ones are recorded as _other
//   - the prompt, the completion, and the key never reach a log line
//   - telemetry never blocks a request: spans leave through otelx's batch
//     processor, metrics are in-memory counters read on scrape
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"

	"tinyllm/otelx"
)

// Bucket bounds of otel/metrics.yaml, in seconds (tokens for usage).
var (
	obsTTFTBounds  = []float64{0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0}
	obsTPOTBounds  = []float64{0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5}
	obsE2EBounds   = []float64{0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92}
	obsUsageBounds = []float64{1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864}
	obsHTTPBounds  = []float64{0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0}
)

const obsCap = 64 // values per capped label (metrics.yaml)

type obsTelemetry struct {
	service string
	tp      *sdktrace.TracerProvider
	log     *slog.Logger
	m       *obsMetrics
}

func newObsTelemetry() (*obsTelemetry, error) {
	service := os.Getenv("OTEL_SERVICE_NAME")
	if service == "" {
		service = "gateway"
	}
	endpoint := os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
	if strings.HasSuffix(endpoint, ":4318") { // the Pass 1 Jaeger OTLP/HTTP port; the SDK speaks gRPC
		endpoint = strings.TrimSuffix(endpoint, ":4318") + ":4317"
	}
	tp, err := otelx.Setup(context.Background(), otelx.Config{
		Endpoint:    endpoint,
		ServiceName: service,
		Namespace:   os.Getenv("TL_SYSTEM"),
		Version:     os.Getenv("TL_SYSTEM_VERSION"),
		SampleRatio: 1,
	})
	if err != nil {
		return nil, err
	}
	h := slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{ReplaceAttr: func(groups []string, a slog.Attr) slog.Attr {
		switch {
		case len(groups) == 0 && a.Key == slog.TimeKey:
			return slog.String("ts", a.Value.Time().UTC().Format(time.RFC3339Nano))
		case len(groups) == 0 && a.Key == slog.LevelKey:
			return slog.String("level", strings.ToLower(a.Value.String()))
		}
		return a
	}})
	return &obsTelemetry{
		service: service,
		tp:      tp,
		log:     slog.New(otelx.LogHandler(h)).With("service", service),
		m:       newObsMetrics(),
	}, nil
}

func (t *obsTelemetry) shutdown(ctx context.Context) { _ = t.tp.Shutdown(ctx) }

// obsRoute names the route of an API path, never the raw path.
func obsRoute(p string) string {
	switch p {
	case "/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/v1/models", "/v1/tokenize":
		return p
	}
	if strings.HasPrefix(p, "/v1/models/") {
		return "/v1/models/{model}"
	}
	return "other"
}

func obsOperation(route string) string {
	switch route {
	case "/v1/chat/completions":
		return "chat"
	case "/v1/completions":
		return "text_completion"
	case "/v1/embeddings":
		return "embeddings"
	}
	return ""
}

// wrap puts the SERVER span, the metrics, and one log line around every API
// request. The proxy library reads traceparent from the request headers; when
// the caller sent none, the span's own context is written into them, so the
// proxy's gateway.proxy span hangs under this SERVER span in one trace.
func (t *obsTelemetry) wrap(next http.Handler) http.Handler {
	return otelx.Middleware(t.tp, func(r *http.Request) string { return obsRoute(r.URL.Path) },
		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			// A caller's traceparent is kept: obs.00's contract makes the
			// gateway.proxy span the direct child of the caller's span. Only
			// an untraced request gets this span's context, so the proxy and
			// the engine join this span's trace instead of starting another.
			if r.Header.Get("traceparent") == "" {
				otelx.Inject(r.Context(), r.Header)
			}
			route := obsRoute(r.URL.Path)
			model := ""
			if r.Body != nil && r.Method == http.MethodPost {
				body, _ := io.ReadAll(io.LimitReader(r.Body, 8<<20))
				r.Body = io.NopCloser(bytes.NewReader(body))
				var req struct {
					Model string `json:"model"`
				}
				_ = json.Unmarshal(body, &req)
				model = req.Model
			}
			rec := &obsWriter{ResponseWriter: w}
			next.ServeHTTP(rec, r)
			t.m.observe(r.Method, route, model, rec, start, time.Now())
			// No prompt, completion, or key in logs: method, route, status, timing only.
			t.log.InfoContext(r.Context(), "request", "method", r.Method, "route", route,
				"status", rec.status(), "duration_ms", time.Since(start).Milliseconds())
		}))
}

// withMetrics serves /metrics on the health port next to h, and keeps the
// worker gauge in step with the upstream's health.
func (t *obsTelemetry) withMetrics(h http.Handler, upstream string) http.Handler {
	client := &http.Client{Timeout: time.Second}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/metrics" {
			h.ServeHTTP(w, r)
			return
		}
		healthy := false
		if resp, err := client.Get(strings.TrimRight(upstream, "/") + "/healthz"); err == nil {
			healthy = resp.StatusCode == http.StatusOK
			resp.Body.Close()
		}
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		_, _ = io.WriteString(w, t.m.exposition(healthy))
	})
}

// obsWriter records the status, the time of the first byte, and the number
// of SSE events (one per token); Flush and Unwrap keep streaming working.
type obsWriter struct {
	http.ResponseWriter
	code   int
	first  time.Time
	events int
}

func (o *obsWriter) WriteHeader(c int) {
	if o.code == 0 {
		o.code = c
	}
	o.ResponseWriter.WriteHeader(c)
}

func (o *obsWriter) Write(b []byte) (int, error) {
	if o.code == 0 {
		o.code = http.StatusOK
	}
	if o.first.IsZero() && len(b) > 0 {
		o.first = time.Now()
	}
	o.events += bytes.Count(b, []byte("data: ")) - bytes.Count(b, []byte("data: [DONE]"))
	return o.ResponseWriter.Write(b)
}

func (o *obsWriter) Flush() {
	if f, ok := o.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (o *obsWriter) Unwrap() http.ResponseWriter { return o.ResponseWriter }

func (o *obsWriter) status() int {
	if o.code == 0 {
		return http.StatusOK
	}
	return o.code
}

// -- a small registry ---------------------------------------------------------

type obsHist struct {
	bounds []float64
	counts map[string][]uint64 // label set -> per-bound counts (not cumulative)
	sums   map[string]float64
	totals map[string]uint64
}

func newObsHist(bounds []float64) *obsHist {
	return &obsHist{bounds: bounds, counts: map[string][]uint64{}, sums: map[string]float64{}, totals: map[string]uint64{}}
}

func (h *obsHist) observe(labels string, v float64) {
	c, ok := h.counts[labels]
	if !ok {
		c = make([]uint64, len(h.bounds))
		h.counts[labels] = c
	}
	for i, b := range h.bounds {
		if v <= b {
			c[i]++
			break
		}
	}
	h.sums[labels] += v
	h.totals[labels]++
}

func (h *obsHist) write(sb *strings.Builder, name string) {
	fmt.Fprintf(sb, "# TYPE %s histogram\n", name)
	for _, labels := range obsSorted(h.totals) {
		sep := ""
		if labels != "" {
			sep = ","
		}
		var cum uint64
		for i, b := range h.bounds {
			cum += h.counts[labels][i]
			fmt.Fprintf(sb, "%s_bucket{%s%sle=\"%s\"} %d\n", name, labels, sep, strconv.FormatFloat(b, 'f', -1, 64), cum)
		}
		fmt.Fprintf(sb, "%s_bucket{%s%sle=\"+Inf\"} %d\n", name, labels, sep, h.totals[labels])
		fmt.Fprintf(sb, "%s_sum{%s} %g\n", name, labels, h.sums[labels])
		fmt.Fprintf(sb, "%s_count{%s} %d\n", name, labels, h.totals[labels])
	}
}

func obsSorted[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// obsLabels renders label pairs in order: k1, v1, k2, v2, ...
func obsLabels(kv ...string) string {
	parts := make([]string, 0, len(kv)/2)
	for i := 0; i+1 < len(kv); i += 2 {
		parts = append(parts, fmt.Sprintf("%s=%q", kv[i], kv[i+1]))
	}
	return strings.Join(parts, ",")
}

type obsMetrics struct {
	mu                             sync.Mutex
	ttft, tpot, e2e, usage, http   *obsHist
	requests, rejections           map[string]uint64
	cacheHits                      uint64
	models, routes, codes, tenants map[string]bool
}

func newObsMetrics() *obsMetrics {
	return &obsMetrics{
		ttft: newObsHist(obsTTFTBounds), tpot: newObsHist(obsTPOTBounds), e2e: newObsHist(obsE2EBounds),
		usage: newObsHist(obsUsageBounds), http: newObsHist(obsHTTPBounds),
		requests: map[string]uint64{}, rejections: map[string]uint64{},
		models: map[string]bool{}, routes: map[string]bool{}, codes: map[string]bool{}, tenants: map[string]bool{},
	}
}

// obsCapped keeps the first obsCap distinct values of a label, then _other.
func obsCapped(seen map[string]bool, v string) string {
	if seen[v] {
		return v
	}
	if len(seen) >= obsCap {
		return "_other"
	}
	seen[v] = true
	return v
}

func (m *obsMetrics) observe(method, route, model string, w *obsWriter, start, end time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()
	code := w.status()
	codeS := obsCapped(m.codes, strconv.Itoa(code))
	routeL := obsCapped(m.routes, route)
	tenant := obsCapped(m.tenants, "default") // one static key until gw.02's tenants
	m.http.observe(obsLabels("http_request_method", method, "http_route", routeL, "http_response_status_code", codeS), end.Sub(start).Seconds())
	m.requests[obsLabels("route", routeL, "code", codeS, "tenant", tenant)]++
	if code == http.StatusTooManyRequests {
		m.rejections[obsLabels("limit", "rpm", "tenant", tenant)]++
	}
	if w.Header().Get("X-TL-Cache") == "hit" {
		m.cacheHits++
	}
	op := obsOperation(route)
	if op == "" || method != http.MethodPost {
		return
	}
	modelL := obsCapped(m.models, model)
	e2eLabels := obsLabels("gen_ai_operation_name", op, "gen_ai_request_model", modelL)
	if code >= 400 {
		e2eLabels = obsLabels("error_type", strconv.Itoa(code), "gen_ai_operation_name", op, "gen_ai_request_model", modelL)
	}
	m.e2e.observe(e2eLabels, end.Sub(start).Seconds())
	if code != http.StatusOK || op == "embeddings" || w.first.IsZero() {
		return
	}
	lat := obsLabels("gen_ai_operation_name", op, "gen_ai_request_model", modelL, "tl_engine_role", "gateway")
	m.ttft.observe(lat, w.first.Sub(start).Seconds())
	if w.events >= 2 {
		m.tpot.observe(lat, end.Sub(w.first).Seconds()/float64(w.events-1))
	}
	if w.events > 0 {
		m.usage.observe(obsLabels("gen_ai_operation_name", op, "gen_ai_request_model", modelL, "gen_ai_token_type", "output"), float64(w.events))
	}
}

func (m *obsMetrics) exposition(upstreamHealthy bool) string {
	m.mu.Lock()
	defer m.mu.Unlock()
	var sb strings.Builder
	m.ttft.write(&sb, "gen_ai_server_time_to_first_token_seconds")
	m.tpot.write(&sb, "gen_ai_server_time_per_output_token_seconds")
	m.e2e.write(&sb, "gen_ai_server_request_duration_seconds")
	m.usage.write(&sb, "gen_ai_client_token_usage")
	m.http.write(&sb, "http_server_request_duration_seconds")
	sb.WriteString("# TYPE tl_gateway_requests_total counter\n")
	for _, k := range obsSorted(m.requests) {
		fmt.Fprintf(&sb, "tl_gateway_requests_total{%s} %d\n", k, m.requests[k])
	}
	sb.WriteString("# TYPE tl_gateway_ratelimit_rejections_total counter\n")
	for _, limit := range []string{"rpm", "tpm"} { // known series start at 0
		if k := obsLabels("limit", limit, "tenant", "default"); m.rejections[k] == 0 {
			m.rejections[k] = 0
		}
	}
	for _, k := range obsSorted(m.rejections) {
		fmt.Fprintf(&sb, "tl_gateway_ratelimit_rejections_total{%s} %d\n", k, m.rejections[k])
	}
	fmt.Fprintf(&sb, "# TYPE tl_gateway_cache_hits_total counter\ntl_gateway_cache_hits_total %d\n", m.cacheHits)
	healthy := 0
	if upstreamHealthy {
		healthy = 1
	}
	fmt.Fprintf(&sb, "# TYPE tl_gateway_workers gauge\ntl_gateway_workers{role=\"unified\",state=\"healthy\"} %d\n", healthy)
	return sb.String()
}
