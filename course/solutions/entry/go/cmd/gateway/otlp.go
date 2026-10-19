// Span export for the tracer gateway (obs.00): the reference of the "tracing
// setup in the gateway entry point".
//
// The proxy library (gw.00) already records one gateway.proxy span per
// request, with W3C ids it shares with the engine through traceparent. This
// file ships those records to an OTLP/HTTP endpoint as JSON, the same wire
// format the engine writes by hand (L10.0), so Pass 1 needs no SDK and no
// third-party module. obs.01 replaces it with the OpenTelemetry Go SDK in
// go/otelx.
//
// Wiring, in main: proxy.Config{..., OnSpan: newSpanExporter()}.
//
// Rules this file follows (content/courses/systems/04-observability/01-one-trace.md):
//   - OTEL_EXPORTER_OTLP_ENDPOINT unset or empty: export nothing (nil hook).
//   - Never block the request path: the hook hands the span to a bounded
//     queue and returns; a full queue drops the span and counts it.
//   - Every POST has a timeout, so a dead collector costs one goroutine,
//     never a request.
package main

import (
	"bytes"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"tinyllm/gateway/proxy"
)

// spanKindClient is SPAN_KIND_CLIENT in opentelemetry/proto/trace/v1/trace.proto:
// the gateway's span covers its call to the engine.
const spanKindClient = 3

var droppedSpans atomic.Int64

// newSpanExporter returns the proxy.Config.OnSpan hook, or nil when no
// endpoint is configured.
func newSpanExporter() func(proxy.Span) {
	endpoint := strings.TrimRight(os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT"), "/")
	if endpoint == "" {
		return nil
	}
	service := os.Getenv("OTEL_SERVICE_NAME")
	if service == "" {
		service = "gateway"
	}
	url := endpoint + "/v1/traces"
	client := &http.Client{Timeout: 2 * time.Second}
	queue := make(chan proxy.Span, 1024)

	go func() {
		for s := range queue {
			body, err := json.Marshal(otlpRequest(service, s))
			if err != nil {
				continue
			}
			resp, err := client.Post(url, "application/json", bytes.NewReader(body))
			if err != nil {
				log.Printf("gateway: span export to %s failed: %v", url, err)
				continue
			}
			resp.Body.Close()
			if resp.StatusCode/100 != 2 {
				log.Printf("gateway: span export to %s answered %d", url, resp.StatusCode)
			}
		}
	}()

	return func(s proxy.Span) {
		select {
		case queue <- s:
		default:
			droppedSpans.Add(1) // telemetry is best effort; requests are not
		}
	}
}

// otlpRequest builds an ExportTraceServiceRequest in the OTLP/HTTP JSON
// mapping: ids are lowercase hex strings, 64-bit integers are decimal strings.
func otlpRequest(service string, s proxy.Span) map[string]any {
	span := map[string]any{
		"traceId":           s.TraceID,
		"spanId":            s.SpanID,
		"name":              s.Name,
		"kind":              spanKindClient,
		"startTimeUnixNano": strconv.FormatInt(s.Start.UnixNano(), 10),
		"endTimeUnixNano":   strconv.FormatInt(s.End.UnixNano(), 10),
		"attributes": []map[string]any{
			{"key": "http.response.status_code", "value": map[string]any{"intValue": strconv.Itoa(s.StatusCode)}},
			{"key": "tl.request_id", "value": map[string]any{"stringValue": s.RequestID}},
		},
	}
	if s.ParentSpanID != "" {
		span["parentSpanId"] = s.ParentSpanID
	}
	if s.StatusCode >= 500 {
		span["status"] = map[string]any{"code": 2} // STATUS_CODE_ERROR
	}
	return map[string]any{
		"resourceSpans": []map[string]any{{
			"resource": map[string]any{"attributes": []map[string]any{
				{"key": "service.name", "value": map[string]any{"stringValue": service}},
			}},
			"scopeSpans": []map[string]any{{
				"scope": map[string]any{"name": "tinyllm/gateway"},
				"spans": []map[string]any{span},
			}},
		}},
	}
}
