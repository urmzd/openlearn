// W3C Trace Context in and out of a process (obs.01): HTTP headers, the
// single-string form the durable engine stores (ActivityTask.trace_context)
// and Python reads (TRACEPARENT), and the ids a log line carries.

package otelx

import (
	"context"
	"net/http"

	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/trace"
)

// w3c is the only propagator the system uses (semconv.md, Propagation):
// traceparent plus tracestate, never B3 or Jaeger headers.
var w3c = propagation.TraceContext{}

// Inject writes the span context of ctx into h as traceparent (and
// tracestate when ctx carries one). A ctx without a valid span context
// writes nothing.
func Inject(ctx context.Context, h http.Header) {
	w3c.Inject(ctx, propagation.HeaderCarrier(h))
}

// Extract returns ctx with the remote span context of h's traceparent, so
// the next span started from it is the caller's child. An invalid or absent
// header returns ctx unchanged (the next span starts a new trace).
func Extract(ctx context.Context, h http.Header) context.Context {
	return w3c.Extract(ctx, propagation.HeaderCarrier(h))
}

// Traceparent is the traceparent of the span in ctx,
// "00-<32 hex trace id>-<16 hex span id>-<2 hex flags>", or "" when ctx has
// no valid span context. Flags are 01 when the span is sampled, else 00.
func Traceparent(ctx context.Context) string {
	carrier := propagation.MapCarrier{}
	w3c.Inject(ctx, carrier)
	return carrier.Get("traceparent")
}

// ContextWithTraceparent is Extract for the single-string form: ctx with the
// remote span context s names. An invalid s (wrong length, version ff,
// upper-case hex, an all-zero id) returns ctx unchanged.
func ContextWithTraceparent(ctx context.Context, s string) context.Context {
	return w3c.Extract(ctx, propagation.MapCarrier{"traceparent": s})
}

// IDs are the lowercase hex trace and span ids of the span in ctx, for log
// lines; ok is false when ctx has no valid span context.
func IDs(ctx context.Context) (traceID, spanID string, ok bool) {
	sc := trace.SpanContextFromContext(ctx)
	if !sc.IsValid() {
		return "", "", false
	}
	return sc.TraceID().String(), sc.SpanID().String(), true
}
