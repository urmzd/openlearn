// HTTP server and client spans (obs.01), named and attributed as
// course/contracts/otel/semconv.md says.

package otelx

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strconv"
	"sync"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

// Middleware wraps next with one SERVER span per request, named
// "<METHOD> <route>" (just "<METHOD>" when route returns ""), the child of
// the request's traceparent when it has a valid one and a new trace's root
// otherwise. route maps a request to its route pattern
// ("/v1/models/{model}"), never the raw path, so span names stay a small
// fixed set. nil means no route: the name is "<METHOD>". (r.Pattern is no
// help here: the ServeMux sets it only after this middleware has run.)
//
// Attributes: http.request.method, http.route (when known),
// http.response.status_code. A status >= 500 sets the span status to Error
// and error.type to the status code; a 4xx is the client's error, not the
// server's, and leaves the status unset.
//
// The ResponseWriter next sees still implements http.Flusher (and Unwrap for
// http.ResponseController), so a streamed SSE response is not buffered.
func Middleware(tp trace.TracerProvider, route func(*http.Request) string, next http.Handler) http.Handler {
	tracer := tp.Tracer(ScopeName)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx := Extract(r.Context(), r.Header)
		rt := ""
		if route != nil {
			rt = route(r)
		}
		name := r.Method
		if rt != "" {
			name = r.Method + " " + rt
		}
		attrs := []attribute.KeyValue{attribute.String("http.request.method", r.Method)}
		if rt != "" {
			attrs = append(attrs, attribute.String("http.route", rt))
		}
		ctx, span := tracer.Start(ctx, name, trace.WithSpanKind(trace.SpanKindServer), trace.WithAttributes(attrs...))
		defer span.End()
		rec := &statusRecorder{ResponseWriter: w}
		next.ServeHTTP(rec, r.WithContext(ctx))
		code := rec.code
		if code == 0 {
			code = http.StatusOK // the handler wrote nothing: net/http answers 200
		}
		span.SetAttributes(attribute.Int("http.response.status_code", code))
		if code >= 500 {
			span.SetStatus(codes.Error, http.StatusText(code))
			span.SetAttributes(attribute.String("error.type", strconv.Itoa(code)))
		}
	})
}

// statusRecorder remembers the first status code written and passes
// everything else through, including Flush.
type statusRecorder struct {
	http.ResponseWriter
	code int
}

func (s *statusRecorder) WriteHeader(code int) {
	if s.code == 0 {
		s.code = code
	}
	s.ResponseWriter.WriteHeader(code)
}

func (s *statusRecorder) Write(b []byte) (int, error) {
	if s.code == 0 {
		s.code = http.StatusOK
	}
	return s.ResponseWriter.Write(b)
}

// Flush sends buffered bytes to the client now (SSE needs one per event).
func (s *statusRecorder) Flush() {
	if s.code == 0 {
		s.code = http.StatusOK
	}
	if f, ok := s.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// Unwrap lets http.NewResponseController reach the real writer.
func (s *statusRecorder) Unwrap() http.ResponseWriter {
	return s.ResponseWriter
}

// Transport wraps base (nil means http.DefaultTransport) with one CLIENT
// span per request, named "<METHOD>", the child of the request context's
// span. The outgoing request carries a traceparent naming this CLIENT span
// as the parent, never the caller's span. The span ends when the response
// body is read to EOF or closed (a stream lasts until its last event).
// Attributes: http.request.method, server.address, server.port, and
// http.response.status_code (or error.type when no response came back).
func Transport(tp trace.TracerProvider, base http.RoundTripper) http.RoundTripper {
	if base == nil {
		base = http.DefaultTransport
	}
	return &transport{tracer: tp.Tracer(ScopeName), base: base}
}

type transport struct {
	tracer trace.Tracer
	base   http.RoundTripper
}

func (t *transport) RoundTrip(r *http.Request) (*http.Response, error) {
	host, port := r.URL.Hostname(), r.URL.Port()
	if port == "" {
		port = map[string]string{"https": "443"}[r.URL.Scheme]
		if port == "" {
			port = "80"
		}
	}
	attrs := []attribute.KeyValue{
		attribute.String("http.request.method", r.Method),
		attribute.String("server.address", host),
	}
	if p, err := strconv.Atoi(port); err == nil {
		attrs = append(attrs, attribute.Int("server.port", p))
	}
	ctx, span := t.tracer.Start(r.Context(), r.Method, trace.WithSpanKind(trace.SpanKindClient), trace.WithAttributes(attrs...))
	// RoundTrip must not modify the caller's request: clone, then inject.
	out := r.Clone(ctx)
	Inject(ctx, out.Header)
	resp, err := t.base.RoundTrip(out)
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
		span.SetAttributes(attribute.String("error.type", errorType(err)))
		span.End()
		return nil, err
	}
	span.SetAttributes(attribute.Int("http.response.status_code", resp.StatusCode))
	if resp.StatusCode >= 500 {
		span.SetStatus(codes.Error, http.StatusText(resp.StatusCode))
		span.SetAttributes(attribute.String("error.type", strconv.Itoa(resp.StatusCode)))
	}
	// The call lasts until the body is consumed: for a streamed completion
	// that is the last SSE event, not the response headers.
	resp.Body = &spanBody{ReadCloser: resp.Body, span: span}
	return resp, nil
}

// spanBody ends the client span once, at EOF, a read error, or Close.
type spanBody struct {
	io.ReadCloser
	span trace.Span
	once sync.Once
}

func (b *spanBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	if err != nil {
		b.end()
	}
	return n, err
}

func (b *spanBody) Close() error {
	err := b.ReadCloser.Close()
	b.end()
	return err
}

func (b *spanBody) end() {
	b.once.Do(func() { b.span.End() })
}

// errorType names a transport failure for error.type.
func errorType(err error) string {
	var ne net.Error
	switch {
	case errors.Is(err, context.Canceled):
		return "canceled"
	case errors.As(err, &ne) && ne.Timeout():
		return "timeout"
	default:
		return fmt.Sprintf("%T", err)
	}
}

// Span is a started span with the SetAttribute(key, any) the gateway's
// chain uses (tinyllm/gateway/server.Span); End ends it.
type Span struct{ trace.Span }

// SetAttribute records one attribute; string, bool, int, int64, float64,
// and []string keep their type, anything else is recorded with fmt's %v.
func (s Span) SetAttribute(key string, value any) {
	var kv attribute.KeyValue
	switch v := value.(type) {
	case string:
		kv = attribute.String(key, v)
	case bool:
		kv = attribute.Bool(key, v)
	case int:
		kv = attribute.Int(key, v)
	case int64:
		kv = attribute.Int64(key, v)
	case float64:
		kv = attribute.Float64(key, v)
	case []string:
		kv = attribute.StringSlice(key, v)
	default:
		kv = attribute.String(key, fmt.Sprint(v))
	}
	s.Span.SetAttributes(kv)
}

// StartServer starts a SERVER span named name as the child of ctx's span
// (or remote span context, after Extract). It is what a gateway.server.Tracer
// adapter calls:
//
//	type tracer struct{ tp trace.TracerProvider }
//	func (t tracer) Start(ctx context.Context, name string) (context.Context, server.Span) {
//		return otelx.StartServer(ctx, t.tp, name)
//	}
func StartServer(ctx context.Context, tp trace.TracerProvider, name string) (context.Context, Span) {
	ctx, span := tp.Tracer(ScopeName).Start(ctx, name, trace.WithSpanKind(trace.SpanKindServer))
	return ctx, Span{span}
}
