// Package otelx is the tracing kit of the serving path (obs.01): one
// TracerProvider per process, HTTP and gRPC middleware that continue and
// propagate W3C trace context, the traceparent helpers the durable engine
// and the Python subprocess contract use, and a log handler that stamps
// trace and span ids on every JSON log line.
//
// Nothing here touches the OpenTelemetry globals (otel.SetTracerProvider,
// otel.SetTextMapPropagator): every function takes the provider it should
// use, and propagation is always W3C Trace Context. A process wires it once
// in its entry point; tests build their own provider per test.
//
// Contract: course/contracts/otel/semconv.md (span names, kinds, attributes,
// propagation, resource attributes, log fields).
// Chapter: content/courses/systems/11-observability/02-tracing-for-the-serving-path.md
// Callers: your go/cmd/gateway (obs.02 wires it), dur.04 (activity spans from
// ActivityTask.trace_context), dur.09 (TRACEPARENT for Python), ag.03.
package otelx

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

// ScopeName is the instrumentation scope of every span this package starts.
const ScopeName = "tinyllm/otelx"

// Config is a process's tracing identity and the [otel] table of runtime.toml.
type Config struct {
	// Endpoint is the collector's OTLP/gRPC URL, for example
	// "http://otel-collector.observability:4317" ([otel].endpoint). An
	// http:// URL is sent without TLS. Empty means no exporter at all: spans
	// still get ids (so propagation and log correlation keep working), but
	// nothing leaves the process.
	Endpoint string
	// ServiceName is service.name: "<system>-<component>", e.g. "forge-gateway".
	ServiceName string
	// Namespace is service.namespace: "<system>" ([otel].service_namespace).
	Namespace string
	// Version is service.version: [system].version of system.toml.
	Version string
	// SampleRatio is [otel].trace_sample_ratio in [0, 1]: the fraction of new
	// traces (requests that arrive without a sampled traceparent) to record.
	// A request whose caller sampled it is always recorded.
	SampleRatio float64
	// Attributes are extra resource attributes, e.g. {"tl.engine.role": "decode"}.
	Attributes map[string]string
	// ExportTimeout bounds one export call; 0 means 5 s.
	ExportTimeout time.Duration
}

// Setup builds the process's TracerProvider: a resource with service.name,
// service.namespace, service.version and Attributes; the sampler
// ParentBased(TraceIDRatioBased(SampleRatio)); and, when Endpoint is set, a
// batch span processor over an OTLP/gRPC exporter, so ending a span never
// waits for the network. opts are appended after these (tests add a span
// processor that records). Call Shutdown on the result before exit to flush.
func Setup(ctx context.Context, cfg Config, opts ...sdktrace.TracerProviderOption) (*sdktrace.TracerProvider, error) {
	if cfg.ServiceName == "" {
		return nil, errors.New("otelx: Config.ServiceName is required (service.name, e.g. forge-gateway)")
	}
	if cfg.SampleRatio < 0 || cfg.SampleRatio > 1 {
		return nil, fmt.Errorf("otelx: SampleRatio %v is outside [0, 1]", cfg.SampleRatio)
	}
	attrs := []attribute.KeyValue{attribute.String("service.name", cfg.ServiceName)}
	if cfg.Namespace != "" {
		attrs = append(attrs, attribute.String("service.namespace", cfg.Namespace))
	}
	if cfg.Version != "" {
		attrs = append(attrs, attribute.String("service.version", cfg.Version))
	}
	keys := make([]string, 0, len(cfg.Attributes))
	for k := range cfg.Attributes {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		attrs = append(attrs, attribute.String(k, cfg.Attributes[k]))
	}
	all := []sdktrace.TracerProviderOption{
		// Schemaless and without resource.Default(): the resource is exactly
		// what the config says, not whatever OTEL_RESOURCE_ATTRIBUTES holds.
		sdktrace.WithResource(resource.NewSchemaless(attrs...)),
		sdktrace.WithSampler(sdktrace.ParentBased(sdktrace.TraceIDRatioBased(cfg.SampleRatio))),
	}
	if cfg.Endpoint != "" {
		timeout := cfg.ExportTimeout
		if timeout <= 0 {
			timeout = 5 * time.Second
		}
		eopts := []otlptracegrpc.Option{otlptracegrpc.WithTimeout(timeout)}
		if strings.HasPrefix(cfg.Endpoint, "http://") || strings.HasPrefix(cfg.Endpoint, "https://") {
			eopts = append(eopts, otlptracegrpc.WithEndpointURL(cfg.Endpoint))
		} else {
			eopts = append(eopts, otlptracegrpc.WithEndpoint(cfg.Endpoint), otlptracegrpc.WithInsecure())
		}
		// New does not dial: the connection is made lazily, so a collector
		// that is down at startup is not a startup failure.
		exp, err := otlptracegrpc.New(ctx, eopts...)
		if err != nil {
			return nil, fmt.Errorf("otelx: OTLP exporter for %s: %w", cfg.Endpoint, err)
		}
		all = append(all, sdktrace.WithBatcher(exp, sdktrace.WithBatchTimeout(time.Second)))
	}
	return sdktrace.NewTracerProvider(append(all, opts...)...), nil
}
