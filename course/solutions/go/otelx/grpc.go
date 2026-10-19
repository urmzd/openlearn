// gRPC client and server spans (obs.01): tl.engine.v1.EngineControl,
// tl.kv.v1, tl.control.v1, tl.durable.v1 all carry trace context in gRPC
// metadata under the same keys as HTTP (traceparent, tracestate).

package otelx

import (
	"context"
	"errors"
	"io"
	"strings"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

// mdCarrier adapts gRPC metadata to the propagator's TextMapCarrier. gRPC
// lower-cases metadata keys; traceparent and tracestate already are.
type mdCarrier metadata.MD

func (m mdCarrier) Get(key string) string {
	if v := metadata.MD(m).Get(key); len(v) > 0 {
		return v[0]
	}
	return ""
}

func (m mdCarrier) Set(key, value string) {
	metadata.MD(m).Set(key, value)
}

func (m mdCarrier) Keys() []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

// rpcAttrs splits "/tl.engine.v1.EngineControl/Prefill" into the span name
// "tl.engine.v1.EngineControl/Prefill" and the rpc.* attributes.
func rpcAttrs(fullMethod string) (string, []attribute.KeyValue) {
	name := strings.TrimPrefix(fullMethod, "/")
	service, method, _ := strings.Cut(name, "/")
	return name, []attribute.KeyValue{
		attribute.String("rpc.system", "grpc"),
		attribute.String("rpc.service", service),
		attribute.String("rpc.method", method),
	}
}

// endRPC records the call's outcome and ends the span: rpc.grpc.status_code
// always; status Error and error.type (the code's name, "NotFound") when
// the code is not OK.
func endRPC(span trace.Span, err error) {
	st, _ := status.FromError(err)
	span.SetAttributes(attribute.Int("rpc.grpc.status_code", int(st.Code())))
	if err != nil {
		span.SetStatus(codes.Error, st.Message())
		span.SetAttributes(attribute.String("error.type", st.Code().String()))
	}
	span.End()
}

// serverContext is the incoming call's context with the caller's span
// context extracted from its metadata.
func serverContext(ctx context.Context) context.Context {
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return ctx
	}
	return w3c.Extract(ctx, mdCarrier(md))
}

// clientContext is ctx with traceparent added to the outgoing metadata, so
// the server's span names the span in ctx as its parent.
func clientContext(ctx context.Context) context.Context {
	// A copy: the caller's metadata must not change under it.
	md, _ := metadata.FromOutgoingContext(ctx)
	md = md.Copy()
	w3c.Inject(ctx, mdCarrier(md))
	return metadata.NewOutgoingContext(ctx, md)
}

// UnaryServerInterceptor starts a SERVER span per unary call, the child of
// the caller's traceparent, named "<service>/<method>".
func UnaryServerInterceptor(tp trace.TracerProvider) grpc.UnaryServerInterceptor {
	tracer := tp.Tracer(ScopeName)
	return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
		name, attrs := rpcAttrs(info.FullMethod)
		ctx, span := tracer.Start(serverContext(ctx), name, trace.WithSpanKind(trace.SpanKindServer), trace.WithAttributes(attrs...))
		resp, err := handler(ctx, req)
		endRPC(span, err)
		return resp, err
	}
}

// StreamServerInterceptor is UnaryServerInterceptor for streaming calls; the
// handler's stream reports the span's context from Context().
func StreamServerInterceptor(tp trace.TracerProvider) grpc.StreamServerInterceptor {
	tracer := tp.Tracer(ScopeName)
	return func(srv any, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		name, attrs := rpcAttrs(info.FullMethod)
		ctx, span := tracer.Start(serverContext(ss.Context()), name, trace.WithSpanKind(trace.SpanKindServer), trace.WithAttributes(attrs...))
		err := handler(srv, &serverStream{ServerStream: ss, ctx: ctx})
		endRPC(span, err)
		return err
	}
}

// serverStream overrides Context so handlers see the server span.
type serverStream struct {
	grpc.ServerStream
	ctx context.Context
}

func (s *serverStream) Context() context.Context {
	return s.ctx
}

// UnaryClientInterceptor starts a CLIENT span per unary call, the child of
// ctx's span, and sends its traceparent in the call's metadata.
func UnaryClientInterceptor(tp trace.TracerProvider) grpc.UnaryClientInterceptor {
	tracer := tp.Tracer(ScopeName)
	return func(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		name, attrs := rpcAttrs(method)
		ctx, span := tracer.Start(ctx, name, trace.WithSpanKind(trace.SpanKindClient), trace.WithAttributes(attrs...))
		err := invoker(clientContext(ctx), method, req, reply, cc, opts...)
		endRPC(span, err)
		return err
	}
}

// StreamClientInterceptor starts a CLIENT span per streaming call and ends
// it when the stream ends: io.EOF from RecvMsg (OK), another error, or a
// failed open.
func StreamClientInterceptor(tp trace.TracerProvider) grpc.StreamClientInterceptor {
	tracer := tp.Tracer(ScopeName)
	return func(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, streamer grpc.Streamer, opts ...grpc.CallOption) (grpc.ClientStream, error) {
		name, attrs := rpcAttrs(method)
		ctx, span := tracer.Start(ctx, name, trace.WithSpanKind(trace.SpanKindClient), trace.WithAttributes(attrs...))
		cs, err := streamer(clientContext(ctx), desc, cc, method, opts...)
		if err != nil {
			endRPC(span, err)
			return nil, err
		}
		return &clientStream{ClientStream: cs, span: span}, nil
	}
}

// clientStream ends its span once, at the first error RecvMsg returns.
type clientStream struct {
	grpc.ClientStream
	span  trace.Span
	ended bool
}

func (c *clientStream) RecvMsg(m any) error {
	err := c.ClientStream.RecvMsg(m)
	if err != nil && !c.ended {
		c.ended = true
		if errors.Is(err, io.EOF) {
			endRPC(c.span, nil) // the server finished the stream: OK
		} else {
			endRPC(c.span, err)
		}
	}
	return err
}
