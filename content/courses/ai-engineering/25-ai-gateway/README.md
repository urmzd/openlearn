# The AI Gateway

An LLM gateway is the one front door to every model server. Authentication, rate limits, usage policy, caching, routing, and metering all live there, so the engines behind it can stay simple and keyless. Put that way, a gateway sounds like any reverse proxy with a middleware chain, and structurally it is. What makes it its own subject is that its payload is a stream. Every feature it gains is a step in front of, or around, one operation: copy the engine's response to the client, chunk by chunk, without delaying the first token.

That one constraint shapes everything in this chapter. Bytes are forwarded and flushed as they arrive, never parsed and re-emitted. Each request is two HTTP exchanges, client to gateway and gateway to engine, so hop-by-hop headers stay on their hop and the gateway's own key never travels further. Before the first byte of a response the gateway can still choose a status (401, 429, 503) or try another worker; after it, the only honest signal of failure is an SSE error event. And the gateway continues the caller's trace with its own span, so one request is one trace from client to engine.

The chapter builds the gateway in the order the request meets it. It starts with a tracer gateway that is the core operation plus a key check and trace propagation, then gives it a skeleton: an ordered middleware chain, `requestid -> otel -> recover -> authn -> policy -> ratelimit -> cache -> route -> proxy -> meter`, a shared request record, graceful shutdown, and configuration. The streaming proxy learns to observe a stream without touching it, the router spreads requests over a fleet of engines with failover, cascades, and disaggregated serving, the usage ledger records who used what and assembles the whole chain, and the policy stage enforces usage restrictions. Two stages of the chain, authentication and rate limiting, are built in [Authorization and access control](../../software-engineering/10-authorization/README.md), and the response cache in [Distributed data and caching](../../systems/06-distributed-data-and-caching/README.md). Each section extends the same `go/gateway/` packages behind the same OpenAI-compatible surface.

> **Learning objectives**
> - Build a streaming reverse proxy that checks keys in constant time, fails closed, flushes every chunk, and propagates W3C trace context and request ids.
> - Compose gateway stages in a contractual order through a composition root, and shut the server down without cutting streams.
> - Observe SSE streams for time to first token and usage without changing a byte, and draw the commitment boundary that decides what may be retried.
> - Route requests across a live fleet with heartbeats, load-aware and cache-aware policies, canaries, cascades, failover, and disaggregated prefill and decode.
> - Meter every request durably in SQLite, serve a fail-closed admin API, and enforce an auditable usage policy.

## Before you begin

You should be comfortable reading Go; [Go: packages, interfaces, goroutines, channels, context, net/http, testing](../../programming-and-algorithms/01-language-primers/06-go.md) in *Programming and Algorithms* covers what the chapter uses. The wire format the gateway forwards is the subject of [Streaming and SSE](../../systems/04-streaming/README.md). The engine the gateway fronts, and its API, comes from [Your first endpoint](../23-model-serving/01-your-first-endpoint.md) in *AI Engineering*. The routing section draws on [Model routing and cascades](../24-model-routing/README.md) for the ideas behind cascades, and the gateway's spans and metrics feed [Observability](../../systems/11-observability/01-one-trace.md) in *Systems Engineering*.

For routing across provider APIs, see [Designing a model router across APIs, caches, and applications](../24-model-routing/01-router-api-cache-application-design.md). It separates application task and tool decisions from provider selection, adapters, prompt caching, and worker scheduling; this chapter implements routing behind one compatible gateway surface.

The full solution code for every section lives in [`course/solutions/go/gateway/`](../../../../course/solutions/go/gateway/).

## In this chapter

1. [The tracer gateway](01-streaming-proxy.md). The thinnest gateway that closes the engine's open port: a static API key checked in constant time and failing closed, SSE pass-through with a flush per chunk, the before-and-after-first-byte failure rules, and `traceparent` and `X-Request-Id` propagation.
2. [The server skeleton](02-server-skeleton.md). The ordered middleware chain and why each stage sits where it does, the composition root that lets the server be built before its stages, the request-scoped `Exchange`, a drain that Kubernetes rollouts cannot cut, and one configuration loader for every Go service.
3. [The SSE streaming proxy](03-sse-streaming-proxy.md). Relaying a stream byte for byte while a splitter observes its events, first byte, and usage on the side, and the commitment boundary that tells the router which failures it may retry.
4. [Routing, cascades, and failover](04-routing.md). A heartbeat registry with exact eviction, three routing policies including prefix affinity with bounded loads, canary splits, failover before the first byte, cascades with a cost and accuracy curve, and disaggregated prefill and decode.
5. [The usage ledger and admin API](05-usage-ledger-and-admin-api.md). A meter that writes one durable SQLite row per request, including stream usage the client never asked for; half-open usage queries; a fail-closed admin front door; and the assembly of every stage into the chain.
6. [Usage policy enforcement](06-usage-policy-enforcement.md). An ordered, first-match policy of model, tenant, token-cap, and classifier rules, with a numerically stable linear head, fail-closed 503s, 451 denials with stable rule ids, and redacted audit events.

## Key terms

- **AI gateway / LLM gateway**: the single front door to a fleet of model servers, where authentication, rate limits, usage policy, caching, routing, and metering live, so the engines behind it stay simple and keyless; its defining constraint is forwarding a token stream without delaying the first token.

## Further reading

- [RFC 9110 HTTP Semantics](https://www.rfc-editor.org/rfc/rfc9110) (free)
- [W3C Trace Context](https://www.w3.org/TR/trace-context/) (free)
- [OpenAI API reference](https://platform.openai.com/docs/api-reference) (free)
- [Envoy AI Gateway](https://aigateway.envoyproxy.io/) (free)
- [LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy) (free)
- [`net/http/httputil.ReverseProxy`](https://pkg.go.dev/net/http/httputil#ReverseProxy) (free)
