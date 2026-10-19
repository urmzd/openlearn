# Gateway

## Overview

- **Primary references**: [RFC 9110 HTTP Semantics](https://www.rfc-editor.org/rfc/rfc9110) (free), [W3C Trace Context](https://www.w3.org/TR/trace-context/) (free), [OpenAI API reference](https://platform.openai.com/docs/api-reference) (free)
- **Supplementary**: [Envoy AI Gateway](https://aigateway.envoyproxy.io/) (free), [LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy) (free), [`net/http/httputil.ReverseProxy`](https://pkg.go.dev/net/http/httputil#ReverseProxy) (free)
- **Prerequisites**: [Streaming & SSE](../03-streaming-sse/), the Go primer [`lang.06`](../../programming-languages/02-language-and-tool-primers/06-go.md), the engine's API (`L10.0`)
- **Estimated time**: Pass 1 for the tracer (`gw.00`); the full gateway arrives in Pass 7

## Key Takeaways

- A gateway is the **one front door** to every model server: authentication, limits, routing, caching, and metering live there, so the engines stay simple and keyless.
- Everything it adds must leave **streaming intact**: bytes are forwarded and flushed as they arrive, and failures after the first byte become SSE error events.
- The gateway **continues the caller's trace** with its own span, so one request is one trace from client to engine.

## How to Study

Read the chapters in order. Each one builds the idea from first principles, then walks through the solution in [`course/solutions/go/gateway/`](../../../../course/solutions/go/gateway/) and explains why it is written that way. The tracer chapter (`gw.00`) is in Pass 1; the server skeleton, streaming observer, routing, and the rest follow in Pass 7, each extending the same `go/gateway/` packages behind the same OpenAI-compatible surface.

---

# Concepts & Techniques

## Core Insight

An LLM gateway is a reverse proxy whose payload is a stream. Every feature it gains (keys, rate limits, caching, routing, usage accounting) is a step in a middleware chain in front of one operation: copy the engine's response to the client, chunk by chunk, without delaying the first token. The tracer gateway is that operation plus a key check and trace propagation; later modules insert the rest of the chain around it.

## 1. The request path

**Key ideas**:
- Two HTTP exchanges per request: client to gateway, gateway to engine. Hop-by-hop headers stay on their hop; the gateway's own key never travels further.
- Middleware order is a contract (`gw.01`): `requestid -> otel -> recover -> authn -> policy -> ratelimit -> cache -> route -> proxy -> meter`.
- Before the first byte the gateway can still choose a status (401, 429, 503); after it, only an SSE error event.

## Connections to Other Tracks

| Track | Connection |
|---|---|
| [Streaming & SSE](../03-streaming-sse/) | the wire format the gateway forwards |
| [Authorization & Access Control](../08-authorization-and-access-control/) | API keys and scopes (`gw.02`, `gw.03`) |
| [Distributed Data & Caching](../04-distributed-data-and-caching/) | the response cache (`gw.06`) |
| [Model Routing & Cascades](../11-model-routing-and-cascades/) | routing and cascades (`gw.05`) |
| [Observability](../../systems/04-observability/) | the `gateway.proxy` span and gateway metrics (`obs.*`) |

## Chapters

| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `gw.00` | [Tracer gateway: static API-key check, SSE pass-through without buffering, traceparent and X-Request-Id propagation](00-streaming-proxy.md) | build | 1 |
| 2 | `gw.01` | [Server skeleton, composition root, go/config, graceful shutdown](01-server-skeleton.md) | build | 7 |
| 3 | `gw.04` | [SSE streaming proxy (upgrades gw.00)](04-sse-streaming-proxy.md) | build | 7 |
| 4 | `gw.05` | [Worker registry, routing, cascades, failover, disaggregated orchestration](05-routing.md) | build | 7 |
| 5 | `gw.07` | [Usage ledger, metering, admin API](07-usage-ledger-and-admin-api.md) | build | 7 |
| 6 | `gw.08` | [Usage policy enforcement](08-usage-policy-enforcement.md) | build | 10 |
