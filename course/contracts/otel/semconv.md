# Observability conventions
<!-- chapter: content/courses/build-an-llm/p10-serving/07-serving-metrics-and-tracing.md -->
<!-- chapter: content/tracks/systems/04-observability/01-tracing-for-the-serving-path.md -->

<!-- modules: L10.0 and obs.00 (tracer spans), L10.7 (engine), obs.01 (go/otelx, serving path), obs.02 (metrics and logs), obs.05 (control plane and Python), dur.04, ag.03
     conformance: otel/ (span tree from a recorded trace) -->

One request is one trace, from the client through the gateway, the engines, and the KV transfer; one `TrainRun` is one trace, from the CLI through workflow and activity spans to sampled Python training steps. This page fixes the span names, kinds, parents, and attributes those traces carry. Metrics are in [`metrics.yaml`](metrics.yaml); SLOs in [`slo.schema.json`](slo.schema.json). Attribute names follow the OpenTelemetry semantic conventions (HTTP, RPC, GenAI) where one exists and use the `tl.*` namespace otherwise.

## Propagation

- **W3C Trace Context** everywhere: the `traceparent` header (and `tracestate` when present) on HTTP, the same keys in gRPC metadata.
- The gateway makes its outbound calls children of its `gateway.proxy` CLIENT span (or `gateway.route` for gRPC calls); an engine's SERVER span is the child of the incoming context.
- The durable server stores the starter's context in `WorkflowExecutionStarted.trace_context` and hands it to every task (`ActivityTask.trace_context`); workflow and activity spans are its descendants. A replayed workflow task emits its spans with `tl.workflow.replay = true`.
- Python subprocess activities receive the activity span's context as `TRACEPARENT` and make `train.run` (or `corpus.stage <stage>`) its child.

## Exporters

| Who | How |
|---|---|
| Go services, Rust engine (from L10.7), Python | traces as OTLP to `OTEL_EXPORTER_OTLP_ENDPOINT` (gRPC `:4317` in Kubernetes; Python may use OTLP/HTTP `:4318`); Python also pushes its metrics this way, because a subprocess cannot be scraped |
| Pass 1 tracer engine (L10.0 to L10.6) | hand-written OTLP/HTTP JSON, one `POST /v1/traces` per request, to `:4318` |
| every Go and Rust service | the metrics of `metrics.yaml` on its health port, Prometheus text format, at `/metrics`, scraped by Prometheus (never also pushed, so no series is duplicated; helm/observability.md) |

**Resource attributes:** `service.name` = `<system>-<component>` (`<system>-gateway`, `<system>-engine`, `<system>-durable`, `<system>-worker`, `<system>-agent`, `<system>-python`), `service.namespace` = `<system>`, `service.version` = the learner's `[system].version`, and for engines `tl.engine.role`.

## Spans

| Span name | Kind | Emitted by | Parent | Required attributes |
|---|---|---|---|---|
| `POST /v1/chat/completions` (also `/v1/completions`, `/v1/embeddings`) | SERVER | gateway, engine | client context, or `gateway.proxy` | `http.request.method`, `http.route`, `http.response.status_code`, `gen_ai.operation.name` (`chat`, `text_completion`, `embeddings`), `gen_ai.request.model`; gateway adds `tl.api_key_id`, `tl.tenant`, `tl.api_version` |
| `gateway.auth`, `gateway.policy`, `gateway.ratelimit`, `gateway.cache` | INTERNAL | gateway | the gateway SERVER span | `tl.ratelimit.decision` (`allow`, `deny`), `tl.policy.decision` (`allow`, `deny`) and `tl.policy.rule`, `tl.cache.hit` (bool) |
| `gateway.route` | INTERNAL | gateway | the gateway SERVER span | `tl.route.worker_id`, `tl.route.reason` (`affinity`, `load`, `cascade`, `canary`, `failover`), `tl.route.cascade_step` (int) |
| `gateway.proxy` | CLIENT | gateway | `gateway.route` | `server.address`, `tl.route.worker_id`, `tl.proxy.first_byte_ms` |
| `tl.engine.v1.EngineControl/Prefill` | CLIENT and SERVER | gateway, prefill engine | `gateway.route`; the CLIENT span | `rpc.system = grpc`, `rpc.service`, `rpc.method`, `rpc.grpc.status_code` |
| `engine.queue` | INTERNAL | engine | the engine SERVER span (or the Prefill SERVER span) | `tl.engine.queue_ms`, `tl.engine.priority` |
| `engine.prefill` | INTERNAL | engine | same | `gen_ai.usage.input_tokens`, `tl.engine.prefix_hit_tokens`, `tl.engine.chunks` |
| `engine.decode` | INTERNAL | engine | same | `gen_ai.usage.output_tokens`, `gen_ai.response.finish_reasons` (string array), `tl.engine.spec_accept_rate`; an event `token` every 32 tokens with `tl.engine.tokens` |
| `kv.transfer` | CLIENT (prefill), SERVER (decode) | engines | `engine.prefill`; the client span | `tl.kv.blocks`, `tl.kv.deduped`, `tl.kv.bytes`, `tl.kv.format` |
| `workflow <type>` | INTERNAL | durable SDK | the starter's context | `tl.workflow.id`, `tl.workflow.run_id`, `tl.workflow.replay` |
| `activity <type>` | INTERNAL | worker | `workflow <type>` | `tl.activity.id`, `tl.activity.attempt`, `tl.idempotency_key` |
| `train.run` | INTERNAL | Python | `activity train` (from `TRACEPARENT`) | `tl.train.steps`, `tl.run.id` |
| `train.step` (sampled: every 50th step) | INTERNAL | Python | `train.run` | `tl.train.step`, `tl.train.loss`, `tl.train.tokens` |
| `train.checkpoint` | INTERNAL | Python | `train.run` | `tl.train.step`, `tl.ckpt.path` |
| `corpus.stage <stage>` | INTERNAL | Python | `activity <type>` | `tl.corpus.stage` (`fetch`, `filter`, `dedup_exact`, `dedup_near`, `pii`, `shard`, `tokenize`), `tl.corpus.rows_in`, `tl.corpus.rows_out` |
| `agent.run` | INTERNAL | agent SDK | the caller, or `workflow AgentRun` | `gen_ai.agent.name`, `tl.agent.iterations` |
| `agent.llm_call` | CLIENT | agent SDK | `agent.run` | `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` |
| `agent.tool <name>` | INTERNAL | agent SDK | `agent.run` | `gen_ai.tool.name`, `tl.tool.outcome` (`ok`, `error`, `denied`, `needs_approval`) |
| `rag.retrieve` | INTERNAL | agent SDK | `agent.tool search_docs` | `tl.rag.k`, `tl.rag.retrievers` (string array: `bm25`, `vector`) |
| `eval.case` | INTERNAL | eval runner | the eval run | `tl.eval.suite`, `tl.eval.case_id`, `tl.eval.subject` |

A failing span sets status `ERROR` and `error.type` (the error `code` for HTTP, the gRPC status name for RPC). No span attribute carries prompt or completion text, an API key, or a secret.

## The serving trace, by example

A disaggregated chat request produces this tree (indentation is parentage):

```
POST /v1/chat/completions            SERVER   gateway
  gateway.auth                       INTERNAL gateway
  gateway.policy                     INTERNAL gateway
  gateway.ratelimit                  INTERNAL gateway
  gateway.cache                      INTERNAL gateway   tl.cache.hit = false
  gateway.route                      INTERNAL gateway   tl.route.reason = affinity
    tl.engine.v1.EngineControl/Prefill  CLIENT gateway
      tl.engine.v1.EngineControl/Prefill SERVER prefill engine
        engine.queue                 INTERNAL
        engine.prefill               INTERNAL
          kv.transfer                CLIENT   prefill engine
            kv.transfer              SERVER   decode engine
    gateway.proxy                    CLIENT   gateway
      POST /v1/chat/completions      SERVER   decode engine
        engine.queue                 INTERNAL
        engine.decode                INTERNAL
```

The tracer (Pass 1) has only `gateway.proxy` and the engine's SERVER span (obs.00).

## Logs

Every service writes JSON lines to stdout: `{"ts": "<RFC 3339>", "level": "debug|info|warn|error", "msg": "...", "service": "<service.name>", "trace_id": "<32 hex>", "span_id": "<16 hex>", ...}`. `trace_id` and `span_id` are present whenever a span is active, so a `kubectl logs | grep <trace_id>` finds a request's lines (obs.02). Request logs redact PII with the data.05 detectors (gw.08) and never contain a key or a prompt.
