# Pass 7: The Serving Platform

This pass replaces the tracer's engine and gateway with production ones and runs them the way a production platform would. The Rust engine, written with async Rust, tokio, and hyper, gains continuous batching, chunked prefill, a prefix-caching block manager, disaggregated prefill and decode connected by gRPC, serving metrics and tracing, speculative decoding, and tool calls. A load generator measures it. The Go gateway in front gains authentication and authorization, rate limits, a streaming proxy, consistent-hash routing with cascades and failover, a response cache, and a usage ledger in SQLite that reconciles with what the engine reports.

The last part of the pass deploys the platform on kind with per-role Helm charts and a Tilt dev loop, adds Prometheus, Tempo, and Grafana with SLOs and burn-rate alerts, and rehearses a decode worker dying mid-stream. It closes with a code review and a design review of everything built so far.

> **Learning objectives**
> - Explain how continuous batching, chunked prefill, and the block manager let many streams share one model.
> - Explain why prefill and decode are disaggregated and how the KV transfer between them stays leak-free.
> - Explain how the gateway authenticates, limits, routes, caches, and meters requests without the engine knowing about tenants.
> - Explain how the SLOs, burn-rate alerts, traces, and dashboards explain a violation.
> - Diagnose a decode worker dying mid-stream and keep the system within its error budget.

## Before you begin

This pass needs [Pass 6](../course-p06-inference-and-kernels/README.md), whose Python inference stack defines the correct output the engine must reproduce.

## In this pass

The pass builds the engine first, then the gateway, then the deployment and its observability:

1. [Candle model runner (Llama + bigram, int4), Rust KV cache, sampler and PCG32](../../11-serving/02-model-runner-and-sampler.md). Afterwards you can explain how the candle model runner runs Llama and the bigram (int4 included), and how the Rust KV cache and PCG32 sampler match Python.
2. [Continuous batching scheduler with priority and preemption](../../11-serving/03-continuous-batching.md). Afterwards you can explain how continuous batching admits, prioritizes, and preempts sequences each step.
3. [Chunked prefill (mixed prefill + decode batches)](../../11-serving/04-chunked-prefill.md). Afterwards you can explain how chunked prefill mixes prefill chunks with decode steps under a token budget.
4. [Block manager with prefix cache (none, hash, radix)](../../11-serving/05-block-manager-and-prefix-cache.md). Afterwards you can explain how the block manager allocates KV blocks and how the none, hash, and radix prefix caches reuse them.
5. [Async Rust and tokio: futures, tasks, channels, cancellation, hyper](../../../programming-languages/02-language-and-tool-primers/09-async-rust-and-tokio.md). Afterwards you can explain how futures, tasks, channels, and cancellation work in tokio and how hyper serves a request.
6. [Speculative decoding in the engine](../../11-serving/09-speculative-decoding-in-the-engine.md). Afterwards you can explain how the engine runs speculative decoding and verifies drafts in one batched step.
7. [tl-serve: OpenAI-compatible HTTP + SSE on tokio/hyper](../../11-serving/06-openai-server-on-tokio.md). Afterwards you can trace how tl-serve parses an OpenAI-compatible request and streams SSE on tokio and hyper.
8. [Protocol Buffers and gRPC: schema evolution, unary and streaming RPCs, stubs](../../../programming-languages/02-language-and-tool-primers/10-protocol-buffers-and-grpc.md). Afterwards you can explain how Protobuf schemas evolve compatibly and how unary and streaming gRPC calls are made.
9. [Disaggregated prefill and decode](../../11-serving/07-disaggregated-prefill-decode.md). Afterwards you can explain how disaggregated prefill and decode transfer the KV cache and how the heartbeat client reports health.
10. [Serving metrics and tracing](../../11-serving/08-serving-metrics-and-tracing.md). Afterwards you can explain how the serving metrics, SLO histograms, and OTel spans are recorded and propagated.
11. [Tool calls and constrained decoding](../../11-serving/10-tool-calls-and-constrained-decoding.md). Afterwards you can explain how tool calls are parsed and how constrained JSON decoding guarantees valid arguments in the engine.
12. [An open-loop load generator](../../11-serving/11-load-generator.md). Afterwards you can explain why the load generator is open-loop, how its log-linear histogram records latency, and how the Go PCG32 port matches.
13. [Run comparison and the regression gate](../../11-serving/12-run-comparison-and-regression-gate.md). Afterwards you can explain how two runs are compared and when the regression gate fails.
14. [The server skeleton](../../../ai-platform-engineering/12-gateway/02-server-skeleton.md). Afterwards you can explain how the gateway's composition root wires its parts from go/config and how graceful shutdown drains requests.
15. [API Keys, Scopes, Model Allowlists, and Tenants](../../../ai-platform-engineering/08-authorization-and-access-control/01-api-keys-and-scopes.md). Afterwards you can explain how API keys authenticate a request and how scopes, the model allowlist, and tenants authorize it.
16. [Rate Limiting: RPM Token Buckets and TPM Reserve/Settle](../../../ai-platform-engineering/08-authorization-and-access-control/02-rate-limiting.md). Afterwards you can explain how the RPM token bucket and the TPM reserve-and-settle limit traffic.
17. [SSE streaming proxy](../../../ai-platform-engineering/12-gateway/03-sse-streaming-proxy.md). Afterwards you can explain how the SSE streaming proxy forwards events, handles disconnects, and how it upgrades [the tracer gateway](../../../ai-platform-engineering/12-gateway/01-streaming-proxy.md).
18. [A consistent hash ring with bounded loads](../../../algorithms/13-systems-data-structures/09-consistent-hash-ring.md). Afterwards you can explain how a consistent hash ring with bounded loads places keys and caps each node's share.
19. [Routing, cascades, and failover](../../../ai-platform-engineering/12-gateway/04-routing.md). Afterwards you can explain how the worker registry routes requests, how cascades and failover work, and how disaggregated requests are orchestrated.
20. [Response cache: LRU, TTL, singleflight, and tenant-scoped keys](../../../ai-platform-engineering/04-distributed-data-and-caching/01-response-cache.md). Afterwards you can explain how the response cache combines LRU and TTL, why singleflight collapses duplicate misses, and why keys are tenant-scoped.
21. [SQL and SQLite: schema, transactions, WAL mode, indexes, aggregates](../../../programming-languages/02-language-and-tool-primers/11-sql-and-sqlite.md). Afterwards you can explain how SQLite transactions, WAL mode, and indexes behave and how an aggregate query is written.
22. [The usage ledger and admin API](../../../ai-platform-engineering/12-gateway/05-usage-ledger-and-admin-api.md). Afterwards you can explain how the usage ledger meters requests and how the admin API reads it.
23. [Contract tests: consumer-driven tests for gateway to engine](../../../software-craftsmanship/03-testing-mentality/06-contract-tests.md). Afterwards you can explain how a consumer-driven contract test pins the gateway-to-engine interface.
24. [Dockerfiles for the gateway and the engine](../../../infrastructure/01-containers-kubernetes/02-dockerfiles-for-gateway-and-engine.md). Afterwards you can explain how the multi-stage Dockerfiles build small gateway and engine images.
25. [kind cluster, local registry, and port mappings](../../../infrastructure/01-containers-kubernetes/03-kind-cluster-and-local-registry.md). Afterwards you can explain how the kind cluster, local registry, and port mappings fit together.
26. [Helm charts and the observability stack](../../../infrastructure/01-containers-kubernetes/04-helm-charts-and-observability-stack.md). Afterwards you can explain how the Helm charts deploy the gateway, the engine, and the observability stack.
27. [The Tilt dev loop](../../../infrastructure/01-containers-kubernetes/05-tilt-dev-loop.md). Afterwards you can explain how the Tilt dev loop rebuilds and redeploys on change.
28. [CI for the learner repo](../../../infrastructure/01-containers-kubernetes/06-ci-for-your-repo.md). Afterwards you can explain what the repository's CI runs and in what order.
29. [Tracing for the serving path](../../../systems/04-observability/02-tracing-for-the-serving-path.md). Afterwards you can trace the spans along the serving path and how their context propagates.
30. [Metrics and log correlation](../../../systems/04-observability/03-metrics-and-log-correlation.md). Afterwards you can explain how metrics and logs are correlated with traces.
31. [SLOs and multi-window burn-rate alerts](../../../systems/04-observability/04-slos-and-burn-rate-alerts.md). Afterwards you can explain how SLOs set error budgets and why multi-window burn-rate alerts fire when they do.
32. [Dashboards as code for the serving path](../../../systems/04-observability/05-dashboards-as-code.md). Afterwards you can explain how the dashboards are defined as code for the serving path.
33. [A decode worker dies mid-stream](../../../systems/05-incident-response-and-chaos/02-kill-decode.md). Afterwards you can diagnose a decode worker dying mid-stream from its symptoms, explain the fix, and explain how recurrence is prevented.
34. [Code review: review a seeded PR against your system](../../../software-craftsmanship/05-code-review-and-ci/02-code-review.md). Afterwards you can identify the defects in the seeded PR and explain why each matters.
35. [Design review: the tracer and the serving platform](../../../systems/01-system-design/01-design-review-tracer-and-serving-platform.md). Afterwards you can explain the design decisions of the tracer and serving platform and the tradeoffs a design review raises.
36. [What the System Does After Pass 7](milestone.md). Afterwards you can explain how the engine, the gateway, and observability work together on kind to hold the SLOs.

The stages are also in this pass's manifest, [`path.tsv`](path.tsv). [Pass 8](../course-p08-durable/README.md) turns to the long-running jobs behind the platform: corpus builds and training runs.
