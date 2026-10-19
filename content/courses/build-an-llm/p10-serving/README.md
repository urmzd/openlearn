# Part 10: Serving

The Rust engine: an OpenAI-compatible HTTP server that streams completions from a checkpoint through Candle tensor operations. Pass 1 has one chapter here: a std-only HTTP/1.1 and SSE server (no async runtime, no dependencies) serving the byte bigram, with one hand-written OTLP span per request. Pass 7 rebuilds it on tokio and hyper with continuous batching, chunked prefill, disaggregation, speculative decoding, and tool calls, behind the same API.

**Course passes**: 1 (L10.0, gate [MS-P1](../paths/course-p01-tracer/milestone.md)), 7 (L10.1 to L10.9, load.01, load.02, gate MS-P7).

**Before you start**: the [Rust](../../../tracks/software-craftsmanship/12-language-and-tool-primers/04-rust.md) and [HTTP and SSE](../../../tracks/software-craftsmanship/12-language-and-tool-primers/05-http-and-sse.md) primers; the API contract [`openai-subset.v0.yaml`](../../../../course/contracts/openapi/openai-subset.v0.yaml) and the engine role in [`spec/cli-roles.md`](../../../../course/contracts/spec/cli-roles.md).

| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `L10.0` | [Your first endpoint: a std-only Rust HTTP/1.1 + SSE server](00-your-first-endpoint.md) | build | 1 |
| 2 | `L10.1` | [Candle model runner (Llama + bigram, int4), Rust KV cache, sampler and PCG32](01-model-runner-and-sampler.md) | build | 7 |
| 3 | `L10.2` | [Continuous batching scheduler with priority and preemption](02-continuous-batching.md) | build | 7 |
| 4 | `L10.3` | [Chunked prefill (mixed prefill + decode batches)](03-chunked-prefill.md) | build | 7 |
| 5 | `L10.4` | [Block manager with prefix cache (none, hash, radix)](04-block-manager-and-prefix-cache.md) | build | 7 |
| 6 | `L10.5` | [tl-serve: OpenAI-compatible HTTP + SSE on tokio/hyper](05-openai-server-on-tokio.md) | build | 7 |
| 7 | `L10.6` | [Disaggregated prefill/decode, KV transfer, heartbeat client](06-disaggregated-prefill-decode.md) | build | 7 |
| 8 | `L10.7` | [Serving metrics, SLO histograms, OTel spans and propagation](07-serving-metrics-and-tracing.md) | build | 7 |
| 9 | `L10.8` | [Speculative decoding in the engine](08-speculative-decoding-in-the-engine.md) | build | 7 |
| 10 | `L10.9` | [Tool calls and constrained JSON decoding in the engine](09-tool-calls-and-constrained-decoding.md) | build | 7 |
| 11 | `load.01` | [Open-loop load generator, log-linear histogram, Go PCG32 port](10-load-generator.md) | build | 7 |
| 12 | `load.02` | [Run comparison and regression gate](11-run-comparison-and-regression-gate.md) | side | 7 |
