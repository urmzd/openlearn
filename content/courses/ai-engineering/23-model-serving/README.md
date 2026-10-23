# Build a Model Server

Everything earlier in this book produces a checkpoint. This chapter turns a checkpoint into a service: a Rust engine behind an OpenAI-compatible HTTP API that streams completions through Candle tensor operations. By the end, the same API that any OpenAI client can call is backed by continuous batching, a paged and prefix-cached KV pool, disaggregated prefill and decode, speculative decoding, constrained tool calls, and the metrics and traces that let you tell how well it all works.

The chapter grows the engine in two movements. It opens with the smallest server that can possibly work: a std-only HTTP/1.1 and SSE server, with no async runtime and no dependencies, serving the byte bigram and emitting one hand-written OTLP span per request. That first endpoint fixes the API and the streaming format. The rest of the chapter rebuilds the engine behind that same API on tokio and hyper, one capability at a time: a real model runner and sampler, a batching scheduler, chunked prefill, the block manager, the production server, disaggregation across engines, observability, speculation, and tool calls. The last two sections step outside the engine to measure it: an open-loop load generator in Go and a statistical regression gate built on its reports.

> **Learning objectives**
> - Serve streamed completions over HTTP and SSE, first with only the standard library and then on tokio and hyper, behind one OpenAI-compatible contract.
> - Schedule many sequences at once with continuous batching, chunked prefill, priorities, and preemption, over a paged KV pool with a prefix cache.
> - Split prefill and decode across engines, moving only the KV blocks the receiver lacks, without leaking blocks on any error path.
> - Expose contract-exact Prometheus metrics and W3C-propagated OTLP traces for every request.
> - Speed up decoding with exact speculative verification, and guarantee schema-valid tool calls with constrained decoding.
> - Measure the engine honestly with an open-loop load generator, and gate regressions with a permutation test.

## Before you begin

You should be comfortable reading Rust and know how HTTP and server-sent events work; the [Rust](../../programming-and-algorithms/01-language-primers/04-rust.md) and [HTTP and SSE](../../programming-and-algorithms/01-language-primers/05-http-and-sse.md) primers in *Programming and Algorithms* cover both, and the later sections also lean on the primers for [async Rust and tokio](../../programming-and-algorithms/01-language-primers/09-async-rust-and-tokio.md), [Protocol Buffers and gRPC](../../programming-and-algorithms/01-language-primers/10-protocol-buffers-and-grpc.md), and [Go](../../programming-and-algorithms/01-language-primers/06-go.md). Many sections port a Python reference from [the inference chapter](../21-inference/README.md), so having read it helps. Two contracts frame the whole chapter: the API in [`openai-subset.v0.yaml`](../../../../course/contracts/openapi/openai-subset.v0.yaml) (and its later `v1`), and the engine role in [`spec/cli-roles.md`](../../../../course/contracts/spec/cli-roles.md).

## In this chapter

1. [Your first endpoint: a std-only Rust HTTP/1.1 and SSE server](01-your-first-endpoint.md). The smallest working engine: hand-parsed HTTP, streamed SSE completions from the byte bigram, and one OTLP span per request, with no dependencies at all.
2. [The model runner and sampler](02-model-runner-and-sampler.md). A Candle runner for Llama-family and bigram models (with int4 weights), the Rust KV cache and block pool, and the sampler with its PCG32 generator, held to the Python reference.
3. [Continuous batching](03-continuous-batching.md). A scheduler that admits and retires sequences every step, with priorities and preemption, so the batch stays full.
4. [Chunked prefill](04-chunked-prefill.md). Splitting long prompts into chunks that share a batch with decodes, so one long prompt no longer stalls every stream.
5. [The block manager and prefix cache](05-block-manager-and-prefix-cache.md). Allocating KV blocks per sequence and reusing shared prefixes, with no cache, a hash cache, or a radix cache.
6. [The OpenAI-compatible server on tokio](06-openai-server-on-tokio.md). The production server, `tl-serve`: the engine step loop behind hyper, chat templates, bounded admission, abort on disconnect, drain on SIGTERM, and health and readiness routes.
7. [Disaggregated prefill and decode](07-disaggregated-prefill-decode.md). Prefill on one engine and decode on another, with deduplicated KV transfer over gRPC, an exact RNG hand-off, and a heartbeat to the gateway's registry.
8. [Serving metrics and tracing](08-serving-metrics-and-tracing.md). A hand-written Prometheus registry with SLO histograms, and W3C trace propagation with OTLP/HTTP JSON export.
9. [Speculative decoding in the engine](09-speculative-decoding-in-the-engine.md). Prompt-lookup and n-gram drafts, exact greedy and accept-or-resample verification, and KV rollback of rejected drafts.
10. [Tool calls and constrained decoding](10-tool-calls-and-constrained-decoding.md). A canonical regex-to-DFA compiler, JSON-schema grammars, token masks over a byte trie, and the OpenAI tool-calling surface with streamed deltas.
11. [An open-loop load generator](11-load-generator.md). A Go load generator that avoids coordinated omission, measures TTFT, ITL, and TPOT from the stream, and records them in log-linear histograms.
12. [Run comparison and the regression gate](12-run-comparison-and-regression-gate.md). An optional section: a seeded one-sided permutation test that fails CI only when a run is both worse beyond a budget and significantly so.
13. [Inference stack: from tokens to an interface](13-inference-stack-contracts-and-configuration.md). Provider-neutral contracts and configuration across preparation, generation, structured output, interpretation, runtime, and delivery, with defaults, precedence, failure symptoms, and measurement methods.
