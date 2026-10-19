# What the System Does After Pass 7

After Pass 7 the tracer's engine and gateway have been replaced by production ones, and the platform runs on kind the way it would in production. The Rust inference engine is conformant, concurrent, observable, speculative, and tool-calling. The gateway conforms to the API, and its usage ledger reconciles with the engine. On kind, the gateway, disaggregated engines, and the observability stack hold the SLOs. Everything from earlier passes keeps working. This page gives the commands and configuration the finished solution runs, what each produces, and why each piece is needed.

The engine is the [Serving](../../p10-serving/README.md) chapter, from [the model runner and sampler](../../p10-serving/01-model-runner-and-sampler.md) through [continuous batching](../../p10-serving/02-continuous-batching.md), [chunked prefill](../../p10-serving/03-chunked-prefill.md), [the block manager](../../p10-serving/04-block-manager-and-prefix-cache.md), [the tokio server](../../p10-serving/05-openai-server-on-tokio.md), [disaggregated prefill and decode](../../p10-serving/06-disaggregated-prefill-decode.md), [metrics and tracing](../../p10-serving/07-serving-metrics-and-tracing.md), [speculative decoding](../../p10-serving/08-speculative-decoding-in-the-engine.md), and [tool calls](../../p10-serving/09-tool-calls-and-constrained-decoding.md), measured by [the load generator](../../p10-serving/10-load-generator.md) and written with the primers on [async Rust and tokio](../../../programming-languages/02-language-and-tool-primers/09-async-rust-and-tokio.md) and [Protocol Buffers and gRPC](../../../programming-languages/02-language-and-tool-primers/10-protocol-buffers-and-grpc.md). The gateway spans *AI Platform Engineering*, from [the server skeleton](../../../ai-platform-engineering/12-gateway/01-server-skeleton.md), [API keys and scopes](../../../ai-platform-engineering/08-authorization-and-access-control/01-api-keys-and-scopes.md), [rate limiting](../../../ai-platform-engineering/08-authorization-and-access-control/02-rate-limiting.md), and [the SSE streaming proxy](../../../ai-platform-engineering/12-gateway/04-sse-streaming-proxy.md) to [routing](../../../ai-platform-engineering/12-gateway/05-routing.md), [the response cache](../../../ai-platform-engineering/04-distributed-data-and-caching/01-response-cache.md), and [the usage ledger](../../../ai-platform-engineering/12-gateway/07-usage-ledger-and-admin-api.md), with [a consistent hash ring](../../../algorithms/16-systems-data-structures/09-consistent-hash-ring.md), [SQL and SQLite](../../../programming-languages/02-language-and-tool-primers/11-sql-and-sqlite.md), and [contract tests](../../../software-craftsmanship/03-testing-mentality/06-contract-tests.md). Production on kind draws on the deploy sections from [Dockerfiles](../../../infrastructure/01-containers-kubernetes/01-dockerfiles-for-gateway-and-engine.md) to [CI for the platform repo](../../../infrastructure/01-containers-kubernetes/05-ci-for-your-repo.md), the observability sections from [tracing for the serving path](../../../systems/04-observability/01-tracing-for-the-serving-path.md) to [dashboards as code](../../../systems/04-observability/04-dashboards-as-code.md), the drill [A decode worker dies mid-stream](../../../systems/05-incident-response-and-chaos/01-kill-decode.md), and the [code review](../../../software-craftsmanship/05-code-review-and-ci/02-code-review.md) and [design review](../../../systems/01-system-design/01-design-review-tracer-and-serving-platform.md) that close the pass.

## The inference engine: conformant, concurrent, observable, speculative, tool-calling

The Rust engine serves the OpenAI subset v1, [`openai-subset.v1.yaml`](../../../../../course/contracts/openapi/openai-subset.v1.yaml), conformantly: streaming, stop strings, seeds, 16 concurrent streams, abort on disconnect, and tool calls. It keeps its KV pool clean under a burst of 64 requests from its own load generator, reports its metrics under the contract's names, keeps greedy output unchanged with prompt-lookup speculation on, and answers a forced tool call with a schema-valid call. A tiny random Llama shows all of this in seconds. It is the `tiny-llama-2l` of [the Llama-family model](../../p07-modern-block/09-llama-family-model.md), in a 1024-position copy, because a tool request's schema prompt is about 470 byte tokens, past the original 256. SmolLM2-135M-Instruct served the same way chooses for itself when to call a tool.

Why this matters: this engine replaces the tracer's in every later pass. Continuous batching is what lets 16 streams share one model. A clean KV pool after a burst is the evidence that aborted and finished requests free their blocks. The metric names are what the SLO rules below query. And greedy equality with speculation on is the same correctness rule as Pass 6.

The roles are wired as [`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md) describes:

```text
engine           tl-serve --config <runtime.toml>; the runtime.toml's
                 [engine].model_dir names the model, and [engine].model_id
                 the name clients ask for ("tracer").
                 Health: http://127.0.0.1:{health_port}/healthz
engine-spec      the same template with
                 [engine].speculative = { draft = "prompt_lookup", k = 4 }
engine-prefill, engine-decode-a, engine-decode-b
                 the same model with [engine].role = "prefill" and
                 "decode", each with a gRPC port and a KV transfer port
loadgen          {loadgen} --target <base url without /v1> --model <m>
                 --prompt <text> --max-tokens <n> --mode burst
                 --burst <n> --duration <d> --out <report.json>:
                 writes the report (formats/loadgen-report.schema.json)
                 to --out and prints it as the final stdout line
```

The runtime templates (`engine`, `engine-spec`, `engine-prefill`, and `engine-decode`) are in [`config/`](../../../../../course/solutions/entry/config/engine.runtime.toml), and the load generator is [`go/cmd/loadgen`](../../../../../course/solutions/entry/go/cmd/loadgen/main.go).

Two comparisons show that the engines agree with each other. The first is greedy equality: the plain and the speculative engine give the same greedy text. The second is disaggregation without the gateway. Tokenize on the prefill engine, call `EngineControl.Prefill` there with `decode_target` set to a decode engine's KV port, then send the same request to that decode engine with `X-TL-KV-Handle`. Greedy output (plain and streamed, alternating the two decode engines) and seeded output equal the unified engine's. An unknown or released handle answers 404 `kv_handle_not_found`, and a transfer reset mid-push fails `Prefill` without leaking a KV block on either side, after which that decode engine serves again.

Disaggregation exists because prefill is compute-bound and decode is memory-bound, so running them on separate pools lets each be sized for its own bottleneck. Finally, the Rust and Python samplers agree on the same logits, and the engine's greedy stream equals the Python model's on the tiny model.

## The gateway conforms to the API, and its ledger reconciles with the engine

The gateway, in front of the engine, is checked as a client sees it. The gateway tier of `openai-subset.v1` passes: auth, limits, cache, headers, and `priority.internal`. The usage-policy cases, such as `policy.451`, wait for Pass 10. The engine tier passes on its own, which is the provider side of the contract tests. And the usage ledger reconciles with the usage the engine reported for a fixed workload, both through the admin API, [`admin.v1.yaml`](../../../../../course/contracts/openapi/admin.v1.yaml), and through the CLI.

The roles are wired like this:

```text
engine     the --config form, serving the byte-level model "tracer"
           (decision D32): "Once" is 4 prompt tokens
gateway    the --config form; health /readyz, started after the engine;
           its config sets [gateway].usage_db to a fresh ledger, routes
           model "tracer" to the engine, and accepts the key in
           $TL_API_KEY with the scopes infer and admin
ctl        the umbrella CLI, with the usage verb below
```

The `usage` verb of the `ctl` role is in [`go/cmd/ctl/usage.go`](../../../../../course/solutions/entry/go/cmd/ctl/usage.go):

```text
{ctl} usage --gateway <base URL> [--tenant T] [--key-id K] [--since T] [--until T]
            [--group-by none|model|tenant|key_id|api_version] [--json]
    GET <base>/admin/v1/usage with the key in $TL_API_KEY. With --json: one
    JSON line per row, then a final line of totals over the rows:
    {"requests": N, "errors": E, "prompt_tokens": P, "completion_tokens": C, "cached_tokens": K}.
    Exit 1 on an HTTP error, with the error's message on stderr.
```

A fixed workload makes the reconciliation exact: 8 completions and 4 streams of `max_tokens` 4 on "Once", each with 4 prompt and 4 completion tokens, so the ledger holds exactly 12 requests, 48 prompt tokens, and 48 completion tokens. Failover before the first byte is part of routing: a backend killed before its first byte is retried elsewhere.

Why this matters: the gateway is where tenants, keys, limits, and billing live, so the engine never has to know about them. The ledger is what usage is billed from, and it is only trustworthy if it agrees with what the engine actually produced.

## Production on kind holds the SLOs

The serving platform runs on the kind cluster as it would in production: the gateway (1 replica) in front of the engines deployed as 1 prefill and 2 decode, the observability stack (collector, Prometheus with the SLO rules, Tempo, and Grafana) scraping and tracing it, the load generator holding the SLO rate within budget, and the drill in which a decode pod is killed mid-load resolved within the error budget.

The Helm charts are under [`deploy/helm`](../../../../../course/solutions/entry/deploy/helm/) and the observability chart under [`deploy/observability`](../../../../../course/solutions/entry/deploy/observability/slo.yaml):

```text
loadgen      {loadgen} --target <base url> --rate <rps> --duration <d>;
             sends the API key from $TL_API_KEY and prints its report
             (formats/loadgen-report.schema.json) as the final JSON line
deploy       kube_context "kind-<system>", namespace "<system>",
             gateway_url  NodePort 30080 (http://127.0.0.1:30080),
             prometheus   NodePort 30090 (http://127.0.0.1:30090),
             traces       Tempo query API, NodePort 30320,
             deployments  <system>-gateway, <system>-engine-prefill,
                          <system>-engine-decode
```

The names the SLO queries rely on are fixed by [`observability.md`](../../../../../course/contracts/helm/observability.md). The Deployments and Services are `<system>-gateway`, `<system>-engine-prefill`, and `<system>-engine-decode` in the namespace. Every Service is scraped on `:9464` by a ServiceMonitor labelled `release: observability`, so its series carry `namespace` and `service` labels. `OTEL_SERVICE_NAME` is `<system>-gateway` or `<system>-engine`. And the PrometheusRules are rendered from `slo.yaml` with the `prod` window profile and the label `release: observability`.

The SLO bounds are a TTFT p95 of 0.5 s, a TPOT p95 of 60 ms at 4 requests per second, and a 5xx ratio under 0.5%. They are calibrated on a fast laptop, so a slower machine can miss them with a correct system.

Why this matters: the SLOs turn "fast enough" into numbers the alerting rules can burn against, the observability stack is how a violation is explained, and the drill shows that the system degrades within budget when a decode pod dies, which is what production will do to it. [Pass 8](../course-p08-durable/README.md) turns to the long-running jobs behind the platform.
