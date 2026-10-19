# What the system does after Pass 7

**Pass result**: async Rust, gRPC, and SQL primers; candle-based Rust engine (continuous batching, chunked prefill, disaggregation, speculative decoding, tool calls), gateway (auth, limits, routing, cascades, cache, ledger), loadgen, gateway and engine charts, Tilt, Prometheus, Tempo, Grafana, SLOs.

Each capability below is built from the listed modules. The sections that follow give the commands and configuration the finished solution runs, what each one produces, and why each piece is needed. Everything from earlier passes keeps working.

| What the system does | Built from |
|---|---|
| The inference engine: conformant, concurrent, observable, speculative, tool-calling. | `L10.1`, `L10.2`, `L10.3`, `L10.4`, `L10.5`, `L10.6`, `L10.7`, `L10.8`, `L10.9`, `load.01` |
| The gateway conforms to the API and its usage ledger reconciles with the engine. | `lang.11`, `ds.09`, `gw.01`, `gw.02`, `gw.03`, `gw.04`, `gw.05`, `gw.06`, `gw.07`, `craft.20` |
| Production on kind: the gateway, disaggregated engines, and observability hold the SLOs. | `load.01`, `dep.01`, `dep.02`, `dep.03`, `dep.04`, `dep.05`, `obs.01`, `obs.02`, `obs.03`, `obs.04`, `ops.01`, `craft.08`, `review.01` |
| The async Rust and gRPC primers the engine is written with. | `lang.09`, `lang.10` |

## Details

## The inference engine: conformant, concurrent, observable, speculative, tool-calling

The Rust engine serves the OpenAI subset v1
([`openai-subset.v1.yaml`](../../../../../course/contracts/openapi/openai-subset.v1.yaml))
conformantly (streaming, stop strings, seeds, 16 concurrent streams, abort
on disconnect, tool calls), keeps its KV pool clean under a burst of 64
requests from its own load generator, reports its metrics in the contract's
names, keeps greedy output unchanged with prompt-lookup speculation on, and
answers a forced tool call with a schema-valid call. A tiny random Llama
(L7.9's tiny-llama-2l, in a 1024-position copy because a tool request's
schema prompt is about 470 byte tokens, past the original 256) shows all of
this in seconds; SmolLM2-135M-Instruct served the same way chooses for
itself when to call a tool.

Why it is needed: this engine replaces the tracer's in every later pass.
Continuous batching is what lets 16 streams share one model; a clean KV
pool after a burst is the evidence that aborted and finished requests free
their blocks; the metric names are what the SLO rules below query; and
greedy equality with speculation on is the same correctness rule as Pass 6.

How the solution is wired ([`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md)):

    engine           `tl-serve --config <runtime.toml>` (L10.5); the
                     runtime.toml's [engine].model_dir names the model, and
                     [engine].model_id the name clients ask for ("tracer").
                     Health: http://127.0.0.1:{health_port}/healthz
    engine-spec      the same template with
                     [engine].speculative = { draft = "prompt_lookup", k = 4 }
    engine-prefill, engine-decode-a, engine-decode-b
                     the same model with [engine].role = "prefill" and
                     "decode" (L10.6), each with a gRPC port and a KV
                     transfer port
    loadgen          {loadgen} --target <base url without /v1> --model <m>
                     --prompt <text> --max-tokens <n> --mode burst
                     --burst <n> --duration <d> --out <report.json>:
                     writes the report (formats/loadgen-report.schema.json)
                     to --out and prints it as the final stdout line

The runtime templates are in
[`config/`](../../../../../course/solutions/entry/config/engine.runtime.toml)
(`engine`, `engine-spec`, `engine-prefill`, `engine-decode`), the load
generator is [`go/cmd/loadgen`](../../../../../course/solutions/entry/go/cmd/loadgen/main.go).

Two comparisons show the engines agree with each other. Greedy equality:
the plain and the speculative engine give one greedy text. Disaggregation
without the gateway: tokenize on the prefill engine, call
`EngineControl.Prefill` there with `decode_target` set to a decode engine's
KV port, then send the same request to that decode engine with
`X-TL-KV-Handle`. Greedy (plain and streamed, alternating the two decode
engines) and seeded output equal the unified engine's; an unknown or
released handle answers 404 `kv_handle_not_found`; and a transfer reset
mid-push fails `Prefill` without leaking a KV block on either side, after
which that decode engine serves again. Disaggregation exists because
prefill is compute-bound and decode is memory-bound, so running them on
separate pools lets each be sized for its own bottleneck.

The Rust and Python samplers agree on the same logits, and the engine's
greedy stream equals the Python model's on the tiny model (L10.1).

## The gateway conforms to the API and its usage ledger reconciles with the engine

The gateway (gw.01 to gw.07) in front of the engine (L10.5), as a client
sees it: the gateway tier of openai-subset.v1 passes (auth, limits, cache,
headers, priority.internal; the usage-policy cases such as policy.451 wait
for Pass 10), the engine tier passes on its own (the provider side of
craft.20's contract), and the usage ledger (gw.07) reconciles with the
usage the engine reported for a fixed workload, through the admin API
([`admin.v1.yaml`](../../../../../course/contracts/openapi/admin.v1.yaml))
and through the CLI.

How the solution is wired:

    engine     the --config form (L10.5), serving the byte-level model
               "tracer" (D32): "Once" is 4 prompt tokens
    gateway    the --config form (gw.01); health /readyz, started after
               the engine; its config sets [gateway].usage_db to a fresh
               ledger, routes model "tracer" to the engine, and accepts the
               key in $TL_API_KEY with the scopes infer and admin
    ctl        the umbrella CLI, with the `usage` verb below

The `usage` verb of the `ctl` role
([`go/cmd/ctl/usage.go`](../../../../../course/solutions/entry/go/cmd/ctl/usage.go)):

    {ctl} usage --gateway <base URL> [--tenant T] [--key-id K] [--since T] [--until T]
                [--group-by none|model|tenant|key_id|api_version] [--json]
      GET <base>/admin/v1/usage with the key in $TL_API_KEY. With --json: one
      JSON line per row, then a final line of totals over the rows:
      {"requests": N, "errors": E, "prompt_tokens": P, "completion_tokens": C, "cached_tokens": K}.
      Exit 1 on an HTTP error, with the error's message on stderr.

A fixed workload makes the reconciliation exact: 8 completions and 4
streams of max_tokens 4 on "Once", each 4 prompt and 4 completion tokens,
so the ledger holds exactly 12 requests, 48 prompt tokens and 48 completion
tokens. Failover before the first byte is part of gw.05: a backend killed
before its first byte is retried elsewhere.

Why it is needed: the gateway is where tenants, keys, limits, and billing
live, so the engine never has to know about them; the ledger is what usage
is billed from, and it is only trustworthy if it agrees with what the
engine actually produced.

## Production on kind: the gateway, disaggregated engines, and observability hold the SLOs

The serving platform runs on the kind cluster the way it would in
production: the gateway (1 replica) in front of the engines deployed as
1 prefill + 2 decode, the observability stack (collector, Prometheus with
the SLO rules, Tempo, Grafana) scraping and tracing it, the load generator
holding the SLO rate within budget, and drill ops.01 (a decode pod killed
mid-load) resolved within the error budget.

How the solution is deployed (Helm charts under
[`deploy/helm`](../../../../../course/solutions/entry/deploy/helm/), the
observability chart under
[`deploy/observability`](../../../../../course/solutions/entry/deploy/observability/slo.yaml)):

    loadgen      {loadgen} --target <base url> --rate <rps> --duration <d>;
                 sends the API key from $TL_API_KEY and prints its report
                 (formats/loadgen-report.schema.json) as the final JSON line
    deploy       kube_context "kind-<system>", namespace "<system>",
                 gateway_url  NodePort 30080 (http://127.0.0.1:30080),
                 prometheus   NodePort 30090 (http://127.0.0.1:30090),
                 traces       Tempo query API, NodePort 30320,
                 deployments  <system>-gateway, <system>-engine-prefill,
                              <system>-engine-decode

The names the SLO queries rely on are fixed by
[`observability.md`](../../../../../course/contracts/helm/observability.md):
Deployments and Services `<system>-gateway`, `<system>-engine-prefill`,
`<system>-engine-decode` in the namespace; every Service scraped on :9464
by a ServiceMonitor labelled `release: observability` (so its series carry
`namespace` and `service` labels); `OTEL_SERVICE_NAME` `<system>-gateway`
and `<system>-engine`; and the PrometheusRules rendered from `slo.yaml`
with the `prod` window profile and the label `release: observability`.

The SLO bounds are TTFT p95 0.5 s, TPOT p95 60 ms at 4 rps, and a 5xx ratio
under 0.5%. They are calibrated on a fast laptop, so a slower machine can
miss them with a correct system.

Why it is needed: the SLOs turn "fast enough" into numbers the alerting
rules can burn against, the observability stack is how a violation is
explained, and the drill shows the system degrades within budget when a
decode pod dies, which is what production will do to it.
