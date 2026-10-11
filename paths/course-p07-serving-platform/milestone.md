# Pass 7 milestones

**Pass result**: async Rust, gRPC, and SQL primers; candle-based Rust engine (continuous batching, chunked prefill, disaggregation, speculative decoding, tool calls), gateway (auth, limits, routing, cascades, cache, ledger), loadgen, gateway and engine charts, Tilt, Prometheus, Tempo, Grafana, SLOs.

The path includes the milestone stages below. Run each component milestone after its modules pass, then run the pass gate. The component gates run before the pass gate, which also reruns the smoke steps of earlier passes.

| Gate | Specification | What it covers |
|---|---|---|
| `MS-L10` | [`MS-L10.toml`](../../course/milestones/MS-L10.toml) | Your inference engine: conformant, concurrent, observable, speculative, tool-calling. Requires `L10.1`, `L10.2`, `L10.3`, `L10.4`, `L10.5`, `L10.6`, `L10.7`, `L10.8`, `L10.9`, `load.01`. |
| `MS-gateway` | [`MS-gateway.toml`](../../course/milestones/MS-gateway.toml) | Your gateway passes conformance and its usage ledger reconciles with the engine. Requires `lang.11`, `ds.09`, `gw.01`, `gw.02`, `gw.03`, `gw.04`, `gw.05`, `gw.06`, `gw.07`, `craft.20`. |
| `MS-prod` | [`MS-prod.toml`](../../course/milestones/MS-prod.toml) | Production on kind: your gateway, disaggregated engines, and observability hold the SLOs. Requires `load.01`, `dep.01`, `dep.02`, `dep.03`, `dep.04`, `dep.05`, `obs.01`, `obs.02`, `obs.03`, `obs.04`, `ops.01`, `craft.08`, `review.01`. |
| `MS-P7` | [`MS-P7.toml`](../../course/milestones/MS-P7.toml) | Serving platform: inference engine, gateway, and production operations. Requires `lang.09`, `lang.10`. |


## Component gate details

## MS-L10: Your inference engine: conformant, concurrent, observable, speculative, tool-calling

5.7 catalog, 7.2). MS-P7 = MS-L10 + MS-gateway + MS-prod.

Your Rust engine, started from your own entry point, serves the OpenAI
subset v1 conformantly (streaming, stop strings, seeds, 16 concurrent
streams, abort on disconnect, tool calls), keeps its KV pool clean under a
burst of 64 requests from your own load generator, reports its metrics in
the contract's names, keeps greedy output unchanged with prompt-lookup
speculation on, and answers a forced tool call with a schema-valid call.
PR CI runs on L7.9's committed tiny-llama-2l, in its 1024-position copy
tiny-llama-2l-1k (the same weights: a tool request's schema prompt is about
470 byte tokens, past the original 256); the nightly step serves
SmolLM2-135M-Instruct (`ss fetch smollm2-135m-instruct`) and lets it choose
to call a tool.

Your system.toml must declare (spec/cli-roles.md, DESIGN 2.16):
  [entry].engine        `--config {config}` (L10.5): your tl-serve main reads
                        runtime.toml; its [engine].model_dir is the model
  [services.engine]     config = a runtime.toml template serving
                        course/fixtures/L7.9/tiny-llama-2l-1k as the model
                        [endpoints].model names (default "tracer")
                        ({fixture:L7.9/tiny-llama-2l-1k} fills the path);
                        health = "http://127.0.0.1:{health_port}/healthz"
  [services.engine-spec]  the same template with
                        [engine].speculative = { draft = "prompt_lookup", k = 4 }
  [services.engine-instruct]  (nightly) model_dir = {asset:smollm2-135m-instruct}
  [services.engine-prefill], [services.engine-decode-a], [services.engine-decode-b]
                        the same model with [engine].role = "prefill" and
                        "decode" (L10.6); the harness fills grpc_listen and
                        kv_listen with allocated ports ({grpc_port}, {kv_port})
  [entry].loadgen       {loadgen} --target <base url without /v1> --model <m>
                        --prompt <text> --max-tokens <n> --mode burst
                        --burst <n> --duration <d> --out <report.json>:
                        writes the report (formats/loadgen-report.schema.json)
                        to --out and prints it as the final stdout line
                        (`{loadgen} compare` belongs to the optional load.02)

Two matchers compare your engines with each other (course/harness disagg.py):
`greedy-equal` (the plain and the speculative engine give one greedy text)
and `disagg`, the disaggregated run without the gateway: tokenize on P,
EngineControl.Prefill on P with decode_target = a decode engine's kv port,
then the same request on that engine with X-TL-KV-Handle. Its cases: Info
roles, greedy (plain and streamed, alternating the two decode engines) and
seeded output equal to the unified engine, an unknown and a released handle
answering 404 kv_handle_not_found, and a transfer reset mid-push (a harness
proxy in front of the kv port) failing Prefill without leaking a KV block
on either side, after which that decode engine serves again. The disagg
step is written below but stays commented until the reference serve loop
serves both roles (DEVIATIONS B92-08, X34).

Covered elsewhere, so not repeated here: Rust and Python samplers agree on
fixture logits (L10.1's course tests, parity/sampler); the engine's greedy
stream equals your Python on the tiny model (L10.1's runner tests and
conformance chat.nonstream.greedy).

`ss milestone MS-L10 --smoke` runs the conformance step only.

## MS-gateway: Your gateway passes conformance and its usage ledger reconciles with the engine

Your gateway (gw.01 to gw.07) in front of your engine (L10.5), started by the
runner on allocated ports, as a client sees it: the gateway tier of
openapi-subset.v1 passes (auth, limits, cache, headers, priority.internal;
cases that declare requires = ["gw.08"], such as policy.451, report
pending until the usage policy lands in Pass 10), the engine tier passes on
its own (the provider side of craft.20's contract), and the usage ledger
(gw.07) reconciles with the usage the engine reported for a fixed workload,
through the admin API and through your CLI.

Your system.toml must declare (spec/cli-roles.md, DESIGN 2.16):
  [entry].engine     the --config form (L10.5), serving the byte-level model
                     "tracer" (D32): "Once" is 4 prompt tokens
  [entry].gateway    the --config form (gw.01)
  [entry].ctl        your umbrella CLI, with the `usage` verb below
  [services.engine]  health = "http://127.0.0.1:{health_port}/healthz"
  [services.gateway] health = "http://127.0.0.1:{health_port}/readyz", after = ["engine"];
                     its config template sets [gateway].usage_db = "{data}/usage.db"
                     (a fresh ledger per run), routes model "tracer" to the
                     engine, and accepts the key in [endpoints].api_key_env
                     with the scopes infer and admin
  [endpoints]        api_key_env (export that key)

This milestone fixes the `usage` verb of your `ctl` role (cli-roles.md,
"Verbs of later passes"):
  {ctl} usage --gateway <base URL> [--tenant T] [--key-id K] [--since T] [--until T]
              [--group-by none|model|tenant|key_id|api_version] [--json]
    GET <base>/admin/v1/usage with the key in $TL_API_KEY. With --json: one
    JSON line per row, then a final line of totals over the rows:
    {"requests": N, "errors": E, "prompt_tokens": P, "completion_tokens": C, "cached_tokens": K}.
    Exit 1 on an HTTP error, with the error's message on stderr.

The workload steps run first, so the ledger holds exactly 12 requests when
it is read: 8 completions and 4 streams of max_tokens 4 on "Once", each
4 prompt and 4 completion tokens.

Failover before the first byte (DESIGN 4.4) is checked by gw.05's course
tests (a backend killed before its first byte is retried elsewhere); a
runner step that stops one engine instance mid-milestone is a harness
addition (DEVIATIONS B94-17).

`ss milestone MS-gateway --smoke` runs every step except the CLI step.

## MS-prod: Production on kind: your gateway, disaggregated engines, and observability hold the SLOs

(DESIGN 4.5 "MS-prod", 5.7 catalog, 7.4). MS-P7 = MS-L10 + MS-gateway + MS-prod.

Your serving platform runs on your kind cluster the way it would in
production: your gateway (1 replica) in front of your engines deployed as
1 prefill + 2 decode, the observability stack (collector, Prometheus with
your SLO rules, Tempo, Grafana) scraping and tracing it, your load
generator holding the SLO rate within budget, and drill ops.01 (a decode
pod killed mid-load) resolved within the error budget. The unified
bring-up that comes first is graded by dep.03's check; this milestone
checks the production topology (DEVIATIONS B96-04).

Your system.toml must declare (spec/cli-roles.md, DESIGN 2.16):
  [entry].loadgen      `{loadgen} --target <base url> --rate <rps> --duration <d>`;
                       it sends the API key from $TL_API_KEY ([endpoints].api_key_env)
                       and prints its report (formats/loadgen-report.schema.json)
                       as the final JSON line of stdout
  [entry].engine, [entry].gateway and [services.*]   as for MS-L10 and MS-gateway
                       (the local smoke step starts them)
  [deploy]             kube_context ("kind-<system>"), namespace ("<system>"),
                       gateway_url  (NodePort 30080: "http://127.0.0.1:30080"),
                       prometheus   (NodePort 30090: "http://127.0.0.1:30090"),
                       traces       (Tempo query API, NodePort 30320),
                       services     = { gateway = "deploy/<system>-gateway",
                                        prefill = "deploy/<system>-engine-prefill",
                                        decode  = "deploy/<system>-engine-decode" }
Names the queries below rely on (DESIGN 2.13, contracts/helm/observability.md):
Deployments and Services <system>-gateway, <system>-engine-prefill,
<system>-engine-decode in [deploy].namespace; every Service scraped on :9464
by a ServiceMonitor labelled release: observability (so its series carry
`namespace` and `service` labels); OTEL_SERVICE_NAME <system>-gateway and
<system>-engine; your PrometheusRules rendered from slo.yaml with the
`prod` window profile and the label release: observability.

SLO bounds are the course defaults of DESIGN 2.11 (TTFT p95 0.5 s, TPOT p95
60 ms at 4 rps, 5xx ratio under 0.5%); scaling them by the in-cluster
calibration (ss bench --calibrate --in-cluster) is a harness change that
has not landed (DEVIATIONS B96-05): until it does, a slower machine than the
reference Mac can miss these bounds with a correct system.

`ss milestone MS-prod --smoke` (PR CI, through MS-P7 --smoke) runs the four
local steps: your SLO and rule files, and a short load through your gateway
started as local processes. The full run (nightly kind job, or your own
cluster) adds every `ci = "kind"` step; without a reachable cluster they are
skipped and the verdict is `incomplete`. The drill and the nightly modules
in `requires` are graded on a cluster too, so --smoke does not wait for them.

## MS-P7: Serving platform: inference engine, gateway, and production operations

Compose the engine, gateway, and production milestones. Rust and protocol
primers are direct requirements; optional C exercises are not included.
Earlier pass-gate smoke steps rerun through the spiral invariant.

Run a gate with `practice/bin/ss milestone <ID> --smoke`; omit `--smoke` for its full local and cluster steps.
