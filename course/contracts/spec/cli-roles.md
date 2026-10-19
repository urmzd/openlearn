# CLI roles

<!-- chapter: content/courses/build-an-llm/11-serving/01-your-first-endpoint.md
     chapter: content/courses/ai-platform-engineering/12-gateway/01-streaming-proxy.md -->

The system never runs its binaries by guessing a path. Each entry point is declared once in `system.toml` under `[entry]` (schema: [`config/system.schema.json`](../config/system.schema.json)) as an argv list, and everything else (scripts, deploy files, the chapters) refers to it by its **role**. A command written `{tinyllm} generate --prompt hi` means the `[entry].tinyllm` argv followed by `generate --prompt hi`.

```toml
[entry]
tinyllm = ["uv", "run", "--project", "python", "python", "python/tinyllm/__main__.py"]
engine  = ["rust/target/release/tl-serve", "--model-dir", "{model_dir}", "--port", "{port}", "--health-port", "{health_port}"]
gateway = ["go", "run", "-C", "go", "./cmd/gateway", "--port", "{port}", "--health-port", "{health_port}", "--upstream", "http://127.0.0.1:{engine.port}"]
```

Placeholders such as `{port}` are filled by whatever starts the process (ports are allocated per run, never fixed on a machine).

This page fixes the verbs, flags, and outputs that the rest of the system and its end-to-end checks depend on. A CLI may have more verbs and flags; the ones here must behave as written. Later modules add verbs and flags; the ones below stay valid.

## Rules for every role

- **Exit codes.** `0` on success; `2` for a usage error (unknown verb or flag); any other non-zero code for a failure, with the reason on stderr.
- **Machine output.** When a verb's output is marked *final JSON line*, the **last line of stdout** is one JSON object on one line. Earlier lines are free (progress, streamed text). Callers read only that line, so they never need the system's tokenizer.
- **Telemetry.** Servers read the standard OTel variables: `OTEL_EXPORTER_OTLP_ENDPOINT` (base URL; unset means export nothing) and `OTEL_SERVICE_NAME` (in Kubernetes the charts set it to `<system>-engine`, `<system>-gateway`, ...).
- **Shutdown.** Servers exit `0` on SIGTERM.

## Roles

| Role | Language (typical) | First used | Contract |
|---|---|---|---|
| `tinyllm` | Python | Pass 1 (L0.0) | [below](#tinyllm) |
| `engine` | Rust | Pass 1 (L10.0); `--config` from L10.5 | [below](#engine) |
| `gateway` | Go | Pass 1 (gw.00); `--config` from gw.01 | [below](#gateway) |
| `corpus` | Python | Pass 3 (data.*) | `{corpus} <verb> --spec ... --progress ...` under `spec/subprocess-activity.md` |
| `tl-tok` | Rust | Pass 3 (L1.5) | `encode`, `bench`; defined by L1.5 |
| `durable` | Go | Pass 8 (dur.02) | `--data {data} --port {grpc_port}`; `--test-clock` enables `/debug/clock` |
| `worker` | Go | Pass 8 (dur.04) | `--queue {queue} --durable <addr>`; `--test-activities` registers the course test activity `tl.test.Append` (appends its idempotency key to the `effects` sink) and the course test workflows used by replay and kill-loop tests |
| `ctl` | Go | Pass 8 (dur.11) | the umbrella CLI: `train`, `eval`, `release`, `data build`, defined by their modules |
| `loadgen` | Go | Pass 7 (load.01) | defined by load.01 and load.02 |
| `agent` | Go | Pass 10 (ag.*) | defined by the agent modules |

## `tinyllm`

The Python CLI: `python/tinyllm/__main__.py`, run as a script from the repo root inside the uv project (`uv run --project python python python/tinyllm/__main__.py`). The project is not an installed package (`[tool.uv] package = false`), so the script puts `python/` on `sys.path` itself before importing `tinyllm`. In Pass 1 it has three verbs.

### `train bigram`

```
{tinyllm} train bigram --data <file> --out <dir> [--alpha <float>]
```

Fits the byte bigram of L0.0 by counting. `--data` is a file whose raw bytes are the token ids (the byte tokenizer, [formats/tokenizer.md](../formats/tokenizer.md)). `--alpha` is the add-one smoothing constant, default `1.0`. Writes a model directory:

- `<dir>/config.json`: `{"tl_arch": "bigram", "tl_tokenizer": "bytes", "vocab_size": 256, "tl_format": 1}` ([formats/config.schema.json](../formats/config.schema.json))
- `<dir>/model.safetensors`: `bigram.weight`, F32 `[256, 256]`, metadata `{"format": "tinyllm"}` ([formats/safetensors.md](../formats/safetensors.md))

Final JSON line: `{"out": "<dir>", "tokens": <int>, "nll": <float>}`, where `nll` is the mean negative log-likelihood in nats per byte of the fitted model on `--data`.

### `generate`

```
{tinyllm} generate --model <dir> --prompt <text> [--max-tokens <n>] [--greedy | --temperature <t>] [--seed <s>]
```

Continues `--prompt` (non-empty) with exactly `--max-tokens` new tokens (default 16; the byte tokenizer has no end-of-sequence token). `--greedy` is `--temperature 0`: the highest logit, ties to the lowest id. Otherwise tokens are sampled at `--temperature` (default 1.0) from a generator seeded with `--seed` (default 0), so the same seed gives the same ids.

Final JSON line: `{"ids": [<int>, ...], "text": "<string>"}`. `ids` holds the **generated ids only**, not the prompt's; `text` is `decode(ids)` with replacement. A token-for-token comparison reads `ids` and compares it with an expected list.

### `logits`

```
{tinyllm} logits --model <dir> --prompt <text> [--prefix-ids <id,id,...>]
```

Teacher forcing: runs the model on the prompt's ids followed by `--prefix-ids` (comma-separated, may be empty) and prints the next-token logits, before temperature.

Final JSON line: `{"logits": [<float>, ...]}`, `vocab_size` values. A token comparison uses this verb to tell a real divergence from a near-tie: it takes `generate`'s argv, replaces the verb with `logits`, and appends `--prefix-ids` with the expected ids up to the first divergence.

## `engine`

The model server: OpenAI subset over HTTP and SSE ([openapi/openai-subset.v0.yaml](../openapi/openai-subset.v0.yaml)).

### Tracer form (L10.0 to L10.4)

```
{engine} --model-dir <dir> --port <n> --health-port <n>
```

The tracer engine is std-only Rust and has no TOML parser, so it takes flags.

- `--model-dir`: a model directory as written by `train bigram`. Contract v0 serves `tl_arch = bigram` with `tl_tokenizer = bytes`; every later engine keeps serving it, so the tracer smoke stays green for the whole course.
- `--port`: serves `POST /v1/completions` (API v0) on `0.0.0.0:<port>`.
- `--health-port`: serves `GET /healthz` on `0.0.0.0:<port>` (`9464` in Kubernetes). The engine must also answer `GET /healthz` on `--port`: the tracer gateway's `/readyz` probes `<upstream>/healthz` on the API port.
- Telemetry: reads `traceparent` and, when `OTEL_EXPORTER_OTLP_ENDPOINT` is set, posts one SERVER span `POST /v1/completions` per request as OTLP/HTTP JSON to `<endpoint>/v1/traces`, as the child of the incoming context.

### Config form (from L10.5)

```
{engine} --config <runtime.toml>
```

Reads `[engine]` from `runtime.toml` (schema `config/runtime.schema.json`), with `TL_ENGINE__<KEY>` overrides. Whatever starts the engine fills `http_listen`, `grpc_listen`, `kv_listen`, and `health_listen` with allocated ports.

## `gateway`

The front door: API keys, then SSE passed through without buffering.

### Tracer form (gw.00)

```
{gateway} --port <n> --health-port <n> --upstream <engine base URL>
```

- `--port`: serves the gateway tier of API v0 on `0.0.0.0:<port>`.
- `--health-port`: serves `GET /healthz` (200 while up) and `GET /readyz` (200 once `<upstream>/healthz` answers, else 503).
- `--upstream`: the engine's base URL without `/v1`, for example `http://127.0.0.1:{engine.port}` (or the engine Service in Kubernetes).
- **Key.** Accepts `Authorization: Bearer <key>` where `<key>` is the value of the environment variable that `[endpoints].api_key_env` names (default `TL_API_KEY`). A client that starts the services passes its environment to each and uses the same variable as its client key. A missing or different key is `401` with `code: invalid_api_key`, before any upstream call.
- Propagates `traceparent` (as the child of its own `gateway.proxy` CLIENT span) and `X-Request-Id` to the engine.

### Config form (from gw.01)

```
{gateway} --config <runtime.toml>
```

Reads `[gateway]` from `runtime.toml` with `TL_GATEWAY__<KEY>` overrides; keys and routes come from the files it names.

## Verbs of later passes

From Pass 2 on the system's end-to-end runs call the verbs below. No single module owns a verb; the stage that first needs it fixes its exact flags and the keys of its final JSON line (see the solution's entry points in [`course/solutions/entry/`](../../solutions/entry/)). This index fixes the names, so two stages never use one verb for different jobs. Every verb follows the rules above (exit codes, final JSON line).

| Role, verb | First needed by | Contract |
|---|---|---|
| `{tinyllm} gradcheck --suite <id>` | L0 | exit 0 when every check passes |
| `{tinyllm} train <arch>` (`mlp`, `gpt`, `bert`, `electra`, `rnnlm`, `seq2seq`, `transformer`, `llama`, ...) | L0 to L11 | flags fixed by each stage (`train rnnlm` in L3, `train seq2seq` in L4, `train transformer` in L5) |
| `{tinyllm} train --spec <file> --progress <file>` | durable orchestration (dur.*, `TrainRun`), C1 | spec/subprocess-activity.md |
| `{tinyllm} generate` (`--cache`, `--spec`, `--out`, `--prompt-file`) | Pass 1, extended in L3 (`--out` for rnnlm directories), L7 (`--prompt-file`: the final `ids` are every prompt's generated ids concatenated, plus `per_prompt`), and L8 | final line `{ids, text, ...}`, generated ids only |
| `{tinyllm} logits` (`--prompts`, `--out`, `--model`) | Pass 1 (near-tie rule), extended in L7 (`--model` takes a model directory or its `config.json`) | final line `{logits}`, or the output path |
| `{tinyllm} eval` (`ppl`, `--suite <id>`, `--spec <file>`, `--tail-frac` for rnnlm directories) | L2, L3, L6, C1 | `formats/eval-results.schema.json` (zoo rows) and `formats/eval-result.schema.json` (C1) outputs |
| `{tinyllm} export --spec <file>` | C1 (`ModelRelease`) | `formats/export-spec.schema.json` |
| `{tinyllm} info` (`--model`, a model directory or its `config.json`) | Pass 1, L7 | final line `{params, ...}` |
| `{tinyllm} translate` (`--model`, `--in`, `--beam`, `--max-len`) | L4, L5 (dispatches on `tl_arch`: `transformer` or `seq2seq`) | fixed by that stage |
| `{tinyllm} pull <hf repo>` | L7 | final line `{dir}` |
| `{tinyllm} tok <encode, train, bench>` | L1 | fixed by that stage |
| `{tinyllm} lm train <ngram, nplm>` | L2 | fixed by that stage |
| `{tinyllm} bench decode` | L8 (`perf` steps) | fixed by that stage |
| `{tinyllm} finetune classify` | L6 | fixed by that stage |
| `{tinyllm} post <sft, dpo, grpo>` | C2 | fixed by that stage |
| `{corpus} run` (`--config`, `--until <stage>`, `--stage <s>`, `--workers`) | the corpus pipeline (data.*, `CorpusBuild`) | `formats/corpus-shard.md`, spec/subprocess-activity.md |
| `{corpus} ledger verify`, `{corpus} datasheet` | the corpus pipeline (data.*) | exit 65 on an unknown license |
| `{tl-tok} encode`, `{tl-tok} bench` | L1 | fixed by that stage |
| `{durable} --data <dir> --port <n>` (`--test-clock`, `--replicas 3`) | durable orchestration (dur.*), durable HA (dur.10) | proto/tl/durable/v1 |
| `{worker} --queue <q> --durable <addr>` (`--test-activities`) | durable orchestration (dur.*) | above |
| `{ctl} <train, eval, release, data build, wf, keys, usage, agent, rag>` | durable orchestration (dur.*) to the agent (ag.*) | each verb calls `tl.durable.v1` or `admin.v1`. `usage` (from the gateway, gw.*) takes `--gateway`, `--tenant`, `--key-id`, `--since`, `--until`, `--group-by`, `--json`, with a final totals line `{requests, errors, prompt_tokens, completion_tokens, cached_tokens}` |
| `{loadgen} --target <url> --rate <rps> --duration <d>`; `{loadgen} compare <base> <head> --metric <m> --max-regress <pct>` | L10, the gateway (gw.*), production (dep.*, obs.*), drills | `formats/loadgen-report.schema.json`; `compare` exits 1 on a regression |
| `{agent}` | the agent (ag.*) | fixed by that stage |
