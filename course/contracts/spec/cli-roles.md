# CLI roles

<!-- chapter: content/courses/build-an-llm/p10-serving/00-your-first-endpoint.md
     chapter: content/tracks/ai-platform-engineering/12-gateway/00-streaming-proxy.md -->

The course never builds or runs your binaries by guessing. You declare each entry point once in `system.toml` under `[entry]` (schema: [`config/system.schema.json`](../config/system.schema.json)) as an argv list, and milestones call it by its **role**. A milestone step written `{tinyllm} generate --prompt hi` runs your `[entry].tinyllm` argv followed by `generate --prompt hi`.

```toml
[entry]
tinyllm = ["uv", "run", "--project", "python", "python", "python/tinyllm/__main__.py"]
engine  = ["rust/target/release/tl-serve", "--model-dir", "{model_dir}", "--port", "{port}", "--health-port", "{health_port}"]
gateway = ["go", "run", "-C", "go", "./cmd/gateway", "--port", "{port}", "--health-port", "{health_port}", "--upstream", "http://127.0.0.1:{engine.port}"]
```

A missing role fails the milestone with `milestone <MS-ID> needs [entry].<role> in system.toml`. Placeholders such as `{port}` are filled by the milestone runner (ports are allocated per run, never fixed on your machine).

This page fixes the verbs, flags, and outputs that milestones depend on. Your CLI may have more verbs and flags; the ones here must behave as written. Later modules add verbs and flags; the ones below stay valid.

## Rules for every role

- **Exit codes.** `0` on success; `2` for a usage error (unknown verb or flag); any other non-zero code for a failure, with the reason on stderr.
- **Machine output.** When a verb's output is marked *final JSON line*, the **last line of stdout** is one JSON object on one line. Earlier lines are free (progress, streamed text). Matchers read only that line, so they never need your tokenizer.
- **Telemetry.** Servers read the standard OTel variables: `OTEL_EXPORTER_OTLP_ENDPOINT` (base URL; unset means export nothing) and `OTEL_SERVICE_NAME` (in Kubernetes your charts set it to `<system>-engine`, `<system>-gateway`, ...).
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

The Python CLI: `python/tinyllm/__main__.py`, run as a script from the repo root inside your uv project (`uv run --project python python python/tinyllm/__main__.py`). The project is not an installed package (`[tool.uv] package = false`), so the script puts `python/` on `sys.path` itself before importing `tinyllm`. In Pass 1 it has three verbs.

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

Final JSON line: `{"ids": [<int>, ...], "text": "<string>"}`. `ids` holds the **generated ids only**, not the prompt's; `text` is `decode(ids)` with replacement. The `tokens-equal` matcher compares `ids` with an expected list.

### `logits`

```
{tinyllm} logits --model <dir> --prompt <text> [--prefix-ids <id,id,...>]
```

Teacher forcing: runs the model on the prompt's ids followed by `--prefix-ids` (comma-separated, may be empty) and prints the next-token logits, before temperature.

Final JSON line: `{"logits": [<float>, ...]}`, `vocab_size` values. The `tokens-equal` matcher's near-tie rule calls this verb: it takes `generate`'s argv, replaces the verb with `logits`, and appends `--prefix-ids` with the expected ids up to the first divergence.

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

Reads `[engine]` from `runtime.toml` (schema `config/runtime.schema.json`), with `TL_ENGINE__<KEY>` overrides. The runner fills `http_listen`, `grpc_listen`, `kv_listen`, and `health_listen` with allocated ports.

## `gateway`

The front door: API keys, then SSE passed through without buffering.

### Tracer form (gw.00)

```
{gateway} --port <n> --health-port <n> --upstream <engine base URL>
```

- `--port`: serves the gateway tier of API v0 on `0.0.0.0:<port>`.
- `--health-port`: serves `GET /healthz` (200 while up) and `GET /readyz` (200 once `<upstream>/healthz` answers, else 503).
- `--upstream`: the engine's base URL without `/v1`, for example `http://127.0.0.1:{engine.port}` (or the engine Service in Kubernetes).
- **Key.** Accepts `Authorization: Bearer <key>` where `<key>` is the value of the environment variable that `[endpoints].api_key_env` names (default `TL_API_KEY`). The harness passes its environment to every service it starts and uses the same variable as its client key. A missing or different key is `401` with `code: invalid_api_key`, before any upstream call.
- Propagates `traceparent` (as the child of its own `gateway.proxy` CLIENT span) and `X-Request-Id` to the engine.

### Config form (from gw.01)

```
{gateway} --config <runtime.toml>
```

Reads `[gateway]` from `runtime.toml` with `TL_GATEWAY__<KEY>` overrides; keys and routes come from the files it names.

## Verbs of later passes

Milestones from Pass 2 on call the verbs below. The CLI is yours (D16), so no module owns a verb; the milestone that first calls it (`course/milestones/<MS-ID>.toml`) fixes its exact flags and the keys of its final JSON line when that milestone is authored. This index fixes the names, so two milestones never use one verb for different jobs. Every verb follows the rules above (exit codes, final JSON line).

| Role, verb | First called by | Contract |
|---|---|---|
| `{tinyllm} gradcheck --suite <id>` | MS-L0 | exit 0 when every check passes |
| `{tinyllm} train <arch>` (`mlp`, `gpt`, `bert`, `electra`, `rnnlm`, `seq2seq`, `transformer`, `llama`, ...) | MS-L0 to MS-L11 | flags per milestone (`train rnnlm` in MS-L3, `train seq2seq` in MS-L4, `train transformer` in MS-L5) |
| `{tinyllm} train --spec <file> --progress <file>` | MS-durable (`TrainRun`), MS-C1 | spec/subprocess-activity.md |
| `{tinyllm} generate` (`--cache`, `--spec`, `--out`, `--prompt-file`) | MS-P1, extended in MS-L3 (`--out` for rnnlm directories), MS-L7 (`--prompt-file`: the final `ids` are every prompt's generated ids concatenated, plus `per_prompt`), and MS-L8 | final line `{ids, text, ...}`, generated ids only |
| `{tinyllm} logits` (`--prompts`, `--out`, `--model`) | MS-P1 (near-tie rule), extended in MS-L7 (`--model` takes a model directory or its `config.json`) | final line `{logits}`, or the output path |
| `{tinyllm} eval` (`ppl`, `--suite <id>`, `--spec <file>`, `--tail-frac` for rnnlm directories) | MS-L2, MS-L3, MS-L6, MS-C1 | `formats/eval-results.schema.json` (zoo rows) and `formats/eval-result.schema.json` (MS-C1) outputs |
| `{tinyllm} export --spec <file>` | MS-C1 (`ModelRelease`) | `formats/export-spec.schema.json` |
| `{tinyllm} info` (`--model`, a model directory or its `config.json`) | MS-P1, MS-L7 | final line `{params, ...}` |
| `{tinyllm} translate` (`--model`, `--in`, `--beam`, `--max-len`) | MS-L4, MS-L5 (dispatches on `tl_arch`: `transformer` or `seq2seq`) | per milestone |
| `{tinyllm} pull <hf repo>` | MS-L7 | final line `{dir}` |
| `{tinyllm} tok <encode, train, bench>` | MS-L1 | per milestone |
| `{tinyllm} lm train <ngram, nplm>` | MS-L2 | per milestone |
| `{tinyllm} bench decode` | MS-L8 (`perf` steps) | per milestone |
| `{tinyllm} finetune classify` | MS-L6 | per milestone |
| `{tinyllm} post <sft, dpo, grpo>` | MS-C2 | per milestone |
| `{corpus} run` (`--config`, `--until <stage>`, `--stage <s>`, `--workers`) | MS-corpus, `CorpusBuild` | `formats/corpus-shard.md`, spec/subprocess-activity.md |
| `{corpus} ledger verify`, `{corpus} datasheet` | MS-corpus | exit 65 on an unknown license |
| `{tl-tok} encode`, `{tl-tok} bench` | MS-L1 | per milestone |
| `{durable} --data <dir> --port <n>` (`--test-clock`, `--replicas 3`) | MS-durable, MS-durable-ha | proto/tl/durable/v1 |
| `{worker} --queue <q> --durable <addr>` (`--test-activities`) | MS-durable | above |
| `{ctl} <train, eval, release, data build, wf, keys, usage, agent, rag>` | MS-durable to MS-agent | each verb calls `tl.durable.v1` or `admin.v1`. `usage` (MS-gateway) takes `--gateway`, `--tenant`, `--key-id`, `--since`, `--until`, `--group-by`, `--json`, with a final totals line `{requests, errors, prompt_tokens, completion_tokens, cached_tokens}` |
| `{loadgen} --target <url> --rate <rps> --duration <d>`; `{loadgen} compare <base> <head> --metric <m> --max-regress <pct>` | MS-L10, MS-gateway, MS-prod, drills | `formats/loadgen-report.schema.json`; `compare` exits 1 on a regression |
| `{agent}` | MS-agent | per milestone |
