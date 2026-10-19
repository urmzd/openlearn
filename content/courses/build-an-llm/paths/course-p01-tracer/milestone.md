# What the system does after Pass 1

Every layer of the final system exists in its thinnest form and runs end to end: the Python CLI trains a byte bigram by counting and writes a checkpoint, the Rust engine streams completions from it, the Go gateway authenticates and proxies the stream, both run on kind behind Helm charts, and one request appears as one trace across both services. Later passes deepen each layer without changing these seams.

It is built from `M03.1`, `L0.0`, `lang.04` to `lang.07`, `L10.0`, `gw.00`, `dep.00`, `obs.00`, `ops.00` (the drill), and `craft.02`. The C primer `lang.03` is optional.

## What runs, and why each piece is needed

| Behaviour | Where | What it shows | Why it is needed |
|---|---|---|---|
| `{tinyllm} train bigram` on a small story corpus | local | NLL at most 3.2004 nats per byte (the count model gives 3.199389) | the checkpoint format every later model reuses |
| greedy generation of 32 tokens from "Once" | local | the exact same ids on every run | deterministic decoding is what makes engine and CLI comparable |
| the engine against `openapi:v0` | local | `POST /v1/completions` and its SSE stream follow the contract | the gateway and every client depend on the wire format, not the implementation |
| the gateway against `openapi:v0` | local | the same contract, in front of the engine | the gateway must be transparent to clients |
| a keyed streaming request through the gateway | local | at least 32 content chunks arrive as they are produced | proves the proxy does not buffer the stream |
| ADR-0001 and the engine-crashloop runbook | docs | [ADR-0001](../../../../../course/solutions/docs/adr/0001-languages-meet-at-process-and-file-boundaries.md), [runbook](../../../../../course/solutions/docs/runbooks/engine-crashloop.md) | the decision that languages meet at process and file boundaries, and the first operational procedure |
| the same stream through the kind NodePort 30080 | kind | the deployed system behaves like the local one | the deploy path is exercised from day one |
| one trace spanning gateway and engine | kind | Jaeger holds one trace whose `gateway.proxy` span is the parent of the engine's `POST /v1/completions` | trace context propagation, which every later debugging session relies on |

## How the solution is wired

The roles and their verbs are fixed by [`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md). The build and the entry points of the solution are:

```toml
[build]
steps = [["cargo", "build", "--release", "--manifest-path", "rust/Cargo.toml", "-p", "tl-serve"],
         ["go", "build", "-C", "go", "-o", "bin/gateway", "./cmd/gateway"],
         ["uv", "run", "--project", "python", "python", "python/tinyllm/__main__.py",
          "train", "bigram", "--data", "corpus.txt", "--out", "artifacts/models/bigram"]]

[entry]
tinyllm = ["uv", "run", "--project", "python", "python", "python/tinyllm/__main__.py"]
engine  = ["rust/target/release/tl-serve", "--model-dir", "artifacts/models/bigram", "--port", "{port}", "--health-port", "{health_port}"]
gateway = ["go/bin/gateway", "--port", "{port}", "--health-port", "{health_port}", "--upstream", "http://127.0.0.1:{engine.port}"]

[services.engine]
health = "http://127.0.0.1:{health_port}/healthz"

[services.gateway]
health = "http://127.0.0.1:{health_port}/readyz"
after  = ["engine"]

[endpoints]
api_base    = "http://127.0.0.1:{gateway.port}/v1"
api_key_env = "TL_API_KEY"                # the gateway reads the accepted key from it

[deploy]
kube_context = "kind-<system>"
namespace    = "<system>"
gateway_url  = "http://127.0.0.1:30080"
traces       = "http://127.0.0.1:30686"
services     = { engine = "deploy/<system>-engine", gateway = "deploy/<system>-gateway" }
```

The order matters: the engine starts first and the gateway waits for the engine's health URL (`after = ["engine"]`), because a gateway whose upstream is not ready would fail its own readiness check. Separate health ports keep liveness probes off the serving port. The CLI entry point is [`__main__.py`](../../../../../course/solutions/entry/python/tinyllm/__main__.py); the gateway's is under [`go/cmd/gateway`](../../../../../course/solutions/entry/go/cmd/gateway/); the deploy files are under [`deploy/`](../../../../../course/solutions/entry/deploy/).
