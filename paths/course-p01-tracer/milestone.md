# MS-P1: Every Layer Is Yours

**Spec**: [`course/milestones/MS-P1.toml`](../../course/milestones/MS-P1.toml). **Requires** a fresh pass of your own for every core Pass 1 module: `M03.1`, `L0.0`, `lang.04` to `lang.07`, `L10.0`, `gw.00`, `dep.00`, `obs.00`, `ops.00` (the drill, graded by `ol drill end`), and `craft.02`. MS-P1 also reruns the smoke steps of MS-P0. The C primer `lang.03` is optional practice.

## What it runs

| Step | Where | Passes when |
|---|---|---|
| train-bigram | local | `{tinyllm} train bigram` on the fixture corpus reaches NLL at most 3.2004 nats per byte (the count model gives 3.199389) |
| greedy-from-your-checkpoint | local | greedy generation of 32 tokens from "Once" equals the fixture ids exactly |
| engine-conforms-openapi-v0 | local | `ol conform openapi:v0:engine` against your engine |
| gateway-conforms-openapi-v0 | local | `ol conform openapi:v0:gateway` against your gateway in front of your engine |
| stream-through-your-gateway | local | a keyed streaming request through your gateway yields at least 32 content chunks |
| adr-0001-present, engine-crashloop-runbook-present | full | the documents exist in your repo |
| stream-through-the-kind-nodeport | kind | the same stream through the gateway NodePort 30080 |
| one-trace-spans-gateway-and-engine | kind | Jaeger holds one trace whose `gateway.proxy` span is the parent of the engine's `POST /v1/completions` |

## Run it

```bash
practice/bin/ol milestone MS-P1 --smoke   # local processes only: what PR CI runs
practice/bin/ol milestone MS-P1           # adds the documents and the kind steps
```

The runner first runs your `[build].steps`, then starts your engine and gateway on allocated ports and waits for their health URLs. Your `system.toml` declares, per [`spec/cli-roles.md`](../../course/contracts/spec/cli-roles.md):

```toml
[build]
steps = [["cargo", "build", "--release", "--manifest-path", "rust/Cargo.toml", "-p", "tl-serve"],
         ["go", "build", "-C", "go", "-o", "bin/gateway", "./cmd/gateway"],
         ["uv", "run", "--project", "python", "python", "python/tinyllm/__main__.py",
          "train", "bigram", "--data", "{fixture:MS-P1/corpus.txt}", "--out", "artifacts/models/bigram"]]

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
api_key_env = "TL_API_KEY"                # export a key in it before you run

[deploy]
kube_context = "kind-<system>"
namespace    = "<system>"
gateway_url  = "http://127.0.0.1:30080"
traces       = "http://127.0.0.1:30686"
services     = { engine = "deploy/<system>-engine", gateway = "deploy/<system>-gateway" }
```

A kind step without a reachable cluster is skipped with the reason and the verdict is `incomplete`, not a pass. `--ref-deps` runs the milestone while some required module is still red and records the verdict as assisted; your entry points always run as they are.
