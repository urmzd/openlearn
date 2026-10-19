# What the System Does After Pass 1

After the tracer, every layer of the final system exists in its thinnest form and runs end to end. The Python CLI trains a byte bigram by counting and writes a checkpoint, the Rust engine streams completions from it, the Go gateway authenticates and proxies the stream, both run on kind behind Helm charts, and one request appears as one trace across both services. Later passes deepen each layer without changing these seams.

The system is built from the sections of [Pass 1](README.md): [matmul in Python](../../../mathematics-for-computing/03-linear-algebra/01-vectors-matrices-and-matmul.md), the [byte bigram](../../p00-foundations/00-byte-bigram.md), the primers on [Rust](../../../programming-languages/02-language-and-tool-primers/04-rust.md), [HTTP and SSE](../../../programming-languages/02-language-and-tool-primers/05-http-and-sse.md), [Go](../../../programming-languages/02-language-and-tool-primers/06-go.md), and [containers and Kubernetes](../../../programming-languages/02-language-and-tool-primers/07-containers-and-kubernetes.md), then [your first endpoint](../../p10-serving/00-your-first-endpoint.md), [the tracer gateway](../../../ai-platform-engineering/12-gateway/00-streaming-proxy.md), [the tracer deploy](../../../infrastructure/01-containers-kubernetes/00-tracer-deploy.md), [one trace](../../../systems/04-observability/00-one-trace.md), [the first drill](../../../systems/05-incident-response-and-chaos/00-first-drill.md), and [architecture decision records](../../../diagramming/02-documentation-writing/01-architecture-decision-records.md). The [C primer](../../../programming-languages/02-language-and-tool-primers/03-c.md) is optional.

## What runs, and why each piece is needed

Start locally. `{tinyllm} train bigram` on a small story corpus reaches an NLL of at most 3.2004 nats per byte; the count model gives 3.199389. That number matters less than the checkpoint it writes, because every later model reuses the format. Greedy generation of 32 tokens from "Once" gives exactly the same ids on every run, and that determinism is what makes the engine and the CLI comparable at all.

Next, the services are checked against the wire contract, `openapi:v0`. The engine's `POST /v1/completions` and its SSE stream follow the contract, and the gateway serves the same contract in front of the engine. The gateway and every client depend on the wire format, not on any implementation, and the gateway must be transparent to clients. A keyed streaming request through the gateway receives at least 32 content chunks as they are produced, which proves the proxy does not buffer the stream.

Two documents exist alongside the code: [ADR-0001](../../../../../course/solutions/docs/adr/0001-languages-meet-at-process-and-file-boundaries.md), the decision that languages meet at process and file boundaries, and the [engine-crashloop runbook](../../../../../course/solutions/docs/runbooks/engine-crashloop.md), the first operational procedure.

Finally, on kind, the same stream arrives through NodePort 30080, so the deployed system behaves like the local one and the deploy path is exercised from day one. Jaeger holds one trace whose `gateway.proxy` span is the parent of the engine's `POST /v1/completions` span. That is trace-context propagation, and every later debugging session relies on it.

## How the solution is wired

The roles and their verbs are fixed by [`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md). The solution's system manifest declares how to build the system and how to start each role:

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

The order matters. The engine starts first and the gateway waits for the engine's health URL (`after = ["engine"]`), because a gateway whose upstream is not ready would fail its own readiness check. Separate health ports keep liveness probes off the serving port. The CLI entry point is [`__main__.py`](../../../../../course/solutions/entry/python/tinyllm/__main__.py), the gateway's is under [`go/cmd/gateway`](../../../../../course/solutions/entry/go/cmd/gateway/), and the deploy files are under [`deploy/`](../../../../../course/solutions/entry/deploy/).

[Pass 2](../course-p02-foundations/README.md) replaces how the bigram is trained and leaves everything else here untouched.
