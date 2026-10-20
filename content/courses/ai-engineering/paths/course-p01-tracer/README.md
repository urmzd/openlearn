# Pass 1: The Tracer

The tracer is a thin system through every layer of the final one, built before any layer is deep. Its model is a byte-level bigram fitted by counting, which is about as simple as a language model can be. But that bigram is written to a real checkpoint, served by a real Rust engine that streams completions over HTTP, fronted by a Go gateway that checks an API key and streams without buffering, packaged as images and Helm charts on a local kind cluster, and observed as one trace across the gateway and the engine in Jaeger. The pass also runs a first incident drill with its runbook and writes a first architecture decision record.

Building every layer thinly first is what makes the rest of the course safe. Each later pass deepens one layer behind a contract this pass fixes, and because the whole system already runs end to end, a change that breaks a seam shows up immediately. The pass introduces Python, Rust, HTTP and SSE, Go, and containers as they are needed. C remains optional depth later in the course.

> **Learning objectives**
> - Explain how a count-based bigram reaches the maximum-likelihood NLL and how it is written to safetensors.
> - Trace an HTTP request and an SSE stream through a hand-written Rust server and a Go streaming proxy.
> - Explain how the system is deployed on kind behind Helm charts and NodePorts.
> - Explain how trace context propagates so one request shows as one trace.
> - Diagnose the engine crashloop drill and explain the decision recorded in ADR-0001.

## Before you begin

This pass needs [Pass 0](../course-p00-setup/README.md): a working toolchain and a repository whose CI gate is green. Stages 1 to 7 run as local processes; stages 8 to 11 run on a local kind cluster.

## In this pass

The sections build the system from the model outward, then deploy and observe it:

1. [Matmul in Python](../../../mathematics/09-linear-algebra/01-vectors-matrices-and-matmul.md). Afterwards you can explain how the triple-loop Python matmul indexes rows and columns and why it matches numpy on the golden cases.
2. [Byte bigram](../../06-numerics-and-autograd/01-byte-bigram.md). Afterwards you can explain why the count-based bigram reaches the maximum-likelihood NLL and how it is written to a safetensors file.
3. [Rust](../../../programming-and-algorithms/01-language-primers/04-rust.md). Afterwards you can explain how the Rust echo server serves two clients at once and who owns each connection.
4. [HTTP and SSE](../../../programming-and-algorithms/01-language-primers/05-http-and-sse.md). Afterwards you can trace how an HTTP request is parsed and how an SSE stream is framed by hand.
5. [Your first endpoint](../../23-model-serving/01-your-first-endpoint.md). Afterwards you can trace how the Rust server loads the checkpoint, streams SSE completions, and exports its span.
6. [Go](../../../programming-and-algorithms/01-language-primers/06-go.md). Afterwards you can explain how the Go streaming proxy propagates a deadline through its context.
7. [Your gateway](../../25-ai-gateway/01-streaming-proxy.md). Afterwards you can explain why the Go gateway rejects bad keys before proxying and how it streams without buffering.
8. [Containers and Kubernetes](../../../programming-and-algorithms/01-language-primers/07-containers-and-kubernetes.md). Afterwards you can explain how the image runs on kind behind a Helm chart and a NodePort.
9. [Images, kind, and Helm](../../../systems/10-containers-kubernetes/01-tracer-deploy.md). Afterwards you can trace how helm install brings up the gateway and the engine on kind and how a curl reaches the engine through the gateway.
10. [One trace](../../../systems/11-observability/01-one-trace.md). Afterwards you can explain how trace context propagates so one request shows as one trace across gateway and engine in Jaeger.
11. [First drill](../../../systems/12-incident-response-and-chaos/01-first-drill.md). Afterwards you can diagnose the engine crashloop from its symptoms, explain the fix, and explain what its runbook says.
12. [First ADR](../../../software-engineering/12-documentation/01-architecture-decision-records.md). Afterwards you can explain why ADR-0001 has the languages meet at process and file boundaries.
13. [What the System Does After Pass 1](milestone.md). Afterwards you can explain how every layer runs end to end, starting with the CLI training and sampling the bigram from a clean shell.

The engine serves exactly the checkpoint the `tinyllm` CLI wrote, so by the end of the pass [every layer exists and runs end to end](milestone.md). The stages are also in this pass's manifest, [`path.tsv`](path.tsv). [Pass 2](../course-p02-foundations/README.md) keeps the engine and replaces how its model is trained.
