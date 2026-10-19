# Language and Tool Primers

Building a whole LLM system from high-school algebra means meeting a lot of languages and tools quickly: Python and numpy, the shell, C, Rust, HTTP, Go, containers and Kubernetes, and later asyncio, async Rust, gRPC, and SQL. This chapter introduces each one from first principles, with just enough depth for the first place the other books rely on it. Each section is placed immediately before that first use, so you can read it when you need it rather than all at once.

Every section follows the same arc. It builds the ideas up from the machine (a process, a socket, a byte on the wire), works a small example by hand, fixes a precise interface, and then walks through a complete, annotated solution, explaining why each line is there. The exercises are not part of the system; they are the smallest artifacts that exercise the ideas the next chapter needs. The full solution code for each lives in [`course/solutions/primers/`](../../../../course/solutions/primers/), so "I read about it" becomes "I can explain every line". Each section ends by naming the place in the other books where its ideas are used next.

> **Learning objectives**
> - Read and write idiomatic code in Python, C, Rust, and Go at the level the system's components need, and explain the tradeoffs each language makes.
> - Drive processes, git, and make from the shell, and reason about exit codes, signals, and the environment.
> - Speak HTTP/1.1, Server-Sent Events, Protocol Buffers, and gRPC at the byte level.
> - Package a service as a container image and run it on Kubernetes with Helm and kind.
> - Write concurrent network code with asyncio and tokio: bounded concurrency, backpressure, cancellation, and graceful shutdown.
> - Query and write a SQLite database correctly at the edges: empty windows, NULLs, ties, replays, and crashes.

## Before you begin

The first section, on Python and numpy, assumes nothing beyond basic algebra and a terminal. Each later section names the earlier ones it builds on in its opening paragraphs; most lean on one or two predecessors, for example the asyncio section on the HTTP section, and the gRPC section on Go, Rust, and async Rust. The type-theoretic background for the languages here is in [Type Systems and Polymorphism](../01-type-systems/README.md), the previous chapter of this book, but you do not need it to start.

## In this chapter

1. [Python and numpy: arrays, dtypes, broadcasting, views vs copies, uv projects](01-python-and-numpy.md). Arrays, dtypes, broadcasting, and the difference between a view and a copy, in a `uv` project; first used by the byte bigram model in [Byte bigram: counts, logits, safetensors v0](../../build-an-llm/01-foundations/01-byte-bigram.md) in *Build Your Own LLM System*.
2. [Shell, git, make, processes: exit codes, signals, environment, pipes](02-shell-git-make.md). How processes start, communicate, and end, and the git and make workflow every later script relies on; first used by [The repository and its CI gate](../../software-craftsmanship/05-code-review-and-ci/01-your-repo-and-ci-gate.md) in *Software Craftsmanship*.
3. [C: memory, pointers, structs, headers, and the C11 toolchain](03-c.md). Memory, pointers, and structs, with sanitizers, headers, and linkage; the foundation for the C runtime and for [Vectors, matrices, row-major layout, and matmul](../../mathematics-for-computing/04-linear-algebra/01-vectors-matrices-and-matmul.md) in *Mathematics for Computing*.
4. [Rust: ownership, traits, Result, cargo workspaces, std TCP](04-rust.md). Ownership, traits, error handling with `Result`, cargo workspaces, and a TCP server from the standard library; first used by [Your first endpoint](../../build-an-llm/11-serving/01-your-first-endpoint.md).
5. [HTTP/1.1, JSON, and Server-Sent Events from the wire up](05-http-and-sse.md). Requests, responses, JSON, and SSE streaming, written and parsed by hand; used by the first endpoint and by the [tracer gateway](../../ai-platform-engineering/12-gateway/01-streaming-proxy.md) in *AI Platform Engineering*.
6. [Go: packages, interfaces, goroutines, channels, context, net/http, testing](06-go.md). Go's packages, interfaces, goroutines, channels, `context`, `net/http`, and `testing`; the language of the gateway, the durable engine, and the load generator.
7. [Containers and Kubernetes: images, layers, Pods, Deployments, Services, Helm, kind](07-containers-and-kubernetes.md). Images as layers plus config, containers as isolated processes, and Deployments and Services on a kind cluster through a Helm chart; applied in [Tracer deploy](../../infrastructure/01-containers-kubernetes/01-tracer-deploy.md) in *Infrastructure*.
8. [Python asyncio: event loop, tasks, cancellation, bounded concurrency](08-python-asyncio.md). A downloader with retries, exponential backoff, per-attempt timeouts, a semaphore, and a task group; the basis of the corpus fetcher in *Data Engineering*.
9. [Async Rust and tokio: futures, tasks, channels, cancellation, hyper](09-async-rust-and-tokio.md). Futures, the tokio runtime, bounded channels for backpressure, cancellation by dropping, and graceful shutdown in a hyper server; grown into the OpenAI-compatible engine server.
10. [Protocol Buffers and gRPC: schema evolution, unary and streaming RPCs, stubs](10-protocol-buffers-and-grpc.md). The protobuf wire format by hand, safe schema evolution, and a client-streaming KV transfer service in Go and Rust.
11. [SQL and SQLite: schema, transactions, WAL mode, indexes, aggregates](11-sql-and-sqlite.md). Five queries over the gateway's usage ledger, an index, WAL and connection pragmas, and an atomic, idempotent write path.
