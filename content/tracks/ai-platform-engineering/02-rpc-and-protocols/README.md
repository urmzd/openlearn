# RPC & Protocols

## Overview

- **Primary references**: [gRPC docs](https://grpc.io/docs/) (free), [Protocol Buffers docs](https://protobuf.dev/) (free)
- **Supplementary**: [Apache Thrift](https://thrift.apache.org/docs/), [Apache Avro spec](https://avro.apache.org/docs/), [Cap'n Proto](https://capnproto.org/), [FlatBuffers](https://flatbuffers.dev/), [MessagePack](https://msgpack.org/), [HTTP/2 (RFC 9113)](https://www.rfc-editor.org/rfc/rfc9113.html), DDIA Ch 4 (encoding & evolution)
- **Prerequisites**: [System Design](../../systems/01-system-design/) (services, APIs, load balancing), basic networking (TCP/HTTP)
- **Estimated time**: 2-3 weeks at 8-10 hrs/week

## Key Takeaways

- **RPC makes a network call look like a function call** — and the leak in that abstraction (partial failure, latency, serialization) is where all the engineering lives.
- **The wire format is an architecture decision.** Schema-based binary formats (Protobuf, Thrift, Avro) give typed contracts, compact bytes, and code generation; schemaless text (JSON) gives human-readability and zero tooling. You're trading bytes and safety for legibility.
- **gRPC = Protocol Buffers + HTTP/2 + generated stubs.** Multiplexed streams, binary framing, and four call types (unary, server-stream, client-stream, bidi) make it the default for internal service-to-service traffic.
- **Schema evolution is the real problem.** Field tags/aliases let producers and consumers deploy independently. Protobuf evolves by *tag number*; Avro by *reader/writer schema resolution*. Get this wrong and a deploy breaks every caller.
- **Match the protocol to the boundary**: gRPC inside the mesh, REST/JSON at the public edge, Avro for the data lake/Kafka, FlatBuffers/Cap'n Proto when you can't afford a parse step.

## How to Study

- Define one `.proto` service, generate stubs in two languages, and call across them. Then add a field and prove old and new clients still interoperate.
- Encode the same record as JSON, Protobuf, and Avro; compare byte size and parse time.
- Implement all four gRPC call types — especially server-streaming, which is how you'd stream model tokens to another service.
- Break compatibility on purpose (reuse a tag number, change a type) and watch what happens to old consumers.

---

# Concepts & Techniques

## Core Insight

Two processes can only exchange bytes. A *protocol* is the agreement that turns a meaningful value on one machine into bytes and back into a meaningful value on another — across languages, versions, and time. *RPC* layers a familiar shape on top: call a method, get a return value, as if it were local. The whole discipline is managing the ways that illusion breaks (the network is slow, lossy, and can fail halfway) and the ways the agreement must change without coordinated, simultaneous deploys. Encoding efficiency, schema evolution, and streaming are the three axes you optimize.

## 1. What RPC Is (and Its Fallacies)

**Key ideas**:
- **The model**: client invokes a method on a *stub*; the stub *marshals* (serializes) arguments, sends them, the server *unmarshals*, runs the method, and returns a result the same way. The network is hidden behind a function signature.
- **Why it leaks**: a local call can't time out, lose the response, or partially execute — a remote one can. You must design for **partial failure**: timeouts, retries (with idempotency keys), backoff, deadlines/cancellation, and circuit breakers.
- **The fallacies of distributed computing**: the network is *not* reliable, zero-latency, infinite-bandwidth, secure, or free. RPC frameworks paper over the syntax, not these realities.
- **Idempotency**: because retries are unavoidable, write operations should be safe to apply more than once (connects to durable execution in [topic 05](../05-durable-orchestration-and-workers/)).

## 2. Serialization Formats

**The bytes on the wire**

| Format | Schema | Encoding | Strength | Weakness |
|--------|--------|----------|----------|----------|
| **JSON** | None | Text | Human-readable, universal | Verbose, slow parse, no types |
| **Protocol Buffers** | Required (`.proto`) | Binary, tag-based | Compact, fast, codegen, evolution | Not human-readable |
| **Apache Thrift** | Required (`.thrift`) | Binary | Protobuf-like + full RPC stack | Two ecosystems (Facebook/Apache) |
| **Apache Avro** | Required (JSON schema) | Binary, schema-resolved | Schema travels with data; great for Kafka/lakes | Needs writer schema to read |
| **MessagePack** | None | Binary | "Binary JSON", compact, schemaless | No contract/evolution |
| **Cap'n Proto / FlatBuffers** | Required | Binary, **zero-copy** | Read fields without parsing | Larger on wire, less ergonomic |

**Key distinctions**:
- **Schema-based vs schemaless**: a schema buys you a typed contract, smaller bytes (field *names* aren't on the wire — tag numbers are), and generated code. Schemaless (JSON, MessagePack) buys you flexibility and no build step.
- **Zero-copy** (FlatBuffers, Cap'n Proto): the serialized bytes *are* the in-memory layout, so you read a field by offset without a parse/allocate step — matters for huge messages or hot paths (game state, ML tensors, mmap'd files).
- **Avro's trick**: the *writer's* schema is stored with the data; a *reader's* schema resolves against it. This makes Avro the standard for evolving records in Kafka and data lakes (pairs with [Data Engineering](../../data-engineering/03-batch-streaming/)).

## 3. Schema Evolution

**Deploying producer and consumer independently — the hardest part**

**Key ideas**:
- **Backward compatible**: new code reads old data. **Forward compatible**: old code reads new data. You usually want *both* (full compatibility) so deploy order doesn't matter.
- **Protobuf rules**: identify fields by **tag number**, never reuse a retired tag (`reserved`), all fields optional, unknown fields are preserved on pass-through. Add fields freely; never change a field's type or number.
- **Avro rules**: compatibility comes from **schema resolution** between reader and writer schemas; defaults fill missing fields; aliases rename. A **schema registry** (Confluent) enforces compatibility at publish time.
- **The discipline**: additive changes are safe; renames/removals/type-changes are breaking. Treat the schema as a versioned API contract under review.

## 4. gRPC

**Protocol Buffers + HTTP/2 + generated stubs**

**Key ideas**:
- **The stack**: define services and messages in `.proto`; `protoc` generates client stubs and server skeletons in every language; messages serialize as Protobuf; transport is HTTP/2.
- **Why HTTP/2**: **multiplexing** (many concurrent streams on one TCP connection, no head-of-line blocking), binary framing, header compression (HPACK), and built-in flow control — all of which RPC needs.
- **Four call types**:
  - **Unary**: one request → one response (classic RPC).
  - **Server streaming**: one request → a stream of responses. *This is how you stream model tokens or query results.*
  - **Client streaming**: a stream of requests → one response (uploads, telemetry).
  - **Bidirectional streaming**: both stream independently (chat, live sync).
- **Features**: deadlines/timeouts, cancellation propagation, interceptors (auth, tracing, retries), per-call metadata, deflate/gzip, mTLS, pluggable load balancing.
- **gRPC-Web / gRPC-Gateway**: browsers can't speak raw gRPC; a proxy bridges to gRPC-Web, or generates a REST/JSON facade from the same `.proto`.

## 5. Choosing the Protocol at Each Boundary

**Key ideas**:
- **Internal service-to-service**: **gRPC** — typed contracts, codegen, streaming, low overhead, mesh-friendly (Envoy/Istio understand HTTP/2). The default east-west protocol.
- **Public / browser-facing edge**: **REST + JSON** (or GraphQL) — debuggable, cacheable, universal client support, no codegen for third parties. (And **SSE** for streaming responses — see [topic 03](../03-streaming-sse/).)
- **Event streams / data lake**: **Avro** (or Protobuf) over Kafka with a schema registry — evolving records, compact, schema-checked.
- **Latency-critical / large payloads**: **FlatBuffers / Cap'n Proto** — skip the parse step entirely.
- **The rule**: binary + schema inside the system where both ends are yours; text + schemaless at the edge where the client isn't. Don't put JSON on a hot internal path or gRPC in a public browser API without a gateway.

## 6. RPC in an AI Platform

**Key ideas**:
- **Internal model calls**: feature service → ranking model → re-ranker are often gRPC unary calls with tight deadlines; the *embedding service* is a high-QPS gRPC endpoint.
- **Streaming generation between services**: a gateway calls an inference service via **gRPC server-streaming** to get tokens, then re-emits them to the browser as **SSE** (topic 03) — gRPC for east-west, SSE for the last mile.
- **Tensors on the wire**: large activations/embeddings benefit from zero-copy formats or Arrow Flight (gRPC + Arrow) to avoid serialization overhead.
- **Contracts as the platform interface**: the `.proto` files *are* the platform's API surface — versioned, reviewed, and code-generated for every team that integrates.

---

## Decision Cheat Sheet

| Need | Reach for |
|------|-----------|
| Internal microservice RPC | gRPC + Protobuf |
| Public / browser API | REST + JSON |
| Stream tokens to a browser | SSE ([topic 03](../03-streaming-sse/)) |
| Stream between backend services | gRPC server-streaming |
| Evolving records on Kafka / lake | Avro + schema registry |
| Zero-parse, latency-critical | FlatBuffers / Cap'n Proto |
| Compact schemaless blob | MessagePack |

## Patterns Worth Internalizing

- **A network call is a function call that can fail halfway** — design every RPC for timeout, retry, and partial failure, not just the happy path.
- **The schema is a versioned contract.** Identify fields by stable tags, only add (never reuse/retype), and you decouple producer and consumer deploys forever.
- **Binary + schema inside, text + schemaless at the edge** — match the format to the boundary; never put JSON on a hot internal path or raw gRPC in a browser.
- **Streaming is a first-class call shape, not a hack** — server-streaming is the right tool for tokens, results, and progress between services.
- **Codegen the contract** — generate stubs from one source of truth so every language stays in sync.

## Connections to Other Tracks

| Concept | Connected Track | Application |
|---------|-----------------|-------------|
| Encoding, schema evolution, dataflow | [Data Engineering](../../data-engineering/) | Avro/Protobuf on Kafka, the lake |
| Services, APIs, load balancing, mesh | [System Design](../../systems/01-system-design/) | Where RPC fits in an architecture |
| Containers, service mesh, mTLS | [Cloud Native](../../systems/03-cloud-native/) | Running gRPC services on K8s |
| Tracing, deadlines, retries, RED metrics | [Observability](../../systems/04-observability/) | Instrumenting RPC calls |
| Idempotency, retries, exactly-once | [Orchestration & Workers](../05-durable-orchestration-and-workers/) | Surviving partial failure |
| Streaming token delivery to clients | [Streaming & SSE](../03-streaming-sse/) | gRPC stream → SSE at the edge |

## How Companies Apply These Patterns

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| Google | gRPC + Protobuf are theirs; everything internal is RPC | Expert |
| Meta | Thrift everywhere; service-to-service at scale | Expert |
| Stripe | Versioned APIs, idempotency keys, contract design | Expert |
| Netflix / Uber | gRPC mesh, schema registries, streaming | Expert |
| Anthropic / OpenAI | Internal inference RPC, streaming generation between tiers | Expert |
| Confluent / Databricks | Avro + schema registry, Arrow Flight | Expert |
