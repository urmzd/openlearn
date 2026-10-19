# RPC and Protocols

Two processes can only exchange bytes. A *protocol* is the agreement that turns a meaningful value on one machine into bytes and back into a meaningful value on another, across languages, versions, and time. *RPC* (remote procedure call) layers a familiar shape on top: call a method, get a return value, as if the callee were local. The whole discipline is managing the ways that illusion breaks, because the network is slow, lossy, and can fail halfway, and the ways the agreement must change without coordinated, simultaneous deploys.

That makes three axes you optimize: encoding efficiency (how many bytes, how fast to parse), schema evolution (how producer and consumer change independently), and streaming (how a result that arrives over time crosses the wire). An AI platform is full of service boundaries, from the gateway to the inference engine to the embedding service, and each boundary is a protocol decision. This chapter gives you the vocabulary and the rules for making those decisions well.

The chapter starts with what RPC is and why it leaks. It then compares serialization formats, the bytes on the wire, and spends a full part on schema evolution, which is the hardest problem in practice. With that groundwork it covers gRPC, the default for internal traffic, then gives a rule for choosing a protocol at each boundary, and closes with how RPC shows up specifically in an AI platform.

> **Learning objectives**
> - Explain why a remote call is not a local call, and design an RPC for timeouts, retries with idempotency keys, and partial failure.
> - Compare JSON, Protocol Buffers, Thrift, Avro, MessagePack, and the zero-copy formats on schema, size, speed, and evolution.
> - State the Protobuf and Avro rules for backward- and forward-compatible schema changes, and identify a breaking change.
> - Describe gRPC's stack (Protobuf, HTTP/2, generated stubs) and its four call types, and say which fits streaming model tokens.
> - Choose the protocol for a given boundary: internal service mesh, public edge, event stream, or latency-critical path.

## Before you begin

You should be comfortable with services, APIs, and load balancing at the level of [System Design](../../systems/01-system-design/) in *Systems & Architecture*, and with basic networking: what TCP and HTTP do. No prior experience with gRPC or Protobuf is assumed.

## In this chapter

- [What RPC is, and why it leaks](#what-rpc-is-and-why-it-leaks): the stub model, partial failure, and the fallacies of distributed computing.
- [Serialization formats](#serialization-formats): the bytes on the wire, schema-based versus schemaless, and zero-copy.
- [Schema evolution](#schema-evolution): backward and forward compatibility and the rules that let producers and consumers deploy independently.
- [gRPC](#grpc): Protocol Buffers over HTTP/2 with generated stubs, and its four call types.
- [Choosing the protocol at each boundary](#choosing-the-protocol-at-each-boundary): binary and schema inside, text at the edge.
- [RPC in an AI platform](#rpc-in-an-ai-platform): model calls, streaming generation between services, and tensors on the wire.

## What RPC is, and why it leaks

The RPC model is simple. A client invokes a method on a *stub*. The stub *marshals* (serializes) the arguments and sends them; the server *unmarshals* them, runs the method, and returns the result the same way. The network is hidden behind a function signature.

The trouble is that the signature hides exactly the things that matter. A local call cannot time out, lose its response, or partially execute; a remote one can do all three. So every RPC must be designed for **partial failure**: timeouts, retries (with idempotency keys), backoff, deadlines and cancellation, and circuit breakers. The classic **fallacies of distributed computing** name the assumptions you must not make: the network is *not* reliable, zero-latency, infinite-bandwidth, secure, or free. RPC frameworks paper over the syntax of a remote call, not these realities.

Retries are unavoidable, which means a request may arrive more than once. Write operations should therefore be **idempotent**: safe to apply more than once, usually by carrying an idempotency key the server deduplicates on. The same idea is the foundation of durable execution in [Durable Orchestration and Workers](../05-durable-orchestration-and-workers/). The rule to carry forward is that a network call is a function call that can fail halfway; design for timeout, retry, and partial failure, not just the happy path.

## Serialization formats

Once you accept that a call crosses the network, the next question is what the bytes look like. The common formats differ in whether they require a schema, how they encode, and what they are good and bad at:

| Format | Schema | Encoding | Strength | Weakness |
|--------|--------|----------|----------|----------|
| **JSON** | None | Text | Human-readable, universal | Verbose, slow parse, no types |
| **Protocol Buffers** | Required (`.proto`) | Binary, tag-based | Compact, fast, codegen, evolution | Not human-readable |
| **Apache Thrift** | Required (`.thrift`) | Binary | Protobuf-like + full RPC stack | Two ecosystems (Facebook/Apache) |
| **Apache Avro** | Required (JSON schema) | Binary, schema-resolved | Schema travels with data; great for Kafka/lakes | Needs writer schema to read |
| **MessagePack** | None | Binary | "Binary JSON", compact, schemaless | No contract/evolution |
| **Cap'n Proto / FlatBuffers** | Required | Binary, **zero-copy** | Read fields without parsing | Larger on wire, less ergonomic |

The first distinction is **schema-based versus schemaless**. A schema buys you a typed contract, smaller messages (field *names* are not on the wire; tag numbers are), and generated code in every language. Schemaless formats such as JSON and MessagePack buy you flexibility and no build step. You are trading bytes and safety for legibility, which makes the wire format an architecture decision rather than a detail.

The second distinction is **zero-copy**. In FlatBuffers and Cap'n Proto, the serialized bytes *are* the in-memory layout, so you read a field by offset with no parse or allocation step. That matters for huge messages and hot paths: game state, ML tensors, memory-mapped files. The cost is larger messages on the wire and less ergonomic APIs.

Avro deserves its own note because it solves evolution differently. The *writer's* schema is stored with the data, and a *reader's* schema is resolved against it at read time. That property makes Avro the standard for evolving records in Kafka and data lakes, as [Batch & Streaming Processing](../../data-engineering/03-batch-streaming/) in *Data Engineering* shows. Its price is that you need the writer's schema to read anything.

## Schema evolution

Choosing a format is a one-time decision; changing the schema is a permanent activity, and it is the hardest part of the whole subject. The goal is to deploy producer and consumer independently, in any order, without breaking either.

Two properties define success. A change is **backward compatible** if new code can read old data, and **forward compatible** if old code can read new data. You usually want both, called full compatibility, so that deploy order does not matter. Get this wrong and one deploy breaks every caller.

Protocol Buffers achieve this by identifying fields by **tag number**. All fields are optional, and unknown fields are preserved when a message passes through code that does not know them. The rules follow: add fields freely, never change a field's type or number, and never reuse a retired tag; mark it `reserved` instead, so no future field can accidentally take it.

Avro achieves compatibility through **schema resolution** between the reader's and writer's schemas. Defaults fill fields the writer did not have, and aliases handle renames. A **schema registry** (such as Confluent's) enforces compatibility at publish time, so a breaking schema never reaches the topic.

The discipline that ties both together: additive changes are safe; renames, removals, and type changes are breaking. Treat the schema as a versioned API contract under review. Identify fields by stable tags, only add, never reuse or retype, and you decouple producer and consumer deploys for good.

## gRPC

With a format and evolution rules in hand, you can assemble a full RPC system, and gRPC is the one most internal service-to-service traffic uses. Its stack is three pieces: you define services and messages in a `.proto` file, `protoc` generates client stubs and server skeletons in every language, messages serialize as Protobuf, and the transport is HTTP/2. Generating stubs from one source of truth keeps every language in sync.

HTTP/2 is chosen deliberately. It provides **multiplexing** (many concurrent streams on one TCP connection, with no HTTP-level head-of-line blocking), binary framing, header compression (HPACK), and built-in flow control, all of which RPC needs.

gRPC supports four call types, and the streaming ones are first-class shapes, not hacks:

- **Unary**: one request, one response; classic RPC.
- **Server streaming**: one request, a stream of responses. This is how you stream model tokens or query results.
- **Client streaming**: a stream of requests, one response; uploads and telemetry.
- **Bidirectional streaming**: both sides stream independently; chat and live sync.

Around the calls, gRPC provides deadlines and timeouts, cancellation propagation, interceptors (for auth, tracing, and retries), per-call metadata, deflate and gzip compression, mTLS, and pluggable load balancing. The one place it does not reach is the browser: browsers cannot speak raw gRPC. **gRPC-Web** bridges this through a proxy, and **gRPC-Gateway** generates a REST/JSON facade from the same `.proto`.

## Choosing the protocol at each boundary

No single protocol is right everywhere, so the practical skill is matching the protocol to the boundary. Inside the system, for **service-to-service** traffic, gRPC is the default east-west protocol: typed contracts, codegen, streaming, low overhead, and mesh-friendly, since Envoy and Istio understand HTTP/2. At the **public or browser-facing edge**, REST with JSON (or GraphQL) wins: it is debuggable, cacheable, universally supported, and needs no codegen from third parties. Streaming responses at the edge use **SSE**, the subject of the [next chapter](../03-streaming-sse/).

For **event streams and the data lake**, use Avro (or Protobuf) over Kafka with a schema registry, which gives you evolving, compact, schema-checked records. On **latency-critical paths with large payloads**, FlatBuffers or Cap'n Proto skip the parse step entirely.

The rule underneath all of these is binary and schema inside the system, where both ends are yours, and text and schemaless at the edge, where the client is not. Do not put JSON on a hot internal path, and do not put raw gRPC in a public browser API without a gateway. The common needs map to protocols as follows:

| Need | Reach for |
|------|-----------|
| Internal microservice RPC | gRPC + Protobuf |
| Public / browser API | REST + JSON |
| Stream tokens to a browser | SSE ([Streaming and SSE](../03-streaming-sse/)) |
| Stream between backend services | gRPC server-streaming |
| Evolving records on Kafka / lake | Avro + schema registry |
| Zero-parse, latency-critical | FlatBuffers / Cap'n Proto |
| Compact schemaless blob | MessagePack |

## RPC in an AI platform

An AI platform uses every one of these patterns. **Internal model calls**, such as feature service to ranking model to re-ranker, are often gRPC unary calls with tight deadlines, and the embedding service is a high-QPS gRPC endpoint. **Streaming generation between services** combines two protocols: a gateway calls an inference service with gRPC server-streaming to receive tokens, then re-emits them to the browser as SSE. gRPC handles the east-west hop and SSE the last mile.

**Tensors on the wire** are a special case. Large activations and embeddings benefit from zero-copy formats or Arrow Flight (gRPC plus Arrow) to avoid serialization overhead. And at the organizational level, **contracts are the platform interface**: the `.proto` files *are* the platform's API surface, versioned, reviewed, and code-generated for every team that integrates.

The industry shows the same split. Google built gRPC and Protobuf, and everything internal there is RPC; Meta uses Thrift for service-to-service traffic at scale. Stripe is known for versioned APIs, idempotency keys, and careful contract design. Netflix and Uber run gRPC meshes with schema registries and streaming. Anthropic and OpenAI use internal inference RPC with streaming generation between tiers, and Confluent and Databricks build on Avro with schema registries and Arrow Flight.

## Where this leads

The next chapter, [Streaming and SSE](../03-streaming-sse/), picks up the last-mile half of the streaming story: how a gRPC stream between services becomes SSE to the browser. [Durable Orchestration and Workers](../05-durable-orchestration-and-workers/) builds on idempotency, retries, and exactly-once effects to survive partial failure. Outside this book, [Data Engineering](../../data-engineering/) covers encoding, schema evolution, and dataflow for Avro and Protobuf on Kafka and the lake; [System Design](../../systems/01-system-design/) places RPC in a service architecture; [Cloud Native](../../systems/03-cloud-native/) runs gRPC services on Kubernetes with a service mesh and mTLS; and [Observability](../../systems/04-observability/) instruments RPC calls with tracing, deadlines, retries, and RED metrics.

### Further reading

- [gRPC docs](https://grpc.io/docs/) and [Protocol Buffers docs](https://protobuf.dev/) (free): the primary references.
- [Apache Thrift](https://thrift.apache.org/docs/), the [Apache Avro spec](https://avro.apache.org/docs/), [Cap'n Proto](https://capnproto.org/), [FlatBuffers](https://flatbuffers.dev/), and [MessagePack](https://msgpack.org/).
- [HTTP/2 (RFC 9113)](https://www.rfc-editor.org/rfc/rfc9113.html).
- Martin Kleppmann, *Designing Data-Intensive Applications*, Chapter 4 (encoding and evolution).

## Key terms

- **RPC**: a remote call presented as a local function call through a generated stub.
- **Marshalling**: serializing a call's arguments or result into bytes for the wire.
- **Partial failure**: a remote call that times out, loses its response, or executes only partly.
- **Idempotency**: the property that applying an operation more than once has the same effect as applying it once.
- **Backward / forward compatibility**: new code reading old data / old code reading new data.
- **Tag number**: the stable integer that identifies a Protobuf field on the wire in place of its name.
- **Schema resolution**: Avro's reconciliation of a reader's schema with the writer's schema stored alongside the data.
- **Zero-copy format**: a serialization whose bytes are the in-memory layout, readable by offset without parsing.
- **Server streaming**: a gRPC call shape with one request and a stream of responses.

## Summary

- RPC makes a network call look like a function call, and the leak in that abstraction (partial failure, latency, serialization) is where the engineering lives.
- The wire format is an architecture decision: schema-based binary formats give typed contracts, compact bytes, and codegen; schemaless text gives legibility and no tooling.
- Schema evolution is the real problem. Protobuf evolves by tag number, Avro by reader/writer schema resolution; additive changes are safe and everything else breaks callers.
- gRPC is Protobuf plus HTTP/2 plus generated stubs, with four call types; server streaming is the tool for tokens and results between services.
- Match the protocol to the boundary: gRPC inside the mesh, REST/JSON at the public edge, Avro on Kafka and the lake, zero-copy formats where you cannot afford a parse.

## Review questions

1. Why must retried write RPCs be idempotent?
   <details><summary>Answer</summary>A client cannot tell a lost request from a lost response, so it retries, and the server may receive the same write twice. An idempotency key lets the server apply it once and return the same result for the duplicate.</details>

2. Why are Protobuf messages smaller than the equivalent JSON?
   <details><summary>Answer</summary>Field names are not on the wire; each field is identified by its tag number, and values are binary-encoded. JSON repeats every field name as text in every message.</details>

3. A team removes a Protobuf field and later adds a new field with the same tag number but a different type. What goes wrong, and how should they have done it?
   <details><summary>Answer</summary>Old producers and stored data still write the old field under that tag, and new consumers decode those bytes as the new type, producing wrong values or decode errors. They should have marked the retired tag `reserved` and given the new field a fresh number.</details>

4. Why is Avro the standard for records in Kafka and data lakes?
   <details><summary>Answer</summary>The writer's schema is stored with (or registered for) the data, and each reader resolves its own schema against it, filling missing fields from defaults and handling renames with aliases. Records written years apart remain readable, and a schema registry rejects incompatible changes at publish time.</details>

5. Which gRPC call type would a gateway use to receive tokens from an inference service, and what does it send to the browser?
   <details><summary>Answer</summary>Server streaming: one request, a stream of token responses. Browsers cannot speak raw gRPC, so the gateway re-emits the tokens to the browser as SSE.</details>

## Exercises

- Define one `.proto` service, generate stubs in two languages, and call across them. Then add a field and confirm that old and new clients still interoperate.
- Encode the same record as JSON, Protobuf, and Avro, and compare byte size and parse time.
- Implement all four gRPC call types, especially server streaming, which is how you would stream model tokens to another service.
- Break compatibility on purpose (reuse a tag number, change a type) and observe what happens to old consumers.
