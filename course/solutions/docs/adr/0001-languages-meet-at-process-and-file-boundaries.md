# ADR-0001: Languages meet at process and file boundaries

## Status

Accepted (2026-10-09)

## Context

Python trains models, Rust serves them, Go fronts the engine, and optional C exercises explore kernels. The programs need reproducible data exchange without coupling their memory layouts or runtimes.

## Decision

Python writes weights as safetensors and tokenizers as `tokenizer.json`. Shared fixture files establish parity between Python, Rust, Go, and standalone C exercises. Go and Rust communicate through HTTP/JSON, Server-Sent Events, and internal gRPC. Each service can be built, tested, and restarted independently.

The Rust engine uses `candle-core` and `candle-nn` for tensors. It implements its own model layers. Optional C kernels run in their own test binaries and do not become an engine backend.

## Consequences

- File formats need versioned schemas and parity fixtures.
- HTTP and gRPC make service boundaries observable and independently testable.
- Optional C modules teach layout and numerical behavior without changing core pass gates.
- Each language owns its memory and errors within its process.

## Alternatives considered

| Option | Why we did not choose it |
|---|---|
| In-process language bindings | They would couple ownership and failure behavior across runtimes. |
| A C backend for the Rust engine | It would make optional C exercises a core serving dependency. |
