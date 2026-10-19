# Language and Tool Primers

Just enough of each language and tool to build the course system, each placed immediately before the first module that needs it (design decision D38). A learner starting from high-school algebra meets Python, the shell, C, Rust, HTTP, Go, containers, and Kubernetes in the first two passes; these primers are where each one is introduced from first principles.

## Overview

- **What a primer is**: a `practice` module (`lang.01` to `lang.11`): one chapter plus a small worked solution in [`course/solutions/primers/<id>/`](../../../../course/solutions/primers/), walked through line by line in the chapter. The exercises are not part of the system; they are the smallest artifacts that exercise the ideas the next module relies on.
- **Prerequisites**: none for `lang.01`; each later primer names its own.
- **Estimated time**: 2 to 4 hours per primer.

## Key Takeaways

- Every primer ends in a complete, annotated solution, so "I read about it" becomes "I can explain why each line is there".
- Each primer teaches only what its first call site needs, and its last section names that call site.
- Primers have no code call site of their own: later modules list them under `reading`, and the first module that uses each idea is named in the primer's last section.

## How to Study

- Read the chapter's sections 1 and 2 first; do section 3 on paper.
- Before reading the solution walkthrough, predict each step from the principles; then compare.
- Answer the "Explain it back" questions without looking, then check the answers.
- Each pitfall in the table names what the solution does to avoid it: find that line in the quoted code.

## Primer plan

| ID | Primer | First use | Pass |
|---|---|---|---|
| lang.01 | Python and numpy: arrays, dtypes, broadcasting, views vs copies, `uv` projects | L0.0, rt.01 loader, L0.1 | P0 |
| lang.02 | Shell, git, make, processes: exit codes, signals, environment, pipes | craft.01, every script that reads an exit code | P0 |
| lang.03 | C: memory, pointers, structs, the C11 toolchain, sanitizers, headers and linkage | rt.01, M03.1 | P1 |
| lang.04 | Rust: ownership, traits, `Result`, cargo workspaces, std TCP | L10.0, ds.05, L1.5 | P1 |
| lang.05 | HTTP/1.1, JSON, and Server-Sent Events from the wire up | L10.0, gw.00 | P1 |
| lang.06 | Go: packages, interfaces, goroutines, channels, `context`, `net/http`, `testing` | gw.00, dur.01, load.01 | P1 |
| lang.07 | Containers and Kubernetes: images, layers, Pods, Deployments, Services, Helm, kind | dep.00 | P1 |
| lang.08 | Python asyncio: event loop, tasks, cancellation, bounded concurrency | data.01 | P3 |
| lang.09 | Async Rust and tokio: futures, tasks, channels, cancellation, hyper | L10.5 | P7 |
| lang.10 | Protocol Buffers and gRPC: schema evolution, unary and streaming RPCs | L10.6, dur.01, dur.04 | P7 |
| lang.11 | SQL and SQLite: schema, transactions, WAL mode, indexes, aggregates | gw.07 | P7 |

## Chapters

| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `lang.01` | [Python and numpy: arrays, dtypes, broadcasting, views vs copies, uv projects](01-python-and-numpy.md) | practice | 0 |
| 2 | `lang.02` | [Shell, git, make, processes: exit codes, signals, environment, pipes](02-shell-git-make.md) | practice | 0 |
| 3 | `lang.03` | [C: memory, pointers, structs, headers, and the C11 toolchain](03-c.md) | practice | 1 |
| 4 | `lang.04` | [Rust: ownership, traits, Result, cargo workspaces, std TCP](04-rust.md) | practice | 1 |
| 5 | `lang.05` | [HTTP/1.1, JSON, and Server-Sent Events from the wire up](05-http-and-sse.md) | practice | 1 |
| 6 | `lang.06` | [Go: packages, interfaces, goroutines, channels, context, net/http, testing](06-go.md) | practice | 1 |
| 7 | `lang.07` | [Containers and Kubernetes: images, layers, Pods, Deployments, Services, Helm, kind](07-containers-and-kubernetes.md) | practice | 1 |
| 8 | `lang.08` | [Python asyncio: event loop, tasks, cancellation, bounded concurrency](08-python-asyncio.md) | practice | 3 |
| 9 | `lang.09` | [Async Rust and tokio: futures, tasks, channels, cancellation, hyper](09-async-rust-and-tokio.md) | practice | 7 |
| 10 | `lang.10` | [Protocol Buffers and gRPC: schema evolution, unary and streaming RPCs, stubs](10-protocol-buffers-and-grpc.md) | practice | 7 |
| 11 | `lang.11` | [SQL and SQLite: schema, transactions, WAL mode, indexes, aggregates](11-sql-and-sqlite.md) | practice | 7 |
