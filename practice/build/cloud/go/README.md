# Go Practice

Master goroutines, channels, interfaces, and simplicity-driven design.

## Language Essentials

| Concept | Details |
|---------|---------|
| **Memory Model** | GC, stack-allocated by default, escape analysis, pointers but no arithmetic |
| **Type System** | Static, structural typing, interfaces (implicit), generics (1.18+) |
| **Error Handling** | `error` interface, multiple returns, `errors.Is`/`As`, no exceptions |
| **Concurrency** | Goroutines, channels, `select`, `sync` package, `context` cancellation |
| **Build** | `go build`, modules (`go.mod`), `go test`, `go vet` |
| **Standard** | Go 1.22+ |

## Key Idioms

- **Accept interfaces, return structs**
- **`if err != nil`** -- explicit error checking
- **Table-driven tests** with `testing` package
- **`context.Context`** threaded through call chains
- **Channel direction** in function signatures (`chan<-`, `<-chan`)
- **Embedding** for composition over inheritance
- **`defer`** for cleanup, runs LIFO

## Reference Documentation

- [A Tour of Go](https://go.dev/tour/) -- free, interactive
- [Effective Go](https://go.dev/doc/effective_go) -- free, essential reading
- [Go standard library](https://pkg.go.dev/std) -- API docs
- [Go Blog](https://go.dev/blog/) -- official articles
- *The Go Programming Language* (Donovan & Kernighan) -- comprehensive

## How this page is organised

The table below is an index of exercise briefs: what to build, the concepts it
exercises, and how hard it is. Exercises are ordered by concept, not by
difficulty. Where a row names a course version (for example `ds.02`), that
module's chapter walks through a full solution of the same problem.

## Exercises

Worked solutions for these exercises are not written yet.

| # | Exercise | Concepts | Difficulty |
|---|----------|----------|------------|
| 01 | Concurrent web crawler | Goroutines, channels, `sync.WaitGroup`, rate limiting; course version: `ag.06` | ⭐⭐⭐ |
| 02 | In-memory key-value store | `sync.RWMutex`, interfaces, HTTP handler; course version: `gw.06` | ⭐⭐ |
| 03 | Worker pool | Bounded concurrency, `context` cancellation; course version: `dur.04` | ⭐⭐⭐ |
| 04 | Generic sorted set | Generics, type constraints, `cmp.Ordered`; course version: `dur.07` | ⭐⭐⭐ |
| 05 | Log-structured storage | `io.Reader`/`Writer`, binary encoding, file I/O; course version: `dur.01` | ⭐⭐⭐⭐ |
| 06 | Pub/sub message broker | Channels, fan-out, `select`, graceful shutdown; course version: `dur.03` | ⭐⭐⭐ |
| 07 | Rate limiter (token bucket) | `time.Ticker`, atomic operations, middleware; course version: `gw.03` | ⭐⭐ |
| 08 | Raft consensus (simplified) | State machines, RPC, leader election; course version: `dur.10` | ⭐⭐⭐⭐⭐ |
| 09 | Dependency injection container | Reflection, interfaces, `sync.Once` | ⭐⭐⭐ |
| 10 | gRPC service + client | Protocol buffers, streaming, interceptors; course version: `dur.04` | ⭐⭐⭐⭐ |
