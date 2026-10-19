# Go

Go is the language this curriculum reaches for when a service has to be simple to read, cheap to run, and correct under concurrency. Its design is deliberately small: a garbage collector so you never write `free`, structural interfaces so types satisfy contracts without declaring it, goroutines and channels so concurrent code reads like sequential code, and explicit error values so every failure path is visible in the source. The exercises in this chapter are chosen to make you lean on each of those choices until you can say why Go made it.

The chapter is a set of exercise briefs rather than worked solutions. Each brief names a classic system to build and the language features it forces you to use. Several of them are the same problem that a section of *AI Platform Engineering* or *Algorithms* solves in full, and for those the brief links to that section, so you can reason through your own design first and then compare it with a complete, tested one.

> **Learning objectives**
> - Describe Go's memory model, type system, error handling, and concurrency primitives, and say what each trades away.
> - Apply the core Go idioms (accept interfaces, return structs; explicit error checks; `context` propagation; `defer` cleanup) when designing a small system.
> - Choose between channels and `sync` primitives for a given concurrency problem and justify the choice.
> - Reason through the design of each exercise before reading the matching worked solution elsewhere in the books.

## Before you begin

You should be comfortable writing a small Go program, running `go test`, and reading a function signature with channels in it. If you are not, read [Go: packages, interfaces, goroutines, channels, context, net/http, testing](../../../../content/courses/programming-languages/02-language-and-tool-primers/06-go.md) in *Programming Languages* first. The concurrency exercises also assume the vocabulary of locks, condition variables, and cancellation that the systems chapters of the books build up.

## The language in brief

Go manages memory with a garbage collector. Values are stack-allocated by default, and the compiler's escape analysis moves them to the heap only when a reference outlives the function. You get pointers, but no pointer arithmetic, so the class of bugs that the C chapter spends its time on simply cannot be written.

The type system is static and structural. An interface is satisfied implicitly: any type with the right method set implements it, with no `implements` clause. Generics arrived in Go 1.18 and are constrained by interfaces, such as `cmp.Ordered` for types that support `<`.

Errors are ordinary values of the `error` interface, returned alongside results. There are no exceptions. You inspect wrapped errors with `errors.Is` and `errors.As`, and the price of that visibility is the familiar `if err != nil` after every call that can fail.

Concurrency is built into the language: goroutines are cheap threads scheduled by the runtime, channels pass values between them, `select` waits on several channel operations at once, the `sync` package supplies mutexes, wait groups, and `sync.Once`, and `context.Context` carries cancellation and deadlines across call chains. The toolchain is `go build`, `go test`, and `go vet`, with dependencies declared in `go.mod`. The exercises target Go 1.22 or later.

## Idioms to practise

These are the habits the exercises reward. Each one is short to state and takes practice to apply well.

- **Accept interfaces, return structs.** Callers can pass anything that fits, and you keep the freedom to add methods to what you return.
- **`if err != nil`.** Check every error explicitly, at the point it happens.
- **Table-driven tests** with the `testing` package: one slice of cases, one loop.
- **Thread `context.Context`** through every call chain that can block or be cancelled.
- **Give channels a direction** in function signatures (`chan<-`, `<-chan`) so the compiler enforces who sends and who receives.
- **Embed for composition** instead of reaching for inheritance.
- **`defer` for cleanup.** Deferred calls run in last-in, first-out order when the function returns.

## In this chapter

The briefs are listed below, ordered by concept rather than by difficulty, so choose by the difficulty rating (1 is easiest, 5 hardest) rather than by number. Worked Go solutions for these exercises have not been written yet; where a full solution of the same problem exists elsewhere in the books, the last column links to it.

| # | Exercise | Concepts | Difficulty (1-5) | Worked elsewhere |
|---|----------|----------|------------------|------------------|
| 01 | Concurrent web crawler | Goroutines, channels, `sync.WaitGroup`, rate limiting | 3 | [RAG Ingest: Crawler, Chunker, Embedder, and Store](../../../../content/courses/ai-platform-engineering/07-retrieval-and-rag/06-rag-ingest.md) |
| 02 | In-memory key-value store | `sync.RWMutex`, interfaces, HTTP handler | 2 | [Response cache: LRU, TTL, singleflight, and tenant-scoped keys](../../../../content/courses/ai-platform-engineering/04-distributed-data-and-caching/01-response-cache.md) |
| 03 | Worker pool | Bounded concurrency, `context` cancellation | 3 | [The task protocol and workers](../../../../content/courses/ai-platform-engineering/05-durable-orchestration-and-workers/04-task-protocol-and-workers.md) |
| 04 | Generic sorted set | Generics, type constraints, `cmp.Ordered` | 3 | [Durable timers](../../../../content/courses/ai-platform-engineering/05-durable-orchestration-and-workers/07-durable-timers.md) |
| 05 | Log-structured storage | `io.Reader`/`Writer`, binary encoding, file I/O | 4 | [The append-only event log](../../../../content/courses/ai-platform-engineering/05-durable-orchestration-and-workers/01-event-log.md) |
| 06 | Pub/sub message broker | Channels, fan-out, `select`, graceful shutdown | 3 | [The task queue](../../../../content/courses/ai-platform-engineering/05-durable-orchestration-and-workers/03-task-queue.md) |
| 07 | Rate limiter (token bucket) | `time.Ticker`, atomic operations, middleware | 2 | [Rate Limiting: RPM Token Buckets and TPM Reserve/Settle](../../../../content/courses/ai-platform-engineering/08-authorization-and-access-control/02-rate-limiting.md) |
| 08 | Raft consensus (simplified) | State machines, RPC, leader election | 5 | [Raft high availability for the durable log](../../../../content/courses/ai-platform-engineering/05-durable-orchestration-and-workers/10-raft-ha.md) |
| 09 | Dependency injection container | Reflection, interfaces, `sync.Once` | 3 | |
| 10 | gRPC service and client | Protocol buffers, streaming, interceptors | 4 | [The task protocol and workers](../../../../content/courses/ai-platform-engineering/05-durable-orchestration-and-workers/04-task-protocol-and-workers.md) |

All of the linked sections are in *AI Platform Engineering*, where the problem is solved as part of a larger platform rather than in isolation. Read the brief, sketch your own design, and only then open the worked version.

## Further reading

- [A Tour of Go](https://go.dev/tour/): free and interactive.
- [Effective Go](https://go.dev/doc/effective_go): free, and essential reading.
- [Go standard library](https://pkg.go.dev/std): the API documentation.
- [Go Blog](https://go.dev/blog/): official articles.
- *The Go Programming Language* by Donovan and Kernighan: the comprehensive book.
