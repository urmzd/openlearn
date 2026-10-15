# Coding & Design Patterns

## Overview

- **Primary references**: [Refactoring Guru — Design Patterns](https://refactoring.guru/design-patterns) (free), [*Mostly Adequate Guide to Functional Programming*](https://mostly-adequate.gitbook.io/mostly-adequate-guide/) (free)
- **Supplementary**: *Design Patterns* (Gang of Four), [SourceMaking](https://sourcemaking.com/design_patterns), [*SICP*](https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/6515/sicp.zip/index.html) (closures & higher-order procedures, free), [Python decorators (PEP 318)](https://peps.python.org/pep-0318/)
- **Prerequisites**: comfort in at least one language with first-class functions
- **Estimated time**: 2-3 weeks at 8-10 hrs/week

## Key Takeaways

- **Patterns are the vocabulary the rest of this track is written in.** Temporal workflows are *decorators*; a serving stack behind one call is a *facade*; JAX transforms and middleware are *closures* and *currying*. Learn the pattern once and you recognize it everywhere.
- **A design pattern is a named solution to a recurring design problem** — not code to copy, but a shape to recognize. The value is shared vocabulary and a checklist of trade-offs.
- **The decorator wraps behavior without touching the wrapped thing.** Retries, caching, auth, tracing, rate-limiting — all are decorators in disguise. It's the single most useful pattern in platform code.
- **The facade hides a messy subsystem behind a simple interface.** `client.generate(prompt)` over a tangle of tokenizer + scheduler + KV cache + sampler is a facade — and so is most of a good SDK.
- **Closures and currying are how configuration and partial application actually work.** A closure captures state in a function; currying turns one configurable function into a family of specialized ones. They are the functional answer to "dependency injection."
- **Functional patterns reduce the surface for bugs**: pure functions, immutability, and composition make code testable and parallelizable — which is exactly why JAX, Spark, and React lean on them.

## How to Study

- For each design pattern, find it in a tool you already use (the decorator in a web framework, the facade in an SDK, the strategy in a sampler) — recognition beats memorization.
- Implement a `@retry`, `@cache`, and `@timed` decorator from scratch; then stack them and reason about order.
- Write `curry`, `compose`, and `pipe` from scratch; rebuild a small data transformation as a point-free pipeline.
- Take a tangled function and refactor it behind a facade; take a class hierarchy and replace it with strategy functions.

---

# Concepts & Techniques

## Core Insight

Software complexity is mostly about *change* and *coupling*: who has to change when a requirement does, and how far the blast radius spreads. Patterns are accumulated answers to that — they name the seams where you can absorb change without rewiring everything. Object-oriented patterns (decorator, facade, strategy, adapter) manage change by **composing objects** behind stable interfaces; functional patterns (closures, currying, composition, HOFs) manage it by **composing functions** and avoiding shared mutable state. They are two dialects of the same goal — small, replaceable pieces behind clear contracts — and the systems in this track are built almost entirely from these few shapes.

## 1. Decorator

**Wrap behavior around something without modifying it**

**Key idea**: a decorator presents the same interface as the thing it wraps but adds behavior before/after delegating. It composes — you stack decorators to layer concerns.

```python
def retry(times):
    def wrap(fn):
        def inner(*a, **k):
            for i in range(times):
                try: return fn(*a, **k)
                except Exception:
                    if i == times - 1: raise
        return inner
    return wrap

@retry(3)
@cache              # decorators stack: cache wraps retry wraps the function
def fetch(url): ...
```

**Where it shows up**:
- **Cross-cutting concerns**: retry, caching, logging, tracing, auth, rate-limiting, metrics — each a decorator, kept out of business logic.
- **Temporal/DBOS**: `@workflow`, `@activity`, `@step` decorators turn a plain function into a durable, retried, observable unit ([topic 05](../05-durable-orchestration-and-workers/)).
- **Frameworks**: route handlers (`@app.get`), `@property`, `@lru_cache`, gRPC interceptors (the decorator's RPC cousin), PyTorch hooks.
- **Pattern relatives**: middleware (a decorator over a request/response), the proxy (a decorator that controls access).

## 2. Facade

**One simple interface over a complex subsystem**

**Key idea**: hide the coordination of many moving parts behind a single, intention-revealing entry point. Callers depend on the facade, not the internals — so the internals can change freely.

**Where it shows up**:
- **Inference SDKs**: `client.generate(prompt)` hides tokenization, scheduling, the KV cache, batching, sampling, and detokenization ([LLM Systems](../../ml/04-llm-systems/)).
- **A platform's public API**: one `embed(texts)` over model loading, batching, normalization, and the vector store ([topic 01](../01-training-and-frameworks/)).
- **`vllm.LLM(...)`, HuggingFace `pipeline(...)`** — facades over enormous subsystems.
- **Pattern relatives**: adapter (make an incompatible interface fit), gateway (a facade over a remote system / set of services).

## 3. Other Structural & Behavioral Patterns (Briefly)

| Pattern | One-line shape | Platform instance |
|---------|----------------|-------------------|
| **Strategy** | Swap an algorithm behind a common interface | Pluggable sampler (greedy/top-k/top-p), pluggable sharding |
| **Adapter** | Translate one interface to another | Wrap a vendor SDK to your internal interface |
| **Factory** | Centralize construction choice | `load_model(name)` returns the right backend |
| **Observer / pub-sub** | Notify subscribers of events | Streaming callbacks, metrics, event buses |
| **Singleton** | One shared instance | A connection pool, a model registry handle (use sparingly) |
| **Proxy** | Stand-in that controls access | Lazy loading, rate-limiting, the RPC client stub |
| **Iterator / generator** | Produce a sequence lazily | Token streaming, dataloaders, paging an API |

**Caution**: patterns are tools, not goals. Reaching for a pattern where a plain function would do (over-engineering) is its own anti-pattern.

## 4. Functional Programming Patterns

**Compose functions instead of mutating state**

### Closures

A function plus the environment it captured. The functional way to bundle state with behavior — and the mechanism under almost everything below.

```javascript
function counter() { let n = 0; return () => ++n; }   // n is captured, private
const next = counter(); next(); next();               // 1, 2
```
**Shows up as**: configured callbacks, memoization tables, private state without classes, the captured `params` in a JAX/Optax update step.

### Currying & Partial Application

Turn a function of many arguments into a chain of one-argument functions, so you can fix some arguments now and the rest later — producing specialized functions from general ones.

```javascript
const add = a => b => a + b;     // curried
const inc = add(1);              // partial application → a specialized fn
[1,2,3].map(inc);                // [2,3,4]
```
**Shows up as**: configuration (`makeLogger(level)(message)`), the functional form of dependency injection, building a family of endpoints/handlers from one template, JAX `partial(jit, static_argnums=...)`.

### Higher-Order Functions & Composition

Functions that take/return functions; `compose`/`pipe` to build pipelines out of small steps.

```javascript
const pipe = (...fns) => x => fns.reduce((v, f) => f(v), x);
const clean = pipe(trim, lower, dedupe);   // a data pipeline as data
```
**Shows up as**: `map`/`filter`/`reduce`, transducers, Spark/Pandas chains, middleware stacks, the whole `jit(vmap(grad(f)))` composition in [JAX](../01-training-and-frameworks/).

### Purity & Immutability

A pure function's output depends only on its inputs and it mutates nothing — so it's trivially testable, cacheable (memoizable), and safe to parallelize.

**Shows up as**: why JAX requires pure functions (so XLA can transform them), why Spark/MapReduce can distribute work, why React renders are pure, why event sourcing works ([topic 04](../04-distributed-data-and-caching/)). **Memoization** is just caching a pure function — the same bargain as [caching](../04-distributed-data-and-caching/), at function granularity.

## 5. Composition over Inheritance — the Through-Line

Both halves of this topic push the same lesson: **build big behavior from small, replaceable pieces with clear contracts.** Decorators compose objects; `pipe` composes functions; the worker pattern composes tasks; microservices compose systems. Deep inheritance hierarchies and god-objects are the anti-pattern — they couple things that should change independently. When in doubt, prefer a small function or a wrapper over a new layer of class hierarchy.

---

## Patterns Worth Internalizing

- **Decorator = wrap, don't modify.** Cross-cutting concerns (retry/cache/auth/trace) belong in wrappers, not in business logic.
- **Facade = one door, many rooms.** Expose intent; hide coordination so internals stay free to change.
- **Closure = state captured in a function; currying = specialization by fixing arguments.** Together they replace most "configuration" and "injection" machinery.
- **Compose small, pure pieces.** Purity buys testability, memoization, and parallelism for free — the reason JAX, Spark, and event sourcing exist.
- **A pattern is a recognition tool, not a mandate.** The skill is naming the shape you already see; over-applying patterns is its own smell.

## Connections to Other Tracks

| Concept | Connected Track | Application |
|---------|-----------------|-------------|
| Pure functions enabling transforms | [Training & Frameworks](../01-training-and-frameworks/) | Why JAX is functional |
| Decorators wrapping durable steps | [Orchestration & Workers](../05-durable-orchestration-and-workers/) | `@workflow`/`@activity` APIs |
| Interceptors, stubs, the proxy | [RPC & Protocols](../02-rpc-and-protocols/) | gRPC middleware and client stubs |
| Iterator/generator, backpressure | [Streaming & SSE](../03-streaming-sse/) | Lazy token sequences |
| Memoization as caching a pure function | [Distributed Data & Caching](../04-distributed-data-and-caching/) | Same bargain, function-level |

## How Companies Apply These Patterns

| Company | The pattern they lean on | Instance |
|---------|--------------------------|----------|
| Google / Meta | Interceptors & decorators for cross-cutting concerns | gRPC interceptors, framework middleware |
| Temporal / DBOS | Decorator-defined durable units | `@workflow` / `@activity` / `@step` |
| Jane Street | Functional purity & composition | OCaml; immutability for correctness |
| React / Meta | Pure functions + composition | Function components, hooks (closures) |
| Databricks | HOFs & lazy composition | Spark transformation chains |
| Any SDK team | Facade over a complex backend | `client.generate(...)`, `pipeline(...)` |
