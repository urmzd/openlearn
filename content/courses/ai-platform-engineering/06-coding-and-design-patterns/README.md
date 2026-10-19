# Coding and Design Patterns

Software complexity is mostly about *change* and *coupling*: who has to change when a requirement does, and how far the blast radius spreads. Design patterns are accumulated answers to that problem. They name the seams where you can absorb change without rewiring everything. A **design pattern** is a named solution to a recurring design problem: not code to copy, but a shape to recognize. Its value is a shared vocabulary and a checklist of trade-offs.

This chapter matters at this point in the book because patterns are the vocabulary the rest of it is written in. Temporal workflows are *decorators*; a serving stack behind one call is a *facade*; JAX transforms and middleware are *closures* and *currying*. Learn a pattern once and you recognize it everywhere. The chapter covers two dialects of the same goal. Object-oriented patterns (decorator, facade, strategy, adapter) manage change by **composing objects** behind stable interfaces. Functional patterns (closures, currying, composition, higher-order functions) manage it by **composing functions** and avoiding shared mutable state. Both aim at small, replaceable pieces behind clear contracts, and the systems in this book are built almost entirely from these few shapes.

The chapter is a single, self-contained reading rather than a set of build sections. It starts with the two structural patterns you will meet most often, the decorator and the facade, surveys the other common patterns briefly, then turns to the functional patterns, and ends with the principle that ties both halves together: composition over inheritance.

> **Learning objectives**
> - Recognize the decorator, facade, strategy, adapter, factory, observer, singleton, proxy, and iterator patterns in tools you already use.
> - Write retry and caching decorators and reason about the order in which stacked decorators apply.
> - Use closures and currying to capture configuration and specialize general functions.
> - Build data pipelines with `compose` and `pipe`, and explain what purity and immutability buy.
> - Judge when a pattern helps and when a plain function would do.

## Before you begin

You need comfort in at least one language with first-class functions. The examples use Python for the object-oriented patterns and JavaScript for the functional ones, but nothing depends on either language. The best way to read this chapter is actively: for each pattern, find it in a tool you already use (the decorator in a web framework, the facade in an SDK, the strategy in a sampler), because recognition beats memorization. Then implement a few from scratch: `@retry`, `@cache`, and `@timed` decorators that you stack and reason about; `curry`, `compose`, and `pipe`, with a small data transformation rebuilt as a point-free pipeline; a tangled function refactored behind a facade; and a class hierarchy replaced with strategy functions.

## The decorator: wrap behavior without modifying it

A decorator presents the same interface as the thing it wraps but adds behavior before or after delegating to it. Because the interface does not change, decorators compose: you stack them to layer concerns.

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

Read the stack carefully, because order is behavior. Decorators apply bottom-up, so `@cache` wraps `fetch` first and `@retry(3)` wraps the cached function. A call goes through the retry loop, then the cache, then (on a miss) the network. The comment in the listing names the layers from the outside in.

The decorator is the single most useful pattern in platform code, because retries, caching, logging, tracing, auth, rate-limiting, and metrics are all **cross-cutting concerns**: each can be a decorator, kept out of the business logic. In durable execution systems such as Temporal and DBOS, the `@workflow`, `@activity`, and `@step` decorators turn a plain function into a durable, retried, observable unit (see [Durable Orchestration & Workers](../05-durable-orchestration-and-workers/)). Frameworks are full of them: route handlers (`@app.get`), `@property`, `@lru_cache`, gRPC interceptors (the decorator's RPC cousin), and PyTorch hooks. Its close relatives are **middleware**, a decorator over a request and response, and the **proxy**, a decorator that controls access.

## The facade: one simple interface over a complex subsystem

Where the decorator adds behavior around one thing, the facade hides the coordination of many. It puts a single, intention-revealing entry point in front of a subsystem with many moving parts. Callers depend on the facade, not the internals, so the internals can change freely: one door, many rooms.

You use facades constantly. In an inference SDK, `client.generate(prompt)` hides tokenization, scheduling, the KV cache, batching, sampling, and detokenization (see [LLM Systems](../../machine-learning/04-llm-systems/) in *Machine Learning*). A platform's public API might offer one `embed(texts)` over model loading, batching, normalization, and the vector store (see [Training & Frameworks](../01-training-and-frameworks/)). `vllm.LLM(...)` and Hugging Face's `pipeline(...)` are facades over enormous subsystems, and so is most of a good SDK. The facade's relatives are the **adapter**, which makes an incompatible interface fit, and the **gateway**, a facade over a remote system or a set of services.

## Other structural and behavioral patterns

The decorator and facade do most of the work, but a handful of other patterns recur often enough to know by name. The table gives each one's shape and a platform instance:

| Pattern | One-line shape | Platform instance |
|---------|----------------|-------------------|
| **Strategy** | Swap an algorithm behind a common interface | Pluggable sampler (greedy/top-k/top-p), pluggable sharding |
| **Adapter** | Translate one interface to another | Wrap a vendor SDK to your internal interface |
| **Factory** | Centralize construction choice | `load_model(name)` returns the right backend |
| **Observer / pub-sub** | Notify subscribers of events | Streaming callbacks, metrics, event buses |
| **Singleton** | One shared instance | A connection pool, a model registry handle (use sparingly) |
| **Proxy** | Stand-in that controls access | Lazy loading, rate-limiting, the RPC client stub |
| **Iterator / generator** | Produce a sequence lazily | Token streaming, dataloaders, paging an API |

> **Warning:** Patterns are tools, not goals. Reaching for a pattern where a plain function would do is over-engineering, and that is its own anti-pattern. The skill is naming the shape you already see, not imposing one.

## Functional patterns: compose functions instead of mutating state

The object-oriented patterns compose objects. The functional patterns reach the same goal by composing functions, and they rest on one mechanism: the closure.

### Closures

A **closure** is a function plus the environment it captured. It is the functional way to bundle state with behavior, and it is the mechanism under almost everything that follows.

```javascript
function counter() { let n = 0; return () => ++n; }   // n is captured, private
const next = counter(); next(); next();               // 1, 2
```

Closures show up as configured callbacks, memoization tables, private state without classes, and the captured `params` in a JAX or Optax update step. The `retry` decorator above is itself a closure: `inner` captures `fn` and `times`.

### Currying and partial application

**Currying** turns a function of many arguments into a chain of one-argument functions, so you can fix some arguments now and the rest later. Fixing some arguments, **partial application**, produces specialized functions from general ones.

```javascript
const add = a => b => a + b;     // curried
const inc = add(1);              // partial application → a specialized fn
[1,2,3].map(inc);                // [2,3,4]
```

This is how configuration actually works in functional code: `makeLogger(level)(message)` fixes the level once and logs many times. It is the functional form of dependency injection, the way to build a family of endpoints or handlers from one template, and the idea behind JAX's `partial(jit, static_argnums=...)`. Together, closures and currying replace most "configuration" and "injection" machinery.

### Higher-order functions and composition

A **higher-order function** takes or returns functions. With `compose` and `pipe` you build pipelines out of small steps:

```javascript
const pipe = (...fns) => x => fns.reduce((v, f) => f(v), x);
const clean = pipe(trim, lower, dedupe);   // a data pipeline as data
```

The same shape appears in `map`, `filter`, and `reduce`, in transducers, in Spark and Pandas chains, in middleware stacks, and in the whole `jit(vmap(grad(f)))` composition of [JAX](../01-training-and-frameworks/).

### Purity and immutability

A **pure function**'s output depends only on its inputs, and it mutates nothing. That makes it trivially testable, cacheable (memoizable), and safe to parallelize. Purity is why JAX requires pure functions (so XLA can transform them), why Spark and MapReduce can distribute work, why React renders are pure, and why event sourcing works (see [Distributed Data & Caching](../04-distributed-data-and-caching/)). **Memoization** is just caching a pure function: the same bargain as [caching](../04-distributed-data-and-caching/), at function granularity. Purity and composition reduce the surface for bugs, which is exactly why JAX, Spark, and React lean on them.

## Composition over inheritance

Both halves of this chapter push the same lesson: **build big behavior from small, replaceable pieces with clear contracts.** Decorators compose objects; `pipe` composes functions; the worker pattern composes tasks; microservices compose systems. Deep inheritance hierarchies and god-objects are the anti-pattern, because they couple things that should change independently. When in doubt, prefer a small function or a wrapper over a new layer of class hierarchy.

## Where this leads

These patterns recur through the rest of the book. Pure functions are what make the transforms in [Training & Frameworks](../01-training-and-frameworks/) possible, which is why JAX is functional. Decorators wrap durable steps in [Durable Orchestration & Workers](../05-durable-orchestration-and-workers/), in the `@workflow` and `@activity` APIs. Interceptors, stubs, and the proxy are gRPC middleware and client stubs in [RPC & Protocols](../02-rpc-and-protocols/). Iterators and generators, with backpressure, deliver lazy token sequences in [Streaming & SSE](../03-streaming-sse/). Memoization as caching a pure function is the same bargain, at function level, as [Distributed Data & Caching](../04-distributed-data-and-caching/).

Industry leans on the same shapes. Google and Meta use interceptors and decorators for cross-cutting concerns (gRPC interceptors, framework middleware). Temporal and DBOS define durable units with `@workflow`, `@activity`, and `@step` decorators. Jane Street relies on functional purity and composition in OCaml, using immutability for correctness. React's function components and hooks are pure functions plus closures. Databricks builds on higher-order functions and lazy composition in Spark transformation chains. And any SDK team ships a facade over a complex backend: `client.generate(...)`, `pipeline(...)`.

## Further reading

- [Refactoring Guru: Design Patterns](https://refactoring.guru/design-patterns) (free).
- [*Mostly Adequate Guide to Functional Programming*](https://mostly-adequate.gitbook.io/mostly-adequate-guide/) (free).
- Gamma, Helm, Johnson, and Vlissides, *Design Patterns* (the Gang of Four book).
- [SourceMaking](https://sourcemaking.com/design_patterns).
- Abelson and Sussman, [*SICP*](https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/6515/sicp.zip/index.html), on closures and higher-order procedures (free).
- [PEP 318, Python decorators](https://peps.python.org/pep-0318/).

## Key terms

- **Design pattern**: a named solution to a recurring design problem, recognized rather than copied.
- **Decorator**: a wrapper with the same interface as the thing it wraps, adding behavior before or after delegating.
- **Facade**: a single, intention-revealing interface over a complex subsystem.
- **Closure**: a function together with the environment it captured.
- **Currying**: rewriting a many-argument function as a chain of one-argument functions.
- **Partial application**: fixing some of a function's arguments to produce a specialized function.
- **Pure function**: a function whose output depends only on its inputs and which mutates nothing.

## Summary

- Patterns are the vocabulary this book is written in; their value is recognition and a checklist of trade-offs, not code to copy.
- Decorator means wrap, don't modify: cross-cutting concerns such as retry, cache, auth, and tracing belong in wrappers, not in business logic.
- Facade means one door, many rooms: expose intent and hide coordination so internals stay free to change.
- A closure is state captured in a function, and currying is specialization by fixing arguments; together they replace most configuration and injection machinery.
- Compose small, pure pieces: purity buys testability, memoization, and parallelism, the reason JAX, Spark, and event sourcing work.
- A pattern is a recognition tool, not a mandate; over-applying patterns is its own smell.

## Review questions

1. In the `@retry(3)` over `@cache` stack, in what order do the layers run on a call, and what would change if you swapped them?
   <details><summary>Answer</summary>Decorators apply bottom-up, so `cache` wraps `fetch` and `retry` wraps the cached function. A call enters the retry loop, then checks the cache, then calls `fetch` on a miss. Swapped, the cache would wrap the retrying function: a hit would skip the retry loop entirely, and a miss would run all the retries inside one cache lookup. With a cache that stores only successes, the observable results are similar here; the order matters more for concerns like timing, auth, or a cache that also stores failures.</details>
2. Why is the facade said to let internals "change freely"?
   <details><summary>Answer</summary>Callers depend only on the facade's interface, such as `client.generate(prompt)`, not on the tokenizer, scheduler, KV cache, or sampler behind it. Any of those can be replaced or reorganized without touching callers, as long as the facade's contract holds.</details>
3. How does currying act as a functional form of dependency injection?
   <details><summary>Answer</summary>Fixing the first arguments of a curried function supplies its dependencies or configuration once (`makeLogger(level)`), returning a specialized function that callers use without knowing those dependencies. That is what an injector does for an object, done with plain functions and closures.</details>
4. Why does JAX require pure functions?
   <details><summary>Answer</summary>Its transforms (`jit`, `vmap`, `grad`) and XLA compilation need a function whose output depends only on its inputs, so it can be traced, compiled, vectorized, and differentiated without hidden state changing the result.</details>
5. When is applying a pattern a mistake?
   <details><summary>Answer</summary>When a plain function would do. Adding a factory, strategy hierarchy, or singleton where there is no recurring design problem adds indirection without absorbing any change; over-engineering is its own anti-pattern.</details>
