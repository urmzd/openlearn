# Python

Python looks simple, and most of the time it is. Underneath the simplicity sits a data model that is unusually open: every operator, attribute lookup, iteration, and `with` block is a call to a dunder method you can define yourself. The exercises in this chapter go below the surface on purpose. They make you build the machinery that frameworks hide (descriptors behind an ORM, generators behind a pipeline, the event loop behind `asyncio`, the import system behind every `import`) so that the next time a library does something surprising, you know which hook it used.

The chapter is a set of exercise briefs rather than worked solutions. Several briefs are the same problem a section elsewhere in the books solves in full, and for those the brief links to that section so you can reason through your own design first and then compare it with a complete, tested one.

> **Learning objectives**
> - Describe CPython's memory model, Python's gradual type system, its exception model, and the tradeoffs between `asyncio`, threads, and processes.
> - Use the data model (dunder methods, descriptors, context managers) to make your own types behave like built-ins.
> - Build lazy pipelines with generators and explain what `send` and `throw` add to plain iteration.
> - Reason through the design of each exercise before reading the matching worked solution elsewhere in the books.

## Before you begin

You should be comfortable writing Python classes, functions, and tests, and running a project with `uv`. If you have not used `asyncio` before, read [Python asyncio: event loop, tasks, cancellation, bounded concurrency](../../../../content/courses/programming-languages/02-language-and-tool-primers/08-python-asyncio.md) in *Programming Languages* before the scheduler brief.

## The language in brief

CPython manages memory by reference counting, backed by a cycle-detecting garbage collector for structures that point at themselves. Everything is an object, including integers, functions, and classes, which is what makes the data model so uniform. `__slots__` trades the per-instance `__dict__` for a fixed layout when memory matters.

Typing is dynamic, with optional gradual static typing through the `typing` module: protocols for structural interfaces and `TypeVar` for generics. The interpreter ignores annotations; checkers such as mypy and pyright enforce them.

Errors are exceptions, handled with `try`, `except`, `else`, and `finally`. Since 3.11, `ExceptionGroup` lets concurrent code raise several failures at once. For concurrency you choose between `asyncio` for I/O-bound work on one thread, `threading` (serialized by the global interpreter lock for Python bytecode), `multiprocessing` for CPU-bound work, and, from 3.13, an experimental free-threaded build without the GIL. Projects are managed with `uv` or `pip` and `pyproject.toml`, and linted with `ruff`. The exercises target CPython 3.12 or later.

## Idioms to practise

- **Dunder methods** (`__init__`, `__repr__`, `__iter__`) so your types conform to Python's protocols.
- **Generators and `yield from`** for lazy sequences.
- **Decorators**, on functions and classes, for cross-cutting concerns.
- **Context managers** (the `with` statement, `__enter__` and `__exit__`) for scoped resources.
- **Comprehensions** (list, dict, set, generator) instead of explicit accumulation loops.
- **`dataclasses` and `attrs`** for structured data.
- **Descriptors** for controlling attribute access.
- **The `match` statement** (3.10 and later) for structural pattern matching.

## In this chapter

The briefs are listed below, ordered by concept rather than by difficulty, so choose by the difficulty rating (1 is easiest, 5 hardest) rather than by number. Worked Python solutions for these exercises have not been written yet; where a full solution of the same problem exists elsewhere in the books, the last column links to it.

| # | Exercise | Concepts | Difficulty (1-5) | Worked elsewhere |
|---|----------|----------|------------------|------------------|
| 01 | Descriptor-based ORM | Descriptors, metaclasses, `__set_name__` | 4 | |
| 02 | Async task scheduler | `asyncio`, event loop, `Future`, cancellation | 3 | [Python asyncio](../../../../content/courses/programming-languages/02-language-and-tool-primers/08-python-asyncio.md); [Async fetch with resume, checksums, and license capture](../../../../content/courses/data-engineering/05-corpus-pipeline/01-async-fetch-resume-checksums.md) |
| 03 | Generator-based pipeline | `yield`, `send`, `throw`, coroutine chaining | 3 | [Extract, normalize, and quality filters](../../../../content/courses/data-engineering/05-corpus-pipeline/02-extract-normalize-and-quality-filters.md) |
| 04 | Property-based test framework | Decorators, `inspect`, random generation, shrinking | 4 | [Property-based tests with Hypothesis, proptest, rapid, and ol_prop.h](../../../../content/courses/software-craftsmanship/03-testing-mentality/02-property-based-tests.md) |
| 05 | LRU cache with TTL | `__hash__`, `__eq__`, doubly-linked list, `threading.Lock` | 3 | [Response cache: LRU, TTL, singleflight, and tenant-scoped keys](../../../../content/courses/ai-platform-engineering/04-distributed-data-and-caching/01-response-cache.md) |
| 06 | Import hook (custom loader) | `importlib`, `sys.meta_path`, `ModuleSpec` | 5 | |
| 07 | Dataclass code generator | `ast` module, metaprogramming, `exec` | 4 | |
| 08 | Actor model (multiprocessing) | `multiprocessing.Queue`, `Process`, message passing | 3 | [Near-duplicate dedup: MinHash, LSH, union-find, and decontamination](../../../../content/courses/data-engineering/05-corpus-pipeline/04-near-dedup-minhash-lsh.md) |
| 09 | Type checker (subset) | `typing`, AST walking, unification algorithm | 5 | |
| 10 | WSGI framework (mini-Flask) | WSGI spec, decorators, routing, middleware | 3 | |

The linked sections come from *Programming Languages*, *Data Engineering*, *Software Craftsmanship*, and *AI Platform Engineering*, where each problem is solved as part of a larger system rather than in isolation.

## Further reading

- [Python documentation](https://docs.python.org/3/): the authoritative reference.
- [Data model reference](https://docs.python.org/3/reference/datamodel.html): essential for every brief in this chapter.
- *Fluent Python* by Luciano Ramalho, 2nd edition: deep Pythonic patterns.
- *Python Cookbook* by Beazley and Jones: recipes.
- [Real Python](https://realpython.com/): tutorials.
