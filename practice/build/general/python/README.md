# Python Practice

Master the data model, generators, decorators, and asyncio.

## Language Essentials

| Concept | Details |
|---------|---------|
| **Memory Model** | Reference counting + cycle GC, everything is an object, `__slots__` |
| **Type System** | Dynamic, gradual typing (`typing` module), protocols, `TypeVar` |
| **Error Handling** | Exceptions (`try`/`except`/`else`/`finally`), `ExceptionGroup` (3.11+) |
| **Concurrency** | `asyncio`, `threading` (GIL), `multiprocessing`, free-threading (3.13+) |
| **Build** | `uv`, `pip`, `pyproject.toml`, `ruff` for linting |
| **Standard** | CPython 3.12+ |

## Key Idioms

- **Dunder methods** (`__init__`, `__repr__`, `__iter__`) for protocol conformance
- **Generators** and `yield from` for lazy sequences
- **Decorators** (function and class) for cross-cutting concerns
- **Context managers** (`with` statement, `__enter__`/`__exit__`)
- **Comprehensions** (list, dict, set, generator) over explicit loops
- **`dataclasses`** and `attrs` for structured data
- **Descriptors** for attribute access control
- **`match` statement** (3.10+) for structural pattern matching

## Reference Documentation

- [Python docs](https://docs.python.org/3/) -- authoritative
- [Data Model reference](https://docs.python.org/3/reference/datamodel.html) -- essential
- *Fluent Python* (Ramalho, 2nd ed.) -- deep Pythonic patterns
- *Python Cookbook* (Beazley & Jones) -- recipes
- [Real Python](https://realpython.com/) -- tutorials

## How this page is organised

The table below is an index of exercise briefs: what to build, the concepts it
exercises, and how hard it is. Exercises are ordered by concept, not by
difficulty. Where a row names a course version (for example `ds.02`), that
module's chapter walks through a full solution of the same problem.

## Exercises

Worked solutions for these exercises are not written yet.

| # | Exercise | Concepts | Difficulty |
|---|----------|----------|------------|
| 01 | Descriptor-based ORM | Descriptors, metaclasses, `__set_name__` | ⭐⭐⭐⭐ |
| 02 | Async task scheduler | `asyncio`, event loop, `Future`, cancellation; course version: `lang.08`, `data.01` | ⭐⭐⭐ |
| 03 | Generator-based pipeline | `yield`, `send`, `throw`, coroutine chaining; course version: `data.02` | ⭐⭐⭐ |
| 04 | Property-based test framework | Decorators, `inspect`, random generation, shrinking; course version: `craft.04` | ⭐⭐⭐⭐ |
| 05 | LRU cache with TTL | `__hash__`, `__eq__`, doubly-linked list, `threading.Lock`; course version: `gw.06` | ⭐⭐⭐ |
| 06 | Import hook (custom loader) | `importlib`, `sys.meta_path`, `ModuleSpec` | ⭐⭐⭐⭐⭐ |
| 07 | Dataclass code generator | `ast` module, metaprogramming, `exec` | ⭐⭐⭐⭐ |
| 08 | Actor model (multiprocessing) | `multiprocessing.Queue`, `Process`, message passing; course version: `data.04` | ⭐⭐⭐ |
| 09 | Type checker (subset) | `typing`, AST walking, unification algorithm | ⭐⭐⭐⭐⭐ |
| 10 | WSGI framework (mini-Flask) | WSGI spec, decorators, routing, middleware | ⭐⭐⭐ |
