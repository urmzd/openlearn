# Scala Practice

> **Optional language.** Scala is not one of the languages this
> curriculum targets, so these exercises have no worked solutions
> yet and nothing here is built in CI. They are kept because the briefs
> are worth reading as design problems.
>
> Toolchain: `brew install scala-cli`. On a chezmoi-managed machine it comes from the
> `install_alt_langs` flag, which is off by default.
>
> `scala-cli` runs a bare directory of sources and manages its own JVM, so no build definition is needed.


Master functional programming, the type system, and pattern matching on the JVM.

## Language Essentials

| Concept | Details |
|---------|---------|
| **Memory Model** | JVM GC, value classes, case classes (immutable by default) |
| **Type System** | Strong static, higher-kinded types, variance, path-dependent types |
| **Error Handling** | `Option`, `Either`, `Try`, `Future` recovery, no checked exceptions |
| **Concurrency** | `Future`/`Promise`, Akka actors, cats-effect IO, ZIO |
| **Build** | `sbt`, `scala-cli`, Mill |
| **Standard** | Scala 3 (Dotty) |

## Key Idioms

- **Case classes** for immutable data
- **Pattern matching** with exhaustiveness checking
- **For-comprehensions** for monadic composition
- **Implicits / given instances** (Scala 3) for type-class derivation
- **Sealed traits** for algebraic data types
- **Extension methods** (Scala 3) over implicit classes
- **Opaque types** for zero-cost newtype wrappers

## Reference Documentation

- [Scala 3 Book](https://docs.scala-lang.org/scala3/book/introduction.html) -- free, official
- [Scala Standard Library](https://www.scala-lang.org/api/current/) -- API docs
- *Functional Programming in Scala* (Chiusano & Bjarnason) -- the red book
- [Scala Exercises](https://www.scala-exercises.org/) -- free interactive
- [Typelevel ecosystem](https://typelevel.org/) -- cats, fs2, http4s

## How this page is organised

The table below is an index of exercise briefs: what to build, the concepts it
exercises, and how hard it is. Exercises are ordered by concept, not by
difficulty. Where a row names a course version (for example `ds.02`), that
module's chapter walks through a full solution of the same problem.

## Exercises

Worked solutions for these exercises are not written yet.

| # | Exercise | Concepts | Difficulty |
|---|----------|----------|------------|
| 01 | Immutable red-black tree | Case classes, pattern matching, recursion | ⭐⭐⭐ |
| 02 | Monad implementation | Higher-kinded types, `flatMap`, for-comprehension | ⭐⭐⭐⭐ |
| 03 | Actor system (simplified) | Message passing, mailboxes, supervision | ⭐⭐⭐⭐ |
| 04 | Type-safe builder (phantom types) | Phantom types, type-level state machine | ⭐⭐⭐⭐ |
| 05 | Streaming CSV parser | `Iterator`, lazy evaluation, `fs2`-style | ⭐⭐⭐ |
| 06 | JSON codec (automatic derivation) | Given instances, `Mirror`, compile-time derivation | ⭐⭐⭐⭐⭐ |
| 07 | Parallel collection processor | `Future`, `ExecutionContext`, `Par` | ⭐⭐⭐ |
| 08 | Free monad interpreter | GADTs, natural transformations, Church encoding | ⭐⭐⭐⭐⭐ |
| 09 | Property-based test framework | Generators, shrinking, `given` Arbitrary | ⭐⭐⭐⭐ |
| 10 | Effect system (mini-ZIO) | Trampolining, fibers, error channels | ⭐⭐⭐⭐⭐ |
