# Scala

Scala is the place to practise functional programming with a type system strong enough to encode real constraints. On the JVM it inherits garbage collection and a vast library ecosystem; on top of that it adds immutable case classes, exhaustive pattern matching, higher-kinded types, and (in Scala 3) given instances for type-class derivation. The exercises in this chapter use those features for what they are good at: making illegal states unrepresentable, composing effects, and deriving code at compile time.

> **Note:** Scala is an optional language here. None of the core books target it, so these exercises have no worked solutions yet and nothing in this chapter is built in CI. The briefs are kept because they are worth reading as design problems. To get a toolchain, run `brew install scala-cli`; on a chezmoi-managed machine it comes from the `install_alt_langs` flag, which is off by default. `scala-cli` runs a bare directory of sources and manages its own JVM, so no build definition is needed.

> **Learning objectives**
> - Describe Scala's type system (variance, higher-kinded and path-dependent types) and its value-based approach to errors.
> - Model data as algebraic data types with sealed traits and case classes, and process it with exhaustive pattern matching.
> - Compose effectful computations with for-comprehensions, and explain what `flatMap` must satisfy for that to work.
> - Reason through the design of each exercise brief before writing code.

## Before you begin

You should be able to write and run a small Scala 3 program and be comfortable with recursion and immutable data. The harder briefs (free monads, an effect system) assume you have used `map` and `flatMap` on `Option` or `Future` and want to know what is underneath.

## The language in brief

Scala runs on the JVM, so memory is garbage collected. Case classes are immutable by default, and value classes let a wrapper type cost nothing at run time.

The type system is strong and static, with declaration-site variance (`+A`, `-A`), higher-kinded types (types that take type constructors, such as `F[_]`), and path-dependent types (types that belong to a particular value). It is expressive enough to encode a state machine in types, which is what the phantom-types brief asks you to do.

Errors are values: `Option` for absence, `Either` for a failure with a reason, `Try` for wrapping code that throws, and recovery combinators on `Future`. There are no checked exceptions. Concurrency comes from `Future` and `Promise` in the standard library and from ecosystems built on top: Akka actors, cats-effect `IO`, and ZIO. Builds use sbt, `scala-cli`, or Mill. The exercises target Scala 3.

## Idioms to practise

- **Case classes** for immutable data.
- **Pattern matching** with exhaustiveness checking, so the compiler warns about a missed case.
- **For-comprehensions** for monadic composition.
- **Given instances** (Scala 3's replacement for implicits) for type-class derivation.
- **Sealed traits** for algebraic data types.
- **Extension methods** (Scala 3) instead of implicit classes.
- **Opaque types** for zero-cost newtype wrappers.

## In this chapter

The briefs are listed below, ordered by concept rather than by difficulty, so choose by the difficulty rating (1 is easiest, 5 hardest) rather than by number.

| # | Exercise | Concepts | Difficulty (1-5) |
|---|----------|----------|------------------|
| 01 | Immutable red-black tree | Case classes, pattern matching, recursion | 3 |
| 02 | Monad implementation | Higher-kinded types, `flatMap`, for-comprehension | 4 |
| 03 | Actor system (simplified) | Message passing, mailboxes, supervision | 4 |
| 04 | Type-safe builder (phantom types) | Phantom types, type-level state machine | 4 |
| 05 | Streaming CSV parser | `Iterator`, lazy evaluation, `fs2`-style | 3 |
| 06 | JSON codec (automatic derivation) | Given instances, `Mirror`, compile-time derivation | 5 |
| 07 | Parallel collection processor | `Future`, `ExecutionContext`, `Par` | 3 |
| 08 | Free monad interpreter | GADTs, natural transformations, Church encoding | 5 |
| 09 | Property-based test framework | Generators, shrinking, `given` Arbitrary | 4 |
| 10 | Effect system (mini-ZIO) | Trampolining, fibers, error channels | 5 |

## Key terms

- **Case class**: an immutable Scala class whose constructor parameters become fields and for which the compiler generates structural equality, copying, and pattern-matching support; Scala's standard way to model data.
- **Sealed trait**: a Scala trait whose subtypes must all be defined in the same file, so the compiler knows every case and checks pattern matches for exhaustiveness; with case classes it encodes algebraic data types.
- **Higher-kinded type**: a type parameterized by a type constructor rather than a type, such as `F[_]`, which lets one piece of code abstract over containers like `List`, `Option`, and `Future`.
- **Path-dependent type**: a type that belongs to a particular value, written `x.T`, so the member types of two different instances are distinct and cannot be mixed.
- **For-comprehension**: Scala syntax that desugars a sequence of generators into nested `flatMap` calls and a final `map`, giving sequential-looking code for `Option`, `Future`, and other monads.
- **Given instance**: a Scala 3 value declared with `given` that the compiler supplies automatically wherever a parameter of its type is required; it replaces implicits for type-class instances and derivation.
- **Opaque type**: a Scala 3 type alias whose underlying representation is visible only where it is defined, giving a distinct type with no run-time cost; a newtype without a wrapper.

## Further reading

- [Scala 3 Book](https://docs.scala-lang.org/scala3/book/introduction.html): free and official.
- [Scala Standard Library](https://www.scala-lang.org/api/current/): the API documentation.
- *Functional Programming in Scala* by Chiusano and Bjarnason: "the red book".
- [Scala Exercises](https://www.scala-exercises.org/): free and interactive.
- [Typelevel ecosystem](https://typelevel.org/): cats, fs2, and http4s.
