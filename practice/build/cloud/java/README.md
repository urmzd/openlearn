# Java Practice

> **Optional language.** Java is not one of the languages this
> curriculum targets, so these exercises have no worked solutions
> yet and nothing here is built in CI. They are kept because the briefs
> are worth reading as design problems.
>
> Toolchain: `mise`, with a `.mise.toml` pin. On a chezmoi-managed machine it comes from the
> `install_alt_langs` flag, which is off by default.
>
> macOS ships `/usr/bin/java` as a stub that exists and only prints "Unable to locate a Java Runtime", so check with `javac -version`, not `command -v javac`.


Master JVM internals, generics, concurrency, and modern Java features.

## Language Essentials

| Concept | Details |
|---------|---------|
| **Memory Model** | JVM GC (G1/ZGC), heap/stack, escape analysis, value types (Valhalla preview) |
| **Type System** | Strong static, generics (type erasure), sealed classes, records |
| **Error Handling** | Checked + unchecked exceptions, `Optional`, try-with-resources |
| **Concurrency** | Virtual threads (21+), `CompletableFuture`, `java.util.concurrent`, structured concurrency |
| **Build** | `gradle` or `maven`, `jshell` for REPL |
| **Standard** | Java 21+ LTS |

## Key Idioms

- **Records** for immutable data transfer objects
- **Sealed interfaces + pattern matching** (switch expressions)
- **Virtual threads** over thread pools for I/O-bound work
- **`Stream` API** for functional collection processing
- **Try-with-resources** for `AutoCloseable` cleanup
- **`Optional`** instead of null returns
- **`var`** for local variable type inference

## Reference Documentation

- [Java Language Specification](https://docs.oracle.com/javase/specs/) -- authoritative
- [Java SE API Docs](https://docs.oracle.com/en/java/javase/21/docs/api/) -- standard library
- [Dev.java](https://dev.java/) -- official tutorials
- *Effective Java* (Bloch, 3rd ed.) -- essential patterns
- *Java Concurrency in Practice* (Goetz) -- concurrency bible

## How this page is organised

The table below is an index of exercise briefs: what to build, the concepts it
exercises, and how hard it is. Exercises are ordered by concept, not by
difficulty. Where a row names a course version (for example `ds.02`), that
module's chapter walks through a full solution of the same problem.

## Exercises

Worked solutions for these exercises are not written yet.

| # | Exercise | Concepts | Difficulty |
|---|----------|----------|------------|
| 01 | `ArrayList<T>` from scratch | Generics, type erasure, `Iterable` | ⭐⭐ |
| 03 | Virtual thread HTTP server | Virtual threads, `ServerSocket`, structured concurrency | ⭐⭐⭐ |
| 04 | Dependency injection (mini-Spring) | Reflection, annotations, `Proxy` | ⭐⭐⭐⭐ |
| 05 | Stream collector library | `Collector` interface, `Spliterator`, parallel streams | ⭐⭐⭐ |
| 06 | Event sourcing framework | Records, sealed interfaces, pattern matching | ⭐⭐⭐⭐ |
| 07 | Garbage collector simulator | Memory model, reference types, mark-sweep | ⭐⭐⭐⭐ |
| 08 | Bytecode interpreter (subset) | ClassFile format, operand stack, JVM internals | ⭐⭐⭐⭐⭐ |
| 09 | Reactive streams publisher | `Flow` API, backpressure, `SubmissionPublisher` | ⭐⭐⭐⭐ |
| 10 | Annotation processor | `javax.annotation.processing`, code generation | ⭐⭐⭐⭐ |

Exercise 02 (concurrent hash map) is consolidated into the course modules `ds.02` and `ds.05` ([Systems Data Structures](../../../../content/courses/algorithms/16-systems-data-structures/)); the row had no reference.
