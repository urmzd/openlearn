# Java

Java earns its place in a book about building systems because the JVM underneath it is one of the most heavily engineered runtimes in existence. Modern Java is not the language of a decade ago: records give you immutable data in one line, sealed interfaces and pattern-matching `switch` give you algebraic data types, and virtual threads make blocking I/O cheap enough that you can write one thread per request again. The exercises in this chapter push you into the JVM's internals (generics by erasure, garbage collection, the class-file format) and into its modern features at the same time.

> **Note:** Java is an optional language here. None of the core books target it, so these exercises have no worked solutions yet and nothing in this chapter is built in CI. The briefs are kept because they are worth reading as design problems. If you want a toolchain, install it with `mise` and pin it in a `.mise.toml`; on a chezmoi-managed machine it comes from the `install_alt_langs` flag, which is off by default. On macOS, `/usr/bin/java` is a stub that exists and only prints "Unable to locate a Java Runtime", so check for a real JDK with `javac -version`, not `command -v javac`.

> **Learning objectives**
> - Describe the JVM's memory model, Java's type system and its use of type erasure, and the language's error-handling and concurrency models.
> - Apply modern Java idioms (records, sealed interfaces with pattern matching, virtual threads, try-with-resources) when designing a small system.
> - Explain what type erasure prevents you from doing at run time, and how a generic container works around it.
> - Reason through the design of each exercise brief before writing code.

## Before you begin

You should be able to write, compile, and run a small Java program and read generic signatures such as `<T extends Comparable<? super T>>`. The concurrency briefs assume you know what a lock and a thread pool are; the JVM-internals briefs assume you have met a stack machine and a mark-sweep collector, at least by name.

## The language in brief

Memory is managed by the JVM's garbage collector (G1 by default, ZGC for low pause times). Objects live on the heap and locals on the stack, and escape analysis lets the JIT keep short-lived objects off the heap entirely. Value types are in preview under Project Valhalla.

The type system is strong and static. Generics are implemented by erasure: `List<String>` and `List<Integer>` are the same class at run time, which is why you cannot write `new T[]`. Sealed classes restrict which types may extend a hierarchy, and records declare immutable data carriers with generated accessors, `equals`, and `hashCode`.

Errors are exceptions, split into checked exceptions (which a method must declare) and unchecked ones. `Optional` expresses "maybe absent" without `null`, and try-with-resources closes any `AutoCloseable` deterministically.

Concurrency spans the full range: virtual threads (Java 21 and later) for I/O-bound work, `CompletableFuture` for asynchronous composition, the `java.util.concurrent` toolkit of locks, queues, and executors, and structured concurrency for scoping a group of tasks to one block. Builds use Gradle or Maven, and `jshell` gives you a REPL. The exercises target Java 21 LTS or later.

## Idioms to practise

- **Records** for immutable data-transfer objects.
- **Sealed interfaces with pattern matching** in `switch` expressions, so the compiler checks that every case is handled.
- **Virtual threads** instead of thread pools for I/O-bound work.
- **The `Stream` API** for functional processing of collections.
- **Try-with-resources** for cleaning up anything `AutoCloseable`.
- **`Optional`** instead of returning `null`.
- **`var`** for local variable type inference where the type is obvious from the right-hand side.

## In this chapter

The briefs are listed below, ordered by concept rather than by difficulty, so choose by the difficulty rating (1 is easiest, 5 hardest) rather than by number.

| # | Exercise | Concepts | Difficulty (1-5) |
|---|----------|----------|------------------|
| 01 | `ArrayList<T>` from scratch | Generics, type erasure, `Iterable` | 2 |
| 03 | Virtual thread HTTP server | Virtual threads, `ServerSocket`, structured concurrency | 3 |
| 04 | Dependency injection (mini-Spring) | Reflection, annotations, `Proxy` | 4 |
| 05 | Stream collector library | `Collector` interface, `Spliterator`, parallel streams | 3 |
| 06 | Event sourcing framework | Records, sealed interfaces, pattern matching | 4 |
| 07 | Garbage collector simulator | Memory model, reference types, mark-sweep | 4 |
| 08 | Bytecode interpreter (subset) | ClassFile format, operand stack, JVM internals | 5 |
| 09 | Reactive streams publisher | `Flow` API, backpressure, `SubmissionPublisher` | 4 |
| 10 | Annotation processor | `javax.annotation.processing`, code generation | 4 |

There is no exercise 02. It was a concurrent hash map, and that problem is now covered in full by [A Swiss table](../15-systems-data-structures/02-swiss-table.md) and [A Robin Hood hash map](../15-systems-data-structures/05-robin-hood-hash-map.md) in the [Systems data structures](../15-systems-data-structures/) chapter of *Algorithms*.

## Key terms

- **Virtual thread**: a lightweight Java thread (Java 21 and later) that the JVM schedules onto a few platform threads, cheap enough that blocking I/O code can use one thread per request.
- **Java record**: a class declared as `record Point(int x, int y)`, an immutable data carrier for which the compiler generates the constructor, accessors, `equals`, and `hashCode`.
- **Sealed class / sealed interface**: a Java class or interface that lists the only types allowed to extend it, so a pattern-matching `switch` over it can be checked for exhaustiveness; with records, Java's way to write algebraic data types.
- **Checked exception**: a Java exception that a method must declare in its `throws` clause and that callers must catch or declare in turn, unlike unchecked exceptions, which need neither.
- **Try-with-resources**: a Java `try` form that closes every `AutoCloseable` resource it declares when the block exits, normally or by exception, in reverse order of opening.
- **Structured concurrency**: scoping a group of concurrent tasks to one block, so the block does not exit until all of them have finished and a failure in one cancels the rest.

## Further reading

- [Java Language Specification](https://docs.oracle.com/javase/specs/): the authoritative definition.
- [Java SE API documentation](https://docs.oracle.com/en/java/javase/21/docs/api/): the standard library.
- [Dev.java](https://dev.java/): the official tutorials.
- *Effective Java* by Joshua Bloch, 3rd edition: the essential patterns.
- *Java Concurrency in Practice* by Brian Goetz: the standard reference on concurrency.
