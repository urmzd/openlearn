# TypeScript

TypeScript is two languages stacked on top of each other. At run time it is JavaScript: a garbage-collected, prototype-based language with a single-threaded event loop. At compile time it is a structural type system expressive enough to compute with, through mapped types, conditional types, and template literal types. The exercises in this chapter work both layers. Some ask you to build run-time machinery (a signal graph, a middleware pipeline, a streaming parser); others ask you to make the type checker reject wrong programs that would otherwise run, which is where TypeScript earns its keep.

> **Learning objectives**
> - Describe the JavaScript runtime model (garbage collection, prototypes, closures, the event loop) that every TypeScript program runs on.
> - Use discriminated unions, `never`, and `satisfies` to make the compiler check that every case is handled.
> - Compute types with mapped, conditional, and template literal types, and explain when that is worth the complexity.
> - Reason through the design of each exercise brief before writing code.

## Before you begin

You should be comfortable writing modern JavaScript, including `async`/`await` and modules, and reading basic generic signatures such as `function first<T>(xs: T[]): T | undefined`. The type-level briefs (the schema validator, the SQL builder, the type checker) assume you have written a conditional type with `infer` at least once.

## The language in brief

At run time, memory belongs to the engine's garbage collector (V8 in Node and Chrome). Objects inherit through prototypes, closures capture their environment, and `WeakRef` and `FinalizationRegistry` give you limited hooks into collection.

The type system is structural and gradual: types are compatible when their shapes match, and `any` opts out entirely. Mapped types transform every property of a type, conditional types choose between types, and template literal types compute string types, which together make the type level a small functional language.

Error handling is `try`/`catch`, and the caught value is untyped (`unknown`). Many codebases therefore return a `Result<T, E>` union instead of throwing, and use `never` to prove that a `switch` is exhaustive. Concurrency is the event loop: `Promise` and `async`/`await` for asynchronous work on one thread, Web Workers for parallelism, and `SharedArrayBuffer` for shared memory between workers. The toolchain is `tsc` for type checking, `tsx` and `esbuild` for running and bundling, and `vitest` for tests. The exercises target TypeScript 5.x with an ES2023 or later target.

## Idioms to practise

- **Discriminated unions** with a `kind` field, for exhaustive switching.
- **The `satisfies` operator**, to check a value against a type without widening it.
- **Branded types** for nominal typing, such as `type USD = number & { __brand: 'USD' }`.
- **`as const`** for literal type inference.
- **Mapped types**, both built-in (`Partial`, `Required`, `Pick`) and your own.
- **Conditional types with `infer`** for type-level computation.
- **Template literal types** to enforce string patterns.
- **The `using` keyword** (TC39 explicit resource management) for deterministic cleanup.

## In this chapter

The briefs are listed below, ordered by concept rather than by difficulty, so choose by the difficulty rating (1 is easiest, 5 hardest) rather than by number. Worked solutions for these exercises have not been written yet.

| # | Exercise | Concepts | Difficulty (1-5) |
|---|----------|----------|------------------|
| 01 | Type-safe event emitter | Generics, mapped types, `Parameters<T>` | 3 |
| 02 | Result/Option monads | Discriminated unions, method chaining, `never` | 2 |
| 03 | Schema validator (like Zod) | Template literals, recursive types, `infer` | 5 |
| 04 | Reactive signals (like Solid) | Proxies, dependency tracking, batched updates | 4 |
| 05 | Type-safe SQL query builder | Template literal types, conditional types | 4 |
| 06 | Middleware pipeline (like Koa) | Generics, `async` composition, context threading | 3 |
| 07 | State machine (typed transitions) | Mapped types, phantom states, `never` transitions | 4 |
| 08 | DI container (type-safe) | `Symbol` keys, `Map`, `WeakMap`, interface registry | 3 |
| 09 | Incremental type checker (subset) | AST, inference, unification, variance | 5 |
| 10 | Streaming JSON parser | `AsyncGenerator`, backpressure, `ReadableStream` | 4 |

## Key terms

- **Structural typing**: type compatibility decided by shape: a value fits a type when it has the required members, whatever it was declared as; TypeScript's types and Go's interfaces both work this way.
- **Prototypal inheritance**: JavaScript's object model, in which each object links to a prototype object and a property lookup that misses on the object walks up that chain; classes are syntax over it.
- **Discriminated union**: a union of object types that share a literal tag field such as `kind`, so checking the tag narrows the value to one member; with `never`, the compiler can verify that a `switch` handles every case.
- **Mapped type**: a type that transforms every property of another type, such as the built-in `Partial<T>`, which makes each property optional.
- **Conditional type**: a type of the form `A extends B ? X : Y` that chooses between two types, with `infer` to capture part of a matched type; it is the branching construct of type-level computation.
- **Template literal type**: a string type built by interpolating other types into a template, so the compiler can check that strings follow a pattern such as an event name or a route.
- **Branded type**: a primitive type intersected with a phantom marker, such as `number & { __brand: 'USD' }`, so values with the same representation but different meanings are not interchangeable; nominal typing inside a structural type system.

## Further reading

- [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/): free and official.
- [TypeScript Playground](https://www.typescriptlang.org/play): experiment online.
- [type-challenges](https://github.com/type-challenges/type-challenges): free type-level puzzles.
- *Programming TypeScript* by Boris Cherny: comprehensive.
- *Effective TypeScript* by Dan Vanderkam, 2nd edition: 83 practical items.
