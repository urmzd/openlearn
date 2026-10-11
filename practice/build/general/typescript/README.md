# TypeScript Practice

Master the type system, generics, mapped types, and runtime safety patterns.

## Language Essentials

| Concept | Details |
|---------|---------|
| **Memory Model** | V8 GC, prototypal inheritance, closures, WeakRef/FinalizationRegistry |
| **Type System** | Structural, gradual, mapped types, conditional types, template literals |
| **Error Handling** | `try`/`catch` (untyped), `Result<T,E>` pattern, `never` for exhaustiveness |
| **Concurrency** | Event loop, `Promise`, `async`/`await`, Web Workers, `SharedArrayBuffer` |
| **Build** | `tsc`, `tsx`, `esbuild`, `vitest` for testing |
| **Standard** | TypeScript 5.x, ES2023+ target |

## Key Idioms

- **Discriminated unions** with `kind` field for exhaustive switching
- **`satisfies`** operator for type checking without widening
- **Branded types** for nominal typing (`type USD = number & { __brand: 'USD' }`)
- **`as const`** for literal type inference
- **Mapped types** (`Partial`, `Required`, `Pick`, custom)
- **Conditional types** with `infer` for type-level computation
- **Template literal types** for string pattern enforcement
- **`using` keyword** (TC39 explicit resource management)

## Reference Documentation

- [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/) -- free, official
- [TypeScript Playground](https://www.typescriptlang.org/play) -- experiment online
- [type-challenges](https://github.com/type-challenges/type-challenges) -- free, type gymnastics
- *Programming TypeScript* (Cherny) -- comprehensive
- *Effective TypeScript* (Vanderkam, 2nd ed.) -- 83 practical items

## Practising these

```bash
practice/bin/ol list  build typescript          # which of these have references yet
practice/bin/ol start build typescript 01       # clone it stubbed into .scratchpad/
practice/bin/ol check build typescript 01       # run it; exit code is the verdict
practice/bin/ol diff  build typescript 01       # your attempt against the reference
```

Exercises are ordered by concept, not by difficulty, so pick by the stars rather
than by the number. An exercise listed below with no reference yet is a row in
this table and nothing more; see the [practice path](../../../README.md) for how the
harness works.

## Exercises

| # | Exercise | Concepts | Difficulty |
|---|----------|----------|------------|
| 01 | Type-safe event emitter | Generics, mapped types, `Parameters<T>` | ⭐⭐⭐ |
| 02 | Result/Option monads | Discriminated unions, method chaining, `never` | ⭐⭐ |
| 03 | Schema validator (like Zod) | Template literals, recursive types, `infer` | ⭐⭐⭐⭐⭐ |
| 04 | Reactive signals (like Solid) | Proxies, dependency tracking, batched updates | ⭐⭐⭐⭐ |
| 05 | Type-safe SQL query builder | Template literal types, conditional types | ⭐⭐⭐⭐ |
| 06 | Middleware pipeline (like Koa) | Generics, `async` composition, context threading | ⭐⭐⭐ |
| 07 | State machine (typed transitions) | Mapped types, phantom states, `never` transitions | ⭐⭐⭐⭐ |
| 08 | DI container (type-safe) | `Symbol` keys, `Map`, `WeakMap`, interface registry | ⭐⭐⭐ |
| 09 | Incremental type checker (subset) | AST, inference, unification, variance | ⭐⭐⭐⭐⭐ |
| 10 | Streaming JSON parser | `AsyncGenerator`, backpressure, `ReadableStream` | ⭐⭐⭐⭐ |
