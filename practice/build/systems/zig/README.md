# Zig Practice

> **Optional language.** Zig is not one of the languages this
> curriculum targets, so these exercises have no reference implementations
> yet and are excluded from `ol verify build` and from CI. They are kept
> because the exercises are worth doing, and every `ol` command works on
> them by name.
>
> Toolchain: `brew install zig`. On a chezmoi-managed machine it comes from the
> `install_alt_langs` flag, which is off by default.
>
> Zig 0.16 removed `async`/`await` from the language, so exercise 06 as listed below needs redefining before it can be written.


Master comptime, explicit allocation, and zero-hidden-control-flow.

## Language Essentials

| Concept | Details |
|---------|---------|
| **Memory Model** | Explicit allocators passed as params, no hidden allocations, `defer` cleanup |
| **Type System** | Strong static, comptime generics, optional types, error unions |
| **Error Handling** | Error unions (`!`), `try`, `catch`, error traces |
| **Concurrency** | `async`/`await` (stackless), `std.Thread`, no runtime |
| **Build** | `zig build`, build.zig (Zig-based build system), C interop out of box |
| **Standard** | Zig 0.13+ (pre-1.0 but stable API subset) |

## Key Idioms

- **Allocator parameter** on every function that allocates
- **`comptime`** for generics and compile-time code generation
- **`defer`/`errdefer`** for cleanup (like Go but with error awareness)
- **Sentinel-terminated slices** (`[:0]u8` for C strings)
- **`@import("std")`** -- single namespace, explicit imports
- **Tagged unions** for sum types
- **No hidden control flow** -- no operator overloading, no hidden allocations

## Reference Documentation

- [Zig Language Reference](https://ziglang.org/documentation/master/) -- free, official
- [ziglearn.org](https://ziglearn.org/) -- free, tutorial-style
- [Zig Standard Library Source](https://github.com/ziglang/zig/tree/master/lib/std) -- read the source
- [Zig News](https://zig.news/) -- community articles
- Andrew Kelley's talks -- design philosophy

## Practising these

```bash
practice/bin/ol list  build zig          # which of these have references yet
practice/bin/ol start build zig 01       # clone it stubbed into .scratchpad/
practice/bin/ol check build zig 01       # run it; exit code is the verdict
practice/bin/ol diff  build zig 01       # your attempt against the reference
```

Exercises are ordered by concept, not by difficulty, so pick by the stars rather
than by the number. An exercise listed below with no reference yet is a row in
this table and nothing more; see the [practice path](../../../README.md) for how the
harness works.

## Exercises

| # | Exercise | Concepts | Difficulty |
|---|----------|----------|------------|
| 01 | ArrayList (generic) | Comptime generics, allocator interface | ⭐⭐ |
| 03 | Memory arena | Allocator interface, page allocation, alignment | ⭐⭐⭐ |
| 04 | JSON parser | Tagged unions, recursive descent, error unions | ⭐⭐⭐⭐ |
| 05 | Compile-time regex | Comptime string processing, state machine generation | ⭐⭐⭐⭐⭐ |
| 06 | Async file server | Async frames, I/O ring, event loop | ⭐⭐⭐⭐ |
| 07 | C library wrapper | `@cImport`, sentinel slices, translate-c | ⭐⭐⭐ |
| 08 | SIMD vector math | `@Vector`, platform intrinsics, benchmarking | ⭐⭐⭐⭐ |
| 09 | Build system plugin | build.zig, custom steps, dependency graph | ⭐⭐⭐ |
| 10 | Toy ELF linker | Binary formats, memory layout, relocations | ⭐⭐⭐⭐⭐ |

Exercise 02 (hash map) is consolidated into the course modules `ds.02` and `ds.05` ([Systems Data Structures](../../../../content/tracks/algorithms/16-systems-data-structures/)); the row had no reference.
