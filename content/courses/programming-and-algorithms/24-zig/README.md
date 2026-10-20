# Zig

Zig is a systems language built around one promise: no hidden control flow. There is no operator overloading, no hidden allocation, no exceptions unwinding the stack behind your back. Every function that allocates takes an allocator as a parameter, so you can see the cost at the call site and swap the strategy (arena, pool, testing allocator that detects leaks) without changing the code. And `comptime` lets ordinary Zig run at compile time, which replaces templates, macros, and generics with one mechanism. The exercises in this chapter put both ideas to work.

> **Note:** Zig is an optional language here. None of the core books target it, so these exercises have no worked solutions yet and nothing in this chapter is built in CI. The briefs are kept because they are worth reading as design problems. To get a toolchain, run `brew install zig`; on a chezmoi-managed machine it comes from the `install_alt_langs` flag, which is off by default. Zig 0.16 removed `async`/`await` from the language, so exercise 06 as listed below needs redefining before it can be written.

> **Learning objectives**
> - Describe Zig's explicit-allocator memory model, its error unions, and its `comptime` approach to generics.
> - Explain why passing an allocator to every allocating function makes memory cost visible and testable.
> - Use `defer` and `errdefer` to release resources correctly on both the success and the error path.
> - Reason through the design of each exercise brief before writing code.

## Before you begin

You should be comfortable with manual memory management at the level of the [C](../18-c/README.md) chapter: what an allocation is, what alignment means, and why a pointer into a buffer can dangle. Familiarity with C helps throughout, since Zig interoperates with it directly.

## The language in brief

Zig has no garbage collector and no global allocator. Allocators are values passed as parameters, nothing allocates behind your back, and `defer` runs cleanup when a scope exits.

The type system is strong and static. Generics are ordinary functions evaluated at compile time that take and return types, which is what `comptime` means. Optional types (`?T`) replace null pointers, and error unions (`E!T`) carry either an error or a value.

Errors are handled with `try` (propagate) and `catch` (handle), and the runtime records an error return trace showing where an error came from and every frame it passed through. `errdefer` runs cleanup only when the scope exits with an error, which is the clean answer to partial-construction failures. For concurrency there is `std.Thread`; the language has no runtime. Earlier versions had stackless `async`/`await`, but it was removed in 0.16. Builds are described in `build.zig`, a Zig program, and C interoperability works out of the box. The exercises were written against Zig 0.13 and later, which is pre-1.0 but has a stable core.

## Idioms to practise

- **An allocator parameter** on every function that allocates.
- **`comptime`** for generics and compile-time code generation.
- **`defer` and `errdefer`** for cleanup, like Go's `defer` but aware of the error path.
- **Sentinel-terminated slices**, such as `[:0]u8` for C strings.
- **`@import("std")`**: one namespace, with every import explicit.
- **Tagged unions** for sum types.
- **No hidden control flow**: no operator overloading and no hidden allocations.

## In this chapter

The briefs are listed below, ordered by concept rather than by difficulty, so choose by the difficulty rating (1 is easiest, 5 hardest) rather than by number.

| # | Exercise | Concepts | Difficulty (1-5) |
|---|----------|----------|------------------|
| 01 | ArrayList (generic) | Comptime generics, allocator interface | 2 |
| 03 | Memory arena | Allocator interface, page allocation, alignment | 3 |
| 04 | JSON parser | Tagged unions, recursive descent, error unions | 4 |
| 05 | Compile-time regex | Comptime string processing, state machine generation | 5 |
| 06 | Async file server | Async frames, I/O ring, event loop | 4 |
| 07 | C library wrapper | `@cImport`, sentinel slices, translate-c | 3 |
| 08 | SIMD vector math | `@Vector`, platform intrinsics, benchmarking | 4 |
| 09 | Build system plugin | build.zig, custom steps, dependency graph | 3 |
| 10 | Toy ELF linker | Binary formats, memory layout, relocations | 5 |

There is no exercise 02. It was a hash map, and that problem is now covered in full by [A Swiss table](../15-systems-data-structures/02-swiss-table.md) and [A Robin Hood hash map](../15-systems-data-structures/05-robin-hood-hash-map.md) in the [Systems data structures](../15-systems-data-structures/) chapter of *Algorithms*. For exercise 03, the C [arena allocator](../18-c/08-arena-allocator/README.md) is a worked version of the same idea.

## Key terms

- **Explicit allocator**: the Zig convention that every function that allocates takes an allocator as a parameter, so memory cost is visible at the call site and the strategy (arena, pool, leak-detecting test allocator) can change without changing the code.
- **Comptime / compile-time execution**: Zig's mechanism for running ordinary Zig code during compilation; generics are functions that take and return types at compile time, replacing templates, macros, and separate generics syntax.
- **Error union**: a Zig type `E!T` that holds either an error from the set `E` or a value of type `T`; `try` propagates the error and `catch` handles it.
- **errdefer**: a Zig statement that schedules cleanup to run only if the enclosing scope exits with an error, which undoes a partial construction without touching the success path.
- **Error return trace**: a record, kept by the Zig runtime, of where an error was created and every frame it was returned through, so a failure can be traced without exceptions or stack unwinding.
- **No hidden control flow**: Zig's design rule that code does only what it visibly says: no operator overloading, no hidden allocations, and no exceptions, so every call, allocation, and early return appears in the source.

## Further reading

- [Zig Language Reference](https://ziglang.org/documentation/master/): free and official.
- [ziglearn.org](https://ziglearn.org/): free, tutorial-style.
- [Zig standard library source](https://github.com/ziglang/zig/tree/master/lib/std): read the source; it is the best documentation of idiomatic Zig.
- [Zig News](https://zig.news/): community articles.
- Andrew Kelley's talks: the design philosophy behind the language.
