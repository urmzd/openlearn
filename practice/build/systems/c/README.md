# C Practice

Master manual memory management, pointer arithmetic, and undefined behavior avoidance.

## Language Essentials

| Concept | Details |
|---------|---------|
| **Memory Model** | Manual allocation (`malloc`/`free`), stack vs heap, no GC |
| **Type System** | Weak static typing, implicit conversions, `void*` generics |
| **Error Handling** | Return codes, `errno`, `setjmp`/`longjmp` (avoid) |
| **Concurrency** | pthreads, atomics (`stdatomic.h`), no language-level async |
| **Build** | `gcc`/`clang`, `make`, `cmake`, header/source separation |
| **Standard** | C11 (prefer) or C17 |

## Key Idioms

- **Opaque pointers** for encapsulation (`struct Foo;` in header, definition in .c)
- **Goto cleanup** pattern for resource management
- **Flexible array members** for variable-length structs
- **Static inline** functions in headers for zero-cost abstractions
- **Compound literals** for temporary struct values
- **Designated initializers** for readable struct initialization

## Reference Documentation

- [C11 Standard Draft (N1570)](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf) -- free
- [cppreference.com/w/c](https://en.cppreference.com/w/c) -- best online reference
- *The C Programming Language* (K&R, 2nd ed.) -- the classic
- *Modern C* by Jens Gustedt -- [free PDF](https://gustedt.gitlabpages.inria.fr/modern-c/)
- *Expert C Programming* by Peter van der Linden -- deep dives

## Practising these

```bash
practice/bin/ol list  build c          # which of these have references yet
practice/bin/ol start build c 01       # clone it stubbed into .scratchpad/
practice/bin/ol check build c 01       # run it; exit code is the verdict
practice/bin/ol diff  build c 01       # your attempt against the reference
```

Exercises are ordered by concept, not by difficulty, so pick by the stars rather
than by the number. An exercise listed below with no reference yet is a row in
this table and nothing more; see the [practice path](../../../README.md) for how the
harness works.

## Exercises

Implement each using only the C standard library. No external dependencies.

| # | Exercise | Concepts | Difficulty |
|---|----------|----------|------------|
| 01 | Dynamic array (`vec.h`) | `realloc`, growth factor, generics via macros; course version: `ds.01`, `lang.03` | ⭐⭐ |
| 02 | Hash map (open addressing) | Hash functions, probing, load factor; course version: `ds.02` (this drill is its worked baseline) | ⭐⭐⭐ |
| 03 | Linked list (intrusive) | Container-of macro, pointer manipulation; course version: `ds.03` | ⭐⭐ |
| 04 | Binary heap | Array-based, `sift_up`/`sift_down`; course version: `ds.04` | ⭐⭐ |
| 05 | Merge sort (in-place) | Pointer arithmetic, recursion | ⭐⭐ |
| 06 | Graph (adjacency list) | Flexible arrays, BFS/DFS | ⭐⭐⭐ |
| 07 | Thread pool | pthreads, condition variables, work queue; course version: `rt.03` | ⭐⭐⭐⭐ |
| 08 | Arena allocator | Memory management, alignment, bump allocation; course version: `rt.02` | ⭐⭐⭐ |
| 09 | String interning | Hash table, `strdup`, lifetime management | ⭐⭐⭐ |
| 10 | Simple HTTP parser | State machine, buffer management, `recv` | ⭐⭐⭐⭐ |
