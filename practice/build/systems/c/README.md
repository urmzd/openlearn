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

## How these are organised

Each exercise directory holds a header (the interface), the reference
implementation, a `main.c` test suite, and a README that works through the
solution: the problem and its constraints, the reasoning that leads to the
design, the key code annotated line by line, why this design beats the
alternatives, and questions to check you can explain it back. Exercises are
ordered by concept, not by difficulty, so pick by the stars rather than by the
number.

Every solution uses only the C standard library (plus pthreads for 07). To
build and run one, compile its sources together from its directory:

```bash
cd practice/build/systems/c/01-dynamic-array
cc -std=c11 -Wall -Wextra -O2 *.c -o vec && ./vec
```

The test program prints `ok ...` and exits 0 when every check holds. Exercise
07 needs `-pthread` and ships a `run.sh` that adds it. `just
run-practice-solutions` from the repository root builds and runs all ten.

## Exercises

| # | Exercise | Concepts | Difficulty |
|---|----------|----------|------------|
| 01 | [Dynamic array](01-dynamic-array/) | `realloc`, growth factor, amortised cost; course version: `ds.01`, `lang.03` | ⭐⭐ |
| 02 | [Hash map (open addressing)](02-hash-map/) | Hash functions, probing, tombstones, load factor; course version: `ds.02` (this exercise is its worked baseline) | ⭐⭐⭐ |
| 03 | [Linked list (intrusive)](03-intrusive-list/) | Container-of macro, pointer manipulation; course version: `ds.03` | ⭐⭐ |
| 04 | [Binary heap](04-binary-heap/) | Array-based, `sift_up`/`sift_down`; course version: `ds.04` | ⭐⭐ |
| 05 | [Merge sort (buffered and in-place)](05-merge-sort/) | Pointer arithmetic, recursion, stability, rotation | ⭐⭐ |
| 06 | [Graph (adjacency list)](06-graph/) | Growable per-vertex arrays, BFS/DFS, topological sort | ⭐⭐⭐ |
| 07 | [Thread pool](07-thread-pool/) | pthreads, condition variables, work queue; course version: `rt.03` | ⭐⭐⭐⭐ |
| 08 | [Arena allocator](08-arena-allocator/) | Memory management, alignment, bump allocation; course version: `rt.02` | ⭐⭐⭐ |
| 09 | [String interning](09-string-interning/) | Hash table, stable addresses, lifetime management | ⭐⭐⭐ |
| 10 | [HTTP request parser](10-http-parser/) | State machine, buffer management, incremental input | ⭐⭐⭐⭐ |
