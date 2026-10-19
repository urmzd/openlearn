# C

C is the language in which every other runtime in this book is ultimately written, and it is the one that forgives nothing. Memory is yours to allocate and yours to free; there is no garbage collector, no bounds check, and no type system strong enough to stop a `void *` from being misread. A mistake does not throw an exception. It is undefined behaviour, which may work perfectly on your machine and corrupt the heap on someone else's. Working in C teaches you what the higher-level languages are doing for you, and what it costs.

This chapter is ten classic exercises, each with a reference implementation and a full walkthrough. The arc runs from single containers (a growable array, a hash map, an intrusive list, a heap) through algorithms on them (merge sort, graph traversal) to the runtime pieces that larger systems are made of: a thread pool, an arena allocator, a string interner, and an incremental HTTP parser. Each section states the problem and its constraints, builds the design up from first principles, walks through the key code with annotations, explains why this design beats the obvious alternatives, and ends with review questions that check you can explain the solution back.

> **Learning objectives**
> - Manage heap memory in C without leaks, double frees, or dangling pointers, and verify it with AddressSanitizer and UndefinedBehaviorSanitizer.
> - Design C interfaces with opaque types, explicit error returns, and clear ownership rules.
> - Implement and analyse the core systems data structures (dynamic array, open-addressing hash map, intrusive list, binary heap, adjacency-list graph).
> - Build the runtime pieces of a server: a pthreads thread pool, an arena allocator, a string interner, and an incremental parser.
> - Explain, for each design, which alternatives were rejected and why.

## Before you begin

You should be able to write and compile a small C program, and know what a pointer, a struct, and `malloc` are. If any of that is shaky, read [C: memory, pointers, structs, headers, and the C11 toolchain](../../../../content/courses/programming-languages/02-language-and-tool-primers/03-c.md) in *Programming Languages* first. The big-O arguments assume the basics of asymptotic analysis.

## The language in brief

C's memory model is manual: you allocate with `malloc` and release with `free`, locals live on the stack and everything else on the heap, and there is no garbage collector. The type system is static but weak. Implicit conversions are everywhere, and generic code goes through `void *`, which the compiler cannot check. Errors are reported through return codes and `errno`; `setjmp` and `longjmp` exist but are best avoided. Concurrency comes from pthreads and, since C11, `stdatomic.h`, with nothing at the language level for asynchronous work. You build with `gcc` or `clang`, usually under `make` or `cmake`, with interfaces in headers and implementations in source files. The exercises use C11, which is preferred here; C17 also works.

## Idioms to practise

- **Opaque pointers** for encapsulation: declare `struct Foo;` in the header and define it only in the `.c` file.
- **The goto-cleanup pattern** for releasing resources on every exit path.
- **Flexible array members** for variable-length structs.
- **`static inline` functions in headers** for zero-cost abstractions.
- **Compound literals** for temporary struct values.
- **Designated initializers** for readable struct initialization.

## Building and running the exercises

Each exercise directory holds a header (the interface), the reference implementation, a `main.c` test suite, and the walkthrough. Every solution uses only the C standard library, plus pthreads for the thread pool. To build and run one, compile its sources together from its directory:

```bash
cd practice/build/systems/c/01-dynamic-array
cc -std=c11 -Wall -Wextra -O2 *.c -o vec && ./vec
```

The test program prints `ok ...` and exits 0 when every check holds. The thread pool needs `-pthread` and ships a `run.sh` that adds it. Running `just run-practice-solutions` from the repository root builds and runs all ten.

## In this chapter

The exercises are ordered by concept, not by difficulty, so if you are choosing where to start, the difficulty rating (1 is easiest, 5 hardest) is a better guide than the number.

1. [Dynamic array](01-dynamic-array/README.md) (difficulty 2). A growable array of `int` that teaches why `realloc` is dangerous, why the growth factor must be geometric, and how to survive allocation failure. The longer treatment is [A growable array](../../../../content/courses/algorithms/13-systems-data-structures/01-growable-array.md) in *Algorithms*.
2. [Hash map with open addressing](02-hash-map/README.md) (difficulty 3). String keys, linear probing, tombstones, and a load factor that counts the dead. It is the baseline that [A Swiss table](../../../../content/courses/algorithms/13-systems-data-structures/02-swiss-table.md) improves on.
3. [Intrusive linked list](03-intrusive-list/README.md) (difficulty 2). The Linux-kernel list: the links live inside your struct, and `container_of` walks back from the link to the object. Compare [An intrusive list and an LRU](../../../../content/courses/algorithms/13-systems-data-structures/03-intrusive-list-and-lru.md).
4. [Binary heap](04-binary-heap/README.md) (difficulty 2). A priority queue in a flat array, with `sift_up`, `sift_down`, and linear-time heapify. [Heap top-k in C](../../../../content/courses/algorithms/13-systems-data-structures/04-binary-heap-top-k.md) puts it to work.
5. [Merge sort, buffered and in place](05-merge-sort/README.md) (difficulty 2). Two stable sorts that trade memory for time: one with a scratch buffer, one with rotations.
6. [Graph as an adjacency list](06-graph/README.md) (difficulty 3). Growable per-vertex arrays, breadth-first and depth-first search, and topological sort.
7. [Thread pool](07-thread-pool/README.md) (difficulty 4). pthreads, condition variables, a work queue, and a shutdown that drains cleanly. [Thread pool and tl_parallel_for](../../../../content/courses/build-an-llm/10-kernels/02-thread-pool-and-parallel-for.md) in *Build Your Own LLM System* uses the same design.
8. [Arena allocator](08-arena-allocator/README.md) (difficulty 3). Bump allocation, alignment, and freeing everything at once. [C runtime support and an arena allocator with marks](../../../../content/courses/build-an-llm/10-kernels/01-arena-allocator-with-marks.md) extends it.
9. [String interning](09-string-interning/README.md) (difficulty 3). A hash table that hands out stable addresses, so string equality becomes pointer equality.
10. [HTTP request parser](10-http-parser/README.md) (difficulty 4). A state machine that consumes input incrementally, in whatever chunks the network delivers.

## Further reading

- [C11 standard draft (N1570)](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf): free.
- [cppreference.com/w/c](https://en.cppreference.com/w/c): the best online reference.
- *The C Programming Language* by Kernighan and Ritchie, 2nd edition: the classic.
- *Modern C* by Jens Gustedt: [free PDF](https://gustedt.gitlabpages.inria.fr/modern-c/).
- *Expert C Programming* by Peter van der Linden: deep dives.
