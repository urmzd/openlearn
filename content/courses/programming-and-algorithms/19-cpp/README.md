# C++

C++ keeps C's control over memory and layout and adds the tools to make that control safe: destructors that release resources automatically, move semantics that transfer ownership without copying, templates that generate code per type at no run-time cost, and, since C++20, concepts and coroutines. The price is that the language exposes every one of those mechanisms. To use `std::vector` well you need to know that allocation is not construction; to use `std::function` well you need to know what type erasure costs. This chapter builds the library pieces themselves so that those costs become visible.

Each of the nine exercises has a reference implementation in a single header and a walkthrough. The arc starts with ownership and object lifetime (a vector and a unique pointer), moves to compile-time programming (a `constexpr` matrix and expression templates), then to concurrency and coroutines (a thread-safe queue and a generator), and finishes with type erasure, concepts, and a custom allocator. Each section states the problem and its constraints, builds the design from first principles, walks through annotated code, explains why the design beats the alternatives, and closes with review questions.

> **Learning objectives**
> - Manage object lifetime explicitly: separate allocation from construction, and implement the rule of five with correct exception guarantees.
> - Use RAII and move-only types to make ownership a compile-time property.
> - Move work and checks to compile time with templates, `constexpr`, `requires`, and concepts.
> - Build concurrent and lazy abstractions with `std::condition_variable` and C++20 coroutines.
> - Implement type erasure and a standard-library-compatible allocator, and explain their costs.

## Before you begin

You should be comfortable with C++ classes, references, and basic templates, and with the C memory model. The [C](../18-c/README.md) chapter is the natural predecessor: several sections here (the vector, the queue, the allocator) are the C++ counterparts of C exercises and refer back to them.

## The language in brief

Memory in C++ is managed through RAII: a resource is acquired in a constructor and released in a destructor, so cleanup happens on every exit path, including exceptions. Smart pointers (`unique_ptr` for one owner, `shared_ptr` for shared ownership) apply that to heap memory, and stack allocation is preferred where it fits.

The type system is strongly and statically typed. Templates generate code per type, C++20 concepts constrain them, and `auto` deduces types where spelling them adds nothing. Errors are reported with exceptions (to be used judiciously), `std::optional` for absence, and, from C++23, `std::expected` for a value or an error. Concurrency comes from `std::thread`, `std::async`, and `std::atomic`, and C++20 adds coroutines. Builds use `cmake` with `g++` or `clang++`; C++20 also has modules, and many libraries are header-only. The exercises use C++20, which is preferred here; C++23 also works.

## Idioms to practise

- **RAII everywhere**: destructors do the cleanup, and there is no manual `delete`.
- **Move semantics**: `std::move`, rvalue references, and the rule of five (or zero).
- **SFINAE and concepts** to constrain templates at compile time.
- **Range-based `for`** with structured bindings.
- **`std::variant` with `std::visit`** for sum types.
- **CRTP** (the curiously recurring template pattern) for static polymorphism.
- **`constexpr`** for compile-time computation.

## Building and running the exercises

Each exercise directory holds a header with the reference implementation, a `main.cpp` test suite, and the walkthrough. Every exercise is a single translation unit. To build and run one:

```bash
cd practice/build/systems/cpp/01-vector
c++ -std=c++20 -Wall -Wextra -O2 main.cpp -o vector && ./vector
```

The test program prints `ok ...` and exits 0 when every check holds. The thread-safe queue needs `-pthread` and ships a `run.sh` that adds it.

## In this chapter

The exercises are ordered by concept, not by difficulty, so if you are choosing where to start, the difficulty rating (1 is easiest, 5 hardest) is a better guide than the number. There is no exercise 03.

1. [Vector](01-vector/README.md) (difficulty 3). A `std::vector` work-alike that separates raw storage from live objects, grows with `move_if_noexcept`, and implements the rule of five with copy-and-swap.
2. [UniquePtr](02-unique-ptr/README.md) (difficulty 2). A move-only owning pointer with custom deleters and an array specialisation, and the exact order of operations that makes `reset` and self-move safe.
4. [Compile-time matrix](04-constexpr-matrix/README.md) (difficulty 3). A matrix whose dimensions live in its type, so shape errors fail to compile, with `requires`-constrained square-only members and `constexpr` arithmetic.
5. [Thread-safe queue](05-threadsafe-queue/README.md) (difficulty 3). A bounded blocking queue built from one mutex and two condition variables, with a `close` that wakes everyone and still delivers the backlog.
6. [Expression templates](06-expression-template/README.md) (difficulty 4). Lazy vector arithmetic with CRTP that computes `a + b + c` in one pass, and the storage rule that keeps `auto` expressions from dangling.
7. [Coroutine generator](07-coroutine-generator/README.md) (difficulty 4). A `Generator<T>` over the C++20 coroutine protocol: lazy start, owned frames, exceptions delivered to the consumer.
8. [Function: type erasure with a small buffer](08-function/README.md) (difficulty 4). A `std::function` clone with a hand-written vtable that stores small callables inline and supports move-only captures.
9. [Graph with concepts](09-graph-concepts/README.md) (difficulty 3). BFS, DFS, shortest path, and Dijkstra written once against a concept and run on three unrelated graph types.
10. [Memory pool allocator](10-pool-allocator/README.md) (difficulty 4). A fixed-size block allocator with an intrusive free list, and the standard-library adaptor that lets `std::list` and `std::map` use it.

Exercise 03 was a hash map. Six hash maps across the exercises were consolidated into two worked sections of *Algorithms*: [A Swiss table](../15-systems-data-structures/02-swiss-table.md), in C, and [A Robin Hood hash map](../15-systems-data-structures/05-robin-hood-hash-map.md), in Rust, both in the [Systems data structures](../15-systems-data-structures/) chapter.

## Further reading

- [cppreference.com](https://en.cppreference.com/w/): the authoritative reference.
- [ISO C++ FAQ](https://isocpp.org/faq): answers to common questions.
- *A Tour of C++* by Bjarne Stroustrup, 3rd edition: a concise modern overview.
- *Effective Modern C++* by Scott Meyers: the essential C++11/14 patterns.
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/): free, maintained by Stroustrup and Sutter.
