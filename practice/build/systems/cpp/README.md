# C++ Practice

Master RAII, move semantics, templates, and the STL.

## Language Essentials

| Concept | Details |
|---------|---------|
| **Memory Model** | RAII, smart pointers (`unique_ptr`, `shared_ptr`), stack preference |
| **Type System** | Strong static typing, templates, concepts (C++20), `auto` |
| **Error Handling** | Exceptions (use judiciously), `std::expected` (C++23), `std::optional` |
| **Concurrency** | `std::thread`, `std::async`, `std::atomic`, coroutines (C++20) |
| **Build** | `cmake`, `g++`/`clang++`, modules (C++20), header-only libraries |
| **Standard** | C++20 (prefer) or C++23 |

## Key Idioms

- **RAII everywhere** -- destructors for cleanup, no manual `delete`
- **Move semantics** -- `std::move`, rvalue references, rule of five/zero
- **SFINAE / Concepts** -- constrain templates at compile time
- **Range-based for** with structured bindings
- **`std::variant` + `std::visit`** for sum types
- **CRTP** (Curiously Recurring Template Pattern) for static polymorphism
- **`constexpr`** for compile-time computation

## Reference Documentation

- [cppreference.com](https://en.cppreference.com/w/) -- authoritative reference
- [ISO C++ FAQ](https://isocpp.org/faq) -- common questions
- *A Tour of C++* (Stroustrup, 3rd ed.) -- concise modern overview
- *Effective Modern C++* (Scott Meyers) -- essential C++11/14 patterns
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/) -- free, maintained by Stroustrup & Sutter

## Practising these

```bash
practice/bin/ol list  build cpp          # which of these have references yet
practice/bin/ol start build cpp 01       # clone it stubbed into .scratchpad/
practice/bin/ol check build cpp 01       # run it; exit code is the verdict
practice/bin/ol diff  build cpp 01       # your attempt against the reference
```

Exercises are ordered by concept, not by difficulty, so pick by the stars rather
than by the number. An exercise listed below with no reference yet is a row in
this table and nothing more; see the [practice path](../../../README.md) for how the
harness works.

## Exercises

| # | Exercise | Concepts | Difficulty |
|---|----------|----------|------------|
| 01 | `Vector<T>` (like `std::vector`) | Templates, move semantics, allocator awareness | ⭐⭐⭐ |
| 02 | `UniquePtr<T>` | RAII, move-only types, custom deleters | ⭐⭐ |
| 04 | Compile-time matrix (`constexpr`) | `constexpr`, template metaprogramming | ⭐⭐⭐ |
| 05 | Thread-safe queue | `std::mutex`, `std::condition_variable`, move semantics | ⭐⭐⭐ |
| 06 | Expression template (lazy eval) | CRTP, operator overloading, zero-cost abstractions | ⭐⭐⭐⭐ |
| 07 | Coroutine generator | C++20 coroutines, `co_yield`, lazy sequences | ⭐⭐⭐⭐ |
| 08 | `std::function` clone | Type erasure, small buffer optimization | ⭐⭐⭐⭐ |
| 09 | Graph with concepts | C++20 concepts, ranges, BFS/DFS | ⭐⭐⭐ |
| 10 | Memory pool allocator | Custom allocator, placement new, alignment | ⭐⭐⭐⭐ |

Exercise 03 (hash map) is consolidated into the course: six hash maps across the drills became two course modules, `ds.02` (Swiss table, C) and `ds.05` (Robin Hood, Rust), whose chapters live in [Systems Data Structures](../../../../content/tracks/algorithms/16-systems-data-structures/).
