# Rust Practice

Master ownership, borrowing, lifetimes, and zero-cost abstractions.

## Language Essentials

| Concept | Details |
|---------|---------|
| **Memory Model** | Ownership + borrowing, no GC, compile-time lifetime checks |
| **Type System** | Strong static, algebraic types (`enum`), traits, generics, no null |
| **Error Handling** | `Result<T, E>`, `Option<T>`, `?` operator, no exceptions |
| **Concurrency** | Fearless concurrency, `Send`/`Sync` traits, `async`/`await`, Tokio |
| **Build** | `cargo`, crates.io, edition system (2021+) |
| **Standard** | Stable Rust, edition 2021 |

## Key Idioms

- **Pattern matching** exhaustively on enums
- **Iterator chains** over explicit loops (`map`, `filter`, `collect`)
- **`impl Trait`** for return-position polymorphism
- **Newtype pattern** for type safety (`struct Miles(f64)`)
- **Builder pattern** for complex construction
- **`From`/`Into`** conversions instead of explicit casts
- **`derive` macros** for common trait implementations

## Reference Documentation

- [The Rust Programming Language](https://doc.rust-lang.org/book/) -- free, the official book
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/) -- free, learn by doing
- [std library docs](https://doc.rust-lang.org/std/) -- API reference
- [Rustonomicon](https://doc.rust-lang.org/nomicon/) -- unsafe Rust deep dive
- *Programming Rust* (Blandy, Orendorff, Tindall) -- comprehensive reference

## How this page is organised

The table below is an index of exercise briefs: what to build, the concepts it
exercises, and how hard it is. Exercises are ordered by concept, not by
difficulty. Where a row names a course version (for example `ds.02`), that
module's chapter walks through a full solution of the same problem.

## Exercises

Worked solutions for these exercises are not written yet.

| # | Exercise | Concepts | Difficulty |
|---|----------|----------|------------|
| 01 | `Vec<T>` from scratch | Unsafe, `RawVec`, `Drop`, `Deref`; recommended before course module `L10.1` | ⭐⭐⭐ |
| 02 | Doubly-linked list | `Rc<RefCell<T>>` or unsafe, borrow checker challenges; course version: `ds.07` | ⭐⭐⭐⭐ |
| 03 | Hash map (Robin Hood) | Hashing, `Entry` API, generics with trait bounds; course version: `ds.05` | ⭐⭐⭐⭐ |
| 04 | Channel (`mpsc` clone) | `Arc<Mutex<T>>`, `Condvar`, `Send` bounds; course version: `lang.09` | ⭐⭐⭐⭐ |
| 05 | Iterator adaptor library | `Iterator` trait, associated types, lazy evaluation; course version: `L1.5` | ⭐⭐⭐ |
| 06 | CLI argument parser | Lifetimes, `&str` vs `String`, builder pattern | ⭐⭐ |
| 07 | Async TCP echo server | `async`/`await`, `tokio`, `Pin`, `Future`; course version: `lang.09`, `L10.5` | ⭐⭐⭐⭐ |
| 08 | ECS (Entity Component System) | Trait objects vs enums, `Any`, downcast, arena allocation | ⭐⭐⭐⭐⭐ |
| 09 | Proc macro for `derive(Builder)` | Procedural macros, `syn`, `quote`, token streams | ⭐⭐⭐⭐ |
| 10 | Lock-free stack | `AtomicPtr`, `compare_exchange`, ABA problem | ⭐⭐⭐⭐⭐ |
