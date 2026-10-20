# Rust

Rust gives you C's control over memory without C's undefined behaviour, and the price is that the compiler makes you prove your program is safe before it will build it. Ownership says who frees each value, borrowing says who may look at it and for how long, and the `Send` and `Sync` traits extend the same reasoning to threads. The exercises in this chapter are chosen because they are exactly where that proof is hard: a growable buffer with raw memory, a doubly linked list where two nodes point at each other, a channel shared between threads, a lock-free stack. Each one teaches you what the borrow checker is protecting you from, and when `unsafe` is the honest answer.

The chapter is a set of exercise briefs rather than worked solutions. Several of them are the same problem a section of *Algorithms*, *Programming and Algorithms*, or *AI Engineering* solves in full, and for those the brief links to that section.

> **Learning objectives**
> - Describe Rust's ownership and borrowing rules, its algebraic type system, and its `Result`-based error handling.
> - Explain how `Send` and `Sync` make data races a compile-time error, and what `unsafe` code must uphold by hand.
> - Apply core Rust idioms (exhaustive matching, iterator chains, newtypes, `From`/`Into`) when designing a small library.
> - Reason through the design of each exercise before reading the matching worked solution elsewhere in the books.

## Before you begin

You should have written some Rust and fought the borrow checker at least once. If not, read [Rust: ownership, traits, Result, cargo workspaces, std TCP](../01-language-primers/04-rust.md) in *Programming and Algorithms* first. The async briefs build on [Async Rust and tokio: futures, tasks, channels, cancellation, hyper](../01-language-primers/09-async-rust-and-tokio.md).

## The language in brief

Rust has no garbage collector. Every value has exactly one owner, and it is dropped when the owner goes out of scope. You may lend it out as any number of shared references (`&T`) or exactly one mutable reference (`&mut T`), and the compiler checks lifetimes so that no reference outlives what it points at.

The type system is strong and static, with algebraic data types (`enum` with payloads), traits for shared behaviour, and generics constrained by trait bounds. There is no null; absence is `Option<T>`. Errors are values too: `Result<T, E>` with the `?` operator to propagate them, and no exceptions.

Concurrency is "fearless" in the sense that the compiler rejects data races: a type can cross threads only if it is `Send`, and be shared between them only if it is `Sync`. Asynchronous code uses `async`/`await` over futures, usually on the Tokio runtime. Builds use `cargo` and crates.io, and the edition system lets the language evolve without breaking old code. The exercises target stable Rust, edition 2021.

## Idioms to practise

- **Exhaustive pattern matching** on enums.
- **Iterator chains** (`map`, `filter`, `collect`) instead of explicit loops.
- **`impl Trait`** for return-position polymorphism.
- **The newtype pattern** for type safety, such as `struct Miles(f64)`.
- **The builder pattern** for complex construction.
- **`From` and `Into` conversions** instead of explicit casts.
- **`derive` macros** for common trait implementations.

## In this chapter

The briefs are listed below, ordered by concept rather than by difficulty, so choose by the difficulty rating (1 is easiest, 5 hardest) rather than by number. Worked Rust solutions for these exercises have not been written yet; where a full solution of the same problem, or one that builds directly on it, exists elsewhere in the books, the last column links to it.

| # | Exercise | Concepts | Difficulty (1-5) | Worked elsewhere |
|---|----------|----------|------------------|------------------|
| 01 | `Vec<T>` from scratch | Unsafe, `RawVec`, `Drop`, `Deref` | 3 | Do this before [the Candle model runner](../../ai-engineering/23-model-serving/02-model-runner-and-sampler.md) |
| 02 | Doubly linked list | `Rc<RefCell<T>>` or unsafe, borrow checker challenges | 4 | [A radix tree over token ids](../15-systems-data-structures/07-radix-tree.md) |
| 03 | Hash map (Robin Hood) | Hashing, `Entry` API, generics with trait bounds | 4 | [A Robin Hood hash map](../15-systems-data-structures/05-robin-hood-hash-map.md) |
| 04 | Channel (`mpsc` clone) | `Arc<Mutex<T>>`, `Condvar`, `Send` bounds | 4 | [Async Rust and tokio](../01-language-primers/09-async-rust-and-tokio.md) |
| 05 | Iterator adaptor library | `Iterator` trait, associated types, lazy evaluation | 3 | [Fast BPE in Rust](../../ai-engineering/07-tokenizers/05-rust-fast-bpe.md) |
| 06 | CLI argument parser | Lifetimes, `&str` vs `String`, builder pattern | 2 | |
| 07 | Async TCP echo server | `async`/`await`, `tokio`, `Pin`, `Future` | 4 | [Async Rust and tokio](../01-language-primers/09-async-rust-and-tokio.md); [the OpenAI-compatible server on tokio](../../ai-engineering/23-model-serving/06-openai-server-on-tokio.md) |
| 08 | ECS (Entity Component System) | Trait objects vs enums, `Any`, downcast, arena allocation | 5 | |
| 09 | Proc macro for `derive(Builder)` | Procedural macros, `syn`, `quote`, token streams | 4 | |
| 10 | Lock-free stack | `AtomicPtr`, `compare_exchange`, ABA problem | 5 | |

## Further reading

- [The Rust Programming Language](https://doc.rust-lang.org/book/): free, the official book.
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/): free, learn by doing.
- [Standard library documentation](https://doc.rust-lang.org/std/): the API reference.
- [The Rustonomicon](https://doc.rust-lang.org/nomicon/): the deep dive into unsafe Rust, essential for exercises 01, 02, and 10.
- *Programming Rust* by Blandy, Orendorff, and Tindall: the comprehensive reference.
