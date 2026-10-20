# Programming Languages

This book is a working tour of the languages and tools a modern systems engineer reads every day, and of the type theory that explains why they behave the way they do. It is for engineers who already program in one language and keep meeting new ones at the edge of their work, such as the Python researcher who now has to read a Rust server. Chapter 1 gives you the vocabulary of type systems; Chapter 2 gives you just enough of each language or tool to build one small, complete program.

## What you will be able to do

- Read typing judgments such as $\Gamma \vdash e : \tau$, state soundness as progress plus preservation, and classify polymorphic code by its mechanism.
- Trace Hindley-Milner inference by hand and check it against a running Algorithm W.
- Predict the runtime cost of generics in a language from whether it monomorphizes, erases, or passes dictionaries.
- Build a small program with each of eleven languages and tools, from C and Rust to gRPC, SQLite, and Kubernetes, and explain every line of it.

## Ways through this book

Chapter 1 takes two to three weeks; each primer in Chapter 2 is a sitting or two, best read just before you need it.

- **Understanding type systems** (two to three weeks): [Chapter 1: Type Systems and Polymorphism](01-type-systems/README.md), running [`hindley_milner.py`](01-type-systems/code/hindley_milner.py) and then [`inference.ml`](01-type-systems/code/inference.ml) as you reach inference, and the other files in [`code/`](01-type-systems/code/) side by side.
- **Just enough of one language, by Monday** (an evening to a weekend): open the primer you need in [Chapter 2](02-language-and-tool-primers/README.md); each names its prerequisites up front. For example, [2.8 Python asyncio](02-language-and-tool-primers/08-python-asyncio.md) builds on [2.5 HTTP and SSE](02-language-and-tool-primers/05-http-and-sse.md), and [2.10 Protocol Buffers and gRPC](02-language-and-tool-primers/10-protocol-buffers-and-grpc.md) builds on [2.4 Rust](02-language-and-tool-primers/04-rust.md), [2.6 Go](02-language-and-tool-primers/06-go.md), and [2.9 async Rust](02-language-and-tool-primers/09-async-rust-and-tokio.md).
- **Reviewing a Rust or Go service** (about two weeks): [2.4 Rust](02-language-and-tool-primers/04-rust.md), [2.9 async Rust and tokio](02-language-and-tool-primers/09-async-rust-and-tokio.md), [2.6 Go](02-language-and-tool-primers/06-go.md), then [Chapter 1](01-type-systems/README.md) with [`traits.rs`](01-type-systems/code/traits.rs) and [`generics.go`](01-type-systems/code/generics.go) open.
- **The whole book, in order** (six to eight weeks): Chapter 1, then the primers 2.1 to 2.11.

## Before you start

You should be comfortable programming in at least one language and using a terminal. Chapter 1 leans on proofs, induction, and relations from [Chapter 6: Discrete Math 1](../mathematics-for-computing/06-discrete-math-1/) in *Mathematics for Computing*. Primer solutions live in [`course/solutions/primers/`](../../../course/solutions/primers/).

Two books pair with this one: [*Practice: Predict and Build*](../../../practice/README.md) tests whether a mental model has stuck with short programs whose output you predict, and [*Build Your Own LLM System*](../build-an-llm/README.md) puts every language here to work in one system.

New to these books? [How to read these books](../../references/how-to-read.md) explains how sections, review questions, and solutions work.
