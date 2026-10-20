# Programming Languages

This book is a working tour of the languages and tools a modern systems engineer reads every day, and of the type theory that explains why they behave the way they do. It is for engineers who already program in one language and keep meeting new ones at the edge of their work: the Python researcher who now has to read a Rust server, the Go developer asked to review C, the backend engineer who wants to know why TypeScript accepts a program that Haskell would reject. Chapter 1 gives you the vocabulary of type systems; Chapter 2 gives you just enough of each language or tool to build and explain a small, complete program in it.

## What you will be able to do

- Read typing judgments such as $\Gamma \vdash e : \tau$, state soundness as progress plus preservation, and classify polymorphic code as parametric, ad-hoc, subtype, coercion, or bounded.
- Trace Hindley-Milner inference by hand on small terms and check your answer against a running implementation of Algorithm W.
- Predict the runtime cost of generics in a language from whether it monomorphizes, erases, or passes dictionaries.
- Build a small, complete program in Python with NumPy, C, Rust, Go, async Python, and async Rust, and work with the shell, git, make, HTTP and server-sent events, Protocol Buffers and gRPC, SQLite, and containers on Kubernetes.
- Explain, line by line, why each of those programs is written the way it is.

## Ways through this book

Chapter 1 takes two to three weeks of steady reading, and each primer in Chapter 2 is a sitting or two, best read just before you first need that language or tool.

- **Understanding type systems** (two to three weeks): [Chapter 1: Type Systems and Polymorphism](01-type-systems/README.md) in order. Run [`hindley_milner.py`](01-type-systems/code/hindley_milner.py) while you read about inference, then [`inference.ml`](01-type-systems/code/inference.ml) to watch OCaml reach the same types, then the TypeScript, Go, Rust, and Haskell files in [`code/`](01-type-systems/code/) side by side.
- **Just enough of one language, by Monday** (an evening to a weekend): open the primer you need in [Chapter 2](02-language-and-tool-primers/README.md). If you are new to the toolchain, start with [2.1 Python and NumPy](02-language-and-tool-primers/01-python-and-numpy.md) and [2.2 Shell, git, make](02-language-and-tool-primers/02-shell-git-make.md). Each primer names its prerequisites in its opening paragraphs: [2.8 Python asyncio](02-language-and-tool-primers/08-python-asyncio.md) builds on [2.5 HTTP and SSE](02-language-and-tool-primers/05-http-and-sse.md), and [2.10 Protocol Buffers and gRPC](02-language-and-tool-primers/10-protocol-buffers-and-grpc.md) builds on [2.4 Rust](02-language-and-tool-primers/04-rust.md), [2.6 Go](02-language-and-tool-primers/06-go.md), and [2.9 async Rust](02-language-and-tool-primers/09-async-rust-and-tokio.md).
- **Reviewing a Rust or Go service** (about two weeks): [2.4 Rust](02-language-and-tool-primers/04-rust.md), [2.9 async Rust and tokio](02-language-and-tool-primers/09-async-rust-and-tokio.md), [2.6 Go](02-language-and-tool-primers/06-go.md), then [Chapter 1](01-type-systems/README.md) with [`traits.rs`](01-type-systems/code/traits.rs) and [`generics.go`](01-type-systems/code/generics.go) open, to see why traits and Go's type-set constraints behave as they do.
- **The whole book, in order** (six to eight weeks): Chapter 1, then the primers 2.1 to 2.11.

## Before you start

You should be comfortable programming in at least one language and using a terminal. Chapter 1 leans on proofs, induction, and relations from [Chapter 6: Discrete Math 1](../mathematics-for-computing/06-discrete-math-1/) in *Mathematics for Computing*; the notation for inference rules is taught as it is needed. The full solution for every primer lives in [`course/solutions/primers/`](../../../course/solutions/primers/).

Two books pair with this one. [*Practice: Predict and Build*](../../../practice/README.md) gives you short programs in Go, Python, Rust, and TypeScript whose output you predict before reading the explanation, which tests whether a mental model from this book has stuck. [*Build Your Own LLM System*](../build-an-llm/README.md) uses every language here in one system, and each primer names the section of that book where its ideas are first put to work.

New to these books? [How to read these books](../../references/how-to-read.md) explains how sections, review questions, and solutions work.
