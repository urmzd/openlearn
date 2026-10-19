# Programming Languages

This book is a working tour of the languages and tools a modern systems engineer reads every day, and of the theory that explains why they behave the way they do. It is not a first programming course. It assumes you can already program in one language, and it shows you what each new language or tool makes easy, what it makes dangerous, and which mental model you need to read it fluently.

The book is for engineers who keep meeting a new language at the boundary of their work: the Python researcher who now has to read a Rust server, the Go developer asked to review a C kernel, the backend engineer who wants to know why TypeScript accepts a program that Haskell would reject. If you have ever argued about "generics versus interfaces" without being sure what the other person meant, the first chapter is for you; if you have ever needed "just enough Rust" or "just enough Kubernetes" by Monday, the second is.

## What you will be able to do

By the end of the book you will be able to:

- Read typing judgments and inference rules, state what a sound type system guarantees, and classify a piece of polymorphic code by the mechanism it uses.
- Trace Hindley-Milner inference by hand and predict the runtime cost of generics in a language from how it compiles them.
- Pick up Python with NumPy, the shell, git and make, C, Rust, Go, async Python, async Rust, HTTP and server-sent events, Protocol Buffers and gRPC, SQL, and containers on Kubernetes at the level needed to build and explain a small, complete program in each.
- Explain, line by line, why each of those programs is written the way it is.

## How this book is organized

The book has two chapters, one conceptual and one practical. You can read them in either order, but the first gives you vocabulary that makes the second faster.

[Type Systems and Polymorphism](01-type-systems/README.md) treats a type system as a lightweight formal method. It defines soundness as progress plus preservation, separates the kinds of polymorphism (parametric, ad-hoc, subtype, coercion, and bounded), follows the lambda-calculus ladder from the simply typed calculus to System F, and derives Hindley-Milner inference with a runnable implementation of Algorithm W. It then turns practical: variance and why `List<Dog>` is not automatically a `List<Animal>`, and why monomorphization, erasure, and dictionary passing give the same generic function three different runtime stories. Companion programs in OCaml, TypeScript, Go, Rust, and Haskell walk one running example through five type systems that make different choices.

[Language and Tool Primers](02-language-and-tool-primers/README.md) takes one language or tool per section and works through a small, complete program, explaining every decision: Python and NumPy, the shell with git and make, C, Rust, HTTP and server-sent events, Go, containers and Kubernetes, Python asyncio, async Rust with Tokio, Protocol Buffers and gRPC, and SQL with SQLite. Each primer teaches only what its first real use needs, and closes by naming where that use is.

## Prerequisites and related books

You should be comfortable programming in at least one language and using a terminal. The type-systems chapter leans on proofs, induction, and relations from [Discrete Math 1](../mathematics-for-computing/05-discrete-math-1/) in *Mathematics for Computing*; the notation for inference rules is taught as it is needed.

Two books pair naturally with this one. [*Practice: Predict and Build*](../../../practice/README.md) gives you short programs in Go, Python, Rust, and TypeScript whose output you predict before reading the explanation, which is the fastest way to test whether a mental model from this book has stuck. [*Build Your Own LLM System*](../build-an-llm/README.md) uses every language in this book in one system, and each primer names the section of that book where its ideas are first put to work.

## Conventions

Code appears in fenced listings tagged with their language, and short identifiers such as `Result` or `context.Context` are set in code type within the prose. Mathematics, mostly typing judgments such as $\Gamma \vdash e : \tau$, is set inline or in display. Call-outs mark material that deserves a pause: **Learning objectives** open each chapter and section, and **Warning** and **Note** call-outs flag pitfalls and side remarks.

Each section ends with key terms, a summary, and review questions. The answers sit in collapsed blocks beneath each question; try to answer aloud before you open them, since being able to explain why a line is there is the skill the book is built to teach.

The runnable code for the type-systems chapter lives beside it in [`01-type-systems/code/`](01-type-systems/code/). The full solution for every primer lives in [`course/solutions/primers/`](../../../course/solutions/primers/), one directory per primer, and the sections quote from it as they walk through each step. A suggested pace for reading the book is in the [study plan](study-plan.md).
