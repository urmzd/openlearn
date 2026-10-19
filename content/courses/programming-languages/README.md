# Programming Languages

This book is a working tour of the languages and tools a modern systems engineer reads every day. It is not a first programming course: it assumes you can already program in one language, and it shows what each new language or tool makes easy, what it makes dangerous, and which mental model you need to read it fluently.

## How this book is organized

[Type systems](01-type-systems/README.md) is the conceptual chapter. It compares how languages use types to rule out whole classes of bugs, from dynamic checks to ownership and affine types, with small programs in several languages.

[Language and tool primers](02-language-and-tool-primers/README.md) is the practical chapter. Each section takes one language or tool and works through a small, complete program, explaining every decision: Python and NumPy, the shell, git and make, C, Rust, HTTP and server-sent events, Go, containers and Kubernetes, Python asyncio, async Rust with Tokio, Protocol Buffers and gRPC, and SQL with SQLite. Every program's full solution is in [`course/solutions/primers/`](../../../course/solutions/primers/).

## Related books

*Practice: Predict and Build* gives you short programs in Go, Python, Rust, and TypeScript whose output you predict before reading the explanation. *Build Your Own LLM System* uses every language in this book in one system.
