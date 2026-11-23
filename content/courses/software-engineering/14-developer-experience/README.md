# Developer Experience

A library, a command-line tool, or a coding agent succeeds only when a person can predict what it will do, check that it did it, and correct it when it did not. Developer experience is the name for that loop. This chapter studies it through published interfaces and runnable examples, not through opinion.

The chapter has two halves. The first studies how people review work that a tool or an agent did for them: which evidence they need, where it should sit, and how a correction reaches the right target. The second studies how people learn an API: a runnable example beside its source, one value to change, and a signature that tells the caller what can fail.

> **Learning objectives**
> - Name the evidence a reviewer needs before accepting an agent's change: the target, the change, and the check that supports it.
> - Distinguish a proposed change from an applied one, and tie an undo action to one specific change.
> - Design a first example that a reader can run, change in one place, and verify by its output.
> - Judge an interface by the work it leaves to its callers, not by the size of its signature.
> - Make each intermediate value in a pipeline inspectable, including the exact request sent to a model.

## Before you begin

Read [The Pragmatic Programmer](../01-pragmatic-programmer/README.md) and [Documentation Writing](../12-documentation/README.md) first. The examples use Go, Rust, `jq`, and `uv`. Each example states its expected output, so you can read the chapter without running them.

## Sections

| Section | Question it answers |
|---|---|
| [Reviewing an agent's work](01-reviewing-agent-work.md) | What does a reviewer need to see to accept, reject, or steer a change? |
| [Learning an API by editing it](02-learning-by-editing.md) | How does an example move a reader from "it runs" to "I changed it and know why"? |
| [Small interfaces that do real work](03-small-interfaces.md) | How do a signature and its errors teach the caller what to do? |
| [Inspecting a pipeline into a model](04-inspecting-pipelines.md) | How do you see what a model actually received? |

## Primary sources

- Saleema Amershi and colleagues, [Guidelines for Human-AI Interaction](https://www.microsoft.com/en-us/research/publication/guidelines-for-human-ai-interaction/), an empirical basis for testing the ideas in the first half.
- John Ousterhout, *A Philosophy of Software Design*, [chapter 6 extract](https://web.stanford.edu/~ouster/cgi-bin/aposd2ndEdExtract.pdf), on general-purpose interfaces.
- Rob Pike, [Errors are values](https://go.dev/blog/errors-are-values), on error contracts a caller can use.
