# Practice: Predict and Build

This is the companion workbook to the rest of the shelf: small, self-contained problems, each shown with its answer and a worked explanation of why that answer is right. Chapter 1 gives you twenty-four short programs in Go, Python, Rust, and TypeScript whose exact output you predict before reading the explanation. Chapters 2 to 10 are classic build exercises, one chapter per language; C and C++ (Chapters 2 and 3) have full reference solutions with walkthroughs, and the other languages give the exercise briefs as design questions to reason through. It is for programmers who want to find out whether a mental model actually stuck.

## How to practise

Commit to a prediction and the rule behind it before revealing an answer. Compare your reasoning with the walkthrough, then choose a small change that would distinguish your explanation from a plausible wrong one. Run that case where executable code is provided; for a design brief, describe the test and expected evidence. Validate against the stated behavior, not just whether the program runs. The aim is to strengthen your mental model, including when a tool generated the code. See [How to read these books](../content/references/how-to-read.md) for the shared method.

## What you will be able to do

- Say what a short program prints and name the language rule that makes it print that.
- Explain why a reference implementation is shaped the way it is, and what breaks with the obvious alternatives.
- Recognize when several surprises across languages are really one missing mental model, such as value versus reference semantics.

## Ways through this book

Chapter 1 takes a few evenings, and each of the C and C++ chapters takes about two weeks at five to six hours a week; the briefs in Chapters 4 to 10 take an evening each to reason through.

- **Checking a language you just learned** (one or two evenings): the matching section of [Chapter 1: Predict-Then-Read](predict/README.md), for example [1.1 Go](predict/go/README.md) or [1.3 Rust](predict/rust/README.md). Commit to an output before you open the fold.
- **Finding your gaps, then filling them** (two to three weeks): all of [Chapter 1](predict/README.md), then the walkthroughs its surprises point to. Three aliasing surprises in three languages is one missing model, and [2.1 Dynamic array](build/systems/c/01-dynamic-array/README.md) and [3.1 Vector](build/systems/cpp/01-vector/README.md) are where it is built.
- **Systems design, defended line by line** (about four weeks): [Chapter 2: C](build/systems/c/README.md) in order, then [Chapter 3: C++](build/systems/cpp/README.md), whose vector, queue, and allocator sections ([3.1](build/systems/cpp/01-vector/README.md), [3.4](build/systems/cpp/05-threadsafe-queue/README.md), [3.9](build/systems/cpp/10-pool-allocator/README.md)) refer back to the C versions in [2.1](build/systems/c/01-dynamic-array/README.md), [2.7](build/systems/c/07-thread-pool/README.md), and [2.8](build/systems/c/08-arena-allocator/README.md).
- **Design questions in a language without solutions yet** (an evening per exercise): pick a brief from Chapters 4 to 10, such as [Chapter 4: Go](build/cloud/go/README.md) or [Chapter 6: Python](build/general/python/README.md), sketch a design, then compare it with the C or C++ walkthrough.

## Before you start

You should be able to read and write small programs in at least one of the languages here. [*Programming Languages*](../content/courses/programming-languages/README.md) teaches the type systems and language primers that Chapter 1 checks, and [*Algorithms*](../content/courses/algorithms/README.md) covers the data structures the build chapters implement. Every solution is a single self-testing program, as are the builds in [*Case Studies*](../content/courses/case-studies/README.md); for the same solution-and-walkthrough shape at the scale of a whole system, read [*Build Your Own LLM System*](../content/courses/build-an-llm/paths/course/README.md). New to these books? [How to read these books](../content/references/how-to-read.md) explains how sections, review questions, and solutions work.
