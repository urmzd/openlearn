# Practice

The rest of this repository is the **learning path**: tracks you read, in
order, with a textbook behind each topic. This directory is the **practice
part of the manual**: small, self-contained problems, each shown with its
answer and a worked explanation of why that answer is right.

Reading on its own produces recognition, which feels like knowledge and is
not. You nod along to a correct solution and come away unable to rebuild it,
defend it, or spot the bug in a near miss. The cure is not to type the code
yourself; it is to commit to an answer before you look, and then to be able to
explain the optimal one from first principles. Every page here is written for
that: the problem first, the answer folded away, and the reasoning laid out so
you can check yours step by step.

## Two parts

| Part | What it shows | What you should be able to do afterwards |
|------|---------------|------------------------------------------|
| [`predict/`](predict/) | Short snippets, their exact output, and a line-by-line explanation of the language rule behind each line | Say what a program prints and name the rule that makes it print that |
| [`build/`](build/) | Reference implementations of classic exercises, with walkthroughs of the design | Explain why the solution is shaped the way it is and what breaks with the obvious alternatives |

`predict` finds holes fastest, because committing to an output is cheap and
the correction is precise. `build` finds different holes, the ones that only
show up when every design decision has to be justified: why a growth factor of
two, why this lock and not that one, why the destructor runs here.

## Predict: worked output explanations

Twenty-four snippets across Go, Rust, Python, and TypeScript, six each, every
one aimed at a belief that is common, load-bearing, and wrong. Start at
[`predict/README.md`](predict/README.md) for how to use them and why the
cross-language pairings are the real payload, or go straight to a language:

| Language | Worked answers |
|----------|----------------|
| Go | [`predict/go/`](predict/go/README.md) |
| Rust | [`predict/rust/`](predict/rust/README.md) |
| Python | [`predict/python/`](predict/python/README.md) |
| TypeScript | [`predict/ts/`](predict/ts/README.md) |

Each page shows the snippet's source link, then hides the output and the
explanation behind a fold. Read the code, write down what you think it prints,
then open the fold. A prediction you change after seeing the answer tells you
nothing.

## Build: reference implementations with walkthroughs

Ten exercises per language, ordered by concept rather than by difficulty, so
read the stars and not the numbers. Standard library only unless an exercise
says otherwise: the point is to see how the thing itself is built, not which
library already does it.

Solutions with walkthroughs exist today for C and C++. The other languages
list the exercise briefs only, which are still useful as design questions to
reason through before reading a solution in a language that has one.

### Core

| Language | Focus | Status |
|----------|-------|--------|
| [C](build/systems/c/README.md) | Manual memory, pointers, undefined behaviour | Solutions and walkthroughs, 10 exercises |
| [C++](build/systems/cpp/README.md) | RAII, templates, move semantics | Solutions and walkthroughs, 9 exercises |
| [Go](build/cloud/go/README.md) | Concurrency, channels, interfaces | Exercise briefs |
| [Rust](build/systems/rust/README.md) | Ownership, lifetimes, zero-cost abstractions | Exercise briefs |
| [Python](build/general/python/README.md) | Data model, metaprogramming, asyncio | Exercise briefs |
| [TypeScript](build/general/typescript/README.md) | Type narrowing, generics, runtime safety | Exercise briefs |

### Optional

Kept because the exercises are worth thinking through, not because the
languages are used elsewhere in this repository.

| Language | Focus | Status |
|----------|-------|--------|
| [Zig](build/systems/zig/README.md) | Comptime, explicit allocators, C interop | Exercise briefs |
| [Scala](build/cloud/scala/README.md) | FP plus OOP, type system, given instances | Exercise briefs |
| [Java](build/cloud/java/README.md) | JVM internals, virtual threads, generics | Exercise briefs |

Every solution is a single self-testing program: one entry point, its own
assertions, nonzero exit on the first failure. The assertions are part of the
answer, since they state exactly what "correct" means for the exercise. The
[case studies](../content/courses/case-studies/) are built the same way.

## Picking something

Start with `predict`. It takes minutes, and the gap between what you expected
and what the explanation says tells you which `build` walkthrough is worth your
time. Look for patterns across languages: three surprises about aliasing in
three languages is not three gaps, it is one missing model of value versus
reference semantics, and the dynamic array and vector walkthroughs in `build/`
are where that model is built.

The full course, which builds a working system chapter by chapter with the
same solution-and-walkthrough shape, starts at
[`paths/course/`](../content/courses/build-an-llm/paths/course/).
