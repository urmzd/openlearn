# Predict-Then-Read

Twenty-four short programs, six each in Go, Rust, Python, and TypeScript, and
for every one of them the exact output and a line-by-line explanation of why
the language produces it. Each snippet is aimed at a belief that is common,
load-bearing, and wrong: about `defer`, integer overflow, what a closure
captures, or what `this` is bound to.

This is one of the two parts of the [practice section](../README.md). The
`build` part explains how whole programs are put together. This part explains
the rules underneath them, one surprising line at a time.

> **Learning objectives**
> - Predict the exact output of short programs in Go, Rust, Python, and TypeScript, and name the rule behind each line.
> - Recognise the common wrong prediction for each snippet and the misconception that produces it.
> - Connect one underlying model (value versus reference, evaluation time, bytes versus characters) across languages.
> - Reproduce any output yourself, and compare what each runtime costs to start.

## Why predicting output builds a correct mental model

Reading a language spec produces recognition ("yes, that looks right"), which
is not the same as understanding. Committing to an exact expected output
before you look forces your model of the language into a form that can be
falsified. When it is wrong, you get a precise, memorable correction instead of
a vague feeling that you should reread the chapter.

The explanations are built for that moment. Each snippet opens with the rule:
the language semantics that decide its output, stated from first principles,
in terms of what the runtime actually stores and does. Then, for each output
line, the explanation applies that rule to the exact code in the snippet and,
where it matters, gives the common wrong prediction: what most people write
down, and the misconception that produces it.

That last part is the one to dwell on. A wrong prediction is almost never a
typo; it is a wrong model, and the same wrong model will produce wrong code.

## How to use these pages

Start by reading the snippet source; each entry links it. Then predict the
full output, line by line, and write it down: exact text, not "roughly this".
Only then open the answer (the folded "Output and explanation" block) and
compare. For every line you missed, read the explanation until you can state
the rule in one sentence without looking. If you can also explain why the
common wrong prediction is wrong, you have the model.

Go slowly. The value is in the minute you spend deciding what you think
happens, and a prediction you change after seeing the answer tells you
nothing.

## In this part

Each language has one page of worked answers, and the snippets live beside it:

- [Predicting Go output](go/README.md): loop variables, nil interfaces, slice
  aliasing, `defer`, strings and runes, and channels. Snippets in
  [`go/predict/`](../../../../practice/predict/go/predict/).
- [Predicting Rust output](rust/README.md): shadowing, drop order, iterator
  laziness, integer overflow, `String` versus `&str`, and moves with `Rc`.
  Snippets in [`rust/predict/src/bin/`](../../../../practice/predict/rust/predict/src/bin/).
- [Predicting Python output](python/README.md): mutable defaults, late-binding
  closures, identity versus equality, aliasing, generators, and attribute
  lookup. Snippets in [`python/predict/`](../../../../practice/predict/python/predict/).
- [Predicting TypeScript output](ts/README.md): number precision, `sort` and
  array holes, equality coercion, `this`, type erasure, and async ordering.
  Snippets in [`ts/predict/`](../../../../practice/predict/ts/predict/).

You can read the languages in any order; the pairings described after the
tables below are a good reason to read across them rather than one at a time.

## The corpus

The tables below list every snippet by its file name, with a hint at the trap it
sets, and link straight to its worked answer.


| # | Go | Rust |
|---|-----|------|
| 01 | [`loop-var-capture`](go/README.md#loop-variables-and-closures) (Go 1.22 changed the answer) | [`shadowing-and-blocks`](rust/README.md#shadowing-and-block-values) |
| 02 | [`nil-interface`](go/README.md#nil-interfaces) (typed nil is not nil) | [`drop-order`](rust/README.md#drop-order) (including `let _ =`) |
| 03 | [`slice-aliasing`](go/README.md#slice-aliasing-and-append) (when append shares, when it copies) | [`iterator-laziness`](rust/README.md#iterator-laziness) |
| 04 | [`defer-order`](go/README.md#defer-ordering) (LIFO, arg evaluation, named returns) | [`integer-overflow`](rust/README.md#integer-overflow-and-casts) (wrap, saturate, `as`) |
| 05 | [`strings-and-runes`](go/README.md#strings-bytes-and-runes) (bytes vs runes vs characters) | [`string-vs-str`](rust/README.md#owned-and-borrowed-strings) (byte-indexed slicing) |
| 06 | [`channels-and-close`](go/README.md#channels-and-close) (closed channels never block) | [`move-and-rc`](rust/README.md#moves-and-reference-counting) (watching the refcount) |

| # | Python | TypeScript |
|---|--------|------------|
| 01 | [`mutable-default-arg`](python/README.md#mutable-default-arguments) (evaluated once, at `def`) | [`number-precision`](ts/README.md#number-precision) (everything is a double) |
| 02 | [`late-binding-closures`](python/README.md#late-binding-closures) | [`array-sort-and-holes`](ts/README.md#array-sort-and-holes) (lexicographic by default) |
| 03 | [`identity-vs-equality`](python/README.md#identity-versus-equality) (`is` asks the allocator) | [`equality-coercion`](ts/README.md#equality-and-coercion) |
| 04 | [`aliasing-and-copies`](python/README.md#aliasing-and-copies) (`[[0]*3]*3`) | [`this-binding`](ts/README.md#how-this-is-bound) |
| 05 | [`generators-and-exhaustion`](python/README.md#generators-and-exhaustion) (what `zip` eats) | [`types-are-erased`](ts/README.md#types-are-erased) |
| 06 | [`class-attrs-and-mro`](python/README.md#class-attributes-and-the-mro) (reads walk, writes do not) | [`async-ordering`](ts/README.md#async-ordering) (microtasks before timers) |

The cross-language pairings are the real payload. Aliasing shows up as Go
slice capacity, Python list multiplication, and Rust move semantics; the same
missing model of value versus reference produces all three. Overflow is a Rust
API choice, a TypeScript silent precision loss, and in Go a compile error you
never see. Closure capture is Go 1.22's per-iteration variable, Python's late
binding, and TypeScript's `this`. Strings are bytes in Go and Rust and both
languages make `len("héllo")` equal 6. Reading the pairs within an hour of each
other is what makes the connection stick.

## Running a snippet yourself

The outputs on these pages were produced by running the snippets, and every
snippet is deterministic: no map iteration order, goroutine interleaving,
timestamps, or addresses reach the output. You can reproduce any of them. Each
language keeps its snippets in the layout its toolchain expects, so no wrapper
is needed, as the following table shows.

| Language | Snippet path | Run it (from the language directory) |
|----------|--------------|--------------------------------------|
| Go | `go/predict/NN-name/main.go` | `go run ./predict/NN-name` |
| Rust | `rust/predict/src/bin/NN-name.rs` | `cargo run --quiet --bin NN-name` (inside `rust/predict/`) |
| Python | `python/predict/NN-name.py` | `python3 predict/NN-name.py` |
| TypeScript | `ts/predict/NN-name.ts` | `node predict/NN-name.ts` (Node 22.18+ strips types natively; older 22.x needs `--experimental-strip-types`) |

Node runs the TypeScript files by stripping the types, not by checking them.
Some snippets are deliberately unsound, because being unsound at runtime is
the lesson, so do not expect `tsc --noEmit` to accept all of them.

## What they cost

What a program prints is one half of a mental model; what it costs is the
other half, and usually the half made of folklore. These snippets all do
nothing (under twenty printed lines, no allocation worth the name), so timing
them measures the price of admission for each runtime. Measured as the minimum
wall time of several runs and the peak resident memory, with Go and Rust built
ahead of time so no compile step is counted:

| Runtime | Wall time | Peak RSS | What you are paying for |
|---------|-----------|----------|-------------------------|
| Rust | ~2ms | ~1.5MB | A static binary with no runtime to start |
| Go | ~3ms | ~4MB | The scheduler and GC coming up before `main` |
| Python | ~19ms | ~15MB | Interpreter startup plus stdlib import |
| Node | ~50ms | ~70MB | V8 reserving its heap |

Minimum, not mean, because every source of noise makes a run slower and never
faster. Peak, not average, because peak memory is what gets a process killed.
A 45x memory spread between the fastest and slowest way to print twelve lines
is worth knowing before you pick a language for something that starts a
process per request.
