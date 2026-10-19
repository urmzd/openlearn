# Predict-Then-Read

Twenty-four short programs, six each in Go, Rust, Python, and TypeScript, and
for every one of them the exact output and a line-by-line explanation of why
the language produces it. Each snippet is aimed at a belief that is common,
load-bearing, and wrong: about `defer`, integer overflow, what a closure
captures, or what `this` is bound to.

This is one of the two parts of the [practice section](../README.md). The
`build` part explains how whole programs are put together. This part explains
the rules underneath them, one surprising line at a time.

## Why predicting output builds a correct mental model

Reading a language spec produces recognition ("yes, that looks right"), which
is not the same as understanding. Committing to an exact expected output
before you look forces your model of the language into a form that can be
falsified. When it is wrong, you get a precise, memorable correction instead of
a vague feeling that you should reread the chapter.

The explanations are built for that moment. For each output line they give:

1. **The rule**: the language semantics that decide the line, stated from first
   principles (what the runtime actually stores and does).
2. **Why it prints this**: the rule applied to the exact code in the snippet.
3. **The common wrong prediction**: what most people write down, and the
   misconception that produces it.

The third part is the one to dwell on. A wrong prediction is almost never a
typo; it is a wrong model, and the same wrong model will produce wrong code.

## How to use these pages

1. **Read** the snippet source (each entry links it).
2. **Predict** the full output, line by line, and write it down. Exact text,
   not "roughly this".
3. **Open the answer** (the folded "Output and explanation" block) and compare.
4. For every line you missed, read the explanation until you can state the rule
   in one sentence without looking. If you can explain why the common wrong
   prediction is wrong, you have the model.

Six snippets is roughly a 40-minute session, and going slower is better: the
value is in the minute you spend deciding what you think happens.

| Language | Worked answers | Snippets |
|----------|----------------|----------|
| Go | [`go/README.md`](go/README.md) | [`go/predict/`](go/predict/) |
| Rust | [`rust/README.md`](rust/README.md) | [`rust/predict/src/bin/`](rust/predict/src/bin/) |
| Python | [`python/README.md`](python/README.md) | [`python/predict/`](python/predict/) |
| TypeScript | [`ts/README.md`](ts/README.md) | [`ts/predict/`](ts/predict/) |

## The corpus

| # | Go | Rust |
|---|-----|------|
| 01 | [`loop-var-capture`](go/README.md#01-loop-var-capture) (Go 1.22 changed the answer) | [`shadowing-and-blocks`](rust/README.md#01-shadowing-and-blocks) |
| 02 | [`nil-interface`](go/README.md#02-nil-interface) (typed nil is not nil) | [`drop-order`](rust/README.md#02-drop-order) (including `let _ =`) |
| 03 | [`slice-aliasing`](go/README.md#03-slice-aliasing) (when append shares, when it copies) | [`iterator-laziness`](rust/README.md#03-iterator-laziness) |
| 04 | [`defer-order`](go/README.md#04-defer-order) (LIFO, arg evaluation, named returns) | [`integer-overflow`](rust/README.md#04-integer-overflow) (wrap, saturate, `as`) |
| 05 | [`strings-and-runes`](go/README.md#05-strings-and-runes) (bytes vs runes vs characters) | [`string-vs-str`](rust/README.md#05-string-vs-str) (byte-indexed slicing) |
| 06 | [`channels-and-close`](go/README.md#06-channels-and-close) (closed channels never block) | [`move-and-rc`](rust/README.md#06-move-and-rc) (watching the refcount) |

| # | Python | TypeScript |
|---|--------|------------|
| 01 | [`mutable-default-arg`](python/README.md#01-mutable-default-arg) (evaluated once, at `def`) | [`number-precision`](ts/README.md#01-number-precision) (everything is a double) |
| 02 | [`late-binding-closures`](python/README.md#02-late-binding-closures) | [`array-sort-and-holes`](ts/README.md#02-array-sort-and-holes) (lexicographic by default) |
| 03 | [`identity-vs-equality`](python/README.md#03-identity-vs-equality) (`is` asks the allocator) | [`equality-coercion`](ts/README.md#03-equality-coercion) |
| 04 | [`aliasing-and-copies`](python/README.md#04-aliasing-and-copies) (`[[0]*3]*3`) | [`this-binding`](ts/README.md#04-this-binding) |
| 05 | [`generators-and-exhaustion`](python/README.md#05-generators-and-exhaustion) (what `zip` eats) | [`types-are-erased`](ts/README.md#05-types-are-erased) |
| 06 | [`class-attrs-and-mro`](python/README.md#06-class-attrs-and-mro) (reads walk, writes do not) | [`async-ordering`](ts/README.md#06-async-ordering) (microtasks before timers) |

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
is needed.

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
