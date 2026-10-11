# Predict-Then-Run

A harness for the one drill that actually finds holes in your mental model of a
language: **write down what you think the code prints, then run it.** The diff
between your prediction and reality is the only signal here. Nothing in this
directory grades style, teaches syntax, or rewards effort.

This is one of the three exercise kinds on the [practice path](../README.md).
The `build` kind asks you to *build* things. This one asks you to *predict*
things, which is a different and much faster way to find out that you were wrong
about `defer`, about integer overflow, or about what `this` is bound to.

## Why prediction and not reading

Reading a language spec produces recognition ("yes, that looks right"), which is
not the same as recall. Committing to an exact expected output before running
forces the prediction into a form that can be falsified. When it is wrong you get
a precise, memorable correction instead of a vague feeling that you should reread
the chapter.

Two rules make this work:

1. **Commit before you run.** A prediction you edit after seeing the output is
   worth nothing.
2. **Log every miss.** A miss you do not write down is a miss you will repeat.
   `LEDGER.md` is where they go.

## The harness

The harness is [`../bin/ol`](../bin/ol), a single bash script with no
dependencies beyond the language toolchains themselves. It drives the other
exercise kinds too; these are the verbs that matter here.

| Command | What it does |
|---------|--------------|
| `ol list predict [lang]` | Show snippets and whether you have predicted them |
| `ol show predict <lang> <id>` | Print a snippet's source |
| `ol start predict <lang> <id>` | Show the source, then open your prediction file in `$EDITOR` |
| `ol check predict <lang> [id]` | Run for real and diff your prediction against actual output |
| `ol score predict [lang]` | Tally passed, missed, and not attempted |
| `ol reveal predict <lang> <id>` | Just run it, burning the exercise |
| `ol reset predict <lang> <id>` | Delete your prediction and start over |
| `ol verify predict [lang]` | Maintainer check: every snippet builds, runs, and prints the same thing three times in a row |
| `ol bench [lang] [--assert]` | Wall time and peak memory per snippet, against the ceilings in `budgets.tsv` |

`<id>` is the numeric prefix of a snippet, so `ol check predict go 03` is
enough.

Your predictions are written to `.scratchpad/predict/<lang>/<snippet>.txt` at
the repo root. They are yours, the whole `.scratchpad/` directory is gitignored,
and the harness never looks at them until you run `ol check`.

## A session

```bash
ol list predict go              # what is here, and what you have attempted
ol start predict go 03          # read the source, write the exact expected output
ol check predict go 03          # PASS, or a diff of prediction against reality
ol score predict                # where you stand across all four languages
```

A miss prints a unified diff and one instruction: record it in
[`LEDGER.md`](LEDGER.md). Six snippets is roughly a 40-minute session, and going
slower is better, because the value is entirely in the minute you spend deciding
what you think happens.

`ol check` runs the real toolchain. First contact with Rust or Go pays for a
compile, so `ol verify predict <lang>` (below) is a reasonable way to warm the
cache before a session.

## Where snippets live

Each language keeps its snippets in whatever layout its toolchain expects, so
that `go run`, `cargo run`, and friends work with no wrapper.

| Language | Snippet path | How the harness runs it |
|----------|--------------|-------------------------|
| Go | `go/predict/NN-name/main.go` | `go run ./predict/NN-name` |
| Rust | `rust/predict/src/bin/NN-name.rs` | `cargo run --bin NN-name` |
| Python | `python/predict/NN-name.py` | `uv run python predict/NN-name.py` |
| TypeScript | `ts/predict/NN-name.ts` | `node predict/NN-name.ts` (native type stripping, Node 22+) |

Every snippet must be **deterministic**. Output that varies between runs (map
iteration order, goroutine interleaving, timestamps, addresses) makes prediction
impossible and is a bug in the snippet, not a lesson. `ol verify` is what
enforces that.

Node runs the TypeScript files by stripping the types, not by checking them.
Some snippets are deliberately unsound, because being unsound at runtime is the
lesson, so do not expect `tsc --noEmit` to be happy with all of them.

## The corpus

Twenty-four snippets, six per language, each aimed at a belief that is common,
load-bearing, and wrong.

| # | Go | Rust |
|---|-----|------|
| 01 | `loop-var-capture` (Go 1.22 changed the answer) | `shadowing-and-blocks` |
| 02 | `nil-interface` (typed nil is not nil) | `drop-order` (including `let _ =`) |
| 03 | `slice-aliasing` (when append shares, when it copies) | `iterator-laziness` |
| 04 | `defer-order` (LIFO, arg evaluation, named returns) | `integer-overflow` (wrap, saturate, `as`) |
| 05 | `strings-and-runes` (bytes vs runes vs characters) | `string-vs-str` (byte-indexed slicing) |
| 06 | `channels-and-close` (closed channels never block) | `move-and-rc` (watching the refcount) |

| # | Python | TypeScript |
|---|--------|------------|
| 01 | `mutable-default-arg` (evaluated once, at `def`) | `number-precision` (everything is a double) |
| 02 | `late-binding-closures` | `array-sort-and-holes` (lexicographic by default) |
| 03 | `identity-vs-equality` (`is` asks the allocator) | `equality-coercion` |
| 04 | `aliasing-and-copies` (`[[0]*3]*3`) | `this-binding` |
| 05 | `generators-and-exhaustion` (what `zip` eats) | `types-are-erased` |
| 06 | `class-attrs-and-mro` (reads walk, writes do not) | `async-ordering` (microtasks before timers) |

The cross-language pairings are the real payload. Aliasing shows up as Go slice
capacity, Python list multiplication, and Rust move semantics; the same missing
model of value versus reference produces all three. Overflow is a Rust API
choice, a TypeScript silent precision loss, and in Go a compile error you never
see. Predicting them within an hour of each other is what makes the connection
stick.

## Validating the corpus

`ol check` validates *you*. `ol verify` validates the *snippets*, and it is the
one that runs in CI:

```bash
$ ol verify predict go
OK    go/01-loop-var-capture (4 lines, stable over 3 runs)
OK    go/02-nil-interface (7 lines, stable over 3 runs)
...
```

For each snippet it builds the language's targets, then runs the program three
times and requires that it exits zero, prints something, and prints the identical
something every time. A snippet that fails any of those is unusable as an
exercise no matter how good the lesson inside it is.

It deliberately prints no snippet output, only line counts, so you can run it on
a fresh checkout without spoiling a single exercise.

## What it costs

`ol check` asks what a program prints. `ol bench` asks what it costs, which is
the other half of a mental model and usually the half made of folklore.

```bash
$ ol bench rust
rust
  01-shadowing-and-blocks             2ms     1504KB  ok
  02-drop-order                       3ms     1488KB  ok
  ...
```

Wall time is the *minimum* of five runs, because every source of noise makes a
run slower and never faster. Memory is the *peak* RSS across those runs, because
peak is what gets a process killed. [`../bin/measure.py`](../bin/measure.py) does the
measuring and normalises the platform difference in `ru_maxrss` (bytes on macOS,
kilobytes on Linux) that would otherwise be a factor-of-1024 error. On Linux,
it requires GNU `/usr/bin/time` (the `time` package): measuring the Python
harness's immediate child includes inherited launcher memory and falsely
inflates small native programs. GNU time measures its own child after the
Python launcher has been replaced. Wall time includes this small wrapper
overhead; the memory ceilings remain unchanged.

Two rules keep the numbers honest:

1. **Measure the program, not the toolchain.** Go and Rust are built ahead of
   time so the measured run contains no compile step. Python and TypeScript have
   no build step, so their interpreter startup is a real part of the cost and is
   left in.
2. **Compare against a ceiling, not against each other.** A breach means
   something regressed. It does not mean one language beat another.

That said, the spread is the most useful thing in this directory, because these
snippets all do nothing. Under twenty printed lines, no allocation worth the
name, so what you are looking at is the price of admission for each runtime:

| Runtime | Wall time | Peak RSS | What you are paying for |
|---------|-----------|----------|-------------------------|
| Rust | ~2ms | ~1.5MB | A static binary with no runtime to start |
| Go | ~3ms | ~4MB | The scheduler and GC coming up before `main` |
| Python | ~19ms | ~15MB | Interpreter startup plus stdlib import |
| Node | ~50ms | ~70MB | V8 reserving its heap |

A 45x memory spread between the fastest and slowest way to print twelve lines is
worth knowing before you pick a language for something that starts a process per
request.

[`budgets.tsv`](budgets.tsv) holds the ceilings, generous enough that CI runner
variance never trips them:

```bash
ol bench --assert          # nonzero exit on a breach; runs in CI
just run-predict-bench     # the same thing from the repo root
```

Ceilings are uniform within a language on purpose. The spread between snippets
in the same language is measurement noise, and a per-snippet number would imply a
precision that is not there.
