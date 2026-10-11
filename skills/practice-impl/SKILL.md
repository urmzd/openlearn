---
name: practice-impl
description: "Author a NEW build exercise in the polyglot practice track: brief, reference implementation with solution markers, and a self-testing entry point. To merely practise an exercise that already exists, use the ss CLI instead, which needs no agent."
invoke: user
arguments:
  - name: args
    description: "<language> <exercise-number-or-name> — e.g., 'rust 03' or 'go worker-pool'"
---

# Practice Exercise Authoring

## First: you probably want the CLI, not this skill

Scaffolding a workspace to *practise in* is what `ol start` does, and it needs no
agent:

```bash
practice/bin/ol start build rust 03    # clone stubbed into .scratchpad/, open it
practice/bin/ol check build rust 03    # run it; exit code is the verdict
practice/bin/ol diff  build rust 03    # your attempt against the reference
```

If the exercise already exists under `practice/build/`, stop here and use that.
`practice/bin/ol list build <lang>` shows what exists.

## Use this skill to author an exercise that does not exist yet

Reference implementations are being filled in incrementally, so a language
README may list an exercise that has no directory. Creating one is the job here.

1. Parse language and exercise identifier from `{args}`.
2. Map the language to its directory. The tier prefix is part of the path:
   - **Systems**: c → `practice/build/systems/c/`, cpp → `practice/build/systems/cpp/`, rust → `practice/build/systems/rust/`, zig → `practice/build/systems/zig/`
   - **Cloud**: go → `practice/build/cloud/go/`, scala → `practice/build/cloud/scala/`, java → `practice/build/cloud/java/`
   - **General**: python → `practice/build/general/python/`, typescript → `practice/build/general/typescript/`
3. Read that language's `README.md` and find the exercise row by number or name.
   Extract its title, concepts, and difficulty stars.
4. Create `<lang-dir>/<NN>-<kebab-name>/` containing the three required pieces
   below.
5. Verify it: `practice/bin/ol verify build <lang>` must report `OK` for the new
   exercise. An exercise that has not been run is not finished.

## The three required pieces

### 1. `README.md`, the brief

Concepts and difficulty from the language README table, what the reader is
building, the contract as a table of functions, and a **What to notice** section.
That last section is the exercise's actual reason to exist: the two or three
things that are counterintuitive, the decision that has more than one defensible
answer, or the bug that is invisible until a specific input. Without it the
exercise is a chore rather than a lesson.

### 2. A reference implementation with solution markers

Wrap every region the reader is meant to write in begin/end markers, in that
language's line-comment or block-comment syntax:

```c
void vec_init(Vec *v) {
  /* SOLUTION-BEGIN */
  v->data = NULL;
  /* SOLUTION-END */
}
```

`ol start` replaces each marked region with a single `TODO` line, so the marker
line must contain nothing but comment punctuation and the marker itself. Prose
that merely mentions the marker by name is ignored, which is why this file can
show the markers without breaking.

Leave signatures, types, and declarations *outside* the markers. A stub that
does not compile because a type went missing teaches nothing.

### 3. A self-testing entry point

No test framework, in any language. One entry point that runs assertions and
exits nonzero on the first failure, because that exit code is the entire
contract `ol check` relies on. The naming that `ol` knows how to run:

| Language | Entry point | How ol runs it |
|----------|-------------|----------------|
| C | `main.c` plus `<thing>.c` | `cc -std=c11 -Wall -Wextra *.c` then run |
| C++ | `main.cpp` | `c++ -std=c++20 -Wall -Wextra *.cpp` then run |
| Rust | `main.rs` | `rustc --edition 2021` then run |
| Zig | `main.zig` | `zig run main.zig` |
| Go | `main.go` | `go run .` (ol writes a `go.mod` into the workspace) |
| Scala | `main.scala` | `scala-cli run .` |
| Java | `Main.java` | `javac` then `java Main` |
| Python | `main.py` | `uv run --no-project python main.py` |
| TypeScript | `main.ts` | `node main.ts`, native type stripping, no build step |

An exercise that needs something this table cannot express, an external crate or
a code generator or a linker invocation, ships an executable `run.sh` and that
wins over the table.

Keep the implementation and the tests in **separate files** wherever the
language allows it. Tests in the same file as the marked region invite editing a
test into passing, and the split is usually a lesson in itself (header versus
translation unit, module versus consumer).

## Rules the exercise must obey

- Standard library only, unless the brief explicitly says otherwise and explains
  why.
- Deterministic. Same output every run: no clocks, addresses, map iteration
  order, or unseeded randomness in anything asserted on.
- Assertions that a plausible wrong implementation actually fails. A test suite
  that a stub passes is worse than no test suite. Write at least one test that
  targets the specific bug the **What to notice** section describes.
- Clean under sanitizers where the language has them. For C and C++, verify with
  `-fsanitize=address,undefined` before considering it done.
