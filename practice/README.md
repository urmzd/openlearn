# Practice

The rest of this repository is the **learning path**: tracks you read, in
order, with a textbook behind each topic. This directory is the **practice
path**. Nothing here teaches you anything. Everything here tells you whether
you actually know it.

The two are not interchangeable and they fail differently. Reading produces
recognition, which feels like knowledge and is not. Practice produces a verdict,
which is uncomfortable and is the only thing that transfers.

```bash
practice/bin/ol learn          # role paths: the learning path in order, for one job
practice/bin/ol where          # what exists, and where it lives
practice/bin/ol list           # every exercise, and which ones you have started
```

## Three kinds of exercise

They differ in what you produce, and that is the only thing they differ in. One
CLI drives all three with the same verbs.

| Kind | You produce | It is wrong when |
|------|-------------|------------------|
| [`predict`](predict/) | The exact output you expect from a snippet | Your prediction differs from what ran |
| [`build`](build/) | An implementation from scratch | The exercise's own assertions fail |
| [`reattempt`](reattempt/) | A second solution to something you already solved | The original's assertions fail |

`predict` finds holes fastest, because committing to an output is cheap and the
correction is precise. `build` finds different holes, the ones that only appear
when you have to make every decision yourself. `reattempt` is the honest one:
code you wrote before, handed back with the solution removed, which is how you
find out whether you learned it or just finished it.

## The loop

```bash
ol start build rust 01     # clone it into .scratchpad/, stubbed, and open it
ol check build rust 01     # run it; the exit code is the verdict
ol diff  build rust 01     # your attempt against the reference
```

`ol start` never touches the repository. It copies the exercise into
`.scratchpad/` at the repo root, which is gitignored in its entirety, and
strips out the parts you are supposed to write. Your attempts are yours: they
are never committed, and nothing in the repo changes as you work.

Add `practice/bin` to your `PATH` and the `practice/bin/` prefix goes away.

| Command | Does |
|---------|------|
| `ol start <kind> <lang> <id>` | Clone the exercise into `.scratchpad/` and open it |
| `ol check <kind> <lang> [id]` | Run your attempt. Exit code is the verdict |
| `ol diff <kind> <lang> <id>` | Your attempt against the reference |
| `ol list [kind] [lang]` | What exists, and what you have started |
| `ol score [kind] [lang]` | Passing, failing, not attempted |
| `ol show <kind> <lang> <id>` | Print the reference. Spoils the exercise |
| `ol reveal <kind> <lang> <id>` | Run the reference. Also spoils it |
| `ol reset <kind> <lang> <id>` | Throw your attempt away and start cold |
| `ol where` | Paths and exercise counts |
| `ol verify [kind] [lang]` | Maintainer check, described below |
| `ol bench [lang] [--assert]` | Cost of each predict snippet against `predict/budgets.tsv` |

`<id>` is the numeric prefix, so `ol check build c 02` is enough. Languages
accept the spelling you would expect: `ts` and `typescript` both work.

## How an exercise is stored

One committed source of truth per exercise, never a reference and a separate
stub that drift apart. The reference marks the regions you are meant to write,
and `ol start` replaces each marked region with a `TODO` on the way into your
scratchpad.

That means the same file is the thing you diff against, the thing CI runs, and
the thing your stub is generated from. Tests, headers, and scaffolding carry no
markers, so they arrive intact and you cannot accidentally edit a test into
passing.

Every `build` exercise is self-testing: one entry point, its own assertions,
nonzero exit on the first failure. No test framework in any of the nine
languages, which is why `ol check` is the same command everywhere. The
[case studies](../content/case-studies/) already worked this way and CI already depended
on it.

## Build exercises

Ten per language, ordered by concept rather than by difficulty, so read the
stars and not the numbers.

Standard library only unless an exercise says otherwise. The point is to write
the thing, not to find the crate that already did.

### Core

The languages this curriculum actually targets. `ol verify build` with no
argument covers exactly these, and so does CI.

| Language | Focus | References |
|----------|-------|-----------|
| [Go](build/cloud/go/) | Concurrency, channels, interfaces | 0/10 |
| [Rust](build/systems/rust/) | Ownership, lifetimes, zero-cost abstractions | 0/10 |
| [Python](build/general/python/) | Data model, metaprogramming, asyncio | 0/10 |
| [TypeScript](build/general/typescript/) | Type narrowing, generics, runtime safety | 0/10 |
| [C](build/systems/c/) | Manual memory, pointers, undefined behaviour | 10/10 |
| [C++](build/systems/cpp/) | RAII, templates, move semantics | 9/9 |

### Optional

Kept because the exercises are worth doing, not because the languages are in
use here. Every command works on them by name, but they are left out of the
bare fan-out so a toolchain you have not installed is never a failure in a run
you did not ask for.

| Language | Focus | Toolchain |
|----------|-------|-----------|
| [Zig](build/systems/zig/) | Comptime, explicit allocators, C interop | `brew install zig` |
| [Scala](build/cloud/scala/) | FP plus OOP, type system, given instances | `brew install scala-cli` |
| [Java](build/cloud/java/) | JVM internals, virtual threads, generics | `mise` plus a `.mise.toml` pin |

On a chezmoi-managed machine all three come from the `install_alt_langs` flag,
which is off by default.

```bash
ol verify build              # core only
ol verify build zig          # by name, and skips cleanly if zig is absent
ol where                     # which languages are core, optional, or missing a toolchain
```

**Reference implementations are being filled in incrementally.** `ol list build`
shows what is ready right now; an exercise with no directory yet is a table row
in its language README and nothing more.

## Predict exercises

Twenty-four snippets across Go, Rust, Python, and TypeScript, six each, every
one aimed at a belief that is common, load-bearing, and wrong. Full detail in
[`predict/README.md`](predict/README.md), including why the cross-language
pairings are the real payload.

Two rules make it work: commit before you run, and log every miss in
[`predict/LEDGER.md`](predict/LEDGER.md). A prediction edited after seeing the
output is worth nothing, and a miss you do not write down is one you will
repeat.

## Verifying the exercises

`ol check` validates you. `ol verify` validates the exercises, and it is what
runs in CI:

| Kind | Verified by |
|------|-------------|
| `predict` | Every snippet builds, exits zero, prints something, and prints the identical something three runs running |
| `build` | Every reference passes its own assertions and carries markers so it can be stubbed |
| `reattempt` | Every manifest row points at a file that exists, carries markers, and still runs clean |

It prints no exercise output, only shapes and counts, so it is safe to run on a
fresh checkout without spoiling a single exercise.

```bash
just run-predict           # ol verify predict
just run-build             # ol verify build, for every installed toolchain
just run-predict-bench     # ol bench --assert
```

## Course modules

The course (start at [`paths/course/`](../content/courses/build-an-llm/paths/course/); design:
[`course/DESIGN.md`](../course/DESIGN.md)) runs on the
same `ol`. A verb whose first argument is a course id (`M03.1`, `L8.3`,
`rt.01`, `lang.02`, `S-M07a`, `sq.multi-lora`, `MS-P1`) goes to the course
harness; the practice kinds never match that grammar, so every command above
keeps its meaning, including `ol bench [lang]`. The heavy lifting is
`python -m olcourse` in the uv project [`course/harness/`](../course/harness/),
which needs only `uv` (stdlib plus PyYAML and SymPy at run time; the root `uv.lock` is untouched).

Instead of a scratchpad copy per exercise, the course gives you one git repo
that grows into a whole system. Each module owns whole source files ("units")
in it, and checks run against your own earlier modules.

```bash
ol course init --name forge   # your repo, at .scratchpad/course/ (or OL_COURSE_HOME)
ol next                       # the next stage of content/courses/build-an-llm/paths/course whose deps pass
ol start M03.1                # stub the module's units into your repo
ol tests M03.1                # what each course test checks, and why
ol check M03.1                # the exit code is the verdict
ol status                     # every module: todo, started, pass, stale, assisted, spoiled, self
```

| Command | Does | Exit codes |
|---------|------|-----------|
| `ol course init --name <system> [--at DIR]` | Create your repo: `system.toml`, `.gitignore`, vendored `contracts/` (with `VERSION`), `git init` | 0, 5 |
| `ol start <ID>` | Write compiling stubs of the module's units, never overwriting a file. Library manifests come along only when absent. A Rust crate root's `mod`s and a Go package's sibling units get stubs too, so your crate and package always compile. For a unit the module takes over (`upgrades`), prints the contract diff instead | 0, 5 |
| `ol check <ID>` | Contract pre-check, smoke tests of every dependency you built, then the course tests through the overlay, then (with `[learner_tests]`) the red-then-green journal and the mutation grade of your tests; for a solve set, the answer checker and the proof rubric | 0 pass, 1 fail, 2 not started, 3 blocked by deps, 4 contract drift, 5 harness or toolchain |
| `ol check <ID> --ref-deps[=all\|ID,...]` | Use the hidden reference for unfinished (or named, or all) deps; the verdict is `assisted` | as above |
| `ol check <ID> --no-cumulative --kind K --json --seed N` | Skip the dependency smoke tests; run only tests of one KIND; machine output; seed | as above |
| `ol check --all [--ci] [--fresh]` | Every started module in pass order, each after its deps; `--ci` forbids `--ref-deps` and runs practice checks with `OL_SMOKE=1` (no cluster tier). A module whose last verdict is a pass on exactly the current files (every owned unit in the repo, the same course commit and harness, no reference code) is reported from that verdict, and a dependency that passed on these files skips its cumulative smoke rerun; `--fresh` checks everything | worst code |
| `ol tests <ID>` | The annotated test catalog: name, KIND, WHY, smoke tests | 0 |
| `ol diff <ID> [--spoil]` | Your units against the reference, after a pass (before one, only with `--spoil`) | 0, 1 |
| `ol show <ID>` / `ol reveal <ID>` | Print the reference; recorded as `spoiled` | 0 |
| `ol reset <ID> [--force]` | Restore the stubs; refuses on uncommitted changes without `--force` | 0, 1 |
| `ol status [--graph\|--counts\|--json]` / `ol next` | State of every module / the next path stage | 0 |
| `ol contracts sync [--to REV]` | Re-vendor `contracts/` at the current openlearn (or REV) | 0 |
| `ol lint [ID..] [--links] [--fix-index]` | Registry invariants, chapter contract, links, no em dashes; `--fix-index` rewrites `course/modules.tsv` and the `## Chapters` tables | 0, 1 |
| `ol mutate <ID> [-j N] [--reveal-survivors]` | The mutation grade of your `[learner_tests]`: they run against the reference with one planted fault per mutant; cached by test, unit, and patch hash | 0, 1, 5 |
| `ol tdd red\|green <ID>` | Rung R3 and up: your tests must fail against your current code, then pass with the same test files; `ol check` requires the red record | 0, 1 |
| `ol milestone <MS-ID> [--smoke] [--ref-deps] [--step NAME] [--seed N]` / `ol milestone list` | Run a milestone through your `system.toml` entry points: `[build]`, then your services on allocated ports, then the steps. Maintainers: `--record-thresholds [--seeds 5]` writes `calibrated` bars | 0 pass, 1 fail or incomplete, 3 blocked, 5 |
| `ol conform openapi[:v0\|v1\|v2][:engine\|gateway][:smoke] [--target T] [--base URL]` | OpenAPI conformance against your service (started for you) or a URL; the gateway tier also runs against a recording fake upstream | 0, 1, 5 |
| `ol parity [<suite>..] [--fuzz] [--ref]` | Every implementation of an algorithm against one golden oracle (or each other on generated inputs) | 0, 1, 5 |
| `ol fetch <asset>.. [--verify]` / `ol fetch --list` | Pinned large assets from `course/fixtures/ASSETS.tsv` into the cache, size and sha256 checked | 0, 5 |
| `ol bench --calibrate [--in-cluster]` / `ol bench <ID>\|course [--assert]` | Time this machine (or a Job in your kind namespace); course perf budgets relative to it | 0, 1, 5 |
| `ol drill list\|start <name> [--seed N]\|status\|end\|reset` / `ol drill run <name> --respond` | Inject faults behind a safety gate (cluster injectors) or onto a scratch-copy branch (`git-branch`, `contract-bump`); grade detection, resolution, postmortem; undo from the journal; `run --respond` is CI's scripted responder | 0, 1, 3, 5 |
| `ol export <DIR> [--remote URL] [--allow-incomplete]` | Clone your repo with its history and vendor the course tests of every passed module, with test glue | 0, 1 |
| `ol doctor [--pass N] [--json]` | The toolchain each pass needs, Docker's CPU and memory from Pass 7 | 0, 5 |
| `ol course ci [--upstream URL]` | Print the learner CI recipe (below) | 0 |
| `ol verify course [ID..] [--changed REF] [--global] [--nightly] [--e2e\|--kind [MS-ID..] [--keep DIR]] [--assemble DIR]` | Maintainer checks 1 to 14 of course/DESIGN.md 5.14; `--e2e` runs a learner assembled from `course/ref` end to end, `--kind` adds the kind steps against a deployed reference (milestone ids limit either to those milestones), `--assemble DIR` only builds that learner. `OL_MUTATION_CACHE=<file>` shares mutation results across runs (CI keeps it between jobs) | 0, 1 |

**Grading your tests.** A module with `[learner_tests]` (rung R2 and up)
grades the tests you write, not your code: they run against the reference
with one planted fault at a time (`course/mutants/<ID>/`), and the score is
the share of faults they catch. The tests may touch only the contract (Python
imports names in `contracts/py`, Go tests are `package <pkg>_test`, Rust tests
are integration tests, C tests include only `tinyllm/*.h` and `ol_*.h`).
`ol check` uses the full grade `ol mutate` cached for your current test files,
or runs the required mutants plus a seeded sample of 8 and calls it an
estimate. A surviving semantic mutant shows only its Pitfall number until the
module passes or `--reveal-survivors` (recorded as spoiled). Perf, model,
agent, and resilience mutants (rungs R7 to R10) are graded by your benchmark
gate, a 5-seed permutation test on your eval metric, non-overlapping 95% CIs,
and your fault suite.

**Solve sets.** `ol start S-M07a` writes `solve/S-M07a.toml` with one table
per question (lettered parts are `[q3.a]`); answers are ASCII math (`x^2`,
`[1, 3) U (5, oo)`, `{1, 2}`, `[[1, 2], [3, 4]]`). `ol check` compares them
with SymPy in a subprocess (5 s per answer) and never shows the expected
answer; proofs are self-graded against their rubric (`course/rubrics/`), y or
n per line, and the verdict is tagged `self`.

**Milestones run your entry points.** `ol milestone` reads `system.toml`
(course/DESIGN.md 2.16): it runs `[build].steps`, starts the `[services.*]` the
steps need in `after` order, each on ports the OS hands out, with a generated
`runtime.toml` (your `config` template with `{port}`, `{health_port}`, and
`{<service>.port}` filled in, the listen keys forced to those ports, and the
same values as `TL_<SECTION>__<KEY>` variables), waits for each `health` URL,
runs the steps, and tears everything down. Logs land in
`.ol/milestones/<MS-ID>/<timestamp>/`. `--smoke` runs only the `smoke = true`
steps that need no cluster (what PR CI runs); a full run executes
`ci = "kind"` steps against `[deploy]` and records `incomplete` when that
cluster is unreachable. A pass gate `MS-P<n>` also reruns the smoke steps of
every earlier gate.

**Drills are gated.** `ol drill start` refuses unless kubectl's current context
equals `[deploy].kube_context`, that context starts with `kind-` or `k3d-`, and
`[deploy].namespace` exists. Every action carries `--context` and `-n`, every
injection writes its undo to `.ol/drills/<run>/journal.jsonl`, and a failed
start undoes what it did. There is no override flag.

**Your own CI.** Your repo is pushed to its own remote, where openlearn is
absent. `ol course ci` prints the steps for your CI file: clone public
openlearn into `.ol/openlearn`, check out the sha in `contracts/VERSION`,
and run `OL_COURSE_HOME=$PWD .ol/openlearn/practice/bin/ol check --all --ci`.
With nothing started it is trivially green; `--ci` never uses `--ref-deps`.

**Where things come from.** Course tests, references, and fixtures always come
from the openlearn commit named in your `contracts/VERSION`: the live
checkout when that is HEAD, otherwise a cached worktree in
`~/.cache/openlearn/worktrees/<sha>`. A maintainer's edit cannot change your
verdicts until you run `ol contracts sync`. Editing `contracts/` yourself is
drift (exit 4).

**The overlay.** A check builds under `<your repo>/.ol/`, never in your files
or in openlearn: Python runs in your own uv environment with the reference or
stub of each non-learner unit shadowing yours on `PYTHONPATH`; C compiles one
object per unit (yours, the reference, or a stub) into an ASan and UBSan test
binary for standalone C tests; Rust and Go use copy farms with
generated manifests (`.ol/rust-farm`, `.ol/overlay/<ID>/go`). Every C test
binary installs a counting allocator through `tl_set_allocator`, so a leak
fails the test on every platform. Verdicts land in `.ol/verdicts.jsonl`.

**Path stages with checks.** A `path.tsv` row may carry a fifth column,
`module:<ID>`, `solve:<ID>`, `milestone:<ID>`, `drill:<ID>`, `conform:<suite>`,
or `all:<ID>,<ID>`. `ol learn <path> --done <stage>` marks such a stage only
when its check passes (running it if there is no verdict yet);
`--force` marks it anyway and logs that. `ol learn <path>` shows `[x]`, `[~]`
(assisted, self-graded, spoiled, or forced), or `[ ]`.

**Harness self-tests.** A miniature course with one sample module per language
lives in [`bin/tests/fixtures/site/`](bin/tests/fixtures/site/), with a small
reference system (a byte bigram engine, a gateway, a CLI, and `system.toml`
under `course/ref/entry/`), milestones, a drill, and an OpenAPI v0 contract.
The tests drive `ol` against it end to end (start, stub compiles and fails,
reference passes, verdicts, upgrades, drift, verify, lint, milestones, conform,
drills against a fake `kubectl`, export, the CI recipe, `verify --e2e`):

```bash
uv run --project course/harness pytest course/harness/tests practice/bin/tests
```

| Variable | Default | Points ol at |
|----------|---------|--------------|
| `OL_COURSE_HOME` | `.scratchpad/course/` | your course repo |
| `OL_COURSE_ROOT` | `course/` | the live course tree |
| `OL_PATHS_DIR` | `content/courses/build-an-llm/paths/` | the learning paths |
| `OL_CACHE` | `~/.cache/openlearn` | course-tree worktrees, the verify venv |
| `OL_GO_RACE` | `1` | `0` drops `-race` from Go course tests |
| `OL_TSAN` | `1` | `0` skips the ThreadSanitizer build of modules with `sanitize = ["thread"]` |

**The testkit.** [`course/testkit/`](../course/testkit/) is the fault and
determinism kit course tests (and your graded tests) import: in Go
(`openlearn.urmzd.com/tl/testkit`: `clock`, `failpoint`, `effects`,
`proc.KillLoop`, `chaosproxy`, `otlpsink`, `promscrape`, `faketool`), Python
(`sstestkit`: `flakyhttp`, `failpoint`, `clock`), Rust (`tl-testkit`:
failpoints and a fake clock), and C (`tinyllm/failpoint.h`). Failpoints share
one spec: `TL_FAILPOINTS="name=crash;other=error(msg);x=3*sleep(20ms)"`.

## Picking something

Start with `predict`. It costs nothing to set up, the corpus is complete, and
the diff between what you expected and what happened tells you which `build`
exercise is worth your time. Then use the patterns in your ledger to choose the
language: three entries about aliasing in three languages is not three misses,
it is one missing model of value versus reference semantics, and that is a
`build` exercise waiting to happen.
