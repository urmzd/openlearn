# openlearn

A Wails desktop app that renders course packs: it reads a pack's documents and role paths,
runs its exercises with its own engine, generates timed exercises from a chapter
with a hidden rubric, coaches while the user works, and scores the result. Beta: prompts
and scoring calibration are still moving.

## Architecture

| Package | Role |
|---------|------|
| `internal/course` | A course pack: manifest, outline, role paths, progress files, search. Reads `path.tsv` and `.done` in the format `ol learn` owns. |
| `internal/packs` | Finds packs (checkout, linked folder, imported zip, built in) and opens one |
| `internal/builtin` | The packs embedded in the binary, filled by `cmd/bundle` (`make content`) |
| `internal/practice` | The exercise engine: lists a pack's exercises from its spec, clones one with the solution stripped, and checks an attempt by running it |
| `internal/workspace` | One open pack with its progress and exercises. The app and the MCP server both open one. |
| `internal/mcpserver` | The pack over MCP, for whatever agent the learner uses. `openlearn mcp` and `cmd/openlearn-mcp` run it. See `docs/mcp.md`. |
| `internal/domain` | Data model, JSON-tagged because everything crosses into TypeScript. Owns the fixed skill taxonomy. |
| `internal/harness` | The three LLM engines: `problem.go` (generation), `coach.go` (live coaching), `review.go` (scoring) |
| `internal/agui` | AG-UI event vocabulary, the typed `Bus`, and the `Emitter` seam |
| `internal/session` | `Runtime`: clock, curveball timeline, coach loop, review transition |
| `internal/store` | Settings and session JSON under `~/.openlearn`, plus profile aggregation |
| `internal/llm` | Provider construction over saige, and `Structured[T]` generation |
| `app.go`, `course.go` | The Wails binding surface. Every exported method on `App` becomes callable from TS. |
| `frontend/src` | React, Tailwind, shadcn/ui. `Workbench.tsx` hosts the views; `agui.ts` reduces the session event stream. See `docs/ui.md`. |

## Non-obvious invariants

**Exercises run in-process, from a spec.** `internal/practice` is the only implementation
of list, start, check, diff, and reset. A pack describes its exercises in a JSON spec
(`practice/exercises.json` here) and the engine does the rest. Do not shell out to a pack
CLI for any of it: that was the old design, and it failed whenever the CLI moved. The one
process the engine starts is the compiler or interpreter an attempt needs, from an argument
list in the spec, with no shell.

**`path.tsv` and `.done` are shared with `ol learn`.** `internal/course` reads and writes
them in the format `practice/bin/ol` uses, so the terminal and the app agree. A stage tied
to a check is only marked through the pack's `stageCheck` command, looked up when it is
needed, because only the pack's own tooling knows whether the check passed.

**Every exported method on `App` is a frontend binding.** Anything else that needs to be
exported (the asset handler, say) goes on its own type.

**There is no assistant in the app.** Tutoring belongs to whatever agent the learner
connects over MCP. Do not add a chat panel or a model call for answering questions; add a
tool to `internal/mcpserver` instead. The only model calls the app makes are the timed
session's: generation, the coach, and the review.

**An agent gets the learner's verdicts, never its own.** No MCP tool sets a verdict or
forces a checked stage. `check_exercise` and `mark_stage` go through `internal/workspace`,
the same path the app's buttons use.

**Pack content is untrusted.** A pack can be imported from a zip. Markdown is sanitized
before rendering, zip entries are checked against path escape and a size budget, and names
used as exercise names are validated as plain names. Keep those checks when touching
`Markdown.tsx`, `internal/packs`, or `internal/practice`.

**`wails dev` needs the toolchain `go.mod` declares.** Wails 2.11 cannot type-check packages
built by Go 1.27. The Makefile pins `GOTOOLCHAIN`; run wails through `make`.

**Rubric criteria must map onto `domain.Area`.** The taxonomy in `taxonomy.go` is fixed
and per-mode. Criteria whose area is outside it are dropped at assembly time. This is what
makes scores comparable across sessions: per-problem rubric wording changes every run,
area IDs do not. Adding an area is fine; renaming one silently orphans existing profile
history.

**Generation is two LLM calls.** Brief first, then rubric derived from that brief. A
single combined schema is too much for a small local model: pushing on one field in the
prompt collapses another, silently (empty requirements, three criteria instead of eight).
Do not merge them back.

**A coaching tick is two calls.** `harness.Decide` returns a structured decision; only if
it says speak does `harness.Speak` stream the body. Keep the decision schema small and
keep silence the default. The system prompt's bias toward silence is the product, not a
limitation to tune away.

**The overall score is computed in Go, not asked for.** `assembleReview` derives it from
per-criterion scores and weights so the number cannot drift from its evidence. Never let
the model return an overall score.

**Pacing events are arithmetic, not LLM output.** `harness.PacingEvent` runs in Go so it
fires at the same points regardless of which model is loaded.

**Message events must open, deliver, and close.** The frontend appends on
`TEXT_MESSAGE_START` and stops the typing caret on `TEXT_MESSAGE_END`. A message that
never ends leaves a caret blinking forever. `runtime.say` is the path for text that is
already known; it emits all three.

**All AG-UI events go out on one Wails channel** (`agui.Channel`). Ordering is what makes
start/content/end reconstructable, and separate channels would not guarantee it.

**`frontend/dist` is tracked but its contents are ignored.** `main.go` has
`//go:embed all:frontend/dist`, which fails on a missing directory, so `.gitkeep` keeps a
fresh clone buildable before the frontend is built. `make clean` must not delete it.

## Commands

```bash
make check      # fmt, vet, lint, test, typecheck
make test       # Go tests only
make live       # harness against a real model; needs ollama. MODEL=... to override
make dev        # live-reloading app
make bindings   # regenerate frontend/wailsjs after changing App's surface
```

## Testing

Unit tests cover the deterministic layer: clock and pause arithmetic, curveball firing,
pacing thresholds, event well-formedness, profile aggregation, store round trips.

Prompt behavior cannot be unit tested, so `internal/harness/live_test.go` hits a real
model behind `OPENLEARN_LIVE=1`. It asserts properties rather than strings: rubric
areas are in the taxonomy, curveballs exist, starter code is present for coding mode, the
coach streams in more than one chunk, and near-empty work scores below 40. Run it after
touching any prompt.

## Code style

Go: standard library first, `slog` for logging, errors wrapped with context. Comments
explain why, not what; the interesting comments in this repo are the ones justifying a
non-obvious choice (two-call generation, silence bias, Go-side scoring).

TypeScript: `reduce` in `agui.ts` stays pure so the same event stream always produces the
same UI. Wails' generator does not emit named string types, so Go enums arrive as
`string`; `src/types.ts` restores the unions and must stay in step with the Go constants.
