# whiteboardy

A Wails desktop app for timeboxed system design and coding practice. The backend
generates an exercise plus a hidden rubric, coaches while the user works, and scores the
result. Beta: prompts and scoring calibration are still moving.

## Architecture

| Package | Role |
|---------|------|
| `internal/domain` | Data model, JSON-tagged because everything crosses into TypeScript. Owns the fixed skill taxonomy. |
| `internal/harness` | The three LLM engines: `problem.go` (generation), `coach.go` (live coaching), `review.go` (scoring) |
| `internal/agui` | AG-UI event vocabulary, the typed `Bus`, and the `Emitter` seam |
| `internal/session` | `Runtime`: clock, curveball timeline, coach loop, review transition |
| `internal/store` | Settings and session JSON under `~/.whiteboardy`, plus profile aggregation |
| `internal/llm` | Provider construction over saige, and `Structured[T]` generation |
| `app.go` | The Wails binding surface. Every exported method becomes callable from TS. |
| `frontend/src` | React. `agui.ts` reduces the event stream; the rest is presentation. |

## Non-obvious invariants

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
model behind `WHITEBOARDY_LIVE=1`. It asserts properties rather than strings: rubric
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
