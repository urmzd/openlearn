# Architecture

whiteboardy is a Wails desktop app: a Go backend that owns the session and talks to a
model, and a React frontend that renders whatever the backend tells it.

## The event stream

The backend drives the UI over an [AG-UI](https://github.com/ag-ui-protocol/ag-ui)-shaped
event stream rather than a request/response API:

| Event | Carries |
|-------|---------|
| `RUN_STARTED` / `RUN_FINISHED` / `RUN_ERROR` | Session lifecycle |
| `STEP_STARTED` / `STEP_FINISHED` | Named slow phases: problem generation, a coaching tick, review |
| `TEXT_MESSAGE_START` / `CONTENT` / `END` | Agent speech, streamed |
| `STATE_SNAPSHOT` / `STATE_DELTA` | Shared state, whole or patched |

The frontend is a pure reduction of that stream (`reduce` in `frontend/src/agui.ts`) and
never polls. The agent decides when to respond; the UI does not ask.

Every event goes out on a single Wails channel. Ordering is what makes start/content/end
reconstructable, and separate channels would not guarantee it. The transport is Wails'
event bus rather than AG-UI's SSE encoding, since this is a single process: a networked
mode would be a transport swap, not a redesign.

The per-second clock is a `STATE_DELTA`, not a bespoke event. The UI holds one state
object and patches it.

## Packages

| Package | Role |
|---------|------|
| `internal/domain` | Data model, JSON-tagged because everything crosses into TypeScript. Owns the fixed skill taxonomy. |
| `internal/harness` | The three LLM engines: problem generation, the live coach, the reviewer |
| `internal/agui` | Event vocabulary, the typed `Bus`, and the `Emitter` seam |
| `internal/session` | `Runtime`: clock, curveball timeline, coach loop, review transition |
| `internal/store` | Settings and session JSON under `~/.whiteboardy`, plus profile aggregation |
| `internal/llm` | Provider construction over saige, and `Structured[T]` generation |
| `app.go` | The Wails binding surface. Every exported method becomes callable from TypeScript. |
| `frontend/src` | React. `agui.ts` reduces the event stream; the rest is presentation. |

## Why generation is two LLM calls

A single schema covering the brief, the rubric, the curveballs, and the reference outline
is too much for a small local model to hold at once. Pushing on one field in the prompt
collapses another, and the failure is silent: empty requirements, or three criteria where
eight were asked for.

Splitting it into a brief call and a rubric call makes each one reliable, and the second
call gets to read the brief that was actually produced. Criteria then grade the exercise
that exists rather than the one that was requested.

## Why a coaching tick is also two calls

The decision (speak or stay quiet, and about what) needs a schema so it comes back
reliably. The message body does not: it is prose the user watches being written, and
streaming it is what makes the coach feel like someone watching rather than a notification
that pops in fully formed.

Silence is the common case and costs only the first call, whose schema is small. Speaking
costs a second call that streams.

## What is computed rather than asked for

Two things deliberately never come from the model:

**The overall score.** `assembleReview` derives it from per-criterion scores and their
weights, so the number cannot drift from the evidence behind it.

**Pacing nudges.** `harness.PacingEvent` is arithmetic over elapsed time and rubric
coverage, so it fires at the same points regardless of which model is loaded.

## The skill taxonomy

Rubric criteria must map onto a fixed, per-mode set of `domain.Area` values. Criteria whose
area falls outside it are dropped at assembly time.

This is what makes scores comparable across sessions: per-problem rubric wording changes
every run, area IDs do not. Adding an area is safe. Renaming one silently orphans existing
profile history.

## Ollama specifics

Two settings are applied to every local call, both learned the hard way:

**`num_ctx` is set explicitly.** Ollama's default context window is far smaller than a
review prompt (problem, hidden rubric, whole board, event log) and it truncates past the
limit silently, turning an oversized prompt into a confidently wrong answer rather than an
error.

**Thinking is disabled** on models that can toggle it. Schema-constrained calls need it off because the format grammar
applies to every token the model emits, including reasoning. For the coach's prose it is
also harmful: a reasoning model spends tens of seconds thinking before its first text
token, so the message bubble opens and then sits visibly empty.

The client is built with [saige](https://github.com/urmzd/saige)'s `provider.Build`, which
sends the thinking toggle only where its model catalog says the model accepts it. Sampling
is set with saige's `creativity` dial (`focused`) rather than a raw temperature: the adapter
compiles it to a temperature where the model takes one and drops it on models that reject
sampling controls. `num_ctx` has no provider-neutral option, so it is added to the built
client's chat options.
