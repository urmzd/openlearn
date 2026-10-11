# Contributing

whiteboardy is in beta. Prompts, scoring calibration, and the saved-session format are all
still moving, so expect churn in `internal/harness` in particular.

## Prerequisites

- [Go 1.26.9+](https://go.dev)
- [Node 22+](https://nodejs.org)
- [Wails v2](https://wails.io): `go install github.com/wailsapp/wails/v2/cmd/wails@v2.11.0`
- [Ollama](https://ollama.com) with a model pulled, to run the live tests

## Getting started

```bash
git clone git@github.com:urmzd/whiteboardy.git
cd whiteboardy
make init
make dev
```

## Development

```bash
make check      # fmt, vet, lint, test, typecheck. Run before opening a PR.
make test       # Go tests only
make live       # harness against a real model. MODEL=qwen3.5:2b to override
make bindings   # regenerate frontend/wailsjs after changing App's method surface
```

`frontend/dist` is tracked but its contents are ignored, because `main.go` embeds it and
`//go:embed` fails on a missing directory. Do not delete the directory or its `.gitkeep`.

## Changing prompts

Prompt behavior cannot be unit tested, so `internal/harness/live_test.go` hits a real
model behind `WHITEBOARDY_LIVE=1` and asserts properties rather than exact strings. If you
touch a prompt or a schema in `internal/harness`, run `make live` and paste the relevant
output in your PR.

Try it against a small model too (`make live MODEL=qwen3.5:2b`). Most prompt regressions
show up as a small model silently dropping a field rather than as an error.

## Capturing the showcase

The README images live in `showcase/` and are shot from the real app running under
`wails dev`, never from a mock.

All four are shot at 1440x920 so the README gallery lines up.

| Image | Screen | Captured by |
|-------|--------|-------------|
| `demo.png` | A session in progress: brief, board, coach | CDP, live session |
| `demo-review.png` | The debrief | CDP, live session |
| `demo-setup.png` | The setup screen | CDP |
| `demo-progress.png` | The cross-session profile | CDP, or `make record` |

`make record` runs [teasr](https://github.com/urmzd/teasr) against the dev server and
re-captures the progress screen. It is the only screen teasr can script: its web backend
cannot type into React-controlled inputs, and a populated board and a debrief both need a
real model and real work.

teasr 0.19.2 also renders at 800x600 regardless of the viewport block in `teasr.toml`, so
`make record` breaks the gallery's matching aspect ratios. Re-shoot the progress screen
over CDP at 1440x920 afterwards if that matters.

Re-shoot whichever images a UI change invalidates. Do not hand-edit them, and do not stage
a board the app could not actually produce.

## Commit convention

Angular conventional commits. `sr` derives releases from them, so the type matters:

```
feat: add a coding-mode test runner
fix: stop the coach repeating itself after a curveball
docs: document the two-call generation split
```

`feat` cuts a minor, `fix`/`perf`/`refactor` cut a patch, everything else cuts nothing.

## Pull requests

1. Branch off `main`.
2. Add tests for anything deterministic; add or update a live assertion for anything
   prompt-shaped.
3. Make sure `make check` passes.
4. Open the PR against `main`.
