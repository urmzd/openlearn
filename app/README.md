<p align="center">
  <h1 align="center">openlearn</h1>
  <p align="center">
    A desktop learning platform: read a course, practice it, and have it graded, with an AI professor alongside.
    <br /><br />
    <a href="https://github.com/urmzd/openlearn/releases">Download</a>
    &middot;
    <a href="https://github.com/urmzd/openlearn/issues">Report Bug</a>
  </p>
</p>

<p align="center">
  <img src="showcase/reader.png" alt="A chapter open in the reader, with search results and the professor panel" width="80%">
</p>

> [!WARNING]
> **Beta, and still in testing.** Prompts, scoring calibration, and the saved-session
> format are all still moving. Scores are directional, not authoritative.

openlearn renders a **course pack**: a directory of markdown with an `openlearn.json`
manifest that names its tracks, its role paths, and its practice CLI. The app knows
nothing about any one course. The [supersource curriculum](../README.md) is the pack built
into the binary, and [languide](../packs/languide/) ships beside it; any other directory
with a manifest loads the same way.

It started as whiteboardy, a timeboxed practice harness, and that harness is still the
part that grades you: it writes an exercise from the chapter you just read, times you,
interrupts when it is worth interrupting, and scores what you produced against a rubric
written before you started. Runs fully local against [Ollama](https://ollama.com), or
against a hosted model.

## Features

- **Course packs.** Tracks, role paths, eval modes, and the practice CLI come from the
  pack's manifest. Open a folder or import a zip and the app teaches that course instead.
- **A reader built for courses.** Math, diagrams, tables, and links between documents,
  with full-text search across the pack.
- **Paths with shared progress.** Stages, what "done" means for each, and which you have
  finished, read from the same `.done` files `ss learn` writes.
- **Exercises that run for real.** Edit an attempt in the app and check it with the pack's
  own CLI. The exit code is the verdict, the same one a terminal would give you.
- **Sessions grounded in a chapter.** Start from any document and the exercise and its
  hidden rubric are generated from that material: a board for system design, an editor for
  coding, a written answer for everything else.
- **A coach that mostly stays quiet**, and a **professor** that answers when asked, about
  the page you have open.
- **A profile that accumulates.** Rubric criteria map onto a fixed taxonomy, so scores
  compare across sessions.

## Installation

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Go | 1.26.9+ |
| Node | 22+ |
| Wails | v2.11 |
| Ollama | any recent, with a model pulled |

### Build

```sh
git clone https://github.com/urmzd/openlearn
cd openlearn/app
make init
make build
```

The app lands in `build/bin/`. For a live-reloading dev loop, use `make dev`.


## Quick Start

```sh
ollama pull qwen3.5:9b
make build && open build/bin/openlearn.app
```

1. Pick a provider and model in **Settings**, then hit **Test connection**.
2. Open a chapter and hit **Practice this**, or start a **Timed session** cold. Choose a
   mode, a level, and a timebox. Start. Generation takes a moment on a local
   model: it writes the brief, then the hidden rubric.
3. Work. The coach reads your board or code on an interval and speaks only when it has
   something worth saying.
4. Hit **Finish & score** for the debrief, then check **Progress** once you have a few
   sessions behind you.

Anything in the 7-9B range and up is usable. Smaller models produce noticeably thinner
rubrics. Sessions are written to `~/.openlearn/sessions` as plain JSON, and nothing
leaves your machine unless you point it at a hosted provider.

<p align="center">
  <img src="showcase/demo.png" width="45%"> &nbsp; <img src="showcase/demo-review.png" width="45%">
</p>
<p align="center"><em>A session in progress &middot; The debrief</em></p>

Every criterion is scored 0-4 with the evidence that earned it, quoted from your board and
your notes. The overall number is computed from those scores and their weights, so it
cannot drift from the reasoning behind it. Ranking an area as a strength or a weakness is
withheld until it has at least two samples, and trend needs three, because calling
something a weakness off one data point is noise dressed as insight.

## Writing a course pack

A pack is a directory of markdown with an `openlearn.json` at its root. Paths in it are
relative to that directory.

```json
{
  "id": "my-course",
  "title": "My Course",
  "tagline": "One line about it.",
  "repo": "https://github.com/me/my-course",
  "paths": "paths",
  "featured": "main",
  "practice": "bin/check",
  "modes": ["written"],
  "start": [{ "file": "README.md", "title": "Overview" }],
  "tracks": [{ "dir": "unit-1", "title": "Unit 1" }]
}
```

| Field | Meaning |
|-------|---------|
| `id`, `title` | Required. The id names the pack on disk, so keep it a plain directory name. |
| `tracks` | Directories of documents, in reading order. Each subdirectory is a topic, titled by its `README.md`. |
| `reference` | Same shape as `tracks`, for material that is browsable but not in the learning order. |
| `start` | Documents a new reader opens first. |
| `paths` | Directory holding one `<name>/path.tsv` per role path. See [paths/README.md](../paths/README.md) for the format. |
| `featured` | The path the home screen leads with. |
| `modes` | Which timed-session modes fit the material: `system_design`, `coding`, `written`. Omit for all three. |
| `practice` | An executable implementing the practice CLI contract documented in [`internal/practice`](internal/practice/practice.go). It only runs when the pack is opened from a folder. |
| `repo` | Where links to files the app does not render (code, configs) open. |

Open it with **Settings > Open folder**, or zip the directory and use **Import zip**. The two
packs in this repository are working examples: [the curriculum](../openlearn.json), which
has everything, and [languide](../packs/languide/openlearn.json), which is documents and
paths only.

## Configuration

Settings live in `~/.openlearn/settings.json` and are editable from the Settings view.

| Provider | Model discovery | Notes |
|----------|-----------------|-------|
| Ollama | Automatic | Default. Fully local, no key. |
| Anthropic | Manual model id | Needs an API key |
| OpenAI | Manual model id | Needs an API key |
| Google | Manual model id | Needs an API key |

Provider plumbing comes from [saige](https://github.com/urmzd/saige)'s agent SDK.

## Architecture

The backend drives the UI over an [AG-UI](https://github.com/ag-ui-protocol/ag-ui)-shaped
event stream: a run is bracketed by `RUN_STARTED` and `RUN_FINISHED`, agent speech arrives
as `TEXT_MESSAGE_START` / `CONTENT` / `END`, and shared state moves as `STATE_SNAPSHOT`
plus `STATE_DELTA`. The frontend is a pure reduction of that stream and never polls, so
the agent decides when to respond rather than the UI asking.

See [docs/architecture.md](docs/architecture.md) for the full guide, including why problem
generation and each coaching tick are two LLM calls rather than one.

## Development

```sh
make check   # fmt, vet, lint, test, typecheck
make test    # Go tests only
make live    # exercise the harness against a real model (needs ollama)
```

`make live` is how prompt changes get validated. It asserts the things that actually
matter: that generated rubrics map onto the taxonomy, that curveballs exist, that the
coach streams rather than emitting one blob, and that near-empty work does not score well.
Override the model with `make live MODEL=qwen3.5:2b`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## Known Gaps

This is where the beta label comes from.

- Scoring calibration is unvalidated across models. A 3B model and a 9B model do not grade
  the same work the same way, and neither has been checked against human judgment.
- Coach silence is a soft bias, not a guarantee. It sometimes speaks earlier than it
  should.
- Coding mode is exercised by the live tests but has not been driven end to end through
  the app the way system design has.
- The saved-session format is not stable yet, so old sessions may stop aggregating.
- No import, export, or replay of sessions.
- Coding mode has no execution and no test runner. Correctness is judged by reading.

## License

[Apache-2.0](../LICENSE)
