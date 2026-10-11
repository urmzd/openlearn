# Connecting an agent

The app has no assistant of its own. It serves the open course pack over the [Model Context Protocol](https://modelcontextprotocol.io), so the agent a learner already uses can read the course, see their progress, and check their exercises.

## Connect

The desktop binary is also the server. Run it with `mcp` and it speaks MCP on stdin and stdout instead of opening a window. **Connect an agent** in the sidebar shows the exact command for your install.

```sh
claude mcp add openlearn -- /Applications/openlearn.app/Contents/MacOS/openlearn mcp
```

For any other client, the same thing as configuration:

```json
{
  "mcpServers": {
    "openlearn": {
      "command": "/Applications/openlearn.app/Contents/MacOS/openlearn",
      "args": ["mcp"]
    }
  }
}
```

| Flag | Effect |
|------|--------|
| (none) | Follow whichever pack the app has open. Switch packs in the app and the next tool call serves the new one. |
| `-pack DIR` | Serve one pack directory, whatever the app has open. |

`go run ./cmd/openlearn-mcp` is the same server without the desktop app's webview dependencies, for a machine that only needs the server.

The server shares the app's store (`~/.openlearn`), so progress and sessions are the learner's own, and it works whether or not the app is running.

## Tools

| Tool | Does |
|------|------|
| `course_overview` | The pack's title, paths with progress, and tracks. The place to start. |
| `current_focus` | The document open in the app right now, so "this page" means something. |
| `list_track` | A track's topics and documents, in reading order. |
| `search` | Full-text search over every document. |
| `read_document` | One document, paged at 12,000 characters. |
| `get_path` | A path's stages: documents, "done when", check, and whether each is finished. |
| `mark_stage` | Mark a stage finished or not. |
| `session_profile` | Timed-session scores by skill area. |
| `list_exercises` | Exercises by kind and language, with which are started. |
| `open_exercise` | The learner's attempt, cloned into the scratchpad if new. |
| `write_exercise_file` | Overwrite one file of the attempt. |
| `check_exercise` | Run the attempt and return the verdict. |

The first eight read. `mark_stage` and the exercise tools change the learner's progress or their attempt.

## What the agent is told

The server sends instructions at initialization (`internal/mcpserver/server.go`). They say how to tutor here: ground answers in the course document, ask rather than hand over when the learner is working toward an answer, and never call an attempt correct without `check_exercise`.

Two things are enforced rather than asked for, because an instruction is not a guarantee:

- **Verdicts come from running the attempt.** There is no tool that sets a verdict. `check_exercise` runs the attempt against the exercise's own tests and reports the exit status.
- **Checked stages cannot be forced.** `mark_stage` goes through the same path the app's checkbox does. A stage tied to a check is refused until the pack's stage check says it passed.

## Trust

The server gives an agent read access to the pack and write access to two things: path progress, and files inside an attempt the practice CLI already created. File writes are confined to files the exercise engine already created, by `internal/practice`. `check_exercise` runs the attempt's compiler and tests on your machine, the same as the app's Check button, so the caution about opening packs you trust applies to agents as well.
