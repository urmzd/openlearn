package mcpserver_test

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/urmzd/openlearn/app/internal/mcpserver"
	"github.com/urmzd/openlearn/app/internal/store"
)

// exercises is a spec whose only toolchain is sh.
const exercises = `{"sets": [{"kind": "build", "lang": "sh", "mode": "attempt", "glob": "ex/[0-9][0-9]-*",
  "run": {"steps": [["sh", "main.sh"]]}}]}`

// solved is a correct attempt at the exercise.
const solved = "greet() { echo hello; }\n[ \"$(greet)\" = hello ]\n"

// gate is the pack's stage check: it lets a checked stage through only once
// the learner's exercise attempt passes.
const gate = `#!/bin/sh
sh .scratchpad/build/sh/01-echo/main.sh 2>/dev/null || { echo "module:M1 has not passed"; exit 1; }
mkdir -p .scratchpad/learn && echo "$2" >> ".scratchpad/learn/$1.done"
`

func write(t *testing.T, path, body string, mode os.FileMode) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), mode); err != nil {
		t.Fatal(err)
	}
}

// connect starts a server over a temp pack and store and returns a client
// session talking to it, the way an agent would.
func connect(t *testing.T) (*mcp.ClientSession, *store.Store) {
	t.Helper()
	pack := t.TempDir()
	write(t, filepath.Join(pack, "openlearn.json"), `{"id":"demo","title":"Demo","paths":"paths","featured":"main","practice":"ex.json","stageCheck":["bin/gate","{path}","{stage}"],"modes":["written"],"tracks":[{"dir":"unit","title":"Unit One"}]}`, 0o600)
	write(t, filepath.Join(pack, "ex.json"), exercises, 0o600)
	write(t, filepath.Join(pack, "ex/01-echo/main.sh"), "greet() {\n  # SOLUTION-BEGIN\n  echo hello\n  # SOLUTION-END\n  :\n}\n[ \"$(greet)\" = hello ]\n", 0o600)
	write(t, filepath.Join(pack, "unit/01-limits/README.md"), "# Limits\n"+strings.Repeat("A limit is a promise about nearby values. ", 400), 0o600)
	write(t, filepath.Join(pack, "paths/main/README.md"), "# Main path\n", 0o600)
	write(t, filepath.Join(pack, "paths/main/path.tsv"), "1\tLimits\tunit/01-limits/README.md\tYou can prove one.\t-\n2\tEcho\tunit/01-limits/README.md\tYour echo passes.\tmodule:M1\n", 0o600)
	write(t, filepath.Join(pack, "bin/gate"), gate, 0o700)
	t.Setenv("OL_SCRATCH", "")
	t.Setenv("OL_COURSE_HOME", "")

	st, err := store.New(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	serverT, clientT := mcp.NewInMemoryTransports()
	srv := mcpserver.New(mcpserver.Options{Store: st, Pack: pack, Version: "test"})
	ss, err := srv.Connect(ctx, serverT, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ss.Close() })
	cs, err := mcp.NewClient(&mcp.Implementation{Name: "test", Version: "0"}, nil).Connect(ctx, clientT, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = cs.Close() })
	return cs, st
}

// call runs a tool and returns its text and whether it reported an error.
func call(t *testing.T, cs *mcp.ClientSession, name string, args map[string]any) (string, bool) {
	t.Helper()
	res, err := cs.CallTool(context.Background(), &mcp.CallToolParams{Name: name, Arguments: args})
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	var b strings.Builder
	for _, c := range res.Content {
		if tc, ok := c.(*mcp.TextContent); ok {
			b.WriteString(tc.Text)
		}
	}
	return b.String(), res.IsError
}

func TestServerTellsTheAgentHowToTeach(t *testing.T) {
	cs, _ := connect(t)
	if got := cs.InitializeResult().Instructions; !strings.Contains(got, "check_exercise") {
		t.Errorf("instructions do not tell the agent where verdicts come from: %q", got)
	}
	tools, err := cs.ListTools(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(tools.Tools) != 12 {
		t.Errorf("tools = %d, want 12", len(tools.Tools))
	}
	for _, tool := range tools.Tools {
		if tool.Description == "" {
			t.Errorf("tool %s has no description", tool.Name)
		}
	}
}

func TestAnAgentCanFindReadAndPageADocument(t *testing.T) {
	cs, _ := connect(t)

	overview, _ := call(t, cs, "course_overview", nil)
	for _, want := range []string{`"title": "Demo"`, `"dir": "unit"`, `"name": "main"`, `"exercises": true`} {
		if !strings.Contains(overview, want) {
			t.Errorf("overview is missing %s:\n%s", want, overview)
		}
	}

	hits, _ := call(t, cs, "search", map[string]any{"query": "promise nearby"})
	if !strings.Contains(hits, "unit/01-limits/README.md") {
		t.Fatalf("search did not find the document: %s", hits)
	}

	page, _ := call(t, cs, "read_document", map[string]any{"path": "unit/01-limits/README.md"})
	if !strings.Contains(page, "title: Limits") || !strings.Contains(page, "next_offset: 12000") {
		t.Errorf("first page header = %q", page[:120])
	}
	rest, _ := call(t, cs, "read_document", map[string]any{"path": "unit/01-limits/README.md", "offset": 12000})
	if strings.Contains(rest, "next_offset") {
		t.Error("the last page must not offer another")
	}

	if msg, isErr := call(t, cs, "read_document", map[string]any{"path": "../../etc/passwd"}); !isErr {
		t.Errorf("a path outside the pack was read: %s", msg)
	}
}

func TestStageMarksAreSharedAndCheckedStagesCannotBeForced(t *testing.T) {
	cs, _ := connect(t)

	if out, isErr := call(t, cs, "mark_stage", map[string]any{"path": "main", "stage": "1", "done": true}); isErr {
		t.Fatalf("mark_stage: %s", out)
	}
	after, _ := call(t, cs, "get_path", map[string]any{"name": "main"})
	if strings.Count(after, `"done": true`) != 1 {
		t.Errorf("the mark did not show up where the app reads progress: %s", after)
	}
	if out, isErr := call(t, cs, "mark_stage", map[string]any{"path": "main", "stage": "9", "done": true}); !isErr {
		t.Errorf("marking a stage that does not exist succeeded: %s", out)
	}

	// Stage 2 is tied to a check. An agent asking nicely is not a pass.
	out, isErr := call(t, cs, "mark_stage", map[string]any{"path": "main", "stage": "2", "done": true})
	if !isErr || !strings.Contains(out, "module:M1 has not passed") {
		t.Fatalf("a checked stage was marked without its check passing: %s", out)
	}
	// Once the learner's attempt really passes, the same call goes through.
	ex := map[string]any{"kind": "build", "lang": "sh", "name": "01-echo"}
	call(t, cs, "open_exercise", ex)
	call(t, cs, "write_exercise_file", map[string]any{"kind": "build", "lang": "sh", "name": "01-echo", "file": "main.sh", "content": solved})
	if out, isErr := call(t, cs, "mark_stage", map[string]any{"path": "main", "stage": "2", "done": true}); isErr {
		t.Fatalf("the check passed but the stage was refused: %s", out)
	}
	done, _ := call(t, cs, "get_path", map[string]any{"name": "main"})
	if strings.Count(done, `"done": true`) != 2 {
		t.Errorf("both stages should be finished now: %s", done)
	}
}

func TestOnlyTheCheckerDecidesAnExercise(t *testing.T) {
	cs, _ := connect(t)
	ex := map[string]any{"kind": "build", "lang": "sh", "name": "01-echo"}

	if out, _ := call(t, cs, "check_exercise", ex); !strings.Contains(out, `"verdict": "none"`) {
		t.Errorf("an unstarted exercise has no verdict: %s", out)
	}
	opened, _ := call(t, cs, "open_exercise", ex)
	if !strings.Contains(opened, "TODO: your implementation goes here.") || strings.Contains(opened, "echo hello") {
		t.Fatalf("attempt = %s", opened)
	}
	if out, _ := call(t, cs, "check_exercise", ex); !strings.Contains(out, `"verdict": "fail"`) {
		t.Errorf("the stub must fail: %s", out)
	}

	write := map[string]any{"kind": "build", "lang": "sh", "name": "01-echo", "file": "main.sh", "content": solved}
	if out, isErr := call(t, cs, "write_exercise_file", write); isErr {
		t.Fatalf("write: %s", out)
	}
	if out, _ := call(t, cs, "check_exercise", ex); !strings.Contains(out, `"verdict": "pass"`) {
		t.Errorf("the fixed attempt must pass: %s", out)
	}

	write["file"] = "../../escape.sh"
	if out, isErr := call(t, cs, "write_exercise_file", write); !isErr {
		t.Errorf("a write outside the attempt succeeded: %s", out)
	}
}

func TestFocusIsWhatTheAppLastRecorded(t *testing.T) {
	cs, st := connect(t)

	out, _ := call(t, cs, "current_focus", nil)
	var none struct{ Open bool }
	if err := json.Unmarshal([]byte(out), &none); err != nil || none.Open {
		t.Errorf("no focus recorded yet, got %s", out)
	}

	if err := st.SaveFocus(store.Focus{Pack: "elsewhere", PackTitle: "Other", Doc: "a/b.md"}); err != nil {
		t.Fatal(err)
	}
	out, _ = call(t, cs, "current_focus", nil)
	if !strings.Contains(out, `"document": "a/b.md"`) || !strings.Contains(out, "different pack") {
		t.Errorf("focus = %s", out)
	}
}
