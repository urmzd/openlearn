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

// checker is a practice CLI small enough to read. It honors the contract
// internal/practice documents and nothing else.
const checker = `#!/bin/sh
scratch="$PWD/.scratchpad"
case "$1" in
  list) printf 'build/sh\n  01-echo   --\n' ;;
  start) mkdir -p "$scratch/$2/$3/$4" && [ -f "$scratch/$2/$3/$4/main.sh" ] || printf 'echo TODO\n' > "$scratch/$2/$3/$4/main.sh" ;;
  check)
    f="$scratch/$2/$3/$4/main.sh"
    [ -f "$f" ] || { echo "skip $4"; exit 2; }
    [ "$(sh "$f")" = hello ] && { echo "PASS $4"; exit 0; }
    echo "FAIL $4"; exit 1 ;;
  learn) mkdir -p "$scratch/learn"; [ "$3" = --done ] && echo "$4" >> "$scratch/learn/$2.done"; echo marked ;;
esac
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
	write(t, filepath.Join(pack, "openlearn.json"), `{"id":"demo","title":"Demo","paths":"paths","featured":"main","practice":"bin/check","modes":["written"],"tracks":[{"dir":"unit","title":"Unit One"}]}`, 0o600)
	write(t, filepath.Join(pack, "unit/01-limits/README.md"), "# Limits\n"+strings.Repeat("A limit is a promise about nearby values. ", 400), 0o600)
	write(t, filepath.Join(pack, "paths/main/README.md"), "# Main path\n", 0o600)
	write(t, filepath.Join(pack, "paths/main/path.tsv"), "1\tLimits\tunit/01-limits/README.md\tYou can prove one.\t-\n", 0o600)
	write(t, filepath.Join(pack, "bin/check"), checker, 0o700)
	t.Setenv("SS_SCRATCH", "")
	t.Setenv("SS_COURSE_HOME", "")

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

func TestStageMarksGoThroughThePacksOwnCLI(t *testing.T) {
	cs, _ := connect(t)

	before, _ := call(t, cs, "get_path", map[string]any{"name": "main"})
	if !strings.Contains(before, `"done": false`) {
		t.Fatalf("stage should start unfinished: %s", before)
	}
	if out, isErr := call(t, cs, "mark_stage", map[string]any{"path": "main", "stage": "1", "done": true}); isErr {
		t.Fatalf("mark_stage: %s", out)
	}
	after, _ := call(t, cs, "get_path", map[string]any{"name": "main"})
	if !strings.Contains(after, `"done": true`) {
		t.Errorf("the mark did not show up where the app reads progress: %s", after)
	}
	if out, isErr := call(t, cs, "mark_stage", map[string]any{"path": "main", "stage": "9", "done": true}); !isErr {
		t.Errorf("marking a stage that does not exist succeeded: %s", out)
	}
}

func TestOnlyTheCheckerDecidesAnExercise(t *testing.T) {
	cs, _ := connect(t)
	ex := map[string]any{"kind": "build", "lang": "sh", "name": "01-echo"}

	if out, _ := call(t, cs, "check_exercise", ex); !strings.Contains(out, `"verdict": "none"`) {
		t.Errorf("an unstarted exercise has no verdict: %s", out)
	}
	opened, _ := call(t, cs, "open_exercise", ex)
	if !strings.Contains(opened, "echo TODO") {
		t.Fatalf("attempt = %s", opened)
	}
	if out, _ := call(t, cs, "check_exercise", ex); !strings.Contains(out, `"verdict": "fail"`) {
		t.Errorf("the stub must fail: %s", out)
	}

	write := map[string]any{"kind": "build", "lang": "sh", "name": "01-echo", "file": "main.sh", "content": "echo hello\n"}
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
