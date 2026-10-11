package practice_test

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/urmzd/openlearn/app/internal/practice"
)

func TestParseListReadsGroupsAndStartedMarks(t *testing.T) {
	got := practice.ParseList("predict/go\n  01-loop-var-capture                    --\n  02-nil-interface                       started\nbuild/c\n  01-dynamic-array                       --\n\nnot a heading\n")
	if len(got) != 2 {
		t.Fatalf("groups = %+v", got)
	}
	if got[0].Kind != "predict" || got[0].Lang != "go" || len(got[0].Exercises) != 2 {
		t.Errorf("group 0 = %+v", got[0])
	}
	if got[0].Exercises[0].Started || !got[0].Exercises[1].Started {
		t.Errorf("started marks = %+v", got[0].Exercises)
	}
	if got[1].Exercises[0].Name != "01-dynamic-array" {
		t.Errorf("group 1 = %+v", got[1])
	}
}

// fakeCLI is a practice CLI small enough to read: it honors the contract the
// package documents and nothing else.
const fakeCLI = `#!/bin/sh
scratch="${SS_SCRATCH:-$PWD/.scratchpad}"
case "$1" in
  list) printf 'build/sh\n  01-echo   --\n' ;;
  start)
    [ "$EDITOR" = true ] || { echo "would open an editor"; exit 9; }
    mkdir -p "$scratch/$2/$3/$4" && printf 'echo TODO\n' > "$scratch/$2/$3/$4/main.sh" ;;
  check)
    f="$scratch/$2/$3/$4/main.sh"
    [ -f "$f" ] || { echo "skip $4"; exit 2; }
    out="$(sh "$f")"
    [ "$out" = hello ] && { echo "PASS $4"; exit 0; }
    echo "FAIL $4"; echo "    got: $out"; exit 1 ;;
  reset) rm -rf "$scratch/$2/$3/$4" && echo cleared ;;
  learn) echo "$2 $3 $4"; [ "$3" = --done ] ;;
  *) echo "unknown verb $1"; exit 64 ;;
esac
`

func runner(t *testing.T) *practice.Runner {
	t.Helper()
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "bin"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "bin", "ss"), []byte(fakeCLI), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("SS_SCRATCH", "")
	r, err := practice.New(root, "bin/ss")
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func TestAttemptLifecycleThroughTheCLI(t *testing.T) {
	r, ctx := runner(t), context.Background()

	groups, err := r.List(ctx)
	if err != nil || len(groups) != 1 || groups[0].Exercises[0].Name != "01-echo" {
		t.Fatalf("list = %+v, %v", groups, err)
	}

	if res, err := r.Check(ctx, "build", "sh", "01-echo"); err != nil || res.Verdict != "none" {
		t.Fatalf("check before start = %+v, %v; want verdict none", res, err)
	}

	ws, err := r.Start(ctx, "build", "sh", "01-echo")
	if err != nil {
		t.Fatal(err)
	}
	if len(ws.Files) != 1 || ws.Files[0].Path != "main.sh" || ws.Files[0].Content != "echo TODO\n" {
		t.Fatalf("workspace = %+v", ws)
	}

	res, err := r.Check(ctx, "build", "sh", "01-echo")
	if err != nil || res.Verdict != "fail" || !strings.Contains(res.Output, "got: TODO") {
		t.Fatalf("check of the stub = %+v, %v", res, err)
	}

	if err := r.Save("build", "sh", "01-echo", "main.sh", "echo hello\n"); err != nil {
		t.Fatal(err)
	}
	if res, err := r.Check(ctx, "build", "sh", "01-echo"); err != nil || res.Verdict != "pass" {
		t.Fatalf("check after the fix = %+v, %v", res, err)
	}

	if _, err := r.Reset(ctx, "build", "sh", "01-echo"); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Workspace(ctx, "build", "sh", "01-echo"); err == nil {
		t.Error("a reset attempt must be gone")
	}
}

func TestSaveOnlyWritesFilesOfTheAttempt(t *testing.T) {
	r, ctx := runner(t), context.Background()
	if _, err := r.Start(ctx, "build", "sh", "01-echo"); err != nil {
		t.Fatal(err)
	}
	for _, rel := range []string{"../../../../escape.sh", "/tmp/abs.sh", "new-file.sh"} {
		if err := r.Save("build", "sh", "01-echo", rel, "x"); err == nil {
			t.Errorf("Save(%q) succeeded", rel)
		}
	}
	for _, name := range []string{"--help", "../01-echo", "a b", ""} {
		if _, err := r.Check(ctx, "build", "sh", name); err == nil {
			t.Errorf("Check with name %q ran", name)
		}
	}
}

func TestLearnReportsWhetherTheCLIAcceptedTheMark(t *testing.T) {
	r, ctx := runner(t), context.Background()
	if res, err := r.Learn(ctx, "course", "part:3", true); err != nil || res.Verdict != "pass" || res.Output != "course --done part:3" {
		t.Errorf("done = %+v, %v", res, err)
	}
	if res, err := r.Learn(ctx, "course", "3", false); err != nil || res.Verdict != "fail" {
		t.Errorf("a nonzero exit must surface as a refusal: %+v, %v", res, err)
	}
}

// The real CLI, when this runs inside the repository: the list parser has to
// keep reading what ss actually prints.
func TestRealCLIListParses(t *testing.T) {
	r, err := practice.New("../../..", "practice/bin/ss")
	if err != nil {
		t.Skip("no practice CLI beside the app:", err)
	}
	groups, err := r.List(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	total := 0
	for _, g := range groups {
		if g.Kind == "" || g.Lang == "" {
			t.Errorf("unparsed heading: %+v", g)
		}
		total += len(g.Exercises)
	}
	if total == 0 {
		t.Error("ss list returned no exercises")
	}
}
