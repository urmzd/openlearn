package practice_test

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/urmzd/openlearn/app/internal/practice"
)

func write(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
}

// A pack whose only toolchain is sh, so the engine is tested without depending
// on any compiler being installed.
const spec = `{
  "sets": [
    {"kind": "predict", "lang": "sh", "mode": "predict", "glob": "snippets/[0-9][0-9]-*.sh",
     "run": {"dir": "snippets", "steps": [["sh", "{name}.sh"]]}},
    {"kind": "build", "lang": "sh", "mode": "attempt", "glob": "builds/[0-9][0-9]-*",
     "seed": {"NOTES.txt": "attempt at {name}\n"},
     "run": {"steps": [["sh", "-c", "cat \"$@\" > \"$0/all\"", "{out}", "*.part"], ["sh", "main.sh", "{out}/all"]]}},
    {"kind": "redo", "lang": "files", "mode": "attempt",
     "items": [{"name": "solo", "source": "elsewhere/solo.sh"}],
     "run": {"steps": [["sh", "{file}"]]}},
    {"kind": "build", "lang": "missing", "mode": "attempt", "glob": "builds/[0-9][0-9]-*",
     "run": {"steps": [["no-such-compiler-anywhere", "main.sh"]]}},
    {"kind": "build", "lang": "empty", "mode": "attempt", "glob": "nothing/*", "run": {"steps": [["true"]]}}
  ]
}`

func runner(t *testing.T) *practice.Runner {
	t.Helper()
	root := t.TempDir()
	write(t, filepath.Join(root, "ex.json"), spec)
	write(t, filepath.Join(root, "snippets/01-hello.sh"), "echo hello\necho world   \n")
	write(t, filepath.Join(root, "builds/01-sum/main.sh"),
		"# test harness, never edited\n. ./lib.sh\n[ \"$(cat \"$1\")\" = ab ] || exit 3\n[ \"$(add 2 3)\" = 5 ] || { echo \"add 2 3 gave $(add 2 3)\"; exit 1; }\n")
	write(t, filepath.Join(root, "builds/01-sum/lib.sh"),
		"add() {\n  # SOLUTION-BEGIN\n  echo $(( $1 + $2 ))\n  # SOLUTION-END\n  :\n}\n")
	write(t, filepath.Join(root, "builds/01-sum/a.part"), "a")
	write(t, filepath.Join(root, "builds/01-sum/b.part"), "b")
	write(t, filepath.Join(root, "elsewhere/solo.sh"), "# SOLUTION-BEGIN\nexit 0\n# SOLUTION-END\nexit 7\n")
	t.Setenv("OL_SCRATCH", "")
	r, err := practice.New(root, "ex.json")
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func TestListFindsExercisesAndOmitsEmptySets(t *testing.T) {
	groups, err := runner(t).List(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, g := range groups {
		for _, e := range g.Exercises {
			got = append(got, g.Kind+"/"+g.Lang+"/"+e.Name)
		}
	}
	want := "predict/sh/01-hello build/sh/01-sum redo/files/solo build/missing/01-sum"
	if strings.Join(got, " ") != want {
		t.Errorf("exercises = %v\nwant        %s", got, want)
	}
}

func TestPredictComparesWhatRanWithWhatWasWritten(t *testing.T) {
	r, ctx := runner(t), context.Background()

	if res, _ := r.Check(ctx, "predict", "sh", "01-hello"); res.Verdict != "none" {
		t.Errorf("an empty prediction is not an attempt: %+v", res)
	}
	ws, err := r.Start(ctx, "predict", "sh", "01-hello")
	if err != nil {
		t.Fatal(err)
	}
	if len(ws.Files) != 2 || !ws.Files[0].ReadOnly || ws.Files[0].Path != "snippets/01-hello.sh" || ws.Files[1].Path != "01-hello.txt" {
		t.Fatalf("workspace = %+v", ws.Files)
	}

	if err := r.Save("predict", "sh", "01-hello", "01-hello.txt", "hello\nmoon\n"); err != nil {
		t.Fatal(err)
	}
	res, _ := r.Check(ctx, "predict", "sh", "01-hello")
	if res.Verdict != "fail" || !strings.Contains(res.Output, "- moon") || !strings.Contains(res.Output, "+ world") {
		t.Errorf("a wrong prediction must show both sides: %+v", res)
	}

	// Trailing whitespace and a stray final newline are not a wrong answer.
	if err := r.Save("predict", "sh", "01-hello", "01-hello.txt", "\nhello\nworld\n\n\n"); err != nil {
		t.Fatal(err)
	}
	if res, _ := r.Check(ctx, "predict", "sh", "01-hello"); res.Verdict != "pass" {
		t.Errorf("verdict = %+v, want pass", res)
	}
	if err := r.Save("predict", "sh", "01-hello", "other.txt", "x"); err == nil {
		t.Error("a prediction has exactly one file to write")
	}
}

func TestAttemptIsStubbedCheckedAndJudgedByExitCode(t *testing.T) {
	r, ctx := runner(t), context.Background()

	ws, err := r.Start(ctx, "build", "sh", "01-sum")
	if err != nil {
		t.Fatal(err)
	}
	files := map[string]string{}
	for _, f := range ws.Files {
		files[f.Path] = f.Content
	}
	if strings.Contains(files["lib.sh"], "$(( $1 + $2 ))") || !strings.Contains(files["lib.sh"], "# TODO: your implementation goes here.") {
		t.Errorf("the solution was not stripped:\n%s", files["lib.sh"])
	}
	if !strings.Contains(files["main.sh"], "never edited") {
		t.Error("a file with no markers must arrive intact")
	}
	if files["NOTES.txt"] != "attempt at 01-sum\n" {
		t.Errorf("seed file = %q", files["NOTES.txt"])
	}

	res, _ := r.Check(ctx, "build", "sh", "01-sum")
	if res.Verdict != "fail" || !strings.Contains(res.Output, "add 2 3 gave") {
		t.Fatalf("the stub must fail with the test's own message: %+v", res)
	}

	diff, err := r.Diff(ctx, "build", "sh", "01-sum")
	if err != nil || !strings.Contains(diff.Output, "+   echo $(( $1 + $2 ))") || strings.Contains(diff.Output, "SOLUTION-BEGIN") {
		t.Errorf("diff = %+v, %v", diff, err)
	}

	if err := r.Save("build", "sh", "01-sum", "lib.sh", "add() { echo $(( $1 + $2 )); }\n"); err != nil {
		t.Fatal(err)
	}
	// Passing also proves the step list: *.part was expanded with no shell, and
	// {out} pointed both steps at the same directory.
	if res, _ := r.Check(ctx, "build", "sh", "01-sum"); res.Verdict != "pass" {
		t.Fatalf("verdict = %+v, want pass", res)
	}

	if _, err := r.Reset(ctx, "build", "sh", "01-sum"); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Workspace(ctx, "build", "sh", "01-sum"); err == nil {
		t.Error("a reset attempt must be gone")
	}
	// Starting again is a cold start: the edit is gone, the stub is back.
	again, _ := r.Start(ctx, "build", "sh", "01-sum")
	for _, f := range again.Files {
		if f.Path == "lib.sh" && !strings.Contains(f.Content, "TODO") {
			t.Error("reset did not bring the stub back")
		}
	}
}

func TestListedItemRunsItsSingleFile(t *testing.T) {
	r, ctx := runner(t), context.Background()
	ws, err := r.Start(ctx, "redo", "files", "solo")
	if err != nil {
		t.Fatal(err)
	}
	if len(ws.Files) != 1 || ws.Files[0].Path != "solo.sh" {
		t.Fatalf("workspace = %+v", ws.Files)
	}
	if res, _ := r.Check(ctx, "redo", "files", "solo"); res.Verdict != "fail" {
		t.Errorf("the stripped file exits 7, so it must fail: %+v", res)
	}
}

func TestAMissingToolchainIsNamedNotMistakenForAWrongAnswer(t *testing.T) {
	r, ctx := runner(t), context.Background()
	if _, err := r.Start(ctx, "build", "missing", "01-sum"); err != nil {
		t.Fatal(err)
	}
	res, _ := r.Check(ctx, "build", "missing", "01-sum")
	if !strings.Contains(res.Output, `needs "no-such-compiler-anywhere"`) {
		t.Errorf("output = %q", res.Output)
	}
}

func TestOnlyFilesOfTheAttemptAreWritten(t *testing.T) {
	r, ctx := runner(t), context.Background()
	if _, err := r.Start(ctx, "build", "sh", "01-sum"); err != nil {
		t.Fatal(err)
	}
	for _, rel := range []string{"../../../../escape.sh", "/tmp/abs.sh", "new-file.sh"} {
		if err := r.Save("build", "sh", "01-sum", rel, "x"); err == nil {
			t.Errorf("Save(%q) succeeded", rel)
		}
	}
	for _, name := range []string{"--help", "../01-sum", "a b", ""} {
		if _, err := r.Check(ctx, "build", "sh", name); err == nil {
			t.Errorf("Check with name %q ran", name)
		}
	}
}

func TestStripSolutionKeepsCommentsClosed(t *testing.T) {
	in := "int f(void) {\n  /* SOLUTION-BEGIN */\n  return 1;\n  /* SOLUTION-END */\n}\n// mentions SOLUTION-BEGIN in prose, which is not a marker\n"
	got := string(practice.StripSolution([]byte(in)))
	want := "int f(void) {\n  /* TODO: your implementation goes here. */\n}\n// mentions SOLUTION-BEGIN in prose, which is not a marker\n"
	if got != want {
		t.Errorf("got:\n%s\nwant:\n%s", got, want)
	}
	if kept := string(practice.DropMarkers([]byte(in))); !strings.Contains(kept, "return 1;") || strings.Contains(kept, "/* SOLUTION") {
		t.Errorf("DropMarkers = %q", kept)
	}
}

// This repository's own spec, against its real exercises. A C compiler is on
// every machine that builds the app, so one real exercise is driven end to
// end: the stub fails, and the reference with its markers dropped passes.
func TestThisPacksSpecRunsARealExercise(t *testing.T) {
	root := "../../.."
	r, err := practice.New(root, "practice/exercises.json")
	if err != nil {
		t.Skip("no exercise spec beside the app:", err)
	}
	t.Setenv("OL_SCRATCH", t.TempDir())
	ctx := context.Background()

	groups, _ := r.List(ctx)
	total := map[string]int{}
	for _, g := range groups {
		total[g.Kind] += len(g.Exercises)
	}
	if total["predict"] == 0 || total["build"] == 0 || total["reattempt"] == 0 {
		t.Fatalf("exercises by kind = %v, want all three kinds", total)
	}

	const name = "01-dynamic-array"
	ws, err := r.Start(ctx, "build", "c", name)
	if err != nil {
		t.Fatal(err)
	}
	if res, _ := r.Check(ctx, "build", "c", name); res.Verdict != "fail" {
		t.Fatalf("the stub should fail, got %s:\n%s", res.Verdict, res.Output)
	}
	for _, f := range ws.Files {
		ref, err := os.ReadFile(filepath.Join(root, "practice/build/systems/c", name, f.Path))
		if err != nil {
			t.Fatal(err)
		}
		if err := r.Save("build", "c", name, f.Path, string(practice.DropMarkers(ref))); err != nil {
			t.Fatal(err)
		}
	}
	if res, _ := r.Check(ctx, "build", "c", name); res.Verdict != "pass" {
		t.Fatalf("the reference should pass, got %s:\n%s", res.Verdict, res.Output)
	}
}

// The spec is a second description of a tree that practice/bin/ol also walks.
// This fails when the two drift: a language directory or a reattempt row with
// no set in the spec, or a set pointing at something that is not there.
func TestThisPacksSpecCoversTheExerciseTree(t *testing.T) {
	root := "../../.."
	data, err := os.ReadFile(filepath.Join(root, "practice/exercises.json"))
	if err != nil {
		t.Skip("no exercise spec beside the app:", err)
	}
	var spec practice.Spec
	if err := json.Unmarshal(data, &spec); err != nil {
		t.Fatal(err)
	}

	type key struct{ kind, lang string }
	sets := map[key]practice.Set{}
	items := map[string]string{} // "track<TAB>name" -> source
	for _, s := range spec.Sets {
		sets[key{s.Kind, s.Lang}] = s
		if s.Glob != "" {
			// The directory a glob looks in must exist, even when it is empty.
			if _, err := os.Stat(filepath.Join(root, filepath.Dir(filepath.FromSlash(s.Glob)))); err != nil {
				t.Errorf("set %s/%s globs a directory that does not exist: %s", s.Kind, s.Lang, s.Glob)
			}
		}
		for _, it := range s.Items {
			items[s.Lang+"\t"+it.Name] = it.Source
			if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(it.Source))); err != nil {
				t.Errorf("set %s/%s lists %s, which does not exist", s.Kind, s.Lang, it.Source)
			}
		}
	}

	// Every language directory on disk has a set: practice/predict/<lang> and
	// practice/build/<tier>/<lang>.
	langDirs := map[string][]string{"predict": {"practice/predict/*"}, "build": {"practice/build/*/*"}}
	for kind, patterns := range langDirs {
		for _, pattern := range patterns {
			dirs, _ := filepath.Glob(filepath.Join(root, filepath.FromSlash(pattern)))
			for _, d := range dirs {
				if info, err := os.Stat(d); err != nil || !info.IsDir() {
					continue
				}
				if _, ok := sets[key{kind, filepath.Base(d)}]; !ok {
					t.Errorf("%s has no %s set in practice/exercises.json", d, kind)
				}
			}
		}
	}

	// Every reattempt manifest row is an item, and every item is a row.
	manifest, err := os.ReadFile(filepath.Join(root, "practice/reattempt/manifest.tsv"))
	if err != nil {
		t.Fatal(err)
	}
	rows := map[string]string{}
	for _, line := range strings.Split(string(manifest), "\n") {
		if strings.TrimSpace(line) == "" || strings.HasPrefix(strings.TrimSpace(line), "#") {
			continue
		}
		cols := strings.Split(line, "\t")
		if len(cols) < 3 {
			t.Errorf("malformed manifest row: %q", line)
			continue
		}
		rows[cols[0]+"\t"+cols[1]] = cols[2]
	}
	for k, src := range rows {
		if items[k] != src {
			t.Errorf("reattempt row %q -> %s is missing from the spec (spec has %q)", k, src, items[k])
		}
	}
	for k := range items {
		if _, ok := rows[k]; !ok && strings.Contains(k, "\t") {
			if _, isReattempt := sets[key{"reattempt", strings.SplitN(k, "\t", 2)[0]}]; isReattempt {
				t.Errorf("spec item %q is not a row of practice/reattempt/manifest.tsv", k)
			}
		}
	}
}
