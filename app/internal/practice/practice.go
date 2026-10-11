// Package practice drives a pack's practice CLI on the learner's behalf. The
// CLI owns the exercises, the scratchpad layout, and the verdicts; this package
// only runs it and reads what it left on disk, so an exercise checked in the
// app and one checked in a terminal are the same exercise.
//
// The contract is the one practice/bin/ss implements: `list`, `start`, `check`,
// `diff`, `show`, `reset`, each taking <kind> <lang> <name>, with `check`
// exiting 0 for a pass, 1 for a fail, and 2 when nothing has been attempted.
package practice

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

// Group is the exercises of one kind in one language.
type Group struct {
	Kind      string     `json:"kind"`
	Lang      string     `json:"lang"`
	Exercises []Exercise `json:"exercises"`
}

// Exercise is one row of a group.
type Exercise struct {
	Name    string `json:"name"`
	Started bool   `json:"started"`
}

// File is one file of an attempt.
type File struct {
	// Path is relative to the workspace.
	Path    string `json:"path"`
	Content string `json:"content"`
	// ReadOnly marks what the learner reads but does not write: the snippet
	// whose output a predict exercise asks them to predict.
	ReadOnly bool `json:"readOnly"`
}

// Workspace is an attempt as the app shows it.
type Workspace struct {
	Kind  string `json:"kind"`
	Lang  string `json:"lang"`
	Name  string `json:"name"`
	Dir   string `json:"dir"`
	Files []File `json:"files"`
}

// Result is the outcome of one command.
type Result struct {
	// Verdict is "pass", "fail", or "none" (nothing attempted, or the command
	// has no verdict).
	Verdict string `json:"verdict"`
	Output  string `json:"output"`
}

// kindPredict is the one kind whose attempt is a single prediction file
// rather than a directory.
const kindPredict = "predict"

// The verdicts a Result carries.
const (
	verdictPass = "pass"
	verdictFail = "fail"
	verdictNone = "none"
)

// Runner runs one pack's practice CLI.
type Runner struct {
	// Root is the pack checkout; commands run with it as their directory.
	Root string
	// CLI is the executable, absolute.
	CLI string
}

// New returns a runner for the CLI at rel inside root.
func New(root, rel string) (*Runner, error) {
	root, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("practice: %w", err)
	}
	cli := filepath.Join(root, filepath.FromSlash(rel))
	info, err := os.Stat(cli)
	if err != nil {
		return nil, fmt.Errorf("practice: %w", err)
	}
	if info.IsDir() || info.Mode()&0o111 == 0 {
		return nil, fmt.Errorf("practice: %s is not executable", cli)
	}
	return &Runner{Root: root, CLI: cli}, nil
}

// timeout bounds one command. Compiling a first Rust exercise is the slow
// case; an exercise that loops forever must not hang the app.
const timeout = 3 * time.Minute

// run executes the CLI and returns its combined output and exit code.
func (r *Runner) run(ctx context.Context, args ...string) (string, int, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, r.CLI, args...) //nolint:gosec // G204: the CLI is the pack's own, arguments are validated names
	cmd.Dir = r.Root
	// EDITOR is what `start` opens the attempt in. The app is the editor, so
	// it is pointed at a command that does nothing.
	cmd.Env = append(shellEnv(), "EDITOR=true", "NO_COLOR=1")
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out

	err := cmd.Run()
	text := strings.TrimRight(out.String(), "\n")
	if ctx.Err() == context.DeadlineExceeded {
		return text, -1, fmt.Errorf("practice: timed out after %s", timeout)
	}
	var exit *exec.ExitError
	if errors.As(err, &exit) {
		return text, exit.ExitCode(), nil
	}
	if err != nil {
		return text, -1, fmt.Errorf("practice: %w", err)
	}
	return text, 0, nil
}

// List returns every exercise, grouped as the CLI groups them.
func (r *Runner) List(ctx context.Context) ([]Group, error) {
	out, code, err := r.run(ctx, "list")
	if err != nil {
		return nil, err
	}
	if code != 0 {
		return nil, fmt.Errorf("practice: list failed: %s", firstLine(out))
	}
	return ParseList(out), nil
}

// ParseList reads `list` output: a "kind/lang" heading, then one indented row
// per exercise ending in "started" or "--".
func ParseList(out string) []Group {
	groups := []Group{}
	sc := bufio.NewScanner(strings.NewReader(out))
	for sc.Scan() {
		line := sc.Text()
		if strings.TrimSpace(line) == "" {
			continue
		}
		if !strings.HasPrefix(line, " ") {
			kind, lang, ok := strings.Cut(strings.TrimSpace(line), "/")
			if !ok {
				continue
			}
			groups = append(groups, Group{Kind: kind, Lang: lang, Exercises: []Exercise{}})
			continue
		}
		fields := strings.Fields(line)
		if len(groups) == 0 || len(fields) < 2 {
			continue
		}
		g := &groups[len(groups)-1]
		g.Exercises = append(g.Exercises, Exercise{Name: fields[0], Started: fields[len(fields)-1] == "started"})
	}
	return groups
}

// Start clones the exercise into the scratchpad (a no-op when it is already
// there) and returns the attempt.
func (r *Runner) Start(ctx context.Context, kind, lang, name string) (Workspace, error) {
	if err := validate(kind, lang, name); err != nil {
		return Workspace{}, err
	}
	if out, code, err := r.run(ctx, "start", kind, lang, name); err != nil {
		return Workspace{}, err
	} else if code != 0 {
		return Workspace{}, fmt.Errorf("practice: start failed: %s", firstLine(out))
	}
	return r.Workspace(ctx, kind, lang, name)
}

// Workspace reads an attempt from the scratchpad.
func (r *Runner) Workspace(ctx context.Context, kind, lang, name string) (Workspace, error) {
	if err := validate(kind, lang, name); err != nil {
		return Workspace{}, err
	}
	ws := Workspace{Kind: kind, Lang: lang, Name: name, Files: []File{}}

	if kind == kindPredict {
		// A prediction is one text file beside its siblings; the snippet it
		// predicts stays in the pack and is shown read-only.
		ws.Dir = filepath.Join(r.scratch(), kind, lang)
		snippet, code, err := r.run(ctx, "show", kind, lang, name)
		if err != nil {
			return Workspace{}, err
		}
		if code != 0 {
			return Workspace{}, fmt.Errorf("practice: show failed: %s", firstLine(snippet))
		}
		title, body, _ := strings.Cut(snippet, "\n")
		ws.Files = append(ws.Files, File{Path: strings.TrimSpace(title), Content: strings.TrimLeft(body, "\n"), ReadOnly: true})
		pred := name + ".txt"
		data, _ := os.ReadFile(filepath.Join(ws.Dir, pred)) //nolint:gosec // G304: validated name under the scratchpad
		ws.Files = append(ws.Files, File{Path: pred, Content: string(data)})
		return ws, nil
	}

	ws.Dir = filepath.Join(r.scratch(), kind, lang, name)
	err := filepath.WalkDir(ws.Dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		info, err := d.Info()
		if err != nil || info.Size() > maxFileBytes {
			return nil
		}
		data, err := os.ReadFile(p) //nolint:gosec // G304: walking the attempt's own directory
		if err != nil || !utf8.Valid(data) {
			return nil // binaries a build left behind are not part of the attempt
		}
		rel, err := filepath.Rel(ws.Dir, p)
		if err != nil {
			return nil
		}
		ws.Files = append(ws.Files, File{Path: filepath.ToSlash(rel), Content: string(data)})
		return nil
	})
	if err != nil {
		return Workspace{}, fmt.Errorf("practice: not started: %s/%s/%s", kind, lang, name)
	}
	sort.Slice(ws.Files, func(i, j int) bool { return ws.Files[i].Path < ws.Files[j].Path })
	return ws, nil
}

// maxFileBytes keeps a stray build artifact from being read into the editor.
const maxFileBytes = 1 << 20

// Save writes one file of an attempt.
func (r *Runner) Save(kind, lang, name, rel, content string) error {
	if err := validate(kind, lang, name); err != nil {
		return err
	}
	if !fs.ValidPath(rel) {
		return fmt.Errorf("practice: bad file path %q", rel)
	}
	dir := filepath.Join(r.scratch(), kind, lang, name)
	if kind == kindPredict {
		if rel != name+".txt" {
			return fmt.Errorf("practice: a prediction is written to %s.txt", name)
		}
		dir = filepath.Join(r.scratch(), kind, lang)
	}
	target := filepath.Join(dir, filepath.FromSlash(rel))
	// Only files the CLI created are written, so the app cannot add files to
	// an attempt or be talked into writing somewhere new.
	if kind != kindPredict {
		if _, err := os.Stat(target); err != nil {
			return fmt.Errorf("practice: %s is not part of this attempt", rel)
		}
	}
	if err := os.MkdirAll(filepath.Dir(target), 0o750); err != nil {
		return fmt.Errorf("practice: %w", err)
	}
	if err := os.WriteFile(target, []byte(content), 0o644); err != nil { //nolint:gosec // G306: source files the learner's own tools also read
		return fmt.Errorf("practice: save %s: %w", rel, err)
	}
	return nil
}

// Check runs the attempt and reports the verdict.
func (r *Runner) Check(ctx context.Context, kind, lang, name string) (Result, error) {
	if err := validate(kind, lang, name); err != nil {
		return Result{}, err
	}
	out, code, err := r.run(ctx, "check", kind, lang, name)
	if err != nil {
		return Result{Verdict: verdictFail, Output: strings.TrimSpace(out + "\n" + err.Error())}, nil
	}
	verdict := verdictFail
	switch code {
	case 0:
		verdict = verdictPass
	case 2:
		verdict = verdictNone
	}
	return Result{Verdict: verdict, Output: out}, nil
}

// Diff compares the attempt with the reference. It spoils the exercise, which
// is the learner's call to make.
func (r *Runner) Diff(ctx context.Context, kind, lang, name string) (Result, error) {
	return r.plain(ctx, "diff", kind, lang, name)
}

// Reset throws the attempt away.
func (r *Runner) Reset(ctx context.Context, kind, lang, name string) (Result, error) {
	return r.plain(ctx, "reset", kind, lang, name)
}

func (r *Runner) plain(ctx context.Context, verb, kind, lang, name string) (Result, error) {
	if err := validate(kind, lang, name); err != nil {
		return Result{}, err
	}
	out, code, err := r.run(ctx, verb, kind, lang, name)
	if err != nil {
		return Result{}, err
	}
	if code != 0 {
		return Result{}, fmt.Errorf("practice: %s", firstLine(out))
	}
	return Result{Verdict: verdictNone, Output: out}, nil
}

// Learn marks a path stage through the CLI, which is what enforces a stage's
// check. It returns what the CLI said either way.
func (r *Runner) Learn(ctx context.Context, top, key string, done bool) (Result, error) {
	if err := validate(top, "x", "x"); err != nil {
		return Result{}, err
	}
	if !argSafe(strings.ReplaceAll(key, ":", "")) {
		return Result{}, fmt.Errorf("practice: bad stage %q", key)
	}
	flag := "--undo"
	if done {
		flag = "--done"
	}
	out, code, err := r.run(ctx, "learn", top, flag, key)
	if err != nil {
		return Result{}, err
	}
	verdict := verdictPass
	if code != 0 {
		verdict = verdictFail
	}
	return Result{Verdict: verdict, Output: out}, nil
}

func (r *Runner) scratch() string {
	if v := os.Getenv("SS_SCRATCH"); v != "" {
		return v
	}
	return filepath.Join(r.Root, ".scratchpad")
}

// validate keeps every argument a plain name. They become both command
// arguments and path elements, so nothing that could be a flag or climb a
// directory is let through.
func validate(parts ...string) error {
	for _, p := range parts {
		if !argSafe(p) {
			return fmt.Errorf("practice: bad name %q", p)
		}
	}
	return nil
}

func argSafe(s string) bool {
	if s == "" || s[0] == '-' || s[0] == '.' {
		return false
	}
	for _, c := range s {
		ok := c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-' || c == '_' || c == '.' || c == '+'
		if !ok {
			return false
		}
	}
	return !strings.Contains(s, "..")
}

func firstLine(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

var (
	envOnce sync.Once
	envVars []string
)

// shellEnv returns the environment commands run in. A desktop app launched
// from Finder or a dock gets a bare PATH with no compilers on it, while the
// learner's toolchains are set up by their shell profile. Asking a login shell
// for its PATH once closes that gap; if the shell cannot be asked, the app's
// own environment is used as is.
func shellEnv() []string {
	envOnce.Do(func() {
		envVars = os.Environ()
		shell := os.Getenv("SHELL")
		if shell == "" {
			return
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		// Markers bracket the value because an interactive profile may print
		// a banner around it.
		out, err := exec.CommandContext(ctx, shell, "-ilc", `printf '<<PATH:%s>>' "$PATH"`).Output() //nolint:gosec // G204: the user's own login shell, fixed script
		if err != nil {
			return
		}
		_, rest, ok := strings.Cut(string(out), "<<PATH:")
		if !ok {
			return
		}
		path, _, ok := strings.Cut(rest, ">>")
		if !ok || path == "" {
			return
		}
		kept := envVars[:0:0]
		for _, kv := range envVars {
			if !strings.HasPrefix(kv, "PATH=") {
				kept = append(kept, kv)
			}
		}
		envVars = append(kept, "PATH="+path)
	})
	return append([]string(nil), envVars...)
}
