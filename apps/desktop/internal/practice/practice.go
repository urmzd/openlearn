// Package practice is the exercise engine: it lists a pack's exercises, clones
// one into the learner's scratchpad with the solution removed, and checks an
// attempt by running it. It is the only implementation of that loop. The app
// and the MCP server call it directly, so nothing has to locate, launch, or
// parse the output of a separate CLI.
//
// A pack describes its exercises in a spec file (see Spec). The engine supplies
// the two ways an exercise can be judged:
//
//	predict   the learner writes the exact output of a snippet; the snippet is
//	          run where it lives and the two are compared.
//	attempt   the learner writes the code; their copy is run and its exit code
//	          is the verdict.
//
// Running an attempt means running a compiler or an interpreter, which is the
// one process this package starts. Those commands are argument lists from the
// spec, executed directly with no shell in between.
package practice

import (
	"bytes"
	"context"
	"encoding/json"
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

// Spec is a pack's exercise description: sets of exercises that are found and
// run the same way.
type Spec struct {
	// Scratch is where attempts live, relative to the pack. It is the
	// learner's own space and should be ignored by the pack's version control.
	Scratch string `json:"scratch"`
	Sets    []Set  `json:"sets"`
}

// Set is the exercises of one kind in one language.
type Set struct {
	Kind string `json:"kind"`
	Lang string `json:"lang"`
	// Mode is "predict" or "attempt".
	Mode string `json:"mode"`
	// Glob finds the exercises: each match is one, named by its base name
	// without extension. A match may be a file or a directory.
	Glob string `json:"glob"`
	// Items lists exercises one by one, for sets that point at files living
	// elsewhere in the pack.
	Items []Item `json:"items"`
	// Show is the file to display for a predict exercise whose match is a
	// directory.
	Show string `json:"show"`
	// Seed files are written into a new attempt when absent, such as the
	// go.mod a Go directory needs once it is copied out of its module.
	Seed map[string]string `json:"seed"`
	Run  Run               `json:"run"`
}

// Item is one explicitly listed exercise.
type Item struct {
	Name   string `json:"name"`
	Source string `json:"source"`
}

// Run is how a set's exercises are executed. Steps run in order and stop at the
// first failure. In an argument, {name} is the exercise name, {out} a build
// directory outside the attempt, and {file} the attempt's single source file;
// an argument containing * is expanded against the working directory.
type Run struct {
	// Dir is the working directory for predict sets, relative to the pack.
	// Attempt sets run in the attempt's own directory.
	Dir   string     `json:"dir"`
	Steps [][]string `json:"steps"`
}

const (
	modePredict = "predict"
	modeAttempt = "attempt"
)

// Group is the exercises of one kind in one language, as the UI lists them.
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

// The verdicts a Result carries.
const (
	VerdictPass = "pass"
	VerdictFail = "fail"
	VerdictNone = "none"
)

// Runner is the engine bound to one pack on disk.
type Runner struct {
	// Root is the pack directory.
	Root string
	spec Spec
}

// New loads the spec at rel inside root.
func New(root, rel string) (*Runner, error) {
	root, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("practice: %w", err)
	}
	data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel))) //nolint:gosec // G304: the spec path comes from the pack's own manifest
	if err != nil {
		return nil, fmt.Errorf("practice: %w", err)
	}
	var spec Spec
	if err := json.Unmarshal(data, &spec); err != nil {
		return nil, fmt.Errorf("practice: parse %s: %w", rel, err)
	}
	if spec.Scratch == "" {
		spec.Scratch = ".scratchpad"
	}
	for i, s := range spec.Sets {
		if !argSafe(s.Kind) || !argSafe(s.Lang) {
			return nil, fmt.Errorf("practice: set %d needs a plain kind and lang", i)
		}
		if s.Mode != modePredict && s.Mode != modeAttempt {
			return nil, fmt.Errorf("practice: set %s/%s: mode must be %q or %q", s.Kind, s.Lang, modePredict, modeAttempt)
		}
		if len(s.Run.Steps) == 0 {
			return nil, fmt.Errorf("practice: set %s/%s has no run steps", s.Kind, s.Lang)
		}
	}
	return &Runner{Root: root, spec: spec}, nil
}

// located is one exercise resolved to where it lives.
type located struct {
	set  Set
	name string
	// source is the exercise's file or directory, absolute.
	source string
}

func (r *Runner) scratch() string {
	if v := os.Getenv("OL_SCRATCH"); v != "" {
		return v
	}
	return filepath.Join(r.Root, filepath.FromSlash(r.spec.Scratch))
}

// names returns a set's exercises, sorted by name.
func (r *Runner) names(s Set) []located {
	var out []located
	if s.Glob != "" {
		matches, _ := filepath.Glob(filepath.Join(r.Root, filepath.FromSlash(s.Glob)))
		for _, m := range matches {
			name := strings.TrimSuffix(filepath.Base(m), filepath.Ext(m))
			if info, err := os.Stat(m); err == nil && info.IsDir() {
				name = filepath.Base(m)
			}
			if argSafe(name) {
				out = append(out, located{set: s, name: name, source: m})
			}
		}
	}
	for _, it := range s.Items {
		if argSafe(it.Name) && fs.ValidPath(it.Source) {
			out = append(out, located{set: s, name: it.Name, source: filepath.Join(r.Root, filepath.FromSlash(it.Source))})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].name < out[j].name })
	return out
}

func (r *Runner) find(kind, lang, name string) (located, error) {
	if !argSafe(kind) || !argSafe(lang) || !argSafe(name) {
		return located{}, fmt.Errorf("practice: bad exercise name %s/%s/%s", kind, lang, name)
	}
	for _, s := range r.spec.Sets {
		if s.Kind != kind || s.Lang != lang {
			continue
		}
		for _, l := range r.names(s) {
			if l.name == name {
				return l, nil
			}
		}
	}
	return located{}, fmt.Errorf("practice: no %s exercise %q in %s", kind, name, lang)
}

// workDir is where an attempt lives. A prediction is a single file beside its
// siblings; anything else is a directory of its own.
func (r *Runner) workDir(l located) string {
	if l.set.Mode == modePredict {
		return filepath.Join(r.scratch(), l.set.Kind, l.set.Lang)
	}
	return filepath.Join(r.scratch(), l.set.Kind, l.set.Lang, l.name)
}

func (r *Runner) predictionFile(l located) string {
	return filepath.Join(r.workDir(l), l.name+".txt")
}

func (r *Runner) started(l located) bool {
	if l.set.Mode == modePredict {
		info, err := os.Stat(r.predictionFile(l))
		return err == nil && info.Size() > 0
	}
	info, err := os.Stat(r.workDir(l))
	return err == nil && info.IsDir()
}

// List returns every exercise, grouped by set. Sets with no exercises are
// omitted, so a pack can declare a language before it has anything in it.
func (r *Runner) List(context.Context) ([]Group, error) {
	groups := []Group{}
	for _, s := range r.spec.Sets {
		found := r.names(s)
		if len(found) == 0 {
			continue
		}
		g := Group{Kind: s.Kind, Lang: s.Lang, Exercises: make([]Exercise, 0, len(found))}
		for _, l := range found {
			g.Exercises = append(g.Exercises, Exercise{Name: l.name, Started: r.started(l)})
		}
		groups = append(groups, g)
	}
	return groups, nil
}

// Start clones the exercise into the scratchpad with its solution removed (a
// no-op when it is already there) and returns the attempt.
func (r *Runner) Start(ctx context.Context, kind, lang, name string) (Workspace, error) {
	l, err := r.find(kind, lang, name)
	if err != nil {
		return Workspace{}, err
	}
	wd := r.workDir(l)
	if l.set.Mode == modePredict {
		if err := os.MkdirAll(wd, 0o750); err != nil {
			return Workspace{}, fmt.Errorf("practice: %w", err)
		}
		return r.Workspace(ctx, kind, lang, name)
	}
	if _, err := os.Stat(wd); err != nil {
		if err := r.clone(l, wd); err != nil {
			_ = os.RemoveAll(wd) // a half-cloned attempt would look started
			return Workspace{}, err
		}
	}
	return r.Workspace(ctx, kind, lang, name)
}

// clone copies an exercise into wd, replacing each marked solution region with
// a TODO. Files with no markers arrive intact, which is how tests and headers
// come along untouched.
func (r *Runner) clone(l located, wd string) error {
	info, err := os.Stat(l.source)
	if err != nil {
		return fmt.Errorf("practice: %w", err)
	}
	copyOne := func(src, dst string) error {
		data, err := os.ReadFile(src) //nolint:gosec // G304: a file of the pack's own exercise
		if err != nil {
			return err
		}
		st, err := os.Stat(src)
		if err != nil {
			return err
		}
		if err := os.MkdirAll(filepath.Dir(dst), 0o750); err != nil {
			return err
		}
		// Executable bits survive so an exercise's own run script still runs.
		mode := fs.FileMode(0o644)
		if st.Mode()&0o100 != 0 {
			mode = 0o755
		}
		return os.WriteFile(dst, StripSolution(data), mode) //nolint:gosec // G306: source files the learner's own tools also read
	}
	if !info.IsDir() {
		if err := copyOne(l.source, filepath.Join(wd, filepath.Base(l.source))); err != nil {
			return fmt.Errorf("practice: %w", err)
		}
	} else {
		err := filepath.WalkDir(l.source, func(p string, d fs.DirEntry, err error) error {
			if err != nil || d.IsDir() {
				return err
			}
			rel, err := filepath.Rel(l.source, p)
			if err != nil {
				return err
			}
			return copyOne(p, filepath.Join(wd, rel))
		})
		if err != nil {
			return fmt.Errorf("practice: %w", err)
		}
	}
	for rel, body := range l.set.Seed {
		target := filepath.Join(wd, filepath.FromSlash(rel))
		if !fs.ValidPath(rel) {
			continue
		}
		if _, err := os.Stat(target); err == nil {
			continue
		}
		if err := os.WriteFile(target, []byte(strings.ReplaceAll(body, "{name}", l.name)), 0o644); err != nil { //nolint:gosec // G306: as above
			return fmt.Errorf("practice: %w", err)
		}
	}
	return nil
}

// maxFileBytes keeps a stray build artifact from being read into the editor.
const maxFileBytes = 1 << 20

// Workspace reads an attempt from the scratchpad.
func (r *Runner) Workspace(_ context.Context, kind, lang, name string) (Workspace, error) {
	l, err := r.find(kind, lang, name)
	if err != nil {
		return Workspace{}, err
	}
	ws := Workspace{Kind: kind, Lang: lang, Name: name, Dir: r.workDir(l), Files: []File{}}

	if l.set.Mode == modePredict {
		// The snippet stays in the pack and is shown read-only; the learner
		// writes only what they expect it to print.
		snippet := l.source
		if info, err := os.Stat(snippet); err == nil && info.IsDir() {
			snippet = filepath.Join(snippet, filepath.FromSlash(l.set.Show))
		}
		data, err := os.ReadFile(snippet) //nolint:gosec // G304: the pack's own snippet
		if err != nil {
			return Workspace{}, fmt.Errorf("practice: %w", err)
		}
		rel, _ := filepath.Rel(r.Root, snippet)
		ws.Files = append(ws.Files, File{Path: filepath.ToSlash(rel), Content: string(data), ReadOnly: true})
		pred, _ := os.ReadFile(r.predictionFile(l)) //nolint:gosec // G304: validated name under the scratchpad
		ws.Files = append(ws.Files, File{Path: name + ".txt", Content: string(pred)})
		return ws, nil
	}

	err = filepath.WalkDir(ws.Dir, func(p string, d fs.DirEntry, err error) error {
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

// Save writes one file of an attempt.
func (r *Runner) Save(kind, lang, name, rel, content string) error {
	l, err := r.find(kind, lang, name)
	if err != nil {
		return err
	}
	if !fs.ValidPath(rel) {
		return fmt.Errorf("practice: bad file path %q", rel)
	}
	if l.set.Mode == modePredict {
		if rel != name+".txt" {
			return fmt.Errorf("practice: a prediction is written to %s.txt", name)
		}
		if err := os.MkdirAll(r.workDir(l), 0o750); err != nil {
			return fmt.Errorf("practice: %w", err)
		}
		return writeFile(r.predictionFile(l), content)
	}
	target := filepath.Join(r.workDir(l), filepath.FromSlash(rel))
	// Only files the clone created are written, so a caller cannot add files to
	// an attempt or be talked into writing somewhere new.
	if _, err := os.Stat(target); err != nil {
		return fmt.Errorf("practice: %s is not part of this attempt", rel)
	}
	return writeFile(target, content)
}

func writeFile(target, content string) error {
	// The file's existing mode is kept, so saving a run script does not strip
	// its executable bit.
	mode := fs.FileMode(0o644)
	if info, err := os.Stat(target); err == nil {
		mode = info.Mode().Perm()
	}
	if err := os.WriteFile(target, []byte(content), mode); err != nil { //nolint:gosec // G306: source files the learner's own tools also read
		return fmt.Errorf("practice: save %s: %w", filepath.Base(target), err)
	}
	return nil
}

// Check runs the attempt and reports the verdict.
func (r *Runner) Check(ctx context.Context, kind, lang, name string) (Result, error) {
	l, err := r.find(kind, lang, name)
	if err != nil {
		return Result{}, err
	}
	if !r.started(l) {
		return Result{Verdict: VerdictNone, Output: "Not started yet."}, nil
	}

	if l.set.Mode == modePredict {
		actual, ok := r.exec(ctx, l, filepath.Join(r.Root, filepath.FromSlash(l.set.Run.Dir)))
		if !ok {
			// The snippet itself failed to run, which says nothing about the
			// prediction: usually a toolchain that is not installed.
			return Result{Verdict: VerdictNone, Output: "The snippet could not be run, so there is nothing to compare against.\n\n" + actual}, nil
		}
		pred, _ := os.ReadFile(r.predictionFile(l)) //nolint:gosec // G304: validated name under the scratchpad
		want, got := normalize(string(pred)), normalize(actual)
		if want == got {
			return Result{Verdict: VerdictPass, Output: "PASS " + name}, nil
		}
		return Result{
			Verdict: VerdictFail,
			Output:  "MISS " + name + "\n\n" + Diff("you predicted", "actually printed", want, got),
		}, nil
	}

	out, ok := r.exec(ctx, l, r.workDir(l))
	if ok {
		return Result{Verdict: VerdictPass, Output: strings.TrimSpace("PASS " + name + "\n\n" + out)}, nil
	}
	return Result{Verdict: VerdictFail, Output: strings.TrimSpace("FAIL " + name + "\n\n" + out)}, nil
}

// Diff compares the attempt with the reference. It spoils the exercise, which
// is the learner's call to make.
func (r *Runner) Diff(_ context.Context, kind, lang, name string) (Result, error) {
	l, err := r.find(kind, lang, name)
	if err != nil {
		return Result{}, err
	}
	if l.set.Mode == modePredict {
		return Result{}, errors.New("practice: a prediction has no reference to compare with; check it")
	}
	wd := r.workDir(l)
	if _, err := os.Stat(wd); err != nil {
		return Result{}, fmt.Errorf("practice: not started: %s/%s/%s", kind, lang, name)
	}
	var out strings.Builder
	compare := func(src, rel string) {
		ref, err := os.ReadFile(src) //nolint:gosec // G304: the pack's own reference
		if err != nil || !utf8.Valid(ref) {
			return
		}
		mine, _ := os.ReadFile(filepath.Join(wd, rel)) //nolint:gosec // G304: a file of the attempt
		a, b := string(mine), string(DropMarkers(ref))
		if a == b {
			return
		}
		out.WriteString(Diff("yours: "+rel, "reference: "+rel, a, b))
		out.WriteString("\n")
	}
	info, err := os.Stat(l.source)
	if err != nil {
		return Result{}, fmt.Errorf("practice: %w", err)
	}
	if !info.IsDir() {
		compare(l.source, filepath.Base(l.source))
	} else {
		_ = filepath.WalkDir(l.source, func(p string, d fs.DirEntry, err error) error {
			if err == nil && !d.IsDir() {
				if rel, rerr := filepath.Rel(l.source, p); rerr == nil {
					compare(p, rel)
				}
			}
			return nil
		})
	}
	text := strings.TrimSpace(out.String())
	if text == "" {
		text = "Your attempt matches the reference."
	}
	return Result{Verdict: VerdictNone, Output: text}, nil
}

// Reset throws the attempt away.
func (r *Runner) Reset(_ context.Context, kind, lang, name string) (Result, error) {
	l, err := r.find(kind, lang, name)
	if err != nil {
		return Result{}, err
	}
	target := r.workDir(l)
	if l.set.Mode == modePredict {
		target = r.predictionFile(l)
	}
	// The target is built from validated names under the scratchpad, and this
	// is the last check before a recursive delete.
	if rel, err := filepath.Rel(r.scratch(), target); err != nil || strings.HasPrefix(rel, "..") || rel == "." {
		return Result{}, fmt.Errorf("practice: refusing to delete outside the scratchpad: %s", target)
	}
	if err := os.RemoveAll(target); err != nil {
		return Result{}, fmt.Errorf("practice: %w", err)
	}
	return Result{Verdict: VerdictNone, Output: "Attempt cleared."}, nil
}

// timeout bounds one check. Compiling a first Rust exercise is the slow case;
// an attempt that loops forever must not hang the app.
const timeout = 3 * time.Minute

// exec runs an exercise's steps in dir and returns their combined output and
// whether every step succeeded. An attempt that ships its own executable
// run.sh is run through that instead, for exercises that need something the
// step list cannot express.
func (r *Runner) exec(ctx context.Context, l located, dir string) (string, bool) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	steps := l.set.Run.Steps
	if l.set.Mode == modeAttempt {
		if info, err := os.Stat(filepath.Join(dir, "run.sh")); err == nil && info.Mode()&0o111 != 0 {
			steps = [][]string{{"./run.sh"}}
		}
	}

	key := strings.NewReplacer("/", "_", string(filepath.Separator), "_").Replace(l.set.Kind + "/" + l.set.Lang + "/" + l.name)
	outDir := filepath.Join(r.scratch(), ".build", key)
	if err := os.MkdirAll(outDir, 0o750); err != nil {
		return err.Error(), false
	}
	file := ""
	if entries, err := os.ReadDir(dir); err == nil && l.set.Mode == modeAttempt {
		for _, e := range entries {
			if !e.IsDir() {
				file = e.Name()
				break
			}
		}
	}
	fill := strings.NewReplacer("{name}", l.name, "{out}", outDir, "{file}", file)

	var out bytes.Buffer
	for _, step := range steps {
		var args []string
		for _, a := range step {
			a = fill.Replace(a)
			// A bare pattern is expanded here because there is no shell to do
			// it: "*.c" means the attempt's C files.
			if strings.ContainsAny(a, "*?") && !strings.ContainsRune(a, filepath.Separator) {
				if matches, _ := filepath.Glob(filepath.Join(dir, a)); len(matches) > 0 {
					for _, m := range matches {
						args = append(args, filepath.Base(m))
					}
					continue
				}
			}
			args = append(args, a)
		}
		if len(args) == 0 {
			continue
		}
		cmd := exec.CommandContext(ctx, args[0], args[1:]...) //nolint:gosec // G204: argument lists from the pack's spec, no shell
		cmd.Dir = dir
		cmd.Env = shellEnv()
		cmd.Stdout = &out
		cmd.Stderr = &out
		err := cmd.Run()
		if ctx.Err() == context.DeadlineExceeded {
			fmt.Fprintf(&out, "\nStopped after %s.", timeout)
			return strings.TrimRight(out.String(), "\n"), false
		}
		var exit *exec.ExitError
		if err != nil && !errors.As(err, &exit) {
			// Not an exit code: the command could not be started at all.
			if errors.Is(err, exec.ErrNotFound) {
				fmt.Fprintf(&out, "This exercise needs %q, which is not installed or not on your PATH.", args[0])
			} else {
				fmt.Fprintf(&out, "Could not run %s: %v", args[0], err)
			}
			return strings.TrimRight(out.String(), "\n"), false
		}
		if err != nil {
			return strings.TrimRight(out.String(), "\n"), false
		}
	}
	return strings.TrimRight(out.String(), "\n"), true
}

// normalize strips trailing whitespace and leading and trailing blank lines, so
// a correct prediction is not called wrong over an extra newline.
func normalize(s string) string {
	lines := strings.Split(strings.ReplaceAll(s, "\r\n", "\n"), "\n")
	for i := range lines {
		lines[i] = strings.TrimRight(lines[i], " \t")
	}
	for len(lines) > 0 && lines[0] == "" {
		lines = lines[1:]
	}
	for len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	return strings.Join(lines, "\n")
}

// argSafe keeps a name a plain name. Names become path elements, so nothing
// that could climb a directory or look like a flag is let through.
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

var (
	envOnce sync.Once
	envVars []string
)

// shellEnv returns the environment attempts run in. A desktop app launched
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
		p, _, ok := strings.Cut(rest, ">>")
		if !ok || p == "" {
			return
		}
		kept := envVars[:0:0]
		for _, kv := range envVars {
			if !strings.HasPrefix(kv, "PATH=") {
				kept = append(kept, kv)
			}
		}
		envVars = append(kept, "PATH="+p)
	})
	return append([]string(nil), envVars...)
}
