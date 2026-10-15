// Package workspace is one open course pack with everything that hangs off it:
// its documents, the learner's progress, and its exercises. The desktop app
// and the MCP server both open a workspace, which is what keeps a stage marked
// by an agent and one marked in the app the same mark.
package workspace

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/urmzd/openlearn/app/internal/builtin"
	"github.com/urmzd/openlearn/app/internal/course"
	"github.com/urmzd/openlearn/app/internal/packs"
	"github.com/urmzd/openlearn/app/internal/practice"
)

// DefaultPack is the pack a first launch opens.
const DefaultPack = "supersource"

// Workspace is a loaded pack.
type Workspace struct {
	// Ref is the pack ref that was opened; empty when nothing could be.
	Ref      string
	Course   *course.Course
	Progress course.DoneFiles
	// Runner is nil when the pack has no exercises or is not on disk.
	Runner *practice.Runner
}

// Registry returns the pack registry for a store rooted at storeRoot, searching
// upward from the working directory for a checkout.
func Registry(storeRoot string) packs.Registry {
	cwd, _ := os.Getwd()
	return packs.Registry{
		Builtin:    builtin.FS(),
		InstallDir: filepath.Join(storeRoot, "packs"),
		Checkout:   cwd,
		Preferred:  DefaultPack,
	}
}

// Open loads the pack ref names, falling back to the default when ref is empty
// or no longer resolves (a linked directory that was moved, say). It always
// returns a usable workspace: with no pack at all, the course is empty.
func Open(reg packs.Registry, ref string, linked []string, storeRoot string, log *slog.Logger) *Workspace {
	c, err := reg.Open(ref)
	if ref == "" || err != nil {
		if ref != "" {
			log.Warn("saved pack is gone, opening the default", "ref", ref, "err", err)
		}
		ref = ""
		if def, ok := reg.Default(linked); ok {
			ref = def.Ref
			c, err = reg.Open(ref)
		}
	}
	if ref == "" || err != nil {
		log.Warn("no course pack available", "err", err)
		c, ref = course.Empty(), ""
	}

	w := &Workspace{Ref: ref, Course: c}
	info := c.Info()
	if info.Practice {
		runner, err := practice.New(c.Dir(), c.Manifest().Practice)
		if err != nil {
			log.Warn("exercises unavailable", "err", err)
		}
		w.Runner = runner
	}
	// A pack on disk keeps progress inside itself, in the layout its own
	// tooling reads, so the two agree. A built-in pack has nowhere of its own
	// to write.
	if c.Dir() != "" {
		w.Progress = course.CheckoutProgress(c.Dir())
	} else {
		w.Progress = course.StoreProgress(filepath.Join(storeRoot, "learn", info.ID))
	}
	return w
}

// Info describes the open pack, with what can actually run here filled in.
func (w *Workspace) Info() course.Info {
	info := w.Course.Info()
	info.Practice = w.Runner != nil
	if len(info.Modes) == 0 {
		info.Modes = []string{"system_design", "coding", "written"}
	}
	return info
}

// Paths summarizes the pack's role paths with progress.
func (w *Workspace) Paths() []course.PathSummary { return w.Course.Paths(w.Progress) }

// Path expands one role path with progress.
func (w *Workspace) Path(name string) (course.Path, error) { return w.Course.Path(name, w.Progress) }

// MarkStage marks a path stage finished or not, and returns what to tell the
// learner. A stage tied to a check is only marked when the pack's own stage
// check says it passed: nothing here can decide that on the pack's behalf.
func (w *Workspace) MarkStage(ctx context.Context, top, key string, done bool) (practice.Result, error) {
	stage, err := w.Course.Stage(top, key)
	if err != nil {
		return practice.Result{}, err
	}
	if done && stage.Check != "" {
		return w.checkedStage(ctx, top, key, stage)
	}
	if err := w.Progress.Mark(stage.Owner, stage.Number, done); err != nil {
		return practice.Result{}, err
	}
	return practice.Result{Verdict: practice.VerdictPass}, nil
}

// stageCheckTimeout bounds the pack's stage check, which may run tests.
const stageCheckTimeout = 5 * time.Minute

// checkedStage hands a check-gated stage to the pack's stage check command.
// The command is the pack's own, named in its manifest and run directly; it is
// looked up at the moment it is needed, so a pack whose tooling moved reports
// that instead of failing on a path remembered from earlier.
func (w *Workspace) checkedStage(ctx context.Context, top, key string, stage course.Stage) (practice.Result, error) {
	refuse := func(format string, args ...any) (practice.Result, error) {
		return practice.Result{Verdict: practice.VerdictFail, Output: fmt.Sprintf(format, args...)}, nil
	}
	argv := w.Course.Manifest().StageCheck
	if w.Course.Dir() == "" || len(argv) == 0 {
		return refuse("This stage is finished by passing %s, which this pack can only verify from a folder on disk with its own tooling.", stage.Check)
	}
	fill := strings.NewReplacer("{path}", top, "{stage}", key)
	args := make([]string, len(argv))
	for i, a := range argv {
		args[i] = fill.Replace(a)
	}
	exe := args[0]
	if strings.ContainsRune(exe, '/') && !filepath.IsAbs(exe) {
		exe = filepath.Join(w.Course.Dir(), filepath.FromSlash(exe))
	}
	if _, err := exec.LookPath(exe); err != nil {
		return refuse("This stage is finished by passing %s, but the pack's stage check (%s) is not there to run.", stage.Check, argv[0])
	}

	ctx, cancel := context.WithTimeout(ctx, stageCheckTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, exe, args[1:]...) //nolint:gosec // G204: the pack's declared stage check, arguments are a path name and a stage key from the pack
	cmd.Dir = w.Course.Dir()
	cmd.Env = append(os.Environ(), "NO_COLOR=1")
	out, err := cmd.CombinedOutput()
	text := strings.TrimSpace(string(out))
	if err != nil {
		if text == "" {
			text = fmt.Sprintf("%s has not passed yet.", stage.Check)
		}
		return practice.Result{Verdict: practice.VerdictFail, Output: text}, nil
	}
	return practice.Result{Verdict: practice.VerdictPass, Output: text}, nil
}
