// Package workspace is one open course pack with everything that hangs off it:
// its documents, the learner's progress, and its practice CLI. The desktop app
// and the MCP server both open a workspace, which is what keeps a stage marked
// by an agent and one marked in the app the same mark.
package workspace

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

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
	// Runner is nil when the pack has no practice CLI or is not on disk.
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
			log.Warn("practice CLI unavailable", "err", err)
		}
		w.Runner = runner
	}
	// A pack with a working practice CLI keeps progress where the CLI does, so
	// the two agree. Any other pack has nowhere of its own to write.
	if w.Runner != nil {
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
// learner. A stage tied to a check is only marked by the practice CLI, which
// is what knows whether the check passed.
func (w *Workspace) MarkStage(ctx context.Context, top, key string, done bool) (practice.Result, error) {
	stage, err := w.Course.Stage(top, key)
	if err != nil {
		return practice.Result{}, err
	}
	if w.Runner != nil {
		return w.Runner.Learn(ctx, top, key, done)
	}
	if done && stage.Check != "" {
		return practice.Result{
			Verdict: "fail",
			Output:  fmt.Sprintf("This stage is finished by passing %s, which needs the pack's practice CLI. Open the pack from a checkout to run it.", stage.Check),
		}, nil
	}
	if err := w.Progress.Mark(stage.Owner, stage.Number, done); err != nil {
		return practice.Result{}, err
	}
	return practice.Result{Verdict: "pass"}, nil
}
