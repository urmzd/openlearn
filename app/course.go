package main

import (
	"errors"
	"fmt"
	"io/fs"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/urmzd/openlearn/app/internal/course"
	"github.com/urmzd/openlearn/app/internal/packs"
	"github.com/urmzd/openlearn/app/internal/practice"
	"github.com/urmzd/openlearn/app/internal/store"
	"github.com/urmzd/openlearn/app/internal/workspace"
	wruntime "github.com/wailsapp/wails/v2/pkg/runtime"
)

// openPack loads the pack ref names; see workspace.Open for the fallbacks.
func (a *App) openPack(ref string) {
	a.mu.RLock()
	linked := a.settings.LinkedPacks
	a.mu.RUnlock()

	w := workspace.Open(a.packs, ref, linked, a.store.Root(), a.log)

	a.mu.Lock()
	a.pack = w
	a.mu.Unlock()
}

func (a *App) current() *workspace.Workspace {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.pack
}

// ---------- packs ----------

// ListPacks returns every course pack the app can open.
func (a *App) ListPacks() []packs.Pack {
	a.mu.RLock()
	linked := a.settings.LinkedPacks
	a.mu.RUnlock()
	return a.packs.List(linked)
}

// OpenPack switches to another pack and remembers the choice.
func (a *App) OpenPack(ref string) (course.Info, error) {
	if _, err := a.packs.Open(ref); err != nil {
		return course.Info{}, err
	}
	a.openPack(ref)
	if err := a.savePackChoice(func(s *store.Settings) { s.Pack = ref }); err != nil {
		return course.Info{}, err
	}
	return a.GetCourse(), nil
}

// ImportPack asks for a pack and adds it: a folder is linked where it stands,
// a zip archive is unpacked into the app's own directory. It returns the new
// pack's ref, or an empty string when the dialog was cancelled.
func (a *App) ImportPack(archive bool) (string, error) {
	var picked string
	var err error
	if archive {
		picked, err = wruntime.OpenFileDialog(a.ctx, wruntime.OpenDialogOptions{
			Title:   "Import a course pack",
			Filters: []wruntime.FileFilter{{DisplayName: "Course pack (*.zip)", Pattern: "*.zip"}},
		})
	} else {
		picked, err = wruntime.OpenDirectoryDialog(a.ctx, wruntime.OpenDialogOptions{Title: "Open a course pack folder"})
	}
	if err != nil || picked == "" {
		return "", err
	}
	return a.addPack(picked, archive)
}

func (a *App) addPack(picked string, archive bool) (string, error) {
	if archive {
		return a.packs.Install(picked)
	}
	dir, err := filepath.Abs(picked)
	if err != nil {
		return "", err
	}
	if _, err := a.packs.Open(dir); err != nil {
		if errors.Is(err, course.ErrNoManifest) {
			return "", fmt.Errorf("%s has no %s, so it is not a course pack", dir, course.ManifestName)
		}
		return "", err
	}
	err = a.savePackChoice(func(s *store.Settings) {
		for _, l := range s.LinkedPacks {
			if l == dir {
				return
			}
		}
		s.LinkedPacks = append(s.LinkedPacks, dir)
	})
	return dir, err
}

func (a *App) savePackChoice(edit func(*store.Settings)) error {
	a.mu.Lock()
	edit(&a.settings)
	s := a.settings
	a.mu.Unlock()
	return a.store.SaveSettings(s)
}

// ---------- reading ----------

// GetCourse describes the open pack.
func (a *App) GetCourse() course.Info { return a.current().Info() }

// GetOutline returns the open pack's browsable tree.
func (a *App) GetOutline() course.Outline { return a.current().Course.Outline() }

// GetDoc reads one document of the open pack.
func (a *App) GetDoc(p string) (course.Doc, error) { return a.current().Course.Doc(p) }

// Search finds documents in the open pack.
func (a *App) Search(query string) []course.Hit { return a.current().Course.Search(query, 40) }

// ListPaths summarizes the open pack's role paths with progress.
func (a *App) ListPaths() []course.PathSummary { return a.current().Paths() }

// GetPath expands one role path with progress.
func (a *App) GetPath(name string) (course.Path, error) { return a.current().Path(name) }

// MarkStage marks a path stage finished or not, and returns what to tell the
// learner.
func (a *App) MarkStage(top, key string, done bool) (practice.Result, error) {
	return a.current().MarkStage(a.ctxOrBackground(), top, key, done)
}

// GetAgentCommand returns the command an MCP client runs to connect an agent
// to this app's course packs: this executable, in its MCP mode.
func (a *App) GetAgentCommand() []string {
	exe, err := os.Executable()
	if err != nil {
		exe = "openlearn"
	}
	return []string{exe, "mcp"}
}

// SetFocus records which document the learner has open, so an agent connected
// over MCP can be asked about "this page" and know which one that is. An empty
// path means no document is open.
func (a *App) SetFocus(doc string) {
	w := a.current()
	if err := a.store.SaveFocus(store.Focus{Pack: w.Ref, PackTitle: w.Course.Info().Title, Doc: doc}); err != nil {
		a.log.Warn("save focus", "err", err)
	}
}

// ---------- practice ----------

var errNoPractice = errors.New("this pack has no practice CLI that can run here")

func (a *App) runner() (*practice.Runner, error) {
	if r := a.current().Runner; r != nil {
		return r, nil
	}
	return nil, errNoPractice
}

// ListExercises returns the open pack's exercises.
func (a *App) ListExercises() ([]practice.Group, error) {
	r, err := a.runner()
	if err != nil {
		return []practice.Group{}, nil //nolint:nilerr // no practice CLI is an empty list, not a failure
	}
	return r.List(a.ctxOrBackground())
}

// StartExercise clones an exercise into the scratchpad and returns the attempt.
func (a *App) StartExercise(kind, lang, name string) (practice.Workspace, error) {
	r, err := a.runner()
	if err != nil {
		return practice.Workspace{}, err
	}
	return r.Start(a.ctxOrBackground(), kind, lang, name)
}

// SaveExerciseFile writes one file of an attempt.
func (a *App) SaveExerciseFile(kind, lang, name, rel, content string) error {
	r, err := a.runner()
	if err != nil {
		return err
	}
	return r.Save(kind, lang, name, rel, content)
}

// CheckExercise runs an attempt and returns the verdict.
func (a *App) CheckExercise(kind, lang, name string) (practice.Result, error) {
	r, err := a.runner()
	if err != nil {
		return practice.Result{}, err
	}
	return r.Check(a.ctxOrBackground(), kind, lang, name)
}

// DiffExercise compares an attempt with the reference.
func (a *App) DiffExercise(kind, lang, name string) (practice.Result, error) {
	r, err := a.runner()
	if err != nil {
		return practice.Result{}, err
	}
	return r.Diff(a.ctxOrBackground(), kind, lang, name)
}

// ResetExercise throws an attempt away.
func (a *App) ResetExercise(kind, lang, name string) (practice.Result, error) {
	r, err := a.runner()
	if err != nil {
		return practice.Result{}, err
	}
	return r.Reset(a.ctxOrBackground(), kind, lang, name)
}

// ---------- assets ----------

// assetPrefix is where the frontend requests a pack's images from.
const assetPrefix = "/pack/"

// packAssets serves images out of the open pack. Wails routes here any request
// its own bundled assets do not answer. It is its own type because every
// exported method on App becomes a frontend binding.
type packAssets struct{ app *App }

func (h packAssets) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	a := h.app
	rel, ok := strings.CutPrefix(r.URL.Path, assetPrefix)
	if !ok {
		http.NotFound(w, r)
		return
	}
	rel = path.Clean(rel)
	types := map[string]string{
		".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg",
		".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp",
	}
	ctype, ok := types[strings.ToLower(path.Ext(rel))]
	if !ok || !fs.ValidPath(rel) {
		http.NotFound(w, r)
		return
	}
	data, err := fs.ReadFile(a.current().Course.FS(), rel)
	if err != nil {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", ctype)
	// An SVG is a document; this keeps one from a pack from running script.
	w.Header().Set("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'")
	_, _ = w.Write(data) //nolint:gosec // G705: image bytes under a fixed image content type and a no-script policy
}
