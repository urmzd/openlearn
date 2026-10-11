package main

import (
	"errors"
	"fmt"
	"io/fs"
	"net/http"
	"path"
	"path/filepath"
	"strings"

	"github.com/google/uuid"
	"github.com/urmzd/openlearn/app/internal/course"
	"github.com/urmzd/openlearn/app/internal/domain"
	"github.com/urmzd/openlearn/app/internal/harness"
	"github.com/urmzd/openlearn/app/internal/llm"
	"github.com/urmzd/openlearn/app/internal/packs"
	"github.com/urmzd/openlearn/app/internal/practice"
	"github.com/urmzd/openlearn/app/internal/store"
	wruntime "github.com/wailsapp/wails/v2/pkg/runtime"
)

// defaultPack is the pack a first launch opens.
const defaultPack = "supersource"

// openPack is the loaded course pack and everything derived from it.
type openPack struct {
	ref      string
	course   *course.Course
	progress course.DoneFiles
	// runner is nil when the pack has no practice CLI or is not on disk.
	runner *practice.Runner
}

// openPack loads the pack ref names, falling back to the default when ref is
// empty or no longer resolves (a linked directory that was moved, say).
func (a *App) openPack(ref string) {
	a.mu.RLock()
	linked := a.settings.LinkedPacks
	a.mu.RUnlock()

	c, err := a.packs.Open(ref)
	if ref == "" || err != nil {
		if ref != "" {
			a.log.Warn("saved pack is gone, opening the default", "ref", ref, "err", err)
		}
		ref = ""
		if def, ok := a.packs.Default(linked); ok {
			ref = def.Ref
			c, err = a.packs.Open(ref)
		}
	}
	if ref == "" || err != nil {
		a.log.Warn("no course pack available", "err", err)
		c, ref = course.Empty(), ""
	}

	p := &openPack{ref: ref, course: c}
	info := c.Info()
	if info.Practice {
		runner, err := practice.New(c.Dir(), c.Manifest().Practice)
		if err != nil {
			a.log.Warn("practice CLI unavailable", "err", err)
		}
		p.runner = runner
	}
	// A pack with a working practice CLI keeps progress where the CLI does, so
	// the two agree. Any other pack has nowhere of its own to write.
	if p.runner != nil {
		p.progress = course.CheckoutProgress(c.Dir())
	} else {
		p.progress = course.StoreProgress(filepath.Join(a.store.Root(), "learn", info.ID))
	}

	a.mu.Lock()
	a.pack = p
	a.mu.Unlock()
}

func (a *App) current() *openPack {
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
func (a *App) GetCourse() course.Info {
	p := a.current()
	info := p.course.Info()
	info.Practice = p.runner != nil
	if len(info.Modes) == 0 {
		info.Modes = []string{string(domain.ModeSystemDesign), string(domain.ModeCoding), string(domain.ModeWritten)}
	}
	return info
}

// GetOutline returns the open pack's browsable tree.
func (a *App) GetOutline() course.Outline { return a.current().course.Outline() }

// GetDoc reads one document of the open pack.
func (a *App) GetDoc(p string) (course.Doc, error) { return a.current().course.Doc(p) }

// Search finds documents in the open pack.
func (a *App) Search(query string) []course.Hit { return a.current().course.Search(query, 40) }

// ListPaths summarizes the open pack's role paths with progress.
func (a *App) ListPaths() []course.PathSummary {
	p := a.current()
	return p.course.Paths(p.progress)
}

// GetPath expands one role path with progress.
func (a *App) GetPath(name string) (course.Path, error) {
	p := a.current()
	return p.course.Path(name, p.progress)
}

// MarkStage marks a path stage finished or not, and returns what to tell the
// learner. A stage tied to a check is only marked by the practice CLI, which
// is what knows whether the check passed.
func (a *App) MarkStage(top, key string, done bool) (practice.Result, error) {
	p := a.current()
	stage, err := p.course.Stage(top, key)
	if err != nil {
		return practice.Result{}, err
	}
	if p.runner != nil {
		return p.runner.Learn(a.ctxOrBackground(), top, key, done)
	}
	if done && stage.Check != "" {
		return practice.Result{
			Verdict: "fail",
			Output:  fmt.Sprintf("This stage is finished by passing %s, which needs the pack's practice CLI. Open the pack from a checkout to run it.", stage.Check),
		}, nil
	}
	if err := p.progress.Mark(stage.Owner, stage.Number, done); err != nil {
		return practice.Result{}, err
	}
	return practice.Result{Verdict: "pass"}, nil
}

// ---------- practice ----------

var errNoPractice = errors.New("this pack has no practice CLI that can run here")

func (a *App) runner() (*practice.Runner, error) {
	if r := a.current().runner; r != nil {
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

// ---------- professor ----------

// ProfessorChannel carries the professor's streamed answer to the frontend.
const ProfessorChannel = "professor"

// ProfessorDelta is one streamed piece of an answer.
type ProfessorDelta struct {
	ID    string `json:"id"`
	Delta string `json:"delta"`
}

// AskProfessor answers a question about the page the learner has open. The
// answer streams on ProfessorChannel under the returned id, and the full text
// is the return value, so a caller that missed the stream still has it.
func (a *App) AskProfessor(question, docPath string, history []harness.Turn) (string, error) {
	if strings.TrimSpace(question) == "" {
		return "", errors.New("ask a question first")
	}
	a.mu.RLock()
	cfg := a.settings.LLM
	a.mu.RUnlock()
	client, err := llm.New(a.ctxOrBackground(), cfg, a.log)
	if err != nil {
		return "", fmt.Errorf("no model configured: %w", err)
	}

	p := a.current()
	in := harness.ProfessorInput{
		Course:   p.course.Info().Title,
		Progress: a.progressLine(p),
		History:  history,
		Question: question,
	}
	if docPath != "" {
		if doc, err := p.course.Doc(docPath); err == nil {
			in.DocTitle, in.Material = doc.Title, doc.Markdown
		}
	}

	id := uuid.NewString()
	wruntime.EventsEmit(a.ctx, ProfessorChannel, ProfessorDelta{ID: id})
	return harness.Ask(a.ctxOrBackground(), client, in, func(delta string) {
		wruntime.EventsEmit(a.ctx, ProfessorChannel, ProfessorDelta{ID: id, Delta: delta})
	})
}

// progressLine says where the learner is on the pack's featured path, in a
// sentence the professor can read.
func (a *App) progressLine(p *openPack) string {
	for _, s := range p.course.Paths(p.progress) {
		if s.Featured && s.Broken == "" {
			return fmt.Sprintf("%d of %d stages finished on the %q path.", s.Done, s.Total, s.Title)
		}
	}
	return ""
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
	data, err := fs.ReadFile(a.current().course.FS(), rel)
	if err != nil {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", ctype)
	// An SVG is a document; this keeps one from a pack from running script.
	w.Header().Set("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'")
	_, _ = w.Write(data) //nolint:gosec // G705: image bytes under a fixed image content type and a no-script policy
}
