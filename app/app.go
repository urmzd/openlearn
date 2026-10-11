package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"sync"

	"github.com/urmzd/whiteboardy/internal/agui"
	"github.com/urmzd/whiteboardy/internal/domain"
	"github.com/urmzd/whiteboardy/internal/llm"
	"github.com/urmzd/whiteboardy/internal/session"
	"github.com/urmzd/whiteboardy/internal/store"
	wruntime "github.com/wailsapp/wails/v2/pkg/runtime"
)

// App is the Wails binding surface. Every exported method here becomes a
// callable from TypeScript, so the signatures are kept flat and JSON-friendly.
type App struct {
	ctx     context.Context
	log     *slog.Logger
	store   *store.Store
	runtime *session.Runtime

	mu       sync.RWMutex
	settings store.Settings
	// clientErr records why the last client build failed, so the UI can show
	// the reason instead of only failing when a session starts.
	clientErr string
}

// NewApp constructs the app. Store failures are fatal: without a writable home
// directory there is nowhere to keep the sessions the whole product is about.
func NewApp() *App {
	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

	st, err := store.New("")
	if err != nil {
		log.Error("cannot open store", "err", err)
		os.Exit(1)
	}
	if err := st.Touch(); err != nil {
		log.Error("store not writable", "err", err)
		os.Exit(1)
	}

	a := &App{log: log, store: st, settings: st.LoadSettings()}
	a.runtime = session.New(log, agui.EmitterFunc(a.emit), a.persist)
	return a
}

func (a *App) startup(ctx context.Context) {
	a.ctx = ctx
	if err := a.rebuildClient(); err != nil {
		a.log.Warn("no model configured yet", "err", err)
	}
}

func (a *App) shutdown(context.Context) {
	a.runtime.Abandon()
}

// emit forwards one AG-UI event to the frontend. Every event goes out on a
// single Wails channel so the frontend receives them in emission order, which
// message start/content/end depends on.
func (a *App) emit(e agui.Event) {
	if a.ctx == nil {
		return
	}
	wruntime.EventsEmit(a.ctx, agui.Channel, e)
}

func (a *App) persist(s domain.Session) {
	if err := a.store.SaveSession(s); err != nil {
		a.log.Error("save session", "err", err)
	}
}

// rebuildClient rebuilds the LLM client from current settings.
func (a *App) rebuildClient() error {
	a.mu.RLock()
	cfg := a.settings.LLM
	a.mu.RUnlock()

	client, err := llm.New(context.Background(), cfg, a.log)
	if err != nil {
		a.mu.Lock()
		a.clientErr = err.Error()
		a.mu.Unlock()
		a.runtime.SetClient(nil)
		return err
	}
	a.mu.Lock()
	a.clientErr = ""
	a.mu.Unlock()
	a.runtime.SetClient(client)
	return nil
}

// maskedKey is the placeholder the UI sees in place of a stored API key.
// Saving it back unchanged means "keep the existing key".
const maskedKey = "__saved__"

// GetSettings returns the saved settings with the API key masked.
func (a *App) GetSettings() store.Settings {
	a.mu.RLock()
	defer a.mu.RUnlock()
	s := a.settings
	if s.LLM.APIKey != "" {
		s.LLM.APIKey = maskedKey
	}
	return s
}

// SaveSettings persists settings and rebuilds the LLM client.
func (a *App) SaveSettings(s store.Settings) error {
	a.mu.Lock()
	if s.LLM.APIKey == maskedKey {
		s.LLM.APIKey = a.settings.LLM.APIKey
	}
	if s.LLM.Kind == llm.KindOllama && strings.TrimSpace(s.LLM.Host) == "" {
		s.LLM.Host = llm.DefaultOllamaHost
	}
	a.settings = s
	a.mu.Unlock()

	if err := a.store.SaveSettings(s); err != nil {
		return err
	}
	return a.rebuildClient()
}

// ListModels returns the models available from the configured backend. Only
// ollama supports discovery without credentials; hosted providers return an
// empty list and the user types the model name.
func (a *App) ListModels() ([]llm.ModelInfo, error) {
	a.mu.RLock()
	cfg := a.settings.LLM
	a.mu.RUnlock()

	if cfg.Kind != llm.KindOllama && cfg.Kind != "" {
		return []llm.ModelInfo{}, nil
	}
	return llm.ListOllamaModels(a.ctxOrBackground(), cfg.Host)
}

// CheckModel does a real round trip so setup problems surface before a session
// rather than 30 seconds into problem generation.
func (a *App) CheckModel() (string, error) {
	a.mu.RLock()
	cfg := a.settings.LLM
	clientErr := a.clientErr
	a.mu.RUnlock()

	if clientErr != "" {
		return "", fmt.Errorf("%s", clientErr)
	}
	client, err := llm.New(a.ctxOrBackground(), cfg, a.log)
	if err != nil {
		return "", err
	}
	out, err := client.Text(a.ctxOrBackground(),
		"You are a connectivity probe. Reply with exactly the word: ready",
		"Reply with the single word ready.")
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%s / %s responded: %s", cfg.Kind, cfg.Model, strings.TrimSpace(firstLine(out))), nil
}

// StartSession generates a problem and starts the timebox.
func (a *App) StartSession(spec domain.SessionSpec) (domain.Problem, error) {
	return a.runtime.Start(a.ctxOrBackground(), spec)
}

// UpdateSnapshot pushes the user's current work to the runtime.
func (a *App) UpdateSnapshot(s domain.Snapshot) {
	a.runtime.UpdateSnapshot(s)
}

// PauseSession stops the clock and the coach.
func (a *App) PauseSession() error { return a.runtime.Pause() }

// ResumeSession restarts the clock.
func (a *App) ResumeSession() error { return a.runtime.Resume() }

// FinishSession ends the timebox and returns the review.
func (a *App) FinishSession(final domain.Snapshot) (*domain.Review, error) {
	return a.runtime.Finish(a.ctxOrBackground(), final)
}

// AbandonSession discards the live session without scoring it.
func (a *App) AbandonSession() { a.runtime.Abandon() }

// GetStatus returns the current session state.
func (a *App) GetStatus() domain.Status {
	st := a.runtime.Status()
	if st.Error == "" {
		a.mu.RLock()
		st.Error = a.clientErr
		a.mu.RUnlock()
	}
	return st
}

// ListSessions returns saved sessions, newest first.
func (a *App) ListSessions() ([]domain.Session, error) { return a.store.ListSessions() }

// DeleteSession removes a saved session.
func (a *App) DeleteSession(id string) error { return a.store.DeleteSession(id) }

// GetProfile aggregates saved sessions for a mode into a skill profile.
func (a *App) GetProfile(mode domain.Mode) (domain.Profile, error) { return a.store.Profile(mode) }

// GetAreas returns the skill taxonomy for a mode so the UI can label things
// without duplicating the list in TypeScript.
func (a *App) GetAreas(mode domain.Mode) []domain.AreaInfo { return domain.AreasFor(mode) }

// GetStoreRoot reports where sessions are written, for the settings screen.
func (a *App) GetStoreRoot() string { return a.store.Root() }

func (a *App) ctxOrBackground() context.Context {
	if a.ctx != nil {
		return a.ctx
	}
	return context.Background()
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}
