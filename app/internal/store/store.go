// Package store persists app settings and finished sessions under the user's
// home directory, and aggregates sessions into a skill profile.
package store

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/urmzd/whiteboardy/internal/domain"
	"github.com/urmzd/whiteboardy/internal/llm"
)

// Settings is everything the app remembers between launches.
type Settings struct {
	LLM      llm.Config         `json:"llm"`
	Defaults domain.SessionSpec `json:"defaults"`
}

// DefaultSettings is the zero-config starting point: local ollama, a 45 minute
// system design session with the coach on.
func DefaultSettings() Settings {
	return Settings{
		LLM: llm.Config{
			Kind: llm.KindOllama,
			Host: llm.DefaultOllamaHost,
		},
		Defaults: domain.SessionSpec{
			Mode:             domain.ModeSystemDesign,
			Level:            domain.LevelSenior,
			DurationSec:      45 * 60,
			Language:         "go",
			CoachEnabled:     true,
			CoachIntervalSec: 45,
		},
	}
}

// Store reads and writes app state on disk.
type Store struct {
	root string
	mu   sync.Mutex
}

// New opens (and creates if needed) the store rooted at dir. An empty dir uses
// ~/.whiteboardy.
func New(dir string) (*Store, error) {
	if dir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil, fmt.Errorf("store: home dir: %w", err)
		}
		dir = filepath.Join(home, ".whiteboardy")
	}
	// 0700: sessions record what the user could not do under time pressure,
	// which is nobody else's business on a shared machine.
	if err := os.MkdirAll(filepath.Join(dir, "sessions"), 0o700); err != nil {
		return nil, fmt.Errorf("store: create %s: %w", dir, err)
	}
	return &Store{root: dir}, nil
}

// Root is the directory the store writes to.
func (s *Store) Root() string { return s.root }

func (s *Store) settingsPath() string { return filepath.Join(s.root, "settings.json") }

// LoadSettings returns saved settings, falling back to defaults when nothing
// is saved yet or the file is unreadable.
func (s *Store) LoadSettings() Settings {
	s.mu.Lock()
	defer s.mu.Unlock()

	data, err := os.ReadFile(s.settingsPath())
	if err != nil {
		return DefaultSettings()
	}
	out := DefaultSettings()
	if err := json.Unmarshal(data, &out); err != nil {
		return DefaultSettings()
	}
	if out.LLM.Kind == "" {
		out.LLM.Kind = llm.KindOllama
	}
	if out.Defaults.DurationSec <= 0 {
		out.Defaults.DurationSec = 45 * 60
	}
	if out.Defaults.CoachIntervalSec <= 0 {
		out.Defaults.CoachIntervalSec = 45
	}
	return out
}

// SaveSettings writes settings atomically.
func (s *Store) SaveSettings(v Settings) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return writeJSON(s.settingsPath(), v)
}

// SaveSession writes a finished session. Sessions are named by start time so
// the directory listing is chronological.
func (s *Store) SaveSession(sess domain.Session) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	name := fmt.Sprintf("%s-%s.json", sess.StartedAt.UTC().Format("20060102-150405"), sess.ID)
	return writeJSON(filepath.Join(s.root, "sessions", name), sess)
}

// ListSessions returns saved sessions, newest first.
func (s *Store) ListSessions() ([]domain.Session, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	dir := filepath.Join(s.root, "sessions")
	entries, err := os.ReadDir(dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, fmt.Errorf("store: list sessions: %w", err)
	}

	out := make([]domain.Session, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		// The name comes from reading our own sessions directory, not from user
		// input, and non-JSON entries are already filtered out above.
		data, err := os.ReadFile(filepath.Join(dir, e.Name())) //nolint:gosec // G304: path is derived from our own ReadDir
		if err != nil {
			continue // a single unreadable session must not hide the rest
		}
		var sess domain.Session
		if err := json.Unmarshal(data, &sess); err != nil {
			continue
		}
		out = append(out, sess)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].StartedAt.After(out[j].StartedAt) })
	return out, nil
}

// DeleteSession removes a saved session by ID.
func (s *Store) DeleteSession(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	dir := filepath.Join(s.root, "sessions")
	entries, err := os.ReadDir(dir)
	if err != nil {
		return fmt.Errorf("store: delete session: %w", err)
	}
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), "-"+id+".json") {
			return os.Remove(filepath.Join(dir, e.Name()))
		}
	}
	return fmt.Errorf("store: session %s not found", id)
}

// Profile aggregates reviewed sessions for one mode into a skill profile. Only
// sessions that were actually reviewed contribute; abandoned runs are ignored
// so the profile is not diluted by sessions the user walked away from.
func (s *Store) Profile(mode domain.Mode) (domain.Profile, error) {
	sessions, err := s.ListSessions()
	if err != nil {
		return domain.Profile{}, err
	}
	return BuildProfile(mode, sessions), nil
}

// BuildProfile is the pure aggregation over sessions, separated from disk so
// it can be tested directly.
func BuildProfile(mode domain.Mode, sessions []domain.Session) domain.Profile {
	profile := domain.Profile{Mode: mode}

	// Oldest first, so "trend" reads left to right in time.
	relevant := make([]domain.Session, 0, len(sessions))
	for _, s := range sessions {
		if s.Spec.Mode == mode && s.Review != nil {
			relevant = append(relevant, s)
		}
	}
	sort.Slice(relevant, func(i, j int) bool { return relevant[i].StartedAt.Before(relevant[j].StartedAt) })

	type acc struct {
		scores []int
	}
	byArea := map[domain.Area]*acc{}
	totalOverall := 0

	for _, s := range relevant {
		profile.Sessions++
		profile.TotalMinutes += s.ElapsedSec / 60
		totalOverall += s.Review.Overall
		for _, sc := range s.Review.Scores {
			a, ok := byArea[sc.Area]
			if !ok {
				a = &acc{}
				byArea[sc.Area] = a
			}
			a.scores = append(a.scores, sc.Score)
		}
	}
	if profile.Sessions > 0 {
		profile.AverageScore = float64(totalOverall) / float64(profile.Sessions)
	}

	for _, info := range domain.AreasFor(mode) {
		a, ok := byArea[info.ID]
		if !ok || len(a.scores) == 0 {
			continue
		}
		stat := domain.AreaStat{
			Area:    info.ID,
			Label:   info.Label,
			Samples: len(a.scores),
			Last:    a.scores[len(a.scores)-1],
			Average: mean(a.scores),
			Trend:   trend(a.scores),
		}
		profile.Areas = append(profile.Areas, stat)
	}

	sort.Slice(profile.Areas, func(i, j int) bool {
		if profile.Areas[i].Average != profile.Areas[j].Average {
			return profile.Areas[i].Average > profile.Areas[j].Average
		}
		return profile.Areas[i].Label < profile.Areas[j].Label
	})

	// Only rank areas with enough evidence to say anything.
	ranked := make([]domain.AreaStat, 0, len(profile.Areas))
	for _, a := range profile.Areas {
		if a.Samples >= 2 {
			ranked = append(ranked, a)
		}
	}
	for i := 0; i < len(ranked) && i < 3; i++ {
		profile.Strongest = append(profile.Strongest, ranked[i].Label)
	}
	for i := len(ranked) - 1; i >= 0 && len(profile.Weakest) < 3; i-- {
		profile.Weakest = append(profile.Weakest, ranked[i].Label)
	}
	return profile
}

func mean(xs []int) float64 {
	if len(xs) == 0 {
		return 0
	}
	sum := 0
	for _, x := range xs {
		sum += x
	}
	return float64(sum) / float64(len(xs))
}

// trend compares the newest third of samples to the oldest third. Fewer than
// three samples is not enough signal to claim a direction.
func trend(xs []int) float64 {
	if len(xs) < 3 {
		return 0
	}
	n := len(xs) / 3
	if n < 1 {
		n = 1
	}
	return mean(xs[len(xs)-n:]) - mean(xs[:n])
}

func writeJSON(path string, v any) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return fmt.Errorf("store: encode %s: %w", filepath.Base(path), err)
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return fmt.Errorf("store: write %s: %w", filepath.Base(path), err)
	}
	if err := os.Rename(tmp, path); err != nil {
		return fmt.Errorf("store: commit %s: %w", filepath.Base(path), err)
	}
	return nil
}

// Touch records that the store is reachable; used by the app on startup to
// surface a permissions problem early rather than at session end.
func (s *Store) Touch() error {
	probe := filepath.Join(s.root, ".probe")
	if err := os.WriteFile(probe, []byte(time.Now().UTC().Format(time.RFC3339)), 0o600); err != nil {
		return fmt.Errorf("store: not writable: %w", err)
	}
	return os.Remove(probe)
}
