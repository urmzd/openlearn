package store_test

import (
	"testing"
	"time"

	"github.com/urmzd/whiteboardy/internal/domain"
	"github.com/urmzd/whiteboardy/internal/store"
)

func session(day int, mode domain.Mode, overall int, scores map[domain.Area]int) domain.Session {
	s := domain.Session{
		ID:         string(rune('a' + day)),
		Spec:       domain.SessionSpec{Mode: mode},
		StartedAt:  time.Date(2026, 1, day+1, 9, 0, 0, 0, time.UTC),
		ElapsedSec: 45 * 60,
		Review:     &domain.Review{Overall: overall},
	}
	for area, score := range scores {
		s.Review.Scores = append(s.Review.Scores, domain.CriterionScore{
			CriterionID: string(area),
			Area:        area,
			Score:       score,
		})
	}
	return s
}

func TestBuildProfileIgnoresUnreviewedAndOtherModes(t *testing.T) {
	sessions := []domain.Session{
		session(0, domain.ModeSystemDesign, 60, map[domain.Area]int{domain.AreaScaling: 3}),
		session(1, domain.ModeCoding, 90, map[domain.Area]int{domain.AreaCorrectness: 4}),
		// Abandoned: no review. It must not dilute the profile.
		{Spec: domain.SessionSpec{Mode: domain.ModeSystemDesign}, ElapsedSec: 3600},
	}

	p := store.BuildProfile(domain.ModeSystemDesign, sessions)
	if p.Sessions != 1 {
		t.Errorf("sessions = %d, want 1", p.Sessions)
	}
	if p.AverageScore != 60 {
		t.Errorf("averageScore = %v, want 60", p.AverageScore)
	}
	if p.TotalMinutes != 45 {
		t.Errorf("totalMinutes = %d, want 45", p.TotalMinutes)
	}
}

func TestBuildProfileAveragesPerAreaAcrossSessions(t *testing.T) {
	sessions := []domain.Session{
		session(0, domain.ModeSystemDesign, 50, map[domain.Area]int{domain.AreaScaling: 1, domain.AreaCaching: 4}),
		session(1, domain.ModeSystemDesign, 70, map[domain.Area]int{domain.AreaScaling: 3, domain.AreaCaching: 4}),
	}

	p := store.BuildProfile(domain.ModeSystemDesign, sessions)

	byArea := map[domain.Area]domain.AreaStat{}
	for _, a := range p.Areas {
		byArea[a.Area] = a
	}

	scaling := byArea[domain.AreaScaling]
	if scaling.Average != 2 {
		t.Errorf("scaling average = %v, want 2", scaling.Average)
	}
	if scaling.Samples != 2 {
		t.Errorf("scaling samples = %d, want 2", scaling.Samples)
	}
	// Last must be the most recent session's score, not the first seen.
	if scaling.Last != 3 {
		t.Errorf("scaling last = %d, want 3", scaling.Last)
	}
	if byArea[domain.AreaCaching].Average != 4 {
		t.Errorf("caching average = %v, want 4", byArea[domain.AreaCaching].Average)
	}

	// Areas are labelled from the taxonomy so the UI needs no lookup table.
	if scaling.Label == "" || scaling.Label == string(domain.AreaScaling) {
		t.Errorf("scaling label = %q, want a human label", scaling.Label)
	}
}

func TestBuildProfileOrdersByStrength(t *testing.T) {
	sessions := []domain.Session{
		session(0, domain.ModeSystemDesign, 50, map[domain.Area]int{
			domain.AreaScaling: 4, domain.AreaSecurity: 1,
		}),
		session(1, domain.ModeSystemDesign, 50, map[domain.Area]int{
			domain.AreaScaling: 4, domain.AreaSecurity: 1,
		}),
	}

	p := store.BuildProfile(domain.ModeSystemDesign, sessions)
	if len(p.Areas) < 2 {
		t.Fatalf("got %d areas", len(p.Areas))
	}
	if p.Areas[0].Area != domain.AreaScaling {
		t.Errorf("strongest area first = %q, want scaling", p.Areas[0].Area)
	}
	if len(p.Strongest) == 0 || len(p.Weakest) == 0 {
		t.Fatalf("strongest=%v weakest=%v", p.Strongest, p.Weakest)
	}
	if p.Strongest[0] == p.Weakest[0] {
		t.Errorf("same area ranked both strongest and weakest: %q", p.Strongest[0])
	}
}

func TestBuildProfileWithdrawsRankingWithoutEvidence(t *testing.T) {
	// One session means one sample per area. Calling anything a strength or a
	// weakness off a single data point would be noise dressed as insight.
	sessions := []domain.Session{
		session(0, domain.ModeSystemDesign, 50, map[domain.Area]int{
			domain.AreaScaling: 4, domain.AreaSecurity: 0,
		}),
	}

	p := store.BuildProfile(domain.ModeSystemDesign, sessions)
	if len(p.Strongest) != 0 || len(p.Weakest) != 0 {
		t.Errorf("ranked areas from one sample: strongest=%v weakest=%v", p.Strongest, p.Weakest)
	}
	// The per-area averages are still reported; only the ranking is withheld.
	if len(p.Areas) != 2 {
		t.Errorf("got %d areas, want 2", len(p.Areas))
	}
}

func TestBuildProfileTrendNeedsThreeSamples(t *testing.T) {
	two := []domain.Session{
		session(0, domain.ModeSystemDesign, 40, map[domain.Area]int{domain.AreaScaling: 0}),
		session(1, domain.ModeSystemDesign, 90, map[domain.Area]int{domain.AreaScaling: 4}),
	}
	p := store.BuildProfile(domain.ModeSystemDesign, two)
	if p.Areas[0].Trend != 0 {
		t.Errorf("trend = %v from two samples, want 0", p.Areas[0].Trend)
	}

	improving := []domain.Session{
		session(0, domain.ModeSystemDesign, 40, map[domain.Area]int{domain.AreaScaling: 0}),
		session(1, domain.ModeSystemDesign, 60, map[domain.Area]int{domain.AreaScaling: 2}),
		session(2, domain.ModeSystemDesign, 90, map[domain.Area]int{domain.AreaScaling: 4}),
	}
	p = store.BuildProfile(domain.ModeSystemDesign, improving)
	if p.Areas[0].Trend <= 0 {
		t.Errorf("trend = %v for an improving area, want positive", p.Areas[0].Trend)
	}
}

func TestBuildProfileOnNoSessions(t *testing.T) {
	p := store.BuildProfile(domain.ModeCoding, nil)
	if p.Sessions != 0 || len(p.Areas) != 0 || p.AverageScore != 0 {
		t.Errorf("empty profile is not empty: %+v", p)
	}
}

func TestSaveAndListSessionsRoundTrip(t *testing.T) {
	st, err := store.New(t.TempDir())
	if err != nil {
		t.Fatalf("new store: %v", err)
	}

	older := session(0, domain.ModeSystemDesign, 55, map[domain.Area]int{domain.AreaScaling: 2})
	newer := session(5, domain.ModeSystemDesign, 75, map[domain.Area]int{domain.AreaScaling: 3})
	for _, s := range []domain.Session{older, newer} {
		if err := st.SaveSession(s); err != nil {
			t.Fatalf("save: %v", err)
		}
	}

	got, err := st.ListSessions()
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d sessions, want 2", len(got))
	}
	// Newest first, so the history view needs no sorting of its own.
	if !got[0].StartedAt.After(got[1].StartedAt) {
		t.Errorf("sessions not newest-first: %v then %v", got[0].StartedAt, got[1].StartedAt)
	}
	if got[0].Review == nil || got[0].Review.Overall != 75 {
		t.Errorf("review did not survive the round trip: %+v", got[0].Review)
	}

	if err := st.DeleteSession(newer.ID); err != nil {
		t.Fatalf("delete: %v", err)
	}
	if got, _ = st.ListSessions(); len(got) != 1 {
		t.Errorf("after delete got %d sessions, want 1", len(got))
	}
}

func TestSettingsRoundTripAndDefaults(t *testing.T) {
	st, err := store.New(t.TempDir())
	if err != nil {
		t.Fatalf("new store: %v", err)
	}

	// Nothing saved yet: defaults, not zeroes.
	got := st.LoadSettings()
	if got.Defaults.DurationSec == 0 || got.LLM.Kind == "" {
		t.Fatalf("defaults are empty: %+v", got)
	}

	got.LLM.Model = "qwen3.5:9b"
	got.Defaults.Level = domain.LevelStaff
	if err := st.SaveSettings(got); err != nil {
		t.Fatalf("save settings: %v", err)
	}

	reloaded := st.LoadSettings()
	if reloaded.LLM.Model != "qwen3.5:9b" || reloaded.Defaults.Level != domain.LevelStaff {
		t.Errorf("settings did not round trip: %+v", reloaded)
	}
}
