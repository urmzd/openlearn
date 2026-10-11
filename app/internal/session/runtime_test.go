package session

import (
	"testing"
	"time"

	"github.com/urmzd/whiteboardy/internal/agui"
	"github.com/urmzd/whiteboardy/internal/domain"
	"github.com/urmzd/whiteboardy/internal/harness"
)

// recorder collects emitted events so tests can assert on the stream the UI
// would actually receive.
type recorder struct {
	events []agui.Event
}

func (r *recorder) Emit(e agui.Event) { r.events = append(r.events, e) }

func (r *recorder) types() []agui.Type {
	out := make([]agui.Type, len(r.events))
	for i, e := range r.events {
		out[i] = e.Type
	}
	return out
}

func (r *recorder) count(t agui.Type) int {
	n := 0
	for _, e := range r.events {
		if e.Type == t {
			n++
		}
	}
	return n
}

func TestSayEmitsAWellFormedMessage(t *testing.T) {
	rec := &recorder{}
	r := New(nil, rec, nil)
	r.bus = agui.NewBus(rec, "run-1")
	r.sess = &domain.Session{ID: "s1", Spec: domain.SessionSpec{DurationSec: 600}}
	r.phase = domain.PhaseActive
	r.resumedAt = time.Now()

	r.say(domain.Event{
		Kind:     domain.KindCurveball,
		Severity: domain.SeverityCritical,
		Title:    "Traffic went 10x",
		Body:     "What breaks first?",
		Areas:    []domain.Area{domain.AreaScaling},
	})

	// Every message must open, deliver, and close, in that order: the UI
	// appends on start and only stops the caret on end.
	want := []agui.Type{
		agui.TypeTextMessageStart,
		agui.TypeTextMessageContent,
		agui.TypeTextMessageEnd,
	}
	got := rec.types()
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("event %d = %s, want %s", i, got[i], want[i])
		}
	}

	start := rec.events[0].Data.(agui.TextMessageStart)
	end := rec.events[2].Data.(agui.TextMessageEnd)
	if start.MessageID != end.MessageID {
		t.Errorf("start id %q != end id %q; the UI could not correlate them", start.MessageID, end.MessageID)
	}
	if start.Title != "Traffic went 10x" {
		t.Errorf("title = %q", start.Title)
	}
	// A curveball is harness-authored, not the coach talking.
	if start.Role != agui.RoleSystem {
		t.Errorf("role = %q, want system", start.Role)
	}
	if len(r.sess.Events) != 1 {
		t.Errorf("event not recorded in the transcript: %d", len(r.sess.Events))
	}
}

func TestCurveballsFireOnceAtTheirMark(t *testing.T) {
	r := New(nil, &recorder{}, nil)
	r.sess = &domain.Session{
		Spec: domain.SessionSpec{DurationSec: 600},
		Problem: domain.Problem{Curveballs: []domain.Curveball{
			{AtPct: 50, Title: "half", Body: "b"},
			{AtPct: 80, Title: "late", Body: "b"},
		}},
	}

	if due := r.dueCurveballsLocked(60, 540); len(due) != 0 {
		t.Fatalf("fired %d curveballs at 10%%", len(due))
	}
	due := r.dueCurveballsLocked(300, 300)
	if len(due) != 1 || due[0].Title != "half" {
		t.Fatalf("at 50%% got %v", due)
	}
	// Re-checking the same moment must not re-deliver it.
	if again := r.dueCurveballsLocked(300, 300); len(again) != 0 {
		t.Fatalf("curveball fired twice: %v", again)
	}
	if late := r.dueCurveballsLocked(500, 100); len(late) != 1 || late[0].Title != "late" {
		t.Fatalf("at 83%% got %v", late)
	}
}

func TestPacingFiresOnceAndOnlyWhenBehind(t *testing.T) {
	p := domain.Problem{Rubric: []domain.Criterion{
		{Area: domain.AreaScaling}, {Area: domain.AreaCaching},
		{Area: domain.AreaSecurity}, {Area: domain.AreaReliability},
	}}

	fired := map[string]bool{}
	// Halfway with 3 of 4 areas covered is on pace: staying quiet is correct.
	onPace := harness.PacingEvent(p, 300, 300,
		[]domain.Area{domain.AreaScaling, domain.AreaCaching, domain.AreaSecurity}, fired)
	if onPace != nil {
		t.Errorf("nagged a candidate who is on pace: %s", onPace.Title)
	}

	fired = map[string]bool{}
	behind := harness.PacingEvent(p, 300, 300, []domain.Area{domain.AreaScaling}, fired)
	if behind == nil {
		t.Fatal("no pacing event at half time with 25% coverage")
	}
	if behind.Kind != domain.KindPacing {
		t.Errorf("kind = %q", behind.Kind)
	}
	if repeat := harness.PacingEvent(p, 310, 290, []domain.Area{domain.AreaScaling}, fired); repeat != nil {
		t.Errorf("pacing event repeated: %s", repeat.Title)
	}
}

func TestClockExcludesPausedTime(t *testing.T) {
	r := New(nil, &recorder{}, nil)
	r.sess = &domain.Session{Spec: domain.SessionSpec{DurationSec: 600}}
	r.phase = domain.PhaseActive
	r.resumedAt = time.Now().Add(-30 * time.Second)

	if err := r.Pause(); err != nil {
		t.Fatalf("pause: %v", err)
	}
	elapsed, remaining := r.clockLocked()
	if elapsed < 29 || elapsed > 32 {
		t.Fatalf("elapsed = %d, want ~30", elapsed)
	}

	// Time spent paused must not burn the timebox.
	time.Sleep(1100 * time.Millisecond)
	stillElapsed, _ := r.clockLocked()
	if stillElapsed != elapsed {
		t.Errorf("clock advanced while paused: %d -> %d", elapsed, stillElapsed)
	}
	if remaining != 600-elapsed {
		t.Errorf("remaining = %d, want %d", remaining, 600-elapsed)
	}
}

func TestNormalizeSpecClampsHostileInput(t *testing.T) {
	got := normalizeSpec(domain.SessionSpec{DurationSec: 5, CoachIntervalSec: 1})
	if got.DurationSec != 5*60 {
		t.Errorf("durationSec = %d, want a 5 minute floor", got.DurationSec)
	}
	if got.CoachIntervalSec != 45 {
		t.Errorf("coachIntervalSec = %d, want the default", got.CoachIntervalSec)
	}
	if got.Mode != domain.ModeSystemDesign {
		t.Errorf("mode = %q, want the system design default", got.Mode)
	}

	coding := normalizeSpec(domain.SessionSpec{Mode: domain.ModeCoding, DurationSec: 99 * 3600})
	if coding.Language == "" {
		t.Error("coding session left with no language")
	}
	if coding.DurationSec != 4*3600 {
		t.Errorf("durationSec = %d, want a 4 hour ceiling", coding.DurationSec)
	}
}

func TestMergeAreasDeduplicates(t *testing.T) {
	got := mergeAreas(
		[]domain.Area{domain.AreaScaling, domain.AreaCaching},
		[]domain.Area{domain.AreaCaching, domain.AreaSecurity},
	)
	want := []domain.Area{domain.AreaScaling, domain.AreaCaching, domain.AreaSecurity}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
}

func TestRecentSpokenDropsHousekeeping(t *testing.T) {
	events := []domain.Event{
		{Kind: domain.KindSystem, Title: "started"},
		{Kind: domain.KindHint, Title: "h1"},
		{Kind: domain.KindPacing, Title: "pace"},
		{Kind: domain.KindProbe, Title: "p1"},
	}
	got := recentSpoken(events, 6)
	if len(got) != 2 {
		t.Fatalf("got %d events, want 2 (hint and probe only)", len(got))
	}
	if got[0].Title != "h1" || got[1].Title != "p1" {
		t.Errorf("got %v", got)
	}

	// The cap keeps the most recent, not the oldest.
	many := make([]domain.Event, 0, 10)
	for i := 0; i < 10; i++ {
		many = append(many, domain.Event{Kind: domain.KindHint, Title: string(rune('a' + i))})
	}
	capped := recentSpoken(many, 3)
	if len(capped) != 3 || capped[0].Title != "h" || capped[2].Title != "j" {
		t.Errorf("cap kept the wrong window: %v", capped)
	}
}

func TestAbandonWithoutASessionIsSafe(t *testing.T) {
	rec := &recorder{}
	r := New(nil, rec, nil)
	r.Abandon() // must not panic or claim a run finished
	if rec.count(agui.TypeRunFinished) != 0 {
		t.Error("emitted RUN_FINISHED with no run in progress")
	}
	if r.Status().Phase != domain.PhaseIdle {
		t.Errorf("phase = %q", r.Status().Phase)
	}
}
