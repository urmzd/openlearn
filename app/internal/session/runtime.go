// Package session runs a practice session: the clock, the curveball timeline,
// the coach loop, and the transition into review. Everything it tells the UI
// goes out as AG-UI events (see internal/agui), so the UI is a projection of
// the event stream rather than a poller.
package session

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/urmzd/whiteboardy/internal/agui"
	"github.com/urmzd/whiteboardy/internal/domain"
	"github.com/urmzd/whiteboardy/internal/harness"
	"github.com/urmzd/whiteboardy/internal/llm"
)

// Step names that bracket the slow phases of a run.
const (
	StepGenerate = "generate_problem"
	StepReview   = "review"
	StepThinking = "coach_thinking"
)

// Runtime owns at most one live session.
type Runtime struct {
	mu     sync.Mutex
	log    *slog.Logger
	out    agui.Emitter
	client *llm.Client

	bus *agui.Bus

	// Live session state.
	phase     domain.Phase
	sess      *domain.Session
	snapshot  domain.Snapshot
	covered   []domain.Area
	fired     map[string]bool
	lastPrint string
	errMsg    string

	// Clock. accumulated is time banked from before the current resume point.
	accumulated time.Duration
	resumedAt   time.Time

	cancel context.CancelFunc
	// wg tracks the clock and coach goroutines so shutdown can wait for them.
	wg sync.WaitGroup
	// onSave is called with the finished session so the app can persist it.
	onSave func(domain.Session)
}

// New builds a Runtime. out and onSave may be nil.
func New(log *slog.Logger, out agui.Emitter, onSave func(domain.Session)) *Runtime {
	if log == nil {
		log = slog.Default()
	}
	if out == nil {
		out = agui.Discard
	}
	if onSave == nil {
		onSave = func(domain.Session) {}
	}
	return &Runtime{
		log:    log,
		out:    out,
		phase:  domain.PhaseIdle,
		fired:  map[string]bool{},
		bus:    agui.NewBus(out, ""),
		onSave: onSave,
	}
}

// SetClient swaps the LLM client. Safe to call between sessions.
func (r *Runtime) SetClient(c *llm.Client) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.client = c
}

// ErrBusy is returned when a session is already running.
var ErrBusy = errors.New("session: a session is already in progress")

// ErrNoSession is returned by operations that need a live session.
var ErrNoSession = errors.New("session: no session in progress")

// ErrNoClient is returned when no model has been configured.
var ErrNoClient = errors.New("session: no model configured")

// Start generates a problem and begins the timebox. It blocks for the duration
// of problem generation, which on a local model is the slowest thing the app
// does; the STEP_STARTED event fires first so the UI can say what it is waiting
// on.
func (r *Runtime) Start(ctx context.Context, spec domain.SessionSpec) (domain.Problem, error) {
	r.mu.Lock()
	if r.phase == domain.PhaseActive || r.phase == domain.PhasePaused || r.phase == domain.PhaseGenerating {
		r.mu.Unlock()
		return domain.Problem{}, ErrBusy
	}
	client := r.client
	if client == nil {
		r.mu.Unlock()
		return domain.Problem{}, ErrNoClient
	}
	spec = normalizeSpec(spec)
	sessionID := uuid.NewString()
	bus := agui.NewBus(r.out, uuid.NewString())

	r.phase = domain.PhaseGenerating
	r.errMsg = ""
	r.bus = bus
	r.mu.Unlock()

	bus.RunStarted(sessionID)
	r.snapshotState()
	bus.StepStarted(StepGenerate)

	problem, err := harness.GenerateProblem(ctx, client, spec)
	if err != nil {
		bus.StepFinished(StepGenerate)
		r.mu.Lock()
		r.phase = domain.PhaseFailed
		r.errMsg = err.Error()
		r.mu.Unlock()
		bus.RunError("generation_failed", err.Error())
		r.snapshotState()
		return domain.Problem{}, err
	}
	bus.StepFinished(StepGenerate)

	runCtx, cancel := context.WithCancel(context.Background())

	r.mu.Lock()
	now := time.Now()
	r.sess = &domain.Session{
		ID:        sessionID,
		Spec:      spec,
		Problem:   problem,
		StartedAt: now,
		Phase:     domain.PhaseActive,
		Provider:  client.ProviderName(),
		Model:     client.Model(),
	}
	r.snapshot = domain.Snapshot{Language: spec.Language}
	if spec.Mode == domain.ModeCoding {
		r.snapshot.Code = problem.Starter
	}
	r.covered = nil
	r.fired = map[string]bool{}
	r.lastPrint = ""
	r.accumulated = 0
	r.resumedAt = now
	r.phase = domain.PhaseActive
	r.cancel = cancel
	r.mu.Unlock()

	r.snapshotState()

	r.wg.Add(1)
	go r.clockLoop(runCtx)
	if spec.CoachEnabled {
		r.wg.Add(1)
		go r.coachLoop(runCtx, client)
	}

	r.say(domain.Event{
		Kind:     domain.KindSystem,
		Severity: domain.SeverityInfo,
		Title:    "Session started",
		Body:     fmt.Sprintf("%d minutes on the clock. The board is yours.", spec.DurationSec/60),
	})
	return problem, nil
}

// UpdateSnapshot receives the user's current work. The frontend pushes this on
// a debounce; the coach reads whatever landed most recently.
func (r *Runtime) UpdateSnapshot(s domain.Snapshot) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.sess == nil {
		return
	}
	if s.Language == "" {
		s.Language = r.sess.Spec.Language
	}
	r.snapshot = s
}

// Pause stops the clock and the coach without ending the session.
func (r *Runtime) Pause() error {
	r.mu.Lock()
	if r.phase != domain.PhaseActive {
		r.mu.Unlock()
		return ErrNoSession
	}
	r.accumulated += time.Since(r.resumedAt)
	r.phase = domain.PhasePaused
	r.mu.Unlock()
	r.snapshotState()
	return nil
}

// Resume restarts the clock after a pause.
func (r *Runtime) Resume() error {
	r.mu.Lock()
	if r.phase != domain.PhasePaused {
		r.mu.Unlock()
		return ErrNoSession
	}
	r.resumedAt = time.Now()
	r.phase = domain.PhaseActive
	r.mu.Unlock()
	r.snapshotState()
	return nil
}

// Finish ends the timebox and scores the work. It blocks for the length of the
// review call.
func (r *Runtime) Finish(ctx context.Context, final domain.Snapshot) (*domain.Review, error) {
	r.mu.Lock()
	if r.sess == nil || (r.phase != domain.PhaseActive && r.phase != domain.PhasePaused) {
		r.mu.Unlock()
		return nil, ErrNoSession
	}
	if r.phase == domain.PhaseActive {
		r.accumulated += time.Since(r.resumedAt)
	}
	if !final.IsEmpty() {
		if final.Language == "" {
			final.Language = r.sess.Spec.Language
		}
		r.snapshot = final
	}
	sess := *r.sess
	snapshot := r.snapshot
	elapsed := int(r.accumulated.Seconds())
	client := r.client
	cancel := r.cancel
	bus := r.bus
	r.phase = domain.PhaseReviewing
	r.mu.Unlock()

	if cancel != nil {
		cancel()
	}
	r.wg.Wait()
	r.snapshotState()
	bus.StepStarted(StepReview)

	sess.EndedAt = time.Now()
	sess.ElapsedSec = elapsed
	sess.Final = snapshot

	review, err := harness.Review(ctx, client, sess.Problem, snapshot, sess.Events, elapsed)
	bus.StepFinished(StepReview)
	if err != nil {
		r.mu.Lock()
		r.phase = domain.PhaseFailed
		r.errMsg = err.Error()
		// The work itself is still worth keeping even when scoring failed.
		sess.Phase = domain.PhaseFailed
		r.sess = &sess
		r.mu.Unlock()
		r.onSave(sess)
		bus.RunError("review_failed", err.Error())
		r.snapshotState()
		return nil, err
	}

	sess.Review = &review
	sess.Phase = domain.PhaseDone

	r.mu.Lock()
	r.sess = &sess
	r.phase = domain.PhaseDone
	r.mu.Unlock()

	r.onSave(sess)
	r.snapshotState()
	bus.RunFinished(review)
	return &review, nil
}

// Abandon throws away the live session without scoring it.
func (r *Runtime) Abandon() {
	r.mu.Lock()
	cancel := r.cancel
	r.cancel = nil
	bus := r.bus
	live := r.sess != nil
	r.mu.Unlock()

	if cancel != nil {
		cancel()
	}
	r.wg.Wait()

	r.mu.Lock()
	r.sess = nil
	r.snapshot = domain.Snapshot{}
	r.covered = nil
	r.phase = domain.PhaseIdle
	r.errMsg = ""
	r.mu.Unlock()

	if live {
		bus.RunFinished(nil)
	}
	r.snapshotState()
}

// Status returns the current state for the UI.
func (r *Runtime) Status() domain.Status {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.statusLocked()
}

func (r *Runtime) statusLocked() domain.Status {
	st := domain.Status{Phase: r.phase, Error: r.errMsg, CoveredAreas: r.covered}
	if r.sess == nil {
		return st
	}
	elapsed, remaining := r.clockLocked()
	st.SessionID = r.sess.ID
	problem := r.sess.Problem
	st.Problem = &problem
	st.ElapsedSec = elapsed
	st.RemainingSec = remaining
	st.DurationSec = r.sess.Spec.DurationSec
	st.CoachEnabled = r.sess.Spec.CoachEnabled
	st.Events = append([]domain.Event(nil), r.sess.Events...)
	st.Review = r.sess.Review
	return st
}

// snapshotState publishes the whole shared state. Used at phase transitions;
// per-second changes go out as deltas instead.
func (r *Runtime) snapshotState() {
	r.mu.Lock()
	bus, st := r.bus, r.statusLocked()
	r.mu.Unlock()
	bus.StateSnapshot(st)
}

func (r *Runtime) clockLocked() (elapsed, remaining int) {
	if r.sess == nil {
		return 0, 0
	}
	d := r.accumulated
	if r.phase == domain.PhaseActive {
		d += time.Since(r.resumedAt)
	}
	elapsed = int(d.Seconds())
	remaining = r.sess.Spec.DurationSec - elapsed
	if remaining < 0 {
		remaining = 0
	}
	return elapsed, remaining
}

// clockLoop drives the per-second tick and the curveball timeline.
func (r *Runtime) clockLoop(ctx context.Context) {
	defer r.wg.Done()
	t := time.NewTicker(time.Second)
	defer t.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			r.mu.Lock()
			if r.phase != domain.PhaseActive || r.sess == nil {
				r.mu.Unlock()
				continue
			}
			elapsed, remaining := r.clockLocked()
			due := r.dueCurveballsLocked(elapsed, remaining)
			pacing := harness.PacingEvent(r.sess.Problem, elapsed, remaining, r.covered, r.fired)
			expired := remaining == 0 && !r.fired["expired"]
			if expired {
				r.fired["expired"] = true
			}
			bus := r.bus
			r.mu.Unlock()

			// The clock is a state delta, not a bespoke event: the UI holds one
			// state object and patches it.
			bus.StateDelta(map[string]any{
				"elapsedSec":   elapsed,
				"remainingSec": remaining,
			})

			for _, cb := range due {
				r.say(domain.Event{
					ElapsedSec: elapsed,
					Kind:       domain.KindCurveball,
					Severity:   domain.SeverityCritical,
					Title:      cb.Title,
					Body:       cb.Body,
				})
			}
			if pacing != nil {
				r.say(*pacing)
			}
			if expired {
				r.say(domain.Event{
					ElapsedSec: elapsed,
					Kind:       domain.KindSystem,
					Severity:   domain.SeverityCritical,
					Title:      "Time is up",
					Body:       "The timebox is done. Finish the session to get scored, or keep going in overtime if you want to see where it lands.",
				})
			}
		}
	}
}

// dueCurveballsLocked returns curveballs whose trigger point has passed, and
// marks them fired. Caller must hold the mutex.
func (r *Runtime) dueCurveballsLocked(elapsed, remaining int) []domain.Curveball {
	total := elapsed + remaining
	if total <= 0 {
		return nil
	}
	pct := elapsed * 100 / total
	var due []domain.Curveball
	for i := range r.sess.Problem.Curveballs {
		cb := &r.sess.Problem.Curveballs[i]
		if !cb.Fired && pct >= cb.AtPct {
			cb.Fired = true
			due = append(due, *cb)
		}
	}
	return due
}

// coachLoop runs the live coach. Ticks are sequential: a slow model coaches
// less often rather than piling up overlapping calls.
func (r *Runtime) coachLoop(ctx context.Context, client *llm.Client) {
	defer r.wg.Done()

	interval := time.Duration(r.spec().CoachIntervalSec) * time.Second
	if interval < 15*time.Second {
		interval = 15 * time.Second
	}
	t := time.NewTicker(interval)
	defer t.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			in, ok := r.coachInput()
			if !ok {
				continue
			}
			// A tick that outlives its own interval is stale; bound it.
			tickCtx, cancel := context.WithTimeout(ctx, 3*interval)
			r.tick(tickCtx, client, in)
			cancel()
		}
	}
}

// tick runs one coaching decision and, if the coach chose to speak, streams
// the message it writes.
func (r *Runtime) tick(ctx context.Context, client *llm.Client, in harness.CoachInput) {
	r.mu.Lock()
	bus := r.bus
	r.mu.Unlock()

	bus.StepStarted(StepThinking)
	decision, err := harness.Decide(ctx, client, in)
	bus.StepFinished(StepThinking)
	if err != nil {
		if ctx.Err() == nil {
			r.log.Warn("coach decision failed", "err", err)
		}
		return
	}

	r.mu.Lock()
	if len(decision.Covered) > 0 {
		r.covered = mergeAreas(r.covered, decision.Covered)
	}
	r.lastPrint = in.Snapshot.Fingerprint()
	covered := append([]domain.Area(nil), r.covered...)
	r.mu.Unlock()

	bus.StateDelta(map[string]any{"coveredAreas": covered})

	if !decision.Speak {
		return
	}

	// The headline is known now, so the bubble can appear styled and titled
	// while the body is still being written.
	messageID := uuid.NewString()
	bus.MessageStart(agui.TextMessageStart{
		MessageID:  messageID,
		Role:       agui.RoleAssistant,
		Kind:       string(decision.Kind),
		Severity:   string(decision.Severity),
		Title:      decision.Title,
		ElapsedSec: in.ElapsedSec,
	})

	streamed := false
	body, err := harness.Speak(ctx, client, in, decision, func(chunk string) {
		streamed = true
		bus.MessageContent(messageID, chunk)
	})
	if err != nil {
		if ctx.Err() == nil {
			r.log.Warn("coach message failed", "err", err)
		}
		// The bubble is already open; close it with the point rather than
		// leaving it hanging.
		body = decision.Point
	}
	// Speak also falls back to the decision's point when the model streamed
	// nothing. Either way the body exists but never went over the wire, so the
	// bubble would stay empty while the transcript held text. Send it now:
	// what the UI shows and what the session records must be the same string.
	if !streamed && body != "" {
		bus.MessageContent(messageID, body)
	}

	areas := make([]string, len(decision.Areas))
	for i, a := range decision.Areas {
		areas[i] = string(a)
	}
	bus.MessageEnd(messageID, areas)

	r.record(domain.Event{
		ID:         messageID,
		ElapsedSec: in.ElapsedSec,
		Kind:       decision.Kind,
		Severity:   decision.Severity,
		Title:      decision.Title,
		Body:       body,
		Areas:      decision.Areas,
	})
}

// coachInput assembles a tick's input, or reports false when this tick should
// be skipped: paused, empty board, or nothing changed since the last tick.
func (r *Runtime) coachInput() (harness.CoachInput, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.phase != domain.PhaseActive || r.sess == nil {
		return harness.CoachInput{}, false
	}
	if r.snapshot.IsEmpty() {
		return harness.CoachInput{}, false
	}
	if fp := r.snapshot.Fingerprint(); fp == r.lastPrint {
		return harness.CoachInput{}, false
	}

	elapsed, remaining := r.clockLocked()
	return harness.CoachInput{
		Problem:      r.sess.Problem,
		Snapshot:     r.snapshot,
		ElapsedSec:   elapsed,
		RemainingSec: remaining,
		RecentEvents: recentSpoken(r.sess.Events, 6),
		CoveredAreas: append([]domain.Area(nil), r.covered...),
	}, true
}

func (r *Runtime) spec() domain.SessionSpec {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.sess == nil {
		return domain.SessionSpec{}
	}
	return r.sess.Spec
}

// say publishes an event whose text is already known, as a message that opens,
// delivers its body in one chunk, and closes. Going through the same message
// events as streamed coach output means the UI has one rendering path.
func (r *Runtime) say(e domain.Event) {
	if e.ID == "" {
		e.ID = uuid.NewString()
	}
	r.mu.Lock()
	bus := r.bus
	if e.ElapsedSec == 0 {
		e.ElapsedSec, _ = r.clockLocked()
	}
	r.mu.Unlock()

	role := agui.RoleSystem
	if e.Kind == domain.KindHint || e.Kind == domain.KindProbe || e.Kind == domain.KindPraise {
		role = agui.RoleAssistant
	}
	bus.MessageStart(agui.TextMessageStart{
		MessageID:  e.ID,
		Role:       role,
		Kind:       string(e.Kind),
		Severity:   string(e.Severity),
		Title:      e.Title,
		ElapsedSec: e.ElapsedSec,
	})
	bus.MessageContent(e.ID, e.Body)
	areas := make([]string, len(e.Areas))
	for i, a := range e.Areas {
		areas[i] = string(a)
	}
	bus.MessageEnd(e.ID, areas)

	r.record(e)
}

// record appends an event to the session transcript.
func (r *Runtime) record(e domain.Event) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.sess == nil {
		return
	}
	if e.ID == "" {
		e.ID = uuid.NewString()
	}
	if e.At.IsZero() {
		e.At = time.Now()
	}
	r.sess.Events = append(r.sess.Events, e)
}

func normalizeSpec(s domain.SessionSpec) domain.SessionSpec {
	if s.Mode != domain.ModeCoding {
		s.Mode = domain.ModeSystemDesign
	}
	if s.Level == "" {
		s.Level = domain.LevelMid
	}
	if s.DurationSec < 5*60 {
		s.DurationSec = 5 * 60
	}
	if s.DurationSec > 4*60*60 {
		s.DurationSec = 4 * 60 * 60
	}
	if s.CoachIntervalSec < 15 {
		s.CoachIntervalSec = 45
	}
	if s.Mode == domain.ModeCoding && s.Language == "" {
		s.Language = "go"
	}
	return s
}

func mergeAreas(existing, incoming []domain.Area) []domain.Area {
	seen := make(map[domain.Area]bool, len(existing))
	out := make([]domain.Area, 0, len(existing)+len(incoming))
	for _, a := range existing {
		if !seen[a] {
			seen[a] = true
			out = append(out, a)
		}
	}
	for _, a := range incoming {
		if !seen[a] {
			seen[a] = true
			out = append(out, a)
		}
	}
	return out
}

// recentSpoken returns the last n coach-authored events, oldest first. System
// and pacing events are excluded: repeating those is not a problem worth
// spending prompt space to prevent.
func recentSpoken(events []domain.Event, n int) []domain.Event {
	var spoken []domain.Event
	for _, e := range events {
		switch e.Kind {
		case domain.KindHint, domain.KindProbe, domain.KindPraise, domain.KindCurveball:
			spoken = append(spoken, e)
		}
	}
	if len(spoken) > n {
		spoken = spoken[len(spoken)-n:]
	}
	return spoken
}
