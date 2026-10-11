// Package domain holds the data model shared by the harness engines, the
// session runtime, and the Wails bindings. Every type here is JSON-tagged
// because it crosses into TypeScript.
package domain

import "time"

// Mode is the kind of practice a session runs.
type Mode string

const (
	// ModeSystemDesign practices architecture on a node-and-edge whiteboard.
	ModeSystemDesign Mode = "system_design"
	// ModeCoding practices implementation in a code editor.
	ModeCoding Mode = "coding"
)

// Level is the seniority bar a problem and its rubric are calibrated to.
type Level string

const (
	LevelJunior Level = "junior"
	LevelMid    Level = "mid"
	LevelSenior Level = "senior"
	LevelStaff  Level = "staff"
)

// SessionSpec is what the user asks for before a session starts.
type SessionSpec struct {
	Mode  Mode  `json:"mode"`
	Level Level `json:"level"`
	// Topic is a free-text hint ("rate limiter", "graphs", "event-driven
	// inventory"). Empty means the generator picks something.
	Topic string `json:"topic"`
	// Language is the coding-mode target language. Ignored for system design.
	Language string `json:"language"`
	// DurationSec is the timebox. The whole harness paces against it.
	DurationSec int `json:"durationSec"`
	// CustomStatement, when set, is used verbatim instead of generating a
	// problem. The rubric is still generated from it.
	CustomStatement string `json:"customStatement"`
	// CoachEnabled turns the live coach on. Off means a silent run, scored
	// only at the end.
	CoachEnabled bool `json:"coachEnabled"`
	// CoachIntervalSec is how often the coach is allowed to consider speaking.
	CoachIntervalSec int `json:"coachIntervalSec"`
}

// Criterion is one row of a problem's hidden rubric.
type Criterion struct {
	ID     string `json:"id"`
	Area   Area   `json:"area"`
	Title  string `json:"title"`
	Detail string `json:"detail"`
	// Weight is 1-5; higher means it matters more to the final score.
	Weight int `json:"weight"`
}

// Curveball is a scheduled complication revealed partway through the timebox,
// the way an interviewer changes the requirements once you look comfortable.
type Curveball struct {
	// AtPct is the percentage of the timebox at which it fires (0-100).
	AtPct int    `json:"atPct"`
	Title string `json:"title"`
	Body  string `json:"body"`
	// Fired is set by the runtime once delivered.
	Fired bool `json:"fired"`
}

// Problem is a generated practice prompt plus everything needed to grade it.
type Problem struct {
	ID    string `json:"id"`
	Mode  Mode   `json:"mode"`
	Level Level  `json:"level"`
	Title string `json:"title"`
	// Statement is markdown shown to the user.
	Statement string `json:"statement"`
	// Requirements are the functional asks, shown to the user.
	Requirements []string `json:"requirements"`
	// Constraints are the scale and non-functional numbers, shown to the user.
	Constraints []string `json:"constraints"`
	// Rubric is hidden until review.
	Rubric []Criterion `json:"rubric"`
	// Curveballs fire on the timeline.
	Curveballs []Curveball `json:"curveballs"`
	// ReferenceOutline is what a strong answer covers. Hidden until review.
	ReferenceOutline []string `json:"referenceOutline"`
	// Starter is coding-mode scaffolding (signature, imports, examples).
	Starter  string `json:"starter"`
	Language string `json:"language"`
}

// EventKind classifies what the coach is doing when it speaks.
type EventKind string

const (
	// KindHint points at something missing without naming the answer.
	KindHint EventKind = "hint"
	// KindProbe is an interviewer-style question about what is on the board.
	KindProbe EventKind = "probe"
	// KindCurveball is a scheduled requirement change.
	KindCurveball EventKind = "curveball"
	// KindPraise confirms a strong move so the user knows to keep going.
	KindPraise EventKind = "praise"
	// KindPacing is a timebox nudge, emitted locally without the LLM.
	KindPacing EventKind = "pacing"
	// KindSystem is harness plumbing (session started, generation failed).
	KindSystem EventKind = "system"
)

// Severity controls how loudly the UI surfaces an event.
type Severity string

const (
	SeverityInfo     Severity = "info"
	SeverityWarn     Severity = "warn"
	SeverityCritical Severity = "critical"
)

// Event is one thing the harness says during a session.
type Event struct {
	ID         string    `json:"id"`
	At         time.Time `json:"at"`
	ElapsedSec int       `json:"elapsedSec"`
	Kind       EventKind `json:"kind"`
	Severity   Severity  `json:"severity"`
	Title      string    `json:"title"`
	Body       string    `json:"body"`
	// Areas are the skill areas this event is about.
	Areas []Area `json:"areas"`
}

// NodeKind is the shape of a whiteboard box.
type NodeKind string

const (
	NodeClient   NodeKind = "client"
	NodeService  NodeKind = "service"
	NodeDatabase NodeKind = "database"
	NodeCache    NodeKind = "cache"
	NodeQueue    NodeKind = "queue"
	NodeStorage  NodeKind = "storage"
	NodeCDN      NodeKind = "cdn"
	NodeBalancer NodeKind = "balancer"
	NodeExternal NodeKind = "external"
	NodeNote     NodeKind = "note"
)

// BoardNode is one box on the whiteboard.
type BoardNode struct {
	ID    string   `json:"id"`
	Kind  NodeKind `json:"kind"`
	Label string   `json:"label"`
	// Detail is the annotation the user typed into the box.
	Detail string  `json:"detail"`
	X      float64 `json:"x"`
	Y      float64 `json:"y"`
}

// BoardEdge is one arrow on the whiteboard.
type BoardEdge struct {
	ID     string `json:"id"`
	Source string `json:"source"`
	Target string `json:"target"`
	Label  string `json:"label"`
}

// Snapshot is the state of the user's work at a moment in time. The coach and
// the reviewer both read it; nothing else in the harness sees the raw board.
type Snapshot struct {
	Nodes []BoardNode `json:"nodes"`
	Edges []BoardEdge `json:"edges"`
	// Notes is the freeform talking track next to the board or editor.
	Notes string `json:"notes"`
	// Code and Language carry coding-mode work.
	Code     string `json:"code"`
	Language string `json:"language"`
}

// CriterionScore is the reviewer's verdict on one rubric row.
type CriterionScore struct {
	CriterionID string `json:"criterionId"`
	Area        Area   `json:"area"`
	Title       string `json:"title"`
	// Score is 0-4: 0 absent, 1 named, 2 partial, 3 solid, 4 excellent.
	Score int `json:"score"`
	// Evidence quotes what in the work earned (or failed to earn) the score.
	Evidence string `json:"evidence"`
}

// Review is the end-of-session assessment.
type Review struct {
	// Overall is 0-100, weight-normalized across criteria.
	Overall int `json:"overall"`
	// Verdict is the calibrated level the work reads at.
	Verdict   Level            `json:"verdict"`
	Summary   string           `json:"summary"`
	Scores    []CriterionScore `json:"scores"`
	Strengths []string         `json:"strengths"`
	Gaps      []string         `json:"gaps"`
	NextSteps []string         `json:"nextSteps"`
	// MissedOutline are reference-outline points the work never reached.
	MissedOutline []string `json:"missedOutline"`
}

// Phase is where a session is in its lifecycle.
type Phase string

const (
	PhaseIdle       Phase = "idle"
	PhaseGenerating Phase = "generating"
	PhaseActive     Phase = "active"
	PhasePaused     Phase = "paused"
	PhaseReviewing  Phase = "reviewing"
	PhaseDone       Phase = "done"
	PhaseFailed     Phase = "failed"
)

// Session is a full practice run, persisted to disk when it ends.
type Session struct {
	ID        string      `json:"id"`
	Spec      SessionSpec `json:"spec"`
	Problem   Problem     `json:"problem"`
	StartedAt time.Time   `json:"startedAt"`
	EndedAt   time.Time   `json:"endedAt"`
	// ElapsedSec is time actually spent, excluding pauses.
	ElapsedSec int      `json:"elapsedSec"`
	Events     []Event  `json:"events"`
	Final      Snapshot `json:"final"`
	Review     *Review  `json:"review"`
	Phase      Phase    `json:"phase"`
	Provider   string   `json:"provider"`
	Model      string   `json:"model"`
}

// AreaStat is one row of the aggregated skill profile.
type AreaStat struct {
	Area  Area   `json:"area"`
	Label string `json:"label"`
	// Average is the mean 0-4 score across sessions that scored this area.
	Average float64 `json:"average"`
	// Samples is how many criteria across all sessions fed the average.
	Samples int `json:"samples"`
	// Trend is the delta between the most recent third and the earliest third
	// of samples; positive means improving.
	Trend float64 `json:"trend"`
	// Last is the most recent score recorded for this area.
	Last int `json:"last"`
}

// Profile is the cross-session view of where the user is at.
type Profile struct {
	Mode         Mode       `json:"mode"`
	Sessions     int        `json:"sessions"`
	TotalMinutes int        `json:"totalMinutes"`
	AverageScore float64    `json:"averageScore"`
	Areas        []AreaStat `json:"areas"`
	// Strongest and Weakest are area labels, ranked by Average.
	Strongest []string `json:"strongest"`
	Weakest   []string `json:"weakest"`
}

// Status is the shared state the UI holds, delivered as AG-UI state snapshots
// and patched by state deltas.
type Status struct {
	Phase        Phase    `json:"phase"`
	SessionID    string   `json:"sessionId"`
	Problem      *Problem `json:"problem"`
	ElapsedSec   int      `json:"elapsedSec"`
	RemainingSec int      `json:"remainingSec"`
	DurationSec  int      `json:"durationSec"`
	// CoachEnabled reflects the running session's spec, so the UI can say the
	// coach is off rather than looking like it is broken.
	CoachEnabled bool    `json:"coachEnabled"`
	Events       []Event `json:"events"`
	// CoveredAreas are areas the coach has observed the user touch.
	CoveredAreas []Area  `json:"coveredAreas"`
	Review       *Review `json:"review"`
	Error        string  `json:"error"`
}
