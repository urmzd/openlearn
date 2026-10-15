// Package agui defines the event vocabulary the backend uses to drive the UI.
//
// It follows the AG-UI protocol's event model rather than inventing one: a run
// is bracketed by RUN_STARTED and RUN_FINISHED, agent speech arrives as
// TEXT_MESSAGE_START / CONTENT / END so the UI can render it as it is produced,
// and shared state moves as STATE_SNAPSHOT plus STATE_DELTA. The frontend is a
// projection of that stream, so the agent decides when to speak and the UI
// never has to poll to find out.
//
// The transport is Wails' event bus rather than AG-UI's SSE encoding: this is a
// single-process desktop app, so the wire format buys nothing. Keeping the
// event vocabulary means a future networked mode is a transport swap, not a
// redesign.
package agui

import "time"

// Type identifies an event. Values match AG-UI's event names so the mapping
// stays obvious.
type Type string

const (
	// TypeRunStarted opens a run. Payload is RunStarted.
	TypeRunStarted Type = "RUN_STARTED"
	// TypeRunFinished closes a run normally. Payload is RunFinished.
	TypeRunFinished Type = "RUN_FINISHED"
	// TypeRunError closes a run that failed. Payload is RunError.
	TypeRunError Type = "RUN_ERROR"

	// TypeStepStarted and TypeStepFinished bracket a named phase inside a run,
	// such as generating the problem or scoring the result.
	TypeStepStarted  Type = "STEP_STARTED"
	TypeStepFinished Type = "STEP_FINISHED"

	// TypeTextMessageStart opens an agent message. Payload is TextMessageStart.
	TypeTextMessageStart Type = "TEXT_MESSAGE_START"
	// TypeTextMessageContent carries one chunk. Payload is TextMessageContent.
	TypeTextMessageContent Type = "TEXT_MESSAGE_CONTENT"
	// TypeTextMessageEnd closes a message. Payload is TextMessageEnd.
	TypeTextMessageEnd Type = "TEXT_MESSAGE_END"

	// TypeStateSnapshot carries the whole shared state. Payload is any.
	TypeStateSnapshot Type = "STATE_SNAPSHOT"
	// TypeStateDelta carries a partial update. Payload is StateDelta.
	TypeStateDelta Type = "STATE_DELTA"

	// TypeCustom carries an app-specific event that has no AG-UI equivalent.
	TypeCustom Type = "CUSTOM"
)

// Channel is the single Wails event name every AG-UI event is published on.
// One channel keeps ordering intact: the frontend sees events in exactly the
// order the backend emitted them, which matters for message start/content/end.
const Channel = "agui"

// Event is one item in the stream.
type Event struct {
	Type Type      `json:"type"`
	At   time.Time `json:"at"`
	// RunID ties the event to a session run.
	RunID string `json:"runId,omitempty"`
	// Name qualifies STEP_* and CUSTOM events.
	Name string `json:"name,omitempty"`
	// Data is the type-specific payload.
	Data any `json:"data,omitempty"`
}

// RunStarted opens a run.
type RunStarted struct {
	RunID string `json:"runId"`
	// ThreadID groups runs that belong to the same practice session.
	ThreadID string `json:"threadId"`
}

// RunFinished closes a run.
type RunFinished struct {
	RunID string `json:"runId"`
	// Result is the run's terminal payload, such as a review.
	Result any `json:"result,omitempty"`
}

// RunError closes a failed run.
type RunError struct {
	RunID   string `json:"runId"`
	Message string `json:"message"`
	// Code is a stable machine-readable reason, e.g. "no_model".
	Code string `json:"code,omitempty"`
}

// Role identifies who authored a message.
type Role string

const (
	// RoleAssistant is the coach.
	RoleAssistant Role = "assistant"
	// RoleSystem is the harness itself: the clock, the curveball timeline.
	RoleSystem Role = "system"
)

// TextMessageStart opens a streamed message.
type TextMessageStart struct {
	MessageID string `json:"messageId"`
	Role      Role   `json:"role"`
	// Kind and Severity let the UI style the message before any content
	// arrives, so a critical curveball does not first render as a neutral hint.
	Kind     string `json:"kind,omitempty"`
	Severity string `json:"severity,omitempty"`
	// Title is the headline. It is known up front; only the body streams.
	Title string `json:"title,omitempty"`
	// ElapsedSec is where in the timebox the message was produced.
	ElapsedSec int `json:"elapsedSec"`
}

// TextMessageContent is one chunk of a streamed message body.
type TextMessageContent struct {
	MessageID string `json:"messageId"`
	Delta     string `json:"delta"`
}

// TextMessageEnd closes a streamed message.
type TextMessageEnd struct {
	MessageID string `json:"messageId"`
	// Areas are the skill areas the finished message is about.
	Areas []string `json:"areas,omitempty"`
}

// StateDelta is a partial update to shared state. Keys are top-level fields of
// the state object; only the fields present are changed.
type StateDelta struct {
	Patch map[string]any `json:"patch"`
}

// Emitter publishes events. The runtime holds one; the app implementation
// forwards to the Wails event bus, and tests use a recorder.
type Emitter interface {
	Emit(Event)
}

// EmitterFunc adapts a function to Emitter.
type EmitterFunc func(Event)

// Emit implements Emitter.
func (f EmitterFunc) Emit(e Event) { f(e) }

// Discard is an Emitter that drops everything, for tests and headless runs.
var Discard Emitter = EmitterFunc(func(Event) {})

// Bus wraps an Emitter with the constructors for each event type, so callers
// never build an Event literal and cannot mismatch a Type with its payload.
type Bus struct {
	out   Emitter
	runID string
	now   func() time.Time
}

// NewBus returns a Bus that stamps every event with runID.
func NewBus(out Emitter, runID string) *Bus {
	if out == nil {
		out = Discard
	}
	return &Bus{out: out, runID: runID, now: time.Now}
}

// RunID reports the run this bus stamps events with.
func (b *Bus) RunID() string { return b.runID }

func (b *Bus) emit(t Type, name string, data any) {
	b.out.Emit(Event{Type: t, At: b.now(), RunID: b.runID, Name: name, Data: data})
}

// RunStarted opens the run.
func (b *Bus) RunStarted(threadID string) {
	b.emit(TypeRunStarted, "", RunStarted{RunID: b.runID, ThreadID: threadID})
}

// RunFinished closes the run with a terminal result.
func (b *Bus) RunFinished(result any) {
	b.emit(TypeRunFinished, "", RunFinished{RunID: b.runID, Result: result})
}

// RunError closes the run with a failure.
func (b *Bus) RunError(code, message string) {
	b.emit(TypeRunError, "", RunError{RunID: b.runID, Code: code, Message: message})
}

// StepStarted opens a named phase.
func (b *Bus) StepStarted(name string) { b.emit(TypeStepStarted, name, nil) }

// StepFinished closes a named phase.
func (b *Bus) StepFinished(name string) { b.emit(TypeStepFinished, name, nil) }

// MessageStart opens a streamed message.
func (b *Bus) MessageStart(m TextMessageStart) { b.emit(TypeTextMessageStart, "", m) }

// MessageContent appends a chunk to an open message.
func (b *Bus) MessageContent(messageID, delta string) {
	b.emit(TypeTextMessageContent, "", TextMessageContent{MessageID: messageID, Delta: delta})
}

// MessageEnd closes a streamed message.
func (b *Bus) MessageEnd(messageID string, areas []string) {
	b.emit(TypeTextMessageEnd, "", TextMessageEnd{MessageID: messageID, Areas: areas})
}

// StateSnapshot replaces the UI's copy of shared state.
func (b *Bus) StateSnapshot(state any) { b.emit(TypeStateSnapshot, "", state) }

// StateDelta patches named fields of shared state.
func (b *Bus) StateDelta(patch map[string]any) {
	if len(patch) == 0 {
		return
	}
	b.emit(TypeStateDelta, "", StateDelta{Patch: patch})
}

// Custom emits an app-specific event.
func (b *Bus) Custom(name string, data any) { b.emit(TypeCustom, name, data) }
