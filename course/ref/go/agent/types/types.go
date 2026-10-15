// Package types is the agent SDK's shared vocabulary (ag.01): messages, tool
// calls, tool definitions, the deltas a Provider streams, the Accumulator
// that turns deltas back into a message, and the two seams later modules
// plug into the loop: Gate (ag.04 implements it) and StepRunner (ag.05).
//
// The seams live here, not in the packages that implement them, so the loop
// (ag.03) imports neither the gate nor the durable runner: Go interfaces
// belong to the package that consumes them, and every consumer of these two
// is downstream of types.
//
// Chapter: content/tracks/ai-platform-engineering/13-agent-sdk/01-types-and-provider.md.
// Wire contract: course/contracts/openapi/openai-subset.v1.yaml (chat
// completions with tools, streaming).
package types

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// Role is who wrote a message.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// Message is one turn of a conversation. An assistant message may carry tool
// calls instead of (or as well as) text; a tool message answers exactly one
// call, named by ToolCallID.
type Message struct {
	Role       Role
	Content    string
	ToolCalls  []ToolCall
	ToolCallID string // role tool: the id of the call this message answers
	Name       string // role tool: the tool's name
}

// ToolCall is the model asking for one tool to run. Args is the JSON object
// the model wrote, exactly as assembled from the stream (it may be invalid:
// the tool registry, ag.02, reports that back to the model).
type ToolCall struct {
	ID   string
	Name string
	Args json.RawMessage
}

// ToolDef describes a tool to the model: Parameters is a JSON Schema object.
type ToolDef struct {
	Name        string
	Description string
	Parameters  json.RawMessage
}

// Usage counts tokens: In for the prompt, Out for the completion.
type Usage struct{ In, Out int }

// Delta is one piece of a streamed answer. The concrete types are below; a
// stream ends with exactly one DoneDelta or one ErrorDelta, and then the
// channel is closed.
type Delta interface{ isDelta() }

// TextDelta is a non-empty piece of the assistant's text.
type TextDelta struct{ Text string }

// ToolCallStartDelta opens tool call Index (its position in tool_calls).
type ToolCallStartDelta struct {
	Index    int
	ID, Name string
}

// ToolCallArgsDelta is a fragment of call Index's serialized arguments.
type ToolCallArgsDelta struct {
	Index    int
	Fragment string
}

// ToolCallEndDelta closes call Index and carries the whole call.
type ToolCallEndDelta struct {
	Index int
	Call  ToolCall
}

// UsageDelta reports the token counts (the stream's usage chunk).
type UsageDelta struct{ In, Out int }

// ErrorDelta ends a stream that failed.
type ErrorDelta struct{ Err error }

// DoneDelta ends a stream that completed; FinishReason is the provider's
// (stop, length, tool_calls, content_filter).
type DoneDelta struct{ FinishReason string }

func (TextDelta) isDelta()          {}
func (ToolCallStartDelta) isDelta() {}
func (ToolCallArgsDelta) isDelta()  {}
func (ToolCallEndDelta) isDelta()   {}
func (UsageDelta) isDelta()         {}
func (ErrorDelta) isDelta()         {}
func (DoneDelta) isDelta()          {}

// IsContent reports whether d is model output (text or any tool-call
// delta). The retry wrapper may retry only before the first one.
func IsContent(d Delta) bool {
	// SOLUTION-BEGIN ag.01
	switch d.(type) {
	case TextDelta, ToolCallStartDelta, ToolCallArgsDelta, ToolCallEndDelta:
		return true
	}
	return false
	// SOLUTION-END
}

// CallOptions are per-call settings; zero values mean "provider default".
type CallOptions struct {
	Model       string
	Temperature *float64
	MaxTokens   int
	Seed        *int64
	ToolChoice  string // "", auto, none, required
}

// CallOption sets one CallOptions field.
type CallOption func(*CallOptions)

func WithModel(m string) CallOption        { return func(o *CallOptions) { o.Model = m } }
func WithTemperature(t float64) CallOption { return func(o *CallOptions) { o.Temperature = &t } }
func WithMaxTokens(n int) CallOption       { return func(o *CallOptions) { o.MaxTokens = n } }
func WithSeed(s int64) CallOption          { return func(o *CallOptions) { o.Seed = &s } }
func WithToolChoice(c string) CallOption   { return func(o *CallOptions) { o.ToolChoice = c } }
func ApplyOptions(opts []CallOption) CallOptions {
	var o CallOptions
	for _, f := range opts {
		f(&o)
	}
	return o
}

// Provider is a chat model behind any OpenAI-compatible endpoint: the
// learner's gateway, an engine, or a frontier API. ChatStream returns an
// error when the request is refused before streaming starts (with the HTTP
// status in it); after that, failures arrive as an ErrorDelta.
type Provider interface {
	ChatStream(ctx context.Context, msgs []Message, tools []ToolDef, opts ...CallOption) (<-chan Delta, error)
}

// ErrIncomplete is the Accumulator's error when the stream closed without a
// DoneDelta or an ErrorDelta.
var ErrIncomplete = errors.New("stream closed without a done or error delta")

// Accumulator rebuilds the assistant message from a stream's deltas. Tool
// calls come out in Index order, whatever order their fragments arrived in.
type Accumulator struct {
	text   strings.Builder
	calls  map[int]*ToolCall
	usage  Usage
	finish string
	err    error
	done   bool
}

// Add folds one delta in.
func (a *Accumulator) Add(d Delta) {
	// SOLUTION-BEGIN ag.01
	if a.calls == nil {
		a.calls = map[int]*ToolCall{}
	}
	call := func(i int) *ToolCall {
		c, ok := a.calls[i]
		if !ok {
			c = &ToolCall{}
			a.calls[i] = c
		}
		return c
	}
	switch v := d.(type) {
	case TextDelta:
		a.text.WriteString(v.Text)
	case ToolCallStartDelta:
		c := call(v.Index)
		c.ID, c.Name = v.ID, v.Name
	case ToolCallArgsDelta:
		c := call(v.Index)
		c.Args = append(c.Args, v.Fragment...)
	case ToolCallEndDelta:
		c := v.Call
		a.calls[v.Index] = &c
	case UsageDelta:
		a.usage = Usage{In: v.In, Out: v.Out}
	case ErrorDelta:
		a.err = v.Err
		a.done = true
	case DoneDelta:
		a.finish = v.FinishReason
		a.done = true
	}
	// SOLUTION-END
}

// Message is the assistant message so far.
func (a *Accumulator) Message() Message {
	// SOLUTION-BEGIN ag.01
	m := Message{Role: RoleAssistant, Content: a.text.String()}
	idx := make([]int, 0, len(a.calls))
	for i := range a.calls {
		idx = append(idx, i)
	}
	sort.Ints(idx)
	for _, i := range idx {
		c := *a.calls[i]
		if len(c.Args) == 0 {
			c.Args = json.RawMessage("{}")
		}
		m.ToolCalls = append(m.ToolCalls, c)
	}
	return m
	// SOLUTION-END
}

func (a *Accumulator) Usage() Usage         { return a.usage }
func (a *Accumulator) FinishReason() string { return a.finish }

// Err is the stream's error: the ErrorDelta's, ErrIncomplete when the stream
// has not ended, or nil after a DoneDelta.
func (a *Accumulator) Err() error {
	// SOLUTION-BEGIN ag.01
	if !a.done {
		return ErrIncomplete
	}
	return a.err
	// SOLUTION-END
}

// Collect drains a stream into one message.
func Collect(ch <-chan Delta) (Message, Usage, error) {
	// SOLUTION-BEGIN ag.01
	var a Accumulator
	for d := range ch {
		a.Add(d)
	}
	return a.Message(), a.Usage(), a.Err()
	// SOLUTION-END
}

// ---- the gate seam (ag.04 implements Gate) ---------------------------------

// VerdictKind is a gate's decision about one tool call.
type VerdictKind int

const (
	Allow        VerdictKind = iota // run it
	Deny                            // do not run it; the model is told why
	NeedApproval                    // do not run it until a human approves Marker
)

func (k VerdictKind) String() string {
	switch k {
	case Allow:
		return "allow"
	case Deny:
		return "deny"
	case NeedApproval:
		return "needs_approval"
	}
	return "verdict(" + strconv.Itoa(int(k)) + ")"
}

// Verdict is a gate's answer. Marker identifies an approval request: the
// same call (same name, same arguments) always has the same marker, so an
// approval given for it is recognized when the call is proposed again.
type Verdict struct {
	Kind   VerdictKind
	Reason string
	Marker string
}

// Gate decides, before dispatch, whether a tool call may run.
type Gate interface {
	Check(ctx context.Context, c ToolCall) (Verdict, error)
}

// CallContext is what the loop knows about the conversation when it asks the
// gate: Tainted is true once any tool result (data from outside the
// program: a web page, a retrieved chunk, a database row) is in the context.
type CallContext struct {
	Tainted bool
	Iter    int
}

type callContextKey struct{}

// WithCallContext attaches cc to ctx for Gate.Check.
func WithCallContext(ctx context.Context, cc CallContext) context.Context {
	return context.WithValue(ctx, callContextKey{}, cc)
}

// CallContextFrom returns the attached CallContext; with none, the zero
// value would claim an untainted context, so it reports Tainted: true (fail
// closed).
func CallContextFrom(ctx context.Context) CallContext {
	// SOLUTION-BEGIN ag.01
	if cc, ok := ctx.Value(callContextKey{}).(CallContext); ok {
		return cc
	}
	return CallContext{Tainted: true}
	// SOLUTION-END
}

// ---- the step seam (ag.05 implements StepRunner) ---------------------------

// StepKind says what a step does.
type StepKind int

const (
	StepLLM  StepKind = iota + 1 // one model call
	StepTool                     // one tool call
)

// StepRunner runs one named step of an agent run and returns its result.
// The direct runner just calls fn; a durable runner (ag.05) records the
// result under name and, on replay, returns the recording without calling fn.
// Names are deterministic: LLMStep and ToolStep below.
type StepRunner interface {
	RunStep(ctx context.Context, name string, kind StepKind, fn func(context.Context) ([]byte, error)) ([]byte, error)
}

// Direct is the StepRunner that records nothing.
type Direct struct{}

func (Direct) RunStep(ctx context.Context, _ string, _ StepKind, fn func(context.Context) ([]byte, error)) ([]byte, error) {
	return fn(ctx)
}

// LLMStep names the model call of iteration iter: "llm/<iter>".
func LLMStep(iter int) string { return fmt.Sprintf("llm/%d", iter) }

// ToolStep names call index i of iteration iter: "tool/<iter>/<i>/<name>".
func ToolStep(iter, i int, name string) string { return fmt.Sprintf("tool/%d/%d/%s", iter, i, name) }

// StepToolName is the tool name in a ToolStep name ("" and false otherwise).
func StepToolName(step string) (string, bool) {
	// SOLUTION-BEGIN ag.01
	parts := strings.SplitN(step, "/", 4)
	if len(parts) != 4 || parts[0] != "tool" || parts[3] == "" {
		return "", false
	}
	return parts[3], true
	// SOLUTION-END
}
