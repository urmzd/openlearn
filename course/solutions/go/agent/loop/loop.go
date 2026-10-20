// Package loop is the agent loop (ag.03): call the model with the messages
// and the tool definitions, run the tool calls it returns, append the
// results, and repeat until it answers without calling a tool or a limit
// stops it. Everything else is policy around that loop: an iteration cap, a
// bound on parallel tools (results stay in call order), a token and
// tool-call budget checked before anything is dispatched, a Gate consulted
// before every call (types.Gate, implemented by ag.04), and a StepRunner
// every model and tool call goes through (types.StepRunner, made durable by
// ag.05).
//
// Chapter: content/courses/ai-engineering/28-agents/03-agent-loop.md.
// Spans: course/contracts/otel/semconv.md (agent.run, agent.llm_call,
// agent.tool <name>).
package loop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"

	"tinyllm/agent/tool"
	"tinyllm/agent/types"
)

// Config is what every run of the agent shares.
type Config struct {
	Provider    types.Provider
	Tools       *tool.Registry // nil: no tools
	System      string         // added as the first message when msgs has no system message
	Name        string         // gen_ai.agent.name; "" means "agent"
	CallOptions []types.CallOption
}

// Budget caps one run; zero fields are unlimited.
type Budget struct {
	MaxTokens    int // prompt plus completion tokens over all model calls
	MaxToolCalls int // tool calls dispatched
}

// Option configures an Agent.
type Option func(*Agent)

// WithMaxIter caps model calls per run (default 8).
func WithMaxIter(n int) Option { return func(a *Agent) { a.maxIter = n } }

// WithMaxParallelTools caps tool calls in flight (default 4).
func WithMaxParallelTools(n int) Option { return func(a *Agent) { a.maxPar = n } }

// WithBudget sets the run budget.
func WithBudget(b Budget) Option { return func(a *Agent) { a.budget = b } }

// WithGate consults g before every tool call.
func WithGate(g types.Gate) Option { return func(a *Agent) { a.gate = g } }

// WithStepRunner runs every model and tool call through s.
func WithStepRunner(s types.StepRunner) Option { return func(a *Agent) { a.steps = s } }

// WithTracer records the run's spans with t.
func WithTracer(t trace.Tracer) Option { return func(a *Agent) { a.tracer = t } }

// WithApprovals marks gate markers a human approved: a NeedApproval verdict
// whose marker is listed runs.
func WithApprovals(markers ...string) Option {
	return func(a *Agent) {
		for _, m := range markers {
			a.approved[m] = true
		}
	}
}

// Agent runs the loop; it is safe to Invoke concurrently.
type Agent struct {
	cfg      Config
	maxIter  int
	maxPar   int
	budget   Budget
	gate     types.Gate
	steps    types.StepRunner
	tracer   trace.Tracer
	approved map[string]bool
}

// New builds an agent.
func New(cfg Config, opts ...Option) *Agent {
	a := &Agent{cfg: cfg, maxIter: 8, maxPar: 4, steps: types.Direct{}, approved: map[string]bool{}}
	for _, o := range opts {
		o(a)
	}
	if a.tracer == nil {
		a.tracer = noop.NewTracerProvider().Tracer("")
	}
	if a.cfg.Name == "" {
		a.cfg.Name = "agent"
	}
	if a.maxPar < 1 {
		a.maxPar = 1
	}
	return a
}

// ErrMaxIter ends a run that used every iteration without a final answer.
var ErrMaxIter = errors.New("loop: iteration limit reached")

// BudgetError ends a run before a dispatch that would exceed the budget.
type BudgetError struct {
	Resource    string // "tokens" or "tool_calls"
	Used, Limit int
}

func (e *BudgetError) Error() string {
	return fmt.Sprintf("loop: %s budget exceeded (%d of %d)", e.Resource, e.Used, e.Limit)
}

// Pending is a call waiting for a human: approve Marker and invoke again
// with the transcript.
type Pending struct {
	Call   types.ToolCall
	Marker string
	Reason string
}

// ApprovalError ends a run whose next calls need approval. None of the
// turn's calls ran; the transcript ends with the assistant message that
// proposed them, and invoking again with that transcript and
// WithApprovals(markers...) dispatches them without calling the model.
type ApprovalError struct{ Pending []Pending }

func (e *ApprovalError) Error() string {
	return fmt.Sprintf("loop: %d tool call(s) need approval", len(e.Pending))
}

// EventKind says what an Event reports.
type EventKind int

const (
	EventText       EventKind = iota + 1 // a piece of assistant text, as it streams
	EventToolCall                        // a call the model proposed, with the gate's verdict
	EventToolResult                      // a call's result, in call order
	EventUsage                           // tokens of one model call
	EventDone                            // the run ended (Err set when it failed)
)

// Event is one thing that happened in a run.
type Event struct {
	Kind    EventKind
	Iter    int
	Text    string
	Call    types.ToolCall
	Verdict types.Verdict
	Result  tool.Result
	Usage   types.Usage
	Err     error
}

// Stream is one run in progress.
type Stream struct {
	events  chan Event
	discard chan struct{}
	once    sync.Once
	done    chan struct{}
	msg     types.Message
	err     error
	msgs    []types.Message
	usage   types.Usage
}

// Events delivers the run's events and is closed when the run ends. Reading
// it is optional: Result stops delivery and waits.
func (s *Stream) Events() <-chan Event { return s.events }

// Result waits for the run to end and returns the final assistant message
// (or the last one, with the error that ended the run).
func (s *Stream) Result() (types.Message, error) {
	s.once.Do(func() { close(s.discard) })
	<-s.done
	return s.msg, s.err
}

// Transcript is every message of the run, the input included, once it ended.
func (s *Stream) Transcript() []types.Message {
	<-s.done
	return append([]types.Message(nil), s.msgs...)
}

// Usage is the run's total token use, once it ended.
func (s *Stream) Usage() types.Usage {
	<-s.done
	return s.usage
}

func (s *Stream) emit(ctx context.Context, e Event) {
	select {
	case s.events <- e:
	case <-s.discard:
	case <-ctx.Done():
	}
}

// Invoke starts a run over msgs.
func (a *Agent) Invoke(ctx context.Context, msgs []types.Message) *Stream {
	s := &Stream{events: make(chan Event, 16), discard: make(chan struct{}), done: make(chan struct{})}
	go func() {
		defer close(s.done)
		defer close(s.events)
		s.msg, s.err = a.run(ctx, s, msgs)
		s.emit(ctx, Event{Kind: EventDone, Err: s.err})
	}()
	return s
}

// llmRecord is what a model step returns (and a durable runner records).
type llmRecord struct {
	Message types.Message
	Usage   types.Usage
}

func (a *Agent) run(ctx context.Context, s *Stream, input []types.Message) (types.Message, error) {
	ctx, span := a.tracer.Start(ctx, "agent.run", trace.WithSpanKind(trace.SpanKindInternal),
		trace.WithAttributes(attribute.String("gen_ai.agent.name", a.cfg.Name)))
	defer span.End()

	msgs := append([]types.Message(nil), input...)
	if a.cfg.System != "" && (len(msgs) == 0 || msgs[0].Role != types.RoleSystem) {
		msgs = append([]types.Message{{Role: types.RoleSystem, Content: a.cfg.System}}, msgs...)
	}
	iter, tainted := 0, false
	for _, m := range msgs {
		switch m.Role {
		case types.RoleAssistant:
			iter++
		case types.RoleTool:
			tainted = true
		}
	}
	var last types.Message
	toolCalls := 0
	finish := func(m types.Message, err error) (types.Message, error) {
		s.msgs = msgs
		span.SetAttributes(attribute.Int("tl.agent.iterations", iter))
		if err != nil {
			span.SetStatus(codes.Error, err.Error())
		}
		return m, err
	}
	var defs []types.ToolDef
	if a.cfg.Tools != nil {
		defs = a.cfg.Tools.Definitions()
	}
	for {
		var reply types.Message
		if n := len(msgs); n > 0 && msgs[n-1].Role == types.RoleAssistant && len(msgs[n-1].ToolCalls) > 0 {
			reply = msgs[n-1] // resume: the calls were proposed but never answered
		} else {
			if iter >= a.maxIter {
				return finish(last, ErrMaxIter)
			}
			if b := a.budget.MaxTokens; b > 0 && s.usage.In+s.usage.Out >= b {
				return finish(last, &BudgetError{"tokens", s.usage.In + s.usage.Out, b})
			}
			iter++
			rec, err := a.callModel(ctx, s, iter, msgs, defs)
			if err != nil {
				return finish(last, fmt.Errorf("loop: model call %d: %w", iter, err))
			}
			s.usage.In += rec.Usage.In
			s.usage.Out += rec.Usage.Out
			s.emit(ctx, Event{Kind: EventUsage, Iter: iter, Usage: rec.Usage})
			reply = rec.Message
			msgs = append(msgs, reply)
			last = reply
			if len(reply.ToolCalls) == 0 {
				return finish(reply, nil)
			}
			if b := a.budget.MaxTokens; b > 0 && s.usage.In+s.usage.Out > b {
				return finish(last, &BudgetError{"tokens", s.usage.In + s.usage.Out, b})
			}
		}
		last = reply
		calls := reply.ToolCalls
		if b := a.budget.MaxToolCalls; b > 0 && toolCalls+len(calls) > b {
			return finish(last, &BudgetError{"tool_calls", toolCalls + len(calls), b})
		}

		// Gate every call before running any of them.
		verdicts := make([]types.Verdict, len(calls))
		var pending []Pending
		gctx := types.WithCallContext(ctx, types.CallContext{Tainted: tainted, Iter: iter})
		for i, c := range calls {
			v := types.Verdict{Kind: types.Allow}
			if a.gate != nil {
				var err error
				if v, err = a.gate.Check(gctx, c); err != nil {
					v = types.Verdict{Kind: types.Deny, Reason: "gate error: " + err.Error()}
				}
			}
			if v.Kind == types.NeedApproval && a.approved[v.Marker] {
				v.Kind = types.Allow
			}
			verdicts[i] = v
			s.emit(ctx, Event{Kind: EventToolCall, Iter: iter, Call: c, Verdict: v})
			if v.Kind == types.NeedApproval {
				pending = append(pending, Pending{Call: c, Marker: v.Marker, Reason: v.Reason})
				a.toolSpan(ctx, c, "needs_approval")
			}
		}
		if len(pending) > 0 {
			return finish(last, &ApprovalError{Pending: pending})
		}

		results := make([]tool.Result, len(calls))
		errs := make([]error, len(calls))
		sem := make(chan struct{}, a.maxPar)
		var wg sync.WaitGroup
		for i, c := range calls {
			if verdicts[i].Kind == types.Deny {
				results[i] = tool.Result{CallID: c.ID, Name: c.Name, Content: "denied by policy: " + verdicts[i].Reason, IsError: true}
				a.toolSpan(ctx, c, "denied")
				continue
			}
			wg.Add(1)
			go func(i int, c types.ToolCall) {
				defer wg.Done()
				sem <- struct{}{}
				defer func() { <-sem }()
				results[i], errs[i] = a.callTool(ctx, iter, i, c)
			}(i, c)
		}
		wg.Wait()
		for _, err := range errs { // a step that could not run (not a tool error) ends the run
			if err != nil {
				return finish(last, err)
			}
		}
		toolCalls += len(calls)
		for i := range calls {
			msgs = append(msgs, results[i].Message())
			s.emit(ctx, Event{Kind: EventToolResult, Iter: iter, Call: calls[i], Result: results[i]})
		}
		tainted = true // tool output is data from outside the program
		if err := ctx.Err(); err != nil {
			return finish(last, err)
		}
	}
}

// callModel runs one model call as step llm/<iter>.
func (a *Agent) callModel(ctx context.Context, s *Stream, iter int, msgs []types.Message, defs []types.ToolDef) (llmRecord, error) {
	model := types.ApplyOptions(a.cfg.CallOptions).Model
	ctx, span := a.tracer.Start(ctx, "agent.llm_call", trace.WithSpanKind(trace.SpanKindClient),
		trace.WithAttributes(attribute.String("gen_ai.request.model", model)))
	defer span.End()
	out, err := a.steps.RunStep(ctx, types.LLMStep(iter), types.StepLLM, func(ctx context.Context) ([]byte, error) {
		ch, err := a.cfg.Provider.ChatStream(ctx, msgs, defs, a.cfg.CallOptions...)
		if err != nil {
			return nil, err
		}
		var acc types.Accumulator
		for d := range ch {
			acc.Add(d)
			if t, ok := d.(types.TextDelta); ok {
				s.emit(ctx, Event{Kind: EventText, Iter: iter, Text: t.Text})
			}
		}
		if err := acc.Err(); err != nil {
			return nil, err
		}
		return json.Marshal(llmRecord{Message: acc.Message(), Usage: acc.Usage()})
	})
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
		return llmRecord{}, err
	}
	var rec llmRecord
	if err := json.Unmarshal(out, &rec); err != nil {
		return llmRecord{}, fmt.Errorf("decoding step %s: %w", types.LLMStep(iter), err)
	}
	span.SetAttributes(attribute.Int("gen_ai.usage.input_tokens", rec.Usage.In), attribute.Int("gen_ai.usage.output_tokens", rec.Usage.Out))
	return rec, nil
}

// callTool runs call i of iteration iter as step tool/<iter>/<i>/<name>.
// A failing tool is a Result with IsError (the model reads it); an error is
// returned only when the step itself could not run (the StepRunner failed,
// for example a durable journal that cannot replay a write, ag.05).
func (a *Agent) callTool(ctx context.Context, iter, i int, c types.ToolCall) (tool.Result, error) {
	ctx, span := a.tracer.Start(ctx, "agent.tool "+c.Name, trace.WithSpanKind(trace.SpanKindInternal),
		trace.WithAttributes(attribute.String("gen_ai.tool.name", c.Name)))
	defer span.End()
	out, err := a.steps.RunStep(ctx, types.ToolStep(iter, i, c.Name), types.StepTool, func(ctx context.Context) ([]byte, error) {
		var res tool.Result
		if a.cfg.Tools == nil {
			res = tool.Result{CallID: c.ID, Name: c.Name, Content: "no tools are registered", IsError: true}
		} else {
			res = a.cfg.Tools.Call(ctx, c)
		}
		return json.Marshal(res)
	})
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
		return tool.Result{}, fmt.Errorf("loop: step %s: %w", types.ToolStep(iter, i, c.Name), err)
	}
	var res tool.Result
	if err := json.Unmarshal(out, &res); err != nil {
		return tool.Result{}, fmt.Errorf("decoding step %s: %w", types.ToolStep(iter, i, c.Name), err)
	}
	res.CallID, res.Name = c.ID, c.Name // a recorded result answers this call
	outcome := "ok"
	if res.IsError {
		outcome = "error"
		span.SetStatus(codes.Error, res.Content)
	}
	span.SetAttributes(attribute.String("tl.tool.outcome", outcome))
	return res, nil
}

// toolSpan records a call that did not run (denied, or waiting for approval).
func (a *Agent) toolSpan(ctx context.Context, c types.ToolCall, outcome string) {
	_, span := a.tracer.Start(ctx, "agent.tool "+c.Name, trace.WithSpanKind(trace.SpanKindInternal),
		trace.WithAttributes(attribute.String("gen_ai.tool.name", c.Name), attribute.String("tl.tool.outcome", outcome)))
	span.End()
}
