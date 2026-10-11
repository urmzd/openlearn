// Course tests for ag.03, the agent loop. The model is a scripted fake
// Provider (a list of replies, one per call) so every test knows exactly
// what the model "says"; one test drives the real provider (ag.01) against
// the course's faketool server end to end. Tools that must overlap block on
// channels the test releases, so concurrency is controlled, never timed.
package ag_03

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"openlearn.urmzd.com/tl/testkit/faketool"
	"tinyllm/agent/loop"
	"tinyllm/agent/provider"
	"tinyllm/agent/tool"
	"tinyllm/agent/types"
)

// reply is one scripted model answer.
type reply struct {
	text  string
	calls []types.ToolCall
	usage types.Usage
	err   error
}

// script is a fake Provider: call i gets replies[i] (the last repeats).
type script struct {
	mu      sync.Mutex
	replies []reply
	seen    [][]types.Message
	tools   [][]types.ToolDef
}

func (s *script) ChatStream(ctx context.Context, msgs []types.Message, tools []types.ToolDef, _ ...types.CallOption) (<-chan types.Delta, error) {
	s.mu.Lock()
	i := len(s.seen)
	s.seen = append(s.seen, append([]types.Message(nil), msgs...))
	s.tools = append(s.tools, tools)
	r := s.replies[min(i, len(s.replies)-1)]
	s.mu.Unlock()
	if r.err != nil {
		return nil, r.err
	}
	ch := make(chan types.Delta, 64)
	go func() {
		defer close(ch)
		for _, w := range strings.SplitAfter(r.text, " ") {
			if w != "" {
				ch <- types.TextDelta{Text: w}
			}
		}
		for j, c := range r.calls {
			ch <- types.ToolCallStartDelta{Index: j, ID: c.ID, Name: c.Name}
			ch <- types.ToolCallArgsDelta{Index: j, Fragment: string(c.Args)}
		}
		ch <- types.UsageDelta{In: r.usage.In, Out: r.usage.Out}
		ch <- types.DoneDelta{FinishReason: "stop"}
	}()
	return ch, nil
}

func (s *script) calls() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.seen)
}

func call(id, name, args string) types.ToolCall {
	return types.ToolCall{ID: id, Name: name, Args: json.RawMessage(args)}
}

var user = []types.Message{{Role: types.RoleUser, Content: "What is the weather in Paris?"}}

func weatherTools(runs *atomic.Int32) *tool.Registry {
	return tool.NewRegistry().MustRegister(tool.New("get_weather", "Current weather",
		`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`,
		func(_ context.Context, a json.RawMessage) (string, error) {
			runs.Add(1)
			return "sunny, 21 C", nil
		}))
}

func TestHandExample(t *testing.T) {
	// WHY: section 3 traced by hand. Iteration 1: the model calls
	//      get_weather({"city":"Paris"}) (10 in, 5 out); the tool answers.
	//      Iteration 2: the model reads the result and answers "It is sunny."
	//      (20 in, 4 out). Four messages, two model calls, one tool run,
	//      usage 30/9, and the second call sees the tool message answering
	//      call_1.
	// KIND: unit
	// CATCHES: s12, s13
	// CHAPTER: ag.03 section 3, worked example
	var runs atomic.Int32
	p := &script{replies: []reply{
		{calls: []types.ToolCall{call("call_1", "get_weather", `{"city":"Paris"}`)}, usage: types.Usage{In: 10, Out: 5}},
		{text: "It is sunny.", usage: types.Usage{In: 20, Out: 4}},
	}}
	st := loop.New(loop.Config{Provider: p, Tools: weatherTools(&runs)}).Invoke(context.Background(), user)
	var kinds []loop.EventKind
	for e := range st.Events() {
		kinds = append(kinds, e.Kind)
	}
	msg, err := st.Result()
	if err != nil || msg.Content != "It is sunny." || runs.Load() != 1 || p.calls() != 2 {
		t.Fatalf("result %q, %v; %d tool runs, %d model calls", msg.Content, err, runs.Load(), p.calls())
	}
	if u := st.Usage(); u != (types.Usage{In: 30, Out: 9}) {
		t.Fatalf("usage %+v, want {30 9}", u)
	}
	tr := st.Transcript()
	if len(tr) != 4 || tr[1].Role != types.RoleAssistant || len(tr[1].ToolCalls) != 1 ||
		tr[2].Role != types.RoleTool || tr[2].ToolCallID != "call_1" || tr[2].Content != "sunny, 21 C" || tr[3].Content != "It is sunny." {
		t.Fatalf("transcript %+v", tr)
	}
	second := p.seen[1]
	if len(second) != 3 || second[1].Role != types.RoleAssistant || second[2].Role != types.RoleTool {
		t.Fatalf("the second model call saw %+v; want user, assistant (the call), tool (the result)", second)
	}
	want := []loop.EventKind{loop.EventUsage, loop.EventToolCall, loop.EventToolResult, loop.EventText, loop.EventText, loop.EventText, loop.EventUsage, loop.EventDone}
	if !reflect.DeepEqual(kinds, want) {
		t.Fatalf("events %v, want %v", kinds, want)
	}
}

func TestSystemPrompt(t *testing.T) {
	// WHY: Config.System is the agent's standing instruction: added once as
	//      the first message, and not again when the caller's transcript
	//      already starts with a system message (a resumed run).
	// KIND: unit
	// CATCHES: s10
	// CHAPTER: ag.03 section 2.1
	p := &script{replies: []reply{{text: "ok"}}}
	a := loop.New(loop.Config{Provider: p, System: "Be terse."})
	if _, err := a.Invoke(context.Background(), user).Result(); err != nil {
		t.Fatal(err)
	}
	if got := p.seen[0]; len(got) != 2 || got[0].Role != types.RoleSystem || got[0].Content != "Be terse." {
		t.Fatalf("first call saw %+v", got)
	}
	withSys := append([]types.Message{{Role: types.RoleSystem, Content: "Custom."}}, user...)
	if _, err := a.Invoke(context.Background(), withSys).Result(); err != nil {
		t.Fatal(err)
	}
	if got := p.seen[1]; len(got) != 2 || got[0].Content != "Custom." {
		t.Fatalf("a transcript with its own system message got %+v", got)
	}
}

func TestMaxIterStops(t *testing.T) {
	// WHY: a model that calls a tool forever must not run forever: with
	//      WithMaxIter(3) there are exactly 3 model calls and 3 tool runs,
	//      then ErrMaxIter, with the last assistant message returned.
	// KIND: boundary
	// CATCHES: s01
	// CHAPTER: ag.03 section 2.2
	var runs atomic.Int32
	p := &script{replies: []reply{{calls: []types.ToolCall{call("c", "get_weather", `{"city":"Paris"}`)}}}}
	msg, err := loop.New(loop.Config{Provider: p, Tools: weatherTools(&runs)}, loop.WithMaxIter(3)).Invoke(context.Background(), user).Result()
	if !errors.Is(err, loop.ErrMaxIter) || p.calls() != 3 || runs.Load() != 3 {
		t.Fatalf("err %v after %d model calls and %d tool runs; want ErrMaxIter after 3 and 3", err, p.calls(), runs.Load())
	}
	if len(msg.ToolCalls) != 1 {
		t.Fatalf("the last assistant message is returned with the error, got %+v", msg)
	}
}

func TestParallelToolsBoundedOrderPreserved(t *testing.T) {
	// WHY: six calls in one turn with WithMaxParallelTools(2): never more
	//      than two run at once, and although the test finishes them in
	//      reverse order, the tool messages come back in call order (the
	//      model matches results to calls by position as well as by id).
	// KIND: unit
	// CATCHES: s02, s03
	// CHAPTER: ag.03 section 2.3
	const n = 6
	var inFlight, maxSeen atomic.Int32
	started := make(chan int, n)
	release := make([]chan struct{}, n)
	for i := range release {
		release[i] = make(chan struct{})
	}
	reg := tool.NewRegistry().MustRegister(tool.New("slow", "", `{"type":"object","properties":{"i":{"type":"integer"}}}`,
		func(_ context.Context, a json.RawMessage) (string, error) {
			var x struct{ I int }
			json.Unmarshal(a, &x)
			cur := inFlight.Add(1)
			for {
				m := maxSeen.Load()
				if cur <= m || maxSeen.CompareAndSwap(m, cur) {
					break
				}
			}
			started <- x.I
			<-release[x.I]
			inFlight.Add(-1)
			return fmt.Sprintf("done %d", x.I), nil
		}))
	var calls []types.ToolCall
	for i := 0; i < n; i++ {
		calls = append(calls, call(fmt.Sprintf("c%d", i), "slow", fmt.Sprintf(`{"i":%d}`, i)))
	}
	p := &script{replies: []reply{{calls: calls}, {text: "all done"}}}
	st := loop.New(loop.Config{Provider: p, Tools: reg}, loop.WithMaxParallelTools(2)).Invoke(context.Background(), user)
	// Keep two running (fewer at the end), always finish the highest index
	// first: completion order is the reverse of call order.
	go func() {
		running := map[int]bool{}
		for released := 0; released < n; released++ {
			for len(running) < min(2, n-released) {
				running[<-started] = true
			}
			if released == 0 {
				// A loop without the bound starts a third call at once; give
				// it the chance (a correct loop never does, so this only
				// costs 200 ms).
				select {
				case i := <-started:
					running[i] = true
				case <-time.After(200 * time.Millisecond):
				}
			}
			hi := -1
			for i := range running {
				hi = max(hi, i)
			}
			delete(running, hi)
			close(release[hi])
		}
	}()
	if _, err := st.Result(); err != nil {
		t.Fatal(err)
	}
	if maxSeen.Load() != 2 {
		t.Fatalf("at most %d tools ran at once, want exactly 2", maxSeen.Load())
	}
	tr := st.Transcript()
	for i := 0; i < n; i++ {
		m := tr[2+i]
		if m.Role != types.RoleTool || m.ToolCallID != fmt.Sprintf("c%d", i) || m.Content != fmt.Sprintf("done %d", i) {
			t.Fatalf("tool message %d = %+v; want c%d / done %d", i, m, i, i)
		}
	}
}

func TestBudgetStopsBeforeDispatch(t *testing.T) {
	// WHY: a budget is a promise about spend, so it is checked before work
	//      starts: three calls against MaxToolCalls 2 run none of them, and a
	//      reply that took the run past MaxTokens dispatches nothing.
	// KIND: boundary
	// CATCHES: s04, s05
	// CHAPTER: ag.03 section 2.4
	var runs atomic.Int32
	three := []types.ToolCall{call("a", "get_weather", `{"city":"A"}`), call("b", "get_weather", `{"city":"B"}`), call("c", "get_weather", `{"city":"C"}`)}
	p := &script{replies: []reply{{calls: three}}}
	_, err := loop.New(loop.Config{Provider: p, Tools: weatherTools(&runs)}, loop.WithBudget(loop.Budget{MaxToolCalls: 2})).Invoke(context.Background(), user).Result()
	var be *loop.BudgetError
	if !errors.As(err, &be) || be.Resource != "tool_calls" || be.Used != 3 || be.Limit != 2 || runs.Load() != 0 {
		t.Fatalf("err %v, %d runs; want BudgetError{tool_calls 3 2} and no run", err, runs.Load())
	}
	p = &script{replies: []reply{{calls: three[:1], usage: types.Usage{In: 60, Out: 50}}}}
	_, err = loop.New(loop.Config{Provider: p, Tools: weatherTools(&runs)}, loop.WithBudget(loop.Budget{MaxTokens: 100})).Invoke(context.Background(), user).Result()
	if !errors.As(err, &be) || be.Resource != "tokens" || be.Used != 110 || runs.Load() != 0 {
		t.Fatalf("err %v, %d runs; want BudgetError{tokens 110 100} and no run", err, runs.Load())
	}
}

// recGate allows everything except the names it denies or holds, and
// records the call context of each check.
type recGate struct {
	mu   sync.Mutex
	deny map[string]bool
	hold map[string]bool
	seen []types.CallContext
}

func (g *recGate) Check(ctx context.Context, c types.ToolCall) (types.Verdict, error) {
	g.mu.Lock()
	g.seen = append(g.seen, types.CallContextFrom(ctx))
	g.mu.Unlock()
	switch {
	case g.deny[c.Name]:
		return types.Verdict{Kind: types.Deny, Reason: "not allowed here"}, nil
	case g.hold[c.Name]:
		return types.Verdict{Kind: types.NeedApproval, Marker: "m-" + c.ID, Reason: "sends email"}, nil
	}
	return types.Verdict{Kind: types.Allow}, nil
}

func TestGateVerdicts(t *testing.T) {
	// WHY: the gate decides before dispatch. Deny: the tool does not run and
	//      the model reads why. NeedApproval: nothing in the turn runs, the
	//      run ends with the marker; invoking again with the transcript and
	//      the approval runs the call once, without asking the model again.
	//      The gate sees Tainted false before any tool output and true after.
	// KIND: unit
	// CATCHES: s06, s07, s08, s09, s14
	// CHAPTER: ag.03 section 2.5
	var weather, sent atomic.Int32
	reg := weatherTools(&weather).MustRegister(
		tool.New("rm", "", `{"type":"object"}`, func(context.Context, json.RawMessage) (string, error) { return "", errors.New("must not run") }),
		tool.New("send_email", "", `{"type":"object"}`, func(context.Context, json.RawMessage) (string, error) { sent.Add(1); return "sent", nil }),
	)
	g := &recGate{deny: map[string]bool{"rm": true}, hold: map[string]bool{"send_email": true}}
	p := &script{replies: []reply{
		{calls: []types.ToolCall{call("w", "get_weather", `{"city":"Paris"}`), call("r", "rm", `{}`)}},
		{calls: []types.ToolCall{call("e", "send_email", `{"to":"boss"}`)}},
		{text: "Done."},
	}}
	cfg := loop.Config{Provider: p, Tools: reg}
	st := loop.New(cfg, loop.WithGate(g)).Invoke(context.Background(), user)
	_, err := st.Result()
	var ae *loop.ApprovalError
	if !errors.As(err, &ae) || len(ae.Pending) != 1 || ae.Pending[0].Marker != "m-e" || sent.Load() != 0 {
		t.Fatalf("err %v, sent %d; want one pending approval m-e and nothing sent", err, sent.Load())
	}
	tr := st.Transcript()
	if denied := tr[3]; denied.ToolCallID != "r" || denied.Content != "error: denied by policy: not allowed here" || weather.Load() != 1 {
		t.Fatalf("denied call message %+v (weather ran %d)", denied, weather.Load())
	}
	if last := tr[len(tr)-1]; last.Role != types.RoleAssistant || last.ToolCalls[0].ID != "e" {
		t.Fatalf("the transcript must end with the proposing assistant message, got %+v", last)
	}
	if len(g.seen) != 3 || g.seen[0].Tainted || g.seen[1].Tainted || !g.seen[2].Tainted {
		t.Fatalf("gate call contexts %+v; want untainted, untainted, tainted", g.seen)
	}
	before := p.calls()
	msg, err := loop.New(cfg, loop.WithGate(g), loop.WithApprovals("m-e")).Invoke(context.Background(), tr).Result()
	if err != nil || msg.Content != "Done." || sent.Load() != 1 || p.calls() != before+1 {
		t.Fatalf("resume: %q, %v, sent %d, model calls %d->%d; want Done., one send, one more call", msg.Content, err, sent.Load(), before, p.calls())
	}
}

// recSteps records step names and kinds; with replay set it returns the
// recorded bytes without calling fn (what a durable runner does on restart).
type recSteps struct {
	mu     sync.Mutex
	names  []string
	kinds  []types.StepKind
	saved  map[string][]byte
	replay bool
}

func (r *recSteps) RunStep(ctx context.Context, name string, kind types.StepKind, fn func(context.Context) ([]byte, error)) ([]byte, error) {
	r.mu.Lock()
	r.names = append(r.names, name)
	r.kinds = append(r.kinds, kind)
	if b, ok := r.saved[name]; ok && r.replay {
		r.mu.Unlock()
		return b, nil
	}
	r.mu.Unlock()
	b, err := fn(ctx)
	r.mu.Lock()
	r.saved[name] = b
	r.mu.Unlock()
	return b, err
}

func TestStepRunnerNames(t *testing.T) {
	// WHY: the durable runner (ag.05) keys its journal by step name, so the
	//      names are a contract: llm/1, tool/1/0/get_weather,
	//      tool/1/1/get_weather (two calls, two entries), llm/2. A
	//      runner that replays recorded results reproduces the run with zero
	//      model calls and zero tool runs.
	// KIND: unit
	// CATCHES: s11
	// CHAPTER: ag.03 section 2.6
	var runs atomic.Int32
	p := &script{replies: []reply{
		{calls: []types.ToolCall{call("call_1", "get_weather", `{"city":"Paris"}`), call("call_2", "get_weather", `{"city":"Lyon"}`)}, usage: types.Usage{In: 10, Out: 5}},
		{text: "It is sunny.", usage: types.Usage{In: 20, Out: 4}},
	}}
	steps := &recSteps{saved: map[string][]byte{}}
	cfg := loop.Config{Provider: p, Tools: weatherTools(&runs)}
	if _, err := loop.New(cfg, loop.WithStepRunner(steps)).Invoke(context.Background(), user).Result(); err != nil {
		t.Fatal(err)
	}
	// The two tool steps run in parallel, so their record order may vary.
	names := append([]string(nil), steps.names...)
	sort.Strings(names[1:3])
	if want := []string{"llm/1", "tool/1/0/get_weather", "tool/1/1/get_weather", "llm/2"}; !reflect.DeepEqual(names, want) {
		t.Fatalf("steps %v, want %v", steps.names, want)
	}
	if want := []types.StepKind{types.StepLLM, types.StepTool, types.StepTool, types.StepLLM}; !reflect.DeepEqual(steps.kinds, want) {
		t.Fatalf("kinds %v, want %v", steps.kinds, want)
	}
	steps.replay = true
	msg, err := loop.New(cfg, loop.WithStepRunner(steps)).Invoke(context.Background(), user).Result()
	if err != nil || msg.Content != "It is sunny." || p.calls() != 2 || runs.Load() != 2 {
		t.Fatalf("replay: %q, %v, model calls %d, tool runs %d; want the same answer and no new calls", msg.Content, err, p.calls(), runs.Load())
	}
}

// failSteps fails every tool step with err (a durable runner that cannot
// replay a write, for example) and runs model steps.
type failSteps struct{ err error }

func (f failSteps) RunStep(ctx context.Context, name string, kind types.StepKind, fn func(context.Context) ([]byte, error)) ([]byte, error) {
	if kind == types.StepTool {
		return nil, f.err
	}
	return fn(ctx)
}

func TestStepRunnerErrorEndsRun(t *testing.T) {
	// WHY: a tool that fails is data for the model (an error result); a
	//      step that could not run at all (the durable journal refuses to
	//      replay a write whose outcome it never recorded, ag.05) is not,
	//      and must end the run with that error instead of being shown to
	//      the model as "error: ..." and papered over.
	// KIND: fault
	// CATCHES: s17
	// CHAPTER: ag.03 section 2.6
	sentinel := errors.New("journal: indeterminate write")
	var runs atomic.Int32
	p := &script{replies: []reply{
		{calls: []types.ToolCall{call("c", "get_weather", `{"city":"Paris"}`)}},
		{text: "should not be reached"},
	}}
	st := loop.New(loop.Config{Provider: p, Tools: weatherTools(&runs)}, loop.WithStepRunner(failSteps{sentinel})).Invoke(context.Background(), user)
	_, err := st.Result()
	if !errors.Is(err, sentinel) || p.calls() != 1 {
		t.Fatalf("err %v after %d model calls; want the step error after 1", err, p.calls())
	}
	if tr := st.Transcript(); tr[len(tr)-1].Role != types.RoleAssistant {
		t.Fatalf("no tool message may be appended for a step that did not run: %+v", tr[len(tr)-1])
	}
}

func TestSpans(t *testing.T) {
	// WHY: the trace is how an operator reads a run (otel/semconv.md): one
	//      agent.run with the iteration count, an agent.llm_call per model
	//      call with token counts, an agent.tool <name> per call with its
	//      outcome (ok, error, denied, needs_approval).
	// KIND: unit
	// CATCHES: s15
	// CHAPTER: ag.03 section 2.7
	rec := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec))
	var runs atomic.Int32
	reg := weatherTools(&runs).MustRegister(
		tool.New("rm", "", `{"type":"object"}`, func(context.Context, json.RawMessage) (string, error) { return "", nil }),
		tool.New("flaky", "", `{"type":"object"}`, func(context.Context, json.RawMessage) (string, error) { return "", errors.New("timeout") }),
	)
	p := &script{replies: []reply{
		{calls: []types.ToolCall{call("w", "get_weather", `{"city":"Paris"}`), call("r", "rm", `{}`), call("f", "flaky", `{}`)}, usage: types.Usage{In: 10, Out: 5}},
		{text: "ok", usage: types.Usage{In: 20, Out: 4}},
	}}
	g := &recGate{deny: map[string]bool{"rm": true}}
	if _, err := loop.New(loop.Config{Provider: p, Tools: reg, Name: "helper"}, loop.WithGate(g), loop.WithTracer(tp.Tracer("ag03"))).Invoke(context.Background(), user).Result(); err != nil {
		t.Fatal(err)
	}
	attrs := map[string]map[string]string{}
	var llm int
	var root string
	for _, s := range rec.Ended() {
		m := map[string]string{}
		for _, kv := range s.Attributes() {
			m[string(kv.Key)] = kv.Value.Emit()
		}
		switch {
		case s.Name() == "agent.run":
			root = s.SpanContext().TraceID().String()
			attrs["run"] = m
		case s.Name() == "agent.llm_call":
			llm++
			attrs[fmt.Sprintf("llm%d", llm)] = m
		default:
			attrs[s.Name()] = m
		}
	}
	if attrs["run"]["gen_ai.agent.name"] != "helper" || attrs["run"]["tl.agent.iterations"] != "2" || root == "" {
		t.Fatalf("agent.run attributes %v", attrs["run"])
	}
	if llm != 2 || attrs["llm1"]["gen_ai.usage.input_tokens"] != "10" || attrs["llm2"]["gen_ai.usage.output_tokens"] != "4" {
		t.Fatalf("llm spans %d: %v %v", llm, attrs["llm1"], attrs["llm2"])
	}
	if attrs["agent.tool get_weather"]["tl.tool.outcome"] != "ok" || attrs["agent.tool rm"]["tl.tool.outcome"] != "denied" ||
		attrs["agent.tool flaky"]["tl.tool.outcome"] != "error" {
		t.Fatalf("tool spans %v", attrs)
	}
}

func TestProviderErrorEndsRun(t *testing.T) {
	// WHY: a model that cannot be reached ends the run with that error
	//      (wrapped with the step), not a silent empty answer, and no tool
	//      runs.
	// KIND: fault
	// CATCHES: s16
	// CHAPTER: ag.03 section 5
	boom := errors.New("connection refused")
	var runs atomic.Int32
	p := &script{replies: []reply{{err: boom}}}
	_, err := loop.New(loop.Config{Provider: p, Tools: weatherTools(&runs)}).Invoke(context.Background(), user).Result()
	if !errors.Is(err, boom) || runs.Load() != 0 {
		t.Fatalf("err %v, %d runs; want the provider's error and no run", err, runs.Load())
	}
}

func TestEventsOptional(t *testing.T) {
	// WHY: a caller that only wants the answer calls Result and never reads
	//      Events; a long streamed answer must not block the run on a full
	//      event channel.
	// KIND: regression
	// CHAPTER: ag.03 section 4
	p := &script{replies: []reply{{text: strings.Repeat("token ", 2000)}}}
	done := make(chan struct{})
	go func() {
		defer close(done)
		loop.New(loop.Config{Provider: p}).Invoke(context.Background(), user).Result()
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Result blocked: the run waits on an event channel nobody reads")
	}
}

func TestFaketoolEndToEnd(t *testing.T) {
	// WHY: the loop over the real provider (ag.01) and the course's fake
	//      model: the tool call streams in fragments, the registry runs it,
	//      the tool message goes back over HTTP, and the fake answers from
	//      the tool's output.
	// KIND: conformance
	// CHAPTER: ag.03 section 6
	srv, err := faketool.Start(&faketool.Rules{Rules: []faketool.Rule{
		{When: faketool.When{Role: "user", Contains: "weather"}, Reply: faketool.Reply{ToolCalls: []faketool.ToolCall{{Name: "get_weather", Arguments: map[string]any{"city": "Paris"}}}}},
		{When: faketool.When{Role: "tool", Contains: "sunny"}, Reply: faketool.Reply{Content: "It is sunny in Paris."}},
	}}, "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()
	var runs atomic.Int32
	prov := provider.NewOpenAI(provider.Config{BaseURL: srv.BaseURL(), Model: "faketool"})
	msg, err := loop.New(loop.Config{Provider: prov, Tools: weatherTools(&runs)}).Invoke(context.Background(), user).Result()
	if err != nil || msg.Content != "It is sunny in Paris." || runs.Load() != 1 {
		t.Fatalf("%q, %v, %d runs", msg.Content, err, runs.Load())
	}
	tools := srv.Requests()[0]["tools"].([]any)
	if len(tools) != 1 || tools[0].(map[string]any)["function"].(map[string]any)["name"] != "get_weather" {
		t.Fatalf("the model was offered %v", tools)
	}
}
