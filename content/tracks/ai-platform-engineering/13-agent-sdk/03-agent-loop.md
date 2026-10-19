# Agent loop

## Overview

| | |
|---|---|
| **Module** | `ag.03` · build · Go · Pass 10 · 4 h |
| **Solution** | [`loop.go`](../../../../course/solutions/go/agent/loop/loop.go): `New`, the options (`WithMaxIter`, `WithMaxParallelTools`, `WithBudget`, `WithGate`, `WithStepRunner`, `WithTracer`, `WithApprovals`), `Agent.Invoke`, `Stream` (`Events`, `Result`, `Transcript`, `Usage`), `ErrMaxIter`, `BudgetError`, `ApprovalError`; the internals `run`, `callModel`, `callTool` |
| **Contract** | the agent spans of [`course/contracts/otel/semconv.md`](../../../../course/contracts/otel/semconv.md) (`agent.run`, `agent.llm_call`, `agent.tool <name>`); the Go API is section 4 |
| **Needs** | [`ag.01`](01-types-and-provider.md) (`Provider`, `Accumulator`, the `Gate` and `StepRunner` seams), [`ag.02`](02-tools-and-schemas.md) (`Registry`) · reading: tracing in Go (`obs.01`) |
| **Used by** | `ag.04` the prompt-injection suite drives this loop with the gate · `ag.05` runs it durably · `ag.09` evaluates it as a subject · the agent track's end-to-end run is this loop over the gateway |
| **Optional depth** | Anthropic, [*Building effective agents*](https://www.anthropic.com/engineering/building-effective-agents) (free); Yao et al., [*ReAct*](https://arxiv.org/abs/2210.03629) (free) |

## Key Takeaways

- An agent is a **loop**: call the model with the messages and tool definitions, run the calls it returns, append the results, repeat until it answers without a call.
- Every limit is checked **before** the work it limits: the iteration cap before a model call, the tool-call and token budgets before dispatch.
- Calls of one turn run **in parallel up to a bound**, and their results go back **in call order**.
- The gate decides before dispatch: denied calls never run and the model reads why; calls that need approval end the run with a marker, and a resumed run dispatches them **without asking the model again**.
- Every model and tool call is a **named step** through a `StepRunner`, which is what makes the loop durable in `ag.05`.

## How to read this chapter

Sections 1 to 3 build the loop from first principles and run one conversation by hand. Section 4 fixes the interface. Section 5 walks through the solution in `loop.go` and explains why each choice is the right one; try to predict each step before reading it.

---

## 1. Why now

There is now a model client (`ag.01`) and a tool registry (`ag.02`), and no program that connects them. Without one, "the agent" is a person copying tool calls into a shell. The loop is short, which is the danger: the obvious version runs forever when the model keeps calling tools, runs ten slow calls one after another, spends past its budget before checking it, and has no place to put a safety gate or a durable journal. The solution writes the loop once, with those places built in.

## 2. Principles

### 2.1 The loop

| Symbol | Meaning |
|---|---|
| $M_t$ | the messages after iteration $t$ ($M_0$ is the input, with the system prompt first) |
| $a_t$ | the assistant message of iteration $t$ |
| $C_t$ | the tool calls in $a_t$, in the order the model wrote them |
| $r_{t,i}$ | the tool message answering call $i$ of $C_t$ |

$$M_t = M_{t-1} \,\Vert\, a_t \,\Vert\, r_{t,0} \,\Vert\, \dots \,\Vert\, r_{t,|C_t|-1}$$

and the run ends at the first $t$ with $C_t$ empty: $a_t$ is the answer. `Config.System` is the first message unless the input already starts with a system message (a resumed transcript has one).

### 2.2 The iteration cap

Each iteration is one model call. `WithMaxIter(n)` (default 8) allows $n$ model calls per run; before call $n+1$ the run ends with `ErrMaxIter` and the last assistant message. A model stuck calling the same tool cannot burn money forever.

### 2.3 Parallel calls, ordered results

A model may return several calls in one turn ("weather in Paris" and "weather in Lyon"). They are independent, so they run concurrently, at most `WithMaxParallelTools(k)` (default 4) at once: a semaphore of $k$ slots, each call waits for a slot. Results are written to slot $i$ of a result array and appended in index order, whatever order the calls finish in. The model matches results to calls by id, but providers also check that the tool messages follow the assistant message in its call order.

### 2.4 Budgets

`WithBudget(Budget{MaxTokens, MaxToolCalls})` caps one run. A budget is a promise about spend, so it is checked before the spend: the tool-call budget before dispatching a turn (three calls against a budget of two run none of them), the token budget after each model call returns its usage (a reply that took the run past the budget dispatches nothing) and before the next model call. Usage is summed over every model call.

### 2.5 The gate and approvals

Before dispatching a turn, the loop asks the gate (`ag.04`) about every call, attaching `CallContext{Tainted, Iter}`. **Tainted** becomes true once any tool message is in the context: tool output is data from outside the program (a web page, a retrieved chunk, a database row), and data can contain instructions an attacker wrote. The verdicts:

| Verdict | Loop does |
|---|---|
| `Allow` | runs the call |
| `Deny{Reason}` | does not run it; its tool message is `error: denied by policy: <reason>` |
| `NeedApproval{Marker}` | runs nothing of this turn; the run ends with `*ApprovalError` listing the pending calls and markers |

The transcript then ends with the assistant message that proposed the calls. Invoking again with that transcript and `WithApprovals(marker)` sees an unanswered assistant message, skips the model call, and dispatches the calls (an approved marker turns `NeedApproval` into `Allow`). The model is not asked again, so it cannot propose a different call in place of the approved one.

### 2.6 Steps

Every model call runs as step `llm/<iter>` and every tool call as `tool/<iter>/<i>/<name>` through the `StepRunner` (`types.Direct` by default). A tool that fails is a `Result` with `IsError`, which the model reads. A step that could not run at all (the runner failed: in `ag.05`, a journal that refuses to replay a write whose outcome it never recorded) ends the run with that error; showing it to the model as `error: ...` would let the run go on as if the write had failed.

### 2.7 Spans

Per `otel/semconv.md`: `agent.run` (INTERNAL, `gen_ai.agent.name`, `tl.agent.iterations`), one `agent.llm_call` (CLIENT, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`) per model call, and one `agent.tool <name>` (INTERNAL, `gen_ai.tool.name`, `tl.tool.outcome` = `ok`, `error`, `denied`, or `needs_approval`) per call. The loop takes an OpenTelemetry `trace.Tracer`; the composition root passes the one `go/otelx` sets up (`obs.01`).

## 3. Worked example by hand

Input: one user message, "What is the weather in Paris?". The tool `get_weather` answers "sunny, 21 C".

| Step | Messages before | Model returns | Loop does | Usage so far |
|---|---|---|---|---|
| `llm/1` | user | call `call_1` = `get_weather({"city":"Paris"})`, 10 in, 5 out | gate (none: allow), run the call as `tool/1/0/get_weather` | 10 / 5 |
| | user, assistant(call_1), tool(call_1: "sunny, 21 C") | | | |
| `llm/2` | the three above | "It is sunny.", 20 in, 4 out | no calls: the answer | 30 / 9 |

Two model calls, one tool run, a transcript of four messages (user, assistant with the call, the tool message answering `call_1`, the answer), and usage 30 in, 9 out. The events, in order: usage of call 1, the tool call, its result, the three streamed text pieces "It ", "is ", "sunny.", usage of call 2, done.

## 4. The interface

```go
package loop // import "tinyllm/agent/loop"

type Config struct { Provider types.Provider; Tools *tool.Registry; System, Name string; CallOptions []types.CallOption }
type Budget struct{ MaxTokens, MaxToolCalls int }
func New(cfg Config, opts ...Option) *Agent
func WithMaxIter(n int) Option
func WithMaxParallelTools(n int) Option
func WithBudget(b Budget) Option
func WithGate(g types.Gate) Option
func WithStepRunner(s types.StepRunner) Option
func WithTracer(t trace.Tracer) Option
func WithApprovals(markers ...string) Option

func (a *Agent) Invoke(ctx context.Context, msgs []types.Message) *Stream
func (s *Stream) Events() <-chan Event       // optional to read; closed at the end
func (s *Stream) Result() (types.Message, error)
func (s *Stream) Transcript() []types.Message
func (s *Stream) Usage() types.Usage
type Event struct { Kind EventKind; Iter int; Text string; Call types.ToolCall; Verdict types.Verdict; Result tool.Result; Usage types.Usage; Err error }

var ErrMaxIter error
type BudgetError struct { Resource string; Used, Limit int }
type ApprovalError struct{ Pending []Pending } // Pending{Call, Marker, Reason}
```

Allowed imports: the standard library, `go.opentelemetry.io/otel` (`attribute`, `codes`, `trace`, `trace/noop`), and the `ag.01` and `ag.02` packages. `Result` must not block on unread events: a caller that only wants the answer never reads `Events`.

### What a correct solution must handle

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| the hand example (section 3) | unit | the answer, 2 model calls, 1 tool run, usage 30/9, the transcript, what the second call saw, the event order | the loop is pinned down |
| the system prompt | unit | added once, and not to a transcript that already has one | resumed runs do not stack instructions |
| `WithMaxIter(3)` with a model that never stops | boundary | 3 model calls, 3 tool runs, `ErrMaxIter` with the last message | a looping model is stopped |
| 6 calls with bound 2 | unit | exactly 2 in flight; finished in reverse, returned in order | throughput without breaking the call order |
| budgets | boundary | 3 calls against 2: none run; 110 tokens against 100: no dispatch | spend is checked before it happens |
| gate verdicts | unit | deny, approval, resume without a model call, tainted only after tool output | `ag.04` plugs in here |
| step names | unit | names and kinds; a replaying runner reproduces the run with no calls | `ag.05` keys its journal by these names |
| a step runner error | fault | ends the run, nothing appended | an unknown write outcome is never papered over |
| spans | unit | `agent.run`, two `agent.llm_call` with tokens, `agent.tool` outcomes ok, error, denied | operators read runs from traces |
| a provider error | fault | ends the run, no tool runs | no silent empty answers |
| 2000 unread events | regression | `Result` without reading `Events` does not block | callers that want only the answer |
| the fake tool-calling model end to end | conformance | over the real provider: call, tool, answer | the PR stand-in for a real model |

## 5. The solution, step by step

### 5.1 From first principles

1. **The loop's state is the message list.** Everything the model knows is in $M_t$, so the loop only appends: the assistant turn, then one tool message per call. Because the transcript is the state, a run can be resumed by passing the transcript back in.
2. **A limit protects only if it is checked before the work.** The iteration cap is checked before each model call; the token budget before each model call and again after a reply that asks for tools; the tool-call budget before dispatching a turn. A check after the work only reports the overspend.
3. **One turn's calls are independent; their order on the wire is not.** So they run concurrently with a semaphore of $k$ slots, each writes its result into slot $i$ of a preallocated array, and the loop appends the array in index order after `wg.Wait()`.
4. **The gate sees every call before any runs.** If any call of a turn needs approval, none of that turn runs: a partially executed turn could not be resumed cleanly.
5. **Resuming means "the last message is an assistant turn with unanswered calls".** That shape cannot occur in the middle of a normal run (results are always appended), so it identifies a resume without extra state, and the loop skips the model call so the approved calls are exactly the ones a human saw.
6. **Taint is monotone.** Once any tool output is in the context, it stays there, so `tainted` only ever goes from false to true, and a resumed transcript with tool messages starts tainted.
7. **Every side effect is a named step.** Model calls and tool calls go through `StepRunner.RunStep` with deterministic names and JSON results, which is all `ag.05` needs to record and replay a run.
8. **Delivery of events must never block the run.** Events are optional; `Result` closes a `discard` channel so `emit` stops waiting.

### 5.2 The code, annotated

**Setup** in `run`: the system prompt goes first unless the input already has one, and the iteration count and taint are recomputed from the input, so a resumed transcript continues where it left off:

```go
msgs := append([]types.Message(nil), input...) // never mutate the caller's slice
if a.cfg.System != "" && (len(msgs) == 0 || msgs[0].Role != types.RoleSystem) {
	msgs = append([]types.Message{{Role: types.RoleSystem, Content: a.cfg.System}}, msgs...)
}
iter, tainted := 0, false
for _, m := range msgs {
	switch m.Role {
	case types.RoleAssistant:
		iter++ // each assistant turn was one model call
	case types.RoleTool:
		tainted = true // step 6
	}
}
```

**The model call and its limits**, steps 2 and 5:

```go
for {
	var reply types.Message
	if n := len(msgs); n > 0 && msgs[n-1].Role == types.RoleAssistant && len(msgs[n-1].ToolCalls) > 0 {
		reply = msgs[n-1] // resume: the calls were proposed but never answered
	} else {
		if iter >= a.maxIter {
			return finish(last, ErrMaxIter) // before call maxIter+1, not after it
		}
		if b := a.budget.MaxTokens; b > 0 && s.usage.In+s.usage.Out >= b {
			return finish(last, &BudgetError{"tokens", s.usage.In + s.usage.Out, b})
		}
		iter++
		rec, err := a.callModel(ctx, s, iter, msgs, defs)
		if err != nil {
			return finish(last, fmt.Errorf("loop: model call %d: %w", iter, err))
		}
		s.usage.In += rec.Usage.In // summed over every call, never overwritten
		s.usage.Out += rec.Usage.Out
		// ... EventUsage
		reply = rec.Message
		msgs = append(msgs, reply)
		last = reply
		if len(reply.ToolCalls) == 0 {
			return finish(reply, nil) // the answer
		}
		if b := a.budget.MaxTokens; b > 0 && s.usage.In+s.usage.Out > b {
			return finish(last, &BudgetError{"tokens", s.usage.In + s.usage.Out, b}) // this reply overspent: dispatch nothing
		}
	}
	last = reply
	calls := reply.ToolCalls
	if b := a.budget.MaxToolCalls; b > 0 && toolCalls+len(calls) > b {
		return finish(last, &BudgetError{"tool_calls", toolCalls + len(calls), b}) // none of the turn runs
	}
	// ... gate, dispatch, append (below)
}
```

`finish` stores the transcript and the span's iteration count for every exit, so `Transcript` always ends with the assistant turn that proposed any pending calls.

**The gate**, step 4. Every call is checked with the current `CallContext`; a gate error is treated as a deny; an approved marker turns `NeedApproval` into `Allow`:

```go
gctx := types.WithCallContext(ctx, types.CallContext{Tainted: tainted, Iter: iter})
for i, c := range calls {
	v := types.Verdict{Kind: types.Allow}
	if a.gate != nil {
		var err error
		if v, err = a.gate.Check(gctx, c); err != nil {
			v = types.Verdict{Kind: types.Deny, Reason: "gate error: " + err.Error()} // fail closed
		}
	}
	if v.Kind == types.NeedApproval && a.approved[v.Marker] { // keyed by the gate's marker
		v.Kind = types.Allow
	}
	verdicts[i] = v
	// ... EventToolCall; collect NeedApproval as Pending with its marker; span "needs_approval"
}
if len(pending) > 0 {
	return finish(last, &ApprovalError{Pending: pending}) // nothing of this turn runs
}
```

**Dispatch**, step 3. Denied calls get their result directly; the rest run in goroutines behind a semaphore:

```go
results := make([]tool.Result, len(calls)) // slot i answers call i
errs := make([]error, len(calls))
sem := make(chan struct{}, a.maxPar)
var wg sync.WaitGroup
for i, c := range calls {
	if verdicts[i].Kind == types.Deny {
		results[i] = tool.Result{CallID: c.ID, Name: c.Name, Content: "denied by policy: " + verdicts[i].Reason, IsError: true}
		a.toolSpan(ctx, c, "denied")
		continue // never runs; the model reads why
	}
	wg.Add(1)
	go func(i int, c types.ToolCall) {
		defer wg.Done()
		sem <- struct{}{} // at most maxPar in flight
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
	msgs = append(msgs, results[i].Message()) // in call order, whatever order they finished in
	// ... EventToolResult
}
tainted = true // tool output is data from outside the program
if err := ctx.Err(); err != nil {
	return finish(last, err)
}
```

Each goroutine writes only its own slot, so the arrays need no lock; `wg.Wait()` is the synchronization point that makes the writes visible.

**Steps**, step 7. `callModel` wraps the provider call in `RunStep(ctx, types.LLMStep(iter), types.StepLLM, ...)`; inside, it folds the stream with an `Accumulator`, forwards each `TextDelta` as an `EventText`, and returns the message and usage as JSON. `callTool` does the same for `Registry.Call` under `types.ToolStep(iter, i, c.Name)`. Two things make replay work:

```go
res.CallID, res.Name = c.ID, c.Name // a recorded result answers this call
outcome := "ok"
if res.IsError {
	outcome = "error" // tl.tool.outcome: failures visible in traces
	span.SetStatus(codes.Error, res.Content)
}
```

and the step name includes the call index `i`, so two calls of the same tool in one turn get two journal entries.

**Optional events**, step 8:

```go
func (s *Stream) emit(ctx context.Context, e Event) {
	select {
	case s.events <- e: // a reader is keeping up (16-slot buffer)
	case <-s.discard: // Result was called: nobody will read
	case <-ctx.Done():
	}
}

func (s *Stream) Result() (types.Message, error) {
	s.once.Do(func() { close(s.discard) })
	<-s.done
	return s.msg, s.err
}
```

Full solution: [`loop.go`](../../../../course/solutions/go/agent/loop/loop.go).

### 5.3 Why this solution

- **Cost.** Per iteration: one model call plus the turn's tool calls, wall time about the slowest $\lceil |C_t|/k \rceil$ batches of tools. Memory is the transcript. The loop's own work (gate checks, appends) is negligible next to the model.
- **The transcript as the only state** rather than a separate run object: resume, approval, and durability (`ag.05`) all fall out of "pass the transcript back in", and there is nothing else to persist.
- **Bounded parallelism** rather than sequential or unbounded: sequential calls make a turn as slow as the sum of its tools; unbounded calls let one turn of ten slow tools overload a backend. A semaphore gives both a speedup and a ceiling.
- **Approval ends the run** rather than blocking inside it: a human may answer in minutes or days, and a blocked goroutine holding a connection is the wrong place to wait. The run returns `ApprovalError` with markers; the caller persists the transcript and invokes again.
- **Tradeoffs accepted:** a token budget can be exceeded by at most one model reply (the loop cannot know a reply's size in advance), but it never dispatches tools after that; denied calls still count against the tool-call budget, since the model proposed them.

### 5.4 Explain it back

1. In section 3, why does the second model call see three messages while the transcript ends with four?
   <details><summary>Answer</summary>The second call is made with $M_1$ = user, assistant(call_1), tool(call_1). Its reply, "It is sunny.", is appended afterwards, making four messages. The model never sees its own reply; it sees the tool's result answering its call.</details>
2. Why does the loop check `iter >= maxIter` before the model call instead of counting after it?
   <details><summary>Answer</summary>A check after the call would allow call number `maxIter + 1` to happen and only then stop. The cap is a promise about how many model calls a run may make, so it must gate the call itself.</details>
3. Three calls finish in the order 2, 0, 1. In what order are their tool messages appended, and why does it matter?
   <details><summary>Answer</summary>0, 1, 2: each goroutine writes slot `i` and the loop appends the slots in index order after `wg.Wait()`. Providers check that tool messages follow the assistant message in its call order, and a deterministic order keeps the prompt (and a durable replay) reproducible.</details>
4. How does a resumed run know not to call the model again, without a flag?
   <details><summary>Answer</summary>The transcript ends with an assistant message that has tool calls and no tool messages after it. A normal run always appends the results right after the proposing turn, so that shape occurs only when a run stopped for approval. The loop uses that message as the reply and dispatches its calls.</details>
5. Why does a `StepRunner` error end the run instead of becoming an `error:` tool message?
   <details><summary>Answer</summary>A tool error means the tool ran and failed; the model can react. A step error means the step could not run or its outcome is unknown (in `ag.05`, a write whose result was never recorded). Telling the model "error" would let it proceed as if the write had failed when it may have succeeded.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. the cap checked after the call | one model call more than configured | `iter >= a.maxIter` is checked before `callModel` |
| 2. unbounded parallel calls, results appended as they finish | ten slow tools at once overload a backend; results misordered | a semaphore of `maxPar` slots; results by index, appended after `wg.Wait()` |
| 3. budgets checked after the spend, usage overwritten instead of summed | tools run past the budget; the budget never trips | token and tool-call checks before dispatch; `s.usage.In +=` |
| 4. verdicts ignored, approvals looked up by the wrong key, a resumed run asking the model again | denied tools run; approved calls stay pending or are replaced | denied calls never reach `callTool`; `approved[v.Marker]`; the resume branch skips `callModel` |
| 5. the context never marked tainted | a gate cannot tell a clean request from one after a poisoned document | `tainted = true` after every turn's results, and from any tool message in the input |
| 6. the system prompt added twice, the proposing assistant turn dropped | stacked instructions; tool messages with no call to answer | the system prompt only when the first message is not a system message; the reply is appended before any check |
| 7. tool step names without the call index, every outcome recorded as ok | two calls of one tool share a journal entry; failures invisible in traces | `ToolStep(iter, i, name)`; `tl.tool.outcome` set from `IsError` |
| 8. errors swallowed | a dead model gives an empty "answer"; a refused replay is shown to the model | provider and step errors return through `finish` with the error |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `ag.01` | the provider, the accumulator, the gate and step seams |
| Back | `ag.02` | `Definitions` for every model call, `Call` for every tool call |
| Forward | `ag.04` | the prompt-injection suite runs this loop with the policy gate |
| Forward | `ag.05` | `AgentRun` passes its journal as the `StepRunner` and resumes from the recorded input |
| Forward | `ag.09` | `AgentSubject` runs the loop on every case of a suite and reads its events |

`obs.01` is reading because the loop takes any `trace.Tracer`.

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| the loop | saige `agent` | a conversation tree with branching and compaction, sub-agents as call and return, handoff | saige `agent/agent.go`, `agent/tree/compact.go` |
| `WithGate` | saige tool policy | separate policies for disclosure, execution, context, results, routing | saige `agent/tool_policy.go` |
| `WithBudget` | saige shared `Budget` | request, token, and cost capacity reserved under one lock and settled once | saige `agent/budget` |
| the pattern | ReAct, plan-and-execute | reasoning traces interleaved with actions; planning before acting | Yao et al., [*ReAct*](https://arxiv.org/abs/2210.03629) (free) |
