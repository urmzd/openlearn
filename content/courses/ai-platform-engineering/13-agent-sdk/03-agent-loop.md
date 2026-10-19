# The agent loop

There is now a model client ([Types, an OpenAI-compatible provider, and a retry wrapper](01-types-and-provider.md)) and a tool registry ([Tools, a registry, and JSON Schema argument validation](02-tools-and-schemas.md)), and no program that connects them. Without one, "the agent" is a person copying tool calls into a shell. An agent is a **loop**: call the model with the messages and tool definitions, run the calls it returns, append the results, and repeat until it answers without a call.

The loop is short, which is the danger. The obvious version runs forever when the model keeps calling tools, runs ten slow calls one after another, spends past its budget before checking it, and has no place to put a safety gate or a durable journal. This section writes the loop once, with those places built in: an iteration cap, bounded parallel dispatch with ordered results, budgets checked before the spend, a gate with human approval, named steps for durability, and tracing spans.

Besides the two previous sections, you will want tracing in Go from [Tracing for the serving path](../../systems/04-observability/01-tracing-for-the-serving-path.md) in *Systems*; the loop takes any OpenTelemetry tracer, so that is background reading rather than a code dependency. The spans follow the agent spans of [`course/contracts/otel/semconv.md`](../../../../course/contracts/otel/semconv.md) (`agent.run`, `agent.llm_call`, `agent.tool <name>`). The full solution is [`loop.go`](../../../../course/solutions/go/agent/loop/loop.go): `New`, the options (`WithMaxIter`, `WithMaxParallelTools`, `WithBudget`, `WithGate`, `WithStepRunner`, `WithTracer`, `WithApprovals`), `Agent.Invoke`, `Stream` (`Events`, `Result`, `Transcript`, `Usage`), `ErrMaxIter`, `BudgetError`, `ApprovalError`, and the internals `run`, `callModel`, and `callTool`.

> **Learning objectives**
> - Write the loop's recurrence over messages and trace a two-iteration run by hand, including usage and events.
> - Explain why every limit is checked before the work it limits.
> - Run a turn's tool calls in parallel up to a bound and return their results in call order.
> - Describe how the gate's verdicts, taint, and approval markers change what the loop does, and how a resumed run avoids asking the model again.
> - Explain why every model and tool call is a named step.

## The loop as a recurrence

Name the pieces of state first:

| Symbol | Meaning |
|---|---|
| $M_t$ | the messages after iteration $t$ ($M_0$ is the input, with the system prompt first) |
| $a_t$ | the assistant message of iteration $t$ |
| $C_t$ | the tool calls in $a_t$, in the order the model wrote them |
| $r_{t,i}$ | the tool message answering call $i$ of $C_t$ |

Each iteration appends the assistant turn and one tool message per call:

$$M_t = M_{t-1} \,\Vert\, a_t \,\Vert\, r_{t,0} \,\Vert\, \dots \,\Vert\, r_{t,|C_t|-1}$$

The run ends at the first $t$ with $C_t$ empty, and $a_t$ is the answer. `Config.System` is the first message unless the input already starts with a system message, as a resumed transcript does. Everything else in this section is about the conditions under which an iteration is allowed to happen and how its calls are dispatched.

## The iteration cap

Each iteration is one model call. `WithMaxIter(n)` (default 8) allows $n$ model calls per run; before call $n+1$ the run ends with `ErrMaxIter` and the last assistant message. A model stuck calling the same tool cannot burn money forever.

## Parallel calls, ordered results

A model may return several calls in one turn ("weather in Paris" and "weather in Lyon"). They are independent, so they run concurrently, at most `WithMaxParallelTools(k)` (default 4) at once: a semaphore of $k$ slots, where each call waits for a slot. Results are written to slot $i$ of a result array and appended in index order, whatever order the calls finish in. The model matches results to calls by id, but providers also check that the tool messages follow the assistant message in its call order, so ordering is not optional.

## Budgets

`WithBudget(Budget{MaxTokens, MaxToolCalls})` caps one run. A budget is a promise about spend, so it is checked before the spend. The tool-call budget is checked before dispatching a turn: three calls against a budget of two run none of them. The token budget is checked after each model call returns its usage (a reply that took the run past the budget dispatches nothing) and again before the next model call. Usage is summed over every model call. The general rule is that every limit is checked **before** the work it limits.

## The gate and approvals

Before dispatching a turn, the loop asks the gate ([Tool gate, SQL safety gate, and prompt-injection suite](04-tool-gate-and-injection.md)) about every call, attaching `CallContext{Tainted, Iter}`. **Tainted** becomes true once any tool message is in the context. Tool output is data from outside the program (a web page, a retrieved chunk, a database row), and data can contain instructions an attacker wrote. The loop acts on each verdict as follows:

| Verdict | Loop does |
|---|---|
| `Allow` | runs the call |
| `Deny{Reason}` | does not run it; its tool message is `error: denied by policy: <reason>` |
| `NeedApproval{Marker}` | runs nothing of this turn; the run ends with `*ApprovalError` listing the pending calls and markers |

After an approval stop, the transcript ends with the assistant message that proposed the calls. Invoking again with that transcript and `WithApprovals(marker)` sees an unanswered assistant message, skips the model call, and dispatches the calls (an approved marker turns `NeedApproval` into `Allow`). The model is not asked again, so it cannot propose a different call in place of the approved one.

## Steps and spans

Every model call runs as step `llm/<iter>` and every tool call as `tool/<iter>/<i>/<name>` through the `StepRunner` (`types.Direct` by default). Naming every side effect is what makes the loop durable in [Durable agent runs](05-durable-agent-runs.md). A tool that fails is a `Result` with `IsError`, which the model reads. A step that could not run at all is different: when the runner itself fails (in the durable runner, a journal that refuses to replay a write whose outcome it never recorded), the run ends with that error. Showing it to the model as `error: ...` would let the run go on as if the write had failed.

The loop also emits spans per `otel/semconv.md`: `agent.run` (INTERNAL, with `gen_ai.agent.name` and `tl.agent.iterations`), one `agent.llm_call` (CLIENT, with `gen_ai.request.model`, `gen_ai.usage.input_tokens`, and `gen_ai.usage.output_tokens`) per model call, and one `agent.tool <name>` (INTERNAL, with `gen_ai.tool.name` and `tl.tool.outcome` set to `ok`, `error`, `denied`, or `needs_approval`) per call. The loop takes an OpenTelemetry `trace.Tracer`; the composition root passes the one that `go/otelx` sets up.

## Running a conversation by hand

Take one user message, "What is the weather in Paris?", and a tool `get_weather` that answers "sunny, 21 C". The run goes like this:

| Step | Messages before | Model returns | Loop does | Usage so far |
|---|---|---|---|---|
| `llm/1` | user | call `call_1` = `get_weather({"city":"Paris"})`, 10 in, 5 out | gate (none: allow), run the call as `tool/1/0/get_weather` | 10 / 5 |
| | user, assistant(call_1), tool(call_1: "sunny, 21 C") | | | |
| `llm/2` | the three above | "It is sunny.", 20 in, 4 out | no calls: the answer | 30 / 9 |

That is two model calls and one tool run, producing a transcript of four messages (user, assistant with the call, the tool message answering `call_1`, and the answer) and usage of 30 in, 9 out. The events, in order, are: usage of call 1, the tool call, its result, the three streamed text pieces "It ", "is ", "sunny.", usage of call 2, and done.

## Designing the API

The package exposes a constructor with options and a `Stream` handle:

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

The allowed imports are the standard library, `go.opentelemetry.io/otel` (`attribute`, `codes`, `trace`, `trace/noop`), and the `types`, `provider`, and `tool` packages from the previous two sections. One rule shapes the `Stream`: `Result` must not block on unread events, because a caller that only wants the answer never reads `Events`.

## Building it step by step

Here is the reasoning chain; try to predict each step before reading it.

1. **The loop's state is the message list.** Everything the model knows is in $M_t$, so the loop only appends: the assistant turn, then one tool message per call. Because the transcript is the state, a run can be resumed by passing the transcript back in.
2. **A limit protects only if it is checked before the work.** The iteration cap is checked before each model call; the token budget before each model call and again after a reply that asks for tools; the tool-call budget before dispatching a turn. A check after the work only reports the overspend.
3. **One turn's calls are independent; their order on the wire is not.** So they run concurrently with a semaphore of $k$ slots, each writes its result into slot $i$ of a preallocated array, and the loop appends the array in index order after `wg.Wait()`.
4. **The gate sees every call before any runs.** If any call of a turn needs approval, none of that turn runs: a partially executed turn could not be resumed cleanly.
5. **Resuming means "the last message is an assistant turn with unanswered calls".** That shape cannot occur in the middle of a normal run (results are always appended), so it identifies a resume without extra state, and the loop skips the model call so the approved calls are exactly the ones a human saw.
6. **Taint is monotone.** Once any tool output is in the context, it stays there, so `tainted` only ever goes from false to true, and a resumed transcript with tool messages starts tainted.
7. **Every side effect is a named step.** Model calls and tool calls go through `StepRunner.RunStep` with deterministic names and JSON results, which is all the durable runner needs to record and replay a run.
8. **Delivery of events must never block the run.** Events are optional; `Result` closes a `discard` channel so `emit` stops waiting.

### Setup

In `run`, the system prompt goes first unless the input already has one, and the iteration count and taint are recomputed from the input, so a resumed transcript continues where it left off:

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

### The model call and its limits

Steps 2 and 5 meet at the top of each iteration:

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

### The gate

Step 4 checks every call with the current `CallContext`. A gate error is treated as a deny, and an approved marker turns `NeedApproval` into `Allow`:

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

### Dispatch

Step 3 is the concurrent part. Denied calls get their result directly; the rest run in goroutines behind a semaphore:

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

### Steps

For step 7, `callModel` wraps the provider call in `RunStep(ctx, types.LLMStep(iter), types.StepLLM, ...)`. Inside, it folds the stream with an `Accumulator`, forwards each `TextDelta` as an `EventText`, and returns the message and usage as JSON. `callTool` does the same for `Registry.Call` under `types.ToolStep(iter, i, c.Name)`. Two details make replay work. First, the result is re-stamped with the call's identity and its outcome is recorded:

```go
res.CallID, res.Name = c.ID, c.Name // a recorded result answers this call
outcome := "ok"
if res.IsError {
	outcome = "error" // tl.tool.outcome: failures visible in traces
	span.SetStatus(codes.Error, res.Content)
}
```

Second, the step name includes the call index `i`, so two calls of the same tool in one turn get two journal entries.

### Optional events

Step 8 keeps a slow or absent reader from stalling the run:

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

The full solution is [`loop.go`](../../../../course/solutions/go/agent/loop/loop.go).

### Why this design

Per iteration, the cost is one model call plus the turn's tool calls, with wall time about the slowest $\lceil |C_t|/k \rceil$ batches of tools. Memory is the transcript. The loop's own work (gate checks, appends) is negligible next to the model.

Keeping the transcript as the only state, rather than a separate run object, is the decision everything else leans on: resume, approval, and durability all fall out of "pass the transcript back in," and there is nothing else to persist. Bounded parallelism beats both alternatives. Sequential calls make a turn as slow as the sum of its tools, and unbounded calls let one turn of ten slow tools overload a backend; a semaphore gives both a speedup and a ceiling. Approval ends the run rather than blocking inside it, because a human may answer in minutes or days, and a blocked goroutine holding a connection is the wrong place to wait. The run returns `ApprovalError` with markers; the caller persists the transcript and invokes again.

Two tradeoffs are accepted. A token budget can be exceeded by at most one model reply, since the loop cannot know a reply's size in advance, but it never dispatches tools after that. And denied calls still count against the tool-call budget, since the model proposed them.

## What can go wrong

A correct loop passes a specific set of checks. The hand example must yield the answer, two model calls, one tool run, usage 30/9, the four-message transcript, the three messages the second call saw, and the event order above. The system prompt is added once, and not to a transcript that already has one, so resumed runs do not stack instructions. `WithMaxIter(3)` with a model that never stops makes three model calls and three tool runs and ends in `ErrMaxIter` with the last message. Six calls with a bound of 2 have exactly two in flight; finished in reverse, they are returned in order. Three calls against a tool-call budget of 2 run none, and 110 tokens against a budget of 100 dispatch nothing. Deny, approval, resume without a model call, and taint only after tool output all behave as described, which is where the gate plugs in. Step names and kinds are right, and a replaying runner reproduces the run with no calls, since the durable runner keys its journal by these names. A step-runner error ends the run with nothing appended, so an unknown write outcome is never papered over. The spans include `agent.run`, two `agent.llm_call` spans with tokens, and `agent.tool` outcomes ok, error, and denied. A provider error ends the run with no tool runs, so there are no silent empty answers. With 2000 unread events, `Result` still does not block. And over the real provider, the fake tool-calling model goes through call, tool, and answer end to end.

The mistakes behind failures of those checks:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| the cap checked after the call | one model call more than configured | `iter >= a.maxIter` is checked before `callModel` |
| unbounded parallel calls, results appended as they finish | ten slow tools at once overload a backend; results misordered | a semaphore of `maxPar` slots; results by index, appended after `wg.Wait()` |
| budgets checked after the spend, usage overwritten instead of summed | tools run past the budget; the budget never trips | token and tool-call checks before dispatch; `s.usage.In +=` |
| verdicts ignored, approvals looked up by the wrong key, a resumed run asking the model again | denied tools run; approved calls stay pending or are replaced | denied calls never reach `callTool`; `approved[v.Marker]`; the resume branch skips `callModel` |
| the context never marked tainted | a gate cannot tell a clean request from one after a poisoned document | `tainted = true` after every turn's results, and from any tool message in the input |
| the system prompt added twice, the proposing assistant turn dropped | stacked instructions; tool messages with no call to answer | the system prompt only when the first message is not a system message; the reply is appended before any check |
| tool step names without the call index, every outcome recorded as ok | two calls of one tool share a journal entry; failures invisible in traces | `ToolStep(iter, i, name)`; `tl.tool.outcome` set from `IsError` |
| errors swallowed | a dead model gives an empty "answer"; a refused replay is shown to the model | provider and step errors return through `finish` with the error |

> **Warning:** If a resumed run calls the model again instead of dispatching the pending calls, the model may propose something different from what the human approved. The resume branch exists precisely to prevent that.

## Where this leads

This loop is the agent the rest of the book runs. [The tool gate](04-tool-gate-and-injection.md)'s prompt-injection suite drives it with the policy gate. [Durable agent runs](05-durable-agent-runs.md) passes its journal as the `StepRunner` and resumes from the recorded input. The [eval runner](../09-llm-evaluation/09-eval-runner.md)'s `AgentSubject` runs the loop on every case of a suite and reads its events. The agent track's end-to-end run is this loop over the gateway.

Production frameworks build further. Saige's `agent` (`agent/agent.go`, `agent/tree/compact.go`) keeps a conversation tree with branching and compaction, sub-agents as call and return, and handoff. Saige's tool policy (`agent/tool_policy.go`) separates policies for disclosure, execution, context, results, and routing. Saige's shared `Budget` (`agent/budget`) reserves request, token, and cost capacity under one lock and settles it once. The loop itself is an instance of patterns like ReAct and plan-and-execute, which interleave reasoning traces with actions or plan before acting.

### Further reading

- Anthropic, [*Building effective agents*](https://www.anthropic.com/engineering/building-effective-agents) (free).
- Yao et al., [*ReAct*](https://arxiv.org/abs/2210.03629) (free).

## Key terms

- **agent loop**: repeated model calls, each followed by running the tool calls it returned and appending their results, until a reply has no calls.
- **iteration cap**: the maximum number of model calls in one run, checked before each call.
- **budget**: a per-run limit on tokens and tool calls, checked before the spend.
- **taint**: the flag, monotone within a run, that says tool output (outside data) is in the context.
- **approval marker**: the gate's identifier for a call needing human approval; passing it to `WithApprovals` turns that verdict into `Allow`.
- **step**: a named, recorded unit of side effect (`llm/<iter>`, `tool/<iter>/<i>/<name>`) run through a `StepRunner`.

## Summary

- The transcript is the loop's only state; each iteration appends the assistant turn and one tool message per call.
- Every limit (iteration cap, token budget, tool-call budget) is checked before the work it limits.
- A turn's calls run in parallel behind a semaphore, and their results are appended in call order.
- The gate decides before dispatch; denied calls never run, and an approval stop ends the run so a resumed run dispatches exactly the approved calls without asking the model again.
- Every model and tool call is a named step, which is what makes the loop durable.

## Review questions

1. In the hand example, why does the second model call see three messages while the transcript ends with four?
   <details><summary>Answer</summary>The second call is made with $M_1$ = user, assistant(call_1), tool(call_1). Its reply, "It is sunny.", is appended afterwards, making four messages. The model never sees its own reply; it sees the tool's result answering its call.</details>
2. Why does the loop check `iter >= maxIter` before the model call instead of counting after it?
   <details><summary>Answer</summary>A check after the call would allow call number `maxIter + 1` to happen and only then stop. The cap is a promise about how many model calls a run may make, so it must gate the call itself.</details>
3. Three calls finish in the order 2, 0, 1. In what order are their tool messages appended, and why does it matter?
   <details><summary>Answer</summary>0, 1, 2: each goroutine writes slot `i` and the loop appends the slots in index order after `wg.Wait()`. Providers check that tool messages follow the assistant message in its call order, and a deterministic order keeps the prompt (and a durable replay) reproducible.</details>
4. How does a resumed run know not to call the model again, without a flag?
   <details><summary>Answer</summary>The transcript ends with an assistant message that has tool calls and no tool messages after it. A normal run always appends the results right after the proposing turn, so that shape occurs only when a run stopped for approval. The loop uses that message as the reply and dispatches its calls.</details>
5. Why does a `StepRunner` error end the run instead of becoming an `error:` tool message?
   <details><summary>Answer</summary>A tool error means the tool ran and failed; the model can react. A step error means the step could not run or its outcome is unknown (in the durable runner, a write whose result was never recorded). Telling the model "error" would let it proceed as if the write had failed when it may have succeeded.</details>
