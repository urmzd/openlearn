<!-- ol:module ag.03 -->
# Agent loop

## Overview

| | |
|---|---|
| **Module** | `ag.03` · build · Go · Pass 10 · 4 h |
| **You build** | `go/agent/loop/loop.go`: `New`, the options (`WithMaxIter`, `WithMaxParallelTools`, `WithBudget`, `WithGate`, `WithStepRunner`, `WithTracer`, `WithApprovals`), `Agent.Invoke`, `Stream` (`Events`, `Result`, `Transcript`, `Usage`), `ErrMaxIter`, `BudgetError`, `ApprovalError` |
| **Contract** | the agent spans of [`course/contracts/otel/semconv.md`](../../course/contracts/otel/semconv.md) (`agent.run`, `agent.llm_call`, `agent.tool <name>`); the Go API is section 4 |
| **Tests** | `course/tests/go/ag_03/` (what they check: section 4) |
| **Needs** | [`ag.01`](01-types-and-provider.md) (`Provider`, `Accumulator`, the `Gate` and `StepRunner` seams), [`ag.02`](02-tools-and-schemas.md) (`Registry`) · reading: tracing in Go (`obs.01`) |
| **Used by** | `ag.04` the prompt-injection suite drives this loop with the gate · `ag.05` runs it durably · `ag.09` evaluates it as a subject |
| **Milestone** | MS-agent |
| **Optional depth** | Anthropic, [*Building effective agents*](https://www.anthropic.com/engineering/building-effective-agents) (free); Yao et al., [*ReAct*](https://arxiv.org/abs/2210.03629) (free) |

## Key Takeaways

- An agent is a **loop**: call the model with the messages and tool definitions, run the calls it returns, append the results, repeat until it answers without a call (`TestHandExample`).
- Every limit is checked **before** the work it limits: the iteration cap before a model call, the tool-call and token budgets before dispatch (`TestMaxIterStops`, `TestBudgetStopsBeforeDispatch`).
- Calls of one turn run **in parallel up to a bound**, and their results go back **in call order** (`TestParallelToolsBoundedOrderPreserved`).
- The gate decides before dispatch: denied calls never run and the model reads why; calls that need approval end the run with a marker, and a resumed run dispatches them **without asking the model again** (`TestGateVerdicts`).
- Every model and tool call is a **named step** through a `StepRunner`, which is what makes the loop durable in `ag.05` (`TestStepRunnerNames`, `TestStepRunnerErrorEndsRun`).

## How to work this chapter

```bash
ol start ag.03
ol tests ag.03
ol check ag.03          # ag.01 and ag.02 smoke tests run first
ol diff  ag.03
```

---

## 1. Why now

You have a model client (`ag.01`) and a tool registry (`ag.02`), and no program that connects them. Without one, "the agent" is a person copying tool calls into a shell. The loop is short, which is the danger: the obvious version runs forever when the model keeps calling tools, runs ten slow calls one after another, spends past its budget before checking it, and has no place to put a safety gate or a durable journal. This module writes the loop once, with those places built in.

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

Per `otel/semconv.md`: `agent.run` (INTERNAL, `gen_ai.agent.name`, `tl.agent.iterations`), one `agent.llm_call` (CLIENT, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`) per model call, and one `agent.tool <name>` (INTERNAL, `gen_ai.tool.name`, `tl.tool.outcome` = `ok`, `error`, `denied`, or `needs_approval`) per call. The loop takes an OpenTelemetry `trace.Tracer`; your composition root passes the one `go/otelx` sets up (`obs.01`).

## 3. Worked example by hand

Input: one user message, "What is the weather in Paris?". The tool `get_weather` answers "sunny, 21 C".

| Step | Messages before | Model returns | Loop does | Usage so far |
|---|---|---|---|---|
| `llm/1` | user | call `call_1` = `get_weather({"city":"Paris"})`, 10 in, 5 out | gate (none: allow), run the call as `tool/1/0/get_weather` | 10 / 5 |
| | user, assistant(call_1), tool(call_1: "sunny, 21 C") | | | |
| `llm/2` | the three above | "It is sunny.", 20 in, 4 out | no calls: the answer | 30 / 9 |

Two model calls, one tool run, a transcript of four messages (user, assistant with the call, the tool message answering `call_1`, the answer), and usage 30 in, 9 out. The events, in order: usage of call 1, the tool call, its result, the three streamed text pieces "It ", "is ", "sunny.", usage of call 2, done. This is `TestHandExample`.

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestHandExample` | unit | section 3: answer, 2 model calls, 1 tool run, usage 30/9, the transcript, what the second call saw, the event order | you and the tests agree on the loop |
| `TestSystemPrompt` | unit | the system prompt is added once, not to a transcript that has one | resumed runs do not stack instructions |
| `TestMaxIterStops` | boundary | `WithMaxIter(3)`: 3 model calls, 3 tool runs, `ErrMaxIter` with the last message | a looping model is stopped |
| `TestParallelToolsBoundedOrderPreserved` | unit | 6 calls, bound 2: exactly 2 in flight; finished in reverse, returned in order | throughput without breaking the call order |
| `TestBudgetStopsBeforeDispatch` | boundary | 3 calls against 2: none run; 110 tokens against 100: no dispatch | spend is checked before it happens |
| `TestGateVerdicts` | unit | deny, approval, resume without a model call, tainted only after tool output | `ag.04` plugs in here |
| `TestStepRunnerNames` | unit | step names and kinds; a replaying runner reproduces the run with no calls | `ag.05` keys its journal by these names |
| `TestStepRunnerErrorEndsRun` | fault | a step runner error ends the run, nothing appended | an unknown write outcome is never papered over |
| `TestSpans` | unit | `agent.run`, two `agent.llm_call` with tokens, `agent.tool` outcomes ok, error, denied | operators read runs from traces |
| `TestProviderErrorEndsRun` | fault | a provider error ends the run, no tool runs | no silent empty answers |
| `TestEventsOptional` | regression | `Result` without reading 2000 events does not block | callers that want only the answer |
| `TestFaketoolEndToEnd` | conformance | over the real provider and the course's fake model: call, tool, answer | the PR stand-in for a real model |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. the cap checked after the call | one model call more than configured | `TestMaxIterStops` (mutant `s01`) |
| 2. unbounded parallel calls, results appended as they finish | ten slow tools at once overload a backend; results misordered | `TestParallelToolsBoundedOrderPreserved` (mutants `s02`, `s03`) |
| 3. budgets checked after the spend, usage overwritten instead of summed | tools run past the budget; the budget never trips | `TestBudgetStopsBeforeDispatch`, `TestHandExample` (mutants `s04`, `s05`, `s13`) |
| 4. verdicts ignored, approvals looked up by the wrong key, a resumed run asking the model again | denied tools run; approved calls stay pending or are replaced | `TestGateVerdicts` (mutants `s06`, `s07`, `s08`, `s14`) |
| 5. the context never marked tainted | a gate cannot tell a clean request from one after a poisoned document | `TestGateVerdicts` (mutant `s09`) |
| 6. the system prompt added twice, the proposing assistant turn dropped | stacked instructions; tool messages with no call to answer | `TestSystemPrompt`, `TestHandExample` (mutants `s10`, `s12`) |
| 7. tool step names without the call index, every outcome recorded as ok | two calls of one tool share a journal entry; failures invisible in traces | `TestStepRunnerNames`, `TestSpans` (mutants `s11`, `s15`) |
| 8. errors swallowed | a dead model gives an empty "answer"; a refused replay is shown to the model | `TestProviderErrorEndsRun`, `TestStepRunnerErrorEndsRun` (mutants `s16`, `s17`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `ag.01` | the provider, the accumulator, the gate and step seams |
| Back | `ag.02` | `Definitions` for every model call, `Call` for every tool call |
| Forward | `ag.04` | the prompt-injection suite runs this loop with the policy gate |
| Forward | `ag.05` | `AgentRun` passes its journal as the `StepRunner` and resumes from the recorded input |
| Forward | `ag.09` | `AgentSubject` runs the loop on every case of a suite and reads its events |

`obs.01` is reading because the loop takes any `trace.Tracer`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| the loop | saige `agent` | a conversation tree with branching and compaction, sub-agents as call and return, handoff | saige `agent/agent.go`, `agent/tree/compact.go` |
| `WithGate` | saige tool policy | separate policies for disclosure, execution, context, results, routing | saige `agent/tool_policy.go` |
| `WithBudget` | saige shared `Budget` | request, token, and cost capacity reserved under one lock and settled once | saige `agent/budget` |
| the pattern | ReAct, plan-and-execute | reasoning traces interleaved with actions; planning before acting | Yao et al., [*ReAct*](https://arxiv.org/abs/2210.03629) (free) |
