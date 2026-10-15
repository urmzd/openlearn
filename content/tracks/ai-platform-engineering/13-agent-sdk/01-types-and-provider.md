<!-- ol:module ag.01 -->
# Types, OpenAI-compatible provider, retry wrapper

## Overview

| | |
|---|---|
| **Module** | `ag.01` · build · Go · Pass 10 · 4 h |
| **You build** | `go/agent/types/types.go` (messages, tool calls, deltas, the `Accumulator`, the `Gate` and `StepRunner` seams), `go/agent/provider/openai.go` (`NewOpenAI`, `RequestBody`, `ChatStream`, `APIError`), `go/agent/provider/sse.go` (`EventReader`, `Decoder`), `go/agent/provider/retry.go` (`RetryPolicy`, `Retryable`, `WithRetry`) |
| **Contract** | chat completions with tools and the Streaming rules of [`course/contracts/openapi/openai-subset.v1.yaml`](../../../../course/contracts/openapi/openai-subset.v1.yaml); the Go API is section 4 of this chapter |
| **Tests** | `course/tests/go/ag_01/` (what they check: section 4), fixtures `course/fixtures/ag.01/streams.json` |
| **Needs** | reading: the Go primer (`lang.06`), the gateway's SSE proxy (`gw.04`), the engine's tool-call stream (`L10.9`) |
| **Used by** | `ag.02` tool definitions · `ag.03` the loop calls the model through `Provider` · `ag.04` the gate's `Verdict` · `ag.05` the `StepRunner` seam · `ag.06` the embedder's retry policy · `ag.09` evaluation subjects stream through `Provider` and `Accumulator` · `ag.11` judge generator |
| **Milestone** | MS-agent |
| **Optional depth** | OpenAI, [function calling guide](https://platform.openai.com/docs/guides/function-calling) (free); WHATWG HTML, [server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html) (free); AWS Architecture Blog, *Exponential backoff and jitter* (free) |

## Key Takeaways

- A streamed answer is a sequence of **typed deltas**: text pieces, tool-call starts, argument fragments, call ends, usage, and exactly one final done or error delta (`TestHandExample`, `TestRecordedStreams`).
- Tool-call arguments arrive as **fragments keyed by index**, possibly interleaved between calls; they are assembled per index and closed in index order (`TestRecordedStreams`, `TestAccumulatorOrdersCallsByIndex`).
- A stream that ends without `[DONE]` is an **error**, never a short success (`TestRecordedStreams`, the `no_done` stream).
- The retry wrapper retries **only before the first content delta**: after the caller has text, a retry would duplicate it (`TestRetryBeforeFirstContent`, `TestNoRetryAfterContent`).
- One `Provider` talks to your gateway, your engine, or a frontier API with the same code, because all three speak the same OpenAI subset (`TestFaketoolToolCall`).

## How to work this chapter

```bash
ol start ag.01          # stubs into your repo
ol tests ag.01          # read the test catalog first
ol check ag.01          # exit code is the verdict
ol diff  ag.01          # after passing: your code against the reference
```

---

## 1. Why now

By Pass 10 your system can serve a model with tool calls (`L10.9`) behind a gateway with keys, limits, and a usage ledger (`gw.*`). Nothing in it can yet *use* a model as a component: every caller so far was `curl` or the load generator, which read bytes and never act on them. An agent needs a client that turns the gateway's event stream back into structure: "the model said these words, then asked for `get_weather` with these arguments". Get the client wrong and every layer above it inherits the bug: a dropped argument fragment is a tool called with `{"city":` and a parse error, a stream cut by a reset looks like a short but complete answer, and a retry after the first token prints the answer twice.

## 2. Principles

### 2.1 Messages, calls, and deltas

| Name | Meaning | Go type |
|---|---|---|
| message | one turn: a role (`system`, `user`, `assistant`, `tool`) and its content | `types.Message` |
| tool call | the model asking for tool `Name` with JSON object `Args`, identified by `ID` | `types.ToolCall` |
| tool definition | a tool's name, description, and JSON Schema for its arguments | `types.ToolDef` |
| delta | one piece of a streamed answer | `types.Delta` |

An assistant message may carry tool calls instead of text. The program runs them and answers each with a **tool message** whose `ToolCallID` names the call, then asks the model again. That cycle is the agent loop (`ag.03`); this module is its vocabulary.

The provider turns the wire stream into these deltas:

| Delta | When |
|---|---|
| `TextDelta{Text}` | a chunk's `delta.content` is non-empty |
| `ToolCallStartDelta{Index, ID, Name}` | the first fragment of tool call `Index` |
| `ToolCallArgsDelta{Index, Fragment}` | every non-empty `function.arguments` fragment |
| `ToolCallEndDelta{Index, Call}` | at `finish_reason` (or `[DONE]`), every open call in index order, with its assembled arguments (`{}` when there were none) |
| `UsageDelta{In, Out}` | the usage chunk (`choices: []`) |
| `DoneDelta{FinishReason}` | `data: [DONE]` |
| `ErrorDelta{Err}` | an error event, a cut stream, a broken connection |

The **content deltas** are text and the three tool-call deltas: they are what the caller shows or acts on. `Accumulator` folds deltas back into one `Message`, with tool calls sorted by index.

### 2.2 Server-sent events

The body of a streamed answer is `text/event-stream`. An **event** is a group of lines ended by a blank line. A line `data: <value>` contributes `<value>` (one leading space removed); several data lines in one event are joined with `\n`. A line starting with `:` is a comment (`: ping` keeps idle connections open). Lines end in `\n` or `\r\n`. The reader therefore keeps state across reads: a TCP read may hold half an event, or three.

`EventReader.Next` returns the data of the next event, `io.EOF` at a clean end, and `io.ErrUnexpectedEOF` when the input ends inside an event. The contract says every stream ends with `data: [DONE]`. A stream that reaches EOF without it was cut (a reset, a crashed engine, a proxy timeout), so the provider reports `ErrNoDone`. An event `data: {"error": {...}}` after the first byte is the server reporting a failure in band; it becomes a `*StreamError`.

### 2.3 When a retry is safe

| Symbol | Meaning |
|---|---|
| $n$ | failures so far ($n = 1$ after the first attempt failed) |
| $d_0$ | `Initial`, the first wait (500 ms) |
| $m$ | `Multiplier` (2) |
| $d_{\max}$ | `Max`, the cap (30 s) |
| $d_n$ | the wait after failure $n$ |

$$d_n = \min\left(d_{\max},\ d_0 \cdot m^{\,n-1}\right)$$

so the waits are 500 ms, 1 s, 2 s, ... When the server sends `Retry-After: s`, the wait is $s$ seconds instead: the server knows when its capacity returns. `MaxAttempts` counts every attempt including the first.

`Retryable` decides whether trying again can help. Overload and server errors (408, 429, 500, 502, 503, 504), a refused or reset connection, a stream cut before `[DONE]`, and a `server_error` event can succeed later. A client error (400, 401, 403, 404, 422) will fail the same way every time, and a cancelled context means the caller gave up.

The rule that makes retries safe for streams is the **commitment boundary** you met in `gw.04`: until the first content delta, nothing has been shown, so the whole request can be repeated. After it, the caller holds text a second attempt would repeat ("Hel" + "Hello world"). `WithRetry` holds deltas back until the first content delta (usage and nothing else may precede it), retries a failure that happens before that point, and from then on passes everything through, including a later `ErrorDelta`.

### 2.4 The seams

Two interfaces are declared here although later modules implement them, because the loop (`ag.03`) consumes them and must not import their implementations:

- `Gate.Check(ctx, call) (Verdict, error)`: `Allow`, `Deny` with a reason, or `NeedApproval` with a `Marker` (`ag.04`). The loop attaches a `CallContext{Tainted, Iter}` to `ctx`; `CallContextFrom` returns `Tainted: true` when none is attached, so a gate called without context errs on the cautious side.
- `StepRunner.RunStep(ctx, name, kind, fn)`: every model call and tool call is a named step (`llm/<iter>`, `tool/<iter>/<i>/<name>`). `Direct` just calls `fn`; `ag.05` records results and replays them.

## 3. Worked example by hand

The engine streams "Hello" in five events (one empty-content chunk while a character is incomplete) and the client reads 7 bytes at a time:

```
data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""},...}]}
data: {"choices":[{"index":0,"delta":{"content":"Hel"},...}]}
data: {"choices":[{"index":0,"delta":{"content":"lo"},...}]}
data: {"choices":[{"index":0,"delta":{"content":""},...}]}
data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}
data: [DONE]
```

| Event | Decoder state | Deltas out |
|---|---|---|
| role chunk, content `""` | nothing | none (empty content is not text) |
| `"Hel"` | | `TextDelta{"Hel"}` |
| `"lo"` | | `TextDelta{"lo"}` |
| `""` | | none |
| `finish_reason: stop` | finish = stop; no open calls | none |
| usage chunk | | `UsageDelta{5, 2}` |
| `[DONE]` | | `DoneDelta{"stop"}`, channel closed |

Four deltas. `Collect` gives `Message{Role: assistant, Content: "Hello"}` and usage {5, 2}. Reading 7 bytes at a time changes nothing: the reader buffers until each blank line. This is `TestHandExample`.

For the interleaved tool calls in the fixture `tool_interleaved`, the decoder keeps a map index to call: call 0 opens (`get_weather`), call 1 opens (`get_time`, first fragment `{"tz"`), then fragments `{"city":` (0), `:"UTC"}` (1), `"Paris"}` (0). At `finish_reason` it closes 0 with `{"city":"Paris"}`, then 1 with `{"tz":"UTC"}`.

## 4. The interface

```go
package types // import "tinyllm/agent/types"

type Message struct { Role Role; Content string; ToolCalls []ToolCall; ToolCallID, Name string }
type ToolCall struct { ID, Name string; Args json.RawMessage }
type ToolDef struct { Name, Description string; Parameters json.RawMessage }
type Delta interface{ isDelta() } // TextDelta ToolCallStartDelta ToolCallArgsDelta ToolCallEndDelta UsageDelta ErrorDelta DoneDelta
func IsContent(d Delta) bool
type Provider interface {
	ChatStream(ctx context.Context, msgs []Message, tools []ToolDef, opts ...CallOption) (<-chan Delta, error)
}
type Accumulator struct{ ... } // Add(Delta); Message() Message; Usage() Usage; FinishReason() string; Err() error
func Collect(ch <-chan Delta) (Message, Usage, error)
type Gate interface{ Check(ctx context.Context, c ToolCall) (Verdict, error) }
func WithCallContext(ctx context.Context, cc CallContext) context.Context
func CallContextFrom(ctx context.Context) CallContext
type StepRunner interface {
	RunStep(ctx context.Context, name string, kind StepKind, fn func(context.Context) ([]byte, error)) ([]byte, error)
}
func LLMStep(iter int) string; func ToolStep(iter, i int, name string) string; func StepToolName(step string) (string, bool)

package provider // import "tinyllm/agent/provider"

func NewOpenAI(cfg Config) *OpenAI // Config{BaseURL, APIKey, Model string; Client *http.Client; Headers map[string]string}
func RequestBody(model string, msgs []types.Message, tools []types.ToolDef, o types.CallOptions) ([]byte, error)
type APIError struct { Status int; Type, Code, Message string; RetryAfter time.Duration }
func NewEventReader(r io.Reader) *EventReader // Next() ([]byte, error)
func NewDecoder() *Decoder                    // Event(data []byte) ([]types.Delta, bool)
type RetryPolicy struct { MaxAttempts int; Initial, Max time.Duration; Multiplier float64; Sleep func(context.Context, time.Duration) error }
func (p RetryPolicy) Delay(n int, err error) time.Duration
func Retryable(err error) bool
func WithRetry(p types.Provider, pol RetryPolicy) types.Provider
```

Use only the standard library. The request is `POST {BaseURL}/chat/completions` with `stream: true` and `stream_options.include_usage: true`.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestHandExample` | unit | section 3: four deltas, `Collect` gives "Hello" and 5/2 | you and the tests agree on the delta vocabulary |
| `TestRecordedStreams` | conformance | six fixture streams at read sizes 1, 7, 65536: text, interleaved calls, CRLF and comments and a two-line event, an error event, EOF before `[DONE]`, a call left open at `[DONE]` | the loop sees the same calls whatever TCP does |
| `TestRequestBody` | unit | path, bearer key, model, stream with usage, tools, assistant `content: null` with string arguments (`{}` when empty), tool messages with `tool_call_id` | real providers reject a wrong shape; the model never sees its own results without the ids |
| `TestAPIErrorNotRetried` | unit | a 400 is returned once as `*APIError` with the body's type and message | no budget wasted on hopeless retries |
| `TestRetryBackoffSchedule` | unit | `Delay` doubles from `Initial` to `Max`; 503, 503, 200 waits 100 ms then 200 ms | the formula of section 2.3 |
| `TestRetryAfterHonored` | unit | 429 with `Retry-After: 3` waits 3 s | the gateway's limiter says when to come back |
| `TestRetryGivesUpAtMaxAttempts` | boundary | `MaxAttempts: 3` makes exactly 3 requests | bounded retries |
| `TestRetryBeforeFirstContent` | fault | a reset after the role chunk is retried; "Hello" arrives once | blips do not fail agent steps |
| `TestNoRetryAfterContent` | fault | a reset after three tokens ends in an error after one request, text "a b c" not duplicated | no repeated or spliced text |
| `TestCancelStopsStream` | fault | cancelling `ctx` closes the channel and the server sees its request cancelled | a cancelled agent stops paying for tokens |
| `TestFaketoolToolCall` | conformance | against the course fake model: a call in two halves is rebuilt, the tool result goes back with its id | the stand-in for a model in every PR check |
| `TestRetryableTable` | unit | which errors are retryable | one place to read the retry policy |
| `TestAccumulatorOrdersCallsByIndex` | unit | calls sorted by index, missing arguments are `{}`, no end delta is `ErrIncomplete` | the loop returns results in call order |
| `TestSeams` | boundary | a missing call context reads as tainted; step names; `IsContent` | the gate and the durable runner rely on them |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. empty content emitted as text, the usage chunk ignored, or `include_usage` never requested | empty events in the UI; the agent's budget never sees a token | `TestHandExample`, `TestRequestBody` (mutants `s04`, `s08`, `s14`) |
| 2. one data line per event, or `\r` left on CRLF lines | multi-line events lose their head; CRLF streams never end an event | `TestRecordedStreams` (mutants `s01`, `s03`) |
| 3. fragments appended to the last opened call, calls closed or listed out of index order, calls left open at `[DONE]` | `{"city":":"UTC"}`-style garbage; results matched to the wrong call | `TestRecordedStreams`, `TestAccumulatorOrdersCallsByIndex` (mutants `s05`, `s06`, `s07`, `s22`) |
| 4. an error event read as an empty chunk, EOF before `[DONE]` treated as done | a cut answer looks complete | `TestRecordedStreams` (mutants `s09`, `s10`) |
| 5. arguments `""` instead of `{}`, `content: ""` instead of null, no `tool_call_id`, the key in the wrong header | 400 from strict providers; the model cannot match results to calls | `TestRequestBody` (mutants `s11`, `s12`, `s13`, `s15`) |
| 6. retrying after content, or never retrying a stream that broke before content | duplicated text; avoidable failures | `TestNoRetryAfterContent`, `TestRetryBeforeFirstContent` (mutants `s16`, `s24`) |
| 7. retrying 4xx, an off-by-one exponent, ignoring `Retry-After`, one attempt too many | wasted calls; a gateway 429 hammered | `TestAPIErrorNotRetried`, `TestRetryBackoffSchedule`, `TestRetryAfterHonored`, `TestRetryGivesUpAtMaxAttempts` (mutants `s17`, `s18`, `s19`, `s20`) |
| 8. the HTTP request built without the caller's `ctx` | a cancelled agent keeps generating | `TestCancelStopsStream` (mutant `s21`) |
| 9. `CallContextFrom` returning the zero value | a gate called without context believes the context is clean | `TestSeams` (mutant `s23`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Forward | `ag.02` | `ToolDef` and `ToolCall` are the registry's inputs and outputs |
| Forward | `ag.03` | the loop calls `Provider.ChatStream`, folds deltas with `Accumulator`, and names steps with `LLMStep` and `ToolStep` |
| Forward | `ag.04` | the gate returns `Verdict`s and reads `CallContextFrom` |
| Forward | `ag.05` | the durable runner implements `StepRunner` and parses `StepToolName` |
| Forward | `ag.06` | the embedder retries with `Retryable` and `RetryPolicy.Delay` |
| Forward | `ag.09` | `ProviderSubject` evaluates a bare model through `Provider` and `Accumulator` |
| Forward | `ag.11` | the judge generator uses the shared provider types |

The provider reaches `gw.04` and `L10.9` over HTTP only, so they are reading, not dependencies: `ol check ag.01` never needs them built.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `types` | saige typed messages | thinking, citation, and route deltas; invalid-control rejection | saige `agent/types` |
| `OpenAI` provider | saige providers | Anthropic, Google, Ollama, a fallback chain, a router and model catalog, structured output | saige `provider/{anthropic,google,ollama,fallback,router,catalog}` |
| `WithRetry` | saige retry provider | the same drain-until-content rule, plus jittered backoff | saige `provider/retry` (`drainUntilContentOrError`) |
| backoff | full jitter | spreading retries of many clients so they do not arrive together | AWS Architecture Blog, *Exponential backoff and jitter* (free) |
