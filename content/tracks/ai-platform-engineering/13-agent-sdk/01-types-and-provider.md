# Types, OpenAI-compatible provider, retry wrapper

## Overview

| | |
|---|---|
| **Module** | `ag.01` · build · Go · Pass 10 · 4 h |
| **Solution** | [`types.go`](../../../../course/solutions/go/agent/types/types.go) (messages, tool calls, deltas, the `Accumulator`, the `Gate` and `StepRunner` seams), [`openai.go`](../../../../course/solutions/go/agent/provider/openai.go) (`NewOpenAI`, `RequestBody`, `ChatStream`, `APIError`), [`sse.go`](../../../../course/solutions/go/agent/provider/sse.go) (`EventReader`, `Decoder`), [`retry.go`](../../../../course/solutions/go/agent/provider/retry.go) (`RetryPolicy`, `Retryable`, `WithRetry`) |
| **Contract** | chat completions with tools and the Streaming rules of [`course/contracts/openapi/openai-subset.v1.yaml`](../../../../course/contracts/openapi/openai-subset.v1.yaml); the Go API is section 4 of this chapter |
| **Needs** | reading: the Go primer (`lang.06`), the gateway's SSE proxy (`gw.04`), the engine's tool-call stream (`L10.9`) |
| **Used by** | `ag.02` tool definitions · `ag.03` the loop calls the model through `Provider` · `ag.04` the gate's `Verdict` · `ag.05` the `StepRunner` seam · `ag.06` the embedder's retry policy · `ag.09` evaluation subjects stream through `Provider` and `Accumulator` · `ag.11` judge generator · the agent track's end-to-end run talks to the model only through this provider |
| **Optional depth** | OpenAI, [function calling guide](https://platform.openai.com/docs/guides/function-calling) (free); WHATWG HTML, [server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html) (free); AWS Architecture Blog, *Exponential backoff and jitter* (free) |

## Key Takeaways

- A streamed answer is a sequence of **typed deltas**: text pieces, tool-call starts, argument fragments, call ends, usage, and exactly one final done or error delta.
- Tool-call arguments arrive as **fragments keyed by index**, possibly interleaved between calls; they are assembled per index and closed in index order.
- A stream that ends without `[DONE]` is an **error**, never a short success.
- The retry wrapper retries **only before the first content delta**: after the caller has text, a retry would duplicate it.
- One `Provider` talks to the gateway, the engine, or a frontier API with the same code, because all three speak the same OpenAI subset.

## How to read this chapter

Sections 1 to 3 build the provider from first principles and decode one stream by hand. Section 4 fixes the interface. Section 5 walks through the solution layer by layer (events, deltas, request, retry) and explains why each choice is the right one; try to predict each step before reading it.

---

## 1. Why now

By Pass 10 the system can serve a model with tool calls (`L10.9`) behind a gateway with keys, limits, and a usage ledger (`gw.*`). Nothing in it can yet *use* a model as a component: every caller so far was `curl` or the load generator, which read bytes and never act on them. An agent needs a client that turns the gateway's event stream back into structure: "the model said these words, then asked for `get_weather` with these arguments". Get the client wrong and every layer above it inherits the bug: a dropped argument fragment is a tool called with `{"city":` and a parse error, a stream cut by a reset looks like a short but complete answer, and a retry after the first token prints the answer twice.

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

The rule that makes retries safe for streams is the **commitment boundary** from `gw.04`: until the first content delta, nothing has been shown, so the whole request can be repeated. After it, the caller holds text a second attempt would repeat ("Hel" + "Hello world"). `WithRetry` holds deltas back until the first content delta (usage and nothing else may precede it), retries a failure that happens before that point, and from then on passes everything through, including a later `ErrorDelta`.

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

Four deltas. `Collect` gives `Message{Role: assistant, Content: "Hello"}` and usage {5, 2}. Reading 7 bytes at a time changes nothing: the reader buffers until each blank line. A correct solution gives exactly these deltas at any read size.

For the interleaved tool calls in the recorded stream `tool_interleaved`, the decoder keeps a map index to call: call 0 opens (`get_weather`), call 1 opens (`get_time`, first fragment `{"tz"`), then fragments `{"city":` (0), `:"UTC"}` (1), `"Paris"}` (0). At `finish_reason` it closes 0 with `{"city":"Paris"}`, then 1 with `{"tz":"UTC"}`.

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

### What a correct solution must handle

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| the hand example (section 3) | unit | four deltas; `Collect` gives "Hello" and 5/2 | the delta vocabulary is pinned down |
| recorded streams at read sizes 1, 7, 65536 | conformance | six streams: text, interleaved calls, CRLF with comments and a two-line event, an error event, EOF before `[DONE]`, a call left open at `[DONE]` | the loop sees the same calls whatever TCP does |
| the request body | unit | path, bearer key, model, stream with usage, tools, assistant `content: null` with string arguments (`{}` when empty), tool messages with `tool_call_id` | real providers reject a wrong shape; the model never sees its own results without the ids |
| a 400 | unit | returned once as `*APIError` with the body's type and message | no budget wasted on hopeless retries |
| the backoff schedule | unit | `Delay` doubles from `Initial` to `Max`; 503, 503, 200 waits 100 ms then 200 ms (with `Initial` 100 ms) | the formula of section 2.3 |
| `Retry-After` | unit | 429 with `Retry-After: 3` waits 3 s | the gateway's limiter says when to come back |
| `MaxAttempts` | boundary | `MaxAttempts: 3` makes exactly 3 requests | bounded retries |
| a reset before content | fault | a reset after the role chunk is retried; "Hello" arrives once | blips do not fail agent steps |
| a reset after content | fault | a reset after three tokens ends in an error after one request; "a b c" is not duplicated | no repeated or spliced text |
| cancellation | fault | cancelling `ctx` closes the channel and the server sees its request cancelled | a cancelled agent stops paying for tokens |
| the fake tool-calling model | conformance | a call sent in two halves is rebuilt; the tool result goes back with its id | the stand-in for a model in every PR check |
| which errors retry | unit | the `Retryable` table of section 2.3 | one place to read the retry policy |
| the accumulator | unit | calls sorted by index, missing arguments are `{}`, no end delta is `ErrIncomplete` | the loop returns results in call order |
| the seams | boundary | a missing call context reads as tainted; step names; `IsContent` | the gate and the durable runner rely on them |

## 5. The solution, step by step

### 5.1 From first principles

1. **Bytes, then events, then deltas: three layers, three jobs.** TCP delivers arbitrary pieces; SSE groups lines into events; the OpenAI chunk format gives each event meaning. Keeping them separate (`EventReader`, `Decoder`, `ChatStream`) means each can be checked alone: the reader against framing rules, the decoder against chunk sequences.
2. **The reader works on lines, so it buffers.** `bufio.Reader.ReadBytes('\n')` returns a whole line however the bytes arrived, which makes read size irrelevant (1 byte or 64 KiB give the same events). A blank line ends an event; EOF in the middle of one is `io.ErrUnexpectedEOF`.
3. **Tool calls are identified by index, not by arrival order.** Fragments of call 0 and call 1 interleave, so the decoder keeps `map[int]*ToolCall` and appends each fragment to its own call. Closing in sorted index order gives a deterministic result whatever the interleaving.
4. **The end of a stream is explicit.** `[DONE]` is the only success; EOF without it, an error event, or a broken connection all become an `ErrorDelta`. Exactly one terminal delta is sent, and the channel is closed after it.
5. **Retry safety depends on what the caller has seen.** Until the first content delta, a retry is invisible; after it, a retry duplicates. So the retry wrapper must buffer the prefix (usage, nothing else can precede content) and decide only once it has either content or a failure.
6. **Waiting should grow, but the server's word wins.** Exponential backoff spreads retries out; `Retry-After` is the server saying exactly when capacity returns, so it replaces the formula.
7. **Seams fail closed.** A gate asked without a call context must assume the worst (tainted), because the zero value of a struct would claim the opposite.

### 5.2 The code, annotated

**The event reader** (`sse.go`), step 2:

```go
func (e *EventReader) Next() ([]byte, error) {
	var data [][]byte
	pending := false // inside an event: some non-blank line since the last blank one
	for {
		line, err := e.br.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				if pending || len(line) > 0 {
					return nil, io.ErrUnexpectedEOF // the input ended inside an event
				}
				return nil, io.EOF // a clean end
			}
			return nil, err
		}
		line = bytes.TrimSuffix(bytes.TrimSuffix(line, []byte("\n")), []byte("\r")) // \n or \r\n
		if len(line) == 0 { // a blank line ends the event
			if len(data) > 0 {
				return bytes.Join(data, []byte("\n")), nil // multi-line data joined with \n
			}
			pending = false // an event with no data (only comments) is skipped
			continue
		}
		pending = true
		if line[0] == ':' {
			continue // a comment, such as ": ping"
		}
		if v, ok := bytes.CutPrefix(line, []byte("data:")); ok {
			v, _ = bytes.CutPrefix(v, []byte(" "))      // one optional space
			data = append(data, append([]byte(nil), v...)) // copy: the bufio buffer is reused
		}
	}
}
```

**The decoder**, steps 3 and 4. `[DONE]` closes any open calls before the `DoneDelta`, so a call left open by a sloppy server still reaches the caller:

```go
func (d *Decoder) Event(data []byte) (out []types.Delta, done bool) {
	if string(bytes.TrimSpace(data)) == "[DONE]" {
		out = append(out, d.endCalls()...)
		return append(out, types.DoneDelta{FinishReason: d.finish}), true
	}
	var c chunk
	if err := json.Unmarshal(data, &c); err != nil {
		return []types.Delta{types.ErrorDelta{Err: fmt.Errorf("provider: bad chunk: %w", err)}}, true
	}
	if c.Error != nil { // an in-band error after the first byte
		// ... *StreamError{Message, Type, Code}
		return []types.Delta{types.ErrorDelta{Err: se}}, true
	}
	for _, ch := range c.Choices {
		if ch.Delta.Content != nil && *ch.Delta.Content != "" { // empty content is not text
			out = append(out, types.TextDelta{Text: *ch.Delta.Content})
		}
		for _, tc := range ch.Delta.ToolCalls {
			call, ok := d.calls[tc.Index]
			if !ok { // the first fragment of this index opens the call
				call = &types.ToolCall{ID: tc.ID, Name: tc.Function.Name}
				d.calls[tc.Index] = call
				out = append(out, types.ToolCallStartDelta{Index: tc.Index, ID: tc.ID, Name: tc.Function.Name})
			}
			if tc.Function.Arguments != "" {
				call.Args = append(call.Args, tc.Function.Arguments...) // to its own call, by index
				out = append(out, types.ToolCallArgsDelta{Index: tc.Index, Fragment: tc.Function.Arguments})
			}
		}
		if ch.FinishReason != nil && *ch.FinishReason != "" {
			d.finish = *ch.FinishReason
			out = append(out, d.endCalls()...)
		}
	}
	if c.Usage != nil { // the choices: [] chunk
		out = append(out, types.UsageDelta{In: c.Usage.PromptTokens, Out: c.Usage.CompletionTokens})
	}
	return out, false
}
```

`endCalls` collects the indexes not yet ended, sorts them, marks each ended (so `[DONE]` after `finish_reason` closes nothing twice), and emits a copy of each call with `{}` for empty arguments.

**The request body** (`openai.go`). The wire format differs from the Go types in three places, each of which a strict provider rejects if wrong:

```go
case types.RoleAssistant:
	if content != "" || len(m.ToolCalls) == 0 {
		w.Content = &content // otherwise nil: "content": null for a tool-calls-only message
	}
	for _, c := range m.ToolCalls {
		args := string(c.Args) // arguments are a JSON *string* on the wire
		if args == "" {
			args = "{}"
		}
		w.ToolCalls = append(w.ToolCalls, wireCall{ID: c.ID, Type: "function", Function: wireFunction{Name: c.Name, Arguments: args}})
	}
case types.RoleTool:
	w.Content = &content
	w.ToolCallID = m.ToolCallID // how the model matches a result to its call
	w.Name = m.Name
```

and every request sets `"stream": true, "stream_options": {"include_usage": true}`, so the usage chunk always arrives.

**Streaming** (`ChatStream`). The HTTP request is built on the caller's `ctx`, a non-2xx status becomes an `*APIError` before any delta (reading `Retry-After` in seconds), and a goroutine pumps events into an unbuffered channel:

```go
go func() {
	defer close(ch)
	defer resp.Body.Close()
	send := func(d types.Delta) bool {
		select {
		case ch <- d:
			return true
		case <-ctx.Done():
			return false // nobody is reading: stop, and the request's ctx aborts the body
		}
	}
	rd := NewEventReader(resp.Body)
	dec := NewDecoder()
	for {
		data, err := rd.Next()
		if err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				err = ErrNoDone // any end without [DONE] is a cut stream (step 4)
			}
			if ctx.Err() != nil {
				err = ctx.Err() // report the cancellation, not its side effect
			}
			send(types.ErrorDelta{Err: err})
			return
		}
		out, done := dec.Event(data)
		// ... send each delta; return when done
	}
}()
```

**Backoff** (`retry.go`), step 6:

```go
func (p RetryPolicy) Delay(n int, err error) time.Duration {
	p = p.withDefaults() // 4 attempts, 500 ms, 30 s, x2
	var ae *APIError
	if errors.As(err, &ae) && ae.RetryAfter > 0 {
		return ae.RetryAfter // the server knows when capacity returns
	}
	d := float64(p.Initial) * math.Pow(p.Multiplier, float64(n-1)) // n = 1 waits Initial
	if d > float64(p.Max) {
		return p.Max
	}
	return time.Duration(d)
}
```

`Retryable` returns false first for a cancelled or expired context, then decides by type: an `*APIError` by status (408, 429, 500, 502, 503, 504), a `*StreamError` only for `server_error`, and `ErrNoDone`, `io.ErrUnexpectedEOF`, `ECONNRESET`, `ECONNREFUSED`, or any `net.Error` as transient.

**The retry wrapper**, step 5. Each attempt is read until it commits or fails, holding the deltas seen so far:

```go
var held []types.Delta
var failed error
committed := false
for d := range ch {
	held = append(held, d)
	if types.IsContent(d) {
		committed = true // the caller is about to see output: no more retries
		break
	}
	if e, ok := d.(types.ErrorDelta); ok {
		failed = e.Err
		break
	}
	if _, ok := d.(types.DoneDelta); ok {
		committed = true // an empty but complete answer
		break
	}
}
if !committed && failed == nil {
	failed = ErrNoDone // closed without an end delta
	held = append(held, types.ErrorDelta{Err: failed})
}
if !committed && Retryable(failed) && attempt < r.pol.MaxAttempts {
	for range ch { // drain the dead attempt
	}
	// ... sleep Delay(attempt, failed), honoring ctx; then the next attempt
	continue
}
// otherwise: a goroutine sends `held`, then forwards the rest of ch unchanged,
// including any later ErrorDelta, which is never retried
```

A refused request (`ChatStream` returning an error) takes the simpler path at the top of the loop: retry if `Retryable` and attempts remain.

**The accumulator** (`types.go`). `Add` folds each delta into text, a `map[int]*ToolCall`, usage, and the terminal state; `Message` sorts the call indexes and fills `{}` for empty arguments; `Err` is `ErrIncomplete` until a `DoneDelta` or `ErrorDelta` was seen. **The seam** fails closed:

```go
func CallContextFrom(ctx context.Context) CallContext {
	if cc, ok := ctx.Value(callContextKey{}).(CallContext); ok {
		return cc
	}
	return CallContext{Tainted: true} // the zero value would claim a clean context (step 7)
}
```

Full solution: [`types.go`](../../../../course/solutions/go/agent/types/types.go), [`openai.go`](../../../../course/solutions/go/agent/provider/openai.go), [`sse.go`](../../../../course/solutions/go/agent/provider/sse.go), [`retry.go`](../../../../course/solutions/go/agent/provider/retry.go).

### 5.3 Why this solution

- **Cost.** Decoding is $O(n)$ in the stream's bytes with one 32 KiB read buffer; tool-call assembly appends to per-call byte slices. The retry wrapper buffers only the prefix before the first content delta (a role chunk, perhaps usage), so its memory is constant.
- **A channel of typed deltas** rather than a callback or returning the full message: the loop can act on text as it arrives (show it, cancel it) and still rebuild the full message with `Accumulator`; a callback would push concurrency questions onto every caller.
- **One OpenAI-compatible provider** rather than one client per backend: the gateway, the engine, and frontier APIs speak the same subset, so the same code reaches all three. Provider-specific features (thinking deltas, citations) are what production SDKs add on top.
- **Retry only before content** rather than "retry and de-duplicate": de-duplication would need the second attempt to reproduce the first exactly, which sampling does not guarantee.
- **Tradeoff accepted:** no jitter in the backoff. Many agents failing together would retry in step; production clients add full jitter (see Going further).

### 5.4 Explain it back

1. Why does `EventReader` copy each data line (`append([]byte(nil), v...)`) before keeping it?
   <details><summary>Answer</summary>`bufio.Reader.ReadBytes` returns a slice that may share the reader's internal buffer, which the next read overwrites. Keeping it without a copy would corrupt the first lines of a multi-line event while later lines are read.</details>
2. In the `tool_interleaved` stream, what would go wrong if fragments were appended to "the most recently opened call"?
   <details><summary>Answer</summary>Call 1 opens after call 0, so the fragments `{"city":` and `"Paris"}` meant for call 0 would land on call 1, giving call 1 the arguments `{"tz"{"city"::"UTC"}"Paris"}` (invalid JSON) and call 0 none at all (`{}`, a valid call with missing arguments). Keying by `Index` puts each fragment on its own call.</details>
3. Why is a `DoneDelta` counted as committing in the retry wrapper, even though it is not content?
   <details><summary>Answer</summary>A stream that ends cleanly with no content is a complete (empty) answer, not a failure. There is nothing to retry; passing it through gives the caller the correct finish reason.</details>
4. A 429 arrives with `Retry-After: 3` on the second failure. How long does `WithRetry` wait, and why not 1 s?
   <details><summary>Answer</summary>3 s. `Delay` returns the server's `Retry-After` whenever the error carries one; the formula's 1 s ($500\text{ ms} \cdot 2^{1}$) would come back before the limiter refills and earn another 429.</details>
5. Why does `ChatStream` replace the error with `ctx.Err()` when the context is done?
   <details><summary>Answer</summary>Cancelling the context aborts the HTTP body, so the reader sees an EOF or a read error that looks like a cut stream (`ErrNoDone`, which is retryable). Reporting `context.Canceled` instead tells the caller and `Retryable` the truth: the caller gave up, so nothing should retry.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. empty content emitted as text, the usage chunk ignored, or `include_usage` never requested | empty events in the UI; the agent's budget never sees a token | the decoder skips empty content and emits `UsageDelta`; `RequestBody` always sets `include_usage` |
| 2. one data line per event, or `\r` left on CRLF lines | multi-line events lose their head; CRLF streams never end an event | `EventReader` joins every `data:` line and trims `\r` before checking for a blank line |
| 3. fragments appended to the last opened call, calls closed or listed out of index order, calls left open at `[DONE]` | `{"city":":"UTC"}`-style garbage; results matched to the wrong call | `map[int]*ToolCall` by index; `endCalls` sorts; `[DONE]` closes open calls |
| 4. an error event read as an empty chunk, EOF before `[DONE]` treated as done | a cut answer looks complete | `c.Error` becomes a `*StreamError`; EOF becomes `ErrNoDone` |
| 5. arguments `""` instead of `{}`, `content: ""` instead of null, no `tool_call_id`, the key in the wrong header | 400 from strict providers; the model cannot match results to calls | `RequestBody`'s assistant and tool cases; `Authorization: Bearer` |
| 6. retrying after content, or never retrying a stream that broke before content | duplicated text; avoidable failures | `WithRetry` holds deltas until `IsContent`, retries a retryable failure before it, and passes everything after |
| 7. retrying 4xx, an off-by-one exponent, ignoring `Retry-After`, one attempt too many | wasted calls; a gateway 429 hammered | `Retryable`'s status list; `Multiplier^(n-1)`; `RetryAfter` first; `attempt < MaxAttempts` |
| 8. the HTTP request built without the caller's `ctx` | a cancelled agent keeps generating | `http.NewRequestWithContext(ctx, ...)`; `send` stops on `ctx.Done()` |
| 9. `CallContextFrom` returning the zero value | a gate called without context believes the context is clean | it returns `Tainted: true` when none is attached |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Forward | `ag.02` | `ToolDef` and `ToolCall` are the registry's inputs and outputs |
| Forward | `ag.03` | the loop calls `Provider.ChatStream`, folds deltas with `Accumulator`, and names steps with `LLMStep` and `ToolStep` |
| Forward | `ag.04` | the gate returns `Verdict`s and reads `CallContextFrom` |
| Forward | `ag.05` | the durable runner implements `StepRunner` and parses `StepToolName` |
| Forward | `ag.06` | the embedder retries with `Retryable` and `RetryPolicy.Delay` |
| Forward | `ag.09` | `ProviderSubject` evaluates a bare model through `Provider` and `Accumulator` |
| Forward | `ag.11` | the judge generator uses the shared provider types |

The provider reaches `gw.04` and `L10.9` over HTTP only, so they are reading, not code dependencies.

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `types` | saige typed messages | thinking, citation, and route deltas; invalid-control rejection | saige `agent/types` |
| `OpenAI` provider | saige providers | Anthropic, Google, Ollama, a fallback chain, a router and model catalog, structured output | saige `provider/{anthropic,google,ollama,fallback,router,catalog}` |
| `WithRetry` | saige retry provider | the same drain-until-content rule, plus jittered backoff | saige `provider/retry` (`drainUntilContentOrError`) |
| backoff | full jitter | spreading retries of many clients so they do not arrive together | AWS Architecture Blog, *Exponential backoff and jitter* (free) |
