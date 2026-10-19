# Contract tests: consumer-driven tests for gateway to engine

The gateway of *AI Platform Engineering* now forwards to the engine of *Build Your Own LLM System* across a process boundary, and both keep changing. The gateway gains metering ([The usage ledger and admin API](../../ai-platform-engineering/12-gateway/07-usage-ledger-and-admin-api.md)) and routing ([Worker registry, routing, cascades, failover](../../ai-platform-engineering/12-gateway/05-routing.md)); the engine is rebuilt from scratch in the [serving chapter](../../build-an-llm/p10-serving/README.md); and later the API itself migrates to v2 ([Interface migration: API v1 to v2](../07-maintenance/02-api-v1-to-v2.md)). The gateway's unit tests run against a fake engine written in the same test file, so they test the gateway against your **belief** about the engine. When the engine renames a field, adds a comment line to its stream, or starts answering 429, every unit test stays green and production breaks.

The two obvious alternatives are worse. An end-to-end test that starts both services is slow, flaky, and says nothing about which side broke. A **contract test** sits between them: the belief is written down once, in a file, the gateway is tested against it, and the file is checked against the engine's published contract. This section teaches the technique on the one seam every later chapter changes: gateway to engine. It builds consumer-driven contracts from first principles, works one streamed interaction by hand, and then walks through the client's request builder and stream reader line by line, the pact's interactions, and the tests that use them. Try to predict each rule of `NewRequest` and each branch of `ReadStream` before reading it.

You should know how a test suite is judged by planted faults ([TDD, unit tests, and mutation testing](01-tdd-unit-tests-and-mutation-grading.md)) and how to read survivors ([Mutation testing in depth](05-mutation-testing-in-depth.md)). The kata needs SSE on the wire from [HTTP/1.1, JSON, and Server-Sent Events from the wire up](../../programming-languages/02-language-and-tool-primers/05-http-and-sse.md) and Go with `httptest` from [Go: packages, interfaces, goroutines](../../programming-languages/02-language-and-tool-primers/06-go.md), both in *Programming Languages*, and it isolates the upstream half of the [tracer gateway](../../ai-platform-engineering/12-gateway/00-streaming-proxy.md).

The solution is three files. [`engineclient.go`](../../../../course/solutions/primers/craft.20/engineclient.go) is the kata: the gateway's client of the engine API (`Client`, `Forward`, `NewRequest`, `ReadCompletion`, `ReadStream`, `Tokenize`, `APIError`, `ErrTruncated`). [`pacts/gateway-engine.json`](../../../../course/solutions/primers/craft.20/pacts/gateway-engine.json) is the **pact**: what the gateway relies on, in nine interactions. [`consumer_test.go`](../../../../course/solutions/primers/craft.20/consumer_test.go) holds the consumer tests, driven by mocks generated from the pact. The provider side the pact is verified against is the engine tier of [`openai-subset.v1.yaml`](../../../../course/contracts/openapi/openai-subset.v1.yaml). The tests run with `go test ./...` in a module that contains the kata, the pact, and a small pact kit (`pact.Load`, `Find`, `Serve`, `Calls`).

> **Learning objectives**
> - Define consumer, provider, interaction, pact, mock provider, and provider verification.
> - Explain why a mock provider must check the consumer's request as well as replay the response.
> - Build an upstream request that strips secrets and internal headers and sets the gateway's own.
> - Read an SSE stream that ends in exactly one of three ways and tell them apart.
> - Verify a pact statically against the provider's published OpenAPI contract.

## Consumers, providers, and pacts

A contract test checks the agreement between two services without running both. The vocabulary, with what each term means here:

| Term | Definition | Here |
|---|---|---|
| **provider** | the service that answers | the engine (`tl-serve`), engine tier of the API |
| **consumer** | the service that calls it and depends on the answers | the gateway |
| **interaction** | one request the consumer sends and the response it expects | "a streamed chat with usage" |
| **pact** | the consumer's list of interactions, as data | `pacts/gateway-engine.json` |
| **mock provider** | a server generated from one interaction that checks the request and replays the response | `pact.Serve` (the pact kit, `pact/pact.go`) |
| **provider verification** | replaying the pact against the provider (or its published contract) | the static check described below |

The idea comes from Pact and from Robinson's consumer-driven contracts. The consumer's tests produce or use the pact, and the provider's build verifies it. Neither side runs the other. A change that breaks an interaction fails the side that changed, at build time, with the interaction's name in the message. It is **consumer-driven** because the consumer writes down what it actually relies on, so the provider knows which promises it can never break and which fields nobody reads. A field no pact mentions can change freely, because no consumer reads it.

An interaction has a **request side** that the consumer must satisfy and a **response side** that the provider must satisfy:

```json
{"description": "a streamed chat with usage",
 "request":  {"method": "POST", "path": "/v1/chat/completions",
              "headers": {"X-TL-Priority": "5", "Accept": "text/event-stream"},
              "absentHeaders": ["Authorization"],
              "body": {"model": "smol", "stream": true, "stream_options": {"include_usage": true}}},
 "response": {"status": 200, "headers": {"Content-Type": "text/event-stream"},
              "events": ["data: {...}\n\n", ": ping\n\n", "...", "data: [DONE]\n\n"], "splitEvents": true}}
```

The kit's matching rules are: method and path exactly; each listed header equal (names case-insensitive); each `absentHeaders` name absent; and the body as a **subset**, meaning every key of the pact body is present with an equal value, recursively, and extra keys are allowed. A subset is deliberate: the pact lists what the consumer relies on, not everything it happens to send. The response is replayed as written, one event per flush, with each event split into two writes when `splitEvents` is true, which tests that the reader reassembles events across reads.

The mock is therefore **two-sided**: it replays the provider's answer and it **checks the consumer's request**. That second half is how a forwarded API key or a client-chosen priority fails a test.

## The request side: what the gateway must and must not send

The gateway's client builds every upstream request in `NewRequest`, and its rules come from `openai-subset.v1.yaml` and from the gateway's job:

| Rule | Why |
|---|---|
| no `Authorization` | the engine tier has no auth; a key forwarded upstream ends up in engine logs |
| `X-TL-Priority` from the **key**, never from the client | internal header (case `priority.internal`): a client must not raise its own priority |
| no client `X-TL-*` header at all (`X-TL-KV-Handle`) | a forged KV handle would resume someone else's request |
| `X-Request-Id` and the gateway's `traceparent` | one id and one trace across both hops, as in [One trace: gateway to engine in Jaeger](../../systems/04-observability/00-one-trace.md) |
| `stream_options.include_usage: true` on every stream | the gateway meters streams, and the engine sends usage only when asked |

Each of these is a line in some interaction's request side. The mock reports a broken one when the test ends, so a test that never asserts on headers still fails when the client leaks a key.

## The response side: everything the gateway must survive

A stream read by `ReadStream` ends in exactly one of three ways, and the gateway must tell them apart. `data: [DONE]` means success. An error event `data: {"error": {...}}` after some chunks becomes an `*APIError` with status 200, because the HTTP status was sent long ago. The connection closing early becomes `ErrTruncated`, and the gateway must never bill, cache, or forward a short stream as a success. Along the way the SSE grammar allows things a naive reader breaks on: comment lines (`: ping` every 15 s while idle), `data:` without the space, and an event split across two network reads. The usage chunk is the one with `choices: []`.

## Provider verification against the published contract

A pact is only worth something if the provider keeps it. Classic Pact verifies the provider by replaying the pact against a running provider. Here the provider publishes its contract as OpenAPI, so the check verifies the pact **statically**. Each request path and method must exist on the engine tier; request bodies must fit the request schema (partially, since a pact lists only what it relies on); each response status must be declared for the path; each JSON body must validate against its schema, each stream event against the `x-tl-chunk` schema (or the `Error` schema for an error event); and a 429 must carry `Retry-After` in seconds. So a pact cannot expect a chunk shape or a status the engine never promised. The gateway chapter closes the loop dynamically: its conformance run checks the same contract against the real engine.

## How a contract suite is judged

The suite is judged the way [TDD, unit tests, and mutation testing](01-tdd-unit-tests-and-mutation-grading.md) and [Mutation testing in depth](05-mutation-testing-in-depth.md) judge every suite: take a correct client, plant one fault, run the tests and pact, and count the faults the tests catch. First the tests must pass on the correct client, since a test that rejects a correct client measures nothing. There are twenty faults: sixteen semantic ones from the pitfalls listed in [What can go wrong](#what-can-go-wrong), all required, and four automatic ones. The bar is a score of at least 0.80 with every semantic fault caught.

## Working one interaction by hand

Take the interaction shown earlier with these six events:

| # | Event | What the reader does |
|---|---|---|
| 1 | `data: {..."choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}` | chunk 1: role, empty content |
| 2 | `: ping` | a comment: ignored |
| 3 | `data: {...,"delta":{"content":"Once"},"finish_reason":null}` | chunk 2: content `Once` |
| 4 | `data: {...,"delta":{"content":" upon"},"finish_reason":"length"}` | chunk 3: content ` upon`, finish `length` |
| 5 | `data: {...,"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":30,"total_tokens":42}}` | the usage chunk: not a content chunk |
| 6 | `data: [DONE]` | success |

The summary is 3 content chunks, text `Once upon`, finish reason `length`, usage $12 + 30 = 42$ tokens, and no error. Every event arrives in two halves (`splitEvents`), so after the first read of event 1 the reader holds `data: {"id":"c","obj` and must wait for the rest.

On the request side, the gateway forwarded a client request that carried `Authorization: Bearer tl_...`, `X-TL-Priority: 99`, and `X-TL-KV-Handle: forged`, for a key whose priority is 5. The mock receives `X-TL-Priority: 5`, no `Authorization`, no `X-TL-KV-Handle`, and a body with `"stream_options":{"include_usage":true}` added: every line of the request side holds.

Provider verification of event 5 checks that `choices` is an array with 0 items (the schema allows 0 to 1) and that `usage` matches `{"$ref": "#/components/schemas/Usage"}`, three non-negative integers. Write `"total_tokens":"42"` and the event fails: a string is not an integer. `TestStreamWithUsage`, shown later, reads exactly this stream.

## The artifacts

The solution's files and what each holds:

| File | Holds |
|---|---|
| `primers/craft.20/engineclient.go` | the kata, `package craft20`: `Client{Base, HTTP}`, `Forward`, `NewRequest`, `ReadCompletion`, `ReadStream`, `Tokenize`, `APIError`, `ErrTruncated`, `Usage`, `Chunk`, `Summary` (the doc comments are the spec) |
| `primers/craft.20/pacts/gateway-engine.json` | the pact: `consumer: gateway`, `provider: engine`, at least six interactions with unique descriptions, covering a completion with usage, a stream with its usage chunk and `[DONE]` (requested with `include_usage`), a stream that fails with an error event, a 429 with `Retry-After`, a `/v1/tokenize` call, and a request side that pins `X-TL-Priority` and an absent `Authorization` |
| `primers/craft.20/consumer_test.go` | `package craft20_test`: each test loads the pact (`pact.Load(t, "pacts/gateway-engine.json")`), starts a mock for one interaction (`pact.Serve`), sends a realistic client request through `NewRequest` (hostile headers included), reads the answer, and asserts what the gateway needs |
| `primers/craft.20/go.mod`, `pact/pact.go` | `module craft20` and the pact kit |

A correct solution handles seven cases:

| Case | Kind of check | Why it matters downstream |
|---|---|---|
| the hand interaction passes provider verification; its usage is 42; a string `total_tokens` fails | unit | the consumer and the verifier agree on the rules |
| the pact names gateway and engine, has six or more interactions with unique descriptions and complete sides | unit | tests find interactions by name |
| the pact covers the six behaviours listed for it in the table above | unit | the gateway's real needs are on file |
| every interaction fits the engine tier of `openai-subset.v1.yaml` | conformance | a pact the engine can keep |
| the kata streams the hand example, sends the request-side headers, asks for usage, ends streams the three ways, accepts `data:` without a space, maps errors and `Retry-After`, reads both completion shapes, tokenizes, and tolerates a base URL with a trailing slash | conformance | the client is right before its tests are judged |
| the tests pass on the kata and on any other correct client (baseline A) | unit | a test that rejects correct code measures nothing |
| 20 planted faults in the client, one at a time: at least 80% caught, every semantic fault caught | fault | a contract test that misses a broken client protects nothing |

## Building it step by step

Eight observations shape the client, the pact, and the tests.

1. **The gateway's correctness depends on two things it does not own:** what the engine accepts and what it returns. So both sides must be written down in one place the gateway's tests read and the engine's contract checks. That place is the pact.
2. **A mock that only replays is half a mock.** The dangerous bugs here (a leaked key, a forged priority, a missing `include_usage`) are in what the gateway sends, and the gateway's own code never looks at its outgoing request. So the mock must assert the request: listed headers equal, `absentHeaders` absent, body as a subset.
3. **The request builder is the only place those rules can live.** If header filtering were spread over call sites, one would forget. So `NewRequest` is a single function that takes the client's headers and the key's facts (`Priority`, `RequestID`, `Traceparent`) and emits the upstream request.
4. **Filtering must be a deny list of a class, not of names.** Any `X-TL-*` header is internal by definition, so the test is a prefix match; a list of known internal names would let tomorrow's header through.
5. **Gateway-set headers must overwrite, not add.** `X-TL-Priority` is `Set` after the copy loop (and the copy loop already skips it), so the client's value can never survive, whether or not it sent one.
6. **A stream has three endings and only one is success.** So the reader returns `nil` only at `[DONE]`, an `*APIError` for an error event, and `ErrTruncated` at EOF before either. Any other mapping lets a truncated answer be billed.
7. **SSE is line-based with a blank-line terminator,** and network reads do not respect event boundaries. So the reader buffers lines with `bufio.Reader.ReadBytes('\n')`, accumulates `data:` lines, and dispatches only on the blank line. That one structure handles split events, comments, and `data:` with or without the space.
8. **The pact lists only what the gateway relies on.** A subset match on the body and only the headers that matter keep the engine free to add fields.

### The request builder

```go
// internal headers a client may never set (openai-subset.v1.yaml).
func internal(name string) bool {
	return strings.HasPrefix(strings.ToLower(name), "x-tl-")   // the whole class, by prefix
}

// hop-by-hop and gateway-only headers that never go upstream.
var dropped = map[string]bool{
	"authorization": true, "connection": true, "keep-alive": true, "proxy-authorization": true,
	"te": true, "trailer": true, "transfer-encoding": true, "upgrade": true, "content-length": true,
	"traceparent": true, "x-request-id": true,   // the gateway sets its own; the client's are dropped
}

func (c *Client) NewRequest(ctx context.Context, f Forward) (*http.Request, error) {
	body := f.Body
	var probe struct {
		Stream bool `json:"stream"`
	}
	if err := json.Unmarshal(body, &probe); err != nil {
		return nil, fmt.Errorf("the request body is not JSON: %w", err)
	}
	if probe.Stream {
		// Rewrite only stream_options.include_usage; every other key passes through
		// as raw JSON, so the gateway never reorders or reformats what it does not own.
		var m map[string]json.RawMessage
		if err := json.Unmarshal(body, &m); err != nil {
			return nil, err
		}
		opts := map[string]json.RawMessage{}
		if raw, ok := m["stream_options"]; ok && string(raw) != "null" {
			if err := json.Unmarshal(raw, &opts); err != nil {   // keep the client's other options
				return nil, err
			}
		}
		opts["include_usage"] = json.RawMessage("true")          // the meter needs usage on every stream
		raw, _ := json.Marshal(opts)
		m["stream_options"] = raw
		b, err := json.Marshal(m)
		if err != nil {
			return nil, err
		}
		body = b
	}
	// TrimRight: a base of "http://e/" must not produce "//v1/...".
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(c.Base, "/")+f.Path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	for k, vs := range f.Header {
		if internal(k) || dropped[strings.ToLower(k)] {
			continue
		}
		for _, v := range vs {
			req.Header.Add(k, v)
		}
	}
	req.Header.Set("Content-Type", "application/json")
	if probe.Stream {
		req.Header.Set("Accept", "text/event-stream")
	}
	req.Header.Set("X-TL-Priority", strconv.Itoa(f.Priority))   // from the key, always, by Set
	if f.RequestID != "" {
		req.Header.Set("X-Request-Id", f.RequestID)
	}
	if f.Traceparent != "" {
		req.Header.Set("traceparent", f.Traceparent)             // the gateway's span, not the client's
	}
	return req, nil
}
```

Each line maps to a row of the request-side rules and to a planted fault. Copying every header leaks `Authorization`; checking `X-TL-Priority` "only when absent" lets a client's 99 through; forgetting `include_usage` bills streams at zero; dropping `TrimRight` sends `//v1/chat/completions`. The client's `traceparent` and `X-Request-Id` are dropped rather than forwarded because the gateway's own values must be the ones the engine sees, so the two hops share one id and one trace.

### The stream reader

```go
func ReadStream(resp *http.Response, onChunk func(Chunk) error) (Summary, error) {
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode > 299 {          // an error before the stream began
		body, _ := io.ReadAll(resp.Body)
		return Summary{}, apiError(resp.StatusCode, body)
	}
	var sum Summary
	br := bufio.NewReader(resp.Body)
	var raw, data []byte                                          // the event so far: bytes as received, and its data
	for {
		line, err := br.ReadBytes('\n')                           // blocks until a whole line, across reads
		if len(line) > 0 {
			raw = append(raw, line...)
			s := bytes.TrimRight(line, "\r\n")
			switch {
			case len(s) == 0: // the blank line ends an event
				if len(data) > 0 {
					done, cerr := dispatch(&sum, raw, data, onChunk)
					if cerr != nil || done {
						return sum, cerr                          // [DONE]: nil; error event: *APIError
					}
				}
				raw, data = nil, nil
			case s[0] == ':': // a comment (": ping" keep-alives)
			default:
				if v, ok := bytes.CutPrefix(s, []byte("data:")); ok {
					if len(data) > 0 {
						data = append(data, '\n')                 // multi-line data joins with \n (SSE spec)
					}
					data = append(data, bytes.TrimPrefix(v, []byte(" "))...)  // the space is optional
				}
			}
		}
		if err != nil {
			if errors.Is(err, io.EOF) {
				return sum, ErrTruncated                          // EOF before [DONE] or an error event
			}
			return sum, err
		}
	}
}
```

The order matters. The `len(line) > 0` branch runs before the error check, so a last line without a trailing newline is still accumulated; but it cannot be dispatched without its blank line, so the stream still ends as `ErrTruncated`. That is right: an event is not complete until the blank line arrives.

`dispatch` turns one complete event into the summary:

```go
func dispatch(sum *Summary, raw, data []byte, onChunk func(Chunk) error) (done bool, err error) {
	if bytes.Equal(data, []byte("[DONE]")) {
		return true, nil                                          // the only success
	}
	var ej errorJSON
	if json.Unmarshal(data, &ej) == nil && ej.Error != nil {
		return true, apiError(http.StatusOK, data)                // Status 200: headers went out long ago
	}
	// ... unmarshal the chunk: model, choices[0].delta or .text, finish_reason, usage
	ch := Chunk{Raw: append([]byte(nil), raw...), Model: cj.Model, Usage: cj.Usage.usage()}  // copy: raw is reused
	if len(cj.Choices) > 0 {                                      // the usage chunk has choices: [] and is not counted
		// ... role, content, finish reason
		sum.Chunks++
	}
	if sum.Model == "" {
		sum.Model = cj.Model
	}
	if ch.Usage != nil {
		sum.Usage = ch.Usage                                      // what the meter bills
	}
	if onChunk != nil {
		if err := onChunk(ch); err != nil {
			return true, err                                      // the caller can stop the stream
		}
	}
	return false, nil
}
```

Trace the hand-worked stream through it: events 1, 3, and 4 have one choice each (3 chunks), event 2 is a comment, event 5 has `choices: []` and sets `Usage` to $12 + 30 = 42$, and event 6 returns `done` with no error. `ReadCompletion` handles the non-streamed case the same way: a non-2xx becomes an `*APIError`, and on a 429 the `Retry-After` header (seconds) becomes `RetryAfter`. `Tokenize` posts `{model, text}` to `/v1/tokenize` and returns `len(ids)`, rejecting an answer with no `ids` field rather than reading it as zero tokens. The full kata is [`engineclient.go`](../../../../course/solutions/primers/craft.20/engineclient.go).

### The pact and the tests

The pact has nine interactions: a chat completion, a streamed chat with usage (the hand-worked one), a stream that fails after its first token, a stream cut off before `[DONE]`, a stream with no space after `data:`, the engine's admission queue full (429, `Retry-After: 2`), an unknown model (404), a text completion, and a token count. The two happy-path chat interactions pin the full request side:

```json
"request": {"method": "POST", "path": "/v1/chat/completions",
            "headers": {"Content-Type": "application/json", "X-TL-Priority": "5", "X-Request-Id": "req-7",
                        "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-b7ad6b7169203331-01",
                        "Accept": "text/event-stream"},
            "absentHeaders": ["Authorization", "X-TL-KV-Handle"],
            "body": {"model": "smol", "stream": true, "stream_options": {"include_usage": true}}}
```

The failure interactions pin only `"body": {"stream": true}`. They are about the response, and over-specifying their request would make every request-side change break five tests at once instead of one.

Every test sends the same hostile client headers, so every request-side rule is checked in every test whose interaction lists it:

```go
// clientHeaders is what a client sends the gateway, hostile parts included:
// its own key, a priority it chose, a KV handle it forged.
func clientHeaders() http.Header {
	h := http.Header{}
	h.Set("Authorization", "Bearer tl_abcdefghijkl_0123456789abcdefghijklmnopqrstuv")
	h.Set("X-TL-Priority", "99")
	h.Set("X-TL-KV-Handle", "forged")
	h.Set("Content-Type", "application/json")
	return h
}

func forward(path, body string) craft20.Forward {
	return craft20.Forward{Path: path, Body: []byte(body), Header: clientHeaders(), Priority: 5,
		RequestID: "req-7", Traceparent: traceparent}
}
```

The streamed test reads the hand-worked stream:

```go
func TestStreamWithUsage(t *testing.T) {
	m := pact.Serve(t, contract(t).Find(t, "a streamed chat with usage"))  // fails the test on a bad request
	resp := do(t, m.URL, forward("/v1/chat/completions", `{"model":"smol","stream":true,"messages":[{"role":"user","content":"Once"}]}`))
	var text string
	sum, err := craft20.ReadStream(resp, func(c craft20.Chunk) error { text += c.Content; return nil })
	if err != nil {
		t.Fatal(err)
	}
	if text != "Once upon" || sum.Chunks != 3 || sum.FinishReason != "length" || sum.Model != "smol-135m@v3" {
		t.Fatalf("text %q, summary %+v", text, sum)
	}
	if sum.Usage == nil || *sum.Usage != (craft20.Usage{PromptTokens: 12, CompletionTokens: 30, TotalTokens: 42}) {
		t.Fatalf("usage %+v", sum.Usage)
	}
}
```

The body the test sends has no `stream_options`, and the pact expects `include_usage: true`. So the mock fails this test if `NewRequest` forgets to add it, and the assertion on `Usage` fails if the reader drops the `choices: []` chunk. The failure tests are as precise about how the stream ended:

```go
func TestStreamFailureIsAnError(t *testing.T) {
	// ...
	var ae *craft20.APIError
	if !errors.As(err, &ae) || ae.Status != 200 || ae.Code != "internal_error" || ae.Type != "server_error" || n != 2 {
		t.Fatalf("err %v (%+v), %d chunks before it", err, ae, n)   // n: the chunks before the error still arrive
	}
}

func TestTruncatedStream(t *testing.T) {
	// ...
	if !errors.Is(err, craft20.ErrTruncated) {
		t.Fatalf("err %v, want ErrTruncated", err)
	}
}
```

The remaining tests each cover one more behavior. `TestChatCompletion` sends to `m.URL + "/"`, which is how the trailing-slash rule is covered without a test of its own. `TestQueueFull` asserts `RetryAfter == 2*time.Second`, `TestUnknownModel` that a null `param` reads as `""`, `TestTextCompletion` the `choices[0].text` shape (with `Priority: 0`, pinned by its interaction as `"0"`), and `TestTokenize` that four ids are four tokens. The full tests are in [`consumer_test.go`](../../../../course/solutions/primers/craft.20/consumer_test.go) and the full pact is [`gateway-engine.json`](../../../../course/solutions/primers/craft.20/pacts/gateway-engine.json).

## Why this design

The tests are cheap. Each one starts one in-process `httptest` server and makes one request: milliseconds. Provider verification is a static schema check: no engine, no GPU, no network.

Contract tests beat end-to-end tests at this seam. An end-to-end run needs a model loaded and both services up, and a failure does not say which side broke. The pact pins the seam; the gateway chapter's conformance run against the live engine is kept as the one slow check.

Consumer-driven beats provider-published alone. Testing the gateway directly against the OpenAPI document would check shapes but not the gateway's actual dependencies (that it needs usage on streams, that it must never forward a key). The pact records those, and lets the provider see which fields are load-bearing.

A subset body match beats an exact one. Exact matching would break the pact on every harmless new field the gateway sends; subset matching pins exactly what is relied on. The accepted cost is that an unexpected extra field is not caught, which is the job of the header deny list and the provider's schema.

`bufio` line reading beats a hand-rolled byte scanner. `ReadBytes('\n')` already reassembles lines across reads, and on top of it the SSE grammar is a four-way switch.

## What can go wrong

Each row is a fault planted in a correct client to judge the tests:

| Pitfall | Symptom in production | How the solution avoids it |
|---|---|---|
| 1. copying every client header upstream | the client's API key in engine logs | `dropped` includes `authorization`; the pact lists `Authorization` in `absentHeaders` |
| 2. letting the client's `X-TL-*` through, or setting the priority only when absent | a client raises its own priority or forges a KV handle | `internal` drops the whole `x-tl-` class, then `Set("X-TL-Priority", ...)` from the key |
| 3. no `include_usage`, or dropping the `choices: []` chunk | every stream is billed zero tokens | `NewRequest` adds it to every stream; `dispatch` keeps `Usage` from any chunk |
| 4. treating the end of the body as the end of the stream | a truncated answer is billed, cached, and shown as complete | only `[DONE]` returns `nil`; EOF returns `ErrTruncated` |
| 5. reading an error event as an ordinary chunk | a failed stream looks like a short success | `dispatch` checks for `{"error": ...}` before parsing a chunk |
| 6. a strict SSE reader | `: ping` keep-alives or `data:` without a space break the stream | a comment case in the switch; `TrimPrefix` of one optional space |
| 7. a non-2xx parsed as a success, or the code and `Retry-After` dropped | 429s look like empty answers; failover cannot back off | status checked first; `apiError` keeps type, code, param; `RetryAfter` from the header |
| 8. losing `X-Request-Id` or `traceparent` on the upstream hop | logs and traces stop at the gateway | both `Set` from `Forward`, pinned in the pact |
| 9. the wrong field name in `/v1/tokenize` | every TPM cost is wrong or a 400 | `{model, text}` matched by the "a token count" interaction's body |
| 10. a base URL with a trailing slash | the engine receives `//v1/chat/completions` | `strings.TrimRight(c.Base, "/")`; `TestChatCompletion` uses a trailing slash |
| 11. a pact that asserts nothing about requests | every request-side fault survives: the mock checks only what the pact lists | the happy-path interactions pin headers, absent headers, and the body |
| 12. a pact the provider never promised | green consumer tests, a broken production call | every interaction is verified against the engine tier of `openai-subset.v1.yaml` |

> **Warning:** A mock that only replays responses lets every request-side fault survive, including the leaked API key. The assertions that matter most live in the pact's request side, not in the Go test.

## Where this leads

This section reused judging tests by planted faults and reading survivors from [TDD, unit tests, and mutation testing](01-tdd-unit-tests-and-mutation-grading.md) and [Mutation testing in depth](05-mutation-testing-in-depth.md), and isolated the upstream request of the [tracer gateway](../../ai-platform-engineering/12-gateway/00-streaming-proxy.md). From here on, the [gateway chapter](../../ai-platform-engineering/12-gateway/README.md) of *AI Platform Engineering* and the [serving chapter](../../build-an-llm/p10-serving/README.md) of *Build Your Own LLM System* use contract tests:

- The meter's `include_usage` in [The usage ledger and admin API](../../ai-platform-engineering/12-gateway/07-usage-ledger-and-admin-api.md) and the router's failover on 429 in [Worker registry, routing, cascades, failover](../../ai-platform-engineering/12-gateway/05-routing.md) are the request and response sides pinned here.
- The gateway chapter's conformance run verifies the provider side live against the real engine.
- [Interface migration: API v1 to v2](../07-maintenance/02-api-v1-to-v2.md) adds a new pact version, verified against `openai-subset.v2.yaml` while v1 is still served.

Each piece has a production equivalent:

- The pact kit `pact/pact.go` corresponds to Pact (pact-go, pact-js), which adds matchers by type and regex, provider states, and a broker that stores pacts and records which versions verified them (`can-i-deploy`); see the [Pact docs](https://docs.pact.io/).
- Static provider verification corresponds to Specmatic and [Schemathesis](https://schemathesis.readthedocs.io/), which generate contract tests from OpenAPI and fuzz a live provider with properties.
- The mock provider corresponds to WireMock and [Prism](https://docs.stoplight.io/docs/prism/), which add record and replay of real traffic and an OpenAPI-driven mock server.

### Further reading

- [Pact documentation: how Pact works](https://docs.pact.io/getting_started/how_pact_works).
- Ian Robinson, ["Consumer-Driven Contracts: A Service Evolution Pattern"](https://martinfowler.com/articles/consumerDrivenContracts.html).
- Martin Fowler, ["ContractTest"](https://martinfowler.com/bliki/ContractTest.html).

## Key terms

- **Contract test**: a test of the agreement between two services that runs neither against the other.
- **Consumer / provider**: the service that calls and depends on the answers, and the service that answers.
- **Interaction**: one request the consumer sends and the response it expects.
- **Pact**: the consumer's list of interactions, kept as data.
- **Mock provider**: a server generated from one interaction that checks the request and replays the response.
- **Provider verification**: checking every interaction against the provider or its published contract.
- **Subset match**: a body match that requires every key the pact lists and allows extra keys.

## Summary

- A contract test checks the agreement between two services without running both: the consumer is tested against a mock of the provider, and the provider is verified against the same document.
- Consumer-driven means the consumer writes down what it actually relies on, so the provider knows which promises it can never break and which fields nobody reads.
- The mock is two-sided: it replays the provider's answer and checks the consumer's request, which is how a forwarded API key or a client-chosen priority fails a test.
- Provider verification checks every interaction against `openai-subset.v1.yaml`, so a pact cannot expect a chunk shape or a status the engine never promised.
- The suite is judged by mutation testing: it catches at least 80% of 20 planted faults in the client, and every planted pitfall.

## Review questions

1. Why does `NewRequest` drop the client's `traceparent` instead of forwarding it?

   <details><summary>Answer</summary>The upstream request is a child of the gateway's span, not of the client's. The gateway sets its own `traceparent` (from `Forward.Traceparent`), so the engine's spans hang off the gateway's and the trace covers both hops. Forwarding the client's would skip the gateway's span, and a client could inject arbitrary trace ids.</details>

2. A planted fault sets `X-TL-Priority` only when the client did not send one. Which test catches it, and how, if no test asserts on headers?

   <details><summary>Answer</summary>Every test sends `X-TL-Priority: 99` in `clientHeaders`, and the interactions pin `X-TL-Priority: 5`. The mock compares the received header with the pact and fails the test at its end. The assertion is in the pact, not in the Go test.</details>

3. Why does an error event produce an `*APIError` with `Status` 200?

   <details><summary>Answer</summary>The engine committed to 200 when it sent the stream's headers; the failure happened later. Reporting 200 is accurate, and callers distinguish the case by the error type and code, not the status. It must still be an error, never a short success.</details>

4. Why does a stream whose last line lacks a trailing blank line end as `ErrTruncated`?

   <details><summary>Answer</summary>An SSE event is complete only at the blank line. Without it the reader cannot know the event was whole, so it never dispatches `[DONE]` and the EOF path returns `ErrTruncated`. Treating a partial last event as done would bill a truncated stream.</details>

5. Why do the failure interactions pin so little of the request?

   <details><summary>Answer</summary>Each interaction should fail for one reason. The request rules are pinned in the two happy-path chat interactions; repeating them in the failure ones would make one header bug fail many tests, and hide which response-side behaviour each test is about.</details>
