<!-- ol:module craft.20 -->
# Contract tests (R6): consumer-driven tests for gateway to engine

## Overview

| | |
|---|---|
| **Module** | `craft.20` · practice · Go · Pass 7 · 4 to 6 h |
| **You build** | `primers/craft.20/engineclient.go` (a kata: the gateway's client of the engine API, written by `ol start`), `primers/craft.20/pacts/gateway-engine.json` (your **pact**: what the gateway relies on), and `primers/craft.20/consumer_test.go` (your consumer tests, driven by mocks generated from the pact) |
| **Contract** | the provider side your pact is verified against: the engine tier of [`course/contracts/openapi/openai-subset.v1.yaml`](../../course/contracts/openapi/openai-subset.v1.yaml) |
| **Tests** | `course/tests/craft.20/check`: provider verification of your pact, the course's contract suite (`course/tests/go/craft_20/`) on your kata, then the grade of your tests by 20 planted faults (section 4) |
| **Needs** | reading: [`craft.03`](01-tdd-unit-tests-and-mutation-grading.md) how tests are graded · [`craft.07`](05-mutation-testing-in-depth.md) reading survivors · [`lang.05`](../12-language-and-tool-primers/05-http-and-sse.md) SSE on the wire · [`lang.06`](../12-language-and-tool-primers/06-go.md) Go and `httptest` · `gw.00` the proxy whose upstream half the kata isolates |
| **Used by** | no call site (a practice): rung R6 grades contract tests in the gateway (`gw.*`) and engine (`L10.*`) modules from here on |
| **Milestone** | MS-gateway (its conformance run is the provider side of the same contract) |
| **Optional depth** | [Pact documentation: how Pact works](https://docs.pact.io/getting_started/how_pact_works) (free); Ian Robinson, ["Consumer-Driven Contracts: A Service Evolution Pattern"](https://martinfowler.com/articles/consumerDrivenContracts.html) (free); Martin Fowler, ["ContractTest"](https://martinfowler.com/bliki/ContractTest.html) (free) |

## Key Takeaways

- A **contract test** checks the agreement between two services without running both: the **consumer** (your gateway) is tested against a mock of the **provider** (the engine), and the provider is verified against the same document.
- **Consumer-driven**: the consumer writes down what it actually relies on (a **pact**), so the provider knows which promises it can never break and which fields nobody reads.
- The mock is **two-sided**: it replays the provider's answer and it **checks the consumer's request**, which is how a forwarded API key or a client-chosen priority fails a test.
- A pact is only worth something if the provider keeps it: **provider verification** checks every interaction against the provider's published contract (`openai-subset.v1.yaml`), so a pact cannot expect a chunk shape or a status the engine never promised (`test_pact_matches_the_provider_contract`).
- The grade is the R6 rung: your tests must catch at least 80% of 20 planted faults in the client, and every planted pitfall (`test_your_tests_catch_the_planted_faults`).

## How to work this chapter

```bash
ol start craft.20                         # writes primers/craft.20/engineclient.go with stub bodies
ol tests craft.20                         # read the test catalog first
ol check craft.20                         # first run: writes go.mod and pact/pact.go, then fails (no pact yet)
# write pacts/gateway-engine.json and consumer_test.go (section 4); run them yourself:
(cd primers/craft.20 && go test ./...)
ol check craft.20                         # verifies the pact, then grades your tests by planted faults
```

Write the pact and the tests before the kata: against the stub every test must fail. `ol check` never shows a planted fault's code; a survivor prints only its one-line description, or "a planted pitfall".

---

## 1. Why now

Your gateway now forwards to your engine across a process boundary, and both keep changing: Pass 7 adds metering (`gw.07`), routing (`gw.05`), and a whole new engine (`L10.1` to `L10.9`), and Pass 11 migrates the API to v2. The gateway's unit tests run against a fake engine you wrote in the same test file, so they test the gateway against your **belief** about the engine. When the engine renames a field, adds a comment line to its stream, or starts answering 429, every unit test stays green and production breaks. The two other options are worse: an end-to-end test that starts both services is slow, flaky, and says nothing about which side broke. A **contract test** sits between them: the belief is written down once, in a file, the gateway is tested against it, and the file is checked against the engine's published contract. This module teaches it on the one seam every later pass changes: gateway to engine.

## 2. Principles

### 2.1 Consumers, providers, and pacts

| Term | Definition | Here |
|---|---|---|
| **provider** | the service that answers | the engine (`tl-serve`), engine tier of the API |
| **consumer** | the service that calls it and depends on the answers | the gateway |
| **interaction** | one request the consumer sends and the response it expects | "a streamed chat with usage" |
| **pact** | the consumer's list of interactions, as data | `pacts/gateway-engine.json` |
| **mock provider** | a server generated from one interaction that checks the request and replays the response | `pact.Serve` (the course's kit, `pact/pact.go`) |
| **provider verification** | replaying the pact against the provider (or its published contract) | `test_pact_matches_the_provider_contract` |

The idea, from Pact and Robinson's consumer-driven contracts: the consumer's tests produce or use the pact, and the provider's build verifies it. Neither side runs the other. A change that breaks an interaction fails the side that changed, at build time, with the interaction's name in the message. A field no pact mentions can change freely, because no consumer reads it.

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

The kit's matching rules: method and path exactly; each listed header equal (names case-insensitive); each `absentHeaders` name absent; the body as a **subset** (every key of the pact body present with an equal value, recursively; extra keys allowed). A subset is deliberate: the pact lists what the consumer relies on, not everything it happens to send. The response is replayed as written, one event per flush, each event split into two writes when `splitEvents` is true, which tests that your reader reassembles events across reads.

### 2.2 The request side: what the gateway must and must not send

The gateway's client builds every upstream request (`NewRequest`), and the rules come from `openai-subset.v1.yaml` and the gateway's job:

| Rule | Why |
|---|---|
| no `Authorization` | the engine tier has no auth; a key forwarded upstream ends up in engine logs |
| `X-TL-Priority` from the **key**, never from the client | internal header (case `priority.internal`): a client must not raise its own priority |
| no client `X-TL-*` header at all (`X-TL-KV-Handle`) | a forged KV handle would resume someone else's request |
| `X-Request-Id` and the gateway's `traceparent` | one id and one trace across both hops (`obs.00`) |
| `stream_options.include_usage: true` on every stream | the gateway meters streams, and the engine sends usage only when asked (`gw.07`) |

Each of these is a line in some interaction's request side. The mock reports a broken one when the test ends, so a test that never asserts on headers still fails when the client leaks a key.

### 2.3 The response side: everything the gateway must survive

A stream (`ReadStream`) ends in exactly one of three ways, and the gateway must tell them apart: `data: [DONE]` (success), an error event `data: {"error": {...}}` after some chunks (an `*APIError` with status 200: the HTTP status was sent long ago), or the connection closing early (`ErrTruncated`: the gateway must never bill, cache, or forward a short stream as a success). Along the way the SSE grammar allows things a naive reader breaks on: comment lines (`: ping` every 15 s while idle), `data:` without the space, and an event split across two network reads. The usage chunk is the one with `choices: []`.

### 2.4 Provider verification against the published contract

Classic Pact verifies the provider by replaying the pact against a running provider. Here the provider publishes its contract as OpenAPI, so the check verifies the pact **statically**: each request path and method exists on the engine tier, request bodies fit the request schema (partially: a pact lists only what it relies on), each response status is declared for the path, each JSON body validates against its schema, each stream event against the `x-tl-chunk` schema (or the `Error` schema for an error event), and a 429 carries `Retry-After` in seconds. MS-gateway closes the loop dynamically: `ol conform openapi:v1 --target engine` runs the same contract against your real engine.

### 2.5 How R6 grades you

Mutation grading (`craft.03`, `craft.07`) on the contract: the check builds a scratch module with the **course's** client plus one planted fault, runs **your** tests and pact, and counts the faults your tests catch. First your tests must pass on the course's correct client (a test that rejects a correct client grades nothing). Twenty faults: sixteen semantic ones from the pitfalls below, all required, and four automatic ones. Pass: a score of at least 0.80 and every required fault caught.

## 3. Worked example by hand

Take the interaction of 2.1 with these six events:

| # | Event | What the reader does |
|---|---|---|
| 1 | `data: {..."choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}` | chunk 1: role, empty content |
| 2 | `: ping` | a comment: ignored |
| 3 | `data: {...,"delta":{"content":"Once"},"finish_reason":null}` | chunk 2: content `Once` |
| 4 | `data: {...,"delta":{"content":" upon"},"finish_reason":"length"}` | chunk 3: content ` upon`, finish `length` |
| 5 | `data: {...,"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":30,"total_tokens":42}}` | the usage chunk: not a content chunk |
| 6 | `data: [DONE]` | success |

So the summary is: 3 content chunks, text `Once upon`, finish reason `length`, usage $12 + 30 = 42$ tokens, no error. Every event arrives in two halves (`splitEvents`), so after the first read of event 1 the reader holds `data: {"id":"c","obj` and must wait for the rest.

On the request side, the gateway forwarded a client request that carried `Authorization: Bearer tl_...`, `X-TL-Priority: 99`, and `X-TL-KV-Handle: forged`, for a key whose priority is 5. The mock receives `X-TL-Priority: 5`, no `Authorization`, no `X-TL-KV-Handle`, and a body with `"stream_options":{"include_usage":true}` added: every line of the request side holds.

Provider verification of event 5: `choices` is an array with 0 items (the schema allows 0 to 1), `usage` matches `{"$ref": "#/components/schemas/Usage"}` (three non-negative integers). Write `"total_tokens":"42"` and the event fails: a string is not an integer. `test_hand_example_interaction` checks this interaction and that mutation; `TestCourseHandExampleStream` reads the same stream with your kata.

## 4. The artifact and its check

| File | Holds |
|---|---|
| `primers/craft.20/engineclient.go` | the kata, `package craft20`: `Client{Base, HTTP}`, `Forward`, `NewRequest`, `ReadCompletion`, `ReadStream`, `Tokenize`, `APIError`, `ErrTruncated`, `Usage`, `Chunk`, `Summary` (the doc comments are the spec) |
| `primers/craft.20/pacts/gateway-engine.json` | your pact: `consumer: gateway`, `provider: engine`, at least six interactions with unique descriptions, covering a completion with usage, a stream with its usage chunk and `[DONE]` (requested with `include_usage`), a stream that fails with an error event, a 429 with `Retry-After`, a `/v1/tokenize` call, and a request side that pins `X-TL-Priority` and an absent `Authorization` |
| `primers/craft.20/consumer_test.go` | `package craft20_test`: each test loads the pact (`pact.Load(t, "pacts/gateway-engine.json")`), starts a mock for one interaction (`pact.Serve`), sends a realistic client request through `NewRequest` (hostile headers included), reads the answer, and asserts what the gateway needs |
| `primers/craft.20/go.mod`, `pact/pact.go` | written by the first `ol check`: `module craft20` and the course's kit (the check always uses its own copy of the kit) |

A test reads like this:

```go
func TestStreamWithUsage(t *testing.T) {
	m := pact.Serve(t, pact.Load(t, "pacts/gateway-engine.json").Find(t, "a streamed chat with usage"))
	req, _ := (&craft20.Client{Base: m.URL}).NewRequest(ctx, craft20.Forward{Path: "/v1/chat/completions",
		Body: []byte(`{"model":"smol","stream":true}`), Header: clientHeaders(), Priority: 5, RequestID: "req-7"})
	resp, _ := http.DefaultClient.Do(req)
	sum, err := craft20.ReadStream(resp, nil)
	// assert err == nil, sum.Chunks == 3, sum.Usage == {12, 30, 42, 0}
}
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_interaction` | unit | section 3's interaction passes provider verification; its usage is 42; a string `total_tokens` fails | you and the verifier agree on the rules |
| `test_files_present` | unit | the kata, the pact, the tests; writes `go.mod` and the kit | |
| `test_pact_is_well_formed` | unit | gateway/engine, six or more interactions, unique descriptions, complete sides | tests find interactions by name |
| `test_pact_covers_what_the_gateway_relies_on` | unit | the six behaviours of the table above | the gateway's real needs are on file |
| `test_pact_matches_the_provider_contract` | conformance | every interaction fits the engine tier of `openai-subset.v1.yaml` | a pact your engine can keep |
| `test_your_kata_passes_the_course_contract_suite` | conformance | `course/tests/go/craft_20/` (below) on your kata | your client is right before your tests are graded |
| `test_your_tests_pass_on_your_kata` | unit | your tests on your kata | |
| `test_your_tests_pass_on_the_course_kata` | unit | baseline A: your tests accept a correct client | a test that rejects correct code grades nothing |
| `test_your_tests_catch_the_planted_faults` | fault | 20 faults, one at a time: score at least 0.80, every semantic fault caught | rung R6 |

The course's contract suite (Go, run on your kata): `TestCourseHandExampleStream` (section 3), `TestCourseRequestHeaders` (2.2), `TestCourseStreamAsksForUsage`, `TestCourseStreamEnds` (2.3), `TestCourseDataWithoutSpace`, `TestCourseErrorsAndRetryAfter`, `TestCourseCompletionShapes`, `TestCourseTokenize`, `TestCourseBaseWithTrailingSlash`.

## 5. Pitfalls

Each row is a fault the check plants in the course's client; your tests must catch it.

| Pitfall | Symptom in production | Caught by |
|---|---|---|
| 1. copying every client header upstream | the client's API key in engine logs | `TestCourseRequestHeaders` (mutant `s01`) |
| 2. letting the client's `X-TL-*` through, or setting the priority only when absent | a client raises its own priority or forges a KV handle | `TestCourseRequestHeaders` (mutants `s02`, `s15`) |
| 3. no `include_usage`, or dropping the `choices: []` chunk | every stream is billed zero tokens | `TestCourseStreamAsksForUsage`, `TestCourseHandExampleStream` (mutants `s04`, `s07`) |
| 4. treating the end of the body as the end of the stream | a truncated answer is billed, cached, and shown as complete | `TestCourseStreamEnds` (mutant `s05`) |
| 5. reading an error event as an ordinary chunk | a failed stream looks like a short success | `TestCourseStreamEnds` (mutant `s06`) |
| 6. a strict SSE reader | `: ping` keep-alives or `data:` without a space break the stream | `TestCourseHandExampleStream`, `TestCourseDataWithoutSpace` (mutants `s08`, `s09`) |
| 7. a non-2xx parsed as a success, or the code and `Retry-After` dropped | 429s look like empty answers; failover cannot back off | `TestCourseErrorsAndRetryAfter` (mutants `s11`, `s12`, `s16`) |
| 8. losing `X-Request-Id` or `traceparent` on the upstream hop | logs and traces stop at the gateway | `TestCourseRequestHeaders` (mutants `s10`, `s03`) |
| 9. the wrong field name in `/v1/tokenize` | every TPM cost is wrong or a 400 | `TestCourseTokenize` (mutant `s13`) |
| 10. a base URL with a trailing slash | the engine receives `//v1/chat/completions` | `TestCourseBaseWithTrailingSlash` (mutant `m04`) |
| 11. a pact that asserts nothing about requests | every request-side fault survives: the mock checks only what the pact lists | `test_your_tests_catch_the_planted_faults` |
| 12. a pact the provider never promised | green consumer tests, a broken production call | `test_pact_matches_the_provider_contract` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `craft.03`, `craft.07` | mutation grading and reading survivors |
| Back | `gw.00` | the proxy whose upstream request the kata isolates |
| Forward | `gw.07`, `gw.05` | the meter's `include_usage` and the router's failover on 429 are the request and response sides you pinned |
| Forward | MS-gateway | `ol conform openapi:v1 --target engine` verifies the provider side live |
| Forward | `craft.14` | the API v2 migration: a new pact version, verified against `openai-subset.v2.yaml` while v1 is still served |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `pact/pact.go` | Pact (pact-go, pact-js) | matchers by type and regex, provider states, a broker that stores pacts and records which versions verified them (`can-i-deploy`) | [Pact docs](https://docs.pact.io/) (free) |
| static provider verification | Specmatic, Schemathesis | contract tests generated from OpenAPI; property-based fuzzing of a live provider | [Schemathesis](https://schemathesis.readthedocs.io/) (free) |
| the mock provider | WireMock, Prism | record and replay of real traffic; an OpenAPI-driven mock server | [Prism](https://docs.stoplight.io/docs/prism/) (free) |
