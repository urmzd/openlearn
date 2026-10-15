<!-- ol:module gw.03 -->
# Rate limiting: RPM token bucket + TPM reserve/settle

## Overview

| | |
|---|---|
| **Module** | `gw.03` · build · Go · Pass 7 · 4 h |
| **You build** | `go/gateway/limit/limit.go`: `Limiter` (`New`, `Reserve`), `Reservation` (`Settle`, `Cancel`, `Status`), `RejectError`, `EstimateCost`, `PromptText`, `TokenizeCounter`, `SetHeaders`, `RetryAfterSeconds`, `Middleware` |
| **Contract** | 429 `rate_limit_error` / `rate_limit_exceeded`, `Retry-After`, and the `x-ratelimit-*` headers in [`course/contracts/openapi/openai-subset.v1.yaml`](../../../../course/contracts/openapi/openai-subset.v1.yaml); `rpm` and `tpm` of `KeyCreate` in [`course/contracts/openapi/admin.v1.yaml`](../../../../course/contracts/openapi/admin.v1.yaml) |
| **Tests** | `course/tests/go/gw_03/` (what they check: section 4); the concurrency test runs under the race detector |
| **Needs** | [`gw.01` server skeleton](../12-gateway/01-server-skeleton.md) (the chain and the `Exchange`), [`gw.02` API keys](01-api-keys-and-scopes.md) (the `Principal` and its limits) · reading: `S-M07d` [queueing and Little's law](../../math/07-probability-statistics/93-problem-set-d.md), the [practice drill go/07](../../../../practice/build/cloud/go/README.md) (a token bucket warm-up) |
| **Used by** | `gw.07`'s `gateway.Deps`, which puts `limit.Middleware` in the chain's `Limiter` slot; your composition root builds the `Limiter` and calls it |
| **Milestone** | MS-gateway |
| **Optional depth** | Tanenbaum, *Computer Networks*, the token bucket; the GCRA in the ATM Forum's traffic management spec; Stripe's *Scaling your API with rate limiters* (free) |

## Key Takeaways

- A **token bucket** holds at most $C$ tokens and refills at $C$ per minute; a request takes tokens if they are there, otherwise it waits exactly as long as the refill needs. A full bucket allows a burst of one minute's budget, never more (`TestBurstThenRefill`).
- Requests per minute and tokens per minute are two buckets per key. Token cost is unknown when a request starts, so the limiter **reserves** an estimate (prompt tokens plus `max_tokens`) and **settles** the actual usage when the response ends: unused tokens come back, overruns are charged (`TestHandExample`, `TestSettleChargesOverrun`).
- A 429 carries `Retry-After` in whole seconds rounded **up**; every authenticated response carries the five `x-ratelimit-*` headers (`TestMiddleware429Headers`).
- Exactly one of `Settle` and `Cancel` counts; a request that failed with a 5xx before any usage is cancelled, so clients do not pay for our outages (`TestCancelAndSettleOnce`, `TestMiddlewareCancelsOn5xx`).
- One mutex around every bucket decision: 1000 goroutines against a budget of 100 admit exactly 100 (`TestConcurrentReserve`).

## How to work this chapter

```bash
ol start gw.03          # writes go/gateway/limit/limit.go with stub bodies
ol tests gw.03          # read the test catalog first
ol check gw.03          # runs the course tests with -race
ol check gw.03 --ref-deps   # only if your gw.01 or gw.02 is not passing yet
ol diff  gw.03          # after passing: your code against the reference
```

Then build `limit.New(clock)` and `limit.TokenizeCounter{BaseURL: engineURL}` in your composition root; until `gw.07`'s `gateway.Deps` assembles the chain, plug `limit.Middleware(limiter, counter)` into the `Limiter` slot yourself.

---

## 1. Why now

Your gateway now knows who is calling (`gw.02`), but every caller can still send as fast as its network allows. One tenant's batch job at 50 requests per second fills the engine's queue (`L10.2`), and every other tenant's time to first token goes from milliseconds to seconds: the noisy-neighbor incident you will meet as drill `ops.09`. Counting requests is not enough for an LLM: one request with a 4000-token prompt and `max_tokens: 2000` costs a hundred times a short chat turn. This module gives each key two budgets, requests per minute (RPM) and tokens per minute (TPM), enforced before the request reaches the cache or an engine, and tells the client exactly when to come back.

## 2. Principles

### 2.1 The token bucket

| Symbol | Meaning | Type |
|---|---|---|
| $C$ | bucket capacity: the per-minute limit (`rpm` or `tpm` of the key) | integer |
| $L(t)$ | the bucket's level at time $t$ | real, may go negative after an overrun |
| $\rho = C / 60$ | refill rate per second | real |
| $n$ | the cost of a request in this bucket | integer |
| $w$ | the wait before $n$ tokens are available | seconds |

The bucket starts full, $L = C$. Between decisions it refills continuously and is capped: $L(t_2) = \min(C,\ L(t_1) + \rho\,(t_2 - t_1))$. A request costing $n$ is admitted when $L \ge n$, and then $L \leftarrow L - n$. Otherwise it must wait

$$w = \frac{n - L}{\rho} = \frac{(n - L) \cdot 60}{C}\ \text{seconds},$$

which is the `Retry-After` the client receives (in whole seconds, rounded up). Two properties follow. A key idle for an hour can burst at most $C$ at once, because of the cap. Over any long window, the admitted rate approaches $\rho$, which is the limit. Each key has its own pair of buckets, so keys never spend each other's budget; a key with limit 0 has no bucket (unlimited, per `admin.v1.yaml`).

Why not a fixed window ("at most 60 per calendar minute")? It admits 120 in two seconds across a minute boundary. Why not a sliding log of timestamps? It is exact but costs memory per request. The bucket is O(1) per key and smooth.

### 2.2 Reserve and settle

A request's token cost is prompt tokens plus completion tokens, and the completion length is only known when the stream ends. So the stage:

1. **Estimates** the cost: prompt tokens counted by the engine's tokenizer (`POST /v1/tokenize`, engine tier) over the prompt text (the completions `prompt`, or the chat contents joined by newlines; the chat template adds a few tokens this leaves out), plus `max_completion_tokens` or `max_tokens`, or 256 when neither is set. If the tokenizer is unreachable, it falls back to $\lceil \text{bytes} / 4 \rceil$ rather than failing the request.
2. **Reserves** that estimate in both buckets (1 request, $n$ tokens), or rejects with 429.
3. Runs the rest of the chain.
4. **Settles** with the actual usage the proxy put on the `Exchange` (`gw.04`): the difference between reserved and actual goes back into the bucket (capped at $C$) or, for an overrun, is taken from it, possibly below zero. A negative level is a debt the key's next requests wait off.

A reservation is settled or cancelled **once**: if the code settles and then a deferred cleanup cancels, the second call must do nothing, or the key gets a refund it never earned. If the request failed with a 5xx and no usage (no engine had capacity), the stage cancels: request and tokens both come back.

### 2.3 What the client sees

On every authenticated response (`openai-subset.v1.yaml`):

| Header | Value |
|---|---|
| `x-ratelimit-limit-requests` / `-tokens` | $C$ of each bucket |
| `x-ratelimit-remaining-requests` / `-tokens` | $\lfloor L \rfloor$ after this request's reservation, at least 0 |
| `x-ratelimit-reset-tokens` | time until the token bucket is full again, as a Go duration (`18s`) |

On a 429 the same headers plus `Retry-After: ⌈w⌉` seconds. Rounding down would invite the client back before the budget exists (and a $w$ of 0.4 s would become `Retry-After: 0`, "retry now"), so the value is $\lceil w \rceil$ and at least 1.

## 3. Worked example by hand

Key `k` has RPM 3 and TPM 1000; the clock starts at $t = 0$.

1. A request reserves 300 tokens (prompt 60, `max_tokens` 240). Request bucket $3 \to 2$, token bucket $1000 \to 700$. Headers: remaining requests 2, remaining tokens 700, reset-tokens $= 300 / (1000/60) = 18$ s.
2. It finishes having used 120 tokens. Settle refunds $300 - 120 = 180$: the token bucket is $700 + 180 = 880$.
3. Still at $t = 0$, a request needs 900 tokens: $880 < 900$, so $w = (900 - 880) \cdot 60 / 1000 = 1.2$ s. The answer is 429 with `Retry-After: 2` and remaining tokens 880.
4. At $t = 1.2$ s the token bucket has refilled $1000 \cdot 1.2 / 60 = 20$ tokens to exactly 900; the request is admitted and leaves 0. The request bucket refilled $3 \cdot 1.2/60 = 0.06$ to 2.06, and after this request holds 1.06: remaining requests 1.

This is `TestHandExample`.

## 4. The interface

```go
package limit // import "tinyllm/gateway/limit"

type Cost struct{ Requests, Tokens int }
type Limits struct{ RPM, TPM int } // 0 = unlimited
type Status struct {
	LimitRequests, RemainingRequests, LimitTokens, RemainingTokens int
	ResetTokens time.Duration
}
type RejectError struct { RetryAfter time.Duration; Status Status }
type Reservation interface { Settle(actual Cost); Cancel(); Status() Status }

func New(clock server.Clock) *Limiter
func (l *Limiter) Reserve(ctx context.Context, key string, lim Limits, c Cost) (Reservation, error)

type Counter interface { CountPrompt(ctx context.Context, model string, body []byte) (int, error) }
type TokenizeCounter struct { BaseURL string; Client *http.Client } // the engine's /v1/tokenize
func PromptText(body []byte) string
func EstimateCost(ctx context.Context, counter Counter, req *server.Request) Cost
func SetHeaders(h http.Header, s Status)
func RetryAfterSeconds(d time.Duration) int
func Middleware(l *Limiter, counter Counter) server.Middleware
```

The catalog sketched `Reserve(ctx, key, c)`; this one also takes the key's `Limits`, because limits live on the key record (`gw.02`) and change when an admin edits the key, so the limiter holds buckets, not key metadata (DEVIATIONS B93-04). Time comes only from the `Clock`, so the tests drive refill with a fake clock.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestHandExample` | unit | section 3, every number: remaining counts, reset 18 s, the settle refund, `RetryAfter` 1.2 s and `Retry-After` 2, admission at 1.2 s | you and the tests agree on the bucket |
| `TestBurstThenRefill` | unit | 60 at once, the 61st waits 1 s, one per second after, 10 idle minutes bank only 60 | the burst limit in the load report |
| `TestSettleChargesOverrun` | unit | reserve 100, use 700: the next 10 tokens wait 11 s | requests without `max_tokens` cannot dodge TPM |
| `TestCancelAndSettleOnce` | unit | cancel returns everything; a second cancel and a cancel after settle do nothing | no double refunds |
| `TestZeroMeansUnlimited` | boundary | limits of 0 never reject | the admin API's "0 = unlimited" |
| `TestOversizeCostRejected` | boundary | 1001 tokens against TPM 1000 is rejected even when the bucket is full; 1000 fits | one request cannot exceed a minute's budget |
| `TestKeysAreIndependent` | unit | key `a` exhausted, key `b` untouched | the noisy-neighbor drill |
| `TestConcurrentReserve` | property | 1000 goroutines, RPM 100: exactly 100 admitted, no race | correctness under load |
| `TestRetryAfterSeconds` | boundary | 0, 1 ms, 1.2 s, 59 s, 59.001 s | clients come back when the budget exists |
| `TestEstimateCost` | unit | prompt + max_tokens; `max_completion_tokens` wins; 256 default; bytes / 4 when the tokenizer fails | reservations close to the truth |
| `TestPromptText` | unit | prompt, chat contents, embeddings input | the text that is tokenized |
| `TestTokenizeCounter` | unit | posts `{model, text}` to `/v1/tokenize` and counts the ids | the production counter |
| `TestMiddleware429Headers` | conformance | headers on every 200; the third request at RPM 2 is 429 with `Retry-After: 30` and the error shape | conformance case `ratelimit.429` |
| `TestMiddlewareSettlesActualUsage` | unit | the next response's remaining tokens reflect the actual usage, not the estimate | TPM tracks real cost |
| `TestMiddlewareCancelsOn5xx` | unit | after a 503 without usage the one-request budget is back | our outages are free |
| `TestNoPrincipalPassesThrough` | boundary | no principal: no headers, no limit | the stage is safe before `gw.02` is plugged in |
| `TestMiddlewareKeysByKeyID` | unit | two keys of one tenant have separate budgets | a CI key cannot starve a developer's |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. refill without the cap | a key idle overnight sends a day's budget at once | `TestBurstThenRefill` (mutant `s01`) |
| 2. refilling per second instead of per minute, or rounding `Retry-After` down | limits 60 times too loose; clients retry into another 429 | `TestHandExample`, `TestRetryAfterSeconds` (mutants `s02`, `s06`) |
| 3. settle mistakes: refunding the whole reservation, never charging an overrun, both settle and cancel counting, settling the estimate | TPM drifts from real usage in either direction | `TestHandExample`, `TestSettleChargesOverrun`, `TestCancelAndSettleOnce`, `TestMiddlewareSettlesActualUsage` (mutants `s03`, `s04`, `s05`, `s11`) |
| 4. treating 0 as an empty budget, or charging requests that failed with a 5xx | unlimited keys are locked out; clients pay for our outages | `TestZeroMeansUnlimited`, `TestMiddlewareCancelsOn5xx` (mutants `s07`, `s10`) |
| 5. admitting a request larger than the limit because the bucket is full | one request takes more than a minute's tokens | `TestOversizeCostRejected` (mutant `s15`) |
| 6. no lock around a decision, or one bucket per tenant | under load more requests are admitted than the limit; keys drain each other | `TestConcurrentReserve`, `TestMiddlewareKeysByKeyID` (mutants `s08`, `s14`) |
| 7. headers only on 429 | clients cannot pace themselves before they are throttled | `TestMiddleware429Headers` (mutant `s09`) |
| 8. a bad estimate: 0 tokens when the tokenizer fails, `max_tokens` ignored | huge requests pass the reservation and overdraw afterwards | `TestEstimateCost` (mutants `s12`, `s13`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `gw.01` | the `Limiter` slot of the chain; the usage the proxy records on the `Exchange` |
| Back | `gw.02` | buckets are keyed by `Principal.KeyID` with the key's `RPM` and `TPM` |
| Forward | `gw.07` | `gateway.Deps` calls `limit.Middleware` between policy and cache, so a cached answer still spends the key's RPM |
| Forward | `load.01` | `{loadgen} --rate 50` against a low-tier key shows 429s at the configured rate |
| Forward | `ops.09` | the noisy-neighbor drill floods one tenant; your limits keep the others inside their SLO |

`load.01` and `ops.09` reach this module over HTTP through your gateway; they are not code call sites, so they are not in the registry's `used_by`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| per-process buckets | Envoy global rate limit service, Redis GCRA | one budget shared by every gateway replica | [Envoy global rate limiting](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/other_features/global_rate_limiting) (free), [GCRA](https://brandur.org/rate-limiting) (free) |
| reserve and settle | saige's shared `Budget` | request, token, and cost capacity reserved under one lock and settled once | saige `budget` package |
| two buckets per key | OpenAI and Anthropic API rate limits | per-model and per-organization tiers, daily caps | [OpenAI rate limits](https://platform.openai.com/docs/guides/rate-limits) (free) |
| 429 and `Retry-After` | client-side backoff with jitter | clients that spread their retries instead of synchronizing | [AWS: Exponential backoff and jitter](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/) (free) |
