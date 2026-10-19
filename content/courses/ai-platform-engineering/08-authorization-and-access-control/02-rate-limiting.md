# Rate limiting: RPM token bucket + TPM reserve/settle

## Overview

| | |
|---|---|
| **Module** | `gw.03` · build · Go · Pass 7 · 4 h |
| **Solution** | [`go/gateway/limit/limit.go`](../../../../course/solutions/go/gateway/limit/limit.go): `Limiter` (`New`, `Reserve`), `Reservation` (`Settle`, `Cancel`, `Status`), `RejectError`, `EstimateCost`, `PromptText`, `TokenizeCounter`, `SetHeaders`, `RetryAfterSeconds`, `Middleware` |
| **Contract** | 429 `rate_limit_error` / `rate_limit_exceeded`, `Retry-After`, and the `x-ratelimit-*` headers in [`course/contracts/openapi/openai-subset.v1.yaml`](../../../../course/contracts/openapi/openai-subset.v1.yaml); `rpm` and `tpm` of `KeyCreate` in [`course/contracts/openapi/admin.v1.yaml`](../../../../course/contracts/openapi/admin.v1.yaml) |
| **Needs** | [`gw.01` server skeleton](../12-gateway/01-server-skeleton.md) (the chain and the `Exchange`), [`gw.02` API keys](01-api-keys-and-scopes.md) (the `Principal` and its limits) · reading: `S-M07d` [queueing and Little's law](../../mathematics-for-computing/07-probability-statistics/93-problem-set-d.md), the [practice drill go/07](../../../../practice/build/cloud/go/README.md) (a token bucket warm-up) |
| **Used by** | `gw.07`'s `gateway.Deps`, which puts `limit.Middleware` in the chain's `Limiter` slot; the gateway's composition root builds the `Limiter`; part of the full gateway |
| **Optional depth** | Tanenbaum, *Computer Networks*, the token bucket; the GCRA in the ATM Forum's traffic management spec; Stripe's *Scaling your API with rate limiters* (free) |

## Key Takeaways

- A **token bucket** holds at most $C$ tokens and refills at $C$ per minute; a request takes tokens if they are there, otherwise it waits exactly as long as the refill needs. A full bucket allows a burst of one minute's budget, never more.
- Requests per minute and tokens per minute are two buckets per key. Token cost is unknown when a request starts, so the limiter **reserves** an estimate (prompt tokens plus `max_tokens`) and **settles** the actual usage when the response ends: unused tokens come back, overruns are charged.
- A 429 carries `Retry-After` in whole seconds rounded **up**; every authenticated response carries the five `x-ratelimit-*` headers.
- Exactly one of `Settle` and `Cancel` counts; a request that failed with a 5xx before any usage is cancelled, so clients do not pay for our outages.
- One mutex around every bucket decision: 1000 goroutines against a budget of 100 admit exactly 100.

## How to read this chapter

Sections 1 to 3 derive the token bucket and the reserve/settle protocol from first principles and run them by hand. Section 4 fixes the interface. Section 5 walks through the solution function by function and explains why each choice is the right one; try to predict each step before reading it. In the finished system, `gw.07`'s `gateway.Deps` ([`go/gateway/gateway.go`](../../../../course/solutions/go/gateway/gateway.go)) wraps `limit.Middleware(limiter, counter)` into the `Limiter` slot, and the composition root ([`entry/go/cmd/gateway/compose.go`](../../../../course/solutions/entry/go/cmd/gateway/compose.go)) builds the limiter with `limit.New(nil)` (the wall clock). It sets no `Counter`, so the shipped gateway estimates prompt tokens as bytes / 4 and relies on `Settle` for the exact count; a `limit.TokenizeCounter{BaseURL: engineURL}` makes the estimate exact.

---

## 1. Why now

The gateway now knows who is calling (`gw.02`), but every caller can still send as fast as its network allows. One tenant's batch job at 50 requests per second fills the engine's queue (`L10.2`), and every other tenant's time to first token goes from milliseconds to seconds: the noisy-neighbor incident you will meet as drill `ops.09`. Counting requests is not enough for an LLM: one request with a 4000-token prompt and `max_tokens: 2000` costs a hundred times a short chat turn. This module gives each key two budgets, requests per minute (RPM) and tokens per minute (TPM), enforced before the request reaches the cache or an engine, and tells the client exactly when to come back.

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

The catalog sketched `Reserve(ctx, key, c)`; this one also takes the key's `Limits`, because limits live on the key record (`gw.02`) and change when an admin edits the key, so the limiter holds buckets, not key metadata Time comes only from the `Clock`, so a fake clock drives refill exactly.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand example | unit | section 3, every number: remaining counts, reset 18 s, the settle refund, `RetryAfter` 1.2 s and `Retry-After` 2, admission at 1.2 s | one exact definition of the bucket |
| burst, then refill | unit | 60 at once, the 61st waits 1 s, one per second after, 10 idle minutes bank only 60 | the burst limit in the load report |
| settle charges an overrun | unit | reserve 100, use 700: the next 10 tokens wait 11 s | requests without `max_tokens` cannot dodge TPM |
| cancel and settle count once | unit | cancel returns everything; a second cancel and a cancel after settle do nothing | no double refunds |
| zero means unlimited | boundary | limits of 0 never reject | the admin API's "0 = unlimited" |
| oversize cost rejected | boundary | 1001 tokens against TPM 1000 is rejected even when the bucket is full; 1000 fits | one request cannot exceed a minute's budget |
| keys are independent | unit | key `a` exhausted, key `b` untouched | the noisy-neighbor drill |
| concurrent reserve | property | 1000 goroutines, RPM 100: exactly 100 admitted, no data race | correctness under load |
| `Retry-After` rounding | boundary | 0, 1 ms, 1.2 s, 59 s, 59.001 s | clients come back when the budget exists |
| the cost estimate | unit | prompt + max_tokens; `max_completion_tokens` wins; 256 default; bytes / 4 when the tokenizer fails | reservations close to the truth |
| the prompt text | unit | prompt, chat contents, embeddings input | the text that is tokenized |
| the tokenizer counter | unit | posts `{model, text}` to `/v1/tokenize` and counts the ids | the production counter |
| headers and the 429 | conformance | headers on every 200; the third request at RPM 2 is 429 with `Retry-After: 30` and the error shape | the contract's `ratelimit.429` behaviour |
| the middleware settles actual usage | unit | the next response's remaining tokens reflect the actual usage, not the estimate | TPM tracks real cost |
| the middleware cancels on a 5xx | unit | after a 503 without usage the one-request budget is back | our outages are free |
| no principal passes through | boundary | no principal: no headers, no limit | the stage is safe before `gw.02` is plugged in |
| buckets keyed by key id | unit | two keys of one tenant have separate budgets | a CI key cannot starve a developer's |

## 5. The solution, step by step

### 5.1 From first principles

1. **The state per key must be O(1).** The limiter sees every request of every key, so it cannot keep a log of timestamps. A token bucket summarizes the whole history in two numbers: the level and the time it was last brought up to date (section 2.1).
2. **Refill can be computed lazily.** Because refill is linear in time and capped, there is no need for a timer: on each decision, add $\rho \cdot \Delta t$ and cap at $C$. That makes the bucket correct at whatever moments the clock is read, and lets a fake clock drive it exactly.
3. **Two budgets, one decision.** A request must fit both the request bucket and the token bucket; if either is short, it waits for the slower one, so the wait is the maximum of the two. Taking from one bucket and then failing the other would leak budget, so the check happens first and the take after, under one lock.
4. **The cost is unknown up front, so split the decision in two.** Reserve an upper-ish estimate before the request runs (prompt tokens plus `max_tokens`), then settle the difference with the real usage when it ends. The bucket may go negative after an overrun: that debt is what makes `max_tokens`-less requests pay.
5. **A reservation is a one-shot.** Settle or Cancel, whichever comes first, wins; the other is a no-op. `sync.Once` is exactly that.
6. **A request larger than $C$ can never fit.** Waiting for $n > C$ tokens is waiting forever, so it is rejected outright (with the time to a full bucket as a hint) rather than admitted when the bucket happens to be full.
7. **Time comes only from an injected clock.** Then the hand example of section 3 is reproducible to the nanosecond, and refill is testable without sleeping.

### 5.2 The code, annotated

**The bucket.** Levels are `float64` so fractional refill (0.06 of a request in section 3, step 4) is kept rather than rounded away.

```go
type bucket struct {
	capacity float64 // C: the per-minute limit
	level    float64 // L(t_last); may be negative after an overrun
	last     time.Time
}

// refill brings the bucket up to now: L <- min(C, L + C * minutes elapsed).
func (b *bucket) refill(now time.Time) {
	if dt := now.Sub(b.last); dt > 0 { // a clock that steps back adds nothing
		b.level = math.Min(b.capacity, b.level+b.capacity*dt.Minutes())
	}
	b.last = now
}

// wait is w = (n - L) / C minutes, rounded up to the next nanosecond so that
// at now + w the bucket really holds n.
func (b *bucket) wait(n float64) time.Duration {
	if b.level >= n {
		return 0
	}
	return time.Duration(math.Ceil((n - b.level) / b.capacity * float64(time.Minute)))
}
```

**Per-key state.** A key's buckets are created full the first time it is seen. Because limits live on the key record and an admin may edit them, `state` resizes the buckets when the limits change, keeping the same fraction of capacity, so raising a limit does not hand out a fresh burst and lowering it does not leave the key over its new budget.

```go
func (l *Limiter) state(key string, lim Limits, now time.Time) *keyState {
	st, ok := l.keys[key]
	if !ok { // first sight: both buckets full
		st = &keyState{
			req: bucket{capacity: float64(lim.RPM), level: float64(lim.RPM), last: now},
			tok: bucket{capacity: float64(lim.TPM), level: float64(lim.TPM), last: now},
			lim: lim,
		}
		l.keys[key] = st
		return st
	}
	if st.lim != lim { // limits edited: keep L/C, change C
		// ... resize each bucket: level = level / old capacity * new capacity
	}
	st.req.refill(now)
	st.tok.refill(now)
	return st
}
```

**Reserve: check both, then take both.** A zero limit skips its bucket entirely (unlimited). The `min(c.Tokens, lim.TPM)` keeps the wait finite for an oversize request, and the explicit `c.Tokens > lim.TPM` test rejects it even when the bucket is full.

```go
func (l *Limiter) Reserve(ctx context.Context, key string, lim Limits, c Cost) (Reservation, error) {
	now := l.clock.Now()
	l.mu.Lock() // one lock around read-refill-decide-take: no two requests spend the same token
	defer l.mu.Unlock()
	st := l.state(key, lim, now)
	var wait time.Duration
	if lim.RPM > 0 {
		wait = max(wait, st.req.wait(float64(min(c.Requests, lim.RPM))))
	}
	if lim.TPM > 0 {
		wait = max(wait, st.tok.wait(float64(min(c.Tokens, lim.TPM))))
	}
	if wait > 0 || (lim.TPM > 0 && c.Tokens > lim.TPM) || (lim.RPM > 0 && c.Requests > lim.RPM) {
		if wait == 0 { // oversize with a full bucket: point at "full again" as the hint
			wait = st.tok.wait(st.tok.capacity)
		}
		return nil, &RejectError{RetryAfter: wait, Status: st.status()} // nothing taken
	}
	if lim.RPM > 0 {
		st.req.level -= float64(c.Requests)
	}
	if lim.TPM > 0 {
		st.tok.level -= float64(c.Tokens)
	}
	return &reservation{l: l, key: key, cost: c, status: st.status()}, nil
}
```

**Settle and Cancel.** Both are a signed adjustment of the buckets: Settle returns `reserved - actual` (negative for an overrun, which takes more), Cancel returns everything. The cap at capacity means a refund never pushes a bucket above $C$. `sync.Once` makes the pair exactly-once.

```go
func (r *reservation) Settle(actual Cost) {
	r.once.Do(func() {
		// reserved 300, used 120: +180 back; reserved 100, used 700: -600 more
		r.l.adjust(r.key, r.cost.Requests-actual.Requests, r.cost.Tokens-actual.Tokens)
	})
}

func (r *reservation) Cancel() {
	r.once.Do(func() { r.l.adjust(r.key, r.cost.Requests, r.cost.Tokens) })
}
```

`adjust` takes the lock, refills both buckets to now, and adds the deltas with `math.Min(capacity, level + d)`.

**The estimate.** Prompt tokens from the engine's tokenizer when it answers, otherwise $\lceil \text{bytes}/4 \rceil$; completion tokens from `max_tokens` (the `Exchange` has already resolved `max_completion_tokens` over `max_tokens` into `MaxTokens`), or 256.

```go
func EstimateCost(ctx context.Context, counter Counter, req *server.Request) Cost {
	prompt := -1
	if counter != nil {
		if n, err := counter.CountPrompt(ctx, req.Model, req.Body); err == nil {
			prompt = n
		}
	}
	if prompt < 0 { // tokenizer down: estimate, never 0 and never fail the request
		prompt = (len(PromptText(req.Body)) + 3) / 4
	}
	completion := req.MaxTokens
	if completion <= 0 {
		completion = DefaultCompletionReserve // 256; Settle charges the truth afterwards
	}
	return Cost{Requests: 1, Tokens: prompt + completion}
}

func RetryAfterSeconds(d time.Duration) int {
	s := int(math.Ceil(d.Seconds())) // 1.2 s -> 2: never invite the client back early
	if s < 1 {
		s = 1 // "0" would mean "retry now"
	}
	return s
}
```

**The middleware.** Reserve before `next`, settle after it, with a deferred `Cancel` as a safety net if the rest of the chain panics.

```go
p, ok := auth.PrincipalFrom(r.Context())
if !ok { // no authn stage in front: nothing to key by, so do not limit
	next.ServeHTTP(w, r)
	return
}
// ... cost = EstimateCost(...) for POSTs, Cost{Requests: 1} otherwise
res, err := l.Reserve(r.Context(), p.KeyID, Limits{RPM: p.RPM, TPM: p.TPM}, cost) // keyed by key, not tenant
var rej *RejectError
if errors.As(err, &rej) {
	SetHeaders(w.Header(), rej.Status)
	w.Header().Set("Retry-After", strconv.Itoa(RetryAfterSeconds(rej.RetryAfter)))
	// ... 429 rate_limit_error / rate_limit_exceeded
	return
}
// ...
SetHeaders(w.Header(), res.Status()) // on every admitted response, not only on 429
settled := false
defer func() {
	if !settled {
		res.Cancel() // a panic below: nothing ran to completion
	}
}()
next.ServeHTTP(w, r)
st := ex.Snapshot() // the usage the proxy recorded (gw.04)
switch {
case st.Usage.TotalTokens > 0 || st.Usage.PromptTokens+st.Usage.CompletionTokens > 0:
	// ... tokens = TotalTokens, or prompt + completion when the total is missing
	res.Settle(Cost{Requests: 1, Tokens: tokens})
case st.Status >= 500:
	res.Cancel() // our failure, no usage: the client pays nothing
default:
	res.Settle(Cost{Requests: 1}) // ran but produced no usage: the request counts, the tokens come back
}
settled = true
```

Full solution: [`limit.go`](../../../../course/solutions/go/gateway/limit/limit.go). `PromptText` (completions `prompt`, chat contents joined by `\n`, or embeddings `input`) and `TokenizeCounter` (a `POST /v1/tokenize` of `{model, text}` that counts the returned `ids`) are in the same file.

### 5.3 Why this solution

- **Cost.** `Reserve`, `Settle`, and `Cancel` are O(1) under one mutex; memory is two buckets per key ever seen. The tokenizer call is one HTTP round trip per POST, outside the lock.
- **Why one global mutex and not one per key?** The critical section is a handful of float operations, far shorter than any request; contention is negligible at gateway rates, and one lock makes the map and the buckets trivially consistent. Per-key locks would add a second map and lock ordering for no measurable gain.
- **Why not delay requests instead of rejecting them?** Holding a request open until budget exists ties up a connection and a goroutine per throttled request, and hides back-pressure from the client. A 429 with an exact `Retry-After` lets well-behaved clients pace themselves.
- **Why reserve an estimate instead of charging only after?** Charging only after the response lets a key start many expensive requests at once before any of them is counted. Reserving first bounds what can be in flight; settling afterwards keeps the long-run total exact.
- **Tradeoff accepted:** buckets are per process. With $r$ gateway replicas and no stickiness, a key can get up to $r$ times its budget. A shared store (Redis GCRA, Envoy's global rate limit service) fixes that at the price of a network hop per request (see Going further).

### 5.4 Explain it back

1. In section 3, step 4, why are remaining requests 1 and not 2, given that 3 requests per minute were allowed and only 2 have been made?

<details><summary>Answer</summary>

The bucket was at 2 after the first request, refilled only $3 \cdot 1.2 / 60 = 0.06$ in 1.2 s (the 429 at step 3 took nothing), so it held 2.06 and the new request leaves 1.06. The header is $\lfloor 1.06 \rfloor = 1$. The bucket counts budget earned over time, not requests left in a calendar minute.

</details>

2. Why does `Reserve` compute the wait with `min(c.Tokens, lim.TPM)` and then reject `c.Tokens > lim.TPM` separately?

<details><summary>Answer</summary>

Without the `min`, the wait for $n > C$ would be finite but meaningless: the bucket is capped at $C$ and never reaches $n$. Clamping keeps the arithmetic sane, and the explicit test rejects the request even when the bucket is full, so one request can never take more than a minute's budget.

</details>

3. A request reserves 300 tokens; the proxy panics before recording usage. What does the key get back, and which line makes that happen?

<details><summary>Answer</summary>

Everything: the deferred function sees `settled == false` and calls `res.Cancel()`, which returns 1 request and 300 tokens. Had `Settle` already run, `sync.Once` would make that `Cancel` a no-op, so no double refund is possible.

</details>

4. Why does a 5xx without usage cancel, while a 4xx without usage settles with one request?

<details><summary>Answer</summary>

A 5xx without usage means the gateway or the engines failed before doing work: charging the client would make it pay for our outage. A 4xx is the client's own bad request: it still consumed a request slot, so the request counts, but no tokens were generated, so the token reservation comes back.

</details>

5. Why are buckets keyed by `Principal.KeyID` rather than by tenant?

<details><summary>Answer</summary>

Limits are set per key (`rpm`, `tpm` on the key record), and one tenant often has several keys with different jobs, such as CI and a developer. Keying by tenant would let a CI job's burst starve the developer's key, the noisy-neighbor problem inside one tenant.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. refill without the cap | a key idle overnight sends a day's budget at once | `refill` takes `math.Min(capacity, ...)`; refunds in `adjust` are capped too |
| 2. refilling per second instead of per minute, or rounding `Retry-After` down | limits 60 times too loose; clients retry into another 429 | refill multiplies by `dt.Minutes()`; `RetryAfterSeconds` uses `math.Ceil` and a floor of 1 |
| 3. settle mistakes: refunding the whole reservation, never charging an overrun, both settle and cancel counting, settling the estimate | TPM drifts from real usage in either direction | `Settle` adjusts by `reserved - actual` (negative for overruns), guarded by `sync.Once`; the middleware settles with the usage from `ex.Snapshot()` |
| 4. treating 0 as an empty budget, or charging requests that failed with a 5xx | unlimited keys are locked out; clients pay for our outages | every bucket operation is guarded by `lim.RPM > 0` / `lim.TPM > 0`; a 5xx without usage calls `Cancel` |
| 5. admitting a request larger than the limit because the bucket is full | one request takes more than a minute's tokens | `Reserve` rejects `c.Tokens > lim.TPM` explicitly |
| 6. no lock around a decision, or one bucket per tenant | under load more requests are admitted than the limit; keys drain each other | one mutex around refill-decide-take; buckets keyed by `p.KeyID` |
| 7. headers only on 429 | clients cannot pace themselves before they are throttled | `SetHeaders` runs on the admitted path as well as the 429 path |
| 8. a bad estimate: 0 tokens when the tokenizer fails, `max_tokens` ignored | huge requests pass the reservation and overdraw afterwards | `EstimateCost` falls back to $\lceil \text{bytes}/4 \rceil$ and adds `MaxTokens` or 256 |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `gw.01` | the `Limiter` slot of the chain; the usage the proxy records on the `Exchange` |
| Back | `gw.02` | buckets are keyed by `Principal.KeyID` with the key's `RPM` and `TPM` |
| Forward | `gw.07` | `gateway.Deps` calls `limit.Middleware` between policy and cache, so a cached answer still spends the key's RPM |
| Forward | `load.01` | `{loadgen} --rate 50` against a low-tier key shows 429s at the configured rate |
| Forward | `ops.09` | the noisy-neighbor drill floods one tenant; the limits keep the others inside their SLO |

`load.01` and `ops.09` reach this module over HTTP through the gateway; they are not code call sites of this package.

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| per-process buckets | Envoy global rate limit service, Redis GCRA | one budget shared by every gateway replica | [Envoy global rate limiting](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/other_features/global_rate_limiting) (free), [GCRA](https://brandur.org/rate-limiting) (free) |
| reserve and settle | saige's shared `Budget` | request, token, and cost capacity reserved under one lock and settled once | saige `budget` package |
| two buckets per key | OpenAI and Anthropic API rate limits | per-model and per-organization tiers, daily caps | [OpenAI rate limits](https://platform.openai.com/docs/guides/rate-limits) (free) |
| 429 and `Retry-After` | client-side backoff with jitter | clients that spread their retries instead of synchronizing | [AWS: Exponential backoff and jitter](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/) (free) |
