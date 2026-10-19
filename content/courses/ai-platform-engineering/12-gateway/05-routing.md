# Worker registry, routing, cascades, failover, disaggregated orchestration

## Overview

| | |
|---|---|
| **Module** | `gw.05` · build · Go · Pass 7 · 6 to 8 h |
| **Solution** | [`registry.go`](../../../../course/solutions/go/gateway/route/registry.go): `Registry` (the `tl.control.v1` heartbeat server, eviction, drain) · [`route.go`](../../../../course/solutions/go/gateway/route/route.go): `Router`, the policies, `PickBackend`, `PrefixKey`, `ParseAcceptIf`, `LogprobsOf`, `CascadeCurve` · [`handler.go`](../../../../course/solutions/go/gateway/route/handler.go): `Middleware` (the route stage), `Proxy` (failover, cascades, disaggregated prefill and decode), `EngineTokenizer` · [`admin.go`](../../../../course/solutions/go/gateway/route/admin.go): `WorkersHandler`, `RoutesHandler`, `DrainHandler` |
| **Contract** | [`course/contracts/proto/tl/control/v1/control.proto`](../../../../course/contracts/proto/tl/control/v1/control.proto) (served) · [`engine.proto`](../../../../course/contracts/proto/tl/engine/v1/engine.proto) and [`kv.proto`](../../../../course/contracts/proto/tl/kv/v1/kv.proto) (called) · `Route`, `Worker`, and the routes and drain rules of [`admin.v1.yaml`](../../../../course/contracts/openapi/admin.v1.yaml) · `[gateway]` `route_policy`, `affinity_load_factor`, `heartbeat_miss_limit`, `routes` in [`runtime.schema.json`](../../../../course/contracts/config/runtime.schema.json) |
| **Needs** | [`ds.09` consistent hash ring](../../algorithms/16-systems-data-structures/09-consistent-hash-ring.md) · [`gw.01` server skeleton](01-server-skeleton.md) · [`gw.02` API keys](../08-authorization-and-access-control/01-api-keys-and-scopes.md) (the debug scope) · [`gw.04` streaming proxy](04-sse-streaming-proxy.md) (`Forward` and the commitment boundary) · `L10.6` (the Rust engine's Prefill and Release gRPC endpoints) · reading: `S-M07d` [queueing and balls into bins](../../mathematics-for-computing/07-probability-statistics/93-problem-set-d.md), [model routing and cascades](../11-model-routing-and-cascades/) |
| **Used by** | `gw.08` (its policy reads `route.InferenceRequest`); `gw.07`'s `gateway.Deps` and `gateway.Admin` (the `Router` slot, the workers, routes, and drain handlers, the cache revision); the gateway's composition root passes `route.Proxy` and serves the gRPC registry; `load.01`, `ops.01`, and `dur.12`'s canary reach it over HTTP · with `gw.04` and `gw.07` it turns the one-upstream proxy into a gateway over a fleet |
| **Optional depth** | Chen et al., *FrugalGPT* (2023, cascades); Zhong et al., *DistServe* (OSDI 2024) and Patel et al., *Splitwise* (ISCA 2024) on disaggregated prefill and decode; Mitzenmacher, *The Power of Two Choices in Randomized Load Balancing* (2001) |

## Key Takeaways

- Engines announce themselves with a heartbeat every 2 s; a worker that misses `heartbeat_miss_limit` (3) in a row is evicted at exactly $3 \times 2$ s after its last one.
- Three policies pick among the routable workers of a model: `least_outstanding` (fewest in flight plus queued), `weighted` (smooth weighted round robin by KV capacity), and `affinity` (`ds.09`'s ring with bounded loads on the first prompt block, so shared prefixes hit a warm cache until that worker is over its fair share).
- Before the first byte, a worker that is down or answers 503 is excluded and the request routed again; after it, nothing is retried or spliced.
- A cascade asks the small model first and keeps its answer only when its mean token logprob passes `accept_if`; sweeping the threshold trades cost for accuracy, and the curve on fixture data is reproduced exactly.
- Disaggregated serving runs `Prefill` on a prefill worker that pushes the prompt's KV to a decode worker, then streams from the decode worker with `X-TL-KV-Handle`; with one seed for both, the output equals a unified engine's.

## How to read this chapter

Sections 1 to 3 build routing from first principles and work one routing sequence by hand. Section 4 fixes the interface. Section 5 walks through the solution, file by file, and explains why each choice is the right one; try to predict each step before reading it. The package uses the generated gRPC code in `contracts/go/gen` (module `openlearn.urmzd.com/tl/contracts`) and `google.golang.org/grpc`. The gateway's composition root serves `controlv1.RegisterWorkerRegistryServer(grpcServer, registry)` on `registry_listen`, puts `route.Middleware(router)` in the chain's `Router` slot and `route.Proxy(router, route.Options{})` in its `Proxy` slot, and uses `router.Ready` as `Deps.Ready`.

---

## 1. Why now

Until now the gateway has one upstream, written in its configuration. In Pass 7 there are several engines: two decode workers and a prefill worker in the disaggregated layout of `L10.6`, a small and a large model, and soon a canary of a new model version (`dur.12`). Engines come and go: a pod is rescheduled, a node is drained, a worker crashes mid-stream (drill `ops.01`). The gateway needs to know who is alive, pick a worker per request in a way that keeps prefix caches warm without overloading anyone, move away from a dead worker before the client notices, and never stitch two workers' tokens into one answer. This module is that router.

## 2. Principles

### 2.1 The registry

Each engine sends `tl.control.v1.WorkerRegistry/Heartbeat(WorkerStatus)` every 2 s: its id, addresses (HTTP, the `EngineControl` gRPC port, the KV port of a decode worker), role (`unified`, `prefill`, `decode`), model, queue depth, running sequences, free and total KV blocks, and whether it is draining. The registry stores the latest status and the time it arrived (from the gateway's `Clock`).

| Symbol | Meaning | Value |
|---|---|---|
| $T$ | heartbeat interval | 2 s |
| $k$ | `heartbeat_miss_limit` | 3 |
| $t_\ell$ | arrival time of a worker's last heartbeat | from the clock |

A worker is **live** while $\mathit{now} - t_\ell < kT$ and is evicted (removed) once $\mathit{now} - t_\ell \ge kT$: at 5.999 s after its last heartbeat it is still there, at 6 s it is gone. A worker is **routable** when it is live, not reporting `draining`, and its model is not being drained by the admin API (`POST /admin/v1/models/{model}:drain`, which also sets `drain: true` in that worker's next `HeartbeatAck`).

### 2.2 Load and the policies

Every policy picks among the routable workers of the right model and role. A worker's **load** is the gateway's own in-flight count for it plus the engine's reported queue depth: in flight reacts at once, the queue depth catches load from other gateways and requests that are waiting rather than running.

| Policy | Picks | Good at |
|---|---|---|
| `least_outstanding` | the minimum load, ties to the smallest id | equalizing queues when requests differ in size |
| `weighted` | smooth weighted round robin with weight = KV capacity (`kv_total_blocks`): add each worker's weight to its counter, pick the largest counter, subtract the total from it | spreading in proportion to capacity without runs |
| `affinity` | `ds.09` `GetBounded` on the prefix key, with load as above and $c$ = `affinity_load_factor` | prefix cache hits, with a cap on hot prefixes |

Smooth weighted round robin with weights 1 (`small`) and 3 (`big`) gives `big, big, small, big`, repeating: the same 1:3 split as a weighted random choice, but deterministic and interleaved.

### 2.3 Affinity and the prefix key

The engine's prefix cache (`L8.4`, `L10.4`) hashes the prompt in KV blocks of 16 tokens; two requests share cached KV only if they share whole leading blocks. The gateway has no tokenizer, so it uses the leading **64 bytes** of the rendered prompt, about one block of 16 tokens at about 4 bytes per token: the completions `prompt`, or the chat messages written as `role\ncontent\n` in order. Requests that open with the same system prompt share their key, land on the same worker (reason `affinity`), and hit its cache. When that worker's load reaches $\lceil c \cdot (m+1)/n \rceil$, bounded loads move the next one clockwise (reason `load`). With 4 workers, $c = 1.25$, and 3 requests in flight on the owner (none elsewhere), the cap is $\lceil 1.25 \cdot 4 / 4 \rceil = 2$, so the owner is skipped.

### 2.4 Routes, canaries, and model ids

The route table (`[[gateway.routes]]`, or `PUT /admin/v1/routes`) maps a **public** model id and its aliases to what serves it:

- `backends`: a weighted split over **served** model ids (a canary: `smol-v3` 0.9, `smol-v4` 0.1). The draw is $u = \mathrm{Hash64}(\text{request id}) / 2^{64}$ scaled to the total weight (`ds.09`'s hash, top 53 bits), so one request id always lands on the same backend and the split is exact in expectation.
- `cascade`: a list of steps (2.6).
- `targets`: `["unified"]` (default) or `["disaggregated"]` (2.7).

A model with no route is its own route. The engine knows only its served id, so the request body's `model` is rewritten per attempt (`withFields`), leaving every other field as the client sent it. Unknown everywhere (no route, no worker) is **404** `model_not_found`; known but with no routable worker is **503** `no_capacity`. The route table uses optimistic concurrency on the admin API: `GET` returns an `ETag`; a `PUT` needs `If-Match` with it (428 `if_match_required` without, 412 `etag_mismatch` when stale), replaces the whole table, and increases the epoch sent in heartbeat acks.

### 2.5 Failover before the first byte

`Proxy` sends an attempt with `gw.04`'s `Forward` and `Retry503` set. If the attempt fails **uncommitted** (refused, reset, or a 503 from a draining or full engine), the worker is added to the request's exclusions and `Route` is asked again; at most 3 workers are tried, then the client gets 503 `no_capacity`. If it fails **committed**, `gw.04` has already sent an SSE error event and the request ends there. A worker that just died is also evicted 6 s later by the registry; failover covers the requests in between.

### 2.6 Cascades

| Symbol | Meaning |
|---|---|
| $\ell_1, \dots, \ell_T$ | token logprobs of the small model's answer |
| $\bar\ell = \frac{1}{T}\sum_t \ell_t$ | mean logprob (`mean_logprob`); `min_logprob` is $\min_t \ell_t$ |
| $\tau$ | the threshold in `accept_if`, for example `mean_logprob > -1.2` |
| $s$ | cost of a small call relative to a large one |

For a non-streamed request, each step except the last asks its model (with `logprobs` forced on) and keeps the answer when `accept_if` holds; otherwise it escalates. An answer with no tokens is never accepted. A streamed request goes straight to the last step: once a stream starts it cannot be taken back. Over an evaluation set, a threshold gives three numbers: the fraction escalated $e(\tau)$, the cost $s + e(\tau)$ (every prompt pays the small call, escalated ones also the large), and the accuracy. On the fixture's 200 prompts with $s = 0.1$:

| $\tau$ | escalated | cost | accuracy |
|---|---|---|---|
| $-3.0$ | 0.000 | 0.100 | 0.540 |
| $-2.0$ | 0.190 | 0.290 | 0.660 |
| $-1.0$ | 0.640 | 0.740 | 0.865 |
| $-0.5$ | 0.820 | 0.920 | 0.935 |
| $0.0$ | 1.000 | 1.100 | 0.935 |

At $\tau = -0.5$ the cascade matches always-large accuracy (0.935) at 92% of its cost; always-large costs 1.0 and the full sweep shows where the knee is for a given pair of models.

### 2.7 Disaggregated prefill and decode

Prefill (processing the prompt) is compute-bound; decode (one token at a time) is memory-bound. Running them on different workers lets each be sized for its job (`L10.6`). For a route with `targets = ["disaggregated"]` the gateway picks a decode worker D (by the policy) and a prefill worker P (least loaded), then:

1. Tokenizes the prompt (`Tokenizer`; the reference `EngineTokenizer` asks P's `POST /v1/tokenize`).
2. Fixes the seed: if the client sent none, the gateway draws one and puts it in both the `Prefill` request and the body sent to D. P samples the first token from its PCG32 stream; D must continue the **same** stream (`rng_draws_consumed` tells it how far), so without a shared seed a sampled disaggregated stream would differ from a unified one.
3. Calls `EngineControl/Prefill` on P with the prompt ids, the sampling parameters (every field, with the contract's defaults for omitted ones: proto3 has no "unset"), and `decode_target` = **D's KV address**. P runs prefill, pushes the prompt's KV blocks to D over `tl.kv.v1`, and returns a `KvHandle`.
4. Sends the request to D's HTTP surface with `X-TL-KV-Handle: <handle_id>`; D resumes from the pushed KV, emits the first token, and continues. The stream passes through `gw.04` like any other.
5. If that stream does not finish, calls `KvTransferService/Release(handle)` on D, so the pushed blocks return to D's pool instead of waiting for a timeout.

## 3. Worked example by hand

Three unified workers of `smol` heartbeat at $t_0$ with queue depths `a = 2`, `b = 0`, `c = 1`; the policy is `least_outstanding`.

1. Request `r1`: loads are $2, 0, 1$ (nothing in flight yet): **b**.
2. Two requests are now in flight on `b`: loads $2, 2, 1$: request `r2` goes to **c**.
3. `a` heartbeats again at $t_0 + 2$ s and $t_0 + 4$ s; `b` and `c` send nothing more.
4. At $t_0 + 5.999$ s, `b` and `c` have missed fewer than 3 heartbeats ($5.999 < 6$): all three are listed.
5. At $t_0 + 6$ s, $6 - 0 \ge 3 \times 2$: `b` and `c` are evicted; `a`'s last heartbeat was 2 s ago. Request `r3` goes to **a**.

## 4. The interface

```go
package route // import "tinyllm/gateway/route"

type Worker struct { ID, HTTPAddress, GRPCAddress, KVAddress, Role, Model string; KVFormat uint32
	QueueDepth, Running, KVFreeBlocks, KVTotalBlocks int; Draining bool; LastHeartbeat time.Time }
func NewRegistry(clock server.Clock, missLimit int) *Registry // implements controlv1.WorkerRegistryServer
func (r *Registry) Heartbeat(ctx context.Context, s *controlv1.WorkerStatus) (*controlv1.HeartbeatAck, error)
func (r *Registry) Snapshot() []Worker; func (r *Registry) Routable() []Worker; func (r *Registry) Drain(model string)

type InferenceRequest struct { Model, RequestID string; PrefixKey []byte; Stream bool }
type Exclusions map[string]bool
type Target struct { Worker Worker; Prefill *Worker; ServedModel, Reason string; CascadeStep int }
func NewRouter(reg *Registry, policy string, c float64, routes []config.Route) *Router
func (rt *Router) Route(ctx context.Context, req *InferenceRequest, ex Exclusions, step int) (Target, error)
func (rt *Router) SetRoutes([]config.Route) uint64; func (rt *Router) Routes() ([]config.Route, uint64)
func (rt *Router) Acquire(id string); func (rt *Router) Release(id string); func (rt *Router) Ready(ctx context.Context) error
func PickBackend(r config.Route, requestID string) string
func PrefixKey(body []byte) []byte
func ParseAcceptIf(s string) (Accept, error); func LogprobsOf(body []byte) ([]float64, error)
func CascadeCurve(samples []CascadeSample, taus []float64, smallCost float64) []CurvePoint

func Middleware(rt *Router) server.Middleware           // the route stage; also answers GET /v1/models
func Proxy(rt *Router, o Options) http.Handler           // the proxy stage once routing exists
type Options struct { Client *http.Client; Tokenizer Tokenizer; MaxAttempts int
	Dial func(addr string) (grpc.ClientConnInterface, error) }
func WorkersHandler(reg *Registry, rt *Router) http.Handler // GET /admin/v1/workers
func RoutesHandler(rt *Router) http.Handler                // GET, PUT /admin/v1/routes
func DrainHandler(reg *Registry) http.Handler              // POST /admin/v1/models/{model}:drain
```

`Route` takes the cascade step and returns `Target`; the cascade's `Accept` is parsed from the route's `accept_if` strings rather than given as a Go function, because routes arrive through the admin API as JSON. `X-TL-Route` carries the chosen worker id only for keys with the `debug` scope.

### What a correct solution must handle

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| the hand example (section 3) | unit | picks b, then c, eviction exactly at 6 s, then a | the definition of load and liveness |
| heartbeats over real gRPC | conformance | a heartbeat makes a worker routable; drain sets `drain` in the ack and removes it; a worker reporting draining is not routable | `L10.6`'s heartbeat client talks to this |
| unknown model, dead model | unit | unknown model 404 `model_not_found`; routed but down 503 `no_capacity`; `Ready` fails with no worker | conformance `route.model`; `/readyz` |
| affinity | unit, property | 20 requests sharing a system prompt land on one worker (`affinity`), 40 distinct prefixes spread over at least 3 of 4 workers, an overloaded owner is skipped (`load`) | warm prefix caches without hot spots |
| weighted by capacity | unit | capacities 1 and 3 give `big, big, small, big` twice | proportional and interleaved |
| canary split | statistical | 0.1 of 10000 request ids within 4 sd; one id, one backend; no backends means the route's model | `dur.12`'s canary weight |
| aliases and served ids | unit | an alias reaches the engine as the served model id | engines answer only their own id |
| failover before the first byte | fault | a refused worker and a 503 worker are each tried once and the request completes on the next | conformance and drill `ops.01` |
| a worker that dies after one token | fault | the client gets that token and an SSE error event; no other worker is called | no repeated or spliced tokens |
| `X-TL-Route` | unit | only for the `debug` scope | internal topology stays internal |
| `GET /v1/models` | unit | lists route and worker models; an unknown id is 404 | conformance `schema.models` on the gateway |
| the prefix key | unit | the rendered chat prefix; 64 bytes at most | the affinity key |
| `accept_if` parsing | boundary | `mean_logprob` and `min_logprob` with `>` and `>=`; empty answers rejected; malformed rules refused | cascade rules from the admin API |
| a cascade | unit | a confident small answer is kept (large never called, logprobs requested); an unsure one escalated; a streamed request goes to the last step | cost savings without wrong answers |
| the cascade curve | conformance, golden | the sweep over the recorded 200-prompt set reproduces escalation, cost, and accuracy at all 13 thresholds | the cost and quality curve in 2.6 |
| disaggregated equals unified | differential | the disaggregated stream equals the unified one; prompt ids, defaults, `decode_target` = D's KV address; a generated seed reaches D | `L10.6`'s disaggregated serving |
| D dies mid-stream | fault | `Release` is called with the handle and nothing stays held | KV blocks do not leak |
| the routing admin API | unit, conformance | workers with the epoch; routes 428, 412, 400 for a bad cascade, 200 with a new ETag; drain 404 and 202 | `gw.07` mounts these handlers |

## 5. The solution, step by step

### 5.1 From first principles

1. **Liveness has to be decided by the gateway's clock, not by the worker.** A dead worker sends nothing, so the only evidence is absence. The registry stores the arrival time of each heartbeat and calls a worker live while $\mathit{now} - t_\ell < kT$. The strict `<` is the whole off-by-one: at exactly $kT$ the worker has missed $k$ heartbeats and is gone.
2. **Eviction can be lazy.** Nobody needs a dead worker removed at the instant it dies; they need it absent from the next routing decision. So `Snapshot` evicts while it reads, and no background timer is needed (and tests can drive the clock).
3. **Load must combine what the gateway knows now with what the engine reported.** In-flight counts move the instant a request is sent; queue depth is up to 2 s stale but sees other gateways' traffic. Their sum is the load every policy uses (section 3, step 1: loads $2, 0, 1$ come from queue depth alone).
4. **Determinism is a feature.** Ties go to the smallest id (candidates are sorted), the weighted policy is a deterministic round robin, and the canary draw is a hash of the request id. The same inputs route the same way, so a retried request id lands on the same canary backend and a routing decision can be explained after the fact.
5. **Retrying is safe only while nothing was sent.** This is `gw.04`'s commitment boundary; routing adds the loop: exclude the failed worker, ask `Route` again, at most `MaxAttempts` times.
6. **A cascade can only judge a complete answer.** It needs the small model's logprobs over the whole answer before deciding, so its answer is captured, not streamed; a streamed request skips straight to the last step.
7. **Disaggregation is two calls that must agree.** P samples the first token and D continues; for the result to equal a unified engine, both must use the same seed and the same sampling parameters, and P must push KV to the address D's KV service listens on. If D's stream fails, nobody else will free the pushed blocks, so the gateway calls `Release`.

### 5.2 The code, annotated

**The registry** (`registry.go`). A heartbeat overwrites the worker's record and stamps it with the gateway's clock (step 1); the ack carries the drain request back:

```go
func (r *Registry) Heartbeat(ctx context.Context, s *controlv1.WorkerStatus) (*controlv1.HeartbeatAck, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.workers[s.GetWorkerId()] = &Worker{
		ID: s.GetWorkerId(), /* ... addresses, role, model, queue, KV blocks ... */
		Draining: s.GetDraining(), LastHeartbeat: r.clock.Now(), // arrival time, not the worker's claim
	}
	return &controlv1.HeartbeatAck{Drain: r.drain[s.GetModel()], RouteEpoch: r.epoch}, nil
}

func (r *Registry) live(w *Worker, now time.Time) bool {
	return now.Sub(w.LastHeartbeat) < time.Duration(r.missLimit)*r.interval // 5.999 s live, 6 s gone
}

func (r *Registry) Snapshot() []Worker {
	now := r.clock.Now()
	// ... lock
	for id, w := range r.workers {
		if !r.live(w, now) {
			delete(r.workers, id) // lazy eviction (step 2)
			continue
		}
		out = append(out, *w) // copies: callers never see a record change under them
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}
```

`Routable` filters a snapshot further: not `Draining` (the worker's own report) and not a model drained through the admin API.

**Choosing a backend and a worker** (`route.go`). The canary draw maps the request id to $[0, 1)$ with the ring's hash and walks the cumulative weights:

```go
func Unit(s string) float64 {
	return float64(ring.Hash64([]byte(s))>>11) / float64(uint64(1)<<53) // 53 bits: exact in a float64
}

func PickBackend(r config.Route, requestID string) string {
	if len(r.Backends) == 0 {
		return r.Model // a model with no route is its own route
	}
	// ... total = sum of weights
	u := Unit(requestID) * total
	acc := 0.0
	for _, b := range r.Backends {
		acc += b.Weight
		if u < acc {
			return b.ServedModel
		}
	}
	return r.Backends[len(r.Backends)-1].ServedModel // guards rounding at the top end
}
```

The three policies live in one `pick`, called with the router's lock held so in-flight counts and round-robin state cannot change mid-decision:

```go
func (rt *Router) load(w Worker) int {
	return rt.inflight[w.ID] + w.QueueDepth // step 3
}

func (rt *Router) pick(cands []Worker, key []byte) (Worker, string) {
	switch rt.policy {
	case PolicyAffinity:
		// ... rebuild the ring (160 virtual nodes each) only when the candidate set changed
		owner, _ := rt.ring.Get(key)
		id, _ := rt.ring.GetBounded(key, func(n string) int { return rt.load(byID[n]) }, rt.c)
		if id != owner {
			return byID[id], "load" // the owner was over its cap: moved clockwise
		}
		return byID[id], "affinity"
	case PolicyWeighted:
		// Smooth weighted round robin (nginx): weight = KV capacity.
		total, best := 0, -1
		for i, w := range cands {
			wt := max(w.KVTotalBlocks, 1) // a worker reporting 0 still gets a turn
			rt.smooth[w.ID] += wt
			total += wt
			if best < 0 || rt.smooth[w.ID] > rt.smooth[cands[best].ID] {
				best = i
			}
		}
		rt.smooth[cands[best].ID] -= total
		return cands[best], "load"
	default: // least_outstanding
		best := 0
		for i, w := range cands {
			if rt.load(w) < rt.load(cands[best]) { // strict <: ties keep the smallest id
				best = i
			}
		}
		return cands[best], "load"
	}
}
```

Trace the weighted case with `small` = 1, `big` = 3 (total 4). Counters start at 0. Round 1: $(1, 3)$, pick big, $(1, -1)$. Round 2: $(2, 2)$, tie, the first in id order with the larger counter stays, so big again (`big` sorts before `small`), $(2, -2)$. Round 3: $(3, 1)$, pick small, $(-1, 1)$. Round 4: $(0, 4)$, pick big, $(0, 0)$: back to the start, giving `big, big, small, big`.

`Route` puts it together: resolve the route, pick the served model (canary or cascade step), separate 404 from 503, then pick:

```go
func (rt *Router) Route(ctx context.Context, req *InferenceRequest, ex Exclusions, step int) (Target, error) {
	r := rt.Lookup(req.Model)
	served := PickBackend(r, req.RequestID)
	// ... reason "canary" with more than one backend
	if len(r.Cascade) > 0 {
		step = min(max(step, 0), len(r.Cascade)-1)
		served, reason, cstep = r.Cascade[step].Model, "cascade", step
	}
	ws := rt.reg.Routable()
	// known: some routable worker serves `served`; routed: the table names req.Model
	if !known && !routed {
		return Target{}, fmt.Errorf("%w: %s", ErrModelNotFound, req.Model) // 404: a typo, do not retry
	}
	disagg := len(r.Targets) > 0 && r.Targets[0] == "disaggregated"
	role := "unified"
	if disagg {
		role = "decode" // the policy picks D; P is chosen below
	}
	cands := candidates(ws, served, role, ex) // minus exclusions, sorted by id
	if len(cands) == 0 {
		return Target{}, fmt.Errorf("%w %s (%s)", ErrNoCapacity, served, role) // 503: try later
	}
	// ... lock; w, why := rt.pick(cands, req.PrefixKey)
	if disagg {
		// ... P = the least-loaded prefill worker of the same model, not excluded
	}
	return t, nil
}
```

**Failover** (`handler.go`). `attempts` is step 5 as a loop. The first attempt uses the target the route stage already picked; later ones route again with the failed workers excluded:

```go
for attempt := 0; attempt < p.o.MaxAttempts; attempt++ {
	if attempt > 0 {
		t, err = p.rt.Route(r.Context(), &rd.ir, excl, step)
		if err != nil {
			lastErr = err
			break // nobody left to try
		}
		note(ex, w, r, t)
	}
	err := p.once(w, r, rd, ex, t, proxy.ExchangeObserver(ex))
	if err == nil {
		return
	}
	var ue *proxy.UpstreamError
	if !errors.As(err, &ue) || ue.Committed || r.Context().Err() != nil {
		return // after the first byte (or the client left): never retried
	}
	lastErr = err
	excl[t.Worker.ID] = true // never the same worker twice
	if t.Prefill != nil {
		excl[t.Prefill.ID] = true
	}
}
if r.Context().Err() == nil {
	server.WriteError(w, http.StatusServiceUnavailable, "server_error", "no_capacity", "", /* ... lastErr */)
}
```

`once` sends one attempt. It counts the worker as in flight for exactly the attempt's duration, sets `Retry503` so a draining engine's 503 comes back unwritten, and rewrites only `model` in the body (`withFields` decodes to `map[string]json.RawMessage`, so every other field passes through byte for byte):

```go
att := proxy.Outbound{Method: rd.out.method, Path: rd.out.path, RawQuery: rd.out.rawQuery, Header: rd.out.header.Clone()}
att.Retry503 = true
set := map[string]any{"model": t.ServedModel} // engines know only their served id
p.rt.Acquire(t.Worker.ID)
defer p.rt.Release(t.Worker.ID)
if t.Prefill == nil {
	att.Body = withFields(rd.req.Body, set)
	return proxy.Forward(r.Context(), w, "http://"+t.Worker.HTTPAddress, &att, p.o.Client, obs)
}
```

The disaggregated branch of `once` is step 7:

```go
seed := uint64(0)
if rd.req.Seed != nil {
	seed = uint64(*rd.req.Seed)
} else {
	// ... 8 random bytes
	seed = binary.LittleEndian.Uint64(b[:]) >> 1 // fits a JSON int64
	set["seed"] = seed                           // D must draw from the same stream as P
}
ids, err := p.o.Tokenizer.Tokenize(r.Context(), *t.Prefill, t.ServedModel, rd.req.Body)
// ... errors before the stream are uncommitted: failover may pick another pair
pre, err := enginev1.NewEngineControlClient(conn).Prefill(r.Context(), &enginev1.PrefillRequest{
	RequestId: ex.RequestID, Model: t.ServedModel, PromptIds: ids,
	Sampling: sampling(rd.req.Body, seed), // every field, contract defaults filled in
	DecodeTarget: t.Worker.KVAddress,      // D's KV port, not its HTTP address
	// ... deadline, priority
})
// ...
handle := pre.GetHandle().GetHandleId()
att.Body = withFields(rd.req.Body, set)
att.Header.Set("X-TL-KV-Handle", handle)
err = proxy.Forward(r.Context(), w, "http://"+t.Worker.HTTPAddress, &att, p.o.Client, obs)
if err != nil {
	// D holds the pushed blocks under the handle until it resumes or is told to let go.
	if kc, derr := p.dial(t.Worker.KVAddress); derr == nil {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second) // the client's ctx may be gone
		_, _ = kvv1.NewKvTransferServiceClient(kc).Release(ctx, &kvv1.ReleaseRequest{HandleId: handle})
		cancel()
	}
}
return err
```

`sampling` starts from `Temperature: 1, TopP: 1, RepetitionPenalty: 1` and overrides only fields the client sent, because proto3 cannot express "unset": a zero temperature would mean greedy decoding, not "default".

**Cascades.** `ParseAcceptIf` turns `"mean_logprob > -1.2"` into a function; the returned closure starts with `len(lp) > 0 &&`, so an empty answer (whose mean would be NaN) is never accepted. `cascade` runs every step but the last with `logprobs` forced on, into a `capture` writer, and writes the first accepted answer:

```go
for step := 0; step < len(steps)-1; step++ {
	accept, err := ParseAcceptIf(steps[step].AcceptIf)
	// ... route to this step's model
	body := rd.req.Body
	if _, has := lpField["logprobs"]; !has {
		body = withFields(body, force) // logprobs: true (chat) or 1 (completions)
	}
	c := &capture{h: http.Header{}} // nothing reaches the client yet (step 6)
	if err := p.once(c, r, &local, ex, t, obs); err != nil || c.code != http.StatusOK {
		continue // a failed step escalates
	}
	lps, err := LogprobsOf(c.body.Bytes())
	if err != nil || !accept(lps) {
		continue
	}
	// ... copy the captured headers (keeping the gateway's X-Request-Id), record usage, write the body
	return true
}
// ... route to the last step; false: the caller's attempts() sends it with failover
```

`CascadeCurve` is the same decision offline: every prompt pays `smallCost`, an escalated prompt also pays 1, and accuracy counts the answer actually kept. At $\tau = -0.5$ it gives 0.82 escalated, cost $0.1 + 0.82 = 0.92$, accuracy 0.935, the row in 2.6.

**The admin routes** (`admin.go`). `RoutesHandler` serializes `PUT`s with a mutex so the `If-Match` check and the replacement are atomic, and the ETag is the epoch (`"routes-<n>"`), which `SetRoutes` increments on every replacement. Without the mutex, two operators holding the same ETag could both pass the check.

Full solution: [`registry.go`](../../../../course/solutions/go/gateway/route/registry.go), [`route.go`](../../../../course/solutions/go/gateway/route/route.go), [`handler.go`](../../../../course/solutions/go/gateway/route/handler.go), [`admin.go`](../../../../course/solutions/go/gateway/route/admin.go).

### 5.3 Why this solution

- **Cost per decision.** `Routable` copies and sorts the worker list ($O(n \log n)$ for $n$ workers, a handful to a few hundred), and every policy is $O(n)$ or, for affinity, a ring lookup ($O(\log(160n))$) plus bounded-load probing. The ring is rebuilt only when the candidate set changes. All of this is microseconds next to a model call.
- **Push heartbeats rather than polling health endpoints.** Engines already know their queue and KV state; pushing them every 2 s gives the router load information for free, and absence is a liveness signal. A poller would need its own timeouts and would still be stale.
- **Load = in flight + queue**, rather than either alone. In flight alone misses work from other gateways and long engine queues (pitfall 3); queue depth alone reacts only every 2 s, so a burst would all go to the worker that looked idle at the last heartbeat.
- **Consistent hashing with bounded loads** rather than plain hashing or least-loaded for affinity: plain hashing makes one hot system prompt a hot spot; least-loaded scatters a shared prefix over every worker and loses the cache. Bounded loads keep both properties, with $c$ as the dial.
- **Smooth weighted round robin** rather than weighted random: the same proportions, but no runs, and reproducible.
- **Hashing the request id for the canary** rather than a random draw: the split is exact in expectation, retries of the same request stay on one version, and a support ticket's request id tells you which version served it.
- **Tradeoffs accepted:** eviction is lazy, so a dead worker is still in memory until the next read (harmless); failover tries at most 3 workers, bounding the added latency of a widespread outage; the cascade adds the small model's full latency to every escalated request, which is why streams skip it.

### 5.4 Explain it back

1. Why is liveness `now - last < k*T` and not `<=`, and why is it measured with the gateway's clock?
   <details><summary>Answer</summary>At exactly $kT$ after the last heartbeat the worker has missed $k$ in a row, which is the eviction rule; `<=` keeps a dead worker routable for one more instant (or, with a coarse clock, one more interval). The gateway's clock is used because a dead or skewed worker cannot be trusted to timestamp its own heartbeats.</details>
2. In the weighted trace, why does round 2 pick `big` on a 2:2 tie, and what would happen if ties went to the last candidate?
   <details><summary>Answer</summary>The scan replaces `best` only on a strictly larger counter, so the first candidate in id order with the maximum wins, which is `big`. With ties to the last, round 2 picks `small`, and the sequence becomes `big, small, big, big`: still 1:3 over four requests, but a different order, so any recorded expectation of the order (and reproducibility across gateways) depends on the tie rule being fixed.</details>
3. Why does `attempts` stop on `r.Context().Err() != nil` even when the error is uncommitted?
   <details><summary>Answer</summary>A cancelled client context makes every upstream call fail before the first byte. Retrying would route the dead request to other workers for nobody, and writing a 503 would go to a closed connection.</details>
4. What goes wrong in disaggregated mode if the client sent no seed and the gateway does not generate one?
   <details><summary>Answer</summary>P samples the first token from its PCG32 stream with one seed, and D, seeded differently, continues from a different stream. The output is a valid sample but not the one a unified engine would produce for the same request, so the disaggregated stream no longer equals the unified one and results are not reproducible.</details>
5. Why does `sampling` default temperature to 1 instead of using the zero value of a missing JSON field?
   <details><summary>Answer</summary>proto3 has no field presence, so the engine cannot tell "unset" from 0. Temperature 0 means greedy decoding, so sending the zero value would silently change the request's meaning; the gateway fills in the contract's defaults instead.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. eviction and draining off by one: `<=` on the deadline, draining workers still routable | a dead worker gets traffic for one more interval; a rollout keeps sending to the pod it drains | `live` uses strict `<`; `Routable` drops `Draining` workers and drained models |
| 2. retrying after the first byte | the client reads "The cat The cat sat": tokens repeated or spliced | `attempts` returns on a committed `UpstreamError` |
| 3. load from in-flight counts only | a worker with a long engine queue keeps receiving work | `load` is in flight plus `QueueDepth` |
| 4. affinity without bounded loads | one hot system prompt saturates one worker while others idle | `GetBounded` with $c$ = `affinity_load_factor`; the reason becomes `load` |
| 5. failover that relays a 503, or retries the same worker | clients see 503s while healthy workers sit idle | `Retry503` on every attempt; the failed worker joins `excl` |
| 6. disaggregation details: `decode_target` set to D's HTTP address, no shared seed, no `Release` on failure | prefill cannot push KV; sampled outputs differ from unified; D's pool leaks blocks | `DecodeTarget: t.Worker.KVAddress`, a generated seed sent to both, `Release` on any decode error |
| 7. canary and model ids: weights ignored, the public id sent upstream, an unknown model answered 503 | the canary gets half the traffic; engines answer 404; clients retry a typo forever | `PickBackend` walks cumulative weights; `withFields` sets the served `model`; `Route` returns `ErrModelNotFound` when neither table nor workers know it |
| 8. cascade mistakes: accepting everything, cascading a stream, accepting an empty answer, a cost that forgets the small call | quality drops silently; streams stall; the curve understates cost | `accept_if` parsed per step; streams start at the last step; `len(lp) > 0` in the closure; `CascadeCurve` adds `smallCost` for every prompt |
| 9. `X-TL-Route` for every key | worker ids leak to every client | `note` sets it only when the principal has the `debug` scope, and deletes it otherwise |
| 10. `PUT /admin/v1/routes` without the `If-Match` check | two operators' edits overwrite each other silently | 428 without `If-Match`, 412 when stale, the check and the swap under one mutex |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `ds.09` | `GetBounded` is the affinity policy; `Hash64` draws the canary backend |
| Back | `gw.01` | the `Router` and `Proxy` slots, the `Exchange` (worker id, route attributes), `Deps.Ready` |
| Back | `gw.02` | the `debug` scope gates `X-TL-Route` |
| Back | `gw.04` | every attempt goes through `Forward`; only uncommitted failures are retried |
| Back | `L10.6` | Prefill and Release gRPC endpoints implement disaggregated decode routing |
| Forward | `gw.07` | `gateway.Admin` builds `WorkersHandler`, `RoutesHandler`, and `DrainHandler`; `gateway.Deps` puts `Middleware` in the chain and keys the cache by `ETag(epoch)` |
| Forward | `gw.08` | the usage policy matches on the parsed `route.InferenceRequest` (model, prompt) |
| Forward | `load.01`, `ops.01`, `dur.12` | the load generator, the kill-decode drill, and the canary release exercise routing over HTTP |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| affinity on a prefix key | llm-d and KServe KV-cache-aware schedulers, SGLang router | scoring workers by the KV blocks they actually hold, not by a hash | [llm-d](https://github.com/llm-d/llm-d) (free), [SGLang router](https://github.com/sgl-project/sglang) (free) |
| backends and failover | saige `provider/router` | immutable model configurations, sticky selection for prompt-cache affinity, failover only before commitment | saige `provider/router` |
| cascades | FrugalGPT, RouteLLM | learned routers that predict which model a prompt needs | [RouteLLM](https://github.com/lm-sys/RouteLLM) (free) |
| disaggregated orchestration | vLLM disaggregated prefill, Dynamo, Mooncake | KV transfer over RDMA, a global KV pool, prefill scheduling across nodes | [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) (free), [Mooncake](https://github.com/kvcache-ai/Mooncake) (free) |
