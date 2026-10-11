<!-- ol:module gw.05 -->
# Worker registry, routing, cascades, failover, disaggregated orchestration

## Overview

| | |
|---|---|
| **Module** | `gw.05` · build · Go · Pass 7 · 6 to 8 h |
| **You build** | `go/gateway/route/registry.go`: `Registry` (the `tl.control.v1` heartbeat server, eviction, drain) · `route.go`: `Router`, the policies, `PickBackend`, `PrefixKey`, `ParseAcceptIf`, `LogprobsOf`, `CascadeCurve` · `handler.go`: `Middleware` (the route stage), `Proxy` (failover, cascades, disaggregated prefill and decode), `EngineTokenizer` · `admin.go`: `WorkersHandler`, `RoutesHandler`, `DrainHandler` |
| **Contract** | [`course/contracts/proto/tl/control/v1/control.proto`](../../course/contracts/proto/tl/control/v1/control.proto) (served) · [`engine.proto`](../../course/contracts/proto/tl/engine/v1/engine.proto) and [`kv.proto`](../../course/contracts/proto/tl/kv/v1/kv.proto) (called) · `Route`, `Worker`, and the routes and drain rules of [`admin.v1.yaml`](../../course/contracts/openapi/admin.v1.yaml) · `[gateway]` `route_policy`, `affinity_load_factor`, `heartbeat_miss_limit`, `routes` in [`runtime.schema.json`](../../course/contracts/config/runtime.schema.json) |
| **Tests** | `course/tests/go/gw_05/` (what they check: section 4) |
| **Needs** | [`ds.09` consistent hash ring](../../algorithms/16-systems-data-structures/09-consistent-hash-ring.md) · [`gw.01` server skeleton](01-server-skeleton.md) · [`gw.02` API keys](../08-authorization-and-access-control/01-api-keys-and-scopes.md) (the debug scope) · [`gw.04` streaming proxy](04-sse-streaming-proxy.md) (`Forward` and the commitment boundary) · `L10.6` (the Rust engine's Prefill and Release gRPC endpoints) · reading: `S-M07d` [queueing and balls into bins](../../math/07-probability-statistics/93-problem-set-d.md), [model routing and cascades](../11-model-routing-and-cascades/) |
| **Used by** | `gw.08` (its policy reads `route.InferenceRequest`); `gw.07`'s `gateway.Deps` and `gateway.Admin` (the `Router` slot, the workers, routes, and drain handlers, the cache revision); your composition root passes `route.Proxy` and serves the gRPC registry; `load.01`, `ops.01`, and `dur.12`'s canary reach it over HTTP |
| **Milestone** | MS-gateway |
| **Optional depth** | Chen et al., *FrugalGPT* (2023, cascades); Zhong et al., *DistServe* (OSDI 2024) and Patel et al., *Splitwise* (ISCA 2024) on disaggregated prefill and decode; Mitzenmacher, *The Power of Two Choices in Randomized Load Balancing* (2001) |

## Key Takeaways

- Engines announce themselves with a heartbeat every 2 s; a worker that misses `heartbeat_miss_limit` (3) in a row is evicted at exactly $3 \times 2$ s after its last one (`TestHandExample`).
- Three policies pick among the routable workers of a model: `least_outstanding` (fewest in flight plus queued), `weighted` (smooth weighted round robin by KV capacity), and `affinity` (`ds.09`'s ring with bounded loads on the first prompt block, so shared prefixes hit a warm cache until that worker is over its fair share) (`TestAffinityKeepsPrefixTogether`).
- Before the first byte, a worker that is down or answers 503 is excluded and the request routed again; after it, nothing is retried or spliced (`TestFailoverBeforeFirstByte`, `TestNoSpliceAfterFirstByte`).
- A cascade asks the small model first and keeps its answer only when its mean token logprob passes `accept_if`; sweeping the threshold trades cost for accuracy, and the curve on fixture data is reproduced exactly (`TestCascadeAcceptsOrEscalates`, `TestCascadeCurveMatchesFixture`).
- Disaggregated serving runs `Prefill` on a prefill worker that pushes the prompt's KV to a decode worker, then streams from the decode worker with `X-TL-KV-Handle`; with one seed for both, the output equals a unified engine's (`TestDisaggregatedEqualsUnified`).

## How to work this chapter

```bash
ol start gw.05          # writes go/gateway/route/{registry,route,handler,admin}.go with stub bodies
ol tests gw.05          # read the test catalog first
ol check gw.05          # exit code is the verdict
ol check gw.05 --ref-deps   # only if ds.09, gw.01, gw.02, or gw.04 is not passing yet
ol diff  gw.05          # after passing: your code against the reference
```

The package uses the generated gRPC code in `contracts/go/gen` (module `openlearn.urmzd.com/tl/contracts`) and `google.golang.org/grpc`. Build in workspace mode, the way the course overlay does: run `go work init ./go ./contracts/go` once at your repo root (your `go.mod` keeps its `replace` for the contracts and needs no `require` for them), and `go -C go get google.golang.org/grpc@v1.83.1` if your `go.mod` predates this module. In your composition root: serve `controlv1.RegisterWorkerRegistryServer(grpcServer, registry)` on `registry_listen`, put `route.Middleware(router)` in the `Router` slot and `route.Proxy(router, route.Options{})` in the `Proxy` slot, and `router.Ready` in `Deps.Ready`.

---

## 1. Why now

Until now your gateway has one upstream, written in its configuration. In Pass 7 you run several engines: two decode workers and a prefill worker in the disaggregated layout of `L10.6`, a small and a large model, and soon a canary of a new model version (`dur.12`). Engines come and go: a pod is rescheduled, a node is drained, a worker crashes mid-stream (drill `ops.01`). The gateway needs to know who is alive, pick a worker per request in a way that keeps prefix caches warm without overloading anyone, move away from a dead worker before the client notices, and never stitch two workers' tokens into one answer. This module is that router.

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

At $\tau = -0.5$ the cascade matches always-large accuracy (0.935) at 92% of its cost; always-large costs 1.0 and the full sweep shows where the knee is for your models.

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

This is `TestHandExample`.

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

`Route` takes the cascade step and returns `Target`, as sketched in the catalog; the cascade's `Accept` is parsed from the route's `accept_if` strings rather than given as a Go function, because routes arrive through the admin API as JSON (DEVIATIONS B93-06). `X-TL-Route` carries the chosen worker id only for keys with the `debug` scope.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestHandExample` | unit | section 3: picks b, then c, eviction exactly at 6 s, then a | the definition of load and liveness |
| `TestRegistryOverGRPC` | conformance | a real gRPC heartbeat makes a worker routable; drain sets `drain` in the ack and removes it; a worker reporting draining is not routable | `L10.6`'s heartbeat client talks to this |
| `TestModelNotFoundAndNoCapacity` | unit | unknown model 404 `model_not_found`; routed but down 503 `no_capacity`; `Ready` fails with no worker | conformance `route.model`; `/readyz` |
| `TestAffinityKeepsPrefixTogether` | unit, property | 20 requests sharing a system prompt on one worker (`affinity`), 40 distinct prefixes on at least 3 of 4 workers, an overloaded owner skipped (`load`) | warm prefix caches without hot spots |
| `TestWeightedByCapacity` | unit | capacities 1 and 3 give `big, big, small, big` twice | proportional and interleaved |
| `TestCanarySplit` | statistical | 0.1 of 10000 request ids within 4 sd; one id, one backend; no backends means the route's model | `dur.12`'s canary weight |
| `TestModelRewrittenForBackend` | unit | an alias reaches the engine as the served model id | engines answer only their own id |
| `TestFailoverBeforeFirstByte` | fault | a refused worker and a 503 worker are each tried once and the request completes on the next | conformance and drill `ops.01` |
| `TestNoSpliceAfterFirstByte` | fault | a worker that dies after one token: the client gets that token and an SSE error event; no other worker is called | no repeated or spliced tokens |
| `TestRouteHeaderOnlyForDebug` | unit | `X-TL-Route` only for the `debug` scope | internal topology stays internal |
| `TestModelsList` | unit | `GET /v1/models` lists route and worker models; an unknown id is 404 | conformance `schema.models` on the gateway |
| `TestPrefixKey` | unit | the rendered chat prefix; 64 bytes at most | the affinity key |
| `TestParseAcceptIf` | boundary | `mean_logprob` and `min_logprob` with `>` and `>=`; empty answers rejected; malformed rules | cascade rules from the admin API |
| `TestCascadeAcceptsOrEscalates` | unit | confident small answer kept (large never called, logprobs requested); unsure one escalated; streamed request to the last step | cost savings without wrong answers |
| `TestCascadeCurveMatchesFixture` | conformance, golden | the sweep over `cascade_sweep.json` equals the oracle's escalation, cost, and accuracy at all 13 thresholds | the cost and quality curve in 2.6 |
| `TestDisaggregatedEqualsUnified` | differential | the disaggregated stream equals the unified one; prompt ids, defaults, `decode_target` = D's KV address; a generated seed reaches D | `L10.6`'s disaggregated milestone step |
| `TestDisaggregatedReleaseOnAbort` | fault | D dies mid-stream: `Release` is called with the handle and nothing stays held | KV blocks do not leak |
| `TestAdminRoutesWorkersDrain` | unit, conformance | workers with the epoch; routes 428, 412, 400 for a bad cascade, 200 with a new ETag; drain 404 and 202 | `gw.07` mounts these handlers |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. eviction and draining off by one: `<=` on the deadline, draining workers still routable | a dead worker gets traffic for one more interval; a rollout keeps sending to the pod it drains | `TestHandExample`, `TestRegistryOverGRPC`, `TestAdminRoutesWorkersDrain` (mutants `s01`, `s15`) |
| 2. retrying after the first byte | the client reads "The cat The cat sat": tokens repeated or spliced | `TestNoSpliceAfterFirstByte` (mutant `s04`) |
| 3. load from in-flight counts only | a worker with a long engine queue keeps receiving work | `TestHandExample` (mutant `s02`) |
| 4. affinity without bounded loads | one hot system prompt saturates one worker while others idle | `TestAffinityKeepsPrefixTogether` (mutant `s03`) |
| 5. failover that relays a 503, or retries the same worker | clients see 503s while healthy workers sit idle | `TestFailoverBeforeFirstByte` (mutants `s05`, `s06`) |
| 6. disaggregation details: `decode_target` set to D's HTTP address, no shared seed, no `Release` on failure | prefill cannot push KV; sampled outputs differ from unified; D's pool leaks blocks | `TestDisaggregatedEqualsUnified`, `TestDisaggregatedReleaseOnAbort` (mutants `s11`, `s12`, `s13`) |
| 7. canary and model ids: weights ignored, the public id sent upstream, an unknown model answered 503 | the canary gets half the traffic; engines answer 404; clients retry a typo forever | `TestCanarySplit`, `TestModelRewrittenForBackend`, `TestModelNotFoundAndNoCapacity` (mutants `s07`, `s10`, `s17`) |
| 8. cascade mistakes: accepting everything, cascading a stream, accepting an empty answer, a cost that forgets the small call | quality drops silently; streams stall; the curve understates cost | `TestCascadeAcceptsOrEscalates`, `TestParseAcceptIf`, `TestCascadeCurveMatchesFixture` (mutants `s08`, `s09`, `s16`, `s18`) |
| 9. `X-TL-Route` for every key | worker ids leak to every client | `TestRouteHeaderOnlyForDebug` (mutant `s14`) |
| 10. `PUT /admin/v1/routes` without the `If-Match` check | two operators' edits overwrite each other silently | `TestAdminRoutesWorkersDrain` (mutant `s19`) |

## 6. Where it's used next

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

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| affinity on a prefix key | llm-d and KServe KV-cache-aware schedulers, SGLang router | scoring workers by the KV blocks they actually hold, not by a hash | [llm-d](https://github.com/llm-d/llm-d) (free), [SGLang router](https://github.com/sgl-project/sglang) (free) |
| backends and failover | saige `provider/router` | immutable model configurations, sticky selection for prompt-cache affinity, failover only before commitment | saige `provider/router` |
| cascades | FrugalGPT, RouteLLM | learned routers that predict which model a prompt needs | [RouteLLM](https://github.com/lm-sys/RouteLLM) (free) |
| disaggregated orchestration | vLLM disaggregated prefill, Dynamo, Mooncake | KV transfer over RDMA, a global KV pool, prefill scheduling across nodes | [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) (free), [Mooncake](https://github.com/kvcache-ai/Mooncake) (free) |
