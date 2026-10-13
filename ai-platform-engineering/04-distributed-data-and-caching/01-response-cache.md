<!-- ol:module gw.06 -->
# Response cache: LRU + TTL, singleflight, tenant-scoped keys

## Overview

| | |
|---|---|
| **Module** | `gw.06` · build · Go · Pass 7 · 4 h |
| **You build** | `go/gateway/cache/cache.go`: `LRU` (`New`, `Get`, `Put`, `Len`, `Purge`), `Canonicalize`, `KeyOf`, `Middleware` (the cache stage with singleflight and stream replay), `PurgeHandler` |
| **Contract** | `X-TL-Cache: hit\|miss` on cacheable requests in [`course/contracts/openapi/openai-subset.v1.yaml`](../../course/contracts/openapi/openai-subset.v1.yaml) · `POST /admin/v1/cache:purge` in [`course/contracts/openapi/admin.v1.yaml`](../../course/contracts/openapi/admin.v1.yaml) · `[gateway]` `cache_entries`, `cache_ttl_s` in [`course/contracts/config/runtime.schema.json`](../../course/contracts/config/runtime.schema.json) |
| **Tests** | `course/tests/go/gw_06/` (what they check: section 4); the singleflight test runs under the race detector |
| **Needs** | [`gw.01` server skeleton](../12-gateway/01-server-skeleton.md) (the chain, the `Exchange`) · [`gw.02` API keys](../08-authorization-and-access-control/01-api-keys-and-scopes.md) (the tenant) · reading: this topic's [README](README.md), the [practice drill go/02](../../practice/build/cloud/go/README.md) |
| **Used by** | `gw.07`'s `gateway.Deps` (the `Cache` slot, `Rev` from the route epoch) and `gateway.Admin` (`PurgeHandler`) |
| **Milestone** | MS-gateway |
| **Optional depth** | the Go team's `golang.org/x/sync/singleflight` source (free); Megiddo and Modha, *ARC: A Self-Tuning, Low Overhead Replacement Cache* (FAST 2003) |

## Key Takeaways

- Only deterministic requests are cached: `temperature` 0, or a fixed `seed`. Anything sampled without a seed would replay one random answer as if it were the only one (`TestNotCacheableHasNoHeader`).
- The key is a hash of the **tenant**, the **configuration revision**, and the **canonical** request: field order, number spelling, explicit defaults, and the `user` field do not change it; the model, the limits, the messages, and `stream` do (`TestHandExample`, `TestCanonicalDistinctions`).
- An LRU of at most `cache_entries` with a TTL per entry: an entry lives exactly $[t_{\text{put}}, t_{\text{put}} + \mathit{ttl})$, and a hit makes it the most recently used (`TestLRUEvictionOrder`, `TestTTLExpiry`).
- **Singleflight**: 100 identical requests arriving together make one upstream call; the others wait for it and are answered from its result (`TestSingleflightOneUpstreamCall`).
- Only whole 200 answers are stored, and a stored stream is replayed byte for byte with its `Content-Type` (`TestStreamReplayByteExact`, `TestErrorsAndCutStreamsNotCached`).

## How to work this chapter

```bash
ol start gw.06          # writes go/gateway/cache/cache.go with stub bodies
ol tests gw.06          # read the test catalog first
ol check gw.06          # runs the course tests with -race
ol check gw.06 --ref-deps   # only if your gw.01 or gw.02 is not passing yet
ol diff  gw.06          # after passing: your code against the reference
```

Then plug `cache.Middleware(cache.New(cfg.Gateway.CacheEntries, clock), cache.Options{TTL: ..., Rev: routeEpoch})` into the `Cache` slot of your composition root.

---

## 1. Why now

The same questions reach your gateway again and again: an agent's evaluation suite (`ag.09`) replays its cases at temperature 0 on every run, a docs assistant gets the same "how do I create a key?" from every new user, and the load generator (`load.01`) sends a fixed prompt set. Each one costs a full prefill and decode on an engine that is the most expensive thing in the system, for an answer the gateway has already seen. A response cache in front of routing turns those repeats into microseconds. It must also never do the three things that make caches dangerous: show one tenant another tenant's answer, serve a stale answer after the model changed, or let a burst of identical misses stampede the engines.

## 2. Principles

### 2.1 LRU with a TTL

| Symbol | Meaning | Type |
|---|---|---|
| $N$ | capacity, `cache_entries` | integer, 4096 by default |
| $\mathit{ttl}$ | lifetime of an entry, `cache_ttl_s` | duration, 300 s by default |
| $t_p$ | when an entry was put | time |

An **LRU** (least recently used) cache keeps entries in a list ordered by last use, with a map from key to list element: `Get` moves the hit to the front, `Put` inserts at the front and, when there are more than $N$ entries, removes from the back. Both are O(1). (Go's `container/list` is the list; the map holds `*list.Element`.) The TTL bounds staleness: an entry is valid while $\mathit{now} < t_p + \mathit{ttl}$; a `Get` at or after that time is a miss and removes the entry. The clock is the gateway's `Clock`, so tests move time with a fake.

### 2.2 What may be cached, and the canonical request

An answer may be replayed only if asking again would give the same answer. `spec/sampling.md` makes that true in two cases: `temperature` 0 (greedy), and any temperature with a fixed `seed` on the same engine. Requests without either are passed through untouched, with no `X-TL-Cache` header.

Two requests that ask the same thing must have the same key, so the body is reduced to a **canonical form**:

1. Parse it as a JSON object.
2. Fill every omitted field that has a contract default (`temperature` 1, `top_p` 1, `top_k` 0, `min_p` 0, `repetition_penalty` 1, `presence_penalty` 0, `frequency_penalty` 0, `n` 1, `stream` false, `logprobs` null, `stop` null), so `{}` and `{"top_p": 1}` agree.
3. Drop fields that do not change the answer: `user`.
4. Re-encode with sorted keys: Go's `encoding/json` sorts map keys, and numbers go through `float64`, so `0` and `0.0` agree.

`stream` stays in: a streamed answer is SSE bytes and a plain one is JSON, so they are different answers. The model, `max_tokens`, and every message stay in.

### 2.3 Scoped keys

$$\mathit{key} = \mathrm{SHA256}(\mathit{tenant} \,\|\, 0 \,\|\, \mathit{rev} \,\|\, 0 \,\|\, \mathit{path} \,\|\, 0 \,\|\, \mathit{canonical})$$

The **tenant** (from `gw.02`'s `Principal`) makes the cache private per tenant: two keys of one company share answers, two companies never do, even for byte-equal requests (the prompt may contain their data; so may the answer). The **revision** (`Options.Rev`, for example the route table's epoch from `gw.05`) changes when a route points a public model at a new version, so answers from the old model become unreachable at once instead of lingering for the TTL. Requests without a principal (no authn stage) are not cached at all.

The selected API version also partitions responses: v2 includes usage fields that v1 omits. The middleware adds a version suffix to the revision for v2. An absent version and explicit v1 share the original partition; v2 gets its own cached answer and singleflight group (`TestAPIVersionIsolation`). Select the version before the cache so a configured v2 default reaches the same partition as an explicit v2 header.

### 2.4 Singleflight

On a miss, the first request for a key becomes the **leader**: it registers the key as in flight and goes upstream. A request for the same key that arrives while the leader runs **waits** for it instead of going upstream too, then is answered from the leader's result as a hit. Without this, a popular question that just expired sends 100 identical prefills to the engines at once (a thundering herd, or cache stampede). Two details make it correct:

- Check the cache **again** after taking the in-flight lock. A leader that finished between a follower's `Get` and its lock has already stored the entry and left the in-flight table; without the second check, the follower would become a second leader.
- If the leader's answer was not cacheable (an error, a cut stream), waiters go upstream themselves rather than replaying a failure.

### 2.5 What is stored and replayed

While the leader's response passes to its client, a recorder keeps a copy (up to 1 MiB). It is stored only when it is a **whole 200**: status 200, and either valid JSON or an SSE stream that ends with `data: [DONE]\n\n`. A stream cut by an engine failure ends with `gw.04`'s error event instead and is not stored; a 503 is not stored. A hit replays the stored status, `Content-Type`, and body, so a streaming client gets the same SSE bytes it would have gotten from the engine (all at once). Hits set `X-TL-Cache: hit`, misses `X-TL-Cache: miss`; hits record no usage on the `Exchange`, so the limiter (`gw.03`) refunds their tokens and the meter (`gw.07`) records them as free.

## 3. Worked example by hand

Two requests from tenant `acme`:

```json
A: {"model":"smol","messages":[{"role":"user","content":"hi"}],"temperature":0,"user":"alice"}
B: {"temperature":0.0,"top_p":1,"messages":[{"role":"user","content":"hi"}],"model":"smol"}
```

Canonicalize A: fill the defaults it omits (`top_p` 1, `top_k` 0, `min_p` 0, `repetition_penalty` 1, both penalties 0, `n` 1, `stream` false, `logprobs` and `stop` null), drop `user`, sort the keys. B already has `top_p`, writes `0.0` for 0, and has no `user`. Both become

```json
{"frequency_penalty":0,"logprobs":null,"messages":[{"content":"hi","role":"user"}],"min_p":0,"model":"smol","n":1,"presence_penalty":0,"repetition_penalty":1,"stop":null,"stream":false,"temperature":0,"top_k":0,"top_p":1}
```

(inside each message the keys are sorted too). `temperature` is 0, so both are cacheable, and with the same tenant and revision they share one key: A is a miss that reaches the engine, B is a hit. The same body at `temperature` 0.7 is not cacheable; at 0.7 with `"seed": 42` it is. This is `TestHandExample`.

## 4. The interface

```go
package cache // import "tinyllm/gateway/cache"

type Key [32]byte
type Entry struct { Status int; ContentType string; Body []byte; Model string }

func New(capacity int, clock server.Clock) *LRU
func (c *LRU) Get(ctx context.Context, k Key) (Entry, bool)
func (c *LRU) Put(ctx context.Context, k Key, e Entry, ttl time.Duration)
func (c *LRU) Len() int
func (c *LRU) Purge(model string) int // "" = everything

type CanonicalRequest struct { Path string; JSON []byte }
func Canonicalize(path string, body []byte) (CanonicalRequest, bool /* cacheable */, error)
func KeyOf(p auth.Principal, cfgRev string, r CanonicalRequest) Key

type Options struct { TTL time.Duration; Rev func() string; MaxEntryBytes int }
func Middleware(c *LRU, o Options) server.Middleware // chat and completions only
func PurgeHandler(c *LRU) http.Handler                // POST /admin/v1/cache:purge
```

Use only the standard library. The stage caches `POST /v1/chat/completions` and `POST /v1/completions`; embeddings and everything else pass through.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestHandExample` | unit | section 3: both bodies give the exact canonical JSON and one key; 0.7 without seed is not cacheable, with seed it is; no temperature means 1 | you and the tests agree on equality |
| `TestCanonicalDistinctions` | unit | model, `max_tokens`, message text, and `stream` each change the canonical form; a non-object is an error | different questions never share an answer |
| `TestKeyOfTenantAndRevision` | unit | same tenant shares across keys; another tenant or revision does not | privacy and freshness |
| `TestLRUEvictionOrder` | unit | capacity 2: put a, b; get a; put c evicts b | the hot set survives |
| `TestTTLExpiry` | fault | hit at 9.999 s, miss at exactly 10 s, entry removed | bounded staleness |
| `TestPurge` | unit | purge one model, then everything | `POST /admin/v1/cache:purge` |
| `TestHitAfterMiss` | unit | miss then hit with the same body and content type, one upstream call | conformance case `cache.hit` |
| `TestSingleflightOneUpstreamCall` | fault | 100 concurrent identical requests: one upstream call, 100 identical bodies | no stampede after an expiry |
| `TestTenantIsolation` | fault | another tenant's identical request is a miss that reaches the upstream | no cross-tenant leaks |
| `TestAPIVersionIsolation` | regression | v1 and v2 each populate one cache entry; implicit and explicit v1 share | no replay across incompatible response schemas |
| `TestStreamReplayByteExact` | unit | a cached stream replays the same SSE bytes with `text/event-stream` | streaming clients cannot tell |
| `TestErrorsAndCutStreamsNotCached` | fault | a 503 and a stream without `[DONE]` are not stored | failures are not replayed for 5 minutes |
| `TestNotCacheableHasNoHeader` | unit | a sampled request: no header, every call upstream | sampling stays random |
| `TestPurgeHandler` | unit, conformance | `{"purged": n}` with and without a model | `gw.07` mounts it |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. a key without the tenant, or without the revision | tenant B reads tenant A's answer; after a model rollout the old model's answers keep coming | `TestTenantIsolation`, `TestKeyOfTenantAndRevision` (mutants `s01`, `s11`) |
| 2. hashing the raw body, skipping defaults, keeping `user`, or dropping `stream` | equal questions miss; a streamed request receives a JSON body | `TestHandExample`, `TestCanonicalDistinctions`, `TestHitAfterMiss` (mutants `s02`, `s03`, `s10`, `s13`, `s15`) |
| 3. caching failures | one engine outage is replayed to everyone for 5 minutes; a cut stream becomes the answer | `TestErrorsAndCutStreamsNotCached` (mutants `s08`, `s09`) |
| 4. caching sampled requests | every "write me a poem" gets the same poem | `TestNotCacheableHasNoHeader`, `TestHandExample` (mutant `s04`) |
| 5. `now > expires` instead of `now >= expires` | entries live one tick past their TTL | `TestTTLExpiry` (mutant `s05`) |
| 6. a `Get` that does not refresh recency | the most used entry is the first evicted (FIFO, not LRU) | `TestLRUEvictionOrder` (mutant `s06`) |
| 7. no singleflight, or no second check under the lock | an expiring popular entry sends a burst of identical prefills to the engines | `TestSingleflightOneUpstreamCall` (mutant `s07`) |
| 8. replaying without `Content-Type` | SSE clients do not parse the replayed stream | `TestStreamReplayByteExact`, `TestHitAfterMiss` (mutant `s12`) |
| 9. a purge that ignores its model | purging one model empties the cache | `TestPurge`, `TestPurgeHandler` (mutant `s14`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `gw.01` | the `Cache` slot, the `Exchange` (`tl.cache.hit`), `ex.Request` for the body |
| Back | `gw.02` | `Principal.Tenant` scopes every key |
| Forward | `gw.07` | `gateway.Deps` calls `Middleware` with `Rev` set to the route table's ETag; `gateway.Admin` builds `PurgeHandler` |
| Forward | `load.01` | the load report's `X-TL-Cache: hit` ratio |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| exact-match response cache | semantic caches (GPTCache, Redis LangCache) | hits for paraphrases by embedding similarity, with the false-hit risk that brings | [GPTCache](https://github.com/zilliztech/GPTCache) (free) |
| the gateway's response cache | provider prompt caching (Anthropic, OpenAI), the engine prefix cache (`L8.4`) | reuse of the prompt's KV, not the answer: works for sampled requests too | [vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html) (free) |
| singleflight | `golang.org/x/sync/singleflight`, groupcache | request coalescing as a library; peer-to-peer cache filling | [groupcache](https://github.com/golang/groupcache) (free) |
| one process's LRU | saige's separate response cache, prompt cache, step record, and journal | identity private by default, each cache its own contract | saige `cache` package |
