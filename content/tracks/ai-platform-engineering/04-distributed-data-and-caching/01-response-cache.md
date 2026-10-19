# Response cache: LRU + TTL, singleflight, tenant-scoped keys

## Overview

| | |
|---|---|
| **Module** | `gw.06` · build · Go · Pass 7 · 4 h |
| **Solution** | [`go/gateway/cache/cache.go`](../../../../course/solutions/go/gateway/cache/cache.go): `LRU` (`New`, `Get`, `Put`, `Len`, `Purge`), `Canonicalize`, `KeyOf`, `Middleware` (the cache stage with singleflight and stream replay), `PurgeHandler` |
| **Contract** | `X-TL-Cache: hit\|miss` on cacheable requests in [`course/contracts/openapi/openai-subset.v1.yaml`](../../../../course/contracts/openapi/openai-subset.v1.yaml) · `POST /admin/v1/cache:purge` in [`course/contracts/openapi/admin.v1.yaml`](../../../../course/contracts/openapi/admin.v1.yaml) · `[gateway]` `cache_entries`, `cache_ttl_s` in [`course/contracts/config/runtime.schema.json`](../../../../course/contracts/config/runtime.schema.json) |
| **Needs** | [`gw.01` server skeleton](../12-gateway/01-server-skeleton.md) (the chain, the `Exchange`) · [`gw.02` API keys](../08-authorization-and-access-control/01-api-keys-and-scopes.md) (the tenant) · reading: this topic's [README](README.md), the [practice drill go/02](../../../../practice/build/cloud/go/README.md) |
| **Used by** | `gw.07`'s `gateway.Deps` (the `Cache` slot, `Rev` from the route epoch) and `gateway.Admin` (`PurgeHandler`); part of the full gateway |
| **Optional depth** | the Go team's `golang.org/x/sync/singleflight` source (free); Megiddo and Modha, *ARC: A Self-Tuning, Low Overhead Replacement Cache* (FAST 2003) |

## Key Takeaways

- Only deterministic requests are cached: `temperature` 0, or a fixed `seed`. Anything sampled without a seed would replay one random answer as if it were the only one.
- The key is a hash of the **tenant**, the **configuration revision**, and the **canonical** request: field order, number spelling, explicit defaults, and the `user` field do not change it; the model, the limits, the messages, and `stream` do.
- An LRU of at most `cache_entries` with a TTL per entry: an entry lives exactly $[t_{\text{put}}, t_{\text{put}} + \mathit{ttl})$, and a hit makes it the most recently used.
- **Singleflight**: 100 identical requests arriving together make one upstream call; the others wait for it and are answered from its result.
- Only whole 200 answers are stored, and a stored stream is replayed byte for byte with its `Content-Type`.

## How to read this chapter

Sections 1 to 3 build the cache from first principles and work the canonical key by hand. Section 4 fixes the interface. Section 5 walks through [`cache.go`](../../../../course/solutions/go/gateway/cache/cache.go) function by function and explains why each choice is the right one. The gateway's composition root (`gw.07`) plugs it in as `cache.Middleware(cache.New(cfg.Gateway.CacheEntries, clock), cache.Options{TTL: ..., Rev: routeEpoch})` in the `Cache` slot. Try to predict each step of the solution before reading it.

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

The selected API version also partitions responses: v2 includes usage fields that v1 omits. The middleware adds a version suffix to the revision for v2. An absent version and explicit v1 share the original partition; v2 gets its own cached answer and singleflight group. Select the version before the cache so a configured v2 default reaches the same partition as an explicit v2 header.

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

(inside each message the keys are sorted too). `temperature` is 0, so both are cacheable, and with the same tenant and revision they share one key: A is a miss that reaches the engine, B is a hit. The same body at `temperature` 0.7 is not cacheable; at 0.7 with `"seed": 42` it is.

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

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example | unit | section 3: both bodies give the exact canonical JSON and one key; 0.7 without seed is not cacheable, with seed it is; no temperature means 1 | every implementation agrees on equality |
| canonical distinctions | unit | model, `max_tokens`, message text, and `stream` each change the canonical form; a non-object is an error | different questions never share an answer |
| key scoped by tenant and revision | unit | same tenant shares across keys; another tenant or revision does not | privacy and freshness |
| LRU eviction order | unit | capacity 2: put a, b; get a; put c evicts b | the hot set survives |
| TTL expiry | fault | hit at 9.999 s, miss at exactly 10 s, entry removed | bounded staleness |
| purge | unit | purge one model, then everything | `POST /admin/v1/cache:purge` |
| hit after miss | unit | miss then hit with the same body and content type, one upstream call | conformance case `cache.hit` |
| singleflight: one upstream call | fault | 100 concurrent identical requests: one upstream call, 100 identical bodies | no stampede after an expiry |
| tenant isolation | fault | another tenant's identical request is a miss that reaches the upstream | no cross-tenant leaks |
| API version isolation | regression | v1 and v2 each populate one cache entry; implicit and explicit v1 share | no replay across incompatible response schemas |
| byte-exact stream replay | unit | a cached stream replays the same SSE bytes with `text/event-stream` | streaming clients cannot tell |
| errors and cut streams not cached | fault | a 503 and a stream without `[DONE]` are not stored | failures are not replayed for 5 minutes |
| sampled request has no header | unit | a sampled request: no header, every call upstream | sampling stays random |
| purge handler | unit, conformance | `{"purged": n}` with and without a model | `gw.07` mounts it |

## 5. The solution, step by step

### From first principles

1. **A cache may only return what the upstream would have returned.** So the first question is not "how do I store it" but "when is the answer a function of the request". By `spec/sampling.md` that is temperature 0 or a fixed seed. Everything else must bypass the cache entirely, including the `X-TL-Cache` header, so clients never think a sampled answer was cached.
2. **Equality must be semantic, not textual.** Requests A and B in section 3 differ in bytes but not in meaning. A key over raw bytes misses; a key over the canonical form (defaults filled, irrelevant fields dropped, keys sorted, numbers normalised) hits. Anything that can change the answer (model, messages, `max_tokens`, `stream`) must stay in.
3. **The key must also carry who asked and which configuration answered.** Tenant isolation is a privacy requirement, so the tenant goes into the hash, not into a check after lookup. The configuration revision goes in too: changing it makes every old key unreachable in O(1), with no scan and no wait for TTLs.
4. **Bounded memory and bounded staleness are separate limits.** Capacity bounds memory (evict the least recently used); TTL bounds staleness (expire by time). LRU needs O(1) lookup and O(1) reordering, which forces a hash map pointing into a doubly linked list.
5. **Concurrency changes what a miss means.** If 100 identical requests miss together, only one should go upstream. That forces a table of in-flight keys guarded by a lock, waiters that block on the leader, and a re-check of the cache under the lock to close the race where the leader finishes between a follower's `Get` and its `Lock`.
6. **Only whole successes are worth replaying.** A recorded answer is stored only when it is status 200, fits the size cap, and is complete (valid JSON, or SSE ending in `[DONE]`). Failures are transient; caching them would make them last for the TTL.

### The code, annotated

**Step 1: the LRU.** The list front is the most recently used entry; the map gives O(1) access to the list element.

```go
func (c *LRU) Get(ctx context.Context, k Key) (Entry, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	el, ok := c.m[k]
	if !ok {
		return Entry{}, false
	}
	it := el.Value.(*item)
	// Valid on [put, put+ttl): at exactly expires the entry is already dead.
	// Expired entries are removed lazily, on the Get that finds them.
	if !c.clock.Now().Before(it.expires) {
		c.ll.Remove(el)
		delete(c.m, k)
		return Entry{}, false
	}
	c.ll.MoveToFront(el) // a hit refreshes recency: this is what makes it LRU, not FIFO
	return it.e, true
}

func (c *LRU) Put(ctx context.Context, k Key, e Entry, ttl time.Duration) {
	if c.capacity <= 0 || ttl <= 0 {
		return // a zero capacity or TTL turns the cache off without special cases elsewhere
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	exp := c.clock.Now().Add(ttl)
	if el, ok := c.m[k]; ok { // overwrite in place: never two list nodes for one key
		el.Value = &item{key: k, e: e, expires: exp}
		c.ll.MoveToFront(el)
		return
	}
	c.m[k] = c.ll.PushFront(&item{key: k, e: e, expires: exp})
	for c.ll.Len() > c.capacity { // evict from the back, keeping map and list in sync
		last := c.ll.Back()
		c.ll.Remove(last)
		delete(c.m, last.Value.(*item).key)
	}
}
```

The item stores its own key so eviction from the back can delete the map entry without a reverse lookup. `Purge` walks the list and removes items whose `Model` matches (or all, for `""`), saving `next` before removing so the walk survives removal.

**Step 2: canonicalisation and cacheability.**

```go
var defaults = map[string]any{
	"temperature": 1.0, "top_p": 1.0, "top_k": 0.0, "min_p": 0.0, "repetition_penalty": 1.0,
	"presence_penalty": 0.0, "frequency_penalty": 0.0, "n": 1.0, "stream": false,
	"logprobs": nil, "stop": nil,
}
var ignored = []string{"user"}

func Canonicalize(path string, body []byte) (CanonicalRequest, bool, error) {
	var m map[string]any
	// Unmarshalling into map[string]any turns every number into float64,
	// so 0 and 0.0 become the same value before re-encoding.
	if err := json.Unmarshal(body, &m); err != nil || m == nil {
		return CanonicalRequest{}, false, errors.New("the request body must be a JSON object")
	}
	for k, v := range defaults { // explicit default == omitted
		if _, ok := m[k]; !ok {
			m[k] = v
		}
	}
	for _, k := range ignored { // fields that cannot change the answer
		delete(m, k)
	}
	if v, ok := m["logprobs"].(bool); ok && !v {
		m["logprobs"] = nil // logprobs:false means the same as absent
	}
	temp, _ := m["temperature"].(float64) // after defaults: missing means 1, not 0
	_, seeded := m["seed"].(float64)
	canon, err := json.Marshal(m) // encoding/json sorts map keys
	if err != nil {
		return CanonicalRequest{}, false, err
	}
	return CanonicalRequest{Path: path, JSON: canon}, temp == 0 || seeded, nil
}
```

The cacheability test runs after defaults are filled, which is why a body without `temperature` is not cacheable: it means 1.

**Step 3: the scoped key.**

```go
func KeyOf(p auth.Principal, cfgRev string, r CanonicalRequest) Key {
	h := sha256.New()
	for _, part := range []string{p.Tenant, cfgRev, r.Path} {
		h.Write([]byte(part))
		h.Write([]byte{0}) // separator: ("ab","c") and ("a","bc") must hash differently
	}
	h.Write(r.JSON)
	var k Key
	copy(k[:], h.Sum(nil))
	return k
}
```

The NUL separators make the concatenation unambiguous. Hashing to a fixed 32-byte array makes the key a cheap, comparable map key regardless of body size. In the middleware, a non-v1 API version is appended to `rev` (`rev += "\x00api:" + version`), so v2 answers get their own keys while an absent header and explicit `1` keep the original one.

**Step 4: singleflight, inside `Middleware`.** After the early exits (unauthenticated, not POST, not chat or completions, unreadable body, not cacheable), the stage loops:

```go
k := KeyOf(p, rev, canon)
for {
	if e, hit := c.Get(r.Context(), k); hit { // fast path, no global lock
		ex.SetAttr("tl.cache.hit", true)
		replay(w, e)
		return
	}
	mu.Lock()
	if cl, waiting := inflight[k]; waiting { // someone is already fetching this key
		mu.Unlock()
		select {
		case <-cl.done:
		case <-r.Context().Done(): // the client left: stop waiting
			return
		}
		if cl.ok {
			ex.SetAttr("tl.cache.hit", true)
			replay(w, cl.entry) // served from the leader's result, even if TTL is 0
			return
		}
		break // the leader got no cacheable answer: go upstream ourselves
	}
	// Second check under the lock: a leader that finished between our Get and
	// this Lock stored its entry before leaving inflight.
	if e, hit := c.Get(r.Context(), k); hit {
		mu.Unlock()
		ex.SetAttr("tl.cache.hit", true)
		replay(w, e)
		return
	}
	cl := &call{done: make(chan struct{})}
	inflight[k] = cl // we are the leader
	mu.Unlock()
	ex.SetAttr("tl.cache.hit", false)
	w.Header().Set("X-TL-Cache", "miss")
	rec := &recorder{ResponseWriter: w, max: o.MaxEntryBytes}
	func() {
		defer func() { // always release waiters, even if next panics
			mu.Lock()
			delete(inflight, k)
			mu.Unlock()
			close(cl.done)
		}()
		next.ServeHTTP(rec, r)
		// ... store when whole (step 5)
	}()
	return
}
// reached only by break: a follower whose leader failed
ex.SetAttr("tl.cache.hit", false)
w.Header().Set("X-TL-Cache", "miss")
next.ServeHTTP(w, r)
```

Ordering matters in the deferred cleanup: `c.Put` happens inside the inner function before the defer runs, so by the time the key leaves `inflight` the entry is already in the LRU. That ordering is what makes the second check sufficient. `cl.entry` and `cl.ok` are written before `close(cl.done)`, and the channel close is the happens-before edge that makes reading them in waiters race-free.

**Step 5: record, and store only whole answers.**

```go
func (r *recorder) Write(b []byte) (int, error) {
	if r.status == 0 {
		r.status = http.StatusOK // implicit 200, as net/http does
	}
	if !r.over {
		if r.buf.Len()+len(b) > r.max {
			r.over = true // too big to cache: drop the copy, keep streaming
			r.buf.Reset()
		} else {
			r.buf.Write(b)
		}
	}
	return r.ResponseWriter.Write(b) // the client always gets every byte, live
}

func complete(contentType string, body []byte) bool {
	if bytes.HasPrefix([]byte(contentType), []byte("text/event-stream")) {
		return bytes.HasSuffix(body, []byte("data: [DONE]\n\n")) // a cut stream lacks this
	}
	return json.Valid(body)
}

// in the leader, after next.ServeHTTP:
ct := w.Header().Get("Content-Type")
if rec.status == http.StatusOK && !rec.over && complete(ct, rec.buf.Bytes()) {
	cl.entry = Entry{Status: rec.status, ContentType: ct, Body: append([]byte(nil), rec.buf.Bytes()...), Model: req.Model}
	cl.ok = true
	c.Put(r.Context(), k, cl.entry, o.TTL)
}
```

The recorder wraps rather than buffers: the leader's client sees tokens as they stream, and the copy is a side effect. `Unwrap` exposes the underlying writer so `http.ResponseController` can still flush through it. The body is copied out of the buffer so the stored entry does not alias memory the recorder owns. `replay` sets `Content-Type` and `X-TL-Cache: hit` before `WriteHeader`, then writes the body in one call.

Full solution: [`cache.go`](../../../../course/solutions/go/gateway/cache/cache.go).

### Why this solution

- **Cost.** `Get`, `Put`, and the in-flight check are O(1) under a mutex; canonicalisation is O(body) for the JSON parse and SHA-256. A hit costs microseconds against a prefill and decode of tens to hundreds of milliseconds. `Purge` is O(N), which is acceptable for a rare admin action. Memory is at most `cache_entries` times `MaxEntryBytes` (4096 times 1 MiB in the worst case; typical answers are a few KiB).
- **One lock for the LRU, one for in-flight.** A single coarse mutex per structure is simpler than sharding and contention is low at gateway request rates; sharding is the next step if profiles show it.
- **Lazy expiry instead of a sweeper.** No background goroutine, no timer per entry; expired entries cost memory until touched or evicted, which `Len` admits.
- **Revision in the key instead of invalidation.** Purging on every route change would need the gateway to know which entries came from which route. Folding the revision into the key makes stale answers unreachable instantly; LRU reclaims their memory later.
- **Alternatives.** `golang.org/x/sync/singleflight` would do the coalescing, but the chapter keeps to the standard library and the waiters need the leader's *cacheable* result, not any result. A semantic (embedding) cache would raise hit rates but can return a wrong answer for a near paraphrase, which an exact-match cache never does.

### Explain it back

1. Why does `Get` treat `now == expires` as expired?

   <details><summary>Answer</summary>

   An entry's lifetime is the half-open interval $[t_p, t_p + \mathit{ttl})$. Using `Before` makes the TTL exact; `now > expires` would let it live one tick past its TTL.

   </details>

2. Why is the cache re-checked after taking `mu`, when it was just checked before?

   <details><summary>Answer</summary>

   Between the first `Get` and `mu.Lock`, a leader may have stored its entry and removed itself from `inflight`. Without the second check, this request would find no in-flight call and become a second leader, sending a duplicate upstream call.

   </details>

3. Why are the tenant and revision hashed into the key rather than checked after lookup?

   <details><summary>Answer</summary>

   Hashing them in means a lookup for tenant B can never even find tenant A's entry, so there is no comparison to get wrong. For the revision, a new value makes all old keys unreachable at once without scanning or purging.

   </details>

4. A leader's upstream returns 503. What do its 99 waiters do, and why not replay the 503?

   <details><summary>Answer</summary>

   `cl.ok` stays false, so each waiter breaks out of the loop and goes upstream itself with `X-TL-Cache: miss`. Replaying the 503 would turn one transient failure into 100; the engine may already have recovered.

   </details>

5. Why does `stream` stay in the canonical form while `user` is dropped?

   <details><summary>Answer</summary>

   `stream` changes the bytes of the answer (SSE versus JSON), so a streamed and a non-streamed request have different correct responses. `user` is an end-user label for abuse tracking that does not influence generation, so keeping it would only split identical answers across keys.

   </details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. a key without the tenant, or without the revision | tenant B reads tenant A's answer; after a model rollout the old model's answers keep coming | `KeyOf` hashes tenant and revision before the body; see step 3 |
| 2. hashing the raw body, skipping defaults, keeping `user`, or dropping `stream` | equal questions miss; a streamed request receives a JSON body | `Canonicalize` fills `defaults`, deletes `ignored`, keeps `stream`, re-encodes; see step 2 |
| 3. caching failures | one engine outage is replayed to everyone for 5 minutes; a cut stream becomes the answer | stores only `status == 200`, under the size cap, and `complete`; see step 5 |
| 4. caching sampled requests | every "write me a poem" gets the same poem | `temp == 0 || seeded` gates caching; otherwise `next` runs with no header |
| 5. `now > expires` instead of `now >= expires` | entries live one tick past their TTL | `Get` treats `!Now().Before(expires)` as expired; see step 1 |
| 6. a `Get` that does not refresh recency | the most used entry is the first evicted (FIFO, not LRU) | `Get` calls `MoveToFront` on a hit |
| 7. no singleflight, or no second check under the lock | an expiring popular entry sends a burst of identical prefills to the engines | the `inflight` table plus a second `Get` under `mu`; see step 4 |
| 8. replaying without `Content-Type` | SSE clients do not parse the replayed stream | `Entry` stores `ContentType` and `replay` sets it |
| 9. a purge that ignores its model | purging one model empties the cache | `Purge` removes only `it.e.Model == model` unless model is empty |

## 7. Where it's used next

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
