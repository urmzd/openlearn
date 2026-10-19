# Response cache: LRU, TTL, singleflight, and tenant-scoped keys

The same questions reach your gateway again and again. An agent's [evaluation runner](../09-llm-evaluation/01-eval-runner.md) replays its cases at temperature 0 on every run. A docs assistant gets the same "how do I create a key?" from every new user. The [load generator](../../build-an-llm/11-serving/11-load-generator.md) from *Build Your Own LLM System* sends a fixed prompt set. Each of these costs a full prefill and decode on an inference engine, the most expensive thing in the system, for an answer the gateway has already seen. A response cache placed in front of routing turns those repeats into microseconds.

A cache in this position must also avoid the three things that make caches dangerous. It must never show one tenant another tenant's answer, never serve a stale answer after the model behind a name has changed, and never let a burst of identical misses stampede the engines. This section builds a cache that meets all three conditions, applying the patterns from the [chapter opening](README.md): versioned keys instead of invalidation, request coalescing against stampedes, and a TTL to bound staleness.

The section builds on the gateway's [server skeleton](../12-gateway/02-server-skeleton.md), which provides the middleware chain and the per-request `Exchange`, and on [API keys and scopes](../08-authorization-and-access-control/01-api-keys-and-scopes.md), which identifies the tenant. If you want a warm-up in Go first, the [cloud practice drills](../../../../practice/build/cloud/go/README.md) cover the language patterns used here. The complete solution is [`go/gateway/cache/cache.go`](../../../../course/solutions/go/gateway/cache/cache.go), which provides the `LRU` (`New`, `Get`, `Put`, `Len`, `Purge`), `Canonicalize`, `KeyOf`, the `Middleware` stage with singleflight and stream replay, and `PurgeHandler`. As you read, try to predict each step of the solution before you see it.

> **Learning objectives**
> - Decide when an LLM response may be cached at all, and explain why sampled requests without a seed must bypass the cache.
> - Reduce a chat request to a canonical form so that semantically equal requests share a key, and build a key scoped by tenant, configuration revision, and API version.
> - Implement an O(1) LRU with an exact half-open TTL.
> - Implement singleflight so that 100 identical concurrent misses make one upstream call, including the second check under the lock.
> - Store only whole successful answers, and replay a stored SSE stream byte for byte.

## What the cache promises

Before the mechanics, fix the observable contract. The cache stage sets an `X-TL-Cache: hit|miss` header on cacheable requests, as specified in [`openai-subset.v1.yaml`](../../../../course/contracts/openapi/openai-subset.v1.yaml). An admin endpoint, `POST /admin/v1/cache:purge`, specified in [`admin.v1.yaml`](../../../../course/contracts/openapi/admin.v1.yaml), empties the cache for one model or entirely. Two runtime settings in the `[gateway]` section of [`runtime.schema.json`](../../../../course/contracts/config/runtime.schema.json) size it: `cache_entries` and `cache_ttl_s`. Everything below is about honoring those promises without ever returning a wrong answer.

## An LRU with a TTL

The storage layer is a least-recently-used cache with a lifetime on each entry. Three quantities describe it:

| Symbol | Meaning | Type |
|---|---|---|
| $N$ | capacity, `cache_entries` | integer, 4096 by default |
| $\mathit{ttl}$ | lifetime of an entry, `cache_ttl_s` | duration, 300 s by default |
| $t_p$ | when an entry was put | time |

An **LRU** cache keeps its entries in a list ordered by last use, alongside a map from key to list element. `Get` moves a hit to the front of the list; `Put` inserts at the front and, when there are more than $N$ entries, removes from the back. Both operations are O(1). In Go, `container/list` provides the list, and the map holds `*list.Element`.

Capacity bounds memory, but it does not bound staleness: a popular entry could otherwise live forever. The TTL fixes that. An entry is valid while $\mathit{now} < t_p + \mathit{ttl}$, so its lifetime is exactly the half-open interval $[t_p, t_p + \mathit{ttl})$. A `Get` at or after $t_p + \mathit{ttl}$ is a miss and removes the entry. The clock is the gateway's `Clock`, which lets tests move time with a fake rather than sleeping.

## What may be cached

An LRU will store anything. The harder question is what it is *allowed* to store, and the answer starts from a correctness rule: an answer may be replayed only if asking again would give the same answer. The sampling contract, [`spec/sampling.md`](../../../../course/contracts/spec/sampling.md), makes that true in exactly two cases: `temperature` 0 (greedy decoding), and any temperature with a fixed `seed` on the same engine. Anything sampled without a seed would replay one random answer as if it were the only one, so such requests pass through untouched, without even an `X-TL-Cache` header.

Among cacheable requests, two that ask the same thing must produce the same key, even if their bytes differ. So the body is reduced to a **canonical form** in four steps:

1. Parse it as a JSON object.
2. Fill every omitted field that has a contract default (`temperature` 1, `top_p` 1, `top_k` 0, `min_p` 0, `repetition_penalty` 1, `presence_penalty` 0, `frequency_penalty` 0, `n` 1, `stream` false, `logprobs` null, `stop` null), so that `{}` and `{"top_p": 1}` agree.
3. Drop fields that do not change the answer: `user`.
4. Re-encode with sorted keys. Go's `encoding/json` sorts map keys, and numbers pass through `float64`, so `0` and `0.0` agree.

Some fields must stay in. `stream` does: a streamed answer is SSE bytes and a plain one is JSON, so they are different answers. The model, `max_tokens`, and every message stay in as well. Field order, number spelling, explicit defaults, and the `user` field do not change the key; the model, the limits, the messages, and `stream` do.

## Scoping the key

The canonical form tells you whether two requests ask the same question. It does not tell you whether they may share an answer. That depends on who asked and which configuration answered, so the key hashes those in too:

$$\mathit{key} = \mathrm{SHA256}(\mathit{tenant} \,\|\, 0 \,\|\, \mathit{rev} \,\|\, 0 \,\|\, \mathit{path} \,\|\, 0 \,\|\, \mathit{canonical})$$

The **tenant**, taken from the `Principal` that [API keys and scopes](../08-authorization-and-access-control/01-api-keys-and-scopes.md) attaches to the request, makes the cache private per tenant. Two keys belonging to one company share answers; two companies never do, even for byte-equal requests, because the prompt may contain their data and so may the answer. Requests without a principal (no authentication stage) are not cached at all.

The **revision**, `Options.Rev`, is for example the route table's epoch from [routing](../12-gateway/04-routing.md). It changes when a route points a public model name at a new version, so answers from the old model become unreachable at once instead of lingering for the TTL. This is the versioned-key strategy from the chapter opening: no invalidation pass, just a key the old entries can no longer match.

The selected **API version** partitions responses too, because v2 includes usage fields that v1 omits. The middleware appends a version suffix to the revision for v2. An absent version and an explicit v1 share the original partition; v2 gets its own cached answer and its own singleflight group. For this to work, version selection must run before the cache, so that a configured v2 default lands in the same partition as an explicit v2 header.

## Coalescing concurrent misses

So far the cache handles one request at a time. Concurrency changes what a miss means. When a popular question's entry has just expired, 100 identical requests can miss together, and without care they would send 100 identical prefills to the engines at once: a thundering herd, or cache stampede. **Singleflight** prevents this.

On a miss, the first request for a key becomes the **leader**. It registers the key as in flight and goes upstream. A request for the same key that arrives while the leader is running **waits** for it instead of going upstream too, and is then answered from the leader's result as a hit. Two details make this correct.

First, check the cache **again** after taking the in-flight lock. A leader that finished between a follower's `Get` and the follower's lock has already stored its entry and left the in-flight table. Without the second check, the follower would find no in-flight call and become a second leader.

Second, if the leader's answer was not cacheable (an error, a cut stream), the waiters go upstream themselves rather than replaying a failure.

## What is stored and replayed

The last piece of the design is deciding what the leader stores. While the leader's response passes through to its own client, a recorder keeps a copy, up to 1 MiB. The copy is stored only when it is a **whole 200**: status 200, and either valid JSON or an SSE stream that ends with `data: [DONE]\n\n`. A stream cut by an engine failure ends with the [SSE streaming proxy](../12-gateway/03-sse-streaming-proxy.md)'s error event instead, so it is not stored; a 503 is not stored either.

A hit replays the stored status, `Content-Type`, and body, so a streaming client receives the same SSE bytes it would have received from the engine, all at once. Hits set `X-TL-Cache: hit` and misses `X-TL-Cache: miss`. Hits record no usage on the `Exchange`, so the [rate limiter](../08-authorization-and-access-control/02-rate-limiting.md) refunds their reserved tokens and the [usage ledger](../12-gateway/05-usage-ledger-and-admin-api.md) records them as free.

## Working it by hand

Take two requests from tenant `acme`:

```json
A: {"model":"smol","messages":[{"role":"user","content":"hi"}],"temperature":0,"user":"alice"}
B: {"temperature":0.0,"top_p":1,"messages":[{"role":"user","content":"hi"}],"model":"smol"}
```

Canonicalize A first. Fill the defaults it omits (`top_p` 1, `top_k` 0, `min_p` 0, `repetition_penalty` 1, both penalties 0, `n` 1, `stream` false, `logprobs` and `stop` null), drop `user`, and sort the keys. B already has `top_p`, writes `0.0` for 0, and has no `user`. Both become

```json
{"frequency_penalty":0,"logprobs":null,"messages":[{"content":"hi","role":"user"}],"min_p":0,"model":"smol","n":1,"presence_penalty":0,"repetition_penalty":1,"stop":null,"stream":false,"temperature":0,"top_k":0,"top_p":1}
```

Notice that the keys inside each message are sorted too. `temperature` is 0, so both requests are cacheable, and with the same tenant and revision they share one key: A is a miss that reaches the engine, and B is a hit. The same body at `temperature` 0.7 is not cacheable; at 0.7 with `"seed": 42`, it is.

## Designing the API

The package exposes the storage layer, the key functions, and the middleware separately, so each can be tested on its own:

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

The implementation uses only the standard library. The stage caches `POST /v1/chat/completions` and `POST /v1/completions`; embeddings and everything else pass through. The gateway's composition root plugs it into the `Cache` slot as `cache.Middleware(cache.New(cfg.Gateway.CacheEntries, clock), cache.Options{TTL: ..., Rev: routeEpoch})`.

## Building it step by step

Before the code, it is worth seeing how the design follows from first principles, because each decision forces the next.

A cache may only return what the upstream would have returned. So the first question is not "how do I store it" but "when is the answer a function of the request?" By the sampling contract, that is temperature 0 or a fixed seed. Everything else must bypass the cache entirely, including the `X-TL-Cache` header, so clients never think a sampled answer was cached.

Equality must then be semantic, not textual. Requests A and B above differ in bytes but not in meaning. A key over raw bytes misses; a key over the canonical form (defaults filled, irrelevant fields dropped, keys sorted, numbers normalized) hits. Anything that can change the answer (model, messages, `max_tokens`, `stream`) must stay in.

The key must also carry who asked and which configuration answered. Tenant isolation is a privacy requirement, so the tenant goes into the hash, not into a check after lookup. The configuration revision goes in too: changing it makes every old key unreachable in O(1), with no scan and no wait for TTLs.

Bounded memory and bounded staleness are separate limits. Capacity bounds memory by evicting the least recently used entry; the TTL bounds staleness by expiring on time. LRU needs O(1) lookup and O(1) reordering, which forces a hash map pointing into a doubly linked list.

Concurrency changes what a miss means. If 100 identical requests miss together, only one should go upstream. That forces a table of in-flight keys guarded by a lock, waiters that block on the leader, and a re-check of the cache under the lock to close the race where the leader finishes between a follower's `Get` and its `Lock`.

Finally, only whole successes are worth replaying. A recorded answer is stored only when it has status 200, fits the size cap, and is complete (valid JSON, or SSE ending in `[DONE]`). Failures are transient; caching them would make them last for the TTL.

### Step 1: the LRU

The front of the list is the most recently used entry, and the map gives O(1) access to any list element.

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

Each item stores its own key, so eviction from the back can delete the map entry without a reverse lookup. `Purge` walks the list and removes items whose `Model` matches (or every item, for `""`), saving `next` before removing so that the walk survives the removal.

### Step 2: canonicalization and cacheability

With storage in place, the next step turns a request body into the canonical form and decides whether it may be cached at all.

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

The cacheability test runs after the defaults are filled, which is why a body without `temperature` is not cacheable: an omitted temperature means 1, not 0.

### Step 3: the scoped key

The canonical form is then hashed together with the tenant, the revision, and the path.

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

The NUL separators make the concatenation unambiguous. Hashing to a fixed 32-byte array makes the key a cheap, comparable map key regardless of the body's size. In the middleware, a non-v1 API version is appended to `rev` (`rev += "\x00api:" + version`), so v2 answers get their own keys while an absent header and an explicit `1` keep the original one.

### Step 4: singleflight inside the middleware

The middleware first takes its early exits: the request is unauthenticated, not a POST, not chat or completions, has an unreadable body, or is not cacheable. For everything that remains, the stage loops:

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

The ordering in the deferred cleanup is what makes this correct. `c.Put` happens inside the inner function before the defer runs, so by the time the key leaves `inflight`, the entry is already in the LRU. That ordering is what makes the second check sufficient. Likewise, `cl.entry` and `cl.ok` are written before `close(cl.done)`, and the channel close is the happens-before edge that makes reading them in the waiters race-free.

### Step 5: record, and store only whole answers

The leader's response is copied as it streams, then stored only if it passes the completeness test.

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

The recorder wraps the writer rather than buffering it: the leader's client sees tokens as they stream, and the copy is a side effect. `Unwrap` exposes the underlying writer so that `http.ResponseController` can still flush through it. The body is copied out of the buffer so that the stored entry does not alias memory the recorder owns. On the other side, `replay` sets `Content-Type` and `X-TL-Cache: hit` before `WriteHeader`, then writes the body in one call.

The full solution is [`cache.go`](../../../../course/solutions/go/gateway/cache/cache.go).

### Why this design

The costs are small where it matters. `Get`, `Put`, and the in-flight check are O(1) under a mutex; canonicalization is O(body) for the JSON parse and SHA-256. A hit costs microseconds, against a prefill and decode of tens to hundreds of milliseconds. `Purge` is O(N), which is acceptable for a rare admin action. Memory is at most `cache_entries` times `MaxEntryBytes`, which is 4096 times 1 MiB in the worst case, though typical answers are a few KiB.

Several simpler-looking alternatives were rejected deliberately. The design uses one lock for the LRU and one for the in-flight table: a single coarse mutex per structure is simpler than sharding, and contention is low at gateway request rates; sharding is the next step if profiles show contention. It uses lazy expiry rather than a sweeper, so there is no background goroutine and no timer per entry; expired entries cost memory until they are touched or evicted, which `Len` admits by counting them. And it puts the revision in the key rather than invalidating: purging on every route change would require the gateway to know which entries came from which route, whereas folding the revision into the key makes stale answers unreachable instantly and lets LRU reclaim their memory later.

Two library alternatives are worth naming. `golang.org/x/sync/singleflight` would do the coalescing, but the solution keeps to the standard library, and its waiters need the leader's *cacheable* result, not any result. A semantic (embedding-based) cache would raise hit rates but can return a wrong answer for a near paraphrase, which an exact-match cache never does.

## What can go wrong

A correct implementation has to survive a specific set of checks. The table lists each, the kind of test that exercises it, and why it matters to the rest of the gateway.

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example | unit | the worked example: both bodies give the exact canonical JSON and one key; 0.7 without seed is not cacheable, with seed it is; no temperature means 1 | every implementation agrees on equality |
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
| purge handler | unit, conformance | `{"purged": n}` with and without a model | the admin API mounts it |

The bugs that break these checks are predictable, and most of them are a single missing line.

> **Warning:** A key without the tenant, or without the revision, is the most dangerous mistake. Without the tenant, tenant B reads tenant A's answer; without the revision, the old model's answers keep coming after a rollout. `KeyOf` hashes both before the body (step 3).

Getting equality wrong is the next family of bugs. Hashing the raw body, skipping defaults, or keeping `user` makes equal questions miss; dropping `stream` is worse, because a streamed request then receives a JSON body. `Canonicalize` fills `defaults`, deletes `ignored`, keeps `stream`, and re-encodes (step 2). Caching sampled requests is the opposite error: every "write me a poem" gets the same poem. The `temp == 0 || seeded` gate prevents it, and anything that fails it runs `next` with no header.

> **Warning:** Caching failures turns one engine outage into five minutes of replayed errors for everyone, and a cut stream becomes the permanent answer. The solution stores only `status == 200`, under the size cap, and `complete` (step 5).

The LRU itself has two classic off-by-one errors. Testing `now > expires` instead of `now >= expires` lets entries live one tick past their TTL; `Get` treats `!Now().Before(expires)` as expired. A `Get` that does not refresh recency turns the cache into FIFO, so the most used entry is the first evicted; `Get` calls `MoveToFront` on every hit.

Concurrency bugs show up only under load. Without singleflight, or without the second check under the lock, an expiring popular entry sends a burst of identical prefills to the engines; the `inflight` table plus a second `Get` under `mu` prevent both (step 4). Finally, two smaller mistakes break clients and operators: replaying without `Content-Type` means SSE clients do not parse the replayed stream (`Entry` stores `ContentType` and `replay` sets it), and a purge that ignores its model empties the whole cache (`Purge` removes only entries with `it.e.Model == model` unless the model is empty).

## Where this leads

This cache depends on two earlier pieces of the gateway. The [server skeleton](../12-gateway/02-server-skeleton.md) provides the `Cache` slot in the chain, the `Exchange` on which the stage sets `tl.cache.hit`, and `ex.Request` for reading the body; [API keys and scopes](../08-authorization-and-access-control/01-api-keys-and-scopes.md) provides the `Principal.Tenant` that scopes every key.

Two later pieces depend on it. The [usage ledger and admin API](../12-gateway/05-usage-ledger-and-admin-api.md) assembles the full gateway: its `gateway.Deps` calls `Middleware` with `Rev` set to the route table's ETag, and `gateway.Admin` mounts `PurgeHandler`. The [load generator](../../build-an-llm/11-serving/11-load-generator.md) in *Build Your Own LLM System* reports the `X-TL-Cache: hit` ratio in its load report.

An exact-match response cache is only one kind of reuse. The engine's [radix prefix cache](../../build-an-llm/09-inference/04-radix-prefix-cache.md), like provider prompt caching from Anthropic and OpenAI, reuses the prompt's KV state rather than the answer, so it works for sampled requests too. Semantic caches reuse answers across paraphrases, at the risk of false hits.

### Further reading

- [GPTCache](https://github.com/zilliztech/GPTCache) (free): a semantic cache that hits on paraphrases by embedding similarity, with the false-hit risk that brings; Redis LangCache is a hosted equivalent.
- [vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html) (free): reuse of the prompt's KV cache instead of the answer.
- The `golang.org/x/sync/singleflight` source (free), and [groupcache](https://github.com/golang/groupcache) (free) for request coalescing as a library and peer-to-peer cache filling.
- Megiddo and Modha, *ARC: A Self-Tuning, Low Overhead Replacement Cache* (FAST 2003), an eviction policy that adapts between recency and frequency.
- saige's `cache` package, which separates the response cache, prompt cache, step record, and journal, keeps identity private by default, and gives each cache its own contract.

## Key terms

- **Cacheable request**: one whose answer is a function of the request: `temperature` 0, or a fixed `seed`.
- **Canonical form**: the request body with defaults filled, ignored fields dropped, numbers normalized, and keys sorted, so equal questions have equal bytes.
- **Scoped key**: the SHA-256 of tenant, revision, path, and canonical body, separated by NUL bytes.
- **Revision**: a configuration epoch folded into every key so a route change makes old entries unreachable.
- **LRU**: an eviction policy that removes the least recently used entry when capacity is exceeded.
- **Half-open TTL**: an entry valid on $[t_p, t_p + \mathit{ttl})$, dead at exactly $t_p + \mathit{ttl}$.
- **Singleflight**: coalescing concurrent misses for one key so a single leader goes upstream and the others wait for its result.
- **Whole 200**: a status-200 response that is valid JSON or an SSE stream ending in `data: [DONE]\n\n`, the only kind that is stored.

## Summary

- Only deterministic requests are cached: `temperature` 0, or a fixed `seed`. Anything sampled without a seed would replay one random answer as if it were the only one.
- The key hashes the tenant, the configuration revision (plus the API version for v2), and the canonical request; field order, number spelling, explicit defaults, and `user` do not change it, while the model, limits, messages, and `stream` do.
- Storage is an LRU of at most `cache_entries` with a TTL per entry; an entry lives exactly $[t_{\text{put}}, t_{\text{put}} + \mathit{ttl})$, and a hit makes it the most recently used.
- Singleflight makes 100 identical concurrent requests produce one upstream call, with a second cache check under the lock to close the race with a finishing leader.
- Only whole 200 answers are stored, and a stored stream is replayed byte for byte with its `Content-Type`.

## Review questions

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

6. A request body omits `temperature` entirely. Is it cacheable, and why?

   <details><summary>Answer</summary>

   No, unless it carries a `seed`. The contract default for `temperature` is 1, and `Canonicalize` fills defaults before testing cacheability, so an omitted temperature is a sampled request.

   </details>

7. Why does the leader's recorder pass every byte through to its own client instead of buffering the response and writing it once?

   <details><summary>Answer</summary>

   Buffering would destroy streaming for the leader's client, who would wait for the whole answer. The recorder keeps its copy as a side effect, dropping it if the response exceeds `MaxEntryBytes`, while the client sees tokens live.

   </details>
