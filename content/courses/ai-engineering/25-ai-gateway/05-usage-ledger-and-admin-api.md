# The usage ledger and admin API

By this point the gateway authenticates every request, limits it, caches it, and routes it, and it forgets each one the moment the response ends. The rate limiter's token buckets live in memory and reset on every restart, so they cannot say how many tokens `acme` used yesterday. Nothing can: not the bill, not the per-tenant panel that the [noisy-neighbor drill](../../systems/12-incident-response-and-chaos/10-noisy-neighbor.md) needs, not the agent's `query_usage` tool in [The tool gate, the SQL safety gate, and the prompt-injection suite](../28-agents/04-tool-gate-and-injection.md), and not the end-to-end check that what the gateway charged equals what the engine produced.

This section adds the one durable record of who used what: a row per request in a SQLite file, written by the last stage of the chain and read through `GET /admin/v1/usage`. It also gives the admin endpoints that the keys, routing, cache, and policy sections build one front door, and it assembles every stage into the chain with code a test can call.

You need [the server skeleton](02-server-skeleton.md) (the chain, the `Exchange`, `WriteError`), [API keys and scopes](../../software-engineering/10-authorization/01-api-keys-and-scopes.md) (the `Principal`), [Rate limiting](../../software-engineering/10-authorization/02-rate-limiting.md), [Routing, cascades, and failover](04-routing.md) (the route stage and its admin handlers), and [the response cache](../../systems/06-distributed-data-and-caching/01-response-cache.md) (the stage and its purge handler). For the SQL, read [SQL and SQLite: schema, transactions, WAL mode, indexes, aggregates](../../programming-and-algorithms/01-language-primers/11-sql-and-sqlite.md) in *Programming and Algorithms*, which works through the same usage table.

The solution is [`ledger.go`](../../../../course/solutions/go/gateway/ledger/ledger.go) (the SQLite store: `DSN`, `Open`, `Record`, `Query`), [`meter.go`](../../../../course/solutions/go/gateway/ledger/meter.go) (`WithIncludeUsage`, `meterWriter`, the `Meter` stage), [`admin.go`](../../../../course/solutions/go/gateway/ledger/admin.go) (`Admin`, the whole `/admin/v1` surface, and `UsageHandler`), and [`gateway.go`](../../../../course/solutions/go/gateway/gateway.go) (`Deps` and `Admin`: the stages assembled into the chain and the admin API). The entry points are the gateway's [`compose.go`](../../../../course/solutions/entry/go/cmd/gateway/compose.go) and the `usage` verb of the platform CLI, [`ctl/usage.go`](../../../../course/solutions/entry/go/cmd/ctl/usage.go). The ledger uses one third-party package, the pure-Go SQLite driver `modernc.org/sqlite`. The contracts are the schema [`course/contracts/formats/usage.v1.sql`](../../../../course/contracts/formats/usage.v1.sql), applied verbatim, and the server side of [`course/contracts/openapi/admin.v1.yaml`](../../../../course/contracts/openapi/admin.v1.yaml).

> **Learning objectives**
> - Meter one row per request from the finished response, including stream usage the client never asked for.
> - Configure SQLite so a `SIGKILL` leaves every acknowledged row complete, concurrent writers wait instead of failing, and retries are no-ops.
> - Write usage queries with half-open windows, bound parameters, and a whitelisted `GROUP BY` column.
> - Build an admin API that fails closed regardless of where it is mounted.
> - Assemble the gateway's stages so a route-table change invalidates the cache.

## Metering: one row per request, from the response

The chain from the server skeleton is `requestid -> otel -> recover -> authn -> policy -> ratelimit -> cache -> route -> proxy -> meter`. The **meter** wraps the proxy directly, so it runs after every other stage has decided and it sees the response go by. It meters the three inference routes (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`) and lets every other path through untouched. It writes one row per request, refused requests included, when the response ends. Each column of the row has one source:

| Column | Source |
|---|---|
| `request_id`, `ts_ms` | the `Exchange`: `RequestID`, `Start` |
| `tenant`, `key_id` | the `Principal` that the authn stage put in the context |
| `model`, `route`, `stream` | the parsed request (`Exchange.Request`) and the path |
| `served_model` | the `model` field of the response: an alias or a canary answers with another id |
| `status`, `error_code` | the status sent to the client; the error body's `code` (its `type` when the code is null); an SSE error event's code for a stream that failed after its first byte |
| `prompt_tokens`, `completion_tokens`, `cached_tokens` | the response's `usage` object (`prompt_tokens_details.cached_tokens` when present) |
| `ttft_ms`, `e2e_ms` | first content byte minus start, NULL when none was sent or the request was refused; end minus start, on the gateway's clock |
| `worker_id`, `cache_hit`, `trace_id` | the router's choice and the cache flag on the `Exchange`; the trace id of the request's `traceparent` |

The token columns hide a trap: where does usage come from? A non-streamed answer carries `usage` in its body. A stream carries it in one extra chunk near the end, `{"choices": [], "usage": {...}}`, and the engine sends that chunk **only when the request says** `"stream_options": {"include_usage": true}` (`openai-subset.v1.yaml`). Most clients do not ask, so a meter that only reads would bill every stream zero tokens. The meter therefore adds `include_usage` to every stream request it forwards. That changes the answer the client receives, so the meter also **removes** that usage chunk on the way back when the client did not ask for it. A client that did ask gets the engine's bytes unchanged.

The goal is an invariant: for every tenant, the ledger and the engine agree. Name the pieces:

| Symbol | Meaning | Type |
|---|---|---|
| $t$ | a tenant | string |
| $L_t$ | the ledger rows of tenant $t$ | set of rows |
| $E_t$ | the requests of $t$ that the engine answered with a `usage` object | set of requests |
| $p_r, c_r$ | prompt and completion tokens of a row or request $r$ | integers $\ge 0$ |

$$\sum_{r \in L_t} p_r = \sum_{e \in E_t} p_e, \qquad \sum_{r \in L_t} c_r = \sum_{e \in E_t} c_e$$

Refused requests are rows with zero tokens, so they count in `requests` and `errors` and add nothing to the sums. A correct solution keeps the invariant over any mix of streamed, non-streamed, and refused requests, and against a real engine.

## The store: SQLite, WAL, one statement per row

The rows have to land somewhere that survives the process. The ledger is a single SQLite file at `[gateway].usage_db`. `Open` applies `usage.v1.sql` exactly as written (every statement is `IF NOT EXISTS` or `OR IGNORE`, so reopening is a no-op), and refuses a file whose `schema_version` is newer than 1: an old gateway must not write into a table that a later migration reshaped. The driver is `modernc.org/sqlite`, SQLite compiled to pure Go, so the gateway still builds with `CGO_ENABLED=0` into a static binary ([Dockerfiles for the gateway and the engine](../../systems/10-containers-kubernetes/02-dockerfiles-for-gateway-and-engine.md)).

Four settings decide whether rows survive; the SQL primer worked through each one:

| Setting | Scope | Why |
|---|---|---|
| `journal_mode = WAL` | stored in the file | a commit appends frames and a commit record to `usage.db-wal`; readers never block the writer; recovery ignores a tail without its commit record |
| `synchronous = NORMAL` | per connection | in WAL mode, a commit survives a **process** kill (the bytes are in the OS page cache); only a power loss can take the last commits. `FULL` would `fsync` on every row |
| `busy_timeout = 5000` | per connection | SQLite has one writer at a time; without a timeout a second writer fails at once with `SQLITE_BUSY` and its row is lost |
| `request_id` primary key, `ON CONFLICT DO NOTHING` | schema and statement | a retried write (a timeout after the commit) leaves the first row alone |

There is a subtlety with the per-connection settings. `database/sql` keeps a **pool** of connections, so a setting made with one `Exec("PRAGMA ...")` reaches one connection only. The settings go into the data source name instead, which the driver applies to every connection it opens: `usage.db?_pragma=busy_timeout(5000)&_pragma=journal_mode(WAL)&_pragma=synchronous(NORMAL)`.

A row is one `INSERT` statement, so it commits whole or not at all. `Record` returns only after the commit: an acknowledgement means the row is in the WAL. Cached tokens larger than the prompt (an engine bug) would violate the table's `CHECK` and lose the whole row, so `Record` clamps them; negative counts and an empty request id are refused with `ErrInvalidRecord`.

## Queries: half-open windows, bound values, whitelisted columns

`Query(UsageFilter)` sums rows into `requests`, `errors` (status at least 400, or an error code), and the three token counts. Four rules govern it.

Windows are half-open, `since <= ts_ms < until`. Then `[10.1:00, 11:00)` and `[11:00, 12:00)` add up to `[10:00, 12:00)`, with no request at exactly `11:00:00.000` counted twice.

Values are bound parameters (`tenant = ?`). The tenant arrives from the admin API's query string; spliced into the SQL text, `x' OR '1'='1` would read every tenant's usage.

A column name cannot be a parameter, so `group_by` is looked up in a fixed map (`model`, `tenant`, `key_id`, `api_version`), and anything else is `ErrBadGroupBy`.

No grouping gives exactly one row, zeros when nothing matches, so a caller never has to special-case an empty answer. Grouped rows are sorted by the group key.

## What a SIGKILL can do

The durability settings are there for one event in particular. The kubelet `SIGKILL`s a pod that exceeds its memory limit or fails its liveness probe, and nothing in the process runs afterwards. What survives is whatever reached the OS before the signal. With WAL and one statement per row, every committed row is in `usage.db-wal` (the OS writes it to disk later even though the process is gone), and a commit that was half-written has no commit record, so the next `Open` ignores it.

Two designs break this. A ledger that collects rows in memory and writes them in batches loses every acknowledged row in the batch. A ledger that writes a row in two statements without a transaction leaves a torn row if the kill lands between them. The test is concrete: killing a writer process after 400 acknowledgements must leave every acknowledged row present and complete, column by column.

## The admin API: one front door, fail closed

`admin.v1.yaml` has eight operations built in five different sections of this book. This section owns the server side: one handler, `Admin(AdminDeps)`, for every `/admin/v1` path. It serves usage itself and hands every other path to the handler its owner builds: keys from [API keys and scopes](../../software-engineering/10-authorization/01-api-keys-and-scopes.md); workers, routes, and drains from [routing](04-routing.md); the cache purge from [the response cache](../../systems/06-distributed-data-and-caching/01-response-cache.md); and the policy from [usage policy enforcement](06-usage-policy-enforcement.md). A path whose handler is nil answers 404 `not_found`, so a partly built gateway still has a well-formed admin surface.

`Admin` checks the principal itself: no principal is 401 `invalid_api_key`, and a key without `admin` is 403 `insufficient_scope`. The authn stage already enforces that, but an admin handler that trusts its mount point serves every tenant's billing data the first time someone mounts it outside the chain.

## Working it by hand

Four requests reach the meter (times are 2026-10-01, UTC):

| `request_id` | start | tenant | key | model | route | status | code | prompt | completion |
|---|---|---|---|---|---|---|---|---|---|
| `req-1` | 12:00 | acme | `acmekeyaaaaa` | smol | chat, stream | 200 | | 12 | 30 |
| `req-2` | 12:01 | acme | `acmekeybbbbb` | smol | chat | 200 | | 20 | 10 |
| `req-3` | 12:02 | acme | `acmekeyaaaaa` | tiny | completions | 429 | `rate_limit_exceeded` | 0 | 0 |
| `req-4` | 12:03 | globex | `globexkeyccc` | smol | chat | 200 | | 7 | 5 |

Start with the stream, `req-1`. The client sends `{"model":"smol","stream":true,"messages":[{"role":"user","content":"Once"}]}`. The meter forwards `{"messages":[...],"model":"smol","stream":true,"stream_options":{"include_usage":true}}`. The engine answers with three content chunks, then `data: {...,"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":30,"total_tokens":42}}`, then `data: [DONE]`. The meter reads 12 and 30 from the usage chunk and drops that chunk, so the client receives the three content chunks and `[DONE]`: exactly what it would have received from an engine it asked directly.

With the four rows stored, run four queries:

1. `acme`, no grouping: the rows are `req-1`, `req-2`, `req-3`. Requests $3$; errors $1$ (`req-3`, status 429); prompt $12 + 20 + 0 = 32$; completion $30 + 10 + 0 = 40$.
2. `acme`, grouped by model, sorted by the key: `smol` is `req-1` and `req-2`, so requests 2, errors 0, prompt 32, completion 40; `tiny` is `req-3`, so requests 1, errors 1, tokens 0.
3. The window `since=12:01, until=12:03`, every tenant: `12:01 <= ts < 12:03` holds for `req-2` and `req-3` and not for `req-4` (12:03 is not before 12:03). Requests 2, errors 1, prompt 20, completion 10.
4. `initech`: no rows, so one row of zeros.

Through the admin API, query 3 grouped by model is

```
GET /admin/v1/usage?since=2026-10-01T12:01:00Z&until=2026-10-01T12:03:00Z&group_by=model
Authorization: Bearer tl_<admin key>

200 {"since":"2026-10-01T12:01:00Z","until":"2026-10-01T12:03:00Z","data":[
  {"model":"smol","requests":1,"errors":0,"prompt_tokens":20,"completion_tokens":10,"cached_tokens":0},
  {"model":"tiny","requests":1,"errors":1,"prompt_tokens":0,"completion_tokens":0,"cached_tokens":0}]}
```

A correct solution must reproduce every number here: the four sums, the hidden usage chunk, and the admin response.

## The interface

The ledger package and the assembly package expose this surface:

```go
package ledger // import "tinyllm/gateway/ledger"

const Schema = `...`        // contracts/formats/usage.v1.sql, byte for byte
const SchemaVersion = 1

type UsageRecord struct {
	RequestID string; Start time.Time; Tenant, KeyID, Model, ServedModel, Route, APIVersion string
	Status int; ErrorCode string; Stream bool
	PromptTokens, CompletionTokens, CachedTokens int
	TTFT *time.Duration; E2E time.Duration; CacheHit bool; WorkerID, TraceID string
}
type UsageFilter struct { Tenant, KeyID string; Since, Until time.Time; GroupBy string }
type UsageRow struct { Tenant, KeyID, Model, APIVersion string; Requests, Errors, PromptTokens, CompletionTokens, CachedTokens int64 }
type Ledger interface {
	Record(ctx context.Context, r UsageRecord) error
	Query(ctx context.Context, f UsageFilter) ([]UsageRow, error)
}
var ErrInvalidRecord, ErrBadGroupBy, ErrNewerSchema error

func DSN(path string) string
func Open(path string) (*DB, error)          // *DB implements Ledger; Close() error
func WithIncludeUsage(body []byte) ([]byte, bool, error)
func Meter(l Ledger, o MeterOptions) server.Middleware   // MeterOptions{Clock, WriteTimeout, Logger}
func Admin(d AdminDeps) http.Handler         // AdminDeps{Usage Ledger; Keys, Workers, Routes, Drain, Policy, Cache http.Handler}
func UsageHandler(l Ledger) http.Handler     // GET /admin/v1/usage

package gateway // import "tinyllm/gateway"

type Stages struct {
	Keys *auth.Store; Logger *slog.Logger                  // API keys and scopes
	Policy server.Middleware; PolicyAdmin http.Handler     // usage policy
	Limiter *limit.Limiter; Counter limit.Counter          // rate limiting
	Cache *cache.LRU; CacheOptions cache.Options           // response cache
	Registry *route.Registry; Router *route.Router         // routing
	Proxy http.Handler                                     // route.Proxy, or the streaming proxy
	Ledger ledger.Ledger; Meter ledger.MeterOptions        // this section
}
func Deps(s Stages) server.Deps          // a nil stage is left out; the cache's Rev defaults to route.ETag(epoch)
func Admin(s Stages) ledger.AdminDeps    // KeysHandler, WorkersHandler, RoutesHandler, DrainHandler, PurgeHandler
```

`gateway.go` is the part of the composition root a test can call. `Deps` calls each stage's middleware constructor (`auth.Middleware`, `limit.Middleware`, `cache.Middleware`, `route.Middleware`, `ledger.Meter`) and leaves a nil stage out; the skeleton's `Handler` puts them in contract order. One wiring decision is made here and nowhere else: with a `Router` and no `CacheOptions.Rev`, the cache revision is the route table's `ETag`, so a `PUT /admin/v1/routes` makes every answer cached under the old table unreachable. `Admin` builds each admin path's handler from the package that owns it. The unexported helpers (`meterWriter`, `inspect`, `errorCode`, `traceID`, and so on) are the solution's decomposition, not part of the interface.

The gateway's composition root ([`compose.go`](../../../../course/solutions/entry/go/cmd/gateway/compose.go)) opens the ledger at `[gateway].usage_db`, builds the stages into a `gateway.Stages`, and serves the admin API through a second, short chain that has only authn in front of it, so admin calls never reach the router. In its simplest form:

```go
st := gateway.Stages{Keys: keys, Limiter: limiter, Counter: counter, Cache: lru, CacheOptions: cache.Options{TTL: ttl},
	Registry: reg, Router: router, Proxy: route.Proxy(router, route.Options{}), Ledger: db}
inference := server.New(cfg, gateway.Deps(st)).Handler()
admin := server.New(cfg, server.Deps{Keys: auth.Middleware(keys, nil), Proxy: ledger.Admin(gateway.Admin(st))}).Handler()
mux := http.NewServeMux()
mux.Handle("/admin/v1/", admin)
mux.Handle("/", inference)
```

The full `compose.go` adds API-version selection and moves the meter out to wrap the cache, so cache hits are metered too. The platform CLI's `usage` verb, `ctl usage --gateway <base URL> [--tenant T] [--key-id K] [--since T] [--until T] [--group-by G]`, calls `GET /admin/v1/usage` with the admin key from `TL_API_KEY`, prints each row as a JSON line, and ends with a line of totals.

## Building it step by step

Eight observations lead to the code:

1. **One row per request, written once the outcome is known.** Tokens, status, and latency are known only when the response ends, so the writer must sit where the whole response passes: the innermost stage, wrapping the proxy, writing after `next.ServeHTTP` returns.
2. **The meter must see usage it would not otherwise get.** An engine sends stream usage only with `include_usage`. So the meter rewrites the request, and to keep the client's answer unchanged it must remove exactly the chunk it caused (a `choices: []` chunk with `usage`), and only when it caused it.
3. **Reading the stream must not delay it.** The meter cuts SSE events the same way the streaming proxy's splitter does, forwards every complete event as soon as it has it, and passes `Flush` through. A half-received event can wait (it is not yet a token); anything else cannot.
4. **The row must outlive the request.** A client that hangs up has still used tokens, and a write on its context would be cancelled with it. So the write uses `context.WithoutCancel(r.Context())` plus its own timeout.
5. **Durability is a property of one statement.** If a row is one `INSERT` in WAL mode, a kill leaves it committed or absent; an acknowledgement after the commit then means "on disk or in the OS's hands". `ON CONFLICT DO NOTHING` on the primary key makes the write safe to repeat.
6. **Pool-wide settings must be in the DSN.** `database/sql` opens connections on demand, so a `PRAGMA` executed once reaches one of them; the driver applies `_pragma` parameters to every connection.
7. **Untrusted input goes in values; structure comes from code.** Filter values are bound parameters; the one piece of SQL structure the caller chooses (the `GROUP BY` column) comes from a whitelist map.
8. **Authorization belongs to the handler, not its mount point.** `Admin` checks the principal and the `admin` scope itself, so mis-mounting it fails closed.

### The data source

Step 6, in `ledger.go`:

```go
func DSN(path string) string {
	q := url.Values{}
	q.Add("_pragma", "busy_timeout(5000)")  // wait for the write lock instead of SQLITE_BUSY
	q.Add("_pragma", "journal_mode(WAL)")   // a commit is a WAL append; readers never block the writer
	q.Add("_pragma", "synchronous(NORMAL)") // survives a process kill; fsync only at checkpoints
	return path + "?" + q.Encode()
}
```

`Open` executes `Schema` (the contract file verbatim, safe to re-run), then reads `MAX(version)` from `schema_version` and refuses anything newer than `SchemaVersion` with `ErrNewerSchema`.

### Recording a row

Step 5: validation first, then one statement.

```go
func (d *DB) Record(ctx context.Context, r UsageRecord) error {
	if r.RequestID == "" || r.PromptTokens < 0 || r.CompletionTokens < 0 || r.CachedTokens < 0 {
		return fmt.Errorf("%w: ...", ErrInvalidRecord /* ... */) // the CHECKs would refuse it anyway; say why
	}
	if r.CachedTokens > r.PromptTokens {
		r.CachedTokens = r.PromptTokens // clamp: an engine bug must not lose the whole row to the CHECK
	}
	if r.APIVersion == "" {
		r.APIVersion = "1"
	}
	var ttft any // nil -> NULL: "no content byte was sent", distinct from 0 ms
	if r.TTFT != nil {
		ttft = ms(*r.TTFT)
	}
	// ... b2i maps bools to 0/1 for the INTEGER CHECK columns
	_, err := d.db.ExecContext(ctx, `INSERT INTO usage (
		request_id, ts_ms, tenant, key_id, model, served_model, route, api_version,
		status, error_code, stream, prompt_tokens, completion_tokens, cached_tokens,
		ttft_ms, e2e_ms, cache_hit, worker_id, trace_id)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT (request_id) DO NOTHING`, // the first row stands; a retry is a no-op
		r.RequestID, r.Start.UnixMilli(), /* ... */ nullString(r.ErrorCode), /* ... */ ttft, ms(r.E2E) /* ... */)
	return err // returns after the commit: an ack means the row is in the WAL
}
```

### Querying

Step 7. The `WHERE` clause is built from fixed fragments with `?` placeholders; only `col`, taken from `groupColumns`, is ever spliced into the text:

```go
col := ""
if f.GroupBy != "" && f.GroupBy != "none" {
	c, ok := groupColumns[f.GroupBy] // model, tenant, key_id, api_version: nothing else
	if !ok {
		return nil, ErrBadGroupBy
	}
	col = c
}
// ... tenant = ?, key_id = ? when set
if !f.Since.IsZero() {
	where = append(where, "ts_ms >= ?") // inclusive
	args = append(args, f.Since.UnixMilli())
}
if !f.Until.IsZero() {
	where = append(where, "ts_ms < ?") // exclusive: adjacent windows never share a row
	args = append(args, f.Until.UnixMilli())
}
sel := `COUNT(*),
	COALESCE(SUM(CASE WHEN status >= 400 OR error_code IS NOT NULL THEN 1 ELSE 0 END), 0),
	COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0), COALESCE(SUM(cached_tokens), 0)`
// ... "SELECT [col,] sel FROM usage [WHERE ...] [GROUP BY col ORDER BY col]"
```

Without `GROUP BY`, an aggregate query returns exactly one row even over no rows; `COUNT(*)` is then 0 and `COALESCE` turns the `NULL` sums into 0, which is how `initech` in the hand example gets one row of zeros. `errors` counts `error_code IS NOT NULL` as well as `status >= 400` because a stream that failed after its first byte was sent with status 200.

### Asking for stream usage

Step 2, in `meter.go`. `WithIncludeUsage` edits the request as a map of raw JSON values, so every other field passes through untouched, and reports whether it changed anything:

```go
func WithIncludeUsage(body []byte) ([]byte, bool, error) {
	var m map[string]json.RawMessage
	// ... unmarshal; read an existing stream_options object, if any
	if string(opts["include_usage"]) == "true" {
		return body, false, nil // the client asked itself: change nothing, strip nothing
	}
	opts["include_usage"] = json.RawMessage("true") // keep the client's other stream_options
	// ... re-marshal opts into m["stream_options"], then m
	return out, true, nil
}
```

The returned `changed` flag becomes `mw.strip`: the meter hides the usage chunk only when it was the one who asked.

### Reading the response on the way through

`meterWriter.Write` buffers a non-SSE body (up to `maxSniff`) for its usage and forwards it unchanged. For SSE it cuts complete events and forwards each one, unless `inspect` says to hide it:

```go
m.pend = append(m.pend, b...)
for {
	i := bytes.Index(m.pend, []byte("\n\n"))
	if i < 0 {
		return len(b), nil // a partial event waits; it is not yet a token
	}
	ev := m.pend[:i+2]
	if m.inspect(ev) { // false only for the usage chunk the meter asked for
		if _, err := m.forward(ev); err != nil {
			return 0, err
		}
	}
	m.pend = m.pend[i+2:]
}
```

`inspect` records what the row needs from each event:

```go
func (m *meterWriter) inspect(ev []byte) bool {
	// ... join the data: lines; [DONE], comments, and unparsable events pass through
	if m.served == "" {
		m.served = e.Model // the id the backend answered with (canary, alias)
	}
	if e.Error != nil {
		m.errCode = errorCode(e.Error.Type, e.Error.Code, m.status) // a mid-stream failure
	}
	if e.Usage != nil {
		m.usage = e.Usage
		if m.strip && e.Choices != nil && strings.TrimSpace(string(*e.Choices)) == "[]" {
			return false // exactly the chunk include_usage added
		}
	}
	return true
}
```

`forward` marks the first content byte on the `Exchange` (only for a status below 400, so refusals have no TTFT), and `Flush` passes straight to the underlying writer so the streaming proxy's per-read flush still reaches the client (step 3). `finish`, called after the handler returns, sends any trailing partial event and parses a non-SSE body's `usage`, `model`, and `error`.

### Writing the row

After `next.ServeHTTP(mw, r)` and `mw.finish()`, `Meter` builds the record from the `Exchange` snapshot, the `Principal`, and what `meterWriter` read; tokens and TTFT are taken only for a status below 400. Then step 4:

```go
ctx, cancel := context.WithTimeout(context.WithoutCancel(r.Context()), timeout) // keeps values, drops cancellation
defer cancel()
if err := l.Record(ctx, rec); err != nil {
	logger.Printf("ledger: request %s not recorded: %v", rec.RequestID, err) // logged; the response is already sent
}
```

### The admin front door

Step 8, in `admin.go`. `Admin` checks the principal before looking at the path, then dispatches; a nil handler is a 404 `not_found`:

```go
p, ok := auth.PrincipalFrom(r.Context())
if !ok {
	server.WriteError(w, http.StatusUnauthorized, "invalid_request_error", "invalid_api_key", "", /* ... */)
	return
}
if !p.HasScope(auth.ScopeAdmin) {
	server.WriteError(w, http.StatusForbidden, "permission_error", "insufficient_scope", "", /* ... */)
	return
}
// ... switch on the path: usage, keys, workers, routes, models/{m}:drain, policy, cache:purge
if h == nil {
	notFound(w, path+" is not served by this gateway")
	return
}
h.ServeHTTP(w, r)
```

`UsageHandler` validates every parameter before querying (`group_by` against the same whitelist, `since` and `until` as RFC 3339, `since` before `until`) and answers 400 with `param` naming the bad one, so the CLI can name the bad flag.

### Assembly

In `gateway.go`, `Deps` turns each built stage into its middleware and leaves nil ones out. The one wiring decision looks like this:

```go
if s.Cache != nil {
	o := s.CacheOptions
	if o.Rev == nil && s.Router != nil {
		rt := s.Router
		o.Rev = func() string {
			_, epoch := rt.Routes()
			return route.ETag(epoch) // a new route table is a new cache namespace
		}
	}
	d.Cache = cache.Middleware(s.Cache, o)
}
```

The full solution is in [`ledger.go`](../../../../course/solutions/go/gateway/ledger/ledger.go), [`meter.go`](../../../../course/solutions/go/gateway/ledger/meter.go), [`admin.go`](../../../../course/solutions/go/gateway/ledger/admin.go), and [`gateway.go`](../../../../course/solutions/go/gateway/gateway.go).

### Why this design

The cost is one `INSERT` per request. With WAL and `synchronous=NORMAL` it is an append to the WAL with no `fsync`, tens of microseconds, off the response path (the response is already sent). Queries use the `(tenant, ts_ms)`, `(key_id, ts_ms)`, and `(model, ts_ms)` indexes, so a tenant's window is a range scan, not a table scan.

SQLite rather than a server database: one gateway process writes one file, there is nothing to operate, and the pure-Go driver keeps the image static. The price is one writer at a time, which `busy_timeout` turns into a short wait. Many gateway replicas would want a shared store, or per-replica files merged downstream.

The rejected alternatives each lose something. Batching rows in memory would cut write cost, but every acknowledged row in an unflushed batch is lost on a `SIGKILL`, which on Kubernetes is routine (OOM, failed liveness probe). `INSERT OR REPLACE` would make retries "succeed" by overwriting the first row; `DO NOTHING` keeps the first, which is the one that was acknowledged. Metering from the `Exchange` only (what the streaming proxy recorded) would miss the served model, the error code of a mid-stream failure, and the usage the meter itself requested; reading the response directly makes the meter self-contained. It still falls back to the `Exchange`'s usage when the response had none of its own.

> **Note:** One tradeoff is accepted. `synchronous=NORMAL` can lose the last commits on power loss (not on a process kill). Usage is billing data, but a lost second of rows on a power cut is cheaper than an `fsync` per request.

## What can go wrong

The tests check the following cases:

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| the hand example | unit | acme 3/1/32/40, by model, the window, the empty tenant | every definition is pinned down |
| the schema | conformance | `Open` creates what `usage.v1.sql` creates (compared through `sqlite_master`), version 1, WAL | the agent tool and the noisy-neighbor drill read these tables |
| reopening, and a newer schema | boundary | reopening keeps rows; `schema_version` 2 is refused with `ErrNewerSchema` | rollouts restart the gateway; the API v2 migration changes the schema |
| the same request recorded twice | boundary | the second record of `req-1` changes nothing | retried writes never double bill |
| bad and odd counts | boundary | cached 50 > prompt 20 is stored as 20; negative counts and an empty id are refused | an engine bug never loses a row |
| every column | unit | every field in its column and unit; NULL code and TTFT; `api_version` defaults to `'1'` | columns, not Go structs, are the contract |
| adjacent windows | boundary | `[10:00,11:00)` + `[11:00,12:00)` = `[10:00,12:00)` | hourly panels add up |
| quotes in filter values | unit | quotes in tenant or key id change nothing | the admin API passes them from the URL |
| `group_by` values | boundary | `status`, `Model`, an injection are `ErrBadGroupBy`; key rows sorted | the admin API's 400 |
| concurrent writers | fault | 16 writers x 64 rows, no `SQLITE_BUSY`, 1024 rows | every request goroutine records at once |
| `SIGKILL` of a writer process | fault | after the kill: integrity ok, WAL, every acknowledged row present and complete | OOM kills on kind; the kill-decode drill |
| ledger against engine | property | 1000 mixed requests: per-tenant ledger sums equal the engine's usage; 59 refusals counted | the end-to-end reconciliation |
| stream usage | unit | `include_usage` added and its chunk hidden; a client that asked gets the bytes unchanged | streams are billed; clients see what they asked for |
| streaming through the meter | unit | the first event reaches the client while the engine holds the rest | TTFT, the gateway's SLO |
| refused requests | unit | 429 and 400 rows with code, zero tokens, NULL TTFT; a mid-stream error event is an error | the noisy-neighbor drill sees who hits limits |
| who, what, and where | unit | tenant, key, model, served model, route, worker, cache flag, trace id, gateway-clock latency | per-tenant and per-worker panels |
| a client that hangs up mid-stream | fault | it still gets a row | abandoned streams still cost tokens |
| unmetered paths | boundary | `GET /v1/models` untouched, no row | only inference is billed |
| `GET /admin/v1/usage` | conformance | `{since, until, data}`, null window, one ungrouped row, rows grouped by model | `ctl usage`, the agent tool |
| bad admin parameters | boundary | bad `since`, `until`, `group_by`, an empty window: 400 naming the param; POST is 405 | the CLI can name the bad flag |
| the admin scope | unit | no principal 401, an `infer` key 403 | billing data stays private |
| every admin path | unit | each path reaches its owner's handler; nil handlers and unknown paths are 404 `not_found` | keys, routing, cache, and policy mount here |
| the assembled gateway | unit | real keys, limits, cache, router, and ledger through `gateway.Deps` and `gateway.Admin`: a repeat is a hit, a route PUT makes it a miss, the 4th request of an RPM-3 key is a 429 before the cache, purge, workers, routes, drain, keys, and usage reach their handlers | the composition root is these two calls |

The mistakes that most often break a ledger, with their symptoms:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| reading usage only from the response | every stream is billed 0 tokens: engines send stream usage only when asked | `WithIncludeUsage` on every metered stream (step 2) |
| writing the row on `r.Context()` | a client that hangs up mid-stream leaves no row: the write is cancelled with the request | `context.WithoutCancel` plus a write timeout (step 4) |
| no `busy_timeout`, or a `PRAGMA` run once on a pooled `*sql.DB` | `database is locked` under load, rows lost | the pragmas are in `DSN`, applied to every connection (step 6) |
| `INSERT OR REPLACE` (or an upsert) on `request_id` | a retried write rewrites the first row with new numbers | `ON CONFLICT (request_id) DO NOTHING` |
| wrapping the `ResponseWriter` without a working `Flush` | tokens wait in a 4 KiB buffer; TTFT becomes the whole answer's time | `meterWriter.Flush` forwards to the underlying writer; complete events are written at once |
| trusting the engine's `cached_tokens` | the `CHECK (cached_tokens <= prompt_tokens)` rejects the row and it is lost | `Record` clamps cached tokens to the prompt tokens |
| building SQL with `fmt.Sprintf` | `tenant=x' OR '1'='1` reads every tenant | every value is a `?` parameter; only the whitelisted column is spliced |
| an admin handler that trusts its mount point | mounted outside authn, it serves billing data to anyone | `Admin` checks the principal and the `admin` scope first |
| acknowledging before the commit, or no WAL | a `SIGKILL` loses acknowledged rows, or a rollback journal blocks readers | `Record` returns after `ExecContext` commits; WAL in the schema and the DSN |
| forwarding the usage chunk the meter asked for | clients that never asked for usage receive a chunk with `choices: []` | `inspect` returns false for that chunk when `strip` is set |
| `until` inclusive | the request at exactly 11:00 is in two hourly windows | `ts_ms < ?` for `until` |
| recording only successes, or a TTFT for refusals | refused traffic is invisible; error rows show a latency they never had | every metered request gets a row; first byte and tokens only for status below 400 |
| a cache revision that ignores the route table | after a `PUT /admin/v1/routes` the cache keeps serving answers from the old route's model | `Deps` sets the cache's `Rev` to `route.ETag(epoch)` |

## Where this leads

The ledger closes the chain that [the server skeleton](02-server-skeleton.md) defined, reading its `Exchange` and using `WriteError` for admin errors. [API keys and scopes](../../software-engineering/10-authorization/01-api-keys-and-scopes.md) supplies the `Principal` (tenant and key id per row) and the `admin` scope that `Admin` checks. `gateway.Deps` puts [rate limiting](../../software-engineering/10-authorization/02-rate-limiting.md)'s `limit.Middleware` in the `Limiter` slot; puts [routing](04-routing.md)'s `route.Middleware` in the chain and keys the cache by `route.ETag`, while `gateway.Admin` builds the workers, routes, and drain handlers; and puts [the response cache](../../systems/06-distributed-data-and-caching/01-response-cache.md)'s `cache.Middleware` in the chain, while `gateway.Admin` builds `PurgeHandler`. The SQL itself (transactions, WAL, indexes, aggregates, and the same usage table) comes from the SQL and SQLite primer.

Going forward, [Dockerfiles for the gateway and the engine](../../systems/10-containers-kubernetes/02-dockerfiles-for-gateway-and-engine.md) builds the gateway image with `CGO_ENABLED=0` around the pure-Go driver, with the usage DB on a volume. The end-to-end gateway check reconciles the ledger against the engine's usage through the admin API. And three later pieces read these tables: the agent's `query_usage` tool in [The tool gate, the SQL safety gate, and the prompt-injection suite](../28-agents/04-tool-gate-and-injection.md); [Interface migration: API v1 to v2](../../software-engineering/08-maintenance/02-api-v1-to-v2.md) (in *Software Engineering*), where API v2 fills `api_version` and `cached_tokens`; and the [noisy-neighbor drill](../../systems/12-incident-response-and-chaos/10-noisy-neighbor.md), which reads per-tenant usage.

### Further reading

- The SQLite ledger in production: [Litestream](https://litestream.io/how-it-works/) (free) streams the WAL to object storage; ClickHouse provides columnar usage analytics; [OpenMeter](https://openmeter.io/docs) (free) treats metering as an event pipeline.
- `Meter` in production: [LiteLLM spend tracking](https://docs.litellm.ai/docs/proxy/cost_tracking) (free) and [Envoy AI Gateway usage limiting](https://aigateway.envoyproxy.io/docs/capabilities/traffic/usage-based-ratelimiting) (free), with per-key budgets enforced from the same usage and cost per model.
- The admin API in production: Stripe usage records and the Kong Admin API, with idempotency keys on writes, pagination, and audit logs. See [Stripe idempotent requests](https://docs.stripe.com/api/idempotent_requests) (free).
- [SQLite: write-ahead logging](https://www.sqlite.org/wal.html), [SQLite: atomic commit](https://www.sqlite.org/atomiccommit.html), [SQLite: the query planner](https://www.sqlite.org/queryplanner.html), and [OpenAI: streaming usage](https://platform.openai.com/docs/api-reference/chat/create#chat-create-stream_options) (all free).

## Key terms

- **meter**: the innermost gateway stage, which writes one usage row per request after the response ends.
- **usage ledger**: the durable per-request record of tenant, key, model, status, tokens, and latency.
- **`include_usage`**: the `stream_options` flag that makes an engine send a final usage chunk in a stream.
- **WAL (write-ahead log)**: SQLite's journal mode in which commits are appended to a separate log, so readers do not block the writer and a torn tail is ignored on recovery.
- **`busy_timeout`**: how long a SQLite connection waits for the write lock before failing with `SQLITE_BUSY`.
- **half-open window**: a time range `[since, until)` that includes its start and excludes its end.
- **fail closed**: rejecting a request when authorization cannot be confirmed, rather than allowing it.

## Summary

- The meter is the innermost stage (`proxy -> meter`): it sees the finished exchange and writes one row per request, refused requests included, when the response ends.
- An engine sends a stream's usage only when asked, so the meter sets `stream_options.include_usage` for every stream and hides that usage chunk from a client that never asked for it.
- The ledger write must outlive the request: a client that hangs up mid-stream still used the tokens, and a write made on the request's context is cancelled with it.
- SQLite in WAL mode with one statement per row makes a `SIGKILL` leave every row complete or absent; `busy_timeout` makes concurrent writers wait instead of failing; `request_id` makes a retried write a no-op.
- Queries are half-open windows `[since, until)` with bound parameters; a `GROUP BY` column comes from a whitelist, because a column name cannot be a parameter.
- The admin API fails closed (no principal is a 401, no `admin` scope a 403, an unbuilt path a well-formed 404), and `gateway.Deps` keys the cache by the route epoch, so a new route table never serves an old answer.

## Review questions

1. Why does the meter strip the usage chunk only when `WithIncludeUsage` reported a change?

   <details><summary>Answer</summary>

   If the client set `include_usage` itself, the chunk is part of the answer it asked for and must arrive byte for byte. The meter removes only what it added, so either way the client sees exactly what an engine asked directly would send.

   </details>

2. In the hand example, why is `req-4` (12:03) not in the window `since=12:01, until=12:03`, and what would break if it were?

   <details><summary>Answer</summary>

   The window is half-open, `ts_ms < until`. If `until` were inclusive, a request at exactly 12:03:00.000 would be counted in both `[12:01, 12:03]` and `[12:03, 12:05]`, so consecutive panels would sum to more than the total.

   </details>

3. What happens to the row of a client that hangs up mid-stream, step by step?

   <details><summary>Answer</summary>

   The request context is cancelled; the proxy's upstream call ends; `next.ServeHTTP` returns. The meter still builds the record from what it read so far and writes it on `context.WithoutCancel(r.Context())` with its own 5 s timeout, so the cancellation does not reach the `INSERT`.

   </details>

4. Why is `busy_timeout` in the DSN and not executed once after `sql.Open`?

   <details><summary>Answer</summary>

   `*sql.DB` is a pool that opens new connections when goroutines need them. A `PRAGMA` executed once applies to whichever connection ran it; every other connection keeps a zero timeout and fails with `SQLITE_BUSY` under concurrent writes. The driver applies DSN `_pragma`s to every connection it opens.

   </details>

5. Why does `Admin` check the scope even though authn already ran in front of it?

   <details><summary>Answer</summary>

   Defense in depth: the handler's safety must not depend on how it is mounted. If it is ever mounted outside the authn chain (a refactor, a debug server), no principal means 401 instead of every tenant's usage served to anyone.

   </details>
