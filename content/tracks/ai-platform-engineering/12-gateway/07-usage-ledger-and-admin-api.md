# Usage ledger, metering, admin API

## Overview

| | |
|---|---|
| **Module** | `gw.07` · build · Go · Pass 7 · 4 to 6 h |
| **Solution** | [`ledger.go`](../../../../course/solutions/go/gateway/ledger/ledger.go) (the SQLite store: `DSN`, `Open`, `Record`, `Query`), [`meter.go`](../../../../course/solutions/go/gateway/ledger/meter.go) (`WithIncludeUsage`, `meterWriter`, the `Meter` stage of the chain), [`admin.go`](../../../../course/solutions/go/gateway/ledger/admin.go) (`Admin`, the whole `/admin/v1` surface, and `UsageHandler`), [`gateway.go`](../../../../course/solutions/go/gateway/gateway.go) (`Deps` and `Admin`: the stages assembled into the chain and the admin API); entry points: the gateway's [`compose.go`](../../../../course/solutions/entry/go/cmd/gateway/compose.go) and the `usage` verb of the platform CLI, [`ctl/usage.go`](../../../../course/solutions/entry/go/cmd/ctl/usage.go) |
| **Contract** | the schema [`course/contracts/formats/usage.v1.sql`](../../../../course/contracts/formats/usage.v1.sql) (applied verbatim) and the server side of [`course/contracts/openapi/admin.v1.yaml`](../../../../course/contracts/openapi/admin.v1.yaml); the Go API is section 4 |
| **Needs** | `gw.01` server skeleton (the chain, `Exchange`, `WriteError`), `gw.02` API keys and scopes (the `Principal`), `gw.03` rate limiting, `gw.05` routing (the route stage and its admin handlers), `gw.06` response cache (the stage and its purge handler), [`lang.11` SQL and SQLite](../../software-craftsmanship/12-language-and-tool-primers/11-sql-and-sqlite.md) (reading) |
| **Used by** | `dep.01` gateway image (pure-Go SQLite, so the image needs no C toolchain) · later `ag.04` the agent's `query_usage` tool, `craft.14` per-version usage, `ops.09` per-tenant usage · the end-to-end gateway check: the ledger reconciles with the engine's usage |
| **Optional depth** | [SQLite: write-ahead logging](https://www.sqlite.org/wal.html) (free), [SQLite: atomic commit](https://www.sqlite.org/atomiccommit.html) (free), [SQLite: the query planner](https://www.sqlite.org/queryplanner.html) (free), [OpenAI: streaming usage](https://platform.openai.com/docs/api-reference/chat/create#chat-create-stream_options) (free) |

## Key Takeaways

- The **meter** is the innermost stage (`proxy -> meter`): it sees the finished exchange and writes **one row per request**, refused requests included, when the response ends.
- An engine sends a stream's usage **only when asked**: the meter sets `stream_options.include_usage` for every stream and hides that usage chunk from a client that never asked for it.
- The ledger write must **outlive the request**: a client that hangs up mid-stream still used the tokens, and a write made on the request's context is cancelled with it.
- SQLite in **WAL mode** with one statement per row makes a `SIGKILL` leave every row complete or absent; `busy_timeout` makes concurrent writers wait instead of failing; `request_id` makes a retried write a no-op.
- Queries are **half-open** windows `[since, until)` with **bound parameters**; a `GROUP BY` column comes from a whitelist, because a column name cannot be a parameter.
- The admin API **fails closed**: no principal is a 401, no `admin` scope a 403, and a path whose module is not built yet is a well-formed 404.
- `gateway.Deps` and `gateway.Admin` **assemble** the stages, so the chain's wiring is code a test can call; the cache keys answers by the **route epoch**, so a new route table never serves an old answer.

## How to read this chapter

Sections 1 to 3 build the ledger from first principles and work four requests by hand. Section 4 fixes the interface and shows how the gateway's entry point wires it. Section 5 walks through the solution and explains why each choice is the right one; try to predict each step before reading it. The ledger uses one third-party package, the pure-Go SQLite driver `modernc.org/sqlite`.

---

## 1. Why now

After `gw.01` to `gw.06` the gateway authenticates every request, limits it, caches it, and routes it, and it forgets each one the moment the response ends. The rate limiter's token buckets live in memory and reset on every restart, so they cannot say how many tokens `acme` used yesterday. Nothing can: not the bill, not the per-tenant panel that `ops.09`'s noisy-neighbor drill needs, not the agent's `query_usage` tool (`ag.04`), and not the end-to-end check that what the gateway charged equals what the engine produced. This module adds the one durable record of who used what: a row per request in a SQLite file, written by the last stage of the chain, read through `GET /admin/v1/usage`. It also gives the admin endpoints that `gw.02`, `gw.05`, `gw.06`, and `gw.08` build one front door.

## 2. Principles

### 2.1 Metering: one row per request, from the response

The chain of `gw.01` is `requestid -> otel -> recover -> authn -> policy -> ratelimit -> cache -> route -> proxy -> meter`. The meter wraps the proxy directly, so it runs after every other stage has decided and it sees the response go by. It meters the three inference routes (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`) and lets every other path through untouched. Each column of the row has one source:

| Column | Source |
|---|---|
| `request_id`, `ts_ms` | the `Exchange` (`gw.01`): `RequestID`, `Start` |
| `tenant`, `key_id` | the `Principal` that `gw.02`'s authn stage put in the context |
| `model`, `route`, `stream` | the parsed request (`Exchange.Request`) and the path |
| `served_model` | the `model` field of the response: an alias or a canary answers with another id |
| `status`, `error_code` | the status sent to the client; the error body's `code` (its `type` when the code is null); an SSE error event's code for a stream that failed after its first byte |
| `prompt_tokens`, `completion_tokens`, `cached_tokens` | the response's `usage` object (`prompt_tokens_details.cached_tokens` when present) |
| `ttft_ms`, `e2e_ms` | first content byte minus start, NULL when none was sent or the request was refused; end minus start, on the gateway's clock |
| `worker_id`, `cache_hit`, `trace_id` | the router's choice and the cache flag on the `Exchange`; the trace id of the request's `traceparent` |

**Where usage comes from.** A non-streamed answer carries `usage` in its body. A stream carries it in one extra chunk near the end, `{"choices": [], "usage": {...}}`, and the engine sends that chunk **only when the request says** `"stream_options": {"include_usage": true}` (`openai-subset.v1.yaml`). Most clients do not ask, so a meter that only reads would bill every stream zero tokens. The meter therefore adds `include_usage` to every stream request it forwards. That changes the answer the client receives, so the meter also **removes** that usage chunk on the way back when the client did not ask for it. A client that did ask gets the engine's bytes unchanged.

**The invariant.** For every tenant, the ledger and the engine agree:

| Symbol | Meaning | Type |
|---|---|---|
| $t$ | a tenant | string |
| $L_t$ | the ledger rows of tenant $t$ | set of rows |
| $E_t$ | the requests of $t$ that the engine answered with a `usage` object | set of requests |
| $p_r, c_r$ | prompt and completion tokens of a row or request $r$ | integers $\ge 0$ |

$$\sum_{r \in L_t} p_r = \sum_{e \in E_t} p_e, \qquad \sum_{r \in L_t} c_r = \sum_{e \in E_t} c_e$$

Refused requests are rows with zero tokens, so they count in `requests` and `errors` and add nothing to the sums. A correct solution keeps the invariant over any mix of streamed, non-streamed, and refused requests, and against a real engine.

### 2.2 The store: SQLite, WAL, one statement per row

The ledger is a single SQLite file at `[gateway].usage_db`. `Open` applies `usage.v1.sql` exactly as written (every statement is `IF NOT EXISTS` or `OR IGNORE`, so reopening is a no-op), and refuses a file whose `schema_version` is newer than 1: an old gateway must not write into a table a later migration reshaped. The driver is `modernc.org/sqlite`, SQLite compiled to pure Go, so the gateway still builds with `CGO_ENABLED=0` into a static binary (`dep.01`).

Four settings decide whether rows survive (`lang.11` worked through each one):

| Setting | Scope | Why |
|---|---|---|
| `journal_mode = WAL` | stored in the file | a commit appends frames and a commit record to `usage.db-wal`; readers never block the writer; recovery ignores a tail without its commit record |
| `synchronous = NORMAL` | per connection | in WAL mode, a commit survives a **process** kill (the bytes are in the OS page cache); only a power loss can take the last commits. `FULL` would `fsync` on every row |
| `busy_timeout = 5000` | per connection | SQLite has one writer at a time; without a timeout a second writer fails at once with `SQLITE_BUSY` and its row is lost |
| `request_id` primary key, `ON CONFLICT DO NOTHING` | schema and statement | a retried write (a timeout after the commit) leaves the first row alone |

`database/sql` keeps a **pool** of connections, so a per-connection setting made with one `Exec("PRAGMA ...")` reaches one connection only. The settings go into the data source name instead, which the driver applies to every connection it opens: `usage.db?_pragma=busy_timeout(5000)&_pragma=journal_mode(WAL)&_pragma=synchronous(NORMAL)`.

A row is one `INSERT` statement, so it commits whole or not at all. `Record` returns only after the commit: an acknowledgement means the row is in the WAL. Cached tokens larger than the prompt (an engine bug) would violate the table's `CHECK` and lose the whole row, so `Record` clamps them; negative counts and an empty request id are refused with `ErrInvalidRecord`.

### 2.3 Queries: half-open windows, bound values, whitelisted columns

`Query(UsageFilter)` sums rows: `requests`, `errors` (status at least 400, or an error code), and the three token counts.

- **Windows are half-open**, `since <= ts_ms < until`. Then `[10:00, 11:00)` and `[11:00, 12:00)` add up to `[10:00, 12:00)` with no request at exactly `11:00:00.000` counted twice.
- **Values are bound parameters** (`tenant = ?`). The tenant arrives from the admin API's query string; spliced into the SQL text, `x' OR '1'='1` reads every tenant's usage.
- **A column name cannot be a parameter**, so `group_by` is looked up in a fixed map (`model`, `tenant`, `key_id`, `api_version`) and anything else is `ErrBadGroupBy`.
- **No grouping gives exactly one row**, zeros when nothing matches, so a caller never has to special-case an empty answer. Grouped rows are sorted by the group key.

### 2.4 What a SIGKILL can do

The kubelet `SIGKILL`s a pod that exceeds its memory limit or fails its liveness probe; nothing in the process runs afterwards. What survives is whatever reached the OS before the signal. With WAL and one statement per row, every committed row is in `usage.db-wal` (the OS writes it to disk later even though the process is gone), and a commit that was half-written has no commit record, so the next `Open` ignores it. Two designs break this: a ledger that collects rows in memory and writes them in batches (every acknowledged row in the batch is lost), and one that writes a row in two statements without a transaction (a kill between them leaves a torn row). Killing a writer process after 400 acknowledgements must leave every acknowledged row present and complete, column by column.

### 2.5 The admin API: one front door, fail closed

`admin.v1.yaml` has eight operations owned by five modules. This module owns the server side: one handler, `Admin(AdminDeps)`, for every `/admin/v1` path. It serves usage itself and hands every other path to the handler its module builds (keys from `gw.02`, workers, routes, and drains from `gw.05`, the cache purge from `gw.06`, the policy from `gw.08`). A path whose handler is nil answers 404 `not_found`, so a gateway in the middle of Pass 7 still has a well-formed admin surface. `Admin` checks the principal itself: no principal is 401 `invalid_api_key` and a key without `admin` is 403 `insufficient_scope`. `gw.02`'s authn stage already enforces that, but an admin handler that trusts its mount point serves every tenant's billing data the first time someone mounts it outside the chain.

## 3. Worked example by hand

Four requests reach the meter (times are 2026-10-01, UTC):

| `request_id` | start | tenant | key | model | route | status | code | prompt | completion |
|---|---|---|---|---|---|---|---|---|---|
| `req-1` | 12:00 | acme | `acmekeyaaaaa` | smol | chat, stream | 200 | | 12 | 30 |
| `req-2` | 12:01 | acme | `acmekeybbbbb` | smol | chat | 200 | | 20 | 10 |
| `req-3` | 12:02 | acme | `acmekeyaaaaa` | tiny | completions | 429 | `rate_limit_exceeded` | 0 | 0 |
| `req-4` | 12:03 | globex | `globexkeyccc` | smol | chat | 200 | | 7 | 5 |

**The stream, `req-1`.** The client sends `{"model":"smol","stream":true,"messages":[{"role":"user","content":"Once"}]}`. The meter forwards `{"messages":[...],"model":"smol","stream":true,"stream_options":{"include_usage":true}}`. The engine answers with three content chunks, then `data: {...,"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":30,"total_tokens":42}}`, then `data: [DONE]`. The meter reads 12 and 30 from the usage chunk, drops that chunk, and the client receives the three content chunks and `[DONE]`: exactly what it would have received from an engine it asked directly.

**The rows, then the queries.**

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

## 4. The interface

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
	Keys *auth.Store; Logger *slog.Logger                  // gw.02
	Policy server.Middleware; PolicyAdmin http.Handler     // gw.08
	Limiter *limit.Limiter; Counter limit.Counter          // gw.03
	Cache *cache.LRU; CacheOptions cache.Options           // gw.06
	Registry *route.Registry; Router *route.Router         // gw.05
	Proxy http.Handler                                     // route.Proxy, or gw.04's proxy
	Ledger ledger.Ledger; Meter ledger.MeterOptions        // this module
}
func Deps(s Stages) server.Deps          // a nil stage is left out; the cache's Rev defaults to route.ETag(epoch)
func Admin(s Stages) ledger.AdminDeps    // KeysHandler, WorkersHandler, RoutesHandler, DrainHandler, PurgeHandler
```

`gateway.go` is the part of the composition root a test can call. `Deps` calls each module's middleware constructor (`auth.Middleware`, `limit.Middleware`, `cache.Middleware`, `route.Middleware`, `ledger.Meter`) and leaves a nil stage out; gw.01's `Handler` puts them in contract order. One wiring decision is made here and nowhere else: with a `Router` and no `CacheOptions.Rev`, the cache revision is the route table's `ETag`, so a `PUT /admin/v1/routes` makes every answer cached under the old table unreachable. `Admin` builds each admin path's handler from the module that owns it.

The unexported helpers (`meterWriter`, `inspect`, `errorCode`, `traceID`, ...) are the solution's decomposition, not part of the interface.

**The entry points.** The gateway's composition root ([`compose.go`](../../../../course/solutions/entry/go/cmd/gateway/compose.go)) opens the ledger at `[gateway].usage_db`, builds the stages into a `gateway.Stages`, and serves the admin API through a second, short chain that has only authn in front of it, so admin calls never reach the router. In its simplest form:

```go
st := gateway.Stages{Keys: keys, Limiter: limiter, Counter: counter, Cache: lru, CacheOptions: cache.Options{TTL: ttl},
	Registry: reg, Router: router, Proxy: route.Proxy(router, route.Options{}), Ledger: db}
inference := server.New(cfg, gateway.Deps(st)).Handler()
admin := server.New(cfg, server.Deps{Keys: auth.Middleware(keys, nil), Proxy: ledger.Admin(gateway.Admin(st))}).Handler()
mux := http.NewServeMux()
mux.Handle("/admin/v1/", admin)
mux.Handle("/", inference)
```

(The full `compose.go` adds API-version selection and moves the meter out to wrap the cache, so cache hits are metered too.) The platform CLI's `usage` verb, `ctl usage --gateway <base URL> [--tenant T] [--key-id K] [--since T] [--until T] [--group-by G]`, calls `GET /admin/v1/usage` with the admin key from `TL_API_KEY`, prints each row as a JSON line, and ends with a line of totals.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestHandWorkedExample` | unit | section 3: acme 3/1/32/40, by model, the window, the empty tenant | you and the tests agree on every definition |
| `TestSchemaMatchesContract` | conformance | `Open` creates what `usage.v1.sql` creates (compared through `sqlite_master`), version 1, WAL | `ag.04` and `ops.09` read these tables |
| `TestReopenKeepsRowsAndRefusesNewerSchema` | boundary | reopening keeps rows; `schema_version` 2 is refused with `ErrNewerSchema` | rollouts restart the gateway; `craft.14` migrates the schema |
| `TestRecordIsIdempotentByRequestID` | boundary | a second record of `req-1` changes nothing | retried writes never double bill |
| `TestRecordValidatesAndClampsCachedTokens` | boundary | cached 50 > prompt 20 is stored as 20; negative counts and an empty id are refused | an engine bug never loses a row |
| `TestRecordStoresEveryColumn` | unit | every field in its column and unit; NULL code and TTFT; `api_version` defaults to `'1'` | columns, not Go structs, are the contract |
| `TestQueryWindowIsHalfOpen` | boundary | `[10:00,11:00)` + `[11:00,12:00)` = `[10:00,12:00)` | hourly panels add up |
| `TestQueryValuesAreBoundParameters` | unit | quotes in tenant or key id change nothing | the admin API passes them from the URL |
| `TestQueryGroupByIsWhitelisted` | boundary | `status`, `Model`, an injection are `ErrBadGroupBy`; key rows sorted | the admin API's 400 |
| `TestConcurrentRecordsAllLand` | fault | 16 writers x 64 rows, no `SQLITE_BUSY`, 1024 rows | every request goroutine records at once |
| `TestHelperLedgerWriter` | fault | not a check: the child process of the next test (skips otherwise) | |
| `TestSIGKILLLeavesRowsCompleteOrAbsent` | fault | after `SIGKILL`: integrity ok, WAL, every acknowledged row present and complete | OOM kills on kind; drill `ops.01` |
| `TestLedgerSumEqualsEngineUsage` | property | 1000 mixed requests: per-tenant ledger sums equal the engine's usage; 59 refusals counted | MS-gateway's reconciliation |
| `TestStreamUsageIsRequestedAndHidden` | unit | `include_usage` added and its chunk hidden; a client that asked gets the bytes unchanged | streams are billed; clients see what they asked for |
| `TestMeterDoesNotBufferTheStream` | unit | the first event reaches the client while the engine holds the rest | TTFT, the gateway's SLO |
| `TestRefusedRequestsAreRecordedWithTheirCode` | unit | 429 and 400 rows with code, zero tokens, NULL TTFT; a mid-stream error event is an error | `ops.09` sees who hits limits |
| `TestMeterRecordsWhoWhatAndWhere` | unit | tenant, key, model, served model, route, worker, cache flag, trace id, gateway-clock latency | per-tenant and per-worker panels |
| `TestClientDisconnectStillRecords` | fault | a client that hangs up mid-stream still gets a row | abandoned streams still cost tokens |
| `TestUnmeteredPathsPassThrough` | boundary | `GET /v1/models` untouched, no row | only inference is billed |
| `TestAdminUsageEndpoint` | conformance | `{since, until, data}`, null window, one ungrouped row, rows grouped by model | `<system> usage`, `ag.04` |
| `TestAdminUsageRejectsBadParameters` | boundary | bad `since`, `until`, `group_by`, an empty window: 400 naming the param; POST is 405 | the CLI can name the bad flag |
| `TestAdminRequiresAdminScope` | unit | no principal 401, an `infer` key 403 | billing data stays private |
| `TestAdminMountsEveryPath` | unit | each path reaches its module's handler; nil handlers and unknown paths are 404 `not_found` | `gw.02`, `gw.05`, `gw.06`, `gw.08` mount here |
| `TestGatewayAssemblesTheStages` | unit | real keys, limits, cache, router, and ledger through `gateway.Deps` and `gateway.Admin`: a repeat is a hit, a route PUT makes it a miss, the 4th request of an RPM-3 key is a 429 before the cache, purge, workers, routes, drain, keys, and usage reach their handlers | your `go/cmd/gateway` is these two calls |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. reading usage only from the response | every stream is billed 0 tokens: engines send stream usage only when asked | `TestStreamUsageIsRequestedAndHidden`, `TestLedgerSumEqualsEngineUsage` (mutant `s02`) |
| 2. writing the row on `r.Context()` | a client that hangs up mid-stream leaves no row: the write is cancelled with the request | `TestClientDisconnectStillRecords` (mutant `s01`) |
| 3. no `busy_timeout`, or a `PRAGMA` run once on a pooled `*sql.DB` | `database is locked` under load, rows lost | `TestConcurrentRecordsAllLand` (mutant `s06`) |
| 4. `INSERT OR REPLACE` (or an upsert) on `request_id` | a retried write rewrites the first row with new numbers | `TestRecordIsIdempotentByRequestID` (mutant `s04`) |
| 5. wrapping the `ResponseWriter` without a working `Flush` | tokens wait in a 4 KiB buffer; TTFT becomes the whole answer's time | `TestMeterDoesNotBufferTheStream` (mutant `s20`) |
| 6. trusting the engine's `cached_tokens` | the `CHECK (cached_tokens <= prompt_tokens)` rejects the row and it is lost | `TestRecordValidatesAndClampsCachedTokens` (mutant `s10`) |
| 7. building SQL with `fmt.Sprintf` | `tenant=x' OR '1'='1` reads every tenant | `TestQueryValuesAreBoundParameters` (mutant `s08`) |
| 8. an admin handler that trusts its mount point | mounted outside authn, it serves billing data to anyone | `TestAdminRequiresAdminScope` (mutant `s13`) |
| 9. acknowledging before the commit, or no WAL | a `SIGKILL` loses acknowledged rows, or a rollback journal blocks readers | `TestSIGKILLLeavesRowsCompleteOrAbsent`, `TestSchemaMatchesContract` (mutants `s16`, `s05`) |
| 10. forwarding the usage chunk you asked for | clients that never asked for usage receive a chunk with `choices: []` | `TestStreamUsageIsRequestedAndHidden` (mutant `s03`) |
| 11. `until` inclusive | the request at exactly 11:00 is in two hourly windows | `TestQueryWindowIsHalfOpen` (mutant `s07`) |
| 12. recording only successes, or a TTFT for refusals | refused traffic is invisible; error rows show a latency they never had | `TestRefusedRequestsAreRecordedWithTheirCode` (mutants `s15`, `s12`, `s22`) |
| 13. a cache revision that ignores the route table | after a `PUT /admin/v1/routes` the cache keeps serving answers from the old route's model | `TestGatewayAssemblesTheStages` (mutant `s23`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `gw.01` | the chain the meter closes, the `Exchange` it reads, `WriteError` for the admin errors |
| Back | `gw.02` | the `Principal`: tenant and key id per row, the `admin` scope `Admin` checks |
| Back | `gw.03` | `gateway.Deps` puts `limit.Middleware` in the `Limiter` slot |
| Back | `gw.05` | `gateway.Deps` puts `route.Middleware` in the chain and keys the cache by `route.ETag`; `gateway.Admin` builds the workers, routes, and drain handlers |
| Back | `gw.06` | `gateway.Deps` puts `cache.Middleware` in the chain; `gateway.Admin` builds `PurgeHandler` |
| Back | `lang.11` | the SQL: transactions, WAL, indexes, aggregates, and the same usage table |
| Forward | `dep.01` | builds the gateway image with `CGO_ENABLED=0` around the pure-Go driver, the usage DB on a volume |
| Forward | end-to-end gateway check | reconciles the ledger against the engine's usage through the admin API |
| Forward | `ag.04` (B12), `craft.14`, `ops.09` | the agent's `query_usage` tool reads these tables; API v2 fills `api_version` and `cached_tokens`; the noisy-neighbor drill reads per-tenant usage |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| the SQLite ledger | Litestream, ClickHouse, OpenMeter | streaming the WAL to object storage; columnar usage analytics; metering as an event pipeline | [Litestream](https://litestream.io/how-it-works/) (free), [OpenMeter](https://openmeter.io/docs) (free) |
| `Meter` | LiteLLM spend tracking, Envoy AI Gateway token usage | per-key budgets enforced from the same usage, cost per model | [LiteLLM spend tracking](https://docs.litellm.ai/docs/proxy/cost_tracking) (free), [Envoy AI Gateway usage limiting](https://aigateway.envoyproxy.io/docs/capabilities/traffic/usage-based-ratelimiting) (free) |
| the admin API | Stripe usage records, Kong Admin API | idempotency keys on writes, pagination, audit logs | [Stripe idempotent requests](https://docs.stripe.com/api/idempotent_requests) (free) |
