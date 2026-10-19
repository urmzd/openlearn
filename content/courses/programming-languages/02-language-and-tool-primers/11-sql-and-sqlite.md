# SQL and SQLite: schema, transactions, WAL mode, indexes, aggregates

Up to now the gateway forgets every request the moment the response ends, so nobody can say how many tokens `acme` used yesterday, which key is burning through its budget, or whether the canary model is slower than the stable one. The gateway needs a memory. In [The usage ledger and admin API](../../ai-platform-engineering/12-gateway/07-usage-ledger-and-admin-api.md) in *AI Platform Engineering* it gets one: a usage ledger with one row per request in a SQLite database, written by the gateway and read through its admin API, by the agent's `query_usage` tool, and by the noisy-neighbor drill.

Every one of those readers is a SQL query, and the ledger is only useful if its writes survive a crash and its reads are right at the edges: an empty window, a tie, a missing value. This section teaches that SQL on the real table, before the gateway's Go code fills it. You will write five queries over the usage ledger, an index, the connection settings, a rollup table, and the transaction that records one request.

You need little background: [Shell, git, make, processes](02-shell-git-make.md) covers the shell and exit codes you use with the `sqlite3` command line. The table the queries read is the contract [`course/contracts/formats/usage.v1.sql`](../../../../course/contracts/formats/usage.v1.sql), the gateway's usage ledger. The solution is nine files in [`course/solutions/primers/lang.11/`](../../../../course/solutions/primers/lang.11/): the queries [`q1_tenant_totals.sql`](../../../../course/solutions/primers/lang.11/q1_tenant_totals.sql) to [`q5_ttft_p95.sql`](../../../../course/solutions/primers/lang.11/q5_ttft_p95.sql), [`indexes.sql`](../../../../course/solutions/primers/lang.11/indexes.sql), [`pragmas.sql`](../../../../course/solutions/primers/lang.11/pragmas.sql), [`rollup.sql`](../../../../course/solutions/primers/lang.11/rollup.sql), and [`record.sql`](../../../../course/solutions/primers/lang.11/record.sql). To try the solution by hand:

```bash
sqlite3 /tmp/u.db < contracts/formats/usage.v1.sql
sqlite3 /tmp/u.db ".param set :tenant acme" ".param set :since 0" ".param set :until 9999999999999" \
  ".read primers/lang.11/q1_tenant_totals.sql"
```

> **Learning objectives**
> - Use constraints so that the database refuses bad rows, and apply a schema idempotently.
> - Write aggregate queries over half-open time windows that are correct over empty input, NULLs, and ties.
> - Compute a nearest-rank percentile with window functions.
> - Choose WAL mode and per-connection pragmas, and design an index whose column order matches the query, checked with `EXPLAIN QUERY PLAN`.
> - Make a two-statement write atomic and idempotent with a transaction, `ON CONFLICT`, and `changes()`.

## Tables, rows, and constraints

A **relational database** stores **tables**. A table has named **columns**, each with a type, and holds **rows**, one value per column. SQLite's types are `INTEGER`, `REAL`, `TEXT`, `BLOB`, and the absent value `NULL`. Open `contracts/formats/usage.v1.sql` and you will find the `usage` table, with one row per request: `request_id TEXT NOT NULL PRIMARY KEY`, `ts_ms INTEGER` (the start time in Unix milliseconds), `tenant`, `key_id`, `model`, `status`, `error_code` (NULL on success), token counts, and `ttft_ms REAL` (NULL when no content byte was sent).

A **constraint** is a rule the database enforces on every write, so that it refuses bad rows instead of storing them. The ledger uses four kinds:

| Constraint | Meaning | In the ledger |
|---|---|---|
| `PRIMARY KEY` | the column (or tuple of columns) identifies one row; a second row with the same key is refused | `request_id`: one row per request |
| `NOT NULL` | the column always has a value | every column but `error_code` and `ttft_ms` |
| `CHECK (expr)` | `expr` must be true for every row | `cached_tokens <= prompt_tokens`, `stream IN (0, 1)` |
| `DEFAULT v` | the value when an `INSERT` leaves the column out | `api_version DEFAULT '1'` |

`CREATE TABLE IF NOT EXISTS` makes a schema file safe to apply on every start: the second time it does nothing. The solution's `rollup.sql` defines a second table, `usage_hourly`, with one row per tenant and hour. Its primary key is the pair `(tenant, hour_ms)`, and a `CHECK (hour_ms % 3600000 = 0)` keeps every `hour_ms` on an hour boundary.

## Connections, pragmas, and the journal

Before reading or writing, a program needs a **connection**, and how that connection is configured decides both durability and concurrency. A **pragma** is a SQLite setting, `PRAGMA name = value`. Some are stored in the database file and apply to every connection; most belong to the one connection that ran them. That difference matters as soon as a program keeps a pool of connections, as Go's `database/sql` does. The ledger needs four:

| Pragma | Scope | Value for the ledger | Why |
|---|---|---|---|
| `journal_mode` | the file | `WAL` | see below |
| `synchronous` | connection | `NORMAL` (1) | with WAL, a commit survives a process crash; only a power cut can lose the last commits, and no `fsync` per commit |
| `busy_timeout` | connection | at least 1000 ms | a writer that finds the database locked waits up to this long instead of failing with `SQLITE_BUSY` |
| `foreign_keys` | connection | `ON` | SQLite enforces `REFERENCES` only when this is on |

The journal deserves its own explanation. A database must survive a crash in the middle of a write. SQLite's default **rollback journal** copies each page it is about to change into `db-journal`, then changes the database in place; a crash is undone from the copy on the next open. While a writer works, readers wait. In **write-ahead logging** (WAL) the database file is not touched by a commit: the changed pages are **appended** to `db-wal`, followed by a commit record. Readers keep reading the old pages plus every committed frame, so readers never block the writer and the writer never blocks readers. On open after a crash, SQLite replays the WAL up to the last commit record and ignores a torn tail. Every so often a **checkpoint** copies the WAL back into the database file. There is still only **one writer at a time**, which is what `busy_timeout` is for.

## Reading: SELECT, WHERE, aggregates, GROUP BY

With the table in place, the queries. A query is evaluated in this order: `FROM` picks the table, `WHERE` keeps the rows whose condition is true, `GROUP BY` collects rows with equal values of the grouping expressions, the `SELECT` list computes one output row per group (or per row without `GROUP BY`), `ORDER BY` sorts, and `LIMIT` keeps the first rows.

An **aggregate** turns the rows of a group into one value. Using this notation:

| Symbol | Meaning |
|---|---|
| $G$ | the rows of one group |
| $\lvert G \rvert$ | the number of rows in $G$ |
| $x_r$ | the value of column $x$ in row $r$, possibly NULL |

the aggregates you need are defined as follows, including what each gives over no rows:

| Aggregate | Value | Over no rows |
|---|---|---|
| `COUNT(*)` | $\lvert G \rvert$ | 0 |
| `COUNT(x)` | the number of rows where $x_r$ is not NULL | 0 |
| `SUM(x)` | $\sum_{r \in G,\ x_r \ne \text{NULL}} x_r$ | **NULL** |
| `AVG(x)`, `MIN(x)`, `MAX(x)` | as named, ignoring NULLs | NULL |

The NULL in the last column is the classic surprise: the total tokens of a tenant with no requests is `NULL`, not 0. `COALESCE(a, b)` returns `a` unless it is NULL, so `COALESCE(SUM(prompt_tokens), 0)` is the total you want. Without `GROUP BY`, an aggregate query returns exactly one row even when no row matched; with `GROUP BY`, an empty input gives no rows.

A comparison is 1 or 0 in SQLite, so `SUM(status >= 400 OR error_code IS NOT NULL)` counts errors: refused requests (status 400 and up) and streams that failed after their first byte (status 200 with an error code). Note `IS NOT NULL`: `error_code != NULL` is NULL for every row, never true.

## Windows of time and buckets

Times are integers (Unix milliseconds), so a window is two comparisons. Use **half-open** windows, `ts_ms >= :since AND ts_ms < :until`. Then the hours `[10:00, 11:00)` and `[11:00, 12:00)` add up to `[10:00, 12:00)`, and a request at exactly 11:00:00.000 is counted once. A bucket is an expression in `GROUP BY`: the start of the hour of `ts_ms` is `ts_ms - ts_ms % 3600000` (`%` is the remainder), because an hour is $3.6 \times 10^6$ ms.

## Window functions and a percentile

Aggregates collapse rows. Sometimes you need a value per row computed from its neighbours, and that is a **window function**: it computes a value for each row from a set of related rows, without collapsing them. `ROW_NUMBER() OVER (PARTITION BY model ORDER BY ttft_ms)` numbers each model's rows 1, 2, 3, ... in TTFT order, and `COUNT(*) OVER (PARTITION BY model)` puts the model's row count on every row.

Those two are enough for a percentile. Take these symbols:

| Symbol | Meaning |
|---|---|
| $n$ | the number of non-NULL TTFTs of one model |
| $v_i$ | the $i$-th smallest TTFT |
| $k$ | the rank of the percentile |

The **nearest-rank percentile** of $n$ values sorted ascending, $v_1 \le \dots \le v_n$, is the $p$-th percentile $v_k$ with $k = \lceil p\,n / 100 \rceil$. For $p = 95$, in integers, $k = \lfloor (95 n + 99) / 100 \rfloor$ (adding 99 before the floor division rounds up).

So `q5_ttft_p95.sql` ranks rows that **have** a TTFT (refusals and embeddings have NULL) and keeps the row where `rn = (95 * n + 99) / 100`. A `WITH name AS (SELECT ...)` clause, a common table expression, names that ranked set so the outer query can filter it.

## Parameters, never pasted values

A query's values come from outside: the admin API's URL, a CLI flag. Pasting them into the SQL text is **SQL injection**: `tenant = 'x' OR '1'='1'` reads every tenant, and a quote in a real name (`o'brien` is in the fixture) breaks the query. A **named parameter** (`:tenant`) is a placeholder; the value is bound separately and is always data. The query files take `:tenant`, `:since`, `:until`, and `:n` and nothing else. A column name cannot be a parameter, which is why the gateway's admin API picks `GROUP BY` columns from a fixed list.

## Indexes and the query plan

Correct queries also have to be fast enough. Without an index, a query with `WHERE served_model = ?` reads every row: a **scan**. An **index** is a B-tree holding chosen columns of every row in sorted order, plus a pointer to the row. With an index on `(served_model, ts_ms)`, all rows for one served model sit next to each other, sorted by time inside, so `served_model = ? AND ts_ms >= ?` is one jump to the first match and a walk to the end of the slice: a **search**.

Order matters. An index on `(ts_ms, served_model)` sorts by time first, so the rows of one model are scattered and only the time bound can use it. The rule is **equality columns first, then one range column**. `EXPLAIN QUERY PLAN <query>` shows the plan: `SCAN usage` or `SEARCH usage USING INDEX usage_served_ts (served_model=? AND ts_ms>?)`. An index costs space and slows every insert a little, so the contract indexes only what its readers filter on.

## Transactions, upserts, and replays

The last idea is how to write. A **transaction** groups statements: `BEGIN` starts one, `COMMIT` makes every change visible and durable at once, `ROLLBACK` discards them. Outside an explicit transaction, every statement is its own transaction (autocommit). An error inside a transaction fails only that statement; the program decides whether to `ROLLBACK`, and a connection that closes with a transaction open rolls it back. `BEGIN IMMEDIATE` takes the write lock at once rather than at the first write, so two writers queue on `busy_timeout` instead of failing halfway.

An **upsert** is `INSERT ... ON CONFLICT (key) DO UPDATE SET ...`: insert the row, or, when the key exists, update the existing row, where `excluded.col` is the value that was about to be inserted. `ON CONFLICT DO NOTHING` ignores the row instead. `changes()` is the number of rows the previous statement changed, so after `INSERT ... ON CONFLICT (request_id) DO NOTHING` it is 1 for a new request and 0 for a replay. `record.sql` uses all three: in one transaction, insert the usage row (ignored on a replay), then upsert the hourly rollup **only when** `changes() = 1`.

## Working it by hand

Take six requests on 2026-10-01 (12:00 is `ts_ms` 1790856000000), all on `/v1/chat/completions`:

| `request_id` | time | tenant | key | model | status | `error_code` | prompt | completion | `ttft_ms` |
|---|---|---|---|---|---|---|---|---|---|
| r1 | 12:05 | acme | `acmekeyaaaaa` | smol | 200 | NULL | 12 | 30 | 180 |
| r2 | 12:20 | acme | `acmekeybbbbb` | smol | 200 | NULL | 20 | 10 | 240 |
| r3 | 12:40 | acme | `acmekeyaaaaa` | tiny | 429 | `rate_limit_exceeded` | 0 | 0 | NULL |
| r4 | 13:10 | acme | `acmekeyaaaaa` | smol | 200 | NULL | 8 | 16 | 200 |
| r5 | 13:15 | globex | `globexkeyccc` | smol | 200 | NULL | 7 | 5 | 900 |
| r6 | 13:50 | acme | `acmekeybbbbb` | smol | 503 | `no_capacity` | 0 | 0 | NULL |

The window is `[12:00, 14:00)`, so every row is in it. Answer the five questions in turn.

1. **acme's totals (q1).** `WHERE tenant = 'acme'` keeps r1, r2, r3, r4, r6. Requests $5$; errors $2$ (r3 and r6); prompt $12 + 20 + 0 + 8 + 0 = 40$; completion $30 + 10 + 0 + 16 + 0 = 56$. The answer is `(5, 2, 40, 56)`.
2. **acme by model (q2)**, sorted by model: smol is r1, r2, r4, r6, so `(smol, 4, 1, 40, 56)`; tiny is r3, so `(tiny, 1, 1, 0, 0)`.
3. **The top 2 keys by tokens (q3)**, prompt plus completion: `acmekeyaaaaa` has $42 + 0 + 24 = 66$, `acmekeybbbbb` $30 + 0 = 30$, `globexkeyccc` $12$. The answer is `(acmekeyaaaaa, acme, 66), (acmekeybbbbb, acme, 30)`.
4. **acme per hour (q4).** $12{:}05 \to 1790856000000$, the 12:00 bucket, holds r1, r2, r3: 3 requests, $30 + 10 + 0 = 40$ completion tokens. The 13:00 bucket ($1790859600000$) holds r4, r6: 2 requests, 16 tokens.
5. **p95 TTFT per model (q5).** smol's non-NULL TTFTs are 180, 240, 200, 900, so $n = 4$, and sorted they are 180, 200, 240, 900. $k = \lfloor (95 \cdot 4 + 99)/100 \rfloor = \lfloor 479/100 \rfloor = 4$, so the p95 is $v_4 = 900$: with four samples the p95 is the maximum. tiny has no TTFT at all, so it has no row. The answer is `(smol, 4, 900.0)`.

The five queries of the solution, run on exactly these rows, give exactly these answers.

## Designing the files

The solution is nine files in `primers/lang.11/`. Each query file holds exactly one `SELECT`, takes its values as named parameters, and returns its columns in the order given:

| File | Holds | Columns, order |
|---|---|---|
| `q1_tenant_totals.sql` | `:tenant`'s totals in `[:since, :until)`; zeros, not NULL, when nothing matches | `requests, errors, prompt_tokens, completion_tokens` |
| `q2_by_model.sql` | `:tenant`'s usage per model in the window | `model, requests, errors, prompt_tokens, completion_tokens`, by model |
| `q3_top_keys.sql` | the `:n` keys with the most tokens (prompt plus completion) in the window, all tenants | `key_id, tenant, tokens`, by tokens descending, then `key_id` ascending |
| `q4_hourly.sql` | `:tenant`'s requests and completion tokens per hour, hours with rows only | `hour_ms, requests, completion_tokens`, by hour |
| `q5_ttft_p95.sql` | the nearest-rank p95 of `ttft_ms` per model in the window, NULLs left out | `model, n, p95_ttft_ms`, by model |
| `indexes.sql` | `CREATE INDEX` statements only, so that the query below searches an index | |
| `pragmas.sql` | the four pragmas of the pragma table above | |
| `rollup.sql` | `CREATE TABLE IF NOT EXISTS usage_hourly (tenant, hour_ms, requests, prompt_tokens, completion_tokens)`, keyed by `(tenant, hour_ms)`, hour-aligned by a `CHECK` | |
| `record.sql` | one request: the `usage` insert and the `usage_hourly` upsert in one transaction, a replayed `request_id` changing nothing; parameters are the usage columns (`:request_id`, `:ts_ms`, ..., `:trace_id`) | |

The query `indexes.sql` must serve is a per-canary panel:

```sql
SELECT worker_id, COUNT(*) AS requests, SUM(completion_tokens) AS completion_tokens
FROM usage WHERE served_model = :served_model AND ts_ms >= :since GROUP BY worker_id
```

`record.sql` is run by splitting it at each line that ends a statement and running the statements in order on one connection in autocommit mode, binding the row's columns by name.

## Building it step by step

### Reasoning from first principles

Try to write each query in your head from this chain before reading it.

1. **Every query is a filter, a grouping, and an order.** Write the `WHERE` first (tenant equality plus the half-open window), then decide what one output row is (the `GROUP BY`), then the order the reader needs.
2. **A total must be a number even over nothing.** q1 has no `GROUP BY`, so it always returns one row, and its `SUM`s are NULL when no row matched; `COALESCE(..., 0)` turns them into zeros. q2 to q4 group, so an empty window yields no rows and every group has at least one row: no `COALESCE` needed.
3. **An error is either refused or failed mid-stream.** `status >= 400 OR error_code IS NOT NULL`, summed as 0/1, counts both; `IS NOT NULL` because `!= NULL` is never true.
4. **A ranking must be deterministic.** "Top keys by tokens" needs a tie-break, so `key_id ASC` after `tokens DESC`.
5. **A percentile is a rank, not an average.** Rank only the rows that have a TTFT, count them per model with a window, and keep the row at $k = \lceil 0.95 n \rceil$, computed in integers as `(95 * n + 99) / 100`.
6. **An index matches the query's shape.** The canary query is `served_model = ? AND ts_ms >= ?`: one equality, one range, so the index is `(served_model, ts_ms)`.
7. **Durability and concurrency are connection settings.** WAL is stored in the file; the other three must run on every connection, which is why they live in one file that every connection runs first.
8. **A two-statement write must be one transaction, and idempotent.** `BEGIN IMMEDIATE` takes the write lock up front; the usage insert ignores a replayed `request_id`; the rollup upsert runs only when that insert changed a row.

### q1: totals, never NULL

```sql
SELECT COUNT(*)                                                             AS requests,
       COALESCE(SUM(status >= 400 OR error_code IS NOT NULL), 0)            AS errors,
       COALESCE(SUM(prompt_tokens), 0)                                      AS prompt_tokens,
       COALESCE(SUM(completion_tokens), 0)                                  AS completion_tokens
FROM usage
WHERE tenant = :tenant AND ts_ms >= :since AND ts_ms < :until;   -- half-open window
```

`COUNT(*)` is already 0 over no rows; the three `SUM`s are not, hence `COALESCE` on each. On the hand-worked rows for acme it gives `(5, 2, 40, 56)`.

### q2: the same per model

Grouped, so every output group is non-empty and `SUM` cannot be NULL:

```sql
SELECT model,
       COUNT(*)                                        AS requests,
       SUM(status >= 400 OR error_code IS NOT NULL)    AS errors,
       SUM(prompt_tokens)                              AS prompt_tokens,
       SUM(completion_tokens)                          AS completion_tokens
FROM usage
WHERE tenant = :tenant AND ts_ms >= :since AND ts_ms < :until
GROUP BY model
ORDER BY model;
```

### q3: a stable leaderboard across tenants

```sql
SELECT key_id, tenant, SUM(prompt_tokens + completion_tokens) AS tokens
FROM usage
WHERE ts_ms >= :since AND ts_ms < :until           -- all tenants: no tenant filter
GROUP BY key_id, tenant                            -- tenant in the key so it can be selected
ORDER BY tokens DESC, key_id ASC                   -- the tie-break makes the order total
LIMIT :n;
```

### q4: hour buckets

The bucket expression is computed once, named, and grouped by name:

```sql
SELECT ts_ms - ts_ms % 3600000 AS hour_ms,         -- start of the hour
       COUNT(*)                AS requests,
       SUM(completion_tokens)  AS completion_tokens
FROM usage
WHERE tenant = :tenant AND ts_ms >= :since AND ts_ms < :until
GROUP BY hour_ms                                   -- hours with no rows simply do not appear
ORDER BY hour_ms;
```

A row at exactly 11:00:00.000 has `ts_ms % 3600000 = 0`, so it is in the 11:00 bucket, and the half-open window counts it in `[11:00, 12:00)` only.

### q5: nearest-rank p95

```sql
WITH ranked AS (
    SELECT model, ttft_ms,
           ROW_NUMBER() OVER (PARTITION BY model ORDER BY ttft_ms) AS rn,   -- 1..n per model
           COUNT(*)     OVER (PARTITION BY model)                  AS n     -- n on every row
    FROM usage
    WHERE ttft_ms IS NOT NULL AND ts_ms >= :since AND ts_ms < :until      -- NULLs out BEFORE ranking
)
SELECT model, n, ttft_ms AS p95_ttft_ms
FROM ranked
WHERE rn = (95 * n + 99) / 100      -- integer division: ceil(0.95 n)
ORDER BY model;
```

Filtering NULLs in the `WHERE` (before the window) is what makes `n` and `rn` count only real TTFTs; `COUNT(*)` here is safe because the NULL rows are already gone. A model with no TTFT at all has no rows in `ranked` and so no output row, like `tiny` in the hand example.

### The index

```sql
-- equality column first, then the range column, so the served-model query
-- searches one contiguous slice of the index.
CREATE INDEX IF NOT EXISTS usage_served_ts ON usage (served_model, ts_ms);
```

`IF NOT EXISTS` makes the file safe to apply on every start, like the contract's own indexes, which it leaves in place.

### The pragmas

```sql
PRAGMA journal_mode = WAL;     -- stored in the file: every later connection sees WAL
PRAGMA synchronous = NORMAL;   -- per connection: no fsync per commit; WAL keeps commits across a process crash
PRAGMA busy_timeout = 5000;    -- per connection: wait up to 5 s for the single writer lock
PRAGMA foreign_keys = ON;      -- per connection: enforce REFERENCES
```

### The rollup table

Constraints carry the invariants, so a bad write fails instead of being stored:

```sql
CREATE TABLE IF NOT EXISTS usage_hourly (
    tenant            TEXT    NOT NULL,
    hour_ms           INTEGER NOT NULL CHECK (hour_ms % 3600000 = 0),   -- on an hour boundary
    requests          INTEGER NOT NULL DEFAULT 0 CHECK (requests >= 0),
    prompt_tokens     INTEGER NOT NULL DEFAULT 0 CHECK (prompt_tokens >= 0),
    completion_tokens INTEGER NOT NULL DEFAULT 0 CHECK (completion_tokens >= 0),
    PRIMARY KEY (tenant, hour_ms)                                       -- one row per tenant and hour
);
```

### Recording one request

```sql
BEGIN IMMEDIATE;                                   -- take the write lock now; queue on busy_timeout
INSERT INTO usage (request_id, ts_ms, tenant, /* ...every usage column... */ trace_id)
VALUES (:request_id, :ts_ms, :tenant, /* ...one named parameter per column... */ :trace_id)
ON CONFLICT (request_id) DO NOTHING;               -- a replay inserts nothing
-- changes() is the number of rows the previous statement inserted: 0 when
-- the request id was already recorded, so a replay adds nothing.
INSERT INTO usage_hourly (tenant, hour_ms, requests, prompt_tokens, completion_tokens)
SELECT :tenant, :ts_ms - :ts_ms % 3600000, 1, :prompt_tokens, :completion_tokens
WHERE changes() = 1                                -- zero rows to insert on a replay
ON CONFLICT (tenant, hour_ms) DO UPDATE SET
    requests          = requests + 1,              -- the existing row's value, plus this request
    prompt_tokens     = prompt_tokens + excluded.prompt_tokens,
    completion_tokens = completion_tokens + excluded.completion_tokens;
COMMIT;
```

The rollup insert is `INSERT ... SELECT ... WHERE` rather than `INSERT ... VALUES` so that it can be conditional: with `changes() = 0` the `SELECT` yields no row and nothing is inserted or updated. (The `WHERE` also resolves a parsing ambiguity: SQLite requires one between `INSERT ... SELECT` and `ON CONFLICT`.) If either insert fails, `COMMIT` is never reached; the caller rolls back (or closes the connection, which rolls back), so the usage row never exists without its rollup.

The full solution is the nine files in [`primers/lang.11/`](../../../../course/solutions/primers/lang.11/).

### Why this design

q1, q2, and q4 filter on `tenant` and `ts_ms`, which the contract's `usage_tenant_ts` index serves as a search. q3 has only the time range, which no contract index leads with, so it scans. q5 sorts each model's TTFTs, $O(n \log n)$. The new `usage_served_ts` index turns the canary panel from a full scan into a search, at the cost of a little space and a slightly slower insert.

Nearest rank beats interpolation here. `percentile_cont` interpolates between neighbours and is not in SQLite; nearest rank always returns an observed value, is exact, and is what an SLO "95% of requests under X" means.

Maintaining the rollup in the write path costs one extra statement per request and makes hourly panels read a handful of rows instead of aggregating the raw table. The alternative, recomputing from `usage` on every read, is simpler but grows with history.

Idempotence comes from the key (`ON CONFLICT DO NOTHING` plus `changes()`) instead of "check, then insert". The check-then-insert version has a race between two writers and costs a read; the primary key already decides.

Finally, WAL with `synchronous = NORMAL` accepts that a power cut may lose the last few commits, in exchange for no `fsync` per request. A process crash loses nothing committed.

## What can go wrong

The solution is checked by running the SQL with Python's built-in `sqlite3` against 240 fixture rows and comparing every answer with one computed in plain Python. These are the cases:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| Every file present | unit | the nine files exist | each later case names its file |
| The hand example | unit | every query on the six hand-worked rows | each definition means what the hand example says |
| Read-only selects with named parameters | unit | one statement each, read-only (an authorizer refuses writes), parameters only from `:tenant :since :until :n`, no tenant or time pasted in | the admin API serves them from URLs; the agent's tool runs queries read-only |
| q1 on the fixture | unit | 4 windows x 4 tenants (one with a quote, one with no rows) | the admin API's ungrouped answer |
| q2 on the fixture | unit | grouped, sorted, conditional error count | `group_by=model` |
| q3 on the fixture | boundary | ties broken by `key_id`; `n` 1, 3, 5 | a stable leaderboard |
| q4 on the fixture | boundary | hour buckets; rows at exactly 11:00:00.000 and one ms before | hourly panels add up |
| q5 on the fixture | unit | nearest rank, NULLs excluded | the TTFT SLO |
| The index serves the served-model query | unit | the plan shows `SEARCH usage USING ... (served_model=? AND ts_ms>?)`; contract indexes kept | canary dashboards stay fast |
| Pragmas set WAL and the connection settings | unit | WAL seen by a second connection; NORMAL, a busy timeout, foreign keys on the first | the gateway's data source name sets the same |
| The rollup schema | unit | applying twice works; key `(tenant, hour_ms)`; duplicate and half-hour rows refused | constraints catch bad writes |
| Record rolls up every row | unit | 240 records give 240 usage rows and the right rollup; no open transaction left | the per-request write path |
| A replayed record is a no-op | boundary | 25 replays change nothing | retried writes never double count |
| Record is atomic | fault | a failing rollup leaves no usage row behind | crash safety of a two-statement write |

Most wrong answers come from one of ten pitfalls, each with a symptom and the line of the solution that prevents it:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| `SUM(x)` as a total | a tenant with no requests has total NULL, and the CLI prints `None` | `COALESCE(SUM(x), 0)` in q1 |
| `error_code != NULL`, or `status >= 400` alone | no errors at all, or streams that failed mid-way uncounted | `SUM(status >= 400 OR error_code IS NOT NULL)` |
| `COUNT(*)` or `AVG` in the percentile | refusals with NULL TTFT shift the rank; an average is not a p95 | NULLs filtered before the window; the row at rank `(95 * n + 99) / 100` |
| `ORDER BY tokens DESC` with no tie-break | two keys with equal totals come back in either order | `ORDER BY tokens DESC, key_id ASC` |
| upserting the rollup without checking `changes()` | a retried write counts the request twice | `WHERE changes() = 1` on the rollup insert |
| two statements without `BEGIN` ... `COMMIT` | a failed rollup leaves the usage row behind | `BEGIN IMMEDIATE` ... `COMMIT` around both inserts |
| index `(ts_ms, served_model)` | the plan uses only the time bound and walks every model's rows | `usage_served_ts ON usage (served_model, ts_ms)` |
| `synchronous` set on one connection and assumed everywhere | the other pooled connections run with the default | every connection runs `pragmas.sql` first |
| `ts_ms <= :until` | a request at the boundary is in two hourly windows | every window is `ts_ms >= :since AND ts_ms < :until` |
| pasting a tenant into the SQL text | `o'brien` breaks the query; `x' OR '1'='1` reads everyone | only named parameters `:tenant :since :until :n` |

> **Warning:** `error_code != NULL` is the bug that passes every casual test. It is NULL for every row, so the error count is simply zero, and zero errors looks like good news. Always compare with NULL using `IS` or `IS NOT`.

## Where this leads

The shell, the `sqlite3` CLI, and exit codes came from [Shell, git, make, processes](02-shell-git-make.md).

In *AI Platform Engineering*, [The usage ledger and admin API](../../ai-platform-engineering/12-gateway/07-usage-ledger-and-admin-api.md) writes `usage` from Go through `modernc.org/sqlite` with these pragmas and serves q1 and q2 as `GET /admin/v1/usage`, and [The tool gate, the SQL safety gate, and the prompt-injection suite](../../ai-platform-engineering/13-agent-sdk/04-tool-gate-and-injection.md) gives the agent a `query_usage` tool that runs read-only queries like these through a SQL safety gate. In *Systems and Architecture*, [A noisy neighbor](../../systems/05-incident-response-and-chaos/09-noisy-neighbor.md) reads per-tenant usage and TTFT, and [SLOs and multi-window burn-rate alerts](../../systems/04-observability/03-slos-and-burn-rate-alerts.md) defines the TTFT SLO the p95 query measures.

Each piece has a production equivalent:

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| SQLite in WAL mode | PostgreSQL, Litestream | many writers (MVCC); streaming the WAL to object storage for backups | [PostgreSQL MVCC](https://www.postgresql.org/docs/current/mvcc-intro.html) (free), [Litestream](https://litestream.io/how-it-works/) (free) |
| the hourly rollup | ClickHouse materialized views | rollups maintained by the database on every insert | [ClickHouse materialized views](https://clickhouse.com/docs/materialized-view) (free) |
| nearest-rank p95 | `percentile_cont`, t-digest | interpolated and streaming percentiles | [PostgreSQL aggregate functions](https://www.postgresql.org/docs/current/functions-aggregate.html) (free) |

### Further reading

- [SQLite: SQL as understood by SQLite](https://www.sqlite.org/lang.html) (free)
- [SQLite: write-ahead logging](https://www.sqlite.org/wal.html) (free)
- [SQLite: query planning](https://www.sqlite.org/queryplanner.html) (free)
- [Use The Index, Luke](https://use-the-index-luke.com/) (free)

## Key terms

- **constraint**: a rule (`PRIMARY KEY`, `NOT NULL`, `CHECK`, `DEFAULT`) the database enforces on every write.
- **pragma**: a SQLite setting, stored either in the file or only on the connection that ran it.
- **write-ahead logging (WAL)**: a journal mode in which commits append pages to a log, so readers and the single writer do not block each other.
- **aggregate**: a function such as `COUNT`, `SUM`, or `MAX` that turns the rows of a group into one value.
- **half-open window**: a time range `[since, until)` that includes its start and excludes its end.
- **window function**: a function computed per row over related rows without collapsing them, such as `ROW_NUMBER() OVER (...)`.
- **nearest-rank percentile**: the value at rank $\lceil p\,n / 100 \rceil$ of the sorted values.
- **named parameter**: a placeholder such as `:tenant` whose value is bound separately and is always data.
- **index**: a sorted B-tree copy of chosen columns that lets a query search instead of scan.
- **upsert**: an `INSERT ... ON CONFLICT ... DO UPDATE` that inserts a row or updates the existing one.

## Summary

- A table is a set of rows with typed columns; constraints (`PRIMARY KEY`, `CHECK`, `NOT NULL`) make the database refuse bad rows instead of storing them.
- `GROUP BY` turns rows into one row per group; `COUNT(*)` counts rows, `SUM` adds values and is NULL over no rows, so ungrouped totals use `COALESCE(SUM(x), 0)`. Windows are half-open, and a p95 is a nearest rank over non-NULL values.
- A transaction makes several statements one unit: either every change is committed or none is, and `ON CONFLICT DO NOTHING` plus `changes()` makes a replayed write a no-op.
- WAL mode lets readers and one writer work at once and keeps every committed transaction across a process crash; it is stored in the file, while `synchronous` and `busy_timeout` are per connection.
- An index is a sorted copy of some columns; put equality columns first and the range column last, and read `EXPLAIN QUERY PLAN` to see `SEARCH` instead of `SCAN`.
- Values always enter as named parameters, never pasted into the SQL text.

## Review questions

1. Why does q1 need `COALESCE` but q2 does not?
   <details><summary>Answer</summary>
   q1 has no `GROUP BY`, so it returns one row even when no row matches, and `SUM` over no rows is NULL. q2 groups, so an empty window gives no rows at all, and every group that exists has at least one row, where `SUM` is a number.
   </details>
2. Why is the NULL filter of q5 in the CTE's `WHERE` and not in the outer query?
   <details><summary>Answer</summary>
   Window functions run after `WHERE`. Filtering inside the CTE means `ROW_NUMBER()` and `COUNT(*)` see only real TTFTs, so $n$ and the ranks are right. Filtering outside would rank NULL rows too (NULLs sort first in SQLite), which shifts every rank and inflates $n$.
   </details>
3. Replay the same `record.sql` call twice. Which statements change what the second time?
   <details><summary>Answer</summary>
   The usage insert hits the primary key and does nothing, so `changes()` is 0. The rollup's `SELECT ... WHERE changes() = 1` yields no row, so the upsert inserts and updates nothing. Both tables are unchanged.
   </details>
4. Why `(served_model, ts_ms)` and not `(ts_ms, served_model)`?
   <details><summary>Answer</summary>
   The query fixes `served_model` by equality and bounds `ts_ms` by a range. With `served_model` first, all its rows are contiguous and sorted by time, so the plan is one seek plus a walk: `SEARCH ... (served_model=? AND ts_ms>?)`. With `ts_ms` first, only the time bound can use the index and the walk passes every model's rows in that range.
   </details>
5. Why does `pragmas.sql` have to be run on every connection, if WAL only needs to be set once?
   <details><summary>Answer</summary>
   `journal_mode = WAL` is persistent in the database file, but `synchronous`, `busy_timeout`, and `foreign_keys` belong to the connection that ran them. A pool (Go's `database/sql`) opens several connections; any one that skipped the file would run with the defaults: `FULL` sync, no busy wait (instant `SQLITE_BUSY`), and no foreign-key checks.
   </details>
6. A model has four TTFTs, 180, 240, 200, and 900 ms. What is its nearest-rank p95, and why?
   <details><summary>Answer</summary>
   Sorted, the values are 180, 200, 240, 900, and $n = 4$. The rank is $k = \lfloor (95 \cdot 4 + 99)/100 \rfloor = \lfloor 479/100 \rfloor = 4$, so the p95 is $v_4 = 900$: with only four samples, the p95 is the maximum.
   </details>
