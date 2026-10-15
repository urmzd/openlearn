<!-- ol:module lang.11 -->
# SQL and SQLite: schema, transactions, WAL mode, indexes, aggregates

## Overview

| | |
|---|---|
| **Module** | `lang.11` · practice · SQL · Pass 7 · 4 to 6 h |
| **You build** | `primers/lang.11/`: five queries over the usage ledger (`q1_tenant_totals.sql` to `q5_ttft_p95.sql`), an index (`indexes.sql`), connection settings (`pragmas.sql`), a rollup table (`rollup.sql`), and the transaction that records one request (`record.sql`) |
| **Contract** | the table your queries read: [`course/contracts/formats/usage.v1.sql`](../../../../course/contracts/formats/usage.v1.sql), the gateway's usage ledger |
| **Tests** | `course/tests/lang.11/check` runs your SQL with Python's built-in `sqlite3` against the 240 fixture rows of `course/fixtures/lang.11/usage_rows.jsonl` and compares every answer with one computed in plain Python (what each test checks: section 4) |
| **Needs** | reading: `lang.02` shell and processes ([primer](02-shell-git-make.md)) for the `sqlite3` command line |
| **Used by** | no call site (a primer): `gw.07` applies it next, writing the same table from Go and serving these queries through `GET /admin/v1/usage` |
| **Milestone** | MS-gateway |
| **Optional depth** | [SQLite: SQL as understood by SQLite](https://www.sqlite.org/lang.html) (free); [SQLite: write-ahead logging](https://www.sqlite.org/wal.html) (free); [SQLite: query planning](https://www.sqlite.org/queryplanner.html) (free); [Use The Index, Luke](https://use-the-index-luke.com/) (free) |

## Key Takeaways

- A table is a set of rows with typed columns; **constraints** (`PRIMARY KEY`, `CHECK`, `NOT NULL`) make the database refuse bad rows instead of storing them (`test_rollup_schema`).
- `GROUP BY` turns rows into one row per group; `COUNT(*)` counts rows, `SUM` adds values and is **NULL over no rows**, so totals use `COALESCE(SUM(x), 0)` (`test_q1_tenant_totals_on_the_fixture`).
- A **transaction** makes several statements one unit: either every change is committed or none is, and `ON CONFLICT DO NOTHING` plus `changes()` makes a replayed write a no-op (`test_record_is_atomic`, `test_record_replay_is_a_no_op`).
- **WAL mode** lets readers and one writer work at once and keeps every committed transaction across a process crash; it is stored in the file, while `synchronous` and `busy_timeout` are **per connection** (`test_pragmas_set_wal_and_connection_settings`).
- An index is a sorted copy of some columns; put **equality columns first and the range column last**, and read `EXPLAIN QUERY PLAN` to see `SEARCH` instead of `SCAN` (`test_index_serves_the_served_model_query`).

## How to work this chapter

```bash
ol start lang.11                         # records that you started; the exercise is primers/lang.11/
ol tests lang.11                         # read the test catalog first
mkdir -p primers/lang.11                 # write the nine files of section 4
sqlite3 /tmp/u.db < contracts/formats/usage.v1.sql      # try things by hand
sqlite3 /tmp/u.db ".param set :tenant acme" ".param set :since 0" ".param set :until 9999999999999" \
  ".read primers/lang.11/q1_tenant_totals.sql"
ol check lang.11                         # exit code is the verdict
```

---

## 1. Why now

In Pass 7 the gateway grows a memory. Until now it forgot every request the moment the response ended, so nobody could say how many tokens `acme` used yesterday, which key is burning through its budget, or whether the canary model is slower than the stable one. `gw.07` adds a usage ledger: one row per request in a SQLite database, written by the gateway and read through its admin API, by the agent's `query_usage` tool (`ag.04`), and by the noisy-neighbor drill (`ops.09`). Every one of those readers is a SQL query, and the ledger is only useful if its writes survive a crash and its reads are right at the edges: an empty window, a tie, a missing value. This primer teaches that SQL on the real table before you write the Go that fills it.

## 2. Principles

### 2.1 Tables, rows, and constraints

A **relational database** stores **tables**. A table has named **columns**, each with a type, and holds **rows**, one value per column. SQLite's types are `INTEGER`, `REAL`, `TEXT`, `BLOB`, and the absent value `NULL`. Open `contracts/formats/usage.v1.sql`: the `usage` table has one row per request, with `request_id TEXT NOT NULL PRIMARY KEY`, `ts_ms INTEGER` (the start time in Unix milliseconds), `tenant`, `key_id`, `model`, `status`, `error_code` (NULL on success), token counts, and `ttft_ms REAL` (NULL when no content byte was sent).

A **constraint** is a rule the database enforces on every write:

| Constraint | Meaning | In the ledger |
|---|---|---|
| `PRIMARY KEY` | the column (or tuple of columns) identifies one row; a second row with the same key is refused | `request_id`: one row per request |
| `NOT NULL` | the column always has a value | every column but `error_code` and `ttft_ms` |
| `CHECK (expr)` | `expr` must be true for every row | `cached_tokens <= prompt_tokens`, `stream IN (0, 1)` |
| `DEFAULT v` | the value when an `INSERT` leaves the column out | `api_version DEFAULT '1'` |

`CREATE TABLE IF NOT EXISTS` makes a schema file safe to apply on every start: the second time it does nothing. Your `rollup.sql` defines a second table, `usage_hourly`, with one row per tenant and hour: its primary key is the pair `(tenant, hour_ms)`, and a `CHECK (hour_ms % 3600000 = 0)` keeps every `hour_ms` on an hour boundary.

### 2.2 Connections, pragmas, and the journal

A program talks to a database through a **connection**. A **pragma** is a SQLite setting, `PRAGMA name = value`. Some are stored in the database file and apply to every connection; most belong to the one connection that ran them. That difference matters as soon as a program keeps a pool of connections (Go's `database/sql` does).

| Pragma | Scope | Value for the ledger | Why |
|---|---|---|---|
| `journal_mode` | the file | `WAL` | see below |
| `synchronous` | connection | `NORMAL` (1) | with WAL, a commit survives a process crash; only a power cut can lose the last commits, and no `fsync` per commit |
| `busy_timeout` | connection | at least 1000 ms | a writer that finds the database locked waits up to this long instead of failing with `SQLITE_BUSY` |
| `foreign_keys` | connection | `ON` | SQLite enforces `REFERENCES` only when this is on |

**The journal.** A database must survive a crash in the middle of a write. SQLite's default **rollback journal** copies each page it is about to change into `db-journal`, then changes the database in place; a crash is undone from the copy on the next open. While a writer works, readers wait. In **write-ahead logging** (WAL) the database file is not touched by a commit: the changed pages are **appended** to `db-wal`, followed by a commit record. Readers keep reading the old pages plus every committed frame, so readers never block the writer and the writer never blocks readers. On open after a crash, SQLite replays the WAL up to the last commit record and ignores a torn tail. Every so often a **checkpoint** copies the WAL back into the database file. There is still only **one writer at a time**, which is what `busy_timeout` is for.

### 2.3 Reading: SELECT, WHERE, aggregates, GROUP BY

A query is evaluated in this order: `FROM` picks the table, `WHERE` keeps the rows whose condition is true, `GROUP BY` collects rows with equal values of the grouping expressions, the `SELECT` list computes one output row per group (or per row without `GROUP BY`), `ORDER BY` sorts, `LIMIT` keeps the first rows.

An **aggregate** turns the rows of a group into one value:

| Symbol | Meaning |
|---|---|
| $G$ | the rows of one group |
| $\lvert G \rvert$ | the number of rows in $G$ |
| $x_r$ | the value of column $x$ in row $r$, possibly NULL |

| Aggregate | Value | Over no rows |
|---|---|---|
| `COUNT(*)` | $\lvert G \rvert$ | 0 |
| `COUNT(x)` | the number of rows where $x_r$ is not NULL | 0 |
| `SUM(x)` | $\sum_{r \in G,\ x_r \ne \text{NULL}} x_r$ | **NULL** |
| `AVG(x)`, `MIN(x)`, `MAX(x)` | as named, ignoring NULLs | NULL |

The NULL in the last column is the classic surprise: the total tokens of a tenant with no requests is `NULL`, not 0. `COALESCE(a, b)` returns `a` unless it is NULL, so `COALESCE(SUM(prompt_tokens), 0)` is the total you want. Without `GROUP BY` an aggregate query returns exactly one row even when no row matched; with `GROUP BY` an empty input gives no rows.

A comparison is 1 or 0 in SQLite, so `SUM(status >= 400 OR error_code IS NOT NULL)` counts errors: refused requests (status 400 and up) and streams that failed after their first byte (status 200 with an error code). Note `IS NOT NULL`: `error_code != NULL` is NULL for every row, never true.

### 2.4 Windows of time and buckets

Times are integers (Unix milliseconds), so a window is two comparisons. Use **half-open** windows, `ts_ms >= :since AND ts_ms < :until`: then the hours `[10:00, 11:00)` and `[11:00, 12:00)` add up to `[10:00, 12:00)` and a request at exactly 11:00:00.000 is counted once. A bucket is an expression in `GROUP BY`: the start of the hour of `ts_ms` is `ts_ms - ts_ms % 3600000` (`%` is the remainder), because an hour is $3.6 \times 10^6$ ms.

### 2.5 Window functions and a percentile

A **window function** computes a value for each row from a set of related rows, without collapsing them: `ROW_NUMBER() OVER (PARTITION BY model ORDER BY ttft_ms)` numbers each model's rows 1, 2, 3, ... in TTFT order, and `COUNT(*) OVER (PARTITION BY model)` puts the model's row count on every row.

The **nearest-rank percentile**: for $n$ values sorted ascending $v_1 \le \dots \le v_n$, the $p$-th percentile is $v_k$ with $k = \lceil p\,n / 100 \rceil$. For $p = 95$, in integers, $k = \lfloor (95 n + 99) / 100 \rfloor$ (adding 99 before the floor division rounds up).

| Symbol | Meaning |
|---|---|
| $n$ | the number of non-NULL TTFTs of one model |
| $v_i$ | the $i$-th smallest TTFT |
| $k$ | the rank of the percentile |

So `q5_ttft_p95.sql` ranks rows that **have** a TTFT (refusals and embeddings have NULL) and keeps the row where `rn = (95 * n + 99) / 100`. A `WITH name AS (SELECT ...)` clause (a common table expression) names that ranked set so the outer query can filter it.

### 2.6 Parameters, never pasted values

A query's values come from outside: the admin API's URL, a CLI flag. Pasting them into the SQL text is **SQL injection**: `tenant = 'x' OR '1'='1'` reads every tenant, and a quote in a real name (`o'brien` is in the fixture) breaks the query. A **named parameter** (`:tenant`) is a placeholder; the value is bound separately and is always data. Your query files take `:tenant`, `:since`, `:until`, and `:n` and nothing else. A column name cannot be a parameter, which is why `gw.07` picks `GROUP BY` columns from a fixed list.

### 2.7 Indexes and the query plan

Without an index, a query with `WHERE served_model = ?` reads every row: a **scan**. An **index** is a B-tree holding chosen columns of every row in sorted order, plus a pointer to the row. With an index on `(served_model, ts_ms)`, all rows for one served model sit next to each other, sorted by time inside, so `served_model = ? AND ts_ms >= ?` is one jump to the first match and a walk to the end of the slice: a **search**. Order matters: an index on `(ts_ms, served_model)` sorts by time first, so the rows of one model are scattered and only the time bound can use it. The rule is **equality columns first, then one range column**. `EXPLAIN QUERY PLAN <query>` shows the plan: `SCAN usage` or `SEARCH usage USING INDEX usage_served_ts (served_model=? AND ts_ms>?)`. An index costs space and slows every insert a little, so the contract indexes only what its readers filter on.

### 2.8 Transactions, upserts, and replays

A **transaction** groups statements: `BEGIN` starts one, `COMMIT` makes every change visible and durable at once, `ROLLBACK` discards them. Outside an explicit transaction, every statement is its own transaction (autocommit). An error inside a transaction fails only that statement; the program decides whether to `ROLLBACK`, and a connection that closes with a transaction open rolls it back. `BEGIN IMMEDIATE` takes the write lock at once rather than at the first write, so two writers queue on `busy_timeout` instead of failing halfway.

An **upsert** is `INSERT ... ON CONFLICT (key) DO UPDATE SET ...`: insert the row, or, when the key exists, update the existing row, where `excluded.col` is the value that was about to be inserted. `ON CONFLICT DO NOTHING` ignores the row instead. `changes()` is the number of rows the previous statement changed, so after `INSERT ... ON CONFLICT (request_id) DO NOTHING` it is 1 for a new request and 0 for a replay. `record.sql` uses all three: in one transaction, insert the usage row (ignored on a replay), then upsert the hourly rollup **only when** `changes() = 1`.

## 3. Worked example by hand

Six requests on 2026-10-01 (12:00 is `ts_ms` 1790856000000), all on `/v1/chat/completions`:

| `request_id` | time | tenant | key | model | status | `error_code` | prompt | completion | `ttft_ms` |
|---|---|---|---|---|---|---|---|---|---|
| r1 | 12:05 | acme | `acmekeyaaaaa` | smol | 200 | NULL | 12 | 30 | 180 |
| r2 | 12:20 | acme | `acmekeybbbbb` | smol | 200 | NULL | 20 | 10 | 240 |
| r3 | 12:40 | acme | `acmekeyaaaaa` | tiny | 429 | `rate_limit_exceeded` | 0 | 0 | NULL |
| r4 | 13:10 | acme | `acmekeyaaaaa` | smol | 200 | NULL | 8 | 16 | 200 |
| r5 | 13:15 | globex | `globexkeyccc` | smol | 200 | NULL | 7 | 5 | 900 |
| r6 | 13:50 | acme | `acmekeybbbbb` | smol | 503 | `no_capacity` | 0 | 0 | NULL |

The window is `[12:00, 14:00)`, so every row is in it.

1. **q1, acme's totals.** `WHERE tenant = 'acme'` keeps r1, r2, r3, r4, r6. Requests $5$; errors $2$ (r3 and r6); prompt $12 + 20 + 0 + 8 + 0 = 40$; completion $30 + 10 + 0 + 16 + 0 = 56$. Answer `(5, 2, 40, 56)`.
2. **q2, acme by model**, sorted by model: smol is r1, r2, r4, r6, so `(smol, 4, 1, 40, 56)`; tiny is r3, so `(tiny, 1, 1, 0, 0)`.
3. **q3, the top 2 keys by tokens** (prompt plus completion): `acmekeyaaaaa` has $42 + 0 + 24 = 66$, `acmekeybbbbb` $30 + 0 = 30$, `globexkeyccc` $12$. Answer `(acmekeyaaaaa, acme, 66), (acmekeybbbbb, acme, 30)`.
4. **q4, acme per hour.** $12{:}05 \to 1790856000000$, the 12:00 bucket, holds r1, r2, r3: 3 requests, $30 + 10 + 0 = 40$ completion tokens. The 13:00 bucket ($1790859600000$) holds r4, r6: 2 requests, 16 tokens.
5. **q5, p95 TTFT per model.** smol's non-NULL TTFTs are 180, 240, 200, 900, so $n = 4$ and sorted they are 180, 200, 240, 900. $k = \lfloor (95 \cdot 4 + 99)/100 \rfloor = \lfloor 479/100 \rfloor = 4$, so the p95 is $v_4 = 900$: with four samples the p95 is the maximum. tiny has no TTFT at all, so it has no row. Answer `(smol, 4, 900.0)`.

`test_hand_example` runs your five queries on exactly these rows and expects these answers.

## 4. The artifact and its check

Nine files in `primers/lang.11/`. Each query file holds exactly one `SELECT`, takes its values as named parameters, and returns its columns in the order given:

| File | Holds | Columns, order |
|---|---|---|
| `q1_tenant_totals.sql` | `:tenant`'s totals in `[:since, :until)`; zeros, not NULL, when nothing matches | `requests, errors, prompt_tokens, completion_tokens` |
| `q2_by_model.sql` | `:tenant`'s usage per model in the window | `model, requests, errors, prompt_tokens, completion_tokens`, by model |
| `q3_top_keys.sql` | the `:n` keys with the most tokens (prompt plus completion) in the window, all tenants | `key_id, tenant, tokens`, by tokens descending, then `key_id` ascending |
| `q4_hourly.sql` | `:tenant`'s requests and completion tokens per hour, hours with rows only | `hour_ms, requests, completion_tokens`, by hour |
| `q5_ttft_p95.sql` | the nearest-rank p95 of `ttft_ms` per model in the window, NULLs left out | `model, n, p95_ttft_ms`, by model |
| `indexes.sql` | `CREATE INDEX` statements only, so that the query below searches an index | |
| `pragmas.sql` | the four pragmas of section 2.2 | |
| `rollup.sql` | `CREATE TABLE IF NOT EXISTS usage_hourly (tenant, hour_ms, requests, prompt_tokens, completion_tokens)`, keyed by `(tenant, hour_ms)`, hour-aligned by a `CHECK` | |
| `record.sql` | one request: the `usage` insert and the `usage_hourly` upsert in one transaction, a replayed `request_id` changing nothing; parameters are the usage columns (`:request_id`, `:ts_ms`, ..., `:trace_id`) | |

The query `indexes.sql` must serve (a per-canary panel):

```sql
SELECT worker_id, COUNT(*) AS requests, SUM(completion_tokens) AS completion_tokens
FROM usage WHERE served_model = :served_model AND ts_ms >= :since GROUP BY worker_id
```

The check splits `record.sql` at each line that ends a statement and runs the statements in order on one connection in autocommit mode, binding the row's columns by name.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_files_present` | unit | the nine files exist | the check names what is missing |
| `test_hand_example` | unit | section 3, every query | you and the check agree on each definition |
| `test_queries_are_read_only_selects_with_named_parameters` | unit | one statement each, read-only (an authorizer refuses writes), parameters only from `:tenant :since :until :n`, no tenant or time pasted in | `gw.07` serves them from URLs; `ag.04` runs queries read-only |
| `test_q1_tenant_totals_on_the_fixture` | unit | 4 windows x 4 tenants (one with a quote, one with no rows) | the admin API's ungrouped answer |
| `test_q2_by_model_on_the_fixture` | unit | grouped, sorted, conditional error count | `group_by=model` |
| `test_q3_top_keys_on_the_fixture` | boundary | ties broken by `key_id`; `n` 1, 3, 5 | a stable leaderboard |
| `test_q4_hourly_on_the_fixture` | boundary | hour buckets; rows at exactly 11:00:00.000 and one ms before | hourly panels add up |
| `test_q5_ttft_p95_on_the_fixture` | unit | nearest rank, NULLs excluded | the TTFT SLO of `obs.03` |
| `test_index_serves_the_served_model_query` | unit | the plan shows `SEARCH usage USING ... (served_model=? AND ts_ms>?)`; contract indexes kept | canary dashboards stay fast |
| `test_pragmas_set_wal_and_connection_settings` | unit | WAL seen by a second connection; NORMAL, a busy timeout, foreign keys on the first | `gw.07`'s data source name sets the same |
| `test_rollup_schema` | unit | applying twice works; key `(tenant, hour_ms)`; duplicate and half-hour rows refused | constraints catch bad writes |
| `test_record_rolls_up_every_row` | unit | 240 records give 240 usage rows and the right rollup; no open transaction left | the per-request write path |
| `test_record_replay_is_a_no_op` | boundary | 25 replays change nothing | retried writes never double count |
| `test_record_is_atomic` | fault | a failing rollup leaves no usage row behind | crash safety of a two-statement write |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. `SUM(x)` as a total | a tenant with no requests has total NULL, and the CLI prints `None` | `test_q1_tenant_totals_on_the_fixture` |
| 2. `error_code != NULL`, or `status >= 400` alone | no errors at all, or streams that failed mid-way uncounted | `test_hand_example`, `test_q2_by_model_on_the_fixture` |
| 3. `COUNT(*)` or `AVG` in the percentile | refusals with NULL TTFT shift the rank; an average is not a p95 | `test_q5_ttft_p95_on_the_fixture` |
| 4. `ORDER BY tokens DESC` with no tie-break | two keys with equal totals come back in either order | `test_q3_top_keys_on_the_fixture` |
| 5. upserting the rollup without checking `changes()` | a retried write counts the request twice | `test_record_replay_is_a_no_op` |
| 6. two statements without `BEGIN` ... `COMMIT` | a failed rollup leaves the usage row behind | `test_record_is_atomic` |
| 7. index `(ts_ms, served_model)` | the plan uses only the time bound and walks every model's rows | `test_index_serves_the_served_model_query` |
| 8. `synchronous` set on one connection and assumed everywhere | the other pooled connections run with the default | `test_pragmas_set_wal_and_connection_settings` |
| 9. `ts_ms <= :until` | a request at the boundary is in two hourly windows | `test_q4_hourly_on_the_fixture` |
| 10. pasting a tenant into the SQL text | `o'brien` breaks the query; `x' OR '1'='1` reads everyone | `test_queries_are_read_only_selects_with_named_parameters` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `lang.02` | the shell, the `sqlite3` CLI, exit codes |
| Forward | `gw.07` | writes `usage` from Go through `modernc.org/sqlite` with these pragmas, and serves q1 and q2 as `GET /admin/v1/usage` |
| Forward | `ag.04` | the agent's `query_usage` tool runs read-only queries like these through a SQL safety gate |
| Forward | `ops.09` | the noisy-neighbor drill reads per-tenant usage and TTFT |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| SQLite in WAL mode | PostgreSQL, Litestream | many writers (MVCC); streaming the WAL to object storage for backups | [PostgreSQL MVCC](https://www.postgresql.org/docs/current/mvcc-intro.html) (free), [Litestream](https://litestream.io/how-it-works/) (free) |
| the hourly rollup | ClickHouse materialized views | rollups maintained by the database on every insert | [ClickHouse materialized views](https://clickhouse.com/docs/materialized-view) (free) |
| nearest-rank p95 | `percentile_cont`, t-digest | interpolated and streaming percentiles | [PostgreSQL aggregate functions](https://www.postgresql.org/docs/current/functions-aggregate.html) (free) |
