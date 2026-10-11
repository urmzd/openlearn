"""lang.11 artifact check: your SQL over the usage ledger, in primers/lang.11/.

Run by `ol check lang.11` in your repo. Everything runs in Python's sqlite3
module against temporary database files built from your vendored contract
contracts/formats/usage.v1.sql and the fixture rows
course/fixtures/lang.11/usage_rows.jsonl (240 synthetic requests over six
hours). Every expected answer is computed here in plain Python from the same
rows, never by a second copy of the SQL.

  queries     q1 to q5 are read-only SELECTs with named parameters whose rows
              equal the Python answers for several windows and tenants
  indexes     indexes.sql makes the served-model query SEARCH an index
  pragmas     pragmas.sql: WAL (persistent), synchronous NORMAL, a busy timeout,
              foreign keys
  schema      rollup.sql: usage_hourly keyed by (tenant, hour_ms), hours aligned
  record      record.sql: inserts a request and its rollup in one transaction,
              ignores a replayed request id, and leaves nothing when the
              rollup write fails
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.practice import Ctx, Fail, run  # noqa: E402

DIR = "primers/lang.11"
QUERIES = ["q1_tenant_totals", "q2_by_model", "q3_top_keys", "q4_hourly", "q5_ttft_p95"]
FILES = [f"{q}.sql" for q in QUERIES] + [
    "indexes.sql",
    "pragmas.sql",
    "rollup.sql",
    "record.sql",
]
HOUR = 3_600_000
T0 = 1790848800000  # 2026-10-01T10:00:00Z, the fixture's first hour
HAND_T0 = 1790856000000  # 2026-10-01T12:00:00Z, the worked example's first hour
PARAMS = {":since", ":until", ":tenant", ":n"}
SERVED_QUERY = (
    "SELECT worker_id, COUNT(*) AS requests, SUM(completion_tokens) AS completion_tokens "
    "FROM usage WHERE served_model = :served_model AND ts_ms >= :since GROUP BY worker_id"
)
COLUMNS = [
    "request_id",
    "ts_ms",
    "tenant",
    "key_id",
    "model",
    "served_model",
    "route",
    "api_version",
    "status",
    "error_code",
    "stream",
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "ttft_ms",
    "e2e_ms",
    "cache_hit",
    "worker_id",
    "trace_id",
]


# -- helpers --------------------------------------------------------------------


def course_tree() -> Path:
    return Path(os.environ.get("OL_COURSE_TREE") or Path(__file__).resolve().parents[2])


def contract_sql(c: Ctx) -> str:
    for p in (
        c.path("contracts/formats/usage.v1.sql"),
        course_tree() / "contracts/formats/usage.v1.sql",
    ):
        if p.is_file():
            return p.read_text()
    raise Fail("contracts/formats/usage.v1.sql is missing: run `ol contracts sync`")


def fixture_rows() -> list[dict]:
    base = Path(os.environ.get("TINYLLM_FIXTURES") or course_tree() / "fixtures")
    p = base / "lang.11" / "usage_rows.jsonl"
    if not p.is_file():
        raise Fail(f"fixture {p} is missing")
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def sql_file(c: Ctx, name: str) -> str:
    text = c.require_file(f"{DIR}/{name}").read_text()
    if not text.strip():
        raise Fail(f"{DIR}/{name} is empty")
    return text


def fresh_db(c: Ctx, rows: list[dict] | None = None) -> tuple[sqlite3.Connection, Path]:
    """A new database file with the contract schema and `rows`; autocommit mode."""
    d = Path(tempfile.mkdtemp(prefix="ol-lang11-"))
    c.cleanups.append(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
    path = d / "usage.db"
    conn = sqlite3.connect(path, isolation_level=None)
    c.cleanups.append(conn.close)
    conn.executescript(contract_sql(c))
    if rows:
        conn.execute("BEGIN")
        conn.executemany(
            f"INSERT INTO usage ({', '.join(COLUMNS)}) VALUES ({', '.join(':' + k for k in COLUMNS)})",
            rows,
        )
        conn.execute("COMMIT")
    return conn, path


def statements(text: str) -> list[str]:
    """Split a script into complete statements (one statement per line end)."""
    out, buf = [], ""
    for line in text.splitlines(keepends=True):
        buf += line
        if sqlite3.complete_statement(buf):
            out.append(buf.strip())
            buf = ""
    rest = "\n".join(
        x for x in buf.splitlines() if not x.strip().startswith("--")
    ).strip()
    if rest:
        raise Fail(
            f"an incomplete statement at the end: {rest[:60]!r} (a missing semicolon?)"
        )
    return out


def read_only(conn: sqlite3.Connection) -> None:
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        getattr(sqlite3, "SQLITE_RECURSIVE", 33),
    }
    conn.set_authorizer(
        lambda action, *_: (
            sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY
        )
    )


def run_query(c: Ctx, conn: sqlite3.Connection, q: str, params: dict) -> list[tuple]:
    text = sql_file(c, f"{q}.sql")
    stmts = statements(text)
    if len(stmts) != 1:
        raise Fail(
            f"{DIR}/{q}.sql holds {len(stmts)} statements; write exactly one SELECT"
        )
    read_only(conn)
    try:
        return [tuple(r) for r in conn.execute(stmts[0], params).fetchall()]
    except sqlite3.DatabaseError as e:
        raise Fail(f"{DIR}/{q}.sql with {params}: {e}") from None
    finally:
        conn.set_authorizer(None)


def is_error(r: dict) -> bool:
    return r["status"] >= 400 or r["error_code"] is not None


# -- the Python answers (section 4 of the chapter, in code) --------------------


def py_q1(rows, p):
    sel = [
        r
        for r in rows
        if r["tenant"] == p["tenant"] and p["since"] <= r["ts_ms"] < p["until"]
    ]
    return [
        (
            len(sel),
            sum(is_error(r) for r in sel),
            sum(r["prompt_tokens"] for r in sel),
            sum(r["completion_tokens"] for r in sel),
        )
    ]


def py_q2(rows, p):
    out = []
    sel = [
        r
        for r in rows
        if r["tenant"] == p["tenant"] and p["since"] <= r["ts_ms"] < p["until"]
    ]
    for m in sorted({r["model"] for r in sel}):
        g = [r for r in sel if r["model"] == m]
        out.append(
            (
                m,
                len(g),
                sum(is_error(r) for r in g),
                sum(r["prompt_tokens"] for r in g),
                sum(r["completion_tokens"] for r in g),
            )
        )
    return out


def py_q3(rows, p):
    tot: dict[tuple, int] = {}
    for r in rows:
        if p["since"] <= r["ts_ms"] < p["until"]:
            k = (r["key_id"], r["tenant"])
            tot[k] = tot.get(k, 0) + r["prompt_tokens"] + r["completion_tokens"]
    ranked = sorted(tot.items(), key=lambda kv: (-kv[1], kv[0][0]))
    return [(k[0], k[1], v) for k, v in ranked[: p["n"]]]


def py_q4(rows, p):
    b: dict[int, list] = {}
    for r in rows:
        if r["tenant"] == p["tenant"] and p["since"] <= r["ts_ms"] < p["until"]:
            h = r["ts_ms"] - r["ts_ms"] % HOUR
            b.setdefault(h, []).append(r)
    return [
        (h, len(g), sum(r["completion_tokens"] for r in g))
        for h, g in sorted(b.items())
    ]


def py_q5(rows, p):
    out = []
    sel = [
        r
        for r in rows
        if r["ttft_ms"] is not None and p["since"] <= r["ts_ms"] < p["until"]
    ]
    for m in sorted({r["model"] for r in sel}):
        v = sorted(r["ttft_ms"] for r in sel if r["model"] == m)
        n = len(v)
        rank = (95 * n + 99) // 100  # ceil(0.95 n) in integers
        out.append((m, n, v[rank - 1]))
    return out


PY = {
    "q1_tenant_totals": py_q1,
    "q2_by_model": py_q2,
    "q3_top_keys": py_q3,
    "q4_hourly": py_q4,
    "q5_ttft_p95": py_q5,
}

WINDOWS = [
    (T0, T0 + 6 * HOUR),
    (T0 + HOUR, T0 + 3 * HOUR),
    (T0 + 3 * HOUR - 1, T0 + 3 * HOUR),
    (T0 + 7 * HOUR, T0 + 8 * HOUR),
]
TENANTS = ["acme", "o'brien", "initech", "nobody"]


def cases(q: str):
    for since, until in WINDOWS:
        if q in ("q1_tenant_totals", "q2_by_model", "q4_hourly"):
            for t in TENANTS:
                yield {"tenant": t, "since": since, "until": until}
        elif q == "q3_top_keys":
            for n in (1, 3, 5):
                yield {"n": n, "since": since, "until": until}
        else:
            yield {"since": since, "until": until}


def check_query(c: Ctx, q: str) -> None:
    rows = fixture_rows()
    if "fixture" not in c.cache:
        c.cache["fixture"] = fresh_db(c, rows)[0]
    conn = c.cache["fixture"]
    for p in cases(q):
        got = run_query(c, conn, q, p)
        want = PY[q](rows, p)
        if got != want:
            shown = {k: v for k, v in p.items()}
            raise Fail(
                f"{DIR}/{q}.sql with {shown}:\n  got  {got[:6]}\n  want {want[:6]}"
            )


# -- tests ----------------------------------------------------------------------


def test_files_present(c: Ctx) -> None:
    # WHY: the check and the chapter agree on one file per exercise; a missing
    #      file is reported by name, not as a confusing failure later.
    # KIND: unit
    # CHAPTER: lang.11 section 4, The interface
    missing = [f for f in FILES if not c.path(f"{DIR}/{f}").is_file()]
    if missing:
        raise Fail(
            f"{DIR}/ lacks {', '.join(missing)} (chapter section 4 lists what each holds)"
        )


HAND = [
    # request_id, minutes after 12:00, tenant, key, model, status, code, prompt, completion, ttft
    ("r1", 5, "acme", "acmekeyaaaaa", "smol", 200, None, 12, 30, 180.0),
    ("r2", 20, "acme", "acmekeybbbbb", "smol", 200, None, 20, 10, 240.0),
    ("r3", 40, "acme", "acmekeyaaaaa", "tiny", 429, "rate_limit_exceeded", 0, 0, None),
    ("r4", 70, "acme", "acmekeyaaaaa", "smol", 200, None, 8, 16, 200.0),
    ("r5", 75, "globex", "globexkeyccc", "smol", 200, None, 7, 5, 900.0),
    ("r6", 110, "acme", "acmekeybbbbb", "smol", 503, "no_capacity", 0, 0, None),
]


def hand_rows() -> list[dict]:
    out = []
    for rid, minute, tenant, key, model, status, code, p, comp, ttft in HAND:
        out.append(
            {
                "request_id": rid,
                "ts_ms": HAND_T0 + minute * 60_000,
                "tenant": tenant,
                "key_id": key,
                "model": model,
                "served_model": model if status == 200 else "",
                "route": "/v1/chat/completions",
                "api_version": "1",
                "status": status,
                "error_code": code,
                "stream": 1,
                "prompt_tokens": p,
                "completion_tokens": comp,
                "cached_tokens": 0,
                "ttft_ms": ttft,
                "e2e_ms": 1000.0,
                "cache_hit": 0,
                "worker_id": "",
                "trace_id": "",
            }
        )
    return out


def test_hand_example(c: Ctx) -> None:
    # WHY: the chapter's worked example (section 3), computed by hand on six
    #      rows: acme in [12:00, 14:00) is 5 requests, 2 errors, 40 prompt and
    #      56 completion tokens; smol 4/1/40/56 and tiny 1/1/0/0; the top two
    #      keys 66 and 30 tokens; hours 12:00 (3, 40) and 13:00 (2, 16); and
    #      smol's p95 TTFT over four samples is the largest, 900 ms.
    # KIND: unit
    # CHAPTER: lang.11 section 3, Worked example by hand
    conn, _ = fresh_db(c, hand_rows())
    w = {"since": HAND_T0, "until": HAND_T0 + 2 * HOUR}
    want = {
        "q1_tenant_totals": ({**w, "tenant": "acme"}, [(5, 2, 40, 56)]),
        "q2_by_model": (
            {**w, "tenant": "acme"},
            [("smol", 4, 1, 40, 56), ("tiny", 1, 1, 0, 0)],
        ),
        "q3_top_keys": (
            {**w, "n": 2},
            [("acmekeyaaaaa", "acme", 66), ("acmekeybbbbb", "acme", 30)],
        ),
        "q4_hourly": (
            {**w, "tenant": "acme"},
            [(HAND_T0, 3, 40), (HAND_T0 + HOUR, 2, 16)],
        ),
        "q5_ttft_p95": (w, [("smol", 4, 900.0)]),
    }
    for q, (p, rows) in want.items():
        got = run_query(c, conn, q, p)
        if got != rows:
            raise Fail(
                f"{DIR}/{q}.sql on the worked example:\n  got  {got}\n  want {rows}"
            )


def test_queries_are_read_only_selects_with_named_parameters(c: Ctx) -> None:
    # WHY: a query file is one SELECT that reads the ledger and takes its
    #      values as named parameters (:tenant, :since, :until, :n), never as
    #      text pasted into the SQL: the gateway's admin API passes them from
    #      a URL, and ag.04's query_usage tool runs them under a read-only gate.
    # KIND: unit
    # CHAPTER: lang.11 section 2.6
    import re

    for q in QUERIES:
        text = sql_file(c, f"{q}.sql")
        code = re.sub(r"--[^\n]*", "", text)
        used = set(re.findall(r":[a-z_]+", code))
        bad = used - PARAMS
        if bad:
            raise Fail(
                f"{DIR}/{q}.sql uses {sorted(bad)}; the parameters are {sorted(PARAMS)}"
            )
        if re.search(r"\b(acme|globex|initech|o''brien)\b", code):
            raise Fail(f"{DIR}/{q}.sql names a tenant in its text; take it as :tenant")
        if re.search(r"\b17908\d{8}\b", code):
            raise Fail(
                f"{DIR}/{q}.sql has a timestamp in its text; take the window as :since and :until"
            )
    conn, _ = fresh_db(c, hand_rows())
    run_query(c, conn, "q1_tenant_totals", {"tenant": "acme", "since": 0, "until": 1})


def test_q1_tenant_totals_on_the_fixture(c: Ctx) -> None:
    # WHY: totals for one tenant in a half-open window; COUNT and SUM over no
    #      rows must give zeros, not NULL (the "nobody" tenant and the empty
    #      window), and an error is status >= 400 or an error code.
    # KIND: unit
    # CHAPTER: lang.11 section 2.3
    check_query(c, "q1_tenant_totals")


def test_q2_by_model_on_the_fixture(c: Ctx) -> None:
    # WHY: GROUP BY with conditional aggregates, ordered by the group key; a
    #      tenant whose name holds a quote (o'brien) works because the value is
    #      a parameter.
    # KIND: unit
    # CHAPTER: lang.11 section 2.3
    check_query(c, "q2_by_model")


def test_q3_top_keys_on_the_fixture(c: Ctx) -> None:
    # WHY: ORDER BY the total descending, then the key ascending, then LIMIT:
    #      without the tie-break, two keys with equal totals (the fixture has a
    #      pair) come back in whatever order the engine chose. SQLite often
    #      returns ties in key order by accident (GROUP BY sorted them), so the
    #      ORDER BY must also name key_id explicitly.
    # KIND: boundary
    # CHAPTER: lang.11 section 5, Pitfall 4
    import re

    check_query(c, "q3_top_keys")
    code = re.sub(r"--[^\n]*", "", sql_file(c, "q3_top_keys.sql"))
    order = re.search(r"ORDER\s+BY(.*?)(LIMIT|;|$)", code, re.I | re.S)
    if not order or "key_id" not in order.group(1):
        raise Fail(
            f"{DIR}/q3_top_keys.sql: ORDER BY needs key_id as the tie-break after the token total"
        )


def test_q4_hourly_on_the_fixture(c: Ctx) -> None:
    # WHY: bucketing by an expression (the start of the hour in milliseconds);
    #      requests at exactly 11:00:00.000 belong to the 11:00 bucket and to
    #      the [11:00, 13:00) window, never to both neighbours.
    # KIND: boundary
    # CHAPTER: lang.11 section 2.4
    check_query(c, "q4_hourly")


def test_q5_ttft_p95_on_the_fixture(c: Ctx) -> None:
    # WHY: a percentile with window functions: nearest rank ceil(0.95 n) per
    #      model over the rows that have a TTFT. Refusals and embeddings have
    #      NULL TTFT and must not count in n; AVG or MAX are not a p95.
    # KIND: unit
    # CHAPTER: lang.11 section 2.5
    check_query(c, "q5_ttft_p95")


def test_index_serves_the_served_model_query(c: Ctx) -> None:
    # WHY: per-canary dashboards filter on served_model and a time bound; with
    #      no index SQLite scans every row. An index with the equality column
    #      first and the range column second lets the plan SEARCH one slice
    #      (EXPLAIN QUERY PLAN shows served_model=? AND ts_ms>?).
    # KIND: unit
    # CHAPTER: lang.11 section 2.7
    conn, _ = fresh_db(c, fixture_rows())
    for stmt in statements(sql_file(c, "indexes.sql")):
        code = "\n".join(
            x for x in stmt.splitlines() if not x.strip().startswith("--")
        ).strip()
        if not code.upper().startswith("CREATE"):
            raise Fail(
                f"{DIR}/indexes.sql may only CREATE indexes; found {stmt[:50]!r}"
            )
        conn.execute(stmt)
    conn.execute("ANALYZE")
    plan = [
        r[3]
        for r in conn.execute(
            "EXPLAIN QUERY PLAN " + SERVED_QUERY,
            {"served_model": "smol-135m@v4", "since": T0},
        )
    ]
    text = " | ".join(plan)
    if not any(
        "SEARCH usage USING" in p and "served_model=?" in p and "ts_ms>?" in p
        for p in plan
    ):
        raise Fail(
            f"the served-model query does not search an index on (served_model, ts_ms): plan is {text}"
        )
    names = {
        r[0]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
    }
    lost = {"usage_tenant_ts", "usage_key_ts", "usage_model_ts"} - names
    if lost:
        raise Fail(f"the contract's indexes {sorted(lost)} are gone after indexes.sql")


def test_pragmas_set_wal_and_connection_settings(c: Ctx) -> None:
    # WHY: WAL is stored in the database file, so a second connection sees it;
    #      synchronous, busy_timeout, and foreign_keys are per connection, so
    #      the connection that ran pragmas.sql must report NORMAL (1), a
    #      timeout of at least a second, and foreign keys on. gw.07 sets the
    #      same four on every pooled connection.
    # KIND: unit
    # CHAPTER: lang.11 section 2.2
    d = Path(tempfile.mkdtemp(prefix="ol-lang11-"))
    path = d / "p.db"
    a = sqlite3.connect(path, isolation_level=None)
    try:
        a.executescript(sql_file(c, "pragmas.sql"))
        got = {
            k: a.execute(f"PRAGMA {k}").fetchone()[0]
            for k in ("journal_mode", "synchronous", "busy_timeout", "foreign_keys")
        }
        b = sqlite3.connect(path, isolation_level=None)
        other = b.execute("PRAGMA journal_mode").fetchone()[0]
        b.close()
    finally:
        a.close()
        __import__("shutil").rmtree(d, ignore_errors=True)
    errs = []
    if got["journal_mode"] != "wal" or other != "wal":
        errs.append(
            f"journal_mode is {got['journal_mode']} (a second connection sees {other}); want wal"
        )
    if got["synchronous"] != 1:
        errs.append(f"synchronous is {got['synchronous']}; want 1 (NORMAL)")
    if got["busy_timeout"] < 1000:
        errs.append(f"busy_timeout is {got['busy_timeout']} ms; want at least 1000")
    if got["foreign_keys"] != 1:
        errs.append("foreign_keys is off")
    if errs:
        raise Fail("\n".join(errs))


def test_rollup_schema(c: Ctx) -> None:
    # WHY: usage_hourly holds one row per (tenant, hour): that pair is the
    #      primary key, so a second row for it is refused, and a CHECK keeps
    #      hour_ms on an hour boundary. Running rollup.sql twice is a no-op,
    #      because every service applies its schema on start.
    # KIND: unit
    # CHAPTER: lang.11 section 2.1
    conn, _ = fresh_db(c)
    for _ in range(2):
        try:
            conn.executescript(sql_file(c, "rollup.sql"))
        except sqlite3.DatabaseError as e:
            raise Fail(f"{DIR}/rollup.sql (applied twice): {e}") from None
    info = conn.execute("PRAGMA table_info(usage_hourly)").fetchall()
    cols = {r[1]: r for r in info}
    need = {"tenant", "hour_ms", "requests", "prompt_tokens", "completion_tokens"}
    if not need <= set(cols):
        raise Fail(
            f"usage_hourly has columns {sorted(cols)}; want at least {sorted(need)}"
        )
    pk = sorted((r[5], r[1]) for r in info if r[5])
    if [n for _, n in pk] != ["tenant", "hour_ms"]:
        raise Fail(f"the primary key is {[n for _, n in pk]}; want (tenant, hour_ms)")
    conn.execute(
        "INSERT INTO usage_hourly (tenant, hour_ms, requests, prompt_tokens, completion_tokens) VALUES ('a', 0, 1, 1, 1)"
    )
    for sql, why in (
        (
            "INSERT INTO usage_hourly (tenant, hour_ms, requests, prompt_tokens, completion_tokens) VALUES ('a', 0, 1, 1, 1)",
            "a duplicate (tenant, hour_ms)",
        ),
        (
            "INSERT INTO usage_hourly (tenant, hour_ms, requests, prompt_tokens, completion_tokens) VALUES ('a', 1800000, 1, 1, 1)",
            "hour_ms 1800000 (half past)",
        ),
    ):
        try:
            conn.execute(sql)
        except sqlite3.IntegrityError:
            continue
        raise Fail(f"usage_hourly accepted {why}")


def run_record(c: Ctx, conn: sqlite3.Connection, row: dict, script: list[str]) -> None:
    for stmt in script:
        conn.execute(stmt, row)


def record_db(c: Ctx) -> tuple[sqlite3.Connection, list[str]]:
    conn, _ = fresh_db(c)
    conn.executescript(sql_file(c, "rollup.sql"))
    script = statements(sql_file(c, "record.sql"))
    return conn, script


def rollup_of(rows: list[dict]) -> list[tuple]:
    agg: dict[tuple, list] = {}
    for r in rows:
        k = (r["tenant"], r["ts_ms"] - r["ts_ms"] % HOUR)
        a = agg.setdefault(k, [0, 0, 0])
        a[0] += 1
        a[1] += r["prompt_tokens"]
        a[2] += r["completion_tokens"]
    return [(k[0], k[1], *v) for k, v in sorted(agg.items())]


def table_rollup(conn) -> list[tuple]:
    return [
        tuple(r)
        for r in conn.execute(
            "SELECT tenant, hour_ms, requests, prompt_tokens, completion_tokens FROM usage_hourly ORDER BY tenant, hour_ms"
        )
    ]


def test_record_rolls_up_every_row(c: Ctx) -> None:
    # WHY: record.sql is what a gateway runs per request: after the 240 fixture
    #      requests, usage holds 240 rows and usage_hourly equals the rollup
    #      computed from them (an upsert: insert the first request of an hour,
    #      add to the row for the rest).
    # KIND: unit
    # CHAPTER: lang.11 section 2.1
    conn, script = record_db(c)
    rows = fixture_rows()
    try:
        for r in rows:
            run_record(c, conn, r, script)
    except sqlite3.DatabaseError as e:
        raise Fail(f"{DIR}/record.sql: {e}") from None
    if conn.in_transaction:
        raise Fail(f"{DIR}/record.sql leaves a transaction open: end it with COMMIT")
    n = conn.execute("SELECT COUNT(*) FROM usage").fetchone()[0]
    if n != len(rows):
        raise Fail(f"usage holds {n} rows after {len(rows)} records")
    got, want = table_rollup(conn), rollup_of(rows)
    if got != want:
        diff = [(g, w) for g, w in zip(got, want) if g != w][:3]
        raise Fail(
            f"usage_hourly differs from the rollup of usage ({len(got)} rows, want {len(want)}): {diff}"
        )


def test_record_replay_is_a_no_op(c: Ctx) -> None:
    # WHY: a gateway retries a write whose acknowledgement it never saw, so the
    #      same request id arrives twice. The usage row is ignored on conflict,
    #      and the rollup must not count it again either (changes() tells the
    #      second statement whether the first one inserted).
    # KIND: boundary
    # CHAPTER: lang.11 section 5, Pitfall 5
    conn, script = record_db(c)
    rows = fixture_rows()[:60]
    for r in rows + rows[:25]:
        try:
            run_record(c, conn, r, script)
        except sqlite3.DatabaseError as e:
            raise Fail(f"replaying {r['request_id']}: {e}") from None
    got, want = table_rollup(conn), rollup_of(rows)
    if got != want:
        raise Fail(
            f"after replaying 25 requests the rollup counts them twice: {[g for g, w in zip(got, want) if g != w][:3]}"
        )


def test_record_is_atomic(c: Ctx) -> None:
    # WHY: if the rollup write fails (here a trigger refuses tenant 'poison'),
    #      the usage row must not stay behind: both statements run inside one
    #      transaction, and an error before COMMIT followed by ROLLBACK (what a
    #      closed connection does) leaves neither.
    # KIND: fault
    # CHAPTER: lang.11 section 5, Pitfall 6
    conn, script = record_db(c)
    conn.execute(
        "CREATE TRIGGER poison_insert BEFORE INSERT ON usage_hourly WHEN NEW.tenant = 'poison' "
        "BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
    )
    row = dict(fixture_rows()[0], request_id="poison-1", tenant="poison")
    try:
        run_record(c, conn, row, script)
        raise Fail("the injected rollup failure did not stop record.sql")
    except sqlite3.DatabaseError:
        pass
    if conn.in_transaction:
        conn.execute("ROLLBACK")
    left = conn.execute(
        "SELECT COUNT(*) FROM usage WHERE request_id = 'poison-1'"
    ).fetchone()[0]
    if left:
        raise Fail(
            "the usage row stayed after its rollup failed: put both statements in one transaction"
        )
    run_record(
        c, conn, fixture_rows()[1], script
    )  # the connection still works afterwards


if __name__ == "__main__":
    raise SystemExit(run(globals()))
