// Course tests for gw.07, the usage ledger and its store (go/gateway/ledger).
//
// The store tests talk to your ledger.DB directly and read the database back
// through a second, raw database/sql connection, so they check what is on
// disk, not what your code believes it wrote. The chain tests (meter_test.go)
// put your Meter behind gw.01's server chain in front of a fake engine.
// Times come from the course's fake clock; nothing sleeps for a fixed time.
package gw_07

import (
	"context"
	"database/sql"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"tinyllm/gateway/ledger"
)

// t0 is the worked example's first request: 2026-10-01 12:00:00 UTC.
var t0 = time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)

func ctxT(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	t.Cleanup(cancel)
	return ctx
}

// open makes a fresh ledger in a temporary directory.
func open(t *testing.T) (*ledger.DB, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "usage.db")
	db, err := ledger.Open(path)
	if err != nil {
		t.Fatalf("ledger.Open(%s): %v", path, err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db, path
}

// raw is an independent connection to the file, through the driver your
// package registered (modernc.org/sqlite registers "sqlite").
func raw(t *testing.T, path string) *sql.DB {
	t.Helper()
	name := ""
	for _, d := range sql.Drivers() {
		if d == "sqlite" || (name == "" && d == "sqlite3") {
			name = d
		}
	}
	if name == "" {
		t.Fatal(`no SQLite database/sql driver is registered: import _ "modernc.org/sqlite" in your ledger package`)
	}
	db, err := sql.Open(name, path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}

func pragma(t *testing.T, db *sql.DB, name string) string {
	t.Helper()
	var v string
	if err := db.QueryRow("PRAGMA " + name).Scan(&v); err != nil {
		t.Fatalf("PRAGMA %s: %v", name, err)
	}
	return v
}

// The four requests of the chapter's worked example (section 3).
func handRecords() []ledger.UsageRecord {
	ttft := 180 * time.Millisecond
	return []ledger.UsageRecord{
		{RequestID: "req-1", Start: t0, Tenant: "acme", KeyID: "acmekeyaaaaa", Model: "smol", ServedModel: "smol",
			Route: "/v1/chat/completions", Status: 200, Stream: true, PromptTokens: 12, CompletionTokens: 30,
			TTFT: &ttft, E2E: 900 * time.Millisecond},
		{RequestID: "req-2", Start: t0.Add(time.Minute), Tenant: "acme", KeyID: "acmekeybbbbb", Model: "smol", ServedModel: "smol",
			Route: "/v1/chat/completions", Status: 200, PromptTokens: 20, CompletionTokens: 10, E2E: 400 * time.Millisecond},
		{RequestID: "req-3", Start: t0.Add(2 * time.Minute), Tenant: "acme", KeyID: "acmekeyaaaaa", Model: "tiny",
			Route: "/v1/completions", Status: 429, ErrorCode: "rate_limit_exceeded", E2E: 2 * time.Millisecond},
		{RequestID: "req-4", Start: t0.Add(3 * time.Minute), Tenant: "globex", KeyID: "globexkeyccc", Model: "smol", ServedModel: "smol",
			Route: "/v1/chat/completions", Status: 200, PromptTokens: 7, CompletionTokens: 5, E2E: 300 * time.Millisecond},
	}
}

func recordAll(t *testing.T, l ledger.Ledger, recs []ledger.UsageRecord) {
	t.Helper()
	for _, r := range recs {
		if err := l.Record(ctxT(t), r); err != nil {
			t.Fatalf("Record(%s): %v", r.RequestID, err)
		}
	}
}

func query(t *testing.T, l ledger.Ledger, f ledger.UsageFilter) []ledger.UsageRow {
	t.Helper()
	rows, err := l.Query(ctxT(t), f)
	if err != nil {
		t.Fatalf("Query(%+v): %v", f, err)
	}
	return rows
}

func TestHandWorkedExample(t *testing.T) {
	// WHY: the chapter's worked example (section 3), computed by hand: acme's
	//      three requests sum to 3 requests, 1 error, 32 prompt and 40
	//      completion tokens; by model, smol 2/0/32/40 and tiny 1/1/0/0; and the
	//      window [12:01, 12:03) holds req-2 and req-3 only.
	// KIND: unit
	// CATCHES: s07, s11, s21
	// CHAPTER: gw.07 section 3, Worked example by hand
	db, _ := open(t)
	recordAll(t, db, handRecords())

	got := query(t, db, ledger.UsageFilter{Tenant: "acme"})
	want := []ledger.UsageRow{{Requests: 3, Errors: 1, PromptTokens: 32, CompletionTokens: 40}}
	if !equalRows(got, want) {
		t.Fatalf("acme, no grouping:\n got %+v\nwant %+v", got, want)
	}
	got = query(t, db, ledger.UsageFilter{Tenant: "acme", GroupBy: "model"})
	want = []ledger.UsageRow{
		{Model: "smol", Requests: 2, PromptTokens: 32, CompletionTokens: 40},
		{Model: "tiny", Requests: 1, Errors: 1},
	}
	if !equalRows(got, want) {
		t.Fatalf("acme by model:\n got %+v\nwant %+v", got, want)
	}
	got = query(t, db, ledger.UsageFilter{Since: t0.Add(time.Minute), Until: t0.Add(3 * time.Minute)})
	want = []ledger.UsageRow{{Requests: 2, Errors: 1, PromptTokens: 20, CompletionTokens: 10}}
	if !equalRows(got, want) {
		t.Fatalf("window [12:01, 12:03):\n got %+v\nwant %+v", got, want)
	}
	got = query(t, db, ledger.UsageFilter{Tenant: "initech"})
	want = []ledger.UsageRow{{}}
	if !equalRows(got, want) {
		t.Fatalf("a tenant with no rows must give one row of zeros:\n got %+v\nwant %+v", got, want)
	}
}

func equalRows(a, b []ledger.UsageRow) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// contractSchema finds contracts/formats/usage.v1.sql above the test's
// directory (course/contracts in openlearn, contracts/ in an export).
func contractSchema(t *testing.T) string {
	t.Helper()
	dir, _ := os.Getwd()
	for {
		p := filepath.Join(dir, "contracts", "formats", "usage.v1.sql")
		if b, err := os.ReadFile(p); err == nil {
			return string(b)
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("contracts/formats/usage.v1.sql not found above the test directory")
		}
		dir = parent
	}
}

type master struct{ typ, name, tbl, sql string }

func schemaOf(t *testing.T, db *sql.DB) []master {
	t.Helper()
	rows, err := db.Query("SELECT type, name, tbl_name, COALESCE(sql, '') FROM sqlite_master ORDER BY type, name")
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	var out []master
	for rows.Next() {
		var m master
		if err := rows.Scan(&m.typ, &m.name, &m.tbl, &m.sql); err != nil {
			t.Fatal(err)
		}
		m.sql = strings.Join(strings.Fields(m.sql), " ")
		out = append(out, m)
	}
	return out
}

func TestSchemaMatchesContract(t *testing.T) {
	// WHY: formats/usage.v1.sql is the contract ag.04's query_usage tool and
	//      ops.09 read: Open must create exactly its tables, columns, CHECKs,
	//      and indexes (compared through sqlite_master with a database built
	//      from the contract file itself), record schema version 1, and leave
	//      the file in WAL mode, which is what makes a SIGKILL safe.
	// KIND: conformance
	// CATCHES: s05
	// CHAPTER: gw.07 section 2.2
	_, path := open(t)
	got := schemaOf(t, raw(t, path))

	ref := raw(t, filepath.Join(t.TempDir(), "contract.db"))
	if _, err := ref.Exec(contractSchema(t)); err != nil {
		t.Fatalf("applying the contract file: %v", err)
	}
	want := schemaOf(t, ref)
	if len(got) != len(want) {
		t.Fatalf("Open created %d schema objects, the contract %d:\n got %v\nwant %v", len(got), len(want), got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("schema object %d differs from the contract:\n got %+v\nwant %+v", i, got[i], want[i])
		}
	}
	db := raw(t, path)
	var v int
	if err := db.QueryRow("SELECT MAX(version) FROM schema_version").Scan(&v); err != nil || v != 1 {
		t.Fatalf("schema_version = %d (%v), want 1", v, err)
	}
	if jm := pragma(t, db, "journal_mode"); jm != "wal" {
		t.Fatalf("journal_mode = %q, want wal", jm)
	}
}

func TestReopenKeepsRowsAndRefusesNewerSchema(t *testing.T) {
	// WHY: the gateway restarts on every rollout: reopening must apply the
	//      schema as a no-op and keep every row; and a gateway that is older
	//      than the file (schema_version 2 from a later migration) must refuse
	//      to write into it rather than corrupt it.
	// KIND: boundary
	// CATCHES: m07
	// CHAPTER: gw.07 section 2.2
	path := filepath.Join(t.TempDir(), "usage.db")
	db, err := ledger.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	recordAll(t, db, handRecords())
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	db, err = ledger.Open(path)
	if err != nil {
		t.Fatalf("reopening: %v", err)
	}
	got := query(t, db, ledger.UsageFilter{})
	if len(got) != 1 || got[0].Requests != 4 {
		t.Fatalf("after reopening: %+v, want 4 requests", got)
	}
	_ = db.Close()
	r := raw(t, path)
	if _, err := r.Exec("INSERT INTO schema_version (version) VALUES (2)"); err != nil {
		t.Fatal(err)
	}
	_ = r.Close()
	if db, err := ledger.Open(path); err == nil {
		_ = db.Close()
		t.Fatal("Open accepted a database with schema_version 2")
	} else if !errors.Is(err, ledger.ErrNewerSchema) {
		t.Fatalf("Open: %v, want an error wrapping ErrNewerSchema", err)
	}
}

func TestRecordIsIdempotentByRequestID(t *testing.T) {
	// WHY: a retried write (the meter's write timed out after the commit, or a
	//      replayed request id) must not count a request twice: the first row
	//      for a request id stands and later ones are ignored without error.
	// KIND: boundary
	// CATCHES: s04
	// CHAPTER: gw.07 section 5, Pitfall 4
	db, _ := open(t)
	r := handRecords()[0]
	recordAll(t, db, []ledger.UsageRecord{r})
	r.PromptTokens, r.CompletionTokens = 999, 999
	if err := db.Record(ctxT(t), r); err != nil {
		t.Fatalf("a second Record of req-1 must be a no-op, got %v", err)
	}
	got := query(t, db, ledger.UsageFilter{})
	want := []ledger.UsageRow{{Requests: 1, PromptTokens: 12, CompletionTokens: 30}}
	if !equalRows(got, want) {
		t.Fatalf("after recording req-1 twice:\n got %+v\nwant %+v (the first row stands)", got, want)
	}
}

func TestRecordValidatesAndClampsCachedTokens(t *testing.T) {
	// WHY: the CHECK constraints of usage.v1.sql reject cached > prompt, so an
	//      engine that reports it would lose the whole row; the ledger clamps
	//      cached tokens to the prompt tokens and keeps the row. Negative
	//      counts and an empty request id are refused with ErrInvalidRecord.
	// KIND: boundary
	// CATCHES: s10
	// CHAPTER: gw.07 section 5, Pitfall 6
	db, path := open(t)
	r := handRecords()[1]
	r.CachedTokens = 50
	if err := db.Record(ctxT(t), r); err != nil {
		t.Fatalf("Record with cached 50 > prompt 20: %v (clamp it)", err)
	}
	var cached int
	if err := raw(t, path).QueryRow("SELECT cached_tokens FROM usage WHERE request_id = 'req-2'").Scan(&cached); err != nil || cached != 20 {
		t.Fatalf("cached_tokens = %d (%v), want 20 (clamped to prompt_tokens)", cached, err)
	}
	for _, bad := range []ledger.UsageRecord{
		{RequestID: "", Route: "/v1/completions", Status: 200},
		{RequestID: "neg", Route: "/v1/completions", Status: 200, PromptTokens: -1},
		{RequestID: "neg2", Route: "/v1/completions", Status: 200, CompletionTokens: -3},
	} {
		if err := db.Record(ctxT(t), bad); !errors.Is(err, ledger.ErrInvalidRecord) {
			t.Fatalf("Record(%+v) = %v, want ErrInvalidRecord", bad, err)
		}
	}
}

func TestRecordStoresEveryColumn(t *testing.T) {
	// WHY: ag.04 and the dashboards read columns, not the Go struct: each field
	//      lands in its usage.v1.sql column in its unit (Unix milliseconds,
	//      milliseconds as REAL), an empty error code is NULL, a missing TTFT
	//      is NULL, and an empty api_version is '1'.
	// KIND: unit
	// CATCHES: m05
	// CHAPTER: gw.07 section 2.2
	db, path := open(t)
	ttft := 1500 * time.Microsecond
	r := ledger.UsageRecord{RequestID: "cols", Start: t0, Tenant: "acme", KeyID: "acmekeyaaaaa", Model: "smol",
		ServedModel: "smol-135m", Route: "/v1/chat/completions", Status: 200, Stream: false,
		PromptTokens: 9, CompletionTokens: 4, CachedTokens: 3, TTFT: &ttft, E2E: 2250 * time.Microsecond,
		CacheHit: true, WorkerID: "w-1", TraceID: "4bf92f3577b34da6a3ce929d0e0e4736"}
	recordAll(t, db, []ledger.UsageRecord{r, {RequestID: "nulls", Start: t0, Route: "/v1/completions", Status: 503, ErrorCode: "no_capacity"}})
	var (
		ts                            int64
		tenant, key, model, served    string
		route, ver, worker, trace     string
		status, stream, p, c, ca, hit int
		code                          sql.NullString
		ttftMS                        sql.NullFloat64
		e2e                           float64
	)
	err := raw(t, path).QueryRow(`SELECT ts_ms, tenant, key_id, model, served_model, route, api_version, status,
		error_code, stream, prompt_tokens, completion_tokens, cached_tokens, ttft_ms, e2e_ms, cache_hit, worker_id, trace_id
		FROM usage WHERE request_id = 'cols'`).Scan(&ts, &tenant, &key, &model, &served, &route, &ver, &status,
		&code, &stream, &p, &c, &ca, &ttftMS, &e2e, &hit, &worker, &trace)
	if err != nil {
		t.Fatal(err)
	}
	if ts != t0.UnixMilli() || tenant != "acme" || key != "acmekeyaaaaa" || model != "smol" || served != "smol-135m" ||
		route != "/v1/chat/completions" || ver != "1" || status != 200 || code.Valid || stream != 0 ||
		p != 9 || c != 4 || ca != 3 || !ttftMS.Valid || ttftMS.Float64 != 1.5 || e2e != 2.25 || hit != 1 ||
		worker != "w-1" || trace != "4bf92f3577b34da6a3ce929d0e0e4736" {
		t.Fatalf("row cols: ts=%d tenant=%q key=%q model=%q served=%q route=%q ver=%q status=%d code=%v stream=%d p=%d c=%d cached=%d ttft=%v e2e=%v hit=%d worker=%q trace=%q",
			ts, tenant, key, model, served, route, ver, status, code, stream, p, c, ca, ttftMS, e2e, hit, worker, trace)
	}
	err = raw(t, path).QueryRow("SELECT error_code, ttft_ms, api_version FROM usage WHERE request_id = 'nulls'").Scan(&code, &ttftMS, &ver)
	if err != nil || code.String != "no_capacity" || ttftMS.Valid || ver != "1" {
		t.Fatalf("row nulls: code=%v ttft=%v api_version=%q (%v); want no_capacity, NULL, '1'", code, ttftMS, ver, err)
	}
}

func TestQueryWindowIsHalfOpen(t *testing.T) {
	// WHY: since is inclusive and until exclusive (admin.v1.yaml), so hourly
	//      windows [10:00, 11:00) and [11:00, 12:00) add up to the two-hour
	//      total with no request counted twice or lost at 11:00:00.000.
	// KIND: boundary
	// CATCHES: s07
	// CHAPTER: gw.07 section 2.3
	db, _ := open(t)
	base := t0.Add(-2 * time.Hour)
	var recs []ledger.UsageRecord
	for i, at := range []time.Duration{0, time.Hour - time.Millisecond, time.Hour, 2*time.Hour - time.Millisecond, 2 * time.Hour} {
		recs = append(recs, ledger.UsageRecord{RequestID: "w" + string(rune('a'+i)), Start: base.Add(at), Tenant: "acme",
			Route: "/v1/completions", Status: 200, PromptTokens: 1 << i})
	}
	recordAll(t, db, recs)
	a := query(t, db, ledger.UsageFilter{Since: base, Until: base.Add(time.Hour)})
	b := query(t, db, ledger.UsageFilter{Since: base.Add(time.Hour), Until: base.Add(2 * time.Hour)})
	all := query(t, db, ledger.UsageFilter{Since: base, Until: base.Add(2 * time.Hour)})
	if a[0].PromptTokens != 1+2 || b[0].PromptTokens != 4+8 || all[0].PromptTokens != 15 {
		t.Fatalf("[10:00,11:00) = %d (want 3), [11:00,12:00) = %d (want 12), [10:00,12:00) = %d (want 15)",
			a[0].PromptTokens, b[0].PromptTokens, all[0].PromptTokens)
	}
}

func TestQueryValuesAreBoundParameters(t *testing.T) {
	// WHY: tenant and key_id arrive from the admin API's query string; spliced
	//      into the SQL text, tenant=x' OR '1'='1 would read every tenant's
	//      usage (and a crafted value could drop the table). Bound parameters
	//      treat them as data.
	// KIND: unit
	// CATCHES: s08
	// CHAPTER: gw.07 section 5, Pitfall 7
	db, _ := open(t)
	recordAll(t, db, handRecords())
	for _, f := range []ledger.UsageFilter{
		{Tenant: "x' OR '1'='1"},
		{KeyID: "x' OR '1'='1"},
		{Tenant: "acme' --"},
	} {
		rows, err := db.Query(ctxT(t), f)
		if err == nil && (len(rows) != 1 || rows[0].Requests != 0) {
			t.Fatalf("Query(%+v) = %+v: a quote in a value changed the query", f, rows)
		}
	}
	if got := query(t, db, ledger.UsageFilter{}); got[0].Requests != 4 {
		t.Fatalf("after the hostile queries the ledger holds %d rows, want 4", got[0].Requests)
	}
}

func TestQueryGroupByIsWhitelisted(t *testing.T) {
	// WHY: a GROUP BY column cannot be a bound parameter, so it must come from
	//      a fixed list; anything else is ErrBadGroupBy (the admin API's 400),
	//      and every listed grouping returns its rows sorted by the key.
	// KIND: boundary
	// CATCHES: m06
	// CHAPTER: gw.07 section 2.3
	db, _ := open(t)
	recordAll(t, db, handRecords())
	for _, g := range []string{"tenant; DROP TABLE usage", "status", "Model"} {
		if _, err := db.Query(ctxT(t), ledger.UsageFilter{GroupBy: g}); !errors.Is(err, ledger.ErrBadGroupBy) {
			t.Fatalf("GroupBy %q: %v, want ErrBadGroupBy", g, err)
		}
	}
	byKey := query(t, db, ledger.UsageFilter{GroupBy: "key_id"})
	keys := []string{}
	for _, r := range byKey {
		keys = append(keys, r.KeyID)
	}
	if !sort.StringsAreSorted(keys) || len(keys) != 3 || keys[0] != "acmekeyaaaaa" {
		t.Fatalf("group_by key_id gave keys %v, want 3 sorted keys", keys)
	}
	byVer := query(t, db, ledger.UsageFilter{GroupBy: "api_version"})
	if len(byVer) != 1 || byVer[0].APIVersion != "1" || byVer[0].Requests != 4 {
		t.Fatalf("group_by api_version = %+v, want one row for version 1", byVer)
	}
}

func TestConcurrentRecordsAllLand(t *testing.T) {
	// WHY: the gateway records from every request goroutine at once. SQLite
	//      allows one writer at a time; without a busy timeout the losers fail
	//      at once with SQLITE_BUSY ("database is locked") and their rows are
	//      lost. 16 writers x 64 rows must all land.
	// KIND: fault
	// CATCHES: s06
	// CHAPTER: gw.07 section 5, Pitfall 3
	db, _ := open(t)
	var wg sync.WaitGroup
	errs := make(chan error, 16*64)
	for w := 0; w < 16; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < 64; i++ {
				r := ledger.UsageRecord{RequestID: idOf(w*64 + i), Start: t0, Tenant: "acme",
					Route: "/v1/completions", Status: 200, PromptTokens: 1, CompletionTokens: 1}
				if err := db.Record(context.Background(), r); err != nil {
					errs <- err
					return
				}
			}
		}(w)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatalf("a concurrent Record failed: %v", err)
	}
	if got := query(t, db, ledger.UsageFilter{}); got[0].Requests != 1024 {
		t.Fatalf("%d rows after 1024 concurrent records", got[0].Requests)
	}
}

func idOf(i int) string {
	const hex = "0123456789abcdef"
	b := []byte("req-00000")
	for j := len(b) - 1; j >= 4 && i > 0; j-- {
		b[j] = hex[i%16]
		i /= 16
	}
	return string(b)
}
