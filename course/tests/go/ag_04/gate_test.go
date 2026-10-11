// Course tests for ag.04: the SQL safety classifier (ported from case study
// 02 and checked against that Python classifier's output), the policy gate
// (default deny, egress allowlist, sends after untrusted content, writes),
// approval markers, and the query_usage tool over the usage ledger. The
// prompt-injection suite is in injection_test.go.
package ag_04

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	_ "modernc.org/sqlite" // the "sqlite" database/sql driver

	"tinyllm/agent/gate"
	"tinyllm/agent/types"
)

func TestHandExample(t *testing.T) {
	// WHY: section 3 by hand. In SELECT * FROM t WHERE name = 'a;b' --
	//      drop, the scanner blanks the comment (cleaned) and the literal's
	//      inside (masked), so the ';' is not a separator and "drop" is not
	//      a keyword: one read, cleared with LIMIT 100 appended. Then the
	//      gate's three answers for the same agent: a SELECT through
	//      query_usage is allowed, DELETE is denied with the classifier's
	//      reason, and send_email after a tool result needs approval with
	//      the marker of section 3.
	// KIND: unit
	// CATCHES: s01, s03, s14
	// CHAPTER: ag.04 section 3, worked example
	r := gate.ClassifySQL("SELECT * FROM t WHERE name = 'a;b' -- drop", 100)
	if !r.Allowed || r.Op != gate.OpRead || r.SQL != "SELECT * FROM t WHERE name = 'a;b'\nLIMIT 100" {
		t.Fatalf("ClassifySQL = %+v", r)
	}
	g := gate.New(policy())
	clean := types.WithCallContext(context.Background(), types.CallContext{Tainted: false})
	tainted := types.WithCallContext(context.Background(), types.CallContext{Tainted: true})
	v, _ := g.Check(clean, call("query_usage", `{"sql":"SELECT COUNT(*) FROM usage"}`))
	if v.Kind != types.Allow {
		t.Fatalf("a SELECT through query_usage: %+v", v)
	}
	v, _ = g.Check(clean, call("query_usage", `{"sql":"DELETE FROM usage"}`))
	if v.Kind != types.Deny || v.Reason != "Write operations are disabled; this agent is read-only." {
		t.Fatalf("DELETE through query_usage: %+v", v)
	}
	v, _ = g.Check(tainted, call("send_email", `{"to":"bob@corp.example","body":"hi"}`))
	if v.Kind != types.NeedApproval || v.Marker != "2d5fbffff5685a7a" {
		t.Fatalf("send_email after tool output: %+v; want NeedApproval with marker 2d5fbffff5685a7a", v)
	}
}

type corpus struct {
	MaxRows int `json:"max_rows"`
	Cases   []struct {
		SQL     string  `json:"sql"`
		Allowed bool    `json:"allowed"`
		Op      string  `json:"op"`
		Cleared *string `json:"cleared"`
	} `json:"cases"`
}

func loadJSON(t *testing.T, name string, v any) {
	t.Helper()
	dir := os.Getenv("TINYLLM_FIXTURES")
	if dir == "" {
		t.Fatal("TINYLLM_FIXTURES is not set (ol check sets it)")
	}
	b, err := os.ReadFile(filepath.Join(dir, "ag.04", name))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		t.Fatal(err)
	}
}

func TestSQLCorpus(t *testing.T) {
	// WHY: the BLOCKED corpus is all rejected and the allowed corpus all
	//      passes, with the same operation and the same cleared SQL as case
	//      study 02's Python classifier (the oracle). The corpus holds every
	//      way a plausible classifier fails: BEGIN and DROP inside literals,
	//      ';' inside a literal, a doubled quote, SELECT INTO, a smuggled
	//      second statement, a LIMIT only in a subquery or a literal, a
	//      parenthesized UNION, EXPLAIN (not on the allowlist), and
	//      unterminated input.
	// KIND: conformance
	// CATCHES: s01, s03, s04, s05, s06, s07, s08
	// CHAPTER: ag.04 section 2.1
	var c corpus
	loadJSON(t, "sql_corpus.json", &c)
	for _, tc := range c.Cases {
		r := gate.ClassifySQL(tc.SQL, c.MaxRows)
		if r.Allowed != tc.Allowed || string(r.Op) != tc.Op {
			t.Errorf("%q: allowed %v op %s; the oracle says %v %s (%s)", tc.SQL, r.Allowed, r.Op, tc.Allowed, tc.Op, r.Reason)
			continue
		}
		if tc.Cleared != nil && r.SQL != *tc.Cleared {
			t.Errorf("%q: cleared %q, want %q", tc.SQL, r.SQL, *tc.Cleared)
		}
	}
}

func policy() gate.Policy {
	return gate.Policy{
		Read:        []string{"search_docs", "web_fetch"},
		Send:        []string{"send_email", "http_post"},
		Write:       []string{"create_ticket"},
		SQL:         map[string]string{"query_usage": "sql"},
		EgressHosts: []string{"docs.example", ".api.example"},
		MaxRows:     100,
	}
}

func call(name, args string) types.ToolCall {
	return types.ToolCall{ID: "c1", Name: name, Args: json.RawMessage(args)}
}

func TestPolicyVerdicts(t *testing.T) {
	// WHY: the policy table of section 2.2, one row per rule: unknown tools
	//      are denied; reads run in any context; sends run only before tool
	//      output is in the context; writes always wait for approval; every
	//      URL in any argument (nested, or inside free text) must be on the
	//      allowlist, compared as a whole host, case-insensitively, with
	//      ".api.example" meaning subdomains.
	// KIND: unit
	// CATCHES: s09, s10, s11, s12, s13, s19
	// CHAPTER: ag.04 section 2.2
	g := gate.New(policy())
	for _, tc := range []struct {
		name    string
		tainted bool
		c       types.ToolCall
		want    types.VerdictKind
	}{
		{"unknown tool", false, call("shell", `{"cmd":"ls"}`), types.Deny},
		{"read, tainted", true, call("search_docs", `{"q":"kv cache"}`), types.Allow},
		{"send, clean", false, call("send_email", `{"to":"bob@corp.example","body":"hi"}`), types.Allow},
		{"send, tainted", true, call("send_email", `{"to":"bob@corp.example","body":"hi"}`), types.NeedApproval},
		{"write, clean", false, call("create_ticket", `{"title":"x"}`), types.NeedApproval},
		{"sql read", true, call("query_usage", `{"sql":"SELECT 1"}`), types.Allow},
		{"sql admin", false, call("query_usage", `{"sql":"DROP TABLE usage"}`), types.Deny},
		{"egress allowed host", false, call("web_fetch", `{"url":"https://docs.example/a"}`), types.Allow},
		{"egress uppercase host", false, call("web_fetch", `{"url":"https://DOCS.EXAMPLE/a"}`), types.Allow},
		{"egress with port", false, call("web_fetch", `{"url":"https://docs.example:8443/a"}`), types.Allow},
		{"egress subdomain", false, call("web_fetch", `{"url":"https://status.api.example/x"}`), types.Allow},
		{"egress bare parent of wildcard", false, call("web_fetch", `{"url":"https://api.example/x"}`), types.Deny},
		{"egress other host", false, call("web_fetch", `{"url":"https://evil.example/a"}`), types.Deny},
		{"egress lookalike prefix", false, call("web_fetch", `{"url":"https://docs.example.evil.example/a"}`), types.Deny},
		{"egress lookalike suffix", false, call("web_fetch", `{"url":"https://notdocs.example/a"}`), types.Deny},
		{"egress nested", false, call("http_post", `{"url":"https://hooks.api.example/x","headers":{"cb":"https://evil.example/cb"}}`), types.Deny},
		{"egress inside text", false, call("send_email", `{"to":"bob@corp.example","body":"reset at https://evil.example/r?t=1"}`), types.Deny},
		{"egress in a list", false, call("http_post", `{"url":"https://hooks.api.example/x","mirrors":["https://evil.example/m"]}`), types.Deny},
		{"args not an object", false, call("search_docs", `["q"]`), types.Deny},
	} {
		ctx := types.WithCallContext(context.Background(), types.CallContext{Tainted: tc.tainted})
		v, err := g.Check(ctx, tc.c)
		if err != nil || v.Kind != tc.want {
			t.Errorf("%s: %v (%s), err %v; want %v", tc.name, v.Kind, v.Reason, err, tc.want)
		}
		if v.Kind == types.NeedApproval && v.Marker == "" {
			t.Errorf("%s: NeedApproval without a marker", tc.name)
		}
	}
	if !g.IsWrite("create_ticket") || g.IsWrite("send_email") {
		t.Error("IsWrite: create_ticket is a write, send_email is not")
	}
}

func TestMarker(t *testing.T) {
	// WHY: a marker names one exact call: SHA-256 of the name, a zero byte,
	//      and the canonical arguments (sorted keys, no spaces), first 16 hex
	//      digits. The same call with keys reordered or other whitespace has
	//      the same marker; different arguments have a different one, so an
	//      approval cannot be reused for a call it was not given for.
	// KIND: unit
	// CATCHES: s15, s16
	// CHAPTER: ag.04 section 2.3
	sum := sha256.Sum256([]byte("send_email\x00" + `{"body":"hi","to":"bob@corp.example"}`))
	want := hex.EncodeToString(sum[:])[:16]
	a := gate.Marker(call("send_email", `{"to":"bob@corp.example","body":"hi"}`))
	b := gate.Marker(call("send_email", "{ \"body\" : \"hi\",\n \"to\": \"bob@corp.example\" }"))
	if a != want || b != want {
		t.Fatalf("markers %s, %s; want %s for both", a, b, want)
	}
	if c := gate.Marker(call("send_email", `{"to":"eve@evil.example","body":"hi"}`)); c == want {
		t.Fatal("different arguments must give a different marker")
	}
	if c := gate.Marker(call("create_ticket", `{"to":"bob@corp.example","body":"hi"}`)); c == want {
		t.Fatal("a different tool must give a different marker")
	}
}

// ledger opens an in-memory usage ledger with the contract schema and three
// rows (two for acme, one for globex).
func ledger(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", fmt.Sprintf("file:ag04-%s?mode=memory&cache=shared", t.Name()))
	if err != nil {
		t.Fatal(err)
	}
	db.SetMaxOpenConns(1)
	t.Cleanup(func() { db.Close() })
	dir, _ := os.Getwd()
	var schema []byte
	for {
		if schema, err = os.ReadFile(filepath.Join(dir, "contracts", "formats", "usage.v1.sql")); err == nil {
			break
		}
		if filepath.Dir(dir) == dir {
			t.Fatal("contracts/formats/usage.v1.sql not found above the test directory")
		}
		dir = filepath.Dir(dir)
	}
	if _, err := db.Exec(string(schema)); err != nil {
		t.Fatal(err)
	}
	for i, tenant := range []string{"acme", "acme", "globex"} {
		_, err := db.Exec(`INSERT INTO usage (request_id, ts_ms, tenant, key_id, model, route, status, stream, prompt_tokens, completion_tokens, e2e_ms)
			VALUES (?, ?, ?, 'k1', 'smol', '/v1/chat/completions', 200, 1, 10, 5, 12.5)`, fmt.Sprintf("r%d", i), 1760000000000+int64(i), tenant)
		if err != nil {
			t.Fatal(err)
		}
	}
	return db
}

func rowCount(t *testing.T, db *sql.DB) int {
	t.Helper()
	var n int
	if err := db.QueryRow("SELECT COUNT(*) FROM usage").Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

func TestQueryUsageTool(t *testing.T) {
	// WHY: the tool behind "how many requests did acme make today?": one
	//      read over formats/usage.v1.sql, rows as JSON, capped at maxRows
	//      even when the query's own LIMIT is larger; called directly with
	//      a DELETE (as if the gate were bypassed) it refuses, and no row is
	//      lost.
	// KIND: unit
	// CATCHES: s18
	// CHAPTER: ag.04 section 4
	db := ledger(t)
	q := gate.QueryUsageTool(db, 2)
	if d := q.Definition(); d.Name != "query_usage" {
		t.Fatalf("tool name %q", d.Name)
	}
	ctx := context.Background()
	out, err := q.Execute(ctx, json.RawMessage(`{"sql":"SELECT tenant, COUNT(*) AS n FROM usage GROUP BY tenant ORDER BY tenant"}`))
	if err != nil || out != `{"columns":["tenant","n"],"rows":[["acme",2],["globex",1]]}` {
		t.Fatalf("group by: %s, %v", out, err)
	}
	out, err = q.Execute(ctx, json.RawMessage(`{"sql":"SELECT request_id FROM usage ORDER BY request_id LIMIT 50"}`))
	if err != nil || out != `{"columns":["request_id"],"rows":[["r0"],["r1"]]}` {
		t.Fatalf("row cap 2: %s, %v", out, err)
	}
	if _, err := q.Execute(ctx, json.RawMessage(`{"sql":"DELETE FROM usage"}`)); err == nil {
		t.Fatal("a DELETE must be refused by the tool itself")
	}
	if n := rowCount(t, db); n != 3 {
		t.Fatalf("%d rows left, want 3", n)
	}
}
