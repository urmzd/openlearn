// Course tests for gw.02: API keys, scopes, the model allowlist, and the
// authn stage (go/gateway/auth).
//
// Key files live in t.TempDir(). Time comes from the course testkit's fake
// clock, so "within the cache TTL" is an exact statement, not a sleep.
package gw_02

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"openlearn.urmzd.com/tl/testkit/clock"
	"tinyllm/gateway/auth"
	"tinyllm/gateway/server"
)

var t0 = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

// The chapter's worked example (section 3).
const (
	pepper    = "pepper-demo"
	handID    = "demo2key7abc"
	handSec   = "S3cretS3cretS3cretS3cretS3cret01"
	handKey   = "tl_" + handID + "_" + handSec
	handHMAC  = "94f4a20cad218d9e113440c90096b20064d78a02a3cb6f72210222c4d6729506"
	plainSHA2 = "118a232e6e27fd7e6a710f31e6f9249ff9f3956d144435cf23c702f2b4712e64" // SHA-256 without the pepper
)

func writeKeys(t *testing.T, recs ...auth.Record) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "keys.jsonl")
	var b bytes.Buffer
	for _, r := range recs {
		j, _ := json.Marshal(r)
		b.Write(j)
		b.WriteByte('\n')
	}
	if err := os.WriteFile(p, b.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func handRecord() auth.Record {
	return auth.Record{KeyID: handID, HMAC: handHMAC, Tenant: "acme", Name: "demo",
		Scopes: []string{"infer"}, RPM: 60, TPM: 100000, Priority: 5, Models: []string{"smol-135m"}, CreatedAt: t0}
}

func open(t *testing.T, path string, c server.Clock, ttl time.Duration) *auth.Store {
	t.Helper()
	s, err := auth.NewStore(path, []byte(pepper), auth.Options{Clock: c, CacheTTL: ttl})
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestHandExampleLookup(t *testing.T) {
	// WHY: the worked example: tl_demo2key7abc_S3cret... is split into id and
	//      secret, the id finds one record in O(1), and HMAC-SHA256 keyed by
	//      the pepper of the secret is exactly the stored hex.
	// KIND: unit
	// CATCHES: s03, s04
	// CHAPTER: gw.02 section 3, worked example
	if got := auth.Digest([]byte(pepper), handSec); got != handHMAC {
		t.Fatalf("Digest = %s, want %s (HMAC-SHA256 keyed by the pepper, over the secret only)", got, handHMAC)
	}
	s := open(t, writeKeys(t, handRecord()), clock.NewFake(t0), 0)
	p, err := s.Lookup(context.Background(), handKey)
	if err != nil {
		t.Fatal(err)
	}
	if p.KeyID != handID || p.Tenant != "acme" || !p.HasScope("infer") || p.Priority != 5 || p.RPM != 60 {
		t.Fatalf("principal %+v", p)
	}
	// A record holding the unpeppered SHA-256 must not verify.
	r := handRecord()
	r.HMAC = plainSHA2
	s = open(t, writeKeys(t, r), clock.NewFake(t0), 0)
	if _, err := s.Lookup(context.Background(), handKey); !errors.Is(err, auth.ErrInvalidKey) {
		t.Fatalf("a plain SHA-256 record verified (%v): the pepper must be part of the digest", err)
	}
}

func TestParseKey(t *testing.T) {
	// WHY: the key format is the contract tl_<12 of [a-z2-7]>_<32 of
	//      [A-Za-z0-9]>; anything else is rejected before any lookup.
	// KIND: boundary
	// CATCHES: s16
	// CHAPTER: gw.02 section 2.1
	if id, sec, ok := auth.ParseKey(handKey); !ok || id != handID || sec != handSec {
		t.Fatalf("ParseKey(hand key) = %q %q %v", id, sec, ok)
	}
	for _, bad := range []string{
		"", "tl_", handKey[:len(handKey)-1], handKey + "x",
		"tl_DEMO2KEY7ABC_" + handSec,              // upper-case id
		"tl_demo1key7abc_" + handSec,              // 1 is not base32
		"tl_" + handID + "-" + handSec,            // wrong separator
		"tk_" + handID + "_" + handSec,            // wrong prefix
		"tl_" + handID + "_" + handSec[:31] + "!", // punctuation in the secret
	} {
		if _, _, ok := auth.ParseKey(bad); ok {
			t.Fatalf("ParseKey(%q) accepted a malformed key", bad)
		}
	}
}

func TestCreateLookupRoundTrip(t *testing.T) {
	// WHY: Create shows the plaintext key once and stores only the HMAC; the
	//      key it returns then authenticates as the tenant and scopes asked for.
	// KIND: unit
	// CATCHES: s12
	// CHAPTER: gw.02 section 2.2
	path := filepath.Join(t.TempDir(), "keys.jsonl")
	s := open(t, path, clock.NewFake(t0), 0)
	plain, rec, err := s.Create(context.Background(), auth.KeySpec{Tenant: "acme", Name: "ci", Scopes: []string{"infer", "embed"}, RPM: 10, Priority: 2})
	if err != nil {
		t.Fatal(err)
	}
	if !regexp.MustCompile(`^tl_[a-z2-7]{12}_[A-Za-z0-9]{32}$`).MatchString(plain) {
		t.Fatalf("key %q does not match tl_<id>_<secret>", plain)
	}
	_, secret, _ := auth.ParseKey(plain)
	file, _ := os.ReadFile(path)
	if bytes.Contains(file, []byte(secret)) || bytes.Contains(file, []byte(plain)) {
		t.Fatal("the keys file contains the plaintext secret")
	}
	if rec.HMAC != auth.Digest([]byte(pepper), secret) || !rec.CreatedAt.Equal(t0) {
		t.Fatalf("record %+v: HMAC of the secret, created at the clock's now", rec)
	}
	p, err := s.Lookup(context.Background(), plain)
	if err != nil || p.Tenant != "acme" || !p.HasScope("embed") || p.RPM != 10 || p.Priority != 2 {
		t.Fatalf("Lookup = %+v, %v", p, err)
	}
	// A second store on the same file sees the key too (the CLI and the
	// gateway are different processes).
	if _, err := open(t, path, clock.NewFake(t0), 0).Lookup(context.Background(), plain); err != nil {
		t.Fatalf("another store on the same file: %v", err)
	}
	if _, _, err := s.Create(context.Background(), auth.KeySpec{Tenant: "acme"}); err == nil {
		t.Fatal("a key without scopes must be refused")
	}
	if _, err := auth.NewStore(path, nil, auth.Options{}); err == nil {
		t.Fatal("an empty pepper must be refused")
	}
}

func TestLookupRejects(t *testing.T) {
	// WHY: every failure is the same ErrInvalidKey (401): a wrong secret, an
	//      unknown id, a revoked key, an expired key. Telling them apart would
	//      tell an attacker which ids exist.
	// KIND: boundary
	// CATCHES: s10
	// CHAPTER: gw.02 section 2.2
	exp := t0.Add(time.Hour)
	expired := handRecord()
	expired.KeyID, expired.ExpiresAt = "expired2key7", &exp
	revoked := handRecord()
	revoked.KeyID, revoked.Revoked = "revoked2key7", true
	fc := clock.NewFake(t0)
	s := open(t, writeKeys(t, handRecord(), expired, revoked), fc, 0)
	cases := map[string]string{
		"wrong last char": "tl_" + handID + "_" + handSec[:31] + "2",
		"prefix secret":   "tl_" + handID + "_" + strings.Repeat("S", 32),
		"unknown id":      "tl_unknown2key7_" + handSec,
		"revoked":         "tl_revoked2key7_" + handSec,
		"malformed":       "Bearer " + handKey,
	}
	for name, k := range cases {
		if _, err := s.Lookup(context.Background(), k); !errors.Is(err, auth.ErrInvalidKey) {
			t.Fatalf("%s: err = %v, want ErrInvalidKey", name, err)
		}
	}
	if _, err := s.Lookup(context.Background(), "tl_expired2key7_"+handSec); err != nil {
		t.Fatalf("an hour before expiry the key works: %v", err)
	}
	fc.Advance(time.Hour)
	if _, err := s.Lookup(context.Background(), "tl_expired2key7_"+handSec); !errors.Is(err, auth.ErrInvalidKey) {
		t.Fatalf("at expires_at the key must stop working, got %v", err)
	}
}

func TestCacheChecksTheSecret(t *testing.T) {
	// WHY: the lookup cache is keyed by id but must hold the digest: a cached
	//      id presented with a wrong secret is still a 401.
	// KIND: unit
	// CATCHES: s01
	// CHAPTER: gw.02 section 5, Pitfalls, item 3
	s := open(t, writeKeys(t, handRecord()), clock.NewFake(t0), 5*time.Second)
	if _, err := s.Lookup(context.Background(), handKey); err != nil {
		t.Fatal(err)
	}
	bad := "tl_" + handID + "_" + strings.Repeat("x", 32)
	if _, err := s.Lookup(context.Background(), bad); !errors.Is(err, auth.ErrInvalidKey) {
		t.Fatalf("a cached id with a wrong secret: err = %v, want ErrInvalidKey", err)
	}
}

func TestRevokedKeyRejectedWithinTTL(t *testing.T) {
	// WHY: revocation is the emergency brake: the store that revokes rejects
	//      at once, and a gateway that cached the key (another process on the
	//      same file) rejects it once its key_cache_ttl_s has passed.
	// KIND: fault
	// CATCHES: s08, s09
	// CHAPTER: gw.02 section 2.3
	path := writeKeys(t, handRecord())
	fc := clock.NewFake(t0)
	gw := open(t, path, fc, 5*time.Second)  // the gateway
	cli := open(t, path, fc, 5*time.Second) // the keys CLI
	if _, err := gw.Lookup(context.Background(), handKey); err != nil {
		t.Fatal(err)
	}
	if err := cli.Revoke(context.Background(), handID); err != nil {
		t.Fatal(err)
	}
	if _, err := cli.Lookup(context.Background(), handKey); !errors.Is(err, auth.ErrInvalidKey) {
		t.Fatalf("the revoking store must reject at once, got %v", err)
	}
	fc.Advance(5 * time.Second)
	if _, err := gw.Lookup(context.Background(), handKey); !errors.Is(err, auth.ErrInvalidKey) {
		t.Fatalf("5 s (the TTL) after revocation the gateway still accepts the key: %v", err)
	}
	if err := cli.Revoke(context.Background(), "nosuch2key77"); !errors.Is(err, auth.ErrNotFound) {
		t.Fatalf("revoking an unknown id: %v, want ErrNotFound", err)
	}
	recs, _ := cli.List(context.Background())
	if len(recs) != 1 || !recs[0].Revoked {
		t.Fatalf("List = %+v, want the one revoked record", recs)
	}
}

func TestAuthorizeRBAC(t *testing.T) {
	// WHY: the authorization table: a route needs its scope (embeddings:
	//      embed, admin: admin, everything else: infer), and a key with a model
	//      allowlist may use only those models; an empty allowlist means all.
	// KIND: unit
	// CATCHES: s05, s06, s07
	// CHAPTER: gw.02 section 2.4
	infer := auth.Principal{Scopes: []string{"infer"}}
	embed := auth.Principal{Scopes: []string{"embed"}}
	admin := auth.Principal{Scopes: []string{"admin"}}
	pinned := auth.Principal{Scopes: []string{"infer", "embed"}, Models: []string{"smol-135m"}}
	for _, tc := range []struct {
		p            auth.Principal
		route, model string
		ok           bool
	}{
		{infer, "/v1/chat/completions", "any", true},
		{infer, "/v1/completions", "any", true},
		{infer, "/v1/models", "", true},
		{infer, "/v1/embeddings", "any", false},
		{infer, "/admin/v1/keys", "", false},
		{embed, "/v1/embeddings", "e5", true},
		{embed, "/v1/chat/completions", "any", false},
		{admin, "/admin/v1/keys", "", true},
		{admin, "/v1/chat/completions", "any", false},
		{pinned, "/v1/chat/completions", "smol-135m", true},
		{pinned, "/v1/chat/completions", "tinystories-10m", false},
		{pinned, "/v1/embeddings", "tinystories-10m", false},
		{pinned, "/v1/models", "", true},
	} {
		err := auth.Authorize(tc.p, tc.route, tc.model)
		if tc.ok && err != nil {
			t.Errorf("%v %s %q: denied (%v), want allowed", tc.p.Scopes, tc.route, tc.model, err)
		}
		if !tc.ok && !errors.Is(err, auth.ErrInsufficient) {
			t.Errorf("%v %v %s %q: %v, want ErrInsufficient", tc.p.Scopes, tc.p.Models, tc.route, tc.model, err)
		}
	}
}

// chain is the gateway with only the authn stage and a recording proxy.
type seen struct {
	header http.Header
	p      auth.Principal
	ok     bool
}

func chain(t *testing.T, s *auth.Store, logger *slog.Logger) (http.Handler, *seen) {
	sn := &seen{}
	proxy := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sn.header = r.Header.Clone()
		sn.p, sn.ok = auth.PrincipalFrom(r.Context())
		io.Copy(io.Discard, r.Body)
		w.WriteHeader(200)
	})
	return server.New(server.Config{}, server.Deps{Keys: auth.Middleware(s, logger), Proxy: proxy}).Handler(), sn
}

func call(h http.Handler, path, key, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest("POST", path, strings.NewReader(body))
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	req.Header.Set("X-TL-Priority", "100") // a client trying to jump the queue
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func errorOf(t *testing.T, rec *httptest.ResponseRecorder) (string, string) {
	t.Helper()
	var e struct {
		Error struct {
			Type string  `json:"type"`
			Code *string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &e); err != nil || e.Error.Code == nil {
		t.Fatalf("not an error body with a code: %q", rec.Body.String())
	}
	return e.Error.Type, *e.Error.Code
}

func TestMiddleware401And403(t *testing.T) {
	// WHY: the stage's answers: no or bad key is 401 invalid_api_key, a key
	//      without the scope or model is 403 insufficient_scope, and a good key
	//      reaches the proxy with its Principal, without Authorization, and
	//      with X-TL-Priority from the key (never from the client).
	// KIND: unit, conformance
	// CATCHES: s13, s14, s15
	// CHAPTER: gw.02 section 4
	s := open(t, writeKeys(t, handRecord()), clock.NewFake(t0), 0)
	h, sn := chain(t, s, nil)
	body := `{"model":"smol-135m","messages":[]}`
	for _, tc := range []struct {
		name, path, key, body string
		status                int
		typ, code             string
	}{
		{"no key", "/v1/chat/completions", "", body, 401, "invalid_request_error", "invalid_api_key"},
		{"bad key", "/v1/chat/completions", "tl_" + handID + "_" + strings.Repeat("A", 32), body, 401, "invalid_request_error", "invalid_api_key"},
		{"no embed scope", "/v1/embeddings", handKey, `{"model":"smol-135m","input":"x"}`, 403, "permission_error", "insufficient_scope"},
		{"model not allowed", "/v1/chat/completions", handKey, `{"model":"tinystories-10m"}`, 403, "permission_error", "insufficient_scope"},
	} {
		rec := call(h, tc.path, tc.key, tc.body)
		if rec.Code != tc.status {
			t.Fatalf("%s: status %d, want %d", tc.name, rec.Code, tc.status)
		}
		if typ, code := errorOf(t, rec); typ != tc.typ || code != tc.code {
			t.Fatalf("%s: %s/%s, want %s/%s", tc.name, typ, code, tc.typ, tc.code)
		}
	}
	rec := call(h, "/v1/chat/completions", handKey, body)
	if rec.Code != 200 || !sn.ok || sn.p.KeyID != handID || sn.p.Tenant != "acme" {
		t.Fatalf("good key: status %d, principal %+v %v", rec.Code, sn.p, sn.ok)
	}
	if sn.header.Get("Authorization") != "" {
		t.Fatal("the gateway key was forwarded to the engine")
	}
	if got := sn.header.Get("X-TL-Priority"); got != "5" {
		t.Fatalf("X-TL-Priority = %q, want the key's priority 5", got)
	}
	if rec := call(h, "/v1/chat/completions", handKey, `{"model":`); rec.Code != 400 {
		t.Fatalf("an unreadable body after a good key: %d, want 400", rec.Code)
	}
}

func TestNoPlaintextKeyInLogs(t *testing.T) {
	// WHY: logs are copied, shipped, and kept for months; a key that reaches
	//      them is leaked. Every log line names key ids only.
	// KIND: property
	// CATCHES: s11
	// CHAPTER: gw.02 section 5, Pitfalls, item 5
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))
	path := writeKeys(t, handRecord())
	s, err := auth.NewStore(path, []byte(pepper), auth.Options{Clock: clock.NewFake(t0), Logger: logger})
	if err != nil {
		t.Fatal(err)
	}
	h, _ := chain(t, s, logger)
	wrong := "tl_" + handID + "_" + strings.Repeat("Q", 32)
	for _, k := range []string{handKey, wrong, "tl_garbage_" + handSec} {
		call(h, "/v1/chat/completions", k, `{"model":"smol-135m"}`)
		call(h, "/v1/embeddings", k, `{"model":"smol-135m"}`)
	}
	plain, _, _ := s.Create(context.Background(), auth.KeySpec{Tenant: "acme", Scopes: []string{"infer"}})
	_, secret, _ := auth.ParseKey(plain)
	s.Revoke(context.Background(), plain[3:15])
	logs := buf.String()
	if !strings.Contains(logs, handID) {
		t.Fatalf("the failures should be logged with the key id; logs:\n%s", logs)
	}
	for _, leak := range []string{handSec, strings.Repeat("Q", 32), secret, pepper} {
		if strings.Contains(logs, leak) {
			t.Fatalf("a secret (%s...) appears in the logs:\n%s", leak[:6], logs)
		}
	}
}

func TestKeysAdminHandler(t *testing.T) {
	// WHY: the admin API's keys paths: POST returns the plaintext once (201,
	//      KeyCreated with the defaults rpm 60 and tpm 100000), GET lists
	//      KeyInfo without any HMAC, DELETE revokes (204) or says 404.
	// KIND: unit, conformance
	// CATCHES: s17
	// CHAPTER: gw.02 section 4
	s := open(t, filepath.Join(t.TempDir(), "keys.jsonl"), clock.NewFake(t0), 0)
	h := auth.KeysHandler(s)
	do := func(method, path, body string) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(method, path, strings.NewReader(body)))
		return rec
	}
	rec := do("POST", "/admin/v1/keys", `{"tenant":"acme","name":"ci","scopes":["infer"]}`)
	var created struct {
		Key   string `json:"key"`
		KeyID string `json:"key_id"`
		RPM   int    `json:"rpm"`
		TPM   int    `json:"tpm"`
		HMAC  string `json:"hmac"`
	}
	if rec.Code != 201 || json.Unmarshal(rec.Body.Bytes(), &created) != nil || created.RPM != 60 || created.TPM != 100000 || created.HMAC != "" {
		t.Fatalf("POST: %d %s", rec.Code, rec.Body.String())
	}
	if _, err := s.Lookup(context.Background(), created.Key); err != nil {
		t.Fatalf("the returned key does not authenticate: %v", err)
	}
	for _, bad := range []string{`{"tenant":"Acme!","name":"x","scopes":["infer"]}`, `{"tenant":"acme","name":"x","scopes":[]}`, `{"tenant":"acme","name":"x","scopes":["root"]}`} {
		if rec := do("POST", "/admin/v1/keys", bad); rec.Code != 400 {
			t.Fatalf("POST %s: %d, want 400", bad, rec.Code)
		}
	}
	rec = do("GET", "/admin/v1/keys?tenant=acme", "")
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), created.KeyID) || strings.Contains(rec.Body.String(), "hmac") {
		t.Fatalf("GET: %d %s", rec.Code, rec.Body.String())
	}
	if rec := do("DELETE", "/admin/v1/keys/"+created.KeyID, ""); rec.Code != 204 {
		t.Fatalf("DELETE: %d", rec.Code)
	}
	if _, err := s.Lookup(context.Background(), created.Key); !errors.Is(err, auth.ErrInvalidKey) {
		t.Fatal("a key revoked through the admin API still works")
	}
	if rec := do("DELETE", "/admin/v1/keys/nosuch2key77", ""); rec.Code != 404 {
		t.Fatalf("DELETE unknown: %d", rec.Code)
	}
}
