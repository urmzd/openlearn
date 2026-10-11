package gw_07

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	controlv1 "openlearn.urmzd.com/tl/contracts/gen/tl/control/v1"
	"openlearn.urmzd.com/tl/testkit/clock"
	"tinyllm/config"
	assemble "tinyllm/gateway"
	"tinyllm/gateway/auth"
	"tinyllm/gateway/cache"
	"tinyllm/gateway/ledger"
	"tinyllm/gateway/limit"
	"tinyllm/gateway/route"
	"tinyllm/gateway/server"
)

// serve sends one request with the bearer key through h and returns the
// recorder.
func serve(h http.Handler, method, target, key, body string, hdr map[string]string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, target, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestGatewayAssemblesTheStages(t *testing.T) {
	// WHY: gateway.Deps and gateway.Admin are the assembly your go/cmd/gateway
	//      calls: real gw.02 keys, gw.03 limits, gw.06 cache, gw.05 router
	//      and registry, and this module's ledger, in the contract's order.
	//      A repeated deterministic request is a cache hit that still spends
	//      the key's RPM (ratelimit runs before cache); a PUT of the route
	//      table moves the cache revision, so the next request misses; the
	//      4th request of an RPM-3 key is a 429 even though its answer is
	//      cached; and every admin path reaches its module's handler.
	// KIND: unit
	// CATCHES: s23, m09
	// CHAPTER: gw.07 section 4, The interface
	ctx := context.Background()
	fake := clock.NewFake(t0)
	store, err := auth.NewStore(filepath.Join(t.TempDir(), "keys.jsonl"), []byte("pepper"), auth.Options{Clock: fake})
	if err != nil {
		t.Fatal(err)
	}
	user, _, err := store.Create(ctx, auth.KeySpec{Tenant: "acme", Name: "app", Scopes: []string{"infer"}, RPM: 3})
	if err != nil {
		t.Fatal(err)
	}
	root, _, err := store.Create(ctx, auth.KeySpec{Tenant: "ops", Name: "root", Scopes: []string{"admin"}})
	if err != nil {
		t.Fatal(err)
	}
	reg := route.NewRegistry(fake, 3)
	if _, err := reg.Heartbeat(ctx, &controlv1.WorkerStatus{WorkerId: "w1", Role: "unified", Model: "smol",
		HttpAddress: "127.0.0.1:1", KvTotalBlocks: 8, KvFreeBlocks: 8, KvFormat: 1}); err != nil {
		t.Fatal(err)
	}
	rt := route.NewRouter(reg, route.PolicyLeastOutstanding, 1.25, []config.Route{{Model: "smol"}})
	db, _ := open(t)
	var upstream atomic.Int32
	engine := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstream.Add(1)
		server.ExchangeFrom(r.Context()).SetUsage(server.Usage{PromptTokens: 4, CompletionTokens: 2, TotalTokens: 6})
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"c1","object":"chat.completion","model":"smol","choices":[{"index":0,`+
			`"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}`)
	})
	st := assemble.Stages{
		Keys: store, Limiter: limit.New(fake), Cache: cache.New(16, fake),
		CacheOptions: cache.Options{TTL: time.Hour}, Registry: reg, Router: rt, Proxy: engine,
		Ledger: db, Meter: ledger.MeterOptions{Clock: fake},
	}
	api := server.New(server.Config{}, assemble.Deps(st)).Handler()
	admin := server.New(server.Config{}, server.Deps{Keys: auth.Middleware(store, nil),
		Proxy: ledger.Admin(assemble.Admin(st))}).Handler()

	const chat = `{"model":"smol","messages":[{"role":"user","content":"hello"}],"temperature":0}`
	infer := func(step, wantCache string, wantUpstream int32) {
		t.Helper()
		rec := serve(api, "POST", "/v1/chat/completions", user, chat, nil)
		if rec.Code != 200 || rec.Header().Get("X-TL-Cache") != wantCache || upstream.Load() != wantUpstream {
			t.Fatalf("%s: %d X-TL-Cache %q, %d upstream calls; want 200 %q and %d (body %s)",
				step, rec.Code, rec.Header().Get("X-TL-Cache"), upstream.Load(), wantCache, wantUpstream, rec.Body)
		}
		if rec.Header().Get("x-ratelimit-limit-requests") != "3" {
			t.Fatalf("%s: x-ratelimit-limit-requests %q, want 3: the limiter is not in the chain",
				step, rec.Header().Get("x-ratelimit-limit-requests"))
		}
	}
	infer("first request", "miss", 1)
	infer("same request", "hit", 1)

	rec := serve(admin, "GET", "/admin/v1/workers", root, "", nil)
	var workers struct {
		Data []struct {
			ID string `json:"worker_id"`
		} `json:"data"`
		RouteEpoch uint64 `json:"route_epoch"`
	}
	if rec.Code != 200 || json.Unmarshal(rec.Body.Bytes(), &workers) != nil || len(workers.Data) != 1 || workers.Data[0].ID != "w1" {
		t.Fatalf("GET /admin/v1/workers: %d %s; want worker w1 from the registry", rec.Code, rec.Body)
	}
	rec = serve(admin, "GET", "/admin/v1/routes", root, "", nil)
	etag := rec.Header().Get("ETag")
	if rec.Code != 200 || etag != route.ETag(workers.RouteEpoch) {
		t.Fatalf("GET /admin/v1/routes: %d ETag %q; want 200 %q", rec.Code, etag, route.ETag(workers.RouteEpoch))
	}
	rec = serve(admin, "PUT", "/admin/v1/routes", root, `{"routes":[{"model":"smol","aliases":["small"]}]}`,
		map[string]string{"If-Match": etag})
	if rec.Code != 200 || rec.Header().Get("ETag") == etag {
		t.Fatalf("PUT /admin/v1/routes: %d ETag %q; want 200 and a new ETag (body %s)", rec.Code, rec.Header().Get("ETag"), rec.Body)
	}
	infer("same request after the route table changed", "miss", 2)

	rec = serve(api, "POST", "/v1/chat/completions", user, chat, nil)
	if rec.Code != 429 || rec.Header().Get("Retry-After") == "" || upstream.Load() != 2 {
		t.Fatalf("4th request of an RPM-3 key: %d Retry-After %q, %d upstream calls; want 429 before the cache answers",
			rec.Code, rec.Header().Get("Retry-After"), upstream.Load())
	}

	rec = serve(admin, "POST", "/admin/v1/cache:purge", root, `{"model":"smol"}`, nil)
	if rec.Code != 200 || strings.TrimSpace(rec.Body.String()) != `{"purged":2}` {
		t.Fatalf("POST /admin/v1/cache:purge: %d %s; want {\"purged\":2} (one answer per route epoch)", rec.Code, rec.Body)
	}
	rec = serve(admin, "GET", "/admin/v1/usage?tenant=acme", root, "", nil)
	var usage struct {
		Data []ledger.UsageRow `json:"data"`
	}
	if rec.Code != 200 || json.Unmarshal(rec.Body.Bytes(), &usage) != nil || len(usage.Data) != 1 {
		t.Fatalf("GET /admin/v1/usage: %d %s", rec.Code, rec.Body)
	}
	// The meter wraps the proxy (gw.01's order): the two answers the engine
	// gave are rows; the cache hit and the 429 never reached it.
	if got := usage.Data[0]; got.Requests != 2 || got.PromptTokens != 8 || got.CompletionTokens != 4 {
		t.Fatalf("acme usage: %+v; want the 2 upstream answers (8 prompt, 4 completion tokens): the meter is not in the chain", got)
	}
	if rec = serve(admin, "GET", "/admin/v1/keys", root, "", nil); rec.Code != 200 {
		t.Fatalf("GET /admin/v1/keys: %d %s; want gw.02's keys handler", rec.Code, rec.Body)
	}
	if rec = serve(admin, "POST", "/admin/v1/models/smol:drain", root, `{"deadline_s":30}`, nil); rec.Code != 202 {
		t.Fatalf("POST /admin/v1/models/smol:drain: %d %s; want 202 from gw.05's drain handler", rec.Code, rec.Body)
	}
	if rec = serve(admin, "GET", "/admin/v1/policy", root, "", nil); rec.Code != 404 {
		t.Fatalf("GET /admin/v1/policy with no policy stage: %d; want 404 not_found", rec.Code)
	}
}
