package gw_07

import (
	"bufio"
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"openlearn.urmzd.com/tl/testkit/clock"
	"tinyllm/gateway/auth"
	"tinyllm/gateway/ledger"
	"tinyllm/gateway/server"
)

const patience = 3 * time.Second

// engineUsage is what the fake engine reports for one request.
type engineUsage struct{ prompt, completion, cached int }

// fakeEngine answers the three metered routes the way openai-subset.v1.yaml
// says an engine does, with the usage plan(body) returns. A stream carries
// its usage chunk only when the request asked for it (include_usage).
type fakeEngine struct {
	srv  *httptest.Server
	plan func(path string, body map[string]any) (engineUsage, int) // usage, status (0 = 200)
	hold chan struct{}                                             // when set, a stream waits on it after its first event

	mu     sync.Mutex
	bodies []string
	total  map[string]engineUsage // by tenant (from the request's "user" field)
}

func newEngine(t *testing.T, plan func(string, map[string]any) (engineUsage, int)) *fakeEngine {
	t.Helper()
	e := &fakeEngine{plan: plan, total: map[string]engineUsage{}}
	e.srv = httptest.NewServer(http.HandlerFunc(e.serve))
	t.Cleanup(e.srv.Close)
	return e
}

func (e *fakeEngine) serve(w http.ResponseWriter, r *http.Request) {
	b, _ := io.ReadAll(r.Body)
	var body map[string]any
	_ = json.Unmarshal(b, &body)
	e.mu.Lock()
	e.bodies = append(e.bodies, string(b))
	e.mu.Unlock()
	u, status := e.plan(r.URL.Path, body)
	if status >= 400 {
		code := map[int]string{400: "", 429: "rate_limit_exceeded", 503: "no_capacity"}[status]
		typ := map[int]string{400: "invalid_request_error", 429: "rate_limit_error", 503: "server_error"}[status]
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		c := "null"
		if code != "" {
			c = `"` + code + `"`
		}
		fmt.Fprintf(w, `{"error":{"message":"refused","type":%q,"param":null,"code":%s}}`, typ, c)
		return
	}
	if tenant, _ := body["user"].(string); tenant != "" {
		e.mu.Lock()
		s := e.total[tenant]
		e.total[tenant] = engineUsage{s.prompt + u.prompt, s.completion + u.completion, s.cached + u.cached}
		e.mu.Unlock()
	}
	usage := fmt.Sprintf(`{"prompt_tokens":%d,"completion_tokens":%d,"total_tokens":%d`, u.prompt, u.completion, u.prompt+u.completion)
	if u.cached > 0 {
		usage += fmt.Sprintf(`,"prompt_tokens_details":{"cached_tokens":%d}`, u.cached)
	}
	usage += "}"
	served := "smol-135m@v3"
	switch {
	case r.URL.Path == "/v1/embeddings":
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"object":"list","model":%q,"data":[{"object":"embedding","index":0,"embedding":[0.6,0.8]}],"usage":{"prompt_tokens":%d,"total_tokens":%d}}`,
			served, u.prompt, u.prompt)
	case body["stream"] == true:
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(200)
		fl := w.(http.Flusher)
		for _, ev := range streamEvents(served, usage, includeUsage(body)) {
			if _, err := io.WriteString(w, ev); err != nil {
				return
			}
			fl.Flush()
			if e.hold != nil && strings.Contains(ev, `"role"`) {
				select {
				case <-e.hold:
				case <-r.Context().Done():
					return
				}
			}
		}
	default:
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"id":"chatcmpl-1","object":"chat.completion","created":1760000000,"model":%q,"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":%s}`,
			served, usage)
	}
}

func includeUsage(body map[string]any) bool {
	so, _ := body["stream_options"].(map[string]any)
	return so["include_usage"] == true
}

func streamEvents(served, usage string, withUsage bool) []string {
	pre := `data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1760000000,"model":"` + served + `","choices":`
	evs := []string{
		pre + `[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}` + "\n\n",
		pre + `[{"index":0,"delta":{"content":"Once"},"finish_reason":null}]}` + "\n\n",
		pre + `[{"index":0,"delta":{"content":" upon"},"finish_reason":"length"}]}` + "\n\n",
	}
	if withUsage {
		evs = append(evs, pre+`[],"usage":`+usage+"}\n\n")
	}
	return append(evs, "data: [DONE]\n\n")
}

// testProxy forwards the (possibly rewritten) request body to the engine and
// streams the answer back, flushing every read: a stand-in for gw.04's proxy.
func testProxy(upstream string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		req, err := http.NewRequestWithContext(r.Context(), r.Method, upstream+r.URL.Path, r.Body)
		if err != nil {
			server.WriteError(w, 500, "server_error", "", "", err.Error())
			return
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultTransport.RoundTrip(req)
		if err != nil {
			server.WriteError(w, 503, "server_error", "no_capacity", "", err.Error())
			return
		}
		defer resp.Body.Close()
		for k, v := range resp.Header {
			w.Header()[k] = v
		}
		w.Header().Del("Content-Length")
		w.WriteHeader(resp.StatusCode)
		rc := http.NewResponseController(w)
		buf := make([]byte, 4096)
		for {
			n, err := resp.Body.Read(buf)
			if n > 0 {
				if _, werr := w.Write(buf[:n]); werr != nil {
					return
				}
				_ = rc.Flush()
			}
			if err != nil {
				return
			}
		}
	})
}

// principals maps the test's bearer tokens to principals: a stand-in for
// gw.02's authn stage.
func fakeAuthn(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		tok := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		tenant, key, ok := strings.Cut(tok, "/")
		if !ok {
			server.WriteError(w, 401, "invalid_request_error", "invalid_api_key", "", "no key")
			return
		}
		r.Header.Del("Authorization")
		next.ServeHTTP(w, r.WithContext(auth.WithPrincipal(r.Context(), auth.Principal{KeyID: key, Tenant: tenant, Scopes: []string{"infer"}})))
	})
}

type chainOpts struct {
	router server.Middleware
	clock  server.Clock
}

// gateway is gw.01's chain with a fake authn, your Meter, and the test proxy.
func gateway(t *testing.T, l ledger.Ledger, upstream string, o chainOpts) *httptest.Server {
	t.Helper()
	var meterClock server.Clock
	if o.clock != nil {
		meterClock = o.clock
	}
	s := server.New(server.Config{}, server.Deps{
		Keys:   fakeAuthn,
		Router: o.router,
		Proxy:  testProxy(upstream),
		Ledger: ledger.Meter(l, ledger.MeterOptions{Clock: meterClock}),
		Clock:  o.clock,
	})
	g := httptest.NewServer(s.Handler())
	t.Cleanup(g.Close)
	return g
}

func post(t *testing.T, url, path, auth string, body string) (*http.Response, string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), patience)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "POST", url+path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+auth)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST %s: %v", path, err)
	}
	b, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	return resp, string(b)
}

// row reads one usage row by request id through the raw connection.
type usageRow struct {
	tenant, key, model, served, route, worker string
	status, stream, p, c, ca, hit             int
	code                                      sql.NullString
	ttft                                      sql.NullFloat64
	e2e                                       float64
}

func rowOf(t *testing.T, db *sql.DB, id string) (usageRow, bool) {
	t.Helper()
	var u usageRow
	err := db.QueryRow(`SELECT tenant, key_id, model, served_model, route, worker_id, status, stream, prompt_tokens,
		completion_tokens, cached_tokens, cache_hit, error_code, ttft_ms, e2e_ms FROM usage WHERE request_id = ?`, id).
		Scan(&u.tenant, &u.key, &u.model, &u.served, &u.route, &u.worker, &u.status, &u.stream, &u.p, &u.c, &u.ca,
			&u.hit, &u.code, &u.ttft, &u.e2e)
	if err == sql.ErrNoRows {
		return u, false
	}
	if err != nil {
		t.Fatal(err)
	}
	return u, true
}

func onlyRow(t *testing.T, db *sql.DB) usageRow {
	t.Helper()
	var id string
	var n int
	if err := db.QueryRow("SELECT COUNT(*), COALESCE(MAX(request_id), '') FROM usage").Scan(&n, &id); err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("the ledger holds %d rows after one request, want 1", n)
	}
	u, _ := rowOf(t, db, id)
	return u
}

func TestLedgerSumEqualsEngineUsage(t *testing.T) {
	// WHY: the ledger is the bill: over 1000 requests (chat, completions, and
	//      embeddings; streamed and not; some with usage requested by the
	//      client, some refused by the engine) the per-tenant sums in the
	//      ledger must equal the usage the engine reported, token for token.
	//      A meter that misses streamed usage, double counts, or bills a
	//      refused request fails here (MS-gateway reconciles the same way).
	// KIND: property
	// CATCHES: s02, s15
	// CHAPTER: gw.07 section 2.1
	db, _ := open(t)
	tenants := []string{"acme", "globex", "initech"}
	eng := newEngine(t, func(path string, body map[string]any) (engineUsage, int) {
		var i int
		fmt.Sscanf(body["model"].(string), "m%d", &i)
		if i%17 == 5 {
			return engineUsage{}, 400
		}
		u := engineUsage{prompt: (i*7)%50 + 1, completion: (i * 13) % 40}
		if i%5 == 0 {
			u.cached = u.prompt / 2
		}
		if path == "/v1/embeddings" {
			u.completion, u.cached = 0, 0
		}
		return u, 0
	})
	g := gateway(t, db, eng.srv.URL, chainOpts{})
	routes := []string{"/v1/chat/completions", "/v1/completions", "/v1/embeddings"}
	var wg sync.WaitGroup
	work := make(chan int)
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range work {
				tenant := tenants[i%3]
				route := routes[(i/3)%len(routes)]
				fields := fmt.Sprintf(`"model":"m%d","user":%q`, i, tenant)
				switch {
				case route == "/v1/embeddings":
					fields += `,"input":"hello"`
				case i%4 == 0:
					fields += `,"stream":true`
				case i%4 == 1:
					fields += `,"stream":true,"stream_options":{"include_usage":true}`
				}
				if route == "/v1/chat/completions" {
					fields += `,"messages":[{"role":"user","content":"hi"}]`
				} else if route == "/v1/completions" {
					fields += `,"prompt":"hi"`
				}
				resp, body := post(t, g.URL, route, tenant+"/"+tenant+"key", "{"+fields+"}")
				if resp.StatusCode != 200 && resp.StatusCode != 400 {
					t.Errorf("request %d: status %d: %s", i, resp.StatusCode, body)
				}
			}
		}()
	}
	for i := 0; i < 1000; i++ {
		work <- i
	}
	close(work)
	wg.Wait()

	rows := query(t, db, ledger.UsageFilter{GroupBy: "tenant"})
	if len(rows) != 3 {
		t.Fatalf("group_by tenant gave %d rows, want 3: %+v", len(rows), rows)
	}
	var requests int64
	for _, r := range rows {
		want := eng.total[r.Tenant]
		requests += r.Requests
		if r.PromptTokens != int64(want.prompt) || r.CompletionTokens != int64(want.completion) || r.CachedTokens != int64(want.cached) {
			t.Fatalf("tenant %s: ledger %d/%d/%d prompt/completion/cached, engine reported %d/%d/%d",
				r.Tenant, r.PromptTokens, r.CompletionTokens, r.CachedTokens, want.prompt, want.completion, want.cached)
		}
	}
	if requests != 1000 {
		t.Fatalf("the ledger counts %d requests, the gateway served 1000", requests)
	}
	all := query(t, db, ledger.UsageFilter{})
	if all[0].Errors != 59 {
		t.Fatalf("the ledger counts %d errors, the engine refused 59 requests", all[0].Errors)
	}
}

func TestStreamUsageIsRequestedAndHidden(t *testing.T) {
	// WHY: an engine sends a stream's usage only when the request sets
	//      stream_options.include_usage. The meter sets it for every stream so
	//      the ledger gets tokens, and hides that usage chunk from a client
	//      that never asked for it; a client that did ask gets the engine's
	//      stream byte for byte, usage chunk included.
	// KIND: unit
	// CATCHES: s02, s03
	// CHAPTER: gw.07 section 3, Worked example by hand
	db, path := open(t)
	eng := newEngine(t, func(string, map[string]any) (engineUsage, int) { return engineUsage{12, 30, 0}, 0 })
	g := gateway(t, db, eng.srv.URL, chainOpts{})

	_, got := post(t, g.URL, "/v1/chat/completions", "acme/acmekeyaaaaa",
		`{"model":"smol","stream":true,"messages":[{"role":"user","content":"Once"}]}`)
	if !strings.Contains(eng.bodies[0], `"include_usage":true`) {
		t.Fatalf("the engine received %s: no stream_options.include_usage, so it sends no usage", eng.bodies[0])
	}
	want := strings.Join(streamEvents("smol-135m@v3", "", false), "")
	if got != want {
		t.Fatalf("the client, which did not ask for usage, received:\n%s\nwant (no usage chunk):\n%s", got, want)
	}
	u := onlyRow(t, raw(t, path))
	if u.p != 12 || u.c != 30 || u.stream != 1 {
		t.Fatalf("ledger row: prompt %d completion %d stream %d, want 12, 30, 1", u.p, u.c, u.stream)
	}

	asked := `{"model":"smol","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"Once"}]}`
	_, got = post(t, g.URL, "/v1/chat/completions", "acme/acmekeyaaaaa", asked)
	if eng.bodies[1] != asked {
		t.Fatalf("a request that already asks for usage must reach the engine unchanged:\n got %s\nwant %s", eng.bodies[1], asked)
	}
	usage := `{"prompt_tokens":12,"completion_tokens":30,"total_tokens":42}`
	if want := strings.Join(streamEvents("smol-135m@v3", usage, true), ""); got != want {
		t.Fatalf("a client that asked for usage must get the stream byte for byte:\n got %q\nwant %q", got, want)
	}
}

func TestMeterDoesNotBufferTheStream(t *testing.T) {
	// WHY: the meter sits between the proxy and the client. If it holds bytes
	//      back to parse them, the first token waits for the whole answer and
	//      time to first token, the gateway's headline SLO, is ruined.
	// KIND: unit
	// CATCHES: s20
	// CHAPTER: gw.07 section 5, Pitfall 5
	db, _ := open(t)
	eng := newEngine(t, func(string, map[string]any) (engineUsage, int) { return engineUsage{3, 2, 0}, 0 })
	eng.hold = make(chan struct{})
	g := gateway(t, db, eng.srv.URL, chainOpts{})
	ctx, cancel := context.WithTimeout(context.Background(), patience)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "POST", g.URL+"/v1/chat/completions",
		strings.NewReader(`{"model":"smol","stream":true,"messages":[]}`))
	req.Header.Set("Authorization", "Bearer acme/acmekeyaaaaa")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		close(eng.hold)
		t.Fatalf("no response headers while the engine holds its second event: %v", err)
	}
	defer resp.Body.Close()
	line, err := bufio.NewReader(resp.Body).ReadString('\n')
	close(eng.hold)
	if err != nil || !strings.Contains(line, `"role":"assistant"`) {
		t.Fatalf("the first event did not reach the client while the engine held the rest: %q, %v", line, err)
	}
}

func TestRefusedRequestsAreRecordedWithTheirCode(t *testing.T) {
	// WHY: rows for refused requests are how ops sees a tenant hammering a
	//      limit: the status and the error's code are recorded (its type when
	//      the code is null), with zero tokens, no served model, and no TTFT.
	//      A stream that fails after its first byte has status 200 but carries
	//      the error event's code, so it still counts as an error.
	// KIND: unit
	// CATCHES: s12, s15, s22, m03
	// CHAPTER: gw.07 section 2.1
	db, path := open(t)
	status := 429
	eng := newEngine(t, func(string, map[string]any) (engineUsage, int) { return engineUsage{5, 5, 0}, status })
	g := gateway(t, db, eng.srv.URL, chainOpts{})
	r := raw(t, path)

	post(t, g.URL, "/v1/completions", "acme/acmekeyaaaaa", `{"model":"tiny","prompt":"x"}`)
	u := onlyRow(t, r)
	if u.status != 429 || u.code.String != "rate_limit_exceeded" || u.p != 0 || u.c != 0 || u.served != "" || u.ttft.Valid {
		t.Fatalf("429 row: status %d code %v tokens %d/%d served %q ttft %v; want 429, rate_limit_exceeded, 0/0, \"\", NULL",
			u.status, u.code, u.p, u.c, u.served, u.ttft)
	}
	if _, err := r.Exec("DELETE FROM usage"); err != nil {
		t.Fatal(err)
	}
	status = 400
	post(t, g.URL, "/v1/completions", "acme/acmekeyaaaaa", `{"model":"tiny","prompt":"x"}`)
	if u := onlyRow(t, r); u.status != 400 || u.code.String != "invalid_request_error" {
		t.Fatalf("400 with a null code: status %d code %v, want 400 and the type invalid_request_error", u.status, u.code)
	}
	if _, err := r.Exec("DELETE FROM usage"); err != nil {
		t.Fatal(err)
	}

	// An upstream that breaks after the first event: the proxy appends an
	// error event (gw.00's rule), the status stays 200.
	broken := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, streamEvents("smol", "", false)[0])
		io.WriteString(w, `data: {"error":{"message":"upstream failed mid-stream","type":"server_error","param":null,"code":"upstream_error"}}`+"\n\n")
	}))
	t.Cleanup(broken.Close)
	g2 := gateway(t, db, broken.URL, chainOpts{})
	post(t, g2.URL, "/v1/chat/completions", "acme/acmekeyaaaaa", `{"model":"smol","stream":true,"messages":[]}`)
	if u := onlyRow(t, r); u.status != 200 || u.code.String != "upstream_error" {
		t.Fatalf("mid-stream failure: status %d code %v, want 200 and upstream_error", u.status, u.code)
	}
	if rows := query(t, db, ledger.UsageFilter{}); rows[0].Errors != 1 {
		t.Fatalf("the mid-stream failure counts as %d errors, want 1", rows[0].Errors)
	}
}

func TestMeterRecordsWhoWhatAndWhere(t *testing.T) {
	// WHY: each column answers an ops question: tenant and key_id from the
	//      authenticated Principal (who pays), the model the client asked for
	//      and the one that answered (an alias or a canary differ), the route,
	//      the worker the router chose, the cache flag, the trace id, and the
	//      latency measured on the gateway's clock.
	// KIND: unit
	// CATCHES: s18
	// CHAPTER: gw.07 section 2.1
	db, path := open(t)
	eng := newEngine(t, func(string, map[string]any) (engineUsage, int) { return engineUsage{4, 6, 0}, 0 })
	fake := clock.NewFake(t0)
	router := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			ex := server.ExchangeFrom(r.Context())
			ex.SetWorker("w-7")
			ex.SetAttr("tl.cache.hit", true)
			fake.Advance(250 * time.Millisecond)
			next.ServeHTTP(w, r)
		})
	}
	g := gateway(t, db, eng.srv.URL, chainOpts{router: router, clock: fake})
	ctx, cancel := context.WithTimeout(context.Background(), patience)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "POST", g.URL+"/v1/chat/completions", strings.NewReader(`{"model":"smol","messages":[]}`))
	req.Header.Set("Authorization", "Bearer acme/acmekeyaaaaa")
	req.Header.Set("traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")
	req.Header.Set("X-Request-Id", "req-who")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	u, ok := rowOf(t, raw(t, path), "req-who")
	if !ok {
		t.Fatal("no row under the request's X-Request-Id req-who")
	}
	var trace string
	_ = raw(t, path).QueryRow("SELECT trace_id FROM usage WHERE request_id = 'req-who'").Scan(&trace)
	if u.tenant != "acme" || u.key != "acmekeyaaaaa" || u.model != "smol" || u.served != "smol-135m@v3" ||
		u.route != "/v1/chat/completions" || u.worker != "w-7" || u.hit != 1 || u.stream != 0 ||
		trace != "4bf92f3577b34da6a3ce929d0e0e4736" || u.e2e != 250 {
		t.Fatalf("row: tenant %q key %q model %q served %q route %q worker %q cache_hit %d stream %d trace %q e2e %v",
			u.tenant, u.key, u.model, u.served, u.route, u.worker, u.hit, u.stream, trace, u.e2e)
	}
}

func TestClientDisconnectStillRecords(t *testing.T) {
	// WHY: a client that hangs up mid-stream still used the tokens the engine
	//      produced. The request's context is cancelled at that moment, so a
	//      ledger write made on it fails and the row is silently lost: the
	//      write must use a context that outlives the request.
	// KIND: fault
	// CATCHES: s01
	// CHAPTER: gw.07 section 5, Pitfall 2
	db, path := open(t)
	eng := newEngine(t, func(string, map[string]any) (engineUsage, int) { return engineUsage{3, 1, 0}, 0 })
	eng.hold = make(chan struct{})
	defer close(eng.hold)
	g := gateway(t, db, eng.srv.URL, chainOpts{})
	ctx, cancel := context.WithTimeout(context.Background(), patience)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "POST", g.URL+"/v1/chat/completions",
		strings.NewReader(`{"model":"smol","stream":true,"messages":[]}`))
	req.Header.Set("Authorization", "Bearer acme/acmekeyaaaaa")
	req.Header.Set("X-Request-Id", "req-gone")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("no response while the engine holds the stream: %v", err)
	}
	if _, err := bufio.NewReader(resp.Body).ReadString('\n'); err != nil {
		t.Fatalf("the first event never arrived: %v", err)
	}
	cancel() // the client hangs up mid-stream
	resp.Body.Close()
	r := raw(t, path)
	deadline := time.Now().Add(patience)
	for {
		if _, ok := rowOf(t, r, "req-gone"); ok {
			return
		}
		if time.Now().After(deadline) {
			t.Fatal("no ledger row for a stream whose client disconnected")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestUnmeteredPathsPassThrough(t *testing.T) {
	// WHY: only the three inference routes are billed; GET /v1/models and
	//      anything else pass through untouched and leave no row.
	// KIND: boundary
	// CATCHES: m08
	// CHAPTER: gw.07 section 2.1
	db, path := open(t)
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"object":"list","data":[]}`)
	}))
	t.Cleanup(up.Close)
	g := gateway(t, db, up.URL, chainOpts{})
	ctx, cancel := context.WithTimeout(context.Background(), patience)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "GET", g.URL+"/v1/models", nil)
	req.Header.Set("Authorization", "Bearer acme/acmekeyaaaaa")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if !bytes.Equal(b, []byte(`{"object":"list","data":[]}`)) {
		t.Fatalf("GET /v1/models body changed: %s", b)
	}
	var n int
	_ = raw(t, path).QueryRow("SELECT COUNT(*) FROM usage").Scan(&n)
	if n != 0 {
		t.Fatalf("GET /v1/models left %d ledger rows, want 0", n)
	}
}
