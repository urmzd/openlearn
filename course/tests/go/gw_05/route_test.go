// Course tests for gw.05: the worker registry, routing policies, failover
// before the first byte, cascades, and disaggregated orchestration
// (go/gateway/route).
//
// Workers are fake engines: httptest servers for the OpenAI surface and
// in-process gRPC servers for tl.engine.v1 and tl.kv.v1. Their "model" is a
// deterministic byte generator, so a unified stream and a disaggregated one
// can be compared token for token. Heartbeats are fed to the registry
// directly (and once over real gRPC); time is the testkit's fake clock.
package gw_05

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	controlv1 "openlearn.urmzd.com/tl/contracts/gen/tl/control/v1"
	enginev1 "openlearn.urmzd.com/tl/contracts/gen/tl/engine/v1"
	kvv1 "openlearn.urmzd.com/tl/contracts/gen/tl/kv/v1"
	"openlearn.urmzd.com/tl/testkit/clock"
	"tinyllm/config"
	"tinyllm/gateway/auth"
	"tinyllm/gateway/route"
	"tinyllm/gateway/server"
)

var t0 = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

// --- the fake model ---------------------------------------------------------

// gen continues ids greedily: a deterministic function of everything so far.
func gen(ids []uint32, n int) []uint32 {
	h := uint64(1469598103934665603)
	for _, x := range ids {
		h = (h ^ uint64(x)) * 1099511628211
	}
	out := make([]uint32, 0, n)
	for i := 0; i < n; i++ {
		next := uint32('a' + h%26)
		out = append(out, next)
		h = (h ^ uint64(next)) * 1099511628211
	}
	return out
}

func bytesOf(s string) []uint32 {
	out := make([]uint32, len(s))
	for i := 0; i < len(s); i++ {
		out[i] = uint32(s[i])
	}
	return out
}

func textOf(ids []uint32) string {
	b := make([]byte, len(ids))
	for i, x := range ids {
		b[i] = byte(x)
	}
	return string(b)
}

type reqBody struct {
	Model     string  `json:"model"`
	Prompt    string  `json:"prompt"`
	MaxTokens int     `json:"max_tokens"`
	Stream    bool    `json:"stream"`
	Seed      *uint64 `json:"seed"`
	Logprobs  any     `json:"logprobs"`
}

// engine is a fake worker's HTTP surface.
type engine struct {
	srv     *httptest.Server
	calls   atomic.Int64
	mode    atomic.Value // "ok" | "503" | "die-after-first"
	logprob float64      // per token, reported when logprobs are asked
	mu      sync.Mutex
	bodies  []reqBody
	handles []string
	resume  func(handle string) ([]uint32, bool) // decode role: the pushed prompt KV
}

func newEngine(t *testing.T) *engine {
	e := &engine{logprob: -0.5}
	e.mode.Store("ok")
	e.srv = httptest.NewServer(http.HandlerFunc(e.serve))
	t.Cleanup(e.srv.Close)
	return e
}

func (e *engine) addr() string { return strings.TrimPrefix(e.srv.URL, "http://") }

func (e *engine) seen() []reqBody {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]reqBody(nil), e.bodies...)
}

func (e *engine) serve(w http.ResponseWriter, r *http.Request) {
	e.calls.Add(1)
	var b reqBody
	json.NewDecoder(r.Body).Decode(&b)
	e.mu.Lock()
	e.bodies = append(e.bodies, b)
	e.handles = append(e.handles, r.Header.Get("X-TL-KV-Handle"))
	e.mu.Unlock()
	if e.mode.Load() == "503" {
		server.WriteError(w, 503, "server_error", "no_capacity", "", "draining")
		return
	}
	prompt := bytesOf(b.Prompt)
	if h := r.Header.Get("X-TL-KV-Handle"); h != "" {
		ids, ok := e.resume(h)
		if !ok {
			server.WriteError(w, 404, "invalid_request_error", "kv_handle_not_found", "", "unknown handle")
			return
		}
		prompt = ids
	}
	out := gen(prompt, b.MaxTokens)
	if !b.Stream {
		lps := make([]map[string]any, len(out))
		for i, x := range out {
			lps[i] = map[string]any{"token": string(rune(x)), "logprob": e.logprob}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id": "c", "object": "text_completion", "model": b.Model,
			"choices": []any{map[string]any{"index": 0, "text": textOf(out), "finish_reason": "length",
				"logprobs": map[string]any{"content": lps}}},
			"usage": map[string]int{"prompt_tokens": len(prompt), "completion_tokens": len(out), "total_tokens": len(prompt) + len(out)},
		})
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	rc := http.NewResponseController(w)
	for i, x := range out {
		fmt.Fprintf(w, "data: {\"id\":\"c\",\"object\":\"text_completion\",\"choices\":[{\"index\":0,\"text\":%q}]}\n\n", string(rune(x)))
		rc.Flush()
		if i == 0 && e.mode.Load() == "die-after-first" {
			panic(http.ErrAbortHandler) // the engine process dies mid-stream
		}
	}
	io.WriteString(w, "data: [DONE]\n\n")
}

// --- registry helpers ---------------------------------------------------------

type worker struct {
	id, role, model string
	http, grpc, kv  string
	queue, cap      int
}

func beat(t *testing.T, reg *route.Registry, w worker) {
	t.Helper()
	_, err := reg.Heartbeat(context.Background(), &controlv1.WorkerStatus{
		WorkerId: w.id, Role: w.role, Model: w.model, HttpAddress: w.http, GrpcAddress: w.grpc,
		KvAddress: w.kv, QueueDepth: int32(w.queue), KvTotalBlocks: int32(w.cap), KvFreeBlocks: int32(w.cap), KvFormat: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
}

func routeOnce(t *testing.T, rt *route.Router, model, reqID string, key []byte) route.Target {
	t.Helper()
	tg, err := rt.Route(context.Background(), &route.InferenceRequest{Model: model, RequestID: reqID, PrefixKey: key}, nil, 0)
	if err != nil {
		t.Fatalf("Route(%s): %v", model, err)
	}
	return tg
}

// gateway is the chain with an optional principal, the route stage, and the
// routing proxy.
func gateway(t *testing.T, rt *route.Router, o route.Options, p *auth.Principal) http.Handler {
	keys := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if p != nil {
				r = r.WithContext(auth.WithPrincipal(r.Context(), *p))
			}
			next.ServeHTTP(w, r)
		})
	}
	return server.New(server.Config{}, server.Deps{Keys: keys, Router: route.Middleware(rt), Proxy: route.Proxy(rt, o)}).Handler()
}

func post(h http.Handler, path, body string) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", path, strings.NewReader(body)))
	return rec
}

func streamText(t *testing.T, sse string) string {
	t.Helper()
	var sb strings.Builder
	for _, ev := range strings.Split(sse, "\n\n") {
		d, ok := strings.CutPrefix(ev, "data: ")
		if !ok || d == "[DONE]" {
			continue
		}
		var c struct {
			Choices []struct {
				Text string `json:"text"`
			} `json:"choices"`
		}
		if json.Unmarshal([]byte(d), &c) == nil && len(c.Choices) > 0 {
			sb.WriteString(c.Choices[0].Text)
		}
	}
	return sb.String()
}

// --- tests ----------------------------------------------------------------------

func TestHandExample(t *testing.T) {
	// WHY: the worked example: queues a=2, b=0, c=1 send least_outstanding to
	//      b; two requests in flight on b make c the least loaded; when b and c
	//      stop heartbeating at t0, they are evicted at t0 + 3 x 2 s.
	// KIND: unit
	// CATCHES: s01, s02
	// CHAPTER: gw.05 section 3, worked example
	fc := clock.NewFake(t0)
	reg := route.NewRegistry(fc, 3)
	for _, w := range []worker{{id: "a", queue: 2}, {id: "b", queue: 0}, {id: "c", queue: 1}} {
		w.role, w.model, w.http = "unified", "smol", "127.0.0.1:1"
		beat(t, reg, w)
	}
	rt := route.NewRouter(reg, route.PolicyLeastOutstanding, 1.25, nil)
	if got := routeOnce(t, rt, "smol", "r1", nil).Worker.ID; got != "b" {
		t.Fatalf("first pick %s, want b (queue 0)", got)
	}
	rt.Acquire("b")
	rt.Acquire("b")
	if got := routeOnce(t, rt, "smol", "r2", nil).Worker.ID; got != "c" {
		t.Fatalf("with 2 in flight on b, pick %s, want c (load 1)", got)
	}
	for _, at := range []time.Duration{2 * time.Second, 4 * time.Second} {
		fc.Advance(at - fc.Now().Sub(t0))
		beat(t, reg, worker{id: "a", role: "unified", model: "smol", http: "127.0.0.1:1", queue: 2})
	}
	fc.Advance(2*time.Second - time.Millisecond) // t0 + 5.999 s
	if n := len(reg.Snapshot()); n != 3 {
		t.Fatalf("at t0+5.999s %d workers, want all 3 (fewer than 3 heartbeats missed)", n)
	}
	fc.Advance(time.Millisecond) // t0 + 6 s: b and c missed 3 in a row
	if ws := reg.Snapshot(); len(ws) != 1 || ws[0].ID != "a" {
		t.Fatalf("at t0+6s %v, want only a", ws)
	}
	if got := routeOnce(t, rt, "smol", "r3", nil).Worker.ID; got != "a" {
		t.Fatalf("after eviction pick %s, want a", got)
	}
}

func TestRegistryOverGRPC(t *testing.T) {
	// WHY: engines reach the registry over tl.control.v1; a heartbeat makes
	//      a worker routable, and draining its model removes it from routing
	//      and tells the engine to drain in the ack.
	// KIND: conformance
	// CATCHES: s15
	// CHAPTER: gw.05 section 2.1
	reg := route.NewRegistry(clock.NewFake(t0), 3)
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	gs := grpc.NewServer()
	controlv1.RegisterWorkerRegistryServer(gs, reg)
	go gs.Serve(l)
	defer gs.Stop()
	conn, err := grpc.NewClient(l.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	c := controlv1.NewWorkerRegistryClient(conn)
	st := &controlv1.WorkerStatus{WorkerId: "w1", Role: "unified", Model: "smol", HttpAddress: "127.0.0.1:1"}
	ack, err := c.Heartbeat(context.Background(), st)
	if err != nil || ack.GetDrain() {
		t.Fatalf("ack %v, %v", ack, err)
	}
	if ws := reg.Routable(); len(ws) != 1 || ws[0].ID != "w1" {
		t.Fatalf("routable %v", ws)
	}
	reg.Drain("smol")
	if ack, _ = c.Heartbeat(context.Background(), st); !ack.GetDrain() {
		t.Fatal("after Drain(smol) the ack must ask the worker to drain")
	}
	if ws := reg.Routable(); len(ws) != 0 {
		t.Fatalf("a draining model's worker is still routable: %v", ws)
	}
	st.Model, st.Draining = "other", true
	c.Heartbeat(context.Background(), st)
	if ws := reg.Routable(); len(ws) != 0 {
		t.Fatal("a worker that reports draining is not routable")
	}
}

func TestModelNotFoundAndNoCapacity(t *testing.T) {
	// WHY: two different answers: a model nobody serves or routes is 404
	//      model_not_found (the client's mistake); a routed model with no live
	//      worker is 503 no_capacity (ours, retry later).
	// KIND: unit
	// CATCHES: s17
	// CHAPTER: gw.05 section 4
	reg := route.NewRegistry(clock.NewFake(t0), 3)
	rt := route.NewRouter(reg, route.PolicyLeastOutstanding, 1.25, []config.Route{{Model: "routed-but-down"}})
	h := gateway(t, rt, route.Options{}, nil)
	for _, tc := range []struct {
		model string
		code  int
		ecode string
	}{{"nosuch", 404, "model_not_found"}, {"routed-but-down", 503, "no_capacity"}} {
		rec := post(h, "/v1/completions", `{"model":"`+tc.model+`","prompt":"x"}`)
		if rec.Code != tc.code || !strings.Contains(rec.Body.String(), `"`+tc.ecode+`"`) {
			t.Fatalf("%s: %d %s, want %d %s", tc.model, rec.Code, rec.Body.String(), tc.code, tc.ecode)
		}
	}
	if err := rt.Ready(context.Background()); err == nil {
		t.Fatal("Ready with no worker must fail (readyz 503)")
	}
}

func TestAffinityKeepsPrefixTogether(t *testing.T) {
	// WHY: affinity sends every request that shares a prompt prefix to one
	//      worker (its prefix cache is warm), spreads different prefixes, and
	//      moves a prefix off its owner only when the owner is over
	//      ceil(c * average) load (reason "load").
	// KIND: unit, property
	// CATCHES: s03
	// CHAPTER: gw.05 section 2.3
	reg := route.NewRegistry(clock.NewFake(t0), 3)
	for i := 0; i < 4; i++ {
		beat(t, reg, worker{id: fmt.Sprintf("w%d", i), role: "unified", model: "smol", http: "127.0.0.1:1"})
	}
	rt := route.NewRouter(reg, route.PolicyAffinity, 1.25, nil)
	sys := `{"messages":[{"role":"system","content":"You are the docs assistant for forge. Answer from the docs and cite them."},{"role":"user","content":"%d"}]}`
	owner := ""
	for i := 0; i < 20; i++ {
		key := route.PrefixKey([]byte(fmt.Sprintf(sys, i)))
		tg := routeOnce(t, rt, "smol", fmt.Sprint(i), key)
		if owner == "" {
			owner = tg.Worker.ID
		}
		if tg.Worker.ID != owner || tg.Reason != "affinity" {
			t.Fatalf("request %d went to %s (%s); the shared prefix belongs on %s", i, tg.Worker.ID, tg.Reason, owner)
		}
	}
	used := map[string]bool{}
	for i := 0; i < 40; i++ {
		used[routeOnce(t, rt, "smol", "x", []byte(fmt.Sprintf("distinct prompt %d", i))).Worker.ID] = true
	}
	if len(used) < 3 {
		t.Fatalf("40 distinct prefixes used only %d of 4 workers", len(used))
	}
	// Overload the owner: 3 in flight while the others have none.
	for i := 0; i < 3; i++ {
		rt.Acquire(owner)
	}
	tg := routeOnce(t, rt, "smol", "y", route.PrefixKey([]byte(fmt.Sprintf(sys, 99))))
	if tg.Worker.ID == owner || tg.Reason != "load" {
		t.Fatalf("owner %s has load 3 > ceil(1.25 * 4 / 4) = 2, yet got %s (%s)", owner, tg.Worker.ID, tg.Reason)
	}
}

func TestWeightedByCapacity(t *testing.T) {
	// WHY: weighted spreads requests in proportion to KV capacity with smooth
	//      weighted round robin: capacities 1 and 3 give 1 and 3 of every 4,
	//      interleaved rather than in runs.
	// KIND: unit
	// CHAPTER: gw.05 section 2.2
	reg := route.NewRegistry(clock.NewFake(t0), 3)
	beat(t, reg, worker{id: "small", role: "unified", model: "m", http: "h", cap: 1})
	beat(t, reg, worker{id: "big", role: "unified", model: "m", http: "h", cap: 3})
	rt := route.NewRouter(reg, route.PolicyWeighted, 1, nil)
	var seq []string
	for i := 0; i < 8; i++ {
		seq = append(seq, routeOnce(t, rt, "m", "r", nil).Worker.ID)
	}
	if got := strings.Join(seq, ","); got != "big,big,small,big,big,big,small,big" {
		t.Fatalf("sequence %s, want big,big,small,big repeated (smooth WRR)", got)
	}
}

func TestCanarySplit(t *testing.T) {
	// WHY: a canary route splits by weight with a draw keyed by the request
	//      id: about 10% of 10000 requests reach the 0.1 backend (within 4
	//      standard deviations), and one id always lands on the same backend.
	// KIND: statistical
	// CATCHES: s07
	// CHAPTER: gw.05 section 2.4
	r := config.Route{Model: "smol", Backends: []config.Backend{{ServedModel: "smol-v3", Weight: 0.9}, {ServedModel: "smol-v4", Weight: 0.1}}}
	n, canary := 10000, 0
	for i := 0; i < n; i++ {
		id := fmt.Sprintf("req-%d", i)
		b := route.PickBackend(r, id)
		if b != route.PickBackend(r, id) {
			t.Fatal("the same request id picked two backends")
		}
		if b == "smol-v4" {
			canary++
		}
	}
	sd := math.Sqrt(float64(n) * 0.1 * 0.9)
	if math.Abs(float64(canary)-1000) > 4*sd {
		t.Fatalf("%d of %d to the canary, want 1000 +- %.0f", canary, n, 4*sd)
	}
	if got := route.PickBackend(config.Route{Model: "plain"}, "x"); got != "plain" {
		t.Fatalf("no backends: %s, want the route's own model", got)
	}
}

func TestModelRewrittenForBackend(t *testing.T) {
	// WHY: clients name the public model or an alias; the engine only knows
	//      the served model id, so the body's model is rewritten per backend.
	// KIND: unit
	// CATCHES: s10
	// CHAPTER: gw.05 section 2.4
	e := newEngine(t)
	reg := route.NewRegistry(clock.NewFake(t0), 3)
	beat(t, reg, worker{id: "w", role: "unified", model: "smol-135m-v3", http: e.addr()})
	rt := route.NewRouter(reg, route.PolicyLeastOutstanding, 1, []config.Route{{Model: "smol", Aliases: []string{"default"},
		Backends: []config.Backend{{ServedModel: "smol-135m-v3", Weight: 1}}}})
	rec := post(gateway(t, rt, route.Options{}, nil), "/v1/completions", `{"model":"default","prompt":"hi","max_tokens":2}`)
	if rec.Code != 200 {
		t.Fatalf("status %d %s", rec.Code, rec.Body.String())
	}
	if got := e.seen()[0].Model; got != "smol-135m-v3" {
		t.Fatalf("the engine was asked for model %q, want the served model smol-135m-v3", got)
	}
}

func TestFailoverBeforeFirstByte(t *testing.T) {
	// WHY: a worker that is down or answers 503 has sent nothing, so the
	//      request is excluded from it and routed to the next worker; the
	//      client sees one clean stream.
	// KIND: fault
	// CATCHES: s05, s06
	// CHAPTER: gw.05 section 2.5
	for _, failure := range []string{"refused", "503"} {
		good, bad := newEngine(t), newEngine(t)
		badAddr := bad.addr()
		if failure == "refused" {
			bad.srv.Close()
		} else {
			bad.mode.Store("503")
		}
		reg := route.NewRegistry(clock.NewFake(t0), 3)
		beat(t, reg, worker{id: "a-bad", role: "unified", model: "m", http: badAddr, queue: 0})
		beat(t, reg, worker{id: "b-good", role: "unified", model: "m", http: good.addr(), queue: 5})
		rt := route.NewRouter(reg, route.PolicyLeastOutstanding, 1, nil)
		rec := post(gateway(t, rt, route.Options{}, nil), "/v1/completions", `{"model":"m","prompt":"Once","max_tokens":6,"stream":true}`)
		if rec.Code != 200 || streamText(t, rec.Body.String()) != textOf(gen(bytesOf("Once"), 6)) {
			t.Fatalf("%s: got %d %q, want the good worker's stream", failure, rec.Code, rec.Body.String())
		}
		if good.calls.Load() != 1 {
			t.Fatalf("%s: the good worker was called %d times", failure, good.calls.Load())
		}
		if failure == "503" && bad.calls.Load() != 1 {
			t.Fatalf("the 503 worker was tried %d times, want once (then excluded)", bad.calls.Load())
		}
	}
}

func TestNoSpliceAfterFirstByte(t *testing.T) {
	// WHY: once a token reached the client, a dying worker's stream ends with
	//      an SSE error event; retrying elsewhere would repeat or splice
	//      tokens, so the other worker is never called.
	// KIND: fault
	// CATCHES: s04
	// CHAPTER: gw.05 section 5, Pitfalls, item 2
	dying, other := newEngine(t), newEngine(t)
	dying.mode.Store("die-after-first")
	reg := route.NewRegistry(clock.NewFake(t0), 3)
	beat(t, reg, worker{id: "a-dying", role: "unified", model: "m", http: dying.addr(), queue: 0})
	beat(t, reg, worker{id: "b-other", role: "unified", model: "m", http: other.addr(), queue: 5})
	rt := route.NewRouter(reg, route.PolicyLeastOutstanding, 1, nil)
	g := httptest.NewServer(gateway(t, rt, route.Options{}, nil))
	defer g.Close()
	resp, err := http.Post(g.URL+"/v1/completions", "application/json", strings.NewReader(`{"model":"m","prompt":"Once","max_tokens":6,"stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if !strings.Contains(string(body), `data: {"error":`) || strings.Contains(string(body), "[DONE]") {
		t.Fatalf("client read %q; want the first token, then an SSE error event", body)
	}
	if other.calls.Load() != 0 {
		t.Fatal("the request was retried on another worker after the first byte")
	}
	if n := strings.Count(streamText(t, string(body)), ""); n-1 != 1 {
		t.Fatalf("client got %d tokens, want exactly the 1 sent before the failure", n-1)
	}
}

func TestRouteHeaderOnlyForDebug(t *testing.T) {
	// WHY: X-TL-Route names an internal worker; only keys with the debug
	//      scope may see it.
	// KIND: unit
	// CATCHES: s14
	// CHAPTER: gw.05 section 4
	e := newEngine(t)
	reg := route.NewRegistry(clock.NewFake(t0), 3)
	beat(t, reg, worker{id: "w-1", role: "unified", model: "m", http: e.addr()})
	rt := route.NewRouter(reg, route.PolicyLeastOutstanding, 1, nil)
	body := `{"model":"m","prompt":"x","max_tokens":1}`
	if rec := post(gateway(t, rt, route.Options{}, &auth.Principal{KeyID: "k", Scopes: []string{"infer"}}), "/v1/completions", body); rec.Header().Get("X-TL-Route") != "" {
		t.Fatal("X-TL-Route shown to a key without the debug scope")
	}
	if rec := post(gateway(t, rt, route.Options{}, &auth.Principal{KeyID: "k", Scopes: []string{"infer", "debug"}}), "/v1/completions", body); rec.Header().Get("X-TL-Route") != "w-1" {
		t.Fatalf("debug key: X-TL-Route %q, want w-1", rec.Header().Get("X-TL-Route"))
	}
}

func TestModelsList(t *testing.T) {
	// WHY: the gateway answers GET /v1/models from its route table and live
	//      workers (conformance case schema.models on the gateway tier).
	// KIND: unit
	// CHAPTER: gw.05 section 4
	reg := route.NewRegistry(clock.NewFake(t0), 3)
	beat(t, reg, worker{id: "w", role: "unified", model: "tinystories-10m", http: "h"})
	rt := route.NewRouter(reg, route.PolicyAffinity, 1.25, []config.Route{{Model: "smol"}})
	h := gateway(t, rt, route.Options{}, nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/v1/models", nil))
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"id":"smol"`) || !strings.Contains(rec.Body.String(), `"id":"tinystories-10m"`) {
		t.Fatalf("GET /v1/models = %d %s", rec.Code, rec.Body.String())
	}
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/v1/models/nope", nil))
	if rec.Code != 404 {
		t.Fatalf("GET /v1/models/nope = %d", rec.Code)
	}
}

func TestPrefixKey(t *testing.T) {
	// WHY: the affinity key is the start of the rendered prompt, so two chats
	//      with the same system message share it, and it covers one KV block
	//      (64 bytes).
	// KIND: unit
	// CHAPTER: gw.05 section 2.3
	k := route.PrefixKey([]byte(`{"messages":[{"role":"system","content":"S"},{"role":"user","content":"U"}]}`))
	if string(k) != "system\nS\nuser\nU\n" {
		t.Fatalf("PrefixKey = %q", k)
	}
	if k := route.PrefixKey([]byte(`{"prompt":"` + strings.Repeat("x", 1000) + `"}`)); len(k) != route.PrefixKeyBytes {
		t.Fatalf("len %d, want %d", len(k), route.PrefixKeyBytes)
	}
}

func TestParseAcceptIf(t *testing.T) {
	// WHY: accept_if is `<metric> <op> <number>`; an answer with no tokens
	//      has no confidence and is never accepted.
	// KIND: boundary
	// CATCHES: s16
	// CHAPTER: gw.05 section 2.6
	acc, err := route.ParseAcceptIf("mean_logprob > -1.2")
	if err != nil {
		t.Fatal(err)
	}
	if !acc([]float64{-1, -1.3}) || acc([]float64{-1.2, -1.2}) || acc(nil) {
		t.Fatal("mean_logprob > -1.2: mean -1.15 accepted, -1.2 rejected, empty rejected")
	}
	mn, _ := route.ParseAcceptIf("min_logprob >= -2")
	if !mn([]float64{-0.1, -2}) || mn([]float64{-0.1, -2.01}) || mn(nil) {
		t.Fatal("min_logprob >= -2")
	}
	for _, bad := range []string{"", "mean_logprob", "p95_logprob > 1", "mean_logprob => 1", "mean_logprob > x"} {
		if _, err := route.ParseAcceptIf(bad); err == nil {
			t.Fatalf("ParseAcceptIf(%q) must fail", bad)
		}
	}
}

func cascadeSetup(t *testing.T, smallLogprob float64) (http.Handler, *engine, *engine) {
	small, large := newEngine(t), newEngine(t)
	small.logprob = smallLogprob
	reg := route.NewRegistry(clock.NewFake(t0), 3)
	beat(t, reg, worker{id: "s", role: "unified", model: "tinystories-10m", http: small.addr()})
	beat(t, reg, worker{id: "l", role: "unified", model: "smol-135m", http: large.addr()})
	rt := route.NewRouter(reg, route.PolicyLeastOutstanding, 1, []config.Route{{Model: "smol",
		Cascade: []config.CascadeStep{{Model: "tinystories-10m", AcceptIf: "mean_logprob > -1.2"}, {Model: "smol-135m"}}}})
	return gateway(t, rt, route.Options{}, nil), small, large
}

func TestCascadeAcceptsOrEscalates(t *testing.T) {
	// WHY: a confident small answer (mean logprob -0.5 > -1.2) is returned
	//      without calling the large model; an unsure one (-2) is escalated;
	//      a streamed request cannot be taken back, so it goes to the last
	//      step directly.
	// KIND: unit
	// CATCHES: s08, s09
	// CHAPTER: gw.05 section 2.6
	body := `{"model":"smol","prompt":"Once","max_tokens":3}`
	h, small, large := cascadeSetup(t, -0.5)
	if rec := post(h, "/v1/completions", body); rec.Code != 200 || small.calls.Load() != 1 || large.calls.Load() != 0 {
		t.Fatalf("confident: %d, small %d, large %d calls", rec.Code, small.calls.Load(), large.calls.Load())
	}
	if lp := small.seen()[0].Logprobs; lp == nil {
		t.Fatal("the cascade must ask the small model for logprobs")
	}
	h, small, large = cascadeSetup(t, -2)
	rec := post(h, "/v1/completions", body)
	if rec.Code != 200 || small.calls.Load() != 1 || large.calls.Load() != 1 || !strings.Contains(rec.Body.String(), `"model":"smol-135m"`) {
		t.Fatalf("unsure: %d %s, small %d, large %d", rec.Code, rec.Body.String(), small.calls.Load(), large.calls.Load())
	}
	h, small, large = cascadeSetup(t, -0.5)
	if rec := post(h, "/v1/completions", `{"model":"smol","prompt":"Once","max_tokens":3,"stream":true}`); rec.Code != 200 || small.calls.Load() != 0 || large.calls.Load() != 1 {
		t.Fatalf("stream: %d, small %d, large %d", rec.Code, small.calls.Load(), large.calls.Load())
	}
}

func TestCascadeCurveMatchesFixture(t *testing.T) {
	// WHY: the threshold sweep over 200 fixture prompts reproduces the
	//      oracle's cost and accuracy at every tau: raising tau escalates
	//      more, costs more, and (here) answers better.
	// KIND: conformance, golden
	// CATCHES: s18
	// CHAPTER: gw.05 section 2.6
	b, err := os.ReadFile(filepath.Join(os.Getenv("TINYLLM_FIXTURES"), "gw.05", "cascade_sweep.json"))
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		SmallCost float64               `json:"small_cost"`
		Taus      []float64             `json:"taus"`
		Samples   []route.CascadeSample `json:"samples"`
		Curve     []route.CurvePoint    `json:"curve"`
	}
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	got := route.CascadeCurve(f.Samples, f.Taus, f.SmallCost)
	if len(got) != len(f.Curve) {
		t.Fatalf("%d points, want %d", len(got), len(f.Curve))
	}
	for i, w := range f.Curve {
		g := got[i]
		if math.Abs(g.Escalated-w.Escalated) > 1e-9 || math.Abs(g.Cost-w.Cost) > 1e-9 || math.Abs(g.Accuracy-w.Accuracy) > 1e-9 {
			t.Fatalf("tau %v: got %+v, want %+v", w.Tau, g, w)
		}
	}
}

// --- disaggregated ---------------------------------------------------------------

// kvStore is the decode worker's received prompt KV, keyed by handle.
type kvStore struct {
	mu       sync.Mutex
	kv       map[string][]uint32
	released []string
}

type prefillServer struct {
	enginev1.UnimplementedEngineControlServer
	store *kvStore
	mu    sync.Mutex
	reqs  []*enginev1.PrefillRequest
}

func (p *prefillServer) Prefill(ctx context.Context, r *enginev1.PrefillRequest) (*enginev1.PrefillResponse, error) {
	p.mu.Lock()
	p.reqs = append(p.reqs, r)
	p.mu.Unlock()
	h := "h-" + r.GetRequestId()
	p.store.mu.Lock()
	p.store.kv[h] = append([]uint32(nil), r.GetPromptIds()...)
	p.store.mu.Unlock()
	first := gen(r.GetPromptIds(), 1)[0]
	return &enginev1.PrefillResponse{FirstToken: first, PromptTokens: int32(len(r.GetPromptIds())),
		Handle: &enginev1.KvHandle{HandleId: h, DecodeTarget: r.GetDecodeTarget(), NTokens: uint32(len(r.GetPromptIds())),
			FirstToken: first, PromptTokens: uint32(len(r.GetPromptIds())), KvFormat: 1}}, nil
}

type kvServer struct {
	kvv1.UnimplementedKvTransferServiceServer
	store *kvStore
}

func (k *kvServer) Release(ctx context.Context, r *kvv1.ReleaseRequest) (*kvv1.ReleaseResponse, error) {
	k.store.mu.Lock()
	k.store.released = append(k.store.released, r.GetHandleId())
	delete(k.store.kv, r.GetHandleId())
	k.store.mu.Unlock()
	return &kvv1.ReleaseResponse{}, nil
}

func grpcServe(t *testing.T, register func(*grpc.Server)) string {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	s := grpc.NewServer()
	register(s)
	go s.Serve(l)
	t.Cleanup(s.Stop)
	return l.Addr().String()
}

type byteTokenizer struct{}

func (byteTokenizer) Tokenize(ctx context.Context, w route.Worker, served string, body []byte) ([]uint32, error) {
	var b reqBody
	json.Unmarshal(body, &b)
	return bytesOf(b.Prompt), nil
}

type disagg struct {
	h            http.Handler
	unified      *engine
	decode       *engine
	pre          *prefillServer
	store        *kvStore
	decodeKVAddr string
}

func disaggSetup(t *testing.T) *disagg {
	d := &disagg{store: &kvStore{kv: map[string][]uint32{}}}
	d.pre = &prefillServer{store: d.store}
	preAddr := grpcServe(t, func(s *grpc.Server) { enginev1.RegisterEngineControlServer(s, d.pre) })
	d.decodeKVAddr = grpcServe(t, func(s *grpc.Server) { kvv1.RegisterKvTransferServiceServer(s, &kvServer{store: d.store}) })
	d.decode = newEngine(t)
	d.decode.resume = func(h string) ([]uint32, bool) {
		d.store.mu.Lock()
		defer d.store.mu.Unlock()
		ids, ok := d.store.kv[h]
		return ids, ok
	}
	d.unified = newEngine(t)
	reg := route.NewRegistry(clock.NewFake(t0), 3)
	beat(t, reg, worker{id: "p", role: "prefill", model: "m-pd", http: "127.0.0.1:1", grpc: preAddr})
	beat(t, reg, worker{id: "d", role: "decode", model: "m-pd", http: d.decode.addr(), kv: d.decodeKVAddr})
	beat(t, reg, worker{id: "u", role: "unified", model: "m", http: d.unified.addr()})
	rt := route.NewRouter(reg, route.PolicyAffinity, 1.25, []config.Route{
		{Model: "m-pd", Targets: []string{"disaggregated"}},
	})
	d.h = gateway(t, rt, route.Options{Tokenizer: byteTokenizer{}}, nil)
	return d
}

func TestDisaggregatedEqualsUnified(t *testing.T) {
	// WHY: prefill on P, KV pushed to D's kv address, then D resumes from the
	//      handle: the client's text equals the unified engine's for the same
	//      greedy request. A request without a seed gets one, sent to both P
	//      and D, so a sampled stream would also continue P's generator.
	// KIND: differential
	// CATCHES: s11, s12
	// CHAPTER: gw.05 section 2.7
	d := disaggSetup(t)
	uni := post(d.h, "/v1/completions", `{"model":"m","prompt":"Once upon a time","max_tokens":12,"stream":true,"temperature":0}`)
	dis := post(d.h, "/v1/completions", `{"model":"m-pd","prompt":"Once upon a time","max_tokens":12,"stream":true,"temperature":0}`)
	if uni.Code != 200 || dis.Code != 200 {
		t.Fatalf("status unified %d, disaggregated %d (%s)", uni.Code, dis.Code, dis.Body.String())
	}
	if a, b := streamText(t, uni.Body.String()), streamText(t, dis.Body.String()); a != b || len(a) != 12 {
		t.Fatalf("unified %q, disaggregated %q", a, b)
	}
	if len(d.pre.reqs) != 1 {
		t.Fatalf("%d prefill calls", len(d.pre.reqs))
	}
	pr := d.pre.reqs[0]
	if pr.GetDecodeTarget() != d.decodeKVAddr {
		t.Fatalf("decode_target %q, want D's kv address %q", pr.GetDecodeTarget(), d.decodeKVAddr)
	}
	if textOf(pr.GetPromptIds()) != "Once upon a time" || pr.GetSampling().GetMaxNewTokens() != 12 || pr.GetSampling().GetTemperature() != 0 || pr.GetSampling().GetTopP() != 1 {
		t.Fatalf("prefill request %v", pr)
	}
	seen := d.decode.seen()[0]
	if seen.Seed == nil || *seen.Seed != pr.GetSampling().GetSeed() || d.decode.handles[0] != "h-"+pr.GetRequestId() {
		t.Fatalf("decode got seed %v (prefill %d) and handle %q", seen.Seed, pr.GetSampling().GetSeed(), d.decode.handles[0])
	}
}

func TestDisaggregatedReleaseOnAbort(t *testing.T) {
	// WHY: when the decode stream does not finish (the decode worker dies),
	//      the gateway calls Release on D so the pushed blocks return to its
	//      pool instead of leaking until a timeout.
	// KIND: fault
	// CATCHES: s13
	// CHAPTER: gw.05 section 5, Pitfalls, item 6
	d := disaggSetup(t)
	d.decode.mode.Store("die-after-first")
	g := httptest.NewServer(d.h)
	defer g.Close()
	resp, err := http.Post(g.URL+"/v1/completions", "application/json", strings.NewReader(`{"model":"m-pd","prompt":"Once","max_tokens":8,"stream":true,"seed":7}`))
	if err != nil {
		t.Fatal(err)
	}
	io.Copy(io.Discard, bufio.NewReader(resp.Body))
	resp.Body.Close()
	d.store.mu.Lock()
	defer d.store.mu.Unlock()
	if len(d.store.released) != 1 || len(d.store.kv) != 0 {
		t.Fatalf("released %v, still held %d handles", d.store.released, len(d.store.kv))
	}
}

func TestAdminRoutesWorkersDrain(t *testing.T) {
	// WHY: the admin API's routing paths: workers with the route epoch;
	//      routes with optimistic concurrency (428 without If-Match, 412 when
	//      stale, a new ETag and epoch on success, 400 for an invalid
	//      cascade); a drain that stops routing to the model (202).
	// KIND: unit, conformance
	// CATCHES: s19
	// CHAPTER: gw.05 section 4
	reg := route.NewRegistry(clock.NewFake(t0), 3)
	beat(t, reg, worker{id: "w", role: "unified", model: "smol", http: "h"})
	rt := route.NewRouter(reg, route.PolicyAffinity, 1.25, nil)
	do := func(h http.Handler, method, path, body string, hdr ...string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(method, path, strings.NewReader(body))
		for i := 0; i+1 < len(hdr); i += 2 {
			req.Header.Set(hdr[i], hdr[i+1])
		}
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		return rec
	}
	if rec := do(route.WorkersHandler(reg, rt), "GET", "/admin/v1/workers", ""); rec.Code != 200 || !strings.Contains(rec.Body.String(), `"worker_id":"w"`) || !strings.Contains(rec.Body.String(), `"route_epoch":1`) {
		t.Fatalf("workers: %d %s", rec.Code, rec.Body.String())
	}
	rh := route.RoutesHandler(rt)
	etag := do(rh, "GET", "/admin/v1/routes", "").Header().Get("ETag")
	table := `{"routes":[{"model":"smol","cascade":[{"model":"tiny","accept_if":"mean_logprob > -1"},{"model":"smol"}]}]}`
	if rec := do(rh, "PUT", "/admin/v1/routes", table); rec.Code != 428 {
		t.Fatalf("PUT without If-Match: %d", rec.Code)
	}
	if rec := do(rh, "PUT", "/admin/v1/routes", table, "If-Match", `"stale"`); rec.Code != 412 || !strings.Contains(rec.Body.String(), "etag_mismatch") {
		t.Fatalf("PUT stale: %d %s", rec.Code, rec.Body.String())
	}
	if rec := do(rh, "PUT", "/admin/v1/routes", `{"routes":[{"model":"smol","cascade":[{"model":"tiny"},{"model":"smol"}]}]}`, "If-Match", etag); rec.Code != 400 {
		t.Fatalf("PUT invalid cascade: %d", rec.Code)
	}
	rec := do(rh, "PUT", "/admin/v1/routes", table, "If-Match", etag)
	if rec.Code != 200 || rec.Header().Get("ETag") == etag || len(rt.Lookup("smol").Cascade) != 2 {
		t.Fatalf("PUT: %d %s (ETag %s)", rec.Code, rec.Body.String(), rec.Header().Get("ETag"))
	}
	dh := route.DrainHandler(reg)
	if rec := do(dh, "POST", "/admin/v1/models/nosuch:drain", `{"deadline_s":5}`); rec.Code != 404 {
		t.Fatalf("drain unknown: %d", rec.Code)
	}
	if rec := do(dh, "POST", "/admin/v1/models/smol:drain", `{"deadline_s":5}`); rec.Code != 202 || !strings.Contains(rec.Body.String(), `"workers":1`) {
		t.Fatalf("drain: %d %s", rec.Code, rec.Body.String())
	}
	if len(reg.Routable()) != 0 {
		t.Fatal("a drained model's worker is still routable")
	}
}
