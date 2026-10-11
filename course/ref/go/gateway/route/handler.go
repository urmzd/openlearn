// The routing stages of the chain (gw.05): Middleware picks the first
// target; Proxy forwards with failover before the first byte, runs cascades,
// and orchestrates disaggregated prefill and decode.

package route

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	enginev1 "supersource.urmzd.com/tl/contracts/gen/tl/engine/v1"
	kvv1 "supersource.urmzd.com/tl/contracts/gen/tl/kv/v1"
	"tinyllm/config"
	"tinyllm/gateway/auth"
	"tinyllm/gateway/proxy"
	"tinyllm/gateway/server"
)

// Tokenizer turns a request into prompt ids for a disaggregated prefill.
type Tokenizer interface {
	Tokenize(ctx context.Context, w Worker, servedModel string, body []byte) ([]uint32, error)
}

// EngineTokenizer asks the prefill worker's POST /v1/tokenize (engine tier)
// with the completions prompt as `text` (and the chat `messages`, which an
// engine that renders its chat template there can use; DEVIATIONS).
type EngineTokenizer struct {
	Client *http.Client
}

func (e EngineTokenizer) Tokenize(ctx context.Context, w Worker, served string, body []byte) ([]uint32, error) {
	// SOLUTION-BEGIN gw.05
	var f struct {
		Prompt   string          `json:"prompt"`
		Messages json.RawMessage `json:"messages"`
	}
	_ = json.Unmarshal(body, &f)
	in := map[string]any{"model": served, "text": f.Prompt, "add_special": true}
	if len(f.Messages) > 0 {
		in["messages"] = f.Messages
	}
	b, _ := json.Marshal(in)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://"+w.HTTPAddress+"/v1/tokenize", bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	c := e.Client
	if c == nil {
		c = &http.Client{Timeout: 10 * time.Second}
	}
	resp, err := c.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("tokenize: status %d", resp.StatusCode)
	}
	var out struct {
		IDs []uint32 `json:"ids"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return out.IDs, nil
	// SOLUTION-END
}

// Options configure Proxy.
type Options struct {
	Client      *http.Client // HTTP to engines; nil = no overall timeout
	Tokenizer   Tokenizer    // disaggregated routes; nil = EngineTokenizer
	MaxAttempts int          // workers tried before the first byte; 0 = 3
	// Dial opens a gRPC connection (EngineControl on a prefill worker,
	// KvTransferService on a decode worker); nil = insecure grpc.NewClient,
	// one connection per address, reused.
	Dial func(addr string) (grpc.ClientConnInterface, error)
	// OnSpan, when set, receives the gateway.proxy span of each request once
	// its response is finished (the entry point exports it, obs.00): the
	// span whose id the upstream traceparent names as parent.
	OnSpan func(SpanRecord)
}

// SpanRecord is the gateway.proxy span of one routed request. Ids are
// lowercase hex as in a W3C traceparent.
type SpanRecord struct {
	TraceID, SpanID, ParentSpanID, RequestID string
	Status                                   int // the status the client received
	Start, End                               time.Time
}

// spanStatus remembers the status written, for the span.
type spanStatus struct {
	http.ResponseWriter
	code int
}

func (s *spanStatus) WriteHeader(code int) {
	if s.code == 0 {
		s.code = code
	}
	s.ResponseWriter.WriteHeader(code)
}

func (s *spanStatus) Write(b []byte) (int, error) {
	if s.code == 0 {
		s.code = http.StatusOK
	}
	return s.ResponseWriter.Write(b)
}

func (s *spanStatus) Unwrap() http.ResponseWriter { return s.ResponseWriter }

type routedKey struct{}

type routed struct {
	ir     InferenceRequest
	target Target
	route  config.Route
	req    *server.Request
	out    outbound
}

// outbound is the part of gw.04's proxy.Outbound that every attempt shares.
// It is copied out of proxy.Outbound so that no declaration outside a
// function body names a proxy type (gw.00's stubbed tree, which has the
// tracer proxy package, must still compile with this file in it).
type outbound struct {
	method, path, rawQuery string
	header                 http.Header
}

// observer is gw.04's proxy.StreamObserver, restated (identical method set).
type observer interface {
	OnFirstByte()
	OnChunk(data []byte)
	OnDone(u server.Usage)
}

func writeRouteError(w http.ResponseWriter, err error) {
	// SOLUTION-BEGIN gw.05
	if errors.Is(err, ErrModelNotFound) {
		server.WriteError(w, http.StatusNotFound, "invalid_request_error", "model_not_found", "model", err.Error())
		return
	}
	server.WriteError(w, http.StatusServiceUnavailable, "server_error", "no_capacity", "", err.Error())
	// SOLUTION-END
}

// modelsList answers GET /v1/models and /v1/models/{id} from the route table
// and the routable workers.
func modelsList(rt *Router, w http.ResponseWriter, r *http.Request) {
	// SOLUTION-BEGIN gw.05
	ids := map[string]bool{}
	routes, _ := rt.Routes()
	for _, rr := range routes {
		ids[rr.Model] = true
	}
	for _, wk := range rt.reg.Routable() {
		ids[wk.Model] = true
	}
	type model struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}
	if id := strings.TrimPrefix(r.URL.Path, "/v1/models/"); id != r.URL.Path {
		if !ids[id] {
			server.WriteError(w, http.StatusNotFound, "invalid_request_error", "model_not_found", "model", "model "+id+" not found")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(model{id, "model", 0, "tinyllm"})
		return
	}
	list := make([]model, 0, len(ids))
	for id := range ids {
		list = append(list, model{id, "model", 0, "tinyllm"})
	}
	sort.Slice(list, func(i, j int) bool { return list[i].ID < list[j].ID })
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": list})
	// SOLUTION-END
}

// Middleware is the route stage: it answers GET /v1/models itself, picks the
// first target of an inference request (404 model_not_found, 503
// no_capacity), records it on the Exchange (tl.route.worker_id,
// tl.route.reason, tl.route.cascade_step), and sets X-TL-Route for keys
// with the debug scope.
func Middleware(rt *Router) server.Middleware {
	// SOLUTION-BEGIN gw.05
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodGet && strings.HasPrefix(r.URL.Path, "/v1/models") {
				modelsList(rt, w, r)
				return
			}
			ex := server.ExchangeFrom(r.Context())
			req, err := ex.Request(r)
			if err != nil {
				server.WriteError(w, http.StatusBadRequest, "invalid_request_error", "", "", err.Error())
				return
			}
			rd := &routed{req: req, route: rt.Lookup(req.Model)}
			rd.ir = InferenceRequest{Model: req.Model, RequestID: ex.RequestID, PrefixKey: PrefixKey(req.Body), Stream: req.Stream}
			rd.target, err = rt.Route(r.Context(), &rd.ir, nil, 0)
			if err != nil {
				writeRouteError(w, err)
				return
			}
			note(ex, w, r, rd.target)
			next.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), routedKey{}, rd)))
		})
	}
	// SOLUTION-END
}

// note records the chosen target on the Exchange and, for debug keys, in
// X-TL-Route.
func note(ex *server.Exchange, w http.ResponseWriter, r *http.Request, t Target) {
	// SOLUTION-BEGIN gw.05
	ex.SetWorker(t.Worker.ID)
	ex.SetAttr("tl.route.worker_id", t.Worker.ID)
	ex.SetAttr("tl.route.reason", t.Reason)
	ex.SetAttr("tl.route.cascade_step", t.CascadeStep)
	if p, ok := auth.PrincipalFrom(r.Context()); ok && p.HasScope(auth.ScopeDebug) {
		w.Header().Set("X-TL-Route", t.Worker.ID)
	} else {
		w.Header().Del("X-TL-Route")
	}
	// SOLUTION-END
}

// withFields returns body with the given top-level fields set (the served
// model, a forced seed or logprobs), leaving every other field as sent.
func withFields(body []byte, set map[string]any) []byte {
	// SOLUTION-BEGIN gw.05
	var m map[string]json.RawMessage
	if json.Unmarshal(body, &m) != nil || m == nil {
		return body
	}
	for k, v := range set {
		b, _ := json.Marshal(v)
		m[k] = b
	}
	out, _ := json.Marshal(m)
	return out
	// SOLUTION-END
}

// capture is a ResponseWriter that keeps a cascade step's answer until it
// is accepted.
type capture struct {
	h    http.Header
	code int
	body bytes.Buffer
}

func (c *capture) Header() http.Header         { return c.h }
func (c *capture) WriteHeader(code int)        { c.code = code }
func (c *capture) Write(b []byte) (int, error) { return c.body.Write(b) }

type proxyStage struct {
	rt    *Router
	o     Options
	mu    sync.Mutex
	conns map[string]grpc.ClientConnInterface
}

// Proxy is the terminal stage that replaces the single-upstream proxy once
// routing exists. Before the first byte, a worker that refuses, resets, or
// answers 503 is excluded and the request is routed again (at most
// MaxAttempts workers); after the first byte nothing is retried or spliced
// (gw.04 has already sent the SSE error event). A non-streamed request on a
// cascade route tries each step's model in turn and returns the first
// answer its accept_if accepts; streamed requests go to the last step. A
// disaggregated route runs Prefill on the prefill worker, then streams from
// the decode worker with X-TL-KV-Handle, and releases the handle when the
// decode stream does not finish.
func Proxy(rt *Router, o Options) http.Handler {
	// SOLUTION-BEGIN gw.05
	if o.MaxAttempts <= 0 {
		o.MaxAttempts = 3
	}
	if o.Client == nil {
		o.Client = &http.Client{}
	}
	if o.Tokenizer == nil {
		o.Tokenizer = EngineTokenizer{}
	}
	return &proxyStage{rt: rt, o: o, conns: map[string]grpc.ClientConnInterface{}}
	// SOLUTION-END
}

func (p *proxyStage) dial(addr string) (grpc.ClientConnInterface, error) {
	// SOLUTION-BEGIN gw.05
	if p.o.Dial != nil {
		return p.o.Dial(addr)
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if c, ok := p.conns[addr]; ok {
		return c, nil
	}
	c, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}
	p.conns[addr] = c
	return c, nil
	// SOLUTION-END
}

func (p *proxyStage) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// SOLUTION-BEGIN gw.05
	rd, ok := r.Context().Value(routedKey{}).(*routed)
	if !ok {
		server.WriteError(w, http.StatusServiceUnavailable, "server_error", "no_capacity", "", "the route stage did not run")
		return
	}
	ex := server.ExchangeFrom(r.Context())
	r.Header.Set("X-Request-Id", ex.RequestID)
	o, sp := proxy.NewOutbound(r, rd.req.Body)
	if p.o.OnSpan != nil {
		sw := &spanStatus{ResponseWriter: w}
		w = sw
		rec := SpanRecord{TraceID: sp.TraceID, SpanID: sp.SpanID, ParentSpanID: sp.ParentSpanID, RequestID: ex.RequestID, Start: time.Now()}
		defer func() {
			rec.End, rec.Status = time.Now(), sw.code
			p.o.OnSpan(rec)
		}()
	}
	rd.out = outbound{method: o.Method, path: o.Path, rawQuery: o.RawQuery, header: o.Header}
	start := 0
	if n := len(rd.route.Cascade); n > 0 {
		start = n - 1
		if !rd.req.Stream {
			if p.cascade(w, r, rd, ex) {
				return
			}
		} else if rd.target.CascadeStep != start {
			t, err := p.rt.Route(r.Context(), &rd.ir, nil, start)
			if err != nil {
				writeRouteError(w, err)
				return
			}
			rd.target = t
			note(ex, w, r, t)
		}
	}
	p.attempts(w, r, rd, ex, start)
	// SOLUTION-END
}

// attempts forwards with failover before the first byte.
func (p *proxyStage) attempts(w http.ResponseWriter, r *http.Request, rd *routed, ex *server.Exchange, step int) {
	// SOLUTION-BEGIN gw.05
	excl := Exclusions{}
	t := rd.target
	var lastErr error
	for attempt := 0; attempt < p.o.MaxAttempts; attempt++ {
		if attempt > 0 {
			var err error
			t, err = p.rt.Route(r.Context(), &rd.ir, excl, step)
			if err != nil {
				lastErr = err
				break
			}
			note(ex, w, r, t)
		}
		err := p.once(w, r, rd, ex, t, proxy.ExchangeObserver(ex))
		if err == nil {
			return
		}
		var ue *proxy.UpstreamError
		if !errors.As(err, &ue) || ue.Committed || r.Context().Err() != nil {
			return // after the first byte (or the client left): never retried
		}
		lastErr = err
		excl[t.Worker.ID] = true
		if t.Prefill != nil {
			excl[t.Prefill.ID] = true
		}
	}
	if r.Context().Err() == nil {
		server.WriteError(w, http.StatusServiceUnavailable, "server_error", "no_capacity", "",
			fmt.Sprintf("no worker could take the request: %v", lastErr))
	}
	// SOLUTION-END
}

// once runs one attempt against t.
func (p *proxyStage) once(w http.ResponseWriter, r *http.Request, rd *routed, ex *server.Exchange, t Target, obs observer) error {
	// SOLUTION-BEGIN gw.05
	att := proxy.Outbound{Method: rd.out.method, Path: rd.out.path, RawQuery: rd.out.rawQuery, Header: rd.out.header.Clone()}
	att.Retry503 = true
	set := map[string]any{"model": t.ServedModel}
	p.rt.Acquire(t.Worker.ID)
	defer p.rt.Release(t.Worker.ID)
	if t.Prefill == nil {
		att.Body = withFields(rd.req.Body, set)
		return proxy.Forward(r.Context(), w, "http://"+t.Worker.HTTPAddress, &att, p.o.Client, obs)
	}
	// Disaggregated: prefill on P, which pushes the prompt KV to D.
	seed := uint64(0)
	if rd.req.Seed != nil {
		seed = uint64(*rd.req.Seed)
	} else {
		var b [8]byte
		_, _ = rand.Read(b[:])
		seed = binary.LittleEndian.Uint64(b[:]) >> 1 // fits a JSON int64
		set["seed"] = seed                           // D must draw from the same stream as P
	}
	ids, err := p.o.Tokenizer.Tokenize(r.Context(), *t.Prefill, t.ServedModel, rd.req.Body)
	if err != nil {
		return &proxy.UpstreamError{Err: fmt.Errorf("tokenize on %s: %w", t.Prefill.ID, err)}
	}
	conn, err := p.dial(t.Prefill.GRPCAddress)
	if err != nil {
		return &proxy.UpstreamError{Err: err}
	}
	prio, _ := strconv.Atoi(r.Header.Get("X-TL-Priority"))
	pre, err := enginev1.NewEngineControlClient(conn).Prefill(r.Context(), &enginev1.PrefillRequest{
		RequestId: ex.RequestID, Model: t.ServedModel, PromptIds: ids,
		Sampling: sampling(rd.req.Body, seed), DecodeTarget: t.Worker.KVAddress,
		DeadlineUnixMs: time.Now().Add(30 * time.Second).UnixMilli(), Priority: int32(prio),
	})
	if err != nil {
		return &proxy.UpstreamError{Err: fmt.Errorf("prefill on %s: %w", t.Prefill.ID, err)}
	}
	handle := pre.GetHandle().GetHandleId()
	att.Body = withFields(rd.req.Body, set)
	att.Header.Set("X-TL-KV-Handle", handle)
	err = proxy.Forward(r.Context(), w, "http://"+t.Worker.HTTPAddress, &att, p.o.Client, obs)
	if err != nil {
		// The decode side holds the pushed blocks under the handle until it
		// resumes or is told to let go.
		if kc, derr := p.dial(t.Worker.KVAddress); derr == nil {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			_, _ = kvv1.NewKvTransferServiceClient(kc).Release(ctx, &kvv1.ReleaseRequest{HandleId: handle})
			cancel()
		}
	}
	return err
	// SOLUTION-END
}

// sampling maps a request body to the proto's SamplingParams with the
// openai-subset.v1.yaml defaults for omitted fields (proto3 has no
// presence: every field is always sent).
func sampling(body []byte, seed uint64) *enginev1.SamplingParams {
	// SOLUTION-BEGIN gw.05
	var f struct {
		Temperature         *float32 `json:"temperature"`
		TopP                *float32 `json:"top_p"`
		TopK                int32    `json:"top_k"`
		MinP                float32  `json:"min_p"`
		RepetitionPenalty   *float32 `json:"repetition_penalty"`
		PresencePenalty     float32  `json:"presence_penalty"`
		FrequencyPenalty    float32  `json:"frequency_penalty"`
		MaxTokens           *int32   `json:"max_tokens"`
		MaxCompletionTokens *int32   `json:"max_completion_tokens"`
		Stop                any      `json:"stop"`
		Logprobs            any      `json:"logprobs"`
		TopLogprobs         int32    `json:"top_logprobs"`
	}
	_ = json.Unmarshal(body, &f)
	sp := &enginev1.SamplingParams{Temperature: 1, TopP: 1, RepetitionPenalty: 1, TopK: f.TopK, MinP: f.MinP,
		PresencePenalty: f.PresencePenalty, FrequencyPenalty: f.FrequencyPenalty, Seed: seed}
	if f.Temperature != nil {
		sp.Temperature = *f.Temperature
	}
	if f.TopP != nil {
		sp.TopP = *f.TopP
	}
	if f.RepetitionPenalty != nil {
		sp.RepetitionPenalty = *f.RepetitionPenalty
	}
	switch {
	case f.MaxCompletionTokens != nil:
		sp.MaxNewTokens = *f.MaxCompletionTokens
	case f.MaxTokens != nil:
		sp.MaxNewTokens = *f.MaxTokens
	}
	switch s := f.Stop.(type) {
	case string:
		sp.Stop = []string{s}
	case []any:
		for _, x := range s {
			if str, ok := x.(string); ok {
				sp.Stop = append(sp.Stop, str)
			}
		}
	}
	switch lp := f.Logprobs.(type) {
	case bool:
		if lp {
			sp.Logprobs = f.TopLogprobs
		}
	case float64:
		sp.Logprobs = int32(lp)
	}
	return sp
	// SOLUTION-END
}

// cascade runs every step but the last on a non-streamed request and writes
// the first accepted answer; false means escalate to the last step.
func (p *proxyStage) cascade(w http.ResponseWriter, r *http.Request, rd *routed, ex *server.Exchange) bool {
	// SOLUTION-BEGIN gw.05
	steps := rd.route.Cascade
	force := map[string]any{"logprobs": true}
	if r.URL.Path == "/v1/completions" {
		force = map[string]any{"logprobs": 1}
	}
	var lpField map[string]json.RawMessage
	_ = json.Unmarshal(rd.req.Body, &lpField)
	for step := 0; step < len(steps)-1; step++ {
		accept, err := ParseAcceptIf(steps[step].AcceptIf)
		if err != nil {
			continue
		}
		t := rd.target
		if step != rd.target.CascadeStep {
			if t, err = p.rt.Route(r.Context(), &rd.ir, nil, step); err != nil {
				continue
			}
		}
		note(ex, w, r, t)
		body := rd.req.Body
		if _, has := lpField["logprobs"]; !has {
			body = withFields(body, force)
		}
		local := *rd
		local.req = &server.Request{Model: rd.req.Model, Body: body, Stream: false}
		c := &capture{h: http.Header{}}
		var usage server.Usage
		obs := usageObserver{&usage}
		if err := p.once(c, r, &local, ex, t, obs); err != nil || c.code != http.StatusOK {
			continue
		}
		lps, err := LogprobsOf(c.body.Bytes())
		if err != nil || !accept(lps) {
			continue
		}
		for k, vs := range c.h {
			if !strings.EqualFold(k, "X-Request-Id") {
				w.Header()[k] = vs
			}
		}
		ex.SetUsage(usage)
		ex.MarkFirstByte()
		w.WriteHeader(c.code)
		_, _ = w.Write(c.body.Bytes())
		return true
	}
	t, err := p.rt.Route(r.Context(), &rd.ir, nil, len(steps)-1)
	if err != nil {
		writeRouteError(w, err)
		return true
	}
	rd.target = t
	note(ex, w, r, t)
	return false
	// SOLUTION-END
}

type usageObserver struct{ u *server.Usage }

func (usageObserver) OnFirstByte()            {}
func (usageObserver) OnChunk([]byte)          {}
func (o usageObserver) OnDone(u server.Usage) { *o.u = u }
