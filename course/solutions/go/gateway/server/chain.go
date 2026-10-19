// Package server is the gateway's composition root library (gw.01): the
// middleware chain in its contractual order, the request-scoped Exchange
// every stage shares, the OpenAI error shape, and the server lifecycle
// (health, readiness, graceful drain) in server.go.
//
// The chain order is a contract:
//
//	requestid -> otel -> recover -> authn -> policy -> ratelimit -> cache -> route -> proxy -> meter
//
// Stages come in through Deps as plain middleware, so this package imports
// none of the stage packages (auth, limit, cache, route, proxy, ledger,
// policy): they import it, and your go/cmd/gateway main wires them together.
// meter wraps the proxy directly: it is the first stage to see the finished
// exchange, which is what "proxy -> meter" orders.
//
// Contract: openapi/openai-subset.v1.yaml (gateway tier), spec/cli-roles.md
// (role gateway, --config form). Chapter:
// content/courses/ai-platform-engineering/12-gateway/01-server-skeleton.md; policy wiring:
// content/courses/ai-platform-engineering/12-gateway/08-usage-policy-enforcement.md.
package server

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"runtime/debug"
	"strings"
	"sync"
	"time"
)

// Stage names one link of the chain.
type Stage string

const (
	StageRequestID Stage = "requestid"
	StageOTel      Stage = "otel"
	StageRecover   Stage = "recover"
	StageAuthN     Stage = "authn"
	StagePolicy    Stage = "policy"
	StageRateLimit Stage = "ratelimit"
	StageCache     Stage = "cache"
	StageRoute     Stage = "route"
	StageProxy     Stage = "proxy"
	StageMeter     Stage = "meter"
)

// Order is the chain order of the contract.
var Order = []Stage{
	StageRequestID, StageOTel, StageRecover, StageAuthN, StagePolicy,
	StageRateLimit, StageCache, StageRoute, StageProxy, StageMeter,
}

// Middleware wraps the rest of the chain.
type Middleware func(next http.Handler) http.Handler

// Clock is the time seam: the gateway reads time only through it, so tests
// move time with a fake (course testkit clock.Fake satisfies it).
type Clock interface {
	Now() time.Time
}

type wallClock struct{}

func (wallClock) Now() time.Time { return time.Now() }

// WallClock is the real clock.
var WallClock Clock = wallClock{}

// Tracer starts the server span of a request (obs.01 supplies one backed by
// OpenTelemetry; nil records nothing).
type Tracer interface {
	Start(ctx context.Context, name string) (context.Context, Span)
}

// Span is the part of a span the chain uses.
type Span interface {
	SetAttribute(key string, value any)
	End()
}

// Usage is the token accounting of one exchange, from the upstream's usage
// object (prompt_tokens, completion_tokens, total_tokens).
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Request is what the stages need from an inference request body.
type Request struct {
	Model       string
	Stream      bool
	MaxTokens   int      // max_completion_tokens, else max_tokens; 0 when absent
	Temperature *float64 // nil when absent (the default is 1)
	Seed        *int64   // nil when absent
	Body        []byte   // the raw body, as the client sent it
}

// Exchange is the record of one request that every stage shares. The
// requestid stage creates it; read it with ExchangeFrom. Fields other than
// RequestID and Start are set by later stages under its lock (Set*) and
// read with Snapshot.
type Exchange struct {
	RequestID string
	Start     time.Time

	mu        sync.Mutex
	status    int
	usage     Usage
	firstByte time.Time
	worker    string
	attrs     map[string]any
	req       *Request
	reqErr    error
	maxBody   int64
	clock     Clock
}

// ExchangeState is a copy of an Exchange's mutable fields.
type ExchangeState struct {
	Status    int
	Usage     Usage
	FirstByte time.Time
	Worker    string
	Attrs     map[string]any
}

type exchangeKey struct{}

// ExchangeFrom returns the request's Exchange, or nil outside the chain.
func ExchangeFrom(ctx context.Context) *Exchange {
	e, _ := ctx.Value(exchangeKey{}).(*Exchange)
	return e
}

// WithExchange returns ctx carrying e (for tests of a single stage).
func WithExchange(ctx context.Context, e *Exchange) context.Context {
	return context.WithValue(ctx, exchangeKey{}, e)
}

// NewExchange makes an Exchange outside the chain (for tests of a single
// stage); maxBody 0 means DefaultMaxBody.
func NewExchange(requestID string, clock Clock, maxBody int64) *Exchange {
	if clock == nil {
		clock = WallClock
	}
	if maxBody <= 0 {
		maxBody = DefaultMaxBody
	}
	return &Exchange{RequestID: requestID, Start: clock.Now(), clock: clock, maxBody: maxBody, attrs: map[string]any{}}
}

// SetUsage records the upstream's usage (the proxy calls it at the end of
// the stream; a cache hit records the cached usage).
func (e *Exchange) SetUsage(u Usage) {
	e.mu.Lock()
	e.usage = u
	e.mu.Unlock()
}

// MarkFirstByte records when the first response byte reached the client
// (once; later calls are ignored).
func (e *Exchange) MarkFirstByte() {
	e.mu.Lock()
	if e.firstByte.IsZero() {
		e.firstByte = e.clock.Now()
	}
	e.mu.Unlock()
}

// SetWorker records the worker the router chose.
func (e *Exchange) SetWorker(id string) {
	e.mu.Lock()
	e.worker = id
	e.mu.Unlock()
}

// SetAttr records a span attribute (tl.ratelimit.decision, tl.cache.hit, ...).
func (e *Exchange) SetAttr(key string, v any) {
	e.mu.Lock()
	e.attrs[key] = v
	e.mu.Unlock()
}

func (e *Exchange) setStatus(code int) {
	e.mu.Lock()
	if e.status == 0 {
		e.status = code
	}
	e.mu.Unlock()
}

// Snapshot copies the mutable fields.
func (e *Exchange) Snapshot() ExchangeState {
	e.mu.Lock()
	defer e.mu.Unlock()
	attrs := make(map[string]any, len(e.attrs))
	for k, v := range e.attrs {
		attrs[k] = v
	}
	return ExchangeState{Status: e.status, Usage: e.usage, FirstByte: e.firstByte, Worker: e.worker, Attrs: attrs}
}

// DefaultMaxBody is the largest request body the gateway reads: 4 MiB, the
// same cap as every gRPC message.
const DefaultMaxBody = 4 << 20

// Request reads r's body once (at most the configured maximum), parses the
// fields the stages need, and puts an identical body back on r so the proxy
// can forward it. Later calls return the cached result. An error means a
// 400: the body is too large or is not a JSON object.
func (e *Exchange) Request(r *http.Request) (*Request, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.req != nil || e.reqErr != nil {
		if e.req != nil {
			r.Body = io.NopCloser(bytes.NewReader(e.req.Body))
		}
		return e.req, e.reqErr
	}
	var body []byte
	if r.Body != nil {
		b, err := io.ReadAll(io.LimitReader(r.Body, e.maxBody+1))
		if err != nil {
			e.reqErr = fmt.Errorf("cannot read the request body: %v", err)
			return nil, e.reqErr
		}
		body = b
	}
	if int64(len(body)) > e.maxBody {
		e.reqErr = fmt.Errorf("the request body is larger than %d bytes", e.maxBody)
		return nil, e.reqErr
	}
	r.Body = io.NopCloser(bytes.NewReader(body))
	req, err := ParseRequest(body)
	if err != nil {
		e.reqErr = err
		return nil, err
	}
	e.req = &req
	return e.req, nil
}

// ParseRequest extracts model, stream, the token limit, temperature, and
// seed from a JSON request body. Unknown fields are ignored (the contract
// says so); a body that is not one JSON object is an error.
func ParseRequest(body []byte) (Request, error) {
	var f struct {
		Model               string   `json:"model"`
		Stream              bool     `json:"stream"`
		MaxTokens           *int     `json:"max_tokens"`
		MaxCompletionTokens *int     `json:"max_completion_tokens"`
		Temperature         *float64 `json:"temperature"`
		Seed                *int64   `json:"seed"`
	}
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return Request{}, errors.New("the request body must be a JSON object")
	}
	if err := json.Unmarshal(trimmed, &f); err != nil {
		return Request{}, fmt.Errorf("malformed JSON: %v", err)
	}
	req := Request{Model: f.Model, Stream: f.Stream, Temperature: f.Temperature, Seed: f.Seed, Body: body}
	switch {
	case f.MaxCompletionTokens != nil:
		req.MaxTokens = *f.MaxCompletionTokens
	case f.MaxTokens != nil:
		req.MaxTokens = *f.MaxTokens
	}
	return req, nil
}

// apiError is the OpenAI error shape: all four keys always present.
type apiError struct {
	Error struct {
		Message string  `json:"message"`
		Type    string  `json:"type"`
		Param   *string `json:"param"`
		Code    *string `json:"code"`
	} `json:"error"`
}

// WriteError writes status with the body {"error": {message, type, param,
// code}} and Content-Type application/json; an empty param or code is null.
// Every gateway stage answers errors through it.
func WriteError(w http.ResponseWriter, status int, typ, code, param, msg string) {
	var e apiError
	e.Error.Message, e.Error.Type = msg, typ
	if code != "" {
		e.Error.Code = &code
	}
	if param != "" {
		e.Error.Param = &param
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(e)
}

// statusWriter records the status (and the first byte) on the Exchange and
// keeps Flush reachable through Unwrap.
type statusWriter struct {
	http.ResponseWriter
	ex *Exchange
}

func (s *statusWriter) WriteHeader(code int) {
	s.ex.setStatus(code)
	s.ResponseWriter.WriteHeader(code)
}

func (s *statusWriter) Write(b []byte) (int, error) {
	s.ex.setStatus(http.StatusOK)
	return s.ResponseWriter.Write(b)
}

func (s *statusWriter) Unwrap() http.ResponseWriter {
	return s.ResponseWriter
}

// validRequestID keeps a caller's X-Request-Id when it is printable ASCII
// (0x21 to 0x7E) of at most 128 bytes, the gw.00 rule.
func validRequestID(s string) bool {
	if s == "" || len(s) > 128 {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] < 0x21 || s[i] > 0x7e {
			return false
		}
	}
	return true
}

func newID() string {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		panic("server: crypto/rand failed: " + err.Error())
	}
	return hex.EncodeToString(b)
}

// requestID is the first stage: it strips the internal X-TL-* headers a
// client may not set (X-TL-KV-Handle, X-TL-Priority: the gateway sets them
// toward engines), picks the request id, sets it on the request (so the
// proxy forwards it) and on the response, and creates the Exchange.
func (s *Server) requestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		for k := range r.Header {
			// API selection is public; routing and KV handles are internal.
			if strings.HasPrefix(strings.ToLower(k), "x-tl-") && !strings.EqualFold(k, "X-TL-API-Version") {
				r.Header.Del(k)
			}
		}
		id := r.Header.Get("X-Request-Id")
		if !validRequestID(id) {
			id = newID()
		}
		r.Header.Set("X-Request-Id", id)
		w.Header().Set("X-Request-Id", id)
		ex := NewExchange(id, s.clock, s.cfg.MaxBodyBytes)
		next.ServeHTTP(&statusWriter{ResponseWriter: w, ex: ex}, r.WithContext(WithExchange(r.Context(), ex)))
	})
}

// otel starts the SERVER span "<METHOD> <path>" and ends it with the status
// and every attribute the later stages recorded on the Exchange.
func (s *Server) otel(next http.Handler) http.Handler {
	if s.deps.Tracer == nil {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx, span := s.deps.Tracer.Start(r.Context(), r.Method+" "+r.URL.Path)
		ex := ExchangeFrom(ctx)
		span.SetAttribute("http.request.method", r.Method)
		span.SetAttribute("http.route", r.URL.Path)
		defer func() {
			st := ex.Snapshot()
			span.SetAttribute("http.response.status_code", st.Status)
			for k, v := range st.Attrs {
				span.SetAttribute(k, v)
			}
			span.End()
		}()
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// recoverer turns a panic in a later stage into a 500 internal_error (when
// nothing was written yet) and keeps the process serving. A handler that
// panics with http.ErrAbortHandler wants the connection aborted: re-panic.
func (s *Server) recoverer(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			v := recover()
			if v == nil {
				return
			}
			if v == http.ErrAbortHandler {
				panic(v)
			}
			ex := ExchangeFrom(r.Context())
			log.Printf("gateway: panic in request %s: %v\n%s", ex.RequestID, v, debug.Stack())
			if ex.Snapshot().Status == 0 {
				WriteError(w, http.StatusInternalServerError, "server_error", "internal_error", "", "internal error; quote X-Request-Id "+ex.RequestID)
			}
		}()
		next.ServeHTTP(w, r)
	})
}

// Handler is the whole API chain, in Order. A nil stage is skipped; a nil
// Proxy answers 503 no_capacity.
func (s *Server) Handler() http.Handler {
	var h http.Handler = s.deps.Proxy
	if h == nil {
		h = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			WriteError(w, http.StatusServiceUnavailable, "server_error", "no_capacity", "", "no upstream is configured")
		})
	}
	// Innermost first: meter wraps the proxy, then route, cache, ... authn.
	for _, m := range []Middleware{s.deps.Ledger, s.deps.Router, s.deps.Cache, s.deps.Limiter} {
		if m != nil {
			h = m(h)
		}
	}
	// gw.08's policy wraps rate limits, after authentication and before route.
	if s.deps.Policy != nil {
		h = s.deps.Policy(h)
	}
	if s.deps.Keys != nil {
		h = s.deps.Keys(h)
	}
	h = s.recoverer(h)
	h = s.otel(h)
	return s.requestID(h)
}
