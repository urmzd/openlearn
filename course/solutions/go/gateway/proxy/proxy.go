// Package proxy is the gateway's request path to an engine. gw.00 built the
// tracer form: one static API key, the request forwarded, the response
// streamed back byte for byte, W3C trace context and X-Request-Id carried
// across. gw.04 takes the package over and adds the streaming observer
// (first byte, every SSE event, the final usage), the commitment boundary
// that routing's failover relies on (Forward), and the chain handler that
// records the stream on the server Exchange.
//
// Contract: course/contracts/openapi/openai-subset.v1.yaml (Streaming,
// Disconnect, Errors; gateway tier) and openai-subset.v0.yaml for NewProxy.
// Chapters: content/courses/ai-platform-engineering/12-gateway/01-streaming-proxy.md (gw.00)
// and content/courses/ai-platform-engineering/12-gateway/03-sse-streaming-proxy.md (gw.04).
package proxy

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"tinyllm/gateway/server"
)

// Config is everything the tracer proxy needs.
type Config struct {
	// Upstream is the engine's base URL without /v1, for example
	// "http://127.0.0.1:8081". The request path is appended to it.
	Upstream string
	// APIKey is the one key accepted as "Authorization: Bearer <APIKey>".
	// An empty APIKey rejects every request (fail closed).
	APIKey string
	// Client sends the upstream request. nil means a client with no overall
	// timeout: a stream may legitimately last minutes, and the client's own
	// disconnect is what ends it.
	Client *http.Client
	// OnSpan, when set, receives the gateway.proxy span once the response is
	// finished. The entry point exports it (obs.00); the library only records.
	OnSpan func(Span)
	// Observer, when set, makes the observer for each request (gw.04). Inside
	// the server chain the default records on the Exchange (ExchangeObserver).
	Observer func(r *http.Request) StreamObserver
}

// Span is the record of one proxied request. Ids are lowercase hex as in a
// W3C traceparent.
type Span struct {
	Name         string // always "gateway.proxy"
	TraceID      string // 32 hex digits, shared with the engine's span
	SpanID       string // 16 hex digits; the engine's server span is its child
	ParentSpanID string // the caller's span id, "" when the gateway started the trace
	RequestID    string // the X-Request-Id sent upstream and returned to the client
	StatusCode   int    // the status the client received
	Start, End   time.Time
}

// SpanName is the name of the span the gateway records for every request.
const SpanName = "gateway.proxy"

// hopHeaders are connection-scoped (RFC 9110 section 7.6.1): they describe
// one hop and must not be forwarded to the next.
var hopHeaders = []string{
	"Connection", "Keep-Alive", "Proxy-Authenticate", "Proxy-Authorization",
	"Te", "Trailer", "Transfer-Encoding", "Upgrade",
}

// NewProxy returns the tracer gateway's handler for every API path: the
// static key check of gw.00, then Handler.
func NewProxy(cfg Config) http.Handler {
	client := cfg.Client
	if client == nil {
		client = &http.Client{}
	}
	base := strings.TrimRight(cfg.Upstream, "/")
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		serve(w, r, cfg, client, base, true)
	})
}

// Handler is the proxy stage of the server chain (gw.04): the same request
// path as NewProxy without the static key, because the authn stage (gw.02)
// already authenticated the request.
func Handler(cfg Config) http.Handler {
	client := cfg.Client
	if client == nil {
		client = &http.Client{}
	}
	base := strings.TrimRight(cfg.Upstream, "/")
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		serve(w, r, cfg, client, base, false)
	})
}

// statusWriter remembers the status code written, for the span.
type statusWriter struct {
	http.ResponseWriter
	code int
}

func (s *statusWriter) WriteHeader(code int) {
	if s.code == 0 {
		s.code = code
	}
	s.ResponseWriter.WriteHeader(code)
}

func (s *statusWriter) Unwrap() http.ResponseWriter {
	return s.ResponseWriter
}

func serve(w http.ResponseWriter, r *http.Request, cfg Config, client *http.Client, base string, checkKey bool) {
	sw := &statusWriter{ResponseWriter: w}
	reqID := requestID(r.Header.Get("X-Request-Id"))
	span := Span{Name: SpanName, Start: time.Now(), RequestID: reqID}
	defer func() {
		if cfg.OnSpan != nil {
			span.End = time.Now()
			span.StatusCode = sw.code
			cfg.OnSpan(span)
		}
	}()
	w.Header().Set("X-Request-Id", reqID)

	// 1. The key, before anything leaves the gateway (tracer form only).
	if checkKey && !authorized(r.Header.Get("Authorization"), cfg.APIKey) {
		writeError(sw, http.StatusUnauthorized, "invalid_request_error", "invalid_api_key",
			"Incorrect or missing API key: send the header Authorization: Bearer KEY.")
		return
	}

	// 2. The outbound request: trace context, request id, no key.
	body, err := io.ReadAll(io.LimitReader(r.Body, server.DefaultMaxBody+1))
	if err != nil || len(body) > server.DefaultMaxBody {
		writeError(sw, http.StatusBadRequest, "invalid_request_error", "", "cannot read the request body")
		return
	}
	r.Header.Set("X-Request-Id", reqID) // so the outbound carries the same id
	out, s := NewOutbound(r, body)
	span.TraceID, span.SpanID, span.ParentSpanID = s.TraceID, s.SpanID, s.ParentSpanID

	// 3. Forward and stream; before the first byte a failure is a 503.
	var obs StreamObserver
	if cfg.Observer != nil {
		obs = cfg.Observer(r)
	} else if ex := server.ExchangeFrom(r.Context()); ex != nil {
		obs = ExchangeObserver(ex)
	}
	err = Forward(r.Context(), sw, base, out, client, obs)
	var ue *UpstreamError
	if errors.As(err, &ue) && !ue.Committed && r.Context().Err() == nil {
		writeError(sw, http.StatusServiceUnavailable, "server_error", "no_capacity",
			"the upstream engine is unavailable: "+ue.Err.Error())
	}
}

// Outbound is one request as it will be sent upstream: the method, path, and
// query of the client's request, its end-to-end headers minus Authorization,
// a traceparent naming the gateway's span as parent, and the body. gw.05
// sends one Outbound to several workers in turn (failover), so it is built
// once.
type Outbound struct {
	Method, Path, RawQuery string
	Header                 http.Header
	Body                   []byte
	// Retry503: an upstream 503 is returned as an uncommitted UpstreamError
	// (nothing written) so the caller can try another worker.
	Retry503 bool
}

// NewOutbound applies the gw.00 header rules to r: hop-by-hop headers and
// Authorization dropped, the caller's trace continued with a new span id (or
// a new sampled trace), tracestate kept only with a valid traceparent, and
// X-Request-Id set. The returned Span carries the ids.
func NewOutbound(r *http.Request, body []byte) (*Outbound, Span) {
	span := Span{Name: SpanName, RequestID: requestID(r.Header.Get("X-Request-Id"))}
	traceID, parent, flags, ok := ParseTraceparent(r.Header.Get("Traceparent"))
	if !ok {
		traceID, parent, flags = randomHex(16), "", "01"
	}
	span.TraceID, span.ParentSpanID, span.SpanID = traceID, parent, randomHex(8)
	h := http.Header{}
	copyHeaders(h, r.Header)
	h.Del("Authorization")
	h.Del("Tracestate")
	if ok && r.Header.Get("Tracestate") != "" {
		h.Set("Tracestate", r.Header.Get("Tracestate"))
	}
	h.Set("Traceparent", "00-"+traceID+"-"+span.SpanID+"-"+flags)
	h.Set("X-Request-Id", span.RequestID)
	return &Outbound{Method: r.Method, Path: r.URL.EscapedPath(), RawQuery: r.URL.RawQuery, Header: h, Body: body}, span
}

// UpstreamError is a failure talking to the upstream. Committed reports
// whether any part of the response (status line included) was already sent
// to the client: an uncommitted failure may be retried on another worker, a
// committed one never is (a retried stream would repeat or splice tokens).
type UpstreamError struct {
	Committed  bool
	StatusCode int // the upstream's status when it answered (Retry503), else 0
	Err        error
}

func (e *UpstreamError) Error() string {
	if e.Committed {
		return "upstream failed after the first byte: " + e.Err.Error()
	}
	return "upstream failed before the first byte: " + e.Err.Error()
}

func (e *UpstreamError) Unwrap() error {
	return e.Err
}

// Forward sends out to base (the engine's URL without /v1) on ctx and relays
// the answer with StreamProxy. A failure before the upstream answers (refused,
// reset, or a 503 with Retry503) returns an uncommitted *UpstreamError and
// writes nothing to w. ctx is the client's request context: when the client
// leaves, the upstream request is cancelled.
func Forward(ctx context.Context, w http.ResponseWriter, base string, out *Outbound, client *http.Client, o StreamObserver) error {
	if client == nil {
		client = &http.Client{}
	}
	target := strings.TrimRight(base, "/") + out.Path
	if out.RawQuery != "" {
		target += "?" + out.RawQuery
	}
	up, err := http.NewRequestWithContext(ctx, out.Method, target, bytes.NewReader(out.Body))
	if err != nil {
		return &UpstreamError{Err: err}
	}
	up.ContentLength = int64(len(out.Body))
	for k, vs := range out.Header {
		up.Header[k] = append([]string(nil), vs...)
	}
	resp, err := client.Do(up)
	if err != nil {
		return &UpstreamError{Err: err}
	}
	defer resp.Body.Close()
	if out.Retry503 && resp.StatusCode == http.StatusServiceUnavailable {
		return &UpstreamError{StatusCode: resp.StatusCode, Err: errors.New("the upstream answered 503")}
	}
	if err := StreamProxy(w, resp.Request, resp, o); err != nil {
		return &UpstreamError{Committed: true, Err: err}
	}
	return nil
}

// StreamObserver watches one response as it passes through: OnFirstByte
// just before the first body byte is written to the client, OnChunk with the
// data of every complete SSE event (comments skipped, "[DONE]" included),
// and OnDone with the usage when the response ended cleanly (zero when the
// upstream reported none). gw.07 meters TTFT and usage with it; gw.06
// records streams for replay.
type StreamObserver interface {
	OnFirstByte()
	OnChunk(data []byte)
	OnDone(u server.Usage)
}

type exchangeObserver struct{ ex *server.Exchange }

func (e exchangeObserver) OnFirstByte() {
	e.ex.MarkFirstByte()
}

func (e exchangeObserver) OnChunk([]byte) {
}

func (e exchangeObserver) OnDone(u server.Usage) {
	e.ex.SetUsage(u)
}

// ExchangeObserver records the first byte and the usage on ex, which is how
// the ratelimit stage settles and the meter stage records (gw.01 chain).
func ExchangeObserver(ex *server.Exchange) StreamObserver {
	return exchangeObserver{ex}
}

// Observers fans one response out to several observers.
type Observers []StreamObserver

func (os Observers) OnFirstByte() {
	for _, o := range os {
		o.OnFirstByte()
	}
}

func (os Observers) OnChunk(d []byte) {
	for _, o := range os {
		o.OnChunk(d)
	}
}

func (os Observers) OnDone(u server.Usage) {
	for _, o := range os {
		o.OnDone(u)
	}
}

// usageOf returns the usage object of one JSON document (a chunk or a whole
// response), and whether it had a non-null one.
func usageOf(doc []byte) (server.Usage, bool) {
	if !bytes.Contains(doc, []byte(`"usage"`)) {
		return server.Usage{}, false
	}
	var v struct {
		Usage *server.Usage `json:"usage"`
	}
	if json.Unmarshal(doc, &v) != nil || v.Usage == nil {
		return server.Usage{}, false
	}
	u := *v.Usage
	if u.TotalTokens == 0 {
		u.TotalTokens = u.PromptTokens + u.CompletionTokens
	}
	return u, true
}

// sseSplitter cuts a byte stream into SSE events at blank lines, whatever
// the read boundaries, and hands each event's data to emit.
type sseSplitter struct {
	buf  []byte
	emit func(data []byte)
}

func (s *sseSplitter) write(p []byte) {
	s.buf = append(s.buf, p...)
	for {
		i := bytes.Index(s.buf, []byte("\n\n"))
		if i < 0 {
			return
		}
		event := s.buf[:i]
		s.buf = s.buf[i+2:]
		var data [][]byte
		for _, line := range bytes.Split(event, []byte("\n")) {
			line = bytes.TrimSuffix(line, []byte("\r"))
			if bytes.HasPrefix(line, []byte("data:")) {
				data = append(data, bytes.TrimPrefix(bytes.TrimPrefix(line, []byte("data:")), []byte(" ")))
			}
		}
		if data != nil {
			s.emit(bytes.Join(data, []byte("\n")))
		}
	}
}

// StreamProxy relays up to w: the end-to-end headers (w's own X-Request-Id
// kept), the status, then every body chunk as it arrives, flushed before the
// next read and never re-framed. For text/event-stream it also cuts the
// passing bytes into events for o.OnChunk and takes the usage from the last
// event that carries one; for a JSON body it takes the usage from the body.
// If the upstream breaks mid-body, an SSE response gets one error event
// (never a silent truncation) and the error is returned; if the client went
// away (r's context done), nothing more is written. o may be nil.
func StreamProxy(w http.ResponseWriter, r *http.Request, up *http.Response, o StreamObserver) error {
	if o == nil {
		o = Observers(nil)
	}
	reqID := w.Header().Get("X-Request-Id")
	copyHeaders(w.Header(), up.Header)
	if reqID != "" {
		w.Header().Set("X-Request-Id", reqID)
	}
	w.WriteHeader(up.StatusCode)
	sse := strings.HasPrefix(up.Header.Get("Content-Type"), "text/event-stream")
	var usage server.Usage
	var whole bytes.Buffer
	split := &sseSplitter{emit: func(d []byte) {
		o.OnChunk(d)
		if u, ok := usageOf(d); ok {
			usage = u
		}
	}}
	rc := http.NewResponseController(w)
	buf := make([]byte, 32*1024)
	first := true
	for {
		n, err := up.Body.Read(buf)
		if n > 0 {
			if first {
				o.OnFirstByte()
				first = false
			}
			if _, werr := w.Write(buf[:n]); werr != nil {
				return werr
			}
			if ferr := rc.Flush(); ferr != nil && !errors.Is(ferr, http.ErrNotSupported) {
				return ferr
			}
			if sse {
				split.write(buf[:n])
			} else if whole.Len() < server.DefaultMaxBody {
				whole.Write(buf[:n])
			}
		}
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			if r.Context().Err() == nil && sse {
				_, _ = io.WriteString(w, `data: {"error":{"message":"the upstream stream failed","type":"server_error","param":null,"code":null}}`+"\n\n")
				_ = rc.Flush()
			}
			return fmt.Errorf("reading the upstream body: %w", err)
		}
	}
	if !sse {
		if u, ok := usageOf(whole.Bytes()); ok {
			usage = u
		}
	}
	o.OnDone(usage)
	return nil
}

// authorized reports whether header is "Bearer <key>" for the configured key,
// comparing in constant time so the response time leaks nothing about it.
func authorized(header, key string) bool {
	if key == "" {
		return false
	}
	const scheme = "bearer "
	if len(header) < len(scheme) || !strings.EqualFold(header[:len(scheme)], scheme) {
		return false
	}
	presented := strings.TrimSpace(header[len(scheme):])
	return subtle.ConstantTimeCompare([]byte(presented), []byte(key)) == 1
}

// ParseTraceparent splits a W3C traceparent header ("00-<32 hex trace id>-
// <16 hex parent id>-<2 hex flags>"). ok is false for anything else: another
// version, uppercase hex, or an all-zero trace or parent id.
func ParseTraceparent(h string) (traceID, parentID, flags string, ok bool) {
	parts := strings.Split(strings.TrimSpace(h), "-")
	if len(parts) != 4 || parts[0] != "00" {
		return "", "", "", false
	}
	traceID, parentID, flags = parts[1], parts[2], parts[3]
	if !lowerHex(traceID, 32) || !lowerHex(parentID, 16) || !lowerHex(flags, 2) {
		return "", "", "", false
	}
	if strings.Trim(traceID, "0") == "" || strings.Trim(parentID, "0") == "" {
		return "", "", "", false
	}
	return traceID, parentID, flags, true
}

func lowerHex(s string, n int) bool {
	if len(s) != n {
		return false
	}
	for _, c := range s {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}

// requestID keeps the caller's X-Request-Id when it is printable ASCII of at
// most 128 bytes, and otherwise makes a new one.
func requestID(in string) string {
	if in != "" && len(in) <= 128 {
		printable := true
		for i := 0; i < len(in); i++ {
			if in[i] < 0x21 || in[i] > 0x7e {
				printable = false
				break
			}
		}
		if printable {
			return in
		}
	}
	return randomHex(16)
}

// randomHex returns n random bytes as 2n lowercase hex digits, never all zero.
func randomHex(n int) string {
	b := make([]byte, n)
	for {
		if _, err := rand.Read(b); err != nil {
			panic("proxy: crypto/rand failed: " + err.Error())
		}
		for _, x := range b {
			if x != 0 {
				return hex.EncodeToString(b)
			}
		}
	}
}

// copyHeaders copies every end-to-end header from src to dst.
func copyHeaders(dst, src http.Header) {
	for k, vs := range src {
		if isHop(k) {
			continue
		}
		for _, v := range vs {
			dst.Add(k, v)
		}
	}
}

func isHop(name string) bool {
	for _, h := range hopHeaders {
		if strings.EqualFold(name, h) {
			return true
		}
	}
	return false
}

// apiError is the OpenAI error shape every non-2xx gateway answer uses.
type apiError struct {
	Error struct {
		Message string  `json:"message"`
		Type    string  `json:"type"`
		Param   *string `json:"param"`
		Code    *string `json:"code"`
	} `json:"error"`
}

func writeError(w http.ResponseWriter, status int, typ, code, msg string) {
	var e apiError
	e.Error.Message, e.Error.Type = msg, typ
	if code != "" {
		e.Error.Code = &code
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(e)
}
