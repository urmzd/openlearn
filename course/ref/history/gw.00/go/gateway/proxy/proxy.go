// Package proxy is the tracer gateway's request path (gw.00): one static API
// key, then the request forwarded to the engine and its response streamed
// back byte for byte, with W3C trace context and X-Request-Id carried across.
//
// Contract: course/contracts/openapi/openai-subset.v0.yaml (gateway tier) and
// course/contracts/spec/cli-roles.md (role `gateway`, tracer form).
// Chapter: content/tracks/ai-platform-engineering/12-gateway/00-streaming-proxy.md.
// gw.04 takes this package over and adds the StreamObserver.
package proxy

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"time"
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

// NewProxy returns the gateway's handler for every API path.
func NewProxy(cfg Config) http.Handler {
	// SOLUTION-BEGIN gw.00
	client := cfg.Client
	if client == nil {
		client = &http.Client{}
	}
	base := strings.TrimRight(cfg.Upstream, "/")
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		serve(w, r, cfg, client, base)
	})
	// SOLUTION-END
}

// statusWriter remembers the status code written, for the span.
type statusWriter struct {
	http.ResponseWriter
	code int
}

func (s *statusWriter) WriteHeader(code int) {
	// SOLUTION-BEGIN gw.00
	if s.code == 0 {
		s.code = code
	}
	s.ResponseWriter.WriteHeader(code)
	// SOLUTION-END
}

func (s *statusWriter) Unwrap() http.ResponseWriter {
	// SOLUTION-BEGIN gw.00
	return s.ResponseWriter
	// SOLUTION-END
}

func serve(w http.ResponseWriter, r *http.Request, cfg Config, client *http.Client, base string) {
	// SOLUTION-BEGIN gw.00
	span := Span{Name: SpanName, Start: time.Now(), RequestID: requestID(r.Header.Get("X-Request-Id"))}
	sw := &statusWriter{ResponseWriter: w}
	defer func() {
		if cfg.OnSpan != nil {
			span.End = time.Now()
			span.StatusCode = sw.code
			cfg.OnSpan(span)
		}
	}()
	w.Header().Set("X-Request-Id", span.RequestID)

	// 1. The key, before anything leaves the gateway.
	if !authorized(r.Header.Get("Authorization"), cfg.APIKey) {
		writeError(sw, http.StatusUnauthorized, "invalid_request_error", "invalid_api_key",
			"Incorrect or missing API key: send the header Authorization: Bearer KEY.")
		return
	}

	// 2. Trace context: continue the caller's trace, or start one.
	traceID, parent, flags, ok := ParseTraceparent(r.Header.Get("Traceparent"))
	if !ok {
		traceID, parent, flags = randomHex(16), "", "01"
	}
	span.TraceID, span.ParentSpanID, span.SpanID = traceID, parent, randomHex(8)

	// 3. The upstream request: same method, path, query, and body, on the
	// client's context so a client that leaves cancels the engine's work.
	target := base + r.URL.EscapedPath()
	if r.URL.RawQuery != "" {
		target += "?" + r.URL.RawQuery
	}
	up, err := http.NewRequestWithContext(r.Context(), r.Method, target, r.Body)
	if err != nil {
		writeError(sw, http.StatusBadGateway, "server_error", "no_capacity", "cannot build the upstream request")
		return
	}
	up.ContentLength = r.ContentLength
	copyHeaders(up.Header, r.Header)
	up.Header.Del("Authorization")
	up.Header.Del("Tracestate")
	if ok && r.Header.Get("Tracestate") != "" {
		up.Header.Set("Tracestate", r.Header.Get("Tracestate"))
	}
	up.Header.Set("Traceparent", "00-"+traceID+"-"+span.SpanID+"-"+flags)
	up.Header.Set("X-Request-Id", span.RequestID)

	resp, err := client.Do(up)
	if err != nil {
		if r.Context().Err() != nil {
			return // the client left; nobody is listening
		}
		writeError(sw, http.StatusServiceUnavailable, "server_error", "no_capacity",
			"the upstream engine is unavailable: "+err.Error())
		return
	}
	defer resp.Body.Close()

	// 4. Status and headers first, then the body as it arrives.
	copyHeaders(w.Header(), resp.Header)
	w.Header().Set("X-Request-Id", span.RequestID)
	sw.WriteHeader(resp.StatusCode)
	err = copyFlush(sw, resp.Body)
	if err != nil && r.Context().Err() == nil &&
		strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		// A failure after the first byte is an SSE error event, never a
		// silent truncation (openai-subset.v0.yaml, Streaming).
		_, _ = io.WriteString(sw, `data: {"error":{"message":"the upstream stream failed","type":"server_error","param":null,"code":null}}`+"\n\n")
		_ = http.NewResponseController(sw).Flush()
	}
	// SOLUTION-END
}

// copyFlush writes every chunk the upstream sends and flushes it before the
// next read, so SSE events reach the client one by one.
func copyFlush(w http.ResponseWriter, body io.Reader) error {
	// SOLUTION-BEGIN gw.00
	rc := http.NewResponseController(w)
	buf := make([]byte, 32*1024)
	for {
		n, err := body.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				return werr
			}
			if ferr := rc.Flush(); ferr != nil && !errors.Is(ferr, http.ErrNotSupported) {
				return ferr
			}
		}
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
	}
	// SOLUTION-END
}

// authorized reports whether header is "Bearer <key>" for the configured key,
// comparing in constant time so the response time leaks nothing about it.
func authorized(header, key string) bool {
	// SOLUTION-BEGIN gw.00
	if key == "" {
		return false
	}
	const scheme = "bearer "
	if len(header) < len(scheme) || !strings.EqualFold(header[:len(scheme)], scheme) {
		return false
	}
	presented := strings.TrimSpace(header[len(scheme):])
	return subtle.ConstantTimeCompare([]byte(presented), []byte(key)) == 1
	// SOLUTION-END
}

// ParseTraceparent splits a W3C traceparent header ("00-<32 hex trace id>-
// <16 hex parent id>-<2 hex flags>"). ok is false for anything else: another
// version, uppercase hex, or an all-zero trace or parent id.
func ParseTraceparent(h string) (traceID, parentID, flags string, ok bool) {
	// SOLUTION-BEGIN gw.00
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
	// SOLUTION-END
}

func lowerHex(s string, n int) bool {
	// SOLUTION-BEGIN gw.00
	if len(s) != n {
		return false
	}
	for _, c := range s {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
	// SOLUTION-END
}

// requestID keeps the caller's X-Request-Id when it is printable ASCII of at
// most 128 bytes, and otherwise makes a new one.
func requestID(in string) string {
	// SOLUTION-BEGIN gw.00
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
	// SOLUTION-END
}

// randomHex returns n random bytes as 2n lowercase hex digits, never all zero.
func randomHex(n int) string {
	// SOLUTION-BEGIN gw.00
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
	// SOLUTION-END
}

// copyHeaders copies every end-to-end header from src to dst.
func copyHeaders(dst, src http.Header) {
	// SOLUTION-BEGIN gw.00
	for k, vs := range src {
		if isHop(k) {
			continue
		}
		for _, v := range vs {
			dst.Add(k, v)
		}
	}
	// SOLUTION-END
}

func isHop(name string) bool {
	// SOLUTION-BEGIN gw.00
	for _, h := range hopHeaders {
		if strings.EqualFold(name, h) {
			return true
		}
	}
	return false
	// SOLUTION-END
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
	// SOLUTION-BEGIN gw.00
	var e apiError
	e.Error.Message, e.Error.Type = msg, typ
	if code != "" {
		e.Error.Code = &code
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(e)
	// SOLUTION-END
}
