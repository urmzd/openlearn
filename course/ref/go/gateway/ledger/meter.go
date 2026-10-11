// The meter stage (gw.07): the last link of the chain, wrapped around the
// proxy (requestid -> ... -> route -> proxy -> meter), so it sees the whole
// exchange and writes one ledger row when the response ends.

package ledger

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"tinyllm/gateway/auth"
	"tinyllm/gateway/server"
)

// MeteredRoutes are the inference routes the meter records (usage.v1.sql
// `route`); every other path passes through unmetered.
var MeteredRoutes = map[string]bool{
	"/v1/chat/completions": true,
	"/v1/completions":      true,
	"/v1/embeddings":       true,
}

// MeterOptions configure the meter.
type MeterOptions struct {
	Clock server.Clock // nil = server.WallClock; the end time of the exchange
	// WriteTimeout bounds one ledger write; 0 = 5 s. The write never uses
	// the request's context: a client that hung up still gets its row.
	WriteTimeout time.Duration
	Logger       *log.Logger // nil = log.Default(); a failed write is logged, never sent to the client
}

// maxSniff caps how much of a non-streamed body the meter keeps to read its
// usage: the gateway's own request limit (DESIGN 2.7).
const maxSniff = server.DefaultMaxBody

// usageJSON is the usage object of a completion, a chunk, or an embedding list.
type usageJSON struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	TotalTokens         int `json:"total_tokens"`
	PromptTokensDetails *struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

// envelope is the part of a response body or SSE payload the meter reads.
type envelope struct {
	Model   string           `json:"model"`
	Choices *json.RawMessage `json:"choices"`
	Usage   *usageJSON       `json:"usage"`
	Error   *struct {
		Type string  `json:"type"`
		Code *string `json:"code"`
	} `json:"error"`
}

// WithIncludeUsage returns body with stream_options.include_usage set to
// true, and whether it changed anything. The engine sends a stream's usage
// only when asked (openai-subset.v1.yaml), and the ledger needs it.
func WithIncludeUsage(body []byte) ([]byte, bool, error) {
	// SOLUTION-BEGIN gw.07
	var m map[string]json.RawMessage
	if err := json.Unmarshal(body, &m); err != nil {
		return body, false, err
	}
	opts := map[string]json.RawMessage{}
	if raw, ok := m["stream_options"]; ok && string(raw) != "null" {
		if err := json.Unmarshal(raw, &opts); err != nil {
			return body, false, err
		}
	}
	if string(opts["include_usage"]) == "true" {
		return body, false, nil
	}
	opts["include_usage"] = json.RawMessage("true")
	raw, err := json.Marshal(opts)
	if err != nil {
		return body, false, err
	}
	m["stream_options"] = raw
	out, err := json.Marshal(m)
	if err != nil {
		return body, false, err
	}
	return out, true, nil
	// SOLUTION-END
}

// meterWriter forwards the response to the client and reads what the ledger
// needs on the way: usage, the served model, an error code, the first byte.
type meterWriter struct {
	http.ResponseWriter
	ex    *server.Exchange
	strip bool // the meter asked for the usage chunk, so the client must not see it

	status    int
	sse       bool
	pend      []byte       // an SSE event not yet complete
	body      bytes.Buffer // a non-SSE body, up to maxSniff
	usage     *usageJSON
	served    string
	errCode   string
	wroteBody bool
}

func (m *meterWriter) WriteHeader(code int) {
	// SOLUTION-BEGIN gw.07
	if m.status == 0 {
		m.status = code
		m.sse = strings.HasPrefix(m.Header().Get("Content-Type"), "text/event-stream")
	}
	m.ResponseWriter.WriteHeader(code)
	// SOLUTION-END
}

func (m *meterWriter) Write(b []byte) (int, error) {
	// SOLUTION-BEGIN gw.07
	if m.status == 0 {
		m.WriteHeader(http.StatusOK)
	}
	if !m.sse {
		if room := maxSniff - m.body.Len(); room > 0 {
			m.body.Write(b[:min(len(b), room)])
		}
		return m.forward(b)
	}
	m.pend = append(m.pend, b...)
	for {
		i := bytes.Index(m.pend, []byte("\n\n"))
		if i < 0 {
			return len(b), nil
		}
		ev := m.pend[:i+2]
		keep := m.inspect(ev)
		if keep {
			if _, err := m.forward(ev); err != nil {
				return 0, err
			}
		}
		m.pend = m.pend[i+2:]
	}
	// SOLUTION-END
}

// forward writes to the client and marks the first content byte.
func (m *meterWriter) forward(b []byte) (int, error) {
	// SOLUTION-BEGIN gw.07
	if len(b) > 0 && m.status < 400 && !m.wroteBody {
		m.wroteBody = true
		m.ex.MarkFirstByte()
	}
	return m.ResponseWriter.Write(b)
	// SOLUTION-END
}

// inspect reads one complete SSE event and reports whether the client
// should receive it: every event but the usage chunk the meter asked for.
func (m *meterWriter) inspect(ev []byte) bool {
	// SOLUTION-BEGIN gw.07
	var data []byte
	for _, line := range bytes.Split(ev, []byte("\n")) {
		if p, ok := bytes.CutPrefix(line, []byte("data:")); ok {
			data = append(data, bytes.TrimPrefix(p, []byte(" "))...)
		}
	}
	if len(data) == 0 || bytes.Equal(data, []byte("[DONE]")) {
		return true
	}
	var e envelope
	if json.Unmarshal(data, &e) != nil {
		return true
	}
	if m.served == "" {
		m.served = e.Model
	}
	if e.Error != nil {
		m.errCode = errorCode(e.Error.Type, e.Error.Code, m.status)
	}
	if e.Usage != nil {
		m.usage = e.Usage
		if m.strip && e.Choices != nil && strings.TrimSpace(string(*e.Choices)) == "[]" {
			return false
		}
	}
	return true
	// SOLUTION-END
}

// Flush sends what is complete; a half-received event waits for its end.
func (m *meterWriter) Flush() {
	// SOLUTION-BEGIN gw.07
	_ = http.NewResponseController(m.ResponseWriter).Flush()
	// SOLUTION-END
}

func (m *meterWriter) Unwrap() http.ResponseWriter {
	// SOLUTION-BEGIN gw.07
	return m.ResponseWriter
	// SOLUTION-END
}

// finish sends a trailing partial event (a stream that ended without its
// blank line keeps every byte) and reads a non-streamed body.
func (m *meterWriter) finish() {
	// SOLUTION-BEGIN gw.07
	if len(m.pend) > 0 {
		_ = m.inspect(m.pend)
		_, _ = m.forward(m.pend)
		m.pend = nil
	}
	if m.sse || m.body.Len() == 0 {
		return
	}
	var e envelope
	if json.Unmarshal(m.body.Bytes(), &e) != nil {
		return
	}
	m.usage = e.Usage
	m.served = e.Model
	if e.Error != nil {
		m.errCode = errorCode(e.Error.Type, e.Error.Code, m.status)
	}
	// SOLUTION-END
}

// errorCode is the error's code, else its type, else "http_<status>".
func errorCode(typ string, code *string, status int) string {
	// SOLUTION-BEGIN gw.07
	switch {
	case code != nil && *code != "":
		return *code
	case typ != "":
		return typ
	default:
		return "http_" + strconv.Itoa(status)
	}
	// SOLUTION-END
}

// traceID is the trace id of a valid traceparent header, else "".
func traceID(h string) string {
	// SOLUTION-BEGIN gw.07
	parts := strings.Split(h, "-")
	if len(parts) != 4 || len(parts[1]) != 32 || strings.Trim(parts[1], "0") == "" {
		return ""
	}
	for _, c := range parts[1] {
		if !strings.ContainsRune("0123456789abcdef", c) {
			return ""
		}
	}
	return parts[1]
	// SOLUTION-END
}

// Meter is the meter stage: for a metered route it asks the engine for a
// stream's usage (and hides that chunk from a client that did not ask for
// it), forwards the response untouched otherwise, and when the response
// ends writes one UsageRecord built from the Exchange, the Principal, and
// what it read from the response.
func Meter(l Ledger, o MeterOptions) server.Middleware {
	// SOLUTION-BEGIN gw.07
	clock := o.Clock
	if clock == nil {
		clock = server.WallClock
	}
	timeout := o.WriteTimeout
	if timeout <= 0 {
		timeout = 5 * time.Second
	}
	logger := o.Logger
	if logger == nil {
		logger = log.Default()
	}
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			ex := server.ExchangeFrom(r.Context())
			if ex == nil || r.Method != http.MethodPost || !MeteredRoutes[r.URL.Path] {
				next.ServeHTTP(w, r)
				return
			}
			req, err := ex.Request(r)
			if err != nil {
				server.WriteError(w, http.StatusBadRequest, "invalid_request_error", "", "", err.Error())
				return
			}
			mw := &meterWriter{ResponseWriter: w, ex: ex}
			if req.Stream && r.URL.Path != "/v1/embeddings" {
				if body, changed, err := WithIncludeUsage(req.Body); err == nil && changed {
					// The routing proxy (gw.05) forwards the Exchange's parsed
					// body, and later stages re-read r.Body from it: change both.
					req.Body = body
					r.Body = io.NopCloser(bytes.NewReader(body))
					r.ContentLength = int64(len(body))
					r.Header.Set("Content-Length", strconv.Itoa(len(body)))
					mw.strip = true
				}
			}
			next.ServeHTTP(mw, r)
			mw.finish()

			st := ex.Snapshot()
			status := mw.status
			if status == 0 {
				status = st.Status
			}
			if status == 0 {
				status = http.StatusOK
			}
			p, _ := auth.PrincipalFrom(r.Context())
			rec := UsageRecord{
				RequestID:  ex.RequestID,
				Start:      ex.Start,
				Tenant:     p.Tenant,
				KeyID:      p.KeyID,
				Model:      req.Model,
				Route:      r.URL.Path,
				APIVersion: "1",
				Status:     status,
				ErrorCode:  mw.errCode,
				Stream:     req.Stream,
				E2E:        clock.Now().Sub(ex.Start),
				WorkerID:   st.Worker,
				TraceID:    traceID(r.Header.Get("traceparent")),
			}
			if r.Header.Get("X-TL-API-Version") == "2" {
				rec.APIVersion = "2"
			}
			if hit, _ := st.Attrs["tl.cache.hit"].(bool); hit {
				rec.CacheHit = true
			}
			if status >= 400 && rec.ErrorCode == "" {
				rec.ErrorCode = "http_" + strconv.Itoa(status)
			}
			if status < 400 {
				rec.ServedModel = mw.served
				u := st.Usage
				if mw.usage != nil {
					u = server.Usage{PromptTokens: mw.usage.PromptTokens, CompletionTokens: mw.usage.CompletionTokens, TotalTokens: mw.usage.TotalTokens}
					if d := mw.usage.PromptTokensDetails; d != nil {
						rec.CachedTokens = d.CachedTokens
					}
				}
				rec.PromptTokens, rec.CompletionTokens = u.PromptTokens, u.CompletionTokens
				if !st.FirstByte.IsZero() {
					d := st.FirstByte.Sub(ex.Start)
					rec.TTFT = &d
				}
			}
			ctx, cancel := context.WithTimeout(context.WithoutCancel(r.Context()), timeout)
			defer cancel()
			if err := l.Record(ctx, rec); err != nil {
				logger.Printf("ledger: request %s not recorded: %v", rec.RequestID, err)
			}
		})
	}
	// SOLUTION-END
}
