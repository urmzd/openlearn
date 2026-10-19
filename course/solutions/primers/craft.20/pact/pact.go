// Package pact is the course's small consumer-driven contract kit for
// craft.20 (a pact file plus a mock provider), in the spirit of Pact.
//
// A pact file is JSON written by the CONSUMER (the gateway) about the
// PROVIDER (the engine): every interaction the gateway relies on, as the
// request it sends and the response it expects back.
//
//	{"consumer": "gateway", "provider": "engine", "interactions": [
//	  {"description": "a streamed chat with usage",
//	   "request":  {"method": "POST", "path": "/v1/chat/completions",
//	                "headers": {"X-TL-Priority": "5"}, "absentHeaders": ["Authorization"],
//	                "body": {"stream": true, "stream_options": {"include_usage": true}}},
//	   "response": {"status": 200, "headers": {"Content-Type": "text/event-stream"},
//	                "events": ["data: {...}\n\n", "data: [DONE]\n\n"], "splitEvents": true}}]}
//
// Matching (the request side, checked by the mock): method and path exactly;
// every listed header equal (names case-insensitive); every absentHeaders
// name absent; body: every key of the pact body present in the request body
// with an equal value, recursively (extra keys allowed). The response side is
// replayed as written: `body` as JSON, or `events` as raw bytes, flushed one
// event at a time (each split in two writes when splitEvents is true).
package pact

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"
)

// Request is the request side of an interaction.
type Request struct {
	Method        string            `json:"method"`
	Path          string            `json:"path"`
	Headers       map[string]string `json:"headers,omitempty"`
	AbsentHeaders []string          `json:"absentHeaders,omitempty"`
	Body          json.RawMessage   `json:"body,omitempty"`
}

// Response is the response side of an interaction.
type Response struct {
	Status      int               `json:"status"`
	Headers     map[string]string `json:"headers,omitempty"`
	Body        json.RawMessage   `json:"body,omitempty"`
	Events      []string          `json:"events,omitempty"`
	SplitEvents bool              `json:"splitEvents,omitempty"`
}

// Interaction is one request and the response the consumer expects.
type Interaction struct {
	Description string   `json:"description"`
	Request     Request  `json:"request"`
	Response    Response `json:"response"`
}

// File is a whole pact.
type File struct {
	Consumer     string        `json:"consumer"`
	Provider     string        `json:"provider"`
	Interactions []Interaction `json:"interactions"`
}

// Load reads a pact file or fails the test.
func Load(t testing.TB, path string) File {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("pact: %v", err)
	}
	var f File
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatalf("pact %s: %v", path, err)
	}
	return f
}

// Find returns the interaction with this description or fails the test.
func (f File) Find(t testing.TB, description string) Interaction {
	t.Helper()
	for _, it := range f.Interactions {
		if it.Description == description {
			return it
		}
	}
	t.Fatalf("pact: no interaction %q", description)
	return Interaction{}
}

// Mock is a provider that serves one interaction.
type Mock struct {
	URL string
	srv *httptest.Server

	mu       sync.Mutex
	calls    int
	mismatch []string
}

// Calls is how many requests reached the mock.
func (m *Mock) Calls() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.calls
}

// Mismatches lists how the requests differed from the pact.
func (m *Mock) Mismatches() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]string(nil), m.mismatch...)
}

// Serve starts a mock provider for it. When the test ends, a request that
// did not match the pact fails the test (the consumer broke the contract).
func Serve(t testing.TB, it Interaction) *Mock {
	t.Helper()
	m := &Mock{}
	m.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		problems := Match(it.Request, r, body)
		m.mu.Lock()
		m.calls++
		m.mismatch = append(m.mismatch, problems...)
		m.mu.Unlock()
		Replay(w, it.Response)
	}))
	m.URL = m.srv.URL
	t.Cleanup(func() {
		m.srv.Close()
		for _, p := range m.Mismatches() {
			t.Errorf("pact %q: the consumer's request broke the contract: %s", it.Description, p)
		}
	})
	return m
}

// Match lists how a request differs from the pact's request side.
func Match(want Request, r *http.Request, body []byte) []string {
	var out []string
	if want.Method != "" && r.Method != want.Method {
		out = append(out, fmt.Sprintf("method %s, want %s", r.Method, want.Method))
	}
	if want.Path != "" && r.URL.Path != want.Path {
		out = append(out, fmt.Sprintf("path %s, want %s", r.URL.Path, want.Path))
	}
	for k, v := range want.Headers {
		if got := r.Header.Get(k); got != v {
			out = append(out, fmt.Sprintf("header %s = %q, want %q", k, got, v))
		}
	}
	for _, k := range want.AbsentHeaders {
		if _, ok := r.Header[http.CanonicalHeaderKey(k)]; ok {
			out = append(out, fmt.Sprintf("header %s must be absent", k))
		}
	}
	if len(want.Body) > 0 {
		var w, g any
		if err := json.Unmarshal(want.Body, &w); err != nil {
			return append(out, "the pact's request body is not JSON")
		}
		if err := json.Unmarshal(body, &g); err != nil {
			return append(out, fmt.Sprintf("the request body is not JSON: %q", body))
		}
		if p := subset(w, g, "body"); p != "" {
			out = append(out, p)
		}
	}
	return out
}

// subset reports where got lacks something of want ("" when it has it all).
func subset(want, got any, at string) string {
	switch w := want.(type) {
	case map[string]any:
		g, ok := got.(map[string]any)
		if !ok {
			return fmt.Sprintf("%s is %v, want an object", at, got)
		}
		for k, v := range w {
			gv, ok := g[k]
			if !ok {
				return fmt.Sprintf("%s.%s is missing", at, k)
			}
			if p := subset(v, gv, at+"."+k); p != "" {
				return p
			}
		}
		return ""
	default:
		if !reflect.DeepEqual(want, got) {
			return fmt.Sprintf("%s = %v, want %v", at, got, want)
		}
		return ""
	}
}

// Replay writes a pact response.
func Replay(w http.ResponseWriter, resp Response) {
	for k, v := range resp.Headers {
		w.Header().Set(k, v)
	}
	if len(resp.Events) == 0 && w.Header().Get("Content-Type") == "" {
		w.Header().Set("Content-Type", "application/json")
	}
	w.WriteHeader(resp.Status)
	if len(resp.Events) == 0 {
		_, _ = w.Write(resp.Body)
		return
	}
	rc := http.NewResponseController(w)
	for _, ev := range resp.Events {
		parts := []string{ev}
		if resp.SplitEvents && len(ev) > 1 {
			parts = []string{ev[:len(ev)/2], ev[len(ev)/2:]}
		}
		for _, p := range parts {
			if _, err := io.WriteString(w, p); err != nil {
				return
			}
			_ = rc.Flush()
		}
	}
}

// Events joins the data payloads of a pact's events (for assertions).
func Events(resp Response) []string {
	var out []string
	for _, ev := range resp.Events {
		for _, line := range strings.Split(ev, "\n") {
			if v, ok := strings.CutPrefix(line, "data:"); ok {
				out = append(out, strings.TrimPrefix(v, " "))
			}
		}
	}
	return out
}
