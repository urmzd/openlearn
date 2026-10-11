// Course tests for ag.01: the agent SDK's types, the OpenAI-compatible
// provider (request body, SSE decoding into typed deltas), and the retry
// wrapper that retries only before the first content delta.
//
// Every server here is an httptest server on 127.0.0.1 or the course's fake
// provider (testkit/faketool): no network, no model. Retry waits go through
// RetryPolicy.Sleep, which these tests replace with a recorder, so no test
// sleeps.
package ag_01

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"openlearn.urmzd.com/tl/testkit/faketool"
	"tinyllm/agent/provider"
	"tinyllm/agent/types"
)

type fixture struct {
	Name      string           `json:"name"`
	SSE       string           `json:"sse"`
	Deltas    []map[string]any `json:"deltas"`
	ReadSizes []int            `json:"read_sizes"`
}

func fixtures(t *testing.T) map[string]fixture {
	t.Helper()
	dir := os.Getenv("TINYLLM_FIXTURES")
	if dir == "" {
		t.Fatal("TINYLLM_FIXTURES is not set (ol check sets it)")
	}
	b, err := os.ReadFile(filepath.Join(dir, "ag.01", "streams.json"))
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Streams []fixture `json:"streams"`
	}
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	out := map[string]fixture{}
	for _, s := range f.Streams {
		out[s.Name] = s
	}
	return out
}

// serveSSE answers every chat request with sse, written readSize bytes at a
// time with a flush after each piece, so the client sees arbitrary splits.
func serveSSE(t *testing.T, sse string, readSize int) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		fl := w.(http.Flusher)
		for i := 0; i < len(sse); i += readSize {
			j := min(i+readSize, len(sse))
			w.Write([]byte(sse[i:j]))
			fl.Flush()
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

// describe renders a delta in the fixture's vocabulary.
func describe(d types.Delta) map[string]any {
	switch v := d.(type) {
	case types.TextDelta:
		return map[string]any{"type": "text", "text": v.Text}
	case types.ToolCallStartDelta:
		return map[string]any{"type": "start", "index": float64(v.Index), "id": v.ID, "name": v.Name}
	case types.ToolCallArgsDelta:
		return map[string]any{"type": "args", "index": float64(v.Index), "fragment": v.Fragment}
	case types.ToolCallEndDelta:
		return map[string]any{"type": "end", "index": float64(v.Index), "id": v.Call.ID, "name": v.Call.Name, "args": string(v.Call.Args)}
	case types.UsageDelta:
		return map[string]any{"type": "usage", "in": float64(v.In), "out": float64(v.Out)}
	case types.DoneDelta:
		return map[string]any{"type": "done", "finish": v.FinishReason}
	case types.ErrorDelta:
		var se *provider.StreamError
		switch {
		case errors.As(v.Err, &se):
			return map[string]any{"type": "error", "kind": "stream_error"}
		case errors.Is(v.Err, provider.ErrNoDone):
			return map[string]any{"type": "error", "kind": "no_done"}
		}
		return map[string]any{"type": "error", "kind": "other: " + v.Err.Error()}
	}
	return map[string]any{"type": fmt.Sprintf("%T", d)}
}

func stream(t *testing.T, p types.Provider, msgs []types.Message, tools []types.ToolDef) []types.Delta {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	ch, err := p.ChatStream(ctx, msgs, tools)
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	var out []types.Delta
	for d := range ch {
		out = append(out, d)
	}
	return out
}

var hello = []types.Message{{Role: types.RoleUser, Content: "Say hello"}}

func TestHandExample(t *testing.T) {
	// WHY: the chapter's worked example (section 3): role chunk, "Hel", "lo",
	//      an empty-content chunk, finish stop, the usage chunk, [DONE],
	//      read 7 bytes at a time. Exactly four deltas come out, the empty
	//      chunk gives nothing, and Collect rebuilds "Hello" with usage 5/2.
	// KIND: unit
	// CATCHES: s04, s08
	// CHAPTER: ag.01 section 3, worked example
	f := fixtures(t)["text"]
	srv := serveSSE(t, f.SSE, 7)
	p := provider.NewOpenAI(provider.Config{BaseURL: srv.URL + "/v1", Model: "m"})
	got := stream(t, p, hello, nil)
	want := []types.Delta{types.TextDelta{Text: "Hel"}, types.TextDelta{Text: "lo"}, types.UsageDelta{In: 5, Out: 2}, types.DoneDelta{FinishReason: "stop"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("deltas = %#v\nwant %#v", got, want)
	}
	ch := make(chan types.Delta, len(got))
	for _, d := range got {
		ch <- d
	}
	close(ch)
	msg, u, err := types.Collect(ch)
	if err != nil || msg.Role != types.RoleAssistant || msg.Content != "Hello" || u != (types.Usage{In: 5, Out: 2}) {
		t.Fatalf("Collect = %+v, %+v, %v; want assistant \"Hello\", {5 2}, nil", msg, u, err)
	}
}

func TestRecordedStreams(t *testing.T) {
	// WHY: the contract's stream shapes, decoded at read sizes 1, 7, and
	//      64 KiB: interleaved tool-call fragments are assembled by index,
	//      calls close in index order at finish_reason (or at [DONE]),
	//      comments and CRLF line ends are framing, data lines join with
	//      "\n", an error event and an EOF before [DONE] both end in an error.
	// KIND: conformance
	// CATCHES: s01, s03, s05, s06, s07, s09, s10
	// CHAPTER: ag.01 section 2.2
	for _, f := range fixtures(t) {
		for _, n := range f.ReadSizes {
			t.Run(fmt.Sprintf("%s/read%d", f.Name, n), func(t *testing.T) {
				srv := serveSSE(t, f.SSE, n)
				p := provider.NewOpenAI(provider.Config{BaseURL: srv.URL + "/v1", Model: "m"})
				got := stream(t, p, hello, nil)
				var desc []map[string]any
				for _, d := range got {
					desc = append(desc, describe(d))
				}
				if !reflect.DeepEqual(desc, f.Deltas) {
					a, _ := json.Marshal(desc)
					b, _ := json.Marshal(f.Deltas)
					t.Fatalf("deltas\n got %s\nwant %s", a, b)
				}
			})
		}
	}
}

func TestRequestBody(t *testing.T) {
	// WHY: the wire shape the engine and the gateway accept: POST
	//      /v1/chat/completions with the bearer key, stream with usage,
	//      tools as functions, an assistant turn that only calls tools has
	//      content null and its arguments as a JSON string, and the tool
	//      turn names the call it answers. A wrong shape is a 400 from a
	//      real provider, or a model that never sees its own results.
	// KIND: unit
	// CATCHES: s11, s12, s13, s14, s15
	// CHAPTER: ag.01 section 4
	var body map[string]any
	var auth, path string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth, path = r.Header.Get("Authorization"), r.URL.Path
		json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()
	p := provider.NewOpenAI(provider.Config{BaseURL: srv.URL + "/v1", APIKey: "tl_k1_s3cret", Model: "smol"})
	msgs := []types.Message{
		{Role: types.RoleSystem, Content: "You are terse."},
		{Role: types.RoleUser, Content: "Weather in Paris?"},
		{Role: types.RoleAssistant, ToolCalls: []types.ToolCall{
			{ID: "call_1", Name: "get_weather", Args: json.RawMessage(`{"city":"Paris"}`)},
			{ID: "call_2", Name: "list_models"}, // no arguments at all: sent as {}
		}},
		{Role: types.RoleTool, ToolCallID: "call_1", Name: "get_weather", Content: "sunny"},
		{Role: types.RoleTool, ToolCallID: "call_2", Name: "list_models", Content: "smol"},
	}
	tools := []types.ToolDef{{Name: "get_weather", Description: "Weather now", Parameters: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`)}}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	ch, err := p.ChatStream(ctx, msgs, tools, types.WithTemperature(0), types.WithMaxTokens(64))
	if err != nil {
		t.Fatal(err)
	}
	for range ch {
	}
	if path != "/v1/chat/completions" || auth != "Bearer tl_k1_s3cret" {
		t.Fatalf("path %q auth %q; want /v1/chat/completions and the bearer key", path, auth)
	}
	want := map[string]any{
		"model":          "smol",
		"stream":         true,
		"stream_options": map[string]any{"include_usage": true},
		"temperature":    0.0,
		"max_tokens":     64.0,
		"messages": []any{
			map[string]any{"role": "system", "content": "You are terse."},
			map[string]any{"role": "user", "content": "Weather in Paris?"},
			map[string]any{"role": "assistant", "content": nil, "tool_calls": []any{
				map[string]any{"id": "call_1", "type": "function", "function": map[string]any{"name": "get_weather", "arguments": `{"city":"Paris"}`}},
				map[string]any{"id": "call_2", "type": "function", "function": map[string]any{"name": "list_models", "arguments": "{}"}},
			}},
			map[string]any{"role": "tool", "content": "sunny", "tool_call_id": "call_1", "name": "get_weather"},
			map[string]any{"role": "tool", "content": "smol", "tool_call_id": "call_2", "name": "list_models"},
		},
		"tools": []any{map[string]any{"type": "function", "function": map[string]any{
			"name": "get_weather", "description": "Weather now",
			"parameters": map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}, "required": []any{"city"}},
		}}},
	}
	if !reflect.DeepEqual(body, want) {
		a, _ := json.MarshalIndent(body, "", " ")
		t.Fatalf("request body:\n%s", a)
	}
}

// statusServer answers request i (from 0) with codes[i] (the last code
// repeats); 200 answers stream "Hello".
func statusServer(t *testing.T, codes []int, header http.Header) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var n atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		i := int(n.Add(1)) - 1
		code := codes[min(i, len(codes)-1)]
		if code != 200 {
			for k, v := range header {
				w.Header()[k] = v
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(code)
			fmt.Fprintf(w, `{"error":{"message":"status %d","type":"server_error","param":null,"code":"no_capacity"}}`, code)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, `data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}]}`+"\n\ndata: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)
	return srv, &n
}

type sleeps struct {
	mu sync.Mutex
	d  []time.Duration
}

func (s *sleeps) Sleep(_ context.Context, d time.Duration) error {
	s.mu.Lock()
	s.d = append(s.d, d)
	s.mu.Unlock()
	return nil
}

func TestAPIErrorNotRetried(t *testing.T) {
	// WHY: a 400 means the request is wrong; sending it again cannot help
	//      and wastes the budget. The status, type, and message reach the
	//      caller as an *APIError.
	// KIND: unit
	// CATCHES: s17
	// CHAPTER: ag.01 section 2.3
	srv, n := statusServer(t, []int{400}, nil)
	s := &sleeps{}
	p := provider.WithRetry(provider.NewOpenAI(provider.Config{BaseURL: srv.URL + "/v1"}), provider.RetryPolicy{Sleep: s.Sleep})
	_, err := p.ChatStream(context.Background(), hello, nil)
	var ae *provider.APIError
	if !errors.As(err, &ae) || ae.Status != 400 || ae.Type != "server_error" || ae.Message != "status 400" {
		t.Fatalf("err = %v; want *APIError 400 with the body's type and message", err)
	}
	if n.Load() != 1 || len(s.d) != 0 {
		t.Fatalf("%d requests, sleeps %v; want 1 request, no retry", n.Load(), s.d)
	}
}

func TestRetryBackoffSchedule(t *testing.T) {
	// WHY: 503 twice then 200: three requests, waits Initial then
	//      Initial*Multiplier; the Delay table doubles from 100 ms and stops
	//      at the 1 s cap. This is the formula of section 2.3.
	// KIND: unit
	// CATCHES: s18
	// CHAPTER: ag.01 section 2.3
	pol := provider.RetryPolicy{Initial: 100 * time.Millisecond, Max: time.Second, Multiplier: 2}
	for n, want := range []time.Duration{100, 200, 400, 800, 1000, 1000} {
		if got := pol.Delay(n+1, errors.New("x")); got != want*time.Millisecond {
			t.Errorf("Delay(%d) = %v, want %v", n+1, got, want*time.Millisecond)
		}
	}
	srv, n := statusServer(t, []int{503, 503, 200}, nil)
	s := &sleeps{}
	pol.Sleep = s.Sleep
	p := provider.WithRetry(provider.NewOpenAI(provider.Config{BaseURL: srv.URL + "/v1"}), pol)
	msg, _, err := types.Collect(must(t)(p.ChatStream(context.Background(), hello, nil)))
	if err != nil || msg.Content != "Hello" {
		t.Fatalf("got %q, %v", msg.Content, err)
	}
	if n.Load() != 3 || !reflect.DeepEqual(s.d, []time.Duration{100 * time.Millisecond, 200 * time.Millisecond}) {
		t.Fatalf("%d requests, sleeps %v; want 3 and [100ms 200ms]", n.Load(), s.d)
	}
}

func must(t *testing.T) func(<-chan types.Delta, error) <-chan types.Delta {
	return func(ch <-chan types.Delta, err error) <-chan types.Delta {
		t.Helper()
		if err != nil {
			t.Fatalf("ChatStream: %v", err)
		}
		return ch
	}
}

func TestRetryAfterHonored(t *testing.T) {
	// WHY: a 429 with Retry-After: 3 is the server saying when capacity
	//      returns; waiting less is refused again, so the wait is 3 s, not
	//      the computed 500 ms.
	// KIND: unit
	// CATCHES: s19
	// CHAPTER: ag.01 section 2.3
	srv, n := statusServer(t, []int{429, 200}, http.Header{"Retry-After": {"3"}})
	s := &sleeps{}
	p := provider.WithRetry(provider.NewOpenAI(provider.Config{BaseURL: srv.URL + "/v1"}), provider.RetryPolicy{Sleep: s.Sleep})
	if _, _, err := types.Collect(must(t)(p.ChatStream(context.Background(), hello, nil))); err != nil {
		t.Fatal(err)
	}
	if n.Load() != 2 || !reflect.DeepEqual(s.d, []time.Duration{3 * time.Second}) {
		t.Fatalf("%d requests, sleeps %v; want 2 and [3s]", n.Load(), s.d)
	}
}

func TestRetryGivesUpAtMaxAttempts(t *testing.T) {
	// WHY: MaxAttempts counts every attempt, the first included: with 3, a
	//      server that is always 503 sees exactly 3 requests, and the last
	//      error reaches the caller.
	// KIND: boundary
	// CATCHES: s20
	// CHAPTER: ag.01 section 2.3
	srv, n := statusServer(t, []int{503}, nil)
	s := &sleeps{}
	p := provider.WithRetry(provider.NewOpenAI(provider.Config{BaseURL: srv.URL + "/v1"}), provider.RetryPolicy{MaxAttempts: 3, Sleep: s.Sleep})
	_, err := p.ChatStream(context.Background(), hello, nil)
	var ae *provider.APIError
	if !errors.As(err, &ae) || ae.Status != 503 {
		t.Fatalf("err = %v, want the 503", err)
	}
	if n.Load() != 3 || len(s.d) != 2 {
		t.Fatalf("%d requests, %d sleeps; want 3 and 2", n.Load(), len(s.d))
	}
}

// resetAfter writes the role chunk and the given tokens, then resets the
// connection (RST) on the first attempt; later attempts get a whole answer.
func resetAfter(t *testing.T, tokens []string) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var n atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempt := n.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		fl := w.(http.Flusher)
		ev := func(content string) {
			fmt.Fprintf(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":%q},\"finish_reason\":null}]}\n\n", content)
			fl.Flush()
		}
		io.WriteString(w, `data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`+"\n\n")
		fl.Flush()
		if attempt == 1 {
			for _, tok := range tokens {
				ev(tok)
			}
			conn, _, err := w.(http.Hijacker).Hijack()
			if err != nil {
				t.Errorf("hijack: %v", err)
				return
			}
			conn.(*net.TCPConn).SetLinger(0) // close with RST
			conn.Close()
			return
		}
		ev("Hello")
		io.WriteString(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)
	return srv, &n
}

func TestRetryBeforeFirstContent(t *testing.T) {
	// WHY: the connection resets after the role chunk but before any text:
	//      nothing has been shown, so one retry is safe and the caller sees
	//      "Hello" once. Not retrying here turns a blip into a failed agent
	//      step.
	// KIND: fault
	// CATCHES: s24
	// CHAPTER: ag.01 section 2.3
	srv, n := resetAfter(t, nil)
	s := &sleeps{}
	p := provider.WithRetry(provider.NewOpenAI(provider.Config{BaseURL: srv.URL + "/v1"}), provider.RetryPolicy{Sleep: s.Sleep})
	msg, _, err := types.Collect(must(t)(p.ChatStream(context.Background(), hello, nil)))
	if err != nil || msg.Content != "Hello" || n.Load() != 2 {
		t.Fatalf("got %q, %v after %d requests; want \"Hello\" after 2", msg.Content, err, n.Load())
	}
}

func TestNoRetryAfterContent(t *testing.T) {
	// WHY: the reset comes after three tokens ("a", " b", " c"). The caller
	//      already has them; a retry would print "a b ca b c..." (duplicated
	//      text). The stream must end with an error and exactly one request.
	// KIND: fault
	// CATCHES: s16
	// CHAPTER: ag.01 section 2.3, the commitment rule
	srv, n := resetAfter(t, []string{"a", " b", " c"})
	s := &sleeps{}
	p := provider.WithRetry(provider.NewOpenAI(provider.Config{BaseURL: srv.URL + "/v1"}), provider.RetryPolicy{Sleep: s.Sleep})
	msg, _, err := types.Collect(must(t)(p.ChatStream(context.Background(), hello, nil)))
	if err == nil {
		t.Fatal("a stream cut after content must end with an error")
	}
	if msg.Content != "a b c" || n.Load() != 1 {
		t.Fatalf("text %q after %d requests; want \"a b c\" after 1 (no retry, no duplicate)", msg.Content, n.Load())
	}
}

func TestCancelStopsStream(t *testing.T) {
	// WHY: an agent that gives up (user pressed Ctrl-C, a budget ran out)
	//      must stop the generation it is paying for: cancelling ctx closes
	//      the delta channel and the server sees its request cancelled.
	// KIND: fault
	// CATCHES: s21
	// CHAPTER: ag.01 section 4
	gone := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, `data: {"choices":[{"index":0,"delta":{"content":"tick"},"finish_reason":null}]}`+"\n\n")
		w.(http.Flusher).Flush()
		select {
		case <-r.Context().Done():
			close(gone)
		case <-time.After(5 * time.Second):
		}
	}))
	defer srv.Close()
	ctx, cancel := context.WithCancel(context.Background())
	p := provider.NewOpenAI(provider.Config{BaseURL: srv.URL + "/v1"})
	ch := must(t)(p.ChatStream(ctx, hello, nil))
	if d := <-ch; d != (types.TextDelta{Text: "tick"}) {
		t.Fatalf("first delta %#v", d)
	}
	cancel()
	deadline := time.After(2 * time.Second)
	for open := true; open; {
		select {
		case _, open = <-ch:
		case <-deadline:
			t.Fatal("the delta channel stayed open after cancel")
		}
	}
	select {
	case <-gone:
	case <-time.After(2 * time.Second):
		t.Fatal("the server never saw the request cancelled")
	}
}

func TestFaketoolToolCall(t *testing.T) {
	// WHY: the course's fake provider (the PR-CI stand-in for a model,
	//      DESIGN 4.4) streams tool-call arguments in two halves, as a real
	//      engine does. The provider must rebuild get_weather({"city":
	//      "Paris"}), then send the tool result back and read the answer.
	// KIND: conformance
	// CATCHES: s13
	// CHAPTER: ag.01 section 6
	rules := &faketool.Rules{Rules: []faketool.Rule{
		{When: faketool.When{Role: "user", Contains: "weather"}, Reply: faketool.Reply{ToolCalls: []faketool.ToolCall{{Name: "get_weather", Arguments: map[string]any{"city": "Paris"}}}}},
		{When: faketool.When{Role: "tool", Tool: "get_weather"}, Reply: faketool.Reply{Content: "It is sunny in Paris."}},
	}}
	srv, err := faketool.Start(rules, "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()
	p := provider.NewOpenAI(provider.Config{BaseURL: srv.BaseURL(), Model: "faketool"})
	msgs := []types.Message{{Role: types.RoleUser, Content: "What is the weather in Paris?"}}
	msg, _, err := types.Collect(must(t)(p.ChatStream(context.Background(), msgs, nil)))
	if err != nil || len(msg.ToolCalls) != 1 || msg.ToolCalls[0].Name != "get_weather" {
		t.Fatalf("turn 1 = %+v, %v; want one get_weather call", msg, err)
	}
	var args map[string]any
	if json.Unmarshal(msg.ToolCalls[0].Args, &args) != nil || args["city"] != "Paris" {
		t.Fatalf("args %s; want {\"city\":\"Paris\"}", msg.ToolCalls[0].Args)
	}
	c := msg.ToolCalls[0]
	msgs = append(msgs, msg, types.Message{Role: types.RoleTool, ToolCallID: c.ID, Name: c.Name, Content: "sunny, 21 C"})
	msg, _, err = types.Collect(must(t)(p.ChatStream(context.Background(), msgs, nil)))
	if err != nil || msg.Content != "It is sunny in Paris." {
		t.Fatalf("turn 2 = %q, %v", msg.Content, err)
	}
	if got := srv.Requests()[1]["messages"].([]any)[2].(map[string]any)["tool_call_id"]; got != c.ID {
		t.Fatalf("the tool message answers %v, want %s", got, c.ID)
	}
}

func TestRetryableTable(t *testing.T) {
	// WHY: which failures are worth a second attempt, as one table:
	//      overload and server errors, broken connections, and cut streams
	//      are; client errors and a cancelled context are not.
	// KIND: unit
	// CATCHES: s17
	// CHAPTER: ag.01 section 2.3
	for _, tc := range []struct {
		err  error
		want bool
	}{
		{&provider.APIError{Status: 429}, true},
		{&provider.APIError{Status: 503}, true},
		{&provider.APIError{Status: 500}, true},
		{&provider.APIError{Status: 400}, false},
		{&provider.APIError{Status: 401}, false},
		{&provider.APIError{Status: 422}, false},
		{provider.ErrNoDone, true},
		{fmt.Errorf("read: %w", syscall.ECONNRESET), true},
		{&provider.StreamError{Type: "server_error"}, true},
		{&provider.StreamError{Type: "invalid_request_error"}, false},
		{context.Canceled, false},
		{context.DeadlineExceeded, false},
		{errors.New("something else"), false},
	} {
		if got := provider.Retryable(tc.err); got != tc.want {
			t.Errorf("Retryable(%v) = %v, want %v", tc.err, got, tc.want)
		}
	}
}

func TestAccumulatorOrdersCallsByIndex(t *testing.T) {
	// WHY: calls finish in any order on the wire; the message lists them by
	//      index, the order the model wrote them, which is the order the
	//      loop (ag.03) returns their results in. A call with no argument
	//      fragments has arguments {}.
	// KIND: unit
	// CATCHES: s22
	// CHAPTER: ag.01 section 2.1
	var a types.Accumulator
	for _, d := range []types.Delta{
		types.ToolCallStartDelta{Index: 1, ID: "b", Name: "second"},
		types.ToolCallStartDelta{Index: 0, ID: "a", Name: "first"},
		types.ToolCallArgsDelta{Index: 1, Fragment: `{"x":`},
		types.ToolCallArgsDelta{Index: 1, Fragment: `1}`},
		types.DoneDelta{FinishReason: "tool_calls"},
	} {
		a.Add(d)
	}
	m := a.Message()
	if a.Err() != nil || a.FinishReason() != "tool_calls" || len(m.ToolCalls) != 2 {
		t.Fatalf("message %+v err %v", m, a.Err())
	}
	if m.ToolCalls[0].ID != "a" || string(m.ToolCalls[0].Args) != "{}" || m.ToolCalls[1].ID != "b" || string(m.ToolCalls[1].Args) != `{"x":1}` {
		t.Fatalf("calls %+v; want a({}), b({\"x\":1})", m.ToolCalls)
	}
	var open types.Accumulator
	open.Add(types.TextDelta{Text: "partial"})
	if !errors.Is(open.Err(), types.ErrIncomplete) {
		t.Fatalf("a stream with no end delta: Err = %v, want ErrIncomplete", open.Err())
	}
}

func TestSeams(t *testing.T) {
	// WHY: the gate (ag.04) reads the call context the loop attaches; a
	//      caller that forgot to attach one must get the cautious answer
	//      (tainted), never "this context is clean". Step names are a
	//      contract the durable runner (ag.05) parses.
	// KIND: boundary
	// CATCHES: s23
	// CHAPTER: ag.01 section 2.4
	if !types.CallContextFrom(context.Background()).Tainted {
		t.Fatal("no CallContext attached must read as tainted")
	}
	cc := types.CallContextFrom(types.WithCallContext(context.Background(), types.CallContext{Tainted: false, Iter: 3}))
	if cc.Tainted || cc.Iter != 3 {
		t.Fatalf("attached context lost: %+v", cc)
	}
	if types.LLMStep(2) != "llm/2" || types.ToolStep(2, 1, "get_weather") != "tool/2/1/get_weather" {
		t.Fatal("step names differ from the contract")
	}
	if name, ok := types.StepToolName("tool/2/1/get_weather"); !ok || name != "get_weather" {
		t.Fatalf("StepToolName = %q, %v", name, ok)
	}
	if _, ok := types.StepToolName("llm/2"); ok {
		t.Fatal("an llm step has no tool name")
	}
	if !types.IsContent(types.TextDelta{Text: "x"}) || types.IsContent(types.UsageDelta{}) {
		t.Fatal("IsContent: text is content, usage is not")
	}
}
