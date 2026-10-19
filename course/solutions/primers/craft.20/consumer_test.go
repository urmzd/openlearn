// Consumer-driven contract tests: the gateway (consumer) against mocks of
// the engine (provider) generated from pacts/gateway-engine.json. Each test
// sends what the gateway really sends, through the client, and asserts what
// the gateway really needs from the answer. The mock fails the test when the
// request breaks the pact (a forwarded key, a client-chosen priority, ...).
package craft20_test

import (
	"context"
	"errors"
	"net/http"
	"testing"
	"time"

	"craft20"
	"craft20/pact"
)

const traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-b7ad6b7169203331-01"

func contract(t *testing.T) pact.File {
	return pact.Load(t, "pacts/gateway-engine.json")
}

// clientHeaders is what a client sends the gateway, hostile parts included:
// its own key, a priority it chose, a KV handle it forged.
func clientHeaders() http.Header {
	h := http.Header{}
	h.Set("Authorization", "Bearer tl_abcdefghijkl_0123456789abcdefghijklmnopqrstuv")
	h.Set("X-TL-Priority", "99")
	h.Set("X-TL-KV-Handle", "forged")
	h.Set("Content-Type", "application/json")
	return h
}

func forward(path, body string) craft20.Forward {
	return craft20.Forward{Path: path, Body: []byte(body), Header: clientHeaders(), Priority: 5,
		RequestID: "req-7", Traceparent: traceparent}
}

func do(t *testing.T, base string, f craft20.Forward) *http.Response {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	t.Cleanup(cancel)
	req, err := (&craft20.Client{Base: base}).NewRequest(ctx, f)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func TestChatCompletion(t *testing.T) {
	m := pact.Serve(t, contract(t).Find(t, "a chat completion"))
	resp := do(t, m.URL+"/", forward("/v1/chat/completions", `{"model":"smol","messages":[{"role":"user","content":"Once"}]}`))
	c, err := craft20.ReadCompletion(resp)
	if err != nil {
		t.Fatal(err)
	}
	if c.Text != "upon a time" || c.FinishReason != "stop" || c.Model != "smol-135m@v3" ||
		c.Usage != (craft20.Usage{PromptTokens: 12, CompletionTokens: 30, TotalTokens: 42}) {
		t.Fatalf("got %+v", c)
	}
	if m.Calls() != 1 {
		t.Fatalf("%d calls", m.Calls())
	}
}

func TestStreamWithUsage(t *testing.T) {
	m := pact.Serve(t, contract(t).Find(t, "a streamed chat with usage"))
	resp := do(t, m.URL, forward("/v1/chat/completions", `{"model":"smol","stream":true,"messages":[{"role":"user","content":"Once"}]}`))
	var text string
	sum, err := craft20.ReadStream(resp, func(c craft20.Chunk) error { text += c.Content; return nil })
	if err != nil {
		t.Fatal(err)
	}
	if text != "Once upon" || sum.Chunks != 3 || sum.FinishReason != "length" || sum.Model != "smol-135m@v3" {
		t.Fatalf("text %q, summary %+v", text, sum)
	}
	if sum.Usage == nil || *sum.Usage != (craft20.Usage{PromptTokens: 12, CompletionTokens: 30, TotalTokens: 42}) {
		t.Fatalf("usage %+v", sum.Usage)
	}
}

func TestStreamFailureIsAnError(t *testing.T) {
	m := pact.Serve(t, contract(t).Find(t, "a stream that fails after its first token"))
	resp := do(t, m.URL, forward("/v1/chat/completions", `{"model":"smol","stream":true,"messages":[]}`))
	n := 0
	_, err := craft20.ReadStream(resp, func(craft20.Chunk) error { n++; return nil })
	var ae *craft20.APIError
	if !errors.As(err, &ae) || ae.Status != 200 || ae.Code != "internal_error" || ae.Type != "server_error" || n != 2 {
		t.Fatalf("err %v (%+v), %d chunks before it", err, ae, n)
	}
}

func TestTruncatedStream(t *testing.T) {
	m := pact.Serve(t, contract(t).Find(t, "a stream cut off before [DONE]"))
	resp := do(t, m.URL, forward("/v1/chat/completions", `{"model":"smol","stream":true,"messages":[]}`))
	_, err := craft20.ReadStream(resp, nil)
	if !errors.Is(err, craft20.ErrTruncated) {
		t.Fatalf("err %v, want ErrTruncated", err)
	}
}

func TestDataWithoutSpace(t *testing.T) {
	m := pact.Serve(t, contract(t).Find(t, "a stream with no space after data:"))
	resp := do(t, m.URL, forward("/v1/chat/completions", `{"model":"smol","stream":true,"messages":[]}`))
	var text string
	sum, err := craft20.ReadStream(resp, func(c craft20.Chunk) error { text += c.Content; return nil })
	if err != nil || text != "Hi" || sum.FinishReason != "stop" {
		t.Fatalf("err %v text %q summary %+v", err, text, sum)
	}
}

func TestQueueFull(t *testing.T) {
	m := pact.Serve(t, contract(t).Find(t, "the engine's admission queue is full"))
	resp := do(t, m.URL, forward("/v1/chat/completions", `{"model":"smol","messages":[]}`))
	_, err := craft20.ReadCompletion(resp)
	var ae *craft20.APIError
	if !errors.As(err, &ae) || ae.Status != 429 || ae.Code != "rate_limit_exceeded" || ae.RetryAfter != 2*time.Second {
		t.Fatalf("err %v (%+v)", err, ae)
	}
}

func TestUnknownModel(t *testing.T) {
	m := pact.Serve(t, contract(t).Find(t, "an unknown model"))
	resp := do(t, m.URL, forward("/v1/chat/completions", `{"model":"nope","messages":[]}`))
	_, err := craft20.ReadCompletion(resp)
	var ae *craft20.APIError
	if !errors.As(err, &ae) || ae.Status != 404 || ae.Code != "model_not_found" || ae.Param != "" {
		t.Fatalf("err %v (%+v)", err, ae)
	}
}

func TestTextCompletion(t *testing.T) {
	m := pact.Serve(t, contract(t).Find(t, "a text completion"))
	f := forward("/v1/completions", `{"model":"smol","prompt":"Once"}`)
	f.Priority = 0
	c, err := craft20.ReadCompletion(do(t, m.URL, f))
	if err != nil || c.Text != " upon a time" || c.Usage.TotalTokens != 8 {
		t.Fatalf("err %v completion %+v", err, c)
	}
}

func TestTokenize(t *testing.T) {
	m := pact.Serve(t, contract(t).Find(t, "a token count"))
	n, err := (&craft20.Client{Base: m.URL}).Tokenize(context.Background(), "smol", "Once upon a time")
	if err != nil || n != 4 {
		t.Fatalf("n %d err %v", n, err)
	}
}
