// Package craft20 is the craft.20 kata: the gateway's side of the
// gateway-to-engine contract, as one small client. The gateway is the
// CONSUMER of the engine's API (openapi/openai-subset.v1.yaml, engine tier);
// this file is everything the gateway relies on: how it builds the upstream
// request, and how it reads a completion, a stream, an error, and a token
// count. The consumer-driven contract (pacts/gateway-engine.json) and the
// tests (consumer_test.go) pin those expectations down, so a change to any
// of them breaks a test.
//
// Chapter: content/tracks/software-craftsmanship/03-testing-mentality/06-contract-tests.md.
package craft20

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Client talks to one engine.
type Client struct {
	Base string       // engine base URL without /v1, e.g. "http://127.0.0.1:8000"
	HTTP *http.Client // nil = http.DefaultClient
}

// Forward is one client request the gateway forwards to an engine.
type Forward struct {
	Path        string      // "/v1/chat/completions", "/v1/completions", or "/v1/embeddings"
	Body        []byte      // the client's JSON body
	Header      http.Header // the client's request headers
	Priority    int         // the key's tier, sent as X-TL-Priority
	RequestID   string      // sent as X-Request-Id
	Traceparent string      // the gateway's span context, sent as traceparent ("" = none)
}

// Usage is the engine's usage object.
type Usage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
	CachedTokens     int // usage.prompt_tokens_details.cached_tokens (v2), else 0
}

// APIError is an engine answer in the OpenAI error shape: a non-2xx status,
// or an error event inside a stream (Status 200).
type APIError struct {
	Status     int
	Type       string
	Code       string // "" when null
	Param      string // "" when null
	Message    string
	RetryAfter time.Duration // from Retry-After on a 429, else 0
}

func (e *APIError) Error() string {
	return fmt.Sprintf("engine %d %s/%s: %s", e.Status, e.Type, e.Code, e.Message)
}

// ErrTruncated is a stream that ended without data: [DONE] and without an
// error event: the engine (or the network) dropped it.
var ErrTruncated = errors.New("the stream ended before data: [DONE]")

// Completion is a non-streamed answer.
type Completion struct {
	ID, Model, Text, FinishReason string
	Usage                         Usage
}

// Chunk is one data event of a stream.
type Chunk struct {
	Raw          []byte // the event as received, "data: ...\n\n"
	Model        string
	Role         string
	Content      string // delta.content (chat) or text (completions)
	FinishReason string
	Usage        *Usage // set only on the usage chunk (choices: [])
}

// Summary is what a whole stream amounted to.
type Summary struct {
	Model        string
	FinishReason string
	Chunks       int    // content chunks: every data event except the usage chunk
	Usage        *Usage // nil when the stream carried no usage chunk
}

// internal headers a client may never set (openai-subset.v1.yaml).
func internal(name string) bool {
	return strings.HasPrefix(strings.ToLower(name), "x-tl-")
}

// hop-by-hop and gateway-only headers that never go upstream.
var dropped = map[string]bool{
	"authorization": true, "connection": true, "keep-alive": true, "proxy-authorization": true,
	"te": true, "trailer": true, "transfer-encoding": true, "upgrade": true, "content-length": true,
	"traceparent": true, "x-request-id": true,
}

// NewRequest builds the upstream request: POST Base+Path with the body, and
// with stream_options.include_usage set when the body streams (the gateway
// meters every stream). Headers: the client's end-to-end headers minus
// Authorization and every X-TL-*; Content-Type application/json; Accept
// text/event-stream for a stream; X-TL-Priority from the key (never the
// client's); X-Request-Id; traceparent when set.
func (c *Client) NewRequest(ctx context.Context, f Forward) (*http.Request, error) {
	body := f.Body
	var probe struct {
		Stream bool `json:"stream"`
	}
	if err := json.Unmarshal(body, &probe); err != nil {
		return nil, fmt.Errorf("the request body is not JSON: %w", err)
	}
	if probe.Stream {
		var m map[string]json.RawMessage
		if err := json.Unmarshal(body, &m); err != nil {
			return nil, err
		}
		opts := map[string]json.RawMessage{}
		if raw, ok := m["stream_options"]; ok && string(raw) != "null" {
			if err := json.Unmarshal(raw, &opts); err != nil {
				return nil, err
			}
		}
		opts["include_usage"] = json.RawMessage("true")
		raw, _ := json.Marshal(opts)
		m["stream_options"] = raw
		b, err := json.Marshal(m)
		if err != nil {
			return nil, err
		}
		body = b
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(c.Base, "/")+f.Path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	for k, vs := range f.Header {
		if internal(k) || dropped[strings.ToLower(k)] {
			continue
		}
		for _, v := range vs {
			req.Header.Add(k, v)
		}
	}
	req.Header.Set("Content-Type", "application/json")
	if probe.Stream {
		req.Header.Set("Accept", "text/event-stream")
	}
	req.Header.Set("X-TL-Priority", strconv.Itoa(f.Priority))
	if f.RequestID != "" {
		req.Header.Set("X-Request-Id", f.RequestID)
	}
	if f.Traceparent != "" {
		req.Header.Set("traceparent", f.Traceparent)
	}
	return req, nil
}

func (c *Client) http() *http.Client {
	if c.HTTP != nil {
		return c.HTTP
	}
	return http.DefaultClient
}

type usageJSON struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	TotalTokens         int `json:"total_tokens"`
	PromptTokensDetails *struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

func (u *usageJSON) usage() *Usage {
	if u == nil {
		return nil
	}
	out := &Usage{PromptTokens: u.PromptTokens, CompletionTokens: u.CompletionTokens, TotalTokens: u.TotalTokens}
	if u.PromptTokensDetails != nil {
		out.CachedTokens = u.PromptTokensDetails.CachedTokens
	}
	return out
}

type errorJSON struct {
	Error *struct {
		Message string  `json:"message"`
		Type    string  `json:"type"`
		Param   *string `json:"param"`
		Code    *string `json:"code"`
	} `json:"error"`
}

// apiError turns an error body into an *APIError ("" for null fields); a
// body that is not the error shape still gives an error with the status.
func apiError(status int, body []byte) *APIError {
	e := &APIError{Status: status, Type: "server_error", Message: strings.TrimSpace(string(body))}
	var ej errorJSON
	if json.Unmarshal(body, &ej) == nil && ej.Error != nil {
		e.Type, e.Message = ej.Error.Type, ej.Error.Message
		if ej.Error.Code != nil {
			e.Code = *ej.Error.Code
		}
		if ej.Error.Param != nil {
			e.Param = *ej.Error.Param
		}
	}
	return e
}

// ReadCompletion reads a non-streamed answer and closes the body. A 2xx is a
// chat.completion (choices[0].message.content) or a text_completion
// (choices[0].text); anything else is an *APIError, with RetryAfter from the
// Retry-After header (seconds) on a 429.
func ReadCompletion(resp *http.Response) (Completion, error) {
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return Completion{}, err
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		e := apiError(resp.StatusCode, body)
		if resp.StatusCode == http.StatusTooManyRequests {
			if s, err := strconv.Atoi(resp.Header.Get("Retry-After")); err == nil && s >= 0 {
				e.RetryAfter = time.Duration(s) * time.Second
			}
		}
		return Completion{}, e
	}
	var cj struct {
		ID      string `json:"id"`
		Model   string `json:"model"`
		Choices []struct {
			Text    *string `json:"text"`
			Message *struct {
				Content *string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage *usageJSON `json:"usage"`
	}
	if err := json.Unmarshal(body, &cj); err != nil {
		return Completion{}, fmt.Errorf("the engine's answer is not JSON: %w", err)
	}
	out := Completion{ID: cj.ID, Model: cj.Model}
	if len(cj.Choices) > 0 {
		ch := cj.Choices[0]
		out.FinishReason = ch.FinishReason
		switch {
		case ch.Message != nil && ch.Message.Content != nil:
			out.Text = *ch.Message.Content
		case ch.Text != nil:
			out.Text = *ch.Text
		}
	}
	if u := cj.Usage.usage(); u != nil {
		out.Usage = *u
	}
	return out, nil
}

// ReadStream reads a text/event-stream answer and closes the body. It calls
// onChunk for every data event in order (Raw holds the event's bytes),
// ignores comment lines (": ping"), accepts "data:" with or without the
// space, and reassembles events split across reads. It returns at
// data: [DONE]. An error event ({"error": ...}) returns an *APIError with
// Status 200; an end of stream before [DONE] returns ErrTruncated. A non-2xx
// status is read as an error body, as ReadCompletion does.
func ReadStream(resp *http.Response, onChunk func(Chunk) error) (Summary, error) {
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		body, _ := io.ReadAll(resp.Body)
		return Summary{}, apiError(resp.StatusCode, body)
	}
	var sum Summary
	br := bufio.NewReader(resp.Body)
	var raw, data []byte
	for {
		line, err := br.ReadBytes('\n')
		if len(line) > 0 {
			raw = append(raw, line...)
			s := bytes.TrimRight(line, "\r\n")
			switch {
			case len(s) == 0: // the blank line ends an event
				if len(data) > 0 {
					done, cerr := dispatch(&sum, raw, data, onChunk)
					if cerr != nil || done {
						return sum, cerr
					}
				}
				raw, data = nil, nil
			case s[0] == ':': // a comment
			default:
				if v, ok := bytes.CutPrefix(s, []byte("data:")); ok {
					if len(data) > 0 {
						data = append(data, '\n')
					}
					data = append(data, bytes.TrimPrefix(v, []byte(" "))...)
				}
			}
		}
		if err != nil {
			if errors.Is(err, io.EOF) {
				return sum, ErrTruncated
			}
			return sum, err
		}
	}
}

// dispatch handles one complete data event; done is true at [DONE].
func dispatch(sum *Summary, raw, data []byte, onChunk func(Chunk) error) (done bool, err error) {
	if bytes.Equal(data, []byte("[DONE]")) {
		return true, nil
	}
	var ej errorJSON
	if json.Unmarshal(data, &ej) == nil && ej.Error != nil {
		return true, apiError(http.StatusOK, data)
	}
	var cj struct {
		Model   string `json:"model"`
		Choices []struct {
			Text  *string `json:"text"`
			Delta *struct {
				Role    string  `json:"role"`
				Content *string `json:"content"`
			} `json:"delta"`
			FinishReason *string `json:"finish_reason"`
		} `json:"choices"`
		Usage *usageJSON `json:"usage"`
	}
	if err := json.Unmarshal(data, &cj); err != nil {
		return true, fmt.Errorf("a stream event is not JSON: %q", data)
	}
	ch := Chunk{Raw: append([]byte(nil), raw...), Model: cj.Model, Usage: cj.Usage.usage()}
	if len(cj.Choices) > 0 {
		c := cj.Choices[0]
		if c.Delta != nil {
			ch.Role = c.Delta.Role
			if c.Delta.Content != nil {
				ch.Content = *c.Delta.Content
			}
		} else if c.Text != nil {
			ch.Content = *c.Text
		}
		if c.FinishReason != nil {
			ch.FinishReason = *c.FinishReason
			sum.FinishReason = ch.FinishReason
		}
		sum.Chunks++
	}
	if sum.Model == "" {
		sum.Model = cj.Model
	}
	if ch.Usage != nil {
		sum.Usage = ch.Usage
	}
	if onChunk != nil {
		if err := onChunk(ch); err != nil {
			return true, err
		}
	}
	return false, nil
}

// Tokenize counts the tokens of text with the engine's tokenizer
// (POST /v1/tokenize {model, text}, engine tier): the gateway's TPM cost.
func (c *Client) Tokenize(ctx context.Context, model, text string) (int, error) {
	body, _ := json.Marshal(map[string]any{"model": model, "text": text})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(c.Base, "/")+"/v1/tokenize", bytes.NewReader(body))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.http().Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, err
	}
	if resp.StatusCode != http.StatusOK {
		return 0, apiError(resp.StatusCode, b)
	}
	var tj struct {
		IDs *[]int `json:"ids"`
	}
	if err := json.Unmarshal(b, &tj); err != nil || tj.IDs == nil {
		return 0, fmt.Errorf("the tokenize answer has no ids: %q", b)
	}
	return len(*tj.IDs), nil
}
