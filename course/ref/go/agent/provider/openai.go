// Package provider is the agent SDK's model client (ag.01): one Provider for
// every OpenAI-compatible endpoint (the learner's gateway, an engine, or a
// frontier API with the same code), the SSE decoder that turns a streamed
// answer into typed deltas, and the retry wrapper that retries only before
// the first content delta.
//
// Chapter: content/tracks/ai-platform-engineering/13-agent-sdk/01-types-and-provider.md.
// Wire contract: course/contracts/openapi/openai-subset.v1.yaml.
package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"tinyllm/agent/types"
)

// Config says where the model is.
type Config struct {
	// BaseURL ends in /v1, for example "http://localhost:30080/v1".
	BaseURL string
	// APIKey is sent as "Authorization: Bearer <APIKey>"; empty sends none.
	APIKey string
	// Model is the default model; types.WithModel overrides it per call.
	Model string
	// Client sends requests. nil means a client with no overall timeout:
	// streams last as long as generation does, and ctx cancels them.
	Client *http.Client
	// Headers are added to every request (X-Request-Id, traceparent, ...).
	Headers map[string]string
}

// OpenAI is a Provider over POST {BaseURL}/chat/completions with stream: true.
type OpenAI struct{ cfg Config }

func NewOpenAI(cfg Config) *OpenAI {
	// SOLUTION-BEGIN ag.01
	if cfg.Client == nil {
		cfg.Client = &http.Client{}
	}
	cfg.BaseURL = strings.TrimRight(cfg.BaseURL, "/")
	return &OpenAI{cfg: cfg}
	// SOLUTION-END
}

// APIError is a request the server refused before streaming: the HTTP status
// and the OpenAI error body. RetryAfter is the Retry-After header (seconds),
// zero when absent.
type APIError struct {
	Status     int
	Type, Code string
	Message    string
	RetryAfter time.Duration
}

func (e *APIError) Error() string {
	return fmt.Sprintf("provider: HTTP %d %s: %s", e.Status, e.Type, e.Message)
}

type wireFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type wireCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function wireFunction `json:"function"`
}

type wireMessage struct {
	Role       string     `json:"role"`
	Content    *string    `json:"content"`
	ToolCalls  []wireCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
}

type wireTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description,omitempty"`
		Parameters  json.RawMessage `json:"parameters,omitempty"`
	} `json:"function"`
}

// RequestBody is the JSON body ChatStream sends. Assistant tool-call
// arguments go out as a string (the wire type), an assistant message with
// only tool calls has `content: null`, and every tool message names the
// call it answers.
func RequestBody(model string, msgs []types.Message, tools []types.ToolDef, o types.CallOptions) ([]byte, error) {
	// SOLUTION-BEGIN ag.01
	wm := make([]wireMessage, 0, len(msgs))
	for _, m := range msgs {
		w := wireMessage{Role: string(m.Role)}
		content := m.Content
		switch m.Role {
		case types.RoleAssistant:
			if content != "" || len(m.ToolCalls) == 0 {
				w.Content = &content
			}
			for _, c := range m.ToolCalls {
				args := string(c.Args)
				if args == "" {
					args = "{}"
				}
				w.ToolCalls = append(w.ToolCalls, wireCall{ID: c.ID, Type: "function", Function: wireFunction{Name: c.Name, Arguments: args}})
			}
		case types.RoleTool:
			w.Content = &content
			w.ToolCallID = m.ToolCallID
			w.Name = m.Name
		default:
			w.Content = &content
		}
		wm = append(wm, w)
	}
	body := map[string]any{
		"model":          model,
		"messages":       wm,
		"stream":         true,
		"stream_options": map[string]any{"include_usage": true},
	}
	if len(tools) > 0 {
		wt := make([]wireTool, len(tools))
		for i, t := range tools {
			wt[i].Type = "function"
			wt[i].Function.Name = t.Name
			wt[i].Function.Description = t.Description
			wt[i].Function.Parameters = t.Parameters
		}
		body["tools"] = wt
	}
	if o.ToolChoice != "" {
		body["tool_choice"] = o.ToolChoice
	}
	if o.Temperature != nil {
		body["temperature"] = *o.Temperature
	}
	if o.MaxTokens > 0 {
		body["max_tokens"] = o.MaxTokens
	}
	if o.Seed != nil {
		body["seed"] = *o.Seed
	}
	return json.Marshal(body)
	// SOLUTION-END
}

// ChatStream sends the request and streams the answer. A non-2xx answer is
// returned as an *APIError before any delta. The channel is closed after a
// DoneDelta or an ErrorDelta, or when ctx is cancelled (which also aborts
// the HTTP request).
func (p *OpenAI) ChatStream(ctx context.Context, msgs []types.Message, tools []types.ToolDef, opts ...types.CallOption) (<-chan types.Delta, error) {
	// SOLUTION-BEGIN ag.01
	o := types.ApplyOptions(opts)
	model := p.cfg.Model
	if o.Model != "" {
		model = o.Model
	}
	body, err := RequestBody(model, msgs, tools, o)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.cfg.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	if p.cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.cfg.APIKey)
	}
	for k, v := range p.cfg.Headers {
		req.Header.Set(k, v)
	}
	resp, err := p.cfg.Client.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode/100 != 2 {
		defer resp.Body.Close()
		return nil, apiError(resp)
	}
	ch := make(chan types.Delta)
	go func() {
		defer close(ch)
		defer resp.Body.Close()
		send := func(d types.Delta) bool {
			select {
			case ch <- d:
				return true
			case <-ctx.Done():
				return false
			}
		}
		rd := NewEventReader(resp.Body)
		dec := NewDecoder()
		for {
			data, err := rd.Next()
			if err != nil {
				if err == io.EOF || err == io.ErrUnexpectedEOF {
					err = ErrNoDone
				}
				if ctx.Err() != nil {
					err = ctx.Err()
				}
				send(types.ErrorDelta{Err: err})
				return
			}
			out, done := dec.Event(data)
			for _, d := range out {
				if !send(d) {
					return
				}
			}
			if done {
				return
			}
		}
	}()
	return ch, nil
	// SOLUTION-END
}

// apiError reads an OpenAI error body ({"error": {message, type, code}}).
func apiError(resp *http.Response) *APIError {
	// SOLUTION-BEGIN ag.01
	e := &APIError{Status: resp.StatusCode}
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	var body struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    any    `json:"code"`
		} `json:"error"`
	}
	if json.Unmarshal(b, &body) == nil && body.Error.Message != "" {
		e.Message, e.Type = body.Error.Message, body.Error.Type
		if s, ok := body.Error.Code.(string); ok {
			e.Code = s
		}
	} else {
		e.Message = strings.TrimSpace(string(b))
	}
	if s := resp.Header.Get("Retry-After"); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n > 0 {
			e.RetryAfter = time.Duration(n) * time.Second
		}
	}
	return e
	// SOLUTION-END
}
