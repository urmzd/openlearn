// Package faketool is a rule-based fake OpenAI-compatible chat provider
// (B12): no network, no model, deterministic. A rule matches the
// LAST message of the conversation (its role and text) and says what the
// assistant answers: plain content or tool calls. Request-hash cassettes
// never match prompts the agent under test builds; last-message rules do.
//
//	{"model": "faketool", "rules": [
//	  {"when": {"role": "user", "contains": "weather"},
//	   "reply": {"tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}]}},
//	  {"when": {"role": "tool", "regex": "sunny"}, "reply": {"content": "It is sunny in Paris."}},
//	  {"when": {}, "reply": {"content": "I do not know."}}]}
//
// Serves POST /v1/chat/completions (stream or not), GET /v1/models, and
// GET /healthz. Responses carry `created: 0` and ids derived from a counter,
// so two runs produce the same bytes. Every request is recorded.
package faketool

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"regexp"
	"strings"
	"sync"
)

type When struct {
	Role     string `json:"role,omitempty"`     // user, tool, system, assistant; empty matches any
	Contains string `json:"contains,omitempty"` // substring of the last message text
	Regex    string `json:"regex,omitempty"`
	Tool     string `json:"tool,omitempty"` // the last message answers this tool (tool role, by name)
	re       *regexp.Regexp
}

type ToolCall struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

type Reply struct {
	Content   string     `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

type Rule struct {
	When  When  `json:"when"`
	Reply Reply `json:"reply"`
}

type Rules struct {
	Model string `json:"model"`
	Rules []Rule `json:"rules"`
}

// Load reads a rules file.
func Load(path string) (*Rules, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var r Rules
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return &r, r.compile()
}

func (r *Rules) compile() error {
	if r.Model == "" {
		r.Model = "faketool"
	}
	for i := range r.Rules {
		if p := r.Rules[i].When.Regex; p != "" {
			re, err := regexp.Compile(p)
			if err != nil {
				return fmt.Errorf("rule %d: %w", i+1, err)
			}
			r.Rules[i].When.re = re
		}
	}
	return nil
}

type Message struct {
	Role       string `json:"role"`
	Content    any    `json:"content"`
	Name       string `json:"name,omitempty"`
	ToolCallID string `json:"tool_call_id,omitempty"`
}

func text(c any) string {
	switch v := c.(type) {
	case string:
		return v
	case []any: // content parts
		var b strings.Builder
		for _, p := range v {
			if m, ok := p.(map[string]any); ok {
				if t, ok := m["text"].(string); ok {
					b.WriteString(t)
				}
			}
		}
		return b.String()
	}
	return ""
}

// Match returns the first rule matching the last message, or nil.
func (r *Rules) Match(msgs []Message) *Rule {
	if len(msgs) == 0 {
		return nil
	}
	last := msgs[len(msgs)-1]
	t := text(last.Content)
	for i := range r.Rules {
		w := r.Rules[i].When
		if w.Role != "" && w.Role != last.Role {
			continue
		}
		if w.Contains != "" && !strings.Contains(t, w.Contains) {
			continue
		}
		if w.re != nil && !w.re.MatchString(t) {
			continue
		}
		if w.Tool != "" && !(last.Role == "tool" && last.Name == w.Tool) {
			continue
		}
		return &r.Rules[i]
	}
	return nil
}

type Server struct {
	rules    *Rules
	mu       sync.Mutex
	n        int
	requests []map[string]any
	srv      *http.Server
	ln       net.Listener
}

// Start serves rules on addr ("127.0.0.1:0" for a free port).
func Start(rules *Rules, addr string) (*Server, error) {
	if err := rules.compile(); err != nil {
		return nil, err
	}
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, err
	}
	s := &Server{rules: rules, ln: ln}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", s.chat)
	mux.HandleFunc("/v1/models", s.models)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte("ok")) })
	s.srv = &http.Server{Handler: mux}
	go s.srv.Serve(ln)
	return s, nil
}

func (s *Server) BaseURL() string { return "http://" + s.ln.Addr().String() + "/v1" }
func (s *Server) Close() error    { return s.srv.Close() }

// Requests are the decoded request bodies, in order.
func (s *Server) Requests() []map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]map[string]any(nil), s.requests...)
}

func errorJSON(w http.ResponseWriter, code int, typ, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"message": msg, "type": typ, "param": nil, "code": nil}})
}

func (s *Server) models(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": []any{map[string]any{"id": s.rules.Model, "object": "model", "created": 0, "owned_by": "faketool"}}})
}

func (s *Server) chat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		errorJSON(w, 405, "invalid_request_error", "POST only")
		return
	}
	var raw map[string]any
	if err := json.NewDecoder(r.Body).Decode(&raw); err != nil {
		errorJSON(w, 400, "invalid_request_error", "body is not JSON")
		return
	}
	b, _ := json.Marshal(raw["messages"])
	var msgs []Message
	if err := json.Unmarshal(b, &msgs); err != nil || len(msgs) == 0 {
		errorJSON(w, 400, "invalid_request_error", "messages must be a non-empty array")
		return
	}
	s.mu.Lock()
	s.n++
	n := s.n
	s.requests = append(s.requests, raw)
	s.mu.Unlock()
	rule := s.rules.Match(msgs)
	if rule == nil {
		errorJSON(w, 422, "invalid_request_error", "faketool: no rule matches the last message")
		return
	}
	if tc, ok := raw["tool_choice"].(string); ok && tc == "none" && len(rule.Reply.ToolCalls) > 0 {
		rule = &Rule{Reply: Reply{Content: "(tool_choice none)"}}
	}
	model, _ := raw["model"].(string)
	if model == "" {
		model = s.rules.Model
	}
	id := fmt.Sprintf("chatcmpl-fake-%d", n)
	calls := make([]any, 0, len(rule.Reply.ToolCalls))
	for i, c := range rule.Reply.ToolCalls {
		args, _ := json.Marshal(c.Arguments)
		calls = append(calls, map[string]any{"id": fmt.Sprintf("call_%d_%d", n, i), "type": "function", "function": map[string]any{"name": c.Name, "arguments": string(args)}})
	}
	finish := "stop"
	if len(calls) > 0 {
		finish = "tool_calls"
	}
	promptTok := 0
	for _, m := range msgs {
		promptTok += len(strings.Fields(text(m.Content)))
	}
	words := strings.Fields(rule.Reply.Content)
	usage := map[string]any{"prompt_tokens": promptTok, "completion_tokens": len(words) + len(calls), "total_tokens": promptTok + len(words) + len(calls)}
	if stream, _ := raw["stream"].(bool); stream {
		w.Header().Set("Content-Type", "text/event-stream")
		fl, _ := w.(http.Flusher)
		send := func(delta map[string]any, fin any) {
			ch := map[string]any{"id": id, "object": "chat.completion.chunk", "created": 0, "model": model,
				"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": fin}}}
			b, _ := json.Marshal(ch)
			fmt.Fprintf(w, "data: %s\n\n", b)
			if fl != nil {
				fl.Flush()
			}
		}
		send(map[string]any{"role": "assistant"}, nil)
		for i, wd := range words {
			if i > 0 {
				wd = " " + wd
			}
			send(map[string]any{"content": wd}, nil)
		}
		for i, c := range calls {
			f := c.(map[string]any)["function"].(map[string]any)
			args := f["arguments"].(string)
			half := len(args) / 2
			send(map[string]any{"tool_calls": []any{map[string]any{"index": i, "id": c.(map[string]any)["id"], "type": "function", "function": map[string]any{"name": f["name"], "arguments": args[:half]}}}}, nil)
			send(map[string]any{"tool_calls": []any{map[string]any{"index": i, "function": map[string]any{"arguments": args[half:]}}}}, nil)
		}
		send(map[string]any{}, finish)
		fmt.Fprint(w, "data: [DONE]\n\n")
		return
	}
	msg := map[string]any{"role": "assistant", "content": nil}
	if rule.Reply.Content != "" || len(calls) == 0 {
		msg["content"] = rule.Reply.Content
	}
	if len(calls) > 0 {
		msg["tool_calls"] = calls
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"id": id, "object": "chat.completion", "created": 0, "model": model,
		"choices": []any{map[string]any{"index": 0, "message": msg, "finish_reason": finish}},
		"usage":   usage,
	})
}
