package faketool_test

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"openlearn.urmzd.com/tl/testkit/faketool"
)

func rules() *faketool.Rules {
	return &faketool.Rules{Rules: []faketool.Rule{
		{When: faketool.When{Role: "user", Contains: "weather"}, Reply: faketool.Reply{ToolCalls: []faketool.ToolCall{{Name: "get_weather", Arguments: map[string]any{"city": "Paris"}}}}},
		{When: faketool.When{Role: "tool", Regex: "sun+y"}, Reply: faketool.Reply{Content: "It is sunny in Paris."}},
		{When: faketool.When{}, Reply: faketool.Reply{Content: "I do not know."}},
	}}
}

func post(t *testing.T, url string, body map[string]any) *http.Response {
	t.Helper()
	b, _ := json.Marshal(body)
	r, err := http.Post(url+"/chat/completions", "application/json", bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func TestToolCallThenAnswer(t *testing.T) {
	s, err := faketool.Start(rules(), "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	r := post(t, s.BaseURL(), map[string]any{"model": "m", "messages": []any{map[string]any{"role": "user", "content": "what is the weather?"}}})
	var out struct {
		Choices []struct {
			Message struct {
				Content   *string
				ToolCalls []struct {
					ID       string
					Function struct{ Name, Arguments string }
				} `json:"tool_calls"`
			}
			FinishReason string `json:"finish_reason"`
		}
	}
	json.NewDecoder(r.Body).Decode(&out)
	c := out.Choices[0]
	if c.FinishReason != "tool_calls" || c.Message.ToolCalls[0].Function.Name != "get_weather" || c.Message.ToolCalls[0].Function.Arguments != `{"city":"Paris"}` {
		t.Fatalf("tool call %+v", c)
	}
	r = post(t, s.BaseURL(), map[string]any{"messages": []any{
		map[string]any{"role": "user", "content": "what is the weather?"},
		map[string]any{"role": "tool", "content": "sunny, 21C", "tool_call_id": c.Message.ToolCalls[0].ID},
	}})
	b, _ := io.ReadAll(r.Body)
	if !strings.Contains(string(b), "It is sunny in Paris.") {
		t.Fatalf("answer %s", b)
	}
	if len(s.Requests()) != 2 {
		t.Fatal("requests not recorded")
	}
}

func TestStreamReassembles(t *testing.T) {
	s, _ := faketool.Start(rules(), "127.0.0.1:0")
	defer s.Close()
	r := post(t, s.BaseURL(), map[string]any{"stream": true, "messages": []any{map[string]any{"role": "user", "content": "weather please"}}})
	if r.Header.Get("Content-Type") != "text/event-stream" {
		t.Fatal(r.Header)
	}
	sc := bufio.NewScanner(r.Body)
	args, done, role := "", false, false
	for sc.Scan() {
		line := sc.Text()
		if line == "data: [DONE]" {
			done = true
			continue
		}
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var ch struct {
			Choices []struct {
				Delta struct {
					Role      string
					ToolCalls []struct{ Function struct{ Arguments string } } `json:"tool_calls"`
				}
			}
		}
		json.Unmarshal([]byte(line[6:]), &ch)
		d := ch.Choices[0].Delta
		role = role || d.Role == "assistant"
		for _, tc := range d.ToolCalls {
			args += tc.Function.Arguments
		}
	}
	if !done || !role || args != `{"city":"Paris"}` {
		t.Fatalf("stream: done %v role %v args %q", done, role, args)
	}
}

func TestNoRuleAndToolChoiceNone(t *testing.T) {
	s, _ := faketool.Start(&faketool.Rules{Rules: []faketool.Rule{{When: faketool.When{Role: "user", Contains: "weather"}, Reply: faketool.Reply{ToolCalls: []faketool.ToolCall{{Name: "w"}}}}}}, "127.0.0.1:0")
	defer s.Close()
	if r := post(t, s.BaseURL(), map[string]any{"messages": []any{map[string]any{"role": "user", "content": "hi"}}}); r.StatusCode != 422 {
		t.Fatalf("no rule: %d", r.StatusCode)
	}
	r := post(t, s.BaseURL(), map[string]any{"tool_choice": "none", "messages": []any{map[string]any{"role": "user", "content": "weather"}}})
	b, _ := io.ReadAll(r.Body)
	if strings.Contains(string(b), "tool_calls\":[") {
		t.Fatalf("tool_choice none still called a tool: %s", b)
	}
}
