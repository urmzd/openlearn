package agentkit

// fake.go: the faketool provider in process. The rules file is the format of
// course/testkit/go/faketool (DESIGN 4.4): a rule matches the LAST message of
// the conversation (role, substring, regex, answering tool) and says what the
// assistant answers, plain content or tool calls. The learner repo cannot
// import the course testkit (it is vendored only into course test overlays),
// so this file reads the same rules and streams the same deltas the HTTP
// faketool's SSE becomes after ag.01's parser: words as text deltas, then
// each call's start, argument fragments, and end, then usage and done. The
// same rules file also works with `faketool --rules` behind --base-url.

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
	"sync"

	"tinyllm/agent/types"
)

// FakeWhen is a rule's condition on the last message.
type FakeWhen struct {
	Role     string `json:"role,omitempty"`
	Contains string `json:"contains,omitempty"`
	Regex    string `json:"regex,omitempty"`
	Tool     string `json:"tool,omitempty"`
	re       *regexp.Regexp
}

// FakeCall is one tool call a rule answers with.
type FakeCall struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

// FakeReply is what the assistant answers.
type FakeReply struct {
	Content   string     `json:"content,omitempty"`
	ToolCalls []FakeCall `json:"tool_calls,omitempty"`
}

// FakeRule is one rule.
type FakeRule struct {
	When  FakeWhen  `json:"when"`
	Reply FakeReply `json:"reply"`
}

// Fake is the rule-based provider.
type Fake struct {
	Model string     `json:"model"`
	Rules []FakeRule `json:"rules"`
	mu    sync.Mutex
	n     int
}

// LoadFake reads a faketool rules file.
func LoadFake(path string) (*Fake, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var f Fake
	if err := json.Unmarshal(b, &f); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	for i := range f.Rules {
		if p := f.Rules[i].When.Regex; p != "" {
			re, err := regexp.Compile(p)
			if err != nil {
				return nil, fmt.Errorf("%s: rule %d: %w", path, i+1, err)
			}
			f.Rules[i].When.re = re
		}
	}
	return &f, nil
}

// match is the first rule matching the last message, or nil.
func (f *Fake) match(msgs []types.Message) *FakeRule {
	if len(msgs) == 0 {
		return nil
	}
	last := msgs[len(msgs)-1]
	for i := range f.Rules {
		w := f.Rules[i].When
		if w.Role != "" && w.Role != string(last.Role) {
			continue
		}
		if w.Contains != "" && !strings.Contains(last.Content, w.Contains) {
			continue
		}
		if w.re != nil && !w.re.MatchString(last.Content) {
			continue
		}
		if w.Tool != "" && !(last.Role == types.RoleTool && last.Name == w.Tool) {
			continue
		}
		return &f.Rules[i]
	}
	return nil
}

// ChatStream implements types.Provider.
func (f *Fake) ChatStream(ctx context.Context, msgs []types.Message, _ []types.ToolDef, opts ...types.CallOption) (<-chan types.Delta, error) {
	rule := f.match(msgs)
	if rule == nil {
		return nil, fmt.Errorf("faketool: no rule matches the last message (HTTP 422)")
	}
	reply := rule.Reply
	if types.ApplyOptions(opts).ToolChoice == "none" && len(reply.ToolCalls) > 0 {
		reply = FakeReply{Content: "(tool_choice none)"}
	}
	f.mu.Lock()
	f.n++
	n := f.n
	f.mu.Unlock()
	prompt := 0
	for _, m := range msgs {
		prompt += len(strings.Fields(m.Content))
	}
	words := strings.Fields(reply.Content)
	ch := make(chan types.Delta, len(words)+4*len(reply.ToolCalls)+2)
	for i, w := range words {
		if i > 0 {
			w = " " + w
		}
		ch <- types.TextDelta{Text: w}
	}
	for i, c := range reply.ToolCalls {
		args, err := json.Marshal(c.Arguments)
		if err != nil {
			return nil, err
		}
		if c.Arguments == nil {
			args = []byte("{}")
		}
		id := fmt.Sprintf("call_%d_%d", n, i)
		half := len(args) / 2
		ch <- types.ToolCallStartDelta{Index: i, ID: id, Name: c.Name}
		ch <- types.ToolCallArgsDelta{Index: i, Fragment: string(args[:half])}
		ch <- types.ToolCallArgsDelta{Index: i, Fragment: string(args[half:])}
		ch <- types.ToolCallEndDelta{Index: i, Call: types.ToolCall{ID: id, Name: c.Name, Args: args}}
	}
	ch <- types.UsageDelta{In: prompt, Out: len(words) + len(reply.ToolCalls)}
	finish := "stop"
	if len(reply.ToolCalls) > 0 {
		finish = "tool_calls"
	}
	ch <- types.DoneDelta{FinishReason: finish}
	close(ch)
	return ch, nil
}
