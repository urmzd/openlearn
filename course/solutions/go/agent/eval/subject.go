package eval

// subject.go: the three things an eval can run on a case. A chat model
// behind any Provider (ag.01), an agent loop (ag.03), and a durable agent
// run (ag.05), each recording the output, its timing, and what scorers
// need to see.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"time"

	"tinyllm/agent/durableagent"
	"tinyllm/agent/loop"
	"tinyllm/agent/types"
)

// ErrInput is returned for a case input that is not a prompt string or a
// {"messages": [...]} chat.
var ErrInput = errors.New("eval: case input must be a string or {\"messages\": [...]}")

// Messages turns a case input into the conversation a subject sends: a
// string is one user message; {"messages": [{role, content}, ...]} is that
// conversation (roles system, user, assistant).
func Messages(input json.RawMessage) ([]types.Message, error) {
	var s string
	if err := json.Unmarshal(input, &s); err == nil {
		return []types.Message{{Role: types.RoleUser, Content: s}}, nil
	}
	var chat struct {
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(input, &chat); err != nil || len(chat.Messages) == 0 {
		return nil, ErrInput
	}
	out := make([]types.Message, 0, len(chat.Messages))
	for _, m := range chat.Messages {
		switch types.Role(m.Role) {
		case types.RoleSystem, types.RoleUser, types.RoleAssistant:
		default:
			return nil, fmt.Errorf("%w: role %q", ErrInput, m.Role)
		}
		out = append(out, types.Message{Role: types.Role(m.Role), Content: m.Content})
	}
	return out, nil
}

// ToolCallRecord is one tool call an agent subject made, as recorded in the
// "tool_calls" annotation (a JSON array, in call order).
type ToolCallRecord struct {
	Name    string          `json:"name"`
	Args    json.RawMessage `json:"args"`
	Verdict string          `json:"verdict"`  // the gate's: allow, deny, needs_approval
	IsError bool            `json:"is_error"` // the result was a tool error
	Result  string          `json:"result,omitempty"`
}

// textOutput is a final answer as the JSON string Output holds.
func textOutput(s string) json.RawMessage {
	b, _ := json.Marshal(s)
	return b
}

func clockOr(now func() time.Time) func() time.Time {
	if now == nil {
		return time.Now
	}
	return now
}

// ProviderSubject sends each case to p as one streamed chat completion.
// TTFT is the first text delta's arrival, TokenTimes every text delta's,
// Total the end of the stream, Tokens the usage's completion tokens. An
// error delta fails the case. now (nil: time.Now) is the clock.
func ProviderSubject(p types.Provider, now func() time.Time, opts ...types.CallOption) Subject {
	now = clockOr(now)
	return func(ctx context.Context, o *Observation) error {
		msgs, err := Messages(o.Input)
		if err != nil {
			return err
		}
		start := now()
		o.Timing = Timing{Start: start}
		ch, err := p.ChatStream(ctx, msgs, nil, opts...)
		if err != nil {
			return err
		}
		var acc types.Accumulator
		for d := range ch {
			acc.Add(d)
			switch v := d.(type) {
			case types.TextDelta:
				at := now().Sub(start)
				if len(o.Timing.TokenTimes) == 0 {
					o.Timing.TTFT = at
				}
				o.Timing.TokenTimes = append(o.Timing.TokenTimes, at)
			case types.UsageDelta:
				o.Tokens = v.Out
			}
		}
		o.Timing.Total = now().Sub(start)
		if err := acc.Err(); err != nil {
			return err
		}
		o.Output = textOutput(acc.Message().Content)
		return nil
	}
}

// AgentSubject runs each case through agent a (ag.03). Besides the timing
// of the streamed text (as ProviderSubject), it records every tool call in
// the "tool_calls" annotation and sums completion tokens over the run's
// model calls. A run that ends in an error fails the case.
func AgentSubject(a *loop.Agent, now func() time.Time) Subject {
	now = clockOr(now)
	return func(ctx context.Context, o *Observation) error {
		msgs, err := Messages(o.Input)
		if err != nil {
			return err
		}
		start := now()
		o.Timing = Timing{Start: start}
		st := a.Invoke(ctx, msgs)
		calls := []ToolCallRecord{}
		byID := map[string]int{}
		for ev := range st.Events() {
			switch ev.Kind {
			case loop.EventText:
				at := now().Sub(start)
				if len(o.Timing.TokenTimes) == 0 {
					o.Timing.TTFT = at
				}
				o.Timing.TokenTimes = append(o.Timing.TokenTimes, at)
			case loop.EventToolCall:
				byID[ev.Call.ID] = len(calls)
				calls = append(calls, ToolCallRecord{Name: ev.Call.Name, Args: ev.Call.Args, Verdict: ev.Verdict.Kind.String()})
			case loop.EventToolResult:
				if i, ok := byID[ev.Result.CallID]; ok {
					calls[i].IsError = ev.Result.IsError
					calls[i].Result = ev.Result.Content
				}
			case loop.EventUsage:
				o.Tokens += ev.Usage.Out
			}
		}
		msg, err := st.Result()
		o.Timing.Total = now().Sub(start)
		b, _ := json.Marshal(calls)
		if o.Annotations == nil {
			o.Annotations = map[string]json.RawMessage{}
		}
		o.Annotations["tool_calls"] = b
		if err != nil {
			return err
		}
		o.Output = textOutput(msg.Content)
		return nil
	}
}

var runIDUnsafe = regexp.MustCompile(`[^A-Za-z0-9_.-]`)

// RunID names the durable run of one case and sample: "<prefix>-<case>-s<n>"
// with every character outside [A-Za-z0-9_.-] replaced by "_"; when that is
// longer than 128 bytes, the case part becomes the first 16 hex digits of
// its SHA-256. The same case and sample always get the same id.
func RunID(prefix, caseID string, sample int) string {
	id := fmt.Sprintf("%s-%s-s%d", prefix, runIDUnsafe.ReplaceAllString(caseID, "_"), sample)
	if len(id) > 128 {
		h := sha256.Sum256([]byte(caseID))
		id = fmt.Sprintf("%s-%s-s%d", prefix, hex.EncodeToString(h[:8]), sample)
	}
	return strings.TrimLeft(id, "_.-")
}

// DurableSubject runs each case as the durable agent run RunID(prefix,
// case, sample) (ag.05): re-running a suite after a crash replays the
// cases that finished instead of calling the model again. A run that ends
// awaiting approval or indeterminate fails the case with its status.
func DurableSubject(cfg durableagent.Config, prefix string, now func() time.Time) Subject {
	now = clockOr(now)
	return func(ctx context.Context, o *Observation) error {
		msgs, err := Messages(o.Input)
		if err != nil {
			return err
		}
		start := now()
		out, err := durableagent.Run(ctx, cfg, RunID(prefix, o.ID, o.Sample), msgs)
		o.Timing = Timing{Start: start, Total: now().Sub(start)}
		if err != nil {
			return err
		}
		if out.Status != durableagent.Completed {
			return fmt.Errorf("durable run %s: %s", RunID(prefix, o.ID, o.Sample), out.Status)
		}
		o.Output = textOutput(out.Message.Content)
		return nil
	}
}
