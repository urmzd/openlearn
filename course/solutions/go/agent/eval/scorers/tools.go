package scorers

// tools.go: scoring tool use from the "tool_calls" annotation the agent
// subject records (eval.ToolCallRecord).

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"tinyllm/agent/eval"
)

// ErrNoToolCalls: the run dispatched no tool calls.
var ErrNoToolCalls = errors.New("scorers: no tool calls were dispatched")

// ToolCalls decodes the "tool_calls" annotation (none: an empty list).
func ToolCalls(o eval.Observation) ([]eval.ToolCallRecord, error) {
	raw, ok := o.Annotations["tool_calls"]
	if !ok {
		return nil, nil
	}
	var calls []eval.ToolCallRecord
	if err := json.Unmarshal(raw, &calls); err != nil {
		return nil, fmt.Errorf("scorers: tool_calls annotation: %w", err)
	}
	return calls, nil
}

// ToolSuccess is the fraction of dispatched tool calls (gate verdict
// allow) that did not end in a tool error ("tool_success"). Calls the gate
// denied or held for approval were never run and do not count; a run that
// dispatched none is an error, not a perfect score.
func ToolSuccess() eval.Scorer {
	return fn{"tool_success", func(_ context.Context, o eval.Observation) (eval.Score, error) {
		calls, err := ToolCalls(o)
		if err != nil {
			return eval.Score{}, err
		}
		ran, ok := 0, 0
		for _, c := range calls {
			if c.Verdict != "allow" {
				continue
			}
			ran++
			if !c.IsError {
				ok++
			}
		}
		if ran == 0 {
			return eval.Score{}, ErrNoToolCalls
		}
		return eval.Score{Value: float64(ok) / float64(ran)}, nil
	}}
}

// ToolCalled scores 1 when the run dispatched the expected tool and the
// call succeeded, else 0 ("tool_called"). The expected name is the case's
// scorer_args.tool_called.name, or its ground truth's "tool".
func ToolCalled() eval.Scorer {
	return fn{"tool_called", func(_ context.Context, o eval.Observation) (eval.Score, error) {
		var args struct {
			ToolCalled struct {
				Name string `json:"name"`
			} `json:"tool_called"`
		}
		json.Unmarshal(o.Annotations["scorer_args"], &args)
		want := args.ToolCalled.Name
		if want == "" {
			var gt struct {
				Tool string `json:"tool"`
			}
			json.Unmarshal(o.GroundTruth, &gt)
			want = gt.Tool
		}
		if want == "" {
			return eval.Score{}, ErrNoGroundTruth
		}
		calls, err := ToolCalls(o)
		if err != nil {
			return eval.Score{}, err
		}
		for _, c := range calls {
			if c.Name == want && c.Verdict == "allow" && !c.IsError {
				return eval.Score{Value: 1}, nil
			}
		}
		return eval.Score{Value: 0, Reason: "no successful call to " + want}, nil
	}}
}
