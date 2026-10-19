package scorers

// state.go: the set-relation grader of case study 05, ported to Go. An
// agent that changes a system (a helpdesk, an inventory) is graded on the
// change, not on its words: with delta = diff(before, after), required
// changes R and allowed changes A, it passes when R ⊆ delta ⊆ A. One half
// catches missing work, the other unrequested work. Absent and null stay
// different values, and a read counts only for the rows it returned.

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strings"

	"tinyllm/agent/eval"
)

// State is a system's rows: "<entity>/<id>" -> field -> JSON value. A
// field that is not in the map is absent, which is not the same as null.
type State map[string]map[string]json.RawMessage

// Absent is the delta value of a field that existed before and is gone
// after, and the required value that asks for that: {"$absent": true}.
const Absent = `{"$absent":true}`

// Row-level delta values.
const (
	Created = `"created"`
	Deleted = `"deleted"`
)

// Task is what a case requires (key -> value it must end with; a key is
// "<entity>/<id>.<field>", or "<entity>/<id>" with Created or Deleted) and
// allows (keys that may change: exact, or "<entity>/<id>.*" for every field
// of a row). ReadBeforeWrite asks that every written row was returned by
// an earlier read.
type Task struct {
	Required        map[string]json.RawMessage `json:"required"`
	Allowed         []string                   `json:"allowed"`
	ReadBeforeWrite bool                       `json:"read_before_write"`
}

// StateConfig names the tools that read and write rows, and the fields that
// are bookkeeping, not task-visible changes.
type StateConfig struct {
	ReadTools  []string // their results are JSON; every "id" in them was returned
	WriteTools []string // their arguments carry the written row's "id"
	Ignore     []string // fields never in the delta; nil means ["updated_at"]
}

// Criteria names, in report order.
const (
	RequiredChanges      = "required_changes"
	OnlyRequestedChanges = "only_requested_changes"
	ReadBeforeWrite      = "read_before_write"
)

// Grade is the checker's verdict: each criterion's evidence (empty means
// it passed) and whether all passed.
type Grade struct {
	Criteria map[string][]string
	Passed   bool
}

// sameJSON compares two JSON values by meaning (5 and 5.0 are equal),
// keeping null and absent apart: absent is the empty RawMessage.
func sameJSON(a, b json.RawMessage) bool {
	if len(a) == 0 || len(b) == 0 {
		return len(a) == 0 && len(b) == 0
	}
	var x, y any
	if json.Unmarshal(a, &x) != nil || json.Unmarshal(b, &y) != nil {
		return string(a) == string(b)
	}
	return reflect.DeepEqual(x, y)
}

// Diff is delta: every row created (Created) or deleted (Deleted), and
// every field of a kept row whose value differs ("<row>.<field>" -> the new
// value, or Absent when the field is gone), ignored fields left out.
func Diff(before, after State, ignore []string) map[string]json.RawMessage {
	if ignore == nil {
		ignore = []string{"updated_at"}
	}
	skip := map[string]bool{}
	for _, f := range ignore {
		skip[f] = true
	}
	delta := map[string]json.RawMessage{}
	for row := range before {
		if _, ok := after[row]; !ok {
			delta[row] = json.RawMessage(Deleted)
		}
	}
	for row, a := range after {
		b, ok := before[row]
		if !ok {
			delta[row] = json.RawMessage(Created)
			continue
		}
		fields := map[string]bool{}
		for f := range a {
			fields[f] = true
		}
		for f := range b {
			fields[f] = true
		}
		for f := range fields {
			if skip[f] {
				continue
			}
			old, oldOK := b[f]
			nw, newOK := a[f]
			switch {
			case oldOK && !newOK:
				delta[row+"."+f] = json.RawMessage(Absent)
			case !sameJSON(old, nw):
				delta[row+"."+f] = nw
			}
		}
	}
	return delta
}

// allowed reports whether delta key k is in the allowed list.
func allowed(k string, list []string) bool {
	for _, a := range list {
		if a == k {
			return true
		}
		if strings.HasSuffix(a, ".*") && strings.HasPrefix(k, strings.TrimSuffix(a, "*")) {
			return true
		}
	}
	return false
}

// collectIDs adds every string value of an "id" key, at any depth of v.
func collectIDs(v any, out map[string]bool) {
	switch x := v.(type) {
	case map[string]any:
		for k, y := range x {
			if s, ok := y.(string); ok && k == "id" {
				out[s] = true
			}
			collectIDs(y, out)
		}
	case []any:
		for _, y := range x {
			collectIDs(y, out)
		}
	}
}

// unreadWrites lists the writes whose row no earlier successful read
// returned. A read is what came back, not what was asked for.
func unreadWrites(calls []eval.ToolCallRecord, cfg StateConfig) []string {
	isRead, isWrite := map[string]bool{}, map[string]bool{}
	for _, n := range cfg.ReadTools {
		isRead[n] = true
	}
	for _, n := range cfg.WriteTools {
		isWrite[n] = true
	}
	seen := map[string]bool{}
	var bad []string
	for i, c := range calls {
		if c.Verdict != "allow" || c.IsError {
			continue
		}
		if isRead[c.Name] {
			var v any
			if json.Unmarshal([]byte(c.Result), &v) == nil {
				collectIDs(v, seen)
			}
		}
		if isWrite[c.Name] {
			var args struct {
				ID string `json:"id"`
			}
			json.Unmarshal(c.Args, &args)
			if !seen[args.ID] {
				bad = append(bad, fmt.Sprintf("call %d wrote %s unread", i, args.ID))
			}
		}
	}
	return bad
}

// GradeState checks one attempt: required_changes (every required key is in
// delta with its value), only_requested_changes (every delta key is
// allowed), and, when the task asks, read_before_write. It contains no
// model and no randomness: the same record always gets the same grade.
func GradeState(before, after State, t Task, calls []eval.ToolCallRecord, cfg StateConfig) Grade {
	delta := Diff(before, after, cfg.Ignore)
	g := Grade{Criteria: map[string][]string{RequiredChanges: {}, OnlyRequestedChanges: {}}}
	for k, want := range t.Required {
		if got, ok := delta[k]; !ok || !sameJSON(got, want) {
			g.Criteria[RequiredChanges] = append(g.Criteria[RequiredChanges], "missing "+k)
		}
	}
	for k := range delta {
		if !allowed(k, t.Allowed) {
			g.Criteria[OnlyRequestedChanges] = append(g.Criteria[OnlyRequestedChanges], "unauthorized "+k)
		}
	}
	if t.ReadBeforeWrite {
		g.Criteria[ReadBeforeWrite] = append([]string{}, unreadWrites(calls, cfg)...)
	}
	g.Passed = true
	for _, ev := range g.Criteria {
		sort.Strings(ev)
		if len(ev) > 0 {
			g.Passed = false
		}
	}
	return g
}

// StateScorer grades observations whose annotations carry "state_before"
// and "state_after" (State JSON) and whose ground truth is a Task; it
// scores 1 when every criterion passes, else 0, with the failed criteria
// and their evidence as the reason ("state").
func StateScorer(cfg StateConfig) eval.Scorer {
	return fn{"state", func(_ context.Context, o eval.Observation) (eval.Score, error) {
		var before, after State
		if err := json.Unmarshal(o.Annotations["state_before"], &before); err != nil {
			return eval.Score{}, fmt.Errorf("scorers: state_before: %w", err)
		}
		if err := json.Unmarshal(o.Annotations["state_after"], &after); err != nil {
			return eval.Score{}, fmt.Errorf("scorers: state_after: %w", err)
		}
		var t Task
		if err := json.Unmarshal(o.GroundTruth, &t); err != nil || (len(t.Required) == 0 && len(t.Allowed) == 0) {
			return eval.Score{}, ErrNoGroundTruth
		}
		calls, err := ToolCalls(o)
		if err != nil {
			return eval.Score{}, err
		}
		g := GradeState(before, after, t, calls, cfg)
		if g.Passed {
			return eval.Score{Value: 1}, nil
		}
		var names []string
		for n, ev := range g.Criteria {
			if len(ev) > 0 {
				names = append(names, n+": "+strings.Join(ev, ", "))
			}
		}
		sort.Strings(names)
		return eval.Score{Value: 0, Reason: strings.Join(names, "; ")}, nil
	}}
}
