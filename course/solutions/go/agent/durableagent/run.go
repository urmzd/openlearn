package durableagent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"

	"tinyllm/agent/gate"
	"tinyllm/agent/loop"
	"tinyllm/agent/types"
)

// Config is one durable agent: the loop's configuration, its gate (whose
// IsWrite also tells the journal which steps are writes), and the store.
type Config struct {
	Agent   loop.Config
	Gate    *gate.PolicyGate // nil: every call allowed, no tool is a write
	Store   Store
	Options []loop.Option // more loop options (WithMaxIter, WithBudget, WithTracer, ...)
}

// Status is where a run stands after Run returns.
type Status string

const (
	Completed        Status = "completed"         // Message is the answer
	AwaitingApproval Status = "awaiting_approval" // send an "approve" signal per Pending marker
	Indeterminate    Status = "indeterminate"     // send a "reconcile" signal for Step
)

// Outcome is the result of one Run call.
type Outcome struct {
	Status  Status
	Message types.Message
	Pending []loop.Pending
	Step    string
}

// ErrInputMismatch: the run id exists with different input messages.
var ErrInputMismatch = errors.New("durableagent: run exists with a different input")

// Run starts the run runID with input, or resumes it: completed steps are
// replayed from the journal (no model call, no tool call), approvals and
// reconciliations sent as signals since the last attempt take effect, and a
// completed run returns its recorded answer. Running the same id twice with
// the same input is safe; with a different input it is ErrInputMismatch.
// input may be nil to resume an existing run.
func Run(ctx context.Context, cfg Config, runID string, input []types.Message) (Outcome, error) {
	if !ValidRunID(runID) {
		return Outcome{}, fmt.Errorf("durableagent: invalid run id %q", runID)
	}
	recs, err := cfg.Store.Load(ctx, runID)
	if err != nil {
		return Outcome{}, err
	}
	var recorded []types.Message
	var approved []string
	for _, r := range recs {
		switch r.Type {
		case RunStarted:
			if err := json.Unmarshal(r.Data, &recorded); err != nil {
				return Outcome{}, fmt.Errorf("durableagent: run %s: bad input record: %w", runID, err)
			}
		case RunCompleted:
			var m types.Message
			if err := json.Unmarshal(r.Data, &m); err != nil {
				return Outcome{}, err
			}
			return Outcome{Status: Completed, Message: m}, nil
		case SignalRecord:
			if r.Step == "approve" {
				var a struct{ Marker string }
				if json.Unmarshal(r.Data, &a) == nil && a.Marker != "" {
					approved = append(approved, a.Marker)
				}
			}
		}
	}
	j := NewJournal(cfg.Store, runID, recs, nil)
	if recorded == nil {
		if input == nil {
			return Outcome{}, fmt.Errorf("durableagent: run %s does not exist", runID)
		}
		data, err := json.Marshal(input)
		if err != nil {
			return Outcome{}, err
		}
		if err := j.append(ctx, Record{Type: RunStarted, Data: data}); err != nil {
			return Outcome{}, err
		}
		recorded = input
	} else if input != nil && !sameMessages(input, recorded) {
		return Outcome{}, ErrInputMismatch
	}
	opts := append([]loop.Option{loop.WithStepRunner(j), loop.WithApprovals(approved...)}, cfg.Options...)
	if cfg.Gate != nil {
		j.isWrite = cfg.Gate.IsWrite
		opts = append(opts, loop.WithGate(cfg.Gate))
	}
	msg, err := loop.New(cfg.Agent, opts...).Invoke(ctx, recorded).Result()
	var ae *loop.ApprovalError
	var ie *IndeterminateError
	switch {
	case errors.As(err, &ae):
		return Outcome{Status: AwaitingApproval, Message: msg, Pending: ae.Pending}, nil
	case errors.As(err, &ie):
		return Outcome{Status: Indeterminate, Message: msg, Step: ie.Step}, nil
	case err != nil:
		return Outcome{}, err // not recorded: the next Run retries from the journal
	}
	data, err := json.Marshal(msg)
	if err != nil {
		return Outcome{}, err
	}
	if err := j.append(ctx, Record{Type: RunCompleted, Data: data}); err != nil {
		return Outcome{}, err
	}
	return Outcome{Status: Completed, Message: msg}, nil
}

// Signal records a signal for runID: "approve" with {"Marker": m}, or
// "reconcile" with a Reconcile. It takes effect at the next Run.
func Signal(ctx context.Context, store Store, runID, name string, payload any) error {
	if name != "approve" && name != "reconcile" {
		return fmt.Errorf("durableagent: unknown signal %q", name)
	}
	recs, err := store.Load(ctx, runID)
	if err != nil {
		return err
	}
	if len(recs) == 0 {
		return fmt.Errorf("durableagent: run %s does not exist", runID)
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	return NewJournal(store, runID, recs, nil).append(ctx, Record{Type: SignalRecord, Step: name, Data: data})
}

func sameMessages(a, b []types.Message) bool {
	x, _ := json.Marshal(a)
	y, _ := json.Marshal(b)
	return reflect.DeepEqual(x, y)
}
