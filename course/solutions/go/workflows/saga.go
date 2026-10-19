package workflows

// Sagas (dur.08): a workflow that changes the world in several steps cannot
// roll them back in one transaction, so each forward step registers how to
// undo it (a compensation), and on cancel or failure the workflow runs the
// compensations in reverse order. A compensation is itself an activity: it
// is retried, recorded in history, and idempotent by its key, so a crash in
// the middle of compensating resumes instead of compensating twice.
//
//	var saga workflows.Saga
//	saga.Add("delete partial shards", ActCorpusCleanup, cleanup, opts)  // before the step that may leave them
//	if err := rt.ExecuteActivity(ActCorpusStage, shard, opts, &done); err != nil {
//		return res, saga.Fail(rt, err) // compensates, then returns err
//	}

import (
	"errors"
	"fmt"
)

// Compensation is one registered undo step.
type Compensation struct {
	Name     string // for errors and the chapter's traces: "delete partial shards"
	Activity string // the activity type that undoes the step
	Input    any
	Options  StepOptions
}

// Saga collects compensations and runs them once, newest first.
type Saga struct {
	steps []Compensation
	done  bool
}

// Add registers a compensation. Its Options.ID, when empty, becomes
// "compensate-<n>" (n counts from 1 in registration order), so the
// compensation's idempotency key is stable across replays.
func (s *Saga) Add(name, activity string, in any, opts StepOptions) {
	if opts.ID == "" {
		opts.ID = fmt.Sprintf("compensate-%d", len(s.steps)+1)
	}
	s.steps = append(s.steps, Compensation{Name: name, Activity: activity, Input: in, Options: opts})
}

// Len is the number of registered compensations.
func (s *Saga) Len() int {
	return len(s.steps)
}

// Compensate runs every registered compensation on rt.Detached(), newest
// first, and returns the failures joined (nil when all succeeded). A failed
// compensation does not stop the others: each undoes a different step. The
// second and later calls do nothing and return nil, so a workflow that
// compensates on a failure path and again on a cancel path compensates once.
func (s *Saga) Compensate(rt Runtime) error {
	if s.done {
		return nil
	}
	s.done = true
	d := rt.Detached()
	var errs []error
	for i := len(s.steps) - 1; i >= 0; i-- {
		c := s.steps[i]
		if err := d.ExecuteActivity(c.Activity, c.Input, c.Options, nil); err != nil {
			errs = append(errs, fmt.Errorf("compensation %q: %w", c.Name, err))
		}
	}
	return errors.Join(errs...)
}

// Fail compensates and returns cause, with any compensation failures joined
// to it (cause stays matchable with errors.Is, ErrCanceled included).
func (s *Saga) Fail(rt Runtime, cause error) error {
	if err := s.Compensate(rt); err != nil {
		return errors.Join(cause, err)
	}
	return cause
}
