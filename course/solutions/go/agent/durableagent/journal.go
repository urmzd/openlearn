// Package durableagent runs the agent loop durably (ag.05): every model call
// and tool call is a step recorded in an append-only journal before its
// result is used, so a run killed at any point resumes without calling the
// model or a tool again for any step that completed. A write whose start was
// recorded but whose result was not is never guessed at: the run stops with
// ErrIndeterminate until a human reconciles it with a signal.
//
// The journal is the event-sourcing idea of the durable engine (dur.01,
// dur.06) applied to one agent run: the run's state is a fold over its
// records, and replay is "return the recorded result".
//
// Chapter: content/courses/ai-platform-engineering/13-agent-sdk/05-durable-agent-runs.md.
package durableagent

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sync"

	"tinyllm/agent/tool"
	"tinyllm/agent/types"
)

// Record types.
const (
	RunStarted    = "run.started"    // Data: the input messages
	StepStarted   = "step.started"   // Step, Kind: about to run
	StepCompleted = "step.completed" // Step, Data: the step's result
	SignalRecord  = "signal"         // Step: the signal name; Data: its payload
	RunCompleted  = "run.completed"  // Data: the final assistant message
)

// Record is one journal entry. Seq numbers a run's records from 1.
type Record struct {
	Seq  int64           `json:"seq"`
	Type string          `json:"type"`
	Step string          `json:"step,omitempty"`
	Kind types.StepKind  `json:"kind,omitempty"`
	Data json.RawMessage `json:"data,omitempty"`
}

// Store is an append-only journal per run. Append returns only after the
// record is durable (a crash after it returns never loses the record).
type Store interface {
	Load(ctx context.Context, runID string) ([]Record, error)
	Append(ctx context.Context, runID string, r Record) error
}

// runIDRE keeps a run id a plain file name: no separators, no "..".
var runIDRE = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$`)

// ValidRunID reports whether id may name a run.
func ValidRunID(id string) bool {
	return runIDRE.MatchString(id) && id != "." && id != ".."
}

// FileStore keeps one JSON-lines file per run, <dir>/<run id>.jsonl, and
// fsyncs every append. A record torn by a crash (the last line without its
// newline) is ignored by Load and cut off by the next Append.
type FileStore struct {
	dir string
	mu  sync.Mutex
}

// OpenFileStore creates dir when needed.
func OpenFileStore(dir string) (*FileStore, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	return &FileStore{dir: dir}, nil
}

func (s *FileStore) path(runID string) (string, error) {
	if !ValidRunID(runID) {
		return "", fmt.Errorf("durableagent: invalid run id %q", runID)
	}
	return filepath.Join(s.dir, runID+".jsonl"), nil
}

// Load returns the run's complete records in order (none for a new run).
func (s *FileStore) Load(_ context.Context, runID string) ([]Record, error) {
	p, err := s.path(runID)
	if err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	b, err := os.ReadFile(p)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var out []Record
	br := bufio.NewReader(bytes.NewReader(b))
	for {
		line, err := br.ReadBytes('\n')
		if err == io.EOF {
			break // a final line without its newline is a torn write: ignore it
		}
		var r Record
		if jerr := json.Unmarshal(line, &r); jerr != nil {
			return nil, fmt.Errorf("durableagent: %s: corrupt record %d: %w", p, len(out)+1, jerr)
		}
		out = append(out, r)
	}
	return out, nil
}

// Append writes r as one line and fsyncs before returning.
func (s *FileStore) Append(_ context.Context, runID string, r Record) error {
	p, err := s.path(runID)
	if err != nil {
		return err
	}
	line, err := json.Marshal(r)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	f, err := os.OpenFile(p, os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	b, err := io.ReadAll(f)
	if err != nil {
		return err
	}
	end := int64(bytes.LastIndexByte(b, '\n') + 1) // cut a torn tail
	if err := f.Truncate(end); err != nil {
		return err
	}
	if _, err := f.WriteAt(append(line, '\n'), end); err != nil {
		return err
	}
	return f.Sync()
}

// ErrIndeterminate: a write step was started but its result was never
// recorded, so nobody knows whether the write happened. Replaying it could
// do it twice; skipping it could lose it. The run waits for a reconcile
// signal.
var ErrIndeterminate = errors.New("durableagent: a write started but its outcome was never recorded")

// IndeterminateError names the step.
type IndeterminateError struct{ Step string }

func (e *IndeterminateError) Error() string { return ErrIndeterminate.Error() + ": " + e.Step }
func (e *IndeterminateError) Unwrap() error { return ErrIndeterminate }

// Reconcile is the payload of a "reconcile" signal: a human checked whether
// the write in Step happened. Outcome "done" records Result as the step's
// result (the write is not run again); "retry" runs it again.
type Reconcile struct {
	Step    string `json:"step"`
	Outcome string `json:"outcome"`
	Result  string `json:"result,omitempty"`
}

// Journal is the StepRunner of one run.
type Journal struct {
	store   Store
	runID   string
	isWrite func(tool string) bool
	mu      sync.Mutex
	seq     int64
	done    map[string]json.RawMessage
	started map[string]bool
	recon   map[string]Reconcile
}

// NewJournal folds records (from Store.Load) into the journal's state.
// isWrite says which tools change state (nil: none do).
func NewJournal(store Store, runID string, records []Record, isWrite func(string) bool) *Journal {
	if isWrite == nil {
		isWrite = func(string) bool { return false }
	}
	j := &Journal{store: store, runID: runID, isWrite: isWrite,
		done: map[string]json.RawMessage{}, started: map[string]bool{}, recon: map[string]Reconcile{}}
	for _, r := range records {
		j.seq = max(j.seq, r.Seq)
		switch r.Type {
		case StepStarted:
			j.started[r.Step] = true
		case StepCompleted:
			j.done[r.Step] = r.Data
		case SignalRecord:
			if r.Step == "reconcile" {
				var rc Reconcile
				if json.Unmarshal(r.Data, &rc) == nil {
					j.recon[rc.Step] = rc
				}
			}
		}
	}
	return j
}

func (j *Journal) append(ctx context.Context, r Record) error {
	j.mu.Lock()
	j.seq++
	r.Seq = j.seq
	j.mu.Unlock()
	return j.store.Append(ctx, j.runID, r)
}

// RunStep returns the recorded result of a completed step without calling
// fn. Otherwise it records the start, calls fn, and records the result
// before returning it. A model call or a read that started but never
// completed simply runs again; a write in that state is ErrIndeterminate
// unless a reconcile signal settled it. An fn error is returned and not
// recorded, so the next attempt runs the step again.
func (j *Journal) RunStep(ctx context.Context, name string, kind types.StepKind, fn func(context.Context) ([]byte, error)) ([]byte, error) {
	j.mu.Lock()
	out, done := j.done[name]
	started := j.started[name]
	rc, reconciled := j.recon[name]
	j.mu.Unlock()
	if done {
		return out, nil
	}
	toolName, isTool := types.StepToolName(name)
	if started && kind == types.StepTool && isTool && j.isWrite(toolName) {
		switch {
		case reconciled && rc.Outcome == "done":
			res, _ := json.Marshal(tool.Result{Content: rc.Result}) // the loop sets the call id
			if err := j.append(ctx, Record{Type: StepCompleted, Step: name, Kind: kind, Data: res}); err != nil {
				return nil, err
			}
			j.mu.Lock()
			j.done[name] = res
			j.mu.Unlock()
			return res, nil
		case reconciled && rc.Outcome == "retry":
			// a human checked: the write did not happen; run it again
		default:
			return nil, &IndeterminateError{Step: name}
		}
	}
	if err := j.append(ctx, Record{Type: StepStarted, Step: name, Kind: kind}); err != nil {
		return nil, err
	}
	out, err := fn(ctx)
	if err != nil {
		return nil, err
	}
	if err := j.append(ctx, Record{Type: StepCompleted, Step: name, Kind: kind, Data: out}); err != nil {
		return nil, err
	}
	j.mu.Lock()
	j.done[name] = out
	j.mu.Unlock()
	return out, nil
}
