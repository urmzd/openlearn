package tool

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"runtime/debug"
	"sort"
	"strings"
	"sync"

	"tinyllm/agent/types"
)

// NamePattern is the contract's tool name rule (openai-subset.v1, Tool).
var NamePattern = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)

// Registry holds the tools an agent may call. It is safe for concurrent
// use: the loop (ag.03) calls tools in parallel.
type Registry struct {
	mu      sync.RWMutex
	tools   map[string]Tool
	schemas map[string]*Schema
}

func NewRegistry() *Registry {
	return &Registry{tools: map[string]Tool{}, schemas: map[string]*Schema{}}
}

// Register adds t. It fails for a name outside NamePattern, a name already
// registered, or parameters that do not compile or whose top-level type is
// not "object" (the model always sends an object).
func (r *Registry) Register(t Tool) error {
	d := t.Definition()
	if !NamePattern.MatchString(d.Name) {
		return fmt.Errorf("tool: name %q does not match %s", d.Name, NamePattern)
	}
	params := d.Parameters
	if len(params) == 0 {
		params = json.RawMessage(`{"type":"object"}`)
	}
	s, err := Compile(params)
	if err != nil {
		return fmt.Errorf("tool %s: %w", d.Name, err)
	}
	if len(s.types) != 1 || s.types[0] != "object" {
		return fmt.Errorf("tool %s: parameters must have type \"object\"", d.Name)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, dup := r.tools[d.Name]; dup {
		return fmt.Errorf("tool: %q is already registered", d.Name)
	}
	r.tools[d.Name] = t
	r.schemas[d.Name] = s
	return nil
}

// MustRegister is Register that panics: for tools wired at startup.
func (r *Registry) MustRegister(ts ...Tool) *Registry {
	for _, t := range ts {
		if err := r.Register(t); err != nil {
			panic(err)
		}
	}
	return r
}

// Get returns the tool named name.
func (r *Registry) Get(name string) (Tool, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	t, ok := r.tools[name]
	return t, ok
}

// Names lists the registered names, sorted.
func (r *Registry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]string, 0, len(r.tools))
	for n := range r.tools {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

// Definitions are the definitions sent to the model, sorted by name so the
// prompt is the same bytes on every run (and a prompt cache can hit).
func (r *Registry) Definitions() []types.ToolDef {
	names := r.Names()
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]types.ToolDef, 0, len(names))
	for _, n := range names {
		out = append(out, r.tools[n].Definition())
	}
	return out
}

// Call runs one tool call and always returns a Result:
//
//	unknown tool            error result listing the available names
//	arguments not JSON      error result, the tool is not run
//	schema violations       error result with every violation, not run
//	Execute returns err     error result with err's text
//	Execute panics          error result "tool panicked: ..."; the stack
//	                        stays in the process, never in the model's context
//
// Empty arguments mean {} (some providers send "" for a call without
// parameters).
func (r *Registry) Call(ctx context.Context, c types.ToolCall) (res Result) {
	res = Result{CallID: c.ID, Name: c.Name}
	fail := func(msg string) Result {
		res.Content, res.IsError = msg, true
		return res
	}
	t, ok := r.Get(c.Name)
	if !ok {
		return fail(fmt.Sprintf("unknown tool %q; available: %s", c.Name, strings.Join(r.Names(), ", ")))
	}
	r.mu.RLock()
	s := r.schemas[c.Name]
	r.mu.RUnlock()
	args := c.Args
	if len(strings.TrimSpace(string(args))) == 0 {
		args = json.RawMessage("{}")
	}
	if errs := s.ValidateJSON(args); len(errs) > 0 {
		lines := make([]string, len(errs))
		for i, e := range errs {
			lines[i] = e.Error()
		}
		return fail(fmt.Sprintf("invalid arguments for %s:\n%s", c.Name, strings.Join(lines, "\n")))
	}
	defer func() {
		if p := recover(); p != nil {
			_ = debug.Stack() // a caller that logs would log this, never return it
			res = Result{CallID: c.ID, Name: c.Name, Content: fmt.Sprintf("tool %s panicked: %v", c.Name, p), IsError: true}
		}
	}()
	out, err := t.Execute(ctx, args)
	if err != nil {
		return fail(err.Error())
	}
	res.Content = out
	return res
}
