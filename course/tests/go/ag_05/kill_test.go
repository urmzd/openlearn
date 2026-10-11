package ag_05

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"openlearn.urmzd.com/tl/testkit/faketool"
	"tinyllm/agent/durableagent"
	"tinyllm/agent/loop"
	"tinyllm/agent/provider"
	"tinyllm/agent/tool"
)

// childEnv tells the test binary, re-executed as a child process, to act as
// the agent worker: it runs (or resumes) run "a1" once and prints the
// outcome as JSON, unless it kills itself first.
const childEnv = "AG05_CHILD_DIR"

// effect appends name to the effects file and fsyncs, so the parent can
// count every tool execution, including those of killed children.
func effect(path, name string) {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		panic(err)
	}
	f.WriteString(name + "\n")
	f.Sync()
	f.Close()
}

func TestHelperAgentWorker(t *testing.T) {
	// WHY: not a check on its own: the child process of
	//      TestKillResumesWithoutRecall. Each tool records its effect, then
	//      SIGKILLs the process when AG05_CRASH names it (after the effect,
	//      before the journal records the result: the worst moment).
	// KIND: fault
	// CHAPTER: ag.05 section 4, What the tests check
	dir := os.Getenv(childEnv)
	if dir == "" {
		t.Skip("helper process for TestKillResumesWithoutRecall")
	}
	crash, effects := os.Getenv("AG05_CRASH"), filepath.Join(dir, "effects.txt")
	mk := func(name, out string) tool.Tool {
		return tool.New(name, "", `{"type":"object"}`, func(context.Context, json.RawMessage) (string, error) {
			effect(effects, name)
			if crash == name {
				syscall.Kill(os.Getpid(), syscall.SIGKILL)
				select {} // the signal is on its way
			}
			return out, nil
		})
	}
	reg := tool.NewRegistry().MustRegister(mk("lookup", "policy: tickets need approval"), mk("create_ticket", "TICKET-7"))
	store, err := durableagent.OpenFileStore(filepath.Join(dir, "runs"))
	if err != nil {
		t.Fatal(err)
	}
	prov := provider.NewOpenAI(provider.Config{BaseURL: os.Getenv("AG05_URL"), Model: "faketool"})
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	out, err := durableagent.Run(ctx, durableagent.Config{Agent: loop.Config{Provider: prov, Tools: reg}, Gate: policyGate(), Store: store}, "a1", input)
	if err != nil {
		t.Fatal(err)
	}
	var markers []string
	for _, p := range out.Pending {
		markers = append(markers, p.Marker)
	}
	b, _ := json.Marshal(map[string]any{"status": out.Status, "step": out.Step, "markers": markers, "answer": out.Message.Content})
	fmt.Println("OUTCOME " + string(b))
}

type childResult struct {
	Killed  bool
	Status  string   `json:"status"`
	Step    string   `json:"step"`
	Markers []string `json:"markers"`
	Answer  string   `json:"answer"`
}

func runChild(t *testing.T, dir, url, crash string) childResult {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^TestHelperAgentWorker$", "-test.count=1")
	cmd.Env = append(os.Environ(), childEnv+"="+dir, "AG05_URL="+url, "AG05_CRASH="+crash)
	var out bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &out
	err := cmd.Run()
	var ee *exec.ExitError
	if errors.As(err, &ee) {
		if ws, ok := ee.Sys().(syscall.WaitStatus); ok && ws.Signaled() && ws.Signal() == syscall.SIGKILL {
			return childResult{Killed: true}
		}
	}
	if err != nil {
		t.Fatalf("child (crash=%q) failed: %v\n%s", crash, err, out.String())
	}
	for _, line := range strings.Split(out.String(), "\n") {
		if rest, ok := strings.CutPrefix(line, "OUTCOME "); ok {
			var r childResult
			if err := json.Unmarshal([]byte(rest), &r); err != nil {
				t.Fatal(err)
			}
			return r
		}
	}
	t.Fatalf("child printed no outcome:\n%s", out.String())
	return childResult{}
}

func effectsOf(t *testing.T, dir string) map[string]int {
	b, _ := os.ReadFile(filepath.Join(dir, "effects.txt"))
	n := map[string]int{}
	for _, l := range strings.Fields(string(b)) {
		n[l]++
	}
	return n
}

func TestKillResumesWithoutRecall(t *testing.T) {
	// WHY: the module's promise under a real SIGKILL (no deferred code, no
	//      flush). Kill 1 lands inside the read (lookup): the next worker
	//      re-runs the read but not the model call before it. Kill 2 lands
	//      inside the approved write, after the ticket was filed: the next
	//      worker reports indeterminate instead of filing again; a reconcile
	//      signal settles it and the run completes. Over five workers the
	//      model is asked exactly three times (llm/1, llm/2, llm/3, by the
	//      fake's request log) and exactly one ticket exists.
	// KIND: fault
	// CATCHES: s01, s02, s14
	// CHAPTER: ag.05 section 2.4
	srv, err := faketool.Start(&faketool.Rules{Rules: []faketool.Rule{
		{When: faketool.When{Role: "user"}, Reply: faketool.Reply{ToolCalls: []faketool.ToolCall{{Name: "lookup", Arguments: map[string]any{"q": "access policy"}}}}},
		{When: faketool.When{Role: "tool", Tool: "lookup"}, Reply: faketool.Reply{ToolCalls: []faketool.ToolCall{{Name: "create_ticket", Arguments: map[string]any{"title": "grant access"}}}}},
		{When: faketool.When{Role: "tool", Tool: "create_ticket"}, Reply: faketool.Reply{Content: "Ticket filed."}},
	}}, "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()
	dir := t.TempDir()
	step := func(crash string) childResult { return runChild(t, dir, srv.BaseURL(), crash) }

	if r := step("lookup"); !r.Killed {
		t.Fatalf("worker 1 should have died inside lookup: %+v", r)
	}
	r := step("")
	if r.Status != "awaiting_approval" || len(r.Markers) != 1 || len(srv.Requests()) != 2 || effectsOf(t, dir)["lookup"] != 2 {
		t.Fatalf("worker 2: %+v, %d model calls, effects %v; want awaiting approval after 2 calls, lookup run twice", r, len(srv.Requests()), effectsOf(t, dir))
	}
	store, _ := durableagent.OpenFileStore(filepath.Join(dir, "runs"))
	ctx := context.Background()
	if err := durableagent.Signal(ctx, store, "a1", "approve", map[string]string{"Marker": r.Markers[0]}); err != nil {
		t.Fatal(err)
	}
	if r := step("create_ticket"); !r.Killed {
		t.Fatalf("worker 3 should have died inside create_ticket: %+v", r)
	}
	r = step("")
	if r.Status != "indeterminate" || r.Step != "tool/2/0/create_ticket" || effectsOf(t, dir)["create_ticket"] != 1 {
		t.Fatalf("worker 4: %+v, effects %v; want indeterminate at tool/2/0/create_ticket with one ticket", r, effectsOf(t, dir))
	}
	durableagent.Signal(ctx, store, "a1", "reconcile", durableagent.Reconcile{Step: r.Step, Outcome: "done", Result: "TICKET-7"})
	r = step("")
	if r.Status != "completed" || r.Answer != "Ticket filed." {
		t.Fatalf("worker 5: %+v", r)
	}
	if n, eff := len(srv.Requests()), effectsOf(t, dir); n != 3 || eff["create_ticket"] != 1 || eff["lookup"] != 2 {
		t.Fatalf("%d model calls, effects %v; want 3 calls, one ticket, two lookups", n, eff)
	}
	last := srv.Requests()[2]["messages"].([]any)
	if got := last[len(last)-1].(map[string]any)["content"]; got != "TICKET-7" {
		t.Fatalf("the model saw %v as the ticket result, want the reconciled TICKET-7", got)
	}
}
