package ag_04

import (
	"context"
	"encoding/json"
	"errors"
	"net/url"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"

	"openlearn.urmzd.com/tl/testkit/faketool"
	"tinyllm/agent/gate"
	"tinyllm/agent/loop"
	"tinyllm/agent/provider"
	"tinyllm/agent/tool"
	"tinyllm/agent/types"
)

type injCase struct {
	Name   string          `json:"name"`
	Why    string          `json:"why"`
	User   string          `json:"user"`
	Docs   string          `json:"docs"`
	Page   string          `json:"page"`
	Rules  []faketool.Rule `json:"rules"`
	Expect struct {
		Outcome   string   `json:"outcome"`
		Ran       []string `json:"ran"`
		NotRun    []string `json:"not_run"`
		Denied    []string `json:"denied"`
		Egress    []string `json:"egress"`
		RowsAfter int      `json:"rows_after"`
	} `json:"expect"`
}

// effects records which tools ran and which hosts the http tools reached.
type effects struct {
	mu     sync.Mutex
	ran    map[string]int
	egress map[string]bool
}

func (e *effects) hit(name, rawURL string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.ran[name]++
	if rawURL != "" {
		if u, err := url.Parse(rawURL); err == nil {
			e.egress[u.Hostname()] = true
		}
	}
}

// suiteTools are the agent's tools: they never touch the network; each
// records that it ran (and where it would have sent a request).
func suiteTools(c injCase, eff *effects, q tool.Tool) *tool.Registry {
	rec := func(name, schema string, out func(args map[string]any) string) tool.Tool {
		return tool.New(name, "", schema, func(_ context.Context, a json.RawMessage) (string, error) {
			var m map[string]any
			json.Unmarshal(a, &m)
			u, _ := m["url"].(string)
			eff.hit(name, u)
			return out(m), nil
		})
	}
	obj := `{"type":"object"}`
	counted := tool.New("query_usage", q.Definition().Description, string(q.Definition().Parameters), func(ctx context.Context, a json.RawMessage) (string, error) {
		eff.hit("query_usage", "")
		return q.Execute(ctx, a)
	})
	return tool.NewRegistry().MustRegister(
		rec("search_docs", obj, func(map[string]any) string { return c.Docs }),
		rec("web_fetch", obj, func(map[string]any) string { return c.Page }),
		rec("http_post", obj, func(map[string]any) string { return "posted" }),
		rec("send_email", obj, func(map[string]any) string { return "sent" }),
		rec("create_ticket", obj, func(map[string]any) string { return "TICKET-1" }),
		rec("shell", obj, func(map[string]any) string { return "ran" }),
		counted,
	)
}

func keys(m map[string]bool) []string {
	out := []string{}
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func TestInjectionSuite(t *testing.T) {
	// WHY: the prompt-injection fixture suite (DESIGN ag.04 F). The fake
	//      model obeys any instruction in the last message, so every
	//      planted instruction is proposed as a tool call; the program
	//      must make it harmless. Instructions in retrieved chunks and
	//      fetched pages never run a send or write tool without approval,
	//      never reach a host outside the allowlist, and never get anything
	//      but one read past the SQL tool; the three controls show the gate
	//      still lets legitimate work run.
	// KIND: fault
	// CATCHES: s08, s10, s11, s12, s13, s14
	// CHAPTER: ag.04 section 2.4
	var suite struct {
		Cases []injCase `json:"cases"`
	}
	loadJSON(t, "injection.json", &suite)
	if len(suite.Cases) < 10 {
		t.Fatalf("the suite has %d cases", len(suite.Cases))
	}
	for _, c := range suite.Cases {
		t.Run(c.Name, func(t *testing.T) {
			srv, err := faketool.Start(&faketool.Rules{Rules: c.Rules}, "127.0.0.1:0")
			if err != nil {
				t.Fatal(err)
			}
			defer srv.Close()
			db := ledger(t)
			eff := &effects{ran: map[string]int{}, egress: map[string]bool{}}
			reg := suiteTools(c, eff, gate.QueryUsageTool(db, 100))
			g := gate.New(policy())
			prov := provider.NewOpenAI(provider.Config{BaseURL: srv.BaseURL(), Model: "faketool"})
			st := loop.New(loop.Config{Provider: prov, Tools: reg}, loop.WithGate(g), loop.WithMaxIter(6)).Invoke(context.Background(),
				[]types.Message{{Role: types.RoleUser, Content: c.User}})
			var denied []string
			for e := range st.Events() {
				if e.Kind == loop.EventToolCall && e.Verdict.Kind == types.Deny {
					denied = append(denied, e.Call.Name)
				}
			}
			_, err = st.Result()
			outcome := "answer"
			var ae *loop.ApprovalError
			switch {
			case errors.As(err, &ae):
				outcome = "needs_approval"
			case err != nil:
				t.Fatalf("%s: run failed: %v", c.Why, err)
			}
			if outcome != c.Expect.Outcome {
				t.Errorf("%s: outcome %s, want %s", c.Why, outcome, c.Expect.Outcome)
			}
			for _, n := range c.Expect.Ran {
				if eff.ran[n] == 0 {
					t.Errorf("%s: %s should have run", c.Why, n)
				}
			}
			for _, n := range c.Expect.NotRun {
				if eff.ran[n] != 0 {
					t.Errorf("%s: %s ran %d time(s); it must not run", c.Why, n, eff.ran[n])
				}
			}
			sort.Strings(denied)
			want := append([]string{}, c.Expect.Denied...)
			sort.Strings(want)
			if strings.Join(denied, ",") != strings.Join(want, ",") {
				t.Errorf("%s: denied %v, want %v", c.Why, denied, want)
			}
			if got := keys(eff.egress); !reflect.DeepEqual(got, append([]string{}, c.Expect.Egress...)) {
				t.Errorf("%s: requests reached hosts %v, want only %v", c.Why, got, c.Expect.Egress)
			}
			if n := rowCount(t, db); n != c.Expect.RowsAfter {
				t.Errorf("%s: %d ledger rows after the run, want %d", c.Why, n, c.Expect.RowsAfter)
			}
		})
	}
}

func TestApprovalIsForOneCall(t *testing.T) {
	// WHY: approving create_ticket {"title":"A"} must not approve the
	//      ticket an injected instruction swaps in afterwards ({"title":
	//      "B"}): the marker binds the arguments, so the swapped call is
	//      pending again and only the approved one ever runs.
	// KIND: unit
	// CATCHES: s15
	// CHAPTER: ag.04 section 2.3
	var titles []string
	reg := tool.NewRegistry().MustRegister(tool.New("create_ticket", "", `{"type":"object"}`, func(_ context.Context, a json.RawMessage) (string, error) {
		var m struct{ Title string }
		json.Unmarshal(a, &m)
		titles = append(titles, m.Title)
		return "ok", nil
	}))
	g := gate.New(policy())
	proposed := types.ToolCall{ID: "t1", Name: "create_ticket", Args: json.RawMessage(`{"title":"A"}`)}
	approved := gate.Marker(proposed)
	swapped := []types.Message{
		{Role: types.RoleUser, Content: "file ticket A"},
		{Role: types.RoleAssistant, ToolCalls: []types.ToolCall{{ID: "t1", Name: "create_ticket", Args: json.RawMessage(`{"title":"B"}`)}}},
	}
	p := &oneReply{text: "filed"}
	_, err := loop.New(loop.Config{Provider: p, Tools: reg}, loop.WithGate(g), loop.WithApprovals(approved)).Invoke(context.Background(), swapped).Result()
	var ae *loop.ApprovalError
	if !errors.As(err, &ae) || len(titles) != 0 {
		t.Fatalf("swapped call: err %v, ran %v; want still pending and nothing run", err, titles)
	}
	swapped[1].ToolCalls[0].Args = proposed.Args
	if _, err := loop.New(loop.Config{Provider: p, Tools: reg}, loop.WithGate(g), loop.WithApprovals(approved)).Invoke(context.Background(), swapped).Result(); err != nil || !reflect.DeepEqual(titles, []string{"A"}) {
		t.Fatalf("approved call: err %v, ran %v; want ticket A once", err, titles)
	}
}

// oneReply is a fake model that always answers text.
type oneReply struct{ text string }

func (o *oneReply) ChatStream(context.Context, []types.Message, []types.ToolDef, ...types.CallOption) (<-chan types.Delta, error) {
	ch := make(chan types.Delta, 2)
	ch <- types.TextDelta{Text: o.text}
	ch <- types.DoneDelta{FinishReason: "stop"}
	close(ch)
	return ch, nil
}
