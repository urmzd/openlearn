package llm

import (
	"context"
	"testing"

	"github.com/urmzd/saige/agent/provider/catalog"
	"github.com/urmzd/saige/agent/types"
)

// TestCreativityCompilesPerModel checks what the creativity dial sends: a
// temperature where the model takes one, nothing where sampling would be
// rejected, and the thinking toggle left as configured.
func TestCreativityCompilesPerModel(t *testing.T) {
	cases := []struct {
		kind        Kind
		model       string
		temperature bool
	}{
		{KindOllama, "qwen3.5:9b", true},
		{KindOpenAI, "gpt-4.1-mini", true},
		{KindOpenAI, "gpt-6-luna", false},
		{KindAnthropic, "claude-haiku-5-5", false},
	}
	for _, tc := range cases {
		caps := catalog.MustLookup(string(tc.kind), tc.model)
		o := requestOptions(tc.kind, tc.model)
		// The dial must send sampling exactly where a raw temperature
		// would be accepted.
		withTemp := o.Clone()
		temp := 0.3
		withTemp.Temperature = &temp
		if accepted := caps.ValidateOptions(withTemp) == nil; accepted != tc.temperature {
			t.Fatalf("%s/%s: raw temperature accepted = %v, test expects %v", tc.kind, tc.model, accepted, tc.temperature)
		}
		raw := o.Clone()
		o.Dials = dials()
		eff, rep, err := types.CompileOptions(caps, o, types.DialContext{Schema: true})
		if err != nil {
			t.Fatalf("%s/%s: %v", tc.kind, tc.model, err)
		}
		if got := eff.Temperature != nil; got != tc.temperature {
			t.Errorf("%s/%s: temperature sent = %v, want %v (report %+v)", tc.kind, tc.model, got, tc.temperature, rep)
		}
		if tc.temperature && *eff.Temperature != 0.3 {
			t.Errorf("%s/%s: temperature = %v, want 0.3", tc.kind, tc.model, *eff.Temperature)
		}
		if (raw.ReasoningEnabled == nil) != (eff.ReasoningEnabled == nil) {
			t.Errorf("%s/%s: thinking toggle changed: %v -> %v", tc.kind, tc.model, raw.ReasoningEnabled, eff.ReasoningEnabled)
		}
	}
}

// TestNewAcceptsEveryKind builds a client for each backend, so a dial a
// model cannot honor never fails construction.
func TestNewAcceptsEveryKind(t *testing.T) {
	for _, cfg := range []Config{
		{Kind: KindOllama, Model: "qwen3.5:9b"},
		{Kind: KindOpenAI, Model: "gpt-6-luna", APIKey: "k"},
		{Kind: KindOpenAI, Model: "gpt-4.1-mini", APIKey: "k"},
		{Kind: KindAnthropic, Model: "claude-haiku-5-5", APIKey: "k"},
		{Kind: KindGoogle, Model: "gemini-3.1-flash-lite", APIKey: "k"},
	} {
		if _, err := New(context.Background(), cfg, nil); err != nil {
			t.Errorf("%s/%s: %v", cfg.Kind, cfg.Model, err)
		}
	}
}
