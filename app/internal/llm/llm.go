// Package llm adapts the saige agent SDK to whiteboardy's needs: building a
// provider from config, and running one-shot structured generations that come
// back as typed Go values.
package llm

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	agentsdk "github.com/urmzd/saige/agent"
	"github.com/urmzd/saige/agent/provider"
	"github.com/urmzd/saige/agent/provider/catalog"
	"github.com/urmzd/saige/agent/provider/ollama"
	"github.com/urmzd/saige/agent/provider/retry"
	"github.com/urmzd/saige/agent/types"
)

// Kind identifies a supported provider backend. The values match saige's
// provider names, so a Kind passes straight to provider.Build.
type Kind string

const (
	KindOllama    Kind = provider.Ollama
	KindOpenAI    Kind = provider.OpenAI
	KindAnthropic Kind = provider.Anthropic
	KindGoogle    Kind = provider.Google
)

// Config describes how to reach a model.
type Config struct {
	Kind Kind `json:"kind"`
	// Model is the model identifier, e.g. "qwen3.5:9b" or "claude-haiku-5-5".
	Model string `json:"model"`
	// Host is the base URL. Only meaningful for ollama; defaults to the local
	// daemon when empty.
	Host string `json:"host"`
	// APIKey authenticates hosted providers. Ignored by ollama.
	APIKey string `json:"apiKey"`
}

// DefaultOllamaHost is where a local ollama daemon listens.
const DefaultOllamaHost = provider.DefaultOllamaHost

// ErrNoModel is returned when a config names no model.
var ErrNoModel = errors.New("llm: no model configured")

// Client wraps a provider and exposes the generation helpers the harness uses.
type Client struct {
	provider types.Provider
	cfg      Config
	log      *slog.Logger
}

// New builds a Client from config. It does not contact the provider, so a bad
// host or key surfaces on first use rather than here.
func New(ctx context.Context, cfg Config, log *slog.Logger) (*Client, error) {
	if log == nil {
		log = slog.Default()
	}
	if strings.TrimSpace(cfg.Model) == "" {
		return nil, ErrNoModel
	}
	kind := cfg.Kind
	if kind == "" {
		kind = KindOllama
	}

	pc := provider.Config{
		Provider: string(kind),
		Model:    cfg.Model,
		Options:  requestOptions(kind, cfg.Model),
		Dials:    dials(),
		// Settings are the only source of credentials and hosts, so the
		// environment is never consulted.
		Getenv: func(string) string { return "" },
	}
	switch kind {
	case KindOllama:
		pc.BaseURL = cfg.Host
	default:
		if cfg.APIKey == "" {
			return nil, fmt.Errorf("llm: %s requires an api key", kind)
		}
		pc.APIKey = cfg.APIKey
	}

	p, err := provider.Build(ctx, pc)
	if err != nil {
		return nil, fmt.Errorf("llm: %w", err)
	}
	switch kind {
	case KindOllama:
		// num_ctx is set explicitly because ollama's default window is far
		// smaller than a review prompt (problem, hidden rubric, whole board,
		// event log) and it truncates past the limit silently. It has no
		// provider-neutral option, so it goes on the client Build made.
		if a, ok := p.(*ollama.Adapter); ok {
			if opts, ok := a.Client.ChatOptions.(map[string]any); ok {
				opts["num_ctx"] = contextBudget
			} else {
				a.Client.ChatOptions = map[string]any{"num_ctx": contextBudget}
			}
		}
	case KindAnthropic, KindOpenAI:
		// These adapters no longer retry inside the SDK.
		p = retry.New(p, retry.DefaultConfig())
	}

	return &Client{provider: p, cfg: cfg, log: log}, nil
}

// requestOptions returns the raw request options for a model. Each control
// is sent only where the catalog says the model accepts it, so Build never
// rejects a model the user picked.
//
// For ollama, thinking is off for every call on models that can toggle it.
// Schema-constrained calls need it off anyway, and for the coach's prose it
// is actively harmful: a reasoning model spends tens of seconds thinking
// before emitting its first text token, so the message bubble opens and then
// sits visibly empty. The judgment already happened in the decision call;
// this one just writes two sentences.
func requestOptions(kind Kind, model string) types.RequestOptions {
	caps := catalog.MustLookup(string(kind), model)
	var o types.RequestOptions
	try := func(set func(*types.RequestOptions)) {
		next := o.Clone()
		set(&next)
		if caps.ValidateOptions(next) == nil {
			o = next
		}
	}
	if kind == KindOllama && caps.Supports(types.CapReasoningToggle) {
		try(func(r *types.RequestOptions) { off := false; r.ReasoningEnabled = &off })
	}
	return o
}

// dials returns the model-neutral generation settings for every call. The
// adapter compiles them for the model on each request, schema-constrained
// calls included: creativity becomes a temperature where the model takes
// one, and is dropped on models that reject sampling controls, such as
// reasoning models that only sample at their defaults.
func dials() types.Dials {
	c := Creativity
	return types.Dials{Creativity: &c}
}

// Config returns the config the client was built from, with the API key blanked.
func (c *Client) Config() Config {
	safe := c.cfg
	if safe.APIKey != "" {
		safe.APIKey = "***"
	}
	return safe
}

// ProviderName reports the backend in use.
func (c *Client) ProviderName() string { return string(c.cfg.Kind) }

// Model reports the model in use.
func (c *Client) Model() string { return c.cfg.Model }

// Text runs a single-turn completion and returns the accumulated text.
func (c *Client) Text(ctx context.Context, system, user string) (string, error) {
	text, err := agentsdk.CollectText(c.agent(system).Invoke(ctx, prompt(user)))
	if err != nil {
		return text, fmt.Errorf("llm: %w", err)
	}
	if strings.TrimSpace(text) == "" {
		return "", errors.New("llm: model returned empty response")
	}
	return text, nil
}

// TextStream runs a single-turn completion and calls onDelta for each chunk as
// it arrives, returning the accumulated text. Used for prose the user watches
// being written; schema-constrained calls go through Structured instead,
// because partial JSON is not something a UI can render.
func (c *Client) TextStream(ctx context.Context, system, user string, onDelta func(string)) (string, error) {
	t, err := agentsdk.Collect(c.agent(system).Invoke(ctx, prompt(user)), func(d types.Delta) {
		if td, ok := d.(types.TextContentDelta); ok && onDelta != nil && td.Content != "" {
			onDelta(td.Content)
		}
	})
	if err != nil {
		return t.Text, fmt.Errorf("llm: %w", err)
	}
	return t.Text, nil
}

// structuredRepairs is how many times an answer that fails to parse or match
// the schema is sent back to the model with the error. Small local models
// occasionally return an empty or malformed answer; one retry recovers it.
const structuredRepairs = 1

// Structured runs a single-turn completion constrained to T's JSON schema and
// unmarshals the result. Schema mutators let callers narrow enums that depend
// on runtime state (for example the skill areas valid for the current mode).
// The answer is extracted tolerantly (code fences, prose, <think> blocks).
func Structured[T any](ctx context.Context, c *Client, system, user string, mutators ...func(*types.ParameterSchema)) (T, error) {
	schema := types.SchemaFrom[T]()
	for _, m := range mutators {
		m(&schema)
	}
	out, _, err := agentsdk.Structured(ctx, c.agent(system), prompt(user), agentsdk.OutputSpec[T]{
		Schema: &schema,
		Repair: structuredRepairs,
	})
	if err != nil {
		return out, fmt.Errorf("llm: %w", err)
	}
	return out, nil
}

// Creativity is the sampling intent used for every generation. Problem
// generation wants some variety so repeated sessions do not converge on the
// same exercise; grading wants determinism. This sits closer to the grading
// end, and variety comes from the topic instead. On a model that takes
// sampling controls, focused is temperature 0.3, with top_p 0.9 where the
// model declares it.
const Creativity = types.CreativityFocused

// contextBudget is the context window whiteboardy asks a local model for.
// Large enough for a full review prompt, small enough that a small model on a
// laptop still fits it in memory.
const contextBudget = 16384

// agent builds a single-turn agent. Each call gets its own, so concurrent
// generations never share a conversation tree.
func (c *Client) agent(system string) *agentsdk.Agent {
	return agentsdk.NewAgent(agentsdk.AgentConfig{
		Name:         "whiteboardy",
		SystemPrompt: system,
		Provider:     c.provider,
	}, agentsdk.WithMaxIter(1), agentsdk.WithLogger(c.log))
}

func prompt(user string) []types.Message {
	return []types.Message{types.NewUserMessage(user)}
}
