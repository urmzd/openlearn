package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/urmzd/saige/agent/types"
)

// ModelInfo describes a model the user can pick in setup.
type ModelInfo struct {
	Name string `json:"name"`
	// SizeBytes is 0 when the backend does not report it.
	SizeBytes int64 `json:"sizeBytes"`
}

// ListOllamaModels asks a local ollama daemon what it has pulled. It is the
// only discovery path that works without credentials, which is why ollama is
// the default backend.
func ListOllamaModels(ctx context.Context, host string) ([]ModelInfo, error) {
	if host == "" {
		host = DefaultOllamaHost
	}
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(host, "/")+"/api/tags", nil)
	if err != nil {
		return nil, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama not reachable at %s: %w", host, err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama returned %s", resp.Status)
	}

	var payload struct {
		Models []struct {
			Name string `json:"name"`
			Size int64  `json:"size"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, fmt.Errorf("ollama tags: %w", err)
	}

	out := make([]ModelInfo, 0, len(payload.Models))
	for _, m := range payload.Models {
		// Embedding models cannot chat; leaving them in the picker only creates
		// confusing failures at generation time.
		if strings.Contains(m.Name, "embed") {
			continue
		}
		out = append(out, ModelInfo{Name: m.Name, SizeBytes: m.Size})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

// WithEnum returns a schema mutator that constrains a top-level property to a
// set of values. Used for enums whose members depend on runtime state.
func WithEnum(property string, values []string) func(*types.ParameterSchema) {
	return func(s *types.ParameterSchema) {
		if pd, ok := s.Properties[property]; ok {
			pd.Enum = values
			s.Properties[property] = pd
		}
	}
}

// WithArrayEnum constrains the element values of an array-of-string property.
func WithArrayEnum(property string, values []string) func(*types.ParameterSchema) {
	return func(s *types.ParameterSchema) {
		arr, ok := s.Properties[property]
		if !ok || arr.Items == nil {
			return
		}
		item := *arr.Items
		item.Enum = values
		arr.Items = &item
		s.Properties[property] = arr
	}
}

// WithItemEnum constrains a property inside the element type of an array
// property, e.g. the "area" field of every item in "criteria".
func WithItemEnum(arrayProperty, itemProperty string, values []string) func(*types.ParameterSchema) {
	return func(s *types.ParameterSchema) {
		arr, ok := s.Properties[arrayProperty]
		if !ok || arr.Items == nil {
			return
		}
		item := *arr.Items
		if item.Properties == nil {
			return
		}
		pd, ok := item.Properties[itemProperty]
		if !ok {
			return
		}
		pd.Enum = values
		item.Properties[itemProperty] = pd
		arr.Items = &item
		s.Properties[arrayProperty] = arr
	}
}
