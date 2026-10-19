package assemble

// tool.go: search_docs, retrieval as a tool the agent calls (ag.02's Tool).

import (
	"context"
	"encoding/json"
	"fmt"

	"tinyllm/agent/rag/retrieve"
	"tinyllm/agent/tool"
)

// SearchDocsName is the tool's name.
const SearchDocsName = "search_docs"

// SearchDocsSchema is the tool's JSON Schema: a non-empty query, an
// optional k (1 to 20, default 5), an optional list of document ids to
// search within. Anything else is rejected by the registry before the tool
// runs.
const SearchDocsSchema = `{
  "type": "object",
  "properties": {
    "query":   {"type": "string", "minLength": 1, "maxLength": 1000, "description": "what to look for"},
    "k":       {"type": "integer", "minimum": 1, "maximum": 20, "description": "how many chunks, default 5"},
    "doc_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 20, "description": "only search these documents"}
  },
  "required": ["query"],
  "additionalProperties": false
}`

// SearchDocsDescription is what the model reads about the tool.
const SearchDocsDescription = "Search the documentation. Returns numbered excerpts; cite the ones you use as [n] in your answer."

// NoResults is the tool's answer when nothing matched: an answer the model
// can act on, not a tool error.
const NoResults = "no results"

// SearchOptions configure SearchDocs.
type SearchOptions struct {
	MaxTokens int     // context budget per call; 0 means 1024
	Ledger    *Ledger // the run's citation ledger; nil gives each call its own numbering
}

// SearchDocs is the search_docs tool over retriever r and counter c. Each
// call retrieves k hits (filtered to doc_ids when given), packs them with a
// Packer that shares opts.Ledger, and returns the context text, or
// NoResults when no hit fits. Retrieval and counting errors are tool errors.
func SearchDocs(r retrieve.Retriever, c Counter, opts SearchOptions) tool.Tool {
	return tool.New(SearchDocsName, SearchDocsDescription, SearchDocsSchema,
		func(ctx context.Context, args json.RawMessage) (string, error) {
			return searchDocs(ctx, r, c, opts, args)
		})
}

func searchDocs(ctx context.Context, r retrieve.Retriever, c Counter, opts SearchOptions, args json.RawMessage) (string, error) {
	var in struct {
		Query  string   `json:"query"`
		K      int      `json:"k"`
		DocIDs []string `json:"doc_ids"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return "", fmt.Errorf("search_docs: %w", err)
	}
	if in.K == 0 {
		in.K = 5
	}
	budget := opts.MaxTokens
	if budget <= 0 {
		budget = 1024
	}
	hits, err := r.Retrieve(ctx, in.Query, in.K, retrieve.Filter{DocIDs: in.DocIDs})
	if err != nil {
		return "", fmt.Errorf("search_docs: retrieve: %w", err)
	}
	out, _, err := Packer{Counter: c, Ledger: opts.Ledger}.Assemble(ctx, in.Query, hits, budget)
	if err != nil {
		return "", fmt.Errorf("search_docs: %w", err)
	}
	if out.Text == "" {
		return NoResults, nil
	}
	return out.Text, nil
}
