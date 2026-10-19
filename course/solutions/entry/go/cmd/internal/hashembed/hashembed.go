// Package hashembed is "hash16", the deterministic lexical embedding the
// course fixtures use (course/oracle/ag.07/make_fixtures.py): every term of
// the text (ag.07's Terms) adds +1 or -1, the sign from the top bit of its
// FNV-1a 64 hash, to coordinate hash mod 16, and the vector is L2-normalized.
//
// It is the stand-in for a model's /v1/embeddings where no embedding model
// runs: the RAG index ctl builds from docs/ in [build] (ag.06), and the
// usage policy's classifier head in PR CI (gw.08), whose head
// (artifacts/models/hash16/heads/usage.json) scripts/fit_policy_head.py fits
// on these same vectors with L6.5's fit_linear_head. The nightly run swaps
// in the served model's embeddings by naming another embedding_model.
package hashembed

import (
	"context"

	"tinyllm/agent/rag/retrieve"
)

// Model is the embedding_model name of these vectors.
const Model = "hash16"

// Dim is their dimension.
const Dim = 16

// Vector embeds one text.
func Vector(text string) []float32 {
	v := make([]float32, Dim)
	for _, t := range retrieve.Terms(text) {
		h := retrieve.FNV1a64(t)
		if h>>63 == 1 {
			v[h%Dim]--
		} else {
			v[h%Dim]++
		}
	}
	return retrieve.Normalize(v)
}

// Embedder embeds texts in order (ag.06's embed.Embedder and ag.07's
// retrieve.Embedder).
type Embedder struct{}

func (Embedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = Vector(t)
	}
	return out, nil
}
