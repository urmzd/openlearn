// Package embed turns chunk texts into vectors (ag.06) through any
// OpenAI-compatible /v1/embeddings: the learner engine's (mean-pooled,
// L2-normalized, openai-subset.v1) or a frontier API's, with the provider's
// retry policy (ag.01) for overload and broken connections.
//
// Chapter: content/tracks/ai-platform-engineering/07-retrieval-and-rag/06-rag-ingest.md.
package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"tinyllm/agent/provider"
)

// Embedder maps texts to vectors, one per text, in order.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// HTTPEmbedder calls POST {BaseURL}/embeddings in batches.
type HTTPEmbedder struct {
	BaseURL   string // ends in /v1
	APIKey    string // bearer key; empty sends none
	Model     string
	BatchSize int // inputs per request (64; the contract allows at most 256)
	Client    *http.Client
	Retry     provider.RetryPolicy
}

// Embed sends len(texts)/BatchSize requests and reassembles the answers by
// their index field (a server may return them in any order). It fails when
// an answer has the wrong count, a duplicate or missing index, or a vector
// whose length differs from the first one.
func (e HTTPEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	// SOLUTION-BEGIN ag.06
	bs := e.BatchSize
	if bs <= 0 {
		bs = 64
	}
	bs = min(bs, 256)
	out := make([][]float32, len(texts))
	dim := -1
	for start := 0; start < len(texts); start += bs {
		batch := texts[start:min(start+bs, len(texts))]
		vecs, err := e.batch(ctx, batch)
		if err != nil {
			return nil, err
		}
		for i, v := range vecs {
			if dim < 0 {
				dim = len(v)
			}
			if len(v) != dim || dim == 0 {
				return nil, fmt.Errorf("embed: vector %d has %d dimensions, want %d", start+i, len(v), dim)
			}
			out[start+i] = v
		}
	}
	return out, nil
	// SOLUTION-END
}

func (e HTTPEmbedder) batch(ctx context.Context, batch []string) ([][]float32, error) {
	// SOLUTION-BEGIN ag.06
	body, err := json.Marshal(map[string]any{"model": e.Model, "input": batch})
	if err != nil {
		return nil, err
	}
	pol := e.Retry
	if pol.MaxAttempts <= 0 {
		pol.MaxAttempts = 4
	}
	client := e.Client
	if client == nil {
		client = http.DefaultClient
	}
	for attempt := 1; ; attempt++ {
		vecs, err := e.post(ctx, client, body, len(batch))
		if err == nil {
			return vecs, nil
		}
		if !provider.Retryable(err) || attempt >= pol.MaxAttempts {
			return nil, err
		}
		sleep := pol.Sleep
		if sleep == nil {
			sleep = func(ctx context.Context, d time.Duration) error {
				t := time.NewTimer(d)
				defer t.Stop()
				select {
				case <-t.C:
					return nil
				case <-ctx.Done():
					return ctx.Err()
				}
			}
		}
		if serr := sleep(ctx, pol.Delay(attempt, err)); serr != nil {
			return nil, serr
		}
	}
	// SOLUTION-END
}

func (e HTTPEmbedder) post(ctx context.Context, client *http.Client, body []byte, n int) ([][]float32, error) {
	// SOLUTION-BEGIN ag.06
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(e.BaseURL, "/")+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if e.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+e.APIKey)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		ae := &provider.APIError{Status: resp.StatusCode, Message: strings.TrimSpace(string(b))}
		if s := resp.Header.Get("Retry-After"); s != "" {
			var secs int
			if _, err := fmt.Sscanf(s, "%d", &secs); err == nil && secs > 0 {
				ae.RetryAfter = time.Duration(secs) * time.Second
			}
		}
		return nil, ae
	}
	var out struct {
		Data []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("embed: bad response: %w", err)
	}
	if len(out.Data) != n {
		return nil, fmt.Errorf("embed: %d embeddings for %d inputs", len(out.Data), n)
	}
	vecs := make([][]float32, n)
	for _, d := range out.Data {
		if d.Index < 0 || d.Index >= n || vecs[d.Index] != nil {
			return nil, fmt.Errorf("embed: bad or repeated index %d", d.Index)
		}
		vecs[d.Index] = d.Embedding
	}
	return vecs, nil
	// SOLUTION-END
}
