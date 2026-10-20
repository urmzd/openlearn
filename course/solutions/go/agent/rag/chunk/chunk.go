// Package chunk splits documents into chunks that fit a token budget (ag.06),
// counting with the tokenizer of the model that will embed them: the
// system engine's /v1/tokenize, or the tracer's byte-level tokenizer.
//
// Chapter: content/courses/ai-engineering/27-retrieval-and-rag/01-rag-ingest.md.
// Format: course/contracts/formats/rag-index.md (chunk_id, n_tokens,
// fingerprint).
package chunk

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"

	"tinyllm/agent/rag/source"
)

// Tokenizer counts the tokens of a text.
type Tokenizer interface {
	Count(ctx context.Context, text string) (int, error)
}

// Bytes is the tracer's tokenizer (D32): one token per UTF-8 byte.
type Bytes struct{}

func (Bytes) Count(_ context.Context, text string) (int, error) { return len(text), nil }

// HTTPTokenizer counts with POST {BaseURL}/tokenize (openai-subset.v1,
// engine tier): the length of the returned ids.
type HTTPTokenizer struct {
	BaseURL string // ends in /v1
	Model   string
	Client  *http.Client
}

func (h HTTPTokenizer) Count(ctx context.Context, text string) (int, error) {
	body, _ := json.Marshal(map[string]any{"model": h.Model, "text": text})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(h.BaseURL, "/")+"/tokenize", bytes.NewReader(body))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	c := h.Client
	if c == nil {
		c = http.DefaultClient
	}
	resp, err := c.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return 0, fmt.Errorf("chunk: tokenize: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}
	var out struct {
		IDs []int `json:"ids"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return 0, fmt.Errorf("chunk: tokenize: %w", err)
	}
	return len(out.IDs), nil
}

// Chunk is one piece of a document. ID is "<doc_id>#<Index>"; Fingerprint
// is the hex SHA-256 of Text.
type Chunk struct {
	ID, DocID   string
	Index       int
	Text        string
	Tokens      int
	Fingerprint string
}

// Chunker splits a document.
type Chunker interface {
	Chunk(ctx context.Context, d source.Doc) ([]Chunk, error)
}

// TokenChunker packs a document's paragraphs (then sentences, then words,
// then characters, when one piece alone is over budget) into chunks of at
// most MaxTokens tokens. Every candidate chunk is counted whole: token
// counts do not add up (a BPE tokenizer merges across the join, a
// tokenizer may count the separator), so summing pieces is wrong both ways.
type TokenChunker struct {
	Tok       Tokenizer
	MaxTokens int
}

var (
	paraRE     = regexp.MustCompile(`\n\s*\n`)
	sentenceRE = regexp.MustCompile(`[^.!?]+(?:[.!?]+|$)`)
)

// Fingerprint is the hex SHA-256 of text.
func Fingerprint(text string) string {
	sum := sha256.Sum256([]byte(text))
	return hex.EncodeToString(sum[:])
}

func (c TokenChunker) Chunk(ctx context.Context, d source.Doc) ([]Chunk, error) {
	if c.MaxTokens < 1 {
		return nil, fmt.Errorf("chunk: MaxTokens must be at least 1")
	}
	type piece struct {
		text string
		sep  string // what joins it to the previous piece in the same chunk
	}
	var pieces []piece
	for _, para := range paraRE.Split(d.Text, -1) {
		para = strings.TrimSpace(para)
		if para == "" {
			continue
		}
		parts, err := c.fit(ctx, para)
		if err != nil {
			return nil, err
		}
		for i, p := range parts {
			sep := " "
			if i == 0 {
				sep = "\n\n"
			}
			pieces = append(pieces, piece{p, sep})
		}
	}
	var out []Chunk
	emit := func(text string, n int) {
		out = append(out, Chunk{ID: d.ID + "#" + strconv.Itoa(len(out)), DocID: d.ID, Index: len(out), Text: text, Tokens: n, Fingerprint: Fingerprint(text)})
	}
	cur, curN := "", 0
	for _, p := range pieces {
		if cur == "" {
			n, err := c.Tok.Count(ctx, p.text)
			if err != nil {
				return nil, err
			}
			cur, curN = p.text, n
			continue
		}
		cand := cur + p.sep + p.text
		n, err := c.Tok.Count(ctx, cand)
		if err != nil {
			return nil, err
		}
		if n <= c.MaxTokens {
			cur, curN = cand, n
			continue
		}
		emit(cur, curN)
		cur = p.text
		if curN, err = c.Tok.Count(ctx, cur); err != nil {
			return nil, err
		}
	}
	if cur != "" {
		emit(cur, curN)
	}
	return out, nil
}

// fit splits text into pieces that each fit the budget: sentences, then
// words, then the longest UTF-8-safe prefixes.
func (c TokenChunker) fit(ctx context.Context, text string) ([]string, error) {
	n, err := c.Tok.Count(ctx, text)
	if err != nil {
		return nil, err
	}
	if n <= c.MaxTokens {
		return []string{text}, nil
	}
	var parts []string
	switch {
	case len(sentenceRE.FindAllString(text, -1)) > 1:
		for _, s := range sentenceRE.FindAllString(text, -1) {
			if s = strings.TrimSpace(s); s != "" {
				parts = append(parts, s)
			}
		}
	case len(strings.Fields(text)) > 1:
		parts = strings.Fields(text)
	default:
		return c.hardSplit(ctx, text)
	}
	var out []string
	for _, p := range parts {
		sub, err := c.fit(ctx, p)
		if err != nil {
			return nil, err
		}
		out = append(out, sub...)
	}
	return out, nil
}

// hardSplit cuts a word that alone is over budget into the longest prefixes
// that fit, never inside a UTF-8 character (binary search on the count).
func (c TokenChunker) hardSplit(ctx context.Context, text string) ([]string, error) {
	var out []string
	for text != "" {
		var bounds []int // byte offsets of every character end
		for i := range text {
			if i > 0 {
				bounds = append(bounds, i)
			}
		}
		bounds = append(bounds, len(text))
		lo, hi := 0, len(bounds) // bounds[:lo] fit; find the most that do
		for lo < hi {
			mid := (lo + hi + 1) / 2
			n, err := c.Tok.Count(ctx, text[:bounds[mid-1]])
			if err != nil {
				return nil, err
			}
			if n <= c.MaxTokens {
				lo = mid
			} else {
				hi = mid - 1
			}
		}
		cut := bounds[max(lo, 1)-1] // one character over budget alone is emitted whole
		out = append(out, text[:cut])
		text = text[cut:]
	}
	return out, nil
}
