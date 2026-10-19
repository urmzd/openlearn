// Package assemble turns retrieved chunks into the context an agent reads
// (ag.08): numbered blocks packed under a token budget, citations that name
// stored chunks, a parser for the [n] markers a model writes in its answer,
// and the search_docs tool that puts retrieval in the agent's toolset.
//
// Chapter: content/courses/ai-platform-engineering/07-retrieval-and-rag/08-context-assembly.md.
package assemble

import (
	"context"
	"crypto/sha256"
	"fmt"
	"strings"

	"tinyllm/agent/rag/retrieve"
)

// Counter counts the tokens of a text with the model's tokenizer (ag.06's
// chunk.Tokenizer satisfies it). Counts of pieces do not add up to the
// count of their concatenation, so only whole texts are counted.
type Counter interface {
	Count(ctx context.Context, text string) (int, error)
}

// Citation names the stored chunk behind one [N] block.
type Citation struct {
	N       int    `json:"n"`
	ChunkID string `json:"chunk_id"`
	DocID   string `json:"doc_id"`
	URI     string `json:"uri"`
}

// Context is the assembled text and what went into it.
type Context struct {
	Text    string
	Tokens  int            // Counter's count of Text
	Used    []retrieve.Hit // in block order
	Dropped int            // hits left out: over budget or duplicates
}

// Assembler packs hits for query q into at most maxTokens tokens.
type Assembler interface {
	Assemble(ctx context.Context, q string, hits []retrieve.Hit, maxTokens int) (Context, []Citation, error)
}

// Block is the text of one context block: "[n] <uri>\n<text>\n".
func Block(n int, h retrieve.Hit) string {
	return fmt.Sprintf("[%d] %s\n%s\n", n, h.URI, h.Text)
}

// Packer is the course's Assembler. Hits are taken in the order given (the
// retriever's rank order). A hit is skipped when its chunk id, or its exact
// text, is already in the context; otherwise its block is appended (blocks
// are separated by one blank line) when the whole text, counted again,
// stays within maxTokens, and skipped when it does not: a later, shorter
// hit may still fit. A chunk is never cut: a cited block holds the stored
// text exactly. With a Ledger, block numbers come from it, so they stay
// unique across every search of one run; without one they count from 1.
type Packer struct {
	Counter Counter
	Ledger  *Ledger // nil: numbers start at 1 in each Assemble
}

func fingerprint(s string) [32]byte { return sha256.Sum256([]byte(s)) }

// Assemble implements Assembler. A counter error is returned, never taken
// as zero tokens.
func (p Packer) Assemble(ctx context.Context, q string, hits []retrieve.Hit, maxTokens int) (Context, []Citation, error) {
	if p.Ledger != nil {
		p.Ledger.mu.Lock()
		defer p.Ledger.mu.Unlock()
	}
	var out Context
	var cites []Citation
	var b strings.Builder
	seenID := map[string]bool{}
	seenText := map[[32]byte]bool{}
	for _, h := range hits {
		fp := fingerprint(h.Text)
		if seenID[h.ChunkID] || seenText[fp] {
			out.Dropped++
			continue
		}
		n := len(cites) + 1
		if p.Ledger != nil {
			n = p.Ledger.peek(h)
		}
		candidate := b.String()
		if candidate != "" {
			candidate += "\n"
		}
		candidate += Block(n, h)
		tokens, err := p.Counter.Count(ctx, candidate)
		if err != nil {
			return Context{}, nil, fmt.Errorf("assemble: count tokens: %w", err)
		}
		if tokens > maxTokens {
			out.Dropped++
			continue
		}
		c := Citation{N: n, ChunkID: h.ChunkID, DocID: h.DocID, URI: h.URI}
		if p.Ledger != nil {
			c = p.Ledger.cite(h)
		}
		b.Reset()
		b.WriteString(candidate)
		out.Tokens = tokens
		out.Used = append(out.Used, h)
		cites = append(cites, c)
		seenID[h.ChunkID] = true
		seenText[fp] = true
	}
	out.Text = b.String()
	return out, cites, nil
}
