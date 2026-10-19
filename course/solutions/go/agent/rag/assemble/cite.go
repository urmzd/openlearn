package assemble

// cite.go: citation numbers that stay unique across one run, and the
// parser that reads them back out of the model's answer.

import (
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"

	"tinyllm/agent/rag/retrieve"
)

// Ledger numbers the chunks one agent run has been shown. A chunk keeps the
// number it got first; a new chunk gets the next number. It is safe for
// concurrent use: parallel search_docs calls share it.
type Ledger struct {
	mu   sync.Mutex
	byID map[string]int
	list []Citation
}

// NewLedger returns an empty ledger.
func NewLedger() *Ledger { return &Ledger{byID: map[string]int{}} }

// Peek is the number h would get, without recording it.
func (l *Ledger) Peek(h retrieve.Hit) int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.peek(h)
}

// Cite records h (once) and returns its citation.
func (l *Ledger) Cite(h retrieve.Hit) Citation {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.cite(h)
}

// peek and cite run with l.mu held. A Packer holds it for a whole
// Assemble, so the number a block shows is the number it is recorded under
// even when two searches run at once.
func (l *Ledger) peek(h retrieve.Hit) int {
	if n, ok := l.byID[h.ChunkID]; ok {
		return n
	}
	return len(l.list) + 1
}

func (l *Ledger) cite(h retrieve.Hit) Citation {
	if l.byID == nil {
		l.byID = map[string]int{}
	}
	if n, ok := l.byID[h.ChunkID]; ok {
		return l.list[n-1]
	}
	c := Citation{N: len(l.list) + 1, ChunkID: h.ChunkID, DocID: h.DocID, URI: h.URI}
	l.byID[h.ChunkID] = c.N
	l.list = append(l.list, c)
	return c
}

// Citations lists every citation recorded, by number.
func (l *Ledger) Citations() []Citation {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]Citation(nil), l.list...)
}

// markerRE matches one bracketed group of citation numbers: [3], [1, 2],
// [1,2,3]. A group holding anything else ([a], [1-3], [ ]) is not a marker.
var markerRE = regexp.MustCompile(`\[(\d+(?:\s*,\s*\d+)*)\]`)

// ParseCitations returns the distinct citation numbers an answer uses, in
// order of first appearance. Numbers below 1 are ignored.
func ParseCitations(answer string) []int {
	var out []int
	seen := map[int]bool{}
	for _, m := range markerRE.FindAllStringSubmatch(answer, -1) {
		for _, f := range strings.Split(m[1], ",") {
			n, err := strconv.Atoi(strings.TrimSpace(f))
			if err != nil || n < 1 || seen[n] {
				continue
			}
			seen[n] = true
			out = append(out, n)
		}
	}
	return out
}

// Resolve maps the numbers an answer cites to the citations given: used in
// first-appearance order, and unknown, sorted, for every number no citation
// carries (a citation the model made up).
func Resolve(answer string, cites []Citation) (used []Citation, unknown []int) {
	byN := map[int]Citation{}
	for _, c := range cites {
		byN[c.N] = c
	}
	for _, n := range ParseCitations(answer) {
		if c, ok := byN[n]; ok {
			used = append(used, c)
		} else {
			unknown = append(unknown, n)
		}
	}
	sort.Ints(unknown)
	return used, unknown
}

// Lookup finds a stored chunk by id (retrieve.Index.Chunk has this shape).
type Lookup func(chunkID string) (retrieve.Chunk, bool)

// Unresolved returns the citations whose chunk is not stored, or whose
// document or URI differ from the stored chunk's.
func Unresolved(cites []Citation, lookup Lookup) []Citation {
	var bad []Citation
	for _, c := range cites {
		ch, ok := lookup(c.ChunkID)
		if !ok || ch.DocID != c.DocID || ch.URI != c.URI {
			bad = append(bad, c)
		}
	}
	return bad
}
