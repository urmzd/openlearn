// Course tests for ag.08: context assembly under a token budget, citations,
// and the search_docs tool. Counters here are fakes with known counts (one
// of them deliberately not additive, like a real BPE tokenizer); the index
// is ag.07's fixture (course/fixtures/ag.07/index), searched by the
// reference-or-yours retriever through its feature-hashing test embedder.
package ag_08

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"sync"
	"testing"

	"tinyllm/agent/rag/assemble"
	"tinyllm/agent/rag/retrieve"
	"tinyllm/agent/tool"
	"tinyllm/agent/types"
)

// words counts whitespace-separated words: additive over blocks.
type words struct{ calls int }

func (w *words) Count(_ context.Context, s string) (int, error) {
	w.calls++
	return len(strings.Fields(s)), nil
}

// joiner counts words plus one token per blank line, so the count of a
// concatenation is more than the sum of its parts (as with BPE merges and
// separators): only counting the whole text keeps the budget.
type joiner struct{}

func (joiner) Count(_ context.Context, s string) (int, error) {
	return len(strings.Fields(s)) + strings.Count(s, "\n\n"), nil
}

type failing struct{}

func (failing) Count(context.Context, string) (int, error) {
	return 0, errors.New("tokenizer unavailable")
}

func hit(id, uri, text string) retrieve.Hit {
	doc := strings.SplitN(id, "#", 2)[0]
	return retrieve.Hit{ChunkID: id, DocID: doc, URI: uri, Text: text}
}

var handHits = []retrieve.Hit{
	hit("docs/a.md#0", "docs/a.md", "alpha beta gamma delta"),
	hit("docs/b.md#0", "docs/b.md", "one two three four five six seven eight"),
	hit("docs/c.md#0", "docs/c.md", "short chunk"),
}

func TestHandExample(t *testing.T) {
	// WHY: the chapter's worked example (section 3): with a word counter
	//      and maxTokens = 12, block [1] (6 words) fits, docs/b.md would
	//      bring the total to 16 and is skipped, and the short docs/c.md
	//      still fits as [2] for a total of 10. Numbers follow inclusion, so
	//      the model never sees a gap.
	// KIND: unit
	// CATCHES: s02, s03, s07
	// CHAPTER: ag.08 section 3, worked example
	got, cites, err := assemble.Packer{Counter: &words{}}.Assemble(context.Background(), "q", handHits, 12)
	if err != nil {
		t.Fatal(err)
	}
	want := "[1] docs/a.md\nalpha beta gamma delta\n\n[2] docs/c.md\nshort chunk\n"
	if got.Text != want || got.Tokens != 10 || got.Dropped != 1 || len(got.Used) != 2 {
		t.Fatalf("Assemble = %q, %d tokens, %d dropped, %d used;\nwant %q, 10, 1, 2", got.Text, got.Tokens, got.Dropped, len(got.Used), want)
	}
	wantC := []assemble.Citation{
		{N: 1, ChunkID: "docs/a.md#0", DocID: "docs/a.md", URI: "docs/a.md"},
		{N: 2, ChunkID: "docs/c.md#0", DocID: "docs/c.md", URI: "docs/c.md"},
	}
	if !reflect.DeepEqual(cites, wantC) {
		t.Fatalf("citations = %+v, want %+v", cites, wantC)
	}
}

func TestBudgetNeverExceeded(t *testing.T) {
	// WHY: the catalog's rule, maxTokens respected, for every budget from 0
	//      to 60 with a counter whose counts do not add up: the reported
	//      Tokens is the counter's count of the final text and never above
	//      the budget. Summing per-block counts passes with an additive
	//      counter and overflows here at the boundaries.
	// KIND: property
	// CATCHES: s01, s04
	// CHAPTER: ag.08 section 2.1
	hits := append([]retrieve.Hit(nil), handHits...)
	hits = append(hits, hit("docs/d.md#0", "docs/d.md", "x y"), hit("docs/e.md#0", "docs/e.md", "p q r s t"))
	for budget := 0; budget <= 60; budget++ {
		got, _, err := assemble.Packer{Counter: joiner{}}.Assemble(context.Background(), "q", hits, budget)
		if err != nil {
			t.Fatal(err)
		}
		n, _ := joiner{}.Count(context.Background(), got.Text)
		if n > budget || got.Tokens != n {
			t.Fatalf("budget %d: text counts %d tokens, Tokens says %d", budget, n, got.Tokens)
		}
	}
	// exactly at the budget is allowed
	got, _, _ := assemble.Packer{Counter: &words{}}.Assemble(context.Background(), "q", handHits[:1], 6)
	if len(got.Used) != 1 {
		t.Fatalf("a block of exactly maxTokens tokens was dropped")
	}
}

func TestChunksNeverCut(t *testing.T) {
	// WHY: a cited block must hold the stored text exactly, or the citation
	//      points at words the source never said. Over many budgets the
	//      text is exactly the used hits' blocks joined by blank lines, each
	//      with its whole chunk.
	// KIND: property
	// CATCHES: s02
	// CHAPTER: ag.08 section 2.2
	stored := map[string]retrieve.Hit{}
	for _, h := range handHits {
		stored[h.ChunkID] = h
	}
	for budget := 0; budget <= 40; budget++ {
		got, cites, _ := assemble.Packer{Counter: &words{}}.Assemble(context.Background(), "q", handHits, budget)
		var blocks []string
		for i := range got.Used {
			blocks = append(blocks, assemble.Block(cites[i].N, stored[cites[i].ChunkID]))
		}
		if got.Text != strings.Join(blocks, "\n") {
			t.Fatalf("budget %d: text %q is not the used blocks %q", budget, got.Text, blocks)
		}
	}
}

func TestDedup(t *testing.T) {
	// WHY: the same chunk from two retrievers, or the same paragraph stored
	//      under two documents, would spend the budget twice on one fact.
	//      A repeated chunk id or an identical text is skipped and counted
	//      as dropped.
	// KIND: unit
	// CATCHES: s05, s06
	// CHAPTER: ag.08 section 2.2
	hits := []retrieve.Hit{
		hit("docs/a.md#0", "docs/a.md", "alpha beta"),
		hit("docs/a.md#0", "docs/a.md", "alpha beta"),
		hit("docs/copy.md#3", "docs/copy.md", "alpha beta"),
		hit("docs/b.md#1", "docs/b.md", "gamma"),
	}
	got, cites, err := assemble.Packer{Counter: &words{}}.Assemble(context.Background(), "q", hits, 100)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Used) != 2 || got.Dropped != 2 || cites[1].ChunkID != "docs/b.md#1" || cites[1].N != 2 {
		t.Fatalf("used %d, dropped %d, citations %+v; want a#0 and b#1 as [1], [2]", len(got.Used), got.Dropped, cites)
	}
}

func TestCounterErrorIsReturned(t *testing.T) {
	// WHY: a tokenizer that is down must fail the assembly. Treating the
	//      error as zero tokens would pack everything and overflow the
	//      model's context window.
	// KIND: fault
	// CATCHES: s14
	// CHAPTER: ag.08 section 5
	if _, _, err := (assemble.Packer{Counter: failing{}}).Assemble(context.Background(), "q", handHits, 50); err == nil {
		t.Fatal("Assemble ignored the counter's error")
	}
}

func fixtureIndex(t *testing.T) *retrieve.Index {
	t.Helper()
	dir := os.Getenv("TINYLLM_FIXTURES")
	if dir == "" {
		t.Fatal("TINYLLM_FIXTURES is not set (ol check sets it)")
	}
	x, err := retrieve.Open(filepath.Join(dir, "ag.07", "index"))
	if err != nil {
		t.Fatal(err)
	}
	return x
}

func TestCitationsResolveToStoredChunks(t *testing.T) {
	// WHY: the catalog's rule, every citation resolves to a stored chunk:
	//      for real queries over the fixture index, each citation's chunk
	//      exists with the same document and URI. A citation built from the
	//      wrong field, or a made-up one, is reported by Unresolved.
	// KIND: unit
	// CATCHES: s08, s17
	// CHAPTER: ag.08 section 2.3
	x := fixtureIndex(t)
	r := &retrieve.Hybrid{Index: x}
	for _, q := range []string{"kv cache eviction", "reciprocal rank fusion", "gateway rate limits per tenant"} {
		hits, err := r.Retrieve(context.Background(), q, 5, retrieve.Filter{})
		if err != nil {
			t.Fatal(err)
		}
		_, cites, err := assemble.Packer{Counter: &words{}}.Assemble(context.Background(), q, hits, 200)
		if err != nil || len(cites) == 0 {
			t.Fatalf("%q: %d citations, %v", q, len(cites), err)
		}
		if bad := assemble.Unresolved(cites, x.Chunk); len(bad) != 0 {
			t.Fatalf("%q: unresolved citations %+v", q, bad)
		}
	}
	fake := []assemble.Citation{{N: 1, ChunkID: "docs/kv-cache.md#9", DocID: "docs/kv-cache.md", URI: "docs/kv-cache.md"},
		{N: 2, ChunkID: "docs/kv-cache.md#0", DocID: "docs/gateway.md", URI: "docs/kv-cache.md"}}
	if bad := assemble.Unresolved(fake, x.Chunk); len(bad) != 2 {
		t.Fatalf("Unresolved(missing chunk, wrong doc) = %+v, want both", bad)
	}
}

func TestParseCitations(t *testing.T) {
	// WHY: the scorers (ag.10) read citations back out of the model's
	//      answer with this parser. Markers are [n] or a comma list [1, 3];
	//      numbers are reported once, in order of first appearance; text in
	//      brackets that is not a number list is not a citation.
	// KIND: boundary
	// CATCHES: s09, s10
	// CHAPTER: ag.08 section 2.4
	cases := []struct {
		in   string
		want []int
	}{
		{"FNV-1a [1].", []int{1}},
		{"see [2][1] and [2]", []int{2, 1}},
		{"both [1, 3] agree [3,4]", []int{1, 3, 4}},
		{"[12] is two digits", []int{12}},
		{"not [a], not [1-3], not [ ], not [0]", nil},
		{"array[2] counts too", []int{2}},
		{"no citations", nil},
	}
	for _, c := range cases {
		if got := assemble.ParseCitations(c.in); !reflect.DeepEqual(got, c.want) && !(len(got) == 0 && len(c.want) == 0) {
			t.Errorf("ParseCitations(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestResolveFlagsUnknown(t *testing.T) {
	// WHY: a small model invents citation numbers. Resolve separates the
	//      citations an answer really uses from numbers no block carried,
	//      which the eval scores as unsupported claims.
	// KIND: unit
	// CATCHES: s11
	// CHAPTER: ag.08 section 2.4
	cites := []assemble.Citation{{N: 1, ChunkID: "a#0"}, {N: 2, ChunkID: "b#0"}, {N: 3, ChunkID: "c#0"}}
	used, unknown := assemble.Resolve("per [3] and [1], also [7] and [4]", cites)
	if len(used) != 2 || used[0].ChunkID != "c#0" || used[1].ChunkID != "a#0" || !reflect.DeepEqual(unknown, []int{4, 7}) {
		t.Fatalf("Resolve = %+v, %v; want c#0, a#0 used and [4 7] unknown", used, unknown)
	}
}

func registry(t *testing.T, r retrieve.Retriever, l *assemble.Ledger) *tool.Registry {
	t.Helper()
	reg := tool.NewRegistry()
	if err := reg.Register(assemble.SearchDocs(r, &words{}, assemble.SearchOptions{MaxTokens: 200, Ledger: l})); err != nil {
		t.Fatal(err)
	}
	return reg
}

func call(reg *tool.Registry, args string) tool.Result {
	return reg.Call(context.Background(), types.ToolCall{ID: "call_1", Name: assemble.SearchDocsName, Args: json.RawMessage(args)})
}

var blockRE = regexp.MustCompile(`(?m)^\[(\d+)\] (\S+)$`)

func TestSearchDocsTool(t *testing.T) {
	// WHY: search_docs is how retrieval reaches the agent's toolset: its
	//      definition validates through ag.02's registry, a good call
	//      returns numbered blocks, a query matching nothing is the answer
	//      "no results" (not a tool error the model may retry forever), and
	//      bad arguments are rejected before retrieval runs.
	// KIND: unit
	// CATCHES: s12
	// CHAPTER: ag.08 section 4
	x := fixtureIndex(t)
	reg := registry(t, &retrieve.Hybrid{Index: x}, nil)
	defs := reg.Definitions()
	if len(defs) != 1 || defs[0].Name != "search_docs" {
		t.Fatalf("definitions = %+v", defs)
	}
	res := call(reg, `{"query": "kv cache eviction", "k": 3}`)
	if res.IsError || len(blockRE.FindAllString(res.Content, -1)) != 3 {
		t.Fatalf("search_docs = %+v; want 3 numbered blocks", res)
	}
	res = call(reg, `{"query": "zebra quantum"}`)
	if res.IsError || res.Content != assemble.NoResults {
		t.Fatalf("no match = %+v; want the answer %q, not an error", res, assemble.NoResults)
	}
	for _, bad := range []string{`{}`, `{"query": ""}`, `{"query": "x", "k": 0}`, `{"query": "x", "k": 99}`, `{"query": "x", "extra": 1}`} {
		if res := call(reg, bad); !res.IsError {
			t.Fatalf("args %s: %+v; want a tool error", bad, res)
		}
	}
}

// recorder is a retriever that records its filter.
type recorder struct{ f retrieve.Filter }

func (r *recorder) Retrieve(_ context.Context, _ string, k int, f retrieve.Filter) ([]retrieve.Hit, error) {
	r.f = f
	return handHits[:min(k, 3)], nil
}

func TestSearchDocsPassesFilterAndK(t *testing.T) {
	// WHY: doc_ids scopes the search ("only the gateway docs") and k sets
	//      how much comes back; a tool that drops them answers a different
	//      question than the model asked. Without k the default is 5.
	// KIND: unit
	// CATCHES: s13
	// CHAPTER: ag.08 section 4
	r := &recorder{}
	reg := registry(t, r, nil)
	res := call(reg, `{"query": "q", "k": 2, "doc_ids": ["docs/a.md"]}`)
	if res.IsError || !reflect.DeepEqual(r.f.DocIDs, []string{"docs/a.md"}) || strings.Count(res.Content, "\n[") != 1 {
		t.Fatalf("filter %+v, result %q; want doc_ids passed and 2 blocks", r.f, res.Content)
	}
}

func TestLedgerAcrossCalls(t *testing.T) {
	// WHY: an agent often searches twice. With the run's ledger, the
	//      second search keeps the number a chunk got in the first and gives
	//      new chunks the next numbers, so [1] means one chunk for the whole
	//      answer and the ledger lists everything the model was shown.
	// KIND: unit
	// CATCHES: s15
	// CHAPTER: ag.08 section 2.3
	x := fixtureIndex(t)
	l := assemble.NewLedger()
	reg := registry(t, &retrieve.Hybrid{Index: x}, l)
	first := call(reg, `{"query": "kv cache eviction", "k": 2}`)
	second := call(reg, `{"query": "eviction policy LRU cache", "k": 3}`)
	if first.IsError || second.IsError {
		t.Fatalf("%+v %+v", first, second)
	}
	nums := map[string]string{}
	all := l.Citations()
	for i, c := range all {
		if c.N != i+1 {
			t.Fatalf("ledger numbers %+v are not 1..%d", all, len(all))
		}
		nums[c.ChunkID] = fmt.Sprint(c.N)
	}
	for _, res := range []tool.Result{first, second} {
		for _, block := range strings.Split(res.Content, "\n\n") {
			m := regexp.MustCompile(`^\[(\d+)\] `).FindStringSubmatch(block)
			if m == nil {
				t.Fatalf("block %q has no number", block)
			}
			text := strings.SplitN(block, "\n", 2)[1]
			found := false
			for id, n := range nums {
				c, _ := x.Chunk(id)
				if n == m[1] && strings.TrimSuffix(text, "\n") == c.Text {
					found = true
				}
			}
			if !found {
				t.Fatalf("block [%s] does not show the chunk the ledger recorded under %s", m[1], m[1])
			}
		}
	}
	if len(all) >= 5 {
		t.Fatalf("%d citations for 2 + 3 hits: the shared chunk was numbered twice", len(all))
	}
}

func TestLedgerConcurrent(t *testing.T) {
	// WHY: the agent loop runs tool calls in parallel (ag.03). Eight
	//      searches sharing one ledger must still give every chunk one
	//      number, numbers 1..n with no gaps, and each block the number it
	//      is recorded under.
	// KIND: fault
	// CATCHES: s16
	// CHAPTER: ag.08 section 5
	x := fixtureIndex(t)
	l := assemble.NewLedger()
	reg := registry(t, &retrieve.Hybrid{Index: x}, l)
	queries := []string{"kv cache", "gateway", "durable timers", "bpe merges", "sampling temperature", "flash attention", "eval bootstrap", "agent tools"}
	var wg sync.WaitGroup
	out := make([]tool.Result, len(queries))
	for i, q := range queries {
		wg.Add(1)
		go func() {
			defer wg.Done()
			out[i] = call(reg, fmt.Sprintf(`{"query": %q, "k": 4}`, q))
		}()
	}
	wg.Wait()
	all := l.Citations()
	seen := map[string]int{}
	for i, c := range all {
		if c.N != i+1 {
			t.Fatalf("numbers %v have a gap or repeat at %d", all, i)
		}
		if _, dup := seen[c.ChunkID]; dup {
			t.Fatalf("chunk %s numbered twice", c.ChunkID)
		}
		seen[c.ChunkID] = c.N
	}
	for _, res := range out {
		for _, m := range blockRE.FindAllStringSubmatch(res.Content, -1) {
			n := 0
			fmt.Sscan(m[1], &n)
			if n < 1 || n > len(all) || all[n-1].URI != m[2] {
				t.Fatalf("block [%s] %s does not match the ledger", m[1], m[2])
			}
		}
	}
}
