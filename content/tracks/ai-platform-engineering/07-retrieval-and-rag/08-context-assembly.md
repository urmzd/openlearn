# Context assembly, citations, search_docs tool

## Overview

| | |
|---|---|
| **Module** | `ag.08` · build · Go · Pass 10 · reading time 1.5 h |
| **Solution** | `go/agent/rag/assemble/`: [`assemble.go`](../../../../course/solutions/go/agent/rag/assemble/assemble.go) (`Packer`, the token-budgeted context), [`cite.go`](../../../../course/solutions/go/agent/rag/assemble/cite.go) (the run's citation `Ledger`, `ParseCitations`, `Resolve`, `Unresolved`), [`tool.go`](../../../../course/solutions/go/agent/rag/assemble/tool.go) (`SearchDocs`, the `search_docs` tool) |
| **Contract** | citations name `chunk_id`s of [`course/contracts/formats/rag-index.md`](../../../../course/contracts/formats/rag-index.md); the Go API in section 4 |
| **Needs** | `ag.07` retrieval (`Hit`, `Retriever`, `Filter`), `ag.02` tools (`tool.New`, the registry that validates `search_docs` arguments) |
| **Used by** | `ag.10` scorers (the citation scorer parses answers with `ParseCitations` and `Resolve`); the agent's toolset, wired in [`entry/go/cmd/internal/agentkit/agentkit.go`](../../../../course/solutions/entry/go/cmd/internal/agentkit/agentkit.go). With it, the agent answers questions about the course docs with citations that resolve to stored chunks. |
| **Optional depth** | Liu et al., [*Lost in the Middle: How Language Models Use Long Contexts*](https://arxiv.org/abs/2307.03172) (free); Gao et al., [*Enabling Large Language Models to Generate Text with Citations*](https://arxiv.org/abs/2305.14627) (free) |

## Key Takeaways

- A token budget is checked by **counting the whole assembled text**: token counts of pieces do not add up to the count of their concatenation.
- A chunk is **included whole or not at all**; a cut chunk makes its citation point at words the source never said.
- When a chunk does not fit, **skip it and keep going**: a later, shorter chunk may still fit.
- Citation numbers come from **one ledger per run**, so `[1]` means one chunk across every search the agent made, even when searches run in parallel.
- "No results" is an **answer**, not a tool error; every number the model cites is **resolved** against what it was actually shown.

## How to read this chapter

Sections 1 to 3 derive the packing rule and the citation ledger and pack one context by hand. Section 4 fixes the API. Section 5 walks through the packer, the ledger, the citation parser, and the tool, and explains why each choice is the right one. Try to predict each step of the solution before reading it.

In the solution the agent kit ([`agentkit.go`](../../../../course/solutions/entry/go/cmd/internal/agentkit/agentkit.go)) opens the index, builds a `retrieve.Hybrid`, and for every agent run registers a fresh `search_docs` in that run's `tool.Registry`:

```go
	hy := &retrieve.Hybrid{Index: x, Embedder: em}
	reg := tool.NewRegistry()
	reg.MustRegister(assemble.SearchDocs(hy, chunk.Bytes{}, assemble.SearchOptions{MaxTokens: 2048, Ledger: assemble.NewLedger()}))
```

One ledger per run is the point: every search in that run shares the numbering. The system prompt tells the model to cite excerpts as `[n]`.

---

## 1. Why now

`ag.07` returns the best chunks for a query, but a list of `Hit` structs is not something a model can read, and nothing stops an agent from stuffing fifty chunks into a 2048-token SmolLM2 context. Worse, when the agent answers "blocks are named by chained FNV-1a", nobody can tell whether that came from `docs/kv-cache.md` or from the model's imagination. This module turns hits into the text the model reads: numbered excerpts that fit a token budget, each tied to a stored chunk, plus a `search_docs` tool so the agent decides when to search. The numbers are what make the answer checkable: `ag.10` reads them back out and verifies each one.

## 2. Principles

| Symbol | Meaning | Type |
|---|---|---|
| $T(s)$ | tokens the model's tokenizer counts in text $s$ | integer |
| $B$ | the token budget, `maxTokens` | integer |
| $b_n$ | block $n$: `[n] <uri>\n<text>\n` | string |
| $C$ | the context: blocks joined by one blank line | string |

### 2.1 The budget is a property of the whole text

The obvious packing loop counts each block and adds the counts. That assumes $T(a \Vert b) = T(a) + T(b)$, which is false for real tokenizers: BPE merges across the join, and the separator between blocks has tokens of its own. The sum can be lower than the true count, and the packed context then overflows the budget at exactly the boundary the packer was trying to respect. The rule: after tentatively appending a block, count the **whole** candidate context, and keep it only when $T(C) \le B$. Exactly $B$ is allowed. The cost is one count per hit (a handful of `/v1/tokenize` calls for $k \le 20$), cheap next to the model call it protects. `ag.06`'s `TokenChunker` follows the same rule for the same reason.

### 2.2 Whole chunks, in rank order, once

Hits arrive in rank order, and the packer keeps that order. For each hit:

1. **Skip duplicates.** A chunk id already in the context, or a text identical to one already in it (the same paragraph stored under two documents), would spend the budget twice on one fact.
2. **Append tentatively** and count the whole text (2.1).
3. **Over budget: skip and continue.** Stopping at the first misfit throws away every later chunk, including short ones that fit.
4. **Never cut a chunk.** Trimming the last chunk to fill the budget is tempting, but the block then quotes a text the store does not hold, and its citation lies. A block is the stored text, whole.

### 2.3 Citations and the run ledger

Each block is numbered, and a **citation** records what the number means: `{N, ChunkID, DocID, URI}`. Numbers count from 1 in the order blocks were included, so the model never sees a gap.

An agent often searches more than once. If every search numbers from 1, the answer's `[1]` is ambiguous. A **ledger** belongs to one agent run and numbers every chunk the run has been shown: a chunk keeps the number it got first, a new chunk gets the next number. The agent loop (`ag.03`) runs tool calls in parallel, so two searches can assemble at the same time; the packer holds the ledger's lock for its whole assembly, so the number printed in a block is the number recorded for it. (Holding a lock across a tokenizer call serializes searches of one run; correctness first.)

A citation is only as good as its target. `Unresolved` checks each citation against the store (`retrieve.Index.Chunk`): the chunk must exist with the same document and URI. A citation that names a document instead of a chunk, or a chunk that was never stored, is reported.

### 2.4 Reading citations back

The model writes markers in its answer: `[2]`, `[1][3]`, or `[1, 3]`. `ParseCitations` returns the distinct numbers in order of first appearance; a bracket holding anything other than a list of positive integers (`[a]`, `[1-3]`, `[0]`) is not a marker. `Resolve` splits those numbers into citations the answer **uses** and numbers no block carried, which are **unknown**: the model made them up. A small model does this often, and dropping unknown numbers silently would hide it from the eval (`ag.10` scores them as unsupported claims).

### 2.5 The search_docs tool

`SearchDocs` wraps retrieval and assembly as an `ag.02` tool. Its schema requires a non-empty `query`, allows `k` from 1 to 20 (default 5) and a list of `doc_ids`, and rejects anything else, so the registry refuses bad arguments before any retrieval runs. A query that matches nothing returns the text `no results`: that is information the model can act on (rephrase, or answer that the docs do not say). Returning a tool error instead invites the model to retry the same call until the loop's iteration limit.

## 3. Worked example by hand

Counter: whitespace-separated words. Budget $B = 12$. Three hits in rank order:

| Rank | Chunk | URI | Text | Block words |
|---|---|---|---|---|
| 1 | `docs/a.md#0` | `docs/a.md` | `alpha beta gamma delta` | 2 + 4 = 6 |
| 2 | `docs/b.md#0` | `docs/b.md` | `one two three four five six seven eight` | 2 + 8 = 10 |
| 3 | `docs/c.md#0` | `docs/c.md` | `short chunk` | 2 + 2 = 4 |

(`[1] docs/a.md` is two words.)

1. Append `[1] docs/a.md` + text: the whole context counts 6, at most 12. Keep it.
2. Append `[2] docs/b.md` + text: the whole context would count 16 > 12. Skip it; dropped = 1. The number 2 is still free.
3. Append `[2] docs/c.md` + text: the whole context counts 10. Keep it.

The context is

```
[1] docs/a.md
alpha beta gamma delta

[2] docs/c.md
short chunk
```

with Tokens 10, Dropped 1, and citations `{1, docs/a.md#0}`, `{2, docs/c.md#0}`. Stopping at the misfit would have returned only block 1; cutting `docs/b.md` to four words would have quoted a sentence its source never contained.

## 4. The interface

```go
package assemble // import "tinyllm/agent/rag/assemble"

type Counter interface{ Count(ctx context.Context, text string) (int, error) } // ag.06's chunk.Tokenizer
type Citation struct { N int; ChunkID, DocID, URI string }
type Context struct { Text string; Tokens int; Used []retrieve.Hit; Dropped int }
type Assembler interface {
	Assemble(ctx context.Context, q string, hits []retrieve.Hit, maxTokens int) (Context, []Citation, error)
}
func Block(n int, h retrieve.Hit) string // "[n] <uri>\n<text>\n"
type Packer struct { Counter Counter; Ledger *Ledger }
func (p Packer) Assemble(ctx context.Context, q string, hits []retrieve.Hit, maxTokens int) (Context, []Citation, error)

type Ledger struct{ /* unexported */ }
func NewLedger() *Ledger
func (l *Ledger) Peek(h retrieve.Hit) int
func (l *Ledger) Cite(h retrieve.Hit) Citation
func (l *Ledger) Citations() []Citation
func ParseCitations(answer string) []int
func Resolve(answer string, cites []Citation) (used []Citation, unknown []int)
type Lookup func(chunkID string) (retrieve.Chunk, bool) // retrieve.Index.Chunk
func Unresolved(cites []Citation, lookup Lookup) []Citation

const SearchDocsName = "search_docs"
const SearchDocsSchema = `{...}` // query (required, 1 to 1000 chars), k (1 to 20), doc_ids; nothing else
const NoResults = "no results"
type SearchOptions struct { MaxTokens int; Ledger *Ledger } // MaxTokens 0 means 1024
func SearchDocs(r retrieve.Retriever, c Counter, opts SearchOptions) tool.Tool
```

`Peek` and `Cite` take the ledger's lock; the packer uses unexported `peek` and `cite` while it holds the lock for a whole `Assemble`. The solution uses only the standard library and the `ag.02` and `ag.07` packages.

### What a correct solution must handle

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| the hand example | unit | section 3: the exact text, 10 tokens, 1 dropped, citations `[1]` and `[2]` | the reference format |
| the budget is never exceeded | property | budgets 0 to 60 with a counter that is not additive: $T(C) \le B$ and `Tokens` $= T(C)$; exactly $B$ fits | the model's context window never overflows |
| chunks are never cut | property | the text is exactly the cited chunks' stored texts in blocks, at every budget | a citation never quotes words the source lacks |
| dedup | unit | a repeated chunk id and a repeated text are each included once | the budget buys distinct facts |
| a counter error is returned | fault | a failing tokenizer fails the assembly | a tokenizer outage never packs an unbounded context |
| citations resolve to stored chunks | unit | real searches over `ag.07`'s fixture index: every citation resolves; a missing chunk and a wrong document are reported | the catalog's rule: every citation resolves |
| `ParseCitations` | boundary | `[n]`, `[1, 3]`, repeats, `[12]`, `[a]`, `[1-3]`, `[0]` | `ag.10` reads exactly what the model cited |
| `Resolve` flags unknown numbers | unit | used in order of appearance; made-up numbers listed, sorted | unsupported claims are scored, not hidden |
| the `search_docs` tool | unit | the definition, three numbered blocks, `no results` as an answer, five kinds of bad arguments rejected | the tool behaves in the agent loop |
| filter and `k` reach the retriever | unit | `doc_ids` reaches the retriever's filter, `k` the count | a scoped question gets a scoped search |
| the ledger across calls | unit | a second search keeps the first search's number for a shared chunk; numbers are 1..n | `[1]` means one chunk for the whole answer |
| the ledger under concurrency | fault | eight parallel searches on one ledger: one number per chunk, no gaps, each block shows its recorded number (race-free) | parallel tool calls (`ag.03`) stay consistent |

## 5. The solution, step by step

### 5.1 From first principles

1. **The context must fit the model's window, measured by the model's tokenizer.** As in `ag.06`, counts are not additive, so the only reliable check is to count the whole candidate text after each tentative append. With a real BPE tokenizer the separators and merges across joins change the total; in the hand example the plain word counter is enough to show the rule, since block 2 would take the whole context from 6 to 16 > 12.
2. **A citation must point at words the store actually holds.** So a block is a stored chunk, whole; a chunk that does not fit is skipped, never trimmed.
3. **Skipping beats stopping.** Rank order is a preference, not a dependency: a later, shorter chunk that fits is still useful evidence (block `docs/c.md` in the hand example).
4. **Numbers must be unambiguous across one answer.** An answer may draw on several searches, so numbering belongs to the run (a ledger), not to one search. A chunk seen twice keeps its first number.
5. **Parallel searches must not interleave numbering.** The number printed in a block must equal the number recorded; holding the ledger's lock for the whole assembly is the simplest way to guarantee it.
6. **Reading citations back must be strict and loss-free.** Only bracketed lists of positive integers are markers; numbers no block carried are reported as unknown, because a made-up citation is a finding, not noise.
7. **"Nothing found" is information.** The tool answers `no results` rather than failing, so the model can rephrase or say the docs are silent.

### 5.2 The code, annotated

**The packer.** The lock is taken first (when there is a ledger), each hit is deduplicated by id and by text hash, and the whole candidate is counted:

```go
func (p Packer) Assemble(ctx context.Context, q string, hits []retrieve.Hit, maxTokens int) (Context, []Citation, error) {
	if p.Ledger != nil {
		p.Ledger.mu.Lock() // held for the whole assembly: printed number == recorded number
		defer p.Ledger.mu.Unlock()
	}
	var out Context
	var cites []Citation
	var b strings.Builder
	seenID := map[string]bool{}
	seenText := map[[32]byte]bool{}
	for _, h := range hits { // rank order, kept
		fp := fingerprint(h.Text) // SHA-256 of the text: the same paragraph under two docs
		if seenID[h.ChunkID] || seenText[fp] {
			out.Dropped++
			continue
		}
		n := len(cites) + 1 // without a ledger: 1, 2, ... in this call, no gaps
		if p.Ledger != nil {
			n = p.Ledger.peek(h) // the number it would get, not yet recorded
		}
		candidate := b.String()
		if candidate != "" {
			candidate += "\n" // blocks end in "\n"; one more makes the blank line
		}
		candidate += Block(n, h) // "[n] <uri>\n<text>\n"
		tokens, err := p.Counter.Count(ctx, candidate) // the whole text, every time
		if err != nil {
			return Context{}, nil, fmt.Errorf("assemble: count tokens: %w", err) // never "0 tokens"
		}
		if tokens > maxTokens { // exactly maxTokens fits
			out.Dropped++
			continue // skip, do not stop: a later hit may fit
		}
		c := Citation{N: n, ChunkID: h.ChunkID, DocID: h.DocID, URI: h.URI}
		if p.Ledger != nil {
			c = p.Ledger.cite(h) // record only what was actually included
		}
		b.Reset()
		b.WriteString(candidate)
		out.Tokens = tokens // the count of exactly out.Text
		out.Used = append(out.Used, h)
		cites = append(cites, c)
		seenID[h.ChunkID] = true
		seenText[fp] = true
	}
	out.Text = b.String()
	return out, cites, nil
}
```

`peek` before the count and `cite` after it is what keeps numbers gap-free: a skipped hit never takes a number, so in the hand example `docs/c.md` gets 2. Full solution: [`assemble.go`](../../../../course/solutions/go/agent/rag/assemble/assemble.go).

**The ledger.** A map from chunk id to number plus the list of citations in number order; `peek` and `cite` assume the lock is held:

```go
func (l *Ledger) peek(h retrieve.Hit) int {
	if n, ok := l.byID[h.ChunkID]; ok {
		return n // seen in an earlier search of this run: same number
	}
	return len(l.list) + 1 // next free number
}

func (l *Ledger) cite(h retrieve.Hit) Citation {
	// ... lazily create byID
	if n, ok := l.byID[h.ChunkID]; ok {
		return l.list[n-1] // numbers are 1-based positions in list
	}
	c := Citation{N: len(l.list) + 1, ChunkID: h.ChunkID, DocID: h.DocID, URI: h.URI}
	l.byID[h.ChunkID] = c.N
	l.list = append(l.list, c)
	return c
}
```

The exported `Peek`, `Cite`, and `Citations` take the lock themselves, for callers outside a packer.

**Reading citations back.** One regexp recognises a marker: a bracket holding one or more integers separated by commas. `[a]`, `[1-3]`, and `[ ]` do not match; `[0]` matches but is filtered:

```go
var markerRE = regexp.MustCompile(`\[(\d+(?:\s*,\s*\d+)*)\]`)

func ParseCitations(answer string) []int {
	var out []int
	seen := map[int]bool{}
	for _, m := range markerRE.FindAllStringSubmatch(answer, -1) {
		for _, f := range strings.Split(m[1], ",") {
			n, err := strconv.Atoi(strings.TrimSpace(f))
			if err != nil || n < 1 || seen[n] { // numbers count from 1; repeats once
				continue
			}
			seen[n] = true
			out = append(out, n) // first-appearance order
		}
	}
	return out
}

func Resolve(answer string, cites []Citation) (used []Citation, unknown []int) {
	byN := map[int]Citation{}
	for _, c := range cites {
		byN[c.N] = c
	}
	for _, n := range ParseCitations(answer) {
		if c, ok := byN[n]; ok {
			used = append(used, c)
		} else {
			unknown = append(unknown, n) // made up by the model: reported, not dropped
		}
	}
	sort.Ints(unknown)
	return used, unknown
}
```

`Unresolved` checks each citation against the store: the chunk must exist and its `DocID` and `URI` must match, so a citation built from the wrong field is caught. Full solution: [`cite.go`](../../../../course/solutions/go/agent/rag/assemble/cite.go).

**The tool.** The schema does the argument validation (the `ag.02` registry rejects bad arguments before the handler runs); the handler applies defaults, passes `doc_ids` as the filter, and turns an empty context into `no results`:

```go
func searchDocs(ctx context.Context, r retrieve.Retriever, c Counter, opts SearchOptions, args json.RawMessage) (string, error) {
	// ... decode {query, k, doc_ids}
	if in.K == 0 {
		in.K = 5
	}
	budget := opts.MaxTokens
	if budget <= 0 {
		budget = 1024
	}
	hits, err := r.Retrieve(ctx, in.Query, in.K, retrieve.Filter{DocIDs: in.DocIDs}) // scoped inside retrieval
	if err != nil {
		return "", fmt.Errorf("search_docs: retrieve: %w", err) // a real failure is a tool error
	}
	out, _, err := Packer{Counter: c, Ledger: opts.Ledger}.Assemble(ctx, in.Query, hits, budget)
	if err != nil {
		return "", fmt.Errorf("search_docs: %w", err)
	}
	if out.Text == "" {
		return NoResults, nil // an answer the model can act on
	}
	return out.Text, nil
}
```

Full solution: [`tool.go`](../../../../course/solutions/go/agent/rag/assemble/tool.go).

### 5.3 Why this solution

- **Cost.** One tokenizer call per hit, on a text that grows to at most the budget: $O(k)$ calls and $O(k \cdot B)$ tokenized characters, with $k \le 20$. That is small next to the model call the context feeds. Dedup is $O(k)$ hashes.
- **Counting the whole over caching per-block counts.** Per-block counts would save calls but are wrong at the boundary that matters; the extra calls are bounded by $k$.
- **Skip-and-continue over greedy stop or knapsack.** Stopping wastes budget; an optimal knapsack over relevance would reorder evidence and cost more for little gain at $k \le 20$. Skipping keeps rank order and fills the budget well.
- **One lock per assembly over finer locking.** Finer locks would let two searches interleave peeks and cites and print a number that another search then records for a different chunk. The cost is that one run's searches serialize through the tokenizer, which is acceptable; different runs have different ledgers.
- **Accepted tradeoffs.** No reordering against "lost in the middle"; no compression of long chunks; `Unresolved` checks existence, not that the cited text supports the claim (`ag.10` and Going further).

### 5.4 Explain it back

1. Why does the packer call `peek` before counting and `cite` only after the block fits?
   <details><summary>Answer</summary>The block text must contain its number before it can be counted, but a block that does not fit must not consume a number. `peek` gives the number without recording it; `cite` records it only for included blocks, so numbers stay 1..n with no gaps (in the hand example `docs/c.md` is `[2]`, not `[3]`).</details>
2. Why is a context of exactly $B$ tokens accepted?
   <details><summary>Answer</summary>The budget is a maximum the context may reach. The check is `tokens > maxTokens` to refuse; using `>=` would waste the last token of every budget and fail the case where a block fits exactly.</details>
3. What would go wrong if the ledger's lock were taken only inside `peek` and `cite`?
   <details><summary>Answer</summary>Two parallel searches could both peek number 5 for different chunks, both print `[5]` in their blocks, and then one cite records 5 and the other records 6. The model would then see `[5]` on two chunks while the ledger maps 5 to only one of them.</details>
4. Why does `Resolve` return unknown numbers instead of silently dropping them?
   <details><summary>Answer</summary>A number no block carried is a citation the model invented. Dropping it makes the answer look less cited rather than wrong; reporting it lets `ag.10` count it as an unsupported claim.</details>
5. Why is "no results" a successful tool answer rather than an error?
   <details><summary>Answer</summary>An error invites the agent loop or the model to retry the same call; an answer tells the model something true about the docs, so it can rephrase or say the docs do not cover the question.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. adding per-block token counts | the context overflows the budget by the separators' tokens | `Assemble` counts the whole candidate text each time |
| 2. cutting the last chunk to fill the budget | the block quotes a sentence the source never contained | a block is `Block(n, h)` with the stored text, or nothing |
| 3. stopping at the first chunk that does not fit | short, relevant chunks after a long one are lost | an over-budget hit is skipped with `continue` |
| 4. `>=` instead of `>` against the budget | a context of exactly $B$ tokens is refused | the refusal test is `tokens > maxTokens` |
| 5. no deduplication by id or by text | the same paragraph fills the budget twice | `seenID` and `seenText` (SHA-256 of the text) |
| 6. numbers from 0, or from 1 in every search | `[1]` names two chunks in one answer | numbers start at `len(cites) + 1` or come from the run's `Ledger` |
| 7. a citation built from the wrong field, or resolved by id only | citations that resolve to nothing, or to another document | `Citation` takes `ChunkID`, `DocID`, `URI` from the hit; `Unresolved` checks all three |
| 8. a parser that misses `[1, 3]` or reports repeats | citation recall understated, counts inflated | `markerRE` accepts comma lists; `seen` keeps each number once |
| 9. dropping unknown numbers silently | made-up citations look like no citations | `Resolve` returns them in `unknown` |
| 10. "no results" as a tool error | the model retries the same search until the iteration limit | `searchDocs` returns `NoResults, nil` |
| 11. ignoring `doc_ids` | a scoped question gets answers from other documents | `doc_ids` becomes `retrieve.Filter{DocIDs: ...}` |
| 12. treating a counter error as zero tokens | a tokenizer outage packs every hit into the prompt | a `Count` error fails the assembly |
| 13. not holding the ledger lock for the whole assembly | parallel searches print one number and record another, or crash on a map write | `Assemble` locks the ledger before the loop and unlocks at return |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `ag.07` | `Hybrid.Retrieve` produces the hits; `Index.Chunk` resolves citations |
| Back | `ag.02` | `tool.New` builds `search_docs`; the registry validates its arguments against `SearchDocsSchema` |
| Forward | `ag.10` | the citation scorer runs `ParseCitations` and `Resolve` over each answer and scores precision and recall against the ground-truth chunks |

The agent's composition root registers `search_docs` next to `ag.04`'s `query_usage` (see `NewRun` in [`agentkit.go`](../../../../course/solutions/entry/go/cmd/internal/agentkit/agentkit.go)), which is what makes the agent's cited answers checkable end to end.

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Packer` | saige `rag/contextassembler/compressing` | compresses or summarizes chunks to fit more evidence, reorders to fight "lost in the middle" | [saige](https://github.com/urmzd/saige) (free) |
| `Ledger` + `[n]` markers | Anthropic citations, OpenAI file search annotations | citations as structured spans of the answer instead of markers in text | provider API docs (free) |
| `Unresolved` | RAG faithfulness checkers (RAGAS, saige `rag/eval`) | check that each cited claim is entailed by the cited text, not only that the chunk exists | [RAGAS](https://docs.ragas.io/) (free) |
