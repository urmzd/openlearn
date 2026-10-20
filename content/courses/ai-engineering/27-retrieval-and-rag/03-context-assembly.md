# Context Assembly, Citations, and the search_docs Tool

[Retrieval](02-retrieval.md) returns the best chunks for a query, but a list of `Hit` structs is not something a model can read, and nothing stops an agent from stuffing fifty chunks into a 2048-token SmolLM2 context. Worse, when the agent answers "blocks are named by chained FNV-1a", nobody can tell whether that came from `docs/kv-cache.md` or from the model's imagination.

This section closes the loop. It turns hits into the text the model reads: numbered excerpts that fit a token budget, each tied to a stored chunk. It adds a `search_docs` tool so the agent decides for itself when to search. And it reads the numbers back out of the answer, which is what makes the answer checkable: the evaluation scorers later in the book verify every citation the model writes.

This section builds on [Retrieval: BM25, Flat and IVF Vectors, RRF, and MMR](02-retrieval.md) (`Hit`, `Retriever`, `Filter`) and on [Tools, a registry, and JSON Schema argument validation](../28-agents/02-tools-and-schemas.md) (`tool.New`, and the registry that validates `search_docs` arguments). The full solution lives in `go/agent/rag/assemble/`: [`assemble.go`](../../../../course/solutions/go/agent/rag/assemble/assemble.go) (`Packer`, the token-budgeted context), [`cite.go`](../../../../course/solutions/go/agent/rag/assemble/cite.go) (the run's citation `Ledger`, `ParseCitations`, `Resolve`, `Unresolved`), and [`tool.go`](../../../../course/solutions/go/agent/rag/assemble/tool.go) (`SearchDocs`, the `search_docs` tool). Citations name `chunk_id`s of [`course/contracts/formats/rag-index.md`](../../../../course/contracts/formats/rag-index.md), and the Go API is given later in this section.

In the solution, the agent kit ([`agentkit.go`](../../../../course/solutions/entry/go/cmd/internal/agentkit/agentkit.go)) opens the index, builds a `retrieve.Hybrid`, and for every agent run registers a fresh `search_docs` in that run's `tool.Registry`:

```go
	hy := &retrieve.Hybrid{Index: x, Embedder: em}
	reg := tool.NewRegistry()
	reg.MustRegister(assemble.SearchDocs(hy, chunk.Bytes{}, assemble.SearchOptions{MaxTokens: 2048, Ledger: assemble.NewLedger()}))
```

One ledger per run is the point: every search in that run shares the numbering. The system prompt tells the model to cite excerpts as `[n]`.

> **Learning objectives**
> - Pack ranked hits into a context under a token budget by counting the whole assembled text, never summing per-block counts.
> - Explain why chunks are included whole or not at all, and why a misfit is skipped rather than ending the packing.
> - Number citations with one ledger per run so that `[n]` is unambiguous across parallel searches.
> - Parse citation markers out of an answer and separate resolved citations from invented ones.
> - Design a retrieval tool whose "no results" is an answer, not an error.

## The budget is a property of the whole text

The notation is small:

| Symbol | Meaning | Type |
|---|---|---|
| $T(s)$ | tokens the model's tokenizer counts in text $s$ | integer |
| $B$ | the token budget, `maxTokens` | integer |
| $b_n$ | block $n$: `[n] <uri>\n<text>\n` | string |
| $C$ | the context: blocks joined by one blank line | string |

The obvious packing loop counts each block and adds the counts. That assumes $T(a \Vert b) = T(a) + T(b)$, which is false for real tokenizers: BPE merges across the join, and the separator between blocks has tokens of its own. The sum can be lower than the true count, and the packed context then overflows the budget at exactly the boundary the packer was trying to respect. So the rule is: after tentatively appending a block, count the **whole** candidate context, and keep it only when $T(C) \le B$. Exactly $B$ is allowed. The cost is one count per hit (a handful of `/v1/tokenize` calls for $k \le 20$), cheap next to the model call it protects. The `TokenChunker` in [RAG ingest](01-rag-ingest.md) follows the same rule for the same reason.

## Whole chunks, in rank order, once

Knowing how to check the budget, the next question is what to do with each hit. Hits arrive in rank order, and the packer keeps that order. For each hit it does four things:

1. **Skip duplicates.** A chunk id already in the context, or a text identical to one already in it (the same paragraph stored under two documents), would spend the budget twice on one fact.
2. **Append tentatively** and count the whole text.
3. **Over budget: skip and continue.** Stopping at the first misfit throws away every later chunk, including short ones that fit.
4. **Never cut a chunk.** Trimming the last chunk to fill the budget is tempting, but the block then quotes a text the store does not hold, and its citation lies. A block is the stored text, whole.

## Citations and the run ledger

Each block is numbered, and a **citation** records what the number means: `{N, ChunkID, DocID, URI}`. Numbers count from 1 in the order blocks were included, so the model never sees a gap.

An agent often searches more than once. If every search numbers from 1, the answer's `[1]` is ambiguous. A **ledger** belongs to one agent run and numbers every chunk the run has been shown: a chunk keeps the number it got first, and a new chunk gets the next number. The [agent loop](../28-agents/03-agent-loop.md) runs tool calls in parallel, so two searches can assemble at the same time. The packer therefore holds the ledger's lock for its whole assembly, so the number printed in a block is the number recorded for it. Holding a lock across a tokenizer call serializes the searches of one run; correctness comes first.

A citation is only as good as its target. `Unresolved` checks each citation against the store (`retrieve.Index.Chunk`): the chunk must exist with the same document and URI. A citation that names a document instead of a chunk, or a chunk that was never stored, is reported.

## Reading citations back

The model writes markers in its answer: `[2]`, `[1][3]`, or `[1, 3]`. `ParseCitations` returns the distinct numbers in order of first appearance; a bracket holding anything other than a list of positive integers (`[a]`, `[1-3]`, `[0]`) is not a marker. `Resolve` then splits those numbers into citations the answer **uses** and numbers no block carried, which are **unknown**: the model made them up. A small model does this often, and dropping unknown numbers silently would hide it from evaluation, where the [scorers](../29-evaluation/02-scorers.md) count them as unsupported claims.

## The search_docs tool

`SearchDocs` wraps retrieval and assembly as a tool. Its schema requires a non-empty `query`, allows `k` from 1 to 20 (default 5) and a list of `doc_ids`, and rejects anything else, so the registry refuses bad arguments before any retrieval runs. A query that matches nothing returns the text `no results`. That is information the model can act on: it can rephrase, or answer that the docs do not say. Returning a tool error instead invites the model to retry the same call until the loop's iteration limit.

## Working it by hand

Use a counter that counts whitespace-separated words, a budget of $B = 12$, and three hits in rank order:

| Rank | Chunk | URI | Text | Block words |
|---|---|---|---|---|
| 1 | `docs/a.md#0` | `docs/a.md` | `alpha beta gamma delta` | 2 + 4 = 6 |
| 2 | `docs/b.md#0` | `docs/b.md` | `one two three four five six seven eight` | 2 + 8 = 10 |
| 3 | `docs/c.md#0` | `docs/c.md` | `short chunk` | 2 + 2 = 4 |

The header line `[1] docs/a.md` is two words. Now pack:

1. Append `[1] docs/a.md` and its text: the whole context counts 6, at most 12. Keep it.
2. Append `[2] docs/b.md` and its text: the whole context would count 16 > 12. Skip it; dropped = 1. The number 2 is still free.
3. Append `[2] docs/c.md` and its text: the whole context counts 10. Keep it.

The context is

```
[1] docs/a.md
alpha beta gamma delta

[2] docs/c.md
short chunk
```

with Tokens 10, Dropped 1, and citations `{1, docs/a.md#0}` and `{2, docs/c.md#0}`. Stopping at the misfit would have returned only block 1; cutting `docs/b.md` to four words would have quoted a sentence its source never contained. With a real BPE tokenizer, separators and merges across joins would also change the total; the word counter is enough to show the rule, since block 2 would take the whole context from 6 to 16.

## Designing the API

The package exposes the packer, the ledger and citation helpers, and the tool:

```go
package assemble // import "tinyllm/agent/rag/assemble"

type Counter interface{ Count(ctx context.Context, text string) (int, error) } // the ingest chunk.Tokenizer
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

`Peek` and `Cite` take the ledger's lock; the packer uses the unexported `peek` and `cite` while it holds the lock for a whole `Assemble`. The solution uses only the standard library and the tool and retrieval packages.

## Building it step by step

Seven observations shape the code:

1. **The context must fit the model's window, measured by the model's tokenizer.** Counts are not additive, so the only reliable check is to count the whole candidate text after each tentative append.
2. **A citation must point at words the store actually holds.** So a block is a stored chunk, whole; a chunk that does not fit is skipped, never trimmed.
3. **Skipping beats stopping.** Rank order is a preference, not a dependency: a later, shorter chunk that fits is still useful evidence, like `docs/c.md` in the hand example.
4. **Numbers must be unambiguous across one answer.** An answer may draw on several searches, so numbering belongs to the run (a ledger), not to one search. A chunk seen twice keeps its first number.
5. **Parallel searches must not interleave numbering.** The number printed in a block must equal the number recorded; holding the ledger's lock for the whole assembly is the simplest way to guarantee it.
6. **Reading citations back must be strict and loss-free.** Only bracketed lists of positive integers are markers, and numbers no block carried are reported as unknown, because a made-up citation is a finding, not noise.
7. **"Nothing found" is information.** The tool answers `no results` rather than failing, so the model can rephrase or say the docs are silent.

### The packer

The lock is taken first (when there is a ledger), each hit is deduplicated by id and by text hash, and the whole candidate is counted:

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

Calling `peek` before the count and `cite` after it is what keeps numbers gap-free: a skipped hit never takes a number, so in the hand example `docs/c.md` gets 2. The full code is in [`assemble.go`](../../../../course/solutions/go/agent/rag/assemble/assemble.go).

### The ledger

The ledger is a map from chunk id to number plus the list of citations in number order. `peek` and `cite` assume the lock is held:

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

### Reading citations back

One regexp recognizes a marker: a bracket holding one or more integers separated by commas. `[a]`, `[1-3]`, and `[ ]` do not match; `[0]` matches but is filtered out:

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

`Unresolved` checks each citation against the store: the chunk must exist and its `DocID` and `URI` must match, so a citation built from the wrong field is caught. The full code is in [`cite.go`](../../../../course/solutions/go/agent/rag/assemble/cite.go).

### The tool

The schema does the argument validation, since the registry rejects bad arguments before the handler runs. The handler applies defaults, passes `doc_ids` as the filter, and turns an empty context into `no results`:

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

The full code is in [`tool.go`](../../../../course/solutions/go/agent/rag/assemble/tool.go).

### Why this design

The cost is one tokenizer call per hit, on a text that grows to at most the budget: $O(k)$ calls and $O(k \cdot B)$ tokenized characters, with $k \le 20$. That is small next to the model call the context feeds, and dedup is $O(k)$ hashes.

Counting the whole text beats caching per-block counts. Per-block counts would save calls but are wrong at the boundary that matters, and the extra calls are bounded by $k$. Skip-and-continue beats both a greedy stop and a knapsack: stopping wastes budget, while an optimal knapsack over relevance would reorder evidence and cost more for little gain at $k \le 20$; skipping keeps rank order and fills the budget well. One lock per assembly beats finer locking, because finer locks would let two searches interleave peeks and cites and print a number that another search then records for a different chunk. The cost is that one run's searches serialize through the tokenizer, which is acceptable since different runs have different ledgers.

Some tradeoffs are accepted on purpose. There is no reordering against "lost in the middle", no compression of long chunks, and `Unresolved` checks that the cited chunk exists, not that its text supports the claim; evaluation and the production tools below go further.

## What can go wrong

A correct solution must pass these checks:

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| the hand example | unit | the exact text, 10 tokens, 1 dropped, citations `[1]` and `[2]` | the reference format |
| the budget is never exceeded | property | budgets 0 to 60 with a counter that is not additive: $T(C) \le B$ and `Tokens` $= T(C)$; exactly $B$ fits | the model's context window never overflows |
| chunks are never cut | property | the text is exactly the cited chunks' stored texts in blocks, at every budget | a citation never quotes words the source lacks |
| dedup | unit | a repeated chunk id and a repeated text are each included once | the budget buys distinct facts |
| a counter error is returned | fault | a failing tokenizer fails the assembly | a tokenizer outage never packs an unbounded context |
| citations resolve to stored chunks | unit | real searches over the retrieval fixture index: every citation resolves; a missing chunk and a wrong document are reported | the catalog's rule: every citation resolves |
| `ParseCitations` | boundary | `[n]`, `[1, 3]`, repeats, `[12]`, `[a]`, `[1-3]`, `[0]` | the scorers read exactly what the model cited |
| `Resolve` flags unknown numbers | unit | used in order of appearance; made-up numbers listed, sorted | unsupported claims are scored, not hidden |
| the `search_docs` tool | unit | the definition, three numbered blocks, `no results` as an answer, five kinds of bad arguments rejected | the tool behaves in the agent loop |
| filter and `k` reach the retriever | unit | `doc_ids` reaches the retriever's filter, `k` the count | a scoped question gets a scoped search |
| the ledger across calls | unit | a second search keeps the first search's number for a shared chunk; numbers are 1..n | `[1]` means one chunk for the whole answer |
| the ledger under concurrency | fault | eight parallel searches on one ledger: one number per chunk, no gaps, each block shows its recorded number (race-free) | parallel tool calls in the agent loop stay consistent |

Most bugs in this code come from one of a few wrong instincts. Adding per-block token counts lets the context overflow the budget by the separators' tokens; `Assemble` counts the whole candidate text each time. Cutting the last chunk to fill the budget makes the block quote a sentence the source never contained; a block is `Block(n, h)` with the stored text, or nothing. Stopping at the first chunk that does not fit loses short, relevant chunks after a long one; an over-budget hit is skipped with `continue`. Comparing with `>=` instead of `>` refuses a context of exactly $B$ tokens; the refusal test is `tokens > maxTokens`. And without deduplication by id or by text, the same paragraph fills the budget twice; the packer keeps `seenID` and `seenText` (SHA-256 of the text).

Numbering has its own traps. Numbers from 0, or from 1 in every search, make `[1]` name two chunks in one answer; numbers start at `len(cites) + 1` or come from the run's `Ledger`. A citation built from the wrong field, or resolved by id only, resolves to nothing or to another document; `Citation` takes `ChunkID`, `DocID`, and `URI` from the hit, and `Unresolved` checks all three. A parser that misses `[1, 3]` or reports repeats understates citation recall and inflates counts; `markerRE` accepts comma lists and `seen` keeps each number once. Dropping unknown numbers silently makes made-up citations look like no citations; `Resolve` returns them in `unknown`.

> **Warning:** Not holding the ledger lock for the whole assembly lets parallel searches print one number and record another, or crash on a concurrent map write. `Assemble` locks the ledger before the loop and unlocks at return.

The tool layer has three more. Treating "no results" as a tool error makes the model retry the same search until the iteration limit; `searchDocs` returns `NoResults, nil`. Ignoring `doc_ids` lets a scoped question get answers from other documents; `doc_ids` becomes `retrieve.Filter{DocIDs: ...}`. And treating a counter error as zero tokens means a tokenizer outage packs every hit into the prompt; a `Count` error fails the assembly.

## Where this leads

This section depends on [Retrieval](02-retrieval.md), whose `Hybrid.Retrieve` produces the hits and whose `Index.Chunk` resolves citations, and on [Tools, a registry, and JSON Schema argument validation](../28-agents/02-tools-and-schemas.md), whose `tool.New` builds `search_docs` and whose registry validates its arguments against `SearchDocsSchema`. Looking forward, the citation scorer in [Agent evaluation scorers](../29-evaluation/02-scorers.md) runs `ParseCitations` and `Resolve` over each answer and scores precision and recall against the ground-truth chunks. The agent's composition root registers `search_docs` next to the `query_usage` tool from [The tool gate, the SQL safety gate, and the prompt-injection suite](../28-agents/04-tool-gate-and-injection.md) (see `NewRun` in [`agentkit.go`](../../../../course/solutions/entry/go/cmd/internal/agentkit/agentkit.go)), which is what makes the agent's cited answers about the course docs checkable end to end.

Production systems extend each piece:

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Packer` | saige `rag/contextassembler/compressing` | compresses or summarizes chunks to fit more evidence, reorders to fight "lost in the middle" | [saige](https://github.com/urmzd/saige) (free) |
| `Ledger` + `[n]` markers | Anthropic citations, OpenAI file search annotations | citations as structured spans of the answer instead of markers in text | provider API docs (free) |
| `Unresolved` | RAG faithfulness checkers (RAGAS, saige `rag/eval`) | check that each cited claim is entailed by the cited text, not only that the chunk exists | [RAGAS](https://docs.ragas.io/) (free) |

### Further reading

- Liu et al., [*Lost in the Middle: How Language Models Use Long Contexts*](https://arxiv.org/abs/2307.03172) (free).
- Gao et al., [*Enabling Large Language Models to Generate Text with Citations*](https://arxiv.org/abs/2305.14627) (free).

## Key terms

- **Token budget**: the maximum token count of the assembled context, checked on the whole text and inclusive.
- **Block**: one numbered excerpt in the context, `[n] <uri>\n<text>\n`, holding a stored chunk whole.
- **Citation**: the record `{N, ChunkID, DocID, URI}` saying which stored chunk a number names.
- **Ledger**: the per-run numbering of every chunk the agent has been shown, shared by all its searches.
- **Unknown citation**: a number in the answer that no block carried, meaning the model invented it.
- **Unresolved citation**: a citation whose chunk is missing from the store or whose document or URI does not match.

## Summary

- A token budget is checked by counting the whole assembled text, because token counts of pieces do not add up to the count of their concatenation.
- A chunk is included whole or not at all; a cut chunk makes its citation point at words the source never said.
- When a chunk does not fit, skip it and keep going, since a later, shorter chunk may still fit.
- Citation numbers come from one ledger per run, so `[1]` means one chunk across every search the agent made, even when searches run in parallel.
- "No results" is an answer, not a tool error, and every number the model cites is resolved against what it was actually shown.

## Review questions

1. Why does the packer call `peek` before counting and `cite` only after the block fits?
   <details><summary>Answer</summary>The block text must contain its number before it can be counted, but a block that does not fit must not consume a number. `peek` gives the number without recording it; `cite` records it only for included blocks, so numbers stay 1..n with no gaps (in the hand example `docs/c.md` is `[2]`, not `[3]`).</details>
2. Why is a context of exactly $B$ tokens accepted?
   <details><summary>Answer</summary>The budget is a maximum the context may reach. The check is `tokens > maxTokens` to refuse; using `>=` would waste the last token of every budget and fail the case where a block fits exactly.</details>
3. What would go wrong if the ledger's lock were taken only inside `peek` and `cite`?
   <details><summary>Answer</summary>Two parallel searches could both peek number 5 for different chunks, both print `[5]` in their blocks, and then one cite records 5 and the other records 6. The model would then see `[5]` on two chunks while the ledger maps 5 to only one of them.</details>
4. Why does `Resolve` return unknown numbers instead of silently dropping them?
   <details><summary>Answer</summary>A number no block carried is a citation the model invented. Dropping it makes the answer look less cited rather than wrong; reporting it lets the citation scorer count it as an unsupported claim.</details>
5. Why is "no results" a successful tool answer rather than an error?
   <details><summary>Answer</summary>An error invites the agent loop or the model to retry the same call; an answer tells the model something true about the docs, so it can rephrase or say the docs do not cover the question.</details>
