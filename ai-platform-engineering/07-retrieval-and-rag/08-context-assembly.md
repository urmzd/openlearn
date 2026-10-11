<!-- ol:module ag.08 -->
# Context assembly, citations, search_docs tool

## Overview

| | |
|---|---|
| **Module** | `ag.08` · build · Go · Pass 10 · 3 to 4 h |
| **You build** | `go/agent/rag/assemble/`: `assemble.go` (`Packer`, the token-budgeted context), `cite.go` (the run's citation `Ledger`, `ParseCitations`, `Resolve`, `Unresolved`), `tool.go` (`SearchDocs`, the `search_docs` tool) |
| **Contract** | citations name `chunk_id`s of [`course/contracts/formats/rag-index.md`](../../course/contracts/formats/rag-index.md); the Go API in section 4 |
| **Tests** | `course/tests/go/ag_08/` (what they check: section 4) |
| **Needs** | `ag.07` retrieval (`Hit`, `Retriever`, `Filter`), `ag.02` tools (`tool.New`, the registry that validates `search_docs` arguments) |
| **Used by** | `ag.10` scorers (the citation scorer parses answers with `ParseCitations` and `Resolve`); your agent's toolset, wired in your entry point |
| **Milestone** | MS-agent |
| **Optional depth** | Liu et al., [*Lost in the Middle: How Language Models Use Long Contexts*](https://arxiv.org/abs/2307.03172) (free); Gao et al., [*Enabling Large Language Models to Generate Text with Citations*](https://arxiv.org/abs/2305.14627) (free) |

## Key Takeaways

- A token budget is checked by **counting the whole assembled text**: token counts of pieces do not add up to the count of their concatenation.
- A chunk is **included whole or not at all**; a cut chunk makes its citation point at words the source never said.
- When a chunk does not fit, **skip it and keep going**: a later, shorter chunk may still fit.
- Citation numbers come from **one ledger per run**, so `[1]` means one chunk across every search the agent made, even when searches run in parallel.
- "No results" is an **answer**, not a tool error; every number the model cites is **resolved** against what it was actually shown.

## How to work this chapter

```bash
ol start ag.08          # stubs go/agent/rag/assemble/*.go into your repo
ol tests ag.08          # read the test catalog first
ol check ag.08          # exit code is the verdict
ol check ag.08 --ref-deps   # only if you skipped ag.07 or ag.02
ol diff  ag.08          # after passing: your code against the reference
```

Then register the tool in your agent's composition root (learner territory): open the index, build a `retrieve.Hybrid` with your `ag.06` embedder, create a `Ledger` per agent run, and register `SearchDocs(hybrid, tokenizer, SearchOptions{MaxTokens: 1024, Ledger: ledger})` in the run's `tool.Registry`. Tell the model in the system prompt to cite excerpts as `[n]`. MS-agent asks your agent questions about the course docs and checks that its answers cite chunks that exist.

---

## 1. Why now

`ag.07` returns the best chunks for a query, but a list of `Hit` structs is not something a model can read, and nothing stops your agent from stuffing fifty chunks into a 2048-token SmolLM2 context. Worse, when the agent answers "blocks are named by chained FNV-1a", nobody can tell whether that came from `docs/kv-cache.md` or from the model's imagination. This module turns hits into the text the model reads: numbered excerpts that fit a token budget, each tied to a stored chunk, plus a `search_docs` tool so the agent decides when to search. The numbers are what make the answer checkable: `ag.10` reads them back out and verifies each one.

## 2. Principles

| Symbol | Meaning | Type |
|---|---|---|
| $T(s)$ | tokens the model's tokenizer counts in text $s$ | integer |
| $B$ | the token budget, `maxTokens` | integer |
| $b_n$ | block $n$: `[n] <uri>\n<text>\n` | string |
| $C$ | the context: blocks joined by one blank line | string |

### 2.1 The budget is a property of the whole text

The obvious packing loop counts each block and adds the counts. That assumes $T(a \Vert b) = T(a) + T(b)$, which is false for real tokenizers: BPE merges across the join, and the separator between blocks has tokens of its own. The sum can be lower than the true count, and the packed context then overflows the budget at exactly the boundary you were trying to respect. The rule: after tentatively appending a block, count the **whole** candidate context, and keep it only when $T(C) \le B$. Exactly $B$ is allowed. The cost is one count per hit (a handful of `/v1/tokenize` calls for $k \le 20$), cheap next to the model call it protects. `ag.06`'s `TokenChunker` follows the same rule for the same reason.

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

with Tokens 10, Dropped 1, and citations `{1, docs/a.md#0}`, `{2, docs/c.md#0}`. Stopping at the misfit would have returned only block 1; cutting `docs/b.md` to four words would have quoted a sentence its source never contained. This is `TestHandExample`.

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

`Peek` and `Cite` take the ledger's lock; the packer uses unexported `peek` and `cite` while it holds the lock for a whole `Assemble`. Use only the standard library and the `ag.02` and `ag.07` packages.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestHandExample` | unit | section 3: the exact text, 10 tokens, 1 dropped, citations `[1]` and `[2]` | you and the tests agree on the format |
| `TestBudgetNeverExceeded` | property | budgets 0 to 60 with a counter that is not additive: $T(C) \le B$ and `Tokens` $= T(C)$; exactly $B$ fits | the model's context window never overflows |
| `TestChunksNeverCut` | property | the text is exactly the cited chunks' stored texts in blocks, at every budget | a citation never quotes words the source lacks |
| `TestDedup` | unit | a repeated chunk id and a repeated text are each included once | the budget buys distinct facts |
| `TestCounterErrorIsReturned` | fault | a failing tokenizer fails the assembly | a tokenizer outage never packs an unbounded context |
| `TestCitationsResolveToStoredChunks` | unit | real searches over `ag.07`'s fixture index: every citation resolves; a missing chunk and a wrong document are reported | the catalog's rule: every citation resolves |
| `TestParseCitations` | boundary | `[n]`, `[1, 3]`, repeats, `[12]`, `[a]`, `[1-3]`, `[0]` | `ag.10` reads exactly what the model cited |
| `TestResolveFlagsUnknown` | unit | used in order of appearance; made-up numbers listed, sorted | unsupported claims are scored, not hidden |
| `TestSearchDocsTool` | unit | the definition, three numbered blocks, `no results` as an answer, five kinds of bad arguments rejected | the tool behaves in the agent loop |
| `TestSearchDocsPassesFilterAndK` | unit | `doc_ids` reaches the retriever's filter, `k` the count | a scoped question gets a scoped search |
| `TestLedgerAcrossCalls` | unit | a second search keeps the first search's number for a shared chunk; numbers are 1..n | `[1]` means one chunk for the whole answer |
| `TestLedgerConcurrent` | fault | eight parallel searches on one ledger: one number per chunk, no gaps, each block shows its recorded number (run under `-race`) | parallel tool calls (`ag.03`) stay consistent |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. adding per-block token counts | the context overflows the budget by the separators' tokens | `TestBudgetNeverExceeded` (mutant `s01`) |
| 2. cutting the last chunk to fill the budget | the block quotes a sentence the source never contained | `TestChunksNeverCut`, `TestHandExample` (mutant `s02`) |
| 3. stopping at the first chunk that does not fit | short, relevant chunks after a long one are lost | `TestHandExample` (mutant `s03`) |
| 4. `>=` instead of `>` against the budget | a context of exactly $B$ tokens is refused | `TestBudgetNeverExceeded` (mutant `s04`) |
| 5. no deduplication by id or by text | the same paragraph fills the budget twice | `TestDedup` (mutants `s05`, `s06`) |
| 6. numbers from 0, or from 1 in every search | `[1]` names two chunks in one answer | `TestHandExample`, `TestLedgerAcrossCalls` (mutants `s07`, `s15`) |
| 7. a citation built from the wrong field, or resolved by id only | citations that resolve to nothing, or to another document | `TestCitationsResolveToStoredChunks` (mutants `s08`, `s17`) |
| 8. a parser that misses `[1, 3]` or reports repeats | citation recall understated, counts inflated | `TestParseCitations` (mutants `s09`, `s10`) |
| 9. dropping unknown numbers silently | made-up citations look like no citations | `TestResolveFlagsUnknown` (mutant `s11`) |
| 10. "no results" as a tool error | the model retries the same search until the iteration limit | `TestSearchDocsTool` (mutant `s12`) |
| 11. ignoring `doc_ids` | a scoped question gets answers from other documents | `TestSearchDocsPassesFilterAndK` (mutant `s13`) |
| 12. treating a counter error as zero tokens | a tokenizer outage packs every hit into the prompt | `TestCounterErrorIsReturned` (mutant `s14`) |
| 13. not holding the ledger lock for the whole assembly | parallel searches print one number and record another, or crash on a map write | `TestLedgerConcurrent` (mutant `s16`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `ag.07` | `Hybrid.Retrieve` produces the hits; `Index.Chunk` resolves citations |
| Back | `ag.02` | `tool.New` builds `search_docs`; the registry validates its arguments against `SearchDocsSchema` |
| Forward | `ag.10` | the citation scorer runs `ParseCitations` and `Resolve` over each answer and scores precision and recall against the ground-truth chunks |

Your agent's composition root registers `search_docs` next to `ag.04`'s `query_usage`; MS-agent checks the cited answers end to end. If you skip this module, `ol check ag.10` reports `needs ag.08: build it, or pass --ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Packer` | saige `rag/contextassembler/compressing` | compresses or summarizes chunks to fit more evidence, reorders to fight "lost in the middle" | [saige](https://github.com/urmzd/saige) (free) |
| `Ledger` + `[n]` markers | Anthropic citations, OpenAI file search annotations | citations as structured spans of the answer instead of markers in text | provider API docs (free) |
| `Unresolved` | RAG faithfulness checkers (RAGAS, saige `rag/eval`) | check that each cited claim is entailed by the cited text, not only that the chunk exists | [RAGAS](https://docs.ragas.io/) (free) |
