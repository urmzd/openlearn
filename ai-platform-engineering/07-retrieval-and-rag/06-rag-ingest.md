<!-- ol:module ag.06 -->
# RAG ingest: crawler with SSRF allowlist, chunker, embedder, store

## Overview

| | |
|---|---|
| **Module** | `ag.06` · build · Go · Pass 10 · 5 h |
| **You build** | `go/agent/rag/source/source.go` (`RawDoc`, `Doc`, `ToDoc`, `DocID`, `DirSource`), `go/agent/rag/source/crawler.go` (`Crawler`, `PublicAddr`), `go/agent/rag/chunk/chunk.go` (`TokenChunker`, `Bytes`, `HTTPTokenizer`, `Fingerprint`), `go/agent/rag/embed/embed.go` (`HTTPEmbedder`), `go/agent/rag/store/store.go` (`Open`, `Index.Upsert`, `Index.Save`, `Ingest`) |
| **Contract** | the index directory of [`course/contracts/formats/rag-index.md`](../../course/contracts/formats/rag-index.md) (`meta.json`, `docs.jsonl`, `vectors.f32`); `/v1/tokenize` and `/v1/embeddings` of [`openai-subset.v1.yaml`](../../course/contracts/openapi/openai-subset.v1.yaml) |
| **Tests** | `course/tests/go/ag_06/` (what they check: section 4) |
| **Needs** | [`ag.01` provider](../13-agent-sdk/01-types-and-provider.md) (the embedder retries with `Retryable` and `RetryPolicy`) · reading: [`ag.02` tools](../13-agent-sdk/02-tools-and-schemas.md), the engine's tokenizer (`L1.5`), the crawler warm-up in [practice Go 01](../../practice/build/cloud/go/) |
| **Used by** | `ag.07` retrieval reads the index this module writes |
| **Milestone** | MS-agent |
| **Optional depth** | OWASP, [Server-Side Request Forgery Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html) (free); [RFC 9309, Robots Exclusion Protocol](https://www.rfc-editor.org/rfc/rfc9309) (free) |

## Key Takeaways

- Chunks are packed to a **token budget counted by the embedding model's own tokenizer**, and every candidate chunk is counted whole: token counts do not add up across a join (`TestHandExample`, `TestChunkBudgetNeverExceeded`).
- An over-long paragraph is cut at the **largest unit that fits**: sentences, then words, then characters, and never inside a UTF-8 character (`TestSplitOrder`, `TestHardSplitUTF8`).
- Ingest is **idempotent by fingerprint**: an unchanged document embeds nothing and writes nothing; a changed one re-embeds only its new chunks (`TestIngestDedupNoOp`).
- A crawler fetches URLs that documents name, so it is an **SSRF** risk: hosts are allowlisted, every resolved address is checked at dial time (no loopback, private, link-local, ...), and a redirect is checked like a new request (`TestSSRF`, `TestPublicAddrTable`).
- It is a polite crawler: **robots.txt** per RFC 9309 and a **per-host rate**, measured on a fake clock (`TestRobots`, `TestCrawlerRateLimitPerHost`).

## How to work this chapter

```bash
ol start ag.06
ol tests ag.06
ol check ag.06          # ag.01 smoke tests run first
ol diff  ag.06
```

---

## 1. Why now

The agent of `ag.03` knows only what the model memorized. MS-agent asks it to answer questions about the course and your own docs, with citations, which means retrieval: find the passages that answer the question and put them in the context. Retrieval (`ag.07`) searches an index; this module builds that index. Each step has a way to quietly ruin the answers: chunks over the embedding model's context are truncated by the engine and lose their ends, a crawler that follows any link fetches `http://169.254.169.254/latest/meta-data/` (cloud credentials) because a document linked it, a nightly re-ingest that re-embeds everything costs hours, and vectors stored unnormalized make cosine similarity wrong.

## 2. Principles

### 2.1 Sources and text

A **source** returns raw documents: `DirSource` reads files under a directory (`.md` and `.txt` by default, in path order), the **crawler** fetches web pages. `ToDoc` turns a raw document into text: HTML loses `<script>`, `<style>`, and `<head>` content and every tag, block tags (`<p>`, `<li>`, `<h1>`, ...) become paragraph breaks, entities are decoded after tags are removed, spaces collapse, and paragraphs are separated by exactly one blank line. The doc id is the URL without its fragment (scheme and host lowercased) or the file path.

The crawler goes breadth first from its seeds, follows `<a href>` links resolved against the page's final URL, drops `#fragments` (so `/a` and `/a#top` are one page), and visits each URL once, up to `MaxPages` and `MaxDepth`. Two politeness rules:

- **robots.txt** (RFC 9309), fetched once per host: the group naming our user agent applies, else the `*` group; within it the **longest** matching rule wins and `Allow` wins a tie; `*` matches any run and a trailing `$` anchors. A 4xx robots.txt means no restrictions; a 5xx or no answer means the site cannot say, so nothing is fetched.
- **rate**: requests to one host (robots.txt included) are at least $1/\text{RPS}$ apart. Each host has its own next free slot; the crawler sleeps only until that host's slot, so two hosts do not wait for each other. Time comes from a `Clock`, which tests replace with one that advances only on `Sleep`.

### 2.2 Chunking to a token budget

| Symbol | Meaning |
|---|---|
| $B$ | the budget, `MaxTokens` |
| $T(x)$ | the token count of text $x$ by the embedding model's tokenizer |
| $p_1, \dots, p_k$ | the pieces of a document, each with $T(p_i) \le B$ |
| $s_i$ | the separator before $p_i$: `"\n\n"` between paragraphs, `" "` inside one |

Pieces: each paragraph, or, when $T(\text{paragraph}) > B$, its sentences, or a sentence's words, or a word's longest prefixes that fit (cut at a character boundary). Packing is greedy: keep a current chunk $c$; for each piece, if $T(c \,\Vert\, s_i \,\Vert\, p_i) \le B$, extend $c$; otherwise emit $c$ and start a new one with $p_i$.

The count is taken on the **joined candidate**, never as $T(c) + T(p_i)$. A BPE tokenizer can merge across the join (the sum overcounts), and a tokenizer that spends tokens on separators or adds a marker per text undercounts (the sum lets chunks overflow). Only the real tokenizer on the real text is right: `HTTPTokenizer` asks the engine (`/v1/tokenize`, the length of `ids`); `Bytes` is the tracer's byte-level tokenizer. The budget is inclusive.

A chunk's id is `<doc_id>#<k>`, $k$ its position in the document from 0, and its fingerprint is the hex SHA-256 of its text.

### 2.3 Embedding

`HTTPEmbedder` posts batches (64 inputs by default; the contract allows 256) to `/v1/embeddings` and places each vector by its `index` field: a server may answer in any order. An answer with the wrong count, a repeated or out-of-range index, or vectors of different lengths is an error, never a misaligned index. Overload (503, 429) and broken connections are retried with the provider's policy from `ag.01`.

### 2.4 The index and fingerprint dedup

`rag-index.md` fixes the files: `docs.jsonl` (one chunk per line, line $i$ is chunk $i$), `vectors.f32` ($n \times d$ little-endian float32, no header), `meta.json` (`n_chunks`, `dim`, the models, the chunker and BM25 settings). Every row is L2-normalized when stored, so retrieval's cosine similarity is a dot product. `bm25.idx` is written by retrieval (`ag.07`) from the same chunks.

`Ingest` chunks each document and compares the chunk fingerprints with the stored ones for that doc id. All equal: the document is **unchanged**, nothing is embedded and the index is not marked dirty. Otherwise the chunks whose fingerprint the document already had keep their stored vectors, only new ones are embedded, and the document's chunks are replaced in place. `Save` writes nothing when nothing changed, and otherwise writes each file to a temporary name, fsyncs, and renames, so a reader never sees half a file.

### 2.5 Server-side request forgery

A crawler is a program that makes HTTP requests to addresses chosen by its input. If the input is a document an attacker wrote, the attacker chooses where your server connects: an admin port on localhost, a database on the private network, the cloud metadata service at `169.254.169.254`. Three checks, in this order:

1. **Host allowlist**, on the URL's host name before any request: exact names only (a prefix match lets `docs.example.evil.example` in).
2. **Address check at dial time**: the crawler resolves the name itself, refuses if **any** address is not public, and dials the address it checked. Checking a name and letting the HTTP library resolve it again opens a race (DNS rebinding: the second answer is `127.0.0.1`). Not public: loopback, private (`10/8`, `172.16/12`, `192.168/16`, `fc00::/7`), link-local (`169.254/16`, `fe80::/10`), unspecified, multicast, carrier-grade NAT (`100.64/10`), NAT64 (`64:ff9b::/96`, which embeds an IPv4 address), and `0/8`. IPv4-mapped IPv6 (`::ffff:10.0.0.1`) is checked as the IPv4 address it is.
3. **Redirects** are new requests: the allowlist is checked again for each target (at most 5), and the target's connection goes through the same dial check.

## 3. Worked example by hand

Document `docs/kv.md`, the byte tokenizer ($T(x)$ = bytes of $x$), budget $B = 60$:

```
Paged attention stores KV in blocks.                                  (36 bytes)

Blocks are fixed size. A block table maps token positions to blocks.  (68 bytes)

Eviction frees whole blocks.                                          (28 bytes)
```

Paragraph 2 alone is $68 > 60$, so it becomes its two sentences, 22 and 45 bytes. Pieces and packing:

| Piece | Separator | Candidate | $T$ | Fits? | Chunk so far |
|---|---|---|---|---|---|
| P1 (36) | | P1 | 36 | yes | P1 |
| S1 (22) | `\n\n` | P1 + `\n\n` + S1 | 60 | yes (inclusive) | P1, S1 |
| S2 (45) | `" "` | ... + `" "` + S2 | 106 | no: emit chunk 0 (60) | S2 |
| P3 (28) | `\n\n` | S2 + `\n\n` + P3 | 75 | no: emit chunk 1 (45) | P3 |
| end | | | | emit chunk 2 (28) | |

Three chunks: `docs/kv.md#0` (60 tokens, paragraph 1 and the first sentence), `#1` (45), `#2` (28), each with the SHA-256 of its text as fingerprint. This is `TestHandExample`.

Re-ingesting the unchanged file gives the same three fingerprints: nothing is embedded, `Save` writes nothing. Changing the last paragraph to "Eviction frees whole blocks at once." changes only chunk 2's fingerprint: one text is embedded (`TestIngestDedupNoOp`).

## 4. The interface

```go
package source // import "tinyllm/agent/rag/source"
type RawDoc struct { URI, Source, ContentType string; Body []byte }
type Doc struct { ID, URI, Source, Title, Text string }
type Source interface{ Fetch(ctx context.Context) ([]RawDoc, error) }
func DocID(r RawDoc) string
func ToDoc(r RawDoc) Doc
type DirSource struct { Root string; Exts []string }
type CrawlerConfig struct { Seeds, AllowHosts []string; MaxPages, MaxDepth int; RPS float64; UserAgent string; MaxBytes int64
	Clock Clock; Resolver func(context.Context, string) ([]netip.Addr, error); AddrAllowed func(netip.Addr) bool }
func NewCrawler(cfg CrawlerConfig) *Crawler // Fetch(ctx); Refused() []Refusal
func PublicAddr(a netip.Addr) bool

package chunk // import "tinyllm/agent/rag/chunk"
type Tokenizer interface{ Count(ctx context.Context, text string) (int, error) }
type Bytes struct{}
type HTTPTokenizer struct { BaseURL, Model string; Client *http.Client }
type Chunk struct { ID, DocID string; Index int; Text string; Tokens int; Fingerprint string }
type TokenChunker struct { Tok Tokenizer; MaxTokens int } // Chunk(ctx, source.Doc) ([]Chunk, error)
func Fingerprint(text string) string

package embed // import "tinyllm/agent/rag/embed"
type Embedder interface{ Embed(ctx context.Context, texts []string) ([][]float32, error) }
type HTTPEmbedder struct { BaseURL, APIKey, Model string; BatchSize int; Client *http.Client; Retry provider.RetryPolicy }

package store // import "tinyllm/agent/rag/store"
func Open(dir string, m Meta) (*Index, error)
func (x *Index) Entries() []Entry
func (x *Index) Upsert(docID string, es []Entry) error
func (x *Index) Save(now time.Time) (bool, error)
func Ingest(ctx context.Context, src source.Source, ch chunk.Chunker, em embed.Embedder, x *Index) (Stats, error)
```

The catalog writes `Chunker.Chunk(d Doc) []Chunk`; here it takes a `ctx` and returns an error, because the tokenizer is a network call (DEVIATIONS B121-05). Use only the standard library and `tinyllm/agent/provider`.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestHandExample` | unit | section 3: three chunks of 60, 45, 28 tokens with ids and fingerprints | you and the tests agree on packing |
| `TestSplitOrder` | unit | sentences before words before characters | chunks end at sentence boundaries when they can |
| `TestChunkBudgetNeverExceeded` | property | 40 documents, a separator-charging tokenizer: every chunk within budget, no word lost or repeated | the engine never truncates a chunk |
| `TestHardSplitUTF8` | boundary | a 60-byte word of `é` at budget 7: 6-byte valid pieces | no invalid UTF-8 reaches the tokenizer |
| `TestHTTPTokenizer` | conformance | the count is the length of `/v1/tokenize`'s ids | the budget is the embedding model's |
| `TestEmbedderBatchesAndOrder` | unit | batches of 2, vectors placed by index; wrong count and mixed dimensions are errors | rows of `vectors.f32` match lines of `docs.jsonl` |
| `TestEmbedderRetry` | fault | a 503 is retried after `Initial` | ingest survives a busy engine |
| `TestIngestDedupNoOp` | unit | unchanged: nothing embedded or written; one changed chunk: one embedding | nightly re-ingest is cheap |
| `TestIndexFormat` | conformance | `meta.json`, `docs.jsonl` ids and fingerprints, `vectors.f32` size and unit rows, reload | `ag.07` and `ag.08` read these files |
| `TestToDoc` | unit | script and style dropped, block breaks, entities, titles, doc ids | embeddings of text, not markup |
| `TestCrawlerFetches` | unit | breadth-first order, fragments, allowlist (lookalike included), robots, refusal reasons | the practice crawler, as a source |
| `TestCrawlerRateLimitPerHost` | unit | at RPS 2, 500 ms between requests to one host; two hosts finish at 2 s of fake time | polite, without serializing hosts |
| `TestRobots` | unit | the UA group, longest match, `*` and `$`, 404 and 503 | RFC 9309 |
| `TestSSRF` | fault | private, link-local, loopback, IP literals, mapped addresses, a name with one private address, redirects | a document cannot aim your server |
| `TestPublicAddrTable` | unit | the address table, including mapped, NAT64, CGNAT | the forms a string check misses |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. counts summed instead of counting the candidate, an exclusive budget, words before sentences, bytes instead of tokens | chunks over budget, truncated by the engine; chunks cut mid-sentence | `TestHandExample`, `TestChunkBudgetNeverExceeded`, `TestSplitOrder`, `TestHTTPTokenizer` (mutants `s01`, `s02`, `s04`, `s26`) |
| 2. long words cut at a byte offset | invalid UTF-8 rejected or mangled by the tokenizer | `TestHardSplitUTF8` (mutant `s03`) |
| 3. chunk ids from 1, fingerprints of a normalized text | citations point at the wrong chunk; dedup compares the wrong hash | `TestHandExample`, `TestIndexFormat` (mutants `s05`, `s06`) |
| 4. unchanged documents re-embedded, every chunk of a changed one re-embedded, unchanged indexes rewritten | the nightly job re-pays for the whole corpus | `TestIngestDedupNoOp` (mutants `s07`, `s08`, `s09`) |
| 5. vectors stored unnormalized | dot products are not cosines; long texts win every search | `TestIndexFormat` (mutant `s10`) |
| 6. embeddings placed by response order, no retry | vectors attached to the wrong chunks; ingest dies on one 503 | `TestEmbedderBatchesAndOrder`, `TestEmbedderRetry` (mutants `s11`, `s13`) |
| 7. link-local allowed, mapped addresses not unmapped, only the first address checked, redirects off the allowlist followed | the metadata service, an intranet host, or localhost fetched on a document's say-so | `TestSSRF`, `TestPublicAddrTable` (mutants `s14`, `s15`, `s16`, `s17`) |
| 8. the allowlist matched by prefix, fragments kept | lookalike hosts crawled; the same page fetched twice | `TestCrawlerFetches` (mutants `s18`, `s24`) |
| 9. one rate slot for every host, or slots booked without waiting | a slow crawl, or a site hammered | `TestCrawlerRateLimitPerHost` (mutants `s19`, `s20`) |
| 10. a 5xx robots.txt read as "allow all", the first rule winning, the UA group ignored | pages a site excluded are fetched | `TestRobots` (mutants `s21`, `s22`, `s23`) |
| 11. script and style left in the text | embeddings of tracking code | `TestToDoc` (mutant `s25`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `ag.01` | `provider.Retryable` and `RetryPolicy.Delay` for the embedder |
| Forward | `ag.07` | reads `docs.jsonl` and `vectors.f32`, writes `bm25.idx`, and embeds queries with the same model |

`ag.02` is reading: the RAG tools (`search_docs`, `ag.08`) plug into the registry, ingest does not. `L1.5` is the tokenizer behind the engine's `/v1/tokenize`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `TokenChunker` | saige semantic chunker | boundaries where the embedding of consecutive sentences changes, not only at punctuation | saige `rag/chunker/semantic` |
| `Ingest` | saige ingest pipeline | parallel stages, retries per document, deletion of documents gone from the source | saige `rag/internal/pipeline/pipeline.go` |
| `Index` | saige `pgstore`, pgvector | an index in Postgres with transactions and filters | saige `rag/pgstore`; [pgvector](https://github.com/pgvector/pgvector) (free) |
| SSRF guard | Smokescreen | an egress proxy that applies the same address rules to every service | [stripe/smokescreen](https://github.com/stripe/smokescreen) (free) |
| `Crawler` | Colly, Scrapy | concurrency per domain, caching, sitemaps | [gocolly/colly](https://github.com/gocolly/colly) (free) |
