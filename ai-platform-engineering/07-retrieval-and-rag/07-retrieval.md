<!-- ol:module ag.07 -->
# Retrieval: BM25, flat and IVF vectors, RRF, MMR

## Overview

| | |
|---|---|
| **Module** | `ag.07` · build · Go · Pass 10 · 5 to 7 h |
| **You build** | `go/agent/rag/retrieve/`: `bm25.go` (terms, inverted index, BM25, the `bm25.idx` file), `vector.go` (flat search, k-means++, IVF), `fuse.go` (RRF, MMR), `retrieve.go` (open an index directory, the `Hybrid` retriever) |
| **Contract** | the index directory of [`course/contracts/formats/rag-index.md`](../../course/contracts/formats/rag-index.md); the Go API in section 4 |
| **Tests** | `course/tests/go/ag_07/` (what they check: section 4) |
| **Needs** | `ag.06` RAG ingest (it writes the index you read and owns the query embedder), `load.01` (your Go PCG32 in `go/ds/rng` seeds k-means++); reading: `M03.6` cosine similarity and top-k (re-implemented here in Go), [infrastructure/04 search and indexing](../../infrastructure/04-search-and-indexing/) (`bm25_from_scratch.py`), [case study 04](../../case-studies/04-kmeans-optimization/) (k-means) |
| **Used by** | `ag.08` context assembly (its `search_docs` tool calls `Retrieve`) |
| **Milestone** | MS-agent |
| **Optional depth** | Robertson and Zaragoza, [*The Probabilistic Relevance Framework: BM25 and Beyond*](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) (free); Cormack, Clarke, Buettcher, [*Reciprocal Rank Fusion*](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) (free); Arthur and Vassilvitskii, [*k-means++*](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf) (free); Carbonell and Goldstein, *The Use of MMR* (SIGIR 1998) |

## Key Takeaways

- **BM25** scores a chunk by the query terms it holds, with term frequency that **saturates** (`k1`) and a penalty for **long chunks** (`b`); the `+1` inside the idf keeps every weight positive.
- An index file is a contract between two programs: the **term rule**, the **byte layout**, and the **varint** postings must match exactly, or ingest and retrieval disagree silently.
- **IVF** trades a little recall for speed by scanning only the `nprobe` lists nearest the query; seeded **k-means++** makes those lists the same on every load.
- **RRF** fuses lexical and dense lists by **rank alone**, so scores on different scales never need calibrating; **MMR** then trades relevance for **diversity**.
- Filters run **inside** each search, before the cut to `k`; a filter after the cut returns fewer results than exist.

## How to work this chapter

```bash
ol start ag.07          # stubs go/agent/rag/retrieve/*.go into your repo
ol tests ag.07          # read the test catalog first
ol check ag.07          # exit code is the verdict
ol check ag.07 --ref-deps   # only if you skipped load.01 (the Go PCG32)
ol diff  ag.07          # after passing: your code against the reference
```

Then add the `rag search` verb to your umbrella CLI (learner territory): open `/artifacts/rag/<index_id>/`, build IVF when the index is large, embed the query through the same model as the index (`meta.json` `embedding_model`), and print the hits. MS-agent runs `<system> rag search "kv cache eviction"` against the course docs you ingested in `ag.06`.

---

## 1. Why now

After `ag.06`, your system has an index of the course docs and your own: chunks with token counts, fingerprints, and embeddings from your engine's `/v1/embeddings`. Nothing reads it. Ask your agent "which hash names a KV block?" and it answers from SmolLM2's weights, which have never seen your system, so it guesses. The agent needs the three chunks that answer the question, picked from hundreds in a few milliseconds. This module is that step: given a query, return the best `k` chunks. It has two halves that fail in opposite ways. Lexical search finds `tl_kv_export` and `30080` exactly but misses a paraphrase; dense search finds "how is a cache entry dropped" for "eviction" but blurs identifiers. You build both, fuse them, and make the result diverse enough that three chunks are not three copies of one sentence.

## 2. Principles

| Symbol | Meaning | Type |
|---|---|---|
| $N$ | number of chunks in the index | integer |
| $t$ | a term (section 2.1) | string |
| $\mathrm{df}(t)$ | number of chunks containing $t$ | integer, $0 \le \mathrm{df} \le N$ |
| $\mathrm{tf}(t, d)$ | occurrences of $t$ in chunk $d$ | integer |
| $\mathrm{dl}(d)$ | terms in chunk $d$, duplicates counted | integer |
| $\overline{\mathrm{dl}}$ | mean of $\mathrm{dl}$ over the index (`avgdl`) | float64 |
| $k_1, b$ | BM25 knobs: saturation (1.2) and length normalization (0.75), from `meta.json` | float64 |
| $q, x_i$ | query vector and chunk vector, both L2-normalized | `[]float32` of length `dim` |
| $c_j$ | IVF centroid $j$ | `[]float32` |
| $D(i)$ | squared distance from row $i$ to its nearest chosen seed | float64 |
| $r_\ell(d)$ | rank of chunk $d$ in list $\ell$, counting from 1 | integer |
| $k_{\mathrm{rrf}}$ | RRF constant (60) | float64 |
| $\lambda$ | MMR trade-off between relevance and novelty | float64 in $(0, 1]$ |

### 2.1 Terms

Retrieval and ingest must cut text into the same pieces. The contract's rule: a **term** is a maximal run of Unicode letters, digits, and `_` (`[\p{L}\p{N}_]+` in Go's RE2), lowercased with `strings.ToLower`. Everything else separates terms, so `tl_kv_export(handle)` gives `tl_kv_export` and `handle`, `k-means++` gives `k` and `means`, and `Café` gives `café`. An ASCII-only rule (`[a-z0-9]+`) would cut `café` into `caf`, and a query typed with the accent would never find it.

### 2.2 The inverted index and BM25

An **inverted index** maps each term to its **postings list**: the chunks containing it, in chunk order, each with $\mathrm{tf}$. Scoring a query only walks the lists of its terms, never the whole index.

BM25 adds, for every query term occurrence $t$ and every chunk $d$ in its list,

$$\mathrm{idf}(t) \cdot \frac{\mathrm{tf}(t,d)\,(k_1+1)}{\mathrm{tf}(t,d) + k_1\left(1 - b + b\,\frac{\mathrm{dl}(d)}{\overline{\mathrm{dl}}}\right)}, \qquad \mathrm{idf}(t) = \ln\!\left(1 + \frac{N - \mathrm{df}(t) + 0.5}{\mathrm{df}(t) + 0.5}\right).$$

Read the fraction from the inside. With $b = 0$ it is $\mathrm{tf}(k_1+1)/(\mathrm{tf}+k_1)$: 1 at $\mathrm{tf} = 1$, then rising towards the ceiling $k_1 + 1 = 2.2$, so the tenth "cache" adds far less than the first (**saturation**: a chunk cannot win by repeating a word). The factor $1 - b + b\,\mathrm{dl}/\overline{\mathrm{dl}}$ is 1 for a chunk of average length, larger for a long one: a long chunk holds more words by chance, so each match counts for less. The idf is large for a rare term and near 0 for a term in every chunk. Without the `1 +` inside the logarithm (the textbook Robertson-Sparck Jones form), a term in more than half the chunks gets a **negative** weight, and matching it pushes a chunk down. A query term written twice counts twice (`"cache cache"` doubles every "cache" contribution), as in Lucene and `bm25_from_scratch.py`.

**The file.** `bm25.idx` stores the index so a restart does not rebuild it: magic `TLBM`, version, $N$, number of terms, $\overline{\mathrm{dl}}$ as f64 bits, every $\mathrm{dl}$, then each term (sorted by its bytes) with $\mathrm{df}$ and its postings. Postings are stored as **gaps**: the first entry is the chunk number, each next one the difference from the previous, so the numbers stay small. Each number is an **unsigned LEB128 varint**: 7 bits per byte, lowest group first, the top bit set on every byte except the last. 300 is binary `100101100`; its low seven bits `0101100` go first with the top bit set (`0xac`), then the remaining `10` (`0x02`): `ac 02`.

### 2.3 Dense vectors: flat search

`vectors.f32` holds one embedding per chunk, L2-normalized, so cosine similarity is just the dot product $q \cdot x_i$ (`M03.6`). **Flat** search scores every row and keeps the top $k$, ties to the lower row. It is exact, and it is the yardstick for anything faster.

### 2.4 IVF and k-means++

An **inverted file** (IVF) index groups rows into `nlist` clusters around centroids $c_j$ and stores each row in the **list** of its nearest centroid. A query computes its distance to every centroid, scans only the `nprobe` nearest lists exactly, and returns the top $k$ among them. With 32 lists and `nprobe = 8` it scores about a quarter of the rows. **Recall@k** measures what that costs: the fraction of the exact top $k$ that the approximate search also returned.

The centroids come from **k-means** (worked through in case study 04): repeat "assign each row to its nearest centroid, move each centroid to the mean of its rows" (**Lloyd** rounds). The result depends on where you start, so the start is **k-means++** seeding:

1. The first seed is a uniform row: `r.Below(n)`.
2. For every row $i$, $D(i)$ is the squared distance to the nearest seed so far.
3. The next seed is row $i$ with probability $D(i) / \sum_j D(j)$: draw $u$ = `r.Float64()` $\cdot \sum_j D(j)$ and take the first row (in row order) whose running sum of $D$ exceeds $u$.
4. Repeat 2 and 3 until there are `nlist` seeds.

Rows far from every seed are likely picks, so seeds spread out, and rows already chosen have $D = 0$ and are never picked twice. Using the distance instead of its square, or always taking the farthest row, gives different (and for outliers, worse) seeds. The index stores no centroids: `rag-index.md` says every load builds them with PCG32 seeded by `fnv1a64(index_id)`, so the same index always gets the same lists, on any machine and in any language that follows the spec.

### 2.5 Fusion and diversity

BM25 scores are unbounded; dot products lie in $[-1, 1]$. Adding them needs a calibration that changes with every query. **Reciprocal rank fusion** uses ranks only:

$$\mathrm{RRF}(d) = \sum_{\ell} \frac{1}{k_{\mathrm{rrf}} + r_\ell(d)},$$

summed over the lists that contain $d$. With $k_{\mathrm{rrf}} = 60$, being first in one list is worth $1/61$, and being in both lists beats being first in one. Ranks count from 1; counting from 0 changes every value.

The fused top $k$ often holds near-duplicates: the same paragraph in two docs, or two adjacent chunks. **Maximal marginal relevance** picks one chunk at a time, each time the one that maximizes

$$\lambda \,(q \cdot x_c) - (1 - \lambda) \max_{p \in \text{picked}} (x_c \cdot x_p),$$

with the max over an empty set equal to 0, so the first pick is the most relevant. $\lambda = 1$ is pure relevance; smaller $\lambda$ buys diversity. The penalty is the **max** over everything already picked: a candidate that copies the first pick is still a copy after the second.

### 2.6 Filters

A filter (`DocIDs`, `Sources`) restricts which chunks may be returned. It must run inside each search, before the cut to the candidate count. Filtering after the cut is the classic bug: the best 50 chunks overall may hold none from the requested document, and the caller gets an empty list although matching chunks exist.

## 3. Worked example by hand

**BM25** with $k_1 = 1.2$, $b = 0.75$ over three chunks:

| Chunk | Text | Terms | $\mathrm{dl}$ |
|---|---|---|---|
| 0 | `kv cache` | kv, cache | 2 |
| 1 | `kv cache eviction policy` | kv, cache, eviction, policy | 4 |
| 2 | `gateway rate limits` | gateway, rate, limits | 3 |

$N = 3$, $\overline{\mathrm{dl}} = 9/3 = 3$. Query `cache eviction`.

- $\mathrm{df}(\text{cache}) = 2$: $\mathrm{idf} = \ln(1 + 1.5/2.5) = \ln 1.6 = 0.470004$.
- $\mathrm{df}(\text{eviction}) = 1$: $\mathrm{idf} = \ln(1 + 2.5/1.5) = \ln(8/3) = 0.980829$.
- Chunk 0, $\mathrm{dl} = 2$: length factor $1 - 0.75 + 0.75 \cdot 2/3 = 0.75$; tf part $= 2.2 / (1 + 1.2 \cdot 0.75) = 2.2/1.9 = 1.157895$. Score $= 0.470004 \cdot 1.157895 = 0.544215$.
- Chunk 1, $\mathrm{dl} = 4$: length factor $0.25 + 0.75 \cdot 4/3 = 1.25$; tf part $= 2.2 / (1 + 1.5) = 0.88$. Score $= 0.470004 \cdot 0.88 + 0.980829 \cdot 0.88 = 0.413603 + 0.863130 = 1.276733$.
- Chunk 2 holds neither term: score 0, and it is not returned at all.

The short chunk earns more per "cache" (0.544 against 0.414), but the long one wins by also holding "eviction". `Search` returns chunk 1, then chunk 0. These are the numbers of `TestHandExampleBM25`.

**RRF**, $k_{\mathrm{rrf}} = 60$: the lexical list is [x, y, z], the dense list [z, x].

| Chunk | Lexical rank | Dense rank | RRF |
|---|---|---|---|
| x | 1 | 2 | $1/61 + 1/62 = 0.032522$ |
| z | 3 | 1 | $1/63 + 1/61 = 0.032266$ |
| y | 2 | none | $1/62 = 0.016129$ |

Fused order: x, z, y (`TestRRFHandExample`).

**MMR**, $\lambda = 0.5$, query $q = (0.8, 0.6)$, candidates $c_0 = (1, 0)$, $c_1 = (0.96, 0.28)$, $c_2 = (0, 1)$. Relevances: $q \cdot c_0 = 0.8$, $q \cdot c_1 = 0.936$, $q \cdot c_2 = 0.6$; similarities $c_0 \cdot c_1 = 0.96$, $c_0 \cdot c_2 = 0$, $c_1 \cdot c_2 = 0.28$.

| Step | $c_0$ | $c_1$ | $c_2$ | Pick |
|---|---|---|---|---|
| 1 | $0.5 \cdot 0.8 - 0 = 0.4$ | $0.5 \cdot 0.936 - 0 = 0.468$ | $0.3$ | $c_1$ (0.468) |
| 2 | $0.4 - 0.5 \cdot 0.96 = -0.08$ | picked | $0.3 - 0.5 \cdot 0.28 = 0.16$ | $c_2$ (0.16) |
| 3 | $0.4 - 0.5 \cdot \max(0.96, 0) = -0.08$ | picked | picked | $c_0$ (-0.08) |

Pure relevance would order $c_1, c_0, c_2$; MMR moves the near-copy $c_0$ to last (`TestMMRHandExample`).

## 4. The interface

```go
package retrieve // import "tinyllm/agent/rag/retrieve"

// bm25.go
func Terms(text string) []string
type Posting struct{ Doc, TF int }
type BM25 struct {
	K1, B    float64
	DocLen   []int
	AvgDL    float64
	Postings map[string][]Posting // sorted by Doc
}
func BuildBM25(texts []string, k1, b float64) *BM25
func (x *BM25) N() int
func (x *BM25) IDF(term string) float64
func (x *BM25) Scores(q string) []float64
func (x *BM25) Search(q string, k int, keep func(doc int) bool) []Scored // matching docs only
func (x *BM25) WriteTo(w io.Writer) (int64, error)
func ReadBM25(r io.Reader, k1, b float64) (*BM25, error) // errors wrap ErrFormat
func PutUvarint(buf []byte, v uint64) []byte
func Uvarint(buf []byte) (v uint64, n int) // n == 0: truncated or overflow
type Scored struct { Doc int; Score float64 }
func TopK(c []Scored, k int) []Scored // score desc, ties to the lower Doc

// vector.go
func Dot(a, b []float32) float64
func Dist2(a, b []float32) float64
func Normalize(v []float32) []float32
type Flat struct{ Vecs [][]float32 }
func (f *Flat) Search(q []float32, k int, keep func(doc int) bool) []Scored
func KMeansPP(vecs [][]float32, k int, r *rng.PCG32) ([]int, error) // ErrK
func Nearest(v []float32, centroids [][]float32) int
type IVF struct { Vecs, Centroids [][]float32; Lists [][]int }
func BuildIVF(vecs [][]float32, nlist, iters int, r *rng.PCG32) (*IVF, error)
func (x *IVF) Search(q []float32, k, nprobe int, keep func(doc int) bool) []Scored
func Recall(approx, exact []Scored) float64

// fuse.go
const DefaultRRFK = 60
func RRF(lists [][]Hit, k float64, n int) []Hit
func MMR(q []float32, cands []Hit, vec func(index int) []float32, lambda float64, k int) []Hit

// retrieve.go
type Chunk struct { ChunkID, DocID, Source, URI, Text string; NTokens int; Fingerprint string } // docs.jsonl
type Meta struct { IndexID string; NChunks, Dim int; BM25 struct{ K1, B float64 }; ... }  // meta.json
type Hit struct { Index int; ChunkID, DocID, URI, Text string; NTokens int; Score float64; Rank int }
type Filter struct { DocIDs, Sources []string }
func (f Filter) Keep(c Chunk) bool
type Embedder interface { Embed(ctx context.Context, texts []string) ([][]float32, error) }
type Retriever interface { Retrieve(ctx context.Context, q string, k int, f Filter) ([]Hit, error) }
type Index struct { Meta Meta; Chunks []Chunk; Vecs [][]float32; BM25 *BM25; IVF *IVF }
func Open(dir string) (*Index, error) // errors wrap ErrIndex (or ErrFormat from bm25.idx)
func (x *Index) Chunk(id string) (Chunk, bool)
func (x *Index) BuildIVF(nlist, iters int) error // PCG32 seeded with FNV1a64(index_id)
func FNV1a64(s string) uint64
type Hybrid struct { Index *Index; Embedder Embedder; Candidates int; RRFK float64; NProbe int; Lambda float64 }
func (h *Hybrid) Retrieve(ctx context.Context, q string, k int, f Filter) ([]Hit, error)
```

`Hybrid.Retrieve` takes `max(Candidates, k)` candidates (default 50) from BM25 and, when `Embedder` is set, from dense search (IVF when `Index.IVF` is built, else flat), fuses them with RRF, then returns the top `k`, or the MMR picks when `Lambda > 0`. It embeds the query once and normalizes it. `Embedder` is the consumer-side interface `ag.06`'s embedder satisfies: this package never imports the ingest packages, and the index files are the only contract between them. Use only the standard library and `tinyllm/ds/rng`.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestHandExampleBM25` | unit | section 3's BM25 numbers to 1e-12; chunk 2 is not returned | you and the tests agree on the formula |
| `TestTerms` | boundary | underscores, hyphens, digits, accents, CJK, empty input | queries and the index cut text the same way |
| `TestBM25MatchesOracle` | conformance | every chunk's score for 11 queries equals an independent Python implementation to 1e-9 | the contract's acceptance bar (`rag-index.md`) |
| `TestBM25IdxGoldenBytes` | golden | `WriteTo` of the fixture corpus equals the oracle's `bm25.idx` byte for byte | your ingest writes a file any reader can open |
| `TestBM25IdxRoundTrip` | property | `ReadBM25(WriteTo(x))` equals `x` | a restarted retriever ranks as before |
| `TestUvarint` | boundary | 0, 127, 128, 300, 16383, 16384, $2^{32}-1$, $2^{64}-1$; truncated input gives `n = 0` | postings decode exactly |
| `TestReadBM25RejectsCorrupt` | fault | every truncation, bad magic, trailing bytes, unsorted terms: `ErrFormat`, never a panic | a crash mid-write never takes the agent down |
| `TestOpenIndex` | unit | the fixture opens; short or long vectors, a missing line, a duplicate id are `ErrIndex` | texts and vectors stay paired |
| `TestFlatMatchesOracle` | conformance | flat top 10 equals the oracle, ties to the lower chunk | the exact baseline IVF is measured against |
| `TestKMeansPPMatchesOracle` | conformance | picks for two seeds equal the oracle's (frozen PCG32 in Python) | centroids reproduce across languages |
| `TestIVFSeededByIndexID` | unit | `FNV1a64` vectors; with zero Lloyd rounds the centroids are the seed rows of `fnv1a64("docs")`; two loads agree | every load builds the same lists |
| `TestIVFRecall` | property | 2000 clustered vectors: recall@10 at `nprobe = 8` is at least 0.95 (0.986 on the reference), and 1 with every list | the catalog's IVF bar |
| `TestRRFHandExample` | unit | section 3's fusion; equal fused scores go to the lower index | the fused order is deterministic |
| `TestMMRHandExample` | unit | section 3's MMR values; $\lambda = 1$ is pure relevance | the agent's context is not three copies |
| `TestHybridMatchesOracle` | conformance | hybrid top 5, with a sources filter, and with MMR, equal the oracle's ids and scores | `ag.08` gets the same chunks the oracle ranks |
| `TestFilterBeforeCut` | boundary | 3 candidates per list and a one-document filter still give 3 hits, all from that document | a scoped search never comes back empty by accident |
| `TestEmbedderMismatch` | boundary | a query vector of the wrong dimension, or none: an error, not a panic | an index embedded by another model is refused |
| `TestLexicalOnlyAndEmpty` | smoke | no embedder: BM25 through RRF; no match: no hits and no error; `k = 0` | "no results" is an answer the tool can return |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. idf without the `1 +` | common terms get negative weights; matching "the" pushes a chunk down | `TestHandExampleBM25`, `TestBM25MatchesOracle` (mutant `s01`) |
| 2. document length as distinct terms | repeated words stop counting toward length; long repetitive chunks win | `TestBM25MatchesOracle` (mutant `s02`) |
| 3. deduplicating query terms | `cache cache` scores like `cache`, unlike every other BM25 | `TestBM25MatchesOracle` (mutant `s03`) |
| 4. a different term rule in the index and the query (no lowercase, ASCII only) | `Café` never finds `café`; the index bytes differ from any other writer | `TestTerms`, `TestBM25IdxGoldenBytes` (mutants `s04`, `s05`) |
| 5. varint with the continuation bit inverted, or postings without gaps | a file only your own reader understands | `TestUvarint`, `TestBM25IdxGoldenBytes`, `TestBM25IdxRoundTrip` (mutants `s06`, `s07`) |
| 6. trusting lengths read from the file | a truncated `bm25.idx` panics the agent at start-up | `TestReadBM25RejectsCorrupt` (mutant `s08`) |
| 7. ties broken by the higher index (or not at all) | two runs of the same query order equal chunks differently | `TestFlatMatchesOracle`, `TestBM25MatchesOracle` (mutant `s09`) |
| 8. k-means++ by farthest point, or weighted by distance instead of its square | different centroids from every other implementation; outliers become seeds | `TestKMeansPPMatchesOracle` (mutants `s10`, `s11`) |
| 9. ignoring `nprobe` | recall collapses (one list) or speed does (all lists) | `TestIVFRecall` (mutant `s12`) |
| 10. seeding IVF with a constant or the clock | lists change between loads; results change after a restart | `TestIVFSeededByIndexID` (mutant `s13`) |
| 11. RRF ranks from 0, or the max instead of the sum | fused order no longer rewards agreement between lists | `TestRRFHandExample`, `TestHybridMatchesOracle` (mutants `s14`, `s15`) |
| 12. MMR penalty against the last pick only, or an unnormalized query | near-copies come back after one diverse pick; relevance and redundancy on different scales | `TestMMRHandExample`, `TestHybridMatchesOracle` (mutants `s16`, `s17`) |
| 13. filtering after the cut | a scoped search returns fewer hits than exist, often none | `TestFilterBeforeCut` (mutant `s18`) |
| 14. not checking the vector file against `n_chunks x dim` | chunk 12's text paired with chunk 13's vector | `TestOpenIndex` (mutant `s19`) |
| 15. not checking the query's dimension | an index embedded by another model gives garbage or a panic | `TestEmbedderMismatch` (mutant `s20`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `ag.06` | writes `docs.jsonl`, `vectors.f32`, `bm25.idx` (calling your `BuildBM25` and `WriteTo` from the ingest entry point) and embeds queries |
| Back | `load.01` | the Go PCG32 (`rng.Seeded`, `Below`, `Float64`) that seeds k-means++ |
| Forward | `ag.08` | the `search_docs` tool calls `Hybrid.Retrieve` and assembles the hits into cited context |

If you skip this module, `ol check ag.08` reports `needs ag.07: build it, or pass --ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `BM25`, `bm25.idx` | Lucene, tantivy | skip lists over postings, block-max WAND to skip chunks that cannot reach the top k, field boosts, analyzers with stemming | [tantivy](https://github.com/quickwit-oss/tantivy) (free), `sq.tantivy` |
| `IVF` | FAISS `IndexIVFPQ`, pgvector `ivfflat`, HNSW | product quantization to shrink vectors, graph indexes with better recall per scan | [FAISS wiki](https://github.com/facebookresearch/faiss/wiki) (free) |
| `RRF` + `MMR` | saige `rag/internal/pipeline/pipeline.go`, cross-encoder rerankers | multi-retriever fusion, a reranker that reads query and chunk together, HyDE query expansion | [saige](https://github.com/urmzd/saige) (free) |
