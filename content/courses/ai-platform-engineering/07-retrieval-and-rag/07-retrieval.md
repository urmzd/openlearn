# Retrieval: BM25, flat and IVF vectors, RRF, MMR

## Overview

| | |
|---|---|
| **Module** | `ag.07` · build · Go · Pass 10 · reading time 2 to 3 h |
| **Solution** | `go/agent/rag/retrieve/`: [`bm25.go`](../../../../course/solutions/go/agent/rag/retrieve/bm25.go) (`Terms`, `BuildBM25`, `Scores`, `Search`, `WriteTo`, `ReadBM25`, the varints), [`vector.go`](../../../../course/solutions/go/agent/rag/retrieve/vector.go) (`Flat`, `KMeansPP`, `BuildIVF`, `IVF.Search`), [`fuse.go`](../../../../course/solutions/go/agent/rag/retrieve/fuse.go) (`RRF`, `MMR`), [`retrieve.go`](../../../../course/solutions/go/agent/rag/retrieve/retrieve.go) (`Open`, `Index.BuildIVF`, `Hybrid.Retrieve`) |
| **Contract** | the index directory of [`course/contracts/formats/rag-index.md`](../../../../course/contracts/formats/rag-index.md); the Go API in section 4 |
| **Needs** | `ag.06` RAG ingest (it writes the index read here and owns the query embedder), `load.01` (the Go PCG32 in `go/ds/rng` seeds k-means++); reading: `M03.6` cosine similarity and top-k (re-implemented here in Go), [infrastructure/04 search and indexing](../../infrastructure/04-search-and-indexing/) (`bm25_from_scratch.py`), [case study 04](../../case-studies/04-kmeans-optimization/) (k-means) |
| **Used by** | `ag.08` context assembly (its `search_docs` tool calls `Retrieve`); the agent's cited answers over the course docs rest on it |
| **Optional depth** | Robertson and Zaragoza, [*The Probabilistic Relevance Framework: BM25 and Beyond*](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) (free); Cormack, Clarke, Buettcher, [*Reciprocal Rank Fusion*](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) (free); Arthur and Vassilvitskii, [*k-means++*](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf) (free); Carbonell and Goldstein, *The Use of MMR* (SIGIR 1998) |

## Key Takeaways

- **BM25** scores a chunk by the query terms it holds, with term frequency that **saturates** (`k1`) and a penalty for **long chunks** (`b`); the `+1` inside the idf keeps every weight positive.
- An index file is a contract between two programs: the **term rule**, the **byte layout**, and the **varint** postings must match exactly, or ingest and retrieval disagree silently.
- **IVF** trades a little recall for speed by scanning only the `nprobe` lists nearest the query; seeded **k-means++** makes those lists the same on every load.
- **RRF** fuses lexical and dense lists by **rank alone**, so scores on different scales never need calibrating; **MMR** then trades relevance for **diversity**.
- Filters run **inside** each search, before the cut to `k`; a filter after the cut returns fewer results than exist.

## How to read this chapter

Sections 1 to 3 build BM25, IVF, RRF, and MMR from first principles and compute each by hand. Section 4 fixes the package API. Section 5 walks through the solution file by file and explains why each choice is the right one. Try to predict each step of the solution before reading it.

Two entry points of the solution use this package: `rag ingest` ([`entry/go/cmd/ctl/rag.go`](../../../../course/solutions/entry/go/cmd/ctl/rag.go)) writes `bm25.idx` with `BuildBM25` and `WriteTo`, and the agent kit ([`entry/go/cmd/internal/agentkit/agentkit.go`](../../../../course/solutions/entry/go/cmd/internal/agentkit/agentkit.go)) opens `/artifacts/rag/<index_id>/`, picks the query embedder that matches `meta.json`'s `embedding_model`, and builds the `Hybrid` retriever behind `search_docs`.

---

## 1. Why now

After `ag.06`, the system has an index of the course docs and its own: chunks with token counts, fingerprints, and embeddings from the engine's `/v1/embeddings`. Nothing reads it. Ask the agent "which hash names a KV block?" and it answers from SmolLM2's weights, which have never seen this system, so it guesses. The agent needs the three chunks that answer the question, picked from hundreds in a few milliseconds. This module is that step: given a query, return the best `k` chunks. It has two halves that fail in opposite ways. Lexical search finds `tl_kv_export` and `30080` exactly but misses a paraphrase; dense search finds "how is a cache entry dropped" for "eviction" but blurs identifiers. The solution builds both, fuses them, and make the result diverse enough that three chunks are not three copies of one sentence.

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

The short chunk earns more per "cache" (0.544 against 0.414), but the long one wins by also holding "eviction". `Search` returns chunk 1, then chunk 0.

**RRF**, $k_{\mathrm{rrf}} = 60$: the lexical list is [x, y, z], the dense list [z, x].

| Chunk | Lexical rank | Dense rank | RRF |
|---|---|---|---|
| x | 1 | 2 | $1/61 + 1/62 = 0.032522$ |
| z | 3 | 1 | $1/63 + 1/61 = 0.032266$ |
| y | 2 | none | $1/62 = 0.016129$ |

Fused order: x, z, y.

**MMR**, $\lambda = 0.5$, query $q = (0.8, 0.6)$, candidates $c_0 = (1, 0)$, $c_1 = (0.96, 0.28)$, $c_2 = (0, 1)$. Relevances: $q \cdot c_0 = 0.8$, $q \cdot c_1 = 0.936$, $q \cdot c_2 = 0.6$; similarities $c_0 \cdot c_1 = 0.96$, $c_0 \cdot c_2 = 0$, $c_1 \cdot c_2 = 0.28$.

| Step | $c_0$ | $c_1$ | $c_2$ | Pick |
|---|---|---|---|---|
| 1 | $0.5 \cdot 0.8 - 0 = 0.4$ | $0.5 \cdot 0.936 - 0 = 0.468$ | $0.3$ | $c_1$ (0.468) |
| 2 | $0.4 - 0.5 \cdot 0.96 = -0.08$ | picked | $0.3 - 0.5 \cdot 0.28 = 0.16$ | $c_2$ (0.16) |
| 3 | $0.4 - 0.5 \cdot \max(0.96, 0) = -0.08$ | picked | picked | $c_0$ (-0.08) |

Pure relevance would order $c_1, c_0, c_2$; MMR moves the near-copy $c_0$ to last.

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

`Hybrid.Retrieve` takes `max(Candidates, k)` candidates (default 50) from BM25 and, when `Embedder` is set, from dense search (IVF when `Index.IVF` is built, else flat), fuses them with RRF, then returns the top `k`, or the MMR picks when `Lambda > 0`. It embeds the query once and normalizes it. `Embedder` is the consumer-side interface `ag.06`'s embedder satisfies: this package never imports the ingest packages, and the index files are the only contract between them. The solution uses only the standard library and `tinyllm/ds/rng`.

### What a correct solution must handle

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| the hand example's BM25 | unit | section 3's BM25 numbers to 1e-12; chunk 2 is not returned | the reference formula |
| terms | boundary | underscores, hyphens, digits, accents, CJK, empty input | queries and the index cut text the same way |
| BM25 against an independent implementation | conformance | every chunk's score for 11 queries equals an independent Python implementation to 1e-9 | the contract's acceptance bar (`rag-index.md`) |
| `bm25.idx` golden bytes | golden | `WriteTo` of the fixture corpus equals the independent writer's `bm25.idx` byte for byte | ingest writes a file any reader can open |
| `bm25.idx` round trip | property | `ReadBM25(WriteTo(x))` equals `x` | a restarted retriever ranks as before |
| varints | boundary | 0, 127, 128, 300, 16383, 16384, $2^{32}-1$, $2^{64}-1$; truncated input gives `n = 0` | postings decode exactly |
| corrupt `bm25.idx` | fault | every truncation, bad magic, trailing bytes, unsorted terms: `ErrFormat`, never a panic | a crash mid-write never takes the agent down |
| opening an index | unit | the fixture opens; short or long vectors, a missing line, a duplicate id are `ErrIndex` | texts and vectors stay paired |
| flat search | conformance | flat top 10 equals the independent implementation, ties to the lower chunk | the exact baseline IVF is measured against |
| k-means++ seeds | conformance | picks for two seeds equal the independent implementation's (frozen PCG32 in Python) | centroids reproduce across languages |
| IVF seeded by the index id | unit | `FNV1a64` vectors; with zero Lloyd rounds the centroids are the seed rows of `fnv1a64("docs")`; two loads agree | every load builds the same lists |
| IVF recall | property | 2000 clustered vectors: recall@10 at `nprobe = 8` is at least 0.95 (0.986 for this solution), and 1 with every list | the catalog's IVF bar |
| the hand example's RRF | unit | section 3's fusion; equal fused scores go to the lower index | the fused order is deterministic |
| the hand example's MMR | unit | section 3's MMR values; $\lambda = 1$ is pure relevance | the agent's context is not three copies |
| hybrid retrieval | conformance | hybrid top 5, with a sources filter, and with MMR, equal the independent implementation's ids and scores | `ag.08` gets the chunks the contract ranks |
| filter before the cut | boundary | 3 candidates per list and a one-document filter still give 3 hits, all from that document | a scoped search never comes back empty by accident |
| embedder mismatch | boundary | a query vector of the wrong dimension, or none: an error, not a panic | an index embedded by another model is refused |
| lexical only, and empty | smoke | no embedder: BM25 through RRF; no match: no hits and no error; `k = 0` | "no results" is an answer the tool can return |

## 5. The solution, step by step

### 5.1 From first principles

1. **Ingest and retrieval are different programs**, so the term rule and the file layout are a contract, not an implementation detail. A Unicode-aware rule (`[\p{L}\p{N}_]+`, lowercased) keeps `café` whole and `tl_kv_export` intact.
2. **Scoring must touch only relevant chunks.** An inverted index maps each term to the chunks holding it, so a query walks $\sum_t \mathrm{df}(t)$ postings rather than $N$ chunks.
3. **Relevance should saturate in tf and discount length**, and no matched term should hurt. That is BM25 with the `1 +` idf: in the hand example "cache" is in 2 of 3 chunks, and the textbook idf $\ln(1.5/2.5)$ would be negative.
4. **Postings are small increasing integers**, so storing gaps and encoding them as varints keeps the file compact; reading must check every length against the bytes left, because the file can be truncated by a crash.
5. **Dense search must be exact by default and approximate by choice.** Flat scan is the yardstick; IVF scans only `nprobe` lists, and its lists must be identical on every load, which forces a deterministic seed (`fnv1a64(index_id)`) and a deterministic seeding rule (k-means++ with $D^2$ weights and a row-order scan).
6. **Lexical and dense scores live on different scales.** Ranks do not, so fusion uses ranks (RRF), and agreement between lists is rewarded by summing.
7. **The top $k$ should not repeat itself.** MMR penalizes similarity to everything already picked, so a copy of the first pick stays penalized.
8. **A filter changes the candidate set, so it must act before the cut.** Every search takes a `keep` predicate.

### 5.2 The code, annotated

**Building the index.** One pass over each text: `DocLen` counts every term occurrence (duplicates included), and postings are appended in document order, so every list is sorted by `Doc` without a sort:

```go
func BuildBM25(texts []string, k1, b float64) *BM25 {
	idx := &BM25{K1: k1, B: b, DocLen: make([]int, len(texts)), Postings: map[string][]Posting{}}
	total := 0
	for d, text := range texts {
		terms := Terms(text)
		idx.DocLen[d] = len(terms) // dl: occurrences, not distinct terms
		total += len(terms)
		tf := map[string]int{}
		var order []string // first-occurrence order: deterministic, unlike map order
		for _, t := range terms {
			if tf[t] == 0 {
				order = append(order, t)
			}
			tf[t]++
		}
		for _, t := range order {
			idx.Postings[t] = append(idx.Postings[t], Posting{Doc: d, TF: tf[t]}) // d ascends
		}
	}
	if len(texts) > 0 {
		idx.AvgDL = float64(total) / float64(len(texts))
	}
	return idx
}
```

**Scoring.** The formula of section 2.2, once per query term occurrence (so `cache cache` counts twice), only over that term's postings:

```go
func (x *BM25) IDF(term string) float64 {
	n := float64(x.N())
	df := float64(len(x.Postings[term]))
	return math.Log(1 + (n-df+0.5)/(df+0.5)) // never negative
}

func (x *BM25) Scores(q string) []float64 {
	s := make([]float64, x.N())
	if x.AvgDL == 0 {
		return s // an empty index: avoid dividing by zero
	}
	for _, t := range Terms(q) { // duplicates kept on purpose
		pl := x.Postings[t]
		if len(pl) == 0 {
			continue
		}
		idf := x.IDF(t)
		for _, p := range pl {
			tf := float64(p.TF)
			dl := float64(x.DocLen[p.Doc])
			s[p.Doc] += idf * tf * (x.K1 + 1) / (tf + x.K1*(1-x.B+x.B*dl/x.AvgDL))
		}
	}
	return s
}
```

`Search` collects only documents that matched at least one term (chunk 2 in the hand example is absent, not a zero), applies `keep`, then `TopK`, which sorts by score descending with ties to the lower `Doc`.

**Varints and the file.** `PutUvarint` emits 7 bits at a time, low group first; 300 gives `ac 02`. `Uvarint` refuses an eleventh byte or a tenth byte above 1, which is what overflowing 64 bits looks like:

```go
func PutUvarint(buf []byte, v uint64) []byte {
	for v >= 0x80 {
		buf = append(buf, byte(v)|0x80) // continuation bit set: more follows
		v >>= 7
	}
	return append(buf, byte(v)) // last byte: top bit clear
}

	// in WriteTo, per term (terms sorted by their bytes):
		prev := 0
		for _, p := range pl {
			post = PutUvarint(post, uint64(p.Doc-prev)) // gap; the first gap is the doc number
			post = PutUvarint(post, uint64(p.TF))
			prev = p.Doc
		}
		b = binary.LittleEndian.AppendUint16(b, uint16(len(t)))
		b = append(b, t...)
		b = binary.LittleEndian.AppendUint32(b, uint32(len(pl)))   // df
		b = binary.LittleEndian.AppendUint32(b, uint32(len(post))) // postings bytes
		b = append(b, post...)
```

`ReadBM25` is the same walk with a guard before every read: the header is at least 24 bytes with magic `TLBM` and version 1; doc lengths must fit in what remains; each term's length plus 8 must fit; terms must be strictly increasing; each posting's gap must be non-zero after the first and land below $N$; a postings block must decode to exactly `df` pairs; and nothing may trail the last term. Each failure is `fmt.Errorf("%w: ...", ErrFormat)`, never an index past the end of the slice. Full solution: [`bm25.go`](../../../../course/solutions/go/agent/rag/retrieve/bm25.go).

**k-means++.** $D$ is kept as the squared distance to the nearest seed and updated after each pick, so each round is $O(n \cdot \text{dim})$:

```go
	picked := []int{int(r.Below(uint64(n)))} // first seed: uniform
	// ... d[i] = Dist2(vecs[i], vecs[picked[0]])
	for len(picked) < k {
		var sum float64
		for _, x := range d {
			sum += x
		}
		next := -1
		if sum == 0 {
			// every remaining row duplicates a seed: lowest row not yet picked
		} else {
			u := r.Float64() * sum
			var run float64
			for i, x := range d { // row order: part of the contract
				run += x
				if run > u { // strictly greater: a row with D = 0 is never chosen
					next = i
					break
				}
			}
			// ... u rounded up to sum: take the last row with weight
		}
		picked = append(picked, next)
		// ... d[i] = min(d[i], Dist2(vecs[i], vecs[next]))
	}
```

**IVF.** `BuildIVF` runs Lloyd rounds (assign to `Nearest`, move each centroid to the float64 mean of its rows, leave an empty centroid where it is, stop when no assignment changes), then builds the lists from the final centroids. `Search` ranks centroids by negative squared distance and scans only the top `nprobe` lists, applying `keep` inside the scan:

```go
func (x *IVF) Search(q []float32, k, nprobe int, keep func(doc int) bool) []Scored {
	cs := make([]Scored, len(x.Centroids))
	for c, cv := range x.Centroids {
		cs[c] = Scored{Doc: c, Score: -Dist2(q, cv)} // nearest centroid = highest score
	}
	cs = TopK(cs, nprobe)
	var cand []Scored
	for _, c := range cs {
		for _, i := range x.Lists[c.Doc] {
			if keep == nil || keep(i) { // the filter runs before the cut to k
				cand = append(cand, Scored{Doc: i, Score: Dot(q, x.Vecs[i])})
			}
		}
	}
	return TopK(cand, k)
}
```

`Index.BuildIVF` seeds the generator with `rng.Seeded(FNV1a64(x.Meta.IndexID))`, so the same index builds the same lists on any machine. Full solution: [`vector.go`](../../../../course/solutions/go/agent/rag/retrieve/vector.go).

**RRF.** Ranks from 1, summed over lists; other fields come from the first list holding the chunk; ties to the lower index:

```go
func RRF(lists [][]Hit, k float64, n int) []Hit {
	score := map[int]float64{}
	first := map[int]Hit{}
	for _, l := range lists {
		for r, h := range l {
			score[h.Index] += 1 / (k + float64(r+1)) // r is 0-based; the rank is r+1
			if _, ok := first[h.Index]; !ok {
				first[h.Index] = h
			}
		}
	}
	// ... collect, sort by score desc then Index asc, cut to n, set Rank = position + 1
}
```

For the hand example: x gets $1/61 + 1/62$, z $1/63 + 1/61$, y $1/62$.

**MMR.** `maxSim` holds, for every unpicked candidate, its largest similarity to anything picked so far; it is updated after each pick, so a step costs $O(|\text{cands}|)$ dot products:

```go
	rel := make([]float64, len(cands))
	for i, c := range cands {
		rel[i] = Dot(q, vec(c.Index))
	}
	maxSim := make([]float64, len(cands)) // max over an empty set: 0
	used := make([]bool, len(cands))
	out := make([]Hit, 0, k)
	for len(out) < k {
		best, bv := -1, 0.0
		for i := range cands {
			if used[i] {
				continue
			}
			v := lambda*rel[i] - (1-lambda)*maxSim[i]
			if best < 0 || v > bv { // strict: ties go to the earlier candidate
				best, bv = i, v
			}
		}
		used[best] = true
		// ... append with Score = bv, Rank = len(out)+1
		pv := vec(h.Index)
		for i := range cands {
			if used[i] {
				continue
			}
			s := Dot(vec(cands[i].Index), pv)
			if len(out) == 1 || s > maxSim[i] { // the first pick sets it, even if negative
				maxSim[i] = s
			}
		}
	}
```

The `len(out) == 1` case matters: after the first pick `maxSim` must become the actual similarity, which may be negative, not stay at the empty-set 0. Full solution: [`fuse.go`](../../../../course/solutions/go/agent/rag/retrieve/fuse.go).

**Opening and retrieving.** `Open` checks that the four files agree: `n_chunks` lines, `n_chunks * dim * 4` vector bytes, `n_chunks` BM25 documents, and unique chunk ids that start with `<doc_id>#`. `Retrieve` builds one `keep` from the filter and passes it into every search:

```go
	keep := func(i int) bool { return f.Keep(x.Chunks[i]) }
	var lists [][]Hit
	// ... lex from x.BM25.Search(q, n, keep)
	lists = append(lists, lex)
	var qv []float32
	if h.Embedder != nil {
		vs, err := h.Embedder.Embed(ctx, []string{q})
		// ... exactly one vector
		if len(vs[0]) != x.Meta.Dim {
			return nil, fmt.Errorf("%w: query embedding has dim %d, index has %d", ErrIndex, len(vs[0]), x.Meta.Dim)
		}
		qv = Normalize(vs[0]) // unit query: dot products are cosines, MMR terms share a scale
		var dense []Scored
		if x.IVF != nil {
			dense = x.IVF.Search(qv, n, nprobe, keep)
		} else {
			dense = (&Flat{Vecs: x.Vecs}).Search(qv, n, keep)
		}
		// ... lists = append(lists, dense hits)
	}
	fused := RRF(lists, rrfk, n)
	if h.Lambda > 0 && qv != nil {
		return MMR(qv, fused, func(i int) []float32 { return x.Vecs[i] }, h.Lambda, k), nil
	}
	// ... fused[:k]
```

Full solution: [`retrieve.go`](../../../../course/solutions/go/agent/rag/retrieve/retrieve.go).

### 5.3 Why this solution

- **Cost.** BM25 is $O(\sum_{t \in q} \mathrm{df}(t))$ plus a sort of the matches; flat search is $O(N \cdot \text{dim})$; IVF is $O(\text{nlist} \cdot \text{dim})$ for the centroids plus about $\text{nprobe}/\text{nlist}$ of the flat cost. Building IVF costs $O(\text{iters} \cdot N \cdot \text{nlist} \cdot \text{dim})$ once per load. RRF is linear in the candidates, MMR $O(k \cdot |\text{cands}|)$ dot products.
- **IVF over HNSW.** A graph index has better recall per scanned row but is far more code, harder to make bit-for-bit reproducible across languages, and pays off only at sizes this index does not reach. IVF with a seeded k-means is small, deterministic, and measurable against flat.
- **RRF over score normalization.** Min-max or z-score normalization of BM25 and cosine scores depends on each query's score distribution and needs tuning; RRF has one constant, 60, that works across queries.
- **Building IVF at load over storing centroids.** No extra file format, and any implementation of the spec reproduces the lists; the cost is a few seconds of k-means per start-up.
- **Accepted tradeoffs.** No stemming (`evict` does not match `eviction`); BM25 is held fully in memory; a filter scans every chunk's metadata.

### 5.4 Explain it back

1. Why is the idf $\ln(1 + \frac{N - \mathrm{df} + 0.5}{\mathrm{df} + 0.5})$ rather than $\ln \frac{N - \mathrm{df} + 0.5}{\mathrm{df} + 0.5}$?
   <details><summary>Answer</summary>For a term in more than half the chunks the inner ratio is below 1 and the plain logarithm is negative, so matching the term would push a chunk down. With `1 +` the argument is always above 1 and every matched term adds a small positive amount.</details>
2. Why does `ReadBM25` check that each gap after the first is non-zero?
   <details><summary>Answer</summary>Postings are strictly increasing chunk numbers, so a gap of 0 would repeat a chunk. It can only come from corruption or a broken writer, and accepting it would double-count a chunk's tf.</details>
3. Why must the k-means++ draw scan rows in row order and compare `run > u` strictly?
   <details><summary>Answer</summary>The spec fixes the pick for a given PCG32 stream; any other order or comparison picks a different row for the same `u`, so the centroids would differ between implementations. The strict comparison also means a row with $D = 0$ (already a seed) can never be chosen.</details>
4. Why does RRF reward a chunk that is second in both lists over one that is first in only one?
   <details><summary>Answer</summary>$2/62 = 0.0323 > 1/61 = 0.0164$. Each list contributes $1/(60 + r)$, which varies slowly with rank, so appearing in a second list matters more than moving up one place. That is the intended bias: agreement between independent signals is strong evidence.</details>
5. What goes wrong if the filter is applied to the fused top `k`?
   <details><summary>Answer</summary>The 50 best candidates overall may contain no chunk of the requested document, so the caller gets fewer hits than exist, often none. Passing `keep` into each search makes the filtered chunks compete only among themselves.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. idf without the `1 +` | common terms get negative weights; matching "the" pushes a chunk down | `IDF` is `ln(1 + (N - df + 0.5)/(df + 0.5))` |
| 2. document length as distinct terms | repeated words stop counting toward length; long repetitive chunks win | `DocLen[d] = len(Terms(text))`, duplicates included |
| 3. deduplicating query terms | `cache cache` scores like `cache`, unlike every other BM25 | `Scores` loops over `Terms(q)` with duplicates |
| 4. a different term rule in the index and the query (no lowercase, ASCII only) | `Café` never finds `café`; the index bytes differ from any other writer | one `Terms` (`[\p{L}\p{N}_]+`, `strings.ToLower`) for both |
| 5. varint with the continuation bit inverted, or postings without gaps | a file only one reader understands | `PutUvarint` sets the top bit on every byte but the last; `WriteTo` writes gaps |
| 6. trusting lengths read from the file | a truncated `bm25.idx` panics the agent at start-up | `ReadBM25` checks every length against the remaining bytes |
| 7. ties broken by the higher index (or not at all) | two runs of the same query order equal chunks differently | `TopK` and `RRF` break ties to the lower index |
| 8. k-means++ by farthest point, or weighted by distance instead of its square | different centroids from every other implementation; outliers become seeds | `KMeansPP` draws proportional to `Dist2` |
| 9. ignoring `nprobe` | recall collapses (one list) or speed does (all lists) | `IVF.Search` scans `TopK(centroids, nprobe)` |
| 10. seeding IVF with a constant or the clock | lists change between loads; results change after a restart | `Index.BuildIVF` seeds with `FNV1a64(index_id)` |
| 11. RRF ranks from 0, or the max instead of the sum | fused order no longer rewards agreement between lists | `1 / (k + r + 1)` summed over lists |
| 12. MMR penalty against the last pick only, or an unnormalized query | near-copies come back after one diverse pick; relevance and redundancy on different scales | `maxSim` keeps the max over all picks; `Retrieve` normalizes the query |
| 13. filtering after the cut | a scoped search returns fewer hits than exist, often none | `keep` is passed into every search |
| 14. not checking the vector file against `n_chunks x dim` | chunk 12's text paired with chunk 13's vector | `Open` requires exactly `n * dim * 4` bytes |
| 15. not checking the query's dimension | an index embedded by another model gives garbage or a panic | `Retrieve` returns `ErrIndex` on a dimension mismatch |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `ag.06` | writes `docs.jsonl`, `vectors.f32`, `bm25.idx` (calling `BuildBM25` and `WriteTo` from the ingest entry point) and embeds queries |
| Back | `load.01` | the Go PCG32 (`rng.Seeded`, `Below`, `Float64`) that seeds k-means++ |
| Forward | `ag.08` | the `search_docs` tool calls `Hybrid.Retrieve` and assembles the hits into cited context |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `BM25`, `bm25.idx` | Lucene, tantivy | skip lists over postings, block-max WAND to skip chunks that cannot reach the top k, field boosts, analyzers with stemming | [tantivy](https://github.com/quickwit-oss/tantivy) (free), `sq.tantivy` |
| `IVF` | FAISS `IndexIVFPQ`, pgvector `ivfflat`, HNSW | product quantization to shrink vectors, graph indexes with better recall per scan | [FAISS wiki](https://github.com/facebookresearch/faiss/wiki) (free) |
| `RRF` + `MMR` | saige `rag/internal/pipeline/pipeline.go`, cross-encoder rerankers | multi-retriever fusion, a reranker that reads query and chunk together, HyDE query expansion | [saige](https://github.com/urmzd/saige) (free) |
