# Retrieval: BM25, Flat and IVF Vectors, RRF, and MMR

After [RAG ingest](06-rag-ingest.md), the system has an index of the course docs and its own: chunks with token counts, fingerprints, and embeddings from the engine's `/v1/embeddings`. Nothing reads it yet. Ask the agent "which hash names a KV block?" and it answers from SmolLM2's weights, which have never seen this system, so it guesses. What the agent needs is the three chunks that answer the question, picked from hundreds in a few milliseconds. This section builds that step: given a query, return the best `k` chunks.

Retrieval has two halves that fail in opposite ways. Lexical search finds `tl_kv_export` and `30080` exactly but misses a paraphrase. Dense search matches "how is a cache entry dropped" to "eviction" but blurs identifiers. So you will build both, fuse them, and make the result diverse enough that three chunks are not three copies of one sentence. Along the way you will treat the index file as what it really is, a contract between two programs, and you will see why a filter has to run inside a search rather than after it.

This section builds on [RAG ingest](06-rag-ingest.md), which writes the index read here and owns the query embedder, and on the Go PCG32 port (in `go/ds/rng`) from [Open-loop load generator, log-linear histogram, Go PCG32 port](../../build-an-llm/p10-serving/10-load-generator.md) in *Build an LLM*, which seeds k-means++. Useful background: [Inner products, projections, cosine similarity, and top-k](../../mathematics-for-computing/03-linear-algebra/06-inner-products-projections-cosine-and-top-k.md) in *Mathematics for Computing* (re-implemented here in Go), the [Search & Indexing](../../infrastructure/04-search-and-indexing/) chapter of *Infrastructure* (`bm25_from_scratch.py`), and the [K-Means Optimization](../../case-studies/04-kmeans-optimization/) case study. The full solution lives in `go/agent/rag/retrieve/`: [`bm25.go`](../../../../course/solutions/go/agent/rag/retrieve/bm25.go) (`Terms`, `BuildBM25`, `Scores`, `Search`, `WriteTo`, `ReadBM25`, the varints), [`vector.go`](../../../../course/solutions/go/agent/rag/retrieve/vector.go) (`Flat`, `KMeansPP`, `BuildIVF`, `IVF.Search`), [`fuse.go`](../../../../course/solutions/go/agent/rag/retrieve/fuse.go) (`RRF`, `MMR`), and [`retrieve.go`](../../../../course/solutions/go/agent/rag/retrieve/retrieve.go) (`Open`, `Index.BuildIVF`, `Hybrid.Retrieve`). The contract is the index directory of [`course/contracts/formats/rag-index.md`](../../../../course/contracts/formats/rag-index.md) plus the Go API shown later in this section.

Two entry points of the solution use this package. `rag ingest` ([`entry/go/cmd/ctl/rag.go`](../../../../course/solutions/entry/go/cmd/ctl/rag.go)) writes `bm25.idx` with `BuildBM25` and `WriteTo`. The agent kit ([`entry/go/cmd/internal/agentkit/agentkit.go`](../../../../course/solutions/entry/go/cmd/internal/agentkit/agentkit.go)) opens `/artifacts/rag/<index_id>/`, picks the query embedder that matches `meta.json`'s `embedding_model`, and builds the `Hybrid` retriever behind `search_docs`.

> **Learning objectives**
> - Compute BM25 by hand and explain what saturation (`k1`), length normalization (`b`), and the `+1` inside the idf each do.
> - Read and write an inverted index file with gap-encoded varint postings, defensively enough that a truncated file never panics.
> - Build a reproducible IVF index with seeded k-means++ and measure its recall against flat search.
> - Fuse lexical and dense lists with RRF and diversify the result with MMR, by hand and in code.
> - Apply filters inside each search, before the cut to `k`.

## The notation

Every formula in this section uses the same small set of symbols:

| Symbol | Meaning | Type |
|---|---|---|
| $N$ | number of chunks in the index | integer |
| $t$ | a term (defined next) | string |
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

## Terms: the rule both programs share

Retrieval and ingest are separate programs, and they must cut text into the same pieces. The contract's rule is that a **term** is a maximal run of Unicode letters, digits, and `_` (`[\p{L}\p{N}_]+` in Go's RE2), lowercased with `strings.ToLower`. Everything else separates terms. So `tl_kv_export(handle)` gives `tl_kv_export` and `handle`, `k-means++` gives `k` and `means`, and `Café` gives `café`. An ASCII-only rule (`[a-z0-9]+`) would cut `café` into `caf`, and a query typed with the accent would never find it.

## The inverted index and BM25

With terms defined, the lexical index is an **inverted index**: a map from each term to its **postings list**, the chunks containing it, in chunk order, each with its $\mathrm{tf}$. Scoring a query walks only the lists of its terms, never the whole index.

BM25 adds, for every query term occurrence $t$ and every chunk $d$ in its list,

$$\mathrm{idf}(t) \cdot \frac{\mathrm{tf}(t,d)\,(k_1+1)}{\mathrm{tf}(t,d) + k_1\left(1 - b + b\,\frac{\mathrm{dl}(d)}{\overline{\mathrm{dl}}}\right)}, \qquad \mathrm{idf}(t) = \ln\!\left(1 + \frac{N - \mathrm{df}(t) + 0.5}{\mathrm{df}(t) + 0.5}\right).$$

Read the fraction from the inside. With $b = 0$ it is $\mathrm{tf}(k_1+1)/(\mathrm{tf}+k_1)$: 1 at $\mathrm{tf} = 1$, then rising towards the ceiling $k_1 + 1 = 2.2$, so the tenth "cache" adds far less than the first. That is **saturation**: a chunk cannot win by repeating a word. The factor $1 - b + b\,\mathrm{dl}/\overline{\mathrm{dl}}$ is 1 for a chunk of average length and larger for a long one; a long chunk holds more words by chance, so each match counts for less.

The idf is large for a rare term and near 0 for a term in every chunk. Without the `1 +` inside the logarithm (the textbook Robertson-Sparck Jones form), a term in more than half the chunks gets a **negative** weight, and matching it pushes a chunk down. Finally, a query term written twice counts twice (`"cache cache"` doubles every "cache" contribution), as in Lucene and `bm25_from_scratch.py`.

### The index file

`bm25.idx` stores the index so a restart does not rebuild it. Its layout is: magic `TLBM`, version, $N$, number of terms, $\overline{\mathrm{dl}}$ as f64 bits, every $\mathrm{dl}$, then each term (sorted by its bytes) with $\mathrm{df}$ and its postings. Postings are stored as **gaps**: the first entry is the chunk number, and each next one is the difference from the previous, so the numbers stay small. Each number is an **unsigned LEB128 varint**: 7 bits per byte, lowest group first, with the top bit set on every byte except the last. For example, 300 is binary `100101100`. Its low seven bits `0101100` go first with the top bit set (`0xac`), then the remaining `10` (`0x02`), giving `ac 02`.

## Dense vectors: flat search and IVF

The other half of retrieval is dense. `vectors.f32` holds one embedding per chunk, L2-normalized, so cosine similarity is just the dot product $q \cdot x_i$. **Flat** search scores every row and keeps the top $k$, with ties going to the lower row. It is exact, and it is the yardstick for anything faster.

An **inverted file** (IVF) index is the faster thing. It groups rows into `nlist` clusters around centroids $c_j$ and stores each row in the **list** of its nearest centroid. A query computes its distance to every centroid, scans only the `nprobe` nearest lists exactly, and returns the top $k$ among them. With 32 lists and `nprobe = 8` it scores about a quarter of the rows. **Recall@k** measures what that costs: the fraction of the exact top $k$ that the approximate search also returned.

The centroids come from **k-means**: repeat "assign each row to its nearest centroid, then move each centroid to the mean of its rows" (**Lloyd** rounds). The result depends on where you start, so the start is fixed by **k-means++** seeding:

1. The first seed is a uniform row: `r.Below(n)`.
2. For every row $i$, $D(i)$ is the squared distance to the nearest seed so far.
3. The next seed is row $i$ with probability $D(i) / \sum_j D(j)$: draw $u$ = `r.Float64()` $\cdot \sum_j D(j)$ and take the first row, in row order, whose running sum of $D$ exceeds $u$.
4. Repeat steps 2 and 3 until there are `nlist` seeds.

Rows far from every seed are likely picks, so seeds spread out, and rows already chosen have $D = 0$ and are never picked twice. Using the distance instead of its square, or always taking the farthest row, gives different seeds, and for outliers worse ones. The index stores no centroids. Instead, `rag-index.md` says every load builds them with PCG32 seeded by `fnv1a64(index_id)`, so the same index always gets the same lists, on any machine and in any language that follows the spec.

## Fusion and diversity

You now have two ranked lists that cannot be compared directly. BM25 scores are unbounded; dot products lie in $[-1, 1]$. Adding them needs a calibration that changes with every query. **Reciprocal rank fusion** sidesteps calibration by using ranks only:

$$\mathrm{RRF}(d) = \sum_{\ell} \frac{1}{k_{\mathrm{rrf}} + r_\ell(d)},$$

summed over the lists that contain $d$. With $k_{\mathrm{rrf}} = 60$, being first in one list is worth $1/61$, and being in both lists beats being first in one. Ranks count from 1; counting from 0 changes every value.

The fused top $k$ often holds near-duplicates: the same paragraph in two docs, or two adjacent chunks. **Maximal marginal relevance** (MMR) picks one chunk at a time, each time the one that maximizes

$$\lambda \,(q \cdot x_c) - (1 - \lambda) \max_{p \in \text{picked}} (x_c \cdot x_p),$$

with the max over an empty set equal to 0, so the first pick is the most relevant. $\lambda = 1$ is pure relevance; smaller $\lambda$ buys diversity. The penalty is the **max** over everything already picked, because a candidate that copies the first pick is still a copy after the second.

## Filters

One last principle concerns scope. A filter (`DocIDs`, `Sources`) restricts which chunks may be returned, and it must run inside each search, before the cut to the candidate count. Filtering after the cut is the classic bug: the best 50 chunks overall may hold none from the requested document, and the caller gets an empty list although matching chunks exist.

## Working it by hand

Start with **BM25**, $k_1 = 1.2$ and $b = 0.75$, over three chunks:

| Chunk | Text | Terms | $\mathrm{dl}$ |
|---|---|---|---|
| 0 | `kv cache` | kv, cache | 2 |
| 1 | `kv cache eviction policy` | kv, cache, eviction, policy | 4 |
| 2 | `gateway rate limits` | gateway, rate, limits | 3 |

So $N = 3$ and $\overline{\mathrm{dl}} = 9/3 = 3$. The query is `cache eviction`.

- $\mathrm{df}(\text{cache}) = 2$: $\mathrm{idf} = \ln(1 + 1.5/2.5) = \ln 1.6 = 0.470004$.
- $\mathrm{df}(\text{eviction}) = 1$: $\mathrm{idf} = \ln(1 + 2.5/1.5) = \ln(8/3) = 0.980829$.
- Chunk 0, $\mathrm{dl} = 2$: length factor $1 - 0.75 + 0.75 \cdot 2/3 = 0.75$; tf part $= 2.2 / (1 + 1.2 \cdot 0.75) = 2.2/1.9 = 1.157895$. Score $= 0.470004 \cdot 1.157895 = 0.544215$.
- Chunk 1, $\mathrm{dl} = 4$: length factor $0.25 + 0.75 \cdot 4/3 = 1.25$; tf part $= 2.2 / (1 + 1.5) = 0.88$. Score $= 0.470004 \cdot 0.88 + 0.980829 \cdot 0.88 = 0.413603 + 0.863130 = 1.276733$.
- Chunk 2 holds neither term: its score is 0, and it is not returned at all.

The short chunk earns more per "cache" (0.544 against 0.414), but the long one wins by also holding "eviction". `Search` returns chunk 1, then chunk 0. Notice also that "cache" is in 2 of 3 chunks, so the textbook idf $\ln(1.5/2.5)$ would have been negative.

Next, **RRF** with $k_{\mathrm{rrf}} = 60$, where the lexical list is [x, y, z] and the dense list is [z, x]:

| Chunk | Lexical rank | Dense rank | RRF |
|---|---|---|---|
| x | 1 | 2 | $1/61 + 1/62 = 0.032522$ |
| z | 3 | 1 | $1/63 + 1/61 = 0.032266$ |
| y | 2 | none | $1/62 = 0.016129$ |

The fused order is x, z, y.

Finally, **MMR** with $\lambda = 0.5$, query $q = (0.8, 0.6)$, and candidates $c_0 = (1, 0)$, $c_1 = (0.96, 0.28)$, $c_2 = (0, 1)$. The relevances are $q \cdot c_0 = 0.8$, $q \cdot c_1 = 0.936$, $q \cdot c_2 = 0.6$, and the similarities are $c_0 \cdot c_1 = 0.96$, $c_0 \cdot c_2 = 0$, $c_1 \cdot c_2 = 0.28$. Each step scores the remaining candidates:

| Step | $c_0$ | $c_1$ | $c_2$ | Pick |
|---|---|---|---|---|
| 1 | $0.5 \cdot 0.8 - 0 = 0.4$ | $0.5 \cdot 0.936 - 0 = 0.468$ | $0.3$ | $c_1$ (0.468) |
| 2 | $0.4 - 0.5 \cdot 0.96 = -0.08$ | picked | $0.3 - 0.5 \cdot 0.28 = 0.16$ | $c_2$ (0.16) |
| 3 | $0.4 - 0.5 \cdot \max(0.96, 0) = -0.08$ | picked | picked | $c_0$ (-0.08) |

Pure relevance would order $c_1, c_0, c_2$; MMR moves the near-copy $c_0$ to last.

## Designing the API

The package API, grouped by file:

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

`Hybrid.Retrieve` takes `max(Candidates, k)` candidates (default 50) from BM25 and, when `Embedder` is set, from dense search (IVF when `Index.IVF` is built, else flat). It fuses them with RRF, then returns the top `k`, or the MMR picks when `Lambda > 0`. It embeds the query once and normalizes it. `Embedder` is a consumer-side interface that the ingest embedder satisfies: this package never imports the ingest packages, and the index files are the only contract between them. Every search takes a `keep` predicate so that filters can act before the cut. The solution uses only the standard library and `tinyllm/ds/rng`.

## Building it step by step

The design follows from eight observations:

1. **Ingest and retrieval are different programs**, so the term rule and the file layout are a contract, not an implementation detail. A Unicode-aware rule (`[\p{L}\p{N}_]+`, lowercased) keeps `café` whole and `tl_kv_export` intact.
2. **Scoring must touch only relevant chunks.** An inverted index maps each term to the chunks holding it, so a query walks $\sum_t \mathrm{df}(t)$ postings rather than $N$ chunks.
3. **Relevance should saturate in tf and discount length**, and no matched term should hurt. That is BM25 with the `1 +` idf.
4. **Postings are small increasing integers**, so storing gaps and encoding them as varints keeps the file compact. Reading must check every length against the bytes left, because a crash can truncate the file.
5. **Dense search must be exact by default and approximate by choice.** Flat scan is the yardstick. IVF scans only `nprobe` lists, and its lists must be identical on every load, which forces a deterministic seed (`fnv1a64(index_id)`) and a deterministic seeding rule (k-means++ with $D^2$ weights and a row-order scan).
6. **Lexical and dense scores live on different scales.** Ranks do not, so fusion uses ranks (RRF), and agreement between lists is rewarded by summing.
7. **The top $k$ should not repeat itself.** MMR penalizes similarity to everything already picked, so a copy of the first pick stays penalized.
8. **A filter changes the candidate set, so it must act before the cut.** Every search takes a `keep` predicate.

### Building the BM25 index

`BuildBM25` makes one pass over each text. `DocLen` counts every term occurrence, duplicates included, and postings are appended in document order, so every list is sorted by `Doc` without a sort:

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

### Scoring

Scoring is the BM25 formula, applied once per query term occurrence (so `cache cache` counts twice) and only over that term's postings:

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

`Search` collects only documents that matched at least one term (chunk 2 in the hand example is absent, not a zero), applies `keep`, then calls `TopK`, which sorts by score descending with ties to the lower `Doc`.

### Varints and the file

`PutUvarint` emits 7 bits at a time, low group first, so 300 gives `ac 02`. `Uvarint` refuses an eleventh byte, or a tenth byte above 1, which is what overflowing 64 bits looks like:

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

`ReadBM25` is the same walk with a guard before every read. The header must be at least 24 bytes with magic `TLBM` and version 1; the doc lengths must fit in what remains; each term's length plus 8 must fit; terms must be strictly increasing; each posting's gap must be non-zero after the first and land below $N$; a postings block must decode to exactly `df` pairs; and nothing may trail the last term. Each failure is `fmt.Errorf("%w: ...", ErrFormat)`, never an index past the end of the slice. The full code is in [`bm25.go`](../../../../course/solutions/go/agent/rag/retrieve/bm25.go).

### k-means++

`KMeansPP` keeps $D$ as the squared distance to the nearest seed and updates it after each pick, so each round is $O(n \cdot \text{dim})$:

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

### IVF

`BuildIVF` runs Lloyd rounds: assign each row to `Nearest`, move each centroid to the float64 mean of its rows, leave an empty centroid where it is, and stop when no assignment changes. It then builds the lists from the final centroids. `Search` ranks centroids by negative squared distance and scans only the top `nprobe` lists, applying `keep` inside the scan:

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

`Index.BuildIVF` seeds the generator with `rng.Seeded(FNV1a64(x.Meta.IndexID))`, so the same index builds the same lists on any machine. The full code is in [`vector.go`](../../../../course/solutions/go/agent/rag/retrieve/vector.go).

### RRF

`RRF` counts ranks from 1 and sums over lists. A hit's other fields come from the first list holding the chunk, and ties go to the lower index:

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

For the hand example, x gets $1/61 + 1/62$, z gets $1/63 + 1/61$, and y gets $1/62$.

### MMR

`maxSim` holds, for every unpicked candidate, its largest similarity to anything picked so far. It is updated after each pick, so a step costs $O(|\text{cands}|)$ dot products:

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

The `len(out) == 1` case matters: after the first pick, `maxSim` must become the actual similarity, which may be negative, rather than stay at the empty-set 0. The full code is in [`fuse.go`](../../../../course/solutions/go/agent/rag/retrieve/fuse.go).

### Opening and retrieving

`Open` checks that the four files agree: `n_chunks` lines, `n_chunks * dim * 4` vector bytes, `n_chunks` BM25 documents, and unique chunk ids that start with `<doc_id>#`. `Retrieve` builds one `keep` from the filter and passes it into every search:

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

The full code is in [`retrieve.go`](../../../../course/solutions/go/agent/rag/retrieve/retrieve.go).

### Why this design

Start with cost. BM25 is $O(\sum_{t \in q} \mathrm{df}(t))$ plus a sort of the matches; flat search is $O(N \cdot \text{dim})$; IVF is $O(\text{nlist} \cdot \text{dim})$ for the centroids plus about $\text{nprobe}/\text{nlist}$ of the flat cost. Building IVF costs $O(\text{iters} \cdot N \cdot \text{nlist} \cdot \text{dim})$ once per load. RRF is linear in the candidates, and MMR costs $O(k \cdot |\text{cands}|)$ dot products.

Why IVF rather than HNSW? A graph index has better recall per scanned row, but it is far more code, harder to make bit-for-bit reproducible across languages, and pays off only at sizes this index does not reach. IVF with a seeded k-means is small, deterministic, and measurable against flat. Why RRF rather than score normalization? Min-max or z-score normalization of BM25 and cosine scores depends on each query's score distribution and needs tuning; RRF has one constant, 60, that works across queries. And why build IVF at load rather than store centroids? There is no extra file format, and any implementation of the spec reproduces the lists; the cost is a few seconds of k-means per start-up.

Some tradeoffs are accepted on purpose: there is no stemming (`evict` does not match `eviction`), BM25 is held fully in memory, and a filter scans every chunk's metadata.

## What can go wrong

A correct solution must pass these checks; each line says what must hold and why it matters to the code that depends on retrieval:

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| the hand example's BM25 | unit | the hand-worked BM25 numbers to 1e-12; chunk 2 is not returned | the reference formula |
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
| the hand example's RRF | unit | the hand-worked fusion; equal fused scores go to the lower index | the fused order is deterministic |
| the hand example's MMR | unit | the hand-worked MMR values; $\lambda = 1$ is pure relevance | the agent's context is not three copies |
| hybrid retrieval | conformance | hybrid top 5, with a sources filter, and with MMR, equal the independent implementation's ids and scores | context assembly gets the chunks the contract ranks |
| filter before the cut | boundary | 3 candidates per list and a one-document filter still give 3 hits, all from that document | a scoped search never comes back empty by accident |
| embedder mismatch | boundary | a query vector of the wrong dimension, or none: an error, not a panic | an index embedded by another model is refused |
| lexical only, and empty | smoke | no embedder: BM25 through RRF; no match: no hits and no error; `k = 0` | "no results" is an answer the tool can return |

The pitfalls fall into three groups. In BM25, an idf without the `1 +` gives common terms negative weights, so matching "the" pushes a chunk down; `IDF` is `ln(1 + (N - df + 0.5)/(df + 0.5))`. Counting document length as distinct terms means repeated words stop counting toward length, so long repetitive chunks win; `DocLen[d] = len(Terms(text))`, duplicates included. Deduplicating query terms makes `cache cache` score like `cache`, unlike every other BM25; `Scores` loops over `Terms(q)` with duplicates. And a different term rule in the index and the query (no lowercasing, or ASCII only) means `Café` never finds `café` and the index bytes differ from any other writer's; one `Terms` (`[\p{L}\p{N}_]+`, `strings.ToLower`) serves both.

> **Warning:** The index file is shared by two programs. A varint with the continuation bit inverted, or postings written without gaps, produces a file only one reader understands; `PutUvarint` sets the top bit on every byte but the last, and `WriteTo` writes gaps. Trusting lengths read from the file lets a truncated `bm25.idx` panic the agent at start-up; `ReadBM25` checks every length against the remaining bytes. Not checking the vector file against `n_chunks x dim` pairs chunk 12's text with chunk 13's vector; `Open` requires exactly `n * dim * 4` bytes.

Reproducibility is the second group. Ties broken by the higher index, or not at all, make two runs of the same query order equal chunks differently; `TopK` and `RRF` break ties to the lower index. k-means++ by farthest point, or weighted by distance instead of its square, gives centroids that differ from every other implementation and makes outliers into seeds; `KMeansPP` draws proportional to `Dist2`. Ignoring `nprobe` collapses either recall (one list) or speed (all lists); `IVF.Search` scans `TopK(centroids, nprobe)`. Seeding IVF with a constant or the clock changes the lists between loads, so results change after a restart; `Index.BuildIVF` seeds with `FNV1a64(index_id)`.

The third group is fusion and scope. RRF with ranks from 0, or the max instead of the sum, no longer rewards agreement between lists; the solution sums `1 / (k + r + 1)` over lists. An MMR penalty against the last pick only, or an unnormalized query, lets near-copies come back after one diverse pick and puts relevance and redundancy on different scales; `maxSim` keeps the max over all picks, and `Retrieve` normalizes the query. Filtering after the cut returns fewer hits than exist, often none; `keep` is passed into every search. Not checking the query's dimension lets an index embedded by another model give garbage or a panic; `Retrieve` returns `ErrIndex` on a dimension mismatch.

## Where this leads

Looking back, [RAG ingest](06-rag-ingest.md) writes `docs.jsonl`, `vectors.f32`, and `bm25.idx` (calling `BuildBM25` and `WriteTo` from the ingest entry point) and embeds queries, and the [Go PCG32 port](../../build-an-llm/p10-serving/10-load-generator.md) supplies `rng.Seeded`, `Below`, and `Float64` for k-means++. Looking forward, the `search_docs` tool in [Context assembly, citations, search_docs tool](08-context-assembly.md) calls `Hybrid.Retrieve` and assembles the hits into cited context; the agent's cited answers over the course docs rest on it.

Each piece has a production counterpart:

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `BM25`, `bm25.idx` | Lucene, tantivy | skip lists over postings, block-max WAND to skip chunks that cannot reach the top k, field boosts, analyzers with stemming | [tantivy](https://github.com/quickwit-oss/tantivy) (free), the tantivy example in [Search & Indexing](../../infrastructure/04-search-and-indexing/) |
| `IVF` | FAISS `IndexIVFPQ`, pgvector `ivfflat`, HNSW | product quantization to shrink vectors, graph indexes with better recall per scan | [FAISS wiki](https://github.com/facebookresearch/faiss/wiki) (free) |
| `RRF` + `MMR` | saige `rag/internal/pipeline/pipeline.go`, cross-encoder rerankers | multi-retriever fusion, a reranker that reads query and chunk together, HyDE query expansion | [saige](https://github.com/urmzd/saige) (free) |

### Further reading

- Robertson and Zaragoza, [*The Probabilistic Relevance Framework: BM25 and Beyond*](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) (free).
- Cormack, Clarke, Buettcher, [*Reciprocal Rank Fusion*](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) (free).
- Arthur and Vassilvitskii, [*k-means++*](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf) (free).
- Carbonell and Goldstein, *The Use of MMR* (SIGIR 1998).

## Key terms

- **Term**: a maximal run of Unicode letters, digits, and `_`, lowercased; the unit both ingest and retrieval index.
- **Inverted index**: a map from each term to its postings list of chunks and term frequencies.
- **BM25**: a lexical ranking function with saturating term frequency (`k1`), length normalization (`b`), and an idf that is never negative.
- **Varint (unsigned LEB128)**: an integer encoding of 7 bits per byte, low group first, with a continuation bit on every byte but the last.
- **IVF (inverted file index)**: a vector index that clusters rows around centroids and scans only the `nprobe` nearest clusters.
- **k-means++**: a seeding rule that picks each new centroid with probability proportional to its squared distance from the nearest seed.
- **Recall@k**: the fraction of the exact top $k$ that an approximate search also returns.
- **RRF (reciprocal rank fusion)**: combining ranked lists by summing $1/(k_{\mathrm{rrf}} + r)$ over the lists containing each item.
- **MMR (maximal marginal relevance)**: greedy selection that trades relevance against similarity to items already picked.

## Summary

- BM25 scores a chunk by the query terms it holds, with term frequency that saturates (`k1`) and a penalty for long chunks (`b`); the `+1` inside the idf keeps every weight positive.
- An index file is a contract between two programs: the term rule, the byte layout, and the varint postings must match exactly, or ingest and retrieval disagree silently.
- IVF trades a little recall for speed by scanning only the `nprobe` lists nearest the query; seeded k-means++ makes those lists the same on every load.
- RRF fuses lexical and dense lists by rank alone, so scores on different scales never need calibrating; MMR then trades relevance for diversity.
- Filters run inside each search, before the cut to `k`; a filter after the cut returns fewer results than exist.

## Review questions

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
6. Encode 300 as an unsigned LEB128 varint.
   <details><summary>Answer</summary>300 is binary `100101100`. The low seven bits `0101100` (0x2c) go first with the continuation bit set, giving `0xac`; the remaining bits `10` give `0x02` with the top bit clear. The encoding is `ac 02`.</details>
