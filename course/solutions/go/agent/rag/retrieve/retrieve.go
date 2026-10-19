// Package retrieve is the agent SDK's retrieval layer (ag.07): it opens a
// RAG index directory (course/contracts/formats/rag-index.md) that ingest
// (ag.06) wrote, and answers a query with ranked chunks from BM25, dense
// vectors (flat or IVF), reciprocal rank fusion, and optional maximal
// marginal relevance.
//
// The query embedder is ag.06's: Embedder below is the consumer-side
// interface its embedder satisfies, so this package never imports the
// ingest packages and the index files are the only contract between them.
//
// Chapter: content/tracks/ai-platform-engineering/07-retrieval-and-rag/07-retrieval.md.
package retrieve

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"tinyllm/ds/rng"
)

// Chunk is one line of docs.jsonl.
type Chunk struct {
	ChunkID     string `json:"chunk_id"`
	DocID       string `json:"doc_id"`
	Source      string `json:"source"`
	URI         string `json:"uri"`
	Text        string `json:"text"`
	NTokens     int    `json:"n_tokens"`
	Fingerprint string `json:"fingerprint"`
}

// Meta is meta.json.
type Meta struct {
	IndexID        string `json:"index_id"`
	CreatedAt      string `json:"created_at"`
	NChunks        int    `json:"n_chunks"`
	Dim            int    `json:"dim"`
	EmbeddingModel string `json:"embedding_model"`
	Tokenizer      string `json:"tokenizer"`
	Chunker        struct {
		MaxTokens int `json:"max_tokens"`
		Overlap   int `json:"overlap"`
	} `json:"chunker"`
	BM25 struct {
		K1 float64 `json:"k1"`
		B  float64 `json:"b"`
	} `json:"bm25"`
}

// Hit is one retrieved chunk. Index is its line in docs.jsonl (its row in
// vectors.f32, its document number in bm25.idx); Rank counts from 1.
type Hit struct {
	Index   int
	ChunkID string
	DocID   string
	URI     string
	Text    string
	NTokens int
	Score   float64
	Rank    int
}

// Filter restricts retrieval; an empty field accepts everything.
type Filter struct {
	DocIDs  []string // keep chunks of these documents
	Sources []string // keep chunks from these sources (file, web, ...)
}

// Keep reports whether c passes the filter.
func (f Filter) Keep(c Chunk) bool {
	in := func(s string, set []string) bool {
		if len(set) == 0 {
			return true
		}
		for _, x := range set {
			if x == s {
				return true
			}
		}
		return false
	}
	return in(c.DocID, f.DocIDs) && in(c.Source, f.Sources)
}

// Embedder turns texts into vectors (ag.06's embedder satisfies it).
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// Retriever answers a query with at most k chunks, best first.
type Retriever interface {
	Retrieve(ctx context.Context, q string, k int, f Filter) ([]Hit, error)
}

// Index is one opened RAG index.
type Index struct {
	Meta   Meta
	Chunks []Chunk
	Vecs   [][]float32
	BM25   *BM25
	IVF    *IVF // nil: dense search is flat
	byID   map[string]int
}

// ErrIndex wraps every inconsistency between the index files.
var ErrIndex = errors.New("rag index: inconsistent")

// Open reads dir/meta.json, docs.jsonl, vectors.f32, and bm25.idx and
// checks that they agree: n_chunks lines, n_chunks * dim * 4 vector bytes,
// n_chunks BM25 documents, unique chunk ids of the form <doc_id>#<k>.
func Open(dir string) (*Index, error) {
	bad := func(f string, a ...any) error { return fmt.Errorf("%w: %s", ErrIndex, fmt.Sprintf(f, a...)) }
	var x Index
	mb, err := os.ReadFile(filepath.Join(dir, "meta.json"))
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(mb, &x.Meta); err != nil {
		return nil, bad("meta.json: %v", err)
	}
	n, dim := x.Meta.NChunks, x.Meta.Dim
	df, err := os.Open(filepath.Join(dir, "docs.jsonl"))
	if err != nil {
		return nil, err
	}
	defer df.Close()
	sc := bufio.NewScanner(df)
	sc.Buffer(make([]byte, 1<<20), 64<<20)
	x.byID = map[string]int{}
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var c Chunk
		if err := json.Unmarshal([]byte(line), &c); err != nil {
			return nil, bad("docs.jsonl line %d: %v", len(x.Chunks)+1, err)
		}
		if !strings.HasPrefix(c.ChunkID, c.DocID+"#") {
			return nil, bad("chunk id %q is not <doc_id>#<k>", c.ChunkID)
		}
		if _, dup := x.byID[c.ChunkID]; dup {
			return nil, bad("chunk id %q appears twice", c.ChunkID)
		}
		x.byID[c.ChunkID] = len(x.Chunks)
		x.Chunks = append(x.Chunks, c)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if len(x.Chunks) != n {
		return nil, bad("docs.jsonl has %d chunks, meta.json says %d", len(x.Chunks), n)
	}
	vb, err := os.ReadFile(filepath.Join(dir, "vectors.f32"))
	if err != nil {
		return nil, err
	}
	if len(vb) != n*dim*4 {
		return nil, bad("vectors.f32 has %d bytes, want %d x %d x 4 = %d", len(vb), n, dim, n*dim*4)
	}
	x.Vecs = make([][]float32, n)
	for i := range x.Vecs {
		row := make([]float32, dim)
		for j := range row {
			row[j] = math.Float32frombits(binary.LittleEndian.Uint32(vb[(i*dim+j)*4:]))
		}
		x.Vecs[i] = row
	}
	bf, err := os.Open(filepath.Join(dir, "bm25.idx"))
	if err != nil {
		return nil, err
	}
	defer bf.Close()
	if x.BM25, err = ReadBM25(bf, x.Meta.BM25.K1, x.Meta.BM25.B); err != nil {
		return nil, err
	}
	if x.BM25.N() != n {
		return nil, bad("bm25.idx has %d documents, meta.json says %d", x.BM25.N(), n)
	}
	return &x, nil
}

// Chunk returns the chunk with id, if the index holds it.
func (x *Index) Chunk(id string) (Chunk, bool) {
	i, ok := x.byID[id]
	if !ok {
		return Chunk{}, false
	}
	return x.Chunks[i], true
}

// FNV1a64 is 64-bit FNV-1a over the bytes of s.
func FNV1a64(s string) uint64 {
	h := uint64(0xcbf29ce484222325)
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= 0x100000001b3
	}
	return h
}

// BuildIVF gives the index IVF dense search with nlist centroids, seeding
// PCG32 with FNV1a64(index_id) (rng.Seeded), so every load of the same index
// builds the same centroids.
func (x *Index) BuildIVF(nlist, iters int) error {
	ivf, err := BuildIVF(x.Vecs, nlist, iters, rng.Seeded(FNV1a64(x.Meta.IndexID)))
	if err != nil {
		return err
	}
	x.IVF = ivf
	return nil
}

// hit is chunk i as a Hit with the given score.
func (x *Index) hit(i int, score float64) Hit {
	c := x.Chunks[i]
	return Hit{Index: i, ChunkID: c.ChunkID, DocID: c.DocID, URI: c.URI, Text: c.Text, NTokens: c.NTokens, Score: score}
}

// Hybrid is the retriever the agent uses: BM25 and dense candidates, fused
// by RRF, then MMR when Lambda > 0.
type Hybrid struct {
	Index      *Index
	Embedder   Embedder // nil: lexical only
	Candidates int      // per list; 0 means 50 (always at least k)
	RRFK       float64  // 0 means DefaultRRFK
	NProbe     int      // IVF lists scanned; 0 means 8
	Lambda     float64  // MMR trade-off in (0, 1]; 0 turns MMR off
}

// Retrieve implements Retriever. The filter is applied inside each search,
// before any cut, so a filtered query still gets k hits when k chunks pass.
// The query vector is L2-normalized and must have the index's dim.
func (h *Hybrid) Retrieve(ctx context.Context, q string, k int, f Filter) ([]Hit, error) {
	x := h.Index
	if k <= 0 {
		return nil, nil
	}
	n := h.Candidates
	if n <= 0 {
		n = 50
	}
	n = max(n, k)
	rrfk := h.RRFK
	if rrfk == 0 {
		rrfk = DefaultRRFK
	}
	nprobe := h.NProbe
	if nprobe <= 0 {
		nprobe = 8
	}
	keep := func(i int) bool { return f.Keep(x.Chunks[i]) }
	var lists [][]Hit
	var lex []Hit
	for _, s := range x.BM25.Search(q, n, keep) {
		lex = append(lex, x.hit(s.Doc, s.Score))
	}
	lists = append(lists, lex)
	var qv []float32
	if h.Embedder != nil {
		vs, err := h.Embedder.Embed(ctx, []string{q})
		if err != nil {
			return nil, fmt.Errorf("embed query: %w", err)
		}
		if len(vs) != 1 {
			return nil, fmt.Errorf("embed query: %d vectors for 1 text", len(vs))
		}
		if len(vs[0]) != x.Meta.Dim {
			return nil, fmt.Errorf("%w: query embedding has dim %d, index has %d", ErrIndex, len(vs[0]), x.Meta.Dim)
		}
		qv = Normalize(vs[0])
		var dense []Scored
		if x.IVF != nil {
			dense = x.IVF.Search(qv, n, nprobe, keep)
		} else {
			dense = (&Flat{Vecs: x.Vecs}).Search(qv, n, keep)
		}
		var dl []Hit
		for _, s := range dense {
			dl = append(dl, x.hit(s.Doc, s.Score))
		}
		lists = append(lists, dl)
	}
	fused := RRF(lists, rrfk, n)
	if h.Lambda > 0 && qv != nil {
		return MMR(qv, fused, func(i int) []float32 { return x.Vecs[i] }, h.Lambda, k), nil
	}
	if len(fused) > k {
		fused = fused[:k]
	}
	return fused, nil
}
