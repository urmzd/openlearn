// Package store is the RAG index on disk and the ingest pipeline that fills
// it (ag.06): source, chunker, embedder, and an index directory in the
// format of course/contracts/formats/rag-index.md (meta.json, docs.jsonl,
// vectors.f32; bm25.idx is written by retrieval, ag.07, from the same
// chunks). Documents are deduplicated by fingerprint: re-ingesting an
// unchanged document embeds nothing and writes nothing.
//
// Chapter: content/tracks/ai-platform-engineering/07-retrieval-and-rag/06-rag-ingest.md.
package store

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"tinyllm/agent/rag/chunk"
	"tinyllm/agent/rag/embed"
	"tinyllm/agent/rag/source"
)

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

// Entry is one chunk of the index: a docs.jsonl line plus its vector.
type Entry struct {
	ChunkID     string    `json:"chunk_id"`
	DocID       string    `json:"doc_id"`
	Source      string    `json:"source"`
	URI         string    `json:"uri"`
	Text        string    `json:"text"`
	NTokens     int       `json:"n_tokens"`
	Fingerprint string    `json:"fingerprint"`
	Vector      []float32 `json:"-"`
}

// Index is an index directory loaded in memory.
type Index struct {
	Dir     string
	Meta    Meta
	entries []Entry
	dirty   bool
}

// Open loads the index in dir, or starts an empty one with meta m when dir
// holds none (m.IndexID defaults to dir's base name).
func Open(dir string, m Meta) (*Index, error) {
	x := &Index{Dir: dir, Meta: m}
	if x.Meta.IndexID == "" {
		x.Meta.IndexID = filepath.Base(dir)
	}
	mb, err := os.ReadFile(filepath.Join(dir, "meta.json"))
	if errors.Is(err, os.ErrNotExist) {
		return x, nil
	}
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(mb, &x.Meta); err != nil {
		return nil, fmt.Errorf("store: meta.json: %w", err)
	}
	db, err := os.ReadFile(filepath.Join(dir, "docs.jsonl"))
	if err != nil {
		return nil, err
	}
	sc := bufio.NewScanner(bytes.NewReader(db))
	sc.Buffer(make([]byte, 1<<20), 64<<20)
	for sc.Scan() {
		var e Entry
		if err := json.Unmarshal(sc.Bytes(), &e); err != nil {
			return nil, fmt.Errorf("store: docs.jsonl line %d: %w", len(x.entries)+1, err)
		}
		x.entries = append(x.entries, e)
	}
	vb, err := os.ReadFile(filepath.Join(dir, "vectors.f32"))
	if err != nil {
		return nil, err
	}
	n, dim := len(x.entries), x.Meta.Dim
	if n != x.Meta.NChunks || len(vb) != n*dim*4 {
		return nil, fmt.Errorf("store: %d chunks, %d vector bytes; meta says %d x %d", n, len(vb), x.Meta.NChunks, dim)
	}
	for i := range x.entries {
		v := make([]float32, dim)
		for j := range v {
			v[j] = math.Float32frombits(binary.LittleEndian.Uint32(vb[(i*dim+j)*4:]))
		}
		x.entries[i].Vector = v
	}
	return x, nil
}

// Entries are the chunks in index order (docs.jsonl line order).
func (x *Index) Entries() []Entry { return append([]Entry(nil), x.entries...) }

// docEntries are the stored chunks of one document, in order.
func (x *Index) docEntries(docID string) []Entry {
	var out []Entry
	for _, e := range x.entries {
		if e.DocID == docID {
			out = append(out, e)
		}
	}
	return out
}

// Upsert replaces docID's chunks with es (in place when the document was
// already indexed, at the end otherwise). Vectors are L2-normalized.
func (x *Index) Upsert(docID string, es []Entry) error {
	for i := range es {
		v := es[i].Vector
		if x.Meta.Dim == 0 {
			x.Meta.Dim = len(v)
		}
		if len(v) != x.Meta.Dim || len(v) == 0 {
			return fmt.Errorf("store: %s has %d dimensions, the index %d", es[i].ChunkID, len(v), x.Meta.Dim)
		}
		var ss float64
		for _, f := range v {
			ss += float64(f) * float64(f)
		}
		norm := math.Sqrt(ss)
		if norm == 0 {
			return fmt.Errorf("store: %s has a zero vector", es[i].ChunkID)
		}
		nv := make([]float32, len(v))
		for j, f := range v {
			nv[j] = float32(float64(f) / norm)
		}
		es[i].Vector = nv
	}
	var out []Entry
	placed := false
	for _, e := range x.entries {
		if e.DocID == docID {
			if !placed {
				out = append(out, es...)
				placed = true
			}
			continue
		}
		out = append(out, e)
	}
	if !placed {
		out = append(out, es...)
	}
	x.entries = out
	x.dirty = true
	return nil
}

// Save writes the three files when anything changed since Open, each to a
// temporary name, fsynced, then renamed over the old one, so a reader never
// sees half a file. It reports whether it wrote.
func (x *Index) Save(now time.Time) (bool, error) {
	if !x.dirty {
		return false, nil
	}
	if err := os.MkdirAll(x.Dir, 0o755); err != nil {
		return false, err
	}
	if x.Meta.CreatedAt == "" {
		x.Meta.CreatedAt = now.UTC().Format(time.RFC3339)
	}
	x.Meta.NChunks = len(x.entries)
	var docs bytes.Buffer
	vec := make([]byte, 0, len(x.entries)*x.Meta.Dim*4)
	for _, e := range x.entries {
		line, err := json.Marshal(e)
		if err != nil {
			return false, err
		}
		docs.Write(line)
		docs.WriteByte('\n')
		for _, f := range e.Vector {
			vec = binary.LittleEndian.AppendUint32(vec, math.Float32bits(f))
		}
	}
	meta, err := json.Marshal(x.Meta)
	if err != nil {
		return false, err
	}
	for _, f := range []struct {
		name string
		data []byte
	}{{"docs.jsonl", docs.Bytes()}, {"vectors.f32", vec}, {"meta.json", append(meta, '\n')}} {
		if err := writeAtomic(filepath.Join(x.Dir, f.name), f.data); err != nil {
			return false, err
		}
	}
	x.dirty = false
	return true, nil
}

func writeAtomic(path string, data []byte) error {
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := f.Write(data); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// Stats counts what one Ingest did.
type Stats struct {
	Docs      int // documents fetched
	Unchanged int // documents whose chunks were all already stored
	Chunks    int // chunks stored for changed documents
	Embedded  int // texts sent to the embedder
}

// Ingest fetches, chunks, embeds, and upserts. A document whose chunk
// fingerprints equal the stored ones is skipped (nothing embedded, nothing
// marked dirty); for a changed document only chunks with a fingerprint the
// document did not have before are embedded, the others keep their vectors.
func Ingest(ctx context.Context, src source.Source, ch chunk.Chunker, em embed.Embedder, x *Index) (Stats, error) {
	var st Stats
	raws, err := src.Fetch(ctx)
	if err != nil {
		return st, err
	}
	for _, raw := range raws {
		st.Docs++
		d := source.ToDoc(raw)
		chunks, err := ch.Chunk(ctx, d)
		if err != nil {
			return st, fmt.Errorf("store: chunking %s: %w", d.ID, err)
		}
		old := x.docEntries(d.ID)
		if len(old) == len(chunks) {
			same := true
			for i := range chunks {
				same = same && old[i].Fingerprint == chunks[i].Fingerprint
			}
			if same {
				st.Unchanged++
				continue
			}
		}
		known := map[string][]float32{}
		for _, e := range old {
			known[e.Fingerprint] = e.Vector
		}
		var need []string
		var needAt []int
		es := make([]Entry, len(chunks))
		for i, c := range chunks {
			es[i] = Entry{ChunkID: c.ID, DocID: d.ID, Source: d.Source, URI: d.URI, Text: c.Text, NTokens: c.Tokens, Fingerprint: c.Fingerprint, Vector: known[c.Fingerprint]}
			if es[i].Vector == nil {
				need = append(need, c.Text)
				needAt = append(needAt, i)
			}
		}
		if len(need) > 0 {
			vecs, err := em.Embed(ctx, need)
			if err != nil {
				return st, fmt.Errorf("store: embedding %s: %w", d.ID, err)
			}
			if len(vecs) != len(need) {
				return st, fmt.Errorf("store: %d vectors for %d texts", len(vecs), len(need))
			}
			for k, i := range needAt {
				es[i].Vector = vecs[k]
			}
			st.Embedded += len(need)
		}
		if err := x.Upsert(d.ID, es); err != nil {
			return st, err
		}
		st.Chunks += len(es)
	}
	return st, nil
}
