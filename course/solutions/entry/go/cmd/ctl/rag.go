package main

// The rag verb (ag.06): build or refresh a RAG index from a directory.
//
//	ctl rag ingest --source <dir> --index <dir> [--max-tokens <n>]
//
// Documents are the .md and .txt files under --source, chunked by bytes (the
// tracer tokenizer) and embedded with hash16 (cmd/internal/hashembed), so the
// index builds with no model running; [build] runs it on docs/ for the
// course-docs suite. Unchanged documents are skipped by fingerprint. The
// index directory gets meta.json, docs.jsonl, vectors.f32, and bm25.idx
// (formats/rag-index.md).
//
// Final JSON line: {index, docs, unchanged, chunks, embedded, n_chunks}.

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"time"

	"tinyllm/agent/rag/chunk"
	"tinyllm/agent/rag/retrieve"
	"tinyllm/agent/rag/source"
	"tinyllm/agent/rag/store"
	"tinyllm/cmd/internal/hashembed"
)

func init() {
	register("rag", func(args []string) error { return subverbs("rag", map[string]verb{"ingest": ragIngest}, args) })
}

func ragIngest(args []string) error {
	fs := newFlags("rag ingest")
	src := fs.String("source", "", "directory of .md and .txt documents")
	dir := fs.String("index", "", "index directory")
	maxTok := fs.Int("max-tokens", 512, "chunk budget in tokens (bytes)")
	jsonFlag(fs)
	pos, err := parse(fs, args)
	if err != nil {
		return err
	}
	if err := exactly("rag ingest", pos, 0, "no positional arguments"); err != nil {
		return err
	}
	if *src == "" || *dir == "" {
		return usagef("rag ingest: --source and --index are required")
	}
	if st, err := os.Stat(*src); err != nil || !st.IsDir() {
		return usagef("rag ingest: --source %s is not a directory", *src)
	}
	var m store.Meta
	m.Dim, m.EmbeddingModel, m.Tokenizer = hashembed.Dim, hashembed.Model, "bytes"
	m.Chunker.MaxTokens = *maxTok
	m.BM25.K1, m.BM25.B = 1.2, 0.75
	x, err := store.Open(*dir, m)
	if err != nil {
		return err
	}
	ctx := context.Background()
	st, err := store.Ingest(ctx, source.DirSource{Root: *src}, chunk.TokenChunker{Tok: chunk.Bytes{}, MaxTokens: *maxTok}, hashembed.Embedder{}, x)
	if err != nil {
		return err
	}
	if _, err := x.Save(time.Now()); err != nil {
		return err
	}
	// bm25.idx follows docs.jsonl: rebuilt whenever the documents are.
	entries := x.Entries()
	texts := make([]string, len(entries))
	for i, e := range entries {
		texts[i] = e.Text
	}
	var b bytes.Buffer
	if _, err := retrieve.BuildBM25(texts, x.Meta.BM25.K1, x.Meta.BM25.B).WriteTo(&b); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(*dir, "bm25.idx"), b.Bytes(), 0o644); err != nil {
		return err
	}
	if _, err := retrieve.Open(*dir); err != nil {
		return err
	}
	return emit(map[string]any{"index": *dir, "docs": st.Docs, "unchanged": st.Unchanged, "chunks": st.Chunks,
		"embedded": st.Embedded, "n_chunks": len(entries)})
}
