//! Tantivy (Rust, Lucene-inspired) BM25 search over a tiny Streamflow corpus.
//!
//! REQUIRES NETWORK TO BUILD (fetches the `tantivy` crate from crates.io):
//!     cd tantivy-example && cargo run
//!
//! Where `../bm25_from_scratch.{py,rs}` build the inverted index and BM25
//! scorer by hand, this shows the *production* path: Tantivy is a real,
//! Lucene-architecture engine -- it has the same moving parts:
//!
//!   * Schema + analyzers      (here: the default `en_stem` text analyzer:
//!                              tokenize -> lowercase -> stopwords -> stemming)
//!   * Immutable SEGMENTS written by an IndexWriter, merged in the background
//!     (the LSM-tree-like write path; `commit()` makes writes searchable)
//!   * An inverted index per segment (term dict as an FST, postings lists)
//!   * BM25 as the default scorer (same formula as the from-scratch files)
//!
//! Tantivy is the embeddable, single-process answer (like Lucene-as-a-library
//! or Bleve in Go) -- you reach for Elasticsearch/OpenSearch/Solr only when you
//! need the *distributed* layer (sharding, replication, a cluster).

use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Schema, Value, STORED, TEXT};
use tantivy::{doc, Index, TantivyDocument};

fn main() -> tantivy::Result<()> {
    // 1. SCHEMA: `body` is analyzed/indexed text; STORED so we can print it.
    let mut schema_builder = Schema::builder();
    let id = schema_builder.add_u64_field("id", STORED);
    let body = schema_builder.add_text_field("body", TEXT | STORED);
    let schema = schema_builder.build();

    // 2. INDEX in a temp dir on disk (segments are written here).
    let dir = tempfile::tempdir()?;
    let index = Index::create_in_dir(dir.path(), schema.clone())?;

    // 3. WRITE: the IndexWriter buffers docs, then commit() flushes a segment.
    let mut writer = index.writer(15_000_000)?; // 15 MB heap budget
    let corpus = [
        "Streamflow order 8801 for two wireless keyboards shipped to Berlin",
        "Order 8802 cancelled: customer returned the wireless mouse and keyboard",
        "Product catalog: ergonomic wireless keyboard, mechanical keyboard, trackball mouse",
        "Refund issued for order 8801 after the keyboard arrived damaged in shipping",
        "New product launch: noise cancelling headphones now in the Streamflow catalog",
    ];
    for (i, text) in corpus.iter().enumerate() {
        writer.add_document(doc!(id => i as u64, body => *text))?;
    }
    writer.commit()?; // <- writes the segment; now searchable (near-real-time)

    // 4. SEARCH: BM25 is Tantivy's default scorer.
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let query_parser = QueryParser::for_index(&index, vec![body]);

    for q in ["wireless keyboard", "order 8801", "headphones"] {
        println!("Query: {q:?}");
        let query = query_parser.parse_query(q)?;
        let top = searcher.search(&query, &TopDocs::with_limit(5))?;
        for (score, addr) in top {
            let retrieved: TantivyDocument = searcher.doc(addr)?;
            let text = retrieved
                .get_first(body)
                .and_then(|v| v.as_str())
                .unwrap_or("");
            println!("  score={score:.4}  {text}");
        }
        println!();
    }

    Ok(())
}
