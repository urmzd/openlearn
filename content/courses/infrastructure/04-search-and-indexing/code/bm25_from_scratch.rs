//! From-scratch tokenizer + inverted index + BM25 scorer (no external crates).
//!
//! Build & run:  rustc --edition 2021 bm25_from_scratch.rs && ./bm25_from_scratch
//!
//! This mirrors `bm25_from_scratch.py`: an Apache-Lucene-style core that does
//! ANALYSIS -> INDEXING -> SCORING, where SCORING is Okapi BM25:
//!
//!   score(D,Q) = sum_{q in Q} IDF(q) * ( f(q,D)*(k1+1) )
//!                / ( f(q,D) + k1*(1 - b + b*|D|/avgdl) )
//!
//!   k1 : term-frequency saturation (extra occurrences matter less and less;
//!        TF factor -> k1+1 as f -> infinity).
//!   b  : length normalization (b=1 fully penalizes long docs, b=0 ignores len).
//!   IDF(q) = ln( 1 + (N - n(q) + 0.5)/(n(q) + 0.5) )   [Lucene's smoothed form]
//!
//! BM25 = TF-IDF + (a) TF saturation (k1) + (b) length normalization (b).

use std::collections::HashMap;

const STOPWORDS: &[&str] = &[
    "a", "an", "and", "the", "of", "to", "for", "in", "on", "is", "are",
];

/// One postings entry: this term occurs `tf` times in document `doc_id`.
struct Posting {
    doc_id: usize,
    tf: u32,
}

/// A deliberately tiny suffix stripper standing in for a Porter stemmer.
fn stem(token: &str) -> String {
    for suffix in ["ing", "ed", "es", "s"] {
        if token.ends_with(suffix) && token.len() - suffix.len() >= 3 {
            return token[..token.len() - suffix.len()].to_string();
        }
    }
    token.to_string()
}

/// Analysis chain: tokenize -> lowercase -> drop stopwords -> stem.
fn analyze(text: &str) -> Vec<String> {
    let lowered = text.to_lowercase();
    let mut out = Vec::new();
    // Tokenizer: maximal runs of ASCII alphanumerics.
    for raw in lowered.split(|c: char| !c.is_ascii_alphanumeric()) {
        if raw.is_empty() || STOPWORDS.contains(&raw) {
            continue;
        }
        out.push(stem(raw));
    }
    out
}

struct InvertedIndex {
    postings: HashMap<String, Vec<Posting>>, // term -> postings list
    doc_len: Vec<usize>,                      // |D| per doc, indexed by doc_id
    docs: Vec<String>,                        // original text for display
}

impl InvertedIndex {
    fn new() -> Self {
        InvertedIndex {
            postings: HashMap::new(),
            doc_len: Vec::new(),
            docs: Vec::new(),
        }
    }

    fn add(&mut self, text: &str) {
        let doc_id = self.docs.len();
        let terms = analyze(text);
        self.doc_len.push(terms.len());
        self.docs.push(text.to_string());

        let mut tf: HashMap<String, u32> = HashMap::new();
        for t in &terms {
            *tf.entry(t.clone()).or_insert(0) += 1;
        }
        for (term, count) in tf {
            self.postings
                .entry(term)
                .or_default()
                .push(Posting { doc_id, tf: count });
        }
    }

    fn n_docs(&self) -> usize {
        self.docs.len()
    }

    fn avgdl(&self) -> f64 {
        if self.doc_len.is_empty() {
            return 0.0;
        }
        self.doc_len.iter().sum::<usize>() as f64 / self.doc_len.len() as f64
    }

    /// Lucene's smoothed, non-negative IDF.
    fn idf(&self, term: &str) -> f64 {
        let n = self.n_docs() as f64;
        let nq = self.postings.get(term).map_or(0, |p| p.len()) as f64;
        (1.0 + (n - nq + 0.5) / (nq + 0.5)).ln()
    }

    fn bm25_search(&self, query: &str, k1: f64, b: f64) -> Vec<(usize, f64)> {
        let avgdl = self.avgdl();
        let mut scores: HashMap<usize, f64> = HashMap::new();

        for term in analyze(query) {
            let Some(plist) = self.postings.get(&term) else {
                continue;
            };
            let idf = self.idf(&term);
            for p in plist {
                let dl = self.doc_len[p.doc_id] as f64;
                let tf = p.tf as f64;
                let tf_component = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avgdl));
                *scores.entry(p.doc_id).or_insert(0.0) += idf * tf_component;
            }
        }

        let mut ranked: Vec<(usize, f64)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked
    }
}

fn main() {
    let corpus = [
        "Streamflow order 8801 for two wireless keyboards shipped to Berlin",
        "Order 8802 cancelled: customer returned the wireless mouse and keyboard",
        "Product catalog: ergonomic wireless keyboard, mechanical keyboard, trackball mouse",
        "Refund issued for order 8801 after the keyboard arrived damaged in shipping",
        "New product launch: noise cancelling headphones now in the Streamflow catalog",
    ];

    let mut idx = InvertedIndex::new();
    for text in corpus {
        idx.add(text);
    }

    println!(
        "Indexed {} docs | avgdl = {:.2} tokens",
        idx.n_docs(),
        idx.avgdl()
    );
    println!("Vocabulary size = {} terms\n", idx.postings.len());

    println!("Postings list for term 'keyboard' (stemmed):");
    if let Some(plist) = idx.postings.get("keyboard") {
        let mut sorted: Vec<&Posting> = plist.iter().collect();
        sorted.sort_by_key(|p| p.doc_id);
        for p in sorted {
            println!("  doc {}: tf={}", p.doc_id, p.tf);
        }
    }
    println!("  IDF('keyboard') = {:.4}\n", idx.idf("keyboard"));

    for query in ["wireless keyboard", "order 8801", "headphones"] {
        println!("Query: {:?}", query);
        let results = idx.bm25_search(query, 1.2, 0.75);
        if results.is_empty() {
            println!("  (no matches)");
        }
        for (doc_id, score) in results {
            println!("  score={:.4}  doc {}: {}", score, doc_id, idx.docs[doc_id]);
        }
        println!();
    }
}
