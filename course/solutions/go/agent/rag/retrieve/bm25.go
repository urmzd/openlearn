package retrieve

// bm25.go: the lexical half of retrieval. Terms, an inverted index with
// varint postings, BM25 scoring, and the bm25.idx file of
// course/contracts/formats/rag-index.md.

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"regexp"
	"sort"
	"strings"
)

// termRE is the contract's term rule: maximal runs of Unicode letters,
// digits, and underscore.
var termRE = regexp.MustCompile(`[\p{L}\p{N}_]+`)

// Terms splits text into its terms, lowercased with strings.ToLower, in
// order, duplicates kept.
func Terms(text string) []string {
	runs := termRE.FindAllString(text, -1)
	out := make([]string, len(runs))
	for i, r := range runs {
		out[i] = strings.ToLower(r)
	}
	return out
}

// Posting is one document in a term's postings list.
type Posting struct {
	Doc int // document (chunk) number
	TF  int // occurrences of the term in that document
}

// BM25 is an inverted index with the statistics BM25 needs. Postings lists
// are sorted by Doc.
type BM25 struct {
	K1, B    float64
	DocLen   []int // terms per document
	AvgDL    float64
	Postings map[string][]Posting
}

// BuildBM25 indexes texts; document i is texts[i].
func BuildBM25(texts []string, k1, b float64) *BM25 {
	idx := &BM25{K1: k1, B: b, DocLen: make([]int, len(texts)), Postings: map[string][]Posting{}}
	total := 0
	for d, text := range texts {
		terms := Terms(text)
		idx.DocLen[d] = len(terms)
		total += len(terms)
		tf := map[string]int{}
		var order []string
		for _, t := range terms {
			if tf[t] == 0 {
				order = append(order, t)
			}
			tf[t]++
		}
		for _, t := range order {
			idx.Postings[t] = append(idx.Postings[t], Posting{Doc: d, TF: tf[t]})
		}
	}
	if len(texts) > 0 {
		idx.AvgDL = float64(total) / float64(len(texts))
	}
	return idx
}

// N is the number of documents.
func (x *BM25) N() int { return len(x.DocLen) }

// IDF is ln(1 + (N - df + 0.5) / (df + 0.5)): never negative, so a term in
// every document still counts a little instead of pushing documents down.
func (x *BM25) IDF(term string) float64 {
	n := float64(x.N())
	df := float64(len(x.Postings[term]))
	return math.Log(1 + (n-df+0.5)/(df+0.5))
}

// Scores is the BM25 score of every document for query q: for each query
// term occurrence (a repeated query term counts each time) and each
// document containing it, idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b *
// dl / avgdl)). A document matching no query term scores 0.
func (x *BM25) Scores(q string) []float64 {
	s := make([]float64, x.N())
	if x.AvgDL == 0 {
		return s
	}
	for _, t := range Terms(q) {
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

// Search returns up to k documents that match at least one query term, best
// first; equal scores go to the lower document number. keep (nil: all)
// says which documents may be returned, and is applied before the cut to k.
func (x *BM25) Search(q string, k int, keep func(doc int) bool) []Scored {
	matched := make([]bool, x.N())
	for _, t := range Terms(q) {
		for _, p := range x.Postings[t] {
			matched[p.Doc] = true
		}
	}
	scores := x.Scores(q)
	var c []Scored
	for d, ok := range matched {
		if ok && (keep == nil || keep(d)) {
			c = append(c, Scored{Doc: d, Score: scores[d]})
		}
	}
	return TopK(c, k)
}

// Scored is one document and its score.
type Scored struct {
	Doc   int
	Score float64
}

// TopK sorts c by score descending, ties to the lower Doc, and keeps the
// first k (all when k <= 0 or k >= len(c)).
func TopK(c []Scored, k int) []Scored {
	sort.Slice(c, func(i, j int) bool {
		if c[i].Score != c[j].Score {
			return c[i].Score > c[j].Score
		}
		return c[i].Doc < c[j].Doc
	})
	if k > 0 && k < len(c) {
		c = c[:k]
	}
	return c
}

// ---- bm25.idx ----------------------------------------------------------------

// Magic opens every bm25.idx file.
const Magic = "TLBM"

// ErrFormat wraps every malformed-index error.
var ErrFormat = errors.New("bm25.idx: malformed")

// PutUvarint appends v as unsigned LEB128: seven bits per byte, low group
// first, the high bit set on every byte but the last.
func PutUvarint(buf []byte, v uint64) []byte {
	for v >= 0x80 {
		buf = append(buf, byte(v)|0x80)
		v >>= 7
	}
	return append(buf, byte(v))
}

// Uvarint decodes one unsigned LEB128 value from the front of buf and
// returns it with the number of bytes read; n == 0 means buf ended in the
// middle of a value or the value does not fit 64 bits.
func Uvarint(buf []byte) (v uint64, n int) {
	var shift uint
	for i, c := range buf {
		if i == 10 || (i == 9 && c > 1) {
			return 0, 0
		}
		v |= uint64(c&0x7f) << shift
		if c < 0x80 {
			return v, i + 1
		}
		shift += 7
	}
	return 0, 0
}

// WriteTo writes x in the bm25.idx layout: header, doc lengths, then the
// terms sorted by their bytes, each with its postings as (doc gap, tf)
// varint pairs, the first gap being the doc number itself.
func (x *BM25) WriteTo(w io.Writer) (int64, error) {
	terms := make([]string, 0, len(x.Postings))
	for t := range x.Postings {
		terms = append(terms, t)
	}
	sort.Strings(terms)
	var b []byte
	b = append(b, Magic...)
	b = binary.LittleEndian.AppendUint32(b, 1)
	b = binary.LittleEndian.AppendUint32(b, uint32(x.N()))
	b = binary.LittleEndian.AppendUint32(b, uint32(len(terms)))
	b = binary.LittleEndian.AppendUint64(b, math.Float64bits(x.AvgDL))
	for _, dl := range x.DocLen {
		b = binary.LittleEndian.AppendUint32(b, uint32(dl))
	}
	for _, t := range terms {
		pl := x.Postings[t]
		var post []byte
		prev := 0
		for _, p := range pl {
			post = PutUvarint(post, uint64(p.Doc-prev))
			post = PutUvarint(post, uint64(p.TF))
			prev = p.Doc
		}
		b = binary.LittleEndian.AppendUint16(b, uint16(len(t)))
		b = append(b, t...)
		b = binary.LittleEndian.AppendUint32(b, uint32(len(pl)))
		b = binary.LittleEndian.AppendUint32(b, uint32(len(post)))
		b = append(b, post...)
	}
	n, err := w.Write(b)
	return int64(n), err
}

// ReadBM25 reads a bm25.idx; k1 and b come from meta.json. Every length is
// checked against the bytes that remain, so a truncated or corrupt file is
// an error wrapping ErrFormat, never a panic or a silently short index.
func ReadBM25(r io.Reader, k1, b float64) (*BM25, error) {
	data, err := io.ReadAll(bufio.NewReader(r))
	if err != nil {
		return nil, err
	}
	bad := func(what string) error { return fmt.Errorf("%w: %s", ErrFormat, what) }
	if len(data) < 24 || string(data[:4]) != Magic {
		return nil, bad("no TLBM header")
	}
	if v := binary.LittleEndian.Uint32(data[4:]); v != 1 {
		return nil, bad(fmt.Sprintf("version %d", v))
	}
	nDocs := int(binary.LittleEndian.Uint32(data[8:]))
	nTerms := int(binary.LittleEndian.Uint32(data[12:]))
	x := &BM25{K1: k1, B: b, AvgDL: math.Float64frombits(binary.LittleEndian.Uint64(data[16:])), Postings: map[string][]Posting{}}
	off := 24
	if uint64(len(data)-off) < 4*uint64(nDocs) {
		return nil, bad("doc lengths truncated")
	}
	x.DocLen = make([]int, nDocs)
	for i := range x.DocLen {
		x.DocLen[i] = int(binary.LittleEndian.Uint32(data[off:]))
		off += 4
	}
	prevTerm := ""
	for i := 0; i < nTerms; i++ {
		if len(data)-off < 2 {
			return nil, bad("term truncated")
		}
		tl := int(binary.LittleEndian.Uint16(data[off:]))
		off += 2
		if len(data)-off < tl+8 {
			return nil, bad("term truncated")
		}
		term := string(data[off : off+tl])
		off += tl
		if i > 0 && term <= prevTerm {
			return nil, bad("terms not sorted")
		}
		prevTerm = term
		df := int(binary.LittleEndian.Uint32(data[off:]))
		pb := int(binary.LittleEndian.Uint32(data[off+4:]))
		off += 8
		if pb < 0 || len(data)-off < pb {
			return nil, bad("postings truncated")
		}
		post := data[off : off+pb]
		off += pb
		pl := make([]Posting, 0, df)
		doc := 0
		for j := 0; j < df; j++ {
			gap, n := Uvarint(post)
			if n == 0 {
				return nil, bad("postings varint")
			}
			post = post[n:]
			tf, n := Uvarint(post)
			if n == 0 {
				return nil, bad("postings varint")
			}
			post = post[n:]
			if j > 0 && gap == 0 {
				return nil, bad("postings not increasing")
			}
			doc += int(gap)
			if doc >= nDocs {
				return nil, bad("posting past the last document")
			}
			pl = append(pl, Posting{Doc: doc, TF: int(tf)})
		}
		if len(post) != 0 {
			return nil, bad("postings longer than df pairs")
		}
		x.Postings[term] = pl
	}
	if off != len(data) {
		return nil, bad("trailing bytes")
	}
	return x, nil
}
