package ag_07

// Frozen test helpers: the fixture loader, the feature-hashing embedder the
// fixture index was built with (course/oracle/ag.07/make_fixtures.py), and a
// SplitMix64 generator for synthetic vectors. None of them calls learner
// code, so a bug in your units cannot change what these tests expect.

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func fixtureDir(t *testing.T) string {
	t.Helper()
	dir := os.Getenv("TINYLLM_FIXTURES")
	if dir == "" {
		t.Fatal("TINYLLM_FIXTURES is not set (ol check sets it)")
	}
	return filepath.Join(dir, "ag.07")
}

type scored [2]float64 // (doc, score) as JSON numbers

type queryCase struct {
	Q          string    `json:"q"`
	Terms      []string  `json:"terms"`
	BM25       []float64 `json:"bm25"`
	LexicalTop []scored  `json:"lexical_top"`
	FlatTop    []scored  `json:"flat_top"`
	Hybrid5    [][2]any  `json:"hybrid5"`
	Hybrid5Web [][2]any  `json:"hybrid5_web"`
	MMR4       [][2]any  `json:"mmr4"`
}

type expected struct {
	K1, B    float64
	Dim      int
	AvgDL    float64 `json:"avgdl"`
	NTerms   int     `json:"n_terms"`
	Queries  []queryCase
	KMeansPP []struct {
		Seed  uint64 `json:"seed"`
		K     int    `json:"k"`
		Picks []int  `json:"picks"`
	} `json:"kmeanspp"`
}

func loadExpected(t *testing.T) expected {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(fixtureDir(t), "expected.json"))
	if err != nil {
		t.Fatal(err)
	}
	var e expected
	if err := json.Unmarshal(b, &e); err != nil {
		t.Fatal(err)
	}
	return e
}

// fixtureTexts reads docs.jsonl in order.
func fixtureTexts(t *testing.T) []string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(fixtureDir(t), "index", "docs.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	var out []string
	for _, line := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		var c struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal([]byte(line), &c); err != nil {
			t.Fatal(err)
		}
		out = append(out, c.Text)
	}
	return out
}

var testTermRE = regexp.MustCompile(`[\p{L}\p{N}_]+`)

func fnv(s string) uint64 {
	h := uint64(0xcbf29ce484222325)
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= 0x100000001b3
	}
	return h
}

// hashEmbedder is the fixture's embedder: each term adds +1 or -1 (the top
// bit of fnv1a64(term)) to coordinate fnv1a64(term) % dim. It returns the raw
// counts; the retriever must normalize the query itself.
type hashEmbedder struct {
	dim   int
	calls int
}

func (h *hashEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	h.calls++
	out := make([][]float32, len(texts))
	for i, text := range texts {
		v := make([]float32, h.dim)
		for _, term := range testTermRE.FindAllString(text, -1) {
			x := fnv(strings.ToLower(term))
			if x>>63 == 1 {
				v[x%uint64(h.dim)]--
			} else {
				v[x%uint64(h.dim)]++
			}
		}
		out[i] = v
	}
	return out, nil
}

// splitmix is a frozen SplitMix64 stream for synthetic test data.
type splitmix struct{ s uint64 }

func (r *splitmix) next() uint64 {
	r.s += 0x9E3779B97F4A7C15
	z := r.s
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

func (r *splitmix) float() float64 { return float64(r.next()>>11) / (1 << 53) }

func (r *splitmix) normal() float64 {
	u1, u2 := r.float(), r.float()
	return math.Sqrt(-2*math.Log(1-u1)) * math.Cos(2*math.Pi*u2)
}

func unit(v []float64) []float32 {
	var s float64
	for _, x := range v {
		s += x * x
	}
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(x / math.Sqrt(s))
	}
	return out
}

// clustered draws n unit vectors of dimension dim around c random centres.
func clustered(seed uint64, n, dim, c int, noise float64) [][]float32 {
	r := &splitmix{s: seed}
	centres := make([][]float64, c)
	for i := range centres {
		centres[i] = make([]float64, dim)
		for j := range centres[i] {
			centres[i][j] = r.normal()
		}
	}
	out := make([][]float32, n)
	for i := range out {
		ctr := centres[r.next()%uint64(c)]
		v := make([]float64, dim)
		for j := range v {
			v[j] = ctr[j] + noise*r.normal()
		}
		out[i] = unit(v)
	}
	return out
}

func near(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol*math.Max(1, math.Max(math.Abs(a), math.Abs(b)))
}
