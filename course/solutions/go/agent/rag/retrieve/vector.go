package retrieve

// vector.go: the dense half of retrieval. Exact (flat) top-k by dot
// product, and an IVF index whose coarse centroids come from k-means++
// seeding and Lloyd iterations, seeded from the index id so every load of
// the same index builds the same centroids.

import (
	"errors"
	"math"

	"tinyllm/ds/rng"
)

// Dot is the dot product, accumulated in float64 in index order.
func Dot(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

// Dist2 is the squared Euclidean distance, accumulated in float64 in index
// order.
func Dist2(a, b []float32) float64 {
	var s float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		s += d * d
	}
	return s
}

// Normalize returns v scaled to unit L2 norm (a zero vector stays zero).
func Normalize(v []float32) []float32 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	out := make([]float32, len(v))
	if s == 0 {
		return out
	}
	inv := 1 / math.Sqrt(s)
	for i, x := range v {
		out[i] = float32(float64(x) * inv)
	}
	return out
}

// Flat is exact search: every row is scored.
type Flat struct {
	Vecs [][]float32 // row i is chunk i, L2-normalized
}

// Search returns the k rows with the largest dot product with q (ties to
// the lower row) among the rows keep accepts (nil: all).
func (f *Flat) Search(q []float32, k int, keep func(doc int) bool) []Scored {
	c := make([]Scored, 0, len(f.Vecs))
	for i, v := range f.Vecs {
		if keep == nil || keep(i) {
			c = append(c, Scored{Doc: i, Score: Dot(q, v)})
		}
	}
	return TopK(c, k)
}

// ErrK is returned when k-means asks for more centroids than rows, or none.
var ErrK = errors.New("k-means: need 1 <= k <= rows")

// KMeansPP picks k seed rows by k-means++: the first uniformly,
// r.Below(n); each next one with probability proportional to D(i), the
// squared distance from row i to its nearest seed so far. Draw u =
// r.Float64() * sum(D) and take the first row (in row order) whose running
// sum of D exceeds u. When every D is 0 (duplicate rows), take the lowest
// row not yet picked. Returns the picked rows in pick order.
func KMeansPP(vecs [][]float32, k int, r *rng.PCG32) ([]int, error) {
	n := len(vecs)
	if k < 1 || k > n {
		return nil, ErrK
	}
	picked := []int{int(r.Below(uint64(n)))}
	chosen := map[int]bool{picked[0]: true}
	d := make([]float64, n)
	for i := range vecs {
		d[i] = Dist2(vecs[i], vecs[picked[0]])
	}
	for len(picked) < k {
		var sum float64
		for _, x := range d {
			sum += x
		}
		next := -1
		if sum == 0 {
			for i := range vecs {
				if !chosen[i] {
					next = i
					break
				}
			}
		} else {
			u := r.Float64() * sum
			var run float64
			for i, x := range d {
				run += x
				if run > u {
					next = i
					break
				}
			}
			if next < 0 { // u rounded up to sum: the last row with weight
				for i := n - 1; i >= 0; i-- {
					if d[i] > 0 {
						next = i
						break
					}
				}
			}
		}
		picked = append(picked, next)
		chosen[next] = true
		for i := range vecs {
			if x := Dist2(vecs[i], vecs[next]); x < d[i] {
				d[i] = x
			}
		}
	}
	return picked, nil
}

// Nearest is the centroid closest to v by squared distance, ties to the
// lower centroid.
func Nearest(v []float32, centroids [][]float32) int {
	best, bd := 0, math.Inf(1)
	for c, cv := range centroids {
		if x := Dist2(v, cv); x < bd {
			best, bd = c, x
		}
	}
	return best
}

// IVF is an inverted-file index: each row lives in the list of its nearest
// centroid, and a query scans only the nprobe lists whose centroids are
// nearest to it.
type IVF struct {
	Vecs      [][]float32
	Centroids [][]float32
	Lists     [][]int // Lists[c]: the rows of centroid c, ascending
}

// BuildIVF seeds nlist centroids with KMeansPP on r, then runs iters Lloyd
// rounds (assign every row to its Nearest centroid; move each centroid to
// the float64 mean of its rows; a centroid with no rows stays where it is),
// stopping early when no assignment changes. The lists hold the final
// assignment.
func BuildIVF(vecs [][]float32, nlist, iters int, r *rng.PCG32) (*IVF, error) {
	seeds, err := KMeansPP(vecs, nlist, r)
	if err != nil {
		return nil, err
	}
	dim := len(vecs[0])
	cent := make([][]float32, nlist)
	for c, i := range seeds {
		cent[c] = append([]float32(nil), vecs[i]...)
	}
	assign := make([]int, len(vecs))
	for i := range assign {
		assign[i] = -1
	}
	for it := 0; it < iters; it++ {
		changed := false
		for i, v := range vecs {
			if c := Nearest(v, cent); c != assign[i] {
				assign[i], changed = c, true
			}
		}
		if !changed {
			break
		}
		sums := make([][]float64, nlist)
		counts := make([]int, nlist)
		for c := range sums {
			sums[c] = make([]float64, dim)
		}
		for i, v := range vecs {
			c := assign[i]
			counts[c]++
			for j, x := range v {
				sums[c][j] += float64(x)
			}
		}
		for c := range cent {
			if counts[c] == 0 {
				continue
			}
			for j := range cent[c] {
				cent[c][j] = float32(sums[c][j] / float64(counts[c]))
			}
		}
	}
	// The lists follow the final centroids, whatever the last round did.
	lists := make([][]int, nlist)
	for i, v := range vecs {
		c := Nearest(v, cent)
		lists[c] = append(lists[c], i)
	}
	return &IVF{Vecs: vecs, Centroids: cent, Lists: lists}, nil
}

// Search scans the nprobe lists whose centroids are nearest to q (by
// squared distance, ties to the lower centroid) and returns the k best rows
// by dot product among those keep accepts, ties to the lower row.
func (x *IVF) Search(q []float32, k, nprobe int, keep func(doc int) bool) []Scored {
	cs := make([]Scored, len(x.Centroids))
	for c, cv := range x.Centroids {
		cs[c] = Scored{Doc: c, Score: -Dist2(q, cv)}
	}
	cs = TopK(cs, nprobe)
	var cand []Scored
	for _, c := range cs {
		for _, i := range x.Lists[c.Doc] {
			if keep == nil || keep(i) {
				cand = append(cand, Scored{Doc: i, Score: Dot(q, x.Vecs[i])})
			}
		}
	}
	return TopK(cand, k)
}

// Recall is |approx ∩ exact| / |exact| over row numbers.
func Recall(approx, exact []Scored) float64 {
	if len(exact) == 0 {
		return 1
	}
	in := map[int]bool{}
	for _, s := range exact {
		in[s.Doc] = true
	}
	hit := 0
	for _, s := range approx {
		if in[s.Doc] {
			hit++
		}
	}
	return float64(hit) / float64(len(exact))
}
