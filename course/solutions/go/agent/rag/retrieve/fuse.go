package retrieve

// fuse.go: combining ranked lists. Reciprocal rank fusion merges the
// lexical and dense lists by rank alone; maximal marginal relevance then
// trades relevance against redundancy when picking the final k.

import "sort"

// DefaultRRFK is the k of Cormack, Clarke, and Buettcher (2009).
const DefaultRRFK = 60

// RRF fuses ranked lists: a chunk's score is the sum, over the lists that
// contain it, of 1 / (k + rank), rank counted from 1 in that list. Hits are
// identified by Index. The result is sorted by fused score descending, ties
// to the lower Index, cut to n (all when n <= 0), and carries the fused
// score in Score and its 1-based position in Rank. Every other field comes
// from the first list that holds the chunk.
func RRF(lists [][]Hit, k float64, n int) []Hit {
	score := map[int]float64{}
	first := map[int]Hit{}
	for _, l := range lists {
		for r, h := range l {
			score[h.Index] += 1 / (k + float64(r+1))
			if _, ok := first[h.Index]; !ok {
				first[h.Index] = h
			}
		}
	}
	out := make([]Hit, 0, len(score))
	for i, s := range score {
		h := first[i]
		h.Score = s
		out = append(out, h)
	}
	sort.Slice(out, func(a, b int) bool {
		if out[a].Score != out[b].Score {
			return out[a].Score > out[b].Score
		}
		return out[a].Index < out[b].Index
	})
	if n > 0 && n < len(out) {
		out = out[:n]
	}
	for i := range out {
		out[i].Rank = i + 1
	}
	return out
}

// MMR picks k of cands one at a time. Each step takes the candidate that
// maximizes lambda * sim(q, c) - (1 - lambda) * max over picked p of
// sim(c, p), with sim the dot product of the vectors vec returns (the max
// over an empty set is 0, so the first pick is the most relevant). Equal
// values go to the candidate that comes first in cands. Each picked hit
// carries its MMR value in Score and its 1-based position in Rank.
func MMR(q []float32, cands []Hit, vec func(index int) []float32, lambda float64, k int) []Hit {
	if k <= 0 || k > len(cands) {
		k = len(cands)
	}
	rel := make([]float64, len(cands))
	for i, c := range cands {
		rel[i] = Dot(q, vec(c.Index))
	}
	maxSim := make([]float64, len(cands)) // max sim to the picked set
	used := make([]bool, len(cands))
	out := make([]Hit, 0, k)
	for len(out) < k {
		best, bv := -1, 0.0
		for i := range cands {
			if used[i] {
				continue
			}
			v := lambda*rel[i] - (1-lambda)*maxSim[i]
			if best < 0 || v > bv {
				best, bv = i, v
			}
		}
		used[best] = true
		h := cands[best]
		h.Score, h.Rank = bv, len(out)+1
		out = append(out, h)
		pv := vec(h.Index)
		for i := range cands {
			if used[i] {
				continue
			}
			s := Dot(vec(cands[i].Index), pv)
			if len(out) == 1 || s > maxSim[i] {
				maxSim[i] = s
			}
		}
	}
	return out
}
