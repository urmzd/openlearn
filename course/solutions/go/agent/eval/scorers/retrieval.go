package scorers

// retrieval.go: ranking metrics over the chunks a RAG agent retrieved
// (the "retrieved" annotation, chunk ids in rank order) against the case's
// relevant chunks, and citation metrics over the [n] markers in its answer
// (resolved with ag.08's parser against the "citations" annotation).

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"

	"tinyllm/agent/eval"
	"tinyllm/agent/rag/assemble"
)

// Relevance reads the case's relevant chunks from its ground truth:
// "chunks" is a list of ids (each grade 1) or an object of id -> grade.
// Grades at or below 0 are not relevant.
func Relevance(gt json.RawMessage) (map[string]float64, error) {
	var obj struct {
		Chunks json.RawMessage `json:"chunks"`
	}
	if json.Unmarshal(gt, &obj) != nil || len(obj.Chunks) == 0 {
		return nil, ErrNoGroundTruth
	}
	rel := map[string]float64{}
	var list []string
	if json.Unmarshal(obj.Chunks, &list) == nil {
		for _, id := range list {
			rel[id] = 1
		}
	} else if err := json.Unmarshal(obj.Chunks, &rel); err != nil {
		return nil, fmt.Errorf("%w: chunks: %v", ErrNoGroundTruth, err)
	}
	for id, g := range rel {
		if g <= 0 {
			delete(rel, id)
		}
	}
	if len(rel) == 0 {
		return nil, fmt.Errorf("%w: no relevant chunks", ErrNoGroundTruth)
	}
	return rel, nil
}

// Retrieved is the "retrieved" annotation with repeats removed (a chunk
// counts at its first rank only).
func Retrieved(o eval.Observation) ([]string, error) {
	var ids []string
	if raw, ok := o.Annotations["retrieved"]; ok {
		if err := json.Unmarshal(raw, &ids); err != nil {
			return nil, fmt.Errorf("scorers: retrieved annotation: %w", err)
		}
	}
	seen := map[string]bool{}
	out := ids[:0]
	for _, id := range ids {
		if !seen[id] {
			seen[id] = true
			out = append(out, id)
		}
	}
	return out, nil
}

func rankInputs(o eval.Observation) ([]string, map[string]float64, error) {
	rel, err := Relevance(o.GroundTruth)
	if err != nil {
		return nil, nil, err
	}
	got, err := Retrieved(o)
	return got, rel, err
}

// HitAtK scores 1 when a relevant chunk is among the first k retrieved
// ("hit@<k>").
func HitAtK(k int) eval.Scorer {
	return fn{fmt.Sprintf("hit@%d", k), func(_ context.Context, o eval.Observation) (eval.Score, error) {
		got, rel, err := rankInputs(o)
		if err != nil {
			return eval.Score{}, err
		}
		for i, id := range got {
			if i >= k {
				break
			}
			if rel[id] > 0 {
				return eval.Score{Value: 1}, nil
			}
		}
		return eval.Score{Value: 0}, nil
	}}
}

// MRR is the reciprocal rank (counted from 1) of the first relevant chunk,
// 0 when none was retrieved ("mrr"); its mean over a suite is the mean
// reciprocal rank.
func MRR() eval.Scorer {
	return fn{"mrr", func(_ context.Context, o eval.Observation) (eval.Score, error) {
		got, rel, err := rankInputs(o)
		if err != nil {
			return eval.Score{}, err
		}
		for i, id := range got {
			if rel[id] > 0 {
				return eval.Score{Value: 1 / float64(i+1)}, nil
			}
		}
		return eval.Score{Value: 0}, nil
	}}
}

// DCG is sum over ranks i = 1..len(grades) of (2^g_i - 1) / log2(i + 1).
func DCG(grades []float64) float64 {
	var s float64
	for i, g := range grades {
		s += (math.Pow(2, g) - 1) / math.Log2(float64(i+2))
	}
	return s
}

// NDCG is DCG of the first k retrieved chunks' grades divided by the DCG
// of the k best grades in the ground truth (the ideal ranking), in [0, 1]
// ("ndcg@<k>").
func NDCG(k int) eval.Scorer {
	return fn{fmt.Sprintf("ndcg@%d", k), func(_ context.Context, o eval.Observation) (eval.Score, error) {
		got, rel, err := rankInputs(o)
		if err != nil {
			return eval.Score{}, err
		}
		var grades []float64
		for i, id := range got {
			if i >= k {
				break
			}
			grades = append(grades, rel[id])
		}
		ideal := make([]float64, 0, len(rel))
		for _, g := range rel {
			ideal = append(ideal, g)
		}
		sort.Sort(sort.Reverse(sort.Float64Slice(ideal)))
		if len(ideal) > k {
			ideal = ideal[:k]
		}
		return eval.Score{Value: DCG(grades) / DCG(ideal)}, nil
	}}
}

// citations is the "citations" annotation: the run's ledger.
func citations(o eval.Observation) ([]assemble.Citation, error) {
	var cites []assemble.Citation
	if raw, ok := o.Annotations["citations"]; ok {
		if err := json.Unmarshal(raw, &cites); err != nil {
			return nil, fmt.Errorf("scorers: citations annotation: %w", err)
		}
	}
	return cites, nil
}

// CitationPrecision is the fraction of the answer's citation markers that
// name a relevant chunk; a number no block carried counts as a wrong
// citation ("citation_precision"). An answer with no markers has no
// precision: that is an error (citation recall scores it 0).
func CitationPrecision() eval.Scorer {
	return fn{"citation_precision", func(_ context.Context, o eval.Observation) (eval.Score, error) {
		rel, err := Relevance(o.GroundTruth)
		if err != nil {
			return eval.Score{}, err
		}
		cites, err := citations(o)
		if err != nil {
			return eval.Score{}, err
		}
		used, unknown := assemble.Resolve(OutputText(o), cites)
		total := len(used) + len(unknown)
		if total == 0 {
			return eval.Score{}, fmt.Errorf("scorers: the answer cites nothing")
		}
		good := 0
		for _, c := range used {
			if rel[c.ChunkID] > 0 {
				good++
			}
		}
		return eval.Score{Value: float64(good) / float64(total), Reason: fmt.Sprintf("%d unknown citation numbers", len(unknown))}, nil
	}}
}

// CitationRecall is the fraction of the relevant chunks the answer cites
// ("citation_recall").
func CitationRecall() eval.Scorer {
	return fn{"citation_recall", func(_ context.Context, o eval.Observation) (eval.Score, error) {
		rel, err := Relevance(o.GroundTruth)
		if err != nil {
			return eval.Score{}, err
		}
		cites, err := citations(o)
		if err != nil {
			return eval.Score{}, err
		}
		used, _ := assemble.Resolve(OutputText(o), cites)
		hit := map[string]bool{}
		for _, c := range used {
			if rel[c.ChunkID] > 0 {
				hit[c.ChunkID] = true
			}
		}
		return eval.Score{Value: float64(len(hit)) / float64(len(rel))}, nil
	}}
}
