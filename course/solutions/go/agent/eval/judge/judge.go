// Package judge provides rubric scoring with strict output parsing and
// position-swapped pairwise comparisons.
package judge

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"tinyllm/agent/eval"
)

var ErrJudgeOutput = errors.New("judge: invalid judge output")

type Generator interface {
	Generate(context.Context, string) (string, error)
}
type judgeScorer struct {
	g      Generator
	rubric string
}

func (s judgeScorer) Name() string { return "llm_judge" }
func (s judgeScorer) Score(ctx context.Context, o eval.Observation) (eval.Score, error) {
	var in struct {
		Output    string `json:"output"`
		Reference string `json:"answer"`
	}
	json.Unmarshal(o.Output, &in.Output)
	json.Unmarshal(o.GroundTruth, &in.Reference)
	prompt := fmt.Sprintf("Rubric:\n%s\nAnswer:\n%s\nReference:\n%s\nReturn JSON {\"score\": number from 0 to 1, \"evidence\": nonempty string}.", s.rubric, in.Output, in.Reference)
	raw, err := s.g.Generate(ctx, prompt)
	if err != nil {
		return eval.Score{}, err
	}
	var out struct {
		Score    *float64 `json:"score"`
		Evidence string   `json:"evidence"`
	}
	dec := json.NewDecoder(strings.NewReader(raw))
	dec.DisallowUnknownFields()
	if dec.Decode(&out) != nil || dec.Decode(new(any)) != io.EOF || out.Score == nil || *out.Score < 0 || *out.Score > 1 || out.Evidence == "" {
		return eval.Score{}, ErrJudgeOutput
	}
	return eval.Score{Value: *out.Score, Reason: out.Evidence}, nil
}
func NewJudgeScorer(g Generator, rubric string, _ ...JudgeOption) eval.Scorer {
	return judgeScorer{g: g, rubric: rubric}
}

type JudgeOption func(*judgeConfig)
type judgeConfig struct{}

func WithPairwise() JudgeOption {
	return func(*judgeConfig) {}
}

type PairwiseResult struct {
	Winner            string
	PositionSensitive bool
	First, Swapped    string
}

func Pairwise(ctx context.Context, g Generator, prompt, a, b string) (PairwiseResult, error) {
	ask := func(left, right string) (string, error) {
		p := fmt.Sprintf("%s\nA: %s\nB: %s\nReturn only JSON {\"winner\":\"A\"|\"B\"|\"tie\"}.", prompt, left, right)
		raw, err := g.Generate(ctx, p)
		if err != nil {
			return "", err
		}
		var x struct {
			Winner string `json:"winner"`
		}
		d := json.NewDecoder(strings.NewReader(raw))
		d.DisallowUnknownFields()
		if d.Decode(&x) != nil || d.Decode(new(any)) != io.EOF || (x.Winner != "A" && x.Winner != "B" && x.Winner != "tie") {
			return "", ErrJudgeOutput
		}
		return x.Winner, nil
	}
	x, e := ask(a, b)
	if e != nil {
		return PairwiseResult{}, e
	}
	y, e := ask(b, a)
	if e != nil {
		return PairwiseResult{}, e
	}
	if y == "A" {
		y = "B"
	} else if y == "B" {
		y = "A"
	}
	winner := x
	if x != y {
		winner = "tie"
	}
	return PairwiseResult{Winner: winner, PositionSensitive: x != y, First: x, Swapped: y}, nil
}

// Kappa computes Cohen's kappa for two equally sized categorical label sets.
func Kappa(a, b []string) (float64, error) {
	if len(a) == 0 || len(a) != len(b) {
		return 0, fmt.Errorf("judge: label sets must have equal nonzero length")
	}
	ag, bg := map[string]int{}, map[string]int{}
	agree := 0
	for i := range a {
		ag[a[i]]++
		bg[b[i]]++
		if a[i] == b[i] {
			agree++
		}
	}
	po := float64(agree) / float64(len(a))
	var pe float64
	for k, n := range ag {
		pe += float64(n*bg[k]) / float64(len(a)*len(a))
	}
	if pe == 1 {
		if po == 1 {
			return 1, nil
		}
		return 0, nil
	}
	return (po - pe) / (1 - pe), nil
}

type sampled struct {
	inner     eval.Scorer
	n         int
	tolerance float64
}

func (s sampled) Name() string {
	return s.inner.Name()
}
func (s sampled) Score(ctx context.Context, o eval.Observation) (eval.Score, error) {
	if s.n < 1 || s.tolerance < 0 || s.tolerance > 1 {
		return eval.Score{}, fmt.Errorf("judge: invalid sampling configuration")
	}
	h := sha256.Sum256([]byte(o.ID))
	bucket := int(h[0]) % s.n
	if bucket != 0 {
		return eval.Score{}, fmt.Errorf("judge: not sampled (tolerance %.4g; id %s)", s.tolerance, hex.EncodeToString(h[:4]))
	}
	return s.inner.Score(ctx, o)
}
func Sampled(inner eval.Scorer, n int, tolerance float64) eval.Scorer {
	return sampled{inner: inner, n: n, tolerance: tolerance}
}
