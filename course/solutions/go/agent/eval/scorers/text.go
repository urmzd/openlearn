// Package scorers holds the deterministic scorers of the agent evals
// (ag.10): text (exact match, regex, JSON Schema), latency (TTFT, TTLT,
// ITL), tool use, retrieval (hit@k, MRR, nDCG, citation precision and
// recall), and the state-change grader of case study 05. Every scorer is
// code: the same observation always gets the same score, and a scorer
// that cannot score an observation says so with an error instead of
// guessing a number.
//
// Chapter: content/courses/ai-platform-engineering/09-llm-evaluation/10-scorers.md.
package scorers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"unicode"

	"tinyllm/agent/eval"
	"tinyllm/agent/tool"
)

// ErrNoGroundTruth: the case has no ground truth this scorer can use.
var ErrNoGroundTruth = errors.New("scorers: no usable ground truth")

// fn adapts a function to eval.Scorer.
type fn struct {
	name string
	f    func(ctx context.Context, o eval.Observation) (eval.Score, error)
}

func (s fn) Name() string { return s.name }
func (s fn) Score(ctx context.Context, o eval.Observation) (eval.Score, error) {
	return s.f(ctx, o)
}

// OutputText is the output as text: a JSON string's value, or the raw JSON
// of anything else.
func OutputText(o eval.Observation) string {
	var s string
	if json.Unmarshal(o.Output, &s) == nil {
		return s
	}
	return string(o.Output)
}

// Normalize lowercases s, trims Unicode spaces and punctuation from both
// ends, and collapses every run of internal whitespace to one space:
// "  Chained FNV-1a 64. " and "chained   fnv-1a 64" are equal. Punctuation
// inside the text is kept ("fnv-1a" is not "fnv1a").
func Normalize(s string) string {
	s = strings.ToLower(s)
	s = strings.TrimFunc(s, func(r rune) bool { return unicode.IsSpace(r) || unicode.IsPunct(r) })
	return strings.Join(strings.Fields(s), " ")
}

// answer is the expected answer: ground truth that is a string, or an
// object's "answer" string.
func answer(gt json.RawMessage) (string, bool) {
	var s string
	if json.Unmarshal(gt, &s) == nil {
		return s, true
	}
	var obj struct {
		Answer *string `json:"answer"`
	}
	if json.Unmarshal(gt, &obj) == nil && obj.Answer != nil {
		return *obj.Answer, true
	}
	return "", false
}

// Exact scores 1 when the normalized output equals the normalized expected
// answer, else 0 ("exact").
func Exact() eval.Scorer {
	return fn{"exact", func(_ context.Context, o eval.Observation) (eval.Score, error) {
		want, ok := answer(o.GroundTruth)
		if !ok {
			return eval.Score{}, ErrNoGroundTruth
		}
		if Normalize(OutputText(o)) == Normalize(want) {
			return eval.Score{Value: 1}, nil
		}
		return eval.Score{Value: 0, Reason: fmt.Sprintf("want %q", want)}, nil
	}}
}

// Regex scores 1 when the output matches pattern (RE2, unanchored), else 0
// ("regex"). An empty pattern takes each case's scorer_args.regex.pattern.
func Regex(pattern string) (eval.Scorer, error) {
	var fixed *regexp.Regexp
	if pattern != "" {
		re, err := regexp.Compile(pattern)
		if err != nil {
			return nil, err
		}
		fixed = re
	}
	return fn{"regex", func(_ context.Context, o eval.Observation) (eval.Score, error) {
		re := fixed
		if re == nil {
			var args struct {
				Regex struct {
					Pattern string `json:"pattern"`
				} `json:"regex"`
			}
			json.Unmarshal(o.Annotations["scorer_args"], &args)
			if args.Regex.Pattern == "" {
				return eval.Score{}, fmt.Errorf("%w: no scorer_args.regex.pattern", ErrNoGroundTruth)
			}
			var err error
			if re, err = regexp.Compile(args.Regex.Pattern); err != nil {
				return eval.Score{}, err
			}
		}
		if re.MatchString(OutputText(o)) {
			return eval.Score{Value: 1}, nil
		}
		return eval.Score{Value: 0, Reason: "no match for " + re.String()}, nil
	}}, nil
}

// Schema scores 1 when the output text is JSON that validates against
// schema (ag.02's validator), else 0 with the violations as the reason
// ("schema").
func Schema(schema json.RawMessage) (eval.Scorer, error) {
	s, err := tool.Compile(schema)
	if err != nil {
		return nil, err
	}
	return fn{"schema", func(_ context.Context, o eval.Observation) (eval.Score, error) {
		errs := s.ValidateJSON([]byte(OutputText(o)))
		if len(errs) == 0 {
			return eval.Score{Value: 1}, nil
		}
		msgs := make([]string, len(errs))
		for i, e := range errs {
			msgs[i] = e.Error()
		}
		return eval.Score{Value: 0, Reason: strings.Join(msgs, "; ")}, nil
	}}, nil
}
