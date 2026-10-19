package scorers

// registry.go: scorers by the names eval specs use
// (formats/eval-spec.schema.json "scorers"), so a workflow (EvalSuite) or a
// CLI flag can name them.

import (
	"fmt"
	"strconv"
	"strings"

	"tinyllm/agent/eval"
)

// ByName builds the scorer a spec names: exact, regex (pattern from each
// case's scorer_args), ttft_ms, ttlt_ms, itl_ms, tool_success, tool_called,
// mrr, citation_precision, citation_recall, hit@<k>, ndcg@<k> (k >= 1).
// schema and state need configuration and are built with Schema and
// StateScorer instead.
func ByName(name string) (eval.Scorer, error) {
	switch name {
	case "exact":
		return Exact(), nil
	case "regex":
		return Regex("")
	case "ttft_ms":
		return TTFT(), nil
	case "ttlt_ms":
		return TTLT(), nil
	case "itl_ms":
		return ITL(), nil
	case "tool_success":
		return ToolSuccess(), nil
	case "tool_called":
		return ToolCalled(), nil
	case "mrr":
		return MRR(), nil
	case "citation_precision":
		return CitationPrecision(), nil
	case "citation_recall":
		return CitationRecall(), nil
	}
	for prefix, mk := range map[string]func(int) eval.Scorer{"hit@": HitAtK, "ndcg@": NDCG} {
		if rest, ok := strings.CutPrefix(name, prefix); ok {
			k, err := strconv.Atoi(rest)
			if err != nil || k < 1 || strconv.Itoa(k) != rest {
				return nil, fmt.Errorf("scorers: bad k in %q", name)
			}
			return mk(k), nil
		}
	}
	return nil, fmt.Errorf("scorers: unknown scorer %q", name)
}
