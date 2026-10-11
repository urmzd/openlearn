package harness

import (
	"context"
	"fmt"
	"strings"

	"github.com/urmzd/whiteboardy/internal/domain"
	"github.com/urmzd/whiteboardy/internal/llm"
)

// reviewDTO is the reviewer's constrained output. The overall score is not
// asked for: it is computed in Go from the per-criterion scores and weights,
// so the number is reproducible and cannot drift from the evidence.
type reviewDTO struct {
	Summary       string     `json:"summary" description:"3-5 sentences addressed to the candidate. What the work actually shows, then the single biggest thing holding it back."`
	Verdict       string     `json:"verdict" description:"The seniority level this work reads at on its own merits, regardless of the level the exercise targeted"`
	Scores        []scoreDTO `json:"scores" description:"Exactly one entry per rubric criterion given, using the same criterion IDs"`
	Strengths     []string   `json:"strengths" description:"2-4 things done well, each naming the specific evidence"`
	Gaps          []string   `json:"gaps" description:"2-4 things missing or wrong, each naming what should have been there"`
	NextSteps     []string   `json:"nextSteps" description:"2-4 concrete things to practice next, specific enough to act on this week"`
	MissedOutline []string   `json:"missedOutline" description:"Points from the reference outline the work never reached"`
}

type scoreDTO struct {
	CriterionID string `json:"criterionId" description:"The exact id from the rubric"`
	Score       int    `json:"score" description:"0 absent, 1 named only, 2 partial, 3 solid, 4 excellent"`
	Evidence    string `json:"evidence" description:"One sentence quoting or naming what in the work earned this score. If the score is 0, say what you looked for and did not find."`
}

const reviewSystem = `You grade a practice exercise the way a fair, experienced interviewer would in a debrief.

Grade what is actually there, not what the candidate probably meant. An unlabeled box, a stub function, or a note saying "would add caching" is not the same as doing the work, and should not score above 1.

Scoring scale, applied strictly:
0 - absent. Nothing in the work addresses this.
1 - named only. Mentioned but not developed.
2 - partial. Started, but a reviewer would still have questions.
3 - solid. Does the job. This is a good score, not a mediocre one.
4 - excellent. Handled better than most candidates at this level would.

Most criteria on a timeboxed session land at 1-3. Reserve 4 for work that genuinely stands out. Do not inflate to be kind: an inflated score costs the candidate the information they came for.

Score every criterion you are given, using its exact id. Take the timebox into account when judging scope, but not when judging correctness.

Be direct and specific. "Your data model has no partition key" is useful. "Consider thinking more about data" is not.

Return only the JSON object.`

// Review scores a finished session against its hidden rubric.
func Review(ctx context.Context, c *llm.Client, p domain.Problem, final domain.Snapshot, events []domain.Event, elapsedSec int) (domain.Review, error) {
	dto, err := llm.Structured[reviewDTO](ctx, c, reviewSystem, buildReviewPrompt(p, final, events, elapsedSec),
		llm.WithEnum("verdict", []string{"junior", "mid", "senior", "staff"}),
	)
	if err != nil {
		return domain.Review{}, fmt.Errorf("review: %w", err)
	}
	return assembleReview(p, dto), nil
}

func buildReviewPrompt(p domain.Problem, final domain.Snapshot, events []domain.Event, elapsedSec int) string {
	var b strings.Builder

	b.WriteString("=== EXERCISE (with hidden rubric) ===\n")
	b.WriteString(p.Render(true))

	fmt.Fprintf(&b, "\nTimebox used: %s.\n", humanDuration(elapsedSec))

	b.WriteString("\n=== WHAT THE CANDIDATE PRODUCED ===\n")
	b.WriteString(final.Render(p.Mode))

	var curveballs, coaching []domain.Event
	for _, e := range events {
		switch e.Kind {
		case domain.KindCurveball:
			curveballs = append(curveballs, e)
		case domain.KindHint, domain.KindProbe:
			coaching = append(coaching, e)
		}
	}

	if len(curveballs) > 0 {
		b.WriteString("\n=== CURVEBALLS THEY WERE GIVEN (their response to these is part of the grade) ===\n")
		for _, e := range curveballs {
			fmt.Fprintf(&b, "- at %s: %s - %s\n", humanDuration(e.ElapsedSec), e.Title, e.Body)
		}
	}
	if len(coaching) > 0 {
		b.WriteString("\n=== HINTS THEY RECEIVED (work they only did after a hint scores lower than work they did unprompted) ===\n")
		for _, e := range coaching {
			fmt.Fprintf(&b, "- at %s: %s - %s\n", humanDuration(e.ElapsedSec), e.Title, e.Body)
		}
	} else {
		b.WriteString("\n=== HINTS THEY RECEIVED ===\n(none: this was unassisted work)\n")
	}

	b.WriteString("\nCriterion IDs you must score, exactly these and no others:\n")
	for _, c := range p.Rubric {
		fmt.Fprintf(&b, "- %s\n", c.ID)
	}
	b.WriteString("\nGrade the work.\n")
	return b.String()
}

func assembleReview(p domain.Problem, dto reviewDTO) domain.Review {
	byID := map[string]scoreDTO{}
	for _, s := range dto.Scores {
		byID[strings.TrimSpace(s.CriterionID)] = s
	}

	rev := domain.Review{
		Summary:       strings.TrimSpace(dto.Summary),
		Verdict:       parseLevel(dto.Verdict),
		Strengths:     cleanList(dto.Strengths),
		Gaps:          cleanList(dto.Gaps),
		NextSteps:     cleanList(dto.NextSteps),
		MissedOutline: cleanList(dto.MissedOutline),
	}

	weighted, totalWeight := 0, 0
	for _, c := range p.Rubric {
		s, ok := byID[c.ID]
		score := 0
		evidence := "The reviewer did not score this criterion, so it counts as not demonstrated."
		if ok {
			score = clamp(s.Score, 0, 4)
			if e := strings.TrimSpace(s.Evidence); e != "" {
				evidence = e
			}
		}
		rev.Scores = append(rev.Scores, domain.CriterionScore{
			CriterionID: c.ID,
			Area:        c.Area,
			Title:       c.Title,
			Score:       score,
			Evidence:    evidence,
		})
		weighted += score * c.Weight
		totalWeight += 4 * c.Weight
	}
	if totalWeight > 0 {
		rev.Overall = weighted * 100 / totalWeight
	}
	return rev
}

func parseLevel(s string) domain.Level {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "junior":
		return domain.LevelJunior
	case "senior":
		return domain.LevelSenior
	case "staff":
		return domain.LevelStaff
	default:
		return domain.LevelMid
	}
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
