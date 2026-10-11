// Package harness holds the three LLM-backed engines that make a session
// useful: generating a problem with a hidden rubric, coaching while the user
// works, and scoring the result against that rubric.
package harness

import (
	"context"
	"fmt"
	"strings"

	"github.com/google/uuid"
	"github.com/urmzd/whiteboardy/internal/domain"
	"github.com/urmzd/whiteboardy/internal/llm"
)

// Generation runs as two calls rather than one.
//
// A single schema covering the brief, the rubric, the curveballs, and the
// reference outline is too much for a small local model to hold at once: push
// on one field in the prompt and another collapses, and the failure is silent
// (empty requirements, three criteria instead of eight). Two focused calls are
// each reliable, and the second one gets to read the actual brief, so the
// criteria grade the exercise that exists rather than the one that was asked
// for. The cost is one extra round trip, paid once per session.

// briefDTO is the half of a problem the candidate actually sees.
type briefDTO struct {
	Title        string   `json:"title" description:"Short punchy name for the exercise, under 8 words. No level prefix."`
	Statement    string   `json:"statement" description:"The scenario as markdown, 2-4 sentences. Set the scene and state the ask. Do NOT list the requirements or constraints here; they have their own fields."`
	Requirements []string `json:"requirements" description:"3-6 functional requirements: things the built system must DO. One short sentence each."`
	Constraints  []string `json:"constraints" description:"3-6 non-functional constraints on the built system, each containing a concrete number: throughput, latency, data volume, budget, retention."`
	Starter      string   `json:"starter" description:"Coding exercises only: starter code with imports, the signature to implement, and one worked example in a comment. Empty string for system design."`
}

// rubricDTO is the hidden half: how the work gets graded and what happens to
// the requirements partway through.
type rubricDTO struct {
	Criteria         []criterionDTO `json:"criteria" description:"Hidden grading criteria, each covering a different skill area"`
	Curveballs       []curveballDTO `json:"curveballs" description:"Complications revealed partway through to test adaptability. Never empty."`
	ReferenceOutline []string       `json:"referenceOutline" description:"5-9 bullet points a strong answer covers, in the order a strong candidate would reach them"`
}

type criterionDTO struct {
	Area   string `json:"area" description:"Skill area ID this criterion measures. Must be one of the allowed values."`
	Title  string `json:"title" description:"Short name of what is being graded"`
	Detail string `json:"detail" description:"What specifically a strong answer does here, concrete enough to grade against"`
	Weight int    `json:"weight" description:"How much this matters, 1 (minor) to 5 (central)"`
}

type curveballDTO struct {
	AtPct int    `json:"atPct" description:"Percentage of the timebox at which to reveal this, between 30 and 85"`
	Title string `json:"title" description:"Short headline, e.g. 'Traffic just went 10x'"`
	Body  string `json:"body" description:"One or two sentences stating the new constraint and what you now want addressed"`
}

const briefSystem = `You write interview-grade practice exercises for working software engineers.

You are writing the exercise, not describing it. The candidate reads this as if a colleague were describing a real system. Never refer to the exercise as an exercise: no timebox, no seniority level, no "the candidate should", no instructions about drawing diagrams or writing code.

Rules:
- Calibrate difficulty to the stated level. A staff problem is ambiguous and forces prioritization; a junior problem is concrete and bounded.
- Every constraint carries a real number. "High scale" is not a constraint; "40,000 writes per second per shard" is.
- Requirements are what the system does. Constraints are the limits it operates under. Keep them in their own fields and do not repeat them inside the statement.
- The problem must be solvable within the stated timebox by one person. Scope down until it is.
- Do not solve the problem and do not hint at the answer.

Return only the JSON object.`

const rubricSystem = `You write the hidden grading rubric for a practice exercise that has already been written.

You are given the exact exercise the candidate will see. Grade that exercise, not a generic one: every criterion must be answerable from the requirements and constraints in front of you.

Rules:
- Each criterion maps to exactly one allowed skill area, and no area appears twice.
- Criteria describe observable work ("names a partition key and justifies it against the write constraint"), not vibes ("shows good judgment").
- Weights must vary. Reserve 5 for what this exercise is really testing; give peripheral criteria 1 or 2. If everything is a 5, nothing is.
- Curveballs are mandatory. Each invalidates an assumption the candidate has probably already made by that point in the timebox. Annoying but fair.
- The reference outline is the path a strong candidate walks, in order.

Return only the JSON object.`

// GenerateProblem produces a problem calibrated to spec, with a hidden rubric
// and a curveball timeline.
func GenerateProblem(ctx context.Context, c *llm.Client, spec domain.SessionSpec) (domain.Problem, error) {
	brief, err := llm.Structured[briefDTO](ctx, c, briefSystem, buildBriefPrompt(spec))
	if err != nil {
		return domain.Problem{}, fmt.Errorf("generate problem brief: %w", err)
	}

	p := assembleBrief(spec, brief)

	areas := domain.AreaIDsFor(spec.Mode)
	rubric, err := llm.Structured[rubricDTO](ctx, c, rubricSystem, buildRubricPrompt(spec, p, areas),
		llm.WithItemEnum("criteria", "area", areas),
	)
	if err != nil {
		return domain.Problem{}, fmt.Errorf("generate problem rubric: %w", err)
	}

	applyRubric(spec, &p, rubric)
	return p, nil
}

func buildBriefPrompt(spec domain.SessionSpec) string {
	var b strings.Builder

	minutes := spec.DurationSec / 60
	kind := "SYSTEM DESIGN"
	if spec.Mode == domain.ModeCoding {
		kind = "CODING"
	}
	fmt.Fprintf(&b, "Write a %s exercise.\n\nLevel: %s\nTimebox: %d minutes\n", kind, spec.Level, minutes)

	if spec.Mode == domain.ModeCoding {
		lang := spec.Language
		if lang == "" {
			lang = "any language"
		}
		fmt.Fprintf(&b, "Language: %s\n", lang)
		fmt.Fprintf(&b, "\nThe candidate works in a plain code editor with no execution and no test runner, so correctness has to be checkable by reading. "+
			"Fill the starter field with %s: imports, the signature to implement, and one worked example in a comment. "+
			"The starter must compile as written and must not contain the solution.\n", lang)
	} else {
		b.WriteString("\nThe candidate answers on a whiteboard of labeled boxes and arrows plus a notes pane, so the exercise should have a diagram as its natural answer. " +
			"Leave the starter field as an empty string.\n")
	}

	if topic := strings.TrimSpace(spec.Topic); topic != "" {
		fmt.Fprintf(&b, "\nThe exercise must be about: %s\n", topic)
	} else {
		b.WriteString("\nPick a topic yourself. Prefer something a working engineer actually ships over a textbook classic.\n")
	}

	if custom := strings.TrimSpace(spec.CustomStatement); custom != "" {
		fmt.Fprintf(&b, "\nUse this text verbatim as the statement field, and derive the title, requirements, and constraints from it:\n---\n%s\n---\n", custom)
	}

	b.WriteString("\nEvery field must be filled. An empty requirements or constraints list is not acceptable.\n")
	return b.String()
}

func buildRubricPrompt(spec domain.SessionSpec, p domain.Problem, areas []string) string {
	var b strings.Builder

	minutes := spec.DurationSec / 60
	fmt.Fprintf(&b, "Write the hidden rubric for this exercise.\n\nLevel: %s\nTimebox: %d minutes\n\n=== THE EXERCISE ===\n%s\n",
		spec.Level, minutes, p.Render(false))

	if p.Starter != "" {
		fmt.Fprintf(&b, "\nStarter code the candidate begins from:\n```\n%s\n```\n", p.Starter)
	}

	b.WriteString("\nAllowed skill area IDs (use each at most once):\n")
	for _, a := range areas {
		fmt.Fprintf(&b, "- %s\n", a)
	}

	// Budgets scale with the timebox: a 15 minute sprint has no room for three
	// requirement changes.
	switch {
	case minutes <= 20:
		b.WriteString("\nEmit exactly 1 curveball and exactly 6 criteria. This is a short session.\n")
	case minutes <= 40:
		b.WriteString("\nEmit exactly 2 curveballs and exactly 8 criteria.\n")
	default:
		b.WriteString("\nEmit 3 curveballs and 9 criteria.\n")
	}
	b.WriteString("\nThe criteria and curveballs lists must not be empty.\n")
	return b.String()
}

func assembleBrief(spec domain.SessionSpec, dto briefDTO) domain.Problem {
	p := domain.Problem{
		ID:           uuid.NewString(),
		Mode:         spec.Mode,
		Level:        spec.Level,
		Title:        strings.TrimSpace(dto.Title),
		Statement:    strings.TrimSpace(dto.Statement),
		Requirements: cleanList(dto.Requirements),
		Constraints:  cleanList(dto.Constraints),
		Language:     spec.Language,
	}
	if p.Title == "" {
		p.Title = "Untitled exercise"
	}
	if custom := strings.TrimSpace(spec.CustomStatement); custom != "" {
		p.Statement = custom
	}
	if spec.Mode == domain.ModeCoding {
		p.Starter = stripFence(dto.Starter)
	}
	return p
}

func applyRubric(spec domain.SessionSpec, p *domain.Problem, dto rubricDTO) {
	p.ReferenceOutline = cleanList(dto.ReferenceOutline)

	// Keep only criteria that name a real area, and only the first per area:
	// duplicates would double-count one skill in the profile.
	seen := map[domain.Area]bool{}
	for _, c := range dto.Criteria {
		area := domain.Area(strings.TrimSpace(c.Area))
		if !domain.ValidArea(spec.Mode, area) || seen[area] {
			continue
		}
		seen[area] = true
		p.Rubric = append(p.Rubric, domain.Criterion{
			ID:     string(area),
			Area:   area,
			Title:  strings.TrimSpace(c.Title),
			Detail: strings.TrimSpace(c.Detail),
			Weight: clampInt(c.Weight, 1, 5),
		})
	}
	if len(p.Rubric) == 0 {
		p.Rubric = fallbackRubric(spec.Mode)
	}

	for _, cb := range dto.Curveballs {
		title := strings.TrimSpace(cb.Title)
		body := strings.TrimSpace(cb.Body)
		if title == "" || body == "" {
			continue
		}
		p.Curveballs = append(p.Curveballs, domain.Curveball{
			AtPct: clampInt(cb.AtPct, 25, 85),
			Title: title,
			Body:  body,
		})
	}
	if len(p.Curveballs) == 0 {
		// Losing the mechanic entirely is worse than a generic complication.
		p.Curveballs = []domain.Curveball{fallbackCurveball(spec.Mode)}
	}
}

func fallbackCurveball(mode domain.Mode) domain.Curveball {
	if mode == domain.ModeCoding {
		return domain.Curveball{
			AtPct: 60,
			Title: "The input no longer fits in memory",
			Body:  "Assume the input arrives as a stream too large to hold at once. What changes, and what does it cost you?",
		}
	}
	return domain.Curveball{
		AtPct: 60,
		Title: "Traffic just went 10x",
		Body:  "Sustained load is now ten times what the requirements state, and it is not going back down. Which part of your design breaks first, and what do you do about it?",
	}
}

// fallbackRubric keeps a session gradeable when a small model returns criteria
// that all fail validation. A generic rubric beats no rubric.
func fallbackRubric(mode domain.Mode) []domain.Criterion {
	infos := domain.AreasFor(mode)
	if len(infos) > 8 {
		infos = infos[:8]
	}
	out := make([]domain.Criterion, 0, len(infos))
	for _, a := range infos {
		out = append(out, domain.Criterion{
			ID:     string(a.ID),
			Area:   a.ID,
			Title:  a.Label,
			Detail: a.Blurb,
			Weight: 3,
		})
	}
	return out
}

func cleanList(in []string) []string {
	out := make([]string, 0, len(in))
	for _, s := range in {
		if s = strings.TrimSpace(s); s != "" {
			out = append(out, s)
		}
	}
	return out
}

// stripFence removes a markdown code fence around generated starter code.
// Models add one often enough that it would otherwise land in the editor.
func stripFence(s string) string {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "```") {
		return s
	}
	if nl := strings.IndexByte(s, '\n'); nl >= 0 {
		s = s[nl+1:]
	}
	if end := strings.LastIndex(s, "```"); end >= 0 {
		s = s[:end]
	}
	return strings.TrimSpace(s)
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
