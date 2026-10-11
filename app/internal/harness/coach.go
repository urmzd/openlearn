package harness

import (
	"context"
	"fmt"
	"strings"

	"github.com/urmzd/whiteboardy/internal/domain"
	"github.com/urmzd/whiteboardy/internal/llm"
)

// A coaching tick is two calls, not one.
//
// The decision (speak or stay quiet, and about what) needs a schema so it comes
// back reliably. The message body does not: it is prose the user watches being
// written, and streaming it is what makes the coach feel like someone watching
// rather than a notification that pops in fully formed.
//
// Silence is the common case and costs only the first call, whose schema is
// small. Speaking costs a second call that streams.

// decisionDTO is the coach's judgment for a tick.
type decisionDTO struct {
	Speak        bool     `json:"speak" description:"true only if there is something worth interrupting for right now. Default to false."`
	Kind         string   `json:"kind" description:"hint if something important is missing, probe to question a choice already made, praise to confirm a strong move"`
	Severity     string   `json:"severity" description:"info for a nudge, warn for something that will cost them, critical for a fundamental miss"`
	Title        string   `json:"title" description:"Under 8 words. The headline the candidate reads first."`
	Point        string   `json:"point" description:"The single idea the message will make, in one compressed sentence. Not the message itself."`
	TargetAreas  []string `json:"targetAreas" description:"Skill area IDs this message is about"`
	CoveredAreas []string `json:"coveredAreas" description:"Skill area IDs the current work already demonstrably addresses. Judge from what is on the board or in the code, not from intent."`
}

const decisionSystem = `You are a senior interviewer watching a candidate work, in real time, on the exercise below.

You interrupt rarely. Most ticks you stay silent, because interrupting a candidate mid-thought is worse than saying nothing.

Speak only when one of these is true:
- They are about to build on an assumption that will not survive the constraints.
- A high-weight rubric area is untouched and the remaining time is running out for it.
- They made a choice worth questioning, and questioning it now changes what they do next.
- They did something genuinely strong and confirming it will make them commit to the approach.

Never speak when:
- They are actively filling in something you would have asked for anyway.
- You already made the same point. Read the recent messages and do not repeat yourself.
- The work is empty or barely started and they have plenty of time left.
- You would only be restating the requirements.

Judge coverage strictly. An unlabeled box named "cache" does not cover caching; a labeled box with an eviction policy and an invalidation note does. An empty function body does not cover correctness.

Return only the JSON object.`

const messageSystem = `You are a senior interviewer talking to a candidate mid-exercise.

Write the message. One or two sentences, spoken directly to them. Ask a question or point at a gap.

Never give the answer. Never list what they should add. Never write their design or code for them. Make exactly the one point you were given and stop.

No preamble, no "I noticed that", no sign-off. Output the message text only, with no quotes around it and no markdown.`

// CoachInput is everything the coach sees on a tick.
type CoachInput struct {
	Problem      domain.Problem
	Snapshot     domain.Snapshot
	ElapsedSec   int
	RemainingSec int
	// RecentEvents are the last few things already said, so the coach can avoid
	// repeating itself.
	RecentEvents []domain.Event
	// CoveredAreas is the running coverage set from prior ticks.
	CoveredAreas []domain.Area
}

// CoachDecision is the outcome of the first call.
type CoachDecision struct {
	// Speak reports whether the coach chose to interrupt.
	Speak bool
	// Covered is the coach's fresh read of which areas the work addresses.
	Covered []domain.Area
	// The rest are meaningful only when Speak is true.
	Kind     domain.EventKind
	Severity domain.Severity
	Title    string
	Point    string
	Areas    []domain.Area
}

// Decide runs the judgment half of a tick. It is cheap enough to run often and
// returns updated coverage even when it chooses silence.
func Decide(ctx context.Context, c *llm.Client, in CoachInput) (CoachDecision, error) {
	areas := domain.AreaIDsFor(in.Problem.Mode)

	dto, err := llm.Structured[decisionDTO](ctx, c, decisionSystem, buildDecisionPrompt(in),
		llm.WithEnum("kind", []string{"hint", "probe", "praise"}),
		llm.WithEnum("severity", []string{"info", "warn", "critical"}),
		llm.WithArrayEnum("targetAreas", areas),
		llm.WithArrayEnum("coveredAreas", areas),
	)
	if err != nil {
		return CoachDecision{}, fmt.Errorf("coach decision: %w", err)
	}

	out := CoachDecision{
		Speak:   dto.Speak,
		Covered: filterAreas(in.Problem.Mode, dto.CoveredAreas),
	}
	if !dto.Speak {
		return out, nil
	}

	out.Title = strings.TrimSpace(dto.Title)
	out.Point = strings.TrimSpace(dto.Point)
	if out.Point == "" {
		// Claimed to speak but named no point; treat as silence rather than
		// asking the next call to invent one.
		out.Speak = false
		return out, nil
	}
	if out.Title == "" {
		out.Title = "One thought"
	}
	out.Kind = coachKind(dto.Kind)
	out.Severity = coachSeverity(dto.Severity)
	out.Areas = filterAreas(in.Problem.Mode, dto.TargetAreas)
	return out, nil
}

// Speak writes the message body for a decision, streaming it through onDelta.
func Speak(ctx context.Context, c *llm.Client, in CoachInput, d CoachDecision, onDelta func(string)) (string, error) {
	body, err := c.TextStream(ctx, messageSystem, buildMessagePrompt(in, d), onDelta)
	if err != nil {
		return "", fmt.Errorf("coach message: %w", err)
	}
	body = strings.TrimSpace(body)
	if body == "" {
		// Fall back to the point itself rather than showing an empty bubble.
		return d.Point, nil
	}
	return body, nil
}

func buildDecisionPrompt(in CoachInput) string {
	var b strings.Builder

	b.WriteString("=== EXERCISE ===\n")
	b.WriteString(in.Problem.Render(true))

	fmt.Fprintf(&b, "\n=== TIME ===\nElapsed: %s. Remaining: %s (%d%% of the timebox is gone).\n",
		humanDuration(in.ElapsedSec), humanDuration(in.RemainingSec), pctElapsed(in.ElapsedSec, in.RemainingSec))

	covered := map[domain.Area]bool{}
	for _, a := range in.CoveredAreas {
		covered[a] = true
	}
	var uncovered []string
	for _, c := range in.Problem.Rubric {
		if !covered[c.Area] {
			uncovered = append(uncovered, fmt.Sprintf("%s (weight %d)", c.Title, c.Weight))
		}
	}
	if len(uncovered) > 0 {
		fmt.Fprintf(&b, "\nRubric areas still untouched as of the last tick: %s\n", strings.Join(uncovered, "; "))
	} else {
		b.WriteString("\nEvery rubric area has been touched at least once.\n")
	}

	b.WriteString("\n=== CANDIDATE'S CURRENT WORK ===\n")
	b.WriteString(in.Snapshot.Render(in.Problem.Mode))

	if len(in.RecentEvents) > 0 {
		b.WriteString("\n=== WHAT YOU ALREADY SAID (do not repeat any of this) ===\n")
		for _, e := range in.RecentEvents {
			fmt.Fprintf(&b, "- [%s at %s] %s: %s\n", e.Kind, humanDuration(e.ElapsedSec), e.Title, e.Body)
		}
	} else {
		b.WriteString("\n=== WHAT YOU ALREADY SAID ===\n(nothing yet)\n")
	}

	b.WriteString("\nDecide whether to speak now. Silence is the default.\n")
	return b.String()
}

func buildMessagePrompt(in CoachInput, d CoachDecision) string {
	var b strings.Builder

	b.WriteString("=== EXERCISE ===\n")
	b.WriteString(in.Problem.Render(false))

	b.WriteString("\n=== THEIR CURRENT WORK ===\n")
	b.WriteString(in.Snapshot.Render(in.Problem.Mode))

	fmt.Fprintf(&b, "\n=== TIME ===\n%s remaining out of %s.\n",
		humanDuration(in.RemainingSec), humanDuration(in.ElapsedSec+in.RemainingSec))

	fmt.Fprintf(&b, "\n=== THE MESSAGE ===\nTone: %s\nHeadline already shown to them: %s\nThe one point to make: %s\n",
		toneFor(d.Kind), d.Title, d.Point)

	b.WriteString("\nWrite the message body now.\n")
	return b.String()
}

func toneFor(k domain.EventKind) string {
	switch k {
	case domain.KindProbe:
		return "questioning a choice they already made"
	case domain.KindPraise:
		return "confirming something they did well, briefly"
	default:
		return "pointing at a gap without naming the fix"
	}
}

// PacingEvent returns a deterministic timebox nudge, or nil. This runs in Go
// rather than through the LLM because pacing is arithmetic: it should fire
// reliably at the same points regardless of which model is loaded.
func PacingEvent(p domain.Problem, elapsedSec, remainingSec int, covered []domain.Area, alreadyFired map[string]bool) *domain.Event {
	total := elapsedSec + remainingSec
	if total <= 0 {
		return nil
	}
	pct := elapsedSec * 100 / total

	coveredSet := map[domain.Area]bool{}
	for _, a := range covered {
		coveredSet[a] = true
	}
	hit := 0
	for _, c := range p.Rubric {
		if coveredSet[c.Area] {
			hit++
		}
	}
	coverage := 0
	if len(p.Rubric) > 0 {
		coverage = hit * 100 / len(p.Rubric)
	}

	switch {
	case pct >= 50 && pct < 60 && !alreadyFired["half"] && coverage < 40:
		alreadyFired["half"] = true
		return &domain.Event{
			ElapsedSec: elapsedSec,
			Kind:       domain.KindPacing,
			Severity:   domain.SeverityWarn,
			Title:      "Half the clock, a third of the ground",
			Body: fmt.Sprintf("You are %d%% through the timebox with roughly %d%% of the rubric touched. "+
				"Get a rough version of everything down before deepening any one part.", pct, coverage),
		}
	case pct >= 80 && !alreadyFired["wrapup"]:
		alreadyFired["wrapup"] = true
		body := "Last stretch. Spend it on whatever a reviewer would notice missing, not on polish."
		if coverage < 70 {
			body = fmt.Sprintf("Last stretch, and about %d%% of the rubric is covered. "+
				"Name the gaps in your notes even if you cannot build them; saying what you would do next counts.", coverage)
		}
		return &domain.Event{
			ElapsedSec: elapsedSec,
			Kind:       domain.KindPacing,
			Severity:   domain.SeverityWarn,
			Title:      "20% of the time left",
			Body:       body,
		}
	}
	return nil
}

func coachKind(s string) domain.EventKind {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "probe":
		return domain.KindProbe
	case "praise":
		return domain.KindPraise
	default:
		return domain.KindHint
	}
}

func coachSeverity(s string) domain.Severity {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "warn":
		return domain.SeverityWarn
	case "critical":
		return domain.SeverityCritical
	default:
		return domain.SeverityInfo
	}
}

func filterAreas(mode domain.Mode, in []string) []domain.Area {
	seen := map[domain.Area]bool{}
	out := make([]domain.Area, 0, len(in))
	for _, s := range in {
		a := domain.Area(strings.TrimSpace(s))
		if !domain.ValidArea(mode, a) || seen[a] {
			continue
		}
		seen[a] = true
		out = append(out, a)
	}
	return out
}

func pctElapsed(elapsed, remaining int) int {
	total := elapsed + remaining
	if total <= 0 {
		return 100
	}
	return elapsed * 100 / total
}

func humanDuration(sec int) string {
	if sec < 0 {
		sec = 0
	}
	m, s := sec/60, sec%60
	if m == 0 {
		return fmt.Sprintf("%ds", s)
	}
	return fmt.Sprintf("%dm%02ds", m, s)
}
