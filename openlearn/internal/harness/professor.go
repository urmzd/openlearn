package harness

import (
	"context"
	"fmt"
	"strings"

	"github.com/urmzd/supersource/openlearn/internal/llm"
)

// The professor is the untimed half of the assistant. The coach watches a
// session and mostly stays quiet; the professor answers when asked, about the
// page the learner has open.

// Turn is one exchange line in a professor conversation.
type Turn struct {
	// Role is "user" or "assistant".
	Role string `json:"role"`
	Text string `json:"text"`
}

// ProfessorInput is everything one answer is grounded in.
type ProfessorInput struct {
	// Course is the pack's title.
	Course string
	// DocTitle and Material are the page the learner has open. Both may be
	// empty when they ask from somewhere with no page.
	DocTitle string
	Material string
	// Progress is a short plain-text account of where the learner is.
	Progress string
	// History is the conversation so far, oldest first.
	History  []Turn
	Question string
}

const professorSystem = `You are a professor guiding one student through a self-paced course. You have the page they are reading in front of you.

How you teach:
- Answer the question that was asked, first, in plain words. Then stop or add one thing that deepens it.
- Ground what you say in the page when it covers the question. When it does not, say so and answer from your own knowledge.
- When the student is working toward an answer themselves, ask the one question that moves them forward instead of handing it over. When they ask you to just explain, explain.
- Use a small concrete example over a general statement.
- If you are not sure, say what you are unsure about. A confident wrong answer costs a learner more than an honest gap.
- Never claim they have finished or passed something. Their progress is recorded by the course, not by you.

Write markdown. Use LaTeX in $...$ for math. Keep it short enough to read in a minute unless they ask for depth.`

// historyTurns bounds how much conversation a prompt carries.
const historyTurns = 8

// Ask streams the professor's answer through onDelta and returns the full text.
func Ask(ctx context.Context, c *llm.Client, in ProfessorInput, onDelta func(string)) (string, error) {
	return c.TextStream(ctx, professorSystem, buildProfessorPrompt(in), onDelta)
}

func buildProfessorPrompt(in ProfessorInput) string {
	var b strings.Builder
	if in.Course != "" {
		fmt.Fprintf(&b, "Course: %s\n", in.Course)
	}
	if p := strings.TrimSpace(in.Progress); p != "" {
		fmt.Fprintf(&b, "Where the student is: %s\n", p)
	}
	if m := strings.TrimSpace(in.Material); m != "" {
		fmt.Fprintf(&b, "\n=== PAGE: %s ===\n%s\n=== END PAGE ===\n", in.DocTitle, Digest(m, materialBudget))
	} else {
		b.WriteString("\nThe student has no page open.\n")
	}

	history := in.History
	if len(history) > historyTurns {
		history = history[len(history)-historyTurns:]
	}
	if len(history) > 0 {
		b.WriteString("\n=== CONVERSATION SO FAR ===\n")
		for _, t := range history {
			who := "Student"
			if t.Role == "assistant" {
				who = "Professor"
			}
			fmt.Fprintf(&b, "%s: %s\n", who, clip(strings.TrimSpace(t.Text), 1200))
		}
	}
	fmt.Fprintf(&b, "\nStudent: %s\n", strings.TrimSpace(in.Question))
	return b.String()
}
