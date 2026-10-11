package domain

import (
	"fmt"
	"sort"
	"strings"
)

// Render turns a snapshot into the compact text the LLM reads. Keeping this in
// one place means the coach and the reviewer always see the work the same way.
func (s Snapshot) Render(mode Mode) string {
	var b strings.Builder
	if mode == ModeCoding {
		lang := s.Language
		if lang == "" {
			lang = "text"
		}
		code := strings.TrimSpace(s.Code)
		if code == "" {
			b.WriteString("CODE: (empty - nothing written yet)\n")
		} else {
			fmt.Fprintf(&b, "CODE (%s, %d lines):\n```%s\n%s\n```\n", lang, strings.Count(code, "\n")+1, lang, code)
		}
	} else {
		b.WriteString(s.renderBoard())
	}
	if notes := strings.TrimSpace(s.Notes); notes != "" {
		fmt.Fprintf(&b, "\nNOTES / TALKING TRACK:\n%s\n", notes)
	} else {
		b.WriteString("\nNOTES / TALKING TRACK: (empty)\n")
	}
	return b.String()
}

func (s Snapshot) renderBoard() string {
	if len(s.Nodes) == 0 {
		return "BOARD: (empty - no components placed yet)\n"
	}
	labels := make(map[string]string, len(s.Nodes))
	for _, n := range s.Nodes {
		label := strings.TrimSpace(n.Label)
		if label == "" {
			label = "(unlabeled " + string(n.Kind) + ")"
		}
		labels[n.ID] = label
	}

	nodes := append([]BoardNode(nil), s.Nodes...)
	sort.Slice(nodes, func(i, j int) bool { return nodes[i].X < nodes[j].X })

	var b strings.Builder
	fmt.Fprintf(&b, "BOARD: %d components, %d connections\n\nCOMPONENTS:\n", len(s.Nodes), len(s.Edges))
	for _, n := range nodes {
		fmt.Fprintf(&b, "- %s [%s]", labels[n.ID], n.Kind)
		if d := strings.TrimSpace(n.Detail); d != "" {
			fmt.Fprintf(&b, " :: %s", strings.ReplaceAll(d, "\n", " / "))
		}
		b.WriteString("\n")
	}

	if len(s.Edges) == 0 {
		b.WriteString("\nCONNECTIONS: (none - components are not wired together)\n")
		return b.String()
	}
	b.WriteString("\nCONNECTIONS:\n")
	for _, e := range s.Edges {
		src, ok := labels[e.Source]
		if !ok {
			src = e.Source
		}
		dst, ok := labels[e.Target]
		if !ok {
			dst = e.Target
		}
		if l := strings.TrimSpace(e.Label); l != "" {
			fmt.Fprintf(&b, "- %s --[%s]--> %s\n", src, l, dst)
		} else {
			fmt.Fprintf(&b, "- %s --> %s (unlabeled)\n", src, dst)
		}
	}
	return b.String()
}

// IsEmpty reports whether the user has produced nothing yet. The coach uses it
// to stay quiet early instead of lecturing an empty canvas.
func (s Snapshot) IsEmpty() bool {
	return len(s.Nodes) == 0 &&
		strings.TrimSpace(s.Code) == "" &&
		strings.TrimSpace(s.Notes) == ""
}

// Fingerprint is a cheap change detector so the coach skips ticks where the
// user has not actually done anything since the last one.
func (s Snapshot) Fingerprint() string {
	var b strings.Builder
	for _, n := range s.Nodes {
		fmt.Fprintf(&b, "%s|%s|%s|%s;", n.ID, n.Kind, n.Label, n.Detail)
	}
	for _, e := range s.Edges {
		fmt.Fprintf(&b, "%s>%s|%s;", e.Source, e.Target, e.Label)
	}
	b.WriteString(s.Notes)
	b.WriteString("\x00")
	b.WriteString(s.Code)
	return b.String()
}

// RenderProblem formats the problem for a prompt, optionally including the
// hidden rubric and reference outline (review only, never during a session).
func (p Problem) Render(includeHidden bool) string {
	var b strings.Builder
	fmt.Fprintf(&b, "TITLE: %s\nLEVEL: %s\n\nSTATEMENT:\n%s\n", p.Title, p.Level, p.Statement)
	if len(p.Requirements) > 0 {
		b.WriteString("\nREQUIREMENTS:\n")
		for _, r := range p.Requirements {
			fmt.Fprintf(&b, "- %s\n", r)
		}
	}
	if len(p.Constraints) > 0 {
		b.WriteString("\nCONSTRAINTS:\n")
		for _, c := range p.Constraints {
			fmt.Fprintf(&b, "- %s\n", c)
		}
	}
	if !includeHidden {
		return b.String()
	}
	if len(p.Rubric) > 0 {
		b.WriteString("\nRUBRIC (hidden from the candidate):\n")
		for _, c := range p.Rubric {
			fmt.Fprintf(&b, "- id=%s area=%s weight=%d :: %s - %s\n", c.ID, c.Area, c.Weight, c.Title, c.Detail)
		}
	}
	if len(p.ReferenceOutline) > 0 {
		b.WriteString("\nREFERENCE OUTLINE (what a strong answer covers):\n")
		for _, r := range p.ReferenceOutline {
			fmt.Fprintf(&b, "- %s\n", r)
		}
	}
	return b.String()
}
