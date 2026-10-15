package harness

import "strings"

// materialBudget is how much of a chapter a prompt carries, in bytes. Chapters
// run to tens of kilobytes and a local model's window is 16k tokens, most of
// which the exercise, the rubric, and the work already need.
const materialBudget = 9000

// Digest fits a markdown document into budget bytes without losing its shape.
// A document that fits is returned whole. Otherwise the result is every heading
// (the outline survives even when the body cannot) followed by as much of the
// opening as the budget leaves, because the opening is where a chapter says
// what it is about.
func Digest(markdown string, budget int) string {
	markdown = strings.TrimSpace(markdown)
	if len(markdown) <= budget {
		return markdown
	}

	var outline strings.Builder
	fence := false
	for _, line := range strings.Split(markdown, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") || strings.HasPrefix(trimmed, "~~~") {
			fence = !fence
			continue
		}
		if !fence && strings.HasPrefix(line, "#") {
			outline.WriteString(line)
			outline.WriteByte('\n')
		}
	}

	head := outline.String()
	if len(head) > budget/2 {
		head = clip(head, budget/2)
	}
	body := clip(markdown, budget-len(head))
	return "OUTLINE:\n" + head + "\nOPENING:\n" + body + "\n[...chapter continues]"
}

// clip cuts s to at most n bytes on a line boundary where one is near, and
// never inside a UTF-8 sequence.
func clip(s string, n int) string {
	if n <= 0 {
		return ""
	}
	if len(s) <= n {
		return s
	}
	for n > 0 && s[n]&0xC0 == 0x80 {
		n--
	}
	cut := s[:n]
	if nl := strings.LastIndexByte(cut, '\n'); nl > n*3/4 {
		cut = cut[:nl]
	}
	return cut
}
