package practice

import (
	"regexp"
	"strings"
)

// A reference marks the parts the learner is meant to write with begin and end
// markers on their own lines, in whatever comment syntax the language uses:
//
//	/* SOLUTION-BEGIN */        # SOLUTION-BEGIN
//	...                         ...
//	/* SOLUTION-END */          # SOLUTION-END
//
// One committed file is therefore both the reference and the source of the
// stub, so the two cannot drift apart.
//
// A marker only counts when its line holds nothing but comment punctuation and
// the marker. Prose that mentions a marker by name, in a header comment or a
// README, is not one.
var (
	beginRe = regexp.MustCompile(`^[^A-Za-z0-9]*SOLUTION-BEGIN( [A-Za-z][A-Za-z0-9.+-]*)?[^A-Za-z0-9]*$`)
	endRe   = regexp.MustCompile(`^[^A-Za-z0-9]*SOLUTION-END[^A-Za-z0-9]*$`)
	tagRe   = regexp.MustCompile(`SOLUTION-BEGIN( [A-Za-z][A-Za-z0-9.+-]*)?`)
)

// todo is what a stripped region becomes.
const todo = "TODO: your implementation goes here."

// StripSolution replaces each marked region with a single TODO line. The TODO
// is made by substituting into the begin marker's own line, not by keeping the
// text before it: otherwise a C-style "/* MARKER */" would become "/* TODO"
// with no closing "*/", an unterminated comment that swallows the next
// function.
func StripSolution(data []byte) []byte {
	lines := strings.Split(string(data), "\n")
	out := make([]string, 0, len(lines))
	skipping := false
	for _, line := range lines {
		bare := strings.TrimRight(line, "\r")
		switch {
		case beginRe.MatchString(bare):
			out = append(out, tagRe.ReplaceAllLiteralString(line, todo))
			skipping = true
		case endRe.MatchString(bare):
			skipping = false
		case !skipping:
			out = append(out, line)
		}
	}
	return []byte(strings.Join(out, "\n"))
}

// DropMarkers removes the marker lines and keeps everything between them: the
// reference as the learner would have written it.
func DropMarkers(data []byte) []byte {
	lines := strings.Split(string(data), "\n")
	out := make([]string, 0, len(lines))
	for _, line := range lines {
		bare := strings.TrimRight(line, "\r")
		if beginRe.MatchString(bare) || endRe.MatchString(bare) {
			continue
		}
		out = append(out, line)
	}
	return []byte(strings.Join(out, "\n"))
}

// Diff renders a line diff of two texts, a's lines prefixed "-" and b's "+",
// with unchanged lines between changes kept for context.
func Diff(aLabel, bLabel, a, b string) string {
	x, y := strings.Split(a, "\n"), strings.Split(b, "\n")
	// Longest common subsequence, by the standard table. Exercise files are a
	// few hundred lines, so the quadratic table is small.
	lcs := make([][]int, len(x)+1)
	for i := range lcs {
		lcs[i] = make([]int, len(y)+1)
	}
	for i := len(x) - 1; i >= 0; i-- {
		for j := len(y) - 1; j >= 0; j-- {
			switch {
			case x[i] == y[j]:
				lcs[i][j] = lcs[i+1][j+1] + 1
			case lcs[i+1][j] >= lcs[i][j+1]:
				lcs[i][j] = lcs[i+1][j]
			default:
				lcs[i][j] = lcs[i][j+1]
			}
		}
	}
	var out strings.Builder
	out.WriteString("--- " + aLabel + "\n+++ " + bLabel + "\n")
	i, j := 0, 0
	for i < len(x) && j < len(y) {
		switch {
		case x[i] == y[j]:
			out.WriteString("  " + x[i] + "\n")
			i++
			j++
		case lcs[i+1][j] >= lcs[i][j+1]:
			out.WriteString("- " + x[i] + "\n")
			i++
		default:
			out.WriteString("+ " + y[j] + "\n")
			j++
		}
	}
	for ; i < len(x); i++ {
		out.WriteString("- " + x[i] + "\n")
	}
	for ; j < len(y); j++ {
		out.WriteString("+ " + y[j] + "\n")
	}
	return strings.TrimRight(out.String(), "\n")
}
