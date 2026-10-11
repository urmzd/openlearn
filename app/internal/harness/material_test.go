package harness

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func TestDigestReturnsShortDocumentsWhole(t *testing.T) {
	if got := Digest("  # Title\nbody\n", 100); got != "# Title\nbody" {
		t.Errorf("got %q", got)
	}
}

func TestDigestKeepsEveryHeadingWhenTheBodyDoesNotFit(t *testing.T) {
	doc := "# Chapter\n" + strings.Repeat("opening prose. ", 200) +
		"\n```\n# a comment, not a heading\n```\n## Late section\n" + strings.Repeat("ünïcödé ", 400)
	got := Digest(doc, 1200)
	if len(got) > 1200+len("OUTLINE:\n\nOPENING:\n\n[...chapter continues]") {
		t.Errorf("digest is %d bytes, over budget", len(got))
	}
	if !strings.Contains(got, "## Late section") {
		t.Error("a heading past the cut must survive in the outline")
	}
	if strings.Contains(got, "# a comment, not a heading") {
		t.Error("a # line inside a code fence is not a heading")
	}
	if !utf8.ValidString(got) {
		t.Error("digest cut a UTF-8 sequence in half")
	}
}
