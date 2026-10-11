package course

import (
	"io/fs"
	"path"
	"sort"
	"strings"
	"sync"
)

// Hit is one document matching a search.
type Hit struct {
	Path  string `json:"path"`
	Title string `json:"title"`
	// Snippet is the first matching line, trimmed for display.
	Snippet string `json:"snippet"`
	// Matches counts occurrences of the first term, for ranking and display.
	Matches int `json:"matches"`
}

type indexed struct {
	ref   DocRef
	lower string
	text  string
}

type index struct {
	once sync.Once
	docs []indexed
}

// Search finds documents containing every whitespace-separated term of query,
// ranked with title matches first. The corpus is a few megabytes of markdown,
// so a scan over lowercased text is fast enough and needs no index to maintain.
func (c *Course) Search(query string, limit int) []Hit {
	terms := strings.Fields(strings.ToLower(query))
	if len(terms) == 0 {
		return []Hit{}
	}
	c.idx.once.Do(c.loadIndex)

	type scored struct {
		hit   Hit
		score int
	}
	var found []scored
	for _, d := range c.idx.docs {
		title := strings.ToLower(d.ref.Title)
		score, ok := 0, true
		for _, t := range terms {
			n := strings.Count(d.lower, t)
			if n == 0 {
				ok = false
				break
			}
			score += n
			if strings.Contains(title, t) {
				score += 1000
			}
		}
		if !ok {
			continue
		}
		found = append(found, scored{
			hit: Hit{
				Path:    d.ref.Path,
				Title:   d.ref.Title,
				Snippet: snippet(d.text, d.lower, terms[0]),
				Matches: strings.Count(d.lower, terms[0]),
			},
			score: score,
		})
	}
	sort.SliceStable(found, func(i, j int) bool { return found[i].score > found[j].score })
	if limit > 0 && len(found) > limit {
		found = found[:limit]
	}
	out := make([]Hit, len(found))
	for i, f := range found {
		out[i] = f.hit
	}
	return out
}

func (c *Course) loadIndex() {
	seen := map[string]bool{}
	add := func(ref DocRef) {
		if seen[ref.Path] {
			return
		}
		seen[ref.Path] = true
		data, err := fs.ReadFile(c.fsys, ref.Path)
		if err != nil {
			return
		}
		text := string(data)
		c.idx.docs = append(c.idx.docs, indexed{ref: ref, lower: strings.ToLower(text), text: text})
	}
	o := c.Outline()
	for _, ref := range o.Start {
		add(ref)
	}
	for _, group := range [][]Track{o.Tracks, o.Reference} {
		for _, t := range group {
			for _, ref := range t.Docs {
				add(ref)
			}
			for _, topic := range t.Topics {
				for _, ref := range topic.Docs {
					add(ref)
				}
			}
		}
	}
	for _, name := range c.PathNames() {
		p := path.Join(c.m.Paths, name, "README.md")
		add(DocRef{Path: p, Title: c.title(p)})
	}
}

// snippet returns the line holding the first match of term, clipped around it.
func snippet(text, lower, term string) string {
	at := strings.Index(lower, term)
	if at < 0 || at > len(text) {
		return ""
	}
	start := strings.LastIndexByte(text[:at], '\n') + 1
	end := len(text)
	if nl := strings.IndexByte(text[at:], '\n'); nl >= 0 {
		end = at + nl
	}
	line := strings.TrimSpace(text[start:end])
	const width = 180
	if len(line) <= width {
		return line
	}
	off := at - start
	from := off - width/3
	if from < 0 {
		from = 0
	}
	to := from + width
	if to > len(line) {
		to = len(line)
	}
	// Step to rune boundaries so a clipped snippet is still valid UTF-8.
	for from > 0 && !isRuneStart(line[from]) {
		from--
	}
	for to < len(line) && !isRuneStart(line[to]) {
		to++
	}
	out := line[from:to]
	if from > 0 {
		out = "…" + out
	}
	if to < len(line) {
		out += "…"
	}
	return out
}

func isRuneStart(b byte) bool { return b&0xC0 != 0x80 }
