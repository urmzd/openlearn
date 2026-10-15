// Package source is where RAG documents come from (ag.06): a Source returns
// raw documents, ToDoc turns one into plain text, DirSource reads a local
// directory, and Crawler (crawler.go) fetches web pages behind an SSRF
// guard, robots.txt, and a per-host rate limit.
//
// Chapter: content/tracks/ai-platform-engineering/07-retrieval-and-rag/06-rag-ingest.md.
// Format: course/contracts/formats/rag-index.md (doc_id, source, uri).
package source

import (
	"context"
	"html"
	"io/fs"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// RawDoc is a document as fetched.
type RawDoc struct {
	URI         string // a URL, or a slash-separated file path
	Source      string // "web" or "file"
	ContentType string // text/html, text/markdown, text/plain
	Body        []byte
}

// Doc is a document as text. ID is its doc_id in the index.
type Doc struct {
	ID, URI, Source, Title, Text string
}

// Source produces raw documents.
type Source interface {
	Fetch(ctx context.Context) ([]RawDoc, error)
}

var (
	dropRE  = regexp.MustCompile(`(?is)<(script|style|noscript|head)\b.*?</(script|style|noscript|head)\s*>`)
	titleRE = regexp.MustCompile(`(?is)<title\b[^>]*>(.*?)</title\s*>`)
	blockRE = regexp.MustCompile(`(?i)</?(p|div|section|article|li|ul|ol|h[1-6]|br|tr|table|pre|blockquote)\b[^>]*>`)
	tagRE   = regexp.MustCompile(`(?s)<[^>]*>`)
	spaceRE = regexp.MustCompile(`[ \t\r\f\v]+`)
	blankRE = regexp.MustCompile(`\n\s*\n+`)
	h1RE    = regexp.MustCompile(`(?m)^#\s+(.+)$`)
)

// DocID is a document's id: a URL without its fragment, scheme and host
// lowercased; a file path as given (slash-separated, cleaned).
func DocID(r RawDoc) string {
	// SOLUTION-BEGIN ag.06
	if r.Source == "file" {
		return filepath.ToSlash(filepath.Clean(r.URI))
	}
	u, err := url.Parse(r.URI)
	if err != nil {
		return r.URI
	}
	u.Fragment, u.RawFragment = "", ""
	u.Scheme, u.Host = strings.ToLower(u.Scheme), strings.ToLower(u.Host)
	return u.String()
	// SOLUTION-END
}

// ToDoc extracts the text. HTML loses script, style, and head content and
// every tag; block tags become paragraph breaks; entities are decoded; runs
// of spaces collapse and paragraphs are separated by one blank line. Other
// types keep their text with CRLF made LF. The title is the HTML <title> or
// the first "# " heading.
func ToDoc(r RawDoc) Doc {
	// SOLUTION-BEGIN ag.06
	d := Doc{ID: DocID(r), URI: r.URI, Source: r.Source}
	text := strings.ReplaceAll(string(r.Body), "\r\n", "\n")
	if strings.Contains(strings.ToLower(r.ContentType), "html") {
		if m := titleRE.FindStringSubmatch(text); m != nil {
			d.Title = strings.TrimSpace(html.UnescapeString(tagRE.ReplaceAllString(m[1], "")))
		}
		text = dropRE.ReplaceAllString(text, " ")
		text = blockRE.ReplaceAllString(text, "\n\n")
		text = html.UnescapeString(tagRE.ReplaceAllString(text, ""))
	} else if m := h1RE.FindStringSubmatch(text); m != nil {
		d.Title = strings.TrimSpace(m[1])
	}
	lines := strings.Split(text, "\n")
	for i, l := range lines {
		lines[i] = strings.TrimSpace(spaceRE.ReplaceAllString(l, " "))
	}
	d.Text = strings.TrimSpace(blankRE.ReplaceAllString(strings.Join(lines, "\n"), "\n\n"))
	return d
	// SOLUTION-END
}

// DirSource reads every file under Root whose extension is in Exts (default
// .md, .txt), in path order. URIs are Root joined with the relative path.
type DirSource struct {
	Root string
	Exts []string
}

func (s DirSource) Fetch(ctx context.Context) ([]RawDoc, error) {
	// SOLUTION-BEGIN ag.06
	exts := s.Exts
	if len(exts) == 0 {
		exts = []string{".md", ".txt"}
	}
	var paths []string
	err := filepath.WalkDir(s.Root, func(p string, e fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if e.IsDir() {
			return nil
		}
		for _, x := range exts {
			if strings.EqualFold(filepath.Ext(p), x) {
				paths = append(paths, p)
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Strings(paths)
	out := make([]RawDoc, 0, len(paths))
	for _, p := range paths {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return nil, err
		}
		ct := "text/plain"
		if strings.EqualFold(filepath.Ext(p), ".md") {
			ct = "text/markdown"
		}
		out = append(out, RawDoc{URI: filepath.ToSlash(p), Source: "file", ContentType: ct, Body: b})
	}
	return out, nil
	// SOLUTION-END
}
