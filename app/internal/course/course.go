// Package course is the platform's content mechanism. A course is a directory
// tree of markdown with a manifest at its root: tracks to read, role paths that
// order them, and exercises. The app renders whatever course it is handed
// and knows nothing about any one of them; X// the course that ships inside the binary.
package course

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"path"
	"sort"
	"strings"
	"sync"
)

// ManifestName is the file that makes a directory a course.
const ManifestName = "openlearn.json"

// Manifest is the course's table of contents. Everything in it is a path
// relative to the course root.
type Manifest struct {
	ID      string `json:"id"`
	Title   string `json:"title"`
	Tagline string `json:"tagline"`
	// Repo is where links to non-markdown files (code, configs) open.
	Repo string `json:"repo"`
	// Paths is the directory holding one <name>/path.tsv per role path.
	Paths string `json:"paths"`
	// Featured names the path the app leads with.
	Featured string `json:"featured"`
	// Practice is the pack's exercise spec (see internal/practice). Exercises
	// only run from a pack on disk.
	Practice string `json:"practice"`
	// StageCheck is an optional command that decides whether a path stage tied
	// to a check may be marked finished. {path} and {stage} are filled in, and
	// exit status 0 means it may. Without it, such stages cannot be marked here.
	StageCheck []string `json:"stageCheck"`
	// Modes are the session kinds that make sense for this material, in the
	// order the app offers them. Empty means all of them.
	Modes []string `json:"modes"`
	// Start lists the documents a new reader opens first.
	Start []Entry `json:"start"`
	// Tracks are read in the order given.
	Tracks []TrackRef `json:"tracks"`
	// Reference trees are browsable but are not part of the learning order.
	Reference []TrackRef `json:"reference"`
}

// Entry is one named document.
type Entry struct {
	File  string `json:"file"`
	Title string `json:"title"`
}

// TrackRef names a directory of topics.
type TrackRef struct {
	Dir   string `json:"dir"`
	Title string `json:"title"`
}

// Info is what the UI needs to know about the loaded course.
type Info struct {
	ID      string `json:"id"`
	Title   string `json:"title"`
	Tagline string `json:"tagline"`
	Repo    string `json:"repo"`
	// Root is the checkout the course was read from. Empty for the built-in
	// course, which has no directory to run exercises in.
	Root     string `json:"root"`
	BuiltIn  bool   `json:"builtIn"`
	Featured string `json:"featured"`
	// Practice reports whether the pack has exercises that can run here.
	Practice bool     `json:"practice"`
	Modes    []string `json:"modes"`
}

// Outline is the browsable tree: start pages, then tracks, then reference.
type Outline struct {
	Start     []DocRef `json:"start"`
	Tracks    []Track  `json:"tracks"`
	Reference []Track  `json:"reference"`
}

// Track is one directory of topics.
type Track struct {
	Dir   string `json:"dir"`
	Title string `json:"title"`
	// Docs are markdown files directly in the track directory.
	Docs   []DocRef `json:"docs"`
	Topics []Topic  `json:"topics"`
}

// Topic is one subdirectory of a track.
type Topic struct {
	Dir   string   `json:"dir"`
	Title string   `json:"title"`
	Docs  []DocRef `json:"docs"`
}

// DocRef points at one markdown document.
type DocRef struct {
	Path  string `json:"path"`
	Title string `json:"title"`
}

// Doc is one markdown document, ready to render.
type Doc struct {
	Path     string `json:"path"`
	Title    string `json:"title"`
	Markdown string `json:"markdown"`
}

// Course is a loaded course. It is safe for concurrent use.
type Course struct {
	fsys fs.FS
	dir  string
	m    Manifest

	once    sync.Once
	outline Outline
	idx     index
}

// ErrNoManifest means the directory is not a course.
var ErrNoManifest = errors.New("course: no " + ManifestName)

// Open loads a course from fsys. dir is the directory fsys was opened on, or
// empty when the course is not on disk.
func Open(fsys fs.FS, dir string) (*Course, error) {
	data, err := fs.ReadFile(fsys, ManifestName)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, ErrNoManifest
		}
		return nil, fmt.Errorf("course: read manifest: %w", err)
	}
	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("course: parse %s: %w", ManifestName, err)
	}
	if m.ID == "" || m.Title == "" {
		return nil, fmt.Errorf("course: %s needs an id and a title", ManifestName)
	}
	if m.Paths == "" {
		m.Paths = "paths"
	}
	return &Course{fsys: fsys, dir: dir, m: m}, nil
}

// Empty returns a course with nothing in it. It stands in when no pack can be
// opened, so callers always have a course to ask instead of a nil to check.
func Empty() *Course {
	return &Course{fsys: nothing{}, m: Manifest{ID: "none", Title: "No course pack", Paths: "paths"}}
}

type nothing struct{}

func (nothing) Open(name string) (fs.File, error) {
	return nil, &fs.PathError{Op: "open", Path: name, Err: fs.ErrNotExist}
}

// Manifest returns the course's manifest.
func (c *Course) Manifest() Manifest { return c.m }

// Dir is the checkout the course was read from, or empty when it is built in.
func (c *Course) Dir() string { return c.dir }

// FS is the tree the course reads from.
func (c *Course) FS() fs.FS { return c.fsys }

// Info describes the loaded course.
func (c *Course) Info() Info {
	return Info{
		ID:       c.m.ID,
		Title:    c.m.Title,
		Tagline:  c.m.Tagline,
		Repo:     c.m.Repo,
		Root:     c.dir,
		BuiltIn:  c.dir == "",
		Featured: c.m.Featured,
		Practice: c.dir != "" && c.m.Practice != "",
		Modes:    c.m.Modes,
	}
}

// Doc reads one markdown document by its course-relative path.
func (c *Course) Doc(p string) (Doc, error) {
	p = path.Clean(p)
	if !fs.ValidPath(p) || !strings.HasSuffix(p, ".md") {
		return Doc{}, fmt.Errorf("course: not a document: %q", p)
	}
	data, err := fs.ReadFile(c.fsys, p)
	if err != nil {
		return Doc{}, fmt.Errorf("course: read %s: %w", p, err)
	}
	return Doc{Path: p, Title: titleOf(data, p), Markdown: string(data)}, nil
}

// Outline returns the browsable tree. It reads every document's title, so it is
// built once and kept.
func (c *Course) Outline() Outline {
	c.once.Do(func() {
		for _, e := range c.m.Start {
			if _, err := fs.Stat(c.fsys, e.File); err == nil {
				c.outline.Start = append(c.outline.Start, DocRef{Path: e.File, Title: e.Title})
			}
		}
		c.outline.Tracks = c.tracks(c.m.Tracks)
		c.outline.Reference = c.tracks(c.m.Reference)
	})
	return c.outline
}

func (c *Course) tracks(refs []TrackRef) []Track {
	out := make([]Track, 0, len(refs))
	for _, ref := range refs {
		t := Track{Dir: ref.Dir, Title: ref.Title, Docs: []DocRef{}, Topics: []Topic{}}
		byTopic := map[string][]DocRef{}
		err := fs.WalkDir(c.fsys, ref.Dir, func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				if skipDir(d.Name()) && p != ref.Dir {
					return fs.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(p, ".md") {
				return nil
			}
			ref := DocRef{Path: p, Title: c.title(p)}
			rest := strings.TrimPrefix(p, t.Dir+"/")
			topic, _, nested := strings.Cut(rest, "/")
			if !nested {
				t.Docs = append(t.Docs, ref)
				return nil
			}
			byTopic[topic] = append(byTopic[topic], ref)
			return nil
		})
		if err != nil {
			continue // a track the manifest names but the tree lacks is skipped
		}
		readmeFirst(t.Docs)
		names := make([]string, 0, len(byTopic))
		for name := range byTopic {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			docs := byTopic[name]
			readmeFirst(docs)
			dir := path.Join(t.Dir, name)
			title := name
			if docs[0].Path == path.Join(dir, "README.md") {
				title = docs[0].Title
			}
			t.Topics = append(t.Topics, Topic{Dir: dir, Title: title, Docs: docs})
		}
		out = append(out, t)
	}
	return out
}

func (c *Course) title(p string) string {
	data, err := fs.ReadFile(c.fsys, p)
	if err != nil {
		return path.Base(p)
	}
	return titleOf(data, p)
}

// skipDir keeps dependency and build trees out of the outline when the course
// is read from a working checkout rather than from a bundle.
func skipDir(name string) bool {
	switch name {
	case "node_modules", "target", "dist", "build", "__pycache__":
		return true
	}
	return strings.HasPrefix(name, ".")
}

// readmeFirst orders docs by path with a README ahead of its siblings, which
// is the order a reader meets them in.
func readmeFirst(docs []DocRef) {
	key := func(p string) string {
		if path.Base(p) == "README.md" {
			return path.Dir(p) + "/"
		}
		return p
	}
	sort.SliceStable(docs, func(i, j int) bool { return key(docs[i].Path) < key(docs[j].Path) })
}

// titleOf returns a document's first level-one heading with inline markup
// removed, or a name derived from its path.
func titleOf(data []byte, p string) string {
	sc := bufio.NewScanner(bytes.NewReader(data))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	fence := ""
	for sc.Scan() {
		line := sc.Text()
		trimmed := strings.TrimSpace(line)
		if marker := fenceMarker(trimmed); marker != "" {
			switch {
			case fence == "":
				fence = marker
			case strings.HasPrefix(marker, fence):
				fence = ""
			}
			continue
		}
		if fence == "" && strings.HasPrefix(line, "# ") {
			if t := stripInline(line[2:]); t != "" {
				return t
			}
		}
	}
	if base := path.Base(p); base != "README.md" {
		return strings.TrimSuffix(base, ".md")
	}
	return path.Base(path.Dir(p))
}

func fenceMarker(line string) string {
	for _, ch := range []string{"`", "~"} {
		if strings.HasPrefix(line, strings.Repeat(ch, 3)) {
			n := len(line) - len(strings.TrimLeft(line, ch))
			return strings.Repeat(ch, n)
		}
	}
	return ""
}

var inlineMarkup = strings.NewReplacer("`", "", "*", "", "_", " ")

// stripInline reduces "[text](url)" to "text" and drops emphasis markers.
func stripInline(s string) string {
	var b strings.Builder
	for len(s) > 0 {
		open := strings.IndexByte(s, '[')
		if open < 0 {
			break
		}
		mid := strings.Index(s[open:], "](")
		if mid < 0 {
			break
		}
		end := strings.IndexByte(s[open+mid:], ')')
		if end < 0 {
			break
		}
		b.WriteString(s[:open])
		b.WriteString(s[open+1 : open+mid])
		s = s[open+mid+end+1:]
	}
	b.WriteString(s)
	return strings.Join(strings.Fields(inlineMarkup.Replace(b.String())), " ")
}
