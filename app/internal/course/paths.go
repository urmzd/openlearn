package course

import (
	"bufio"
	"bytes"
	"fmt"
	"io/fs"
	"path"
	"sort"
	"strings"
)

// A role path is <paths>/<name>/path.tsv, one tab-separated row per stage:
//
//	stage  title  files  done-when  [check]
//
// A row whose first column is @<other> includes every stage of that path. The
// format is shared with `ol learn`, which reads the same files;
// this is a second reader of it, not a second definition.

// Stage is one step of a path, with includes already resolved.
type Stage struct {
	// Owner is the path whose path.tsv declares the stage. Progress is kept
	// per owner, so a stage finished in a part counts in every path including it.
	Owner string `json:"owner"`
	// Number is the stage's id within its owner.
	Number string `json:"number"`
	// Key addresses the stage from the path being read: "3" or "part:3".
	Key      string   `json:"key"`
	Title    string   `json:"title"`
	Files    []string `json:"files"`
	DoneWhen string   `json:"doneWhen"`
	// Check ties the stage to a verdict ("module:M01.1"); empty when the
	// reader decides for themselves that it is done.
	Check string `json:"check"`
	Done  bool   `json:"done"`
}

// Path is one role path, expanded.
type Path struct {
	Name   string  `json:"name"`
	Title  string  `json:"title"`
	Readme string  `json:"readme"`
	Stages []Stage `json:"stages"`
}

// PathSummary is one row of the path list.
type PathSummary struct {
	Name     string `json:"name"`
	Title    string `json:"title"`
	Total    int    `json:"total"`
	Done     int    `json:"done"`
	Parts    int    `json:"parts"`
	Featured bool   `json:"featured"`
	// Part reports that another path includes this one whole. A part is still
	// a path in its own right; a list that wants only the top level skips it.
	Part bool `json:"part"`
	// Broken explains why the path could not be expanded.
	Broken string `json:"broken"`
}

// Progress answers which stages of an owning path are finished.
type Progress interface {
	Done(owner string) map[string]bool
}

// PathNames lists every path the course defines, sorted.
func (c *Course) PathNames() []string {
	entries, err := fs.ReadDir(c.fsys, c.m.Paths)
	if err != nil {
		return nil
	}
	var names []string
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		if _, err := fs.Stat(c.fsys, path.Join(c.m.Paths, e.Name(), "path.tsv")); err == nil {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	return names
}

// Path expands one path and marks its finished stages.
func (c *Course) Path(name string, prog Progress) (Path, error) {
	stages, err := c.expand(name, name, nil)
	if err != nil {
		return Path{}, err
	}
	done := map[string]map[string]bool{}
	for i := range stages {
		owner := stages[i].Owner
		if _, ok := done[owner]; !ok && prog != nil {
			done[owner] = prog.Done(owner)
		}
		stages[i].Done = done[owner][stages[i].Number]
	}
	readme := path.Join(c.m.Paths, name, "README.md")
	return Path{Name: name, Title: c.title(readme), Readme: readme, Stages: stages}, nil
}

// Paths summarizes every path, featured one first.
func (c *Course) Paths(prog Progress) []PathSummary {
	names := c.PathNames()
	out := make([]PathSummary, 0, len(names))
	included := map[string]bool{}
	for _, name := range names {
		sum := PathSummary{Name: name, Featured: name == c.m.Featured}
		p, err := c.Path(name, prog)
		if err != nil {
			sum.Title = name
			sum.Broken = err.Error()
			out = append(out, sum)
			continue
		}
		sum.Title = p.Title
		parts := map[string]bool{}
		for _, s := range p.Stages {
			sum.Total++
			if s.Done {
				sum.Done++
			}
			if s.Owner != name {
				parts[s.Owner] = true
				included[s.Owner] = true
			}
		}
		sum.Parts = len(parts)
		out = append(out, sum)
	}
	for i := range out {
		out[i].Part = included[out[i].Name]
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Featured && !out[j].Featured })
	return out
}

// Stage finds one stage of a path by its key.
func (c *Course) Stage(top, key string) (Stage, error) {
	stages, err := c.expand(top, top, nil)
	if err != nil {
		return Stage{}, err
	}
	for _, s := range stages {
		if s.Key == key {
			return s, nil
		}
	}
	return Stage{}, fmt.Errorf("course: path %q has no stage %q", top, key)
}

func (c *Course) expand(top, name string, stack []string) ([]Stage, error) {
	for _, seen := range stack {
		if seen == name {
			return nil, fmt.Errorf("course: include cycle: %s -> %s", strings.Join(stack, " -> "), name)
		}
	}
	if !fs.ValidPath(name) || strings.Contains(name, "/") {
		return nil, fmt.Errorf("course: bad path name %q", name)
	}
	data, err := fs.ReadFile(c.fsys, path.Join(c.m.Paths, name, "path.tsv"))
	if err != nil {
		return nil, fmt.Errorf("course: unknown path %q", name)
	}
	stack = append(stack, name)

	var out []Stage
	sc := bufio.NewScanner(bytes.NewReader(data))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := sc.Text()
		if strings.TrimSpace(line) == "" || strings.HasPrefix(line, "#") {
			continue
		}
		cols := strings.Split(line, "\t")
		if inc, ok := strings.CutPrefix(cols[0], "@"); ok {
			sub, err := c.expand(top, strings.TrimSpace(inc), stack)
			if err != nil {
				return nil, err
			}
			out = append(out, sub...)
			continue
		}
		for len(cols) < 5 {
			cols = append(cols, "")
		}
		s := Stage{Owner: name, Number: cols[0], Key: cols[0], Title: cols[1], DoneWhen: cols[3]}
		if name != top {
			s.Key = name + ":" + cols[0]
		}
		for _, f := range strings.Split(cols[2], ",") {
			if f = strings.TrimSpace(f); f != "" {
				s.Files = append(s.Files, f)
			}
		}
		if check := strings.TrimSpace(cols[4]); check != "-" {
			s.Check = check
		}
		out = append(out, s)
	}
	return out, sc.Err()
}
