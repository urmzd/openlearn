package course_test

import (
	"os"
	"path/filepath"
	"testing"
	"testing/fstest"

	"github.com/urmzd/openlearn/app/internal/course"
)

const manifest = `{
  "id": "demo", "title": "Demo", "paths": "paths", "featured": "whole",
  "start": [{"file": "README.md", "title": "Overview"}, {"file": "MISSING.md", "title": "Gone"}],
  "tracks": [{"dir": "math", "title": "Mathematics"}, {"dir": "absent", "title": "Absent"}]
}`

func demo(t *testing.T) *course.Course {
	t.Helper()
	fsys := fstest.MapFS{
		"openlearn.json":              {Data: []byte(manifest)},
		"README.md":                   {Data: []byte("# Demo course\n")},
		"math/README.md":              {Data: []byte("# Math track\n")},
		"math/01-limits/README.md":    {Data: []byte("```\n# not a title\n```\n# [Limits](x.md) and `epsilon`\nA limit is a promise about nearby values.\n")},
		"math/01-limits/02-proofs.md": {Data: []byte("# Proofs\nlimit limit limit\n")},
		"math/01-limits/01-intro.md":  {Data: []byte("no heading here\n")},
		"math/02-series/README.md":    {Data: []byte("# Series\n")},
		"math/node_modules/x/a.md":    {Data: []byte("# Vendored\n")},
		"paths/part/README.md":        {Data: []byte("# The part\n")},
		"paths/part/path.tsv":         {Data: []byte("# comment\n1\tLimits\tmath/01-limits/README.md, math/01-limits/02-proofs.md\tYou can prove one.\t-\n2\tSeries\tmath/02-series/README.md\tYou can sum one.\tmodule:M1\n")},
		"paths/whole/README.md":       {Data: []byte("# The whole\n")},
		"paths/whole/path.tsv":        {Data: []byte("0\tBrief\tREADME.md\tYou know the plan.\n@part\n")},
		"paths/loop/README.md":        {Data: []byte("# Loop\n")},
		"paths/loop/path.tsv":         {Data: []byte("@loop\n")},
	}
	c, err := course.Open(fsys, "")
	if err != nil {
		t.Fatal(err)
	}
	return c
}

type doneMap map[string]map[string]bool

func (d doneMap) Done(owner string) map[string]bool { return d[owner] }

func TestOpenRequiresAManifest(t *testing.T) {
	if _, err := course.Open(fstest.MapFS{}, ""); err != course.ErrNoManifest {
		t.Fatalf("err = %v, want ErrNoManifest", err)
	}
}

func TestOutlineOrdersTopicsAndSkipsWhatIsMissing(t *testing.T) {
	o := demo(t).Outline()
	if len(o.Start) != 1 || o.Start[0].Title != "Overview" {
		t.Errorf("start = %+v, want only the entry whose file exists", o.Start)
	}
	if len(o.Tracks) != 1 {
		t.Fatalf("tracks = %d, want 1: a track with no directory is skipped", len(o.Tracks))
	}
	math := o.Tracks[0]
	if len(math.Docs) != 1 || math.Docs[0].Path != "math/README.md" {
		t.Errorf("track docs = %+v", math.Docs)
	}
	if len(math.Topics) != 2 {
		t.Fatalf("topics = %+v, want limits and series (node_modules skipped)", math.Topics)
	}
	limits := math.Topics[0]
	if limits.Title != "Limits and epsilon" {
		t.Errorf("topic title = %q, want the README heading with markup stripped", limits.Title)
	}
	var got []string
	for _, d := range limits.Docs {
		got = append(got, d.Path)
	}
	want := []string{"math/01-limits/README.md", "math/01-limits/01-intro.md", "math/01-limits/02-proofs.md"}
	if len(got) != len(want) {
		t.Fatalf("docs = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("docs[%d] = %s, want %s", i, got[i], want[i])
		}
	}
	if limits.Docs[1].Title != "01-intro" {
		t.Errorf("untitled doc = %q, want its file name", limits.Docs[1].Title)
	}
}

func TestDocRejectsPathsOutsideTheCourse(t *testing.T) {
	c := demo(t)
	for _, p := range []string{"../secret.md", "/etc/passwd", "math/README.txt", "openlearn.json"} {
		if _, err := c.Doc(p); err == nil {
			t.Errorf("Doc(%q) succeeded, want an error", p)
		}
	}
	d, err := c.Doc("math/01-limits/02-proofs.md")
	if err != nil || d.Title != "Proofs" {
		t.Errorf("Doc = %+v, %v", d, err)
	}
}

func TestPathExpandsIncludesAndSharesProgressWithParts(t *testing.T) {
	c := demo(t)
	p, err := c.Path("whole", doneMap{"part": {"2": true}})
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Stages) != 3 {
		t.Fatalf("stages = %d, want 3", len(p.Stages))
	}
	if p.Stages[0].Key != "0" || p.Stages[1].Key != "part:1" || p.Stages[2].Key != "part:2" {
		t.Errorf("keys = %s %s %s", p.Stages[0].Key, p.Stages[1].Key, p.Stages[2].Key)
	}
	if len(p.Stages[1].Files) != 2 || p.Stages[1].Files[1] != "math/01-limits/02-proofs.md" {
		t.Errorf("files = %v, want both, trimmed", p.Stages[1].Files)
	}
	if p.Stages[1].Check != "" || p.Stages[2].Check != "module:M1" {
		t.Errorf("checks = %q %q, want a dash to mean no check", p.Stages[1].Check, p.Stages[2].Check)
	}
	if p.Stages[1].Done || !p.Stages[2].Done {
		t.Error("progress made in the part must show in the path that includes it")
	}
}

func TestPathsLeadsWithFeaturedAndReportsBrokenOnes(t *testing.T) {
	sums := demo(t).Paths(doneMap{"part": {"1": true}})
	if len(sums) != 3 || sums[0].Name != "whole" || !sums[0].Featured {
		t.Fatalf("paths = %+v, want the featured path first", sums)
	}
	if sums[0].Total != 3 || sums[0].Done != 1 || sums[0].Parts != 1 || sums[0].Part {
		t.Errorf("whole = %+v", sums[0])
	}
	for _, s := range sums {
		if s.Name == "part" && !s.Part {
			t.Error("a path another path includes must be marked as a part")
		}
		if s.Name == "loop" && s.Broken == "" {
			t.Error("an include cycle must be reported, not followed")
		}
	}
}

func TestSearchRanksTitleMatchesFirstAndNeedsEveryTerm(t *testing.T) {
	c := demo(t)
	hits := c.Search("limit", 10)
	if len(hits) < 2 || hits[0].Path != "math/01-limits/README.md" {
		t.Fatalf("hits = %+v, want the doc titled Limits first", hits)
	}
	if hits[0].Snippet != "# [Limits](x.md) and `epsilon`" {
		t.Errorf("snippet = %q", hits[0].Snippet)
	}
	if got := c.Search("limit promise", 10); len(got) != 1 {
		t.Errorf("two terms matched %d docs, want the one holding both", len(got))
	}
	if got := c.Search("   ", 10); len(got) != 0 {
		t.Errorf("blank query matched %d docs", len(got))
	}
}

func TestDoneFilesRoundTripInTheFormatSSReads(t *testing.T) {
	dir := t.TempDir()
	p := course.StoreProgress(dir)
	for _, s := range []string{"1", "3", "1"} {
		if err := p.Mark("part", s, true); err != nil {
			t.Fatal(err)
		}
	}
	if err := p.Mark("part", "1", false); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(dir, "part.done"))
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "3\n" {
		t.Errorf("done file = %q, want one stage number per line", data)
	}
	if done := p.Done("part"); !done["3"] || done["1"] {
		t.Errorf("done = %v", done)
	}
}

// The repository this app lives in is itself a pack, read by both this package
// and `ss learn`. Every path must expand here the way it does there, and every
// stage must name documents that exist.
func TestTheBuiltInPacksAreWellFormed(t *testing.T) {
	for _, root := range []string{"../../..", "../../../packs/languide"} {
		c, err := course.Open(os.DirFS(root), root)
		if err != nil {
			t.Fatalf("%s: %v", root, err)
		}
		if len(c.Outline().Tracks) == 0 {
			t.Errorf("%s: no tracks", root)
		}
		for _, name := range c.PathNames() {
			p, err := c.Path(name, nil)
			if err != nil {
				t.Errorf("%s: path %s: %v", root, name, err)
				continue
			}
			for _, s := range p.Stages {
				for _, f := range s.Files {
					if _, err := c.Doc(f); err != nil {
						t.Errorf("%s: path %s stage %s: %v", root, name, s.Key, err)
					}
				}
			}
		}
	}
}
