// Command bundle copies course packs into internal/builtin/content so they can
// be embedded in the binary.
//
//	go run ./cmd/bundle -out internal/builtin/content <pack dir>...
//
// Only what the app renders is copied: the manifest, the markdown and images
// under the pack's tracks, reference trees, and paths, and any document a path
// stage names. Code, build output, and exercises stay behind, because they only
// work from a checkout anyway.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"

	"github.com/urmzd/openlearn/app/internal/course"
)

func main() {
	out := flag.String("out", "internal/builtin/content", "directory to write packs into")
	flag.Parse()
	if flag.NArg() == 0 {
		fail("usage: bundle -out DIR <pack dir>...")
	}
	if err := clean(*out); err != nil {
		fail(err.Error())
	}
	for _, dir := range flag.Args() {
		n, id, err := bundle(dir, *out)
		if err != nil {
			fail(fmt.Sprintf("%s: %v", dir, err))
		}
		fmt.Printf("bundled %s: %d files\n", id, n)
	}
}

func fail(msg string) {
	fmt.Fprintln(os.Stderr, "bundle:", msg)
	os.Exit(1)
}

// clean empties out but keeps the tracked .gitkeep.
func clean(out string) error {
	entries, err := os.ReadDir(out)
	if err != nil {
		return os.MkdirAll(out, 0o750)
	}
	for _, e := range entries {
		if e.Name() == ".gitkeep" {
			continue
		}
		if err := os.RemoveAll(filepath.Join(out, e.Name())); err != nil {
			return err
		}
	}
	return nil
}

func bundle(dir, out string) (int, string, error) {
	c, err := course.Open(os.DirFS(dir), dir)
	if err != nil {
		return 0, "", err
	}
	m := c.Manifest()

	files, err := candidates(dir)
	if err != nil {
		return 0, "", err
	}

	var trees []string
	for _, t := range m.Tracks {
		trees = append(trees, t.Dir)
	}
	for _, t := range m.Reference {
		trees = append(trees, t.Dir)
	}
	trees = append(trees, m.Paths)

	want := map[string]bool{course.ManifestName: true}
	for _, e := range m.Start {
		want[e.File] = true
	}
	for _, name := range c.PathNames() {
		p, err := c.Path(name, nil)
		if err != nil {
			return 0, "", err
		}
		for _, s := range p.Stages {
			for _, f := range s.Files {
				want[f] = true
			}
		}
	}

	n := 0
	for _, f := range files {
		if !want[f] && (!rendered(f) || !under(f, trees)) {
			continue
		}
		if err := copyFile(filepath.Join(dir, filepath.FromSlash(f)), filepath.Join(out, m.ID, filepath.FromSlash(f))); err != nil {
			return 0, "", err
		}
		n++
	}
	return n, m.ID, nil
}

// candidates lists the pack's files as slash paths. Inside a git checkout that
// is what git tracks, which keeps scratch work and dependencies out for free.
func candidates(dir string) ([]string, error) {
	cmd := exec.Command("git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--deduplicate") //nolint:gosec // G204: fixed arguments
	cmd.Dir = dir
	if data, err := cmd.Output(); err == nil && len(data) > 0 {
		var files []string
		for _, f := range bytes.Split(data, []byte{0}) {
			if len(f) > 0 {
				if _, err := os.Stat(filepath.Join(dir, string(f))); err != nil {
					continue
				}
				files = append(files, string(f))
			}
		}
		return files, nil
	}
	var files []string
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if p != dir && (strings.HasPrefix(d.Name(), ".") || d.Name() == "node_modules") {
				return filepath.SkipDir
			}
			return nil
		}
		rel, err := filepath.Rel(dir, p)
		if err != nil {
			return err
		}
		files = append(files, filepath.ToSlash(rel))
		return nil
	})
	return files, err
}

func rendered(f string) bool {
	switch strings.ToLower(path.Ext(f)) {
	case ".md", ".tsv", ".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp":
		return true
	}
	return false
}

func under(f string, trees []string) bool {
	for _, t := range trees {
		if t != "" && strings.HasPrefix(f, t+"/") {
			return true
		}
	}
	return false
}

func copyFile(src, dst string) error {
	data, err := os.ReadFile(src) //nolint:gosec // G304: a build tool reading the packs it was pointed at
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o750); err != nil {
		return err
	}
	return os.WriteFile(dst, data, 0o600) //nolint:gosec // G306/G703: build output inside the repository
}
