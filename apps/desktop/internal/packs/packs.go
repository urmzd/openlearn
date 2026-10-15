// Package packs finds course packs and opens the one the user picked. A pack
// is any directory holding a course manifest. It reaches the app one of four
// ways: compiled in, found in the checkout the app is running from, linked from
// a directory the user pointed at, or installed by unpacking a zip archive.
package packs

import (
	"archive/zip"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"

	"github.com/urmzd/openlearn/app/internal/course"
)

// builtinPrefix marks a ref that names a compiled-in pack rather than a
// directory.
const builtinPrefix = "builtin:"

// Pack is one course the app can open.
type Pack struct {
	// Ref is what the app stores to remember the choice: "builtin:<id>" or
	// an absolute directory.
	Ref     string `json:"ref"`
	ID      string `json:"id"`
	Title   string `json:"title"`
	Tagline string `json:"tagline"`
	// Source is "builtin", "checkout", "linked", or "installed".
	Source string `json:"source"`
	// Practice reports whether the pack's exercises can run from here.
	Practice bool `json:"practice"`
}

// Registry knows where packs live.
type Registry struct {
	// Builtin holds one directory per compiled-in pack id.
	Builtin fs.FS
	// InstallDir is where imported archives are unpacked.
	InstallDir string
	// Checkout is a directory to search upward from for a pack the app is
	// being developed or run inside. Empty disables the search.
	Checkout string
	// Preferred is the id of the pack to list first.
	Preferred string
}

// List returns every pack the app can open. A pack reachable on disk hides the
// compiled-in copy with the same id, because the copy on disk is newer and is
// the only one its exercises can run from.
func (r Registry) List(linked []string) []Pack {
	var out []Pack
	seenDir := map[string]bool{}
	onDisk := map[string]bool{}
	add := func(dir, source string) {
		abs, err := filepath.Abs(dir)
		if err != nil || seenDir[abs] {
			return
		}
		c, err := course.Open(os.DirFS(abs), abs)
		if err != nil {
			return
		}
		seenDir[abs] = true
		onDisk[c.Info().ID] = true
		out = append(out, describe(c, abs, source))
	}

	if root := r.checkoutRoot(); root != "" {
		add(root, "checkout")
		if entries, err := os.ReadDir(filepath.Join(root, "packs")); err == nil {
			for _, e := range entries {
				if e.IsDir() {
					add(filepath.Join(root, "packs", e.Name()), "checkout")
				}
			}
		}
	}
	for _, dir := range linked {
		add(dir, "linked")
	}
	if entries, err := os.ReadDir(r.InstallDir); err == nil {
		for _, e := range entries {
			if e.IsDir() {
				add(filepath.Join(r.InstallDir, e.Name()), "installed")
			}
		}
	}

	var builtin []Pack
	if r.Builtin != nil {
		if entries, err := fs.ReadDir(r.Builtin, "."); err == nil {
			for _, e := range entries {
				if !e.IsDir() || onDisk[e.Name()] {
					continue
				}
				sub, err := fs.Sub(r.Builtin, e.Name())
				if err != nil {
					continue
				}
				if c, err := course.Open(sub, ""); err == nil {
					builtin = append(builtin, describe(c, builtinPrefix+e.Name(), "builtin"))
				}
			}
		}
	}
	out = append(out, builtin...)
	// The preferred pack leads so it is what a first launch opens; the rest
	// keep discovery order.
	sort.SliceStable(out, func(i, j int) bool { return out[i].ID == r.Preferred && out[j].ID != r.Preferred })
	return out
}

// Open loads the pack a ref names.
func (r Registry) Open(ref string) (*course.Course, error) {
	if id, ok := strings.CutPrefix(ref, builtinPrefix); ok {
		if r.Builtin == nil || !fs.ValidPath(id) || strings.Contains(id, "/") {
			return nil, fmt.Errorf("packs: no built-in pack %q", id)
		}
		sub, err := fs.Sub(r.Builtin, id)
		if err != nil {
			return nil, err
		}
		return course.Open(sub, "")
	}
	if !filepath.IsAbs(ref) {
		return nil, fmt.Errorf("packs: %q is not a pack", ref)
	}
	return course.Open(os.DirFS(ref), ref)
}

// Default picks the pack to open when the user has not chosen one, or when
// their choice is gone: the first one List returns.
func (r Registry) Default(linked []string) (Pack, bool) {
	all := r.List(linked)
	if len(all) == 0 {
		return Pack{}, false
	}
	return all[0], true
}

func describe(c *course.Course, ref, source string) Pack {
	info := c.Info()
	return Pack{Ref: ref, ID: info.ID, Title: info.Title, Tagline: info.Tagline, Source: source, Practice: info.Practice}
}

// checkoutRoot walks up from Checkout to the nearest directory with a manifest.
func (r Registry) checkoutRoot() string {
	if r.Checkout == "" {
		return ""
	}
	dir, err := filepath.Abs(r.Checkout)
	if err != nil {
		return ""
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, course.ManifestName)); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
}

// Limits on an imported archive. A course is markdown and small images; an
// archive past these is a mistake or an attack, and either way is refused
// before it fills the disk.
const (
	maxArchiveFiles = 20000
	maxArchiveBytes = 512 << 20
)

// ErrNotAPack means the import source has no manifest.
var ErrNotAPack = errors.New("packs: no " + course.ManifestName + " found")

// Install unpacks a zip archive into InstallDir and returns the installed
// pack's directory. The manifest may sit at the archive's root or inside its
// single top-level directory, which is how GitHub and most tools zip a folder.
func (r Registry) Install(archive string) (string, error) {
	zr, err := zip.OpenReader(archive)
	if err != nil {
		return "", fmt.Errorf("packs: open archive: %w", err)
	}
	defer func() { _ = zr.Close() }()

	prefix, err := manifestPrefix(zr.File)
	if err != nil {
		return "", err
	}
	if len(zr.File) > maxArchiveFiles {
		return "", fmt.Errorf("packs: archive has %d entries, limit is %d", len(zr.File), maxArchiveFiles)
	}
	if err := os.MkdirAll(r.InstallDir, 0o700); err != nil {
		return "", fmt.Errorf("packs: install dir: %w", err)
	}
	tmp, err := os.MkdirTemp(r.InstallDir, ".import-")
	if err != nil {
		return "", fmt.Errorf("packs: staging dir: %w", err)
	}
	defer func() { _ = os.RemoveAll(tmp) }()

	var total int64
	for _, f := range zr.File {
		name, ok := strings.CutPrefix(f.Name, prefix)
		if !ok || name == "" || strings.HasSuffix(name, "/") {
			continue
		}
		// fs.ValidPath rejects "..", absolute paths, and empty elements, which
		// is every way an entry could land outside the staging directory.
		if !fs.ValidPath(name) || f.Mode()&fs.ModeSymlink != 0 {
			return "", fmt.Errorf("packs: unsafe archive entry %q", f.Name)
		}
		n, err := extract(f, filepath.Join(tmp, filepath.FromSlash(name)), maxArchiveBytes-total)
		if err != nil {
			return "", err
		}
		total += n
	}

	c, err := course.Open(os.DirFS(tmp), tmp)
	if err != nil {
		return "", err
	}
	id := c.Info().ID
	if !fs.ValidPath(id) || strings.ContainsAny(id, `/\`) || strings.HasPrefix(id, ".") {
		return "", fmt.Errorf("packs: manifest id %q cannot be a directory name", id)
	}
	dest := filepath.Join(r.InstallDir, id)
	// Importing the same pack again replaces it: that is how a pack is updated.
	if err := os.RemoveAll(dest); err != nil {
		return "", fmt.Errorf("packs: replace %s: %w", id, err)
	}
	if err := os.Rename(tmp, dest); err != nil {
		return "", fmt.Errorf("packs: install %s: %w", id, err)
	}
	return dest, nil
}

func manifestPrefix(files []*zip.File) (string, error) {
	best, found := "", false
	for _, f := range files {
		if path.Base(f.Name) != course.ManifestName {
			continue
		}
		prefix := strings.TrimSuffix(f.Name, course.ManifestName)
		if strings.Count(prefix, "/") > 1 {
			continue
		}
		if !found || len(prefix) < len(best) {
			best, found = prefix, true
		}
	}
	if !found {
		return "", ErrNotAPack
	}
	return best, nil
}

func extract(f *zip.File, dest string, budget int64) (int64, error) {
	if err := os.MkdirAll(filepath.Dir(dest), 0o700); err != nil {
		return 0, fmt.Errorf("packs: extract: %w", err)
	}
	src, err := f.Open()
	if err != nil {
		return 0, fmt.Errorf("packs: extract %s: %w", f.Name, err)
	}
	defer func() { _ = src.Close() }()

	// Executable bits survive so a pack's practice CLI still runs; nothing else
	// about the archived mode is trusted.
	mode := fs.FileMode(0o600)
	if f.Mode()&0o100 != 0 {
		mode = 0o700
	}
	out, err := os.OpenFile(dest, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode) //nolint:gosec // G304: dest is under the staging dir, checked by the caller
	if err != nil {
		return 0, fmt.Errorf("packs: extract %s: %w", f.Name, err)
	}
	// The header's size is the archive's own claim, so the copy is what is
	// bounded: one byte past the budget is proof the archive is over.
	n, err := io.Copy(out, io.LimitReader(src, budget+1))
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return n, fmt.Errorf("packs: extract %s: %w", f.Name, err)
	}
	if n > budget {
		return n, fmt.Errorf("packs: archive unpacks to more than %d MB", maxArchiveBytes>>20)
	}
	return n, nil
}
