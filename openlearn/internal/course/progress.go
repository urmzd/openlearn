package course

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// DoneFiles keeps progress the way `ss learn` does: one <owner>.done file per
// owning path, one finished stage number per line. Sharing the format means the
// terminal and the app always agree about what is finished.
type DoneFiles struct {
	// DirFor returns the directory holding an owner's done file.
	DirFor func(owner string) string
}

// CheckoutProgress reads progress from a course checkout, where the practice
// CLI writes it. The layout mirrors learn_dir in practice/bin/ss: paths named
// course* belong to the learner's course repo once one exists.
func CheckoutProgress(root string) DoneFiles {
	scratch := envOr("SS_SCRATCH", filepath.Join(root, ".scratchpad"))
	home := envOr("SS_COURSE_HOME", filepath.Join(root, ".scratchpad", "course"))
	return DoneFiles{DirFor: func(owner string) string {
		if strings.HasPrefix(owner, "course") {
			if _, err := os.Stat(filepath.Join(home, "system.toml")); err == nil {
				return filepath.Join(home, ".ss", "learn")
			}
		}
		return filepath.Join(scratch, "learn")
	}}
}

// StoreProgress keeps progress in dir, for a pack with no checkout to hold it.
func StoreProgress(dir string) DoneFiles {
	return DoneFiles{DirFor: func(string) string { return dir }}
}

func (d DoneFiles) file(owner string) string {
	return filepath.Join(d.DirFor(owner), filepath.Base(owner)+".done")
}

// Done returns the finished stage numbers of one owning path.
func (d DoneFiles) Done(owner string) map[string]bool {
	out := map[string]bool{}
	f, err := os.Open(d.file(owner))
	if err != nil {
		return out
	}
	defer func() { _ = f.Close() }()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if s := strings.TrimSpace(sc.Text()); s != "" {
			out[s] = true
		}
	}
	return out
}

// Mark records a stage as finished or not.
func (d DoneFiles) Mark(owner, stage string, done bool) error {
	current := d.Done(owner)
	if current[stage] == done {
		return nil
	}
	file := d.file(owner)
	if err := os.MkdirAll(filepath.Dir(file), 0o700); err != nil {
		return fmt.Errorf("course: progress dir: %w", err)
	}
	// Rewritten in file order so an undo leaves the other marks untouched.
	var lines []string
	if data, err := os.ReadFile(file); err == nil { //nolint:gosec // G304: path is built from the progress dir and a path name
		for _, l := range strings.Split(string(data), "\n") {
			if l = strings.TrimSpace(l); l != "" && l != stage {
				lines = append(lines, l)
			}
		}
	}
	if done {
		lines = append(lines, stage)
	}
	body := strings.Join(lines, "\n")
	if body != "" {
		body += "\n"
	}
	if err := os.WriteFile(file, []byte(body), 0o600); err != nil { //nolint:gosec // G703: same path as the read above
		return fmt.Errorf("course: write progress: %w", err)
	}
	return nil
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
