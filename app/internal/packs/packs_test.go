package packs_test

import (
	"archive/zip"
	"os"
	"path/filepath"
	"testing"
	"testing/fstest"

	"github.com/urmzd/openlearn/app/internal/packs"
)

func manifest(id string) string {
	return `{"id":"` + id + `","title":"` + id + ` course","practice":"bin/ol"}`
}

func writePack(t *testing.T, dir, id string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "openlearn.json"), []byte(manifest(id)), 0o600); err != nil {
		t.Fatal(err)
	}
}

func writeZip(t *testing.T, entries map[string]string) string {
	t.Helper()
	name := filepath.Join(t.TempDir(), "pack.zip")
	f, err := os.Create(name)
	if err != nil {
		t.Fatal(err)
	}
	zw := zip.NewWriter(f)
	for path, body := range entries {
		w, err := zw.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := w.Write([]byte(body)); err != nil {
			t.Fatal(err)
		}
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	return name
}

func TestListPrefersThePackOnDiskOverItsBuiltInCopy(t *testing.T) {
	root := t.TempDir()
	writePack(t, root, "alpha")
	writePack(t, filepath.Join(root, "packs", "beta"), "beta")
	linked := filepath.Join(t.TempDir(), "gamma")
	writePack(t, linked, "gamma")

	r := packs.Registry{
		Builtin: fstest.MapFS{
			"alpha/openlearn.json": {Data: []byte(manifest("alpha"))},
			"delta/openlearn.json": {Data: []byte(manifest("delta"))},
		},
		InstallDir: filepath.Join(t.TempDir(), "none"),
		// A directory below the checkout root, as when the app runs from app/.
		Checkout: filepath.Join(root, "packs"),
	}
	got := r.List([]string{linked, linked, filepath.Join(root, "missing")})

	want := []struct{ id, source string }{
		{"alpha", "checkout"}, {"beta", "checkout"}, {"gamma", "linked"}, {"delta", "builtin"},
	}
	if len(got) != len(want) {
		t.Fatalf("packs = %+v, want %d", got, len(want))
	}
	for i, w := range want {
		if got[i].ID != w.id || got[i].Source != w.source {
			t.Errorf("packs[%d] = %s/%s, want %s/%s", i, got[i].ID, got[i].Source, w.id, w.source)
		}
	}
	if !got[0].Practice || got[3].Practice {
		t.Error("exercises run from a directory, never from a built-in pack")
	}
	for _, p := range got {
		if _, err := r.Open(p.Ref); err != nil {
			t.Errorf("Open(%q): %v", p.Ref, err)
		}
	}
	if _, err := r.Open("relative/dir"); err == nil {
		t.Error("a relative ref must not open")
	}
}

func TestInstallUnpacksAnArchiveWithATopLevelFolder(t *testing.T) {
	r := packs.Registry{InstallDir: filepath.Join(t.TempDir(), "packs")}
	archive := writeZip(t, map[string]string{
		"mycourse-main/openlearn.json":     manifest("mine"),
		"mycourse-main/track/README.md":    "# Track\n",
		"mycourse-main/sub/openlearn.json": manifest("nested"),
	})

	dir, err := r.Install(archive)
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Base(dir) != "mine" {
		t.Errorf("installed to %s, want a directory named for the manifest id", dir)
	}
	if _, err := os.Stat(filepath.Join(dir, "track", "README.md")); err != nil {
		t.Errorf("content missing: %v", err)
	}
	// Installing again is an update, not an error.
	if _, err := r.Install(archive); err != nil {
		t.Errorf("reinstall: %v", err)
	}
	if got := r.List(nil); len(got) != 1 || got[0].Source != "installed" {
		t.Errorf("packs = %+v", got)
	}
}

func TestInstallRefusesArchivesThatAreNotSafePacks(t *testing.T) {
	r := packs.Registry{InstallDir: filepath.Join(t.TempDir(), "packs")}

	if _, err := r.Install(writeZip(t, map[string]string{"README.md": "# no manifest\n"})); err != packs.ErrNotAPack {
		t.Errorf("err = %v, want ErrNotAPack", err)
	}
	escape := writeZip(t, map[string]string{
		"openlearn.json": manifest("evil"),
		"../outside.md":  "escaped",
	})
	if _, err := r.Install(escape); err == nil {
		t.Error("an entry that climbs out of the pack must be refused")
	}
	if _, err := os.Stat(filepath.Join(filepath.Dir(r.InstallDir), "outside.md")); err == nil {
		t.Error("the escaping entry was written")
	}
	if left, _ := os.ReadDir(r.InstallDir); len(left) != 0 {
		t.Errorf("a refused import left %d entries behind", len(left))
	}
	badID := writeZip(t, map[string]string{"openlearn.json": manifest("../up")})
	if _, err := r.Install(badID); err == nil {
		t.Error("a manifest id that is not a directory name must be refused")
	}
}
