package store

import (
	"os"
	"path/filepath"
	"testing"
)

func TestAdoptMovesALegacyStoreOnce(t *testing.T) {
	home := t.TempDir()
	legacy, dir := filepath.Join(home, legacyDir), filepath.Join(home, ".openlearn")
	if err := os.MkdirAll(filepath.Join(legacy, "sessions"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(legacy, "sessions", "a.json"), []byte("{}"), 0o600); err != nil {
		t.Fatal(err)
	}

	if err := adopt(legacy, dir); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "sessions", "a.json")); err != nil {
		t.Errorf("sessions did not move: %v", err)
	}
	if _, err := os.Stat(legacy); !os.IsNotExist(err) {
		t.Error("legacy store still present after adoption")
	}
}

func TestAdoptNeverOverwritesAnExistingStore(t *testing.T) {
	home := t.TempDir()
	legacy, dir := filepath.Join(home, legacyDir), filepath.Join(home, ".openlearn")
	for _, d := range []string{legacy, dir} {
		if err := os.MkdirAll(d, 0o700); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "settings.json"), []byte("kept"), 0o600); err != nil {
		t.Fatal(err)
	}

	if err := adopt(legacy, dir); err != nil {
		t.Fatal(err)
	}
	if data, _ := os.ReadFile(filepath.Join(dir, "settings.json")); string(data) != "kept" {
		t.Errorf("existing store was replaced: %q", data)
	}
	if _, err := os.Stat(legacy); err != nil {
		t.Error("legacy store should be left alone when a new one exists")
	}
}
