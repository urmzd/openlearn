package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"tinyllm/config"
	"tinyllm/gateway/server"
)

func TestRuntimeAPIVersionWindow(t *testing.T) {
	path := filepath.Join(t.TempDir(), "runtime.toml")
	text := "[ext.api]\ndefault_version = '2'\ndeprecated_at = 90\nsunset = 100\n"
	if err := os.WriteFile(path, []byte(text), 0600); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.Load(path, nil)
	if err != nil {
		t.Fatal(err)
	}
	api, err := apiVersionConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	api.Now = func() time.Time { return time.Unix(100, 0) }
	dispatched := 0
	h := server.VersionedAPI(api, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		dispatched++
		if r.Header.Get(server.APIVersionHeader) != "2" {
			t.Error("selected version did not reach downstream metering")
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	for _, tc := range []struct {
		version string
		status  int
	}{{"", 204}, {"2", 204}, {"1", 410}, {"3", 400}} {
		r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
		if tc.version != "" {
			r.Header.Set(server.APIVersionHeader, tc.version)
		}
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		if w.Code != tc.status {
			t.Fatalf("version %q: status %d, want %d: %s", tc.version, w.Code, tc.status, w.Body.String())
		}
	}
	if dispatched != 2 {
		t.Fatalf("expired and unknown versions reached downstream: %d calls", dispatched)
	}
}

func TestRuntimeAPIVersionRejectsInvalidConfiguration(t *testing.T) {
	for _, api := range []any{
		"2",
		map[string]any{"default_version": int64(2)},
		map[string]any{"default_version": "3"},
		map[string]any{"default_vesion": "2"},
		map[string]any{"sunset": int64(100)},
		map[string]any{"deprecated_at": int64(90)},
		map[string]any{"deprecated_at": int64(100), "sunset": int64(100)},
		map[string]any{"deprecated_at": int64(100), "sunset": int64(90)},
		map[string]any{"deprecated_at": int64(0), "sunset": int64(100)},
	} {
		if _, err := apiVersionConfig(config.Config{Ext: map[string]any{"api": api}}); err == nil {
			t.Errorf("accepted invalid migration settings: %#v", api)
		}
	}
	api, err := apiVersionConfig(config.Config{})
	if err != nil || api.Default != "1" || !api.Sunset.IsZero() {
		t.Fatalf("existing deployments must default to v1 without a sunset: %#v, %v", api, err)
	}
}
