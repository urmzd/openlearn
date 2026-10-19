package effects_test

import (
	"bytes"
	"net/http"
	"strings"
	"testing"

	"openlearn.urmzd.com/tl/testkit/effects"
)

func post(t *testing.T, url, key string) {
	t.Helper()
	r, err := http.Post(url, "application/json", bytes.NewBufferString(`{"key":"`+key+`","value":"v"}`))
	if err != nil {
		t.Fatal(err)
	}
	r.Body.Close()
}

func TestExactlyOnceWithRetries(t *testing.T) {
	s, err := effects.Start()
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	for _, k := range []string{"a", "b", "a"} {
		post(t, s.URL(), k)
	}
	s.AssertExactlyOnce(t, []string{"a", "b"})
	if msg := s.Check([]string{"a", "b"}, true); !strings.Contains(msg, "a delivered 2 times") {
		t.Fatalf("strict check: %q", msg)
	}
	if msg := s.Check([]string{"a", "b", "c"}, false); !strings.Contains(msg, "missing c") {
		t.Fatalf("missing: %q", msg)
	}
	if msg := s.Check([]string{"a"}, false); !strings.Contains(msg, "unexpected b") {
		t.Fatalf("unexpected: %q", msg)
	}
	if recs := s.Records(); len(recs) != 2 || recs[0].Key != "a" || recs[0].Deliveries != 2 {
		t.Fatalf("records %+v", recs)
	}
}
