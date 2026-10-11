package dur_90

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"openlearn.urmzd.com/tl/testkit/clock"
	"openlearn.urmzd.com/tl/testkit/promscrape"
	"tinyllm/ds/demo"
)

func TestTestkitReachesCourseTests(t *testing.T) {
	// WHY: course Go tests import the testkit module through the overlay's go.work.
	// KIND: unit
	f := clock.NewFake(time.Unix(0, 0))
	f.Advance(time.Second)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("demo_sum 6\n"))
	}))
	defer srv.Close()
	m, err := promscrape.Scrape(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := m.Get("demo_sum", nil); v != demo.Sum([]float64{1, 2, 3}) || f.Now().Unix() != 1 {
		t.Fatalf("got %v at %v", v, f.Now())
	}
}
