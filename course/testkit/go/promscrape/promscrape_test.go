package promscrape_test

import (
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"openlearn.urmzd.com/tl/testkit/promscrape"
)

const body = `# HELP tl_engine_active_sequences live sequences
# TYPE tl_engine_active_sequences gauge
tl_engine_active_sequences 3
# TYPE http_server_request_duration_seconds histogram
http_server_request_duration_seconds_bucket{http_response_status_code="200",le="0.1"} 10
http_server_request_duration_seconds_bucket{http_response_status_code="200",le="0.5"} 30
http_server_request_duration_seconds_bucket{http_response_status_code="200",le="+Inf"} 40
http_server_request_duration_seconds_count{http_response_status_code="200"} 40
http_server_request_duration_seconds_sum{http_response_status_code="200"} 9.5
http_server_request_duration_seconds_bucket{http_response_status_code="503",le="0.1"} 1
http_server_request_duration_seconds_bucket{http_response_status_code="503",le="0.5"} 1
http_server_request_duration_seconds_bucket{http_response_status_code="503",le="+Inf"} 1
http_server_request_duration_seconds_count{http_response_status_code="503"} 1
tl_gateway_requests{route="a",code="200",tenant="x\"y"} 7
`

func TestScrapeAndQuantile(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(body)) }))
	defer srv.Close()
	m, err := promscrape.Scrape(srv.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	if v, ok := m.Get("tl_engine_active_sequences", nil); !ok || v != 3 {
		t.Fatalf("gauge %v %v", v, ok)
	}
	if m.Types["http_server_request_duration_seconds"] != "histogram" {
		t.Fatal("TYPE line")
	}
	if v, _ := m.Get("tl_gateway_requests", map[string]string{"tenant": `x"y`}); v != 7 {
		t.Fatal("escaped label")
	}
	if c := m.Sum("http_server_request_duration_seconds_count", nil); c != 41 {
		t.Fatalf("count %v", c)
	}
	ok := map[string]string{"http_response_status_code": "200"}
	// rank 0.5*40 = 20 falls in (0.1, 0.5]: 0.1 + 0.4 * (20-10)/(30-10) = 0.3
	if q := m.Quantile(0.5, "http_server_request_duration_seconds", ok); math.Abs(q-0.3) > 1e-12 {
		t.Fatalf("p50 %v", q)
	}
	if q := m.Quantile(0.99, "http_server_request_duration_seconds", ok); q != 0.5 {
		t.Fatalf("p99 in the +Inf bucket is the last finite bound: %v", q)
	}
	if _, err := promscrape.Parse(strings.NewReader("bad{x=1} 2\n")); err == nil {
		t.Fatal("unquoted label value parsed")
	}
}
