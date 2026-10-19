// Package effects is the external side-effect sink of the durable tests
// (dur.*): the worker's `--test-activities` registers
// `tl.test.Append`, which POSTs here with its idempotency key, and the test
// asserts every key was applied exactly once, however often the worker was
// killed and the activity retried.
//
//	POST /effects   {"key": "...", "value": "..."}   200 {"applied": true|false}
//	GET  /effects   every record in arrival order
//
// The sink dedupes by key (as a correct external system would) and counts
// every delivery, so a test sees both the state and the retries.
package effects

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"sort"
	"sync"
	"testing"
)

type Record struct {
	Key        string `json:"key"`
	Value      string `json:"value"`
	Deliveries int    `json:"deliveries"`
}

type Sink struct {
	mu    sync.Mutex
	order []string
	recs  map[string]*Record
	srv   *http.Server
	ln    net.Listener
}

// Start serves on 127.0.0.1 at a free port.
func Start() (*Sink, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, err
	}
	s := &Sink{recs: map[string]*Record{}, ln: ln}
	mux := http.NewServeMux()
	mux.HandleFunc("/effects", s.handle)
	s.srv = &http.Server{Handler: mux}
	go s.srv.Serve(ln)
	return s, nil
}

func (s *Sink) URL() string { return "http://" + s.ln.Addr().String() + "/effects" }

func (s *Sink) Close() error { return s.srv.Close() }

func (s *Sink) handle(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	switch r.Method {
	case http.MethodPost:
		var in struct{ Key, Value string }
		if err := json.NewDecoder(r.Body).Decode(&in); err != nil || in.Key == "" {
			http.Error(w, `{"error":"want {\"key\": ..., \"value\": ...}"}`, http.StatusBadRequest)
			return
		}
		rec, seen := s.recs[in.Key]
		if !seen {
			rec = &Record{Key: in.Key, Value: in.Value}
			s.recs[in.Key] = rec
			s.order = append(s.order, in.Key)
		}
		rec.Deliveries++
		json.NewEncoder(w).Encode(map[string]bool{"applied": !seen})
	case http.MethodGet:
		json.NewEncoder(w).Encode(s.recordsLocked())
	default:
		w.WriteHeader(http.StatusMethodNotAllowed)
	}
}

func (s *Sink) recordsLocked() []Record {
	out := make([]Record, 0, len(s.order))
	for _, k := range s.order {
		out = append(out, *s.recs[k])
	}
	return out
}

// Records in arrival order of their first delivery.
func (s *Sink) Records() []Record {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.recordsLocked()
}

// Check returns a description of every key that is missing, unexpected, or
// (with strict) delivered more than once; "" when the set is exactly keys.
func (s *Sink) Check(keys []string, strict bool) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	want := map[string]bool{}
	var problems []string
	for _, k := range keys {
		want[k] = true
		rec, ok := s.recs[k]
		if !ok {
			problems = append(problems, "missing "+k)
		} else if strict && rec.Deliveries > 1 {
			problems = append(problems, fmt.Sprintf("%s delivered %d times", k, rec.Deliveries))
		}
	}
	for k := range s.recs {
		if !want[k] {
			problems = append(problems, "unexpected "+k)
		}
	}
	sort.Strings(problems)
	if len(problems) == 0 {
		return ""
	}
	return fmt.Sprintf("%d problem(s): %v", len(problems), problems)
}

// AssertExactlyOnce: every key applied, none unexpected. Retries are fine
// (the sink dedupes, as idempotency keys promise); use AssertDeliveredOnce
// when the contract forbids redelivery too.
func (s *Sink) AssertExactlyOnce(t testing.TB, keys []string) {
	t.Helper()
	if msg := s.Check(keys, false); msg != "" {
		t.Fatalf("effects not exactly once: %s", msg)
	}
}

// AssertDeliveredOnce: every key delivered exactly one time.
func (s *Sink) AssertDeliveredOnce(t testing.TB, keys []string) {
	t.Helper()
	if msg := s.Check(keys, true); msg != "" {
		t.Fatalf("effects not delivered once: %s", msg)
	}
}
