// Package otlpsink is an in-test OTLP/HTTP trace receiver: POST /v1/traces in OTLP JSON or protobuf, gzip or not. Tests read
// back flat spans and assert span trees: the named spans, the services that
// emitted them, and parent-child edges, all inside one trace. Hang mode
// accepts requests and never answers, to prove an exporter never blocks the
// request path.
package otlpsink

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
)

// Span is one received span, flattened.
type Span struct {
	TraceID, SpanID, ParentSpanID string // lowercase hex
	Name, Service                 string
	Kind                          int
	Start, End                    uint64 // unix nanos
	Attrs                         map[string]any
	StatusCode                    int
}

type Sink struct {
	mu       sync.Mutex
	spans    []Span
	requests int
	hang     bool
	release  chan struct{}
	srv      *http.Server
	ln       net.Listener
	errs     []string
}

func Start() (*Sink, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, err
	}
	s := &Sink{ln: ln, release: make(chan struct{})}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/traces", s.handle)
	s.srv = &http.Server{Handler: mux}
	go s.srv.Serve(ln)
	return s, nil
}

// Endpoint is the OTLP/HTTP base URL (OTEL_EXPORTER_OTLP_ENDPOINT).
func (s *Sink) Endpoint() string { return "http://" + s.ln.Addr().String() }

func (s *Sink) Close() error {
	s.SetHang(false)
	return s.srv.Close()
}

// SetHang: when true, requests are read and then held unanswered.
func (s *Sink) SetHang(h bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.hang && !h {
		close(s.release)
		s.release = make(chan struct{})
	}
	s.hang = h
}

func (s *Sink) Requests() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.requests
}

// Errors are request bodies the sink could not decode.
func (s *Sink) Errors() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.errs...)
}

func (s *Sink) Spans() []Span {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]Span(nil), s.spans...)
}

// WaitForSpans polls until at least n spans arrived or the timeout passes.
func (s *Sink) WaitForSpans(n int, timeout time.Duration) []Span {
	deadline := time.Now().Add(timeout)
	for {
		sp := s.Spans()
		if len(sp) >= n || time.Now().After(deadline) {
			return sp
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func (s *Sink) handle(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	var body io.Reader = r.Body
	if strings.EqualFold(r.Header.Get("Content-Encoding"), "gzip") {
		gz, err := gzip.NewReader(r.Body)
		if err != nil {
			s.fail(w, "gzip: "+err.Error())
			return
		}
		body = gz
	}
	data, err := io.ReadAll(body)
	if err != nil {
		s.fail(w, err.Error())
		return
	}
	var spans []Span
	ctype := r.Header.Get("Content-Type")
	if strings.Contains(ctype, "protobuf") {
		spans, err = decodeProto(data)
	} else {
		spans, err = decodeJSON(data)
	}
	if err != nil {
		s.fail(w, err.Error())
		return
	}
	s.mu.Lock()
	s.requests++
	s.spans = append(s.spans, spans...)
	hang, release := s.hang, s.release
	s.mu.Unlock()
	if hang {
		select {
		case <-release:
		case <-r.Context().Done():
			return
		}
	}
	if strings.Contains(ctype, "protobuf") {
		w.Header().Set("Content-Type", "application/x-protobuf")
		w.WriteHeader(http.StatusOK)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte("{}"))
}

func (s *Sink) fail(w http.ResponseWriter, msg string) {
	s.mu.Lock()
	s.errs = append(s.errs, msg)
	s.mu.Unlock()
	http.Error(w, msg, http.StatusBadRequest)
}

// ---------------------------------------------------------------------------
// OTLP JSON (ids are hex strings; ints may be strings, as protojson writes them)

func decodeJSON(data []byte) ([]Span, error) {
	var doc struct {
		ResourceSpans []struct {
			Resource struct {
				Attributes []jsonKV `json:"attributes"`
			} `json:"resource"`
			ScopeSpans []struct {
				Spans []struct {
					TraceID           string      `json:"traceId"`
					SpanID            string      `json:"spanId"`
					ParentSpanID      string      `json:"parentSpanId"`
					Name              string      `json:"name"`
					Kind              int         `json:"kind"`
					StartTimeUnixNano json.Number `json:"startTimeUnixNano"`
					EndTimeUnixNano   json.Number `json:"endTimeUnixNano"`
					Attributes        []jsonKV    `json:"attributes"`
					Status            struct {
						Code int `json:"code"`
					} `json:"status"`
				} `json:"spans"`
			} `json:"scopeSpans"`
		} `json:"resourceSpans"`
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	if err := dec.Decode(&doc); err != nil {
		return nil, fmt.Errorf("OTLP JSON: %w", err)
	}
	var out []Span
	for _, rs := range doc.ResourceSpans {
		res := kvMap(rs.Resource.Attributes)
		svc, _ := res["service.name"].(string)
		for _, ss := range rs.ScopeSpans {
			for _, sp := range ss.Spans {
				st, _ := sp.StartTimeUnixNano.Int64()
				en, _ := sp.EndTimeUnixNano.Int64()
				out = append(out, Span{
					TraceID: strings.ToLower(sp.TraceID), SpanID: strings.ToLower(sp.SpanID),
					ParentSpanID: strings.ToLower(sp.ParentSpanID), Name: sp.Name, Service: svc, Kind: sp.Kind,
					Start: uint64(st), End: uint64(en), Attrs: kvMap(sp.Attributes), StatusCode: sp.Status.Code,
				})
			}
		}
	}
	return out, nil
}

type jsonKV struct {
	Key   string `json:"key"`
	Value struct {
		StringValue *string      `json:"stringValue"`
		BoolValue   *bool        `json:"boolValue"`
		IntValue    *json.Number `json:"intValue"`
		DoubleValue *float64     `json:"doubleValue"`
	} `json:"value"`
}

func kvMap(kvs []jsonKV) map[string]any {
	m := map[string]any{}
	for _, kv := range kvs {
		switch v := kv.Value; {
		case v.StringValue != nil:
			m[kv.Key] = *v.StringValue
		case v.BoolValue != nil:
			m[kv.Key] = *v.BoolValue
		case v.IntValue != nil:
			n, _ := v.IntValue.Int64()
			m[kv.Key] = n
		case v.DoubleValue != nil:
			m[kv.Key] = *v.DoubleValue
		}
	}
	return m
}

// ---------------------------------------------------------------------------
// OTLP protobuf: a minimal wire-format reader for ExportTraceServiceRequest

type field struct {
	num  int
	wire int
	v    uint64
	b    []byte
}

func fields(b []byte) ([]field, error) {
	var out []field
	for len(b) > 0 {
		key, n := binary.Uvarint(b)
		if n <= 0 {
			return nil, errors.New("protobuf: bad tag")
		}
		b = b[n:]
		f := field{num: int(key >> 3), wire: int(key & 7)}
		switch f.wire {
		case 0:
			v, n := binary.Uvarint(b)
			if n <= 0 {
				return nil, errors.New("protobuf: bad varint")
			}
			f.v, b = v, b[n:]
		case 1:
			if len(b) < 8 {
				return nil, errors.New("protobuf: short fixed64")
			}
			f.v, b = binary.LittleEndian.Uint64(b), b[8:]
		case 2:
			l, n := binary.Uvarint(b)
			if n <= 0 || uint64(len(b)-n) < l {
				return nil, errors.New("protobuf: bad length")
			}
			f.b, b = b[n:n+int(l)], b[n+int(l):]
		case 5:
			if len(b) < 4 {
				return nil, errors.New("protobuf: short fixed32")
			}
			f.v, b = uint64(binary.LittleEndian.Uint32(b)), b[4:]
		default:
			return nil, fmt.Errorf("protobuf: wire type %d", f.wire)
		}
		out = append(out, f)
	}
	return out, nil
}

func protoKV(b []byte) (string, any, error) {
	fs, err := fields(b)
	if err != nil {
		return "", nil, err
	}
	var key string
	var val any
	for _, f := range fs {
		switch f.num {
		case 1:
			key = string(f.b)
		case 2:
			vs, err := fields(f.b)
			if err != nil {
				return "", nil, err
			}
			for _, v := range vs {
				switch v.num {
				case 1:
					val = string(v.b)
				case 2:
					val = v.v != 0
				case 3:
					val = int64(v.v)
				case 4:
					val = math.Float64frombits(v.v)
				}
			}
		}
	}
	return key, val, nil
}

func decodeProto(data []byte) ([]Span, error) {
	top, err := fields(data)
	if err != nil {
		return nil, err
	}
	var out []Span
	for _, rsf := range top {
		if rsf.num != 1 {
			continue
		}
		rs, err := fields(rsf.b)
		if err != nil {
			return nil, err
		}
		svc := ""
		for _, f := range rs {
			if f.num == 1 { // Resource
				res, err := fields(f.b)
				if err != nil {
					return nil, err
				}
				for _, a := range res {
					if a.num == 1 {
						k, v, err := protoKV(a.b)
						if err != nil {
							return nil, err
						}
						if k == "service.name" {
							svc, _ = v.(string)
						}
					}
				}
			}
		}
		for _, f := range rs {
			if f.num != 2 { // ScopeSpans
				continue
			}
			ss, err := fields(f.b)
			if err != nil {
				return nil, err
			}
			for _, sf := range ss {
				if sf.num != 2 {
					continue
				}
				sp, err := fields(sf.b)
				if err != nil {
					return nil, err
				}
				s := Span{Service: svc, Attrs: map[string]any{}}
				for _, x := range sp {
					switch x.num {
					case 1:
						s.TraceID = hex.EncodeToString(x.b)
					case 2:
						s.SpanID = hex.EncodeToString(x.b)
					case 4:
						s.ParentSpanID = hex.EncodeToString(x.b)
					case 5:
						s.Name = string(x.b)
					case 6:
						s.Kind = int(x.v)
					case 7:
						s.Start = x.v
					case 8:
						s.End = x.v
					case 9:
						k, v, err := protoKV(x.b)
						if err != nil {
							return nil, err
						}
						s.Attrs[k] = v
					case 15:
						st, err := fields(x.b)
						if err != nil {
							return nil, err
						}
						for _, y := range st {
							if y.num == 3 {
								s.StatusCode = int(y.v)
							}
						}
					}
				}
				out = append(out, s)
			}
		}
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// span trees

// Tree describes what one trace must hold.
type Tree struct {
	Spans    []string    // span names
	Services []string    // service.name values
	Parents  [][2]string // [child, parent] span-name edges
}

// FindTree returns the id of a trace holding the tree, or "" and why not.
func FindTree(spans []Span, want Tree) (string, string) {
	byTrace := map[string][]Span{}
	for _, s := range spans {
		byTrace[s.TraceID] = append(byTrace[s.TraceID], s)
	}
	ids := make([]string, 0, len(byTrace))
	for id := range byTrace {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		sp := byTrace[id]
		names, svcs, byID := map[string]bool{}, map[string]bool{}, map[string]Span{}
		for _, s := range sp {
			names[s.Name], svcs[s.Service], byID[s.SpanID] = true, true, s
		}
		ok := true
		for _, n := range want.Spans {
			ok = ok && names[n]
		}
		for _, n := range want.Services {
			ok = ok && svcs[n]
		}
		for _, e := range want.Parents {
			found := false
			for _, s := range sp {
				if s.Name == e[0] {
					if p, has := byID[s.ParentSpanID]; has && p.Name == e[1] {
						found = true
					}
				}
			}
			ok = ok && found
		}
		if ok {
			return id, ""
		}
	}
	return "", fmt.Sprintf("no trace among %d holds spans %v, services %v, edges %v", len(byTrace), want.Spans, want.Services, want.Parents)
}

// AssertTree waits up to timeout for a trace holding the tree.
func (s *Sink) AssertTree(t testing.TB, want Tree, timeout time.Duration) string {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for {
		id, why := FindTree(s.Spans(), want)
		if id != "" {
			return id
		}
		if time.Now().After(deadline) {
			t.Fatalf("otlpsink: %s", why)
		}
		time.Sleep(20 * time.Millisecond)
	}
}
