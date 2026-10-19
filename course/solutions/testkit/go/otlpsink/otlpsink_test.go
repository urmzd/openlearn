package otlpsink_test

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/binary"
	"net/http"
	"testing"
	"time"

	"openlearn.urmzd.com/tl/testkit/otlpsink"
)

const jsonBody = `{"resourceSpans":[{"resource":{"attributes":[{"key":"service.name","value":{"stringValue":"forge-gateway"}}]},
 "scopeSpans":[{"spans":[
  {"traceId":"0af7651916cd43dd8448eb211c80319c","spanId":"b7ad6b7169203331","name":"POST /v1/completions","kind":2,
   "startTimeUnixNano":"1","endTimeUnixNano":"2","attributes":[{"key":"http.response.status_code","value":{"intValue":"200"}}]},
  {"traceId":"0af7651916cd43dd8448eb211c80319c","spanId":"00f067aa0ba902b7","parentSpanId":"b7ad6b7169203331","name":"gateway.proxy","kind":3}]}]}]}`

func TestJSONAndTree(t *testing.T) {
	s, err := otlpsink.Start()
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	gz.Write([]byte(jsonBody))
	gz.Close()
	req, _ := http.NewRequest("POST", s.Endpoint()+"/v1/traces", &buf)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Content-Encoding", "gzip")
	r, err := http.DefaultClient.Do(req)
	if err != nil || r.StatusCode != 200 {
		t.Fatalf("post: %v %v", err, r)
	}
	sp := s.WaitForSpans(2, time.Second)
	if sp[0].Service != "forge-gateway" || sp[0].Attrs["http.response.status_code"] != int64(200) {
		t.Fatalf("span %+v", sp[0])
	}
	s.AssertTree(t, otlpsink.Tree{
		Spans:    []string{"POST /v1/completions", "gateway.proxy"},
		Services: []string{"forge-gateway"},
		Parents:  [][2]string{{"gateway.proxy", "POST /v1/completions"}},
	}, time.Second)
	if id, _ := otlpsink.FindTree(sp, otlpsink.Tree{Parents: [][2]string{{"POST /v1/completions", "gateway.proxy"}}}); id != "" {
		t.Fatal("a reversed edge matched")
	}
}

// protobuf helpers: tag, length-delimited bytes, fixed64
func tag(n, w int) []byte { return binary.AppendUvarint(nil, uint64(n<<3|w)) }
func ld(n int, b []byte) []byte {
	return append(append(tag(n, 2), binary.AppendUvarint(nil, uint64(len(b)))...), b...)
}
func fx(n int, v uint64) []byte { return binary.LittleEndian.AppendUint64(tag(n, 1), v) }

func TestProtobufAndHang(t *testing.T) {
	s, _ := otlpsink.Start()
	defer s.Close()
	kv := ld(1, []byte("service.name"))
	kv = append(kv, ld(2, ld(1, []byte("forge-engine")))...)
	span := ld(1, bytes.Repeat([]byte{0xab}, 16))
	span = append(span, ld(2, bytes.Repeat([]byte{0xcd}, 8))...)
	span = append(span, ld(5, []byte("engine.decode"))...)
	span = append(span, fx(7, 10)...)
	span = append(span, fx(8, 20)...)
	rs := append(ld(1, ld(1, kv)), ld(2, ld(2, span))...)
	body := ld(1, rs)
	r, err := http.Post(s.Endpoint()+"/v1/traces", "application/x-protobuf", bytes.NewReader(body))
	if err != nil || r.StatusCode != 200 {
		t.Fatalf("post: %v %v %v", err, r, s.Errors())
	}
	sp := s.Spans()
	if len(sp) != 1 || sp[0].Name != "engine.decode" || sp[0].Service != "forge-engine" || sp[0].End != 20 || sp[0].SpanID != "cdcdcdcdcdcdcdcd" {
		t.Fatalf("spans %+v", sp)
	}
	s.SetHang(true)
	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "POST", s.Endpoint()+"/v1/traces", bytes.NewReader([]byte(jsonBody)))
	if _, err := http.DefaultClient.Do(req); err == nil {
		t.Fatal("hang mode answered")
	}
	if s.Requests() != 2 {
		t.Fatalf("requests %d", s.Requests())
	}
}
