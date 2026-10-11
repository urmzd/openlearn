package chaosproxy_test

import (
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"openlearn.urmzd.com/tl/testkit/chaosproxy"
)

func upstream(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(strings.Repeat("x", 10000)))
	}))
}

func get(addr string, timeout time.Duration) (string, error) {
	c := &http.Client{Timeout: timeout, Transport: &http.Transport{DisableKeepAlives: true}}
	r, err := c.Get("http://" + addr + "/")
	if err != nil {
		return "", err
	}
	defer r.Body.Close()
	b, err := io.ReadAll(r.Body)
	return string(b), err
}

func TestForwardLatencyAndBandwidth(t *testing.T) {
	up := upstream(t)
	defer up.Close()
	p, err := chaosproxy.Start(strings.TrimPrefix(up.URL, "http://"), chaosproxy.Faults{})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	body, err := get(p.Addr(), 5*time.Second)
	if err != nil || len(body) != 10000 {
		t.Fatalf("faithful forward: %d bytes, %v", len(body), err)
	}
	p.SetFaults(chaosproxy.Faults{Latency: 50 * time.Millisecond, BandwidthBps: 100000})
	t0 := time.Now()
	if body, err = get(p.Addr(), 5*time.Second); err != nil || len(body) != 10000 {
		t.Fatalf("slow forward: %v", err)
	}
	if d := time.Since(t0); d < 100*time.Millisecond {
		t.Fatalf("latency and bandwidth limit not applied: %v", d)
	}
}

func TestResetMidStreamDropAndHalfOpen(t *testing.T) {
	up := upstream(t)
	defer up.Close()
	p, _ := chaosproxy.Start(strings.TrimPrefix(up.URL, "http://"), chaosproxy.Faults{ResetAfterBytes: 500})
	defer p.Close()
	if body, err := get(p.Addr(), 5*time.Second); err == nil && len(body) == 10000 {
		t.Fatal("a reset mid-stream delivered the whole body")
	}
	p.SetFaults(chaosproxy.Faults{Drop: true})
	if _, err := get(p.Addr(), 2*time.Second); err == nil {
		t.Fatal("drop: the request succeeded")
	}
	p.SetFaults(chaosproxy.Faults{HalfOpen: true})
	c, err := net.Dial("tcp", p.Addr())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	c.Write([]byte("GET / HTTP/1.1\r\nHost: x\r\n\r\n"))
	c.SetReadDeadline(time.Now().Add(200 * time.Millisecond))
	buf := make([]byte, 10)
	if _, err := c.Read(buf); err == nil {
		t.Fatal("half-open: got a response")
	} else if ne, ok := err.(net.Error); !ok || !ne.Timeout() {
		t.Fatalf("half-open: want a read timeout, got %v", err)
	}
	if p.Accepted() < 3 {
		t.Fatalf("accepted %d", p.Accepted())
	}
}
