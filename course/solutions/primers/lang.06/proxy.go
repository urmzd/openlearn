// Package proxy is the lang.06 primer exercise: a streaming HTTP reverse
// proxy with a deadline. It is practice, not part of the system; gw.00 builds
// the gateway's real proxy on the same ideas.
//
// Chapter: content/courses/programming-and-algorithms/01-language-primers/06-go.md.
package proxy

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"time"
)

// New returns a handler that forwards every request to upstream (a base URL
// such as "http://127.0.0.1:8080") with the same method, path, query,
// headers, and body, and streams the upstream's status, headers, and body
// back, flushing each chunk as it arrives.
//
// Every forwarded request has a deadline of timeout, counted from the moment
// the handler starts. If the upstream has not answered with headers by then,
// the client gets 504 Gateway Timeout. If the deadline passes while the body
// is streaming, the body ends there. If the client goes away first, the
// upstream request is cancelled. An upstream that cannot be reached is 502
// Bad Gateway.
func New(upstream string, timeout time.Duration) http.Handler {
	base := strings.TrimRight(upstream, "/")
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), timeout)
		defer cancel()

		target := base + r.URL.EscapedPath()
		if r.URL.RawQuery != "" {
			target += "?" + r.URL.RawQuery
		}
		req, err := http.NewRequestWithContext(ctx, r.Method, target, r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		req.ContentLength = r.ContentLength
		req.Header = r.Header.Clone()

		resp, err := http.DefaultClient.Do(req)
		switch {
		case err == nil:
		case r.Context().Err() != nil:
			return // the client left; there is nobody to answer
		case errors.Is(err, context.DeadlineExceeded):
			http.Error(w, "upstream timed out", http.StatusGatewayTimeout)
			return
		default:
			http.Error(w, "upstream unreachable: "+err.Error(), http.StatusBadGateway)
			return
		}
		defer resp.Body.Close()

		for k, vs := range resp.Header {
			for _, v := range vs {
				w.Header().Add(k, v)
			}
		}
		w.WriteHeader(resp.StatusCode)
		_, _ = CopyFlush(w, resp.Body)
	})
}

// CopyFlush copies src to w until src is exhausted, flushing w after every
// chunk it writes. It returns the number of bytes written and the error that
// stopped it: nil when src ended cleanly with io.EOF.
func CopyFlush(w http.ResponseWriter, src io.Reader) (int64, error) {
	rc := http.NewResponseController(w)
	buf := make([]byte, 32*1024)
	var total int64
	for {
		n, err := src.Read(buf)
		if n > 0 {
			m, werr := w.Write(buf[:n])
			total += int64(m)
			if werr != nil {
				return total, werr
			}
			if ferr := rc.Flush(); ferr != nil && !errors.Is(ferr, http.ErrNotSupported) {
				return total, ferr
			}
		}
		if errors.Is(err, io.EOF) {
			return total, nil
		}
		if err != nil {
			return total, err
		}
	}
}
