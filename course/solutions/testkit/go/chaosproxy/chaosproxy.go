// Package chaosproxy is a TCP proxy that misbehaves on purpose:
// latency on every chunk, connections refused (drop), a reset after N bytes
// downstream (a stream cut mid-response), half-open (the proxy accepts and
// then never forwards or closes), and a bandwidth limit. Faults can change
// while connections are open; each new chunk reads the current faults.
package chaosproxy

import (
	"io"
	"net"
	"sync"
	"time"
)

// Faults to apply; the zero value forwards faithfully.
type Faults struct {
	Latency         time.Duration // before forwarding each chunk, both directions
	Drop            bool          // close new connections at once (connection refused-like)
	ResetAfterBytes int64         // > 0: RST the client after this many bytes downstream
	HalfOpen        bool          // accept, read, never forward, never close
	BandwidthBps    int64         // > 0: cap each direction at this many bytes per second
}

type Proxy struct {
	upstream string
	ln       net.Listener
	mu       sync.Mutex
	f        Faults
	conns    map[net.Conn]struct{}
	accepted int
	closed   bool
	wg       sync.WaitGroup
}

// Start listens on 127.0.0.1 at a free port and forwards to upstream.
func Start(upstream string, f Faults) (*Proxy, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, err
	}
	p := &Proxy{upstream: upstream, ln: ln, f: f, conns: map[net.Conn]struct{}{}}
	p.wg.Add(1)
	go p.serve()
	return p, nil
}

func (p *Proxy) Addr() string { return p.ln.Addr().String() }

// SetFaults changes the faults for every chunk from now on.
func (p *Proxy) SetFaults(f Faults) {
	p.mu.Lock()
	p.f = f
	p.mu.Unlock()
}

func (p *Proxy) faults() Faults {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.f
}

// Accepted is how many client connections the proxy has accepted.
func (p *Proxy) Accepted() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.accepted
}

func (p *Proxy) Close() error {
	p.mu.Lock()
	p.closed = true
	for c := range p.conns {
		c.Close()
	}
	p.mu.Unlock()
	err := p.ln.Close()
	p.wg.Wait()
	return err
}

func (p *Proxy) track(c net.Conn, on bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if on {
		p.conns[c] = struct{}{}
	} else {
		delete(p.conns, c)
	}
}

func (p *Proxy) serve() {
	defer p.wg.Done()
	for {
		c, err := p.ln.Accept()
		if err != nil {
			return
		}
		p.mu.Lock()
		p.accepted++
		p.mu.Unlock()
		p.wg.Add(1)
		go p.handle(c)
	}
}

func (p *Proxy) handle(client net.Conn) {
	defer p.wg.Done()
	p.track(client, true)
	defer p.track(client, false)
	f := p.faults()
	if f.Drop {
		client.Close()
		return
	}
	if f.HalfOpen {
		io.Copy(io.Discard, client) // hold the connection until the client or Close ends it
		client.Close()
		return
	}
	up, err := net.Dial("tcp", p.upstream)
	if err != nil {
		client.Close()
		return
	}
	p.track(up, true)
	defer p.track(up, false)
	done := make(chan struct{}, 2)
	go func() { p.pipe(up, client, false); done <- struct{}{} }()
	go func() { p.pipe(client, up, true); done <- struct{}{} }()
	<-done
	client.Close()
	up.Close()
	<-done
}

// pipe copies src to dst chunk by chunk under the current faults.
func (p *Proxy) pipe(dst, src net.Conn, downstream bool) {
	buf := make([]byte, 4096)
	var sent int64
	for {
		n, err := src.Read(buf)
		if n > 0 {
			f := p.faults()
			chunk := buf[:n]
			if f.Latency > 0 {
				time.Sleep(f.Latency)
			}
			if downstream && f.ResetAfterBytes > 0 && sent+int64(len(chunk)) >= f.ResetAfterBytes {
				keep := f.ResetAfterBytes - sent
				if keep > 0 {
					dst.Write(chunk[:keep])
				}
				if tc, ok := dst.(*net.TCPConn); ok {
					tc.SetLinger(0) // close with RST, not FIN
				}
				dst.Close()
				src.Close()
				return
			}
			if f.BandwidthBps > 0 {
				time.Sleep(time.Duration(float64(len(chunk)) / float64(f.BandwidthBps) * float64(time.Second)))
			}
			if _, werr := dst.Write(chunk); werr != nil {
				return
			}
			sent += int64(len(chunk))
		}
		if err != nil {
			if tc, ok := dst.(*net.TCPConn); ok {
				tc.CloseWrite()
			}
			return
		}
	}
}
