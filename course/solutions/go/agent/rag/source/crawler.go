package source

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Clock is the crawler's time: Now, and Sleep for the rate limit. Tests
// pass a fake that advances on Sleep.
type Clock interface {
	Now() time.Time
	Sleep(d time.Duration)
}

type wallClock struct{}

func (wallClock) Now() time.Time        { return time.Now() }
func (wallClock) Sleep(d time.Duration) { time.Sleep(d) }

// CrawlerConfig configures a Crawler; zero fields take the defaults in
// parentheses.
type CrawlerConfig struct {
	Seeds      []string // start URLs
	AllowHosts []string // the only host names it may request (exact, case-insensitive)
	MaxPages   int      // pages fetched in total (100)
	MaxDepth   int      // links followed from a seed (3)
	RPS        float64  // requests per second per host, robots.txt included (1)
	UserAgent  string   // ("tinyllm-rag")
	MaxBytes   int64    // body bytes read per page (2 MiB)
	Clock      Clock    // (the wall clock)
	// Resolver maps a host name to addresses (net.DefaultResolver).
	Resolver func(ctx context.Context, host string) ([]netip.Addr, error)
	// AddrAllowed says which addresses may be dialed (PublicAddr).
	AddrAllowed func(netip.Addr) bool
}

// Refusal is a URL the crawler would not fetch, and why.
type Refusal struct{ URL, Reason string }

// ErrBlockedAddr is a dial to an address AddrAllowed refuses.
var ErrBlockedAddr = errors.New("source: address not allowed")

var cgnat = netip.MustParsePrefix("100.64.0.0/10")
var nat64 = netip.MustParsePrefix("64:ff9b::/96")

// PublicAddr is false for every address a crawler must never reach on
// behalf of a document: loopback, private (10/8, 172.16/12, 192.168/16,
// fc00::/7), link-local (169.254/16, where cloud metadata lives, and
// fe80::/10), unspecified, multicast, carrier-grade NAT (100.64/10), and
// NAT64 (64:ff9b::/96, which embeds an IPv4 address). IPv4-mapped IPv6
// addresses are checked as IPv4.
func PublicAddr(a netip.Addr) bool {
	a = a.Unmap()
	return a.IsValid() && !a.IsLoopback() && !a.IsPrivate() && !a.IsLinkLocalUnicast() &&
		!a.IsLinkLocalMulticast() && !a.IsInterfaceLocalMulticast() && !a.IsMulticast() &&
		!a.IsUnspecified() && !cgnat.Contains(a) && !nat64.Contains(a) &&
		!(a.Is4() && a.As4()[0] == 0)
}

// Crawler fetches pages breadth first from its seeds.
type Crawler struct {
	cfg     CrawlerConfig
	client  *http.Client
	mu      sync.Mutex
	next    map[string]time.Time // per host: the earliest time of its next request
	robots  map[string]*robots
	refused []Refusal
}

// NewCrawler applies the defaults and builds the guarded HTTP client: every
// connection, redirects included, is dialed through dialGuarded.
func NewCrawler(cfg CrawlerConfig) *Crawler {
	if cfg.MaxPages <= 0 {
		cfg.MaxPages = 100
	}
	if cfg.MaxDepth <= 0 {
		cfg.MaxDepth = 3
	}
	if cfg.RPS <= 0 {
		cfg.RPS = 1
	}
	if cfg.UserAgent == "" {
		cfg.UserAgent = "tinyllm-rag"
	}
	if cfg.MaxBytes <= 0 {
		cfg.MaxBytes = 2 << 20
	}
	if cfg.Clock == nil {
		cfg.Clock = wallClock{}
	}
	if cfg.Resolver == nil {
		cfg.Resolver = func(ctx context.Context, host string) ([]netip.Addr, error) {
			return net.DefaultResolver.LookupNetIP(ctx, "ip", host)
		}
	}
	if cfg.AddrAllowed == nil {
		cfg.AddrAllowed = PublicAddr
	}
	c := &Crawler{cfg: cfg, next: map[string]time.Time{}, robots: map[string]*robots{}}
	tr := &http.Transport{DialContext: c.dialGuarded, Proxy: nil, MaxIdleConnsPerHost: 2}
	c.client = &http.Client{Transport: tr, Timeout: 30 * time.Second, CheckRedirect: c.checkRedirect}
	return c
}

// dialGuarded resolves the host itself, refuses when ANY address is not
// allowed (a name that resolves to one public and one private address is
// an attack), and dials the address it checked, so a second DNS answer
// (rebinding) can never swap in another one.
func (c *Crawler) dialGuarded(ctx context.Context, network, hostport string) (net.Conn, error) {
	host, port, err := net.SplitHostPort(hostport)
	if err != nil {
		return nil, err
	}
	var addrs []netip.Addr
	if a, err := netip.ParseAddr(host); err == nil {
		addrs = []netip.Addr{a}
	} else if addrs, err = c.cfg.Resolver(ctx, host); err != nil {
		return nil, err
	}
	if len(addrs) == 0 {
		return nil, fmt.Errorf("source: %s has no addresses", host)
	}
	for _, a := range addrs {
		if !c.cfg.AddrAllowed(a) {
			return nil, fmt.Errorf("%w: %s resolves to %s", ErrBlockedAddr, host, a)
		}
	}
	var d net.Dialer
	return d.DialContext(ctx, network, net.JoinHostPort(addrs[0].Unmap().String(), port))
}

// hostAllowed checks the scheme and the host name against AllowHosts.
func (c *Crawler) hostAllowed(u *url.URL) bool {
	if u.Scheme != "http" && u.Scheme != "https" {
		return false
	}
	h := strings.ToLower(u.Hostname())
	for _, a := range c.cfg.AllowHosts {
		if h == strings.ToLower(a) {
			return true
		}
	}
	return false
}

// checkRedirect follows at most 5 redirects, each to an allowed host: a
// redirect is a new request and gets every check a first request gets (the
// dial-time address check applies too).
func (c *Crawler) checkRedirect(req *http.Request, via []*http.Request) error {
	if len(via) >= 5 {
		return errors.New("source: too many redirects")
	}
	if !c.hostAllowed(req.URL) {
		return fmt.Errorf("source: redirect to %s refused: host not allowed", req.URL.Redacted())
	}
	return nil
}

func (c *Crawler) refuse(u, why string) {
	c.mu.Lock()
	c.refused = append(c.refused, Refusal{u, why})
	c.mu.Unlock()
}

// Refused lists the URLs not fetched and why, in order.
func (c *Crawler) Refused() []Refusal {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]Refusal(nil), c.refused...)
}

// wait spaces requests to one host 1/RPS apart: it sleeps until the host's
// slot and books the next one.
func (c *Crawler) wait(host string) {
	gap := time.Duration(float64(time.Second) / c.cfg.RPS)
	c.mu.Lock()
	now := c.cfg.Clock.Now()
	at := c.next[host]
	if at.Before(now) {
		at = now
	}
	c.next[host] = at.Add(gap)
	c.mu.Unlock()
	if d := at.Sub(now); d > 0 {
		c.cfg.Clock.Sleep(d)
	}
}

func (c *Crawler) get(ctx context.Context, u *url.URL) (*http.Response, error) {
	c.wait(strings.ToLower(u.Hostname()))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", c.cfg.UserAgent)
	return c.client.Do(req)
}

var hrefRE = regexp.MustCompile(`(?i)<a\b[^>]*?\bhref\s*=\s*("([^"]*)"|'([^']*)'|([^\s>]+))`)

// Fetch crawls: seeds first, then the links of each page in document order,
// each URL (without its fragment) at most once, up to MaxPages and MaxDepth.
// A URL off the allowlist, disallowed by robots.txt, or refused at dial
// time is skipped and listed in Refused; only a cancelled ctx is an error.
func (c *Crawler) Fetch(ctx context.Context) ([]RawDoc, error) {
	type item struct {
		u     *url.URL
		depth int
	}
	var queue []item
	seen := map[string]bool{}
	push := func(raw string, base *url.URL, depth int) {
		u, err := url.Parse(strings.TrimSpace(raw))
		if err != nil {
			c.refuse(raw, "unparsable URL")
			return
		}
		if base != nil {
			u = base.ResolveReference(u)
		}
		u.Fragment, u.RawFragment = "", ""
		key := u.String()
		if seen[key] {
			return
		}
		seen[key] = true
		if !c.hostAllowed(u) {
			c.refuse(key, "host not allowed")
			return
		}
		queue = append(queue, item{u, depth})
	}
	for _, s := range c.cfg.Seeds {
		push(s, nil, 0)
	}
	var out []RawDoc
	for len(queue) > 0 && len(out) < c.cfg.MaxPages {
		if err := ctx.Err(); err != nil {
			return out, err
		}
		it := queue[0]
		queue = queue[1:]
		key := it.u.String()
		rb := c.robotsFor(ctx, it.u)
		if rb.err != nil {
			c.refuse(key, "robots.txt unreachable: "+rb.err.Error())
			continue
		}
		if !rb.allowed(it.u.EscapedPath()) {
			c.refuse(key, "disallowed by robots.txt")
			continue
		}
		resp, err := c.get(ctx, it.u)
		if err != nil {
			if ctx.Err() != nil {
				return out, ctx.Err()
			}
			c.refuse(key, err.Error())
			continue
		}
		body, err := io.ReadAll(io.LimitReader(resp.Body, c.cfg.MaxBytes))
		resp.Body.Close()
		if err != nil || resp.StatusCode != http.StatusOK {
			c.refuse(key, "HTTP "+strconv.Itoa(resp.StatusCode))
			continue
		}
		ct := strings.ToLower(resp.Header.Get("Content-Type"))
		if !strings.HasPrefix(ct, "text/html") && !strings.HasPrefix(ct, "text/plain") && !strings.HasPrefix(ct, "text/markdown") {
			c.refuse(key, "content type "+ct)
			continue
		}
		final := resp.Request.URL
		out = append(out, RawDoc{URI: key, Source: "web", ContentType: ct, Body: body})
		if strings.HasPrefix(ct, "text/html") && it.depth < c.cfg.MaxDepth {
			for _, m := range hrefRE.FindAllSubmatch(body, -1) {
				link := string(bytes.Join([][]byte{m[2], m[3], m[4]}, nil))
				push(htmlUnescape(link), final, it.depth+1)
			}
		}
	}
	return out, nil
}

func htmlUnescape(s string) string { return strings.ReplaceAll(s, "&amp;", "&") }

// robots is one host's robots.txt rules for our user agent.
type robots struct {
	all   bool // true: everything allowed (no file); false with no rules: nothing
	rules []rule
	err   error // why robots.txt could not be fetched (then nothing is allowed)
}

type rule struct {
	allow bool
	re    *regexp.Regexp
	n     int // pattern length, for longest match
}

// allowed applies RFC 9309: the longest matching rule wins, Allow wins a
// tie, and no matching rule means allowed.
func (r *robots) allowed(path string) bool {
	if r.all {
		return true
	}
	if r.rules == nil {
		return false
	}
	if path == "" {
		path = "/"
	}
	best, allow := -1, true
	for _, x := range r.rules {
		if x.re.MatchString(path) && (x.n > best || (x.n == best && x.allow)) {
			best, allow = x.n, x.allow
		}
	}
	return allow
}

// robotsFor fetches and caches a host's robots.txt (rate-limited like any
// request). 2xx: parse it. 4xx: no restrictions. 5xx or no answer: the
// whole host is disallowed (RFC 9309 2.3.1.4: unreachable means "assume
// complete disallow").
func (c *Crawler) robotsFor(ctx context.Context, u *url.URL) *robots {
	host := strings.ToLower(u.Host)
	c.mu.Lock()
	r, ok := c.robots[host]
	c.mu.Unlock()
	if ok {
		return r
	}
	ru := &url.URL{Scheme: u.Scheme, Host: u.Host, Path: "/robots.txt"}
	r = &robots{}
	resp, err := c.get(ctx, ru)
	switch {
	case err != nil:
		r.err = err // unreachable: disallow everything
	case resp.StatusCode >= 200 && resp.StatusCode < 300:
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 500<<10))
		r = parseRobots(body, c.cfg.UserAgent)
	case resp.StatusCode >= 400 && resp.StatusCode < 500:
		r = &robots{all: true}
	}
	if resp != nil {
		resp.Body.Close()
	}
	c.mu.Lock()
	c.robots[host] = r
	c.mu.Unlock()
	return r
}

// parseRobots keeps the group whose user-agent matches ours (the product
// token, case-insensitive), else the "*" group. Patterns support "*" (any
// run) and a trailing "$" (end of path). An empty Disallow allows all.
func parseRobots(body []byte, ua string) *robots {
	token := strings.ToLower(strings.SplitN(ua, "/", 2)[0])
	type group struct {
		agents []string
		rules  []rule
	}
	var groups []*group
	var cur *group
	lastAgent := false
	sc := bufio.NewScanner(bytes.NewReader(body))
	for sc.Scan() {
		line := sc.Text()
		if i := strings.IndexByte(line, '#'); i >= 0 {
			line = line[:i]
		}
		k, v, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		k, v = strings.ToLower(strings.TrimSpace(k)), strings.TrimSpace(v)
		switch k {
		case "user-agent":
			if cur == nil || !lastAgent {
				cur = &group{}
				groups = append(groups, cur)
			}
			cur.agents = append(cur.agents, strings.ToLower(v))
			lastAgent = true
		case "allow", "disallow":
			lastAgent = false
			if cur == nil || v == "" {
				continue
			}
			pat := regexp.QuoteMeta(v)
			pat = strings.ReplaceAll(pat, `\*`, ".*")
			if strings.HasSuffix(pat, `\$`) {
				pat = strings.TrimSuffix(pat, `\$`) + "$"
			}
			cur.rules = append(cur.rules, rule{allow: k == "allow", re: regexp.MustCompile("^" + pat), n: len(v)})
		default:
			lastAgent = false
		}
	}
	var star *group
	for _, g := range groups {
		for _, a := range g.agents {
			if a == "*" {
				if star == nil {
					star = g
				}
			} else if a == token {
				return &robots{rules: append([]rule{}, g.rules...), all: len(g.rules) == 0}
			}
		}
	}
	if star == nil {
		return &robots{all: true}
	}
	return &robots{rules: append([]rule{}, star.rules...), all: len(star.rules) == 0}
}
