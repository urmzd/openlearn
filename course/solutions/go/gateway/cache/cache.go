// Package cache is the gateway's response cache (gw.06): an LRU with a TTL
// per entry, keys scoped to the tenant and the configuration revision,
// canonical request bodies, singleflight so a burst of identical misses
// makes one upstream call, and stream replay (a cached SSE answer is sent
// back byte for byte).
//
// Only deterministic requests are cached: temperature 0 or a fixed seed
// (openai-subset.v1.yaml: X-TL-Cache: hit|miss on cacheable requests).
// Contract: openapi/openai-subset.v1.yaml, openapi/admin.v1.yaml
// (POST /admin/v1/cache:purge), config/runtime.schema.json ([gateway]
// cache_entries, cache_ttl_s). Chapter:
// content/courses/systems/06-distributed-data-and-caching/01-response-cache.md.
package cache

import (
	"bytes"
	"container/list"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"net/http"
	"sync"
	"time"

	"tinyllm/gateway/auth"
	"tinyllm/gateway/server"
)

// Key identifies one cacheable answer: SHA-256 of tenant, configuration
// revision, and canonical request.
type Key [32]byte

// Entry is a cached answer, replayed exactly.
type Entry struct {
	Status      int
	ContentType string
	Body        []byte // the whole JSON body, or the whole SSE stream
	Model       string // the public model id (for purges)
}

// item is one LRU slot.
type item struct {
	key     Key
	e       Entry
	expires time.Time
}

// LRU holds at most capacity entries, least recently used evicted first;
// an entry expires ttl after it was put. Safe for concurrent use.
type LRU struct {
	capacity int
	clock    server.Clock

	mu sync.Mutex
	ll *list.List // front = most recently used
	m  map[Key]*list.Element
}

// New returns an empty cache ([gateway].cache_entries); clock nil = wall.
func New(capacity int, clock server.Clock) *LRU {
	if clock == nil {
		clock = server.WallClock
	}
	return &LRU{capacity: capacity, clock: clock, ll: list.New(), m: map[Key]*list.Element{}}
}

// Get returns the entry for k unless it is missing or expired (an expired
// entry is removed). A hit makes k the most recently used.
func (c *LRU) Get(ctx context.Context, k Key) (Entry, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	el, ok := c.m[k]
	if !ok {
		return Entry{}, false
	}
	it := el.Value.(*item)
	if !c.clock.Now().Before(it.expires) {
		c.ll.Remove(el)
		delete(c.m, k)
		return Entry{}, false
	}
	c.ll.MoveToFront(el)
	return it.e, true
}

// Put stores e under k for ttl, evicting the least recently used entries
// beyond capacity. A capacity or ttl of 0 stores nothing.
func (c *LRU) Put(ctx context.Context, k Key, e Entry, ttl time.Duration) {
	if c.capacity <= 0 || ttl <= 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	exp := c.clock.Now().Add(ttl)
	if el, ok := c.m[k]; ok {
		el.Value = &item{key: k, e: e, expires: exp}
		c.ll.MoveToFront(el)
		return
	}
	c.m[k] = c.ll.PushFront(&item{key: k, e: e, expires: exp})
	for c.ll.Len() > c.capacity {
		last := c.ll.Back()
		c.ll.Remove(last)
		delete(c.m, last.Value.(*item).key)
	}
}

// Len is the number of entries held (expired ones included until touched).
func (c *LRU) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.ll.Len()
}

// Purge removes every entry of model, or every entry when model is ""
// (POST /admin/v1/cache:purge). It returns how many it removed.
func (c *LRU) Purge(model string) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	n := 0
	for el := c.ll.Front(); el != nil; {
		next := el.Next()
		it := el.Value.(*item)
		if model == "" || it.e.Model == model {
			c.ll.Remove(el)
			delete(c.m, it.key)
			n++
		}
		el = next
	}
	return n
}

// CanonicalRequest is a request reduced to what determines its answer.
type CanonicalRequest struct {
	Path string
	JSON []byte // the canonical body: sorted keys, defaults filled, `user` dropped
}

// defaults are the openai-subset.v1.yaml values an omitted field takes;
// filling them makes `{"temperature": 1}` and `{}` the same request.
var defaults = map[string]any{
	"temperature": 1.0, "top_p": 1.0, "top_k": 0.0, "min_p": 0.0, "repetition_penalty": 1.0,
	"presence_penalty": 0.0, "frequency_penalty": 0.0, "n": 1.0, "stream": false,
	"logprobs": nil, "stop": nil,
}

// ignored fields do not change the answer.
var ignored = []string{"user"}

// Canonicalize reduces a chat or completions body to its canonical form and
// reports whether it is cacheable: temperature 0 (after defaults) or a seed.
// The canonical JSON has sorted keys and numbers re-encoded, so field order
// and spelling (0 vs 0.0) do not change the key.
func Canonicalize(path string, body []byte) (CanonicalRequest, bool, error) {
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil || m == nil {
		return CanonicalRequest{}, false, errors.New("the request body must be a JSON object")
	}
	for k, v := range defaults {
		if _, ok := m[k]; !ok {
			m[k] = v
		}
	}
	for _, k := range ignored {
		delete(m, k)
	}
	if v, ok := m["logprobs"].(bool); ok && !v {
		m["logprobs"] = nil
	}
	temp, _ := m["temperature"].(float64)
	_, seeded := m["seed"].(float64)
	canon, err := json.Marshal(m) // encoding/json sorts map keys
	if err != nil {
		return CanonicalRequest{}, false, err
	}
	return CanonicalRequest{Path: path, JSON: canon}, temp == 0 || seeded, nil
}

// KeyOf is the cache key: the tenant (never shared across tenants; keys of
// one tenant share), the configuration revision (a route or model change
// makes old answers unreachable), and the canonical request.
func KeyOf(p auth.Principal, cfgRev string, r CanonicalRequest) Key {
	h := sha256.New()
	for _, part := range []string{p.Tenant, cfgRev, r.Path} {
		h.Write([]byte(part))
		h.Write([]byte{0})
	}
	h.Write(r.JSON)
	var k Key
	copy(k[:], h.Sum(nil))
	return k
}

// call is one in-flight miss that later identical requests wait for.
type call struct {
	done  chan struct{}
	entry Entry
	ok    bool
}

// Options configure the stage.
type Options struct {
	TTL time.Duration // cache_ttl_s
	// Rev returns the configuration revision (for example the route epoch);
	// nil means "".
	Rev func() string
	// MaxEntryBytes caps one cached body; 0 = 1 MiB.
	MaxEntryBytes int
}

// recorder passes the response to the client and keeps a copy.
type recorder struct {
	http.ResponseWriter
	status int
	buf    bytes.Buffer
	over   bool
	max    int
}

func (r *recorder) WriteHeader(code int) {
	if r.status == 0 {
		r.status = code
	}
	r.ResponseWriter.WriteHeader(code)
}

func (r *recorder) Write(b []byte) (int, error) {
	if r.status == 0 {
		r.status = http.StatusOK
	}
	if !r.over {
		if r.buf.Len()+len(b) > r.max {
			r.over = true
			r.buf.Reset()
		} else {
			r.buf.Write(b)
		}
	}
	return r.ResponseWriter.Write(b)
}

func (r *recorder) Unwrap() http.ResponseWriter {
	return r.ResponseWriter
}

// complete reports whether a recorded answer is whole: a JSON body, or an
// SSE stream that ended with [DONE] (a stream cut by an error is not).
func complete(contentType string, body []byte) bool {
	if bytes.HasPrefix([]byte(contentType), []byte("text/event-stream")) {
		return bytes.HasSuffix(body, []byte("data: [DONE]\n\n"))
	}
	return json.Valid(body)
}

// replay writes a cached entry as a hit.
func replay(w http.ResponseWriter, e Entry) {
	w.Header().Set("Content-Type", e.ContentType)
	w.Header().Set("X-TL-Cache", "hit")
	w.WriteHeader(e.Status)
	_, _ = w.Write(e.Body)
}

// Middleware is the cache stage. For an authenticated, cacheable chat or
// completions request it answers a hit from c (X-TL-Cache: hit), or on a
// miss (X-TL-Cache: miss) lets one request through per key and stores its
// answer when it is a whole 200; identical requests that arrive meanwhile
// wait for it and are answered as hits. Everything else passes untouched.
func Middleware(c *LRU, o Options) server.Middleware {
	if o.MaxEntryBytes <= 0 {
		o.MaxEntryBytes = 1 << 20
	}
	var mu sync.Mutex
	inflight := map[Key]*call{}
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			p, authed := auth.PrincipalFrom(r.Context())
			if !authed || r.Method != http.MethodPost || (r.URL.Path != "/v1/chat/completions" && r.URL.Path != "/v1/completions") {
				next.ServeHTTP(w, r)
				return
			}
			ex := server.ExchangeFrom(r.Context())
			req, err := ex.Request(r)
			if err != nil {
				next.ServeHTTP(w, r)
				return
			}
			canon, ok, err := Canonicalize(r.URL.Path, req.Body)
			if err != nil || !ok {
				next.ServeHTTP(w, r)
				return
			}
			rev := ""
			if o.Rev != nil {
				rev = o.Rev()
			}
			// The same prompt has a different response schema in API v2.
			// Absent and explicit v1 headers retain the original key.
			if version := r.Header.Get("X-TL-API-Version"); version != "" && version != "1" {
				rev += "\x00api:" + version
			}
			k := KeyOf(p, rev, canon)
			for {
				if e, hit := c.Get(r.Context(), k); hit {
					ex.SetAttr("tl.cache.hit", true)
					replay(w, e)
					return
				}
				mu.Lock()
				if cl, waiting := inflight[k]; waiting {
					mu.Unlock()
					select {
					case <-cl.done:
					case <-r.Context().Done():
						return
					}
					if cl.ok {
						ex.SetAttr("tl.cache.hit", true)
						replay(w, cl.entry)
						return
					}
					break // the leader got no cacheable answer: go upstream ourselves
				}
				// Check again under the lock: a leader that finished between our
				// Get and this Lock stored its entry before leaving inflight.
				if e, hit := c.Get(r.Context(), k); hit {
					mu.Unlock()
					ex.SetAttr("tl.cache.hit", true)
					replay(w, e)
					return
				}
				cl := &call{done: make(chan struct{})}
				inflight[k] = cl
				mu.Unlock()
				ex.SetAttr("tl.cache.hit", false)
				w.Header().Set("X-TL-Cache", "miss")
				rec := &recorder{ResponseWriter: w, max: o.MaxEntryBytes}
				func() {
					defer func() {
						mu.Lock()
						delete(inflight, k)
						mu.Unlock()
						close(cl.done)
					}()
					next.ServeHTTP(rec, r)
					ct := w.Header().Get("Content-Type")
					if rec.status == http.StatusOK && !rec.over && complete(ct, rec.buf.Bytes()) {
						cl.entry = Entry{Status: rec.status, ContentType: ct, Body: append([]byte(nil), rec.buf.Bytes()...), Model: req.Model}
						cl.ok = true
						c.Put(r.Context(), k, cl.entry, o.TTL)
					}
				}()
				return
			}
			ex.SetAttr("tl.cache.hit", false)
			w.Header().Set("X-TL-Cache", "miss")
			next.ServeHTTP(w, r)
		})
	}
}

// PurgeHandler serves POST /admin/v1/cache:purge: {"model"?} in, {"purged"}
// out (openapi/admin.v1.yaml). gw.07 mounts it behind its admin scope check.
func PurgeHandler(c *LRU) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			server.WriteError(w, http.StatusMethodNotAllowed, "invalid_request_error", "", "", "use POST")
			return
		}
		var in struct {
			Model string `json:"model"`
		}
		if r.ContentLength != 0 {
			if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
				server.WriteError(w, http.StatusBadRequest, "invalid_request_error", "", "", "malformed JSON: "+err.Error())
				return
			}
		}
		n := c.Purge(in.Model)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]int{"purged": n})
	})
}
