// Package auth is the gateway's authentication and authorization (gw.02):
// API keys "tl_<id>_<secret>" looked up in O(1) by id and verified against
// HMAC-SHA256(pepper, secret), never stored in plaintext; scopes, a model
// allowlist, and a tenant per key; and the authn middleware of the chain.
//
// Contract: openapi/openai-subset.v1.yaml (401 invalid_api_key, 403
// insufficient_scope), openapi/admin.v1.yaml (KeyCreate, KeyInfo, KeyCreated),
// config/runtime.schema.json ([gateway] keys_file, pepper_env,
// key_cache_ttl_s). Chapter:
// content/courses/software-engineering/10-authorization/01-api-keys-and-scopes.md.
package auth

import (
	"bufio"
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"tinyllm/gateway/server"
)

// Scopes of admin.v1.yaml.
const (
	ScopeInfer = "infer"
	ScopeEmbed = "embed"
	ScopeAdmin = "admin"
	ScopeDebug = "debug"
)

// Principal is who a request acts as: the key and everything attached to it.
// Later stages read it with PrincipalFrom (limits by KeyID, the cache scopes
// by Tenant, the ledger records Tenant and KeyID).
type Principal struct {
	KeyID    string
	Tenant   string
	Name     string
	Scopes   []string
	Models   []string // allowed models; empty = all
	RPM, TPM int      // 0 = unlimited
	Priority int      // sent to engines as X-TL-Priority
}

// HasScope reports whether p holds scope s.
func (p Principal) HasScope(s string) bool {
	for _, x := range p.Scopes {
		if x == s {
			return true
		}
	}
	return false
}

// Record is one line of the keys file (JSON Lines): admin.v1.yaml KeyInfo
// plus the HMAC. The plaintext secret is never in it.
type Record struct {
	KeyID     string     `json:"key_id"`
	HMAC      string     `json:"hmac"` // hex HMAC-SHA256(pepper, secret)
	Tenant    string     `json:"tenant"`
	Name      string     `json:"name"`
	Scopes    []string   `json:"scopes"`
	RPM       int        `json:"rpm"`
	TPM       int        `json:"tpm"`
	Priority  int        `json:"priority"`
	Models    []string   `json:"models"`
	CreatedAt time.Time  `json:"created_at"`
	ExpiresAt *time.Time `json:"expires_at"`
	Revoked   bool       `json:"revoked"`
}

// Principal is the record as a Principal.
func (r Record) Principal() Principal {
	return Principal{KeyID: r.KeyID, Tenant: r.Tenant, Name: r.Name, Scopes: r.Scopes,
		Models: r.Models, RPM: r.RPM, TPM: r.TPM, Priority: r.Priority}
}

// KeySpec is what Create needs (admin.v1.yaml KeyCreate).
type KeySpec struct {
	Tenant    string
	Name      string
	Scopes    []string
	RPM, TPM  int
	Priority  int
	Models    []string
	ExpiresAt *time.Time
}

// Errors of Lookup and Authorize.
var (
	ErrInvalidKey   = errors.New("invalid API key")    // 401 invalid_api_key
	ErrInsufficient = errors.New("insufficient scope") // 403 insufficient_scope
	ErrNotFound     = errors.New("no such key")        // admin: 404
)

const (
	idAlphabet     = "abcdefghijklmnopqrstuvwxyz234567"
	secretAlphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
	idLen          = 12
	secretLen      = 32
)

// ParseKey splits "tl_<id>_<secret>": id is 12 characters of [a-z2-7],
// secret 32 of [A-Za-z0-9]. ok is false for anything else.
func ParseKey(presented string) (id, secret string, ok bool) {
	if len(presented) != 3+idLen+1+secretLen || !strings.HasPrefix(presented, "tl_") || presented[3+idLen] != '_' {
		return "", "", false
	}
	id, secret = presented[3:3+idLen], presented[3+idLen+1:]
	for i := 0; i < len(id); i++ {
		if !strings.ContainsRune(idAlphabet, rune(id[i])) {
			return "", "", false
		}
	}
	for i := 0; i < len(secret); i++ {
		if !strings.ContainsRune(secretAlphabet, rune(secret[i])) {
			return "", "", false
		}
	}
	return id, secret, true
}

// Digest is the stored form of a secret: hex HMAC-SHA256 keyed by the
// pepper. Without the pepper, a leaked keys file does not let anyone test
// guesses offline.
func Digest(pepper []byte, secret string) string {
	m := hmac.New(sha256.New, pepper)
	m.Write([]byte(secret))
	return hex.EncodeToString(m.Sum(nil))
}

// randomString draws n characters uniformly from alphabet (len <= 256) with
// crypto/rand, rejecting bytes past the largest multiple of len(alphabet).
func randomString(alphabet string, n int) string {
	limit := 256 - 256%len(alphabet)
	out := make([]byte, 0, n)
	buf := make([]byte, 2*n)
	for len(out) < n {
		if _, err := rand.Read(buf); err != nil {
			panic("auth: crypto/rand failed: " + err.Error())
		}
		for _, b := range buf {
			if int(b) < limit && len(out) < n {
				out = append(out, alphabet[int(b)%len(alphabet)])
			}
		}
	}
	return string(out)
}

// Store is the file-backed key store: every Create and Revoke rewrites the
// keys file atomically; Lookup caches a verified key for the cache TTL, so a
// key revoked by another process (your `keys revoke` CLI, another gateway)
// stops working within that TTL.
type Store struct {
	path   string
	pepper []byte
	clock  server.Clock
	ttl    time.Duration
	log    *slog.Logger

	mu    sync.Mutex
	recs  map[string]Record // by key id, as last read or written
	cache map[string]cached // by key id
}

type cached struct {
	digest string
	p      Principal
	until  time.Time
}

// Options are the optional parts of NewStore.
type Options struct {
	Clock    server.Clock  // nil = server.WallClock
	CacheTTL time.Duration // key_cache_ttl_s; 0 = no cache
	Logger   *slog.Logger  // nil = slog.Default()
}

// NewStore opens (or creates) the keys file at path. An empty pepper is an
// error: the gateway refuses to start rather than store bare hashes.
func NewStore(path string, pepper []byte, o Options) (*Store, error) {
	if len(pepper) == 0 {
		return nil, errors.New("auth: the pepper is empty (set the variable named by gateway.pepper_env)")
	}
	if o.Clock == nil {
		o.Clock = server.WallClock
	}
	if o.Logger == nil {
		o.Logger = slog.Default()
	}
	s := &Store{path: path, pepper: append([]byte(nil), pepper...), clock: o.Clock, ttl: o.CacheTTL,
		log: o.Logger, cache: map[string]cached{}}
	if err := s.reload(); err != nil {
		return nil, err
	}
	return s, nil
}

// reload reads the keys file; a missing file is an empty store. Later lines
// win over earlier lines with the same key id.
func (s *Store) reload() error {
	recs := map[string]Record{}
	f, err := os.Open(s.path)
	if errors.Is(err, os.ErrNotExist) {
		s.recs = recs
		return nil
	}
	if err != nil {
		return fmt.Errorf("auth: %w", err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64<<10), 1<<20)
	for n := 1; sc.Scan(); n++ {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var r Record
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			return fmt.Errorf("auth: %s line %d: %v", s.path, n, err)
		}
		recs[r.KeyID] = r
	}
	if err := sc.Err(); err != nil {
		return fmt.Errorf("auth: %w", err)
	}
	s.recs = recs
	return nil
}

// save writes every record to a temporary file and renames it over the keys
// file, so a reader never sees half a file.
func (s *Store) save() error {
	tmp, err := os.CreateTemp(filepath.Dir(s.path), ".keys-*.jsonl")
	if err != nil {
		return fmt.Errorf("auth: %w", err)
	}
	w := bufio.NewWriter(tmp)
	ids := make([]string, 0, len(s.recs))
	for id := range s.recs {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		b, _ := json.Marshal(s.recs[id])
		w.Write(b)
		w.WriteByte('\n')
	}
	if err := w.Flush(); err != nil {
		tmp.Close()
		os.Remove(tmp.Name())
		return fmt.Errorf("auth: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		os.Remove(tmp.Name())
		return fmt.Errorf("auth: %w", err)
	}
	tmp.Close()
	if err := os.Rename(tmp.Name(), s.path); err != nil {
		os.Remove(tmp.Name())
		return fmt.Errorf("auth: %w", err)
	}
	return nil
}

// Create makes a key, stores only its HMAC, and returns the plaintext key
// (shown once) and the stored record.
func (s *Store) Create(ctx context.Context, spec KeySpec) (plain string, rec Record, err error) {
	if spec.Tenant == "" || len(spec.Scopes) == 0 {
		return "", Record{}, errors.New("auth: a key needs a tenant and at least one scope")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.reload(); err != nil {
		return "", Record{}, err
	}
	id := randomString(idAlphabet, idLen)
	for _, taken := s.recs[id]; taken; _, taken = s.recs[id] {
		id = randomString(idAlphabet, idLen)
	}
	secret := randomString(secretAlphabet, secretLen)
	rec = Record{
		KeyID: id, HMAC: Digest(s.pepper, secret), Tenant: spec.Tenant, Name: spec.Name,
		Scopes: append([]string(nil), spec.Scopes...), RPM: spec.RPM, TPM: spec.TPM,
		Priority: spec.Priority, Models: append([]string{}, spec.Models...),
		CreatedAt: s.clock.Now().UTC(), ExpiresAt: spec.ExpiresAt,
	}
	s.recs[id] = rec
	if err := s.save(); err != nil {
		delete(s.recs, id)
		return "", Record{}, err
	}
	s.log.Info("api key created", "key_id", id, "tenant", spec.Tenant)
	return "tl_" + id + "_" + secret, rec, nil
}

// Revoke marks a key revoked; this store rejects it at once, other stores on
// the same file within their cache TTL.
func (s *Store) Revoke(ctx context.Context, keyID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.reload(); err != nil {
		return err
	}
	r, ok := s.recs[keyID]
	if !ok {
		return ErrNotFound
	}
	r.Revoked = true
	s.recs[keyID] = r
	delete(s.cache, keyID)
	if err := s.save(); err != nil {
		return err
	}
	s.log.Info("api key revoked", "key_id", keyID)
	return nil
}

// List returns every record (KeyInfo without the HMAC is the admin view).
func (s *Store) List(ctx context.Context) ([]Record, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.reload(); err != nil {
		return nil, err
	}
	ids := make([]string, 0, len(s.recs))
	for id := range s.recs {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	out := make([]Record, 0, len(ids))
	for _, id := range ids {
		out = append(out, s.recs[id])
	}
	return out, nil
}

// Lookup verifies a presented key: parse it, find the record by id, compare
// HMAC-SHA256(pepper, secret) with the stored one in constant time, and
// reject revoked and expired keys. Every failure is ErrInvalidKey, so the
// answer does not reveal which part was wrong. A verified key is cached for
// the TTL; the cache holds the digest, so a wrong secret never hits it.
func (s *Store) Lookup(ctx context.Context, presented string) (Principal, error) {
	id, secret, ok := ParseKey(presented)
	if !ok {
		return Principal{}, ErrInvalidKey
	}
	digest := Digest(s.pepper, secret)
	now := s.clock.Now()
	s.mu.Lock()
	defer s.mu.Unlock()
	if c, ok := s.cache[id]; ok && now.Before(c.until) {
		if hmac.Equal([]byte(c.digest), []byte(digest)) {
			return c.p, nil
		}
		return Principal{}, ErrInvalidKey
	}
	delete(s.cache, id)
	if err := s.reload(); err != nil {
		return Principal{}, err
	}
	r, ok := s.recs[id]
	if !ok || !hmac.Equal([]byte(r.HMAC), []byte(digest)) {
		return Principal{}, ErrInvalidKey
	}
	if r.Revoked || (r.ExpiresAt != nil && !now.Before(*r.ExpiresAt)) {
		return Principal{}, ErrInvalidKey
	}
	p := r.Principal()
	if s.ttl > 0 {
		s.cache[id] = cached{digest: digest, p: p, until: now.Add(s.ttl)}
	}
	return p, nil
}

// RequiredScope is the scope a path needs: /v1/embeddings needs embed,
// /admin/v1/... admin, every other path (chat, completions, models) infer.
func RequiredScope(route string) string {
	switch {
	case strings.HasPrefix(route, "/admin/"):
		return ScopeAdmin
	case route == "/v1/embeddings":
		return ScopeEmbed
	default:
		return ScopeInfer
	}
}

// Authorize decides whether p may call route for model ("" when the request
// names none, as GET /v1/models). It returns ErrInsufficient when p lacks the
// route's scope, or when p has a model allowlist that does not contain model.
func Authorize(p Principal, route, model string) error {
	if !p.HasScope(RequiredScope(route)) {
		return ErrInsufficient
	}
	if model != "" && len(p.Models) > 0 {
		for _, m := range p.Models {
			if m == model {
				return nil
			}
		}
		return ErrInsufficient
	}
	return nil
}

type principalKey struct{}

// WithPrincipal returns ctx carrying p.
func WithPrincipal(ctx context.Context, p Principal) context.Context {
	return context.WithValue(ctx, principalKey{}, p)
}

// PrincipalFrom returns the authenticated principal; ok is false before the
// authn stage (or without it).
func PrincipalFrom(ctx context.Context) (Principal, bool) {
	p, ok := ctx.Value(principalKey{}).(Principal)
	return p, ok
}

// Authenticator is what the middleware needs from a key store.
type Authenticator interface {
	Lookup(ctx context.Context, presented string) (Principal, error)
}

// bearer returns the token of "Authorization: Bearer <token>" (scheme case
// ignored), or "".
func bearer(h string) string {
	const scheme = "bearer "
	if len(h) <= len(scheme) || !strings.EqualFold(h[:len(scheme)], scheme) {
		return ""
	}
	return strings.TrimSpace(h[len(scheme):])
}

// Middleware is the authn stage: 401 invalid_api_key for a missing or bad
// key, 400 for an unreadable body, 403 insufficient_scope from Authorize.
// On success it removes Authorization (the engine tier has no auth), sets
// X-TL-Priority from the key, records tl.api_key_id on the Exchange, and
// passes the Principal on in the context. It logs key ids, never keys.
func Middleware(store Authenticator, logger *slog.Logger) server.Middleware {
	if logger == nil {
		logger = slog.Default()
	}
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			ex := server.ExchangeFrom(r.Context())
			token := bearer(r.Header.Get("Authorization"))
			p, err := store.Lookup(r.Context(), token)
			if token == "" || err != nil {
				id, _, _ := ParseKey(token)
				logger.Warn("authentication failed", "request_id", ex.RequestID, "key_id", id)
				server.WriteError(w, http.StatusUnauthorized, "invalid_request_error", "invalid_api_key", "",
					"Incorrect or missing API key: send the header Authorization: Bearer tl_<id>_<secret>.")
				return
			}
			model := ""
			if r.Method == http.MethodPost {
				req, err := ex.Request(r)
				if err != nil {
					server.WriteError(w, http.StatusBadRequest, "invalid_request_error", "", "", err.Error())
					return
				}
				model = req.Model
			}
			if err := Authorize(p, r.URL.Path, model); err != nil {
				logger.Warn("authorization failed", "request_id", ex.RequestID, "key_id", p.KeyID, "route", r.URL.Path, "model", model)
				server.WriteError(w, http.StatusForbidden, "permission_error", "insufficient_scope", "",
					fmt.Sprintf("key %s may not call %s for model %q", p.KeyID, r.URL.Path, model))
				return
			}
			r.Header.Del("Authorization")
			r.Header.Set("X-TL-Priority", strconv.Itoa(p.Priority))
			ex.SetAttr("tl.api_key_id", p.KeyID)
			next.ServeHTTP(w, r.WithContext(WithPrincipal(r.Context(), p)))
		})
	}
}
