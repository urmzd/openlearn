# API Keys, Scopes, Model Allowlists, and Tenants

So far the gateway has only the tracer's single static key from [the tracer gateway](../12-gateway/01-streaming-proxy.md): one string in `TL_API_KEY` that every client shares. That is enough to prove the path works and no more. You cannot tell two clients apart, so you cannot give them separate rate limits, keep their cached answers apart, or bill them. You cannot revoke one client without breaking all of them. And anyone who reads the gateway's environment or logs holds the key.

This section replaces it with per-client keys, stored safely, scoped to what each client may do, and attached to a tenant. That identity, the **principal**, is what every later stage of the gateway uses to decide who the request belongs to. You will design a key format that makes lookup O(1) while keeping the stored form useless to a thief, cache verified keys without letting the cache become a bypass, and draw an exact line between authentication (401) and authorization (403).

This section builds on [The server skeleton](../12-gateway/02-server-skeleton.md), which supplies the `Middleware` type, the `Exchange`, and `WriteError`. The full solution is [`go/gateway/auth/auth.go`](../../../../course/solutions/go/gateway/auth/auth.go) (`ParseKey`, `Digest`, `Store` with `NewStore`, `Create`, `Revoke`, `List`, `Lookup`, plus `Authorize`, `RequiredScope`, `Principal`, `WithPrincipal`, `PrincipalFrom`, `Middleware`) and [`go/gateway/auth/admin.go`](../../../../course/solutions/go/gateway/auth/admin.go) (`KeysHandler` for the admin API's keys paths). The contract is the 401 `invalid_api_key` and 403 `insufficient_scope` errors in [`openai-subset.v1.yaml`](../../../../course/contracts/openapi/openai-subset.v1.yaml); `KeyCreate`, `KeyInfo`, `KeyCreated`, and the keys rules in [`admin.v1.yaml`](../../../../course/contracts/openapi/admin.v1.yaml); and the `[gateway]` settings `keys_file`, `pepper_env`, and `key_cache_ttl_s` in [`runtime.schema.json`](../../../../course/contracts/config/runtime.schema.json).

In the finished system, the gateway's composition root ([`entry/go/cmd/gateway/compose.go`](../../../../course/solutions/entry/go/cmd/gateway/compose.go)) opens the keys file with `auth.NewStore` (the pepper comes from the variable named by `pepper_env`) and plugs `auth.Middleware` into the `Keys` slot of the chain. Keys are minted through `POST /admin/v1/keys`, which returns the plaintext key once.

> **Learning objectives**
> - Design a two-part key, `tl_<id>_<secret>`, that is found in O(1) by its public id and verified against a peppered HMAC of its secret.
> - Generate unbiased random secrets with rejection sampling.
> - Make every authentication failure indistinguishable and every comparison constant-time.
> - Cache verified keys with a bounded TTL so that revocation still takes effect, without letting a cached id accept any secret.
> - Implement authorization as a pure table of route scopes and model allowlists, answering 403 rather than 401.

## The key format

A key has four ingredients:

| Symbol | Meaning | Type |
|---|---|---|
| $\mathit{id}$ | public key id: 12 characters of `[a-z2-7]` (lower-case base32) | string |
| $s$ | secret: 32 characters of `[A-Za-z0-9]` | string |
| $P$ | the pepper: a server-side secret from the variable named by `pepper_env` | bytes |
| $d$ | the stored digest $\mathrm{HMAC\text{-}SHA256}(P, s)$, hex | 64 hex digits |

A presented key is `tl_` + $\mathit{id}$ + `_` + $s$, exactly 48 characters, and anything else is rejected before any lookup. The id is not secret: it appears in logs, the ledger, and the admin API. The secret carries all the entropy, $32 \log_2 62 \approx 190.5$ bits. It is generated with `crypto/rand` by **rejection sampling**: a random byte $b$ is used only when $b < 248 = 4 \cdot 62$, then reduced mod 62. Taking `b % 62` over all 256 values would make the first 8 characters more likely than the rest.

## Storing a digest, not the key

The keys file (`[gateway].keys_file`, JSON Lines) holds one record per key: `key_id`, `hmac` ($d$), `tenant`, `name`, `scopes`, `rpm`, `tpm`, `priority`, `models`, `created_at`, `expires_at`, `revoked`. It never holds $s$. To verify a presented key, the gateway recomputes $d' = \mathrm{HMAC}(P, s)$ and compares $d'$ with the stored $d$.

Why HMAC with a pepper rather than a plain hash? A plain $\mathrm{SHA256}(s)$ is enough to stop anyone reversing a 190-bit secret, but the file is still a list of targets that anyone who copies it can test guesses against forever. Keying the hash with $P$, which lives only in the gateway's environment (a Kubernetes `Secret`), makes the file alone useless. HMAC is the standard way to key a hash. For a key $K$ padded to the block size,

$$\mathrm{HMAC}(K, m) = H((K \oplus \mathit{opad}) \,\|\, H((K \oplus \mathit{ipad}) \,\|\, m)),$$

which, unlike $H(K \| m)$, is not open to length extension.

The comparison uses `hmac.Equal`, which takes the same time whatever the contents; the argument about early-exit comparisons from the tracer gateway applies to digests too. And every failure returns the same `ErrInvalidKey`.

## The lookup cache and revocation

Every request needs a lookup, and re-reading the file each time is slow. So the store caches a verified key for `key_cache_ttl_s` seconds (at most 5, per the schema). The cache entry is keyed by id but holds the digest, so a cached id presented with a wrong secret still fails.

Revocation (`Revoke`, or `DELETE /admin/v1/keys/{id}`) rewrites the file, to a temporary file and then `rename`, so a reader never sees half a file, and drops the revoking store's own cache entry at once. Another process on the same file, such as a second gateway replica or an operator's tool revoking keys through the file, still holds its cached entry. It rejects the key once that entry is $\ge$ TTL old, when it re-reads the file. That bound, "within the key-cache TTL", is the contract. Expiry uses the same clock: a key is valid while $\mathit{now} < \mathit{expires\_at}$.

## Authorization

Once the caller is known, the question becomes what they may do. Each route needs a scope:

| Route | Scope required |
|---|---|
| `/v1/chat/completions`, `/v1/completions`, `/v1/models` | `infer` |
| `/v1/embeddings` | `embed` |
| `/admin/v1/...` | `admin` |

Then comes the model allowlist. A key with a non-empty `models` list may use only those models; an **empty** list means all models. Requests that name no model (`GET /v1/models`) pass the allowlist.

A failure here is **403** `permission_error` / `insufficient_scope`. The caller is known but not allowed, which is different from a 401 (unknown caller), and clients treat the two differently: a 401 means "fix your key", a 403 "ask for access". The `debug` scope grants nothing on its own; it unlocks the `X-TL-Route` response header in [routing](../12-gateway/04-routing.md).

The **tenant** is the unit that owns keys: one company or team. Later stages scope by it (the cache never shares answers across tenants) or by the key (rate limits).

## Working it by hand

Suppose the gateway's pepper is `pepper-demo`, and a key was created for tenant `acme` with scopes `[infer]`, `rpm` 60, `priority` 5, and allowlist `[smol-135m]`. The client presents

```
Authorization: Bearer tl_demo2key7abc_S3cretS3cretS3cretS3cretS3cret01
```

1. Parse: 48 characters, prefix `tl_`, `_` at position 15. So $\mathit{id}$ = `demo2key7abc` (12 characters, all in `[a-z2-7]`) and $s$ = `S3cretS3cretS3cretS3cretS3cret01` (32, all in `[A-Za-z0-9]`).
2. Find the record by id: one map lookup.
3. Compute $d' = \mathrm{HMAC\text{-}SHA256}(\texttt{pepper-demo}, s)$:

   ```
   94f4a20cad218d9e113440c90096b20064d78a02a3cb6f72210222c4d6729506
   ```

   You can check it with `python3 -c "import hmac,hashlib;print(hmac.new(b'pepper-demo',b'S3cretS3cretS3cretS3cretS3cret01',hashlib.sha256).hexdigest())"`. It equals the record's `hmac`, so the key is valid. The unpeppered `SHA256(s)` would be `118a232e...712e64`, which must **not** verify.
4. The request is `POST /v1/chat/completions` with `"model": "smol-135m"`. The route needs `infer`, which the key holds, and the model is on the allowlist, so it is authorized.
5. The stage passes on `Principal{KeyID: "demo2key7abc", Tenant: "acme", Scopes: [infer], RPM: 60, Priority: 5, ...}`, deletes `Authorization`, and sets `X-TL-Priority: 5`. This overwrites nothing, because the server already stripped the client's own `X-TL-*` headers.

The same key on `/v1/embeddings` is a 403 (no `embed` scope), and with `"model": "tinystories-10m"` it is also a 403 (not on the allowlist).

## Designing the package

The API separates the pure pieces (parsing, digesting, authorizing) from the stateful store and the HTTP glue:

```go
package auth // import "tinyllm/gateway/auth"

type Principal struct {
	KeyID, Tenant, Name string
	Scopes, Models      []string // Models empty = all
	RPM, TPM, Priority  int
}
func (p Principal) HasScope(s string) bool

func ParseKey(presented string) (id, secret string, ok bool)
func Digest(pepper []byte, secret string) string // hex HMAC-SHA256

type Options struct { Clock server.Clock; CacheTTL time.Duration; Logger *slog.Logger }
func NewStore(path string, pepper []byte, o Options) (*Store, error) // empty pepper: error
func (s *Store) Create(ctx context.Context, spec KeySpec) (plain string, rec Record, err error)
func (s *Store) Revoke(ctx context.Context, keyID string) error // ErrNotFound
func (s *Store) List(ctx context.Context) ([]Record, error)
func (s *Store) Lookup(ctx context.Context, presented string) (Principal, error) // ErrInvalidKey

func RequiredScope(route string) string
func Authorize(p Principal, route, model string) error // ErrInsufficient
func WithPrincipal(ctx context.Context, p Principal) context.Context
func PrincipalFrom(ctx context.Context) (Principal, bool)
func Middleware(store Authenticator, logger *slog.Logger) server.Middleware
func KeysHandler(s *Store) http.Handler // POST, GET /admin/v1/keys; DELETE /admin/v1/keys/{key_id}
```

The catalog sketched `Create(ctx, tenant, scopes)`. Here `Create` takes a `KeySpec`, because the admin API's `KeyCreate` also sets `rpm`, `tpm`, `priority`, `models`, and `expires_at`. Logging uses `log/slog`, and every line names the key id, never the presented key.

## Building it step by step

Seven requirements, taken in order, produce the whole design:

1. **A request must be tied to one client, and checking it must be cheap.** Every request passes this stage, so verification must be O(1): the presented key has to say which record to check. That forces a two-part key, a public id that indexes a map and a secret that proves possession.
2. **The stored form must be useless on its own.** If the file held the secret, a copy of the file would be a copy of every key. Storing a hash fixes reversal, but a copied file still allows offline guessing. Keying the hash with a pepper that never touches the file removes that too, and HMAC is the standard keyed hash. So the record holds $d = \mathrm{HMAC}(P, s)$ and nothing else secret.
3. **The format is fixed, so parse strictly before doing any work.** A 48-character, fixed-alphabet format can be checked with a length test and two character loops. Anything that fails never reaches the map or the HMAC, which keeps garbage and oversized input out of the hot path.
4. **Failures must be indistinguishable.** Malformed, unknown id, wrong secret, revoked, expired: if any of these answered differently, in status, message, or timing of the comparison, an attacker could enumerate ids. So there is one error value, `ErrInvalidKey`, and `hmac.Equal` for the digest comparison.
5. **Re-reading the file per request is too slow, but revocation must still bite.** A short TTL cache bounds staleness: a revoked key works at most TTL longer in other processes. The cache must hold the digest, because a cache keyed by id alone would accept any secret for a recently seen id.
6. **Authentication and authorization are different answers.** Who you are (401) is decided by the key; what you may do (403) is a pure function of the principal, the route, and the model. Keeping `Authorize` pure makes it a table you can test exhaustively and keeps the 401/403 split exact.
7. **What leaves the stage is the principal, not the key.** Later stages need the tenant, the key id, and the limits; the engines need a priority and nothing secret. So the stage strips `Authorization`, sets `X-TL-Priority`, and puts a `Principal` in the context.

### Parsing

`ParseKey` checks length and separator first, then each half against its alphabet. Every exit is the same `ok == false`:

```go
func ParseKey(presented string) (id, secret string, ok bool) {
	// 3 ("tl_") + 12 (id) + 1 ("_") + 32 (secret) = 48: one comparison
	// rejects every wrong length, and the separator must sit at index 15.
	if len(presented) != 3+idLen+1+secretLen || !strings.HasPrefix(presented, "tl_") || presented[3+idLen] != '_' {
		return "", "", false
	}
	id, secret = presented[3:3+idLen], presented[3+idLen+1:]
	for i := 0; i < len(id); i++ { // id: lower-case base32, so no '1', no upper case
		if !strings.ContainsRune(idAlphabet, rune(id[i])) {
			return "", "", false
		}
	}
	for i := 0; i < len(secret); i++ { // secret: [A-Za-z0-9] only
		if !strings.ContainsRune(secretAlphabet, rune(secret[i])) {
			return "", "", false
		}
	}
	return id, secret, true
}
```

### The digest and the random strings

`Digest` is the whole storage scheme in four lines. `randomString` is the rejection sampler, written for any alphabet of up to 256 characters:

```go
func Digest(pepper []byte, secret string) string {
	m := hmac.New(sha256.New, pepper) // keyed by the pepper, over the secret only (not the whole key)
	m.Write([]byte(secret))
	return hex.EncodeToString(m.Sum(nil))
}

func randomString(alphabet string, n int) string {
	limit := 256 - 256%len(alphabet) // 248 for 62 characters, 256 for 32 (no rejection)
	out := make([]byte, 0, n)
	buf := make([]byte, 2*n)
	for len(out) < n {
		if _, err := rand.Read(buf); err != nil {
			panic("auth: crypto/rand failed: " + err.Error()) // never fall back to a weak source
		}
		for _, b := range buf {
			// Bytes >= limit would make the first 256%len characters likelier;
			// dropping them keeps every character exactly equally likely.
			if int(b) < limit && len(out) < n {
				out = append(out, alphabet[int(b)%len(alphabet)])
			}
		}
	}
	return string(out)
}
```

### The store refuses to run without a pepper

`NewStore` fails on an empty pepper rather than silently storing bare hashes; the composition root then logs that `keys_file` is not opened.

```go
func NewStore(path string, pepper []byte, o Options) (*Store, error) {
	if len(pepper) == 0 {
		return nil, errors.New("auth: the pepper is empty (set the variable named by gateway.pepper_env)")
	}
	// ... default the clock and logger, copy the pepper, read the file
}
```

### Writes are atomic

`save` writes every record to a temporary file in the same directory, `fsync`s it, and renames it over the keys file. `rename` within one file system is atomic, so a concurrent reader (another replica, or this store's next `reload`) sees either the old file or the new one, never half of each. `Create` and `Revoke` both `reload` first under the mutex, so they edit the current file rather than a stale copy:

```go
func (s *Store) Create(ctx context.Context, spec KeySpec) (plain string, rec Record, err error) {
	if spec.Tenant == "" || len(spec.Scopes) == 0 {
		return "", Record{}, errors.New("auth: a key needs a tenant and at least one scope")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.reload(); err != nil { // another process may have added keys
		return "", Record{}, err
	}
	id := randomString(idAlphabet, idLen)
	for _, taken := s.recs[id]; taken; _, taken = s.recs[id] { // ids are unique in the file
		id = randomString(idAlphabet, idLen)
	}
	secret := randomString(secretAlphabet, secretLen)
	rec = Record{
		KeyID: id, HMAC: Digest(s.pepper, secret), // the only trace of the secret
		// ... tenant, name, copies of scopes and models, limits, created_at, expires_at
	}
	s.recs[id] = rec
	if err := s.save(); err != nil {
		delete(s.recs, id) // memory never claims a key the file does not hold
		return "", Record{}, err
	}
	s.log.Info("api key created", "key_id", id, "tenant", spec.Tenant)
	return "tl_" + id + "_" + secret, rec, nil // the plaintext exists only in this return value
}

func (s *Store) Revoke(ctx context.Context, keyID string) error {
	// ... lock, reload, ErrNotFound for an unknown id
	r.Revoked = true
	s.recs[keyID] = r
	delete(s.cache, keyID) // this store rejects the key at once
	// ... save, log the id
}
```

### Lookup

In `Lookup`, the order of operations is the security argument: parse, hash, then consult the cache, and accept a cache hit only if its digest matches.

```go
func (s *Store) Lookup(ctx context.Context, presented string) (Principal, error) {
	id, secret, ok := ParseKey(presented)
	if !ok {
		return Principal{}, ErrInvalidKey
	}
	digest := Digest(s.pepper, secret) // computed before the lock: HMAC is the costly part
	now := s.clock.Now()
	s.mu.Lock()
	defer s.mu.Unlock()
	if c, ok := s.cache[id]; ok && now.Before(c.until) {
		// A fresh cache hit still has to prove the secret.
		if hmac.Equal([]byte(c.digest), []byte(digest)) {
			return c.p, nil
		}
		return Principal{}, ErrInvalidKey
	}
	delete(s.cache, id)                   // stale or absent: go back to the file
	if err := s.reload(); err != nil {    // picks up revocations by other processes
		return Principal{}, err
	}
	r, ok := s.recs[id]
	if !ok || !hmac.Equal([]byte(r.HMAC), []byte(digest)) { // unknown id and wrong secret: same answer
		return Principal{}, ErrInvalidKey
	}
	// Valid while now < expires_at, so a key is already dead at exactly expires_at.
	if r.Revoked || (r.ExpiresAt != nil && !now.Before(*r.ExpiresAt)) {
		return Principal{}, ErrInvalidKey
	}
	p := r.Principal()
	if s.ttl > 0 {
		s.cache[id] = cached{digest: digest, p: p, until: now.Add(s.ttl)}
	}
	return p, nil
}
```

### Authorization is a pure table

The prefix check on `/admin/` comes first; every other `/v1` path needs `infer` except embeddings. An empty allowlist and an empty model both mean that no model restriction applies:

```go
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

func Authorize(p Principal, route, model string) error {
	if !p.HasScope(RequiredScope(route)) {
		return ErrInsufficient
	}
	if model != "" && len(p.Models) > 0 { // empty allowlist = all models
		for _, m := range p.Models {
			if m == model {
				return nil
			}
		}
		return ErrInsufficient
	}
	return nil
}
```

### The middleware

The middleware answers 401 before reading the body, 400 for an unreadable body, and 403 from `Authorize`. Only then is the request changed and passed on:

```go
return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	ex := server.ExchangeFrom(r.Context())
	token := bearer(r.Header.Get("Authorization")) // "Bearer " scheme, case-insensitive
	p, err := store.Lookup(r.Context(), token)
	if token == "" || err != nil {
		id, _, _ := ParseKey(token) // "" unless the key is well formed: never log the token itself
		logger.Warn("authentication failed", "request_id", ex.RequestID, "key_id", id)
		server.WriteError(w, http.StatusUnauthorized, "invalid_request_error", "invalid_api_key", "",
			"Incorrect or missing API key: send the header Authorization: Bearer tl_<id>_<secret>.")
		return
	}
	model := ""
	if r.Method == http.MethodPost {
		req, err := ex.Request(r) // parsed once by the server skeleton's Exchange, reused by later stages
		if err != nil {
			server.WriteError(w, http.StatusBadRequest, "invalid_request_error", "", "", err.Error())
			return
		}
		model = req.Model
	}
	if err := Authorize(p, r.URL.Path, model); err != nil {
		// ... log key id, route, model; 403 permission_error / insufficient_scope
		return
	}
	r.Header.Del("Authorization")                            // the engine tier never sees keys
	r.Header.Set("X-TL-Priority", strconv.Itoa(p.Priority)) // the engine scheduler's input
	ex.SetAttr("tl.api_key_id", p.KeyID)                     // for traces and the ledger
	next.ServeHTTP(w, r.WithContext(WithPrincipal(r.Context(), p)))
})
```

### The admin handler

`KeysHandler` in `admin.go` is the admin view of the same store. `POST` validates the tenant against `^[a-z0-9][a-z0-9-]{0,62}$`, requires a name and known scopes, defaults `rpm` to 60 and `tpm` to 100000 when absent, and returns 201 with the plaintext key next to the `KeyInfo`. `GET` lists `KeyInfo` values, newest first, optionally filtered by tenant. `DELETE /admin/v1/keys/{key_id}` revokes the key (204, or 404 for an unknown id). `KeyInfo` has no `hmac` field at all, so no listing can leak a digest.

The full code is in [`auth.go`](../../../../course/solutions/go/gateway/auth/auth.go) and [`admin.go`](../../../../course/solutions/go/gateway/auth/admin.go).

### Why this design

`Lookup` costs one HMAC-SHA256 over 32 bytes plus one map lookup: microseconds. A cache miss adds a file read, bounded by one per key per TTL. `Create` and `Revoke` rewrite the whole file, O(number of keys), but they are rare admin operations, which is the right place to pay.

Several alternatives were rejected deliberately. **bcrypt or Argon2?** Slow password hashes exist because passwords have little entropy. A 190-bit random secret cannot be brute-forced even with a fast hash, and a slow hash on every request would cost milliseconds of CPU per call; the pepper handles the stolen-file case that slowness would otherwise address. **Opaque random tokens looked up by hash?** That works too (hash the whole token and index by the hash), but then the identifier you log and show in the admin API is a hash, and you cannot tell from a leaked token which record it is without the pepper. The public id makes logs, ledgers, and revocation readable. **JWTs?** Self-contained tokens need no lookup, but they cannot be revoked before they expire without a deny list, which is a lookup again, and revocation within 5 s is the requirement here. **A database instead of a file?** One small JSON Lines file, rewritten atomically, is enough for hundreds of keys, needs no extra service, and is easy to inspect. The accepted tradeoff is that concurrent writers in different processes can lose an update (each holds its own mutex), and large key counts make every write O(n). A production gateway moves this behind a key service, as discussed at the end of this section.

## What can go wrong

A correct solution must pass these checks:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand example | unit | the digest, the lookup, the principal; an unpeppered record fails | the file format every tool and the gateway share |
| parsing | boundary | the hand key parses; 9 malformed keys (case, `1`, separators, length, punctuation) do not | garbage never reaches a lookup |
| create then look up | unit | `Create` returns `tl_<id>_<secret>`; the file holds only the HMAC; another store on the file sees it; no scopes and empty pepper refused | minting keys through the admin API |
| every rejection | boundary | wrong last character, unknown id, revoked, malformed: `ErrInvalidKey`; expiry exactly at `expires_at` (fake clock) | one 401 for every failure |
| the cache checks the secret | unit | a cached id with a wrong secret is still rejected | the cache cannot become a bypass |
| revocation within the TTL | fault | the revoking store rejects at once; another store within 5 s; `ErrNotFound`; `List` | revocation is the emergency brake |
| the authorization table | unit | the route-scope table and the allowlist, 13 cases | 403 decisions |
| the middleware's 401 and 403 | unit, conformance | no key, bad key: 401; missing scope, model not allowed: 403; a good key reaches the proxy with its principal, no `Authorization`, `X-TL-Priority: 5`; a bad body after a good key: 400 | the contract's `auth.401`, `auth.403`, `priority.internal` behaviour |
| no plaintext key in logs | property | after good, wrong, and malformed keys, creates and revokes, the logs name key ids and contain no secret or pepper | logs are not a leak |
| the keys admin handler | unit, conformance | `POST` 201 with defaults and no HMAC; bad tenants and scopes 400; `GET` lists without HMACs; `DELETE` 204 then the key fails; unknown id 404 | the admin API mounts it at `/admin/v1/keys` |

> **Warning:** Storing too much, or hashing the wrong thing, defeats the scheme. The secret in the file, no pepper, an HMAC over the whole key instead of the secret, a lax parser, or the HMAC in admin output each mean that a copied keys file or admin listing is enough to use or attack keys, or that keys never verify. `Record` holds only `Digest(pepper, secret)` of the secret half, `NewStore` refuses an empty pepper, `ParseKey` checks length, separator, and both alphabets, and `KeyInfo` has no HMAC field.

Revocation and expiry that do not bite are the next family: a revoked flag that is not checked, a cache that never expires, or an ignored `expires_at` lets a leaked key keep working after revocation. `Lookup` checks `Revoked` and `now < expires_at` on every file read, cache entries carry `until`, and `Revoke` drops its own entry. A cache keyed by id alone lets any secret work for a recently used id; cache hits compare the cached digest with `hmac.Equal`.

Authorization slips are subtler. Ignoring the allowlist, reading an empty allowlist as "nothing allowed", filing embeddings under `infer`, or answering 401 for a scope failure lets keys use models they were not granted, or makes nothing work. `Authorize` is a pure table: `RequiredScope` maps embeddings to `embed`, `len(p.Models) > 0` guards the allowlist, and the middleware answers 403 for `ErrInsufficient`.

Finally, two leaks. Logging the presented key on failure puts every typo of a real key into your log store; the failure log names only the parsed id (empty for a malformed key), never the token. Forwarding `Authorization`, or failing to set `X-TL-Priority` from the key, puts the key into engine logs and makes the engine scheduler from [Continuous batching scheduler with priority and preemption](../../build-an-llm/11-serving/03-continuous-batching.md) treat every tenant alike; the middleware deletes `Authorization` and sets `X-TL-Priority` before calling `next`.

## Where this leads

This stage is a `server.Middleware` from the [server skeleton](../12-gateway/02-server-skeleton.md); it reads the model with `ex.Request` and answers with `WriteError`. Every later gateway stage builds on the principal it produces. [Rate limiting](02-rate-limiting.md) keys its buckets by `Principal.KeyID` with the key's `RPM` and `TPM`. [Routing](../12-gateway/04-routing.md) shows `X-TL-Route` only to principals with the `debug` scope. The [response cache](../04-distributed-data-and-caching/01-response-cache.md) includes `Principal.Tenant` in its keys. The [usage ledger and admin API](../12-gateway/05-usage-ledger-and-admin-api.md) records tenant and key id in each row, checks the `admin` scope, and mounts `KeysHandler`. And [usage policy enforcement](../12-gateway/06-usage-policy-enforcement.md) reads the `Principal` and refuses requests that arrive without one. Together these stages make up the full gateway. For a security reviewer's view of this code, see [Secrets and authorization review](../../software-craftsmanship/08-security/03-secrets-and-authorization-review.md) in *Software Craftsmanship*.

Each piece has a production counterpart:

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `tl_<id>_<secret>` keys | Stripe and GitHub token formats | prefixes that secret scanners recognize, checksums that catch typos offline | [GitHub's token format](https://github.blog/engineering/platform-security/behind-githubs-new-authentication-token-formats/) (free) |
| the keys file | Envoy external authorization, Vault | a separate auth service, short-lived credentials, audit devices | [Envoy ext_authz](https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/ext_authz_filter) (free) |
| scopes and allowlists | LiteLLM virtual keys, OpenAI projects | per-key budgets and model access, team hierarchies | [LiteLLM virtual keys](https://docs.litellm.ai/docs/proxy/virtual_keys) (free) |
| `Authorize` | OPA, Cedar | policy as code, evaluated and tested apart from the service | [Cedar](https://www.cedarpolicy.com/) (free) |

### Further reading

- RFC 2104, HMAC (free).
- OWASP, *Authentication Cheat Sheet* and *API Security Top 10* (free).

## Key terms

- **API key**: a credential `tl_<id>_<secret>` whose public id locates a record and whose secret proves possession.
- **Pepper**: a server-side secret that keys the stored hash, kept out of the keys file.
- **HMAC**: a keyed hash construction, $H((K \oplus \mathit{opad}) \| H((K \oplus \mathit{ipad}) \| m))$, resistant to length extension.
- **Rejection sampling**: discarding random values outside a multiple of the alphabet size so every character is equally likely.
- **Principal**: the authenticated identity passed to later stages: key id, tenant, scopes, models, and limits.
- **Scope**: a named permission (`infer`, `embed`, `admin`, `debug`) that a route requires.
- **Model allowlist**: a per-key list of models it may use; empty means all.
- **Tenant**: the unit, such as a company or team, that owns keys and scopes caching and billing.

## Summary

- A key is `tl_<id>_<secret>`: the public id finds one record in O(1), and the secret is checked against `HMAC-SHA256(pepper, secret)`, the only thing stored, so a leaked keys file without the pepper does not let anyone test guesses offline.
- Every way a key can be wrong (malformed, unknown id, wrong secret, revoked, expired) is the same 401, because telling them apart tells an attacker which ids exist.
- Verified keys are cached for at most `key_cache_ttl_s` (5 s), and the cache stores the digest, not just the id; revocation by another process takes effect within that TTL.
- Authorization is a table: a route needs a scope (`infer`, `embed`, `admin`), and a key with a model allowlist may use only those models; failing it is a 403, not a 401.
- The stage passes a `Principal` on, removes `Authorization`, sets `X-TL-Priority` from the key, and logs key ids, never keys.

## Review questions

1. Why is the cache entry keyed by id but holding the digest?
   <details><summary>Answer</summary>Because the id is public. If a fresh cache hit returned the principal on the id alone, anyone who saw an id in a log could call the gateway with any secret for up to the TTL. Comparing the presented secret's digest with the cached one keeps the cache a speed-up, never a bypass.</details>
2. Why does `randomString` drop bytes $\ge 248$ for the secret alphabet but none for the id alphabet?
   <details><summary>Answer</summary>$256 \bmod 62 = 8$, so taking `b % 62` over all 256 byte values maps 5 bytes to each of the first 8 characters and 4 to each of the rest: a bias. Rejecting $b \ge 248 = 4 \cdot 62$ leaves exactly 4 byte values per character. For the 32-character id alphabet, $256 \bmod 32 = 0$, so `limit` is 256 and nothing is rejected.</details>
3. A replica revokes a key. Another replica served that key 2 s ago. When does the second replica start rejecting it, and why not sooner?
   <details><summary>Answer</summary>When its cache entry expires: 5 s after it was cached with the default TTL, so about 3 s from now. Until then it answers from cache without reading the file. Rejecting sooner would need a notification between replicas or a file read per request; the TTL is the accepted bound, and it is the contract ("within the key-cache TTL").</details>
4. Why does a missing scope return 403 and not 401, and why is the body parsed only after the key is verified?
   <details><summary>Answer</summary>A 401 says "I do not know who you are" (fix the key); a 403 says "I know you, and you may not do this" (ask for access). Mixing them sends clients to the wrong fix. Parsing after verification means an unauthenticated caller cannot make the gateway read and decode a large body: unknown callers are turned away for the cost of one HMAC.</details>
5. What would break if the stage forwarded `Authorization` to the engines?
   <details><summary>Answer</summary>The engine tier has no authentication and logs request headers freely, so every key would end up in engine logs, which are not protected like the keys file. Removing the header keeps the secret inside the one stage that needs it; the engines get only `X-TL-Priority`.</details>
