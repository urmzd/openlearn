<!-- ol:module gw.02 -->
# AuthN (API keys) and AuthZ (scopes, model allowlist, tenants)

## Overview

| | |
|---|---|
| **Module** | `gw.02` · build · Go · Pass 7 · 4 h |
| **You build** | `go/gateway/auth/auth.go`: `ParseKey`, `Digest`, `Store` (`NewStore`, `Create`, `Revoke`, `List`, `Lookup`), `Authorize`, `RequiredScope`, `Principal`, `WithPrincipal`, `PrincipalFrom`, `Middleware` · `go/gateway/auth/admin.go`: `KeysHandler` for the admin API's keys paths |
| **Contract** | 401 `invalid_api_key` and 403 `insufficient_scope` in [`course/contracts/openapi/openai-subset.v1.yaml`](../../../../course/contracts/openapi/openai-subset.v1.yaml) · `KeyCreate`, `KeyInfo`, `KeyCreated` and the keys rules in [`course/contracts/openapi/admin.v1.yaml`](../../../../course/contracts/openapi/admin.v1.yaml) · `[gateway]` `keys_file`, `pepper_env`, `key_cache_ttl_s` in [`course/contracts/config/runtime.schema.json`](../../../../course/contracts/config/runtime.schema.json) |
| **Tests** | `course/tests/go/gw_02/` (what they check: section 4) |
| **Needs** | [`gw.01` server skeleton](../12-gateway/01-server-skeleton.md) (the `Middleware` type, the `Exchange`, `WriteError`) |
| **Used by** | `gw.03` keys budgets by `Principal.KeyID` · `gw.05` shows `X-TL-Route` to debug keys · `gw.06` scopes the cache by `Principal.Tenant` · `gw.07` records tenant and key id and checks the admin scope · later: `gw.08` |
| **Milestone** | MS-gateway |
| **Optional depth** | RFC 2104 (HMAC, free); OWASP *Authentication Cheat Sheet* and *API Security Top 10* (free); `craft.19` reviews this module as a security review |

## Key Takeaways

- A key is `tl_<id>_<secret>`: the public **id** finds one record in O(1), and the **secret** is checked against `HMAC-SHA256(pepper, secret)`, the only thing stored. A leaked keys file without the pepper does not let anyone test guesses offline (`TestHandExampleLookup`).
- Every way a key can be wrong (malformed, unknown id, wrong secret, revoked, expired) is the same 401: telling them apart tells an attacker which ids exist (`TestLookupRejects`).
- Verified keys are cached for at most `key_cache_ttl_s` (5 s), and the cache stores the digest, not just the id; revocation by another process takes effect within that TTL (`TestCacheChecksTheSecret`, `TestRevokedKeyRejectedWithinTTL`).
- Authorization is a table: a route needs a scope (`infer`, `embed`, `admin`), and a key with a model allowlist may use only those models; failing it is a 403, not a 401 (`TestAuthorizeRBAC`, `TestMiddleware401And403`).
- The stage passes a `Principal` on, removes `Authorization`, sets `X-TL-Priority` from the key, and logs key ids, never keys (`TestNoPlaintextKeyInLogs`).

## How to work this chapter

```bash
ol start gw.02          # writes go/gateway/auth/{auth,admin}.go with stub bodies
ol tests gw.02          # read the test catalog first
ol check gw.02          # exit code is the verdict
ol check gw.02 --ref-deps   # only if your gw.01 is not passing yet
ol diff  gw.02          # after passing: your code against the reference
```

Then add a `keys` command to your umbrella CLI (`<system> keys create --tenant acme --scopes infer`) that opens the same keys file with `auth.NewStore` and prints the plaintext key once, and plug `auth.Middleware(store, logger)` into the `Keys` slot of your composition root.

---

## 1. Why now

Your gateway still has the tracer's single static key from `gw.00`: one string in `TL_API_KEY` that every client shares. You cannot tell two clients apart, so you cannot give them separate rate limits (`gw.03`), keep their cached answers apart (`gw.06`), or bill them (`gw.07`); you cannot revoke one client without breaking all of them; and anyone who reads the gateway's environment or logs holds the key. This module replaces it with per-client keys stored safely, scoped to what each client may do, and attached to a tenant, which every later stage uses as the identity of the request.

## 2. Principles

### 2.1 The key format

| Symbol | Meaning | Type |
|---|---|---|
| $\mathit{id}$ | public key id: 12 characters of `[a-z2-7]` (lower-case base32) | string |
| $s$ | secret: 32 characters of `[A-Za-z0-9]` | string |
| $P$ | the pepper: a server-side secret from the variable named by `pepper_env` | bytes |
| $d$ | the stored digest $\mathrm{HMAC\text{-}SHA256}(P, s)$, hex | 64 hex digits |

A presented key is `tl_` + $\mathit{id}$ + `_` + $s$, exactly 48 characters. Anything else is rejected before any lookup. The id is not secret: it appears in logs, the ledger, and the admin API. The secret carries all the entropy: $32 \log_2 62 \approx 190.5$ bits, generated with `crypto/rand` by **rejection sampling** (a random byte $b$ is used only when $b < 248 = 4 \cdot 62$, then reduced mod 62), because `b % 62` on all 256 values would make the first 8 characters more likely than the rest.

### 2.2 Storing a digest, not the key

The keys file (`[gateway].keys_file`, JSON Lines) holds one record per key: `key_id`, `hmac` ($d$), `tenant`, `name`, `scopes`, `rpm`, `tpm`, `priority`, `models`, `created_at`, `expires_at`, `revoked`. It never holds $s$. To verify a presented key, recompute $d' = \mathrm{HMAC}(P, s)$ and compare $d'$ with the stored $d$.

Why HMAC with a pepper rather than a plain hash? A plain $\mathrm{SHA256}(s)$ is enough against reversing a 190-bit secret, but the file is a list of targets anyone who copies it can test guesses against forever. Keying the hash with $P$, which lives only in the gateway's environment (a Kubernetes `Secret`), means the file alone is useless. HMAC is the standard way to key a hash: for a key $K$ padded to the block size, $\mathrm{HMAC}(K, m) = H((K \oplus \mathit{opad}) \,\|\, H((K \oplus \mathit{ipad}) \,\|\, m))$, which, unlike $H(K \| m)$, is not open to length extension.

The comparison uses `hmac.Equal`, which takes the same time whatever the contents (the `gw.00` argument about early-exit comparisons applies to digests too). All failures return the same `ErrInvalidKey`.

### 2.3 The lookup cache and revocation

Every request needs a lookup; re-reading the file each time is slow. The store caches a verified key for `key_cache_ttl_s` seconds (at most 5, per the schema). The cache entry is keyed by id but holds the digest, so a cached id presented with a wrong secret still fails. Revocation (`Revoke`, or `DELETE /admin/v1/keys/{id}`) rewrites the file (to a temporary file, then `rename`, so a reader never sees half a file) and drops the revoking store's own cache entry at once. Another process on the same file, such as a second gateway replica or your `keys revoke` CLI, still holds its cached entry: it rejects the key once that entry is $\ge$ TTL old, when it re-reads the file. That bound, "within the key-cache TTL", is the contract. Expiry uses the same clock: a key is valid while $\mathit{now} < \mathit{expires\_at}$.

### 2.4 Authorization

| Route | Scope required |
|---|---|
| `/v1/chat/completions`, `/v1/completions`, `/v1/models` | `infer` |
| `/v1/embeddings` | `embed` |
| `/admin/v1/...` | `admin` |

Then the model allowlist: a key with a non-empty `models` list may use only those models; an **empty** list means all models. Requests that name no model (`GET /v1/models`) pass the allowlist. A failure here is **403** `permission_error` / `insufficient_scope`: the caller is known but not allowed, which is different from 401 (unknown caller), and clients treat them differently (a 401 means "fix your key", a 403 "ask for access"). The `debug` scope grants nothing on its own; it unlocks the `X-TL-Route` response header in `gw.05`.

The **tenant** is the unit that owns keys: one company or team. Later stages scope by it (the cache never shares answers across tenants) or by the key (rate limits).

## 3. Worked example by hand

The gateway's pepper is `pepper-demo`. A key was created for tenant `acme` with scopes `[infer]`, `rpm` 60, `priority` 5, and allowlist `[smol-135m]`. The client presents

```
Authorization: Bearer tl_demo2key7abc_S3cretS3cretS3cretS3cretS3cret01
```

1. Parse: 48 characters, prefix `tl_`, `_` at position 15. $\mathit{id}$ = `demo2key7abc` (12 characters, all in `[a-z2-7]`), $s$ = `S3cretS3cretS3cretS3cretS3cret01` (32, all in `[A-Za-z0-9]`).
2. Find the record by id: one map lookup.
3. Compute $d' = \mathrm{HMAC\text{-}SHA256}(\texttt{pepper-demo}, s)$:

```
94f4a20cad218d9e113440c90096b20064d78a02a3cb6f72210222c4d6729506
```

(check it: `python3 -c "import hmac,hashlib;print(hmac.new(b'pepper-demo',b'S3cretS3cretS3cretS3cretS3cret01',hashlib.sha256).hexdigest())"`). It equals the record's `hmac`, so the key is valid. The unpeppered `SHA256(s)` would be `118a232e...712e64`, which must **not** verify.
4. The request is `POST /v1/chat/completions` with `"model": "smol-135m"`: the route needs `infer` (held) and the model is on the allowlist, so it is authorized.
5. The stage passes on `Principal{KeyID: "demo2key7abc", Tenant: "acme", Scopes: [infer], RPM: 60, Priority: 5, ...}`, deletes `Authorization`, and sets `X-TL-Priority: 5` (overwriting nothing: the server already stripped the client's own `X-TL-*` headers).

The same key on `/v1/embeddings` is a 403 (no `embed` scope), and with `"model": "tinystories-10m"` a 403 (not on the allowlist). These are `TestHandExampleLookup`, `TestAuthorizeRBAC`, and `TestMiddleware401And403`.

## 4. The interface

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

The catalog sketched `Create(ctx, tenant, scopes)`; `Create` takes a `KeySpec` because the admin API's `KeyCreate` also sets `rpm`, `tpm`, `priority`, `models`, and `expires_at` (DEVIATIONS B93-04). Log with `log/slog`: every line names the key id, never the presented key.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestHandExampleLookup` | unit | section 3: the digest, the lookup, the principal; an unpeppered record fails | the file format your CLI and the gateway share |
| `TestParseKey` | boundary | the hand key parses; 9 malformed keys (case, `1`, separators, length, punctuation) do not | garbage never reaches a lookup |
| `TestCreateLookupRoundTrip` | unit | `Create` returns `tl_<id>_<secret>`; the file holds only the HMAC; another store on the file sees it; no scopes and empty pepper refused | `<system> keys create` |
| `TestLookupRejects` | boundary | wrong last character, unknown id, revoked, malformed: `ErrInvalidKey`; expiry exactly at `expires_at` (fake clock) | one 401 for every failure |
| `TestCacheChecksTheSecret` | unit | a cached id with a wrong secret is still rejected | the cache cannot become a bypass |
| `TestRevokedKeyRejectedWithinTTL` | fault | the revoking store rejects at once; another store within 5 s; `ErrNotFound`; `List` | revocation is the emergency brake |
| `TestAuthorizeRBAC` | unit | the route-scope table and the allowlist, 13 cases | 403 decisions |
| `TestMiddleware401And403` | unit, conformance | no key, bad key: 401; missing scope, model not allowed: 403; a good key reaches the proxy with its principal, no `Authorization`, `X-TL-Priority: 5`; a bad body after a good key: 400 | conformance `auth.401`, `auth.403`, `priority.internal` |
| `TestNoPlaintextKeyInLogs` | property | after good, wrong, and malformed keys, creates and revokes, the logs name key ids and contain no secret or pepper | logs are not a leak |
| `TestKeysAdminHandler` | unit, conformance | `POST` 201 with defaults and no HMAC; bad tenants and scopes 400; `GET` lists without HMACs; `DELETE` 204 then the key fails; unknown id 404 | `gw.07` mounts it at `/admin/v1/keys` |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. storing too much or hashing the wrong thing: the secret in the file, no pepper, the HMAC over the whole key, a lax parser, the HMAC in admin output | a copied keys file or admin listing is enough to use or attack keys; keys never verify | `TestHandExampleLookup`, `TestCreateLookupRoundTrip`, `TestParseKey`, `TestKeysAdminHandler` (mutants `s03`, `s04`, `s12`, `s16`, `s17`) |
| 2. revocation and expiry that do not bite: the flag not checked, a cache that never expires, `expires_at` ignored | a leaked key keeps working after revocation | `TestRevokedKeyRejectedWithinTTL`, `TestLookupRejects` (mutants `s08`, `s09`, `s10`) |
| 3. a cache keyed by id alone | any secret works for a recently used id | `TestCacheChecksTheSecret` (mutant `s01`) |
| 4. authorization slips: the allowlist ignored, an empty allowlist read as "nothing", embeddings under `infer`, 401 for a scope failure | keys use models they were not granted, or nothing works | `TestAuthorizeRBAC`, `TestMiddleware401And403` (mutants `s05`, `s06`, `s07`, `s15`) |
| 5. logging the presented key on failure | every typo of a real key lands in your log store | `TestNoPlaintextKeyInLogs` (mutant `s11`) |
| 6. forwarding `Authorization`, or no `X-TL-Priority` from the key | the key reaches engine logs; the engine scheduler (`L10.2`) treats every tenant alike | `TestMiddleware401And403` (mutants `s13`, `s14`) |

## 6. Where it's used next
| Forward | `gw.08` | Registered module relationship. |

| Direction | Module | How it uses this |
|---|---|---|
| Back | `gw.01` | the stage is a `server.Middleware`; it reads the model with `ex.Request` and answers with `WriteError` |
| Forward | `gw.03` | buckets are keyed by `Principal.KeyID` with the key's `RPM` and `TPM` |
| Forward | `gw.05` | `X-TL-Route` only for principals with the `debug` scope |
| Forward | `gw.06` | cache keys include `Principal.Tenant` |
| Forward | `gw.07` | ledger rows carry tenant and key id; the admin API checks `admin` and mounts `KeysHandler` |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `tl_<id>_<secret>` keys | Stripe and GitHub token formats | prefixes that secret scanners recognize, checksums that catch typos offline | [GitHub's token format](https://github.blog/engineering/platform-security/behind-githubs-new-authentication-token-formats/) (free) |
| the keys file | Envoy external authorization, Vault | a separate auth service, short-lived credentials, audit devices | [Envoy ext_authz](https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/ext_authz_filter) (free) |
| scopes and allowlists | LiteLLM virtual keys, OpenAI projects | per-key budgets and model access, team hierarchies | [LiteLLM virtual keys](https://docs.litellm.ai/docs/proxy/virtual_keys) (free) |
| `Authorize` | OPA, Cedar | policy as code, evaluated and tested apart from the service | [Cedar](https://www.cedarpolicy.com/) (free) |
