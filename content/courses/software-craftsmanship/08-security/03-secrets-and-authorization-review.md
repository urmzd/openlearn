# Secrets and authorization review

API keys and tenant scopes are what stand between the public internet and the system's models and usage data. The [threat model](01-threat-model-stride.md) named the threats against them (a stolen key impersonating a tenant, a caller tampering with the tenant scope of a request, an operator pushing a bad route) and pointed at controls. This section checks those controls in detail. You review the authorization design as a matrix of principals, actions, and resources, and then inspect every place a secret could leak.

The review builds on two earlier pieces: the spoofing, tampering, and control-plane rows of the threat model, and the key format, scopes, and store described in [API Keys, Scopes, Model Allowlists, and Tenants](../../ai-platform-engineering/08-authorization-and-access-control/01-api-keys-and-scopes.md) in *AI Platform Engineering*. The model answer is [`security-review.md`](../../../../course/solutions/docs/security-review.md) in the solutions tree: an authorization matrix, a secret inventory, the evidence required before release, and a finding policy. Its claims are checked against the gateway's key handling in [`auth.go`](../../../../course/solutions/go/gateway/auth/auth.go). Like the other artifacts in this chapter, it is standalone and has no code call site.

Try filling in the matrix yourself before reading the model's. The cells you hesitate over are usually the ones the review exists to settle.

> **Learning objectives**
> - Build an authorization matrix of principals against actions and state the condition on every allow.
> - Explain why a caller-supplied tenant id is a selector and never proof of authorization.
> - Inventory secrets with their source, runtime use, logging rule, and rotation path.
> - Trace keyed hashing, constant-time comparison, and verified-key caching through the gateway's code.
> - Define release-blocking findings and the evidence a release must attach.

## The principles behind the review

The working rules are short. Store only keyed hashes of API secrets, compare credentials in constant time, scope every operation, and audit denies without logging presented secrets or prompt content. Rotate and revoke through a documented path. Each rule has a reason, and the reasons are what let you apply them to a system you have not seen before.

Authorization is a function of (principal, action, resource). A matrix with principals as rows and actions as columns makes every cell an explicit decision, so a missing rule shows up as a blank cell instead of a default nobody chose. The principal must come from authentication, and the resource's owner from the system's own records. Anything the caller sends, such as a tenant id in a URL or body, only selects which resource is meant; if it also decided access, any caller could claim any tenant. And a denial must not leak information: if "not yours" and "does not exist" answered differently, a caller could enumerate other tenants' identifiers, so both get the same external error.

Secrets follow a parallel line of reasoning. A stored secret is a liability, because whoever reads the store can use it. So the gateway stores a keyed hash (HMAC under a pepper kept elsewhere), which cannot be reversed and cannot be guessed offline without the pepper. Comparison time must not depend on how much of a guess matches, or the response time leaks the secret byte by byte, so comparisons are constant time. Logs, images, and diagnostic bundles are copied widely, and a secret that reaches any of them is effectively public, so logs carry identifiers (key id, request id, tenant id) and never the presented secret, headers, prompts, or tool arguments. Finally, caching verified keys trades revocation latency for speed. That trade is acceptable only if the window is stated and measured.

## Checking it by hand

The review starts from three concrete checks: an infer-only key cannot call admin routes, tenant A cannot retrieve tenant B's usage, and revocation takes effect within the documented cache window. Alongside them, you search source, logs, image layers, and CI artifacts for credential patterns.

Walk through the first check. An infer-only key calls `GET /admin/v1/...`. The gateway's `RequiredScope` maps any `/admin/` path to the `admin` scope. `Authorize` finds that the principal lacks it and returns `ErrInsufficient`. The middleware answers 403 `insufficient_scope` and logs the key id and route, not the key.

Now the third. With `key_cache_ttl_s = 30`, a key revoked at time $t$ may still be accepted by a gateway that cached it at $t - 1$, until $t + 29$. The cache window is that TTL, and the review must both state it and measure it.

## What the review must contain

The artifact is `docs/security-review.md`. The following table lists its required elements and what each one buys a later reader.

| Element | What it states | Why it matters downstream |
|---|---|---|
| Scope, date, reviewers | What was reviewed, when, by whom. | Findings are tied to a version of the system. |
| Authorization matrix | Every principal against every action, with the condition on each allow. | Gaps and over-grants are visible at a glance. |
| Secret inventory | Each secret's source, runtime use, logging rule, rotation and recovery. | Leaks and stale credentials have a known response. |
| Required evidence | Tests and scans to attach before release, including the measured revocation window. | Claims are checked, not asserted. |
| Finding policy | Owner, severity, due date, retest; what blocks release. | Two reviewers make the same release decision. |

## Building the review step by step

With the principles and required contents fixed, you can walk through the model review section by section, checking each claim against the gateway's code as you go.

### The authorization matrix

The model's matrix covers four principals and five actions:

| Principal | Inference | Read own usage | Read another tenant | Update routes | Rotate/revoke keys |
|---|---:|---:|---:|---:|---:|
| `tenant-key:infer` | allow, quota checked | deny | deny | deny | deny |
| `tenant-key:admin` | allow, quota checked | allow, same tenant only | deny | deny | allow, same tenant only |
| `operator:release` | deny | deny | deny | allow, audited and conditional on current ETag | deny |
| anonymous | deny before dispatch | deny | deny | deny | deny |

Read it cell by cell, as a matrix is meant to be read. The "Read another tenant" column is all deny: that is the cross-tenant rule expressed as a column, and it is what release blocks on. Every allow has a condition (quota for inference, "same tenant only" for usage and keys, ETag and audit for routes); an unconditional allow would be a smell. `operator:release` can change routes but cannot infer or read usage, a separation of duties that means a compromised release credential cannot read tenant data. Anonymous callers are denied "before dispatch": no handler runs, so there is no unauthenticated code path to have a bug in.

Under the matrix, the review states its rules:

> Every allow decision is derived from the authenticated principal and resource owner. A caller supplied tenant id is a selector, never proof of authorization. A mismatch returns the same external error shape as a missing resource to avoid confirming another tenant's identifiers.

These are the principal-from-authentication and no-information-in-denials principles, written as rules a reviewer can hold the code to.

### Checking the matrix against the code

The gateway's [`auth.go`](../../../../course/solutions/go/gateway/auth/auth.go) is where the scope side of the matrix is enforced:

```go
// RequiredScope is the scope a path needs: /v1/embeddings needs embed,
// /admin/v1/... admin, every other path (chat, completions, models) infer.
func RequiredScope(route string) string {
	switch {
	case strings.HasPrefix(route, "/admin/"):
		return ScopeAdmin
	case route == "/v1/embeddings":
		return ScopeEmbed
	default:
		return ScopeInfer          // the default is the least privileged scope
	}
}

func Authorize(p Principal, route, model string) error {
	if !p.HasScope(RequiredScope(route)) {
		return ErrInsufficient     // 403 insufficient_scope
	}
	if model != "" && len(p.Models) > 0 {   // an optional per-key model allowlist
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

The scope comes from the path, and `p` comes from `Lookup`, so neither is asserted by the caller. Later stages read the principal from the request context with `PrincipalFrom`: the response cache scopes entries by its tenant, and the ledger records its tenant and key id. The admin handler in [`ledger/admin.go`](../../../../course/solutions/go/gateway/ledger/admin.go) fails closed: no principal is a 401 and no admin scope is a 403, even if the handler were mounted outside the authentication stage by mistake. Whether each tenant-scoped cell ("same tenant only") holds for a given route is what the review's cross-tenant test establishes. The matrix states the target; the test is the evidence.

### The secret inventory

The review inventories three secrets:

| Secret | Source | Runtime use | Logging rule | Rotation / recovery |
|---|---|---|---|---|
| Tenant API key | one-time admin issuance | keyed hash lookup and constant-time comparison | never log input, hash, or header | issue replacement, verify it, revoke old key, confirm cache expiry |
| Service identity | workload identity | gateway to engine | log principal id only | rotate through platform identity controller |
| Route signing credential | secret manager | operator control plane | never place in command output or audit payload | revoke credential, restore last known route using current ETag |

The API key row maps directly onto the code:

```go
// Digest is the stored form of a secret: hex HMAC-SHA256 keyed by the
// pepper. Without the pepper, a leaked keys file does not let anyone test
// guesses offline.
func Digest(pepper []byte, secret string) string {
	m := hmac.New(sha256.New, pepper)
	m.Write([]byte(secret))
	return hex.EncodeToString(m.Sum(nil))
}

func (s *Store) Lookup(ctx context.Context, presented string) (Principal, error) {
	id, secret, ok := ParseKey(presented)   // "tl_<id>_<secret>": id for O(1) lookup
	if !ok {
		return Principal{}, ErrInvalidKey
	}
	digest := Digest(s.pepper, secret)
	now := s.clock.Now()
	// ...
	if c, ok := s.cache[id]; ok && now.Before(c.until) {
		if hmac.Equal([]byte(c.digest), []byte(digest)) {   // constant-time comparison
			return c.p, nil                                 // served from cache until the TTL
		}
		return Principal{}, ErrInvalidKey
	}
	// ... reload the keys file
	r, ok := s.recs[id]
	if !ok || !hmac.Equal([]byte(r.HMAC), []byte(digest)) {
		return Principal{}, ErrInvalidKey
	}
	if r.Revoked || (r.ExpiresAt != nil && !now.Before(*r.ExpiresAt)) {
		return Principal{}, ErrInvalidKey   // revoked and expired look the same as wrong
	}
	// ... cache the principal for s.ttl
}
```

Each secret-handling principle has a line here. Keyed hashing is `Digest`. Constant-time comparison is `hmac.Equal`. The revocation tradeoff is the cache: a cached entry is returned without consulting the revoked flag until `c.until`, which is why "confirm cache expiry" is part of rotation and why the review measures the window. `NewStore` refuses an empty pepper, so the gateway never falls back to bare hashes. On failure the middleware logs `"key_id", id` only, which keeps the presented secret out of the logs.

The inventory closes with a statement about where secrets may never appear:

> No secret is baked into an image, committed to config, or copied into a diagnostic bundle. Logs retain request id, tenant id, key id, decision, and policy reason; they omit authorization headers, prompts, and tool arguments. Incident responders can revoke by key id without recovering the presented secret.

The last sentence explains why keys carry a public id at all: revocation needs an identifier that is safe to log and safe to type into a runbook.

### Required evidence before release

The review then lists what must be attached before a release ships:

> - Tests show an infer-only key cannot call administrative routes.
> - Cross-tenant usage lookup is denied even when the caller supplies another tenant id.
> - Revocation takes effect within the declared cache window; the measured window is attached to the release record.
> - Secret scanning covers source, container layers, CI artifacts, and logs. Findings include the scan revision and disposition.
> - Route updates require the current ETag, are audited, and have a tested rollback.

There is one check per high-risk cell of the matrix and per leak path in the inventory. Container layers are listed separately from source because a secret deleted in a later layer is still present in the earlier one.

### The finding policy

Finally, the review says what blocks a release:

> Release is blocked for cross-tenant access, exposed credentials, or unauthenticated route mutation. A false positive is closed only with evidence and reviewer approval.

The three blockers are the three failures that harm others and cannot be undone after the fact. Everything else is triaged by severity and due date.

The full model answer is [`security-review.md`](../../../../course/solutions/docs/security-review.md).

### Why this design

A matrix beats a prose list of rules because blanks and over-grants become visible, and each cell maps to a test. A keyed hash (HMAC with a pepper) beats a plain hash: API secrets here are 32 random characters, so brute force is already hard, but the pepper means a leaked keys file alone is useless. A slow password hash such as bcrypt would add latency on every request for no gain on high-entropy secrets. A short verified-key cache beats a store read per request, since the keys file is reread only on a cache miss; the cost is a bounded revocation delay, stated and measured. The one tradeoff the design accepts knowingly is that returning the same error for "not found" and "not yours" makes some client debugging harder, in exchange for not confirming other tenants' identifiers.

## What can go wrong

Reviews like this one fail in a few recurring ways. A review can name a control without the evidence that supports it, leaving reviewers unable to verify the claim against the repository or system; the model's "Required evidence before release" lists the test or scan for each high-risk cell, and the code locations are named. A review can omit the failure or recovery case, so the happy path passes while an operator has no safe response under failure; the inventory's last column gives rotation and recovery for every secret. And a review can leave the owner, threshold, or scope implicit, so two reviewers reach different release or risk decisions from the same evidence; the model puts scope, date, and reviewers at the top and names the three release blockers in its finding policy.

> **Warning:** Never authorize from a caller-supplied id. Doing so produces cross-tenant reads. The matrix rule (a tenant id is a selector, never proof) and the required cross-tenant test exist to catch exactly this.

## Where this leads

This review closes the loop on two earlier pieces. It checks in detail the spoofing, tampering, and control-plane threats from the [threat model](01-threat-model-stride.md), and it verifies the behaviour of the key format, scopes, and store from [API Keys, Scopes, Model Allowlists, and Tenants](../../ai-platform-engineering/08-authorization-and-access-control/01-api-keys-and-scopes.md).

### Further reading

- [OWASP ASVS](https://owasp.org/www-project-application-security-verification-standard/): verification requirements for authentication, session, and access control, usable as a checklist against the matrix.
- [`auth.go`](../../../../course/solutions/go/gateway/auth/auth.go): `NewStore`, `Lookup`, `Authorize`, and `Middleware`, every claim in the review's API-key row, in code.

## Key terms

- **Authorization matrix**: a table of principals against actions in which every cell is an explicit allow (with its condition) or deny.
- **Selector**: a caller-supplied identifier that picks which resource is meant but never decides access.
- **Pepper**: a secret key, stored apart from the hashes, under which API secrets are HMAC-hashed.
- **Constant-time comparison**: a comparison whose running time does not depend on how much of the input matches.
- **Revocation window**: the time a revoked key may still be accepted, bounded here by the verified-key cache TTL.

## Summary

- An authorization review is a matrix of principals, actions, and resources; every allow is derived from the authenticated principal and the resource owner, never from a caller-supplied id.
- Denials for "not yours" and "does not exist" look the same, so callers cannot enumerate other tenants.
- Secrets are inventoried with their source, runtime use, logging rule, and rotation path; the presented secret is never logged, and revocation works by public key id.
- The gateway stores HMAC digests under a pepper, compares with `hmac.Equal`, and caches verified keys for a stated, measured window.
- Release is blocked on named failures (cross-tenant access, exposed credentials, unauthenticated route mutation), and each finding has an owner and retest evidence.

## Review questions

1. Why is a tenant id in the request URL not enough to authorize reading that tenant's usage?

   <details><summary>Answer</summary>

   The caller controls the URL. If the id decided access, any authenticated key could read any tenant by changing it. The id selects the resource; access is decided by comparing the authenticated principal's tenant with the resource's owner.

   </details>

2. A key is revoked, but requests with it succeed for another 20 seconds. Is that a bug?

   <details><summary>Answer</summary>

   Not if the declared cache window (the key cache TTL) is at least 20 seconds: `Lookup` serves a cached principal until the entry expires without rereading the revoked flag. It is a bug if it exceeds the declared window, which is why the release record attaches the measured window.

   </details>

3. Why does `Lookup` compare digests with `hmac.Equal` instead of `==`?

   <details><summary>Answer</summary>

   String equality returns as soon as a byte differs, so its running time reveals how long the matching prefix is. `hmac.Equal` takes the same time whatever the contents, so timing tells an attacker nothing.

   </details>

4. Why does an incident responder never need the presented secret?

   <details><summary>Answer</summary>

   Every key has a public id, logged on each decision, and revocation works by id. The responder can find which key misbehaved and revoke it from logs alone, so no process or person ever needs to see or store the secret.

   </details>
